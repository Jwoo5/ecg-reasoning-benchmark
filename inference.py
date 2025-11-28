import argparse
import glob
import io
import json
import logging
import os
import re
from typing import Dict, List, Optional, Tuple, Union

import ecg_plot
import matplotlib.pyplot as plt
import numpy as np
import torch
import wfdb
from PIL import Image
from tqdm import tqdm

from models import BaseModel, build_model, get_model_name
from utils import Conversation, base64_image_encoder, make_letter_indexed

logger = logging.getLogger(__name__)

# number of samples in each source-dataset
N_MIMIC_IV_ECG = 3355
N_PTBXL = 3076

system_prompt = """You are an expert cardiologist assistant evaluating a specific ECG case.

**Instructions:**
1. Read the question and the provided options.
2. Analyze the ECG systematically to answer the question.
3. Select the answer from the given options that correspond to the findings **visible in the \
ECG image** or the correct diagnostic criterion requested.

"""


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "root",
        type=str,
        metavar="DIR",
        default="data",
        help=(
            "path to the root directory containing ecg-reasoning-benchmark data samples for each "
            "of mimic_iv_ecg and ptbxl datasets (e.g., ecg-reasoning-benchmark/data)"
        ),
    )
    parser.add_argument(
        "--dataset", type=str, help="name of the dataset to run inference on (e.g., mimic_iv_ecg, ptbxl)"
    )
    parser.add_argument(
        "--model", type=str, help="name of the model to run inference on (e.g., gem, pulse, etc.)"
    )
    parser.add_argument(
        "--hf-model-variant",
        type=str,
        default=None,
        help="model variant for any huggingface models (e.g., '4b-it', '27b-it' for medgemma_hf)",
    )
    parser.add_argument(
        "--ecg-base-dir",
        type=str,
        metavar="DIR",
        help=(
            "base directory to load raw ECG signal files from (e.g., for ptbxl, directory that "
            "contains 'records500' subdirectory)"
        ),
    )
    parser.add_argument(
        "--output-dir", type=str, default="results", help="output directory to save the results"
    )
    parser.add_argument(
        "--debug", action="store_true", help="whether to run in debug mode with verbose logging"
    )

    return parser


class Inferencer:
    def __init__(self, model: BaseModel, debug: bool = False):
        self.model = model
        self.model_name = get_model_name(model)
        self.debug = debug

    def get_ecg_signal(
        self,
        source_dataset: str,
        ecg_id: str,
        base_dir: str,
        subject_id: Optional[str] = None,  # for mimic-iv-ecg
    ) -> Tuple[torch.Tensor, int]:
        if source_dataset.lower() == "ptbxl":
            assert os.path.exists(os.path.join(base_dir, "records500")), (
                f"PTB-XL dataset directory structure not found in {base_dir}. "
                "Expected to find 'records500' subdirectory."
            )

            dir_num = int(ecg_id) // 1000 * 1000
            path = os.path.join(base_dir, "records500", f"{dir_num:05d}", f"{int(ecg_id):05d}_hr")

            ecg_record, header = wfdb.rdsamp(path)
            sampling_rate = header["fs"]

            ecg_tensor = torch.tensor(ecg_record.T.astype(np.float32))
        elif source_dataset.lower() == "mimic_iv_ecg":
            assert os.path.exists(os.path.join(base_dir, "files")), (
                f"MIMIC-IV-ECG dataset directory structure not found in {base_dir}. "
                "Expected to find 'files' subdirectory."
            )

            path = os.path.join(
                base_dir, "files", f"p{subject_id:.4s}", f"p{subject_id}", f"s{ecg_id}", f"{ecg_id}"
            )

            ecg_record, header = wfdb.rdsamp(path)
            sampling_rate = header["fs"]

            ecg_record = ecg_record.T

            # adjust lead order to standard 12-lead ECG
            lead_order = header["sig_name"]
            adjusted_lead_order = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
            if lead_order != adjusted_lead_order:
                lead_indices = [lead_order.index(lead) for lead in adjusted_lead_order]
                ecg_record = ecg_record[lead_indices, :]

            ecg_tensor = torch.tensor(ecg_record.astype(np.float32))
        else:
            raise ValueError(f"Unsupported dataset: {source_dataset}")

        return ecg_tensor, sampling_rate

    def visualize_ecg(self, ecg_signal: Union[torch.Tensor, np.ndarray], sampling_rate: int) -> Image.Image:
        if isinstance(ecg_signal, torch.Tensor):
            ecg_signal = ecg_signal.numpy()
        ecg_plot.plot(ecg_signal, sample_rate=sampling_rate, row_height=8)

        fig = plt.gcf()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        image = Image.open(buf).convert("RGB")
        plt.close(fig)

        width, height = image.size
        image = image.resize((width // 2, height // 2))

        return image

    def get_response(self, conversation: Conversation, verbose: bool = False) -> str:
        return self.model.get_response(conversation, verbose=verbose)

    def proceed_step(
        self,
        step: Dict,
        conversation: Conversation,
        ecg_signal: Optional[torch.Tensor] = None,
        ecg_image: Optional[Image.Image] = None,
        return_response: bool = False,
        required_base64_image: bool = False,
        verbose: bool = False,
    ) -> Optional[str]:
        question = step["question"]
        # indexed_options = make_letter_indexed(step["options"])
        indexed_options = step["options"]

        if required_base64_image and ecg_image is not None:
            ecg_image = base64_image_encoder(ecg_image)

        conversation.add_user_turn(question, indexed_options, ecg_signal=ecg_signal, ecg_image=ecg_image)
        response = self.get_response(conversation, verbose=verbose)
        step["model_response"] = response

        # add model turn to conversation with the ground truth answer
        if "answer" in step:
            if isinstance(step["answer"], list):
                answer_str = ", ".join(step["answer"])
            else:
                answer_str = step["answer"]

            conversation.add_model_turn(answer_str)
        else:
            conversation.add_model_turn(response)

        if return_response:
            return response

    # def parse_response(self, response: str) -> Union[int, List[int]]:
    #     # parse the response to get the selected option index, where the response is expected to be
    #     # (a) ..., (b) ..., etc.
    #     # if the response is comma separated options, return all selected options as a list of indices
    #     # TODO we can parse the answer by exact match including the text to make it more concrete
    #     # e.g., match with "(a) left anterior fascicular block", not just "(a)"

    #     pattern = r"\(([a-z])\)"
    #     matches = re.findall(pattern, response)
    #     if matches:
    #         indices = [ord(match) - ord("a") for match in matches]
    #         if len(indices) == 1:
    #             return indices[0]
    #         else:
    #             return indices
    #     else:
    #         return -1

    def inference(self, sample: Dict, ecg_base_dir: str) -> Dict:
        sample_result = sample.copy()
        sample_result["metadata"]["model"] = self.model_name

        ecg_tensor, sampling_rate = self.get_ecg_signal(
            source_dataset=sample["metadata"]["data_source"],
            ecg_id=sample["metadata"]["ecg_id"],
            base_dir=ecg_base_dir,
            subject_id=sample["metadata"].get("subject_id", None),
        )
        ecg_image = self.visualize_ecg(ecg_tensor, sampling_rate)

        global system_prompt
        if self.model_name not in ["opentslm"]:
            system_prompt += (
                "Note that the provided ECG image shows the 10-second 12-lead ECG recording, where "
                "each red square grid represents 0.2 seconds horizontally and 0.5 mV vertically.\n\n"
            )

        conversation = Conversation(system_prompt)


        sample["data"]["initial_diagnostic_question"]["question"] += (
            " If you don't know how to analyze the ECG to answer this question, choose "
            "'I don't know'. Then, you will receive guidance on how to systematically "
            "analyze the ECG to improve your decision-making skills. NEVER choose 'I don't know' "
            "because you are uncertain or want to avoid answering the question. "
        )
        required_base64_image = False
        if self.model_name in ["qwen3-vl-hf", "qwen3-vl-vllm"]:
            required_base64_image = True

        response = self.proceed_step(
            step=sample["data"]["initial_diagnostic_question"],
            conversation=conversation,
            ecg_signal=ecg_tensor,
            ecg_image=ecg_image,
            return_response=True,
            required_base64_image=required_base64_image,
            verbose=self.debug,
        )
        sample_result["data"]["initial_diagnostic_question"]["model_response"] = response
        if (
            response.strip(".").lower() in ["yes", "no"]
            or response.strip(".").lower().startswith("yes")
            or response.strip(".").lower().startswith("**yes**")
            or response.strip(".").lower().endswith("yes")
            or response.strip(".").lower().endswith("**yes**")
            or response.strip(".").lower().startswith("no")
            or response.strip(".").lower().startswith("**no**")
            or response.strip(".").lower().endswith("no")
            or response.strip(".").lower().endswith("**no**")
        ):
            eval_path = 1
        elif (
            response.strip(".").lower() == "i don't know"
            or response.strip(".").lower() == "i don't know"
            or response.strip(".").lower().startswith("i don't know")
            or response.strip(".").lower().startswith("**i don't know**")
            or response.strip(".").lower().endswith("i don't know")
            or response.strip(".").lower().endswith("**i don't know**")
        ):
            eval_path = 2
        else:
            logger.warning(f"Could not parse response: {response}")
            sample_result["data"]["initial_diagnostic_question"]["eval_path"] = -1
            sample_result["metadata"]["parsing_error"] = True
            return sample_result

        if eval_path == 1:
            pass
            # del sample_result["data"]["path_2"]
        elif eval_path == 2:
            del sample_result["data"]["path_1"]

        sample_result["data"]["initial_diagnostic_question"]["eval_path"] = eval_path

        if eval_path == 2:
            logger.info(
                "Model responded with 'I don't know' for initial diagnosis question. "
                "Stop further questioning for this sample as the path 2 (guided reasoning) "
                "is not implemented yet."
            )
            return sample_result

        for stage in sample_result["data"][f"path_{eval_path}"]:
            for step in stage.values():
                if isinstance(step, list):
                    # it is hit for grounding steps
                    for g_step in step:
                        self.proceed_step(
                            # g_step, conversation, return_response=True, verbose=self.debug
                            g_step, conversation
                        )
                else:
                    self.proceed_step(
                        # step, conversation, return_response=False, verbose=self.debug
                        step, conversation
                    )

        return sample_result


def main(args):
    model = build_model(args.model, hf_model_variant=args.hf_model_variant)
    inferencer = Inferencer(model, debug=args.debug)

    root_dir = args.root
    source_dataset = args.dataset
    ecg_base_dir = args.ecg_base_dir
    output_dir = args.output_dir

    model_name = args.model
    if args.hf_model_variant:
        model_name += f"_{args.hf_model_variant}"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    assert os.path.exists(
        os.path.join(root_dir, source_dataset)
    ), f"Dataset directory not found: {os.path.join(root_dir, source_dataset)}"

    n = N_MIMIC_IV_ECG if source_dataset.lower() == "mimic_iv_ecg" else N_PTBXL
    n_path1 = 0
    n_failed = 0
    with tqdm(total=n, ncols=140) as pbar:
        for dx in sorted(os.listdir(os.path.join(root_dir, source_dataset))):
            for fname in sorted(glob.glob(os.path.join(root_dir, source_dataset, dx, "*.json"))):
                with open(fname, "r") as f:
                    sample = json.load(f)

                result = inferencer.inference(sample, ecg_base_dir)
                if result["data"]["initial_diagnostic_question"]["eval_path"] == 1:
                    n_path1 += 1
                elif result["data"]["initial_diagnostic_question"]["eval_path"] == -1:
                    n_failed += 1

                pbar.set_description(
                    f"Processing {dx} | Sample {os.path.basename(fname)} | {n_path1}/{pbar.n+1-n_failed} | Failed: {n_failed}"
                )

                save_path = os.path.join(output_dir, model_name, source_dataset, dx, os.path.basename(fname))
                if not os.path.exists(os.path.dirname(save_path)):
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)

                with open(save_path, "w") as f:
                    json.dump(result, f, indent=4)

                pbar.update(1)


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
