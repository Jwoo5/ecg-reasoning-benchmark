import argparse
import glob
import io
import json
import logging
import os
import random
import shutil
from typing import Dict, Optional, Tuple, Union

import ecg_plot
import matplotlib.pyplot as plt
import numpy as np
import torch
import wfdb
from PIL import Image
from tqdm import tqdm

from models import BaseModel, build_model, get_model_name
from utils import Conversation, base64_image_encoder

logger = logging.getLogger(__name__)

system_prompt = """You are an expert cardiologist specializing in advanced electrocardiography. \
You are participating in a rigorous clinical reasoning examination designed to evaluate your \
ability to interpret ECGs based on formal, authoritative guidelines (e.g., AHA/ACC/HRS \
Recommendations).

**Instructions:**
1. Read the question and the provided options.
2. Analyze the ECG systematically to answer the question.
3. Select the answer from the given options that correspond to the findings **visible in the \
ECG image** or the correct diagnostic criterion requested.
4. When making a decision, you must base your judgment strictly on established diagnostic \
criteria defined in standard textbooks (e.g., Goldberger, Marriott, or AHA/ACC Guidelines). \
You must finalize your diagnosis only when sufficient evidence exists, while explicitly \
acknowledging that further findings are required if only necessary conditions are met.

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
        "--model-variant",
        type=str,
        default=None,
        help="model variant for any models (e.g., '4b-it', '27b-it' for medgemma_hf)",
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
        "--enable-condensed-chat",
        action="store_true",
        help=(
            "whether to enable condensed chat mode to reduce the context length. if enabled, "
            "answer options will only be provided in the last turn of the conversation, and previous "
            "turns will contain only the answer text without the options"
        ),
    )
    parser.add_argument(
        "--output-dir", type=str, default="results", help="output directory to save the results"
    )
    parser.add_argument(
        "--rebase", action="store_true", help="whether to re-base and overwrite existing results"
    )
    parser.add_argument(
        "--debug", action="store_true", help="whether to run in debug mode with verbose logging"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="whether to enable verbose logging during inference"
    )
    parser.add_argument(
        "--do-not-skip-empty-output",
        action="store_true",
        help=(
            "whether to NOT skip samples that already have empty output files, which indicates "
            "(1) previous run was interrupted during processing the sample, or (2) another process "
            "is currently processing the same sample concurrently. Enabling this option forces "
            "re-processing such samples to handle (1) case."
        )
    )

    return parser


class Inferencer:
    def __init__(self, model: BaseModel, debug: bool = False, verbose: bool = False):
        self.model = model
        self.model_name = get_model_name(model)
        self.debug = debug
        self.verbose = verbose

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
            if self.model_name == "opentslm":
                path = os.path.join(base_dir, "records100", f"{dir_num:05d}", f"{int(ecg_id):05d}_lr")
            else:
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

        return image

    def get_response(
        self,
        conversation: Conversation,
        enable_condensed_chat: bool = False,
        verbose: bool = False,
        target_dx: Optional[str] = None,
    ) -> str:
        return self.model.get_response(
            conversation, enable_condensed_chat=enable_condensed_chat, verbose=verbose, target_dx=target_dx
        )

    def proceed_step(
        self,
        step: Dict,
        conversation: Conversation,
        ecg_signal: Optional[torch.Tensor] = None,
        ecg_image: Optional[Image.Image] = None,
        return_response: bool = False,
        require_base64_image: bool = False,
        enable_condensed_chat: bool = False,
        verbose: bool = False,
        target_dx: Optional[str] = None,
    ) -> Optional[str]:
        question = step["question"]
        # indexed_options = make_letter_indexed(step["options"])
        indexed_options = step["options"]

        if require_base64_image and ecg_image is not None:
            ecg_image = base64_image_encoder(ecg_image)

        conversation.add_user_turn(question, indexed_options, ecg_signal=ecg_signal, ecg_image=ecg_image)
        response = self.get_response(
            conversation, enable_condensed_chat=enable_condensed_chat, verbose=verbose, target_dx=target_dx
        )
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

    def inference(self, sample: Dict, ecg_base_dir: str, enable_condensed_chat: bool = False) -> Dict:
        dx = sample["metadata"]["target_dx"].replace("_", " ")

        sample_result = sample.copy()
        sample_result["metadata"]["model"] = self.model_name

        ecg_tensor, sampling_rate = self.get_ecg_signal(
            source_dataset=sample["metadata"]["data_source"],
            ecg_id=sample["metadata"]["ecg_id"],
            base_dir=ecg_base_dir,
            subject_id=sample["metadata"].get("subject_id", None),
        )
        ecg_image = self.visualize_ecg(ecg_tensor, sampling_rate)

        conversation = Conversation(system_prompt)

        if self.model.ecg_modality_base == "image":
            sample["data"]["initial_diagnostic_question"]["question"] += (
                " Note that the red grid in the provided ECG image follows standard calibration: "
                "one large square (5 mm) represents 0.2 seconds on the horizontal axis and 0.5 mV "
                "on the vertical axis."
            )

        response = self.proceed_step(
            step=sample["data"]["initial_diagnostic_question"],
            conversation=conversation,
            ecg_signal=ecg_tensor,
            ecg_image=ecg_image,
            return_response=True,
            require_base64_image=self.model.require_base64_image(),
            enable_condensed_chat=enable_condensed_chat,
            verbose=self.verbose,
            target_dx=dx,
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
            pass
        else:
            logger.warning(f"Could not parse response: {response}")
            sample_result["metadata"]["parsing_error"] = True
            # return sample_result

        for stage in sample_result["data"]["reasoning"]:
            for step in stage.values():
                if isinstance(step, list):
                    # it is hit for grounding steps
                    for g_step in step:
                        self.proceed_step(
                            g_step,
                            conversation,
                            return_response=False,
                            enable_condensed_chat=enable_condensed_chat,
                        )
                else:
                    self.proceed_step(
                        step,
                        conversation,
                        return_response=False,
                        enable_condensed_chat=enable_condensed_chat,
                    )

        return sample_result


def main(args):
    model = build_model(args.model, model_variant=args.model_variant)
    inferencer = Inferencer(model, debug=args.debug, verbose=args.verbose)

    root_dir = args.root
    source_dataset = args.dataset
    ecg_base_dir = args.ecg_base_dir
    output_dir = args.output_dir

    model_name = args.model
    if args.model_variant:
        model_name += f"_{args.model_variant}"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    assert os.path.exists(
        os.path.join(root_dir, source_dataset + ".jsonl")
    ), f"Dataset not found: {os.path.join(root_dir, source_dataset + '.jsonl')}"

    # load benchmark data from jsonl file
    data = []
    with open(os.path.join(root_dir, source_dataset + ".jsonl"), "r") as f:
        for line in f:
            data.append(json.loads(line))

    if args.rebase and os.path.exists(os.path.join(output_dir, model_name, source_dataset)):
        shutil.rmtree(os.path.join(output_dir, model_name, source_dataset))

    if args.debug:
        random.seed(42)
        # for debugging, process only 10% of the data
        subset_len = len(data) // 10 if len(data) >= 10 else 1
        data = random.sample(data, subset_len)

    n = len(data)
    n_failed = 0
    n_total = 0

    with tqdm(total=n, ncols=140) as pbar:
        for sample in data:
            sample_id = sample["metadata"]["id"]
            dx = sample["metadata"]["target_dx"]


            save_path = os.path.join(output_dir, model_name, source_dataset, dx, f"{sample_id}.json")
            if not os.path.exists(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
            elif os.path.exists(save_path):
                if args.do_not_skip_empty_output and os.path.getsize(save_path) == 0:
                    # re-process the sample if the existing output file is empty
                    pass
                else:
                    pbar.update(1)
                    continue

            # write empty file to reserve the spot so that other processes do not process the same file
            with open(save_path, "w") as f:
                f.write("")

            result = inferencer.inference(
                sample,
                ecg_base_dir,
                enable_condensed_chat=args.enable_condensed_chat,
            )

            n_total += 1
            if "parsing_error" in result["metadata"] and result["metadata"]["parsing_error"]:
                n_failed += 1

            pbar.set_description(
                f"Processing {dx} | Sample {sample_id} | {n_total-n_failed}/{n_total} "
                f"| Failed: {n_failed}"
            )

            # save the actual result
            with open(save_path, "w") as f:
                json.dump(result, f, indent=4)

            pbar.update(1)


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
