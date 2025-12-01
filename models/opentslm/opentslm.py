import logging

import numpy as np
import torch
from huggingface_hub import hf_hub_download

from .. import BaseModel, register_model
from .OpenTSLM.model.llm.OpenTSLMFlamingo import OpenTSLMFlamingo
from .OpenTSLM.prompt.full_prompt import FullPrompt
from .OpenTSLM.prompt.text_prompt import TextPrompt
from .OpenTSLM.prompt.text_time_series_prompt import TextTimeSeriesPrompt

logger = logging.getLogger(__name__)


@register_model("opentslm")
class OpenTSLMModel(BaseModel):
    def __init__(self):
        # Configuration matches the provided working script
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.base_llm_id = "meta-llama/Llama-3.2-3B"
        self.checkpoint_repo_id = "OpenTSLM/llama3b-ecg-flamingo"
        self.checkpoint_filename = "best_model.pt"

        # Initialize the specific OpenTSLM architecture
        self.model = OpenTSLMFlamingo(device="cpu", llm_id=self.base_llm_id)

        checkpoint_path = hf_hub_download(repo_id=self.checkpoint_repo_id, filename=self.checkpoint_filename)

        # Load weights
        self.model.load_from_file(checkpoint_path)
        self.model = self.model.to(self.device)
        if hasattr(self.model, "device"):
            self.model.device = self.device
        self.model.eval()

    @classmethod
    def build_model(cls, **kwargs):
        return cls()

    def _process_signal(self, ecg_tensor: torch.Tensor, target_len=1000):
        """
        Adapts the load_and_process_ecg_ptbxl logic from OpenTSLM repo.
        Input: ecg_tensor (12, N) from the benchmark loader.
        Output: norm_ecg (numpy), means, stds
        """
        # Convert to numpy
        ecg = ecg_tensor.cpu().numpy().astype(np.float32)

        L = ecg.shape[1]

        # Downsample if needed (OpenTSLM expects ~100Hz / length 1000)
        if L > target_len:
            stride = L // target_len
            if stride <= 0:
                stride = 1
            ecg = ecg[:, ::stride]
            L = ecg.shape[1]

        # Pad or cut to exactly target_len
        if L < target_len:
            new_ecg = np.zeros((ecg.shape[0], target_len), dtype=np.float32)
            new_ecg[:, :L] = ecg
            ecg = new_ecg
        elif L > target_len:
            ecg = ecg[:, :target_len]

        # Per-lead normalization
        norm_ecg = np.zeros_like(ecg)
        means = []
        stds = []
        for i in range(ecg.shape[0]):
            lead = ecg[i]
            m = float(lead.mean())
            s = float(lead.std())
            if s < 1e-6:
                norm = lead - m
            else:
                norm = (lead - m) / s
            norm_ecg[i] = norm
            means.append(m)
            stds.append(s)

        return norm_ecg, means, stds

    def _split_reasoning_and_answer(self, output: str):
        """
        Helper to parse OpenTSLM CoT output.
        """
        if output is None:
            return "", ""

        marker = "Answer:"
        idx = output.rfind(marker)
        if idx == -1:
            # No "Answer:" found, return whole text as reasoning
            return output.strip(), ""

        reasoning = output[:idx].strip()
        answer_part = output[idx + len(marker) :].strip()

        # Clean up valid sentence endings from the answer part if strictly extracting options
        answer_part = answer_part.split("\n", 1)[0].strip()
        if answer_part.endswith((".", "!", "?")):
            answer_part = answer_part[:-1].strip()

        return reasoning, answer_part

    def get_prompt(self, conversation, norm_ecg, means, stds):
        """
        Constructs the OpenTSLM specific FullPrompt object.
        """
        assert (
            conversation.conversation[0]["role"] == "system"
        ), "The first turn in the conversation must be from the system."
        assert (
            conversation.conversation[-1]["role"] == "user"
        ), "The last turn in the conversation must be from the user."
        assert (
            "image" in conversation.conversation[1]
        ), "The conversation must contain an ECG image in the first user turn."

        pre_prompt = "You are an expert cardiologist analyzing an ECG (electrocardiogram).\n\n"

        if len(conversation.conversation) > 2:
            clinical_context = (
                "The following is a previous dialogue history between a user and an AI assistant "
                "about the provided 12-lead clinical ECG recording.\n"
            )
            for turn in conversation.conversation[1:-1]:
                if turn["role"] == "user":
                    clinical_context += f"- {turn['question']} "
                elif turn["role"] == "model":
                    clinical_context += f"{turn['text']}\n"
        else:
            clinical_context = "12-lead clinical ECG recording."
        pre_prompt += f"Clinical Context: {clinical_context}\n\n"
        pre_prompt += "Your task is to examine the ECG signal and answer the following medical question:\n\n"

        last_turn = conversation.conversation[-1]
        question = last_turn["question"]
        pre_prompt += f"Question: {question}\n\n"

        pre_prompt += "Instructions:\n"
        pre_prompt += "- Begin by analyzing the time series without assuming a specific answer.\n"
        pre_prompt += (
            "- Think step-by-step about what the observed patterns suggest regarding the question.\n"
        )
        pre_prompt += (
            "- Write your rationale as a single, natural paragraph — do not use bullet points, "
            "numbered steps, or section headings.\n"
        )
        pre_prompt += "- Do **not** mention any final answer until the very end.\n"
        pre_prompt += (
            "- Consider the ECG morphology, intervals, and any abnormalities that relate to "
            "the question.\n"
        )

        lead_names = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
        ts_prompts = []
        for idx, lead_signal in enumerate(norm_ecg):
            lead_name = lead_names[idx] if idx < len(lead_names) else f"Lead_{idx}"
            mean_val = means[idx]
            std_val = stds[idx]
            label = f"This is ECG Lead {lead_name}, it has mean {mean_val:.4f} " f"and std {std_val:.4f}:"
            ts_prompts.append(TextTimeSeriesPrompt(label, lead_signal.tolist()))

        # for the very first question (initial diagnostic question)
        if len(conversation.conversation) == 2:
            post_prompt = "Options:\n"
        elif "select all possible leads" in question.lower():
            post_prompt = "This question may have multiple correct answers from the following options:\n"
        else:
            post_prompt = "This question has one of the following options as the correct answer:\n"

        options = last_turn["options"]
        for option in options:
            post_prompt += f"- {option}\n"

        post_prompt += (
            'Make sure that your last word is the answer. you MUST end your response with "Answer: "'
        )

        full_prompt = FullPrompt(TextPrompt(pre_prompt), ts_prompts, TextPrompt(post_prompt))
        return full_prompt

    def get_response(self, conversation, target_dx: str, verbose: bool = False, **kwargs) -> str:
        ecg_signal = conversation.conversation[1]["signal"]
        norm_ecg, means, stds = self._process_signal(ecg_signal)
        prompt = self.get_prompt(conversation, norm_ecg, means, stds)

        if verbose:
            print(f"\nQuestion: {conversation.conversation[-1]['question']}")

        response = self.generate(prompt)

        # For the initial diagnostic question, we specifically post-process to extract the final
        # answer as this model tends to phrase the option rather than select it directly.
        if len(conversation.conversation) == 2:
            if response == target_dx:
                response = "yes"
            elif response == "none":
                response = "no"

        if verbose:
            print(f"Response: {response}\n")

        return response

    def generate(self, prompt, **kwargs):
        with torch.inference_mode():
            output = self.model.eval_prompt(
                prompt,
                max_new_tokens=300,
            )

        _, response = self._split_reasoning_and_answer(output)

        return response
