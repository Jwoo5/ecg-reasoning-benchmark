import logging
import torch
import numpy as np
from huggingface_hub import hf_hub_download

from .. import BaseModel, register_model

# Assuming the OpenTSLM repo source files (model, prompt) are copied into models/opentslm/
from .prompt.full_prompt import FullPrompt
from .prompt.text_prompt import TextPrompt
from .prompt.text_time_series_prompt import TextTimeSeriesPrompt
from .model.llm.OpenTSLMFlamingo import OpenTSLMFlamingo
import pdb 

logger = logging.getLogger(__name__)


@register_model("opentslm")
class OpenTSLMModel(BaseModel):
    def __init__(self, device_map="auto", torch_dtype=torch.float16, **kwargs):
        # Configuration matches the provided working script
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.base_llm_id = "meta-llama/Llama-3.2-3B"
        self.checkpoint_repo_id = "OpenTSLM/llama3b-ecg-flamingo"
        self.checkpoint_filename = "best_model.pt"

        logger.info(f"Initializing OpenTSLM architecture with base: {self.base_llm_id}...")
        
        # Initialize the specific OpenTSLM architecture
        self.model = OpenTSLMFlamingo(
            device="cpu",
            llm_id=self.base_llm_id
        )

        logger.info(f"Downloading checkpoint from {self.checkpoint_repo_id}...")
        checkpoint_path = hf_hub_download(
            repo_id=self.checkpoint_repo_id,
            filename=self.checkpoint_filename
        )
        
        # Load weights
        self.model.load_from_file(checkpoint_path)
        self.model = self.model.to(self.device) 
        if hasattr(self.model, "device"):
            self.model.device = self.device
        self.model.eval()

    @classmethod
    def build_model(cls, device_map="auto", torch_dtype=torch.float16, **kwargs):
        return cls(device_map=device_map, torch_dtype=torch_dtype, **kwargs)

    def _process_signal(self, ecg_tensor: torch.Tensor, target_len=1000):
        """
        Adapts the load_and_process_ecg_ptbxl logic from OpenTSLM repo.
        Input: ecg_tensor (12, N) from the benchmark loader.
        Output: norm_ecg (numpy), means, stds
        """
        # Convert to numpy
        ecg = ecg_tensor.cpu().numpy().astype(np.float32)

        # Check dimensions (Benchmark usually provides (12, N))
        if ecg.shape[0] != 12:
            # Fallback if transpose is needed, though benchmark standard is (12, N)
            if ecg.shape[1] == 12:
                ecg = ecg.T
            else:
                raise ValueError(f"Expected 12 leads, got shape {ecg.shape}")

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
        pdb.set_trace()
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
        answer_part = output[idx + len(marker):].strip()

        # Clean up valid sentence endings from the answer part if strictly extracting options
        answer_part = answer_part.split("\n", 1)[0].strip()
        if answer_part.endswith((".", "!", "?")):
            answer_part = answer_part[:-1].strip()

        return reasoning, answer_part

    def get_prompt(self, conversation, norm_ecg, means, stds):
        """
        Constructs the OpenTSLM specific FullPrompt object.
        """
        # 1. Extract the current (last) user question and options
        # The benchmark appends the latest user question to the end of the conversation list.
        
        last_turn = conversation.conversation[-1]
        assert last_turn["role"] == "user", "Last turn must be user."
        
        question_text = last_turn["question"]
            
        options = last_turn.get("options", [])

        # 2. Build Pre-Prompt (Clinical Context + Instructions)
        # Note: OpenTSLM uses a specific prompting style. We ignore the generic system prompt from conversation[0].
        clinical_context = "12-lead clinical ECG recording."
        
        pre_prompt_str = f"""You are an expert cardiologist analyzing an ECG (electrocardiogram). 

Clinical Context: {clinical_context}

Your task is to examine the ECG signal and answer the following medical question:

Question: {question_text}

Instructions:
- Begin by analyzing the time series without assuming a specific answer.
- Think step-by-step about what the observed patterns suggest regarding the cardiac condition.
- Write your rationale as a single, natural paragraph — do not use bullet points, numbered steps, or section headings.
- Do **not** mention any final answer until the very end.
- Consider the ECG morphology, intervals, and any abnormalities that relate to the question.
"""

        # 3. Build Time-Series Prompts (Per lead stats)
        lead_names = ["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"]
        ts_prompts = []
        for idx, lead_signal in enumerate(norm_ecg):
            lead_name = lead_names[idx] if idx < len(lead_names) else f"Lead_{idx}"
            mean_val = means[idx]
            std_val = stds[idx]
            label = (
                f"This is ECG Lead {lead_name}, it has mean {mean_val:.4f} "
                f"and std {std_val:.4f}:"
            )
            ts_prompts.append(
                TextTimeSeriesPrompt(label, lead_signal.tolist())
            )

        # 4. Build Post-Prompt (Options + formatting constraint)
        if options:
            # Handle if options are dict/list. Benchmark often passes a list of strings.
            if isinstance(options, dict):
                options_text = ", ".join([f"({k}) {v}" for k, v in options.items()])
            else:
                options_text = ", ".join(options)

            post_prompt_str = f"""
Based on your analysis of the ECG data, select your answer from the following options. You must select from the options:
{options_text}

Make sure that your last word is the answer. You MUST end your response with "Answer: "
""".strip()
        else:
            post_prompt_str = """
Based on your analysis of the ECG data, provide your answer.
Make sure that your last word is the answer. You MUST end your response with "Answer: "
""".strip()

        full_prompt = FullPrompt(
            TextPrompt(pre_prompt_str),
            ts_prompts,
            TextPrompt(post_prompt_str)
        )
        return full_prompt

    def get_response(self, conversation) -> str:
        # 1. Retrieve ECG signal from the conversation
        # The benchmark stores the signal in the first user turn (index 1 usually, or searched)
        ecg_signal = None
        for turn in conversation.conversation:
            if "signal" in turn and turn["signal"] is not None:
                ecg_signal = turn["signal"]
                break
        
        if ecg_signal is None:
            raise ValueError("No ECG signal found in conversation turns.")

        # 2. Process ECG (Normalize/Downsample)
        norm_ecg, means, stds = self._process_signal(ecg_signal)

        # 3. Build the specific OpenTSLM Prompt
        full_prompt = self.get_prompt(conversation, norm_ecg, means, stds)

        with torch.inference_mode():
            answer_text = self.model.eval_prompt(
                full_prompt,
                max_new_tokens=500
            )

        reasoning, answer = self._split_reasoning_and_answer(answer_text)
        # pdb.set_trace()
        return answer