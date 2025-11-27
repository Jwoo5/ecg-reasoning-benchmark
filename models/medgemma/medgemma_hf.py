# AutoModelForImageTextToText is available in transformers >= 4.50.0,
# while other models are available in earlier versions (e.g., gem, pulse)
# so we catch the import error here for backward compatibility.
try:
    from transformers import AutoModelForImageTextToText, AutoProcessor
except:
    pass
import logging

import torch
from transformers import BitsAndBytesConfig

from .. import BaseModel, register_model

logger = logging.getLogger(__name__)


@register_model("medgemma-hf")
class MedGemmaHFModel(BaseModel):
    def __init__(
        self, hf_model_variant: str = "4b-it", use_quantization: bool = True, is_thinking: bool = False
    ):
        self.hf_model_variant = hf_model_variant
        self.is_thinking = is_thinking

        if "27b" in hf_model_variant and is_thinking:
            self.max_new_tokens = 1300
        else:
            self.max_new_tokens = 300

        model_id = f"google/medgemma-{hf_model_variant}"

        model_kwargs = dict(
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

        if use_quantization:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)

        self.model = AutoModelForImageTextToText.from_pretrained(model_id, **model_kwargs)
        self.processor = AutoProcessor.from_pretrained(model_id)

    def get_response(self, conversation, verbose: bool = False):
        assert (
            conversation.conversation[0]["role"] == "system"
        ), "The first turn in the conversation must be from the system."
        assert (
            conversation.conversation[-1]["role"] == "user"
        ), "The last turn in the conversation must be from the user."
        assert (
            "image" in conversation.conversation[1]
        ), "The conversation must contain an ECG image in the first user turn."

        system = conversation.conversation[0]["text"]

        if "27b" in self.hf_model_variant and self.is_thinking:
            system += f"SYSTEM INSTRUCTION: think silently if needed."

        messages = [{"role": "system", "content": [{"type": "text", "text": system}]}]
        for i, turn in enumerate(conversation.conversation[1:]):
            if turn["role"] == "user":
                user_text = f"Question: {turn['question']}\n\n"
                if i == 0:
                    user_text += "Options:\n"
                elif "select all possible leads" in turn["question"].lower():
                    user_text += (
                        "This question may have multiple correct answers from the following options:\n"
                    )
                else:
                    user_text += "This question has one of the following options as the correct answer:\n"
                for option in turn["options"]:
                    user_text += f"- {option}\n"
                user_text += "Your response must be **ONLY** the full text of the selected option. Do not "
                user_text += "include any uncertainty, explanation, reasoning, or extra words."

                if i == 0:
                    user = {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": turn["image"]},
                            {"type": "text", "text": user_text},
                        ],
                    }
                else:
                    user = {"role": "user", "content": [{"type": "text", "text": user_text}]}
                messages.append(user)
            elif turn["role"] == "model":
                messages.append({"role": "assistant", "content": [{"type": "text", "text": turn["text"]}]})

        if verbose:
            print(f"\nQuestion: {conversation.conversation[-1]['question']}")

        response = self.generate(messages)

        if verbose:
            print(f"Response: {response}")

        return response

    def generate(self, messages, **kwargs):
        inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
        ).to(self.model.device, dtype=torch.bfloat16)

        input_len = inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
            output = output[0][input_len:]

        response = self.processor.decode(output, skip_special_tokens=True).strip()

        return response

    @classmethod
    def build_model(cls, hf_model_variant="4b-it", use_quantization=True, is_thinking=False, **kwargs):
        return cls(
            hf_model_variant=hf_model_variant, use_quantization=use_quantization, is_thinking=is_thinking
        )