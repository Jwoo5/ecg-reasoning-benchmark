# AutoModelForImageTextToText is available in transformers >= 4.50.0,
# while other models are available in earlier versions (e.g., gem, pulse)
# so we catch the import error here for backward compatibility.
try:
    from transformers import AutoModelForCausalLM, AutoProcessor
except:
    pass

import logging

import torch

from .. import BaseModel, register_model

logger = logging.getLogger(__name__)


@register_model("hulumed-hf")
class HuluMedHFModel(BaseModel):
    def __init__(
        self,
        hf_model_variant: str = "7B",
    ):
        # Check transformers version for VideoInput
        try:
            from transformers.image_utils import VideoInput
        except:
            import transformers

            raise ImportError(
                "Hulu-Med HF model requires to import VideoInput from transformers.image_utils, "
                "which is available in transformers==4.51.2 but not in your current version "
                f"({transformers.__version__}). Please consider reinstalling transformers with the "
                "correct version."
            )

        self.hf_model_variant = hf_model_variant

        model_id = f"ZJU-AI4H/Hulu-Med-{hf_model_variant}"

        model_kwargs = dict(
            trust_remote_code=True,
            torch_dtype="bfloat16",
            device_map="auto",
            attn_implementation="flash_attention_2",
        )

        self.model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

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
        messages = [{"role": "system", "content": system}]
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
                        "content": [{"type": "image"}, {"type": "text", "text": user_text}],
                    }
                else:
                    user = {"role": "user", "content": [{"type": "text", "text": user_text}]}
                messages.append(user)
            elif turn["role"] == "model":
                messages.append({"role": "assistant", "content": [{"type": "text", "text": turn["text"]}]})

        if verbose:
            print(f"\nQuestion: {conversation.conversation[-1]['question']}")

        response = self.generate(messages, conversation.conversation[1]["image"])

        if verbose:
            print(f"Response: {response}")

        return response

    def generate(self, messages, ecg_image, **kwargs):
        input_text = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
        )

        inputs = self.processor(
            images=[ecg_image], text=input_text, add_special_tokens=False, return_tensors="pt"
        ).to(self.model.device, dtype=torch.bfloat16)

        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=300,
                do_sample=False,
            )

        response = self.processor.decode(output[0], skip_special_tokens=True).strip(".")

        return response

    @classmethod
    def build_model(cls, hf_model_variant="32B-Instruct", **kwargs):
        return cls(hf_model_variant=hf_model_variant)

    def load_state_dict(self, **kwargs):
        raise ValueError(
            "MedGemma HF model does not support loading state dicts directly as it loads "
            "pretrained weights in the constructor."
        )
