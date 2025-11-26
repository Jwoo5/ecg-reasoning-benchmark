# AutoModelForImageTextToText is available in transformers >= 4.50.0,
# while other models are available in earlier versions (e.g., gem, pulse)
# so we catch the import error here for backward compatibility.
try:
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
except:
    pass
import logging
import torch

from utils import base64_image_encoder
from .. import BaseModel
from .. import register_model

logger = logging.getLogger(__name__)

@register_model("qwen3-vl-hf")
class Qwen3VLHFModel(BaseModel):
    def __init__(
        self,
        hf_model_variant: str = "32B-Instruct",
    ):
        self.hf_model_variant = hf_model_variant

        model_id = f"Qwen/Qwen3-VL-{hf_model_variant}"

        model_kwargs = dict(
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )

        self.model = Qwen3VLForConditionalGeneration.from_pretrained(model_id, **model_kwargs)
        self.processor = AutoProcessor.from_pretrained(model_id)

    def get_response(self, conversation):
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
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system}]
            }
        ]
        for i, turn in enumerate(conversation.conversation[1:]):
            if turn["role"] == "user":
                user_text = f"{turn['question']} Choose from the following options:\n"
                for option in turn["options"]:
                    user_text += f"- {option}\n"
                if i == 0:
                    base64_image = base64_image_encoder(turn["image"])
                    user = {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": f"data:image/png;base64,{base64_image}"},
                            {"type": "text", "text": user_text}
                        ]
                    }
                else:
                    user = {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": user_text}
                        ]
                    }
                messages.append(user)
            elif turn["role"] == "model":
                messages.append(
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": turn["text"]}
                        ]
                    }
                )
        
        response = self.generate(messages)

        return response

    def generate(self, messages, **kwargs):
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device, dtype=torch.bfloat16)

        input_len = inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=300, do_sample=False)
            output = output[0][input_len:]

        response = self.processor.decode(output, skip_special_tokens=True).strip()

        return response

    @classmethod
    def build_model(cls, hf_model_variant="32B-Instruct", **kwargs):
        return cls(hf_model_variant=hf_model_variant)

    def load_state_dict(self, **kwargs):
        raise ValueError(
            "MedGemma HF model does not support loading state dicts directly as it loads "
            "pretrained weights in the constructor."
        )