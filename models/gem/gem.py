import logging
import re

import torch

from .. import BaseModel, register_model
from .llava.constants import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    IMAGE_TOKEN_INDEX,
)
from .llava.mm_utils import process_images, tokenizer_image_token
from .llava.model.builder import load_pretrained_model

logger = logging.getLogger(__name__)


@register_model("gem")
class GEMLlavaModel(BaseModel):
    def __init__(self, device_map="auto", torch_dtype=torch.float16):
        self.tokenizer, self.model, self.image_processor, _ = load_pretrained_model(
            "LANSG/GEM",
            model_base=None,
            model_name="llava_llama",
            device_map=device_map,
            torch_dtype=torch_dtype,
        )

    @classmethod
    def build_model(cls, device_map="auto", torch_dtype=torch.float16, **kwargs):
        return cls(device_map=device_map, torch_dtype=torch_dtype)

    def get_prompt(self, conversation) -> str:
        seps = [" ", "</s>"]

        assert (
            conversation.conversation[0]["role"] == "system"
        ), "The first turn in the conversation must be from the system."
        assert (
            conversation.conversation[-1]["role"] == "user"
        ), "The last turn in the conversation must be from the user."
        assert (
            "image" in conversation.conversation[1]
        ), "The conversation must contain an ECG image in the first user turn."

        prompt = conversation.conversation[0]["text"] + seps[0]
        for i, turn in enumerate(conversation.conversation[1:]):
            if turn["role"] == "user":
                if i == 0:
                    prompt += f"USER: {DEFAULT_IMAGE_TOKEN}\n"
                else:
                    prompt += f"USER: "
                prompt += f"{turn['question']} "
                prompt += "Choose from the following options:\n"
                for option in turn["options"]:
                    prompt += f"- {option}\n"
                prompt += "\n" + seps[i % 2]
            elif turn["role"] == "model":
                prompt += f"ASSISTANT: {turn['text']}\n\n" + seps[i % 2]

        prompt += "ASSISTANT: "

        return prompt

    def get_response(self, conversation) -> str:
        prompt = self.get_prompt(conversation)
        ecg_signal = conversation.conversation[1]["signal"]
        ecg_image = conversation.conversation[1]["image"]

        response = self.generate(prompt, ecg_signal, ecg_image)

        return response

    def generate(self, prompt, ecg_signal, ecg_image, **kwargs):
        image_tensor = process_images([ecg_image], self.image_processor, self.model.config).to(
            self.model.device, dtype=torch.float16
        )

        input_ids = (
            tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0)
            .to(self.model.device)
        )

        ecg_tensor = ecg_signal.unsqueeze(0).to(self.model.device, dtype=torch.float16)

        with torch.inference_mode():
            output = self.model.generate(
                inputs=input_ids, images=image_tensor, ecgs=ecg_tensor, max_new_tokens=300
            )

        return self.tokenizer.decode(output[0, input_ids.shape[0] :], skip_special_tokens=True).strip()

    def load_state_dict(self, **kwargs):
        raise ValueError(
            "GEM model does not support loading state dicts directly as it loads "
            "pretrained weights in the constructor."
        )
