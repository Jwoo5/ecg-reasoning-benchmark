import logging
import re

import torch

from .. import BaseModel, register_model
from .llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
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

    def get_response(self, conversation) -> str:
        prompt = ""

        first_user_turn_idx = 0
        if conversation.conversation[0]["role"] == "system":
            first_user_turn_idx = 1

            prompt += conversation.conversation[0]["text"] + "\n\n"

        assert (
            conversation.conversation[-1]["role"] == "user"
        ), "The last turn in the conversation must be from the user."
        assert (
            "image" in conversation.conversation[first_user_turn_idx]
        ), "The conversation must contain an ECG image in the first user turn."

        if len(conversation.conversation) > first_user_turn_idx + 1:
            prompt += "**Dialogue history is as follows:**\n\n"
            for turn in conversation.conversation[first_user_turn_idx:-1]:
                if turn["role"] == "user":
                    prompt += turn["text"]
                elif turn["role"] == "model":
                    prompt += turn["text"] + "\n\n"
            prompt += "**Dialogue history ends.**\n\n"

        prompt += conversation.conversation[-1]["text"]

        ecg_signal = conversation.conversation[first_user_turn_idx]["signal"]
        ecg_image = conversation.conversation[first_user_turn_idx]["image"]

        response = self.generate(prompt, ecg_signal, ecg_image)

        return response

    def generate(self, prompt, ecg_signal, ecg_image, **kwargs):
        image_tensor = process_images([ecg_image], self.image_processor, self.model.config).to(
            self.model.device, dtype=torch.float16
        )

        full_prompt = DEFAULT_IMAGE_TOKEN + "\n" + prompt
        input_ids = (
            tokenizer_image_token(full_prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
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
