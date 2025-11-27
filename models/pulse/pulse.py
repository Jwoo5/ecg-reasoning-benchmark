import logging
import re

import torch

from .. import BaseModel, register_model
from .LLaVA.llava.constants import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    IMAGE_TOKEN_INDEX,
)
from .LLaVA.llava.conversation import conv_templates
from .LLaVA.llava.mm_utils import (
    get_model_name_from_path,
    process_images,
    tokenizer_image_token,
)
from .LLaVA.llava.model.builder import load_pretrained_model

logger = logging.getLogger(__name__)


@register_model("pulse")
class PulseModel(BaseModel):
    def __init__(self, device_map="auto", torch_dtype=torch.float16):
        self.model_cfg_name = "PULSE-ECG/PULSE-7B"
        model_name = get_model_name_from_path(self.model_cfg_name)

        self.tokenizer, self.model, self.image_processor, _ = load_pretrained_model(
            self.model_cfg_name,
            model_base=None,
            model_name=model_name,
            device_map=device_map,
            torch_dtype=torch_dtype,
        )

    @classmethod
    def build_model(cls, device_map="auto", torch_dtype=torch.float16, **kwargs):
        return cls(device_map=device_map, torch_dtype=torch_dtype)

    def get_response(self, conversation, verbose: bool = False) -> str:
        assert (
            conversation.conversation[0]["role"] == "system"
        ), "The first turn in the conversation must be from the system."
        assert (
            conversation.conversation[-1]["role"] == "user"
        ), "The last turn in the conversation must be from the user."
        assert (
            "image" in conversation.conversation[1]
        ), "The conversation must contain an ECG image in the first user turn."


        conv = conv_templates["llava_v1"].copy()
        conv.system = conversation.conversation[0]["text"]

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
                user_text += "include any uncertainty, explanation, reasoning, or extra words.\n\n"

                if i == 0:
                    user_text = DEFAULT_IMAGE_TOKEN + "\n" + user_text
                
                conv.append_message(conv.roles[0], user_text)
            elif turn["role"] == "model":
                conv.append_message(conv.roles[1], turn["text"])
        conv.append_message(conv.roles[1], None)

        prompt = conv.get_prompt()

        ecg_image = conversation.conversation[1]["image"]

        if verbose:
            print(f"\nQuestion: {conversation.conversation[-1]['question']}")

        response = self.generate(prompt, ecg_image)

        if verbose:
            print(f"Response: {response}")

        return response

    def generate(self, prompt: str, ecg_image, **kwargs):
        images_tensor = process_images([ecg_image], self.image_processor, self.model.config).to(
            self.model.device, dtype=torch.float16
        )

        input_ids = (
            tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0)
            .to(self.model.device)
        )

        with torch.inference_mode():
            output = self.model.generate(
                input_ids,
                images=images_tensor,
                image_sizes=[ecg_image.size],
                do_sample=False,
                max_new_tokens=300,
            )

        return self.tokenizer.batch_decode(output, skip_special_tokens=True)[0].strip()

    def load_state_dict(self, **kwargs):
        raise ValueError(
            "PULSE model does not support loading state dicts directly as it loads "
            "pretrained weights in the constructor."
        )
