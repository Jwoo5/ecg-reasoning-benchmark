import re
import logging
import torch

from .LLaVA.llava.model.builder import load_pretrained_model
from .LLaVA.llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path
from .LLaVA.llava.conversation import conv_templates
from .LLaVA.llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER
)

from .. import BaseModel
from .. import register_model

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
            torch_dtype=torch_dtype
        )
    
    @classmethod
    def build_model(cls, device_map="auto", torch_dtype=torch.float16, **kwargs):
        return cls(device_map=device_map, torch_dtype=torch_dtype)

    def generate(self, prompt, ecg_signal, ecg_image):
        images_tensor = process_images(
            [ecg_image], self.image_processor, self.model.config
        ).to(self.model.device, dtype=torch.float16)

        qs = prompt
        image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN

        if IMAGE_PLACEHOLDER in qs:
            if self.model.config.mm_use_im_start_end:
                qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
            else:
                qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
        else:
            if self.model.config.mm_use_im_start_end:
                qs = image_token_se + "\n" + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

        conv_mode = "llava_v1"

        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt_formatted = conv.get_prompt()

        input_ids = tokenizer_image_token(
            prompt_formatted, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).to(self.model.device)

        with torch.inference_mode():
            output = self.model.generate(
                input_ids,
                images=images_tensor,
                image_sizes=[ecg_image.size],
                do_sample=False, 
                temperature=0.0, 
                max_new_tokens=300,
                use_cache=True,
            )

        return self.tokenizer.batch_decode(output, skip_special_tokens=True)[0].strip()

    def load_state_dict(self, **kwargs):
        raise ValueError(
            "PULSE model does not support loading state dicts directly as it loads "
            "pretrained weights in the constructor."
        )