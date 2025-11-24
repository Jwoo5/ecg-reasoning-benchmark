import re
import logging
import torch

from .llava.model.builder import load_pretrained_model
from .llava.mm_utils import process_images, tokenizer_image_token
from .llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN

from .. import BaseModel
from .. import register_model

logger = logging.getLogger(__name__)

@register_model("gem")
class GEMLlavaModel(BaseModel):
    def __init__(self, device_map="auto", torch_dtype=torch.float16):
        self.tokenizer, self.model, self.image_processor, _ = load_pretrained_model(
            "LANSG/GEM",
            model_base=None,
            model_name="llava_llama",
            device_map=device_map,
            torch_dtype=torch_dtype
        )

    @classmethod
    def build_model(cls, device_map="auto", torch_dtype=torch.float16, **kwargs):
        return cls(device_map=device_map, torch_dtype=torch_dtype)

    def generate(self, prompt, ecg_signal, ecg_image, **kwargs):
        image_tensor = process_images(
            [ecg_image], self.image_processor, self.model.config
        ).to(self.model.device, dtype=torch.float16)

        full_prompt = DEFAULT_IMAGE_TOKEN + "\n" + prompt
        input_ids = tokenizer_image_token(
            full_prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).to(self.model.device)

        ecg_tensor = ecg_signal.unsqueeze(0).to(self.model.device, dtype=torch.float16)

        with torch.inference_mode():
            output = self.model.generate(
                inputs=input_ids, images=image_tensor, ecgs=ecg_tensor, max_new_tokens=300
            )
        
        return self.tokenizer.decode(output[0, input_ids.shape[0]:], skip_special_tokens=True).strip()

    def load_state_dict(self, **kwargs):
        raise ValueError(
            "GEM model does not support loading state dicts directly as it loads "
            "pretrained weights in the constructor."
        )