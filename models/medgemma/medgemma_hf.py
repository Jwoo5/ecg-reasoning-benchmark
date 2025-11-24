from transformers import AutoProcessor, AutoModelForImageTextToText
from transformers import BitsAndBytesConfig
import logging
import torch

from .. import BaseModel
from .. import register_model

logger = logging.getLogger(__name__)

@register_model("medgemma_hf")
class MedGemmaHFModel(BaseModel):
    def __init__(
        self,
        model_variant: str = "4b-it",
        use_quantization: bool = True,
        is_thinking: bool = False
    ):
        self.model_variant = model_variant
        self.is_thinking = is_thinking

        model_id = f"google/medgemma-{model_variant}"

        model_kwargs = dict(
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

        if use_quantization:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)

        self.model = AutoModelForImageTextToText.from_pretrained(model_id, **model_kwargs)
        self.processor = AutoProcessor.from_pretrained(model_id)

    def generate(self, prompt, ecg_image, **kwargs):
        if "27b" in self.model_variant and self.is_thinking:
            system_instruction = f"SYSTEM INSTRUCTION: think silently if needed. You are an expert cardiologist."
            max_new_tokens = 1300
        else:
            system_instruction = f"You are an expert cardiologist."
            max_new_tokens = 300
        
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_instruction}]
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image", "image": ecg_image}
                ]
            }
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(self.model.device, dtype=torch.bfloat16)

        input_len = inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            output = output[0][input_len:]

        response = self.processor.decode(output, skip_special_tokens=True).strip()

        return response

    @classmethod
    def build_model(cls, model_variant="4b-it", use_quantization=True, is_thinking=False, **kwargs):
        return cls(
            model_variant=model_variant,
            use_quantization=use_quantization,
            is_thinking=is_thinking
        )

    def load_state_dict(self, **kwargs):
        raise ValueError(
            "MedGemma HF model does not support loading state dicts directly as it loads "
            "pretrained weights in the constructor."
        )