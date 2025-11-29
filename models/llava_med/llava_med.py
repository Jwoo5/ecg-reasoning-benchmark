import logging
import torch

from .. import BaseModel, register_model
from .LLaVA_Med.llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN
)
from .LLaVA_Med.llava.conversation import conv_templates
from .LLaVA_Med.llava.model.builder import load_pretrained_model
from .LLaVA_Med.llava.utils import disable_torch_init
from .LLaVA_Med.llava.mm_utils import (
    tokenizer_image_token,
    get_model_name_from_path,
    process_images
)

@register_model("llava-med")
class LLaVAMedModel(BaseModel):
    def __init__(self, device_map="auto", torch_dtype=torch.float16):
        # Suppress warnings for this model as it outputs unnecessary warnings
        logging.getLogger("transformers.generation.utils").setLevel(logging.ERROR)

        self.model_name = "microsoft/llava-med-v1.5-mistral-7b"
        model_name = get_model_name_from_path(self.model_name)

        disable_torch_init()
        self.tokenizer, self.model, self.image_processor, _ = load_pretrained_model(
            self.model_name,
            model_base=None,
            model_name=model_name,
            device_map=device_map,
        )

        if self.tokenizer.pad_token_id is not None:
            pad_token_id = self.tokenizer.pad_token_id
        else:
            pad_token_id = self.tokenizer.eos_token_id
        
        self.model.config.pad_token_id = pad_token_id

    @classmethod
    def build_model(cls, device_map="auto", torch_dtype=torch.float16, **kwargs):
        return cls(device_map=device_map, torch_dtype=torch_dtype)
    
    def get_response(self, conversation, enable_condensed_chat: bool = False, verbose: bool = False) -> str:
        assert (
            conversation.conversation[0]["role"] == "system"
        ), "The first turn in the conversation must be from the system."
        assert (
            conversation.conversation[-1]["role"] == "user"
        ), "The last turn in the conversation must be from the user."
        assert (
            "image" in conversation.conversation[1]
        ), "The conversation must contain an ECG image in the first user turn."

        if self.model.config.mm_use_im_start_end:
            image_token = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        else:
            image_token = DEFAULT_IMAGE_TOKEN

        # 'mistral_instruct' format does not work well for llava-med-v1.5 due to an unknown reason
        # conv = conv_templates["mistral_instruct"].copy()
        conv = conv_templates["llava_v1"].copy()
        conv.system = conversation.conversation[0]["text"]

        for i, turn in enumerate(conversation.conversation[1:]):
            if turn["role"] == "user":
                user_text = f"Question: {turn['question']}\n\n"

                do_add_options = False
                # do not add options in previous turns to reserve context length
                if enable_condensed_chat:
                    if i == len(conversation.conversation[1:]) - 1:
                        do_add_options = True
                else:
                    do_add_options = True

                if do_add_options:
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
                    user_text = image_token + "\n" + user_text

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
                do_sample=False,
                max_new_tokens=300,
                use_cache=True,
            )

        response = self.tokenizer.batch_decode(output, skip_special_tokens=True)[0].strip().strip(".")
        return response