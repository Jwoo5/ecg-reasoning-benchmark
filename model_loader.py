# model_loader.py
import torch
import sys
import os
import re
import pdb

class BaseModel:
    def generate(self, prompt, ecg_signal, ecg_image):
        raise NotImplementedError

class GEMLlavaModel(BaseModel):
    def __init__(self, device_map="auto", torch_dtype=torch.float16):
        print("Loading GEM Model...")
        gem_lib_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'gem'))
        if gem_lib_path not in sys.path:
            sys.path.insert(0, gem_lib_path)
        
        #import for GEM
        from llava.model.builder import load_pretrained_model
        from llava.mm_utils import process_images, tokenizer_image_token
        from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN

        # 3. Store references
        self.process_images = process_images
        self.tokenizer_image_token = tokenizer_image_token
        
        # Store Constants
        self.IMAGE_TOKEN_INDEX = IMAGE_TOKEN_INDEX
        self.DEFAULT_IMAGE_TOKEN = DEFAULT_IMAGE_TOKEN
    
        self.tokenizer, self.model, self.image_processor, _ = load_pretrained_model(
            "LANSG/GEM", None, "llava_llama", 
            device_map=device_map, torch_dtype=torch_dtype
        )

    def generate(self, prompt, ecg_signal, ecg_image):
        image_tensor = self.process_images([ecg_image], self.image_processor, self.model.config)
        
        full_prompt = self.DEFAULT_IMAGE_TOKEN + "\n" + prompt
        input_ids = self.tokenizer_image_token(full_prompt, self.tokenizer, self.IMAGE_TOKEN_INDEX, return_tensors='pt')
        
        input_ids = input_ids.unsqueeze(0).to(self.model.device)
        image_tensor = image_tensor.to(self.model.device, dtype=torch.float16)
        ecg_tensor = ecg_signal.unsqueeze(0).to(self.model.device, dtype=torch.float16)

        with torch.inference_mode():
            output = self.model.generate(
                inputs=input_ids, images=image_tensor, ecgs=ecg_tensor, max_new_tokens=300
            )
        return self.tokenizer.decode(output[0, input_ids.shape[0]:], skip_special_tokens=True).strip()

class PulseModel(BaseModel):
    def __init__(self, device_map="auto", torch_dtype=torch.float16):
        print("Loading PULSE Model with isolated LLaVA import...")
        
        pulse_lib_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'pulse'))
        
        if pulse_lib_path not in sys.path:
            sys.path.insert(0, pulse_lib_path)

        # 2. Lazy Import
        from llava.model.builder import load_pretrained_model
        from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path
        from llava.conversation import conv_templates
        from llava.constants import (
            IMAGE_TOKEN_INDEX, 
            DEFAULT_IMAGE_TOKEN, 
            DEFAULT_IM_START_TOKEN, 
            DEFAULT_IM_END_TOKEN,
            IMAGE_PLACEHOLDER
        )
        
        # 3. Store references
        self.process_images = process_images
        self.tokenizer_image_token = tokenizer_image_token
        self.conv_templates = conv_templates
        
        # Store Constants
        self.IMAGE_TOKEN_INDEX = IMAGE_TOKEN_INDEX
        self.DEFAULT_IMAGE_TOKEN = DEFAULT_IMAGE_TOKEN
        self.DEFAULT_IM_START_TOKEN = DEFAULT_IM_START_TOKEN
        self.DEFAULT_IM_END_TOKEN = DEFAULT_IM_END_TOKEN
        self.IMAGE_PLACEHOLDER = IMAGE_PLACEHOLDER

        self.model_path = "PULSE-ECG/PULSE-7B" 
        model_name = get_model_name_from_path(self.model_path)

        self.tokenizer, self.model, self.image_processor, _ = load_pretrained_model(
            self.model_path, None, model_name, 
            device_map=device_map, 
            torch_dtype=torch_dtype
        )

    def generate(self, prompt, ecg_signal, ecg_image):
        # 1. Prepare Image Tensor
        images_tensor = self.process_images([ecg_image], self.image_processor, self.model.config).to(self.model.device, dtype=torch.float16)

        # 2. Prepare Prompt with Conversation Template
        qs = prompt
        image_token_se = self.DEFAULT_IM_START_TOKEN + self.DEFAULT_IMAGE_TOKEN + self.DEFAULT_IM_END_TOKEN
        
        if self.IMAGE_PLACEHOLDER in qs:
            if self.model.config.mm_use_im_start_end:
                qs = re.sub(self.IMAGE_PLACEHOLDER, image_token_se, qs)
            else:
                qs = re.sub(self.IMAGE_PLACEHOLDER, self.DEFAULT_IMAGE_TOKEN, qs)
        else:
            if self.model.config.mm_use_im_start_end:
                qs = image_token_se + "\n" + qs
            else:
                qs = self.DEFAULT_IMAGE_TOKEN + "\n" + qs

        conv_mode = "llava_v1"
        
        conv = self.conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt_formatted = conv.get_prompt()

        # 3. Tokenize
        input_ids = (
            self.tokenizer_image_token(prompt_formatted, self.tokenizer, self.IMAGE_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0)
            .to(self.model.device)
        )

        # 4. Generate
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=images_tensor,
                image_sizes=[ecg_image.size],
                do_sample=False, 
                temperature=0.0, 
                max_new_tokens=300,
                use_cache=True,
            )

        return self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

def get_model_loader(model_name):
    name = model_name.lower()
    if "gem" in name and "gemini" not in name: 
        return GEMLlavaModel()
    elif "pulse" in name:
        return PulseModel()

    elif "gpt" in name:
        return None 
    else:
        raise ValueError(f"Model {model_name} not implemented in model_loader.py")
