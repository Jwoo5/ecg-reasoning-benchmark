# model_loader.py
import torch
from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from utils import base64_image_encoder
from openai import OpenAI
from google import genai

class BaseModel:
    def generate(self, prompt, ecg_signal, ecg_image):
        raise NotImplementedError

class GEMLlavaModel(BaseModel):
    def __init__(self, device_map="auto", torch_dtype=torch.float16):
        print("Loading GEM Model...")
        self.tokenizer, self.model, self.image_processor, _ = load_pretrained_model(
            "LANSG/GEM", None, "llava_llama", 
            device_map=device_map, torch_dtype=torch_dtype
        )

    def generate(self, prompt, ecg_signal, ecg_image):
        image_tensor = process_images([ecg_image], self.image_processor, self.model.config)
        
        full_prompt = DEFAULT_IMAGE_TOKEN + "\n" + prompt
        input_ids = tokenizer_image_token(full_prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
        
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
        print("Loading PULSE Model...")
        
        self.model_path = "PULSE-ECG/PULSE-7B" 
        model_name = get_model_name_from_path(self.model_path)

        self.tokenizer, self.model, self.image_processor, _ = load_pretrained_model(
            self.model_path, 
            None, 
            model_name, 
            device_map=device_map, 
            torch_dtype=torch_dtype
        )

    def generate(self, prompt, ecg_signal, ecg_image):
        """
        PULSE inference based on the provided reference script.
        Note: PULSE uses the ECG Image, so 'ecg_signal' tensor is ignored.
        """
        
        # 1. Process the Image
        # process_images returns a tensor, we take the first element [0]
        image_tensor = process_images([ecg_image], self.image_processor, self.model.config)[0]
        
        # 2. Construct Prompt with Image Tokens
        if self.model.config.mm_use_im_start_end:
            qs = f"{DEFAULT_IM_START_TOKEN}{DEFAULT_IMAGE_TOKEN}{DEFAULT_IM_END_TOKEN}\n{prompt}"
        else:
            qs = f"{DEFAULT_IMAGE_TOKEN}\n{prompt}"

        # 3. Tokenize
        input_ids = tokenizer_image_token(qs, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(self.model.device)

        # 4. Handle Image Tensor Dimensions & Type
        if len(image_tensor.shape) == 3:
            image_tensor = image_tensor.unsqueeze(0)
        
        image_tensor = image_tensor.to(self.model.device, dtype=torch.float16)

        # 5. Generate
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=[ecg_image.size], 
                do_sample=False,              # Deterministic for QA evaluation
                temperature=0.0,              # Deterministic
                max_new_tokens=300,
                use_cache=True
            )

        # 6. Decode
        return self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

class GPTRetriever(BaseModel):
    def __init__(self, model):
        self.client = OpenAI()
        self.model = model

    def generate(self, prompt, ecg_signal, ecg_image): 
        
        base64_image = base64_image_encoder(ecg_image)
        response = self.client.responses.create(
            model=self.model,
            input=[
                {
                    "role": "user",
                    "content": [
                        { "type": "input_text", "text": f"{prompt}, {ecg_signal}" },
                        {
                            "type": "input_image",
                            "image_url": f"data:image/png;base64,{base64_image}"
                        }
                    ]
                }
            ]
        )
        return response.output[0].content[0]["text"]
    
class GeminiRetriever(BaseModel):
    def __init__(self, model):
        self.client = genai.Client()
        self.model = model

    def generate(self, prompt, ecg_signal, ecg_image): 
        
        base64_image = base64_image_encoder(ecg_image)
        response = self.client.models.generate_content(
            model=self.model,
            contents=[
                genai.types.Part.from_bytes(
                    data=base64_image,
                    mime_type='image/png',
                ),
            f'{prompt}, {ecg_signal}'
            ]
        )
        return response.output[0].content[0]["text"]

def get_model_loader(model_name):
    name = model_name.lower()
    if "gem" in name and "gemini" not in name: 
        return GEMLlavaModel()
    elif "pulse" in name:
        return PulseModel()
    elif "gpt" in name:
        return GPTRetriever(name) #maybe version of LLM might differ
    elif "gemini" in name:
        return GeminiRetriever(name)
    else:
        raise ValueError(f"Model {model_name} not implemented in model_loader.py")

