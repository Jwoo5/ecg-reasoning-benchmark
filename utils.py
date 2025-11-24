import base64
import io
from PIL import Image
from constants import prompt
from google import genai

class MultiTurnGenerator():
    def __init__(self, model_name):
        self.model_name = model_name
        self.conversation = []
    
    def init_chat(self, ecg, ecg_image, question, answer):
        if "gpt" in self.model_name:
            base64_image = base64_image_encoder(ecg_image)
            self.conversation.append({"role": "user", "content": [{ "type": "input_text", "text": f"{prompt}, {question}, {ecg}"},
                        {
                            "type": "input_image",
                            "image_url": f"data:image/png;base64,{base64_image}"
                        }]
            })
            self.conversation.append({"role" : "assistant", "content" : f"{answer}"})

        if "gemini" in self.model_name:
            base64_image = base64_image_encoder(ecg_image)
            self.conversation.append({"role" : "user", "parts": [
                {"text" : f"{prompt}, {question}, {ecg}"}, 
                genai.types.Part.from_bytes(
                    data=base64_image,
                    mime_type='image/png',
                ),
                ]
            })
            self.conversation.append({"role": "model", "parts":[{"text" : f"{answer}"}]})

    def add_chat(self, question, option, answer_idx):

        if "gpt" in self.model_name:
            self.conversation.append({"role": "user", "content" : f"question : {question}, option : {option}"})
            self.conversation.append({"role": "assistant", "content" : option[answer_idx]})

        if "gemini" in self.model_name:
            self.conversation.append({"role": "user", "parts" : [{"text" : f"question : {question}, option : {option}"}]})
            self.conversation.append({"role": "model", "parts" : [{"text" : f"{option[answer_idx]}"}]})


def base64_image_encoder(image: Image.Image):
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    img_bytes = buf.getvalue()
    return base64.b64encode(img_bytes).decode('utf-8')