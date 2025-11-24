import base64
import io
from typing import Dict, List, Optional, Union

import torch
from PIL import Image

# from google import genai


class Conversation:
    def __init__(self, prompt: Optional[str] = None):
        self.conversation = []
        if prompt:
            self.conversation.append({"role": "system", "text": prompt})

    def add_user_turn(
        self, message: str, ecg_signal: Optional[torch.Tensor] = None, ecg_image: Optional[Image.Image] = None
    ) -> None:
        turn = {"role": "user", "text": message}
        if ecg_signal is not None:
            turn["signal"] = ecg_signal
        if ecg_image:
            turn["image"] = ecg_image
        self.conversation.append(turn)

    def add_model_turn(self, message: str) -> None:
        turn = {"role": "model", "text": message}
        self.conversation.append(turn)

    def add_single_turn(
        self, user_message: str, model_message: str, ecg_image: Optional[Image.Image] = None
    ) -> None:
        self.add_user_turn(user_message, ecg_image)
        self.add_model_turn(model_message)

    def add_multi_turn(self, turns: List[Dict[str, Union[str, Image.Image]]]) -> None:
        for turn in turns:
            self.conversation.append(turn)

    # def init_chat(self, ecg, ecg_image, question, answer):

    #     self.conversation.append({"question" : question, "ecg_image" : ecg_image, "ecg" : ecg, "answer" : answer })
    #     # if "gpt" in self.model_name:
    #     #     base64_image = base64_image_encoder(ecg_image)
    #     #     self.conversation.append({"role": "user", "content": [{ "type": "input_text", "text": f"{prompt}, {question}, {ecg}"},
    #     #                 {
    #     #                     "type": "input_image",
    #     #                     "image_url": f"data:image/png;base64,{base64_image}"
    #     #                 }]
    #     #     })
    #     #     self.conversation.append({"role" : "assistant", "content" : f"{answer}"})

    #     # if "gemini" in self.model_name:
    #     #     base64_image = base64_image_encoder(ecg_image)
    #     #     self.conversation.append({"role" : "user", "parts": [
    #     #         {"text" : f"{prompt}, {question}, {ecg}"},
    #     #         genai.types.Part.from_bytes(
    #     #             data=base64_image,
    #     #             mime_type='image/png',
    #     #         ),
    #     #         ]
    #     #     })
    #     #     self.conversation.append({"role": "model", "parts":[{"text" : f"{answer}"}]})

    # def add_chat(self, question, option, answer_idx):

    #     self.conversation.append({"question" : question, "answer" : option[answer_idx]})
    #     # if "gpt" in self.model_name:
    #     #     self.conversation.append({"role": "user", "content" : f"question : {question}, option : {option}"})
    #     #     self.conversation.append({"role": "assistant", "content" : option[answer_idx]})

    #     # if "gemini" in self.model_name:
    #     #     self.conversation.append({"role": "user", "parts" : [{"text" : f"question : {question}, option : {option}"}]})
    #     #     self.conversation.append({"role": "model", "parts" : [{"text" : f"{option[answer_idx]}"}]})


def base64_image_encoder(image: Image.Image):
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    img_bytes = buf.getvalue()
    return base64.b64encode(img_bytes).decode("utf-8")
