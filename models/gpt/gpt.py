import logging
from openai import OpenAI
from .. import BaseModel
from .. import register_model
from utils import base64_image_encoder
logger = logging.getLogger(__name__)

@register_model("gemini")
class GeminiModel(BaseModel):
    def __init__(
        self,
        hf_model_variant: str = "5-mini",
        is_thinking: bool = False
    ):
        self.hf_model_variant = hf_model_variant
        self.is_thinking = is_thinking

        self.model_id = f"gpt-{self.hf_model_variant}"
        self.model = OpenAI()


    def get_response(self, conversation):

        content = []

        first_user_turn_idx = 0
        if conversation.conversation[0]["role"] == "system":
            first_user_turn_idx = 1
            system_instruction = conversation.conversation[0]["text"]
        else : 
            system_instruction = "You are an expert cardiologist."
        
        assert (
            conversation.conversation[-1]["role"] == "user"
        ), "The last turn in the conversation must be from the user."
        assert (
            "image" in conversation.conversation[first_user_turn_idx]
        ), "The conversation must contain an ECG image in the first user turn."

        base64_image = base64_image_encoder(conversation.conversation[first_user_turn_idx]["image"])

        for turn in conversation.conversation[first_user_turn_idx:]:
            if turn["role"] == "user":
                content.append[{"role" : "user", "content" : [{"type": "input_text", "text" : turn["text"]}]}]
            elif turn["role"] == "model":
                content.append[{"role" : "assistant", "content" : [{"type": "input_text", "text" : turn["text"]}]}]
                
        content[0]["content"].append({"type" : "input_image", "image_url": f"data:image/png;base64,{base64_image}"})
        response = self.generate(content, system_instruction)
        return response, 
    
    def generate(self, content, system_instruction, **kwargs):
        
        response = self.model.responses.create(
            instructions=system_instruction,
            model=self.model_id,
            input=content,
        )
        return response.output[0].content[0]["text"]


    @classmethod
    def build_model(cls, hf_model_variant="5-mini", is_thinking=False, **kwargs):
        return cls(
            hf_model_variant=hf_model_variant,
            is_thinking=is_thinking
        )
