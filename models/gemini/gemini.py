import logging
from google import genai
from .. import BaseModel
from .. import register_model
from utils import base64_image_encoder
logger = logging.getLogger(__name__)

@register_model("gemini")
class GeminiModel(BaseModel):
    def __init__(
        self,
        hf_model_variant: str = "3-flash",
        is_thinking: bool = False
    ):
        self.hf_model_variant = hf_model_variant
        self.is_thinking = is_thinking

        self.model_id = f"gemini-{self.hf_model_variant}"
        self.model = genai.Client()


    def get_response(self, conversation):
        #1. 들어올때 

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
            content.append[{"role" : turn["role"], "parts" : [{"type": "text", "text" : turn["text"]}]}]


        content[0]["parts"].append(genai.types.Part.from_bytes(data=base64_image, mime_type='image/png',))
        response = self.generate(content, system_instruction)
        return response, 
    
    def generate(self, content, system_instruction, **kwargs):
        
        response = self.model.models.generate_content(
            systemInstruction=system_instruction,
            model=self.model_id,
            contents=content,
             config=genai.types.GenerateContentConfig(
                thinking_config=genai.types.ThinkingConfig(
                    include_thoughts=True
                )
            )
        )
        return response.text


    @classmethod
    def build_model(cls, hf_model_variant="3-flash", is_thinking=False, **kwargs):
        return cls(
            hf_model_variant=hf_model_variant,
            is_thinking=is_thinking
        )
