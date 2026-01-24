import getpass
import os

from openai import (
    APIConnectionError,
    BadRequestError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from .. import BaseModel, register_model


@register_model("gpt")
class GPTModel(BaseModel):
    def __init__(self, model_variant: str = "5-mini", gpt_api_key: str = None):
        self.model_variant = model_variant
        self.gpt_api_key = gpt_api_key
        if model_variant.startswith("gpt-"):
            model_variant = model_variant[len("gpt-") :]

        self.model_id = f"gpt-{self.model_variant}"

        self.api_key = gpt_api_key
        if not self.api_key:
            if "GPT_API_KEY" in os.environ:
                self.api_key = os.environ["GPT_API_KEY"]
            else:
                self.api_key = getpass.getpass(
                    prompt="Enter your GPT API key (you can also set it via the GPT_API_KEY environment variable): "
                )

        # XXX to be moved to model-specific arguments
        self.enable_reasoning = True

        try:
            self.model = OpenAI(api_key=self.api_key)
        except ImportError as e:
            raise RuntimeError(f"Failed to initialize OpenAI client: {e}")

    @classmethod
    def build_model(cls, model_variant="5-mini", gpt_api_key=None, **kwargs):
        return cls(model_variant=model_variant, gpt_api_key=gpt_api_key)

    @retry(
        retry=retry_if_exception_type(
            (RateLimitError, APIConnectionError, InternalServerError, BadRequestError)
        ),
        wait=wait_exponential(multiplier=2, min=4, max=60),
        stop=stop_after_attempt(10),
    )
    def _call_openai_api(self, contents):
        if self.enable_reasoning and "gpt-5" in self.model_id:
            # NOTE turning on reasoning makes it impossible to control the temperature
            model_kwargs = {"reasoning_effort": "medium"}
        else:
            model_kwargs = {
                "reasoning_effort": "none",
                "temperature": 0,
                "frequency_penalty": 0,
                "presence_penalty": 0,
            }

        return self.model.chat.completions.create(
            model=self.model_id,
            messages=contents,
            service_tier="flex",
            **model_kwargs,
        )

    def get_response(
        self, conversation, enable_condensed_chat: bool = False, verbose: bool = False, **kwargs
    ) -> str:
        assert (
            conversation.conversation[0]["role"] == "system"
        ), "The first turn in the conversation must be from the system."
        assert (
            conversation.conversation[-1]["role"] == "user"
        ), "The last turn in the conversation must be from the user."
        assert (
            "image" in conversation.conversation[1]
        ), "The conversation must contain an ECG image in the first user turn."

        contents = []

        system = conversation.conversation[0]["text"]
        contents.append({"role": "system", "content": system})

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
                    user_text += (
                        "Your response must be **ONLY** the full text of the selected option. Do not "
                    )
                    user_text += "include any uncertainty, explanation, reasoning, or extra words."

                if i == 0:
                    user = {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{turn['image']}",
                                    "detail": "high",
                                },
                            },
                            {"type": "text", "text": user_text},
                        ],
                    }
                else:
                    user = {"role": "user", "content": [{"type": "text", "text": user_text}]}
                contents.append(user)
            elif turn["role"] == "model":
                contents.append({"role": "assistant", "content": [{"type": "text", "text": turn["text"]}]})

        if verbose:
            print(f"\nQuestion: {conversation.conversation[-1]['question']}")

        try:
            response = self._call_openai_api(contents)
            response = response.choices[0].message.content.strip()
        except Exception as e:
            raise RuntimeError(f"Failed to get response: {e}")

        if verbose:
            print(f"Response: {response}")

        return response

    def generate(self, **kwargs):
        raise NotImplementedError("Use get_response method for GPTModel.")

    def require_base64_image(self) -> bool:
        """Indicate if the model requires ECG images in base64 format."""
        return True