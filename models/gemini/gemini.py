import getpass
import os

from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from .. import BaseModel, register_model

try:
    from google import genai
    from google.genai import errors, types
except ImportError:
    pass


def is_retryable_error(exception):
    """Determines if an exception is 429 (Resource Exhausted) or 5xx (Server Error)."""
    if isinstance(exception, errors.ClientError):
        if exception.code == 429 or "RESOURCE_EXHAUSTED" in str(exception):
            print(f"\n Quota exceeded (429). Retrying... Error: {exception}")
            return True
    elif isinstance(exception, errors.ServerError):
        if exception.code and exception.code >= 500:
            print(f"\n Server error ({exception.code}). Retrying... Error: {exception}")
            return True

    return False


@register_model("gemini")
class GeminiModel(BaseModel):
    def __init__(
        self,
        model_variant: str = "2.5-flash",
        thinking_budget: int = 256,
        gemini_api_key: str = None,
    ):
        try:
            from google import genai
        except ImportError as e:
            raise ImportError(
                "Google Gemini SDK is not installed. Please install it with `pip install google-genai`."
            ) from e

        if model_variant.startswith("gemini-"):
            model_variant = model_variant[len("gemini-") :]

        self.model_variant = model_variant
        self.thinking_budget = thinking_budget
        self.model_id = f"gemini-{self.model_variant}"
        # only enable thinking mode for pro models
        if self.model_id not in ["gemini-2.5-pro", "gemini-3-pro", "gemini-3-pro-preview"]:
            self.thinking_budget = 0

        self.api_key = gemini_api_key
        if not self.api_key:
            if "GOOGLE_API_KEY" in os.environ:
                self.api_key = os.environ["GOOGLE_API_KEY"]
            else:
                print("Gemini API Key not provided.")
                self.api_key = getpass.getpass(
                    prompt=(
                        "Enter your Google Gemini API key (you can also set it via the "
                        "GOOGLE_API_KEY environment variable): "
                    )
                )

        try:
            self.model = genai.Client(api_key=self.api_key)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Gemini client: {e}")

    @classmethod
    def build_model(
        cls, model_variant: str = "3-flash", thinking_budget: int = 256, gemini_api_key: str = None, **kwargs
    ):
        return cls(
            model_variant=model_variant, thinking_budget=thinking_budget, gemini_api_key=gemini_api_key
        )

    @retry(
        retry=retry_if_exception(is_retryable_error),
        wait=wait_exponential(multiplier=2, min=2, max=60),
        stop=stop_after_attempt(10),
    )
    def _call_gemini_api(self, contents, config):
        return self.model.models.generate_content(
            model=self.model_id,
            contents=contents,
            config=config,
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

        system = conversation.conversation[0]["text"]

        contents = []
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
                    parts = [types.Part.from_bytes(data=turn["image"], mime_type="image/png")]
                    parts.append(types.Part(text=user_text))
                else:
                    parts = [types.Part(text=user_text)]
                contents.append(types.Content(role="user", parts=parts))
            elif turn["role"] == "model":
                parts = [types.Part(text=turn["text"])]
                contents.append(types.Content(role="model", parts=parts))

        config = types.GenerateContentConfig(
            system_instruction=system,
            thinking_config=types.ThinkingConfig(thinking_budget=self.thinking_budget),
            temperature=0.0,
        )

        if verbose:
            print(f"\nQuestion: {conversation.conversation[-1]['question']}")

        try:
            response = self._call_gemini_api(contents, config)
            # it seems a bug that response.text can be None, which seems something to do with:
            # https://discuss.ai.google.dev/t/gemini-2-5-pro-with-empty-response-text/81175
            # as of now, we just re-try until we get a non-empty response
            retry_count = 0
            while response.text is None:
                response = self._call_gemini_api(contents, config)
                retry_count += 1
            if retry_count > 0:
                print(f"\n**Note: had to retry {retry_count} times to get a non-empty response.**\n")
            response = response.text.strip()
        except Exception as e:
            raise RuntimeError(f"Failed to get response: {e}")

        if verbose:
            print(f"Response: {response}")

        return response

    def generate(self, **kwargs):
        raise NotImplementedError("Use get_response method for GeminiModel.")

    def require_base64_image(self) -> bool:
        """Indicate if the model requires ECG images in base64 format."""
        return True