class BaseModel:
    ecg_modality_base = "image"

    @classmethod
    def build_model(cls, **kwargs):
        """Build a new model instance."""
        raise NotImplementedError("Model Must implement the build_model method.")

    def get_response(self, conversation, enable_condensed_chat: bool = False, verbose: bool = False, **kwargs) -> str:
        """Generate a response based on the conversation history."""
        raise NotImplementedError("Model Must implement the get_response method.")

    def require_base64_image(self) -> bool:
        """Indicate if the model requires ECG images in base64 format."""
        return False