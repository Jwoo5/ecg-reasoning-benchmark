class BaseModel:
    @classmethod
    def build_model(cls, **kwargs):
        """Build a new model instance."""
        raise NotImplementedError("Model Must implement the build_model method.")

    def get_response(self, conversation) -> str:
        """Generate a response based on the conversation history."""
        raise NotImplementedError("Model Must implement the get_response method.")

    def generate(self, prompt, ecg_signal, ecg_image):
        """Generate a response given the prompt, ECG signal, and ECG image."""
        raise NotImplementedError("Model Must implement the generate method.")