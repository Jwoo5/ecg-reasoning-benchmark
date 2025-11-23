class BaseModel:
    @classmethod
    def build_model(cls, **kwargs):
        """Build a new model instance."""
        raise NotImplementedError("Model Must implement the build_model method.")

    def generate(self, prompt, ecg_signal, ecg_image):
        raise NotImplementedError("Model Must implement the generate method.")
    
    def load_state_dict(self, state_dict, strict=True):
        """Copies parameters from *state_dict* into this module and its descendants."""
        raise NotImplementedError("Model Must implement the load_state_dict method.")