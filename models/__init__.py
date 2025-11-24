import importlib
import os

import torch

from .model import BaseModel

MODEL_REGISTRY = {}

__all__ = [
    "BaseModel",
]


def build_model(
    model_name: str, from_checkpoint: bool = False, checkpoint_path: str = None, **kwargs
) -> BaseModel:
    model = None

    if model_name in MODEL_REGISTRY:
        model = MODEL_REGISTRY[model_name]

    assert model is not None, (
        f"Could not infer model from model name '{model_name}'. "
        f"Available models are: {str(MODEL_REGISTRY.keys())}"
    )

    model_instance = model.build_model(**kwargs)
    if checkpoint_path is not None and from_checkpoint:
        with open(checkpoint_path, "rb") as f:
            state = torch.load(f, map_location=torch.device("cpu"))
        model_instance.load_state_dict(state, strict=True)

    return model_instance


def get_model_name(model: BaseModel) -> str:
    for name, cls in MODEL_REGISTRY.items():
        if isinstance(model, cls):
            return name
    raise ValueError("Model class not found in registry.")


def register_model(name):
    """
    New models can be added with the :func:`register_model` function decorator.

    For example::

        @register_model("my_model")
        class MyModel(BaseModel):
            (...)

    .. note:: All models *must* implement the :class:`BaseModel` interface.

    Args:
        name (str): the name of the model
    """

    def register_model_cls(cls):
        if name in MODEL_REGISTRY:
            raise ValueError(f"Cannot register duplicate model ({name})")
        if not issubclass(cls, BaseModel):
            raise ValueError(f"Model ({name}: {cls.__name__}) must extend BaseModel")
        MODEL_REGISTRY[name] = cls
        return cls

    return register_model_cls


def import_models(models_dir, namespace):
    for file in os.listdir(models_dir):
        path = os.path.join(models_dir, file)
        if (
            not file.startswith("_")
            and not file.startswith(".")
            and (file.endswith(".py") or os.path.isdir(path))
        ):
            model_name = file[: file.find(".py")] if file.endswith(".py") else file
            importlib.import_module(f"{namespace}.{model_name}")


# automatically import any Python files in the models/ directory
models_dir = os.path.dirname(__file__)
import_models(models_dir, "models")
