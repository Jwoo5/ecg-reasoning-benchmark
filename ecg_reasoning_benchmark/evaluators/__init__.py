import importlib
import logging
import os

from .evaluator import Evaluator

logger = logging.getLogger(__name__)

EVALUATOR_REGISTRY = {}

__all__ = [
    "Evaluator",
]


def get_evaluator_cls(evaluator_name: str) -> Evaluator:
    evaluator = None

    if evaluator_name in EVALUATOR_REGISTRY:
        evaluator = EVALUATOR_REGISTRY[evaluator_name]
    assert evaluator is not None, (
        f"Could not infer evaluator from evaluator name '{evaluator_name}'. "
        f"Available evaluators are: {str(EVALUATOR_REGISTRY.keys())}"
    )
    return evaluator


def register_evaluator(name):
    """
    New evaluators can be added with the :func:`register_evaluator` function decorator.

    For example::

        @register_evaluator("my_evaluator")
        class MyEvaluator(Evaluator):
            (...)

    .. note:: All evaluators *must* implement the :class:`Evaluator` interface.

    Args:
        name (str): the name of the evaluator
    """

    def register_evaluator_cls(cls):
        if name in EVALUATOR_REGISTRY:
            raise ValueError(f"Cannot register duplicate evaluator ({name})")
        if not issubclass(cls, Evaluator):
            raise ValueError(f"Evaluator ({name}: {cls.__name__}) must extend Evaluator")
        EVALUATOR_REGISTRY[name] = cls
        return cls

    return register_evaluator_cls


def import_evaluators(evaluators_dir, namespace):
    for file in os.listdir(evaluators_dir):
        path = os.path.join(evaluators_dir, file)
        if (
            not file.startswith("_")
            and not file.startswith(".")
            and (file.endswith(".py") or os.path.isdir(path))
        ):
            evaluator_name = file[: file.find(".py")] if file.endswith(".py") else file
            try:
                importlib.import_module(f"{namespace}.{evaluator_name}")
            except (ImportError, ModuleNotFoundError) as e:
                # Skip evaluators whose dependencies are not installed, but surface
                # a warning so users can see which evaluators were omitted and why.
                # This diagnoses silent registration failures (e.g., Gemini-based
                # evaluators missing because `google-genai` is not installed).
                logger.warning(f"Skipped evaluator '{evaluator_name}': {e}")


# automatically import any Python files in the evaluators/ directory
evaluators_dir = os.path.dirname(__file__)
import_evaluators(evaluators_dir, __name__)
