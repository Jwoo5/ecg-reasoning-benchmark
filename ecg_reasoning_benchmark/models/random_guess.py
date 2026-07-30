"""Uniform random-guessing baseline.

A content-blind control that ignores the ECG and the question entirely and picks
uniformly at random among the offered options at every step. It exists to
establish the chance-level floor for every metric in the benchmark: any real
model's score is only meaningful relative to this baseline.

Because it never looks at the signal, it declares ``ecg_modality_base = "text"``
so :class:`Inferencer` skips ECG signal loading and image rasterization (only a
wfdb path string is constructed, with no file I/O). No GPU, no API, and no
PhysioNet ECG files are required to run it.

Guessing policy (matches the option cardinality of each step):
  * single-answer steps (IDQ, criterion_selection, finding_identification,
    wave/measurement grounding, diagnostic_decision): pick one option uniformly
    at random (1/N).
  * the only multi-answer step -- lead grounding ("select all possible leads")
    -- includes each option independently with probability 1/2 (uniform over the
    power set) and joins the selection with ", ". The empty selection is allowed
    (it simply scores as incorrect), matching a true uniform-over-subsets draw.

Runs are made reproducible via a seed: pass ``--seed`` (or encode it in
``--model-variant`` as ``run<k>``/``seed<k>``). Distinct seeds across runs give
the run-to-run variation used for confidence-interval estimation.
"""
import random
from typing import List, Optional

from . import BaseModel, register_model


@register_model("random-guess")
class RandomGuessModel(BaseModel):
    # Ignore the ECG: "text" modality makes Inferencer skip signal loading and
    # image rendering (only a wfdb path string is built, so no file I/O).
    ecg_modality_base = "text"

    def __init__(self, seed: int = 0, model_variant: Optional[str] = None):
        # Only set model_variant when provided so get_model_name() reports plain
        # "random-guess" for a variant-less run (instead of "random-guess-None").
        if model_variant is not None:
            self.model_variant = model_variant
        self.rng = random.Random(seed)

    @classmethod
    def build_model(cls, model_variant: Optional[str] = None, seed: Optional[int] = None, **kwargs):
        # Seed precedence: explicit --seed, else derived from a "run<k>"/"seed<k>"
        # variant string, else 0.
        if seed is None:
            seed = cls._seed_from_variant(model_variant)
        return cls(seed=int(seed), model_variant=model_variant)

    @staticmethod
    def _seed_from_variant(model_variant: Optional[str]) -> int:
        if model_variant is None:
            return 0
        digits = "".join(ch for ch in str(model_variant) if ch.isdigit())
        return int(digits) if digits else 0

    def _current_turn(self, conversation) -> dict:
        turn = conversation.conversation[-1]
        assert turn.get("role") == "user", "Last turn must be a user turn carrying the options."
        return turn

    def get_response(
        self, conversation, enable_condensed_chat: bool = False, verbose: bool = False, **kwargs
    ) -> str:
        turn = self._current_turn(conversation)
        options: List[str] = turn["options"]
        question: str = turn.get("question", "")

        # Only lead grounding ("select all possible leads") is multi-answer; every
        # other step takes exactly one option. Mirrors the detection used by the
        # natural-language models for this benchmark.
        if "select all" in question.lower():
            chosen = [opt for opt in options if self.rng.random() < 0.5]
            # Empty selection is a legitimate uniform-over-subsets outcome; it is
            # emitted as an empty string and scored as incorrect downstream.
            return ", ".join(chosen)

        return self.rng.choice(options)
