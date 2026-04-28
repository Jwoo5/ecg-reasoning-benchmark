"""Evaluator for forced-commit inference outputs, using Gemini as the validator.

Reads the JSON results produced by ``inference.py`` in ``--inference-mode forced-commit``
and computes diagnosis accuracy at each reasoning depth. For each sample:

* Validates the IDQ ``model_response`` against ``metadata.dx_label`` (reported separately
  as the "no-reasoning baseline").
* For each reasoning loop, validates the forced-commit ``diagnostic_decision.model_response``
  against the same ``dx_label``, and buckets the data point by ``(loop_idx+1)/total_loops``
  into one of four quartile bins: ``(0, 25], (25, 50], (50, 75], (75, 100]``.

Inherits :meth:`validate` (and its Gemini API + cache machinery) from
:class:`GeminiEvaluator`, overriding only the walk logic and the CSV schema.
"""

from typing import List

from . import register_evaluator
from .gemini import GeminiEvaluator

# Right-inclusive bins covering (0, 100] in percentage units.
_BIN_EDGES: List[tuple] = [(0, 25), (25, 50), (50, 75), (75, 100)]


@register_evaluator("gemini-forced-commit")
class GeminiForcedCommitEvaluator(GeminiEvaluator):
    def __init__(self, args):
        super().__init__(args)
        # Distinguish the cache / save-dir namespace from the default Gemini evaluator.
        # The validation prompt is identical, so cache hits carry over — but saving
        # evaluation CSVs under their own subdir avoids overwriting default-mode runs.
        self.name = f"gemini-forced-commit_{self.model_name}"

    # ------------------------------------------------------------------ metrics
    def init_metrics(self, name: str, reset: bool = False) -> None:
        if not hasattr(self, "metrics") or self.metrics is None:
            self.metrics = {}

        if reset and name in self.metrics:
            self.metrics[name] = None

        self.metrics[name] = {
            "initial_diagnostic_question": {"total": 0, "correct": 0},
            "per_depth": {
                "bins": [
                    {"range": edges, "total": 0, "correct": 0}
                    for edges in _BIN_EDGES
                ],
            },
        }

    def reduce_metrics(self, name: str) -> dict:
        if not hasattr(self, "metrics") or self.metrics is None or name not in self.metrics:
            raise ValueError(
                f"Metrics for '{name}' have not been initialized. "
                f"Call init_metrics('{name}') before reduce_metrics()."
            )

        reduced = {
            "initial_diagnostic_question": dict(self.metrics[name]["initial_diagnostic_question"]),
            "per_depth": {"bins": []},
        }

        idq = reduced["initial_diagnostic_question"]
        idq["accuracy"] = idq["correct"] / idq["total"] if idq["total"] > 0 else 0.0

        for b in self.metrics[name]["per_depth"]["bins"]:
            entry = {
                "range": b["range"],
                "total": b["total"],
                "correct": b["correct"],
                "accuracy": b["correct"] / b["total"] if b["total"] > 0 else 0.0,
            }
            reduced["per_depth"]["bins"].append(entry)

        return reduced

    # ------------------------------------------------------------------- walk
    @staticmethod
    def _bin_for_ratio(ratio_pct: float) -> int:
        """Return bin index for a ratio in percentage units (0, 100]."""
        for i, (lo, hi) in enumerate(_BIN_EDGES):
            if ratio_pct <= hi:
                return i
        return len(_BIN_EDGES) - 1  # guard

    def evaluate(self, result: dict):
        mode = result.get("metadata", {}).get("inference_mode")
        assert mode == "forced-commit", (
            f"gemini-forced-commit evaluator requires results produced with "
            f"--inference-mode forced-commit (metadata.inference_mode='forced-commit'); "
            f"got {mode!r} instead."
        )

        dx = result["metadata"]["target_dx"]
        dx_label_str = "yes" if result["metadata"]["dx_label"] else "no"

        if "total" not in self.metrics:
            raise ValueError(
                "Metrics for 'total' have not been initialized. "
                "Call init_metrics('total') before evaluate()."
            )
        if dx not in self.metrics:
            raise ValueError(
                f"Metrics for '{dx}' have not been initialized. "
                f"Call init_metrics('{dx}') before evaluate() on samples with target_dx='{dx}'."
            )

        total_input_tokens = 0 if self.estimate_cost else None

        # --- IDQ ---
        idq = result["data"]["initial_diagnostic_question"]
        idq_mr = idq.get("model_response")
        if idq_mr is not None:
            outcome = self.validate(
                question=idq["question"],
                gt=dx_label_str,
                model_response=idq_mr,
                question_type="initial_diagnostic_question",
            )
            if self.estimate_cost:
                total_input_tokens += int(outcome)
            else:
                for nm in ("total", dx):
                    self.metrics[nm]["initial_diagnostic_question"]["total"] += 1
                    if outcome:
                        self.metrics[nm]["initial_diagnostic_question"]["correct"] += 1

        # --- per-loop decisions ---
        reasoning = result["data"]["reasoning"]
        n_loops = len(reasoning)
        for loop_idx, loop in enumerate(reasoning):
            dec = loop["diagnostic_decision"]
            mr = dec.get("model_response")
            if mr is None:
                continue
            outcome = self.validate(
                question=dec["question"],
                gt=dx_label_str,
                model_response=mr,
                question_type="diagnostic_decision",
            )
            if self.estimate_cost:
                total_input_tokens += int(outcome)
                continue

            ratio_pct = (loop_idx + 1) / n_loops * 100.0
            bin_idx = self._bin_for_ratio(ratio_pct)
            for nm in ("total", dx):
                b = self.metrics[nm]["per_depth"]["bins"][bin_idx]
                b["total"] += 1
                if outcome:
                    b["correct"] += 1

        if self.estimate_cost:
            return total_input_tokens

    # ----------------------------------------------------------------- CSV out
    def to_csv_row(self, reduced_metrics: dict, model_name: str) -> dict:
        row = {"model": model_name}

        idq = reduced_metrics["initial_diagnostic_question"]
        row["idq_total"] = idq["total"]
        row["idq_correct"] = idq["correct"]
        row["idq_accuracy"] = idq["accuracy"]

        for i, b in enumerate(reduced_metrics["per_depth"]["bins"], start=1):
            lo, hi = b["range"]
            row[f"bin{i}_lo"] = lo
            row[f"bin{i}_hi"] = hi
            row[f"bin{i}_total"] = b["total"]
            row[f"bin{i}_correct"] = b["correct"]
            row[f"bin{i}_accuracy"] = b["accuracy"]

        return row
