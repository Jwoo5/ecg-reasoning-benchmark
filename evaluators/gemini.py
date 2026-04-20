import argparse
import hashlib
import json
import os
import pickle
from typing import List, Union

from openai import (
    OpenAI,
)
from utils import get_cache_dir

from . import Evaluator, register_evaluator

prompt = """You are a board-certified Cardiologist and an expert in ECG interpretation.
Your task is to evaluate whether the [Model Response] is **clinically aligned** with the [Ground Truth].

**[Context]**
- Question: {}
- Ground Truth (GT): {}
- Model Response: {}

**[Evaluation Criteria]**
1. **Clinical Equivalence**: Do not just look for keyword matching. Look for clinical semantic equivalence.
2. **Specific Terminology**: In ECG interpretation, specific terminology distinguishes different pathologies.
    - Example: "Prolongation of PR interval" (1st degree block) is CLINICALLY DIFFERENT from \
"Progressive prolongation of PR interval" (Wenckebach).
    - If the Ground Truth specifies a specific pattern (e.g., "Progressive") and the Model Response \
is generic (e.g., just "Prolongation"), this is NOT aligned.
3. **Contradiction**: If the response implies a different diagnosis, it is **FALSE**.

**[Output Format]**
You must output exactly two lines.
- The first line must be exactly "TRUE" if aligned, or "FALSE" if not aligned.
- The second line should be a short reasoning (one sentence).
"""


@register_evaluator("gemini")
class GeminiEvaluator(Evaluator):
    @staticmethod
    def parse_arguments(args) -> argparse.Namespace:
        parser = Evaluator.add_default_arguments()

        parser.add_argument("--api-key", type=str, help="Google Gemini API key")
        parser.add_argument(
            "--gemini-model", type=str, default="gemini-2.0-flash", help="Gemini model to use"
        )
        parser.add_argument(
            "--estimate-cost",
            action="store_true",
            help=(
                "If set, calculates input tokens without calling the generation API. "
                "Note that caching is not taken into account for cost estimation as "
                "it requires the actual API calls to determine cache hits."
            ),
        )
        parser.add_argument(
            "--use-cache",
            action="store_true",
            help="If set, enables response caching to reduce API calls.",
        )
        parser.add_argument(
            "--cache-size",
            type=int,
            default=-1,
            help=(
                "Maximum number of entries to store in the cache. -1 for unlimited. "
                "If the cache exceeds this size, the oldest entry will be removed. "
                "Only applicable if --use-cache is set."
            ),
        )
        parser.add_argument(
            "--save-cache",
            action="store_true",
            help="If set, saves the response cache to disk.",
        )
        parser.add_argument(
            "--save-cache-interval",
            type=int,
            default=10,
            help="Interval (in number of entries) to save the cache to disk.",
        )
        parser.add_argument(
            "--load-cache",
            action="store_true",
            help="If set, loads the response cache from disk.",
        )

        return parser.parse_args(args)

    def __init__(self, args: argparse.Namespace):
        super().__init__(args)

        self.api_key = args.api_key
        self.model_name = args.gemini_model
        self.estimate_cost = args.estimate_cost
        self.name = self.model_name
        self.client = OpenAI(api_key=self.api_key, base_url="https://openrouter.ai/api/v1")

        self.use_cache = args.use_cache
        self.cache_size = args.cache_size
        self.save_cache = args.save_cache
        self.save_cache_interval = args.save_cache_interval
        self.load_cache = args.load_cache

        if self.use_cache:
            if self.save_cache or self.load_cache:
                self.cache_dir = get_cache_dir()
                key = {
                    "model": "gemini",
                    "model_name": self.model_name,
                }
                serialized_key = json.dumps(key, sort_keys=True, default=str)
                self.cache_file = hashlib.sha256(serialized_key.encode("utf-8")).hexdigest() + ".pkl"

            self.cache = None
            if self.load_cache:
                if os.path.exists(os.path.join(self.cache_dir, self.cache_file)):
                    with open(os.path.join(self.cache_dir, self.cache_file), "rb") as f:
                        self.cache = pickle.load(f)
                    print(
                        f"Loaded cache with {len(self.cache)} entries from {self.cache_file} for "
                        f"evaluator {self.model_name}."
                    )
            if self.cache is None:
                self.cache = {}

    def _get_evaluation_prompt(self, question, gt_answer, model_response):
        """Constructs the system prompt for clinical evaluation."""
        prompt_text = prompt.format(question, gt_answer, model_response)
        return prompt_text

    def _generate_with_retry(self, prompt_text):
        return self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt_text}],
            temperature=0.0,
        )

    def validate(
        self, question: str, gt: Union[str, List[str]], model_response: str, question_type: str, **kwargs
    ) -> Union[bool, int]:
        """Validates the model response against the ground truth using Gemini.
        If `self.estimate_cost` is True, returns the token count instead of performing evaluation.

        Args:
            question (str): The question being evaluated.
            gt (Union[str, List[str]]): The ground truth answer(s).
            model_response (str): The model's response to evaluate.
            question_type (str): The type of question (not used in this evaluator).

        Returns:
            Union[bool, int]: True if the model response is clinically aligned with the ground truth,
                False otherwise, or an integer token count if estimating cost.
        """
        if question_type.endswith("grounding"):
            assert isinstance(gt, list), (
                "Ground truth must be a list for ecg_grounding (lead_grounding, wave_grounding, "
                "measurement_grounding) question type."
            )
            gt = ", ".join(gt)

        prompt_text = self._get_evaluation_prompt(question, gt, model_response)

        try:
            cache_key = (gt, model_response)
            if self.use_cache and cache_key in self.cache:
                return self.cache[cache_key]

            # generate content with deterministic configuration
            result = self._generate_with_retry(prompt_text)

            result_text = result.choices[0].message.content.strip()
            lines = result_text.split("\n")

            decision_line = lines[0].strip().upper()
            # reasoning = lines[1].strip() if len(lines) > 1 else "No reasoning provided."

            is_aligned = False
            if "TRUE" in decision_line:
                is_aligned = True
            elif "FALSE" in decision_line:
                is_aligned = False
            else:
                # fallback check if the model was verbose
                is_aligned = "TRUE" in result_text.upper()
                if not is_aligned:
                    print(f"Warning: Unexpected evaluation output format:\n{result_text}")

            if self.use_cache:
                if self.cache_size == -1 or len(self.cache) < self.cache_size:
                    self.cache[cache_key] = is_aligned
                    if self.save_cache and len(self.cache) % self.save_cache_interval == 0:
                        with open(os.path.join(self.cache_dir, self.cache_file), "wb") as f:
                            pickle.dump(self.cache, f)
                else:
                    # remove the the oldest entry from the cache
                    self.cache.pop(next(iter(self.cache)))
                    self.cache[cache_key] = is_aligned

            return is_aligned

        except Exception as e:
            raise RuntimeError(f"Failed to generate evaluation with Gemini: {e}")
