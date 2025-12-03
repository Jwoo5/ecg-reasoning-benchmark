import argparse
import getpass
import hashlib
import json
import os
import pickle
from typing import List, Union

from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

try:
    from google import genai
    from google.genai import errors, types
except ImportError as e:
    raise ImportError(
        "Google Gemini SDK is not installed. Please install it with `pip install google-genai`."
    ) from e

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


@register_evaluator("gemini")
class GeminiEvaluator(Evaluator):
    @staticmethod
    def parse_arguments(args) -> argparse.Namespace:
        parser = Evaluator.add_default_arguments()

        parser.add_argument("--api-key", type=str, default=None, help="Google Gemini API key")
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
            default=100,
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

        if not self.api_key:
            if "GOOGLE_API_KEY" in os.environ:
                self.api_key = os.environ["GOOGLE_API_KEY"]
            else:
                print("API Key not provided in arguments.")
                self.api_key = getpass.getpass("Please enter your Google Gemini API Key: ")

        # initialize the Gemini client
        try:
            self.client = genai.Client(api_key=self.api_key)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Gemini Client: {e}")

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

    @retry(
        retry=retry_if_exception(is_retryable_error),
        wait=wait_exponential(multiplier=2, min=2, max=60),
        stop=stop_after_attempt(10),
    )
    def _generate_with_retry(self, prompt_text):
        return self.client.models.generate_content(
            model=self.model_name,
            contents=prompt_text,
            config=types.GenerateContentConfig(
                temperature=0.0,
            ),
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
        if question_type == "lead_grounding":
            assert isinstance(gt, list), "Ground truth must be a list for lead_grounding question type."
            gt = ", ".join(gt)

        prompt_text = self._get_evaluation_prompt(question, gt, model_response)

        try:
            if self.estimate_cost:
                # use count_tokens method to get input token usage
                token_info = self.client.models.count_tokens(
                    model=self.model_name,
                    contents=prompt_text,
                )
                return token_info.total_tokens

            if self.use_cache and prompt_text in self.cache:
                return self.cache[prompt_text]

            # generate content with deterministic configuration
            result = self._generate_with_retry(prompt_text)

            result_text = result.text.strip()
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
                    self.cache[prompt_text] = is_aligned
                    if self.save_cache and len(self.cache) % self.save_cache_interval == 0:
                        with open(os.path.join(self.cache_dir, self.cache_file), "wb") as f:
                            pickle.dump(self.cache, f)
                else:
                    # remove the the oldest entry from the cache
                    self.cache.pop(next(iter(self.cache)))
                    self.cache[prompt_text] = is_aligned

            return is_aligned

        except Exception as e:
            raise RuntimeError(f"Failed to generate evaluation with Gemini: {e}")
