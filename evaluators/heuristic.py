import argparse
import re
from typing import List, Union

from . import Evaluator, register_evaluator


@register_evaluator("heuristic")
class HeuristicEvaluator(Evaluator):
    @staticmethod
    def parse_arguments(args) -> argparse.Namespace:
        parser = Evaluator.add_default_arguments()
        return parser.parse_args(args)

    def __init__(self, args: argparse.Namespace):
        super().__init__(args)

        self.name = "heuristic"

    def validate(
        self, gt: Union[str, List[str]], model_response: str, question_type: str, **kwargs
    ) -> bool:
        """Validate the model response using heuristic rules.

        Args:
            gt (Union[str, List[str]]): The ground truth answer(s).
            model_response (str): The model's response to validate.
            question_type (str): The type of question being evaluated.

        Returns:
            bool: True if the model response is considered correct, False otherwise.
        """
        callback = getattr(self, f"_validate_{question_type}", None)
        assert callback is not None, f"No validation function found for question type: {question_type}"

        return callback(gt, model_response)

    def _validate_initial_diagnostic_question(self, gt: str, response: str) -> bool:
        gt = gt.strip().lower()
        response = response.strip().strip(".").lower()
        if (
            response == "yes"
            or response.startswith("yes")
            or response.startswith("**yes**")
            or response.endswith("yes")
            or response.endswith("**yes**")
        ):
            response = "yes"
        elif (
            response == "no"
            or response.startswith("no")
            or response.startswith("**no**")
            or response.endswith("no")
            or response.endswith("**no**")
        ):
            response = "no"
        else:
            print(f"Unable to parse model response: {response}")
            return False

        return gt == response

    def _validate_criterion_selection(self, gt: str, response: str) -> bool:
        gt = gt.strip().lower()
        response = response.strip("-").strip().lower()
        if gt == response:
            return True
        elif response in gt:
            if response == "":
                return False
            elif response == "a":
                return False
            elif "presence" in gt and response == gt.replace("presence of ", ""):
                return True

            # breakpoint()

        elif gt in response:
            # case by case handling as they seem hard to be parsed generally
            # just include known failure cases here
            if (
                gt == "prolongation of the pr interval"
                and response == "progressive prolongation of the pr interval"
            ):
                return False
            elif (
                (
                    gt == "presence of st-segment elevation in lateral leads"
                    and response
                    == (
                        "the correct diagnostic criterion for lateral myocardial infarction is the "
                        "presence of st-segment elevation in lateral leads"
                    )
                )
                or (
                    gt == "presence of st-segment depression in lateral leads"
                    and response
                    == (
                        "the correct diagnostic criterion for lateral ischemia is the presence of "
                        "st-segment depression in lateral leads"
                    )
                )
                or (
                    gt == "presence of left axis deviation"
                    and response
                    == (
                        "the correct diagnostic criterion for left anterior fascicular block is "
                        "the presence of left axis deviation"
                    )
                )
                or (
                    gt == "presence of premature beats"
                    and response
                    == (
                        "the correct diagnostic criterion for premature atrial complexes is the "
                        "presence of premature beats"
                    )
                )
                or (
                    gt == "presence of premature beats"
                    and response
                    == (
                        "the correct diagnostic criteria for premature atrial complexes are the "
                        "presence of premature beats"
                    )
                )
                or (
                    gt == "presence of right bundle branch block"
                    and response
                    == (
                        "the presence of right bundle branch block should be evaluated first to "
                        "determine which set of diagnostic criteria for right ventricular "
                        "hypertrophy should be applied"
                    )
                )
                or (
                    gt == "presence of right bundle branch block"
                    and response
                    == (
                        "the correct answer is: to determine which set of diagnostic criteria "
                        "for right ventricular hypertrophy should be applied, the presence of "
                        "right bundle branch block (rbbb) must be evaluated first"
                    )
                )
                or (
                    gt == "presence of right bundle branch block"
                    and response
                    == (
                        "the correct answer is: to determine which set of diagnostic criteria "
                        "for right ventricular hypertrophy should be applied, the presence of "
                        "right bundle branch block must be evaluated first"
                    )
                )
                or (
                    gt == "presence of right bundle branch block"
                    and response == "the correct answer is: presence of right bundle branch block"
                )
                or (
                    gt == "sum of s amplitude in v1 and r amplitude in v5/v6 > 3.5mv"
                    and response == "the sum of s amplitude in v1 and r amplitude in v5/v6 > 3.5mv"
                )
                or (
                    gt == "sum of s amplitude in v1 and r amplitude in v5/v6 > 3.5mv"
                    and response
                    == (
                        "the next voltage criterion to evaluate is the sum of s amplitude in v1 "
                        "and r amplitude in v5/v6 > 3.5mv"
                    )
                )
                or (
                    gt == "r wave amplitude in lead avl > 1.1mv"
                    and response == "the correct answer is: r wave amplitude in lead avl > 1.1mv"
                )
                or (
                    gt == "sum of s amplitude in v1 and r amplitude in v5/v6 > 3.5mv"
                    and response
                    == (
                        "the sum of s amplitude in v1 and r amplitude in v5/v6 > 3.5mv should "
                        "also be evaluated to diagnose left ventricular hypertrophy"
                    )
                )
                or (
                    gt == "regularity of the rr intervals"
                    and response
                    == (
                        "the correct diagnostic criterion for third degree av block is the "
                        "presence of regularity of the rr intervals"
                    )
                )
                or (
                    gt == "presence of st-segment elevation in anterior leads"
                    and response
                    == (
                        "the correct diagnostic criterion for anterior myocardial infarction "
                        "is the presence of st-segment elevation in anterior leads"
                    )
                )
                or (
                    gt == "presence of an rsr' pattern in leads v1 and v2"
                    and response
                    == (
                        "the correct diagnostic criterion for complete right bundle branch "
                        "block is the presence of an rsr' pattern in leads v1 and v2"
                    )
                )
                or (
                    gt == "presence of st-segment elevation in inferior leads"
                    and response
                    == (
                        "the correct diagnostic criterion for inferior myocardial infarction "
                        "is the presence of st-segment elevation in inferior leads"
                    )
                )
                or (
                    gt == "presence of st-segment elevation in inferior leads"
                    and response
                    == (
                        "the correct diagnostic criteria for inferior myocardial infarction "
                        "include the presence of st-segment elevation in inferior leads"
                    )
                )
                or (
                    gt == "presence of st-segment elevation in inferior leads"
                    and response
                    == (
                        "the correct diagnostic criteria for inferior myocardial infarction "
                        "are the presence of st-segment elevation in inferior leads"
                    )
                )
                or (
                    gt == "presence of st-segment depression in anterior leads"
                    and response
                    == (
                        "the correct diagnostic criterion for anterior ischemia is the presence "
                        "of st-segment depression in anterior leads"
                    )
                )
                or (
                    gt == "presence of premature beats"
                    and response
                    == (
                        "the correct diagnostic criterion for premature ventricular complexes is "
                        "the presence of premature beats"
                    )
                )
                or (
                    gt == "presence of premature beats"
                    and response
                    == (
                        "the correct diagnostic criteria for premature ventricular complexes are "
                        "the presence of premature beats"
                    )
                )
                or (
                    gt == "presence of st-segment depression in inferior leads"
                    and response
                    == (
                        "the correct diagnostic criterion for inferior ischemia is the presence of "
                        "st-segment depression in inferior leads"
                    )
                )
                or (
                    gt == "presence of inverted t waves in inferior leads"
                    and response
                    == (
                        "presence of inverted t waves in inferior leads, along with the finding "
                        "you just identified, should be evaluated to diagnose inferior ischemia"
                    )
                )
                or (
                    gt == "presence of right axis deviation"
                    and response
                    == (
                        "the correct diagnostic criterion for left posterior fascicular block is "
                        "the presence of right axis deviation"
                    )
                )
            ):
                return True
            else:
                # breakpoint()
                return False
        else:
            return False

    def _validate_finding(self, gt: str, response: str) -> bool:
        gt = gt.strip().lower()
        response = response.strip().strip(".").lower()
        if (
            response == "yes"
            or response.startswith("yes")
            or response.startswith("**yes**")
            or response.endswith("yes")
            or response.endswith("**yes**")
        ):
            response = "yes"
        elif (
            response == "no"
            or response.startswith("no")
            or response.startswith("**no**")
            or response.endswith("no")
            or response.endswith("**no**")
        ):
            response = "no"
        else:
            return False

        return gt == response

    def _validate_lead_grounding(self, gt: List[str], response: str) -> bool:
        gt = set([lead.strip().lower() for lead in gt])

        response = response.lower()
        if response.startswith("ads ") or response.startswith("ad "):
            # prepend "le" to "ads" to make "leads"
            response = "le" + response
        # strip "-", ":", etc from response until no more can be stripped
        while True:
            new_response = response.strip("-").strip(":").strip()
            if new_response == response:
                break
            response = new_response

        lead_names_pattern = r"\b(?:iii|ii|i|avr|avl|avf|v[1-6])\b"
        separator = r"(?:\s*,\s*(?:and\s+)?|\s+and\s+)"
        full_phrase_pattern = rf"(?i)\bleads?\s+{lead_names_pattern}(?:{separator}{lead_names_pattern})*"

        matches = list(re.finditer(full_phrase_pattern, response))
        if len(matches) > 0:
            response = set()
            for match in matches:
                full_phrase = match.group()
                extracted_leads = re.findall(lead_names_pattern, full_phrase)
                cleaned_leads = [lead for lead in extracted_leads]
                response.update(cleaned_leads)
        else:
            response = response.replace("and", ",").replace("leads", "lead")
            if "," in response:
                response = set(
                    [lead.strip("-").strip() for lead in response.split(",") if lead.strip() != ""]
                )
            elif ";" in response:
                response = set(
                    [lead.strip("-").strip() for lead in response.split(";") if lead.strip() != ""]
                )
            elif "/" in response:
                response = set(
                    [lead.strip("-").strip() for lead in response.split("/") if lead.strip() != ""]
                )
            elif "\n" in response:
                response = set(
                    [lead.strip("-").strip() for lead in response.split("\n") if lead.strip() != ""]
                )
            else:
                response = set([response])

        refined_response = []
        for x in response:
            if x in ["i", "ii", "iii", "avr", "avl", "avf", "v1", "v2", "v3", "v4", "v5", "v6"]:
                x = "lead " + x
            refined_response.append(x)
        response = set(refined_response)

        if set(gt) == set(response):
            return True
        elif set(gt).issubset(set(response)):
            # TODO partial credit?
            return False
        elif set(response).issubset(set(gt)):
            # TODO partial credit?
            return False
        else:
            return False

    def _validate_wave_grounding(self, gt: str, response: str) -> bool:
        # gt is expected to be a list of one element for wave grounding question type
        gt = gt[0].strip().lower()
        response = response.strip().lower()

        if gt == response:
            return True
        else:
            # NOTE the model response can contain additional information. for example, the model
            # might be formatting the answer as "the correct answer is: <gt>", ...
            # However, for now we will be strict and only accept exact matches as we prompted
            # the model to only output the exact answer without any additional text.
            # NOTE This can be improved in the future by using gpt or gemini as a judge model to assess
            # the correctness of the response compared to the ground truth.
            return False

    def _validate_measurement_grounding(self, gt: str, response: str) -> bool:
        # gt is expected to be a list of one element for measurement grounding question type
        gt = gt[0].strip().lower()
        response = response.strip().lower()

        if gt == response:
            return True
        else:
            pattern = r"\[?\s*-?\d+\.?\d*\s*[^\d\s]*\s*-\s*-?\d+\.?\d*\s*[^\d\s]*\s*\]?"

            # NOTE the model response can contain additional information. for example, the model
            # might be formatting the answer as "the correct answer is: <gt>", ...
            # However, for now we will be strict and only accept exact matches as we prompted
            # the model to only output the exact answer without any additional text.
            # NOTE This can be improved in the future by using gpt or gemini as a judge model to assess
            # the correctness of the response compared to the ground truth.

            gt_match = re.match(pattern, gt)
            if gt_match:
                response_match = re.match(pattern, response)
                if response_match:
                    extracted_response = response_match.group()
                    if not extracted_response.startswith("["):
                        extracted_response = "[" + extracted_response
                    if not extracted_response.endswith("]"):
                        extracted_response = extracted_response + "]"

                    if gt == extracted_response:
                        return True
            return False

    def _validate_decision(self, gt: str, response: str) -> bool:
        gt = gt.strip().lower()
        response = response.strip().strip(".").lower()
        # refine response for yes/no
        if (
            response == "yes"
            or response.startswith("yes")
            or response.startswith("**yes**")
            or response.endswith("yes")
            or response.endswith("**yes**")
        ):
            response = "yes"
        elif (
            response == "no"
            or response.startswith("no")
            or response.startswith("**no**")
            or response.endswith("no")
            or response.endswith("**no**")
        ):
            response = "no"

        return gt == response
