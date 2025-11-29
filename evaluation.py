import os
import json 
import glob
from typing import Union, List
import pandas as pd
import argparse   
from tqdm import tqdm

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "root",
        type=str,
        metavar="DIR",
        help=(
            "path to the root directory that contains all the json results from inference, "
            "which should have the same structure as the data directory. For example, "
            "given the directory 'results/', the json files are expected to be found in "
            "'results/<model>/<dataset>/<dx>/*.json'."
        )
    )
    parser.add_argument(
        "--dataset",
        type=str,
        nargs="+",
        default=["ptbxl"],
        help="list of dataset names to evaluate.",
    )
    parser.add_argument(
        "--model",
        type=str,
        nargs="+",
        required=True,
        help="list of model names to evaluate.",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        required=True,
        help="directory to save evaluation metrics.",
    )

    return parser

class Evaluator():
    def __init__(self, root_dir: str):
        self.root_dir = root_dir

        self.init_metrics()

    def init_metrics(self) -> None:
        initial_diagnostic_question_metrics = {
            "total": 0,
            "correct": 0,
        }
        stepwise_metrics = {
            "total_w_gt": 0,
            "correct_w_gt": 0,
            "total_wo_gt": 0,
            "correct_wo_gt": 0,
        }
        self.metrics = {
            "initial_diagnostic_question": initial_diagnostic_question_metrics,
            "path_1": {
                "total": 0,
                "completed": 0,
                "aligned": 0,
                "per_loop": {
                    "total": 0,
                    "depth_sum": 0,
                    "criterion_selection": stepwise_metrics.copy(),
                    "finding": stepwise_metrics.copy(),
                    "lead_grounding": stepwise_metrics.copy(),
                    "wave_grounding": stepwise_metrics.copy(),
                    "measurement_grounding": stepwise_metrics.copy(),
                    "decision": stepwise_metrics.copy(),
                }
            },
            "path_2": {
                "total": 0,
                # TODO add more detailed metrics for path 2 when it is implemented
            },
            "failed": {
                "total": 0,
            }
        }

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
            breakpoint()
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
            elif (
                "presence" in gt
                and response == gt.replace("presence of ", "")
            ):
                return True

            breakpoint()

        elif gt in response:
            # case by case handling for ensuring the assessment is correct
            if (
                (gt == "prolongation of the pr interval" and response == "progressive prolongation of the pr interval")
            ):
                return False
            elif (
                (gt == "presence of st-segment elevation in lateral leads" and response == "the correct diagnostic criterion for lateral myocardial infarction is the presence of st-segment elevation in lateral leads")
                or (gt == "presence of st-segment depression in lateral leads" and response == "the correct diagnostic criterion for lateral ischemia is the presence of st-segment depression in lateral leads")
                or (gt == "presence of st-segment depression in lateral leads" and response == "the correct diagnostic criteria for lateral ischemia include the presence of st-segment depression in leads ii, iii, and avf, as well as the presence of st-segment depression in lateral leads")
                or (gt == "presence of left axis deviation" and response == "the correct diagnostic criterion for left anterior fascicular block is the presence of left axis deviation")
                or (gt == "presence of premature beats" and response == "the correct diagnostic criterion for premature atrial complexes is the presence of premature beats")
                or (gt == "presence of premature beats" and response == "the correct diagnostic criteria for premature atrial complexes are the presence of premature beats")
                or (gt == "presence of right bundle branch block" and response == "the presence of right bundle branch block should be evaluated first to determine which set of diagnostic criteria for right ventricular hypertrophy should be applied")
                or (gt == "presence of right bundle branch block" and response == "the correct answer is: to determine which set of diagnostic criteria for right ventricular hypertrophy should be applied, the presence of right bundle branch block (rbbb) must be evaluated first")
                or (gt == "presence of right bundle branch block" and response == "the correct answer is: to determine which set of diagnostic criteria for right ventricular hypertrophy should be applied, the presence of right bundle branch block must be evaluated first")
                or (gt == "presence of right bundle branch block" and response == "the correct answer is: presence of right bundle branch block")
                or (gt == "sum of s amplitude in v1 and r amplitude in v5/v6 > 3.5mv" and response == "the sum of s amplitude in v1 and r amplitude in v5/v6 > 3.5mv")
                or (gt == "sum of s amplitude in v1 and r amplitude in v5/v6 > 3.5mv" and response == "the next voltage criterion to evaluate is the sum of s amplitude in v1 and r amplitude in v5/v6 > 3.5mv")
                or (gt == "r wave amplitude in lead avl > 1.1mv" and response == "the correct answer is: r wave amplitude in lead avl > 1.1mv")
                or (gt == "sum of s amplitude in v1 and r amplitude in v5/v6 > 3.5mv" and response == "the sum of s amplitude in v1 and r amplitude in v5/v6 > 3.5mv should also be evaluated to diagnose left ventricular hypertrophy")
                or (gt == "regularity of the rr intervals" and response == "the correct diagnostic criterion for third degree av block is the presence of regularity of the rr intervals")
                or (gt == "presence of st-segment elevation in anterior leads" and response == "the correct diagnostic criterion for anterior myocardial infarction is the presence of st-segment elevation in anterior leads")
                or (gt == "presence of an rsr' pattern in leads v1 and v2" and response == "the correct diagnostic criterion for complete right bundle branch block is the presence of an rsr' pattern in leads v1 and v2")
                or (gt == "presence of st-segment elevation in inferior leads" and response == "the correct diagnostic criterion for inferior myocardial infarction is the presence of st-segment elevation in inferior leads")
                or (gt == "presence of st-segment elevation in inferior leads" and response == "the correct diagnostic criteria for inferior myocardial infarction include the presence of st-segment elevation in inferior leads")
                or (gt == "presence of st-segment elevation in inferior leads" and response == "the correct diagnostic criteria for inferior myocardial infarction are the presence of st-segment elevation in inferior leads")
                or (gt == "presence of st-segment depression in anterior leads" and response == "the correct diagnostic criterion for anterior ischemia is the presence of st-segment depression in anterior leads")
                or (gt == "presence of premature beats" and response == "the correct diagnostic criterion for premature ventricular complexes is the presence of premature beats")
                or (gt == "presence of premature beats" and response == "the correct diagnostic criteria for premature ventricular complexes are the presence of premature beats")
                or (gt == "presence of st-segment depression in inferior leads" and response == "the correct diagnostic criterion for inferior ischemia is the presence of st-segment depression in inferior leads")
                or (gt == "presence of inverted t waves in inferior leads" and response == "presence of inverted t waves in inferior leads, along with the finding you just identified, should be evaluated to diagnose inferior ischemia")
                or (gt == "presence of right axis deviation" and response == "the correct diagnostic criterion for left posterior fascicular block is the presence of right axis deviation")
            ):
                return True

            breakpoint()

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
            print(f"Unable to parse model response: {response}")
            breakpoint()
            return False
        
        return gt == response

    def validate(
        self, gt: Union[str, List[str]], model_response: str, question_type: str
    ) -> bool:
        callback = getattr(self, f"_validate_{question_type}", None)
        # assert callback is not None, (
        #     f"No validation function found for question type: {question_type}"
        # )
        #XXX
        if callback is None:
            # print(question_type)
            # breakpoint()
            return False

        return callback(gt, model_response)

    def evaluate(self, model: str, dataset: str) -> None:
        data_dir = os.path.join(self.root_dir, model, dataset)
        assert os.path.exists(data_dir), f"Data directory {data_dir} does not exist."

        self.init_metrics()
        with tqdm() as pbar:
            for dx in os.listdir(data_dir):
                for fname in glob.glob(os.path.join(data_dir, dx, "*.json")):
                    pbar.set_description(f"Evaluating {model} on {dataset} - {dx} | {os.path.basename(fname)}")
                    pbar.update(1)
                    
                    #XXX
                    if pbar.n < 2860:
                        continue

                    with open(fname, "r") as f:
                        result = json.load(f)

                    # assert result["metadata"]["model"] == model, (
                    #     f"Model name mismatch: {result['metadata']['model']} vs {model}"
                    # )

                    dx_label = result["metadata"]["dx_label"]
                    dx_gt = "yes" if dx_label else "no"

                    # Evaluate initial diagnostic question
                    initial_diagnostic_question_result = result["data"]["initial_diagnostic_question"]
                    self.metrics["initial_diagnostic_question"]["total"] += 1

                    eval_path = initial_diagnostic_question_result["eval_path"]
                    if eval_path == -1:
                        self.metrics["failed"]["total"] += 1
                        continue
                    elif eval_path == 2:
                        self.metrics["path_2"]["total"] += 1
                        # TODO add more detailed metrics for path 2 when it is implemented
                        continue
                    else:
                        self.metrics["path_1"]["total"] += 1
                        model_response = initial_diagnostic_question_result["model_response"]
                        initial_dx_correct = self.validate(dx_gt, model_response, "initial_diagnostic_question")
                        if initial_dx_correct:
                            self.metrics["initial_diagnostic_question"]["correct"] += 1

                        terminated_early = False
                        # Evaluate stepwise reasoning for path 1
                        for loop in result["data"]["path_1"]:
                            for step_name, step in loop.items():
                                if step_name == "grounding":
                                    for g_step in step:
                                        terminated_early = not self.eval_step(g_step, terminated_early)
                                else:
                                    terminated_early = not self.eval_step(step, terminated_early)

        # breakpoint()

    def eval_step(self, step, terminated_early: bool = False) -> bool:
        step_name = step["question_type"]
        gt = step["answer"]
        model_response = step["model_response"]

        self.metrics["path_1"]["per_loop"][step_name]["total_w_gt"] += 1
        if not terminated_early:
            self.metrics["path_1"]["per_loop"][step_name]["total_wo_gt"] += 1

        if self.validate(gt, model_response, step_name):
            self.metrics["path_1"]["per_loop"][step_name]["correct_w_gt"] += 1
            if not terminated_early:
                self.metrics["path_1"]["per_loop"][step_name]["correct_wo_gt"] += 1
            return True
        else:
            return False

    # def make_csv(self):
    #     paths = [str(path) for path in list(Path(self.dir).rglob("*.json"))]
    #     rows = []
    #     for path in paths:
    #         row = defaultdict(int)
    #         with open(path) as f:
    #             continue_flag = True
    #             result = json.load(f)

    #             row["id"] = os.path.basename(path)
    #             row["data_source"] = result["metadata"]["data_source"]
    #             row["target_dx"] = result["metadata"]["target_dx"]
    #             row["model"] = result["metadata"]["model"]
    #             row["label"] = result["metadata"]["dx_label"]
    #             row["path_idx"] = result["metadata"]["path_idx"]
    #             for question in result["data"]:
    #                 row[f"total_{question['question_type']}s"] += 1
    #                 if self.AnswerValidator(question['answer_str'], question['response_raw']) & continue_flag:
    #                     row[f"total_correct_{question['question_type']}s"] += 1
    #                     row[f"consecutive_correct_{question['question_type']}s"] += 1
    #                 elif self.AnswerValidator(question['answer_str'], question['response_raw']) :
    #                     row[f"total_correct_{question['question_type']}s"] += 1
    #                 else:
    #                     continue_flag = False
            
    #         rows.append(row)
    #     df = pd.DataFrame(rows).fillna(0)
    #     return df

    # def aggregator(self, df, keys):
    #     metric_df = df.groupby(keys).sum()
        
    #     for metric in ['finding', 'wave_grounding', 'lead_grounding', 'measurement_grounding']:
    #         if f'total_{metric}s' in metric_df.columns:
    #             metric_df[f'average_depth_{metric}']= metric_df[f'consecutive_correct_{metric}s'] / metric_df[f'total_{metric}s']
    #             metric_df[f'total_accuracy_{metric}']= metric_df[f'total_correct_{metric}s'] / metric_df[f'total_{metric}s']
    #     return metric_df.fillna(0)

def main(args):
    evaluator = Evaluator(args.root)
    for model in args.model:
        for dataset in args.dataset:
            evaluator.evaluate(model, dataset)

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)