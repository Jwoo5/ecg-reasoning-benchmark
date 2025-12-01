import argparse
import glob
import json
import os

import pandas as pd
from tqdm import tqdm

from evaluators import get_evaluator_cls


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
        ),
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
        "--evaluator",
        type=str,
        default="heuristic",
        help=(
            "name of the evaluator to use for evaluation. default is 'heuristic' evaluator where "
            "strict string matching rules are applied to compute evaluation metrics."
        ),
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        required=True,
        help="directory to save evaluation metrics.",
    )

    return parser


def main():
    parser = get_parser()
    args, remaining_args = parser.parse_known_args()

    output_dir = os.path.join(args.save_dir, args.evaluator)

    evaluator_cls = get_evaluator_cls(args.evaluator)
    evaluator_args = evaluator_cls.parse_arguments(remaining_args)
    evaluator = evaluator_cls(evaluator_args)

    is_dry_run = getattr(evaluator.args, "estimate_cost", False)

    total_input_tokens = 0
    rows = {}
    for model in args.model:
        for dataset in args.dataset:
            if dataset not in rows:
                rows[dataset] = {}
            data_dir = os.path.join(args.root, model, dataset)
            assert os.path.exists(data_dir), f"Data directory {data_dir} does not exist."
            with tqdm() as pbar:
                evaluator.init_metrics("total", reset=True)
                for dx in os.listdir(data_dir):
                    evaluator.init_metrics(dx, reset=True)
                    for fname in glob.glob(os.path.join(data_dir, dx, "*.json")):
                        pbar.set_description(
                            f"Evaluating {model} on {dataset} - {dx} | {os.path.basename(fname)}"
                        )
                        pbar.update(1)

                        with open(fname, "r") as f:
                            result = json.load(f)

                        # XXX to be activated
                        # assert result["metadata"]["model"] == model, (
                        #     f"Model name mismatch: {result['metadata']['model']} vs {model}"
                        # )

                        if is_dry_run:
                            total_input_tokens += evaluator.evaluate(result)
                        else:
                            evaluator.evaluate(result)
                pbar.set_description(f"Evaluating {model} on {dataset} - Done")

            if is_dry_run:
                print(f"[Dry Run] Total input tokens for {model} on {dataset}: {total_input_tokens}")
                continue

            # aggregate metrics into rows to make dataframes
            # NOTE only the built-in metrics are taken accounted here
            for name in evaluator.metrics.keys():
                reduced_metrics = evaluator.reduce_metrics(name)

                row = {"model": model}
                if name not in rows[dataset]:
                    rows[dataset][name] = []

                # save metrics for initial diagnostic question
                idq_metric = reduced_metrics["initial_diagnostic_question"]
                row["idq_total"] = idq_metric["total"]
                row["idq_correct"] = idq_metric["correct"]
                row["idq_accuracy"] = idq_metric["accuracy"]
                row["path_1_total"] = reduced_metrics["path_1"]["total"]
                row["path_1_ratio"] = idq_metric["path_1_ratio"]
                row["path_2_total"] = reduced_metrics["path_2"]["total"]
                row["path_2_ratio"] = idq_metric["path_2_ratio"]
                row["failed_total"] = reduced_metrics["failed"]["total"]
                row["failed_ratio"] = idq_metric["failed_ratio"]

                # save metrics for path 1
                for key in reduced_metrics["path_1"]:
                    if key in ["total", "per_loop"]:
                        continue

                    row[key] = reduced_metrics["path_1"][key]

                # save per-loop metrics for path 1
                row["depth"] = reduced_metrics["path_1"]["per_loop"]["depth_avg"]
                row["per_loop_total"] = reduced_metrics["path_1"]["per_loop"]["total"]
                row["per_loop_depth_total"] = reduced_metrics["path_1"]["per_loop"]["depth_total"]
                row["per_loop_depth_sum"] = reduced_metrics["path_1"]["per_loop"]["depth_sum"]
                for step in reduced_metrics["path_1"]["per_loop"]:
                    if step in ["total", "depth_total", "depth_sum", "depth_avg"]:
                        continue

                    for key in reduced_metrics["path_1"]["per_loop"][step]:
                        row[f"{step}_{key}"] = reduced_metrics["path_1"]["per_loop"][step][key]

                rows[dataset][name].append(row)

    # save calculated metrics to csv files
    if not is_dry_run:
        for dataset in args.dataset:
            for name in evaluator.metrics.keys():
                df = pd.DataFrame(rows[dataset][name])
                if not os.path.exists(os.path.join(output_dir, dataset)):
                    os.makedirs(os.path.join(output_dir, dataset))
                df.to_csv(os.path.join(output_dir, dataset, f"{name}.csv"), index=False)


if __name__ == "__main__":
    main()
