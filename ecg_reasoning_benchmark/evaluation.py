import argparse
import glob
import json
import os

import pandas as pd
from tqdm import tqdm

from .evaluators import get_evaluator_cls


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


def _reduce_to_row(model_name: str, reduced_metrics: dict) -> dict:
    """Convert reduced metrics into a flat dict suitable for a DataFrame row."""
    row = {"model": model_name}

    # initial diagnostic question
    idq_metric = reduced_metrics["initial_diagnostic_question"]
    row["idq_total"] = idq_metric["total"]
    row["idq_correct"] = idq_metric["correct"]
    row["idq_accuracy"] = idq_metric["accuracy"]

    # gt-reasoning-based diagnosis
    gtd_metric = reduced_metrics["gt_reasoning_based_diagnosis"]
    row["gt_reasoning_based_diagnosis_total"] = gtd_metric["total"]
    row["gt_reasoning_based_diagnosis_correct"] = gtd_metric["correct"]
    row["gt_reasoning_based_diagnosis_accuracy"] = gtd_metric["accuracy"]

    # reasoning
    row["reasoning_total"] = reduced_metrics["reasoning"]["total"]
    for key in reduced_metrics["reasoning"]:
        if key in ["total", "per_loop"]:
            continue
        row[key] = reduced_metrics["reasoning"][key]

    # per-loop metrics
    per_loop = reduced_metrics["reasoning"]["per_loop"]
    row["per_loop_total"] = per_loop["total"]
    row["per_loop_depth_total"] = per_loop["depth_total"]
    row["per_loop_depth_sum"] = per_loop["depth_sum"]
    row["depth"] = per_loop["depth_avg"]
    for step in per_loop:
        if step in ["total", "depth_total", "depth_sum", "depth_avg"]:
            continue
        for key in per_loop[step]:
            row[f"{step}_{key}"] = per_loop[step][key]

    return row


def evaluate_results(
    results_dir: str,
    model: str,
    dataset: str,
    evaluator,
) -> dict:
    """Evaluate inference results from disk and return metrics.

    Args:
        results_dir: Root directory containing ``<model>/<dataset>/<dx>/*.json``.
        model: Model name (subdirectory under *results_dir*).
        dataset: Dataset name (subdirectory under model dir).
        evaluator: An instantiated :class:`Evaluator` subclass.

    Returns:
        Dict mapping metric names to lists of row dicts (one per model).
    """
    rows = {}
    data_dir = os.path.join(results_dir, model, dataset)
    assert os.path.exists(data_dir), f"Data directory {data_dir} does not exist."

    evaluator.init_metrics("total", reset=True)
    with tqdm() as pbar:
        for dx in os.listdir(data_dir):
            dx_dir = os.path.join(data_dir, dx)
            if not os.path.isdir(dx_dir):
                continue
            evaluator.init_metrics(dx, reset=True)
            for fname in glob.glob(os.path.join(dx_dir, "*.json")):
                pbar.set_description(
                    f"Evaluating {model} on {dataset} - {dx} | {os.path.basename(fname)}"
                )
                pbar.update(1)
                try:
                    with open(fname, "r") as f:
                        result = json.load(f)
                except json.decoder.JSONDecodeError:
                    continue
                evaluator.evaluate(result)

        pbar.set_description(f"Evaluating {model} on {dataset} - Done")

    for name in evaluator.metrics.keys():
        reduced_metrics = evaluator.reduce_metrics(name)
        if name not in rows:
            rows[name] = []
        rows[name].append(_reduce_to_row(model, reduced_metrics))

    return rows


def main():
    parser = get_parser()
    args, remaining_args = parser.parse_known_args()

    evaluator_cls = get_evaluator_cls(args.evaluator)
    evaluator_args = evaluator_cls.parse_arguments(remaining_args)
    evaluator = evaluator_cls(evaluator_args)
    output_dir = os.path.join(args.save_dir, evaluator.name)

    is_dry_run = getattr(evaluator.args, "estimate_cost", False)

    total_input_tokens = 0
    all_rows = {}
    for model in args.model:
        for dataset in args.dataset:
            if dataset not in all_rows:
                all_rows[dataset] = {}

            if is_dry_run:
                # dry-run mode: count tokens only
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
                            try:
                                with open(fname, "r") as f:
                                    result = json.load(f)
                            except json.decoder.JSONDecodeError:
                                continue
                            total_input_tokens += evaluator.evaluate(result)
                    pbar.set_description(f"Evaluating {model} on {dataset} - Done")
                print(f"[Dry Run] Total input tokens for {model} on {dataset}: {total_input_tokens}")
                continue

            rows = evaluate_results(args.root, model, dataset, evaluator)
            for name, row_list in rows.items():
                if name not in all_rows[dataset]:
                    all_rows[dataset][name] = []
                all_rows[dataset][name].extend(row_list)

    # save calculated metrics to csv files
    if not is_dry_run:
        for dataset in args.dataset:
            for name in all_rows.get(dataset, {}):
                df = pd.DataFrame(all_rows[dataset][name])
                ds_dir = os.path.join(output_dir, dataset)
                if not os.path.exists(ds_dir):
                    os.makedirs(ds_dir)
                df.to_csv(os.path.join(ds_dir, f"{name}.csv"), index=False)


def cli_main():
    """CLI entry point for ``ecg-reasoning-benchmark-evaluate``."""
    main()


if __name__ == "__main__":
    cli_main()
