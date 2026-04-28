"""Plot per-depth diagnosis accuracy curves from a forced-commit evaluation.

Reads CSVs produced by :mod:`ecg_reasoning_benchmark.evaluators.gemini_forced_commit`
(one row per model, four quartile bins plus IDQ baseline) and renders the
"finding coverage → diagnosis accuracy" curve.

Two layouts are available:

* ``combined`` (default): all requested models are plotted on a single axes,
  each with its own curve and transparent dashed horizontal line at that
  model's IDQ accuracy.
* ``separate``: subplot grid with one subplot per model, each showing a single
  curve + its IDQ baseline.

Figures are styled for paper inclusion: colorblind-safe palette, distinct
markers per model, embedded TrueType fonts (``pdf.fonttype=42``), y-only
dashed grid, top/right spines removed, and a compact IDQ-baseline legend
entry.

CLI entry point is registered as ``ecg-reasoning-benchmark-forced-commit-plot``.
"""
from __future__ import annotations

import argparse
import math
import os
from typing import List, Optional

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D


# Colorblind-safe qualitative palette (Wong 2011 / seaborn "colorblind"),
# ordered for high pairwise contrast in paper figures.
_PALETTE = [
    "#0173B2",  # blue
    "#DE8F05",  # orange
    "#029E73",  # green
    "#D55E00",  # vermillion
    "#CC78BC",  # purple
    "#CA9161",  # brown
    "#56B4E9",  # sky blue
    "#FBAFE4",  # pink
    "#949494",  # gray
    "#ECE133",  # yellow
]

# Distinct markers paired with colors so lines remain separable in B/W print.
_MARKERS = ["o", "s", "D", "^", "v", "P", "X", "*", "h", "<"]


def _apply_paper_style() -> None:
    """Set matplotlib rcParams for a publication-quality appearance.

    Applied once per ``render()`` call. Safe to call repeatedly.
    """
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "Liberation Sans"],
            "mathtext.fontset": "dejavusans",
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.titleweight": "bold",
            "axes.labelsize": 13,
            "axes.labelweight": "medium",
            "axes.linewidth": 1.1,
            "axes.edgecolor": "#333333",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.width": 1.0,
            "ytick.major.width": 1.0,
            "xtick.major.size": 4.0,
            "ytick.major.size": 4.0,
            "legend.fontsize": 10,
            "legend.title_fontsize": 10,
            "legend.frameon": True,
            "legend.framealpha": 0.95,
            "legend.edgecolor": "#BBBBBB",
            "legend.fancybox": False,
            "legend.borderpad": 0.5,
            "legend.handlelength": 2.2,
            "grid.color": "#CCCCCC",
            "grid.linewidth": 0.7,
            "grid.linestyle": "--",
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            # Embed TrueType fonts so PDFs are compliant with paper submission systems.
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Render the forced-commit diagnosis-accuracy-vs-reasoning-depth curve from "
            "a CSV produced by the 'gemini-forced-commit' evaluator."
        )
    )
    parser.add_argument(
        "--eval-dir",
        type=str,
        required=True,
        help=(
            "root directory containing the forced-commit evaluator CSVs. Expected to "
            "contain '<dataset>/<dx>.csv' files (e.g., the --save-dir passed to the "
            "evaluator, followed by the evaluator subdir 'gemini-forced-commit_<model>')."
        ),
    )
    parser.add_argument(
        "--dataset", type=str, required=True, help="dataset subdirectory under --eval-dir"
    )
    parser.add_argument(
        "--dx",
        type=str,
        default="total",
        help="which CSV to read (default: 'total'). Use a specific dx name for per-dx plot.",
    )
    parser.add_argument(
        "--all-dx",
        action="store_true",
        help=(
            "render a separate figure for every CSV under '<eval-dir>/<dataset>/'. "
            "When set, --dx is ignored. If --output is given it is treated as a "
            "directory; otherwise each PDF is written next to its source CSV."
        ),
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=None,
        help="list of models to include in the plot. Default: all rows found in the CSV.",
    )
    parser.add_argument(
        "--layout",
        type=str,
        choices=["combined", "separate"],
        default="combined",
        help=(
            "'combined' plots all models on one axes (default); 'separate' produces a "
            "subplot grid with one subplot per model."
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "path to save the figure. Default: '<eval-dir>/<dataset>/<dx>.pdf' for "
            "combined layout, or '<eval-dir>/<dataset>/<dx>_separate.pdf' for separate."
        ),
    )
    parser.add_argument(
        "--ylim",
        type=float,
        nargs=2,
        default=(0.0, 1.0),
        metavar=("LOW", "HIGH"),
        help="y-axis limits. Default: 0.0 1.0.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="figure title override. Default: '<dataset> — <dx>'.",
    )
    return parser


def _extract_bin_columns(df: pd.DataFrame) -> List[int]:
    """Return sorted list of bin indices (1-based) present as ``bin{i}_accuracy`` columns."""
    idxs = []
    for col in df.columns:
        if col.startswith("bin") and col.endswith("_accuracy"):
            try:
                idxs.append(int(col[3 : -len("_accuracy")]))
            except ValueError:
                continue
    return sorted(idxs)


def _bin_midpoints_and_labels(df: pd.DataFrame, bin_idxs: List[int]) -> tuple:
    """Compute bin midpoints (x positions) and human-readable labels from lo/hi columns."""
    if not bin_idxs:
        raise ValueError("No bin_* columns found in CSV.")
    row = df.iloc[0]  # lo/hi are identical across rows
    mids, labels = [], []
    for i in bin_idxs:
        lo = int(row[f"bin{i}_lo"])
        hi = int(row[f"bin{i}_hi"])
        mids.append((lo + hi) / 2)
        labels.append(f"({lo}, {hi}]")
    return mids, labels


def _plot_one_axes(
    ax,
    df: pd.DataFrame,
    bin_idxs: List[int],
    mids,
    labels,
    color_map: dict,
    marker_map: dict,
    title: str,
    ylim: tuple,
    show_idq_legend_entry: bool = True,
) -> None:
    """Plot every row in ``df`` on the same axes (curve + IDQ dashed line)."""
    for _, row in df.iterrows():
        model = row["model"]
        accs = [row[f"bin{i}_accuracy"] for i in bin_idxs]
        color = color_map[model]
        marker = marker_map[model]
        ax.plot(
            mids,
            accs,
            marker=marker,
            markersize=9,
            markeredgecolor="white",
            markeredgewidth=1.2,
            linewidth=2.4,
            color=color,
            label=model,
            zorder=3,
        )
        idq_acc = row.get("idq_accuracy")
        if idq_acc is not None and not (isinstance(idq_acc, float) and math.isnan(idq_acc)):
            ax.axhline(
                idq_acc,
                color=color,
                linestyle=(0, (5, 3)),
                alpha=0.55,
                linewidth=1.6,
                zorder=2,
            )

    ax.set_xticks(mids)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Finding Coverage (%)")
    ax.set_ylabel("Diagnosis Accuracy")
    ax.set_ylim(*ylim)
    # Soft horizontal padding so markers don't touch the spines.
    pad = (mids[-1] - mids[0]) * 0.08 if len(mids) > 1 else 5.0
    ax.set_xlim(mids[0] - pad, mids[-1] + pad)
    ax.set_title(title, pad=10)
    ax.grid(True, axis="y", alpha=0.7)
    ax.set_axisbelow(True)

    # Build legend: solid colored lines per model + one neutral dashed entry
    # that documents the IDQ-baseline convention.
    handles, labels_ = ax.get_legend_handles_labels()
    if show_idq_legend_entry:
        handles.append(
            Line2D(
                [0],
                [0],
                color="#666666",
                linestyle=(0, (5, 3)),
                linewidth=1.6,
                alpha=0.7,
            )
        )
        labels_.append("IDQ baseline")
    ax.legend(handles, labels_, loc="best")


def render(
    csv_path: str,
    output_path: str,
    layout: str,
    dataset_name: str,
    dx_name: str,
    models: Optional[List[str]] = None,
    ylim: tuple = (0.0, 1.0),
    title: Optional[str] = None,
) -> None:
    _apply_paper_style()

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    if models is not None:
        df = df[df["model"].isin(models)].reset_index(drop=True)
        missing = set(models) - set(df["model"])
        if missing:
            raise ValueError(f"Requested models not found in CSV: {sorted(missing)}")
    if df.empty:
        raise ValueError(f"No rows to plot (CSV={csv_path}, requested models={models}).")

    bin_idxs = _extract_bin_columns(df)
    mids, labels = _bin_midpoints_and_labels(df, bin_idxs)

    # Stable color + marker assignment across layouts.
    models_ordered = df["model"].tolist()
    color_map = {m: _PALETTE[i % len(_PALETTE)] for i, m in enumerate(models_ordered)}
    marker_map = {m: _MARKERS[i % len(_MARKERS)] for i, m in enumerate(models_ordered)}

    base_title = title or f"{dataset_name} — {dx_name}"
    if layout == "combined":
        fig, ax = plt.subplots(figsize=(8.0, 5.2))
        _plot_one_axes(
            ax, df, bin_idxs, mids, labels, color_map, marker_map, base_title, ylim
        )
    else:  # separate
        n_models = len(df)
        cols = min(3, n_models)
        rows = math.ceil(n_models / cols)
        fig, axes = plt.subplots(
            rows,
            cols,
            figsize=(4.8 * cols, 3.8 * rows),
            squeeze=False,
            sharey=True,
        )
        for idx, (_, row) in enumerate(df.iterrows()):
            ax = axes[idx // cols][idx % cols]
            single = df.iloc[[idx]]
            _plot_one_axes(
                ax,
                single,
                bin_idxs,
                mids,
                labels,
                color_map,
                marker_map,
                title=str(row["model"]),
                ylim=ylim,
                show_idq_legend_entry=True,
            )
        for j in range(n_models, rows * cols):
            axes[j // cols][j % cols].axis("off")
        fig.suptitle(base_title, y=1.02, fontsize=15, fontweight="bold")

    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to: {output_path}")


def main(args=None):
    parser = get_parser()
    args = parser.parse_args(args)

    dataset_dir = os.path.join(args.eval_dir, args.dataset)
    suffix = "" if args.layout == "combined" else "_separate"

    if args.all_dx:
        if not os.path.isdir(dataset_dir):
            raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
        csv_files = sorted(f for f in os.listdir(dataset_dir) if f.endswith(".csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found under {dataset_dir}")
        if args.output is not None:
            os.makedirs(args.output, exist_ok=True)
        for csv_name in csv_files:
            dx_name = csv_name[: -len(".csv")]
            csv_path = os.path.join(dataset_dir, csv_name)
            if args.output is not None:
                output_path = os.path.join(args.output, f"{dx_name}{suffix}.pdf")
            else:
                output_path = os.path.join(dataset_dir, f"{dx_name}{suffix}.pdf")
            render(
                csv_path=csv_path,
                output_path=output_path,
                layout=args.layout,
                dataset_name=args.dataset,
                dx_name=dx_name,
                models=args.models,
                ylim=tuple(args.ylim),
                title=args.title,
            )
        return

    csv_path = os.path.join(dataset_dir, f"{args.dx}.csv")

    if args.output is None:
        args.output = os.path.join(dataset_dir, f"{args.dx}{suffix}.pdf")

    render(
        csv_path=csv_path,
        output_path=args.output,
        layout=args.layout,
        dataset_name=args.dataset,
        dx_name=args.dx,
        models=args.models,
        ylim=tuple(args.ylim),
        title=args.title,
    )


def cli_main():
    main()


if __name__ == "__main__":
    cli_main()
