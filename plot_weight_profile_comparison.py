#!/usr/bin/env python3
"""
Plot old vs new detector comparison metrics from compare_weight_profiles.py.

Usage:
    python plot_weight_profile_comparison.py --results-dir results
    python plot_weight_profile_comparison.py --summary-csv results/compare_weight_profiles_summary.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def load_summary(results_dir: Path | None, summary_csv: Path | None) -> tuple[pd.DataFrame, Path]:
    if summary_csv is not None:
        df = pd.read_csv(summary_csv)
        out_dir = summary_csv.parent
        return df, out_dir

    if results_dir is None:
        raise ValueError("Either --results-dir or --summary-csv is required")

    summary_path = results_dir / "compare_weight_profiles_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(
            f"Summary CSV not found at {summary_path}. Run compare_weight_profiles.py first."
        )
    return pd.read_csv(summary_path), results_dir


def plot_grouped_bars(ax, labels, old_vals, new_vals, title, ylabel, better="higher"):
    x = np.arange(len(labels))
    width = 0.36
    ax.bar(x - width / 2, old_vals, width, label="old", color="#9aa5b1")
    ax.bar(x + width / 2, new_vals, width, label="new", color="#d65a31")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    if better == "lower":
        ax.invert_yaxis()


def main() -> int:
    ap = argparse.ArgumentParser(description="Plot old/new detector comparison metrics")
    ap.add_argument("--results-dir", default=None, help="Directory containing compare_weight_profiles_summary.csv")
    ap.add_argument("--summary-csv", default=None, help="Explicit path to compare_weight_profiles_summary.csv")
    args = ap.parse_args()

    results_dir = Path(args.results_dir).resolve() if args.results_dir else None
    summary_csv = Path(args.summary_csv).resolve() if args.summary_csv else None
    df, out_dir = load_summary(results_dir, summary_csv)
    import matplotlib.pyplot as plt

    labels = df["folder"].astype(str).tolist()

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
        "axes.labelsize": 11,
        "font.size": 10,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })

    # fig1, axes1 = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    # plot_grouped_bars(axes1[0], labels, df["acc_old"]*100, df["acc_new"]*100, "Accuracy", "Accuracy (%)")
    # plot_grouped_bars(axes1[1], labels, df["f1_old"], df["f1_new"], "F1 Score", "F1")
    # plot_grouped_bars(axes1[2], labels, df["gap_old"], df["gap_new"], "Score Gap", "avg_fake_score - max_legit_score")
    # axes1[0].legend(loc="best")

    fig1, axes1 = plt.subplots(1, 1, figsize=(14, 6), constrained_layout=True)
    plot_grouped_bars(axes1, labels, df["recall_old"]*100, df["recall_new"]*100, "Recall (Detection Prob)", "Recall (%)")
    axes1.legend(loc="best")

    fig1_path = out_dir / "weight_profile_detection_metrics.png"
    fig1.savefig(fig1_path, dpi=180, bbox_inches="tight")

    # fig2, axes2 = plt.subplots(1, 2, figsize=(10.5, 4.5), constrained_layout=True)
    # plot_grouped_bars(axes2[0], labels, df["mean_err_old"], df["mean_err_new"], "Mean UE Error", "meters")
    # plot_grouped_bars(axes2[1], labels, df["p90_err_old"], df["p90_err_new"], "P90 UE Error", "meters")
    # axes2[0].legend(loc="best")

    fig2, axes2 = plt.subplots(1, 1, figsize=(14, 6), constrained_layout=True)
    plot_grouped_bars(axes2, labels, df["mean_err_old"], df["mean_err_new"], "Mean UE Error", "meters")
    axes2.legend(loc="best")

    fig2_path = out_dir / "weight_profile_localization_errors.png"
    fig2.savefig(fig2_path, dpi=180, bbox_inches="tight")

    # fig3, axes3 = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    # x = np.arange(len(labels))
    # axes3[0].bar(x, df["delta_acc"]*100, color="#1f7a8c")
    # axes3[0].axhline(0.0, color="black", linewidth=1.0)
    # axes3[0].set_title("Delta Accuracy")
    # axes3[0].set_ylabel("new - old (%)")
    # axes3[0].set_xticks(x)
    # axes3[0].set_xticklabels(labels, rotation=20, ha="right")
    # axes3[0].grid(axis="y", alpha=0.25)
    #
    # axes3[1].bar(x, df["delta_f1"]*100, color="#1f7a8c")
    # axes3[1].axhline(0.0, color="black", linewidth=1.0)
    # axes3[1].set_title("Delta F1")
    # axes3[1].set_ylabel("new - old (%)")
    # axes3[1].set_xticks(x)
    # axes3[1].set_xticklabels(labels, rotation=20, ha="right")
    # axes3[1].grid(axis="y", alpha=0.25)
    #
    # axes3[2].bar(x, df["delta_gap"], color="#1f7a8c")
    # axes3[2].axhline(0.0, color="black", linewidth=1.0)
    # axes3[2].set_title("Delta Score Gap")
    # axes3[2].set_ylabel("new - old")
    # axes3[2].set_xticks(x)
    # axes3[2].set_xticklabels(labels, rotation=20, ha="right")
    # axes3[2].grid(axis="y", alpha=0.25)

    fig3, axes3 = plt.subplots(1, 1, figsize=(14, 6), constrained_layout=True)
    x = np.arange(len(labels))
    
    df["delta_recall"] = df["recall_new"] - df["recall_old"]
    axes3.bar(x, df["delta_recall"]*100, color="#1f7a8c")
    axes3.axhline(0.0, color="black", linewidth=1.0)
    axes3.set_title("Delta Recall")
    axes3.set_ylabel("new - old (%)")
    axes3.set_xticks(x)
    axes3.set_xticklabels(labels, rotation=20, ha="right")
    axes3.grid(axis="y", alpha=0.25)

    fig3_path = out_dir / "weight_profile_detection_deltas.png"
    fig3.savefig(fig3_path, dpi=180, bbox_inches="tight")

    print(f"Saved {fig1_path}")
    print(f"Saved {fig2_path}")
    print(f"Saved {fig3_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
