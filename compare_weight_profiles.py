#!/usr/bin/env python3
"""
Compare old vs new sliding-window detector weighting profiles.

For each results folder, this script:
1. Runs sliding_window_detector.py with --weight-profile old
2. Saves its outputs with a .old.csv suffix
3. Runs sliding_window_detector.py with --weight-profile new
4. Saves its outputs with a .new.csv suffix
5. Prints a compact metric table comparing classification and UE error

Usage:
    python compare_weight_profiles.py --results-dir results/fbs-medium
    python compare_weight_profiles.py --results-dir results
"""

from __future__ import annotations

import argparse
import math
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd


OUTPUT_BASENAMES = [
    "detection_results.csv",
    "ue_position_errors.csv",
    "window_summary.csv",
]


def discover_result_folders(path: Path) -> list[Path]:
    if (path / "measurements.csv").exists() and (path / "cell_database.csv").exists():
        return [path]

    folders = []
    for child in sorted(path.iterdir()):
        if child.is_dir() and (child / "measurements.csv").exists() and (child / "cell_database.csv").exists():
            folders.append(child)
    return folders


def run_profile(project_root: Path, folder: Path, profile: str) -> None:
    cmd = [
        sys.executable,
        "detector/sliding_window_detector.py",
        "--results-dir",
        str(folder.relative_to(project_root)),
        "--weight-profile",
        profile,
        "--log-level",
        "WARNING",
    ]
    result = subprocess.run(cmd, cwd=project_root, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"{folder.name} profile={profile} failed with exit={result.returncode}\n"
            f"{result.stderr.strip()}"
        )

    for basename in OUTPUT_BASENAMES:
        src = folder / basename
        if src.exists():
            dst = folder / basename.replace(".csv", f".{profile}.csv")
            shutil.copy2(src, dst)


def classification_metrics(det_df: pd.DataFrame) -> dict[str, float]:
    if det_df.empty:
        return {k: math.nan for k in [
            "accuracy", "precision", "recall", "f1",
            "avg_fake_score", "max_legit_score", "score_gap",
        ]}

    y_true = det_df["is_fake_gt"].astype(int)
    y_pred = (det_df["status"] == "FAKE").astype(int)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    total = max(len(det_df), 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = (2.0 * precision * recall) / max(precision + recall, 1e-12)
    accuracy = (tp + tn) / total

    fake_scores = det_df.loc[det_df["is_fake_gt"] == 1, "score"]
    legit_scores = det_df.loc[det_df["is_fake_gt"] == 0, "score"]
    avg_fake_score = float(fake_scores.mean()) if not fake_scores.empty else math.nan
    max_legit_score = float(legit_scores.max()) if not legit_scores.empty else math.nan
    score_gap = avg_fake_score - max_legit_score if not math.isnan(avg_fake_score) and not math.isnan(max_legit_score) else math.nan

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "avg_fake_score": avg_fake_score,
        "max_legit_score": max_legit_score,
        "score_gap": score_gap,
    }


def error_metrics(err_df: pd.DataFrame) -> dict[str, float]:
    if err_df.empty or "error_m" not in err_df.columns:
        return {k: math.nan for k in ["mean_err", "median_err", "p90_err", "max_err"]}

    err = err_df["error_m"].astype(float)
    return {
        "mean_err": float(err.mean()),
        "median_err": float(err.median()),
        "p90_err": float(err.quantile(0.9)),
        "max_err": float(err.max()),
    }


def load_metrics(folder: Path, profile: str) -> dict[str, float]:
    det_path = folder / f"detection_results.{profile}.csv"
    err_path = folder / f"ue_position_errors.{profile}.csv"

    det_df = pd.read_csv(det_path) if det_path.exists() else pd.DataFrame()
    err_df = pd.read_csv(err_path) if err_path.exists() else pd.DataFrame()

    metrics = {}
    metrics.update(classification_metrics(det_df))
    metrics.update(error_metrics(err_df))
    return metrics


def fmt(value: float) -> str:
    return "nan" if pd.isna(value) else f"{value:.4f}"


def main() -> int:
    ap = argparse.ArgumentParser(description="Compare old/new detector weighting profiles")
    ap.add_argument("--results-dir", default="results", help="One results folder or a parent directory")
    args = ap.parse_args()

    project_root = Path(__file__).resolve().parent
    target = (project_root / args.results_dir).resolve() if not Path(args.results_dir).is_absolute() else Path(args.results_dir)
    folders = discover_result_folders(target)

    if not folders:
        print(f"No result folders found under {target}")
        return 1

    rows: list[dict[str, object]] = []
    for folder in folders:
        print(f"[RUN] {folder.name}: old")
        run_profile(project_root, folder, "old")
        print(f"[RUN] {folder.name}: new")
        run_profile(project_root, folder, "new")

        old = load_metrics(folder, "old")
        new = load_metrics(folder, "new")

        rows.append({
            "folder": folder.name,
            # "acc_old": old["accuracy"],
            # "acc_new": new["accuracy"],
            # "f1_old": old["f1"],
            # "f1_new": new["f1"],
            # "gap_old": old["score_gap"],
            # "gap_new": new["score_gap"],
            "recall_old": old["recall"],
            "recall_new": new["recall"],
            "mean_err_old": old["mean_err"],
            "mean_err_new": new["mean_err"],
            # "p90_err_old": old["p90_err"],
            # "p90_err_new": new["p90_err"],
        })

    print()
    print(
        f"{'folder':15s} | {'recall_old':>10s} | {'recall_new':>10s} | "
        f"{'mean_err_old':>12s} | {'mean_err_new':>12s}"
    )
    print("-" * 112)
    for row in rows:
        print(
            f"{row['folder']:15s} | {fmt(row['recall_old']):>10s} | {fmt(row['recall_new']):>10s} | "
            f"{fmt(row['mean_err_old']):>12s} | {fmt(row['mean_err_new']):>12s}"
        )

    print()
    print("Saved side-by-side outputs:")
    for folder in folders:
        print(f"  {folder}/detection_results.old.csv")
        print(f"  {folder}/detection_results.new.csv")
        print(f"  {folder}/ue_position_errors.old.csv")
        print(f"  {folder}/ue_position_errors.new.csv")

    # Lưu file CSV tổng hợp để plot_weight_profile_comparison.py có thể đọc
    summary_csv = target / "compare_weight_profiles_summary.csv"
    pd.DataFrame(rows).to_csv(summary_csv, index=False)
    print(f"\nSaved summary CSV: {summary_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
