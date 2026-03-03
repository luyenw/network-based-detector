#!/usr/bin/env python3
"""
run_all_and_plot.py
===================
1. Chạy sliding_window_detector.py trên từng subfolder trong results/
   (nếu detection_results.csv chưa tồn tại hoặc --force được truyền vào).
2. Đọc score của fake cell (is_fake_gt == 1) từ detection_results.csv.
3. Vẽ box plot phân bổ score theo khoảng cách FBS-LBS.

Usage:
    python run_all_and_plot.py [--results-dir results] [--force] [--no-show]

    --force   : luôn chạy lại detector kể cả khi detection_results.csv đã có
    --no-run  : bỏ qua bước chạy detector, chỉ đọc CSV và vẽ biểu đồ
"""

import argparse
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# IEEE/ACM Paper standards
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
    "axes.labelsize": 12,
    "font.size": 11,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11
})


# ---------------------------------------------------------------------------
# Mapping: folder name → (distance_m, display_label)
# Lấy từ comments trong docker-compose.yml
# ---------------------------------------------------------------------------
FOLDER_DIST_MAP = {
    "fbs-zero":    (0,    "0 m\n(colocated)"),
    "fbs-close":   (100,  "100 m"),
    "fbs-close2":  (200,  "200 m"),
    "fbs-medium":  (500,  "≈500 m"),
    "fbs-far":     (1000, "≈1000 m"),
    "fbs-veryfar": (1721, "≈1721 m"),
    "fbs-edge":    (2614, "≈2614 m\n(edge)"),
}


def run_detector(folder: Path, project_root: Path) -> bool:
    """Gọi sliding_window_detector.py cho folder này. Trả về True nếu thành công."""
    detector_script = project_root / "detector" / "sliding_window_detector.py"
    if not detector_script.exists():
        print(f"  [ERROR] Không tìm thấy {detector_script}")
        return False

    # Dùng đường dẫn tương đối — chuẩn giống câu lệnh thủ công:
    # python detector/sliding_window_detector.py --results-dir results/fbs-zero
    rel_folder = folder.relative_to(project_root)

    cmd = [
        sys.executable,
        str(detector_script.relative_to(project_root)),
        "--results-dir", str(rel_folder),
        "--log-level", "WARNING",
    ]
    print(f"  $ {' '.join(str(c) for c in cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(project_root))
    if result.returncode != 0:
        print(f"  [FAIL] exit={result.returncode}")
        if result.stderr:
            lines = result.stderr.strip().splitlines()
            for ln in lines[-5:]:
                print(f"    {ln}")
        return False
    return True


def load_scores(folder: Path) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Đọc detection_results.csv và trả về:
      1. Series chứa tất cả các score của LBS (is_fake_gt == 0)
      2. Series chứa score của FBS (is_fake_gt == 1) tại mỗi time window
      3. Series chứa threshold
    """
    csv = folder / "detection_results.csv"
    empty_series = pd.Series(dtype=float)
    if not csv.exists():
        return empty_series, empty_series, empty_series

    df = pd.read_csv(csv, skipinitialspace=True)
    if df.empty:
        return empty_series, empty_series, empty_series

    if "is_fake_gt" in df.columns:
        fake_mask = df["is_fake_gt"] == 1
    elif "status" in df.columns:
        fake_mask = df["status"] == "FAKE"
    else:
        return empty_series, empty_series, empty_series

    # Tập hợp các scores của LBS
    lbs_scores = df[~fake_mask]["score"]

    # Lấy max score của FBS tại mỗi time window (trường hợp bị record nhiều cell ID do trùng lặp)
    fake_rows = df[fake_mask]
    if not fake_rows.empty:
        fbs_scores = fake_rows.groupby("time_sec")["score"].max()
    else:
        fbs_scores = empty_series

    # Lấy threshold tại mỗi window (ngưỡng giống nhau cho mọi cell trong cùng 1 window)
    thresholds = df.groupby("time_sec")["threshold"].first()

    return lbs_scores, fbs_scores, thresholds


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Batch detector + box plot theo khoảng cách FBS")
    ap.add_argument("--results-dir", default="results",
                    help="Thư mục gốc chứa các subfolder (default: results)")
    ap.add_argument("--force", action="store_true",
                    help="Chạy lại detector kể cả khi detection_results.csv đã có")
    ap.add_argument("--no-run", action="store_true",
                    help="Bỏ qua bước chạy detector (chỉ vẽ từ CSV sẵn có)")
    ap.add_argument("--output", default=None,
                    help="Lưu biểu đồ ra file (vd: results/score_boxplot.png)")
    ap.add_argument("--no-show", action="store_true",
                    help="Không hiện cửa sổ tương tác")
    args = ap.parse_args()

    results_root = Path(args.results_dir).resolve()
    project_root = Path(__file__).parent.resolve()

    if not results_root.exists():
        print(f"[ERROR] Thư mục '{results_root}' không tồn tại.")
        sys.exit(1)

    # ── Bước 1: Tìm các subfolder khớp với FOLDER_DIST_MAP ─────────────────
    found_folders = sorted(
        [d for d in results_root.iterdir() if d.is_dir() and d.name in FOLDER_DIST_MAP],
        key=lambda d: FOLDER_DIST_MAP[d.name][0],
    )

    if not found_folders:
        print(f"[ERROR] Không tìm thấy subfolder nào phù hợp trong '{results_root}'.")
        print("  Các folder hợp lệ:", list(FOLDER_DIST_MAP.keys()))
        sys.exit(1)

    print(f"Tìm thấy {len(found_folders)} folder: {[d.name for d in found_folders]}\n")

    # ── Bước 2: Chạy detector nếu cần ─────────────────────────────────────
    if not args.no_run:
        for folder in found_folders:
            csv_path = folder / "detection_results.csv"
            if csv_path.exists() and not args.force:
                print(f"[SKIP] {folder.name}  (detection_results.csv đã có, dùng --force để chạy lại)")
            else:
                print(f"[RUN ] {folder.name}")
                ok = run_detector(folder, project_root)
                status = "OK" if ok else "FAIL"
                print(f"  → {status}\n")
    else:
        print("[--no-run] Bỏ qua bước chạy detector.\n")

    # ── Bước 3: Đọc score và vẽ box plot ──────────────────────────────────
    lbs_data_list: list[list[float]] = []
    fbs_means: list[float] = []
    thresh_means: list[float] = []
    labels_ordered: list[str] = []
    distances_ordered: list[int] = []

    for folder in found_folders:
        dist_m, label = FOLDER_DIST_MAP[folder.name]
        lbs_scores, fbs_scores, thresholds = load_scores(folder)
        
        if fbs_scores.empty or lbs_scores.empty:
            print(f"[WARN] {folder.name}: không có dữ liệu score hoàn chỉnh.")
            continue
            
        lbs_data_list.append(lbs_scores.tolist())
        
        # Lưu mean của riêng FBS
        fbs_mean_val = fbs_scores.mean()
        fbs_means.append(fbs_mean_val)
        
        thresh_means.append(thresholds.mean())
        
        labels_ordered.append(label)
        distances_ordered.append(dist_m)
        print(f"  {folder.name:15s}  dist={dist_m:5d} m  |  LBS n={len(lbs_scores):5d} med={lbs_scores.median():.4f}  |  FBS_mean={fbs_mean_val:.4f}  |  thresh_mean={thresh_means[-1]:.4f}")

    if not labels_ordered:
        print("\n[ERROR] Không có dữ liệu để vẽ.")
        sys.exit(1)

    # ── Vẽ biểu đồ ─────────────────────────────────────────────────────────
    # Size (10, 6) for a generous two-column width fit
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("white")

    n_groups = len(labels_ordered)
    positions = np.arange(1, n_groups + 1)

    # Highlight background as safe margin behind the baseline (LBS median)
    
    # 1. Vẽ box plot duy nhất cho tập LBS với màu sắc chuẩn mực
    bp_lbs = ax.boxplot(
        lbs_data_list,
        positions=positions,
        patch_artist=True,
        medianprops=dict(color="black", linewidth=1.5),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.5),
        flierprops=dict(marker=".", markersize=3, alpha=0.5, color='gray', markeredgecolor='none'),
        widths=0.45,
    )
    for patch in bp_lbs["boxes"]:
        patch.set_facecolor("lightgray")
        patch.set_edgecolor("black")
        patch.set_linewidth(1.2)
        patch.set_alpha(0.8)

    # 2. Đường nối trung bình Threshold (MAD)
    line_thresh, = ax.plot(
        positions, thresh_means, 
        color='#e74c3c', marker='o', linestyle='--', linewidth=2, markersize=5, 
        label=r'$\tau(T)$ (MAD Threshold)', zorder=4
    )

    # 3. Đường nối trung bình FBS Score
    line_fbs, = ax.plot(
        positions, fbs_means, 
        color='#2980b9', marker='D', linestyle='-', linewidth=2, markersize=7, 
        label=r'$\mathrm{Score}_{FBS}$', zorder=5
    )

    # Decorate axes
    ax.set_xticks(positions)
    ax.set_xticklabels(labels_ordered)
    ax.set_xlabel(r'Distance from Fake BS to Spoofed Legal BS', labelpad=8)
    ax.set_ylabel(r'Anomaly Score', labelpad=8)
    ax.set_title("Distribution of LBS Anomaly Scores vs. FBS Average Score", weight='bold', pad=12)

    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    # Legend
    proxy_lbs = mpatches.Patch(facecolor='lightgray', edgecolor='black', alpha=0.8, label=r'LBS Scores ($\mathrm{Score}_{LBS}$)')
    ax.legend(handles=[proxy_lbs, line_fbs, line_thresh], loc="upper left", framealpha=0.9, edgecolor='gray')

    # Annotate FBS Means & Thresholds slightly above/below their points
    for idx, (fbs_val, thr_val) in enumerate(zip(fbs_means, thresh_means)):
        pos = positions[idx]
        ax.text(pos, fbs_val + 0.015, f"{fbs_val:.3f}", ha="center", va="bottom",
                fontsize=9, color="#2980b9", fontweight="bold")
        
        # Tránh the text overlapping if they are too close
        if abs(fbs_val - thr_val) > 0.03:
            ax.text(pos, thr_val - 0.015, f"{thr_val:.3f}", ha="center", va="top",
                    fontsize=9, color="#e74c3c", fontweight="bold")

    plt.tight_layout()

    # Lưu / hiển thị
    out_path = Path(args.output) if args.output else results_root / "score_boxplot.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"\nBiểu đồ đã lưu → {out_path}")

    if not args.no_show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
