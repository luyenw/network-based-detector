#!/usr/bin/env python3
"""
plot_layout.py — Cellular Network Layout + Fake-BS Capture Chart

Figure layout:
  Layer 1  UE–BS association lines  (skipped with --no-ue)
             gray       → UE served by legal BS
             light red  → UE served by fake BS (same ECGI, closer source)
             magenta    → UE camped on fake BS (from fake_served_ues.csv)
  Layer 2  UE markers               (skipped with --no-ue)
             no border  → served by legal BS
             red border → served by fake BS (from measurements.csv)
             red border → camped on fake BS (from fake_served_ues.csv)
  Layer 3  Legal BS markers (large circles, unique colour, no label)
  Layer 4  Fake BS marker (red upward triangle)

Serving source resolution (map):
  If a UE's serving ECGI is shared with a fake BS, compare path-loss distance
  to fake vs legal. The closer source wins — matching PeriodicMeasure() logic.

Inputs  (results/):
    cell_database.csv    — ecgi, pos_x, pos_y, tx_power_dbm, is_fake
    measurements.csv     — time_sec, imsi, ue_x, ue_y, ecgi, cell_role, ...  (skipped with --no-ue)
    fake_served_ues.csv  — time_sec, imsi, ue_x, ue_y, serving_ecgi,
                           rsrp_dbm, timing_advance  (optional, skipped with --no-ue)

Output:
    results/layout.png  (and optional interactive window)

Usage:
    python plot_layout.py [--results-dir results]
                          [--time 1.0]
                          [--output results/layout.png]
                          [--no-ue]          # skip UE layer (only BSs shown)
                          [--no-show]
"""

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import hsv_to_rgb


# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------
def _bs_colors(n: int) -> list:
    """n visually distinct colours; red (hue≈0) is reserved for the fake BS."""
    hues = np.linspace(0.08, 0.82, max(n, 1))
    return [hsv_to_rgb([h, 0.65, 0.88]) for h in hues]


# ---------------------------------------------------------------------------
# Serving-source resolver
# ---------------------------------------------------------------------------
def _resolve_source(
    ue_x: float,
    ue_y: float,
    ecgi: str,
    legal_pos: dict,          # ecgi → (x, y)
    fake_pos:  dict,          # ecgi → (x, y)  — only for impersonated ECGIs
) -> tuple:
    """
    Returns (bx, by, is_fake_source).

    When the serving ECGI is shared with a fake BS, the source is whichever
    station has a shorter path (lower path-loss), matching PeriodicMeasure().
    """
    lx, ly = legal_pos[ecgi]

    if ecgi not in fake_pos:
        return lx, ly, False

    fx, fy  = fake_pos[ecgi]
    d_legal = math.hypot(ue_x - lx, ue_y - ly)
    d_fake  = math.hypot(ue_x - fx, ue_y - fy)

    if d_fake < d_legal:
        return fx, fy, True
    return lx, ly, False


# ---------------------------------------------------------------------------
# Main plot
# ---------------------------------------------------------------------------
def plot_layout(
    results_dir: Path,
    snap_time:   float = 1.0,
    output_path: Path  = None,
    show:        bool  = True,
    no_ue:       bool  = False,   # skip UE layer — works with just cell_database.csv
) -> None:

    # ── Load mandatory data ──────────────────────────────────────────────
    db   = pd.read_csv(results_dir / "cell_database.csv",  skipinitialspace=True)
    meas = None if no_ue else pd.read_csv(results_dir / "measurements.csv",   skipinitialspace=True)

    # ── Load optional fake-served data ────────────────────────────────────────────
    fs_path = results_dir / "fake_served_ues.csv"
    fs = None
    if not no_ue:
        fs = pd.read_csv(fs_path, skipinitialspace=True) if fs_path.exists() else None
    has_fs  = fs is not None and not fs.empty

    legal = db[db["is_fake"] == 0].drop_duplicates(subset="ecgi").copy()
    fake  = db[db["is_fake"] == 1].copy()

    # ── Position lookup dicts ─────────────────────────────────────────────────
    legal_pos = {
        row["ecgi"]: (row["pos_x"], row["pos_y"])
        for _, row in legal.iterrows()
    }
    fake_pos = {
        row["ecgi"]: (row["pos_x"], row["pos_y"])
        for _, row in fake.iterrows()
    }

    # ── Assign unique colour per legal ECGI ───────────────────────────────────
    ecgi_sorted = sorted(
        legal["ecgi"].unique(),
        key=lambda e: int(e.split("-")[-1]),
    )
    palette    = _bs_colors(len(ecgi_sorted))
    ecgi_color = {ecgi: palette[i] for i, ecgi in enumerate(ecgi_sorted)}

    # ── UE snapshot (from measurements.csv) ──────────────────────────────────────────
    resolved: list = []
    fs_snap_rows: list = []
    if not no_ue and meas is not None:
        t_avail = sorted(meas["time_sec"].unique())
        t_snap  = min(t_avail, key=lambda t: abs(t - snap_time))

        snap = meas[meas["time_sec"] == t_snap]
        serving = (
            snap[snap["cell_role"] == "S"][["imsi", "ue_x", "ue_y", "ecgi"]]
            .drop_duplicates(subset="imsi")
            .reset_index(drop=True)
        )

        for _, row in serving.iterrows():
            ecgi = row["ecgi"]
            if ecgi not in legal_pos:
                continue
            bx, by, is_fake_src = _resolve_source(
                row["ue_x"], row["ue_y"], ecgi, legal_pos, fake_pos
            )
            resolved.append({
                "ue_x":        row["ue_x"],
                "ue_y":        row["ue_y"],
                "ecgi":        ecgi,
                "bx":          bx,
                "by":          by,
                "is_fake_src": is_fake_src,
            })

        # Fake-served UEs at snapshot time (from fake_served_ues.csv)
        if has_fs:
            fs_snap = fs[fs["time_sec"] == t_snap]
            for _, row in fs_snap.iterrows():
                ecgi = row["serving_ecgi"]
                if ecgi in fake_pos:
                    fx, fy = fake_pos[ecgi]
                    fs_snap_rows.append({
                        "ue_x": row["ue_x"],
                        "ue_y": row["ue_y"],
                        "bx":   fx,
                        "by":   fy,
                    })

    # ── Axis bounds ────────────────────────────────────────────────────────────
    margin = 250
    all_x  = list(db["pos_x"]) + [r["ue_x"] for r in resolved]
    all_y  = list(db["pos_y"]) + [r["ue_y"] for r in resolved]
    if has_fs:
        all_x += list(fs["ue_x"])
        all_y += list(fs["ue_y"])
    x_min, x_max = min(all_x) - margin, max(all_x) + margin
    y_min, y_max = min(all_y) - margin, max(all_y) + margin

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 9))
    fig.patch.set_facecolor("white")
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_facecolor("#F5F5F5")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # ── Grid lines through cell centers, clipped to grid extent ──────────────
    xs = sorted(legal["pos_x"].unique())
    ys = sorted(legal["pos_y"].unique())
    for xv in xs:
        ax.plot([xv, xv], [ys[0], ys[-1]], color="gray", linewidth=0.5,
                linestyle="--", alpha=0.3, zorder=0)
    for yh in ys:
        ax.plot([xs[0], xs[-1]], [yh, yh], color="gray", linewidth=0.5,
                linestyle="--", alpha=0.3, zorder=0)

    # ── Layer 1 : UE–BS association lines (legal-served UEs) ──────────────────
    for r in resolved:
        color = "#FF9999" if r["is_fake_src"] else "#BBBBBB"
        alpha = 0.55      if r["is_fake_src"] else 0.35
        lw    = 0.8       if r["is_fake_src"] else 0.6
        ax.plot(
            [r["ue_x"], r["bx"]], [r["ue_y"], r["by"]],
            color=color, linewidth=lw, alpha=alpha, zorder=1,
        )

    # ── Layer 1b : UE–FakeBS lines for fake-camped UEs at snapshot ────────────
    for r in fs_snap_rows:
        ax.plot(
            [r["ue_x"], r["bx"]], [r["ue_y"], r["by"]],
            color="#FF00FF", linewidth=0.7, alpha=0.40, zorder=1,
        )

    # ── Layer 2 : UE markers (legal-served) ───────────────────────────────────
    for r in resolved:
        c = ecgi_color.get(r["ecgi"], (0.5, 0.5, 0.5))
        if r["is_fake_src"]:
            ax.scatter(
                r["ue_x"], r["ue_y"],
                c=[c], s=30, alpha=0.95,
                edgecolors="black", linewidths=0.4,
                zorder=2,
            )
        else:
            ax.scatter(
                r["ue_x"], r["ue_y"],
                c=[c], s=25, alpha=0.88,
                edgecolors="black", linewidths=0.4,
                zorder=2,
            )

    # ── Layer 2b : UE markers (camped on fake BS at snapshot) ─────────────────
    for r in fs_snap_rows:
        ax.scatter(
            r["ue_x"], r["ue_y"],
            c=["#FFAAAA"], s=30, alpha=0.95,
            edgecolors="black", linewidths=0.4,
            zorder=3,
        )

    # ── Layer 3 : Legal BS markers ────────────────────────────────────────────
    for ecgi in ecgi_sorted:
        row = legal[legal["ecgi"] == ecgi].iloc[0]
        c   = ecgi_color[ecgi]

        ax.scatter(
            row["pos_x"], row["pos_y"],
            c=[c], s=360, marker="o",
            edgecolors="black", linewidths=0.8,
            zorder=4,
        )

    # ── Layer 4 : Fake BS marker ──────────────────────────────────────────────
    for _, frow in fake.iterrows():
        ax.scatter(
            frow["pos_x"], frow["pos_y"],
            c="red", marker="^", s=400,
            edgecolors="black", linewidths=0.8,
            zorder=6,
        )

    n_fs_snap = len(fs_snap_rows)

    if no_ue:
        # In no-ue mode, add a text label showing the FBS-LBS configuration
        import math
        for _, frow in fake.iterrows():
            for ecgi, (lx, ly) in legal_pos.items():
                if ecgi in fake_pos:  # this is the impersonated ECGI
                    d = math.hypot(frow["pos_x"] - lx, frow["pos_y"] - ly)
                    ax.text(
                        frow["pos_x"] + 30, frow["pos_y"] + 30,
                        f"d={d:.0f} m",
                        fontsize=8, color="red", fontweight="bold",
                        zorder=7,
                    )
                    break

    # ── Save / show ───────────────────────────────────────────────────────────
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Layout saved → {output_path}")
    if show:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Plot cellular network layout",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python plot_layout.py                          # uses 'results/'\n"
            "  python plot_layout.py results/fbs-close2      # custom folder\n"
            "  python plot_layout.py results/fbs-close2 --no-ue\n"
        ),
    )
    ap.add_argument(
        "results_dir", nargs="?", default=None,
        help="Folder with cell_database.csv and measurements.csv (positional shortcut)",
    )
    ap.add_argument(
        "--results-dir", dest="results_dir_flag", default=None,
        help="Same as positional results_dir (for script/pipeline use)",
    )
    ap.add_argument(
        "--time", type=float, default=1.0,
        help="Snapshot time [s] — nearest available timestamp is used (default: 1.0)",
    )
    ap.add_argument(
        "--output", default=None,
        help="Output image path (default: <results_dir>/layout.png)",
    )
    ap.add_argument(
        "--no-ue", action="store_true",
        help="Skip UE layer — plot BSs only, works without measurements.csv",
    )
    ap.add_argument(
        "--no-show", action="store_true",
        help="Suppress interactive display",
    )
    args = ap.parse_args()

    # Resolve results directory: positional arg wins over --results-dir flag
    rdir = Path(args.results_dir or args.results_dir_flag or "results")

    # Default output to <results_dir>/layout.png
    out = Path(args.output) if args.output else rdir / "layout.png"

    plot_layout(
        results_dir = rdir,
        snap_time   = args.time,
        output_path = out,
        show        = not args.no_show,
        no_ue       = args.no_ue,
    )

