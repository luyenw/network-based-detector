#!/usr/bin/env python3
"""
detector.py — Physics-Constrained Fake Base Station Detector

Reads:
    /output/measurements.csv    — UE measurement reports (RSRP, Timing Advance)
    /output/cell_database.csv   — Global network cell topology

Writes:
    /output/detection_results.csv — Per-cell anomaly scores + detection verdicts

Architecture is deliberately modular so that:
  • multilateration can be added in compute_estimated_position()
  • additional physics constraints can be added in score_cell()
  • real-time streaming can replace batch reads later

Usage (called by ns-3 via system()):
    python3 /detector/detector.py
"""

import os
import sys
import math
import pandas as pd
import numpy as np
from datetime import datetime

# ---------------------------------------------------------------------------
# Paths (Docker volume mounts)
# ---------------------------------------------------------------------------
OUTPUT_DIR        = "/output"
MEASUREMENTS_PATH = os.path.join(OUTPUT_DIR, "measurements.csv")
CELL_DB_PATH      = os.path.join(OUTPUT_DIR, "cell_database.csv")
RESULTS_PATH      = os.path.join(OUTPUT_DIR, "detection_results.csv")

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
SPEED_OF_LIGHT_M_S  = 3.0e8
LTE_TA_STEP_METERS  = (SPEED_OF_LIGHT_M_S * 16 * 512) / (30720e3 * 2)
# One TA unit in LTE = 16 × Ts × c / 2, Ts = 1/(30.72 MHz)
# ≈ 78.12 m per TA unit (rounded distance each side)

# Detection thresholds (tune for your scenario)
TA_DEVIATION_THRESHOLD_M   = 250.0   # metres: acceptable TA vs geometry error
RSRP_DEVIATION_THRESHOLD   = 10.0    # dB: acceptable RSRP vs path-loss deviation
MIN_OBSERVATIONS_FOR_SCORE = 3       # minimum UE reports needed to score a cell


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_cell_database(path: str) -> pd.DataFrame:
    """Load global cell topology database."""
    if not os.path.exists(path):
        print(f"[WARN] cell_database.csv not found at {path}", flush=True)
        return pd.DataFrame(columns=[
            "time_sec", "cell_id", "pos_x", "pos_y",
            "tx_power_dbm", "is_fake"
        ])
    df = pd.read_csv(path, skipinitialspace=True)
    df.columns = df.columns.str.strip()
    return df


def load_measurements(path: str) -> pd.DataFrame:
    """Load UE measurement reports."""
    if not os.path.exists(path):
        print(f"[WARN] measurements.csv not found at {path}", flush=True)
        return pd.DataFrame(columns=[
            "time_sec", "imsi", "ue_x", "ue_y",
            "serving_cell_id", "rsrp_dbm", "timing_advance"
        ])
    df = pd.read_csv(path, skipinitialspace=True)
    df.columns = df.columns.str.strip()
    return df


# ---------------------------------------------------------------------------
# Physics helpers
# ---------------------------------------------------------------------------

def ta_to_distance_m(ta: int) -> float:
    """Convert a Timing Advance value to round-trip distance in metres."""
    return max(0.0, ta) * LTE_TA_STEP_METERS


def geometric_distance_m(ue_x: float, ue_y: float,
                          cell_x: float, cell_y: float) -> float:
    """Euclidean distance between UE and cell on 2-D plane."""
    return math.sqrt((ue_x - cell_x) ** 2 + (ue_y - cell_y) ** 2)


def log_distance_path_loss_dbm(tx_power_dbm: float,
                                distance_m: float,
                                path_loss_exp: float = 3.5,
                                ref_dist_m: float = 1.0,
                                freq_mhz: float = 2100.0) -> float:
    """
    Expected RSRP using log-distance path loss model.
    Matches the simplified model used in fake-bs-sim.cc.
    """
    if distance_m < ref_dist_m:
        distance_m = ref_dist_m
    pl = 20.0 * math.log10(distance_m) + 38.4   # simplified from sim
    return tx_power_dbm - pl


# ---------------------------------------------------------------------------
# Multilateration stub
# Returns estimated (x, y) of cell given a set of (ue_x, ue_y, ta) triplets.
# Currently returns centroid of TA-circle intersections (placeholder).
# Replace with scipy.optimize.minimize + physics constraints for real paper.
# ---------------------------------------------------------------------------

def compute_estimated_position(obs: pd.DataFrame,
                                cell_info: dict) -> tuple:
    """
    Estimate cell position from UE observations using Timing Advance.

    Parameters
    ----------
    obs       : DataFrame rows for this cell_id
    cell_info : dict with keys pos_x, pos_y, tx_power_dbm from cell database

    Returns
    -------
    (est_x, est_y) in metres, or (nan, nan) if insufficient data
    """
    if len(obs) < MIN_OBSERVATIONS_FOR_SCORE:
        return (float("nan"), float("nan"))

    # Filter valid TA values (TA == -1 means TA not available from this report)
    ta_obs = obs[obs["timing_advance"] >= 0].copy()
    if len(ta_obs) == 0:
        return (float("nan"), float("nan"))

    # Weighted centroid of TA circles (placeholder for full multilateration)
    # Each UE casts a vote for the cell centre based on TA-radius from its position.
    # Direction: from UE toward known cell position (if available), else use grid centroid.
    cx, cy = cell_info.get("pos_x", 0), cell_info.get("pos_y", 0)

    est_xs, est_ys = [], []
    for _, row in ta_obs.iterrows():
        ue_x, ue_y = row["ue_x"], row["ue_y"]
        ta_dist = ta_to_distance_m(int(row["timing_advance"]))
        # Direction from UE toward (cx, cy)
        dx, dy = cx - ue_x, cy - ue_y
        dist_to_nominal = math.sqrt(dx ** 2 + dy ** 2)
        if dist_to_nominal < 1.0:
            est_xs.append(ue_x)
            est_ys.append(ue_y)
        else:
            unit_x, unit_y = dx / dist_to_nominal, dy / dist_to_nominal
            est_xs.append(ue_x + unit_x * ta_dist)
            est_ys.append(ue_y + unit_y * ta_dist)

    return (float(np.mean(est_xs)), float(np.mean(est_ys)))


# ---------------------------------------------------------------------------
# Per-cell anomaly scoring
# ---------------------------------------------------------------------------

def score_cell(cell_id: int,
               cell_info: dict,
               obs: pd.DataFrame) -> dict:
    """
    Compute an anomaly score for one cell.

    Score components (equally weighted, each 0–1):
      1. ta_geometry_score : TA-implied distance vs geometric distance discrepancy
      2. rsrp_physics_score: measured RSRP vs path-loss model expectation

    Returns dict with all fields for detection_results.csv.
    """
    n = len(obs)
    if n < MIN_OBSERVATIONS_FOR_SCORE:
        return {
            "cell_id":          cell_id,
            "score":            0.0,
            "is_fake_detected": 0,
            "est_x":            float("nan"),
            "est_y":            float("nan"),
            "n_obs":            n,
            "ta_score":         0.0,
            "rsrp_score":       0.0,
        }

    cell_x    = cell_info.get("pos_x", 0.0)
    cell_y    = cell_info.get("pos_y", 0.0)
    tx_power  = cell_info.get("tx_power_dbm", 46.0)

    ta_errors   = []
    rsrp_errors = []

    for _, row in obs.iterrows():
        ue_x, ue_y = row["ue_x"], row["ue_y"]
        ta         = row["timing_advance"]
        rsrp       = row["rsrp_dbm"]

        geo_dist = geometric_distance_m(ue_x, ue_y, cell_x, cell_y)

        # --- TA physics check ---
        if ta >= 0:
            ta_dist  = ta_to_distance_m(int(ta))
            ta_error = abs(ta_dist - geo_dist)
            ta_errors.append(ta_error)

        # --- RSRP physics check ---
        if rsrp > -140.0:   # valid range
            expected_rsrp = log_distance_path_loss_dbm(tx_power, max(geo_dist, 1.0))
            rsrp_error    = abs(rsrp - expected_rsrp)
            rsrp_errors.append(rsrp_error)

    # Normalise errors into [0, 1] scores
    # Score → 1 means "highly anomalous" (fake-like)
    mean_ta_err   = float(np.mean(ta_errors))   if ta_errors   else 0.0
    mean_rsrp_err = float(np.mean(rsrp_errors)) if rsrp_errors else 0.0

    ta_score   = min(1.0, mean_ta_err   / TA_DEVIATION_THRESHOLD_M)
    rsrp_score = min(1.0, mean_rsrp_err / RSRP_DEVIATION_THRESHOLD)

    combined_score = 0.5 * ta_score + 0.5 * rsrp_score

    # Threshold for detection verdict
    is_fake_detected = int(combined_score >= 0.5)

    est_x, est_y = compute_estimated_position(obs, cell_info)

    return {
        "cell_id":          cell_id,
        "score":            round(combined_score, 4),
        "is_fake_detected": is_fake_detected,
        "est_x":            round(est_x, 2) if not math.isnan(est_x) else float("nan"),
        "est_y":            round(est_y, 2) if not math.isnan(est_y) else float("nan"),
        "n_obs":            n,
        "ta_score":         round(ta_score, 4),
        "rsrp_score":       round(rsrp_score, 4),
    }


# ---------------------------------------------------------------------------
# Main detector pipeline
# ---------------------------------------------------------------------------

def run_detector():
    timestamp = datetime.utcnow().isoformat()
    sim_time  = 0.0

    print(f"[detector] Running at UTC {timestamp}", flush=True)

    # -- Load inputs --
    cell_db      = load_cell_database(CELL_DB_PATH)
    measurements = load_measurements(MEASUREMENTS_PATH)

    if measurements.empty:
        print("[detector] No measurements yet — writing empty results.", flush=True)
        pd.DataFrame(columns=[
            "time_sec", "cell_id", "score", "is_fake_detected",
            "est_x", "est_y", "n_obs", "ta_score", "rsrp_score"
        ]).to_csv(RESULTS_PATH, index=False)
        return

    sim_time = float(measurements["time_sec"].max())

    # Build cell lookup dict
    cell_lookup = {}
    if not cell_db.empty:
        for _, row in cell_db.iterrows():
            cid = int(row["cell_id"])
            cell_lookup[cid] = {
                "pos_x":        float(row["pos_x"]),
                "pos_y":        float(row["pos_y"]),
                "tx_power_dbm": float(row["tx_power_dbm"]),
                "is_fake":      int(row["is_fake"]),
            }

    # -- Score each observed cell --
    results = []
    for cell_id, group in measurements.groupby("serving_cell_id"):
        cell_id   = int(cell_id)
        cell_info = cell_lookup.get(cell_id, {"pos_x": 0, "pos_y": 0,
                                               "tx_power_dbm": 46.0,
                                               "is_fake": 0})
        result = score_cell(cell_id, cell_info, group)
        result["time_sec"] = round(sim_time, 3)
        results.append(result)

    # -- Write detection results --
    cols = ["time_sec", "cell_id", "score", "is_fake_detected",
            "est_x", "est_y", "n_obs", "ta_score", "rsrp_score"]
    df_results = pd.DataFrame(results)[cols]
    df_results.to_csv(RESULTS_PATH, index=False)

    # -- Console summary --
    print(f"[detector] Scored {len(results)} cells at t={sim_time:.3f}s", flush=True)
    flagged = df_results[df_results["is_fake_detected"] == 1]
    if flagged.empty:
        print("[detector] No fake BS detected.", flush=True)
    else:
        print(f"[detector] ALERT — {len(flagged)} cell(s) flagged:", flush=True)
        for _, row in flagged.iterrows():
            print(f"  cell_id={int(row['cell_id'])}  score={row['score']:.4f}"
                  f"  est_pos=({row['est_x']}, {row['est_y']})", flush=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        run_detector()
    except Exception as exc:
        print(f"[detector] ERROR: {exc}", file=sys.stderr, flush=True)
        sys.exit(1)
