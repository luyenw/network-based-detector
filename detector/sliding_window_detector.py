#!/usr/bin/env python3
"""
sliding_window_detector.py

Per-second fake cell detector implementing the residual method
described in fake_cell_detection.md.

Algorithm (§9 – run once per measurement timestamp T):
  For T = 1, 2, ..., T_max:
    1.  M_T ← measurements with time_sec == T  (current second only)
    2.  Group M_T by IMSI
    3.  For each IMSI:
          a. r_c   = mean RSRP per ECGI                             (§4)
          b. μ, σ  = mean / std of {r_c} across cells              (§5)
             r̃_c  = (r_c − μ) / σ
          c. u*    = argmin_u R(u)                                  (§6)
             R(u)  = 1/N Σ_c (r̃_c − r̃̂_c(u))²
             r̂_c  = P_c − PL(dist(u, p_c))
             r̃̂_c  = (r̂_c − μ) / σ
          d. e_c   = |r̃_c − r̃̂_c(u*)|                             (§7)
    4.  Score_c(T) = mean(e_c) over all valid IMSIs in second T     (§7)
    5.  Cell_c     = FAKE  if Score_c(T) > τ,  LEGIT otherwise      (§8)

UE position estimation (replaces grid search):
    Multi-start L-BFGS-B minimising R(u).
    Starts: centroid of legal cells + n_ue_starts random points
    inside [cell_bbox ± margin].

Path-loss (COST-231 urban macro 2 GHz, matches fake-bs-sim.cc):
    PL [dB] = 128.1 + 37.6 · log10(d_km)

Inputs  (results/):
    measurements.csv   — time_sec, imsi, ue_x, ue_y, cell_id, ecgi,
                         cell_role, rsrp_dbm, timing_advance
    cell_database.csv  — ecgi, pos_x, pos_y, pos_z, tx_power_dbm, is_fake

Outputs (results/):
    detection_results.csv — time_sec, ecgi, score, threshold, status, is_fake_gt
    window_summary.csv    — time_sec, num_measurements, avg_residual,
                            num_cells, num_fake_cells, runtime_ms

Threshold methods (adaptive, from natural score distribution each second):
    zscore  μ + k·σ              (mean + k standard deviations)
    iqr     Q3 + k·IQR           (Tukey's fence)
    mad     median + k·MAD·1.4826 (most robust to outliers, default)

Usage:
    python sliding_window_detector.py [--results-dir results]
                                      [--n-ue-starts 5]
                                      [--method mad]
                                      [--k-sigma 3.0]
                                      [--min-cells 2]
                                      [--log-level INFO|DEBUG]
"""

import argparse
import concurrent.futures
import logging
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize as _scipy_minimize

from fake_localizer import localize_fake, write_localization_report  # kept for potential external use

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s.%(msecs)03d] [%(levelname)-5s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("SlidingWindowDetector")


# ---------------------------------------------------------------------------
# Path-loss model  (§2)
#   PL [dB] = 128.1 + 37.6 · log10(d_km)
#   r̂_c(u)  = P_c − PL(dist(u, p_c))
# ---------------------------------------------------------------------------
def _pl(dist_m: np.ndarray) -> np.ndarray:
    return 128.1 + 37.6 * np.log10(np.maximum(dist_m, 1.0) / 1000.0)


# ---------------------------------------------------------------------------
# UE position estimation  (§6)
#
# Minimises R(u) = 1/N Σ_c (r̃_c − r̃̂_c(u))² using multi-start L-BFGS-B.
#
# Starting points:
#   1. Centroid of legal cell positions.
#   2. n_starts − 1 random points drawn uniformly from
#      [x_min − margin, x_max + margin] × [y_min − margin, y_max + margin].
#
# Args:
#   r_tilde     (N,)     normalised measured RSRP
#   mu, sigma            normalisation params
#   cell_pos    (N, 2)   legal cell 2-D positions [m]
#   cell_h      (N,)     BS antenna heights [m]
#   tx_powers   (N,)     transmit power [dBm]
#   n_starts    int      total number of optimiser starts (default 5)
#   rng                  np.random.Generator (for reproducibility)
#   excl_margin float    margin around cell bbox for random starts [m]
#
# Returns:
#   u_star               (2,)   best UE position estimate
#   R_star               float  minimum R(u*)
#   r_hat_tilde_star     (N,)   normalised predicted RSRP at u*
# ---------------------------------------------------------------------------
def _estimate_ue_pos(
    r_tilde:    np.ndarray,
    mu:         float,
    sigma:      float,
    cell_pos:   np.ndarray,
    cell_h:     np.ndarray,
    tx_powers:  np.ndarray,
    n_starts:   int = 5,
    rng:        "np.random.Generator | None" = None,
    excl_margin: float = 500.0,
    s_pos:      "np.ndarray | None" = None,
    s_dist_ta:  "float | None" = None,
    lambda_ta:  float = 0.0,
) -> tuple:
    N = len(r_tilde)
    if N == 0:
        return np.zeros(3), np.inf, np.zeros(0)

    if rng is None:
        rng = np.random.default_rng()

    x_min, x_max = cell_pos[:, 0].min(), cell_pos[:, 0].max()
    y_min, y_max = cell_pos[:, 1].min(), cell_pos[:, 1].max()
    ue_z_fixed = 2.0   # handheld device height [m]

    def objective(p: np.ndarray) -> float:
        dist    = np.sqrt((p[0] - cell_pos[:, 0]) ** 2 +
                          (p[1] - cell_pos[:, 1]) ** 2 +
                          (ue_z_fixed - cell_h) ** 2)
        r_hat   = tx_powers - _pl(np.maximum(dist, 1.0))
        r_hat_t = (r_hat - mu) / sigma
        mse     = float(np.mean((r_tilde - r_hat_t) ** 2))
        
        if lambda_ta > 0.0 and s_pos is not None and s_dist_ta is not None:
            dist_s  = np.sqrt((p[0] - s_pos[0])**2 + (p[1] - s_pos[1])**2 + (ue_z_fixed - s_pos[2])**2)
            penalty = lambda_ta * ((dist_s - s_dist_ta) ** 2)
            return mse + float(penalty)
            
        return mse

    # Starting points
    centroid = cell_pos.mean(axis=0)
    starts   = [centroid]
    for _ in range(n_starts - 1):
        px = rng.uniform(x_min - excl_margin, x_max + excl_margin)
        py = rng.uniform(y_min - excl_margin, y_max + excl_margin)
        starts.append(np.array([px, py]))

    best_u = centroid.copy()
    best_R = np.inf
    for p0 in starts:
        res = _scipy_minimize(
            objective, p0, method="L-BFGS-B",
            options={"maxiter": 200, "ftol": 1e-12, "gtol": 1e-9},
        )
        p = res.x
        R = objective(p)
        if R < best_R:
            best_R = R
            best_u = p

    # Predicted RSRP at best u
    dist_star    = np.sqrt((best_u[0] - cell_pos[:, 0]) ** 2 +
                           (best_u[1] - cell_pos[:, 1]) ** 2 +
                           (ue_z_fixed - cell_h) ** 2)
    r_hat_star   = tx_powers - _pl(np.maximum(dist_star, 1.0))
    r_hat_t_star = (r_hat_star - mu) / sigma

    # Append the fixed z-coordinate so u_star is fully 3D
    u_star_3d = np.array([best_u[0], best_u[1], ue_z_fixed])

    return u_star_3d, best_R, r_hat_t_star


# ---------------------------------------------------------------------------
# Process one UE in one window  (§4 – §7)
#
# Returns:
#   e_dict   {ecgi: e_c}   per-cell absolute error, or None if skipped
#   u_star   (2,)          estimated UE position
#   R_star   float         minimum residual
#   cellid_dict            {ecgi: cell_id}
# ---------------------------------------------------------------------------
def process_ue(
    ue_meas:  pd.DataFrame,
    legal:    pd.DataFrame,     # legal cells indexed by ecgi
    min_cells: int,
    n_starts:  int = 5,
    rng:       "np.random.Generator | None" = None,
    lambda_ta: float = 0.0,
) -> tuple:
    s_pos = None
    s_dist_ta = None
    if lambda_ta > 0.0:
        serving_rows = ue_meas[ue_meas["cell_role"] == "S"]
        if len(serving_rows) > 0:
            s_ecgi = serving_rows["ecgi"].iloc[0]
            if s_ecgi in legal.index:
                s_ta = serving_rows["timing_advance"].iloc[0]
                if not pd.isna(s_ta):
                    s_dist_ta = float(s_ta) * 78.12
                    s_pos = legal.loc[s_ecgi, ["pos_x", "pos_y", "pos_z"]].values

    # §4 — mean RSRP per ECGI
    r_c = ue_meas.groupby("ecgi")["rsrp_dbm"].mean()

    ecgi_to_cellid = (
        ue_meas.drop_duplicates(subset="ecgi", keep="last")
        .set_index("ecgi")["cell_id"]
        .to_dict()
    )

    # Only keep ECGIs that have a known legal position
    common = r_c.index.intersection(legal.index)
    r_c    = r_c[common]

    if len(r_c) < min_cells:
        return None, None, None, None

    # §5 — normalise
    mu    = float(r_c.mean())
    sigma = float(r_c.std(ddof=0))
    if sigma < 1e-9:
        return None, None, None, None

    r_tilde   = ((r_c - mu) / sigma).values
    cell_pos  = legal.loc[common, ["pos_x", "pos_y"]].values
    cell_h    = legal.loc[common, "pos_z"].values
    tx_powers = legal.loc[common, "tx_power_dbm"].values
    ecgis     = common.tolist()

    # §6 — multi-start optimiser for u*
    u_star, R_star, r_hat_tilde_star = _estimate_ue_pos(
        r_tilde, mu, sigma, cell_pos, cell_h, tx_powers,
        n_starts=n_starts, rng=rng,
        s_pos=s_pos, s_dist_ta=s_dist_ta, lambda_ta=lambda_ta,
    )

    # Note: process_ue now also needs ground truth to calculate errors later, but we calculate
    # error within the main loop to keep the function signatures unchanged where possible.

    # §7 — per-cell error
    e_c        = np.abs(r_tilde - r_hat_tilde_star)
    e_dict     = dict(zip(ecgis, e_c.tolist()))
    cellid_dict = {ecgi: ecgi_to_cellid.get(ecgi, -1) for ecgi in ecgis}

    return e_dict, u_star, R_star, cellid_dict


# ---------------------------------------------------------------------------
# Adaptive threshold from natural score distribution  (§8)
# ---------------------------------------------------------------------------
def adaptive_threshold(scores: dict, method: str = "mad", k: float = 3.0) -> float:
    vals = np.array(list(scores.values()), dtype=float)
    if len(vals) < 3:
        return np.inf

    if method == "zscore":
        mu    = float(np.mean(vals))
        sigma = float(np.std(vals, ddof=0))
        return (mu + k * sigma) if sigma > 1e-9 else np.inf

    if method == "iqr":
        q1, q3 = np.percentile(vals, [25, 75])
        return float(q3 + k * (q3 - q1))

    if method == "mad":
        median = float(np.median(vals))
        mad    = float(np.median(np.abs(vals - median)))
        return median + k * mad * 1.4826

    raise ValueError(f"Unknown method: {method!r}  (choose zscore | iqr | mad)")


# ---------------------------------------------------------------------------
# Collect UE data for localization
#
# Re-estimates u* for each UE by excluding the fake ECGI from the cell set,
# then running the same multi-start optimizer.  This removes the bias caused
# by the fake signal being included in the detection u*.
# ---------------------------------------------------------------------------
def _collect_ue_loc_data(
    M_T:        "pd.DataFrame",
    ecgi_f:     str,
    legal:      "pd.DataFrame",
    u_star_map: dict,
    n_starts:   int = 5,
    rng:        "np.random.Generator | None" = None,
    lambda_ta:  float = 0.0,
) -> list[dict]:
    records: list[dict] = []
    log.debug("  [collect_ue_loc] ecgi_f=%s  u_star_map size=%d", ecgi_f, len(u_star_map))
    n_no_fake = n_no_serving = n_illegal_serving = 0

    for imsi, u_pos_detection in u_star_map.items():
        ue_rows = M_T[M_T["imsi"] == imsi]

        # RSRP observed for ecgi_f
        fake_rows = ue_rows[ue_rows["ecgi"] == ecgi_f]
        if len(fake_rows) == 0:
            n_no_fake += 1
            log.debug("  [collect_ue_loc]   IMSI=%d  skip: no measurement for ecgi_f", imsi)
            continue
        r_if = float(fake_rows["rsrp_dbm"].mean())

        # Serving cell
        serving_rows = ue_rows[ue_rows["cell_role"] == "S"]
        if len(serving_rows) == 0:
            n_no_serving += 1
            log.debug("  [collect_ue_loc]   IMSI=%d  skip: no serving row", imsi)
            continue
        s_ecgi = serving_rows["ecgi"].iloc[0]
        if s_ecgi not in legal.index:
            n_illegal_serving += 1
            log.debug("  [collect_ue_loc]   IMSI=%d  skip: serving %s not in legal", imsi, s_ecgi)
            continue
        r_is  = float(serving_rows["rsrp_dbm"].mean())
        s_row = legal.loc[s_ecgi]

        s_pos_ta = None
        s_dist_ta = None
        if lambda_ta > 0.0:
            s_ta = serving_rows["timing_advance"].iloc[0]
            if not pd.isna(s_ta):
                s_dist_ta = float(s_ta) * 78.12
                s_pos_ta = s_row[["pos_x", "pos_y", "pos_z"]].values

        # Re-estimate UE position excluding the fake ECGI
        rc_clean = (
            ue_rows.groupby("ecgi")["rsrp_dbm"]
            .mean()
            .drop(ecgi_f, errors="ignore")
        )
        common   = rc_clean.index.intersection(legal.index)
        rc_clean = rc_clean[common]

        u_pos = u_pos_detection        # fallback
        if len(rc_clean) >= 2:
            mu    = float(rc_clean.mean())
            sigma = float(rc_clean.std(ddof=0))
            if sigma >= 1e-9:
                r_tilde   = ((rc_clean - mu) / sigma).values
                cell_pos  = legal.loc[common, ["pos_x", "pos_y"]].values
                cell_h    = legal.loc[common, "pos_z"].values
                tx_powers = legal.loc[common, "tx_power_dbm"].values
                u_opt, _, _ = _estimate_ue_pos(
                    r_tilde, mu, sigma, cell_pos, cell_h, tx_powers,
                    n_starts=n_starts, rng=rng,
                    s_pos=s_pos_ta, s_dist_ta=s_dist_ta, lambda_ta=lambda_ta,
                )
                u_pos = u_opt

        records.append({
            "imsi":         int(imsi),
            "u_pos":        (float(u_pos[0]), float(u_pos[1]), float(u_pos[2])),
            "serving_rsrp": r_is,
            "serving_pos":  (
                float(s_row["pos_x"]),
                float(s_row["pos_y"]),
                float(s_row["pos_z"]),
            ),
            "fake_rsrp":    r_if,
        })
    log.debug("  [collect_ue_loc] result: kept=%d  dropped(no_fake=%d, no_serving=%d, illegal_serving=%d)",
              len(records), n_no_fake, n_no_serving, n_illegal_serving)
    return records


# ---------------------------------------------------------------------------
# Collect UE data for Bayesian localizer
#
# Simpler than _collect_ue_loc_data: only needs u* and r_{i,f}.
# No serving-cell or legal-consistency requirements.
# ---------------------------------------------------------------------------
def _collect_bayesian_ue_data(
    M_T:        "pd.DataFrame",
    ecgi_f:     str,
    u_star_map: dict,
    legal:      "pd.DataFrame | None" = None,
    sigma_total: float = 9.1,
) -> list[dict]:
    """
    Collect UE data for the Bayesian/DEC-MAP localizer.

    Returns list of dicts with keys:
        u_pos          : (x, y, z)  estimated UE position [m]
        fake_rsrp      : float      r_{i,f} [dBm]
        anomaly_weight : float      w_i ∈ [0,1]  (DEC-MAP source-discrimination weight)

    Anomaly weight formula (§3.2 of the DEC-MAP formulation):
        w_i = 1 - exp(-(RSRP_obs - P̂_LBS)^2 / (2 * sigma_total^2))
        w_i ≈ 0  → UE likely attached to LBS
        w_i ≈ 1  → UE likely attached to FBS
    """
    records: list[dict] = []
    for imsi, u_pos in u_star_map.items():
        ue_rows   = M_T[M_T["imsi"] == imsi]
        fake_rows = ue_rows[ue_rows["ecgi"] == ecgi_f]
        if len(fake_rows) == 0:
            continue
        fake_rsrp = float(fake_rows["rsrp_dbm"].mean())

        # ── Anomaly weight w_i ────────────────────────────────────────────
        w_i = 1.0   # default: treat as FBS UE (full Bayesian update)
        if legal is not None and ecgi_f in legal.index:
            lbs_row   = legal.loc[ecgi_f]
            lbs_tx    = float(lbs_row["tx_power_dbm"])
            # distance from estimated UE pos to the legal BS
            d_lbs     = float(np.sqrt(
                (u_pos[0] - float(lbs_row["pos_x"])) ** 2 +
                (u_pos[1] - float(lbs_row["pos_y"])) ** 2 +
                (u_pos[2] - float(lbs_row["pos_z"])) ** 2
            ))
            d_lbs     = max(d_lbs, 1.0)
            # predicted RSRP if UE were attached to LBS (COST-231)
            p_hat_lbs = lbs_tx - _pl(np.array([d_lbs]))[0]
            delta     = fake_rsrp - p_hat_lbs
            w_i       = float(1.0 - np.exp(-delta ** 2 / (2.0 * sigma_total ** 2)))
            w_i       = float(np.clip(w_i, 0.0, 1.0))

        records.append({
            "imsi":          int(imsi),
            "u_pos":         (float(u_pos[0]), float(u_pos[1]), float(u_pos[2])),
            "fake_rsrp":     fake_rsrp,
            "anomaly_weight": w_i,
        })
    log.debug("  [bayes_data] ecgi_f=%s  u_star_map=%d  collected=%d",
              ecgi_f, len(u_star_map), len(records))
    return records


# ---------------------------------------------------------------------------
# Helper: compute area bounds from legal cell positions + margin
# ---------------------------------------------------------------------------
def _area_bounds(legal: "pd.DataFrame", margin: float = 500.0) -> tuple:
    return (
        float(legal["pos_x"].min()) - margin,
        float(legal["pos_x"].max()) + margin,
        float(legal["pos_y"].min()) - margin,
        float(legal["pos_y"].max()) + margin,
    )

# ---------------------------------------------------------------------------
# Module-level per-window task
# Must be at module scope for ProcessPoolExecutor on Windows (spawn mode).
# ---------------------------------------------------------------------------
def _process_window_task(
    T:            float,
    M_T:          "pd.DataFrame",
    legal:        "pd.DataFrame",
    ecgi_gt_fake: dict,
    min_cells:    int,
    n_ue_starts:  int,
    lambda_ta:    float,
    method:       str,
    k_sigma:      float,
    sigma_total:  float,
    window_idx:   int,   # used as RNG seed → deterministic + independent per window
) -> dict:
    """
    Process one time window T independently.
    Returns a dict with all per-window results to be aggregated by the main process.
    """
    import time as _time
    t0   = _time.perf_counter()
    rng  = np.random.default_rng(window_idx)
    imsis = M_T["imsi"].unique()

    win_errors:    dict = {}
    sum_R_star     = 0.0
    valid_ues      = 0
    u_star_map:    dict = {}
    ue_loc_err_t:  list = []
    ue_loc_stat_t: list = []

    for imsi in imsis:
        ue_meas = M_T[M_T["imsi"] == imsi]
        e_dict, u_star, R_star, _ = process_ue(
            ue_meas, legal, min_cells,
            n_starts=n_ue_starts, rng=rng, lambda_ta=lambda_ta,
        )
        if e_dict is None:
            continue
        u_star_map[imsi] = u_star

        gt_x = float(ue_meas["ue_x"].iloc[0])
        gt_y = float(ue_meas["ue_y"].iloc[0])
        gt_z = float(ue_meas["ue_z"].iloc[0])
        err  = float(np.sqrt(
            (u_star[0] - gt_x) ** 2 +
            (u_star[1] - gt_y) ** 2 +
            (u_star[2] - gt_z) ** 2
        ))
        ue_loc_err_t.append(err)
        ue_loc_stat_t.append({
            "time_sec": T, "imsi": imsi,
            "true_x": round(gt_x, 2), "true_y": round(gt_y, 2), "true_z": round(gt_z, 2),
            "est_x":  round(u_star[0], 2), "est_y": round(u_star[1], 2), "est_z": round(u_star[2], 2),
            "error_m": round(err, 2),
        })
        for ecgi, e_c in e_dict.items():
            win_errors.setdefault(ecgi, []).append(e_c)
        sum_R_star += R_star
        valid_ues  += 1

    # Localization input: for each UE with u_star, log all its MR rows
    loc_input_t: list = []
    for imsi, u_star in u_star_map.items():
        ue_meas = M_T[M_T["imsi"] == imsi]
        for _, mr in ue_meas.iterrows():
            loc_input_t.append({
                "time_sec":       T,
                "imsi":           int(imsi),
                "estimated_ue_x": round(float(u_star[0]), 2),
                "estimated_ue_y": round(float(u_star[1]), 2),
                "estimated_ue_z": round(float(u_star[2]), 2),
                "ecgi":           mr["ecgi"],
                "cell_role":      mr["cell_role"],
                "rsrp_dbm":       round(float(mr["rsrp_dbm"]), 3),
                "timing_advance": int(mr["timing_advance"]),
            })

    scores = {ecgi: float(np.mean(e_list)) for ecgi, e_list in win_errors.items()}
    tau    = adaptive_threshold(scores, method=method, k=k_sigma)
    fakes  = [e for e, s in scores.items() if s > tau]
    avg_R  = sum_R_star / valid_ues if valid_ues > 0 else 0.0
    rt_ms  = (_time.perf_counter() - t0) * 1000.0

    # Detection log entries for FBS-flagged ECGIs in this window
    det_log_t: list = []
    for ecgi_f in fakes:
        if ecgi_f not in legal.index:
            continue
        bayes_data = _collect_bayesian_ue_data(
            M_T, ecgi_f, u_star_map, legal=legal, sigma_total=sigma_total,
        )
        for rec in bayes_data:
            det_log_t.append({
                "timestamp":        T,
                "ue_id":            rec.get("imsi", -1),
                "estimated_ue_x":   round(rec["u_pos"][0], 2),
                "estimated_ue_y":   round(rec["u_pos"][1], 2),
                "estimated_ue_z":   round(rec["u_pos"][2], 2),
                "rsrp_observed":    round(rec["fake_rsrp"], 3),
                "ecgi":             ecgi_f,
                "anomaly_score_wi": round(rec["anomaly_weight"], 6),
            })

    window_row = {
        "time_sec":         T,
        "num_measurements": len(M_T),
        "avg_residual":     round(avg_R, 6),
        "num_cells":        len(scores),
        "num_fake_cells":   len(fakes),
        "runtime_ms":       round(rt_ms, 2),
    }
    cell_rows_t = [
        {
            "time_sec":   T, "ecgi": ecgi,
            "score":      round(score, 6),
            "threshold":  round(tau, 6) if np.isfinite(tau) else None,
            "status":     "FAKE" if score > tau else "LEGIT",
            "is_fake_gt": int(ecgi_gt_fake.get(ecgi, 0)),
        }
        for ecgi, score in scores.items()
    ]
    vote_entries = [(ecgi, score > tau) for ecgi, score in scores.items()]

    return {
        "T": T, "window_row": window_row, "cell_rows": cell_rows_t,
        "vote_entries": vote_entries, "det_log": det_log_t,
        "ue_loc_errors": ue_loc_err_t, "ue_loc_stats": ue_loc_stat_t,
        "loc_input": loc_input_t,
        "rt_ms": rt_ms, "valid_ues": valid_ues, "avg_R": avg_R,
        "fakes": fakes, "tau": tau, "scores": scores,
    }


# ---------------------------------------------------------------------------
# Main sliding-window pipeline  (§9)
# ---------------------------------------------------------------------------
def run_detection(
    results_dir:  Path,
    n_ue_starts:  int   = 5,
    method:       str   = "mad",
    k_sigma:      float = 3.0,
    min_cells:    int   = 2,
    sigma_total:  float = 9.1,
    lambda_ta:    float = 0.001,
    n_workers:    int   = 0,       # 0 = use os.cpu_count()
) -> None:
    """
    Stage 1 — Fake cell detection.

    Writes:
        detection_results.csv  — per-ECGI scores per second
        window_summary.csv     — per-second stats
        detection_log.csv      — UE records (pos, RSRP, anomaly weight) for confirmed fakes
                                 → consumed by dec_map_localizer.py (Stage 2)
    """

    # ── Load inputs ──────────────────────────────────────────────────────────
    log.info("=== Sliding-Window Fake Cell Detector (Stage 1 — Detection) ===")
    log.info("results_dir   : %s", results_dir)
    log.info("n_ue_starts   : %d", n_ue_starts)
    log.info("threshold     : adaptive  method=%s  k=%.2f", method, k_sigma)
    log.info("σ_total       : %.1f dB  (for anomaly weight w_i in detection_log)", sigma_total)

    meas = pd.read_csv(results_dir / "measurements.csv",  skipinitialspace=True)
    
    if "ue_z" not in meas.columns:
        meas["ue_z"] = 0.0

    db   = pd.read_csv(results_dir / "cell_database.csv", skipinitialspace=True)

    log.info("measurements.csv : %d rows, T ∈ [%.1f, %.1f] s",
             len(meas), meas["time_sec"].min(), meas["time_sec"].max())
    log.info("cell_database.csv: %d rows  (%d legal, %d fake)",
             len(db), int((db["is_fake"] == 0).sum()), int(db["is_fake"].sum()))

    cell_id_counts = meas.groupby("cell_id")["ecgi"].first()
    log.info("Unique cell_ids in measurements: %d", len(cell_id_counts))
    for cid, ecgi in sorted(cell_id_counts.items()):
        n_rows = int((meas["cell_id"] == cid).sum())
        log.debug("  cell_id=%-4s  ecgi=%-22s  rows=%d", cid, ecgi, n_rows)

    legal = (
        db[db["is_fake"] == 0]
        .drop_duplicates(subset="ecgi", keep="first")
        .set_index("ecgi")
    )
    log.info("Legal cells (used for predictions): %d unique ECGIs", len(legal))
    for ecgi, row in legal.iterrows():
        log.debug("  ECGI=%-22s  pos=(%.1f, %.1f)  Ptx=%.1f dBm",
                  ecgi, row["pos_x"], row["pos_y"], row["tx_power_dbm"])

    ecgi_gt_fake = db.groupby("ecgi")["is_fake"].max().to_dict()

    fake_true_pos = (
        db[db["is_fake"] == 1]
        .set_index("ecgi")[["pos_x", "pos_y"]]
        .to_dict(orient="index")
    )


    windows = sorted(meas["time_sec"].unique())
    workers = n_workers if n_workers > 0 else (os.cpu_count() or 4)
    workers = min(workers, len(windows))
    log.info("Parallel workers    : %d  (of %d windows)", workers, len(windows))


    vote_count:  dict = {}
    window_rows: list = []
    cell_rows:   list = []
    detection_log_rows: list = []
    loc_input_rows:    list = []
    ue_loc_errors: list[float] = []
    ue_loc_stats:  list[dict]  = []

    # Pre-slice measurements by window to avoid re-filtering inside workers
    window_slices = [
        (k, T, meas[meas["time_sec"] == T])
        for k, T in enumerate(windows)
    ]

    log.info("Starting detection: %d windows  T_max=%.1f s  workers=%d",
             len(windows), windows[-1], workers)

    completed = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        futs = {
            executor.submit(
                _process_window_task,
                T, M_T, legal, ecgi_gt_fake,
                min_cells, n_ue_starts, lambda_ta,
                method, k_sigma, sigma_total, k,
            ): (k, T)
            for k, T, M_T in window_slices
        }

        for fut in concurrent.futures.as_completed(futs):
            k, T = futs[fut]
            try:
                res = fut.result()
            except Exception as exc:
                log.error("Window T=%.1f failed: %s", T, exc)
                continue

            completed += 1
            window_rows.append(res["window_row"])
            cell_rows.extend(res["cell_rows"])
            detection_log_rows.extend(res["det_log"])
            loc_input_rows.extend(res["loc_input"])
            ue_loc_errors.extend(res["ue_loc_errors"])
            ue_loc_stats.extend(res["ue_loc_stats"])
            for ecgi, is_fake in res["vote_entries"]:
                v = vote_count.setdefault(ecgi, [0, 0])
                v[1] += 1
                if is_fake:
                    v[0] += 1

            # Progress log (print immediately from main process)
            T_r, fakes_r, tau_r, avg_R_r = (
                res["T"], res["fakes"], res["tau"], res["avg_R"]
            )
            log.info(
                "T=%5.1f s | meas=%4d | UEs=%3d | R*=%.4f"
                " | cells=%2d | fakes=%d | τ=%.4f | %.1f ms  [%d/%d]",
                T_r, res["window_row"]["num_measurements"],
                res["valid_ues"], avg_R_r,
                len(res["scores"]), len(fakes_r), tau_r, res["rt_ms"],
                completed, len(windows),
            )
            for ecgi in fakes_r:
                log.warning(
                    "  [ALERT] T=%.1f  ECGI=%-22s  Score=%.4f > τ=%.4f (%s, k=%.1f)  →  FAKE",
                    T_r, ecgi, res["scores"][ecgi], tau_r, method, k_sigma,
                )

    # ── Final summary ─────────────────────────────────────────────────────────
    log.info("=== Final Detection Summary (method=%s, k=%.2f) ===", method, k_sigma)
    
    if ue_loc_errors:
        med_err = np.median(ue_loc_errors)
        avg_err = np.mean(ue_loc_errors)
        p90_err = np.percentile(ue_loc_errors, 90)
        log.info("UE Position Estimation Accuracy (N=%d samples):", len(ue_loc_errors))
        log.info("  Mean err:   %.1f m", avg_err)
        log.info("  Median err: %.1f m", med_err)
        log.info("  90th %%ile: %.1f m", p90_err)
        
        # Save exact error breakdown to CSV output
        df_ue_err = pd.DataFrame(ue_loc_stats)
        ue_err_path = results_dir / "ue_position_errors.csv"
        df_ue_err.to_csv(ue_err_path, index=False)
        log.info("ue_position_errors.csv → %s  (%d rows)", ue_err_path, len(df_ue_err))
        log.info("  90th %%ile: %.1f m", p90_err)
    
    tp = fp = tn = fn = 0
    for ecgi in sorted(vote_count):
        detected, total = vote_count[ecgi]
        gt     = int(ecgi_gt_fake.get(ecgi, 0))
        status = "FAKE" if detected > total / 2 else "LEGIT"
        if   status == "FAKE"  and     gt: tp += 1; tag = "TP"
        elif status == "FAKE"  and not gt: fp += 1; tag = "FP"
        elif status == "LEGIT" and     gt: fn += 1; tag = "FN"
        else:                              tn += 1; tag = "TN"
        log.info("  ECGI=%-22s  detected=%d/%d s  Status=%-5s  GT=%-5s  [%s]",
                 ecgi, detected, total, status, "FAKE" if gt else "LEGIT", tag)

    log.info("Confusion matrix:  TP=%d  FP=%d  FN=%d  TN=%d", tp, fp, fn, tn)
    if tp + fn > 0:
        dr  = 100.0 * tp / (tp + fn)
        fpr = 100.0 * fp / (fp + tn) if fp + tn > 0 else 0.0
        log.info("Detection rate: %.0f %%   False-positive rate: %.0f %%", dr, fpr)

    # ── Save outputs ──────────────────────────────────────────────────────────
    det_path = results_dir / "detection_results.csv"
    win_path = results_dir / "window_summary.csv"

    pd.DataFrame(cell_rows)[
        ["time_sec", "ecgi", "score", "threshold", "status", "is_fake_gt"]
    ].to_csv(det_path, index=False)
    pd.DataFrame(window_rows)[
        ["time_sec", "num_measurements", "avg_residual",
         "num_cells", "num_fake_cells", "runtime_ms"]
    ].to_csv(win_path, index=False)

    log.info("detection_results.csv → %s  (%d rows)", det_path, len(cell_rows))
    log.info("window_summary.csv    → %s  (%d rows)", win_path, len(window_rows))

    # ── Save localization input (Stage 1 → Stage 2 handoff) ───────────────────
    if loc_input_rows:
        loc_input_path = results_dir / "localization_input.csv"
        pd.DataFrame(loc_input_rows)[
            ["time_sec", "imsi",
             "estimated_ue_x", "estimated_ue_y", "estimated_ue_z",
             "ecgi", "cell_role", "rsrp_dbm", "timing_advance"]
        ].sort_values(["time_sec", "imsi"]).to_csv(loc_input_path, index=False)
        log.info("localization_input.csv → %s  (%d rows)",
                 loc_input_path, len(loc_input_rows))

    # ── Identify confirmed fake ECGIs (majority vote) ─────────────────────────
    # Only keep detection_log records for ECGIs that were fake in > 50 % of windows
    confirmed_fake_ecgis = {
        ecgi
        for ecgi, (detected_w, total_w) in vote_count.items()
        if detected_w > total_w / 2
    }
    log.info("Confirmed fake ECGIs (majority vote): %s", sorted(confirmed_fake_ecgis))

    # ── Save detection log (Stage 1 → Stage 2 handoff) ────────────────────────
    detection_log_rows = [
        r for r in detection_log_rows
        if r["ecgi"] in confirmed_fake_ecgis
    ]
    if detection_log_rows:
        det_log_path = results_dir / "detection_log.csv"
        pd.DataFrame(detection_log_rows)[
            ["timestamp", "ue_id",
             "estimated_ue_x", "estimated_ue_y", "estimated_ue_z",
             "rsrp_observed", "ecgi", "anomaly_score_wi"]
        ].to_csv(det_log_path, index=False)
        log.info(
            "detection_log.csv → %s  (%d rows)  "
            "→ run dec_map_localizer.py for Stage 2 localization",
            det_log_path, len(detection_log_rows),
        )
    else:
        log.info(
            "detection_log.csv: 0 rows — no confirmed fake ECGIs detected "
            "(confirmed=%s  raw_rows=%d)",
            sorted(confirmed_fake_ecgis), len(detection_log_rows),
        )

    log.info("=== Done ===")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Sliding-window fake cell detector (fake_cell_detection.md)"
    )
    ap.add_argument(
        "--results-dir", default="results",
        help="Folder containing measurements.csv and cell_database.csv (default: results)",
    )
    ap.add_argument(
        "--n-ue-starts", type=int, default=5,
        help="Number of optimiser starts for UE position estimation (default: 5)",
    )
    ap.add_argument(
        "--method", default="mad", choices=["mad", "iqr", "zscore"],
        help="Adaptive threshold method (default: mad)",
    )
    ap.add_argument(
        "--k-sigma", type=float, default=3.0,
        help="Multiplier k for the threshold formula (default: 3.0)",
    )
    ap.add_argument(
        "--min-cells", type=int, default=2,
        help="Minimum visible cells per UE sample (default: 2)",
    )
    ap.add_argument(
        "--lambda-ta", type=float, default=0.0,
        help="Weight for the TA distance constraint in UE estimation (default: 0.0)",
    )
    ap.add_argument(
        "--n-workers", type=int, default=0,
        help="Worker processes for parallel window detection "
             "(0 = auto = cpu_count, default: 0)",
    )
    ap.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO)",
    )
    ap.add_argument(
        "--sigma-total", type=float, default=9.1,
        help="σ_total [dB] for anomaly weight w_i in detection_log (default: 9.1)",
    )
    args = ap.parse_args()

    logging.getLogger().setLevel(args.log_level)
    log.setLevel(args.log_level)

    run_detection(
        results_dir=Path(args.results_dir),
        n_ue_starts=args.n_ue_starts,
        method=args.method,
        k_sigma=args.k_sigma,
        min_cells=args.min_cells,
        sigma_total=args.sigma_total,
        lambda_ta=args.lambda_ta,
        n_workers=args.n_workers,
    )
