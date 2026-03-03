#!/usr/bin/env python3
"""
fake_localizer.py

Neighbour-based fake base station localization.
Runs AFTER the geometry-consistency detector identifies a fake ECGI.

Algorithm
---------
Step 1  Legal-consistency filtering
        For each UE i with measured RSRP r_{i,f} for the fake ECGI:
            r̂_{i,L} = r_{i,s} − [PL(p_L, u_i) − PL(p_s, u_i)]
        Remove UE i if  |r_{i,f} − r̂_{i,L}| ≤ ε
        Remaining set: U_f'

Step 2  Multi-start Levenberg-Marquardt optimisation
        p̂_f = argmin_p  Σ_{u_i ∈ U_f'}  w_i · [Δr_i − ΔPL_i(p)]²
        where  Δr_i     = r_{i,f} − r_{i,s}
               ΔPL_i(p) = PL(p_s, u_i) − PL(p, u_i)
               w_i  ∝  |r_{i,f} − r̂_{i,L}|   (anomaly weight, optional)
        subject to  ‖p − p_L‖ > d_min

        Starting points (no grid search):
          1. Heuristic: anomaly-weighted centroid of UE estimated positions
          2. n_restarts random points uniformly sampled inside
             [legal_pos ± grid_margin] \ B(legal_pos, d_min)
        The run with the lowest final objective is kept.

Path-loss model (COST-231 urban macro, 2 GHz)
        PL [dB] = 128.1 + 37.6 · log10(d_km)

Public API
----------
    localize_fake(ecgi_f, ue_data, config) → dict
    write_localization_report(entries, out_path)

ue_data items (each a dict)
    u_pos        : (x, y)    UE estimated 2-D position [m]
    serving_rsrp : float     r_{i,s}  serving-cell mean RSRP [dBm]
    serving_pos  : (x,y,z)   serving cell 3-D position [m]
    fake_rsrp    : float     r_{i,f}  mean RSRP observed for ecgi_f [dBm]
    imsi         : int       UE identifier (logging only)

config keys
    Required:
        legal_pos        (x, y, z)  known 3-D position for ecgi_f [m]
    Optional (defaults in _DEFAULTS):
        epsilon           5.0  dB   legal-consistency filter threshold
        d_min            50.0  m    minimum offset from legal position
        grid_margin      500.0  m   random-start search radius around legal_pos
        n_restarts        10        number of random starting points
        random_seed       None      RNG seed (None = non-deterministic)
        max_iterations    200       LM max function evaluations per start
        use_huber        True       Huber IRLS loss
        huber_delta       1.0  dB   Huber soft-margin δ
        use_anomaly_weights True    w_i ∝ |r_{i,f} − r̂_{i,L}|
"""

import logging
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import least_squares

log = logging.getLogger("FakeLocalizer")

# 1/ln(10) — used in path-loss Jacobian
_LOG10E = 1.0 / np.log(10)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
_DEFAULTS: dict[str, Any] = {
    "epsilon":              5.0,
    "d_min":               50.0,
    "grid_margin":         500.0,
    "n_restarts":           10,
    "random_seed":          None,
    "max_iterations":       200,
    "use_huber":            True,
    "huber_delta":          1.0,
    "use_anomaly_weights":  True,
}


def _cfg(config: dict, key: str) -> Any:
    return config.get(key, _DEFAULTS[key])


# ---------------------------------------------------------------------------
# Path-loss  (COST-231 urban macro, matches ns-3 simulation)
#   PL [dB] = 128.1 + 37.6 · log10(d_km)
# ---------------------------------------------------------------------------
def _pl(dist_m: np.ndarray) -> np.ndarray:
    """COST-231 path-loss [dB]. Works on scalars and arrays."""
    return 128.1 + 37.6 * np.log10(np.maximum(dist_m, 1.0) / 1000.0)


def _dist3d(pos_xyz: tuple, u_xy: tuple) -> float:
    """3-D distance from BS at (x,y,z) to UE at (x,y,0)."""
    return float(np.sqrt(
        (pos_xyz[0] - u_xy[0]) ** 2 +
        (pos_xyz[1] - u_xy[1]) ** 2 +
        float(pos_xyz[2]) ** 2
    ))


# ---------------------------------------------------------------------------
# Step 1: Legal-consistency filtering
# ---------------------------------------------------------------------------
def _filter_ues(
    ue_data:   list[dict],
    legal_pos: tuple,
    epsilon:   float,
) -> tuple[list[dict], list[dict]]:
    """
    Partition ue_data into (kept, dropped).

    Kept UEs have |r_{i,f} − r̂_{i,L}| > ε — anomalous w.r.t. legal position,
    hence informative for locating the impersonating transmitter.

    Each returned dict is augmented with:
        _residual_legal : |r_{i,f} − r̂_{i,L}|
        _r_hat_legal    : r̂_{i,L}
    """
    kept, dropped = [], []
    for ue in ue_data:
        pl_L  = float(_pl(np.array([_dist3d(legal_pos,         ue["u_pos"])])))
        pl_s  = float(_pl(np.array([_dist3d(ue["serving_pos"], ue["u_pos"])])))
        r_hat = ue["serving_rsrp"] - (pl_L - pl_s)
        res   = abs(ue["fake_rsrp"] - r_hat)
        aug   = dict(ue, _residual_legal=res, _r_hat_legal=r_hat)
        (dropped if res <= epsilon else kept).append(aug)
    return kept, dropped


# ---------------------------------------------------------------------------
# Step 2: Starting-point generators
# ---------------------------------------------------------------------------
def _heuristic_start(
    ue_valid:   list[dict],
    legal_pos:  tuple,
    d_min:      float,
    weights:    "np.ndarray | None",
) -> np.ndarray:
    """
    Anomaly-weighted centroid of UE estimated positions.

    UEs with larger |r_{i,f} − r̂_{i,L}| are more strongly influenced by the
    fake BS (closer or in favourable geometry), so their weighted centroid is
    a physically-motivated starting estimate.

    If the centroid falls inside d_min, it is pushed radially outward.
    """
    u_xyz = np.array([u["u_pos"] for u in ue_valid], dtype=np.float64)
    w    = weights if weights is not None else np.ones(len(ue_valid)) / len(ue_valid)
    centroid = np.sum(w[:, np.newaxis] * u_xyz[:, :2], axis=0)

    lx, ly = float(legal_pos[0]), float(legal_pos[1])
    dist   = float(np.sqrt((centroid[0] - lx) ** 2 + (centroid[1] - ly) ** 2))
    if dist <= d_min:
        direction = centroid - np.array([lx, ly])
        if np.linalg.norm(direction) < 1.0:
            direction = np.array([1.0, 0.0])
        direction /= np.linalg.norm(direction)
        centroid = np.array([lx, ly]) + direction * (d_min + 10.0)

    return centroid


def _random_starts(
    n:          int,
    legal_pos:  tuple,
    grid_margin: float,
    d_min:      float,
    rng:        np.random.Generator,
) -> list[np.ndarray]:
    """
    Sample n starting points uniformly inside [legal_pos ± grid_margin]
    excluding the d_min exclusion zone around legal_pos.
    """
    lx, ly = float(legal_pos[0]), float(legal_pos[1])
    points: list[np.ndarray] = []
    attempts = 0
    max_attempts = n * 50
    while len(points) < n and attempts < max_attempts:
        px = lx + rng.uniform(-grid_margin, grid_margin)
        py = ly + rng.uniform(-grid_margin, grid_margin)
        if np.sqrt((px - lx) ** 2 + (py - ly) ** 2) > d_min:
            points.append(np.array([px, py]))
        attempts += 1
    return points


# ---------------------------------------------------------------------------
# Step 2: Levenberg-Marquardt optimizer  (scipy.optimize.least_squares)
#
# Minimises  L(p) = Σ_i w_i · ρ(r_i(p))  where
#   r_i(p) = Δr_i + PL(‖p − u_i‖) − PL(‖p_s − u_i‖)
#   ρ = squared loss  (use_huber=False)
#       Huber loss     (use_huber=True, method='trf', loss='huber')
#
# Anomaly weights are encoded by scaling residuals:  f_i = √w_i · r_i
# so that Σ f_i² = Σ w_i · r_i².
#
# Jacobian: ∂r_i/∂p_x = (37.6/(ln10·d_i)) · (p_x − u_x_i)/d_i
# ---------------------------------------------------------------------------
def _pl_jacobian(p: np.ndarray, u_xyz: np.ndarray, fake_z: float) -> np.ndarray:
    """Jacobian of PL(‖p − u_i‖) w.r.t. p. Shape: (N, 2)."""
    dx   = p[0] - u_xyz[:, 0]
    dy   = p[1] - u_xyz[:, 1]
    dz   = fake_z - u_xyz[:, 2]
    dist = np.maximum(np.sqrt(dx ** 2 + dy ** 2 + dz ** 2), 1.0)
    dpl  = 37.6 * _LOG10E / dist                      # dPL/dd [dB/m]
    return np.column_stack([dpl * dx / dist, dpl * dy / dist])


def _lm(
    p0:          np.ndarray,
    ue_valid:    list[dict],
    d_min:       float,
    legal_pos:   tuple,
    max_iter:    int,
    use_huber:   bool,
    huber_delta: float,
    weights:     "np.ndarray | None",
) -> tuple[np.ndarray, float, bool, str]:
    """
    Single-start optimizer using scipy.optimize.least_squares (TRF method).

    Anomaly weights are absorbed into √w_i-scaled residuals so that
    scipy minimises Σ w_i · ρ(r_i²).  Huber loss is handled natively by
    scipy (loss='huber', f_scale=huber_delta).

    Returns (p_opt, mse, converged, message).
    If the result violates d_min it is projected back to the boundary.
    """
    delta_r = np.array([u["fake_rsrp"] - u["serving_rsrp"] for u in ue_valid],
                       dtype=np.float64)
    pl_s    = np.array([
        float(_pl(np.array([_dist3d(u["serving_pos"], u["u_pos"])])))
        for u in ue_valid
    ], dtype=np.float64)
    u_xyz   = np.array([u["u_pos"] for u in ue_valid], dtype=np.float64)
    aw     = weights if weights is not None else np.ones(len(ue_valid)) / len(ue_valid)
    sqrt_w = np.sqrt(aw)
    fake_z = float(legal_pos[2])  # assume fake BS has same Z as legal BS

    def residuals(p: np.ndarray) -> np.ndarray:
        dist = np.maximum(
            np.sqrt((p[0] - u_xyz[:, 0]) ** 2 + (p[1] - u_xyz[:, 1]) ** 2 + (fake_z - u_xyz[:, 2]) ** 2), 1.0
        )
        return sqrt_w * (delta_r + _pl(dist) - pl_s)

    def jacobian(p: np.ndarray) -> np.ndarray:
        return sqrt_w[:, np.newaxis] * _pl_jacobian(p, u_xyz, fake_z)   # (N, 2)

    result = least_squares(
        fun=residuals,
        x0=p0.astype(np.float64),
        jac=jacobian,
        method="trf",
        loss="huber" if use_huber else "linear",
        f_scale=huber_delta if use_huber else 1.0,
        max_nfev=max_iter,
        ftol=1e-7,
        xtol=1e-5,
        gtol=1e-7,
    )

    p         = result.x.copy()
    converged = bool(result.success)
    msg       = result.message

    # Enforce d_min: project onto boundary if violated
    lx, ly = float(legal_pos[0]), float(legal_pos[1])
    dist_from_legal = float(np.sqrt((p[0] - lx) ** 2 + (p[1] - ly) ** 2))
    if dist_from_legal <= d_min:
        direction = p - np.array([lx, ly])
        if np.linalg.norm(direction) < 1.0:
            direction = np.array([1.0, 0.0])
        direction /= np.linalg.norm(direction)
        p         = np.array([lx, ly]) + direction * (d_min + 1.0)
        converged = False
        msg       = "projected to d_min boundary"

    # MSE at final position (unweighted, for comparison across starts)
    dist_f = np.maximum(
        np.sqrt((p[0] - u_xyz[:, 0]) ** 2 + (p[1] - u_xyz[:, 1]) ** 2 + (fake_z - u_xyz[:, 2]) ** 2), 1.0
    )
    mse = float(np.mean((delta_r + _pl(dist_f) - pl_s) ** 2))
    return p, mse, converged, msg


# ---------------------------------------------------------------------------
# Public API: localize_fake
# ---------------------------------------------------------------------------
def localize_fake(
    ecgi_f:  str,
    ue_data: list[dict],
    config:  dict,
) -> dict:
    """
    Locate a fake base station identified by ecgi_f.

    Parameters
    ----------
    ecgi_f  : detected fake ECGI string (e.g. "001-01-0000003").
    ue_data : list of per-UE dicts — see module docstring for required keys.
    config  : configuration dict — must include 'legal_pos' : (x, y, z).

    Returns
    -------
    dict with keys:
        estimated_position  (x, y)   estimated 2-D position of fake BS [m]
        residual_fake        float   MSE at optimised position [dB²]
        residual_legal       float   mean |r_{i,f} − r̂_{i,L}| for U_f' [dB]
        num_ue_total         int     UEs before filtering
        num_ue_used          int     UEs in U_f'
        converged            bool    LM convergence flag of best run
        runtime_ms           dict    {filtering, optimization, total}
        _debug               dict    internal details for report generation
    """
    t_total_0 = time.perf_counter()

    legal_pos   = config["legal_pos"]
    epsilon     = _cfg(config, "epsilon")
    d_min       = _cfg(config, "d_min")
    grid_margin = _cfg(config, "grid_margin")
    n_restarts  = int(_cfg(config, "n_restarts"))
    seed        = _cfg(config, "random_seed")
    max_iter    = _cfg(config, "max_iterations")
    use_huber   = _cfg(config, "use_huber")
    huber_delta = _cfg(config, "huber_delta")
    use_aw      = _cfg(config, "use_anomaly_weights")

    rng = np.random.default_rng(seed)

    def ts() -> str:
        return datetime.now().strftime("%H:%M:%S.%f")[:-3]

    n_total = len(ue_data)
    log.info("[%s] ── Localization ── ECGI=%s  UEs=%d", ts(), ecgi_f, n_total)
    log.info(
        "  ε=%.1f dB  d_min=%.0f m  margin=%.0f m  "
        "restarts=%d  use_huber=%s  δ=%.1f  max_iter=%d  anomaly_w=%s",
        epsilon, d_min, grid_margin,
        n_restarts, use_huber, huber_delta, max_iter, use_aw,
    )
    log.info("  Legal position p_L = (%.1f, %.1f, %.1f) m", *legal_pos)

    # ── Step 1: legal-consistency filtering ──────────────────────────────────
    t0 = time.perf_counter()
    ue_valid, ue_dropped = _filter_ues(ue_data, legal_pos, epsilon)
    t_filter = (time.perf_counter() - t0) * 1e3
    n_used = len(ue_valid)

    log.info(
        "[%s] Filtering: %.2f ms  kept=%d/%d  dropped=%d",
        ts(), t_filter, n_used, n_total, len(ue_dropped),
    )
    for ue in sorted(ue_data, key=lambda u: u.get("imsi", 0)):
        aug = next(
            (u for u in ue_valid + ue_dropped if u.get("imsi") == ue.get("imsi")),
            ue,
        )
        tag = "KEPT   " if aug in ue_valid else "DROPPED"
        log.debug(
            "    [%s] IMSI=%-12d  |r_f−r̂_L|=%6.3f dB  (ε=%.1f)",
            tag, ue.get("imsi", 0), aug.get("_residual_legal", 0.0), epsilon,
        )

    if n_used == 0:
        log.warning("[%s] All UEs filtered out — cannot localize ECGI=%s.", ts(), ecgi_f)
        t_total = (time.perf_counter() - t_total_0) * 1e3
        return {
            "estimated_position": (float(legal_pos[0]), float(legal_pos[1])),
            "residual_fake":      float("inf"),
            "residual_legal":     0.0,
            "num_ue_total":       n_total,
            "num_ue_used":        0,
            "converged":          False,
            "runtime_ms": {
                "filtering":    round(t_filter, 3),
                "optimization": 0.0,
                "total":        round(t_total, 3),
            },
            "_debug": {
                "status":           "no_ue_after_filter",
                "n_starts":         0,
                "best_start_idx":   -1,
                "lm_method":        "N/A",
                "lm_message":       "",
                "dist_from_legal":  0.0,
                "all_objectives":   [],
            },
        }

    res_legal = float(np.mean([u["_residual_legal"] for u in ue_valid]))
    log.info(
        "[%s] Mean |r_f−r̂_L| for U_f' (residual under legal hypothesis): %.4f dB",
        ts(), res_legal,
    )

    # ── Anomaly weights ───────────────────────────────────────────────────────
    anomaly_vals = np.array([u["_residual_legal"] for u in ue_valid], dtype=np.float64)
    if use_aw and anomaly_vals.sum() > 1e-9:
        aw = anomaly_vals / anomaly_vals.sum()
        eff_n = float(1.0 / (aw ** 2).sum())
        log.info(
            "[%s] Anomaly weights: min=%.3f  max=%.3f  eff_N=%.1f / %d",
            ts(), float(aw.min()), float(aw.max()), eff_n, n_used,
        )
    else:
        aw = None

    # ── Step 2: multi-start LM ────────────────────────────────────────────────
    t0 = time.perf_counter()

    p_heuristic = _heuristic_start(ue_valid, legal_pos, d_min, aw)
    p_randoms   = _random_starts(n_restarts, legal_pos, grid_margin, d_min, rng)
    candidates  = [p_heuristic] + p_randoms
    n_starts    = len(candidates)

    log.info(
        "[%s] Multi-start LM: 1 heuristic + %d random = %d total starts",
        ts(), len(p_randoms), n_starts,
    )

    best_p    = candidates[0].copy()
    best_obj  = float("inf")
    best_conv = False
    best_msg  = ""
    best_idx  = 0
    all_obj: list[float] = []

    lm_label = "LM+Huber(IRLS)" if use_huber else "LM"

    for idx, p0 in enumerate(candidates):
        p_opt, obj, conv, msg = _lm(
            p0, ue_valid, d_min, legal_pos,
            max_iter, use_huber, huber_delta, weights=aw,
        )
        all_obj.append(obj)
        label = "heuristic" if idx == 0 else f"random[{idx}]"
        log.debug(
            "    start[%d] %-14s p0=(%.0f,%.0f)  →  p=(%.1f,%.1f)  "
            "MSE=%.4f  conv=%s  %s",
            idx, label, p0[0], p0[1],
            p_opt[0], p_opt[1], obj, conv, msg,
        )
        if obj < best_obj:
            best_p, best_obj, best_conv, best_msg = p_opt, obj, conv, msg
            best_idx = idx

    t_opt   = (time.perf_counter() - t0) * 1e3
    t_total = (time.perf_counter() - t_total_0) * 1e3

    dist_from_legal = float(np.sqrt(
        (best_p[0] - legal_pos[0]) ** 2 + (best_p[1] - legal_pos[1]) ** 2
    ))
    best_label = "heuristic" if best_idx == 0 else f"random[{best_idx}]"
    log.info(
        "[%s] Optimization: %.2f ms  method=%s  starts=%d  best=%s  conv=%s  msg=%r",
        ts(), t_opt, lm_label, n_starts, best_label, best_conv, best_msg,
    )
    log.info(
        "[%s] ── RESULT ──  p̂_f=(%.2f, %.2f) m  "
        "dist_from_legal=%.1f m  MSE=%.6f dB²  total=%.2f ms",
        ts(), best_p[0], best_p[1], dist_from_legal, best_obj, t_total,
    )

    return {
        "estimated_position": (float(best_p[0]), float(best_p[1])),
        "residual_fake":      float(best_obj),
        "residual_legal":     float(res_legal),
        "num_ue_total":       n_total,
        "num_ue_used":        n_used,
        "converged":          bool(best_conv),
        "runtime_ms": {
            "filtering":    round(t_filter, 3),
            "optimization": round(t_opt,    3),
            "total":        round(t_total,  3),
        },
        "_debug": {
            "status":          "ok",
            "n_starts":        n_starts,
            "best_start_idx":  best_idx,
            "lm_method":       lm_label,
            "lm_message":      best_msg,
            "dist_from_legal": float(dist_from_legal),
            "all_objectives":  [round(v, 6) for v in all_obj],
        },
    }


# ---------------------------------------------------------------------------
# Report generator
# ---------------------------------------------------------------------------
def write_localization_report(
    entries:  list[dict],   # each: {ecgi, time_sec, result, config}
    out_path: Path,
) -> None:
    """
    Generate localization_report.md from all collected localization results.
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines: list[str] = []
    a = lines.append

    a("# Fake Base Station Localization Report")
    a("")
    a(f"Generated: {now}")
    a("")

    # ── Configuration ────────────────────────────────────────────────────────
    if entries:
        cfg = entries[0]["config"]
        loss_str = (
            f"LM + Huber IRLS (δ = {_cfg(cfg, 'huber_delta')} dB)"
            if _cfg(cfg, "use_huber") else "LM (linear)"
        )
        a("## Configuration")
        a("")
        a("| Parameter | Value |")
        a("|-----------|-------|")
        a(f"| ε — legal-consistency threshold | {_cfg(cfg, 'epsilon')} dB |")
        a(f"| d_min — minimum offset from legal position | {_cfg(cfg, 'd_min')} m |")
        a(f"| Search margin (±) | {_cfg(cfg, 'grid_margin')} m |")
        a(f"| Random restarts | {_cfg(cfg, 'n_restarts')} |")
        a(f"| Max LM iterations (per start) | {_cfg(cfg, 'max_iterations')} |")
        a(f"| Loss function | {loss_str} |")
        a(f"| Anomaly weighting | {_cfg(cfg, 'use_anomaly_weights')} |")
        a("")

    # ── Mathematical formulation ──────────────────────────────────────────────
    a("## Mathematical Formulation")
    a("")
    a("### Path-Loss Model")
    a("")
    a("$$")
    a(r"\mathrm{PL}(d)\;=\;128.1 + 37.6\,\log_{10}\!\left(\frac{d}{1\,\mathrm{km}}\right)")
    a("$$")
    a("")
    a("### Step 1 — Legal-Consistency Filtering")
    a("")
    a(r"Predicted RSRP under the legal hypothesis for UE $i$:")
    a("")
    a("$$")
    a(r"\hat{r}_{i,L} \;=\; r_{i,s} \;-\; "
      r"\bigl[\,\mathrm{PL}(\mathbf{p}_L,\,\mathbf{u}_i) "
      r"- \mathrm{PL}(\mathbf{p}_s,\,\mathbf{u}_i)\,\bigr]")
    a("$$")
    a("")
    a(r"UE $i$ is **removed** if  $|\,r_{i,f} - \hat{r}_{i,L}\,| \le \varepsilon$.")
    a("")
    a("### Step 2 — Multi-Start Optimisation")
    a("")
    a("$$")
    a(r"\hat{\mathbf{p}}_f \;=\; \arg\min_{\mathbf{p}} "
      r"\sum_{u_i\,\in\,\mathcal{U}_f'} "
      r"w_i\,\left[\,\underbrace{r_{i,f} - r_{i,s}}_{\Delta r_i} "
      r"\;-\; \underbrace{\bigl(\mathrm{PL}(\mathbf{p}_s,\mathbf{u}_i) "
      r"- \mathrm{PL}(\mathbf{p},\mathbf{u}_i)\bigr)}_{\Delta\mathrm{PL}_i(\mathbf{p})}"
      r"\,\right]^2")
    a("$$")
    a("")
    a(r"where $w_i \propto |\,r_{i,f} - \hat{r}_{i,L}\,|$ (anomaly weight). "
      r"Subject to $\|\mathbf{p} - \mathbf{p}_L\| > d_{\min}$.")
    a("")
    a("Starting points:")
    a("1. **Heuristic** — anomaly-weighted centroid of UE estimated positions.")
    a("2. **Random** — $n_{\\text{restarts}}$ points sampled uniformly inside "
      "$\\mathbf{p}_L \\pm \\text{margin}$ excluding $d_{\\min}$.")
    a("")
    a("Each start runs full Levenberg–Marquardt. Best objective is kept.")
    a("")

    # ── Per-ECGI results ──────────────────────────────────────────────────────
    a("## Localisation Results")
    a("")

    by_ecgi: dict[str, list] = defaultdict(list)
    for e in entries:
        by_ecgi[e["ecgi"]].append(e)

    for ecgi, ecgi_entries in sorted(by_ecgi.items()):
        a(f"### ECGI `{ecgi}`")
        a("")
        cfg = ecgi_entries[0]["config"]
        lp  = cfg.get("legal_pos", (0.0, 0.0, 0.0))
        a(f"- Known legal position: **({lp[0]:.1f}, {lp[1]:.1f})** m  "
          f"(antenna height {lp[2]:.1f} m)")
        a("")

        a("#### Per-Window Results")
        a("")
        a("| Time (s) | UEs before filter | UEs used | "
          "Estimated position (m) | Residual fake (dB²) | "
          "Residual legal (dB) | Converged | Runtime (ms) |")
        a("|:--------:|:-----------------:|:--------:|"
          ":---------------------:|:-------------------:|"
          ":------------------:|:---------:|:------------:|")
        for e in sorted(ecgi_entries, key=lambda x: x["time_sec"]):
            r  = e["result"]
            px, py = r["estimated_position"]
            cv = "✓" if r["converged"] else "✗"
            res_f = (
                f"{r['residual_fake']:.4f}"
                if r["residual_fake"] < 1e10 else "∞"
            )
            a(f"| {e['time_sec']:.1f} | {r['num_ue_total']} | {r['num_ue_used']} | "
              f"({px:.1f}, {py:.1f}) | {res_f} | "
              f"{r['residual_legal']:.4f} | {cv} | "
              f"{r['runtime_ms']['total']:.1f} |")
        a("")

        last = ecgi_entries[-1]["result"]
        a("#### Residual Comparison (last window)")
        a("")
        a("| Hypothesis | Value |")
        a("|-----------|-------|")
        a(f"| $\\mathcal{{H}}_0$ — mean $|r_{{i,f}} - \\hat{{r}}_{{i,L}}|$ for $\\mathcal{{U}}_f'$ | "
          f"{last['residual_legal']:.4f} dB |")
        res_fake_str = (
            f"{last['residual_fake']:.6f} dB²"
            if last["residual_fake"] < 1e10 else "∞"
        )
        a(f"| $\\mathcal{{H}}_1$ — MSE at $\\hat{{\\mathbf{{p}}}}_f$ | {res_fake_str} |")
        a("")

        a("#### Runtime Breakdown (last window)")
        a("")
        rt = last["runtime_ms"]
        a("| Stage | Time (ms) |")
        a("|-------|----------:|")
        a(f"| Filtering | {rt['filtering']:.2f} |")
        a(f"| Multi-start optimisation | {rt['optimization']:.2f} |")
        a(f"| **Total** | **{rt['total']:.2f}** |")
        a("")

        dbg = last.get("_debug", {})
        a(f"- Optimiser: **{dbg.get('lm_method', 'N/A')}**")
        a(f"- Starts: {dbg.get('n_starts', '?')}  "
          f"(best = start [{dbg.get('best_start_idx', '?')}])")
        a(f"- Convergence message: *{dbg.get('lm_message', '')}*")
        a(f"- Distance from legal position: {dbg.get('dist_from_legal', 0):.1f} m")
        objs = dbg.get("all_objectives", [])
        if objs:
            a(f"- All start objectives: {objs}")
        a("")

    a("## Summary")
    a("")
    if entries:
        conv_n = sum(1 for e in entries if e["result"]["converged"])
        avg_rt = float(np.mean(
            [e["result"]["runtime_ms"]["total"] for e in entries]
        ))
        a(f"- Time windows processed: **{len(entries)}**")
        a(f"- Unique fake ECGIs localised: **{len(by_ecgi)}**")
        a(f"- Converged windows: **{conv_n} / {len(entries)}**")
        a(f"- Mean runtime per window: **{avg_rt:.2f} ms**")
    a("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    log.info("Localization report written → %s", out_path)


# ---------------------------------------------------------------------------
# BayesianLocalizer
#
# Implements the recursive Bayesian grid-based algorithm in
# fake_cell_localization.md §3–§10.
#
# State: log-posterior  log P(x_j | Z_{1:t})  over a 2D grid G.
#
# Update rule (log domain, numerically stable):
#   log P(x_j | Z_{1:t}) ← log P(x_j | Z_{1:t-1}) + log L_total(x_j)
#
# Likelihood for one UE i at grid cell x_j  (§7):
#   r̂_i(x_j) = P_tx − PL(d_ij)              predicted RSRP
#   σ_loc     = (37.6/ln10) · σ_u / d_ij     location-induced std [dB]
#   σ_tot²    = σ_r² + σ_loc²                 total variance
#   log L_i   = −(r_{i,f} − r̂_i)² / (2σ_tot²)
#
# Joint (§8):  log L_total(x_j) = Σ_i log L_i(x_j)
# MAP   (§10): x_f* = argmax_j log P(x_j | Z_{1:t})
# ---------------------------------------------------------------------------
class BayesianLocalizer:
    """
    Recursive Bayesian grid-based fake BS localization.

    Parameters (config dict)
    ------------------------
    area_bounds   : (x_min, x_max, y_min, y_max)  grid extent [m]
    grid_res      : float   grid cell size [m]          default 50
    tx_power      : float   P_tx [dBm]
    sigma_signal  : float   σ_r — RSRP noise std [dB]   default 8.0
    sigma_ue      : float   σ_u — UE position std [m]   default 50.0

    Usage
    -----
    bl = BayesianLocalizer(cfg)
    for each time window:
        ue_batch = [{u_pos: (x,y,...), fake_rsrp: float}, ...]
        bl.update(ue_batch)
    x_est, y_est = bl.map_estimate()
    """

    def __init__(self, config: dict) -> None:
        self.cfg = config
        self._build_grid()
        # log-prior: uniform → log P = 0 everywhere (normalisation constant absorbed)
        self._log_post  = np.zeros(self.M, dtype=np.float64)
        self.n_updates  = 0
        # history: list of (n_ue_samples, entropy, map_x, map_y) — one entry per UE update
        self._history: list = []

    # ── Grid construction ─────────────────────────────────────────────────────
    def _build_grid(self) -> None:
        x_min, x_max, y_min, y_max = self.cfg["area_bounds"]
        res = float(self.cfg.get("grid_res", 50.0))
        xs  = np.arange(x_min, x_max + res * 0.5, res)
        ys  = np.arange(y_min, y_max + res * 0.5, res)
        XX, YY       = np.meshgrid(xs, ys)
        self._gx     = XX.ravel().astype(np.float64)   # (M,)
        self._gy     = YY.ravel().astype(np.float64)   # (M,)
        self.M       = int(len(self._gx))
        self.grid_res = res
        # Store for 2D heatmap reconstruction
        self._xs     = xs
        self._ys     = ys
        self._nx     = len(xs)
        self._ny     = len(ys)
        log.debug("BayesianLocalizer: grid %dx%d = %d cells, res=%.0f m",
                  len(xs), len(ys), self.M, res)

    # ── DEC-MAP Bayesian update ───────────────────────────────────────────────
    def update(self, ue_batch: list[dict], w_i_threshold: float = 0.0) -> None:
        """
        DEC-MAP weighted Bayesian update (one UE at a time).

        Each item in ue_batch must contain:
            u_pos          : (x, y, ...)   estimated UE 2D position [m]
            fake_rsrp      : float         r_{i,f} [dBm]
            anomaly_weight : float         w_i in [0,1]  (optional, default=1.0)

        DEC-MAP formula (max-normalized, matching reference implementation):
            effective_L(g) = w_i * L_maxnorm(g) + (1 - w_i)
            P_new(g)       ∝ effective_L(g) * P_old(g)

        where L_maxnorm = L / max(L)  so L_maxnorm in (0, 1], max = 1.

        Properties:
            w_i = 1.0  -> effective_L = L_maxnorm  (standard Bayesian update)
            w_i = 0.0  -> effective_L = 1 constant -> P_new = P_old (no update)
            w_i = 0.5  -> soft mixture: only high-likelihood cells gain weight

        w_i_threshold : skip observations with w_i < threshold (default 0.0)
        After each accepted UE, (n_updates, H, map_x, map_y) is recorded in _history.
        """
        if not ue_batch:
            return

        P_tx    = float(self.cfg["tx_power"])
        sigma_r = float(self.cfg.get("sigma_signal", 8.0))
        sigma_u = float(self.cfg.get("sigma_ue",     50.0))
        # Configurable path loss: default COST-231 urban macro
        # Override via cfg: n_pl (exponent) and pl_intercept
        n_pl         = float(self.cfg.get("n_pl",         3.76))
        pl_intercept = float(self.cfg.get("pl_intercept", P_tx - 128.1 + 37.6 * 3))

        for ue in ue_batch:
            u_xy = np.array([ue["u_pos"][0], ue["u_pos"][1]], dtype=np.float64)
            r_if = float(ue["fake_rsrp"])
            w_i  = float(np.clip(ue.get("anomaly_weight", 1.0), 0.0, 1.0))

            # Skip low-confidence observations (reduce LBS noise)
            if w_i < w_i_threshold:
                continue

            # Distance from UE to each grid cell  shape: (M,)
            d_ij = np.maximum(
                np.sqrt((u_xy[0] - self._gx) ** 2 + (u_xy[1] - self._gy) ** 2),
                1.0,
            )

            # Expected RSRP at each grid cell
            # r_hat = intercept - 10*n_pl*log10(d)
            r_hat = pl_intercept - 10.0 * n_pl * np.log10(d_ij)  # (M,)

            # Combined noise: sigma_r (signal) + sigma_loc (UE position error)
            sigma_loc    = (10.0 * n_pl / np.log(10)) * (sigma_u / d_ij)  # (M,) [dB]
            sigma_tot_sq = sigma_r ** 2 + sigma_loc ** 2

            # Log-likelihood  log L(z_i | g_j)
            residual = r_if - r_hat                              # (M,)
            log_L    = -0.5 * residual ** 2 / sigma_tot_sq      # (M,)

            # ── DEC-MAP mixture with max-normalized likelihood ────────────────
            # L_maxnorm(g) = exp(log_L(g) - max(log_L))  in (0, 1],  max = 1
            # effective_L(g) = w_i * L_maxnorm(g) + (1 - w_i)       in [(1-w_i), 1]
            #
            # When w_i = 0: effective_L = 1 everywhere -> no update (correct!)
            # When w_i = 1: effective_L = L_maxnorm    -> standard Bayesian
            #
            # Taking log: log(effective_L) = log(w_i * exp(log_L - max_L) + (1 - w_i))
            # Always finite because argument is in (0, 1] when w_i in [0,1].
            log_L_shifted = log_L - log_L.max()                 # max = 0
            L_maxnorm     = np.exp(log_L_shifted)               # max = 1
            effective_mix = w_i * L_maxnorm + (1.0 - w_i)      # in [(1-w_i), 1]

            log_effective  = np.log(np.maximum(effective_mix, 1e-300))
            self._log_post += log_effective

            # Numerical stability: keep max = 0
            self._log_post -= self._log_post.max()

            self.n_updates += 1
            mx, my = self.map_estimate()
            H      = self.entropy()
            self._history.append((self.n_updates, H, mx, my))


    # ── MAP estimate (§10) ────────────────────────────────────────────────────
    def map_estimate(self) -> tuple[float, float]:
        """x_f* = argmax_j P(x_j | Z_{1:t}).  Returns (x, y) [m]."""
        j_star = int(np.argmax(self._log_post))
        return float(self._gx[j_star]), float(self._gy[j_star])

    # ── Entropy (§11 Step 5) ─────────────────────────────────────────────────
    def entropy(self) -> float:
        """Shannon entropy H = −Σ_j p_j log p_j of the posterior."""
        p = np.exp(self._log_post - self._log_post.max())
        p /= p.sum()
        p = p[p > 1e-300]
        return float(-np.sum(p * np.log(p)))

    # ── Normalised posterior ──────────────────────────────────────────────────
    def posterior(self) -> np.ndarray:
        """Normalised posterior P(x_j | Z_{1:t}) — shape (M,)."""
        p = np.exp(self._log_post - self._log_post.max())
        return p / p.sum()

    # ── Theoretical convergence bound ─────────────────────────────────────────
    @staticmethod
    def theoretical_rmse(
        N:             np.ndarray,
        sigma_d_total: float = 209.5,
        G_gdop:        float = 1.5,
    ) -> np.ndarray:
        """
        ε(N) = G · σ_d_total / √N

        Default values from DEC-MAP theoretical derivation:
            σ_d_total ≈ 209.5 m   (§2.3 in the formulation)
            G         = 1.5       (geometric dilution of precision)
        """
        N = np.asarray(N, dtype=float)
        return G_gdop * sigma_d_total / np.sqrt(np.maximum(N, 1))

    # ── Validation plots (3-panel) ─────────────────────────────────────────────
    def plot_results(
        self,
        true_pos:      "tuple | None" = None,
        sigma_d_total: float = 209.5,
        G_gdop:        float = 1.5,
        save_path:     "str | None" = None,
        title_suffix:  str = "",
    ) -> None:
        """
        Generate a 3-panel DEC-MAP validation figure:
            Panel 1 — 2D posterior heatmap (with true & MAP positions)
            Panel 2 — Empirical RMSE vs N  +  theoretical bound ε(N)
            Panel 3 — Posterior entropy H(N) vs N

        Requires true_pos (x, y) to compute RMSE.
        """
        try:
            import matplotlib
            matplotlib.use("Agg")   # non-interactive backend (safe for Docker)
            import matplotlib.pyplot as plt
            from matplotlib.gridspec import GridSpec
        except ImportError:
            log.warning("matplotlib not available — skipping plot_results()")
            return

        if not self._history:
            log.warning("BayesianLocalizer.plot_results: no history recorded (no updates yet)")
            return

        ns     = np.array([h[0] for h in self._history], dtype=float)
        Hs     = np.array([h[1] for h in self._history], dtype=float)
        map_xs = np.array([h[2] for h in self._history], dtype=float)
        map_ys = np.array([h[3] for h in self._history], dtype=float)

        # ── Convergence curves ──────────────────────────────────────────────
        has_true = true_pos is not None
        if has_true:
            rmse        = np.sqrt((map_xs - true_pos[0]) ** 2 +
                                  (map_ys - true_pos[1]) ** 2)
            theo_bound  = self.theoretical_rmse(ns, sigma_d_total, G_gdop)
        else:
            rmse = theo_bound = None

        # ── Layout ──────────────────────────────────────────────────────────
        fig = plt.figure(figsize=(17, 5))
        gs  = GridSpec(1, 3, figure=fig, wspace=0.38)

        # ── Panel 1: Heatmap ────────────────────────────────────────────────
        ax1  = fig.add_subplot(gs[0, 0])
        # Use log-posterior for display: after many updates the distribution
        # collapses to a tight peak — linear scale shows everything else as 0 (black).
        # Log scale reveals the full probability landscape.
        log_post_2d = self._log_post.reshape(self._ny, self._nx).copy()
        log_post_2d -= log_post_2d.max()              # shift max to 0
        vmin_db = -30.0                               # show top-30 dB of dynamic range
        im   = ax1.imshow(
            log_post_2d,
            origin="lower",
            extent=[self._xs[0], self._xs[-1], self._ys[0], self._ys[-1]],
            cmap="hot",
            aspect="auto",
            vmin=vmin_db,
            vmax=0.0,
        )
        plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04, label="log P(g_j) [dB-rel]")
        mx, my = self.map_estimate()
        ax1.scatter(mx, my, c="lime", marker="x", s=180, linewidths=2.5,
                    zorder=6, label=f"MAP ({mx:.0f},{my:.0f})")
        if has_true:
            ax1.scatter(true_pos[0], true_pos[1], c="cyan", marker="*", s=250,
                        zorder=7, label=f"True ({true_pos[0]:.0f},{true_pos[1]:.0f})")
        ax1.set_title("DEC-MAP Log-Posterior Heatmap", fontsize=11)
        ax1.set_xlabel("X [m]")
        ax1.set_ylabel("Y [m]")
        ax1.legend(loc="upper right", fontsize=7)

        # ── Panel 2: RMSE vs N ───────────────────────────────────────────────
        ax2 = fig.add_subplot(gs[0, 1])
        if has_true:
            ax2.plot(ns, rmse, "b-", linewidth=1.4, label="Empirical RMSE")
            ax2.plot(ns, theo_bound, "r--", linewidth=2.0,
                     label=f"Theoretical ε(N) = G·σ/√N\n"
                           f"(G={G_gdop}, σ_d={sigma_d_total:.0f} m)")
            # Annotate N=100
            if ns[-1] >= 100:
                idx100 = np.argmin(np.abs(ns - 100))
                ax2.annotate(
                    f"N=100\nRMSE≈{rmse[idx100]:.0f} m\nTheory≈{theo_bound[idx100]:.0f} m",
                    xy=(ns[idx100], rmse[idx100]),
                    xytext=(ns[idx100] * 1.05, rmse[idx100] * 1.3),
                    fontsize=7,
                    arrowprops=dict(arrowstyle="->", color="gray"),
                )
        else:
            ax2.text(0.5, 0.5, "No true position\n(RMSE unavailable)",
                     ha="center", va="center", transform=ax2.transAxes)
        ax2.set_xlabel("N (FBS UE samples)")
        ax2.set_ylabel("Localization Error [m]")
        ax2.set_title("RMSE vs N  +  Theoretical Bound", fontsize=11)
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.35)

        # ── Panel 3: Entropy vs N ────────────────────────────────────────────
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.plot(ns, Hs, "g-", linewidth=1.4)
        ax3.set_xlabel("N (FBS UE samples)")
        ax3.set_ylabel("Shannon Entropy H(N)")
        ax3.set_title("Posterior Entropy vs N", fontsize=11)
        ax3.grid(True, alpha=0.35)

        suf = f" — {title_suffix}" if title_suffix else ""
        fig.suptitle(f"DEC-MAP Localization: Theoretical Validation{suf}",
                     fontsize=13, fontweight="bold")

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            log.info("DEC-MAP validation plot saved → %s", save_path)
        plt.close(fig)
