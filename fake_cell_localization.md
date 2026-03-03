# FBS Localization Algorithm (ECGI-Filtered Version)
*(Aligned with Detection Output: Known Fake ECGI)*

---

# 1. Assumption

Detection module has already identified:

    ecgi_f

Localization module will:

- Select ONLY UEs whose Measurement Report contains `ecgi_f`
- Use the reported RSRP `r_{i,f}` corresponding to that neighbour cell
- Ignore all other UEs

This ensures geometric consistency and avoids contamination from unrelated data.

---

# 2. UE Selection

Let:

    U_all = {all UEs in system}
    U_f   = { UE_i | ecgi_f appears in neighbour list of UE_i }

Only `U_f` is used in localization.

If |U_f| < N_min:
    Do not run localization (insufficient observability)

---

# 3. Grid Definition

Monitoring area:

    A ⊂ R²

Discretized into:

    G = { x_1, x_2, ..., x_M }

Each grid centre:

    x_j = (x_j, y_j)

Grid resolution: `grid_res` [m] (default 50 m).

---

# 4. Observation Model

For each selected UE_i in U_f:

Observed RSRP from `ecgi_f`:

    r_{i,f}

Estimated UE position (re-estimated without ecgi_f, see §5):

    u_i = (x_i, y_i)

---

# 5. UE Position Re-Estimation

Before running the Bayesian update, each UE position is re-estimated
**excluding `ecgi_f`** to remove bias from the fake signal:

    r_c = mean RSRP per ECGI,  drop ecgi_f entry
    μ   = mean({r_c}),   σ = std({r_c})
    r̃_c = (r_c − μ) / σ

    u_i* = argmin_u  R(u) = (1/N) Σ_c ( r̃_c − r̂̃_c(u) )²
           r̂_c(u)   = P_c − PL(dist(p_c, u))
           r̂̃_c(u)   = ( r̂_c(u) − μ ) / σ

Solved by multi-start L-BFGS-B (same optimizer as detection step).

---

# 6. Propagation Model

Path-loss (COST-231 urban macro, 2 GHz):

    PL(d) = 128.1 + 37.6 · log10(d / 1000)   [dB],  d in metres

Expected received power if FBS at grid centre x_j:

    r̂_i(x_j) = P_tx − PL(d_ij)

where:

    d_ij = || u_i − x_j ||        (2D Euclidean distance)
    P_tx  = transmit power [dBm]  (same as legal cells by default)

---

# 7. Robust Likelihood (UE Position Uncertainty)

UE position uncertainty is propagated through the path-loss model:

    σ_loc = (37.6 / ln10) · (σ_u / d_ij)

where `σ_u` [m] is the UE position estimation uncertainty (default 50 m).

Total variance (measurement noise + location-induced):

    σ_tot² = σ_r² + σ_loc²

where `σ_r` [dB] is the RSRP measurement noise std (default 8 dB).

Log-likelihood for UE i at grid cell x_j:

    log L_i(x_j) =
        − (r_{i,f} − r̂_i(x_j))²
          / (2 · σ_tot²)

---

# 8. Multi-UE Fusion (Single Time Snapshot)

For a batch of U_f at time t:

Joint log-likelihood (sum over UEs, numerically stable):

    log L_total(x_j) =
        Σ_{i ∈ U_f}  log L_i(x_j)

    = Σ_{i ∈ U_f}  − (r_{i,f} − r̂_i(x_j))²
                     / (2 · σ_tot²)

---

# 9. Recursive Bayesian Update

Posterior (log domain):

    log P(x_j | Z_{1:t}) =
        log P(x_j | Z_{1:t-1}) + log L_total(x_j)

Normalization (numerical stability — subtract max before exp):

    log P ← log P − max(log P)

Initialization (uniform prior):

    log P(x_j | Z_{1:0}) = 0   ∀ j      (uniform: log(1/M) + const)

---

# 10. MAP Estimation

Estimated FBS position:

    x_f* = argmax_{x_j ∈ G}  P(x_j | Z_{1:t})
         = argmax_{x_j ∈ G}  log P(x_j | Z_{1:t})

---

# 11. Algorithm Specification

Algorithm: DEC-MAP-ECGI (Bayesian Grid)

INPUT:
- `ecgi_f`           detected fake ECGI
- Grid G             built from area bounds + `grid_res`
- Measurement stream per-window batches of (u_i, r_{i,f})
- `P_tx`             transmit power [dBm]
- `σ_r`              RSRP noise std [dB]
- `σ_u`              UE position uncertainty std [m]

OUTPUT:
- x_f*               estimated FBS 2D location [m]
- Posterior P(x_j)   probability map over grid G
- Entropy H          posterior concentration metric
- n_updates          number of time windows processed

---

Step 1 — UE Filtering (per window)

    U_f = []
    For each UE_i:
        If ecgi_f in neighbour list of UE_i:
            Re-estimate u_i without ecgi_f  (§5)
            Append (u_i, r_{i,f}) to U_f

    If len(U_f) < N_min:
        Skip window (insufficient UE support)

---

Step 2 — Initialization (once, before first window)

    For each x_j ∈ G:
        log_P[x_j] = 0      (uniform log-prior)

---

Step 3 — Bayesian Update (per window)

    For each grid cell x_j:

        log_L = 0

        For each UE_i in U_f:

            d_ij      = || u_i − x_j ||

            r̂_i      = P_tx − 128.1 − 37.6 · log10(d_ij / 1000)

            σ_loc     = (37.6 / ln10) · (σ_u / d_ij)

            σ_tot²    = σ_r² + σ_loc²

            log_L    +=  − (r_{i,f} − r̂_i)² / (2 · σ_tot²)

        log_P[x_j]  += log_L

    log_P ← log_P − max(log_P)     # numerical stability

---

Step 4 — MAP Estimate (after each window or at end)

    x_f* = x_j  where  log_P[x_j] = max(log_P)

---

Step 5 — Entropy Logging

    p   = softmax(log_P)      (normalize to probability)
    H   = − Σ_j  p_j · log(p_j)

    Log:
    - x_f* (MAP estimate)
    - H (posterior entropy — decreases as estimate concentrates)
    - |U_f| per window
    - Runtime per window

---

# 12. Observability Condition

Localization reliable when:

- |U_f| ≥ 3  per window
- UEs are geometrically distributed around x_f*
- Not collinear

Accuracy improves with:

    σ_loc ∝ σ_r / sqrt(|U_f|)    (per-window CRLB approximation)

Bayesian accumulation over T windows reduces effective noise:

    σ_eff ∝ σ_r / sqrt(|U_f| · T)

---

# 13. Complexity

Per update:

    O(M × |U_f|)    (vectorised: broadcast (N, M) operations)

Total:

    O(M × |U_f| × T)

Memory:

    O(M)    (log-posterior array only)

---

END OF FILE
