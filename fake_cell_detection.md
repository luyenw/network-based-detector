# Fake Cell Detection using Residual-Based Method (Sliding Window)

## 1. Input Measurement Format

    time_sec, imsi, ue_x, ue_y, ecgi, cell_role(N/S), rsrp_dbm, timing_advance

Optional quality fields when available:

    rsrq | rsrq_db | rsrq_dbm
    snr  | snr_db  | snr_dbm

Cell database:

$$
\mathcal{C} = \{ (c, \mathbf{p}_c, P_c) \}
$$

Where:

-   $c$ = cell ECGI\
-   $\mathbf{p}_c = (x_c, y_c)$ = cell position\
-   $P_c$ = transmit power (dBm)

------------------------------------------------------------------------

## 2. Pathloss Model

Distance:

$$
d_c(u) = \| \mathbf{u} - \mathbf{p}_c \|
$$

Pathloss:

$$
PL(d) = 128.1 + 37.6 \log_{10}(d_{km})
$$

Predicted RSRP:

$$
\hat{r}_c(u) = P_c - PL(d_c(u))
$$

------------------------------------------------------------------------

## 3. Sliding Window Definition

For time window:

$$
W_T = [0, T]
$$

Collect all measurements:

$$
\mathcal{M}_T =
\{ m_i \mid time\_sec \le T \}
$$

Group by IMSI.

------------------------------------------------------------------------

## 4. RSRP Aggregation

Mean RSRP per cell:

$$
r_c =
\frac{1}{N_c}
\sum_{i=1}^{N_c}
r_{c,i}
$$

------------------------------------------------------------------------

## 5. Normalization

Mean:

$$
\mu =
\frac{1}{N}
\sum_c r_c
$$

Std:

$$
\sigma =
\sqrt{
\frac{1}{N}
\sum_c (r_c - \mu)^2
}
$$

Normalized measured RSRP:

$$
\tilde{r}_c =
\frac{r_c - \mu}{\sigma}
$$

Normalized predicted RSRP:

$$
\tilde{\hat{r}}_c(u) =
\frac{\hat{r}_c(u) - \mu}{\sigma}
$$

------------------------------------------------------------------------

## 6. Measurement Reliability Weights

For each cell measurement \(i\), define:

$$
w_i^{(\mathrm{RSRQ})}
=
\exp\left(
- \frac{(\mathrm{RSRQ}_{\mathrm{ref}} - \mathrm{RSRQ}_i)^2}
{\sigma_{\mathrm{RSRQ}}^2}
\right)
$$

$$
w_i^{(\mathrm{SNR})}
=
\frac{1}{1 + e^{-(\mathrm{SNR}_i - \theta)}}
$$

$$
w_i^{(\mathrm{var})}
=
\exp\left(
- \frac{\mathrm{Var}(\mathrm{RSRP}_i(t))}
{\sigma_{\mathrm{var}}^2}
\right)
$$

Composite RSRP reliability:

$$
W_i = w_i^{(\mathrm{RSRQ})} \cdot w_i^{(\mathrm{SNR})} \cdot w_i^{(\mathrm{var})}
$$

Timing Advance consistency weight:

$$
w_i^{(\mathrm{TA})}
=
\exp\left(
- \frac{(d_i^{\circ}(\mathrm{RSRP}) - d_i^{(\mathrm{TA})})^2}
{\Delta^2}
\right)
$$

If `RSRQ` or `SNR` are not present in the measurement file, their weights default to 1.

------------------------------------------------------------------------

## 7. Weighted UE Position Objective

Distance inferred from mean RSRP:

$$
d_i^{\circ}(\mathrm{RSRP})
=
PL^{-1}(P_i - r_i)
$$

Weighted objective at position \(u\):

$$
R(u) =
\sum_i
W_i
\left(
\lVert \mathbf{u} - \mathbf{b}_i \rVert - d_i^{\circ}(\mathrm{RSRP})
\right)^2
+
\lambda_{\mathrm{TA}}
\sum_i
w_i^{(\mathrm{TA})}
\left(
\lVert \mathbf{u} - \mathbf{b}_i \rVert - d_i^{(\mathrm{TA})}
\right)^2
$$

Optimal UE position:

$$
u^* =
\arg\min_u R(u)
$$

Minimum residual:

$$
R^* = R(u^*)
$$

------------------------------------------------------------------------

## 8. Per-Cell Residual

Per-cell error:

$$
e_c =
\left|
\tilde{r}_c - \tilde{\hat{r}}_c(u^*)
\right|
$$

Accumulated score over windows:

$$
Score_c =
\frac{1}{K}
\sum_{k=1}^{K}
e_{c,k}
$$

------------------------------------------------------------------------

## 9. Fake Cell Detection Rule

Threshold:

$$
\tau \in [0.5, 1.5]
$$

Decision:

$$
\text{Cell}_c =
\begin{cases}
Fake & Score_c > \tau \\
Legit & Score_c \le \tau
\end{cases}
$$

------------------------------------------------------------------------

## 10. Sliding Window Algorithm

For:

$$
T = 1,2,3,...,T_{max}
$$

Steps:

1.  Collect measurements in:

$$
[0, T]
$$

2.  Aggregate per-cell mean RSRP and quality statistics

3.  Build adaptive weights \(W_i\) and \(w_i^{(\mathrm{TA})}\)

4.  Estimate UE position:

$$
u^* = \arg\min R(u)
$$

5.  Compute normalized per-cell residual

6.  Accumulate score

7.  Detect fake cells

------------------------------------------------------------------------

## 11. Runtime Logging

Log per window:

    WindowEndTime
    NumMeasurements
    Residual
    NumCells
    NumFakeCells
    Runtime_ms

Per cell log:

    WindowEndTime, ecgi, Score, Status

------------------------------------------------------------------------

## 12. Full Pipeline Summary

$$
Measurement
\rightarrow
\mathrm{Weighting}
\rightarrow
Estimate\ UE
\rightarrow
Compute\ Residual
\rightarrow
Accumulate
\rightarrow
Detect\ Fake\ Cell
$$
