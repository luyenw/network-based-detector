# Fake Cell Detection using Residual-Based Method (Sliding Window)

## 1. Input Measurement Format

    time_sec, imsi, ue_x, ue_y, ecgi, cell_role(N/S), rsrp_dbm, timing_advance

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

## 6. Residual Function

Residual at position $u$:

$$
R(u) =
\frac{1}{N}
\sum_c
(\tilde{r}_c - \tilde{\hat{r}}_c(u))^2
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

## 7. Per-Cell Residual

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

## 8. Fake Cell Detection Rule

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

## 9. Sliding Window Algorithm

For:

$$
T = 1,2,3,...,T_{max}
$$

Steps:

1.  Collect measurements in:

$$
[0, T]
$$

2.  Normalize RSRP

3.  Estimate UE position:

$$
u^* = \arg\min R(u)
$$

4.  Compute per-cell residual

5.  Accumulate score

6.  Detect fake cells

------------------------------------------------------------------------

## 10. Runtime Logging

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

## 11. Full Pipeline Summary

$$
Measurement
\rightarrow
Normalize
\rightarrow
Estimate\ UE
\rightarrow
Compute\ Residual
\rightarrow
Accumulate
\rightarrow
Detect\ Fake\ Cell
$$
