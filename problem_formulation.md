# Problem Formulation

## Notation

Let $\mathcal{C}_L = \{1, \dots, N\}$ denote the set of $N$ legitimate cells. Each legitimate cell $c \in \mathcal{C}_L$ is characterised by its 3D position

$$
\mathbf{p}_c = (x_c,\; y_c,\; h)^{\top} \in \mathbb{R}^3
$$

where $h = 25\,\text{m}$ is the common antenna height, and transmit power $P_c$ (dBm). A fake base station (FBS) is deployed at an unknown position

$$
\mathbf{p}_f = (x_f,\; y_f,\; 0)^{\top}
$$

and broadcasts ECGI $c^* \in \mathcal{C}_L$, impersonating the legitimate cell $c^*$. UEs move at ground level ($z = 0$) within the coverage area $\mathcal{A} \subset \mathbb{R}^2$.

---

## Measurement Model

At each reporting interval $T \in \{1, 2, \dots, T_{\max}\}$, every UE $u$ that is served by a legitimate cell submits a measurement report. The 3D distance from UE position $\mathbf{u} = (x_u, y_u, 0)^{\top}$ to cell $c$ is

$$
d_c(\mathbf{u}) = \|\mathbf{u} - \mathbf{p}_c\|_2 = \sqrt{(x_u - x_c)^2 + (y_u - y_c)^2 + h^2}
$$

The received power is modelled by the COST-231 path-loss formula

$$
\text{PL}(d) = 128.1 + 37.6\,\log_{10}\!\left(\frac{d}{1000}\right) \quad [\text{dB}]
$$

$$
r_{u,c} = P_c - \text{PL}\bigl(d_c(\mathbf{u})\bigr) + \eta_{u,c}
$$

where

$$
\eta_{u,c} \sim \mathcal{N}(0,\, \sigma_s^2)
$$

is log-normal shadowing. When the FBS is active, a UE near $\mathbf{p}_f$ observes RSRP attributed to ECGI $c^*$ that originates from $\mathbf{p}_f$, not from the legitimate position $\mathbf{p}_{c^*}$:

$$
r_{u,c^*} = P_f - \text{PL}\!\left(\|\mathbf{u} - \mathbf{p}_f\|_2\right) + \eta_{u,f}
$$

The network operator collects the measurement set at time $T$:

$$
\mathcal{M}_T = \bigl\{(u,\; c,\; r_{u,c},\; \tau_{u,c})\bigr\}
$$

where the Timing Advance is

$$
\tau_{u,c} = \left\lfloor \frac{\|\mathbf{u} - \mathbf{p}_c\|_2}{78.125} \right\rfloor
$$

---

## Detection Problem

Given the trusted cell database $\{(\mathbf{p}_c, P_c)\}_{c \in \mathcal{C}_L}$ and the collected measurements $\mathcal{M}_T$, the goal is to decide, for each ECGI $c$, between the two hypotheses:

$$
\mathcal{H}_0^{(c)}:\; \text{RSRP at ECGI } c \text{ originates from } \mathbf{p}_c \quad \text{(legitimate)}
$$

$$
\mathcal{H}_1^{(c)}:\; \text{RSRP at ECGI } c \text{ does not originate from } \mathbf{p}_c \quad \text{(fake)}
$$

---

## Location Estimation and Anomaly Formulation

### Normalisation Rationale

The raw RSRP vector $\mathbf{r}_u = (r_{u,1}, \dots, r_{u,N_u})^{\top}$ depends on the unknown absolute path gain at the UE position. To eliminate this dependency and work with the relative signal pattern across cells, each RSRP vector is standardised:

$$
\tilde{r}_{u,c} = \frac{r_{u,c} - \mu_u}{\sigma_u}, \qquad
\mu_u = \frac{1}{N_u}\sum_{c} r_{u,c}, \quad
\sigma_u = \sqrt{\frac{1}{N_u}\sum_{c}(r_{u,c}-\mu_u)^2}
$$

The same transformation is applied to the predicted RSRP at any candidate position $\mathbf{u}$:

$$
\hat{r}_{u,c}(\mathbf{u}) = P_c - \text{PL}\bigl(d_c(\mathbf{u})\bigr)
$$

$$
\tilde{\hat{r}}_{u,c}(\mathbf{u}) = \frac{\hat{r}_{u,c}(\mathbf{u}) - \mu_u}{\sigma_u}
$$

Because $\mu_u$ and $\sigma_u$ are computed from the measured vector, the normalised predicted pattern $\tilde{\hat{r}}_{u,c}(\mathbf{u})$ uses the same shift and scale, making measured and predicted directly comparable regardless of the unknown absolute path loss.

### Continuous Optimisation Problem

Given the normalised measurements $\tilde{\mathbf{r}}_u$, the UE position is estimated by solving

$$
\mathbf{u}^* = \underset{\mathbf{u} \in \mathcal{A}}{\arg\min}\ R(\mathbf{u})
$$

$$
R(\mathbf{u}) = \frac{1}{N_u}\sum_{c=1}^{N_u} \Bigl(\tilde{r}_{u,c} - \tilde{\hat{r}}_{u,c}(\mathbf{u})\Bigr)^2
$$

Expanding $R(\mathbf{u})$ and using the definition of normalisation, the residual can be rewritten as

$$
R(\mathbf{u}) = \frac{1}{N_u}\sum_{c} \left(\tilde{r}_{u,c} - \frac{\hat{r}_{u,c}(\mathbf{u}) - \mu_u}{\sigma_u}\right)^2
= \frac{1}{\sigma_u^2} \cdot \frac{1}{N_u}\sum_{c} \Bigl(r_{u,c} - \hat{r}_{u,c}(\mathbf{u}) - (\mu_u - \bar{\hat{r}}_u(\mathbf{u}))\Bigr)^2
$$

where $\bar{\hat{r}}_u(\mathbf{u}) = \frac{1}{N_u}\sum_c \hat{r}_{u,c}(\mathbf{u})$. This shows that $R(\mathbf{u})$ is proportional to the variance of the prediction error after removing its mean — i.e., it measures the **shape mismatch** between measured and predicted RSRP patterns, independent of any constant offset.

### Behaviour Under Each Hypothesis

**Under $\mathcal{H}_0$ (all legitimate):** At the true UE position $\mathbf{u}_0$,

$$
r_{u,c} = \hat{r}_{u,c}(\mathbf{u}_0) + \eta_{u,c}
$$

so $R(\mathbf{u}_0) = \sigma_s^2 / \sigma_u^2 \approx 0$ for small shadowing $\sigma_s$. The estimator recovers $\mathbf{u}^* \approx \mathbf{u}_0$ and the per-cell errors $e_{u,c}$ are uniformly small.

**Under $\mathcal{H}_1$ (fake at $c^*$):** The measurement for the impersonated cell deviates from the legitimate prediction:

$$
r_{u,c^*} = P_f - \text{PL}\!\left(\|\mathbf{u} - \mathbf{p}_f\|_2\right) + \eta_{u,f}
\;\neq\;
\hat{r}_{u,c^*}(\mathbf{u}_0)
$$

No single $\mathbf{u} \in \mathcal{A}$ can simultaneously explain the legitimate cells and cell $c^*$, because the legitimate cells constrain $\mathbf{u}^*$ near $\mathbf{u}_0$ while the fake RSRP implies a different geometry. As a consequence,

$$
e_{u,c^*} = \bigl|\tilde{r}_{u,c^*} - \tilde{\hat{r}}_{u,c^*}(\mathbf{u}^*)\bigr| \gg e_{u,c}, \quad c \neq c^*
$$

revealing $c^*$ as an anomalous cell.

### Discrete Grid Search

Since $R(\mathbf{u})$ is non-convex in general, the continuous problem is solved over a uniform grid $\mathcal{G} \subset \mathcal{A}$ with resolution $\Delta$:

$$
\mathcal{G} = \bigl\{(x_0 + i\Delta,\; y_0 + j\Delta)\bigr\}_{i,j}
$$

$$
\mathbf{u}^* = \underset{\mathbf{u} \in \mathcal{G}}{\arg\min}\ R(\mathbf{u})
$$

The approximation error introduced by discretisation is bounded by $O(\Delta)$ in position, and its effect on $e_{u,c}$ is negligible when $\Delta \ll d_c(\mathbf{u})$.

---

## Residual-Based Test Statistic

For each UE $u$ observing $N_u$ cells, define the normalised measured RSRP:

$$
\tilde{r}_{u,c} = \frac{r_{u,c} - \mu_u}{\sigma_u}
$$

$$
\mu_u = \frac{1}{N_u}\sum_c r_{u,c}, \qquad
\sigma_u = \sqrt{\frac{1}{N_u}\sum_c (r_{u,c} - \mu_u)^2}
$$

The optimal UE position estimate $\mathbf{u}^*$ minimises the residual over a discrete search grid $\mathcal{G}$:

$$
\mathbf{u}^* = \underset{\mathbf{u} \in \mathcal{G}}{\arg\min}\; R(\mathbf{u})
$$

$$
R(\mathbf{u}) = \frac{1}{N_u} \sum_c \Bigl(\tilde{r}_{u,c} - \tilde{\hat{r}}_{u,c}(\mathbf{u})\Bigr)^2
$$

where the normalised predicted RSRP is

$$
\tilde{\hat{r}}_{u,c}(\mathbf{u}) = \frac{\hat{r}_{u,c}(\mathbf{u}) - \mu_u}{\sigma_u}, \qquad
\hat{r}_{u,c}(\mathbf{u}) = P_c - \text{PL}\bigl(d_c(\mathbf{u})\bigr)
$$

The per-cell error for UE $u$ at $\mathbf{u}^*$ is:

$$
e_{u,c} = \bigl|\tilde{r}_{u,c} - \tilde{\hat{r}}_{u,c}(\mathbf{u}^*)\bigr|
$$

Aggregating over all valid UEs at time $T$:

$$
\text{Score}_c(T) = \frac{1}{|\mathcal{U}_T^{(c)}|} \sum_{u \in \mathcal{U}_T^{(c)}} e_{u,c}
$$

where $\mathcal{U}_T^{(c)}$ is the set of UEs that observe cell $c$ at time $T$.

---

## Detection Rule

The adaptive threshold $\tau(T)$ is derived from the empirical distribution of scores using the Median Absolute Deviation (MAD):

$$
\tau(T) = \operatorname{median}_c\!\bigl[\text{Score}_c(T)\bigr]
         + k \cdot 1.4826 \cdot \operatorname{MAD}_c\!\bigl[\text{Score}_c(T)\bigr]
$$

The per-second decision is:

$$
\hat{\mathcal{H}}^{(c)}(T) =
\begin{cases}
\mathcal{H}_1^{(c)} & \text{if } \text{Score}_c(T) > \tau(T) \\
\mathcal{H}_0^{(c)} & \text{otherwise}
\end{cases}
$$

The final verdict uses majority voting over all observed seconds:

$$
\hat{c}^* = \left\{c \in \mathcal{C}_L \;\middle|\; \sum_{T=1}^{T_{\max}} \mathbf{1}\!\left[\hat{\mathcal{H}}^{(c)}(T) = \mathcal{H}_1^{(c)}\right] > \frac{T_{\max}}{2}\right\}
$$
