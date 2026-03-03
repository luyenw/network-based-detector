We consider a cellular network deployed over a two-dimensional urban area of interest $\mathcal{A} \subset \mathbb{R}^2$, consisting of a set $\mathcal{C}_L = \{1, \dots, N\}$ of $N$ legitimate base stations deployed at fixed and known locations. Each cell $c \in \mathcal{C}_L$ is uniquely identifiable by a globally standardized identifier, namely the E-UTRAN Cell Global Identifier (ECGI) in 4G LTE networks or the NR Cell Global Identifier (NCGI) in 5G NR networks, as defined by 3GPP specifications. Each cell is associated with a known 3D position

$$
\mathbf{p}_c = (x_c,\; y_c,\; h)^{\top}
$$

where $h = 25\,\text{m}$ is the common antenna height, and a transmit power $P_c$ (dBm), which are maintained in a trusted operator-side cell database. According to the RRC procedures specified in 3GPP TS 36.331 (LTE) and TS 38.331 (NR), base stations can configure UEs to periodically report measurement information, including cell identities and Reference Signal Received Power (RSRP). A set of $M$ UEs move at ground level ($z = 0$) within $\mathcal{A}$; the position of UE $u$ at time $T$ is denoted $\mathbf{u} = (x_u, y_u, 0)^{\top}$.

We assume the presence of a FBS that performs an impersonation attack. The FBS is deployed at an unknown position

$$
\mathbf{p}_f = (x_f,\; y_f,\; 0)^{\top}
$$

and broadcasts the identity $c^* \in \mathcal{C}_L$ of a legitimate cell in order to deceive nearby UEs. However, unlike the legitimate eNB, the FBS transmits from a different physical location ($\mathbf{p}_f \neq \mathbf{p}_{c^*}$). As a result, signal measurements associated with the impersonated identity may originate from an inconsistent physical location.

The network operator is assumed to have full knowledge of the legitimate cell identities, locations, and transmit parameters through the trusted cell database

$$
\mathcal{C} = \bigl\{(c,\; \mathbf{p}_c,\; P_c)\bigr\}_{c \in \mathcal{C}_L}
$$

At each reporting interval $T \in \{1, 2, \dots, T_{\max}\}$, the operator collects measurement reports from UEs served by legitimate cells. Each report consists of the tuple $(u, c, r_{u,c}, \tau_{u,c})$, where $r_{u,c}$ is the measured RSRP (dBm) from cell $c$ and $\tau_{u,c}$ is the Timing Advance. The operator has no prior knowledge of the existence, identity, or location of the FBS. The objective is to detect such malicious behavior based solely on the measurement observations collected from UEs and the known legitimate network configuration.