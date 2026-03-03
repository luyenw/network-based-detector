You are modifying an existing ns-3 LTE/NR simulation codebase that already builds and runs correctly. The pipeline for simulation execution, build system, and measurement collection already exists. Your task is to implement the **Cell-Conditioned Physics Consistency Verification (CC-PCV) fake cell detection algorithm**, covering Steps 1–7 (grouping, ratio transform, consistency residual computation, and detection decision), WITHOUT implementing fake cell localization.

IMPORTANT CONSTRAINTS:

• DO NOT break existing simulation functionality
• DO NOT modify core PHY/MAC protocol logic
• ONLY add new data collection and detection logic
• USE modular, self-contained classes
• KEEP runtime overhead minimal
• ALL new code must compile cleanly with existing ns-3 build pipeline

You must implement the following pipeline.

---

## STEP 0 — Definitions and Assumptions

Measurement report format (already available or must be extracted):

Each UE measurement sample t contains:

• UE ID
• Serving Cell ID
• Neighbor Cell ID list
• RSRP per Cell ID (linear or dBm, use dBm consistently)
• Simulation timestamp

Represent each measurement sample as:

struct MeasurementSample
{
uint64_t ueId;
double time;
std::map<uint16_t, double> rsrpDbm; // key = CellId
};

All samples are stored in:

std::vector<MeasurementSample> g_allSamples;

Cell position map is known from simulation topology:

std::map<uint16_t, Vector> g_cellPositions;

Path loss exponent constant:

double g_pathLossExponent = 3.0;

Detection threshold:

double g_detectionThreshold = configurable;

---

## STEP 1 — Cell-Conditioned Grouping

Create a structure grouping samples by Cell ID:

std::map<uint16_t, std::vector<const MeasurementSample*>> g_cellGroups;

For each sample:

for each CellId C in sample.rsrpDbm:
g_cellGroups[C].push_back(&sample);

This produces group G_C for each cell C.

---

## STEP 2 — Compute RSRP Difference

For each group G_C and each sample t in G_C:

Let rsrp_C = sample.rsrpDbm[C]

For each other cell j in sample:

delta_r = rsrp_C - rsrp_j

Store delta_r.

---

## STEP 3 — Compute Distance Ratio Alpha

Compute:

alpha_Cj = pow(10.0, delta_r / (10.0 * g_pathLossExponent));

Store in temporary structure:

struct RatioConstraint
{
uint16_t cellC;
uint16_t cellJ;
double alpha;
};

Attach list of RatioConstraint to each sample per group.

---

## STEP 4 — Consistency Residual per Sample

For each group G_C:

For each sample t:

Estimate UE position u by minimizing residual:

Residual(u) =
sum over j:
(
|u - p_j| − alpha_Cj × |u − p_C|
)^2

Implement minimization using simple numerical approach:

Option A (preferred):
Use grid search over simulation area

Option B:
Use gradient descent

Grid resolution configurable (example 20m).

Store minimum residual:

double sampleResidual;

---

## STEP 5 — Cell Consistency Residual Aggregation

Compute per-cell residual:

R(C) =
mean(sampleResidual over all samples in group G_C)

Store result:

std::map<uint16_t, double> g_cellResiduals;

---

## STEP 6 — Detection Decision

For each cell C:

if (g_cellResiduals[C] > g_detectionThreshold)
mark cell as suspicious

Store detection result:

std::set<uint16_t> g_detectedFakeCells;

---

## STEP 7 — Output Detection Results

Print detection summary:

CellId
Residual value
Detection decision

Example output format:

[CCPCV] CellId=3 Residual=12.53 Status=FAKE
[CCPCV] CellId=7 Residual=0.82 Status=NORMAL

Also export CSV file:

time, cellId, residual, status

---

## STEP 8 — Integration with ns-3 Simulation

Detection execution must occur AFTER simulation ends:

Example:

Simulator::Run();

RunCcpcvDetection();

Simulator::Destroy();

---

## STEP 9 — Required New Files

Create:

ccpcv-detector.h
ccpcv-detector.cc

Class:

class CcpcvDetector
{
public:

```
void AddSample(const MeasurementSample& sample);

void RunDetection();

const std::set<uint16_t>& GetDetectedFakeCells();
```

private:

```
void BuildGroups();
void ComputeRatios();
double ComputeSampleResidual(...);
void ComputeCellResiduals();
void MakeDetectionDecision();
```

};

---

## STEP 10 — Performance Constraints

Detection must scale to:

• 10,000 measurement samples
• 50 cells

Execution time under 5 seconds.

---

## STEP 11 — DO NOT MODIFY

Do NOT modify:

• LTE PHY
• Spectrum propagation model
• Scheduler
• Handover logic

Only add measurement extraction and CCPCV detection.

---

## EXPECTED FINAL RESULT

The simulation runs normally.

At the end of simulation, CC-PCV detection executes and outputs:

• Residual per cell
• Fake cell detection decision

No localization required.

---

Implement clean, readable, modular ns-3 compatible C++ code.
