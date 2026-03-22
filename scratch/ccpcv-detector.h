#pragma once
/* =============================================================================
 * ccpcv-detector.h
 *
 * Cell-Conditioned Physics Consistency Verification (CC-PCV)
 * Fake Base Station Detection — Steps 1–7
 *
 * Cell identity: ECGI (E-UTRAN Cell Global Identifier)
 *   ECGI = PLMN (MCC 3-digit + MNC 2-digit) + ECI (28-bit)
 *   Encoded as uint64_t: ((mcc*1000+mnc) << 28) | eci
 *
 * Detection principle:
 *   Only LEGAL cell positions are registered (keyed by ECGI).
 *   Fake cell impersonates a legal ECGI → UE reports strong RSRP at that ECGI
 *   from the fake's position → does not match expected RSRP from the legal
 *   cell's registered position → high residual → FAKE detected.
 * =============================================================================
 */

#include "ns3/vector.h"

#include <cstdint>
#include <iomanip>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace ns3 {

// ---------------------------------------------------------------------------
// ECGI — E-UTRAN Cell Global Identifier
//   Encoding: ((mcc*1000+mnc) << 28) | eci
//   ECI range: 0 – 268 435 455  (28 bits)
//   PLMN range: 0 – 999 999     (mcc 0..999, mnc 0..999)
// ---------------------------------------------------------------------------
typedef uint64_t EcgiT;

inline EcgiT
MakeEcgi(uint32_t mcc, uint32_t mnc, uint32_t eci)
{
    return (static_cast<uint64_t>(mcc * 1000u + mnc) << 28)
           | (static_cast<uint64_t>(eci) & 0x0FFFFFFFull);
}

inline std::string
EcgiToStr(EcgiT ecgi)
{
    uint32_t eci  = static_cast<uint32_t>(ecgi & 0x0FFFFFFFull);
    uint32_t plmn = static_cast<uint32_t>(ecgi >> 28);
    uint32_t mcc  = plmn / 1000u;
    uint32_t mnc  = plmn % 1000u;
    std::ostringstream ss;
    ss << std::setw(3) << std::setfill('0') << mcc << "-"
       << std::setw(2) << std::setfill('0') << mnc << "-"
       << std::setw(7) << std::setfill('0') << eci;
    return ss.str();
}

// ---------------------------------------------------------------------------
// MeasurementSample (STEP 0)
// One UE snapshot: RSRP [dBm] from every visible ECGI at time t.
// Key = ECGI.  When fake and legal share an ECGI, only the stronger
// signal survives aggregation (in PeriodicMeasure, before AddSample).
// ---------------------------------------------------------------------------
struct MeasurementSample
{
    uint64_t                ueId;
    double                  time;
    double                  ueX;
    double                  ueY;
    double                  ueZ;
    std::map<EcgiT, double> rsrpDbm; // ecgi → RSRP [dBm]
};

// ---------------------------------------------------------------------------
// RatioConstraint (STEP 3)
// Pre-computed distance ratio for one (ecgiC, ecgiJ) pair.
// ---------------------------------------------------------------------------
struct RatioConstraint
{
    EcgiT  ecgiC;  // ECGI under test
    EcgiT  ecgiJ;  // reference ECGI
    double alpha;  // 10^( (rsrp_C - rsrp_J) / (10 * n) )
};

// ---------------------------------------------------------------------------
// CcpcvDetector
//
// Usage:
//   1. Set public config fields
//   2. SetCellPosition(ecgi, pos)  for every LEGAL cell  (NOT the fake)
//   3. AddSample()                 for every UE measurement snapshot
//   4. outputCsvPath               = desired output file path
//   5. RunDetection()              after Simulator::Run()
// ---------------------------------------------------------------------------
class CcpcvDetector
{
public:
    // ---- Configuration -----------------------------------------------------
    double   gridResolutionM       = 50.0;
    double   pathLossExponent      = 3.0;
    double   txPowerDbm            = 46.0;
    double   detectionThreshold    = 1000.0;
    double   areaMinX              = 0.0;
    double   areaMaxX              = 3000.0;
    double   areaMinY              = 0.0;
    double   areaMaxY              = 3000.0;
    uint32_t minSamplesForDecision = 3;
    uint32_t maxSamplesPerGroup    = 500;   // 0 = unlimited
    std::string outputCsvPath      = "ccpcv_results.csv";

    // ---- Topology — register LEGAL cell positions keyed by ECGI -----------
    void SetCellPosition(EcgiT ecgi, const Vector& pos);

    // ---- Sample ingestion -------------------------------------------------
    void AddSample(const MeasurementSample& s);

    // ---- Run full CC-PCV pipeline (Steps 1–7) ----------------------------
    void RunDetection();

    // ---- Query results (valid after RunDetection) ------------------------
    const std::set<EcgiT>&         GetDetectedFakeEcgis() const;
    const std::map<EcgiT, double>& GetEcgiResiduals()     const;
    size_t SampleCount() const { return m_samples.size(); }

private:
    // Step 0: known legal ECGI positions
    std::map<EcgiT, Vector> m_ecgiPos;

    // Step 1: samples + per-ECGI groups (after subsampling)
    std::vector<MeasurementSample>         m_samples;
    std::map<EcgiT, std::vector<size_t>>   m_groups;   // ecgi → sample indices

    // Steps 2–3: ratio constraints per ECGI group
    std::map<EcgiT, std::vector<std::vector<RatioConstraint>>> m_groupRatios;

    // Step 5: per-ECGI mean residual
    std::map<EcgiT, double> m_ecgiResiduals;

    // Step 6: detection result
    std::set<EcgiT> m_detectedFakeEcgis;

    // ---- Internal pipeline ------------------------------------------------
    void   BuildGroups();
    void   ComputeRatios();
    double ComputeSampleResidual(EcgiT ecgi,
                                 const MeasurementSample& s) const;
    void   ComputeEcgiResiduals();
    void   MakeDetectionDecision();
    void   WriteResults()  const;  // Step 7a
    void   PrintSummary()  const;  // Step 7b
};

} // namespace ns3
