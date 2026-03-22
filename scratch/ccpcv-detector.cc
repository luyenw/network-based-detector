/* =============================================================================
 * ccpcv-detector.cc
 *
 * Cell-Conditioned Physics Consistency Verification (CC-PCV)
 * Implementation — Steps 1–7
 *
 * All cells identified by ECGI (E-UTRAN Cell Global Identifier).
 * Only LEGAL cell positions are registered in m_ecgiPos.
 * When a fake cell impersonates a legal ECGI, the measured RSRP at that ECGI
 * comes from the fake's position → inconsistent with the known legal position
 * → large residual → detected.
 * =============================================================================
 */

#include "ccpcv-detector.h"
#include "sim-logger.h"
#include "ns3/simulator.h"

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>

namespace ns3 {

// ===========================================================================
// Public API
// ===========================================================================

void
CcpcvDetector::SetCellPosition(EcgiT ecgi, const Vector& pos)
{
    m_ecgiPos[ecgi] = pos;
    SIM_LOG(DEBUG, "CcpcvDetector",
            "RegisterPos: ECGI=" << EcgiToStr(ecgi)
            << " pos=(" << pos.x << "," << pos.y << ")");
}

void
CcpcvDetector::AddSample(const MeasurementSample& s)
{
    if (s.rsrpDbm.size() < 2) return; // need ≥ 1 neighbour ECGI
    m_samples.push_back(s);
}

void
CcpcvDetector::RunDetection()
{
    SIM_LOG(INFO, "CcpcvDetector",
            "=== RunDetection start ==="
            "  samples=" << m_samples.size() <<
            "  ecgis=" << m_ecgiPos.size() <<
            "  gridRes=" << gridResolutionM << "m"
            "  threshold=" << detectionThreshold <<
            "  maxSampPerGroup=" << maxSamplesPerGroup);

    BuildGroups();           // Step 1 + subsampling
    ComputeRatios();         // Steps 2–3
    ComputeEcgiResiduals();  // Steps 4–5
    MakeDetectionDecision(); // Step 6
    WriteResults();          // Step 7a
    PrintSummary();          // Step 7b
}

const std::set<EcgiT>&
CcpcvDetector::GetDetectedFakeEcgis() const { return m_detectedFakeEcgis; }

const std::map<EcgiT, double>&
CcpcvDetector::GetEcgiResiduals() const { return m_ecgiResiduals; }

// ===========================================================================
// Step 1 — Cell-conditioned grouping + subsampling
// ===========================================================================
void
CcpcvDetector::BuildGroups()
{
    m_groups.clear();

    for (size_t i = 0; i < m_samples.size(); ++i)
        for (const auto& kv : m_samples[i].rsrpDbm)
            m_groups[kv.first].push_back(i);

    if (maxSamplesPerGroup > 0)
    {
        for (auto& grp : m_groups)
        {
            auto& v = grp.second;
            if (v.size() <= maxSamplesPerGroup) continue;

            std::vector<size_t> sub;
            sub.reserve(maxSamplesPerGroup);
            double step = static_cast<double>(v.size()) / maxSamplesPerGroup;
            for (uint32_t i = 0; i < maxSamplesPerGroup; ++i)
                sub.push_back(v[static_cast<size_t>(i * step)]);
            v = std::move(sub);
        }
    }

    SIM_LOG(INFO, "CcpcvDetector",
            "BuildGroups: " << m_groups.size() << " ECGI groups"
            << (maxSamplesPerGroup > 0
                ? ("  (capped at " + std::to_string(maxSamplesPerGroup) + " each)")
                : ""));
    for (const auto& grp : m_groups)
    {
        SIM_LOG(DEBUG, "CcpcvDetector",
                "  ECGI=" << EcgiToStr(grp.first)
                << "  samples=" << grp.second.size());
    }
}

// ===========================================================================
// Steps 2–3 — RSRP difference → distance ratio alpha
// Populates m_groupRatios[ecgiC][i] for every ECGI group and sample.
// ===========================================================================
void
CcpcvDetector::ComputeRatios()
{
    m_groupRatios.clear();

    for (const auto& grp : m_groups)
    {
        EcgiT ecgiC    = grp.first;
        auto& ratioVec = m_groupRatios[ecgiC];
        ratioVec.reserve(grp.second.size());

        for (size_t idx : grp.second)
        {
            const MeasurementSample& s = m_samples[idx];
            std::vector<RatioConstraint> constraints;

            auto it_C = s.rsrpDbm.find(ecgiC);
            if (it_C == s.rsrpDbm.end())
            {
                ratioVec.push_back({});
                continue;
            }
            double rsrpC = it_C->second;

            for (const auto& kv : s.rsrpDbm)
            {
                EcgiT ecgiJ = kv.first;
                if (ecgiJ == ecgiC) continue;
                if (m_ecgiPos.find(ecgiJ) == m_ecgiPos.end()) continue;

                double deltaR = rsrpC - kv.second;
                double alpha  = std::pow(10.0, deltaR / (10.0 * pathLossExponent));
                constraints.push_back({ecgiC, ecgiJ, alpha});
            }
            ratioVec.push_back(std::move(constraints));
        }

        size_t total = 0;
        for (const auto& cv : ratioVec) total += cv.size();
        double avg = ratioVec.empty() ? 0.0
                     : static_cast<double>(total) / ratioVec.size();
        SIM_LOG(DEBUG, "CcpcvDetector",
                "ComputeRatios: ECGI=" << EcgiToStr(ecgiC)
                << "  samples=" << ratioVec.size()
                << "  avgConstraints/sample=" << std::fixed
                << std::setprecision(1) << avg);
    }
    SIM_LOG(INFO, "CcpcvDetector",
            "ComputeRatios done for " << m_groupRatios.size() << " ECGIs.");
}

// ===========================================================================
// Step 4 — Consistency residual for one (ECGI, sample) pair
//
// Compares measured RSRP at ecgi against path-loss prediction using the
// LEGAL cell's registered position for that ECGI.
//
//   residual = |measured_rsrp − predicted_rsrp|
//   PL [dB] = 128.1 + 37.6 * log10(d_km)   (COST-231 urban macro, 2 GHz)
//
// If the fake is impersonating this ECGI and is closer to the UE, the
// measured RSRP will exceed the prediction from the legal position → large
// residual → detection.
// ===========================================================================
double
CcpcvDetector::ComputeSampleResidual(EcgiT ecgi,
                                      const MeasurementSample& s) const
{
    auto posIt = m_ecgiPos.find(ecgi);
    if (posIt == m_ecgiPos.end())
    {
        SIM_LOG(WARN, "CcpcvDetector",
                "ComputeSampleResidual: ECGI=" << EcgiToStr(ecgi)
                << " has no registered position — skipped");
        return 0.0;
    }

    double dx      = s.ueX - posIt->second.x;
    double dy      = s.ueY - posIt->second.y;
    double dz      = s.ueZ - posIt->second.z;
    double dist_m  = std::max(std::sqrt(dx*dx + dy*dy + dz*dz), 1.0);
    double dist_km = dist_m / 1000.0;
    double pl      = 128.1 + 37.6 * std::log10(dist_km);
    double predicted = txPowerDbm - pl;

    auto rsrpIt = s.rsrpDbm.find(ecgi);
    if (rsrpIt == s.rsrpDbm.end()) return 0.0;

    double residual = std::fabs(rsrpIt->second - predicted);

    SIM_LOG(DEBUG, "CcpcvDetector",
            "SampleResidual: ECGI=" << EcgiToStr(ecgi)
            << " dist=" << std::fixed << std::setprecision(1) << dist_m << "m"
            << " pred=" << std::setprecision(2) << predicted << "dBm"
            << " meas=" << rsrpIt->second << "dBm"
            << " residual=" << std::setprecision(4) << residual);

    return residual;
}

// ===========================================================================
// Step 5 — Per-ECGI residual aggregation (mean over group)
// ===========================================================================
void
CcpcvDetector::ComputeEcgiResiduals()
{
    m_ecgiResiduals.clear();

    for (const auto& grp : m_groups)
    {
        EcgiT ecgi = grp.first;
        if (grp.second.size() < minSamplesForDecision)
        {
            SIM_LOG(WARN, "CcpcvDetector",
                    "ECGI=" << EcgiToStr(ecgi) << " skipped: only "
                    << grp.second.size() << " samples"
                    << " (min=" << minSamplesForDecision << ")");
            continue;
        }

        double   sum   = 0.0;
        uint32_t count = 0;
        for (size_t i = 0; i < grp.second.size(); ++i)
        {
            sum += ComputeSampleResidual(ecgi, m_samples[grp.second[i]]);
            ++count;
        }
        m_ecgiResiduals[ecgi] = (count > 0) ? sum / count : 0.0;
    }

    SIM_LOG(INFO, "CcpcvDetector",
            "ComputeEcgiResiduals: scored " << m_ecgiResiduals.size() << " ECGIs:");
    for (const auto& kv : m_ecgiResiduals)
    {
        SIM_LOG(INFO, "CcpcvDetector",
                "  ECGI=" << EcgiToStr(kv.first)
                << "  meanResidual=" << std::fixed << std::setprecision(4)
                << kv.second);
    }
}

// ===========================================================================
// Step 6 — Detection decision
// ===========================================================================
void
CcpcvDetector::MakeDetectionDecision()
{
    m_detectedFakeEcgis.clear();
    for (const auto& kv : m_ecgiResiduals)
    {
        bool fake = (kv.second > detectionThreshold);
        SIM_LOG(DEBUG, "CcpcvDetector",
                "Decision: ECGI=" << EcgiToStr(kv.first)
                << "  residual=" << std::fixed << std::setprecision(4) << kv.second
                << "  threshold=" << detectionThreshold
                << "  → " << (fake ? "FAKE" : "NORMAL"));
        if (fake)
            m_detectedFakeEcgis.insert(kv.first);
    }
}

// ===========================================================================
// Step 7a — Write CSV
// Columns: time_sec, ecgi, residual, status
// ===========================================================================
void
CcpcvDetector::WriteResults() const
{
    std::ofstream ofs(outputCsvPath);
    if (!ofs.is_open())
    {
        SIM_LOG(ERR, "CcpcvDetector", "Cannot open output: " << outputCsvPath);
        return;
    }
    double now = Simulator::Now().GetSeconds();
    ofs << "time_sec,ecgi,residual,status\n";
    for (const auto& kv : m_ecgiResiduals)
    {
        bool fake = (m_detectedFakeEcgis.count(kv.first) > 0);
        ofs << std::fixed << std::setprecision(4)
            << now                    << ","
            << EcgiToStr(kv.first)    << ","
            << kv.second              << ","
            << (fake ? "FAKE" : "NORMAL") << "\n";
    }
    SIM_LOG(INFO, "CcpcvDetector",
            "Results written → " << outputCsvPath
            << "  (" << m_ecgiResiduals.size() << " rows)");
}

// ===========================================================================
// Step 7b — Console summary
// ===========================================================================
void
CcpcvDetector::PrintSummary() const
{
    SIM_LOG(INFO, "CcpcvDetector", "===== CC-PCV Detection Results =====");
    for (const auto& kv : m_ecgiResiduals)
    {
        bool fake = (m_detectedFakeEcgis.count(kv.first) > 0);
        if (fake)
            SIM_LOG(WARN, "CcpcvDetector",
                    "[CCPCV] ECGI=" << EcgiToStr(kv.first)
                    << "  Residual=" << std::fixed << std::setprecision(2)
                    << kv.second
                    << "  Status=*** FAKE ***");
        else
            SIM_LOG(INFO, "CcpcvDetector",
                    "[CCPCV] ECGI=" << EcgiToStr(kv.first)
                    << "  Residual=" << std::fixed << std::setprecision(2)
                    << kv.second
                    << "  Status=NORMAL");
    }
    SIM_LOG(INFO, "CcpcvDetector",
            "===================================="
            "  detected=" << m_detectedFakeEcgis.size() << " fake ECGI(s)");
}

} // namespace ns3
