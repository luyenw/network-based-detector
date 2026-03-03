#!/usr/bin/env bash
# =============================================================================
# run.sh — Container entrypoint for fake-bs-sim
#
# Logs written to /output/:
#   run.log         — shell-level log (build steps, timing, this script)
#   simulation.log  — C++ SimLogger output (parameters, callbacks, CC-PCV)
# =============================================================================

set -euo pipefail

NS3_DIR="/opt/ns-allinone-3.39/ns-3.39"
SCRATCH_DIR="${NS3_DIR}/scratch"
VOLUME_OUTPUT="/output"
RUN_LOG="${VOLUME_OUTPUT}/run.log"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
ts() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(ts)] [run.sh] $*" | tee -a "${RUN_LOG}"; }
log_section() {
    local bar="============================================================"
    echo "[$(ts)] [run.sh] ${bar}" | tee -a "${RUN_LOG}"
    echo "[$(ts)] [run.sh]  $*"    | tee -a "${RUN_LOG}"
    echo "[$(ts)] [run.sh] ${bar}" | tee -a "${RUN_LOG}"
}

# ---------------------------------------------------------------------------
# Step 0 — Ensure output directory exists and start log file
# ---------------------------------------------------------------------------
mkdir -p "${VOLUME_OUTPUT}"
# Truncate run.log for fresh run
: > "${RUN_LOG}"

log_section "Fake-BS Simulation Pipeline"
log "Args: $*"
log "NS3_DIR: ${NS3_DIR}"
log "Output : ${VOLUME_OUTPUT}"

# ---------------------------------------------------------------------------
# Step 1 — Copy source files into ns-3 scratch
# ---------------------------------------------------------------------------
log_section "Step 1: Copying source files"
cp /src/fake-bs-sim.cc   "${SCRATCH_DIR}/fake-bs-sim.cc"
cp /src/ccpcv-detector.h "${SCRATCH_DIR}/ccpcv-detector.h"  2>/dev/null || true
cp /src/ccpcv-detector.cc "${SCRATCH_DIR}/ccpcv-detector.cc" 2>/dev/null || true
cp /src/sim-logger.h     "${SCRATCH_DIR}/sim-logger.h"       2>/dev/null || true
log "Source files copied to ${SCRATCH_DIR}/"

# ---------------------------------------------------------------------------
# Step 2 — Incremental compile (log build output to run.log too)
# ---------------------------------------------------------------------------
log_section "Step 2: Compiling (incremental)"
cd "${NS3_DIR}"

BUILD_START=$(date +%s)
./ns3 build fake-bs-sim 2>&1 | tee -a "${RUN_LOG}"
BUILD_END=$(date +%s)
log "Compilation finished in $((BUILD_END - BUILD_START))s"

# ---------------------------------------------------------------------------
# Step 3 — Run simulation
# All output goes to stdout (docker logs) AND to run.log via tee.
# The C++ code also writes /output/simulation.log independently.
# ---------------------------------------------------------------------------
log_section "Step 3: Running simulation"
log "Command args: $*"

SIM_START=$(date +%s)
./ns3 run "fake-bs-sim $*" 2>&1 | tee -a "${RUN_LOG}"
SIM_EXIT=${PIPESTATUS[0]}
SIM_END=$(date +%s)

log "Simulation exited with code ${SIM_EXIT} in $((SIM_END - SIM_START))s"

# ---------------------------------------------------------------------------
# Step 4 — Generate initial layout plot (BS positions only, before measurements)
# ---------------------------------------------------------------------------
log_section "Step 4: Generating initial layout plot (no UE)"
DETECTOR_DIR="/src/detector"
INITIAL_LAYOUT="${VOLUME_OUTPUT}/initial_layout.png"

if [ -f "${DETECTOR_DIR}/plot_layout.py" ] && [ -f "${VOLUME_OUTPUT}/cell_database.csv" ]; then
    python3 "${DETECTOR_DIR}/plot_layout.py" \
        --results-dir "${VOLUME_OUTPUT}" \
        --output      "${INITIAL_LAYOUT}" \
        --no-ue \
        --no-show \
        2>&1 | tee -a "${RUN_LOG}" || true
    if [ -f "${INITIAL_LAYOUT}" ]; then
        log "  [OK] initial_layout.png generated"
    else
        log "  [WARN] initial_layout.png not generated (plot_layout.py error)"
    fi
else
    log "  [SKIP] plot_layout.py or cell_database.csv not found"
fi

# ---------------------------------------------------------------------------
# Step 5 — Summary of output files
# ---------------------------------------------------------------------------
log_section "Step 5: Output files"
for f in simulation.log measurements.csv cell_database.csv ccpcv_results.csv initial_layout.png; do
    FPATH="${VOLUME_OUTPUT}/${f}"
    if [ -f "${FPATH}" ]; then
        LINES=$(wc -l < "${FPATH}" 2>/dev/null || echo "—")
        SIZE=$(du -sh "${FPATH}" 2>/dev/null | cut -f1)
        log "  [OK] ${f}  (${LINES} lines, ${SIZE})"
    else
        log "  [MISSING] ${f}"
    fi
done

log_section "Pipeline complete"
log "All logs available in ${VOLUME_OUTPUT}/"

if [ "${SIM_EXIT}" -ne 0 ]; then
    log "ERROR: simulation returned non-zero exit code ${SIM_EXIT}"
    exit "${SIM_EXIT}"
fi
