#!/usr/bin/env bash
# run_speed_comparison.sh
# Runs flow matching vs CNF step-time benchmark on German Credit low-data.
# CNF is benchmarked by timing its ODE integration cost per step.
# FM is fully trained for 6000 steps to also get final test NLL.
#
# Produces:
#   out_speed_comparison/speed_comparison_summary.json
#   out_speed_comparison/speed_comparison.csv
#   out_speed_comparison/exp7_speed_comparison.pdf
#
# Run from project root.

set -euo pipefail

OUTDIR="out_speed_comparison"
DATA_DIR="data"
PROBLEM="german_credit"
PRIORS="gaussian,gaussian_wide,cauchy,laplace,student_t"

STEPS=6000
BATCH=1024
HIDDEN=256
DEPTH=6
LR=0.001
SEED=0
MAX_TRAIN=75

HMC_SAMPLES=3000
HMC_WARMUP=1000
HMC_STEP_SIZE=0.005
HMC_LEAPFROG=15

N_TIMING_TRIALS=200
CNF_STEPS=10

mkdir -p "${OUTDIR}/logs"

# ── 1. Check torchdiffeq ───────────────────────────────────────────────────
echo "============================================"
echo "  Checking dependencies"
echo "============================================"
python3 -c "import torchdiffeq; print('torchdiffeq OK:', torchdiffeq.__version__)"

# ── 2. Run benchmark + full FM training ───────────────────────────────────
echo ""
echo "============================================"
echo "  Running speed benchmark + FM training"
echo "============================================"
python3 train_cnf_baseline.py \
  --problem          "${PROBLEM}" \
  --priors           "${PRIORS}" \
  --steps            "${STEPS}" \
  --batch            "${BATCH}" \
  --hidden           "${HIDDEN}" \
  --depth            "${DEPTH}" \
  --lr               "${LR}" \
  --seed             "${SEED}" \
  --max_train        "${MAX_TRAIN}" \
  --data_dir         "${DATA_DIR}" \
  --outdir           "${OUTDIR}" \
  --hmc_samples      "${HMC_SAMPLES}" \
  --hmc_warmup       "${HMC_WARMUP}" \
  --hmc_step_size    "${HMC_STEP_SIZE}" \
  --hmc_leapfrog     "${HMC_LEAPFROG}" \
  --n_timing_trials  "${N_TIMING_TRIALS}" \
  --cnf_steps        "${CNF_STEPS}" \
  2>&1 | tee "${OUTDIR}/logs/speed_benchmark.log"

# ── 3. Generate figure ─────────────────────────────────────────────────────
echo ""
echo "============================================"
echo "  Generating figure"
echo "============================================"
python3 plot_speed_comparison.py \
  --outdir "${OUTDIR}" \
  --priors "${PRIORS}" \
  2>&1 | tee "${OUTDIR}/logs/plot.log"

echo ""
echo "============================================"
echo "  Done."
echo "  Figure: ${OUTDIR}/exp7_speed_comparison.pdf"
echo "  JSON:   ${OUTDIR}/speed_comparison_summary.json"
echo "  CSV:    ${OUTDIR}/speed_comparison.csv"
echo "============================================"