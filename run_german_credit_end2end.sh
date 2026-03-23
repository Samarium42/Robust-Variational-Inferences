#!/usr/bin/env bash
set -euo pipefail

# =============================================================
# German Credit Bayesian Logistic Regression — End-to-End
#
# ~25D problem with genuinely contentious prior choices.
# Uses train_bayesian.py (prior-dependent targets) and
# predictive log-likelihood evaluation (not KDE-NLL).
# =============================================================

OUTDIR="out_fm_solver"
SEED=0
DATA_DIR="data"

PROBLEM="german_credit"
PRIORS="gaussian,gaussian_wide,cauchy,laplace,student_t"

# Flow training
STEPS=6000
BATCH=1024
HIDDEN=256
DEPTH=6
LR=0.001
WEIGHT_DECAY=0.01
PRIOR_EMB_DIM=16
DROPOUT=0.05
PRINT_EVERY=200

# Sampling
NSAMPLES=10000
STEP_SIZE=0.01   # smaller step for higher-D

# HMC (reference posteriors) — light settings, sufficient for logistic regression
HMC_SAMPLES=3000
HMC_WARMUP=1000
HMC_STEP_SIZE=0.005
HMC_LEAPFROG=15

# Weight optimisation
RUNS=10

mkdir -p "${OUTDIR}/models" "${OUTDIR}/samples" "${OUTDIR}/weight_experiments" \
         "${OUTDIR}/logs" "${DATA_DIR}"

echo "============================================"
echo "German Credit BLR — End-to-End"
echo "============================================"
echo "OUTDIR=${OUTDIR}"
echo "PROBLEM=${PROBLEM}"
echo "PRIORS=${PRIORS}"
echo "SEED=${SEED}"
echo ""

# ============================================================
# 1. Train conditional flow + sample from each prior
# ============================================================
MODEL_PATH="${OUTDIR}/models/fm_${PROBLEM}_seed${SEED}.pt"

python3 train_bayesian.py \
  --problem "${PROBLEM}" \
  --priors "${PRIORS}" \
  --steps "${STEPS}" \
  --batch "${BATCH}" \
  --hidden "${HIDDEN}" \
  --depth "${DEPTH}" \
  --lr "${LR}" \
  --weight_decay "${WEIGHT_DECAY}" \
  --prior_emb_dim "${PRIOR_EMB_DIM}" \
  --dropout "${DROPOUT}" \
  --seed "${SEED}" \
  --outdir "${OUTDIR}" \
  --print_every "${PRINT_EVERY}" \
  --n_samples "${NSAMPLES}" \
  --step_size "${STEP_SIZE}" \
  --model_path "${MODEL_PATH}" \
  --data_dir "${DATA_DIR}" \
  --hmc_samples "${HMC_SAMPLES}" \
  --hmc_warmup "${HMC_WARMUP}" \
  --hmc_step_size "${HMC_STEP_SIZE}" \
  --hmc_leapfrog "${HMC_LEAPFROG}" \
  2>&1 | tee "${OUTDIR}/logs/train_${PROBLEM}_seed${SEED}.log"

# ============================================================
# 2. Weight optimisation (all algorithms, predictive eval)
# ============================================================
echo ""
echo "=== Running Bayesian weight optimisation ==="

python3 -m weight_opt.run_repeats_bayesian \
  --problem "${PROBLEM}" \
  --outdir "${OUTDIR}" \
  --priors "${PRIORS}" \
  --seed "${SEED}" \
  --hidden "${HIDDEN}" \
  --depth "${DEPTH}" \
  --lr "${LR}" \
  --data_dir "${DATA_DIR}" \
  --out_root "${OUTDIR}/weight_experiments" \
  --run_name "${PROBLEM}_seed${SEED}" \
  --seed_base "${SEED}" \
  --runs "${RUNS}" \
  2>&1 | tee "${OUTDIR}/logs/weight_opt_${PROBLEM}_seed${SEED}.log"

# ============================================================
# 3. Full predictive evaluation
# ============================================================
echo ""
echo "=== Predictive evaluation ==="

python3 evaluation_predictive.py \
  --problem "${PROBLEM}" \
  --outdir "${OUTDIR}" \
  --seed "${SEED}" \
  --hidden "${HIDDEN}" \
  --depth "${DEPTH}" \
  --lr "${LR}" \
  --priors "${PRIORS}" \
  --data_dir "${DATA_DIR}" \
  --split test \
  2>&1 | tee "${OUTDIR}/logs/eval_${PROBLEM}_seed${SEED}.log"

echo ""
echo "============================================"
echo "Done: ${PROBLEM}"
echo "============================================"