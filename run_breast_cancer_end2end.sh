#!/usr/bin/env bash
set -euo pipefail

OUTDIR="out_fm_solver_bc"
DATA_DIR="data"

PROBLEM="breast_cancer"
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
STEP_SIZE=0.01

# HMC
HMC_SAMPLES=3000
HMC_WARMUP=1000
HMC_STEP_SIZE=0.005
HMC_LEAPFROG=15

# Weight optimisation
RUNS=10

# Low-data regime — same as German Credit primary experiment
MAX_TRAIN=75

mkdir -p "${OUTDIR}/models" "${OUTDIR}/samples" "${OUTDIR}/weight_experiments" \
         "${OUTDIR}/logs" "${DATA_DIR}"

echo "============================================"
echo "Breast Cancer BLR — End-to-End (5 seeds)"
echo "============================================"
echo "OUTDIR=${OUTDIR}  PROBLEM=${PROBLEM}  MAX_TRAIN=${MAX_TRAIN}"
echo ""

for SEED in 1 2 3 4; do

    echo "============================================"
    echo "Seed ${SEED}"
    echo "============================================"

    MODEL_PATH="${OUTDIR}/models/fm_${PROBLEM}_seed${SEED}.pt"

    # 1. Train conditional flow + sample
    python3 train_bayesian.py \
      --problem        "${PROBLEM}" \
      --priors         "${PRIORS}" \
      --steps          "${STEPS}" \
      --batch          "${BATCH}" \
      --hidden         "${HIDDEN}" \
      --depth          "${DEPTH}" \
      --lr             "${LR}" \
      --weight_decay   "${WEIGHT_DECAY}" \
      --prior_emb_dim  "${PRIOR_EMB_DIM}" \
      --dropout        "${DROPOUT}" \
      --seed           "${SEED}" \
      --outdir         "${OUTDIR}" \
      --print_every    "${PRINT_EVERY}" \
      --n_samples      "${NSAMPLES}" \
      --step_size      "${STEP_SIZE}" \
      --model_path     "${MODEL_PATH}" \
      --data_dir       "${DATA_DIR}" \
      --hmc_samples    "${HMC_SAMPLES}" \
      --hmc_warmup     "${HMC_WARMUP}" \
      --hmc_step_size  "${HMC_STEP_SIZE}" \
      --hmc_leapfrog   "${HMC_LEAPFROG}" \
      --max_train      "${MAX_TRAIN}" \
      2>&1 | tee "${OUTDIR}/logs/train_${PROBLEM}_seed${SEED}.log"

    # 2. Weight optimisation
    python3 -m weight_opt.run_repeats_bayesian \
      --problem        "${PROBLEM}" \
      --outdir         "${OUTDIR}" \
      --priors         "${PRIORS}" \
      --seed           "${SEED}" \
      --hidden         "${HIDDEN}" \
      --depth          "${DEPTH}" \
      --lr             "${LR}" \
      --data_dir       "${DATA_DIR}" \
      --out_root       "${OUTDIR}/weight_experiments" \
      --run_name       "${PROBLEM}_seed${SEED}" \
      --seed_base      "${SEED}" \
      --runs           "${RUNS}" \
      --max_train      "${MAX_TRAIN}" \
      2>&1 | tee "${OUTDIR}/logs/weight_opt_${PROBLEM}_seed${SEED}.log"

    # 3. Predictive evaluation
    python3 evaluation_predictive.py \
      --problem        "${PROBLEM}" \
      --outdir         "${OUTDIR}" \
      --seed           "${SEED}" \
      --hidden         "${HIDDEN}" \
      --depth          "${DEPTH}" \
      --lr             "${LR}" \
      --priors         "${PRIORS}" \
      --data_dir       "${DATA_DIR}" \
      --max_train      "${MAX_TRAIN}" \
      --split          test \
      2>&1 | tee "${OUTDIR}/logs/eval_${PROBLEM}_seed${SEED}.log"

    echo "Done: Seed ${SEED}"
    echo ""

done

echo "All 4 seeds complete. Results in ${OUTDIR}/weight_experiments/"