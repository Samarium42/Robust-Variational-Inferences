#!/usr/bin/env bash
set -euo pipefail

DATASETS=("eight_ring" "spirals" "moons")
PRIORS=("gaussian" "gaussian_narrow" "gaussian_wide" "student_t" "ringmix")
SEEDS=(0 1 2 3 4 5 6 7 8 9)

STEPS=6000
BATCH=2048
HIDDEN=256
DEPTH=6
LR=0.0005
WEIGHT_DECAY=0.01

NSAMPLES=20000
STEP_SIZE=0.02

OUTDIR="out_fm_solver"
mkdir -p "${OUTDIR}/models" "${OUTDIR}/samples"

for SEED in "${SEEDS[@]}"; do
  for d in "${DATASETS[@]}"; do
    echo "=============================="
    echo "Training conditional model: dataset=${d}, seed=${SEED}"
    echo "=============================="

    python3 train_flow.py \
      --dataset "${d}" \
      --priors "$(IFS=,; echo "${PRIORS[*]}")" \
      --steps "${STEPS}" \
      --batch "${BATCH}" \
      --hidden "${HIDDEN}" \
      --depth "${DEPTH}" \
      --lr "${LR}" \
      --weight_decay "${WEIGHT_DECAY}" \
      --seed "${SEED}" \
      --outdir "${OUTDIR}" \
      --print_every 200

    MODEL_PATH="${OUTDIR}/models/fm_${d}_seed${SEED}.pt"
    mv "${OUTDIR}/fm_${d}_cond_h${HIDDEN}_d${DEPTH}_lr${LR}.pt" "${MODEL_PATH}"

    for p in "${PRIORS[@]}"; do
      echo "Sampling: dataset=${d}, prior=${p}, seed=${SEED}"

      python3 sample_flow.py \
        --dataset "${d}" \
        --prior "${p}" \
        --model_path "${MODEL_PATH}" \
        --n_samples "${NSAMPLES}" \
        --step_size "${STEP_SIZE}" \
        --seed "${SEED}" \
        --out "${OUTDIR}/samples/samples_${d}_${p}_seed${SEED}.npy"
    done
  done
done
