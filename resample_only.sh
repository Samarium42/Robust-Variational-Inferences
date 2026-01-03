#!/usr/bin/env bash
set -euo pipefail

DATASETS=("eight_ring" "moons" "spirals")
PRIORS=("gaussian" "gaussian_narrow" "gaussian_wide" "student_t" "ringmix")
SEEDS=(0 1 2 3 4 5 6 7 8 9)

OUTDIR="out_fm_solver"
NSAMPLES=20000
STEP_SIZE=0.02

mkdir -p "${OUTDIR}/samples"

for d in "${DATASETS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    MODEL="${OUTDIR}/models/fm_${d}_seed${SEED}.pt"

    for p in "${PRIORS[@]}"; do
      echo "Sampling ${d} | seed=${SEED} | prior=${p}"

      python3 sample_flow.py \
        --dataset "${d}" \
        --prior "${p}" \
        --model_path "${MODEL}" \
        --n_samples "${NSAMPLES}" \
        --step_size "${STEP_SIZE}" \
        --seed "${SEED}" \
        --outdir "${OUTDIR}"
    done
  done
done
