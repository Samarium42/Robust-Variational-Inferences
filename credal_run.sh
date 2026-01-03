#!/usr/bin/env bash
set -euo pipefail

DATASETS=("eight_ring" "spirals" "moons")
PRIORS=("gaussian" "gaussian_narrow" "gaussian_wide" "student_t" "ringmix")
SEEDS=(0 1 2 3 4 5 6 7 8 9)

OUTDIR="out_fm_solver"
mkdir -p "${OUTDIR}/credal" "${OUTDIR}/logs"

STEPS=3000
BATCH=4096
LR=0.001

for SEED in "${SEEDS[@]}"; do
  for d in "${DATASETS[@]}"; do
    MANIFEST="${OUTDIR}/manifest_${d}_seed${SEED}.txt"
    : > "${MANIFEST}"

    for p in "${PRIORS[@]}"; do
      echo "${d},${p},${OUTDIR}/fm_${d}_cond_h256_d6_lr0.001.pt,${OUTDIR}/samples_${d}_${p}_cond_h256_d6_lr0.001_seed${SEED}.npy" >> "${MANIFEST}"
    done

    LOGFILE="${OUTDIR}/logs/opt_credal_${d}_seed${SEED}.log"

    python3 opt_credal_kl.py \
      --manifest "${MANIFEST}" \
      --steps "${STEPS}" \
      --batch "${BATCH}" \
      --lr "${LR}" \
      --seed "${SEED}" \
      --out "${OUTDIR}/credal/weights_${d}_seed${SEED}.json" \
      2>&1 | tee "${LOGFILE}"
  done
done
