#!/usr/bin/env bash
set -euo pipefail

# ---- directories ----
OUTDIR="out_fm_solver"
LOGDIR="${OUTDIR}/logs"
mkdir -p "${LOGDIR}"

# ---- script parameters ----
MANIFEST="${OUTDIR}/manifest.txt"
STEPS=3000
BATCH=4096
LR=1e-3
SEED=1337
OUT_WEIGHTS="${OUTDIR}/credal_weights.json"

# ---- log file ----
LOGFILE="${LOGDIR}/credal_opt.log"

echo "======================================"
echo " Running credal-set optimisation"
echo " Manifest: ${MANIFEST}"
echo " Steps:    ${STEPS}"
echo " Batch:    ${BATCH}"
echo " LR:       ${LR}"
echo " Output:   ${OUT_WEIGHTS}"
echo "======================================"

# ---- run the Python optimisation ----
python3 opt_credal_kl.py \
    --manifest "${MANIFEST}" \
    --steps "${STEPS}" \
    --batch "${BATCH}" \
    --lr "${LR}" \
    --out "${OUT_WEIGHTS}" \
    2>&1 | tee "${LOGFILE}"

echo "======================================"
echo " Credal-set weights saved to ${OUT_WEIGHTS}"
echo " Full logs: ${LOGFILE}"
echo "======================================"
