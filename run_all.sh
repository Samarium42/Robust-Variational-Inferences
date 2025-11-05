#!/usr/bin/env bash
set -euo pipefail

# ---- grids ----
DATASETS=("eight_ring" "spirals" "moons")
PRIORS=("gaussian" "gaussian_narrow" "gaussian_wide" "student_t" "ringmix")

# ---- common hparams (tweak here) ----
STEPS=6000
BATCH=1024
HIDDEN=256
LR=0.001                # we found 1e-3 stable with the new net
SEED=1337
NSAMPLES=20000
STEP_SIZE=0.02          # ODE step size; feel free to tune per dataset
OUTDIR="out_fm_solver"

mkdir -p "${OUTDIR}"
LOGDIR="${OUTDIR}/logs"
mkdir -p "${LOGDIR}"

run_one() {
  local d="$1"
  local p="$2"

  echo "=== Training: dataset=${d} prior=${p} ==="
  # Train
  python3 train_flow.py \
    --dataset "${d}" \
    --prior   "${p}" \
    --steps   "${STEPS}" \
    --batch   "${BATCH}" \
    --hidden  "${HIDDEN}" \
    --lr      "${LR}" \
    --seed    "${SEED}" \
    --outdir  "${OUTDIR}" \
    --print_every 200 \
    2>&1 | tee "${LOGDIR}/train_${d}_${p}_h${HIDDEN}_lr${LR}.log"

  echo "=== Sampling: dataset=${d} prior=${p} ==="
  # Sample using the trained checkpoint (same tag used by train_flow.py)
  python3 train_flow.py \
    --dataset "${d}" \
    --prior   "${p}" \
    --hidden  "${HIDDEN}" \
    --lr      "${LR}" \
    --outdir  "${OUTDIR}" \
    --sample_only \
    --n_samples "${NSAMPLES}" \
    --step_size "${STEP_SIZE}" \
    2>&1 | tee "${LOGDIR}/sample_${d}_${p}_h${HIDDEN}_lr${LR}.log"
}

# ---- serial run (portable) ----
for d in "${DATASETS[@]}"; do
  for p in "${PRIORS[@]}"; do
    # Skip if model already exists to make restarts cheap
    TAG="${d}_${p}_h${HIDDEN}_lr${LR}"
    if [[ -f "${OUTDIR}/fm_${TAG}.pt" ]]; then
      echo "--- Found ${OUTDIR}/fm_${TAG}.pt; skipping train ---"
    else
      run_one "${d}" "${p}"
    fi

    # Ensure samples exist
    if [[ ! -f "${OUTDIR}/samples_${TAG}.npy" ]]; then
      echo "--- No samples for ${TAG}; sampling now ---"
      python3 train_flow.py \
        --dataset "${d}" \
        --prior   "${p}" \
        --hidden  "${HIDDEN}" \
        --lr      "${LR}" \
        --outdir  "${OUTDIR}" \
        --sample_only \
        --n_samples "${NSAMPLES}" \
        --step_size "${STEP_SIZE}" \
        2>&1 | tee "${LOGDIR}/sample_${d}_${p}_h${HIDDEN}_lr${LR}.log"
    fi
  done
done

# ---- manifest for downstream credal set work ----
MANIFEST="${OUTDIR}/manifest.txt"
: > "${MANIFEST}"
for d in "${DATASETS[@]}"; do
  for p in "${PRIORS[@]}"; do
    TAG="${d}_${p}_h${HIDDEN}_lr${LR}"
    echo "${d},${p},${OUTDIR}/fm_${TAG}.pt,${OUTDIR}/samples_${TAG}.npy" >> "${MANIFEST}"
  done
done
echo "Wrote manifest to ${MANIFEST}"
