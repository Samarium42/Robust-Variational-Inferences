#!/usr/bin/env bash
set -euo pipefail

OUTDIR="out_fm_solver"
SEED=0

DATASET_TRAIN="radon_mn"
DATASET_VAL="radon_mn_val"
DATASET_TEST="radon_mn_test"

PRIORS=("gaussian" "gaussian_narrow" "student_t")

STEPS=6000
BATCH=2048
HIDDEN=256
DEPTH=6
LR=0.001
WEIGHT_DECAY=0.01
PRINT_EVERY=200

NSAMPLES=20000
STEP_SIZE=0.02

# Weight-optimisation params (run_repeats)
WEIGHT_RUNS=10
N_FIT=8000
N_EVAL=10000
BANDWIDTH=0.2

mkdir -p "${OUTDIR}/models" "${OUTDIR}/samples" "${OUTDIR}/logs" "data"

echo "OUTDIR=${OUTDIR}"
echo "TRAIN=${DATASET_TRAIN}  VAL=${DATASET_VAL}  TEST=${DATASET_TEST}"
echo "SEED=${SEED}"
echo "PRIORS=$(IFS=,; echo "${PRIORS[*]}")"

# ----------------------------------------------------------------
# 1. Train conditional flow model
# ----------------------------------------------------------------
MODEL_PATH="${OUTDIR}/models/fm_${DATASET_TRAIN}_seed${SEED}.pt"

if [[ -f "${MODEL_PATH}" ]]; then
  echo "Found model, skipping training: ${MODEL_PATH}"
else
  python3 train_flow.py \
    --dataset "${DATASET_TRAIN}" \
    --priors "$(IFS=,; echo "${PRIORS[*]}")" \
    --steps "${STEPS}" \
    --batch "${BATCH}" \
    --hidden "${HIDDEN}" \
    --depth "${DEPTH}" \
    --lr "${LR}" \
    --weight_decay "${WEIGHT_DECAY}" \
    --seed "${SEED}" \
    --outdir "${OUTDIR}" \
    --print_every "${PRINT_EVERY}" \
    --model_path "${MODEL_PATH}" \
    2>&1 | tee "${OUTDIR}/logs/train_${DATASET_TRAIN}_seed${SEED}.log"
fi

# ----------------------------------------------------------------
# 2. Sample from each prior
# ----------------------------------------------------------------
for p in "${PRIORS[@]}"; do
  SAMPLE_PATH="${OUTDIR}/samples/samples_${DATASET_TRAIN}_${p}_cond_h${HIDDEN}_d${DEPTH}_lr${LR}_seed${SEED}.npy"

  if [[ -f "${SAMPLE_PATH}" ]]; then
    echo "Found samples, skipping: ${SAMPLE_PATH}"
  else
    python3 sample_flow.py \
      --dataset "${DATASET_TRAIN}" \
      --prior "${p}" \
      --model_path "${MODEL_PATH}" \
      --n_samples "${NSAMPLES}" \
      --step_size "${STEP_SIZE}" \
      --lr "${LR}" \
      --seed "${SEED}" \
      --outdir "${OUTDIR}/samples" \
      2>&1 | tee "${OUTDIR}/logs/sample_${DATASET_TRAIN}_${p}_seed${SEED}.log"
  fi
done

# ----------------------------------------------------------------
# 3. Build manifest
# ----------------------------------------------------------------
MANIFEST="${OUTDIR}/manifest_${DATASET_TRAIN}_seed${SEED}.txt"
: > "${MANIFEST}"
for p in "${PRIORS[@]}"; do
  SAMPLE_PATH="${OUTDIR}/samples/samples_${DATASET_TRAIN}_${p}_cond_h${HIDDEN}_d${DEPTH}_lr${LR}_seed${SEED}.npy"
  echo "${DATASET_VAL},${p},${MODEL_PATH},${SAMPLE_PATH}" >> "${MANIFEST}"
done
echo "Wrote manifest: ${MANIFEST}"
cat "${MANIFEST}"

# ----------------------------------------------------------------
# 4. Weight optimisation — direct NLL, all algorithms
#    (replaces opt_credal_kl.py which used a critic-based
#     adversarial lower bound — wrong algorithm vs write-up)
# ----------------------------------------------------------------
WEIGHT_OUT="${OUTDIR}/weight_experiments"

echo ""
echo "Running weight optimisation (all algorithms, ${WEIGHT_RUNS} runs)..."

python3 -m weight_opt.run_repeats \
  --manifest "${MANIFEST}" \
  --train_dataset "${DATASET_TRAIN}" \
  --val_dataset "${DATASET_VAL}" \
  --test_dataset "${DATASET_TEST}" \
  --out_root "${WEIGHT_OUT}" \
  --run_name "seed${SEED}" \
  --seed_base "${SEED}" \
  --runs "${WEIGHT_RUNS}" \
  --bandwidth "${BANDWIDTH}" \
  --n_fit "${N_FIT}" \
  --n_eval "${N_EVAL}" \
  2>&1 | tee "${OUTDIR}/logs/weight_opt_${DATASET_TRAIN}_seed${SEED}.log"

# ----------------------------------------------------------------
# 5. Print summary
# ----------------------------------------------------------------
echo ""
echo "============================================"
echo "Results: ${DATASET_TRAIN} (seed ${SEED})"
echo "============================================"

python3 - <<'PY'
import os, csv, glob

weight_out = os.environ.get("WEIGHT_OUT", "out_fm_solver/weight_experiments")
# run_repeats names the dir from the dataset_key in the manifest
candidates = glob.glob(os.path.join(weight_out, "*_repeats_seed0", "aggregate.csv"))
if not candidates:
    candidates = glob.glob(os.path.join(weight_out, "*", "aggregate.csv"))

if not candidates:
    print("ERROR: No aggregate.csv found. Check weight_opt log.")
    exit(1)

agg_csv = sorted(candidates)[-1]  # most recent
per_run_csv = os.path.join(os.path.dirname(agg_csv), "per_run.csv")

print(f"Reading: {agg_csv}\n")
print(f"{'Algorithm':<16s}  {'Test NLL mean':>14s}  {'Test NLL std':>14s}  {'Val NLL mean':>14s}")
print("-" * 64)

with open(agg_csv) as f:
    for row in csv.DictReader(f):
        print(
            f"{row['algo']:<16s}  "
            f"{float(row['nll_test_mean']):>14.6f}  "
            f"{float(row['nll_test_std']):>14.6f}  "
            f"{float(row['nll_val_mean']):>14.6f}"
        )

print(f"\nPer-run CSV : {per_run_csv}")
print(f"Results dir : {os.path.dirname(agg_csv)}")
PY