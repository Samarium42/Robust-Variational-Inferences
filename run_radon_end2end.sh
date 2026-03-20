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

CREDAL_STEPS=3000
CREDAL_BATCH=4096
CREDAL_LR=0.001
CREDAL_PRINT_EVERY=200

N_EVAL=10000
BANDWIDTH=0.2

mkdir -p "${OUTDIR}/models" "${OUTDIR}/samples" "${OUTDIR}/credal" "${OUTDIR}/logs" "data"

echo "OUTDIR=${OUTDIR}"
echo "TRAIN=${DATASET_TRAIN}  VAL=${DATASET_VAL}  TEST=${DATASET_TEST}"
echo "SEED=${SEED}"
echo "PRIORS=$(IFS=,; echo "${PRIORS[*]}")"

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

MANIFEST="${OUTDIR}/manifest_${DATASET_TRAIN}_seed${SEED}.txt"
: > "${MANIFEST}"
for p in "${PRIORS[@]}"; do
  SAMPLE_PATH="${OUTDIR}/samples/samples_${DATASET_TRAIN}_${p}_cond_h${HIDDEN}_d${DEPTH}_lr${LR}_seed${SEED}.npy"
  echo "${DATASET_VAL},${p},${MODEL_PATH},${SAMPLE_PATH}" >> "${MANIFEST}"
done
echo "Wrote manifest: ${MANIFEST}"
cat "${MANIFEST}"

WEIGHTS_PATH="${OUTDIR}/credal/weights_${DATASET_TRAIN}_seed${SEED}.json"
if [[ -f "${WEIGHTS_PATH}" ]]; then
  echo "Found weights, skipping: ${WEIGHTS_PATH}"
else
  python3 opt_credal_kl.py \
    --manifest "${MANIFEST}" \
    --steps "${CREDAL_STEPS}" \
    --batch "${CREDAL_BATCH}" \
    --lr "${CREDAL_LR}" \
    --seed "${SEED}" \
    --print_every "${CREDAL_PRINT_EVERY}" \
    --out "${WEIGHTS_PATH}" \
    2>&1 | tee "${OUTDIR}/logs/opt_${DATASET_TRAIN}_seed${SEED}.log"
fi

python3 - <<PY
import os, json
import numpy as np
from sklearn.neighbors import KernelDensity
from train_flow import make_dataset

OUTDIR = "${OUTDIR}"
DATASET_TRAIN = "${DATASET_TRAIN}"
DATASET_VAL = "${DATASET_VAL}"
DATASET_TEST = "${DATASET_TEST}"
HIDDEN = ${HIDDEN}
DEPTH = ${DEPTH}
LR = ${LR}
SEED = ${SEED}
N_EVAL = ${N_EVAL}
BANDWIDTH = ${BANDWIDTH}

weights_path = os.path.join(OUTDIR, "credal", f"weights_{DATASET_TRAIN}_seed{SEED}.json")
with open(weights_path) as f:
    wobj = json.load(f)

if DATASET_VAL in wobj:
    priors = wobj[DATASET_VAL]["priors"]
    weights = wobj[DATASET_VAL]["weights"]
else:
    raise ValueError(f"Expected key {DATASET_VAL} in {weights_path}. Keys: {list(wobj.keys())}")

def load_samples(prior_name: str):
    fname = f"samples_{DATASET_TRAIN}_{prior_name}_cond_h{HIDDEN}_d{DEPTH}_lr{LR}_seed{SEED}.npy"
    path = os.path.join(OUTDIR, "samples", fname)
    return np.load(path)

def sample_data(dist, n):
    x = dist.sample(n)
    if hasattr(x, "cpu"):
        x = x.cpu().numpy()
    return x

def estimate_nll(samples, data_dist, n_eval=N_EVAL, bandwidth=BANDWIDTH):
    kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth).fit(samples)
    x_eval = sample_data(data_dist, n_eval)
    log_q = kde.score_samples(x_eval)
    return float(-log_q.mean())

Ptest = make_dataset(DATASET_TEST)

nlls = {}
for p in priors:
    nlls[p] = estimate_nll(load_samples(p), Ptest)

best_p = min(nlls, key=nlls.get)
best_single = nlls[best_p]

mix_parts = []
for p, w in zip(priors, weights):
    samp = load_samples(p)
    n = max(1, int(w * len(samp)))
    mix_parts.append(samp[:n])
mix_samples = np.vstack(mix_parts)
mix_nll = estimate_nll(mix_samples, Ptest)

print("")
print("Per prior NLL")
for p in sorted(nlls, key=nlls.get):
    print(f"  {p:16s}  {nlls[p]:.6f}")
print("")
print("Best single")
print(f"  {best_p:16s}  {best_single:.6f}")
print("")
print("Mixture")
print(f"  mix_nll={mix_nll:.6f}")
print("")
print("Weights")
for p, w in zip(priors, weights):
    print(f"  {p:16s}  {w:.4f}")
PY
