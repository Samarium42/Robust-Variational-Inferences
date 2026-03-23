#!/usr/bin/env bash
set -euo pipefail

DATASETS=("eight_ring" "spirals" "moons")
PRIORS=("gaussian" "gaussian_narrow" "gaussian_wide" "student_t" "ringmix")
SEEDS=(0 1 2 3 4 5 6 7 8 9)

OUTDIR="out_fm_solver"
WEIGHT_OUT="${OUTDIR}/weight_experiments"

# Weight-optimisation params
WEIGHT_RUNS=10
N_FIT=8000
N_EVAL=10000
BANDWIDTH=0.2

HIDDEN=256
DEPTH=6
LR=0.001

mkdir -p "${OUTDIR}/logs"

for SEED in "${SEEDS[@]}"; do
  for d in "${DATASETS[@]}"; do
    # ---- Build manifest (same format as before) ----
    MANIFEST="${OUTDIR}/manifest_${d}_seed${SEED}.txt"
    : > "${MANIFEST}"

    MODEL="${OUTDIR}/models/fm_${d}_seed${SEED}.pt"

    for p in "${PRIORS[@]}"; do
      SAMPLE="${OUTDIR}/samples/samples_${d}_${p}_cond_h${HIDDEN}_d${DEPTH}_lr${LR}_seed${SEED}.npy"
      echo "${d},${p},${MODEL},${SAMPLE}" >> "${MANIFEST}"
    done

    LOGFILE="${OUTDIR}/logs/weight_opt_${d}_seed${SEED}.log"

    echo "Weight opt: dataset=${d}, seed=${SEED} (all algorithms)"

    # ---- Run all weight-opt algorithms via run_repeats ----
    # For generative toy datasets, train/val/test are the same
    # dataset name — draw_dataset_points varies the random seed.
    python3 -m weight_opt.run_repeats \
      --manifest "${MANIFEST}" \
      --train_dataset "${d}" \
      --val_dataset "${d}" \
      --test_dataset "${d}" \
      --out_root "${WEIGHT_OUT}" \
      --run_name "seed${SEED}" \
      --seed_base "${SEED}" \
      --runs "${WEIGHT_RUNS}" \
      --bandwidth "${BANDWIDTH}" \
      --n_fit "${N_FIT}" \
      --n_eval "${N_EVAL}" \
      2>&1 | tee "${LOGFILE}"
  done
done

# ----------------------------------------------------------------
# Collect all aggregate CSVs into one combined table
# ----------------------------------------------------------------
echo ""
echo "============================================"
echo "Collecting results across all datasets/seeds"
echo "============================================"

python3 - <<'PY'
import os, csv, glob

weight_out = "out_fm_solver/weight_experiments"
pattern = os.path.join(weight_out, "*_repeats_*", "per_run.csv")
all_files = sorted(glob.glob(pattern))

if not all_files:
    print("WARNING: No per_run.csv files found.")
    exit(0)

combined = []
for fpath in all_files:
    # Extract dataset and training seed from directory name
    dirname = os.path.basename(os.path.dirname(fpath))
    # e.g. "eight_ring_repeats_seed3" -> dataset="eight_ring", train_seed=3
    parts = dirname.rsplit("_repeats_seed", 1)
    if len(parts) == 2:
        ds_name = parts[0]
        train_seed = parts[1]
    else:
        ds_name = dirname
        train_seed = "?"

    with open(fpath) as f:
        for row in csv.DictReader(f):
            row["dataset"] = ds_name
            row["train_seed"] = train_seed
            combined.append(row)

out_csv = os.path.join(weight_out, "all_results.csv")
fields = ["dataset", "train_seed", "run", "seed", "algo",
          "nll_train", "nll_val", "nll_test", "weights"]
with open(out_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    for row in combined:
        writer.writerow({k: row.get(k, "") for k in fields})

print(f"Combined results: {out_csv}")
print(f"Total rows: {len(combined)}")
PY