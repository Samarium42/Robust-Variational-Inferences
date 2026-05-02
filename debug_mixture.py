"""
debug_mixture.py — run this in your project root to diagnose the +10.8 NLL issue.

Usage:
    python3 debug_mixture.py
"""
import csv
import os
import numpy as np

# ── Config — matches your evaluation call ─────────────────────────────────
OUTDIR      = "out_fm_solver_bc"
PROBLEM     = "breast_cancer"
SEED        = 0
HIDDEN      = 256
DEPTH       = 6
LR          = 0.001
PRIOR_NAMES = ["gaussian", "gaussian_wide", "cauchy", "laplace", "student_t"]
MAX_TRAIN   = 75
DATA_DIR    = "data"

# ── 1. Load the problem ────────────────────────────────────────────────────
from train_bayesian import make_problem
problem = make_problem(PROBLEM, data_dir=DATA_DIR, seed=SEED, max_train=MAX_TRAIN)

print(f"X_test shape:  {problem.X_test.shape}")
print(f"y_test shape:  {problem.y_test.shape}")
print(f"y_test unique: {np.unique(problem.y_test)}")
print()

# ── 2. Load flow samples ───────────────────────────────────────────────────
sample_arrays = {}
for p in PRIOR_NAMES:
    tag  = f"{PROBLEM}_{p}_cond_h{HIDDEN}_d{DEPTH}_lr{LR}_seed{SEED}"
    path = os.path.join(OUTDIR, "samples", f"samples_{tag}.npy")
    arr  = np.load(path)
    sample_arrays[p] = arr
    print(f"  {p:16s}  shape={arr.shape}  mean={arr.mean():.4f}  std={arr.std():.4f}")

print()

# ── 3. Find per_run.csv ────────────────────────────────────────────────────
base       = os.path.join(OUTDIR, "weight_experiments")
candidates = sorted([d for d in os.listdir(base) if PROBLEM in d])
print(f"Weight experiment directories found:")
for c in candidates:
    print(f"  {c}")

weight_exp_dir = os.path.join(base, candidates[-1])
per_run_csv    = os.path.join(weight_exp_dir, "per_run.csv")
print(f"\nUsing: {per_run_csv}")
print()

# ── 4. Read first few rows of per_run.csv ─────────────────────────────────
with open(per_run_csv) as f:
    reader = csv.DictReader(f)
    rows   = list(reader)

print(f"Columns: {list(rows[0].keys())}")
print(f"First row: {rows[0]}")
print()

# ── 5. Parse weights for best_single ──────────────────────────────────────
bs_rows = [r for r in rows if r["algo"] == "best_single"]
print(f"best_single rows: {len(bs_rows)}")
if bs_rows:
    w_str = bs_rows[0]["weights"]
    print(f"  weights string: '{w_str}'")
    w_dict = {}
    for part in w_str.split():
        pname, val = part.split(":")
        w_dict[pname] = float(val)
    print(f"  parsed weights: {w_dict}")
    print(f"  weight keys match sample_arrays keys: "
          f"{set(w_dict.keys()) == set(sample_arrays.keys())}")
print()

# ── 6. Manually compute mixture pred-LL for best_single ───────────────────
X, y   = problem.X_test, problem.y_test
N      = X.shape[0]
S_max  = 5000

log_mixture = np.full(N, -np.inf)

for prior_name, w in w_dict.items():
    if w < 1e-12:
        continue
    theta  = sample_arrays[prior_name][:S_max]
    S      = theta.shape[0]
    logits = theta @ X.T                                      # (S, N)
    log_p  = y[None, :] * logits - np.logaddexp(0, logits)   # (S, N)

    max_lp    = log_p.max(axis=0, keepdims=True)
    log_mean_p = (max_lp.squeeze(0)
                  + np.log(np.exp(log_p - max_lp).mean(axis=0)))  # (N,)

    print(f"  {prior_name:16s}  w={w:.4f}  "
          f"logits range=[{logits.min():.2f}, {logits.max():.2f}]  "
          f"log_mean_p range=[{log_mean_p.min():.4f}, {log_mean_p.max():.4f}]  "
          f"mean log_mean_p={log_mean_p.mean():.6f}")

    term        = np.log(w) + log_mean_p
    log_mixture = np.logaddexp(log_mixture, term)

result = float(log_mixture.mean())
print(f"\nMixture pred_LL for best_single: {result:.6f}")
print(f"Expected (should match gaussian per-prior): "
      f"{problem.predictive_log_lik(sample_arrays['gaussian'][:S_max], split='test'):.6f}")
