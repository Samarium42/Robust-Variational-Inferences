python3 - <<'PY'
import os, json
import numpy as np
from sklearn.neighbors import KernelDensity
from train_flow import make_dataset

OUTDIR = "out_fm_solver"
DATASET_TRAIN = "radon_mn"
DATASET_VAL = "radon_mn_val"
DATASET_TEST = "radon_mn_test"

HIDDEN = 256
DEPTH = 6
LR = 0.001
SEED = 0

N_EVAL = 10000
BANDWIDTH = 0.2

weights_path = os.path.join(OUTDIR, "credal", f"weights_{DATASET_TRAIN}_seed{SEED}.json")
with open(weights_path) as f:
    wobj = json.load(f)

if DATASET_VAL not in wobj:
    raise ValueError(f"Expected key {DATASET_VAL} in {weights_path}. Keys: {list(wobj.keys())}")

priors = wobj[DATASET_VAL]["priors"]
weights = np.array(wobj[DATASET_VAL]["weights"], dtype=float)
weights = weights / weights.sum()

def load_samples(prior_name: str):
    fname = f"samples_{DATASET_TRAIN}_{prior_name}_cond_h{HIDDEN}_d{DEPTH}_lr{LR}_seed{SEED}.npy"
    path = os.path.join(OUTDIR, "samples", fname)
    return np.load(path)

def sample_data(dist, n):
    x = dist.sample(n)
    if hasattr(x, "cpu"):
        x = x.cpu().numpy()
    return x

def estimate_nll(kde_samples, data_dist):
    kde = KernelDensity(kernel="gaussian", bandwidth=BANDWIDTH).fit(kde_samples)
    x_eval = sample_data(data_dist, N_EVAL)
    return float(-kde.score_samples(x_eval).mean())

rng = np.random.default_rng(0)

def mixture_samples(weight_vec, total):
    parts = []
    for p, w in zip(priors, weight_vec):
        samp = load_samples(p)
        m = int(round(w * total))
        if m <= 0:
            continue
        idx = rng.choice(len(samp), size=m, replace=False if m <= len(samp) else True)
        parts.append(samp[idx])
    if not parts:
        raise ValueError("Mixture produced no samples")
    X = np.vstack(parts)
    if len(X) > total:
        X = X[:total]
    if len(X) < total:
        extra = rng.choice(len(X), size=(total - len(X)), replace=True)
        X = np.vstack([X, X[extra]])
    return X

splits = [DATASET_TRAIN, DATASET_VAL, DATASET_TEST]
total_kde = 20000

for dname in splits:
    P = make_dataset(dname)
    per = {}
    for p in priors:
        per[p] = estimate_nll(load_samples(p), P)

    best_p = min(per, key=per.get)
    best_nll = per[best_p]

    mix = estimate_nll(mixture_samples(weights, total_kde), P)
    unif = estimate_nll(mixture_samples(np.ones_like(weights) / len(weights), total_kde), P)

    print("")
    print(f"Dataset {dname}")
    for p in sorted(per, key=per.get):
        print(f"  {p:16s}  {per[p]:.6f}")
    print(f"  best_single       {best_p:16s}  {best_nll:.6f}")
    print(f"  mixture_learned                 {mix:.6f}")
    print(f"  mixture_uniform                 {unif:.6f}")
    print("  weights", {p: float(w) for p, w in zip(priors, weights)})

PY
