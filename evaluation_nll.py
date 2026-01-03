import os
import json
import numpy as np
import pandas as pd

from sklearn.neighbors import KernelDensity


from datasets.eight_rings import EightGaussianRingDataset
from datasets.spirals import TwoArmSpiralsDataset
from datasets.moons import TwoMoonsDataset


# ------------------------------------------------------------
# Configuration (must match training / sampling)
# ------------------------------------------------------------

OUTDIR = "out_fm_solver"
TAG = "cond_h256_d6_lr0.001"

DATASETS = ["eight_ring", "spirals", "moons"]
PRIORS = ["gaussian", "gaussian_narrow", "gaussian_wide", "student_t", "ringmix"]
SEEDS = range(10)


# ------------------------------------------------------------
# Dataset factory
# ------------------------------------------------------------

def make_dataset(name, seed):
    """
    Returns an instantiated dataset object with .sample(n).
    """
    if name == "eight_ring":
        return EightGaussianRingDataset(seed=seed)
    elif name == "spirals":
        return TwoArmSpiralsDataset(seed=seed)
    elif name == "moons":
        return TwoMoonsDataset(seed=seed)
    else:
        raise ValueError(f"Unknown dataset: {name}")


def sample_data(dist, n):
    """
    Samples from dataset and returns numpy array [n, 2].
    """
    x = dist.sample(n)
    if hasattr(x, "cpu"):
        x = x.cpu().numpy()
    return x


# ------------------------------------------------------------
# Robust sample loader (matches your actual filenames)
# ------------------------------------------------------------

def load_samples(outdir, dataset, prior, seed):
    fname = f"samples_{dataset}_{prior}_{TAG}_seed{seed}.npy"

    # try samples/ subdirectory
    p1 = os.path.join(outdir, "samples", fname)
    if os.path.isfile(p1):
        return np.load(p1)

    # fallback: flat outdir
    p2 = os.path.join(outdir, fname)
    if os.path.isfile(p2):
        return np.load(p2)

    raise FileNotFoundError(
        f"Missing sample file:\n"
        f"  {p1}\n"
        f"  {p2}"
    )


# ------------------------------------------------------------
# NLL estimation via KDE
# ------------------------------------------------------------

def estimate_nll(samples, data_dist, n_eval=10_000, bandwidth=0.2):
    """
    Estimates -E_P[log q(x)] using Gaussian KDE on flow samples.

    This approximates KL(P || q) up to an additive constant.
    """
    kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth)
    kde.fit(samples)

    x_eval = sample_data(data_dist, n_eval)
    log_q = kde.score_samples(x_eval)

    return -log_q.mean()


# ------------------------------------------------------------
# Main evaluation loop
# ------------------------------------------------------------

rows = []

for d in DATASETS:
    print(f"\nEvaluating dataset: {d}")

    for s in SEEDS:
        print(f"  Seed {s}")

        # fresh held-out data per seed
        P = make_dataset(d, seed=10_000 + s)

        nlls = {}

        # ------------------------
        # Single-prior models
        # ------------------------
        for p in PRIORS:
            samples = load_samples(OUTDIR, d, p, s)
            nlls[p] = estimate_nll(samples, P)

        best_single = min(nlls.values())

        # ------------------------
        # Credal mixture
        # ------------------------
        weights_path = os.path.join(
            OUTDIR, "credal", f"weights_{d}_seed{s}.json"
        )
        if not os.path.isfile(weights_path):
            raise FileNotFoundError(
                f"Missing credal weights file: {weights_path}"
            )

        with open(weights_path) as f:
            wobj = json.load(f)

        # Handle both possible JSON formats safely
        if "weights" in wobj:
            weights = wobj["weights"]
        elif d in wobj and "weights" in wobj[d]:
            weights = wobj[d]["weights"]
        else:
            raise ValueError(f"Unrecognised weight file format: {weights_path}")

        if len(weights) != len(PRIORS):
            raise ValueError("Number of weights does not match number of priors")

        mix_samples = []
        for p, wp in zip(PRIORS, weights):
            samp = load_samples(OUTDIR, d, p, s)
            n = max(1, int(wp * len(samp)))
            mix_samples.append(samp[:n])

        mix_samples = np.vstack(mix_samples)
        mix_nll = estimate_nll(mix_samples, P)

        rows.append({
            "dataset": d,
            "seed": s,
            "best_single_nll": best_single,
            "mixture_nll": mix_nll,
        })


# ------------------------------------------------------------
# Save and summarise
# ------------------------------------------------------------

df = pd.DataFrame(rows)

out_csv = os.path.join(OUTDIR, "summary_table.csv")
df.to_csv(out_csv, index=False)

print("\n======================================")
print("Summary (mean ± std over seeds)")
print("======================================")
print(df.groupby("dataset").agg(["mean", "std"]))
print(f"\nSaved summary table to: {out_csv}")
