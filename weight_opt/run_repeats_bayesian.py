"""
weight_opt/run_repeats_bayesian.py

Weight optimisation for Bayesian inference problems.

Same algorithms as run_repeats.py, but instead of KDE log-densities,
the logpk matrix is computed from predictive log-likelihoods:

    logpk[i, k] = log [ (1/S) Σ_s p(y_i | x_i, θ_{k,s}) ]

This means all 8 weight-optimisation algorithms (EM, mirror descent,
Frank-Wolfe, etc.) work unchanged — they just see a different logpk.

Usage:
    python -m weight_opt.run_repeats_bayesian \
        --problem german_credit \
        --outdir out_fm_solver \
        --seed 0 \
        --runs 10
"""

import argparse
import csv
import os
from typing import Dict, List

import numpy as np

from weight_opt.algorithms import (
    coordinate_pair_search,
    em_weights,
    frank_wolfe,
    grid_search_simplex,
    mirror_descent_exponentiated,
    projected_gd,
    weights_best_single,
    weights_uniform,
)
from weight_opt.utils import (
    now_run_id,
    mixture_nll_from_logpk,
    write_json,
    write_weights_json,
)

from datasets.german_credit import GermanCreditBLR


def compute_predictive_logpk(
    problem: GermanCreditBLR,
    sample_arrays: Dict[str, np.ndarray],
    prior_names: List[str],
    split: str = "val",
    max_samples: int = 5000,
) -> np.ndarray:
    """
    Compute logpk matrix for Bayesian predictive evaluation.

    logpk[i, k] = log [ (1/S_k) Σ_s p(y_i | x_i, θ_{k,s}) ]

    Returns: (N, K) array
    """
    if split == "val":
        X, y = problem.X_val, problem.y_val
    elif split == "test":
        X, y = problem.X_test, problem.y_test
    else:
        X, y = problem.X_train, problem.y_train

    N = X.shape[0]
    K = len(prior_names)
    logpk = np.zeros((N, K), dtype=np.float64)

    for k, pname in enumerate(prior_names):
        theta = sample_arrays[pname][:max_samples].astype(np.float64)
        S = theta.shape[0]

        # logits: (S, N)
        logits = theta @ X.T
        # log p(y_i | x_i, θ_s) = y_i * logit - log(1 + exp(logit))
        log_p = y[None, :] * logits - np.logaddexp(0, logits)  # (S, N)

        # log mean_s exp(log_p) = logsumexp(log_p, axis=0) - log(S)
        max_lp = log_p.max(axis=0, keepdims=True)
        logpk[:, k] = max_lp.squeeze(0) + np.log(np.exp(log_p - max_lp).mean(axis=0))

    return logpk


def run_one_algo(algo_name: str, logpk_fit: np.ndarray, cfg: Dict, run_seed: int) -> np.ndarray:
    K = logpk_fit.shape[1]

    if algo_name == "uniform":
        return weights_uniform(K)
    if algo_name == "best_single":
        return weights_best_single(logpk_fit)
    if algo_name == "grid":
        return grid_search_simplex(logpk_fit, step=float(cfg["grid_step"]))
    if algo_name == "em":
        return em_weights(logpk_fit, iters=int(cfg["em_iters"]), tol=float(cfg["em_tol"]))
    if algo_name == "proj_gd":
        return projected_gd(
            logpk_fit,
            iters=int(cfg["pgd_iters"]),
            lr=float(cfg["pgd_lr"]),
            momentum=float(cfg["pgd_momentum"]),
        )
    if algo_name == "mirror":
        return mirror_descent_exponentiated(
            logpk_fit,
            iters=int(cfg["md_iters"]),
            lr=float(cfg["md_lr"]),
        )
    if algo_name == "frank_wolfe":
        return frank_wolfe(
            logpk_fit,
            iters=int(cfg["fw_iters"]),
            gamma_rule=str(cfg["fw_gamma"]),
            line_search_points=int(cfg["fw_ls_points"]),
        )
    if algo_name == "coord":
        return coordinate_pair_search(
            logpk_fit,
            iters=int(cfg["cd_iters"]),
            grid=int(cfg["cd_grid"]),
            seed=run_seed,
        )
    raise ValueError(f"Unknown algo: {algo_name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", default="german_credit")
    ap.add_argument("--outdir", default="out_fm_solver")
    ap.add_argument("--priors", default="gaussian,gaussian_wide,cauchy,laplace,student_t")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--data_dir", default="data")

    ap.add_argument("--out_root", default="out_fm_solver/weight_experiments")
    ap.add_argument("--run_name", default="")
    ap.add_argument("--seed_base", type=int, default=0)
    ap.add_argument("--runs", type=int, default=10)

    # Algo hyperparams (same defaults as run_repeats.py)
    ap.add_argument("--grid_step", type=float, default=0.05)
    ap.add_argument("--em_iters", type=int, default=200)
    ap.add_argument("--em_tol", type=float, default=1e-10)
    ap.add_argument("--pgd_iters", type=int, default=400)
    ap.add_argument("--pgd_lr", type=float, default=0.05)
    ap.add_argument("--pgd_momentum", type=float, default=0.9)
    ap.add_argument("--md_iters", type=int, default=400)
    ap.add_argument("--md_lr", type=float, default=0.05)
    ap.add_argument("--fw_iters", type=int, default=200)
    ap.add_argument("--fw_gamma", type=str, default="harmonic")
    ap.add_argument("--fw_ls_points", type=int, default=50)
    ap.add_argument("--cd_iters", type=int, default=300)
    ap.add_argument("--cd_grid", type=int, default=50)

    # Predictive eval params
    ap.add_argument("--max_samples_per_component", type=int, default=5000)
    ap.add_argument("--max_train", type=int, default=0,
                    help="Must match the value used during training.")

    args = ap.parse_args()
    cfg = vars(args)

    prior_names = [p.strip() for p in args.priors.split(",")]

    # Load problem
    problem = GermanCreditBLR(data_dir=args.data_dir, seed=args.seed,
                              max_train=args.max_train)

    # Load flow samples
    sample_arrays = {}
    for p in prior_names:
        tag = f"{args.problem}_{p}_cond_h{args.hidden}_d{args.depth}_lr{args.lr}_seed{args.seed}"
        path = os.path.join(args.outdir, "samples", f"samples_{tag}.npy")
        sample_arrays[p] = np.load(path)
        print(f"  Loaded {sample_arrays[p].shape[0]} samples for prior={p}")

    # Setup output
    run_id = args.run_name.strip() if args.run_name.strip() else now_run_id()
    out_dir = os.path.join(args.out_root, f"{args.problem}_repeats_{run_id}")
    os.makedirs(out_dir, exist_ok=True)

    algos = ["uniform", "best_single", "grid", "em", "proj_gd", "mirror", "frank_wolfe", "coord"]
    results: Dict[str, List[Dict]] = {a: [] for a in algos}

    for r in range(args.runs):
        run_seed = args.seed_base + r
        run_subdir = os.path.join(out_dir, f"run_{r:02d}_seed{run_seed}")
        os.makedirs(run_subdir, exist_ok=True)

        # Compute logpk on val for weight fitting
        # Use bootstrap resampling of the flow samples for variance across runs
        rng = np.random.default_rng(10_000 + run_seed)
        resampled = {}
        for p in prior_names:
            arr = sample_arrays[p]
            idx = rng.integers(0, arr.shape[0], size=arr.shape[0])
            resampled[p] = arr[idx]

        logpk_val = compute_predictive_logpk(
            problem, resampled, prior_names,
            split="val", max_samples=args.max_samples_per_component,
        )

        # Also compute on test for reporting
        logpk_test = compute_predictive_logpk(
            problem, resampled, prior_names,
            split="test", max_samples=args.max_samples_per_component,
        )

        logpk_train = compute_predictive_logpk(
            problem, resampled, prior_names,
            split="train", max_samples=args.max_samples_per_component,
        )

        for algo in algos:
            # Fit weights on val logpk
            # Note: algorithms minimise NLL = -mean(log mixture density)
            # For predictive: logpk[i,k] is already log p(y_i|x_i, component_k)
            # so mixture NLL = -mean log Σ_k w_k exp(logpk[i,k])
            # This is exactly what mixture_nll_from_logpk computes!
            w = run_one_algo(algo, logpk_val, cfg, run_seed=run_seed)

            weights_path = os.path.join(run_subdir, f"weights_{algo}.json")
            write_weights_json(weights_path, dataset_key=args.problem, priors=prior_names, w=w)

            # NLL here means -mean_log_predictive_likelihood (lower = better)
            nll_train = mixture_nll_from_logpk(logpk_train, w)
            nll_val = mixture_nll_from_logpk(logpk_val, w)
            nll_test = mixture_nll_from_logpk(logpk_test, w)

            rec = {
                "run": r,
                "seed": run_seed,
                "algo": algo,
                "nll_train": float(nll_train),
                "nll_val": float(nll_val),
                "nll_test": float(nll_test),
                "weights": [float(x) for x in (np.asarray(w, dtype=float) / np.sum(w))],
            }
            results[algo].append(rec)
            write_json(os.path.join(run_subdir, f"metrics_{algo}.json"), rec)

        print(f"Finished run {r+1}/{args.runs} seed={run_seed}")

    # ---- Write CSVs (same format as run_repeats.py) ----
    per_run_csv = os.path.join(out_dir, "per_run.csv")
    with open(per_run_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["run", "seed", "algo", "nll_train", "nll_val", "nll_test", "weights"],
        )
        writer.writeheader()
        for algo in algos:
            for rec in results[algo]:
                writer.writerow({
                    "run": rec["run"],
                    "seed": rec["seed"],
                    "algo": rec["algo"],
                    "nll_train": rec["nll_train"],
                    "nll_val": rec["nll_val"],
                    "nll_test": rec["nll_test"],
                    "weights": " ".join(
                        f"{p}:{w:.4f}" for p, w in zip(prior_names, rec["weights"])
                    ),
                })

    agg_csv = os.path.join(out_dir, "aggregate.csv")
    with open(agg_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "algo", "nll_train_mean", "nll_train_std",
            "nll_val_mean", "nll_val_std",
            "nll_test_mean", "nll_test_std",
        ])
        writer.writeheader()
        for algo in algos:
            arr_train = np.array([r["nll_train"] for r in results[algo]])
            arr_val = np.array([r["nll_val"] for r in results[algo]])
            arr_test = np.array([r["nll_test"] for r in results[algo]])
            writer.writerow({
                "algo": algo,
                "nll_train_mean": float(arr_train.mean()),
                "nll_train_std": float(arr_train.std(ddof=1)),
                "nll_val_mean": float(arr_val.mean()),
                "nll_val_std": float(arr_val.std(ddof=1)),
                "nll_test_mean": float(arr_test.mean()),
                "nll_test_std": float(arr_test.std(ddof=1)),
            })

    meta = {
        "problem": args.problem,
        "priors": prior_names,
        "runs": args.runs,
        "seed_base": args.seed_base,
        "dim": problem.dim,
        "n_train": len(problem.y_train),
        "n_val": len(problem.y_val),
        "n_test": len(problem.y_test),
        "algos": algos,
        "out_dir": out_dir,
    }
    write_json(os.path.join(out_dir, "meta.json"), meta)

    print(f"\nSaved to: {out_dir}")
    print(f"Per-run CSV: {per_run_csv}")
    print(f"Aggregate CSV: {agg_csv}")


if __name__ == "__main__":
    main()