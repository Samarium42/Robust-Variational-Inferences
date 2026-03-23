"""
evaluation_predictive.py — Evaluate flow-based posteriors via predictive log-likelihood.

Replaces KDE-NLL evaluation (which fails in >5D) with proper Bayesian predictive
evaluation on held-out data:

    pred_LL = (1/N) Σ_i log [ (1/S) Σ_s p(y_i | x_i, θ_s) ]

For each weight-optimisation algorithm, the mixture predictive LL is:
    pred_LL_mix = (1/N) Σ_i log [ Σ_k w_k · (1/S_k) Σ_s p(y_i | x_i, θ_{k,s}) ]

Reads sample .npy files produced by train_bayesian.py and weight files
produced by weight_opt/run_repeats.py.

Usage:
    python evaluation_predictive.py \
        --problem german_credit \
        --outdir out_fm_solver \
        --seed 0
"""

import argparse
import csv
import json
import os
from typing import Dict, List

import numpy as np
from scipy import stats as scipy_stats

from datasets.german_credit import GermanCreditBLR


def load_flow_samples(outdir: str, problem_name: str, prior_name: str,
                      hidden: int, depth: int, lr: float, seed: int) -> np.ndarray:
    """Load flow posterior samples for a given prior."""
    tag = f"{problem_name}_{prior_name}_cond_h{hidden}_d{depth}_lr{lr}_seed{seed}"
    path = os.path.join(outdir, "samples", f"samples_{tag}.npy")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing samples: {path}")
    return np.load(path)


def predictive_log_lik_mixture(
    problem: GermanCreditBLR,
    sample_arrays: Dict[str, np.ndarray],
    weights: Dict[str, float],
    split: str = "test",
    max_samples_per_component: int = 5000,
) -> float:
    """
    Compute mixture predictive log-likelihood.

    For each data point i:
      p(y_i | x_i) = Σ_k w_k · (1/S_k) Σ_s p(y_i | x_i, θ_{k,s})

    Returns mean log p(y|x) over the split.
    """
    if split == "test":
        X, y = problem.X_test, problem.y_test
    elif split == "val":
        X, y = problem.X_val, problem.y_val
    else:
        X, y = problem.X_train, problem.y_train

    N = X.shape[0]

    # For numerical stability, work in log space
    # log p(y_i|x_i) = log Σ_k w_k · mean_s exp(log p(y_i|x_i,θ_s))
    # = logsumexp_k [ log(w_k) + log mean_s exp(log_lik_{k,s,i}) ]

    log_mixture = np.full(N, -np.inf)

    for prior_name, w in weights.items():
        if w < 1e-12:
            continue
        theta = sample_arrays[prior_name][:max_samples_per_component]
        S = theta.shape[0]

        # logits: (S, N)
        logits = theta @ X.T
        # log p(y_i | x_i, θ_s)
        log_p = y[None, :] * logits - np.logaddexp(0, logits)  # (S, N)

        # log mean_s exp(log_p) = logsumexp(log_p, axis=0) - log(S)
        max_lp = log_p.max(axis=0, keepdims=True)
        log_mean_p = max_lp.squeeze(0) + np.log(np.exp(log_p - max_lp).mean(axis=0))

        # Add log(w_k) + log_mean_p to mixture (logsumexp across components)
        term = np.log(w) + log_mean_p
        log_mixture = np.logaddexp(log_mixture, term)

    return float(log_mixture.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", default="german_credit")
    ap.add_argument("--outdir", default="out_fm_solver")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--data_dir", default="data")

    ap.add_argument("--priors", default="gaussian,gaussian_wide,cauchy,laplace,student_t")

    # Where weight_opt/run_repeats.py saved results
    ap.add_argument("--weight_exp_dir", default="",
                    help="Path to weight_experiments results directory. "
                         "If empty, auto-detect from outdir.")

    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--max_train", type=int, default=0,
                    help="Must match the value used during training.")
    args = ap.parse_args()

    prior_names = [p.strip() for p in args.priors.split(",")]

    # Load problem for predictive evaluation
    problem = GermanCreditBLR(data_dir=args.data_dir, seed=args.seed,
                              max_train=args.max_train)

    # Load flow samples for each prior
    sample_arrays = {}
    for p in prior_names:
        sample_arrays[p] = load_flow_samples(
            args.outdir, args.problem, p,
            args.hidden, args.depth, args.lr, args.seed,
        )
        print(f"  Loaded {sample_arrays[p].shape[0]} samples for prior={p}")

    # ---- Per-prior predictive LL ----
    print(f"\n{'='*60}")
    print(f"Per-prior predictive log-likelihood ({args.split} split)")
    print(f"{'='*60}")

    single_lls = {}
    for p in prior_names:
        ll = problem.predictive_log_lik(sample_arrays[p], split=args.split)
        acc = problem.accuracy(sample_arrays[p], split=args.split)
        single_lls[p] = ll
        print(f"  {p:16s}  pred_LL={ll:.6f}  acc={acc:.4f}")

    best_prior = max(single_lls, key=single_lls.get)
    best_single_ll = single_lls[best_prior]
    print(f"\n  Best single: {best_prior} (pred_LL={best_single_ll:.6f})")

    # ---- MCMC reference baseline ----
    print(f"\n{'='*60}")
    print(f"MCMC reference predictive log-likelihood ({args.split} split)")
    print(f"{'='*60}")

    problem.ensure_posteriors(prior_names)
    for p in prior_names:
        mcmc_samples = problem.sample_posterior(p, 5000, seed=999).numpy()
        ll_mcmc = problem.predictive_log_lik(mcmc_samples, split=args.split)
        acc_mcmc = problem.accuracy(mcmc_samples, split=args.split)
        print(f"  {p:16s}  MCMC pred_LL={ll_mcmc:.6f}  acc={acc_mcmc:.4f}")

    # ---- Mixture evaluation (from weight_opt results) ----
    weight_exp_dir = args.weight_exp_dir
    if not weight_exp_dir:
        # Auto-detect: look for the directory
        base = os.path.join(args.outdir, "weight_experiments")
        candidates = [d for d in os.listdir(base) if args.problem in d] if os.path.isdir(base) else []
        if candidates:
            weight_exp_dir = os.path.join(base, sorted(candidates)[-1])

    if weight_exp_dir and os.path.isdir(weight_exp_dir):
        per_run_csv = os.path.join(weight_exp_dir, "per_run.csv")
        if os.path.isfile(per_run_csv):
            print(f"\n{'='*60}")
            print(f"Mixture predictive log-likelihood by algorithm ({args.split} split)")
            print(f"{'='*60}")

            # Read per_run.csv to get weights per algorithm per run
            import csv as csv_mod
            algo_weights: Dict[str, List[Dict[str, float]]] = {}
            with open(per_run_csv) as f:
                reader = csv_mod.DictReader(f)
                for row in reader:
                    algo = row["algo"]
                    # Parse weights like "gaussian:0.2000 gaussian_wide:0.3000 ..."
                    w_str = row["weights"]
                    w_dict = {}
                    for part in w_str.split():
                        pname, val = part.split(":")
                        w_dict[pname] = float(val)
                    if algo not in algo_weights:
                        algo_weights[algo] = []
                    algo_weights[algo].append(w_dict)

            results = []
            for algo in sorted(algo_weights.keys()):
                lls = []
                for w_dict in algo_weights[algo]:
                    ll = predictive_log_lik_mixture(
                        problem, sample_arrays, w_dict, split=args.split
                    )
                    lls.append(ll)

                mean_ll = np.mean(lls)
                std_ll = np.std(lls, ddof=1) if len(lls) > 1 else 0.0
                delta = mean_ll - best_single_ll
                results.append({
                    "algo": algo,
                    "pred_ll_mean": mean_ll,
                    "pred_ll_std": std_ll,
                    "delta_vs_best_single": delta,
                    "n_runs": len(lls),
                })
                print(f"  {algo:16s}  pred_LL={mean_ll:.6f} ± {std_ll:.6f}  "
                      f"Δ vs best_single={delta:+.6f}")

            # Save results
            out_csv = os.path.join(args.outdir, f"predictive_summary_{args.problem}.csv")
            with open(out_csv, "w", newline="") as f:
                writer = csv_mod.DictWriter(f, fieldnames=[
                    "algo", "pred_ll_mean", "pred_ll_std",
                    "delta_vs_best_single", "n_runs",
                ])
                writer.writeheader()
                writer.writerows(results)
            print(f"\nSaved summary: {out_csv}")

            # Paired sign test: is the best mixture algorithm better than best_single?
            best_algo = max(results, key=lambda r: r["pred_ll_mean"])
            print(f"\n  Best algorithm: {best_algo['algo']} "
                  f"(pred_LL={best_algo['pred_ll_mean']:.6f})")

    else:
        print(f"\nNo weight experiment results found. Run weight_opt/run_repeats.py first.")


if __name__ == "__main__":
    main()