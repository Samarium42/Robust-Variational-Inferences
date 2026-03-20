import argparse
import csv
import os
from typing import Dict, List

import numpy as np

from train_flow import make_dataset

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
    precompute_logpk,
    read_manifest,
    draw_dataset_points,
    fit_kde,
    mixture_nll_from_logpk,
    write_json,
    write_weights_json,
)


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

    ap.add_argument("--manifest", required=True)
    ap.add_argument("--train_dataset", required=True)
    ap.add_argument("--val_dataset", required=True)
    ap.add_argument("--test_dataset", required=True)

    ap.add_argument("--out_root", default="out_fm_solver/weight_experiments")
    ap.add_argument("--run_name", default="")
    ap.add_argument("--seed_base", type=int, default=0)
    ap.add_argument("--runs", type=int, default=10)

    ap.add_argument("--bandwidth", type=float, default=0.2)
    ap.add_argument("--n_fit", type=int, default=8000)
    ap.add_argument("--n_eval", type=int, default=10000)

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

    args = ap.parse_args()
    cfg = vars(args)

    entries = read_manifest(args.manifest)
    dataset_key = entries[0].dataset_name
    priors = [e.prior_name for e in entries]
    sample_paths = [e.sample_path for e in entries]

    run_id = args.run_name.strip() if args.run_name.strip() else now_run_id()
    out_dir = os.path.join(args.out_root, f"{dataset_key}_repeats_{run_id}")
    os.makedirs(out_dir, exist_ok=True)

    # Fit KDEs once
    kdes = []
    for sp in sample_paths:
        samples = np.load(sp)
        kdes.append(fit_kde(samples, bandwidth=args.bandwidth))

    algos = ["uniform", "best_single", "grid", "em", "proj_gd", "mirror", "frank_wolfe", "coord"]

    # Store results: algo -> list of (train, val, test)
    results: Dict[str, List[Dict]] = {a: [] for a in algos}

    for r in range(args.runs):
        run_seed = args.seed_base + r
        run_subdir = os.path.join(out_dir, f"run_{r:02d}_seed{run_seed}")
        os.makedirs(run_subdir, exist_ok=True)

        # Fit points for weight fitting
        x_fit = draw_dataset_points(make_dataset, args.val_dataset, args.n_fit, seed=10_000 + run_seed)
        logpk_fit = precompute_logpk(kdes, x_fit)

        # Eval points for reporting
        x_train = draw_dataset_points(make_dataset, args.train_dataset, args.n_eval, seed=20_000 + run_seed)
        x_val = draw_dataset_points(make_dataset, args.val_dataset, args.n_eval, seed=30_000 + run_seed)
        x_test = draw_dataset_points(make_dataset, args.test_dataset, args.n_eval, seed=40_000 + run_seed)

        logpk_train = precompute_logpk(kdes, x_train)
        logpk_val = precompute_logpk(kdes, x_val)
        logpk_test = precompute_logpk(kdes, x_test)

        for algo in algos:
            w = run_one_algo(algo, logpk_fit, cfg, run_seed=run_seed)

            weights_path = os.path.join(run_subdir, f"weights_{algo}.json")
            write_weights_json(weights_path, dataset_key=dataset_key, priors=priors, w=w)

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

    # Write per run CSV
    per_run_csv = os.path.join(out_dir, "per_run.csv")
    with open(per_run_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["run", "seed", "algo", "nll_train", "nll_val", "nll_test", "weights"],
        )
        writer.writeheader()
        for algo in algos:
            for rec in results[algo]:
                writer.writerow(
                    {
                        "run": rec["run"],
                        "seed": rec["seed"],
                        "algo": rec["algo"],
                        "nll_train": rec["nll_train"],
                        "nll_val": rec["nll_val"],
                        "nll_test": rec["nll_test"],
                        "weights": " ".join([f"{p}:{w:.4f}" for p, w in zip(priors, rec["weights"])]),
                    }
                )

    # Aggregate mean and std
    agg_csv = os.path.join(out_dir, "aggregate.csv")
    with open(agg_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "algo",
                "nll_train_mean",
                "nll_train_std",
                "nll_val_mean",
                "nll_val_std",
                "nll_test_mean",
                "nll_test_std",
            ],
        )
        writer.writeheader()
        for algo in algos:
            arr_train = np.array([r["nll_train"] for r in results[algo]], dtype=float)
            arr_val = np.array([r["nll_val"] for r in results[algo]], dtype=float)
            arr_test = np.array([r["nll_test"] for r in results[algo]], dtype=float)

            writer.writerow(
                {
                    "algo": algo,
                    "nll_train_mean": float(arr_train.mean()),
                    "nll_train_std": float(arr_train.std(ddof=1)),
                    "nll_val_mean": float(arr_val.mean()),
                    "nll_val_std": float(arr_val.std(ddof=1)),
                    "nll_test_mean": float(arr_test.mean()),
                    "nll_test_std": float(arr_test.std(ddof=1)),
                }
            )

    meta = {
        "dataset_key": dataset_key,
        "priors": priors,
        "runs": args.runs,
        "seed_base": args.seed_base,
        "bandwidth": args.bandwidth,
        "n_fit": args.n_fit,
        "n_eval": args.n_eval,
        "train_dataset": args.train_dataset,
        "val_dataset": args.val_dataset,
        "test_dataset": args.test_dataset,
        "algos": algos,
        "out_dir": out_dir,
    }
    write_json(os.path.join(out_dir, "meta.json"), meta)

    print(f"Saved repeats to: {out_dir}")
    print(f"Per run CSV: {per_run_csv}")
    print(f"Aggregate CSV: {agg_csv}")


if __name__ == "__main__":
    main()