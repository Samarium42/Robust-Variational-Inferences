import argparse
import numpy as np
from train_flow import make_dataset
from weight_opt.utils import read_manifest, draw_dataset_points, fit_kde, precompute_logpk

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--val_dataset", required=True)
    ap.add_argument("--bandwidth", type=float, default=0.2)
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    entries = read_manifest(args.manifest)
    priors = [e.prior_name for e in entries]

    kdes = []
    for e in entries:
        samp = np.load(e.sample_path)
        kdes.append(fit_kde(samp, bandwidth=args.bandwidth))

    x = draw_dataset_points(make_dataset, args.val_dataset, args.n, seed=12345 + args.seed)
    logpk = precompute_logpk(kdes, x)

    winner = np.argmax(logpk, axis=1)
    frac = [(priors[k], float((winner == k).mean())) for k in range(len(priors))]
    frac.sort(key=lambda t: -t[1])

    print("Winner fraction on val points")
    for p, f in frac:
        print(f"  {p:16s}  {f:.4f}")

    margins = np.sort(logpk, axis=1)[:, -1] - np.sort(logpk, axis=1)[:, -2]
    print(f"Median margin (top1 minus top2): {float(np.median(margins)):.4f}")

if __name__ == "__main__":
    main()