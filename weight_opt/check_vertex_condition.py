import argparse
import numpy as np
from train_flow import make_dataset
from weight_opt.utils import read_manifest, draw_dataset_points, fit_kde, precompute_logpk

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--val_dataset", required=True)
    ap.add_argument("--bandwidth", type=float, default=0.2)
    ap.add_argument("--n", type=int, default=20000)
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

    mean_ll = logpk.mean(axis=0)
    b = int(np.argmax(mean_ll))
    best = priors[b]

    print(f"Best by mean log density on val: {best}")
    print("Mean log density per component")
    for p, m in sorted(zip(priors, mean_ll), key=lambda t: -t[1]):
        print(f"  {p:16s}  {float(m):.6f}")

    print("")
    print("Vertex optimality check: E[ p_k / p_best ]")
    for k, p in enumerate(priors):
        if k == b:
            continue
        ratio = float(np.exp(logpk[:, k] - logpk[:, b]).mean())
        print(f"  {p:16s}  {ratio:.6f}")

    print("")
    print("Tail loss check: quantiles of (log p_best - log p_k)")
    qs = [0.5, 0.9, 0.95, 0.99]
    for k, p in enumerate(priors):
        if k == b:
            continue
        diff = (logpk[:, b] - logpk[:, k])
        vals = np.quantile(diff, qs)
        print(f"  vs {p:12s}  " + "  ".join([f"q{int(q*100):02d}:{float(v):.4f}" for q, v in zip(qs, vals)]))

if __name__ == "__main__":
    main()
    