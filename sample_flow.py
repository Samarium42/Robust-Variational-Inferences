import os
import argparse
import numpy as np
import torch
import random

from flow_matching.solver import ODESolver
from train_flow import Device, VelocityResNet, ConditionalWrapper, make_prior


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset",
        default="eight_ring",
        choices=["eight_ring", "spirals", "moons", "radon_mn", "old_faithful", "hybrid"],
    )
    ap.add_argument(
        "--prior",
        default="gaussian",
        choices=["gaussian", "gaussian_narrow", "gaussian_wide", "student_t", "ringmix"],
    )
    ap.add_argument("--outdir", default="out_fm_solver")
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--n_samples", type=int, default=20000)
    ap.add_argument("--step_size", type=float, default=0.02)
    ap.add_argument("--model_path", default="", help="Path to trained conditional model checkpoint.")
    ap.add_argument("--seed", type=int, default=0, help="Random seed for reproducible sampling.")
    return ap.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()
    set_seed(args.seed)

    os.makedirs(args.outdir, exist_ok=True)

    # Resolve model path
    if args.model_path:
        model_path = args.model_path
    else:
        tag = f"{args.dataset}_cond_h{args.hidden}_d{args.depth}_lr{args.lr}"
        model_path = os.path.join(args.outdir, f"fm_{tag}.pt")

    ckpt = torch.load(model_path, map_location=Device)
    prior_names = ckpt["prior_names"]

    if args.prior not in prior_names:
        raise ValueError(
            f"prior '{args.prior}' not found in checkpoint prior_names={prior_names}"
        )

    prior_id = prior_names.index(args.prior)

    model = VelocityResNet(
        hidden=ckpt["hidden"],
        depth=ckpt["depth"],
        num_priors=len(prior_names),
        prior_emb_dim=ckpt["prior_emb_dim"],
        dropout=ckpt["dropout"],
    ).to(Device)

    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    solver = ODESolver(ConditionalWrapper(model, prior_id))

    # Sample from prior
    x0 = make_prior(args.prior).sample(args.n_samples).to(Device)

    with torch.no_grad():
        samples = solver.sample(x0, step_size=args.step_size).cpu().numpy()

    # Save with seed in filename
    samp_tag = (
        f"{args.dataset}_{args.prior}_cond_h{ckpt['hidden']}_"
        f"d{ckpt['depth']}_lr{args.lr}_seed{args.seed}"
    )
    samp_path = os.path.join(args.outdir, f"samples_{samp_tag}.npy")
    np.save(samp_path, samples)

    print(f"Saved samples to {samp_path}")


if __name__ == "__main__":
    main()
