"""
train_bayesian.py — Flow matching training for Bayesian inference problems.

Key difference from train_flow.py:
  - The target distribution x1 is PRIOR-DEPENDENT (each prior has its own posterior)
  - input_dim is inferred from the problem, not hardcoded to 2
  - Priors come from the problem class, not from make_prior()

Usage:
    python train_bayesian.py \
        --problem german_credit \
        --priors gaussian,gaussian_wide,cauchy,laplace,student_t \
        --steps 8000 --batch 1024 --hidden 256 --depth 6 \
        --seed 0 --outdir out_fm_solver

Then sample:
    python train_bayesian.py \
        --problem german_credit --sample_only \
        --prior gaussian --model_path out_fm_solver/models/fm_german_credit_seed0.pt \
        --n_samples 20000 --seed 0 --outdir out_fm_solver
"""

import argparse
import os
import random

import numpy as np
import torch
from torch import nn, Tensor

from flow_matching.path import AffineProbPath
from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.solver import ODESolver
from flow_matching.utils import ModelWrapper

# Import the model architecture from train_flow (reuse VelocityResNet etc.)
from train_flow import VelocityResNet, ConditionalWrapper, TimeEmbedding, FiLMBlock

from datasets.german_credit import GermanCreditBLR, PRIOR_REGISTRY

Device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------
# Problem factory
# ---------------------------------------------------------------

def make_problem(name: str, data_dir: str = "data", seed: int = 0, **kwargs):
    if name == "german_credit":
        return GermanCreditBLR(data_dir=data_dir, seed=seed, **kwargs)
    raise ValueError(f"Unknown problem: {name}")


# ---------------------------------------------------------------
# Training loop (prior-dependent targets)
# ---------------------------------------------------------------

def train_conditional_bayesian(
    problem,
    prior_names: list,
    steps: int = 8000,
    batch_size: int = 1024,
    lr: float = 1e-3,
    hidden: int = 256,
    depth: int = 6,
    prior_emb_dim: int = 16,
    dropout: float = 0.05,
    weight_decay: float = 1e-2,
    print_every: int = 200,
    seed: int = 0,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    input_dim = problem.dim
    K = len(prior_names)

    print(f"Training conditional flow: input_dim={input_dim}, K={K} priors, "
          f"hidden={hidden}, depth={depth}")

    model = VelocityResNet(
        input_dim=input_dim,
        hidden=hidden,
        depth=depth,
        num_priors=K,
        prior_emb_dim=prior_emb_dim,
        dropout=dropout,
    ).to(Device)

    path = AffineProbPath(scheduler=CondOTScheduler())
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    eps = 1e-4
    grad_clip = 1.0          # max gradient norm
    source_clip = 15.0       # clip heavy-tailed prior samples to [-c, c] per dim

    # Ensure posteriors are available
    problem.ensure_posteriors(prior_names)

    # Precompute posterior stats for monitoring
    post_stds = {}
    for pn in prior_names:
        s = problem.sample_posterior(pn, min(2000, problem.hmc_samples)).numpy()
        post_stds[pn] = float(np.std(s))

    huber = nn.SmoothL1Loss(beta=5.0)  # Huber loss — robust to outlier velocities

    running_loss = 0.0
    running_count = 0

    for step in range(1, steps + 1):
        # Pick a random prior
        k_idx = random.randrange(K)
        prior_name = prior_names[k_idx]

        # x1 = posterior samples under this prior (TARGET — prior-dependent)
        x1 = problem.sample_posterior(prior_name, batch_size).to(Device)

        # x0 = prior samples (SOURCE)
        # Clip to prevent extreme outliers from Cauchy / StudentT blowing up velocities
        x0 = problem.sample_prior(prior_name, batch_size).to(Device)
        x0 = x0.clamp(-source_clip, source_clip)

        # Flow matching: interpolate and regress velocity
        u = torch.rand(batch_size, device=Device)
        t = u * (1 - 2 * eps) + eps

        ps = path.sample(t=t, x_0=x0, x_1=x1)

        prior_id = torch.full((batch_size,), k_idx, device=Device, dtype=torch.long)
        pred = model(ps.x_t, ps.t, prior_id)
        loss = huber(pred, ps.dx_t)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        opt.zero_grad()

        running_loss += loss.item()
        running_count += 1

        if step % print_every == 0 or step == 1:
            avg = running_loss / running_count
            print(f"Step {step:6d}/{steps}  Loss: {loss.item():.4f}  "
                  f"AvgLoss: {avg:.4f}  prior: {prior_name}")
            running_loss = 0.0
            running_count = 0

    return model, prior_names


# ---------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------

def sample_from_model(
    model_path: str,
    problem,
    prior_name: str,
    n_samples: int = 20000,
    step_size: float = 0.01,
    seed: int = 0,
) -> np.ndarray:
    torch.manual_seed(seed)

    ckpt = torch.load(model_path, map_location=Device)
    prior_names = ckpt["prior_names"]
    input_dim = ckpt["input_dim"]

    if prior_name not in prior_names:
        raise ValueError(f"prior '{prior_name}' not in checkpoint: {prior_names}")

    prior_id = prior_names.index(prior_name)

    model = VelocityResNet(
        input_dim=input_dim,
        hidden=ckpt["hidden"],
        depth=ckpt["depth"],
        num_priors=len(prior_names),
        prior_emb_dim=ckpt["prior_emb_dim"],
        dropout=ckpt["dropout"],
    ).to(Device)

    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    solver = ODESolver(ConditionalWrapper(model, prior_id))
    x0 = problem.sample_prior(prior_name, n_samples).to(Device)
    x0 = x0.clamp(-15.0, 15.0)  # match training clip

    with torch.no_grad():
        samples = solver.sample(x0, step_size=step_size).cpu().numpy()

    return samples


# ---------------------------------------------------------------
# CLI
# ---------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", default="german_credit",
                    choices=["german_credit"])
    ap.add_argument("--priors", default="gaussian,gaussian_wide,cauchy,laplace,student_t",
                    help="Comma-separated priors to train jointly.")
    ap.add_argument("--prior", default="gaussian",
                    help="Single prior for sampling (with --sample_only).")

    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--prior_emb_dim", type=int, default=16)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--weight_decay", type=float, default=1e-2)

    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", default="out_fm_solver")
    ap.add_argument("--print_every", type=int, default=200)

    ap.add_argument("--sample_only", action="store_true")
    ap.add_argument("--n_samples", type=int, default=20000)
    ap.add_argument("--step_size", type=float, default=0.01,
                    help="ODE step size (smaller may be needed for higher-D)")
    ap.add_argument("--model_path", default="")

    ap.add_argument("--data_dir", default="data")

    # HMC parameters (passed through to problem)
    ap.add_argument("--hmc_samples", type=int, default=10000)
    ap.add_argument("--hmc_warmup", type=int, default=2000)
    ap.add_argument("--hmc_step_size", type=float, default=0.005)
    ap.add_argument("--hmc_leapfrog", type=int, default=25)

    # Data regime
    ap.add_argument("--max_train", type=int, default=0,
                    help="Subsample training data to this many observations. "
                         "0 = use all. Low values (50-100) create prior-sensitive regime.")

    return ap.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    problem = make_problem(
        args.problem,
        data_dir=args.data_dir,
        seed=args.seed,
        hmc_samples=args.hmc_samples,
        hmc_warmup=args.hmc_warmup,
        hmc_step_size=args.hmc_step_size,
        hmc_leapfrog=args.hmc_leapfrog,
        max_train=args.max_train,
    )

    prior_names_list = [p.strip() for p in args.priors.split(",") if p.strip()]

    tag = f"{args.problem}_cond_h{args.hidden}_d{args.depth}_lr{args.lr}"
    default_model_path = os.path.join(args.outdir, "models", f"fm_{args.problem}_seed{args.seed}.pt")
    model_path = args.model_path if args.model_path else default_model_path

    if not args.sample_only:
        print(f"Config => problem={args.problem}  dim={problem.dim}  "
              f"priors={','.join(prior_names_list)}  hidden={args.hidden}  "
              f"depth={args.depth}  lr={args.lr}  seed={args.seed}  steps={args.steps}")

        model, prior_names = train_conditional_bayesian(
            problem,
            prior_names_list,
            steps=args.steps,
            batch_size=args.batch,
            lr=args.lr,
            hidden=args.hidden,
            depth=args.depth,
            prior_emb_dim=args.prior_emb_dim,
            dropout=args.dropout,
            weight_decay=args.weight_decay,
            print_every=args.print_every,
            seed=args.seed,
        )

        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        ckpt = {
            "state_dict": model.state_dict(),
            "prior_names": prior_names,
            "hidden": args.hidden,
            "depth": args.depth,
            "prior_emb_dim": args.prior_emb_dim,
            "dropout": args.dropout,
            "input_dim": problem.dim,
        }
        torch.save(ckpt, model_path)
        print(f"Saved model: {model_path}")

    # ---- Sampling (single prior) ----
    if args.sample_only or True:  # always sample after training
        samples_dir = os.path.join(args.outdir, "samples")
        os.makedirs(samples_dir, exist_ok=True)

        priors_to_sample = [args.prior] if args.sample_only else prior_names_list

        for p in priors_to_sample:
            samp_tag = (f"{args.problem}_{p}_cond_h{args.hidden}_d{args.depth}"
                        f"_lr{args.lr}_seed{args.seed}")
            samp_path = os.path.join(samples_dir, f"samples_{samp_tag}.npy")

            if os.path.isfile(samp_path) and args.sample_only:
                print(f"Samples exist, skipping: {samp_path}")
                continue

            print(f"Sampling: prior={p}, n={args.n_samples}")
            samples = sample_from_model(
                model_path, problem, p,
                n_samples=args.n_samples,
                step_size=args.step_size,
                seed=args.seed,
            )
            np.save(samp_path, samples)
            print(f"Saved: {samp_path}")


if __name__ == "__main__":
    main()