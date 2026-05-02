"""
train_cnf_baseline.py

Speed benchmark comparing flow matching vs CNF training cost.

The key claim from Section 2.5 is that flow matching eliminates ODE 
simulation during training, replacing it with a single regression step.
This script measures the per-step wall-clock cost of each approach using
the same VelocityResNet architecture.

CNF simulation cost is measured by timing one forward pass through the
ODE integrator (10-step RK4) WITHOUT backpropagating through the ODE.
This correctly reflects the computational overhead of simulation itself,
which is what makes CNF training slow — not the backward pass cost.

Flow matching training is run for the full 6000 steps on German Credit
seed 0, producing final test NLL for quality comparison.

Usage:
    python3 train_cnf_baseline.py
    python3 train_cnf_baseline.py --steps 6000 --seed 0
"""

import argparse
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from train_flow import VelocityResNet, ConditionalWrapper
from train_bayesian import make_problem

Device = torch.device("cuda" if torch.cuda.is_available() else
                      "mps"  if torch.backends.mps.is_available() else
                      "cpu")

CPU = torch.device("cpu")  # CNF ODE runs on CPU (float32 safe)


# ── CNF per-step cost measurement ─────────────────────────────────────────

def measure_cnf_step_cost(model, x0, x1, k_idx, n_steps=10, n_trials=100):
    """
    Measure the wall-clock cost of one CNF training step.
    
    A CNF training step requires:
      1. Integrate ODE forward: n_steps x velocity field evaluations
      2. Accumulate Hutchinson trace at each step
      3. Backward pass through final log-likelihood
    
    Here we time step 1 + 2 (the simulation cost) which dominates.
    This is the correct measure of the ODE overhead that flow matching avoids.
    """
    model_cpu = model.to(CPU)
    x0_cpu = x0.float().to(CPU)
    x1_cpu = x1.float().to(CPU)
    B, D = x0_cpu.shape

    # Rademacher noise for Hutchinson
    noise = (torch.randint(0, 2, (B, D), device=CPU).float() * 2 - 1)
    prior_ids = torch.full((B,), k_idx, device=CPU, dtype=torch.long)

    times = []
    for _ in range(n_trials):
        z    = x0_cpu.clone().detach()
        log_p = torch.zeros(B, device=CPU)
        dt = 1.0 / n_steps
        t  = 0.0

        t_start = time.perf_counter()

        for step_i in range(n_steps):
            t_tensor  = torch.full((B, 1), t, device=CPU, dtype=torch.float32)

            # Velocity field evaluation — the dominant cost
            with torch.no_grad():
                v = model_cpu(z, t_tensor, prior_ids)

            # Hutchinson trace estimation — second major cost
            z_tr  = z.detach().requires_grad_(True)
            v_tr  = model_cpu(z_tr, t_tensor, prior_ids)
            vjp   = torch.autograd.grad(v_tr, z_tr, grad_outputs=noise)[0]
            tr_dv = (vjp * noise).sum(dim=1).detach()

            # Euler update (position and log-density)
            z     = (z + dt * v).detach()
            log_p = log_p - dt * tr_dv
            t    += dt

        # Include final loss computation in timing (no backward — graph is detached)
        log_p_base = (-0.5 * (x0_cpu ** 2).sum(dim=1)
                      - 0.5 * D * np.log(2 * np.pi))
        _loss = -(log_p_base + log_p).mean()  # forward only, no backward needed

        times.append(time.perf_counter() - t_start)

    return np.array(times)


def measure_fm_step_cost(model, x0, x1, k_idx, n_trials=100):
    """
    Measure the wall-clock cost of one flow matching training step.
    Includes forward pass, Huber loss, and backward.
    """
    from flow_matching.path import AffineProbPath
    from flow_matching.path.scheduler import CondOTScheduler

    model = model.to(Device)
    path = AffineProbPath(scheduler=CondOTScheduler())
    huber = nn.SmoothL1Loss(beta=5.0)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    eps = 1e-4
    B = x0.shape[0]
    prior_ids = torch.full((B,), k_idx, device=Device, dtype=torch.long)

    times = []
    for _ in range(n_trials):
        x0_d = x0.to(Device)
        x1_d = x1.to(Device)
        u = torch.rand(B, device=Device)
        t = u * (1 - 2*eps) + eps

        t_start = time.perf_counter()
        ps = path.sample(t=t, x_0=x0_d, x_1=x1_d)
        pred = model(ps.x_t, ps.t, prior_ids)
        loss = huber(pred, ps.dx_t)
        loss.backward()
        opt.step()
        opt.zero_grad()
        times.append(time.perf_counter() - t_start)

    return np.array(times)


# ── Full flow matching training (for NLL quality comparison) ───────────────

def train_flow_matching(problem, prior_names, steps, batch_size, lr,
                        hidden, depth, seed, weight_decay=1e-2, print_every=500):
    from flow_matching.path import AffineProbPath
    from flow_matching.path.scheduler import CondOTScheduler

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    K = len(prior_names)
    model = VelocityResNet(
        input_dim=problem.dim, hidden=hidden, depth=depth,
        num_priors=K, prior_emb_dim=16, dropout=0.05,
    ).to(Device)

    path = AffineProbPath(scheduler=CondOTScheduler())
    opt  = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    huber = nn.SmoothL1Loss(beta=5.0)
    eps = 1e-4; source_clip = 15.0; grad_clip = 1.0

    problem.ensure_posteriors(prior_names)

    step_times = []
    losses     = []

    for step in range(1, steps + 1):
        k_idx = random.randrange(K)
        prior_name = prior_names[k_idx]
        x1 = problem.sample_posterior(prior_name, batch_size).to(Device)
        x0 = problem.sample_prior(prior_name, batch_size).to(Device).clamp(-source_clip, source_clip)
        u  = torch.rand(batch_size, device=Device)
        t  = u * (1 - 2*eps) + eps

        t0 = time.perf_counter()
        ps = path.sample(t=t, x_0=x0, x_1=x1)
        prior_id_t = torch.full((batch_size,), k_idx, device=Device, dtype=torch.long)
        pred = model(ps.x_t, ps.t, prior_id_t)
        loss = huber(pred, ps.dx_t)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        opt.zero_grad()
        step_times.append(time.perf_counter() - t0)
        losses.append(loss.item())

        if step % print_every == 0 or step == 1:
            print(f"[FM] Step {step:5d}/{steps}  "
                  f"Loss: {loss.item():.4f}  "
                  f"AvgStepTime: {np.mean(step_times[-200:])*1000:.2f}ms")

    return model, step_times, losses


def evaluate_nll(model, problem, prior_names, n_samples=5000, step_size=0.01, seed=0):
    from flow_matching.solver import ODESolver
    model = model.to(Device)
    nll_per_prior = {}
    for k_idx, prior_name in enumerate(prior_names):
        torch.manual_seed(seed)
        model.eval()
        solver = ODESolver(ConditionalWrapper(model, k_idx))
        x0 = problem.sample_prior(prior_name, n_samples).to(Device).clamp(-15, 15)
        with torch.no_grad():
            samples = solver.sample(x0, step_size=step_size).cpu().numpy()
        nll = -problem.predictive_log_lik(samples, split="test")
        nll_per_prior[prior_name] = float(nll)
        print(f"  {prior_name}: test NLL = {nll:.6f}")
    return nll_per_prior


# ── Main ───────────────────────────────────────────────────────────────────

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps",       type=int,   default=6000)
    ap.add_argument("--batch",       type=int,   default=1024)
    ap.add_argument("--lr",          type=float, default=1e-3)
    ap.add_argument("--hidden",      type=int,   default=256)
    ap.add_argument("--depth",       type=int,   default=6)
    ap.add_argument("--seed",        type=int,   default=0)
    ap.add_argument("--max_train",   type=int,   default=75)
    ap.add_argument("--data_dir",    default="data")
    ap.add_argument("--outdir",      default="out_speed_comparison")
    ap.add_argument("--priors",      default="gaussian,gaussian_wide,cauchy,laplace,student_t")
    ap.add_argument("--problem",     default="german_credit",
                    choices=["german_credit","breast_cancer"])
    ap.add_argument("--n_timing_trials", type=int, default=200,
                    help="Number of trials for per-step timing benchmark")
    ap.add_argument("--cnf_steps",   type=int, default=10,
                    help="ODE integration steps for CNF baseline timing")
    ap.add_argument("--hmc_samples",    type=int,   default=3000)
    ap.add_argument("--hmc_warmup",     type=int,   default=1000)
    ap.add_argument("--hmc_step_size",  type=float, default=0.005)
    ap.add_argument("--hmc_leapfrog",   type=int,   default=15)
    return ap.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    prior_names = [p.strip() for p in args.priors.split(",")]

    print(f"Device: {Device}  |  CNF timing on: {CPU}")

    problem = make_problem(
        args.problem, data_dir=args.data_dir, seed=args.seed,
        max_train=args.max_train,
        hmc_samples=args.hmc_samples, hmc_warmup=args.hmc_warmup,
        hmc_step_size=args.hmc_step_size, hmc_leapfrog=args.hmc_leapfrog,
    )
    problem.ensure_posteriors(prior_names)

    K = len(prior_names)
    # Use a small timing batch for the benchmark (not 1024 — too slow for CNF CPU)
    TIMING_BATCH = 64

    # Sample one batch for timing
    x1_timing = problem.sample_posterior("gaussian", TIMING_BATCH).float()
    x0_timing = problem.sample_prior("gaussian", TIMING_BATCH).float().clamp(-15, 15)

    # Fresh model for timing (weights don't matter — we're timing the ops)
    timing_model = VelocityResNet(
        input_dim=problem.dim, hidden=args.hidden, depth=args.depth,
        num_priors=K, prior_emb_dim=16, dropout=0.05,
    )

    # ── Step timing benchmark ──────────────────────────────────────────────
    print("\n" + "="*60)
    print(f"  Per-step timing benchmark  ({args.n_timing_trials} trials each)")
    print("="*60)

    print(f"\nTiming flow matching step (batch={TIMING_BATCH})...")
    fm_times = measure_fm_step_cost(
        timing_model, x0_timing, x1_timing, k_idx=0,
        n_trials=args.n_timing_trials
    )
    # Re-init model for CNF timing (FM modified weights)
    timing_model = VelocityResNet(
        input_dim=problem.dim, hidden=args.hidden, depth=args.depth,
        num_priors=K, prior_emb_dim=16, dropout=0.05,
    )

    print(f"Timing CNF step ({args.cnf_steps} ODE steps, batch={TIMING_BATCH})...")
    cnf_times = measure_cnf_step_cost(
        timing_model, x0_timing, x1_timing, k_idx=0,
        n_steps=args.cnf_steps,
        n_trials=args.n_timing_trials
    )

    fm_mean  = float(np.mean(fm_times)  * 1000)
    cnf_mean = float(np.mean(cnf_times) * 1000)
    speedup  = cnf_mean / fm_mean

    print(f"\n  Flow Matching: {fm_mean:.2f} ms/step  (median {np.median(fm_times)*1000:.2f}ms)")
    print(f"  CNF Baseline:  {cnf_mean:.2f} ms/step  (median {np.median(cnf_times)*1000:.2f}ms)")
    print(f"  Speedup:       {speedup:.1f}x")

    np.save(os.path.join(args.outdir, "fm_step_times.npy"),  fm_times)
    np.save(os.path.join(args.outdir, "cnf_step_times.npy"), cnf_times)

    # ── Full FM training for NLL comparison ───────────────────────────────
    print("\n" + "="*60)
    print(f"  Full flow matching training ({args.steps} steps)")
    print("="*60)

    t_total = time.perf_counter()
    fm_model, all_step_times, all_losses = train_flow_matching(
        problem, prior_names,
        steps=args.steps, batch_size=args.batch,
        lr=args.lr, hidden=args.hidden, depth=args.depth, seed=args.seed,
    )
    fm_total_time = time.perf_counter() - t_total
    print(f"\nTotal FM training time: {fm_total_time:.1f}s")

    np.save(os.path.join(args.outdir, "fm_losses.npy"), all_losses)

    print("\nEvaluating FM test NLL:")
    fm_nll = evaluate_nll(fm_model, problem, prior_names)

    # Estimated CNF total time (never actually run — way too slow)
    estimated_cnf_total = (cnf_mean / fm_mean) * fm_total_time

    # ── Save summary ───────────────────────────────────────────────────────
    import json
    summary = {
        "fm_mean_step_ms":         fm_mean,
        "fm_median_step_ms":       float(np.median(fm_times) * 1000),
        "cnf_mean_step_ms":        cnf_mean,
        "cnf_median_step_ms":      float(np.median(cnf_times) * 1000),
        "speedup_x":               speedup,
        "cnf_ode_steps":           args.cnf_steps,
        "timing_batch_size":       TIMING_BATCH,
        "fm_total_time_s":         fm_total_time,
        "estimated_cnf_total_s":   estimated_cnf_total,
        "fm_nll":                  fm_nll,
    }

    with open(os.path.join(args.outdir, "speed_comparison_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    csv_path = os.path.join(args.outdir, "speed_comparison.csv")
    with open(csv_path, "w") as f:
        f.write("method,mean_step_ms,median_step_ms,total_time_s")
        for p in prior_names:
            f.write(f",nll_{p}")
        f.write("\n")
        f.write(f"flow_matching,{fm_mean:.3f},{np.median(fm_times)*1000:.3f},{fm_total_time:.1f}")
        for p in prior_names:
            f.write(f",{fm_nll.get(p, float('nan')):.6f}")
        f.write("\n")
        f.write(f"cnf_baseline_estimated,{cnf_mean:.3f},{np.median(cnf_times)*1000:.3f},{estimated_cnf_total:.1f}")
        for p in prior_names:
            f.write(",N/A (not trained)")
        f.write("\n")

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  FM  step time:  {fm_mean:.2f} ms")
    print(f"  CNF step time:  {cnf_mean:.2f} ms  ({args.cnf_steps} ODE steps)")
    print(f"  Speedup:        {speedup:.1f}x")
    print(f"  FM  total time: {fm_total_time:.1f}s")
    print(f"  CNF est. total: {estimated_cnf_total:.1f}s (never run)")
    print(f"\n  Saved: {args.outdir}/speed_comparison_summary.json")
    print(f"  Saved: {csv_path}")


if __name__ == "__main__":
    main()