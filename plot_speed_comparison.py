"""
plot_speed_comparison.py

Reads results from out_speed_comparison/ and produces:
  1. exp7_speed_comparison.pdf  — two-panel figure (step time dist + NLL comparison)
  2. Prints the LaTeX table for Section 4.5.6

Usage:
    python3 plot_speed_comparison.py --outdir out_speed_comparison
"""

import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_results(outdir):
    summary_path = os.path.join(outdir, "speed_comparison_summary.json")
    fm_times  = np.load(os.path.join(outdir, "fm_step_times.npy"))
    cnf_times = np.load(os.path.join(outdir, "cnf_step_times.npy"))
    fm_losses = np.load(os.path.join(outdir, "fm_losses.npy"))
    # cnf_losses not saved — CNF is only timed, not trained to convergence

    with open(summary_path) as f:
        summary = json.load(f)

    return summary, fm_times, cnf_times, fm_losses


def make_figure(summary, fm_times, cnf_times, fm_losses,
                prior_names, outdir):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # ── Panel 1: Step time distributions ─────────────────────────────────
    ax = axes[0]
    ax.hist(np.array(fm_times) * 1000, bins=60, alpha=0.7,
            label=f"Flow Matching\n(mean {summary.get('fm_mean_step_ms', 0):.1f}ms)",
            color="#4878CF")
    ax.hist(np.array(cnf_times) * 1000, bins=60, alpha=0.7,
            label=f"CNF Baseline\n(mean {summary.get('cnf_mean_step_ms', 0):.1f}ms)",
            color="#D65F5F")
    ax.set_xlabel("Step time (ms)")
    ax.set_ylabel("Count")
    ax.set_title("Training Step Time Distribution")
    ax.legend(fontsize=9)
    speedup = summary.get("speedup_x", 1.0)
    ax.text(0.97, 0.97, f"{speedup:.1f}x speedup",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=10, fontweight="bold",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    # ── Panel 2: Training loss curves (smoothed) ──────────────────────────
    ax = axes[1]
    window = 50

    def smooth(x, w):
        return np.convolve(x, np.ones(w) / w, mode="valid")

    fm_smooth = smooth(fm_losses, window)
    steps = np.arange(len(fm_smooth))

    ax.plot(steps, fm_smooth, color="#4878CF", lw=1.5,
            label="Flow Matching")
    ax.text(0.5, 0.5, "CNF not trained\n(ODE simulation\ntoo slow)",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=9, color="#D65F5F", alpha=0.7,
            bbox=dict(boxstyle="round", facecolor="#fff0f0", alpha=0.8))
    ax.set_xlabel("Training step")
    ax.set_ylabel("Loss (smoothed)")
    ax.set_title("Training Loss Convergence")
    ax.legend(fontsize=9)

    # ── Panel 3: NLL comparison per prior ────────────────────────────────
    ax = axes[2]
    fm_nlls  = [summary.get("fm_nll", {}).get(p, float("nan")) for p in prior_names]
    cnf_nlls = [float("nan")] * len(prior_names)  # CNF not trained
    x = np.arange(len(prior_names))
    w = 0.35
    ax.bar(x - w/2, fm_nlls,  w, label="Flow Matching", color="#4878CF", alpha=0.85)
    ax.bar(x + w/2, cnf_nlls, w, label="CNF Baseline",  color="#D65F5F", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([p.replace("_", "\n") for p in prior_names], fontsize=8)
    ax.set_ylabel("Test NLL (lower is better)")
    ax.set_title("Test NLL by Prior")
    ax.legend(fontsize=9)

    plt.tight_layout()
    out_path = os.path.join(outdir, "exp7_speed_comparison.pdf")
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Saved figure: {out_path}")
    return out_path


def print_latex_table(summary, prior_names):
    fm_nlls  = summary.get("fm_nll", {})
    cnf_nlls = {}  # CNF not trained — not in summary
    speedup  = summary.get("speedup_x", 1.0)

    print("\n" + "="*60)
    print("LaTeX TABLE for Section 4.5.6")
    print("="*60)
    print(r"""
\begin{table}[h]
\centering
\caption{Speed comparison between flow matching and CNF
baseline training on German Credit low-data regime
($n = 75$, seed 0, 6{,}000 steps, identical
\texttt{VelocityResNet} architecture).}
\label{tab:speed_comparison}
{\singlespacing
\begin{tabular}{lrr}
\hline
Metric & Flow Matching & CNF Baseline \\
\hline""")

    print(f"Total training time (s) & {summary.get('fm_total_time_s', 0):.1f} "
          f"& {summary.get('estimated_cnf_total_s', summary.get('cnf_total_time_s', 0)):.1f} \\\\")
    print(f"Mean step time (ms) & {summary.get('fm_mean_step_ms', 0):.1f} "
          f"& {summary.get('cnf_mean_step_ms', 0):.1f} \\\\")
    print(f"Speedup & \\multicolumn{{2}}{{c}}{{{speedup:.1f}$\\times$}} \\\\")
    print(r"\hline")
    print(r"& \multicolumn{2}{c}{Test NLL (lower is better)} \\")
    print(r"\hline")
    for p in prior_names:
        fn = fm_nlls.get(p, float("nan"))
        cn = cnf_nlls.get(p, float("nan"))
        p_display = p.replace("_", "\\_")
        best_fm  = "\\textbf{" + f"{fn:.4f}" + "}" if fn <= cn else f"{fn:.4f}"
        best_cnf = "\\textbf{" + f"{cn:.4f}" + "}" if cn < fn else f"{cn:.4f}"
        print(f"{p_display} & {best_fm} & {best_cnf} \\\\")

    print(r"""\hline
\end{tabular}}
\end{table}""")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir",  default="out_speed_comparison")
    ap.add_argument("--priors",  default="gaussian,gaussian_wide,cauchy,laplace,student_t")
    args = ap.parse_args()

    prior_names = [p.strip() for p in args.priors.split(",")]

    print("Loading results...")
    summary, fm_times, cnf_times, fm_losses = load_results(args.outdir)

    print("\nSummary:")
    print(f"  Flow Matching: {summary.get('fm_mean_step_ms', 0):.2f}ms/step, "
          f"{summary.get('fm_total_time_s', 0):.1f}s total")
    print(f"  CNF Baseline:  {summary.get('cnf_mean_step_ms', 0):.2f}ms/step, "
          f"{summary.get('estimated_cnf_total_s', summary.get('cnf_total_time_s', 0)):.1f}s total")
    print(f"  Speedup:       {summary.get('speedup_x', 1.0):.1f}x")

    make_figure(summary, fm_times, cnf_times, fm_losses,
                prior_names, args.outdir)
    print_latex_table(summary, prior_names)


if __name__ == "__main__":
    main()