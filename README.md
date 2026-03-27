\# Robust Variational Inference via Credal Sets

A research codebase exploring **credal mixtures for flow-based variational inference under prior uncertainty**. Instead of committing to a single prior, we maintain a credal set — a convex mixture of K approximate posteriors learned under different priors — and optimise the mixture weights to minimise held-out negative log-likelihood.

---

## Core Idea

Given a model with uncertain prior specification, we:

1. Train a single **conditional flow** `q(θ | prior_id)` that learns approximate posteriors under K different priors simultaneously, using a shared `VelocityResNet` with FiLM-conditioned prior embeddings
2. Fit **credal mixture weights** `w` on the simplex by minimising a KL-divergence lower bound via an adversarial critic
3. Show the optimised credal mixture **dominates any single-prior posterior** in held-out predictive log-likelihood, and is robust to prior misspecification

Experiments run on synthetic 2D datasets (eight-ring, spirals, moons), real regression data (Minnesota Radon, Old Faithful), and Bayesian logistic regression (German Credit).

---

## Repository Layout
```
├── train_flow.py              # Conditional flow training (synthetic datasets)
├── train_bayesian.py          # Conditional flow training (Bayesian inference problems)
├── sample_flow.py             # ODE sampling from trained checkpoints
├── opt_credal_kl.py           # Credal weight optimisation via adversarial critic
├── evaluation_nll.py          # KDE-NLL evaluation (2D datasets)
├── evaluation_predictive.py   # Predictive log-likelihood evaluation (classification)
│
├── datasets/                  # Dataset classes (eight_ring, spirals, moons, radon, german_credit)
├── priors/                    # Prior distributions (Gaussian, StudentT, Laplace, Cauchy, RingMix)
├── weight_opt/                # Multi-algorithm weight optimisation (EM, PGD, Frank-Wolfe)
│
├── run_all.sh                 # Train all flows end-to-end
├── credal_run.sh              # Optimise credal weights across datasets and seeds
├── run_german_credit_end2end.sh
├── run_radon_end2end.sh
├── run_old_faithful_end2end.sh
│
├── credal_sets_report.ipynb   # Main results notebook
├── Preliminary_research_note.pdf
└── requirements.txt
```

---

## Setup
```bash
git clone https://github.com/Samarium42/Robust-Variational-Inferences
cd Robust-Variational-Inferences
pip install -r requirements.txt   # Python 3.10+
```

---

## Reproducing Experiments

### 1 — Train conditional flows
```bash
bash run_all.sh                          # synthetic datasets (eight_ring, spirals, moons)
bash run_german_credit_end2end.sh        # Bayesian logistic regression
bash run_radon_end2end.sh                # hierarchical regression
```

### 2 — Optimise credal mixture weights
```bash
bash credal_run.sh
```

This runs EM, PGD, and Frank-Wolfe on the simplex across 10 seeds and saves per-run results to `out_fm_solver/weight_experiments/`.

### 3 — Evaluate
```bash
# Predictive log-likelihood + calibration (German Credit)
python evaluation_predictive.py --problem german_credit --outdir out_fm_solver

# KDE-NLL for 2D synthetic datasets
python evaluation_nll.py
```

### 4 — New experiments (calibration, LOPO, ablations)

See `new_experiments.ipynb` for:
- Prior sensitivity curves
- Weight stability analysis across seeds
- Leave-One-Prior-Out (LOPO)
- Calibration curve, ECE, and Brier score
- Optimiser convergence comparison
- Shared flow vs separate models ablation

---

## Key Results

| Dataset | Best single prior NLL | Credal mixture NLL | Δ |
|---|---|---|---|
| Eight-ring | — | — | — |
| German Credit | — | — | — |

*(Fill in from `out_fm_solver/weight_experiments/all_results.csv` after running)*

---

## Method Overview
```
K priors  →  Conditional Flow  →  K approximate posteriors q_k(θ)
                                           ↓
                              Credal weight optimisation
                              (adversarial critic, simplex PGD)
                                           ↓
                              Mixture  q*(θ) = Σ w_k q_k(θ)
                                           ↓
                              Robust predictive distribution
```

The flow uses a `VelocityResNet` with **FiLM conditioning** on a learned prior embedding, so all K posteriors share parameters — one model, not K separate models. Mixture weights are fit by maximising a Donsker-Varadhan lower bound on the KL divergence using a learned critic network.

---

## Notes

- `evaluation_nll.py` is the legacy KDE-based evaluator for 2D datasets
- `evaluation_predictive.py` is the correct evaluator for higher-dimensional Bayesian problems (German Credit etc.) — uses proper predictive log-likelihood in log-space
- Hardware: trained on MPS (Apple Silicon) / CUDA; CPU fallback available
