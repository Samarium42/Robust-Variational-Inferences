"""
breast_cancer_blr.py  —  Breast Cancer Wisconsin (Diagnostic) dataset
for Bayesian logistic regression, mirroring the GermanCreditBLR interface.

Dataset: 30 features, 569 observations, binary classification
         (malignant=+1, benign=-1)
With max_train=75: 75/31 ≈ 2.4:1 data-to-parameter ratio
→ genuine prior-sensitive regime, comparable to German Credit (n=75, 25 params)

Usage:
    problem = BreastCancerBLR(data_dir="data", seed=0, max_train=75)
    problem.ensure_posteriors(prior_names)
"""

import os
import numpy as np
import torch
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler


class BreastCancerBLR:
    """
    Bayesian logistic regression on Breast Cancer Wisconsin (Diagnostic).

    Labels: malignant -> +1, benign -> -1
    Features: 30 continuous features, standardised to zero mean unit variance.
    Bias term appended -> D = 31 parameters total.

    Interface mirrors GermanCreditBLR exactly so all existing training,
    evaluation, and weight-optimisation scripts work without modification.
    """

    name = "breast_cancer"

    def __init__(
        self,
        data_dir: str = "data",
        seed: int = 0,
        max_train: int = 75,
        train_frac: float = 0.70,
        val_frac: float = 0.15,
        hmc_samples: int = 3000,
        hmc_warmup: int = 1000,
        hmc_step_size: float = 0.005,
        hmc_leapfrog: int = 15,
    ):
        self.data_dir = data_dir
        self.seed = seed
        self.max_train = max_train
        self.hmc_samples = hmc_samples
        self.hmc_warmup = hmc_warmup
        self.hmc_step_size = hmc_step_size
        self.hmc_leapfrog = hmc_leapfrog

        # ── Load and preprocess ───────────────────────────────────────────
        data = load_breast_cancer()
        X = data.data.astype(np.float32)          # (569, 30)
        # malignant=0 in sklearn -> map to +1; benign=1 -> -1
        y = np.where(data.target == 0, 1.0, -1.0).astype(np.float32)

        # Standardise features
        scaler = StandardScaler()
        X = scaler.fit_transform(X).astype(np.float32)

        # Append bias column
        X = np.concatenate(
            [X, np.ones((len(X), 1), dtype=np.float32)], axis=1
        )  # (569, 31)

        self.D = X.shape[1]  # 31
        self.dim = self.D    # alias used by train_bayesian.py

        # ── Train / val / test split ──────────────────────────────────────
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(X))

        n_val  = int(val_frac   * len(X))   # ~85
        n_test = int((1.0 - train_frac - val_frac) * len(X))  # ~85
        # remaining goes to train pool; we subsample max_train from it
        n_train_pool = len(X) - n_val - n_test

        train_pool_idx = idx[:n_train_pool]
        val_idx        = idx[n_train_pool : n_train_pool + n_val]
        test_idx       = idx[n_train_pool + n_val :]

        # Subsample max_train observations from the training pool
        train_idx = rng.choice(
            train_pool_idx, size=min(max_train, n_train_pool), replace=False
        )

        self.X_train = X[train_idx]
        self.y_train = y[train_idx]
        self.X_val   = X[val_idx]
        self.y_val   = y[val_idx]
        self.X_test  = X[test_idx]
        self.y_test  = y[test_idx]

        # ── Posterior cache (filled by ensure_posteriors) ─────────────────
        self._posterior_samples: dict[str, np.ndarray] = {}
        os.makedirs(data_dir, exist_ok=True)

    # ── Log-posterior (used by HMC) ───────────────────────────────────────

    def log_likelihood(self, theta: torch.Tensor) -> torch.Tensor:
        """
        Bayesian logistic regression log-likelihood.
        theta: (D,) or (S, D)
        Returns scalar or (S,)
        """
        X = torch.tensor(self.X_train, dtype=torch.float32)
        y = torch.tensor(self.y_train, dtype=torch.float32)
        if theta.dim() == 1:
            logits = X @ theta          # (N,)
        else:
            logits = X @ theta.T        # (N, S)
            logits = logits.T           # (S, N)
        return torch.nn.functional.logsigmoid(y * logits).sum(-1)

    def log_prior(self, theta: torch.Tensor, prior_name: str) -> torch.Tensor:
        """
        Log-prior for the given prior name.
        Supported: gaussian, gaussian_wide, gaussian_narrow,
                   laplace, cauchy, student_t
        """
        if prior_name == "gaussian":
            return -0.5 * (theta ** 2).sum(-1)
        elif prior_name == "gaussian_wide":
            return -0.5 * ((theta / 1.5) ** 2).sum(-1)
        elif prior_name == "gaussian_narrow":
            return -0.5 * ((theta / 0.5) ** 2).sum(-1)
        elif prior_name == "laplace":
            return -theta.abs().sum(-1)
        elif prior_name == "cauchy":
            return -torch.log1p(theta ** 2).sum(-1)
        elif prior_name == "student_t":
            nu = 3.0
            return (
                -(nu + 1) / 2
                * torch.log1p(theta ** 2 / nu)
            ).sum(-1)
        else:
            raise ValueError(f"Unknown prior: {prior_name}")

    def log_posterior(
        self, theta: torch.Tensor, prior_name: str
    ) -> torch.Tensor:
        return self.log_likelihood(theta) + self.log_prior(theta, prior_name)

    # ── HMC posterior sampling ────────────────────────────────────────────

    def ensure_posteriors(
        self,
        prior_names: list[str],
        n_samples: int = None,
        warmup: int = None,
        step_size: float = None,
        n_leapfrog: int = None,
    ) -> None:
        """
        Run HMC for each prior in prior_names (or load from cache).
        Mirrors GermanCreditBLR.ensure_posteriors exactly.
        """
        n_samples  = n_samples  or self.hmc_samples
        warmup     = warmup     or self.hmc_warmup
        step_size  = step_size  or self.hmc_step_size
        n_leapfrog = n_leapfrog or self.hmc_leapfrog
        for pname in prior_names:
            if pname in self._posterior_samples:
                continue

            cache_path = os.path.join(
                self.data_dir,
                f"breast_cancer_posteriors",
                f"posterior_{pname}_seed{self.seed}.npy",
            )
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)

            if os.path.exists(cache_path):
                self._posterior_samples[pname] = np.load(cache_path)
                print(f"  Loaded cached posterior for prior={pname}")
                continue

            print(f"  Running HMC for prior={pname} ...")
            samples = self._run_hmc(
                pname, n_samples, warmup, step_size, n_leapfrog
            )
            np.save(cache_path, samples)
            self._posterior_samples[pname] = samples
            print(f"  Cached {n_samples} samples -> {cache_path}")

    def _run_hmc(
        self,
        prior_name: str,
        n_samples: int,
        warmup: int,
        step_size: float,
        n_leapfrog: int,
    ) -> np.ndarray:
        """Simple HMC sampler using PyTorch autograd."""
        theta = torch.zeros(self.D, dtype=torch.float32)
        samples = []
        accepted = 0

        def grad_U(t):
            """Return (U(t), grad_U(t)) without retaining graph."""
            t_ = t.detach().requires_grad_(True)
            lp = self.log_posterior(t_, prior_name)
            u  = -lp
            u.backward()
            return u.item(), t_.grad.detach().clone()

        for step in range(warmup + n_samples):
            p = torch.randn(self.D)
            q = theta.clone()
            p_cur = p.clone()

            # Half-step for momentum
            _, g = grad_U(q)
            p_cur = p_cur - 0.5 * step_size * g

            # Full leapfrog steps
            for _ in range(n_leapfrog - 1):
                q = q + step_size * p_cur
                _, g = grad_U(q)
                p_cur = p_cur - step_size * g

            # Final position update + half momentum step
            q = q + step_size * p_cur
            _, g = grad_U(q)
            p_cur = p_cur - 0.5 * step_size * g

            # MH acceptance
            current_U, _ = grad_U(theta)
            prop_U,    _ = grad_U(q)
            current_H = current_U + 0.5 * (p ** 2).sum().item()
            prop_H    = prop_U    + 0.5 * (p_cur ** 2).sum().item()

            if np.log(np.random.uniform() + 1e-10) < current_H - prop_H:
                theta = q.clone()
                accepted += 1

            if step >= warmup:
                samples.append(theta.numpy().copy())

        accept_rate = accepted / (warmup + n_samples)
        print(f"    HMC accept rate: {accept_rate:.3f}")
        return np.array(samples)

    def sample_prior(self, prior_name: str, n: int) -> torch.Tensor:
        """Sample n draws from the named prior distribution."""
        theta = torch.zeros(n, self.D)
        if prior_name == "gaussian":
            return theta + torch.randn(n, self.D)
        elif prior_name == "gaussian_wide":
            return theta + 1.5 * torch.randn(n, self.D)
        elif prior_name == "gaussian_narrow":
            return theta + 0.5 * torch.randn(n, self.D)
        elif prior_name == "laplace":
            return torch.distributions.Laplace(0., 1.).sample((n, self.D))
        elif prior_name == "cauchy":
            return torch.distributions.Cauchy(0., 1.).sample((n, self.D))
        elif prior_name == "student_t":
            return torch.distributions.StudentT(df=3.).sample((n, self.D))
        else:
            raise ValueError(f"Unknown prior: {prior_name}")

    def sample_posterior(
        self, prior_name: str, n: int, seed: int = 0
    ) -> torch.Tensor:
        arr = self._posterior_samples[prior_name]
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, len(arr), size=n)
        return torch.tensor(arr[idx], dtype=torch.float32)

    # ── Evaluation helpers ────────────────────────────────────────────────

    def predictive_log_lik(
        self,
        theta_samples: np.ndarray,
        split: str = "test",
        max_s: int = 5000,
    ) -> float:
        """Mean log p(y|x) over the split using Monte Carlo."""
        if split == "test":
            X, y = self.X_test, self.y_test
        elif split == "val":
            X, y = self.X_val, self.y_val
        else:
            X, y = self.X_train, self.y_train

        theta = theta_samples[:max_s].astype(np.float64)
        X = X.astype(np.float64)
        logits = X @ theta.T          # (N, S)
        log_p  = -np.logaddexp(0, -y[:, None] * logits)  # (N, S) log sigmoid(y*logit)
        # log mean_s exp(log_p) per observation
        log_mean = (
            np.logaddexp.reduce(log_p, axis=1) - np.log(logits.shape[1])
        )
        return float(log_mean.mean())

    def accuracy(
        self,
        theta_samples: np.ndarray,
        split: str = "test",
        max_s: int = 5000,
    ) -> float:
        if split == "test":
            X, y = self.X_test, self.y_test
        elif split == "val":
            X, y = self.X_val, self.y_val
        else:
            X, y = self.X_train, self.y_train
        theta = theta_samples[:max_s].astype(np.float64)
        logits = X.astype(np.float64) @ theta.T   # (N, S)
        pred   = np.sign(logits.mean(1))
        return float((pred == y).mean())