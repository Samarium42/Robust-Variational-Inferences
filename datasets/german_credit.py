
import os
import math
import json
import hashlib
from urllib.request import Request, urlopen

import numpy as np
import torch
from torch import Tensor


# ---------------------------------------------------------------
# Data download & preprocessing
# ---------------------------------------------------------------

GERMAN_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "statlog/german/german.data-numeric"
)


def _download(url: str, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.isfile(path):
        return
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req, timeout=30) as r, open(path, "wb") as f:
        f.write(r.read())


def _load_german(data_dir: str):
    """Load German credit, return (X, y) with X standardised and y in {0,1}."""
    path = os.path.join(data_dir, "german.data-numeric")
    _download(GERMAN_URL, path)

    raw = np.loadtxt(path)
    X = raw[:, :-1].astype(np.float64)       # 24 features
    y = (raw[:, -1] == 2).astype(np.float64)  # 1=bad, 2=good -> 0/1

    # Standardise features (zero-mean, unit-variance)
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True)
    sd = np.maximum(sd, 1e-8)
    X = (X - mu) / sd

    # Add intercept column
    X = np.hstack([np.ones((X.shape[0], 1)), X])  # now 25 features

    return X, y


# ---------------------------------------------------------------
# Priors
# ---------------------------------------------------------------

class _Prior:
    """Base prior on R^d."""
    def __init__(self, dim: int):
        self.dim = dim

    def sample(self, n: int) -> torch.Tensor:
        raise NotImplementedError

    def log_prob(self, theta: torch.Tensor) -> torch.Tensor:
        """theta: (batch, d) or (d,) -> scalar or (batch,)"""
        raise NotImplementedError


class GaussianPriorND(_Prior):
    def __init__(self, dim: int, sigma: float = 1.0):
        super().__init__(dim)
        self.sigma = sigma

    def sample(self, n: int) -> torch.Tensor:
        return torch.randn(n, self.dim) * self.sigma

    def log_prob(self, theta: torch.Tensor) -> torch.Tensor:
        return -0.5 * (theta ** 2).sum(dim=-1) / (self.sigma ** 2) \
               - self.dim * math.log(self.sigma * math.sqrt(2 * math.pi))


class CauchyPriorND(_Prior):
    def __init__(self, dim: int, scale: float = 2.5):
        super().__init__(dim)
        self.scale = scale

    def sample(self, n: int) -> torch.Tensor:
        return torch.distributions.Cauchy(0, self.scale).sample((n, self.dim))

    def log_prob(self, theta: torch.Tensor) -> torch.Tensor:
        # sum of independent Cauchy log-pdfs
        lp = -torch.log(torch.tensor(math.pi * self.scale)) \
             - torch.log(1 + (theta / self.scale) ** 2)
        return lp.sum(dim=-1)


class LaplacePriorND(_Prior):
    def __init__(self, dim: int, scale: float = 1.0):
        super().__init__(dim)
        self.scale = scale

    def sample(self, n: int) -> torch.Tensor:
        return torch.distributions.Laplace(0, self.scale).sample((n, self.dim))

    def log_prob(self, theta: torch.Tensor) -> torch.Tensor:
        lp = -math.log(2 * self.scale) - torch.abs(theta) / self.scale
        return lp.sum(dim=-1)


class StudentTPriorND(_Prior):
    def __init__(self, dim: int, df: float = 3.0, scale: float = 2.5):
        super().__init__(dim)
        self.df = df
        self.scale = scale

    def sample(self, n: int) -> torch.Tensor:
        return torch.distributions.StudentT(self.df, 0, self.scale).sample((n, self.dim))

    def log_prob(self, theta: torch.Tensor) -> torch.Tensor:
        dist = torch.distributions.StudentT(self.df, 0, self.scale)
        return dist.log_prob(theta).sum(dim=-1)


PRIOR_REGISTRY = {
    "gaussian":      lambda d: GaussianPriorND(d, sigma=1.0),
    "gaussian_wide": lambda d: GaussianPriorND(d, sigma=10.0),
    "cauchy":        lambda d: CauchyPriorND(d, scale=2.5),
    "laplace":       lambda d: LaplacePriorND(d, scale=1.0),
    "student_t":     lambda d: StudentTPriorND(d, df=3.0, scale=2.5),
}


# ---------------------------------------------------------------
# HMC sampler (self-contained, uses PyTorch autograd)
# ---------------------------------------------------------------

def _log_likelihood(theta: torch.Tensor, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Bernoulli log-likelihood for logistic regression.
    theta: (d,)   X: (n, d)   y: (n,)
    """
    logits = X @ theta                               # (n,)
    return (y * logits - torch.logaddexp(torch.zeros_like(logits), logits)).sum()


def _hmc_step(theta, log_prob_fn, step_size, n_leapfrog):
    """Single HMC transition. Returns (new_theta, accepted)."""
    d = theta.shape[0]
    p = torch.randn(d, dtype=theta.dtype, device=theta.device)

    # Current Hamiltonian
    theta_curr = theta.detach().clone().requires_grad_(True)
    lp_curr = log_prob_fn(theta_curr)
    H_curr = -lp_curr + 0.5 * (p ** 2).sum()

    # Leapfrog
    theta_prop = theta_curr.detach().clone().requires_grad_(True)
    p_prop = p.clone()

    grad = torch.autograd.grad(log_prob_fn(theta_prop), theta_prop)[0]
    p_prop = p_prop + 0.5 * step_size * grad

    for _ in range(n_leapfrog - 1):
        theta_new = theta_prop.detach() + step_size * p_prop
        theta_prop = theta_new.requires_grad_(True)
        grad = torch.autograd.grad(log_prob_fn(theta_prop), theta_prop)[0]
        p_prop = p_prop + step_size * grad

    theta_new = theta_prop.detach() + step_size * p_prop
    theta_prop = theta_new.requires_grad_(True)
    grad = torch.autograd.grad(log_prob_fn(theta_prop), theta_prop)[0]
    p_prop = p_prop + 0.5 * step_size * grad

    # Proposed Hamiltonian
    lp_prop = log_prob_fn(theta_prop)
    H_prop = -lp_prop + 0.5 * (p_prop ** 2).sum()

    # Metropolis
    log_alpha = -(H_prop - H_curr)
    if torch.isnan(log_alpha):
        return theta.detach(), False
    accept = torch.log(torch.rand(1, dtype=theta.dtype)) < log_alpha
    if accept:
        return theta_prop.detach(), True
    else:
        return theta.detach(), False


def run_hmc(
    log_prob_fn,
    init: torch.Tensor,
    n_samples: int = 5000,
    warmup: int = 1000,
    step_size: float = 0.01,
    n_leapfrog: int = 20,
    target_accept: float = 0.65,
    seed: int = 0,
) -> np.ndarray:
    """
    Run HMC with dual-averaging step-size adaptation during warmup.
    Returns (n_samples, d) numpy array.
    """
    torch.manual_seed(seed)
    theta = init.clone().detach()
    d = theta.shape[0]

    # Dual averaging state (Hoffman & Gelman 2014)
    log_eps = math.log(step_size)
    log_eps_bar = 0.0
    H_bar = 0.0
    gamma = 0.05
    t0 = 10.0
    kappa = 0.75
    mu = math.log(10 * step_size)

    samples = []
    n_accept = 0
    total = warmup + n_samples

    for i in range(1, total + 1):
        eps = math.exp(log_eps) if i <= warmup else math.exp(log_eps_bar)
        theta_new, accepted = _hmc_step(theta, log_prob_fn, eps, n_leapfrog)
        theta = theta_new
        n_accept += int(accepted)

        # Dual averaging during warmup
        if i <= warmup:
            alpha = 1.0 if accepted else 0.0
            w = 1.0 / (i + t0)
            H_bar = (1 - w) * H_bar + w * (target_accept - alpha)
            log_eps = mu - math.sqrt(i) / gamma * H_bar
            m = i ** (-kappa)
            log_eps_bar = m * log_eps + (1 - m) * log_eps_bar
        else:
            samples.append(theta.cpu().numpy().copy())

    accept_rate = n_accept / total
    print(f"    HMC: {n_samples} samples, accept rate {accept_rate:.3f}, "
          f"final step_size {math.exp(log_eps_bar):.5f}")

    return np.array(samples, dtype=np.float64)


# ---------------------------------------------------------------
# Main problem class
# ---------------------------------------------------------------

class GermanCreditBLR:
    """
    Bayesian logistic regression on German credit data.

    Provides prior-specific MCMC posteriors and predictive evaluation.
    """

    PRIOR_NAMES = ["gaussian", "gaussian_wide", "cauchy", "laplace", "student_t"]

    def __init__(
        self,
        data_dir: str = "data",
        seed: int = 0,
        split_seed: int = 42,
        train_frac: float = 0.7,
        val_frac: float = 0.15,
        max_train: int = 0,
        hmc_samples: int = 10000,
        hmc_warmup: int = 2000,
        hmc_step_size: float = 0.005,
        hmc_leapfrog: int = 25,
    ):
        self.data_dir = data_dir
        self.seed = seed
        self.hmc_samples = hmc_samples
        self.hmc_warmup = hmc_warmup
        self.hmc_step_size = hmc_step_size
        self.hmc_leapfrog = hmc_leapfrog
        self.max_train = max_train

        X, y = _load_german(data_dir)
        n = X.shape[0]

        # Deterministic split
        rng = np.random.default_rng(split_seed)
        perm = rng.permutation(n)
        n_train = int(train_frac * n)
        n_val = int(val_frac * n)

        self.X_train = X[perm[:n_train]]
        self.y_train = y[perm[:n_train]]
        self.X_val = X[perm[n_train:n_train + n_val]]
        self.y_val = y[perm[n_train:n_train + n_val]]
        self.X_test = X[perm[n_train + n_val:]]
        self.y_test = y[perm[n_train + n_val:]]

        # Subsample training data to create low-data regime
        # where prior choice genuinely matters
        if max_train > 0 and max_train < len(self.y_train):
            sub_rng = np.random.default_rng(split_seed + 1000)
            sub_idx = sub_rng.choice(len(self.y_train), size=max_train, replace=False)
            self.X_train = self.X_train[sub_idx]
            self.y_train = self.y_train[sub_idx]
            print(f"  Subsampled training data: {max_train} / {n_train} observations")

        self.dim = X.shape[1]  # 25 (24 features + intercept)

        # Build prior objects
        self.priors = {name: PRIOR_REGISTRY[name](self.dim) for name in self.PRIOR_NAMES}

        # Posterior samples cache (in memory)
        self._posteriors: dict = {}  # prior_name -> (n_samples, dim) numpy

        # Cache directory on disk
        self._cache_dir = os.path.join(data_dir, "german_credit_posteriors")

    @property
    def prior_names(self):
        return list(self.PRIOR_NAMES)

    def _cache_key(self, prior_name: str) -> str:
        """Deterministic filename based on config."""
        h = hashlib.md5(
            f"{prior_name}_s{self.seed}_n{self.hmc_samples}_w{self.hmc_warmup}"
            f"_eps{self.hmc_step_size}_L{self.hmc_leapfrog}_mt{self.max_train}".encode()
        ).hexdigest()[:12]
        return f"posterior_{prior_name}_{h}.npy"

    def _run_hmc_for_prior(self, prior_name: str) -> np.ndarray:
        """Run HMC for a single prior, return (n_samples, dim) array."""
        prior = self.priors[prior_name]
        X_t = torch.tensor(self.X_train, dtype=torch.float64)
        y_t = torch.tensor(self.y_train, dtype=torch.float64)

        def log_prob(theta):
            ll = _log_likelihood(theta, X_t, y_t)
            lp = prior.log_prob(theta.unsqueeze(0)).squeeze()
            return ll + lp

        # Initialise at MLE (or zero)
        init = torch.zeros(self.dim, dtype=torch.float64)

        print(f"  Running HMC for prior={prior_name} ...")
        samples = run_hmc(
            log_prob,
            init,
            n_samples=self.hmc_samples,
            warmup=self.hmc_warmup,
            step_size=self.hmc_step_size,
            n_leapfrog=self.hmc_leapfrog,
            seed=self.seed + hash(prior_name) % 10000,
        )
        return samples

    def ensure_posteriors(self, prior_names=None):
        """Run HMC for each prior (skips if cached on disk)."""
        if prior_names is None:
            prior_names = self.PRIOR_NAMES

        os.makedirs(self._cache_dir, exist_ok=True)

        for pname in prior_names:
            if pname in self._posteriors:
                continue

            cache_path = os.path.join(self._cache_dir, self._cache_key(pname))
            if os.path.isfile(cache_path):
                print(f"  Loading cached posterior: {pname}")
                self._posteriors[pname] = np.load(cache_path)
                continue

            samples = self._run_hmc_for_prior(pname)
            np.save(cache_path, samples)
            self._posteriors[pname] = samples
            print(f"  Cached posterior to {cache_path}")

    def sample_posterior(self, prior_name: str, n: int, seed: int = None) -> torch.Tensor:
        """Draw n posterior samples for the given prior. Returns (n, dim) float32 tensor."""
        if prior_name not in self._posteriors:
            self.ensure_posteriors([prior_name])

        arr = self._posteriors[prior_name]
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, arr.shape[0], size=n)
        return torch.tensor(arr[idx], dtype=torch.float32)

    def sample_prior(self, prior_name: str, n: int) -> torch.Tensor:
        """Draw n samples from the named prior."""
        return self.priors[prior_name].sample(n).float()

    def make_prior(self, prior_name: str):
        """Return the prior object (has .sample(n) and .log_prob(theta))."""
        return self.priors[prior_name]

    def predictive_log_lik(
        self, theta_samples: np.ndarray, split: str = "test"
    ) -> float:
        """
        Monte Carlo predictive log-likelihood on the specified split.

        pred_ll = (1/N) Σ_i log [ (1/S) Σ_s p(y_i | x_i, θ_s) ]

        theta_samples: (S, dim) array of posterior samples
        Returns: scalar (higher = better)
        """
        if split == "train":
            X, y = self.X_train, self.y_train
        elif split == "val":
            X, y = self.X_val, self.y_val
        elif split == "test":
            X, y = self.X_test, self.y_test
        else:
            raise ValueError(f"Unknown split: {split}")

        theta = np.asarray(theta_samples, dtype=np.float64)
        # logits: (S, N)
        logits = theta @ X.T

        # log p(y_i | x_i, θ_s) for each sample and data point
        # = y_i * logit - log(1 + exp(logit))
        log_p = y[None, :] * logits - np.logaddexp(0, logits)  # (S, N)

        # log-mean-exp over S samples for each data point
        S = theta.shape[0]
        # log (1/S Σ_s exp(log_p_s)) = logsumexp(log_p, axis=0) - log(S)
        max_lp = log_p.max(axis=0, keepdims=True)
        lme = max_lp.squeeze(0) + np.log(np.exp(log_p - max_lp).mean(axis=0))

        return float(lme.mean())

    def accuracy(self, theta_samples: np.ndarray, split: str = "test") -> float:
        """Posterior predictive accuracy."""
        if split == "test":
            X, y = self.X_test, self.y_test
        elif split == "val":
            X, y = self.X_val, self.y_val
        else:
            X, y = self.X_train, self.y_train

        theta = np.asarray(theta_samples, dtype=np.float64)
        logits = theta @ X.T  # (S, N)
        mean_prob = 1.0 / (1.0 + np.exp(-logits))  # (S, N)
        avg_prob = mean_prob.mean(axis=0)  # (N,)
        preds = (avg_prob >= 0.5).astype(float)
        return float((preds == y).mean())


# ---------------------------------------------------------------
# Thin wrappers compatible with train_flow.py interface
# ---------------------------------------------------------------

class GermanCreditPosteriorDataset:
    """
    Wraps GermanCreditBLR for a specific prior, presenting posterior
    samples as the 'target distribution' for flow training.

    Usage in train_flow.py:
        dataset = GermanCreditPosteriorDataset(problem, "gaussian")
        x1 = dataset.sample(batch_size)  # posterior samples
    """
    def __init__(self, problem: GermanCreditBLR, prior_name: str, seed: int = 0):
        self.problem = problem
        self.prior_name = prior_name
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        problem.ensure_posteriors([prior_name])

    def sample(self, n: int) -> torch.Tensor:
        seed = int(self.rng.integers(0, 2**31))
        return self.problem.sample_posterior(self.prior_name, n, seed=seed)


class GermanCreditMultiPosteriorDataset:
    """
    Provides prior-dependent posterior samples for conditional flow training.

    The training loop should call:
        x1 = dataset.sample(batch_size, prior_name=prior_name)
    """
    def __init__(self, problem: GermanCreditBLR, seed: int = 0):
        self.problem = problem
        self.rng = np.random.default_rng(seed)
        problem.ensure_posteriors()

    def sample(self, n: int, prior_name: str = None) -> torch.Tensor:
        if prior_name is None:
            prior_name = self.rng.choice(self.problem.prior_names)
        seed = int(self.rng.integers(0, 2**31))
        return self.problem.sample_posterior(prior_name, n, seed=seed)