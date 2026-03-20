import json
import math
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
from sklearn.neighbors import KernelDensity


EPS = 1e-12


def now_run_id() -> str:
    # no hyphens, only underscores
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def logsumexp(a: np.ndarray, axis: int = -1) -> np.ndarray:
    a_max = np.max(a, axis=axis, keepdims=True)
    out = a_max + np.log(np.sum(np.exp(a - a_max), axis=axis, keepdims=True) + EPS)
    return np.squeeze(out, axis=axis)


def project_to_simplex(v: np.ndarray) -> np.ndarray:
    # Duchi et al projection onto simplex
    v = np.asarray(v, dtype=float)
    if v.ndim != 1:
        raise ValueError("project_to_simplex expects 1D array")
    n = v.size
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, n + 1) > (cssv - 1.0))[0]
    if rho.size == 0:
        # fallback to uniform
        return np.ones_like(v) / n
    rho = rho[-1]
    theta = (cssv[rho] - 1.0) / (rho + 1)
    w = np.maximum(v - theta, 0.0)
    s = w.sum()
    if s <= 0:
        return np.ones_like(v) / n
    return w / s


@dataclass
class ManifestEntry:
    dataset_name: str
    prior_name: str
    model_path: str
    sample_path: str


def read_manifest(path: str) -> List[ManifestEntry]:
    out: List[ManifestEntry] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != 4:
                raise ValueError(f"Bad manifest line: {line}")
            out.append(ManifestEntry(*parts))
    if not out:
        raise ValueError(f"Empty manifest: {path}")
    return out


def load_samples(sample_path: str) -> np.ndarray:
    arr = np.load(sample_path)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D samples array in {sample_path}, got shape {arr.shape}")
    if not np.isfinite(arr).all():
        arr = arr[np.isfinite(arr).all(axis=1)]
    return arr.astype(np.float64)


def fit_kde(samples: np.ndarray, bandwidth: float) -> KernelDensity:
    kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth)
    kde.fit(samples)
    return kde


def draw_dataset_points(make_dataset_fn, dataset_name: str, n: int, seed: int) -> np.ndarray:
    ds = make_dataset_fn(dataset_name)
    x = ds.sample(n)
    if hasattr(x, "cpu"):
        x = x.cpu().numpy()
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"Dataset sample returned shape {x.shape}, expected (n, d)")
    x = x[np.isfinite(x).all(axis=1)]
    if x.shape[0] == 0:
        raise ValueError("No finite dataset samples")

    # bootstrap resample so different seeds give different point sets
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.shape[0], size=n)
    x = x[idx]

    rng.shuffle(x, axis=0)
    return x


def precompute_logpk(kdes: List[KernelDensity], x: np.ndarray) -> np.ndarray:
    # returns shape (n, K)
    logpk = np.stack([k.score_samples(x) for k in kdes], axis=1)
    return logpk.astype(np.float64)


def mixture_nll_from_logpk(logpk: np.ndarray, w: np.ndarray) -> float:
    w = np.asarray(w, dtype=np.float64)
    w = np.clip(w, EPS, 1.0)
    w = w / w.sum()
    logw = np.log(w)
    ll = logsumexp(logpk + logw[None, :], axis=1)
    return float(-np.mean(ll))


def responsibilities(logpk: np.ndarray, w: np.ndarray) -> np.ndarray:
    w = np.asarray(w, dtype=np.float64)
    w = np.clip(w, EPS, 1.0)
    w = w / w.sum()
    logw = np.log(w)
    logr = logpk + logw[None, :]
    logz = logsumexp(logr, axis=1)
    r = np.exp(logr - logz[:, None])
    r = np.clip(r, EPS, 1.0)
    r = r / r.sum(axis=1, keepdims=True)
    return r


def write_weights_json(out_path: str, dataset_key: str, priors: List[str], w: np.ndarray):
    w = np.asarray(w, dtype=float)
    w = np.clip(w, EPS, 1.0)
    w = w / w.sum()
    obj = {dataset_key: {"priors": list(priors), "weights": [float(x) for x in w]}}
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(obj, f, indent=2)


def write_json(path: str, obj: Dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)