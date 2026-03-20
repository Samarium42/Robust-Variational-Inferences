import os
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
import torch


RADON_URLS = [
    "https://raw.githubusercontent.com/pymc-devs/pymc-examples/main/examples/data/radon.csv",
    "https://raw.githubusercontent.com/pymc-devs/pymc-examples/master/examples/data/radon.csv",
]


def _ensure_download(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.isfile(path):
        return

    last_err = None
    for url in RADON_URLS:
        try:
            req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urlopen(req) as r, open(path, "wb") as f:
                f.write(r.read())
            return
        except Exception as e:
            last_err = e

    raise RuntimeError(f"Failed to download radon.csv to {path}. Last error: {last_err}")


def _get_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"Could not find any of columns {candidates} in {list(df.columns)}")


class MinnesotaRadon2DDataset:
    """
    Produces 2D points so the rest of the repo can stay unchanged.

    Uses (log_radon, log(U)) where U is uranium ppm, if available.
    Different radon.csv variants expose uranium as one of:
      - log_uranium
      - Uppm
      - uranium

    Filters to state == MN if that column exists.

    Splits: train, val, test.
    Standardisation: mean and std fit on train split only, then applied to all splits.
    """

    def __init__(
        self,
        split: str = "train",
        seed: int = 0,
        split_seed: int = 0,
        data_dir: str = "data",
        standardize: bool = True,
        train_frac: float = 0.8,
        val_frac: float = 0.1,
    ):
        assert split in {"train", "val", "test"}
        self.split = split
        self.rng = np.random.default_rng(seed)

        csv_path = os.path.join(data_dir, "radon.csv")
        _ensure_download(csv_path)

        df = pd.read_csv(csv_path)

        if "state" in df.columns:
            df = df[df["state"].astype(str).str.upper() == "MN"].copy()

        # y: log radon
        y_col = _get_col(df, ["log_radon", "log.radon"])
        y = pd.to_numeric(df[y_col], errors="coerce").to_numpy()

        # u: uranium (log scale), robust across csv variants
        if "log_uranium" in df.columns:
            u = pd.to_numeric(df["log_uranium"], errors="coerce").to_numpy()
        elif "Uppm" in df.columns:
            u_raw = pd.to_numeric(df["Uppm"], errors="coerce").to_numpy()
            u = np.log(np.clip(u_raw, 1e-12, None))
        elif "uranium" in df.columns:
            u_raw = pd.to_numeric(df["uranium"], errors="coerce").to_numpy()
            u = np.log(np.clip(u_raw, 1e-12, None))
        else:
            # Last resort fallback if uranium isn't available in some variant
            # Keeps pipeline running but is less meaningful scientifically.
            if "floor" in df.columns:
                u = pd.to_numeric(df["floor"], errors="coerce").to_numpy()
            else:
                raise ValueError(f"No uranium column found. Available columns: {list(df.columns)}")

        X = np.stack([y, u], axis=1)
        X = X[np.isfinite(X).all(axis=1)]

        n = X.shape[0]
        perm = np.random.default_rng(split_seed).permutation(n)

        n_train = int(train_frac * n)
        n_val = int(val_frac * n)

        idx_train = perm[:n_train]
        idx_val = perm[n_train : n_train + n_val]
        idx_test = perm[n_train + n_val :]

        X_train = X[idx_train]

        if standardize:
            mu = X_train.mean(axis=0, keepdims=True)
            sd = X_train.std(axis=0, keepdims=True)
            sd = np.maximum(sd, 1e-6)
            X = (X - mu) / sd

        if split == "train":
            self.X = X[idx_train]
        elif split == "val":
            self.X = X[idx_val]
        else:
            self.X = X[idx_test]

    def sample(self, n: int):
        idx = self.rng.integers(0, self.X.shape[0], size=n)
        return torch.tensor(self.X[idx], dtype=torch.float32)
