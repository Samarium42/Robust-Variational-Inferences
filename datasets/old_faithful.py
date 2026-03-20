import os
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
import torch


FAITHFUL_URLS = [
    "https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/datasets/faithful.csv",
    "https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/datasets/faithful.csv",
]


def _download_csv(urls, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.isfile(path):
        return
    last_err = None
    for url in urls:
        try:
            req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urlopen(req) as r, open(path, "wb") as f:
                f.write(r.read())
            return
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Failed to download faithful.csv to {path}. Last error: {last_err}")


class OldFaithful2DDataset:
    """
    Old Faithful geyser dataset.
    2D points: (eruptions, waiting).

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

        csv_path = os.path.join(data_dir, "faithful.csv")
        _download_csv(FAITHFUL_URLS, csv_path)

        df = pd.read_csv(csv_path)

        # Rdatasets has an index column named 'Unnamed: 0'
        if "Unnamed: 0" in df.columns:
            df = df.drop(columns=["Unnamed: 0"])

        # columns are usually: eruptions, waiting
        if "eruptions" not in df.columns or "waiting" not in df.columns:
            raise ValueError(f"Unexpected columns in faithful.csv: {list(df.columns)}")

        eruptions = pd.to_numeric(df["eruptions"], errors="coerce").to_numpy()
        waiting = pd.to_numeric(df["waiting"], errors="coerce").to_numpy()

        X = np.stack([eruptions, waiting], axis=1)
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