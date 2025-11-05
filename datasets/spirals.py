import numpy as np
import torch

class TwoArmSpiralsDataset:
    def __init__(self, R_max: float = 5.0, alpha: float = 1.5, noise_std: float = 0.1,
                 seed: int | None = None, device=None, dtype=None):
        self.R_max = R_max
        self.alpha = alpha
        self.noise_std = noise_std
        self.rng = np.random.default_rng(seed)
        self.device = device
        self.dtype = dtype if dtype is not None else torch.float32

    def sample(self, n: int) -> torch.Tensor:
        n1 = n // 2
        r = self.rng.uniform(0, self.R_max, size=n1)
        theta = self.alpha * r
        arm1 = np.stack([r * np.cos(theta),  r * np.sin(theta)], axis=1)
        arm2 = np.stack([-r * np.cos(theta), -r * np.sin(theta)], axis=1)
        X = np.vstack([arm1, arm2])
        X += self.rng.normal(0, self.noise_std, size=X.shape)
        if X.shape[0] < n:  # pad if odd
            X = np.vstack([X, X[: n - X.shape[0]]])
        return torch.tensor(X, device=self.device, dtype=self.dtype)
