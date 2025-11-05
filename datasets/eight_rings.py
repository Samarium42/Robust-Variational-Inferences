import math
import numpy as np
import torch

class EightGaussianRingDataset:
    def __init__(self, radius: float = 5.0, std: float = 0.2, seed: int | None = None, device=None, dtype=None):
        self.radius = radius
        self.std = std
        self.rng = np.random.default_rng(seed)
        self.device = device
        self.dtype = dtype if dtype is not None else torch.float32

        k = 8
        angles = np.array([2 * math.pi * j / k for j in range(k)])
        self.centers = np.stack([radius * np.cos(angles), radius * np.sin(angles)], axis=1)

    def sample(self, n: int) -> torch.Tensor:
        idx = self.rng.integers(0, len(self.centers), size=n)
        x = self.centers[idx] + self.rng.normal(0.0, self.std, size=(n, 2))
        return torch.tensor(x, device=self.device, dtype=self.dtype)
