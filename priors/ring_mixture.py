import numpy as np
import torch
import math


class RingMixturePrior():
    def __init__(self, k: int = 8, R: float = 3.0, sigma: float = 0.7):
        self.k = k
        self.R = R
        self.sigma = sigma
        self.centers = []
        for j in range(k):
            phi = 2 * math.pi * j / k
            self.centers.append([R * math.cos(phi), R * math.sin(phi)])
        self.centers = torch.tensor(self.centers, dtype=torch.float32)

    def sample(self, n: int) -> torch.Tensor:
        idx = torch.randint(0, self.k, (n,))
        eps = self.sigma * torch.randn(n, 2)
        return self.centers[idx] + eps