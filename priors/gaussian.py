import numpy as np
import torch
import math

class GaussianPrior():
    def __init__(self, mean=(0., 0.), sigma=1.0, anisotropy=(1.0, 1.0), angle_deg: float = 0.0):
        self.m = torch.tensor(mean, dtype=torch.float32)
        S = torch.diag(torch.tensor(anisotropy, dtype=torch.float32)) * (sigma ** 2)
        theta = math.radians(angle_deg)
        R = torch.tensor([[math.cos(theta), -math.sin(theta)],
                          [math.sin(theta),  math.cos(theta)]], dtype=torch.float32)
        self.C = R @ S @ R.T  # covariance
        self.A = torch.linalg.cholesky(self.C + 1e-6 * torch.eye(2))

    def sample(self, n: int) -> torch.Tensor:
        z = torch.randn(n, 2)
        return self.m + z @ self.A.T