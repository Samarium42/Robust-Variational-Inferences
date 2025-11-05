import numpy as np
import torch
import math


class StudentTPrior():
    def __init__(self, df: float = 5.0, scale: float = 1.0):
        self.df = df
        self.scale = scale

    def sample(self, n: int) -> torch.Tensor:
        # Sample Student-t by normal / sqrt(gamma/df)
        g = torch.distributions.Gamma(self.df/2., 1./2.).sample((n, 1))
        z = torch.randn(n, 2) / torch.sqrt(g / self.df)
        return self.scale * z