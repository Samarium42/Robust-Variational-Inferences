import torch
from sklearn.datasets import make_moons

class TwoMoonsDataset:
    def __init__(self, noise: float = 0.08, seed: int | None = 42, device=None, dtype=None):
        self.noise = noise
        self.seed = seed
        self.device = device
        self.dtype = dtype if dtype is not None else torch.float32

    def sample(self, n: int) -> torch.Tensor:
        X, _ = make_moons(n_samples=n, noise=self.noise, random_state=self.seed)
        return torch.tensor(X, device=self.device, dtype=self.dtype)
