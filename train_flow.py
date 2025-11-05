# train_flow.py
import os, argparse, random, math
import numpy as np
import torch
from torch import nn, Tensor

# ---- Meta Flow Matching (raw API) ----
from flow_matching.path import AffineProbPath
from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.utils import ModelWrapper
from flow_matching.solver import ODESolver


from datasets.eight_rings import EightGaussianRingDataset
from datasets.spirals import TwoArmSpiralsDataset
from datasets.moons import TwoMoonsDataset


from priors.gaussian import GaussianPrior
from priors.student_t import StudentTPrior
from priors.ring_mixture import RingMixturePrior  
Device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


# -------------------------
# Model
# -------------------------
class Swish(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return x * torch.sigmoid(x)

class VelocityMLP(nn.Module):
    """(x,t) -> velocity(x,t) in R^2. Time fed as a scalar feature."""
    def __init__(self, input_dim: int = 2, time_dim: int = 1, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim + time_dim, hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
       if t.ndim == 0:                      
          t = t.repeat(x.shape[0]).unsqueeze(1)   
       else:
          t = t.reshape(-1)                        
          if t.numel() == 1:                       
            t = t.repeat(x.shape[0])             
          t = t.unsqueeze(1)                       

       return self.net(torch.cat([x, t], dim=1))

class TimeEmbedding(nn.Module):
    def __init__(self, n_f=8):  # 8 frequencies -> 16 sin/cos + t itself
        super().__init__()
        freqs = 2.0 ** torch.arange(n_f)  # [1,2,4,...]
        self.register_buffer("freqs", freqs, persistent=False)
        self.proj = nn.Sequential(
            nn.Linear(1 + 2*n_f, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.SiLU(),
        )
    def forward(self, t): 
        angles = t * self.freqs[None, :] * 2*math.pi
        emb = [t, torch.sin(angles), torch.cos(angles)]
        h = torch.cat([x for x in emb], dim=1)
        return self.proj(h)

class FiLMBlock(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1  = nn.Linear(dim, hidden)
        self.fc2  = nn.Linear(hidden, dim)
        self.act  = nn.SiLU()
        self.gamma = nn.Linear(64, hidden)  # from time embedding
        self.beta  = nn.Linear(64, hidden)
    def forward(self, x, t_emb):
        y = self.norm(x)
        y = self.fc1(y)
        g = self.gamma(t_emb)
        b = self.beta(t_emb)
        y = self.act(y * (1 + g) + b)
        y = self.fc2(y)
        return x + y

class VelocityResNet(nn.Module):
    """(x,t) -> v(x,t) in R^2 with FiLM time conditioning."""
    def __init__(self, input_dim=2, hidden=256, depth=4):
        super().__init__()
        self.tok = nn.Linear(input_dim, hidden)
        self.temb = TimeEmbedding(n_f=8)
        self.blocks = nn.ModuleList([FiLMBlock(hidden, 4*hidden) for _ in range(depth)])
        self.head = nn.Linear(hidden, input_dim)
    def forward(self, x, t):
        # ensure t is [B,1]
        if t.ndim == 0: t = t.repeat(x.shape[0]).unsqueeze(1)
        elif t.ndim == 1: t = t.unsqueeze(1)
        h = self.tok(x)
        te = self.temb(t)
        for blk in self.blocks:
            h = blk(h, te)
        return self.head(h)


# -------------------------
# Factories
# -------------------------
def make_dataset(name: str):
    if name == "eight_ring":
        return EightGaussianRingDataset(radius=5.0, std=0.2, seed=123)
    if name == "spirals":
        return TwoArmSpiralsDataset(R_max=5.0, alpha=1.5, noise_std=0.1, seed=123)
    if name == "moons":
        return TwoMoonsDataset(noise=0.08, seed=42)
    raise ValueError(f"unknown dataset '{name}'")

def make_prior(name: str):
    if name == "gaussian":
        return GaussianPrior(mean=(0, 0), sigma=1.0)
    if name == "gaussian_narrow":
        return GaussianPrior(sigma=0.5)
    if name == "gaussian_wide":
        return GaussianPrior(sigma=1.5)
    if name == "student_t":
        return StudentTPrior(df=5.0, scale=1.0)
    if name == "ringmix":
        return RingMixturePrior(k=8, R=3.0, sigma=0.7)
    raise ValueError(f"unknown prior '{name}'")


# -------------------------
# Training loop (AffineProbPath + CondOTScheduler)
# -------------------------
def train(
    dataset,
    prior,
    steps: int = 6000,
    batch_size: int = 1024,
    lr: float = 2e-3,
    hidden: int = 128,
    print_every: int = 500,
    seed: int = 1337,
):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

    model   = VelocityResNet(hidden=hidden, depth = 4).to(Device)
    opt     = torch.optim.Adam(model.parameters(), lr=lr)
    path = AffineProbPath(scheduler=CondOTScheduler())

    for step in range(1, steps + 1):
        x0 = prior.sample(batch_size).to(Device)
        x1 = dataset.sample(batch_size).to(Device)
        u = torch.distributions.Beta(2.0, 2.0).sample((batch_size,)).to(Device)
        eps = 0.02
        t = u * (1 - 2*eps) + eps
        ps = path.sample(t = t, x_0=x0, x_1=x1)   
        
        loss = torch.pow(model(ps.x_t, ps.t) - ps.dx_t, 2).mean()
        loss.backward()
        opt.step()
        opt.zero_grad()

        if step % print_every == 0 or step == 1:
            print(f"Step {step:6d} / {steps}    Loss: {loss.item():.6f}")

    return model

class NewWrapper(ModelWrapper):
    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        return self.model(x, t)

# -------------------------
# CLI
# -------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="eight_ring", choices=["eight_ring","spirals","moons"])
    ap.add_argument("--prior",   default="gaussian",
                    choices=["gaussian","gaussian_narrow","gaussian_wide","student_t","ringmix"])
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--outdir", default="out_fm_solver")
    ap.add_argument("--print_every", type=int, default=100)
    ap.add_argument("--sample_only", action="store_true")
    ap.add_argument("--n_samples", type=int, default=20000)
    ap.add_argument("--step_size", type=float, default=1e-2, help="ODE step size for sampling")
    return ap.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    print(f"Config => dataset={args.dataset}  prior={args.prior}  hidden={args.hidden}  "
          f"lr={args.lr}  seed={args.seed}  steps={args.steps}  batch={args.batch}")

    dataset = make_dataset(args.dataset)
    prior   = make_prior(args.prior)

    
    tag = f"{args.dataset}_{args.prior}_h{args.hidden}_lr{args.lr}"
    model_path = os.path.join(args.outdir, f"fm_{tag}.pt")
    samp_path  = os.path.join(args.outdir, f"samples_{tag}.npy")

    if not args.sample_only:
        print(f"Training FM solver…")
        model = train(
            dataset, prior,
            steps=args.steps, batch_size=args.batch,
            lr=args.lr, hidden=args.hidden,
            print_every=args.print_every, seed=args.seed,
        )
        torch.save(model.state_dict(), model_path)
    else:
        model = VelocityResNet(hidden=args.hidden, depth=4).to(Device)
        model.load_state_dict(torch.load(model_path, map_location=Device))
        model.eval()

    # ---- Sampling ----
    solver = ODESolver(NewWrapper(model))
    x0 = prior.sample(args.n_samples).to(Device)
    samples = solver.sample(x0, step_size= args.step_size).cpu().numpy()
    np.save(samp_path, samples)
    print("Saved:", model_path)
    print("Saved:", samp_path)


if __name__ == "__main__":
    main()
