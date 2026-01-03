# train_flow.py
import os, argparse, random, math
import numpy as np
import torch
from torch import nn, Tensor

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


class TimeEmbedding(nn.Module):
    def __init__(self, n_f=12):
        super().__init__()
        freqs = 2.0 ** torch.arange(n_f)
        self.register_buffer("freqs", freqs, persistent=False)
        self.proj = nn.Sequential(
            nn.Linear(1 + 2 * n_f, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.SiLU(),
        )

    def forward(self, t: Tensor) -> Tensor:
        angles = t * self.freqs[None, :] * 2 * math.pi
        emb = [t, torch.sin(angles), torch.cos(angles)]
        h = torch.cat([x for x in emb], dim=1)
        return self.proj(h)


class FiLMBlock(nn.Module):
    def __init__(self, dim: int, hidden: int, cond_dim: int, dropout: float = 0.05):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.act = nn.SiLU()
        self.drop = nn.Dropout(dropout)

        self.gamma = nn.Linear(cond_dim, hidden)
        self.beta = nn.Linear(cond_dim, hidden)

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        y = self.norm(x)
        y = self.fc1(y)
        g = self.gamma(cond)
        b = self.beta(cond)
        y = self.act(y * (1 + g) + b)
        y = self.drop(y)
        y = self.fc2(y)
        return x + y


class VelocityResNet(nn.Module):
    def __init__(
        self,
        input_dim: int = 2,
        hidden: int = 256,
        depth: int = 6,
        num_priors: int = 5,
        prior_emb_dim: int = 16,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden = hidden
        self.depth = depth

        self.temb = TimeEmbedding(n_f=12)
        self.pemb = nn.Embedding(num_priors, prior_emb_dim)

        self.cond_dim = 64 + prior_emb_dim

        self.tok = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.blocks = nn.ModuleList(
            [FiLMBlock(hidden, 4 * hidden, cond_dim=self.cond_dim, dropout=dropout) for _ in range(depth)]
        )
        self.head = nn.Linear(hidden, input_dim)

    def forward(self, x: Tensor, t: Tensor, prior_id: Tensor) -> Tensor:
        if t.ndim == 0:
            t = t.repeat(x.shape[0]).unsqueeze(1)
        elif t.ndim == 1:
            t = t.unsqueeze(1)

        if prior_id.ndim == 0:
            prior_id = prior_id.repeat(x.shape[0])
        prior_id = prior_id.long()

        te = self.temb(t)
        pe = self.pemb(prior_id)
        cond = torch.cat([te, pe], dim=1)

        h = self.tok(x)
        for blk in self.blocks:
            h = blk(h, cond)
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


def parse_priors_csv(s: str):
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if not parts:
        raise ValueError("empty priors list")
    return parts


# -------------------------
# Training loop (AffineProbPath + CondOTScheduler)
# -------------------------
def train_conditional(
    dataset,
    priors,
    steps: int = 6000,
    batch_size: int = 1024,
    lr: float = 1e-3,
    hidden: int = 256,
    depth: int = 6,
    prior_emb_dim: int = 16,
    dropout: float = 0.05,
    weight_decay: float = 1e-2,
    print_every: int = 500,
    seed: int = 1337,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    prior_names = list(priors.keys())
    K = len(prior_names)

    model = VelocityResNet(hidden=hidden, depth=depth, num_priors=K, prior_emb_dim=prior_emb_dim, dropout=dropout).to(Device)

    path = AffineProbPath(scheduler=CondOTScheduler())
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    eps = 1e-4

    for step in range(1, steps + 1):
        x1 = dataset.sample(batch_size).to(Device)

        k_idx = random.randrange(K)
        prior_name = prior_names[k_idx]
        prior = priors[prior_name]
        x0 = prior.sample(batch_size).to(Device)

        u = torch.rand(batch_size, device=Device)
        t = u * (1 - 2 * eps) + eps   # shape [batch_size]

        ps = path.sample(t=t, x_0=x0, x_1=x1)


        prior_id = torch.full((batch_size,), k_idx, device=Device, dtype=torch.long)
        pred = model(ps.x_t, ps.t, prior_id)
        loss = torch.pow(pred - ps.dx_t, 2).mean()

        loss.backward()
        opt.step()
        opt.zero_grad()

        if step % print_every == 0 or step == 1:
            print(f"Step {step:6d} / {steps}    Loss: {loss.item():.6f}")

    return model, prior_names


class ConditionalWrapper(ModelWrapper):
    def __init__(self, model: nn.Module, prior_id: int):
        super().__init__(model)
        self._prior_id = int(prior_id)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        prior_id = torch.full((x.shape[0],), self._prior_id, device=x.device, dtype=torch.long)
        return self.model(x, t, prior_id)


# -------------------------
# CLI
# -------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="eight_ring", choices=["eight_ring", "spirals", "moons"])

    ap.add_argument("--priors", default="gaussian,gaussian_narrow,gaussian_wide,student_t,ringmix",
                    help="Comma-separated list of priors to train jointly.")
    ap.add_argument("--prior", default="gaussian",
                    choices=["gaussian", "gaussian_narrow", "gaussian_wide", "student_t", "ringmix"],
                    help="Used for sampling when --sample_only is set.")

    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--prior_emb_dim", type=int, default=16)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--weight_decay", type=float, default=1e-2)

    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--outdir", default="out_fm_solver")
    ap.add_argument("--print_every", type=int, default=100)

    ap.add_argument("--sample_only", action="store_true")
    ap.add_argument("--n_samples", type=int, default=20000)
    ap.add_argument("--step_size", type=float, default=0.02, help="ODE step size for sampling")
    ap.add_argument("--model_path", default="", help="Optional path to a trained conditional model checkpoint.")
    return ap.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    priors_list = parse_priors_csv(args.priors)
    priors = {p: make_prior(p) for p in priors_list}

    tag = f"{args.dataset}_cond_h{args.hidden}_d{args.depth}_lr{args.lr}"
    default_model_path = os.path.join(args.outdir, f"fm_{tag}.pt")
    model_path = args.model_path if args.model_path else default_model_path

    samp_tag = f"{args.dataset}_{args.prior}_cond_h{args.hidden}_d{args.depth}_lr{args.lr}"
    samp_path = os.path.join(args.outdir, f"samples_{samp_tag}.npy")

    if not args.sample_only:
        print(
            f"Config => dataset={args.dataset}  priors={','.join(priors_list)}  hidden={args.hidden}  depth={args.depth}  "
            f"lr={args.lr}  seed={args.seed}  steps={args.steps}  batch={args.batch}"
        )
        dataset = make_dataset(args.dataset)
        model, prior_names = train_conditional(
            dataset,
            priors,
            steps=args.steps,
            batch_size=args.batch,
            lr=args.lr,
            hidden=args.hidden,
            depth=args.depth,
            prior_emb_dim=args.prior_emb_dim,
            dropout=args.dropout,
            weight_decay=args.weight_decay,
            print_every=args.print_every,
            seed=args.seed,
        )

        ckpt = {
            "state_dict": model.state_dict(),
            "prior_names": prior_names,
            "hidden": args.hidden,
            "depth": args.depth,
            "prior_emb_dim": args.prior_emb_dim,
            "dropout": args.dropout,
        }
        torch.save(ckpt, model_path)
        print("Saved model:", model_path)

    # ---- Sampling ----
    ckpt = torch.load(model_path, map_location=Device)
    prior_names = ckpt["prior_names"]
    if args.prior not in prior_names:
        raise ValueError(f"prior '{args.prior}' not found in checkpoint prior_names={prior_names}")

    prior_id = prior_names.index(args.prior)

    model = VelocityResNet(
        hidden=ckpt["hidden"],
        depth=ckpt["depth"],
        num_priors=len(prior_names),
        prior_emb_dim=ckpt["prior_emb_dim"],
        dropout=ckpt["dropout"],
    ).to(Device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    solver = ODESolver(ConditionalWrapper(model, prior_id))
    x0 = make_prior(args.prior).sample(args.n_samples).to(Device)
    samples = solver.sample(x0, step_size=args.step_size).cpu().numpy()

    np.save(samp_path, samples)
    print("Saved samples:", samp_path)
    print("Model used:", model_path)


if __name__ == "__main__":
    main()
