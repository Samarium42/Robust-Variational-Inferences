import os, json, math, argparse, numpy as np, torch
from torch import nn
from collections import defaultdict

from train_flow import make_dataset, Device


def load_manifest(path):
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d, p, ckpt, samp = line.split(",")
            rows.append((d, p, ckpt, samp))
    return rows


class Critic(nn.Module):
    def __init__(self, hidden=256, depth=4):
        super().__init__()
        layers = [nn.Linear(2, hidden), nn.SiLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.SiLU()]
        layers += [nn.Linear(hidden, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def batch_expectation_exp(values):
    m = values.max()
    return torch.exp(values - m).mean() * torch.exp(m)


def train_one_dataset(
    dataset_name,
    prior2samples,
    *,
    steps=3000,
    batch=2048,
    lr=1e-3,
    seed=123,
    critic_steps=5,
    alpha_steps=1,
    ema_rate=0.95,
    weight_decay=1e-2,
    print_every=200,
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    P = make_dataset(dataset_name)

    q_samples = {}
    for prior_name, npy_path in prior2samples.items():
        arr = np.load(npy_path)
        q_samples[prior_name] = torch.tensor(arr, dtype=torch.float32, device=Device)

    priors = sorted(q_samples.keys())
    K = len(priors)

    critic = Critic(hidden=256, depth=4).to(Device)

    alpha = torch.zeros(K, device=Device, requires_grad=True)

    opt_critic = torch.optim.AdamW(critic.parameters(), lr=lr, weight_decay=weight_decay)
    opt_alpha = torch.optim.Adam([alpha], lr=lr)

    ptrs = {k: 0 for k in priors}

    def sample_from_Qk(k, B):
        i = ptrs[k]
        x = q_samples[k][i:i + B]
        if x.shape[0] < B:
            rem = B - x.shape[0]
            x = torch.cat([x, q_samples[k][0:rem]], dim=0)
            ptrs[k] = rem
        else:
            ptrs[k] = i + B
        return x

    ema_mix_term = None

    def compute_terms():
        x_p = P.sample(batch).to(Device)
        T_p = critic(x_p)

        E_expT = []
        for k in priors:
            x_qk = sample_from_Qk(k, batch)
            T_qk = critic(x_qk)
            E_expT.append(batch_expectation_exp(T_qk))
        E_expT = torch.stack(E_expT)

        w = torch.softmax(alpha, dim=0)
        mix_term = torch.sum(w * E_expT)
        return T_p, mix_term, w

    for step in range(1, steps + 1):
        for _ in range(critic_steps):
            T_p, mix_term, w = compute_terms()
            loss = -(T_p.mean() - torch.log(mix_term + 1e-12))
            opt_critic.zero_grad()
            loss.backward()
            opt_critic.step()

        for _ in range(alpha_steps):
            T_p, mix_term, w = compute_terms()
            loss = -(T_p.mean() - torch.log(mix_term + 1e-12))
            opt_alpha.zero_grad()
            loss.backward()
            opt_alpha.step()

        with torch.no_grad():
            _, mix_term_eval, w_eval = compute_terms()
            mix_term_val = mix_term_eval.item()
            if ema_mix_term is None:
                ema_mix_term = mix_term_val
            else:
                ema_mix_term = ema_rate * ema_mix_term + (1 - ema_rate) * mix_term_val

        if step % print_every == 0 or step == 1:
            kl_lb = (T_p.mean() - torch.log(mix_term + 1e-12)).item()
            w_list = w_eval.detach().cpu().tolist()
            print(
                f"[{dataset_name}] step {step:5d}/{steps}  KL_lb≈{kl_lb:.4f}  ema_mix≈{ema_mix_term:.4f}  "
                + "w=" + " ".join(f"{p}:{wi:.2f}" for p, wi in zip(priors, w_list))
            )

    with torch.no_grad():
        w = torch.softmax(alpha, dim=0).cpu().numpy().tolist()
    return priors, w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="out_fm_solver/manifest.txt")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--out", default="out_fm_solver/credal_weights.json")
    ap.add_argument("--critic_steps", type=int, default=5)
    ap.add_argument("--alpha_steps", type=int, default=1)
    ap.add_argument("--print_every", type=int, default=200)
    args = ap.parse_args()

    rows = load_manifest(args.manifest)

    by_dataset = defaultdict(dict)
    for d, p, ckpt, samp in rows:
        by_dataset[d][p] = samp

    out = {}
    for d, prior2samples in by_dataset.items():
        print(f"\n=== Optimising credal weights for dataset: {d} ===")
        priors, w = train_one_dataset(
            d,
            prior2samples,
            steps=args.steps,
            batch=args.batch,
            lr=args.lr,
            seed=args.seed,
            critic_steps=args.critic_steps,
            alpha_steps=args.alpha_steps,
            print_every=args.print_every,
        )
        out[d] = {"priors": priors, "weights": w}

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved credal weights to {args.out}")


if __name__ == "__main__":
    main()
