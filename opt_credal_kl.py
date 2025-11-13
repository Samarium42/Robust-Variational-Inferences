
import os, json, math, argparse, numpy as np, torch
from torch import nn
from collections import defaultdict

# ---- import your dataset factories ----
from train_flow import make_dataset, Device

def load_manifest(path):
    rows = []
    with open(path, "r") as f:
        for line in f:
            line=line.strip()
            if not line: continue
            d,p,ckpt,samp = line.split(",")
            rows.append((d,p,ckpt,samp))
    return rows

class Critic(nn.Module):
    # Small MLP T(x): R^2 -> R (works for your 2D toys)
    def __init__(self, hidden=128, depth=3):
        super().__init__()
        layers = [nn.Linear(2, hidden), nn.SiLU()]
        for _ in range(depth-1):
            layers += [nn.Linear(hidden, hidden), nn.SiLU()]
        layers += [nn.Linear(hidden, 1)]
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x).squeeze(-1)  # [B]

def batch_expectation(values):
    # numerically stable mean of e^{T(x)} using log-sum-exp
    # input: tensor [B]; returns scalar E[e^{T(x)}] ~ (1/B) * sum e^{T}
    m = values.max()
    return torch.exp(values - m).mean() * torch.exp(m)

def train_one_dataset(dataset_name, prior2samples, *, steps=3000, batch=2048, lr=1e-3, seed=123):
    torch.manual_seed(seed); np.random.seed(seed)

    # ---- target data sampler P ----
    P = make_dataset(dataset_name)

    # ---- cache: tensors per prior k ----
    # each entry: torch.Tensor [N,2] on Device
    q_samples = {}
    for prior_name, npy_path in prior2samples.items():
        arr = np.load(npy_path)  # shape [N,2]
        q_samples[prior_name] = torch.tensor(arr, dtype=torch.float32, device=Device)

    priors = sorted(q_samples.keys())
    K = len(priors)
    # softmax params for weights
    alpha = torch.zeros(K, device=Device, requires_grad=True)  # start ~uniform
    critic = Critic(hidden=128, depth=3).to(Device)

    opt = torch.optim.AdamW(list(critic.parameters()) + [alpha], lr=lr, weight_decay=1e-2)

    # simple cyclic dataloaders from cached samples
    ptrs = {k: 0 for k in priors}
    Ns = {k: q_samples[k].shape[0] for k in priors}

    def sample_from_Qk(k, B):
        i = ptrs[k]
        x = q_samples[k][i:i+B]
        if x.shape[0] < B:  # wrap-around
            rem = B - x.shape[0]
            x = torch.cat([x, q_samples[k][0:rem]], dim=0)
            ptrs[k] = rem
        else:
            ptrs[k] = i + B
        return x

    # training
    for step in range(1, steps+1):
        # --- sample from P (true data) on the fly
        x_p = P.sample(batch).to(Device)  # [B,2]
        T_p = critic(x_p)                 # [B]

        # --- compute E_{Q_k}[exp(T(x))] for each k with its own batch
        E_expT = []
        for k in priors:
            x_qk = sample_from_Qk(k, batch)          # [B,2]
            T_qk = critic(x_qk)                       # [B]
            E_expT_k = batch_expectation(T_qk)        # scalar
            E_expT.append(E_expT_k)
        E_expT = torch.stack(E_expT)                  # [K]

        w = torch.softmax(alpha, dim=0)               # [K], simplex weights

        # variational lower bound: E_P[T] - log( sum_k w_k * E_{Q_k}[e^{T}] )
        term_p = T_p.mean()
        mix_term = torch.sum(w * E_expT)
        loss = -( term_p - torch.log(mix_term + 1e-12) )  # negative to minimize

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
        opt.step()

        if step % 200 == 0 or step == 1:
            with torch.no_grad():
                val = (term_p - torch.log(mix_term + 1e-12)).item()
                print(f"[{dataset_name}] step {step:5d}/{steps}  KL_lb≈{val:.4f}  "
                      f"w=" + " ".join(f"{p}:{wi:.2f}" for p,wi in zip(priors, w.tolist())))

    with torch.no_grad():
        w = torch.softmax(alpha, dim=0).cpu().numpy().tolist()
    return priors, w

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="out_fm_solver/manifest.txt")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default="out_fm_solver/credal_weights.json")
    args = ap.parse_args()

    rows = load_manifest(args.manifest)
    # group by dataset
    by_dataset = defaultdict(dict)  # dataset -> prior_name -> samples_path
    for d,p,ckpt,samp in rows:
        by_dataset[d][p] = samp

    results = {}
    for d, p2s in by_dataset.items():
        print(f"\n=== Optimising credal weights for dataset: {d} ===")
        priors, w = train_one_dataset(d, p2s, steps=args.steps, batch=args.batch, lr=args.lr)
        results[d] = {"priors": priors, "weights": w}

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved weights to {args.out}")

if __name__ == "__main__":
    main()
