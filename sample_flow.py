import os, argparse, numpy as np, torch
from flow_matching.solver import ODESolver
from train_flow import Device, VelocityMLP, make_prior

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="eight_ring", choices=["eight_ring","spirals","moons"])
    ap.add_argument("--prior",   default="gaussian",
                    choices=["gaussian","gaussian_narrow","gaussian_wide","student_t","ringmix"])
    ap.add_argument("--outdir", default="out_fm_solver")
    ap.add_argument("--n_samples", type=int, default=20000)
    return ap.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    model_path = os.path.join(args.outdir, f"fm_{args.dataset}_{args.prior}.pt")
    samp_path  = os.path.join(args.outdir, f"samples_{args.dataset}_{args.prior}_resampled.npy")

    model = VelocityMLP(hidden_dim=128).to(Device)
    model.load_state_dict(torch.load(model_path, map_location=Device))
    model.eval()

    prior = make_prior(args.prior)
    x0 = prior.sample(args.n_samples).to(Device)

    solver = ODESolver(model)
    samples = solver.sample(x0).cpu()

    np.save(samp_path, samples.numpy())
    print("Saved samples to", samp_path)

if __name__ == "__main__":
    main()
