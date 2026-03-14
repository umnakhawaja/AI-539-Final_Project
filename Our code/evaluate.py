import argparse
import csv
import os
from pathlib import Path
from scipy.linalg import sqrtm

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from model import RectifiedFlow, VelocityUNet
from schedules import (
    GradientAdjustingSchedule,
    GreedyChoosingSchedule,
    L_lipschitz,
    L_early_stopping,
    uniform_schedule,
    compute_EB,
    to_tensor,
)

# FID - lightweight approx. using inception features. 

def compute_fid_approx(real: torch.Tensor, fake: torch.Tensor) -> float:
    r = real.view(real.size(0), -1).float().numpy()
    f = fake.view(fake.size(0), -1).float().numpy()

    mu_r, mu_f = r.mean(0), f.mean(0)
    sigma_r = np.cov(r, rowvar=False) + np.eye(r.shape[1]) * 1e-6
    sigma_f = np.cov(f, rowvar=False) + np.eye(f.shape[1]) * 1e-6

    diff  = mu_r - mu_f
    mean_term = float(diff @ diff)

    # corrected matrix square root using scipy oops
    covmean = sqrtm(sigma_r @ sigma_f)

    # we dont want the imaginary parts 
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    trace_term = float(np.trace(sigma_r + sigma_f -2 * covmean))
    return mean_term + trace_term

# fme grid measurement 
@torch.no_grad()
def measure_fme_grid(
    model: RectifiedFlow,
    data_loader: DataLoader,
    t_grid: list,
    n_samples: int = 1024,
    device: torch.device = torch.device("cpu"),
) -> dict:
    # Gather a fixed batch of real images
    all_x = []
    for x, _ in data_loader:
        all_x.append(x)
        if sum(b.size(0) for b in all_x) >= n_samples:
            break
    x0 = torch.cat(all_x, 0)[:n_samples].to(device)

    model.net.eval()
    fme_dict = {}
    for t in t_grid:
        fme_dict[t] = model.compute_flow_matching_error(x0, t, n_samples=n_samples)
        print(f"  FME at t={t:.3f}: {fme_dict[t]:.4f}")
    return fme_dict

# visualization helpers
def plot_samples(samples: torch.Tensor, title: str, ax: plt.Axes, n: int = 16):
    samples = samples[:n].cpu()
    grid_size = int(n ** 0.5)
    img_size = 28
    canvas = np.zeros((grid_size * img_size, grid_size * img_size))
    for i, s in enumerate(samples):
        r, c = divmod(i, grid_size)
        # [-1,1] -> [0,1]
        img = (s.squeeze().numpy() + 1) / 2 
        canvas[r*img_size:(r+1)*img_size, c*img_size:(c+1)*img_size] = img
    ax.imshow(canvas, cmap="gray", vmin=0, vmax=1)
    ax.set_title(title, fontsize=10)
    ax.axis("off")

def plot_schedule(schedule: list, eb: float, label: str, ax: plt.Axes, color="C0"):
    ax.scatter(schedule, [0] * len(schedule), marker="|", s=200,
               linewidths=1.5, color=color, label=f"{label} (EB={eb:.4f})")
    ax.set_xlim(-0.02, 1.02)
    ax.set_yticks([])
    ax.legend(fontsize=8)

def plot_fme(t_grid: list, fme_vals: list, ax: plt.Axes):
    ax.plot(t_grid, fme_vals, color="pink", linewidth=1.5)
    ax.set_xlabel("t", fontsize=9)
    ax.set_ylabel("FME (ε(t))", fontsize=9)
    ax.set_title("Flow Matching Error across time", fontsize=10)
    ax.grid(True, alpha=0.3)


def plot_eb_curves(histories: dict, ax: plt.Axes):
    for label, hist in histories.items():
        ax.plot(hist, marker="o", markersize=3, label=label)
    ax.set_xlabel("Iteration", fontsize=9)
    ax.set_ylabel("EB", fontsize=9)
    ax.set_title("EB convergence", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

# cli bit
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--n_steps",type=int, default=10,
                   help="Number of sampling steps (N)")
    p.add_argument("--n_fme", type=int, default=50,
                   help="Number of t-grid points for FME measurement")
    p.add_argument("--n_samples", type=int, default=512,
                   help="Samples for FID and FME estimation")
    p.add_argument("--ga_iters", type=int, default=200)
    p.add_argument("--ga_lr", type=float, default=5e-4)
    p.add_argument("--ga_C", type=float, default=1.0)
    p.add_argument("--gc_iters", type=int, default=5)
    p.add_argument("--gc_C", type=float, default=10.0)
    p.add_argument("--gc_pool", type=int, default=100,
                   help="Number of available discrete time points for GC")
    p.add_argument("--gc_max_shift", type=float, default=0.15)
    p.add_argument("--data", type=str, default="./data")
    p.add_argument("--out", type=str, default="./results")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--base_ch", type=int, default=32)
    p.add_argument("--time_dim", type=int, default=128)
    p.add_argument("--smoothness", type=str, default="lipschitz",
                   choices=["lipschitz", "early_stopping"])
    return p.parse_args()

# mainnn

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # load the model
    ckpt = torch.load(args.ckpt, map_location=device)
    net  = VelocityUNet(in_channels=1, base_ch=args.base_ch, time_dim=args.time_dim).to(device)
    net.load_state_dict(ckpt["state_dict"])
    model = RectifiedFlow(net).to(device)
    net.eval()
    print(f"Loaded checkpoint from epoch {ckpt['epoch']}")

    # data loader
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    real_ds = datasets.MNIST(args.data, train=False, download=True, transform=tf)
    real_dl = DataLoader(real_ds, batch_size=256, shuffle=True, num_workers=2)

    # collect real samples for fid
    real_imgs = []
    for x, _ in real_dl:
        real_imgs.append(x)
        if sum(b.size(0) for b in real_imgs) >= args.n_samples:
            break
    real_imgs = torch.cat(real_imgs, 0)[:args.n_samples]

    # measure fme grid
    print("\n--- Measuring FME across time grid ---")
    t_grid = [i / (args.n_fme - 1) for i in range(args.n_fme)]
    # keep away from boundaries
    t_grid = [max(1e-3, min(1 - 1e-3, t)) for t in t_grid] 
    fme_dict = measure_fme_grid(model, real_dl, t_grid, n_samples=args.n_samples, device=device)

    # build eps_fn and L_fn from measurements
    fme_vals = [fme_dict[t] for t in t_grid]

    def eps_fn(t: float) -> float:
        """Interpolate FME at arbitrary t."""
        if t <= t_grid[0]:  return fme_vals[0]
        if t >= t_grid[-1]: return fme_vals[-1]
        for i in range(len(t_grid) - 1):
            if t_grid[i] <= t <= t_grid[i + 1]:
                alpha = (t - t_grid[i]) / (t_grid[i + 1] - t_grid[i])
                return fme_vals[i] * (1 - alpha) + fme_vals[i + 1] * alpha
        return fme_vals[-1]

    L_fn = L_lipschitz if args.smoothness == "lipschitz" else L_early_stopping

    # baseline - uniform schedule
    print("\n--- Baseline: Uniform schedule ---")
    N = args.n_steps
    uniform = uniform_schedule(N)
    uniform_eb = compute_EB(uniform, eps_fn, L_fn, args.ga_C)
    print(f"  Uniform EB = {uniform_eb:.6f}")

    # GA algo
    print(f"\n--- GA: Gradient-based Adjusting ({args.ga_iters} iters) ---")
    ga = GradientAdjustingSchedule(
        init_schedule=uniform[:],
        eps_fn=eps_fn,
        L_fn=L_fn,
        C=args.ga_C,
        lr=args.ga_lr,
        n_iters=args.ga_iters,
    )
    ga_schedule, ga_eb_hist = ga.run(verbose=True)
    ga_eb_final = ga_eb_hist[-1]
    print(f"  GA final EB = {ga_eb_final:.6f}")

    # GC algo
    print(f"\n--- GC: Greedy Choosing ({args.gc_iters} iters, pool={args.gc_pool}) ---")
    # discrete pool: uniformly spaced (simulates finite time embeddings)
    pool = [i / (args.gc_pool - 1) for i in range(args.gc_pool)]
    pool = [max(1e-4, min(1 - 1e-4, t)) for t in pool]
    pool[0] = 0.0; pool[-1] = 1.0

    # build eps_fn for GC at discrete points only
    def eps_fn_gc(t: float) -> float:
        # same interpolated FME
        return eps_fn(t)   

    # initial GC schedule: uniform points snapped to pool
    def snap_to_pool(pts, pool_set):
        pool_sorted = sorted(pool_set)
        snapped = []
        for p in pts:
            best = min(pool_sorted, key=lambda q: abs(q - p))
            snapped.append(best)
        return sorted(list(set(snapped)))

    pool_set = set(pool)
    gc_init = snap_to_pool(uniform, pool_set)
    # ensure we have exactly N+1 points
    while len(gc_init) < N + 1:
        # add extra point from pool not already in schedule
        candidates = sorted(pool_set - set(gc_init))
        gc_init.append(candidates[len(gc_init) // 2])
        gc_init = sorted(gc_init)
    gc_init = gc_init[:N + 1]

    gc = GreedyChoosingSchedule(
        available_points=pool,
        init_schedule=gc_init,
        eps_fn=eps_fn_gc,
        L_fn=L_fn,
        C=args.gc_C,
        n_iters=args.gc_iters,
        max_shift=args.gc_max_shift,
    )
    gc_schedule, gc_eb_hist = gc.run(verbose=True)
    gc_eb_final = gc_eb_hist[-1]
    print(f"  GC final EB = {gc_eb_final:.6f}")

    # generate samples
    print("\n--- Generating samples ---")
    n_gen = min(args.n_samples, 256)

    def gen(sched_list):
        t_tensor = to_tensor(sched_list, device)
        return model.sample(n_gen, t_tensor, device).cpu()

    samples_uniform = gen(uniform)
    samples_ga = gen(ga_schedule)
    samples_gc = gen(gc_schedule)

    # fid - pixel proxy 
    print("\n--- Computing FID (pixel approximation) ---")
    fid_uniform = compute_fid_approx(real_imgs[:n_gen], samples_uniform)
    fid_ga = compute_fid_approx(real_imgs[:n_gen], samples_ga)
    fid_gc = compute_fid_approx(real_imgs[:n_gen], samples_gc)

    print(f" FID (Uniform) = {fid_uniform:.4f}")
    print(f" FID (GA) = {fid_ga:.4f}")
    print(f" FID (GC) = {fid_gc:.4f}")

    # save results in csv
    csv_path = out_dir / "results.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["schedule", "EB", "FID_pixel", "n_steps"])
        w.writerow(["Uniform", uniform_eb, fid_uniform, N])
        w.writerow(["GA", ga_eb_final, fid_ga, N])
        w.writerow(["GC", gc_eb_final, fid_gc, N])
    print(f"\nResults saved to {csv_path}")

    # save schedules csv file
    sched_path = out_dir / "schedules.csv"
    with open(sched_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "uniform", "GA", "GC"])
        for i, (u, g, c) in enumerate(zip(uniform, ga_schedule, gc_schedule)):
            w.writerow([i, u, g, c])
    print(f"Schedules saved to {sched_path}")

    # the big figure lol
    fig = plt.figure(figsize=(22, 18))
    fig.suptitle(
        f"Adaptive Time-Stepping for Rectified Flow on MNIST\n"
        f"(checkpoint: {Path(args.ckpt).name}, N={N} steps, smoothness={args.smoothness})",
        fontsize=13, fontweight="bold",
    )

    # row 1: sample grids
    ax11 = fig.add_subplot(4, 3, 1)
    ax12 = fig.add_subplot(4, 3, 2)
    ax13 = fig.add_subplot(4, 3, 3)
    plot_samples(samples_uniform, f"Uniform (FID≈{fid_uniform:.1f})",  ax11)
    plot_samples(samples_ga, f"GA (FID≈{fid_ga:.1f})", ax12)
    plot_samples(samples_gc, f"GC (FID≈{fid_gc:.1f})", ax13)

    # row 2: schedule visualization
    ax21 = fig.add_subplot(4, 1, 2)
    ax21.set_title("Time-stepping schedules", fontsize=10)
    colors = ["gray", "C0", "C1"]
    ys = [0.8, 0.5, 0.2]
    labels_eb = [
        f"Uniform  (EB={uniform_eb:.4f})",
        f"GA       (EB={ga_eb_final:.4f})",
        f"GC       (EB={gc_eb_final:.4f})",
    ]
    for sched, y, col, lbl in zip(
        [uniform, ga_schedule, gc_schedule], ys, colors, labels_eb
    ):
        ax21.scatter(sched, [y] * len(sched), marker="|", s=300,
                     linewidths=2.0, color=col, label=lbl)
    ax21.set_xlim(-0.02, 1.02)
    ax21.set_ylim(0, 1)
    ax21.set_yticks(ys)
    ax21.set_yticklabels(["Uniform", "GA", "GC"], fontsize=9)
    ax21.set_xlabel("t", fontsize=9)
    ax21.legend(loc="upper right", fontsize=8)
    ax21.grid(axis="x", alpha=0.3)

    # row 3: fme curve
    ax31 = fig.add_subplot(4, 2, 5)
    plot_fme(t_grid, fme_vals, ax31)

    # row 3: eb convergence
    ax32 = fig.add_subplot(4, 2, 6)
    eb_histories = {
        f"GA (C={args.ga_C})": ga_eb_hist,
        f"GC (C={args.gc_C})": gc_eb_hist,
    }
    plot_eb_curves(eb_histories, ax32)

    # row4: bar chart comparison
    ax41 = fig.add_subplot(4, 2, 7)
    names = ["Uniform", "GA", "GC"]
    eb_vals = [uniform_eb, ga_eb_final, gc_eb_final]
    bars = ax41.bar(names, eb_vals, color=["gray", "C0", "C1"])
    ax41.set_title("EB comparison", fontsize=10)
    ax41.set_ylabel("EB", fontsize=9)
    for bar, val in zip(bars, eb_vals):
        ax41.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
                  f"{val:.4f}", ha="center", va="bottom", fontsize=8)

    ax42 = fig.add_subplot(4, 2, 8)
    fid_vals = [fid_uniform, fid_ga, fid_gc]
    bars2 = ax42.bar(names, fid_vals, color=["gray", "C0", "C1"])
    ax42.set_title("FID (pixel proxy) comparison", fontsize=10)
    ax42.set_ylabel("FID↓", fontsize=9)
    for bar, val in zip(bars2, fid_vals):
        ax42.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
                  f"{val:.1f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    fig_path = out_dir / "adaptive_rf_results.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"Figure saved to {fig_path}")
    plt.close()

    # print the summary 
    print("\n" + "=" * 55)
    print("SUMMARY")
    print("=" * 55)
    print(f"{'Schedule':<12} {'EB':>12} {'FID (proxy)':>14}")
    print("-" * 55)
    for name, eb, fid in [("Uniform", uniform_eb, fid_uniform),
                           ("GA",      ga_eb_final, fid_ga),
                           ("GC",      gc_eb_final, fid_gc)]:
        print(f"{name:<12} {eb:>12.6f} {fid:>14.4f}")
    print("=" * 55)

if __name__ == "__main__":
    main()
