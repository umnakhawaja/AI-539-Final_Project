import torch
import torch.optim as optim
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from pathlib import Path

from model import RectifiedFlow, VelocityUNet
from schedules import (
    GradientAdjustingSchedule,
    GreedyChoosingSchedule,
    L_lipschitz,
    uniform_schedule,
    compute_EB,
    to_tensor,
)

# configurations

# increase to 50-100 for better quality
EPOCHS = 20
BATCH = 128
LR = 2e-4
BASE_CH = 32
TIME_DIM = 128
# number of sampling steps to compare
N_STEPS = 10
GA_ITERS = 100
GA_LR = 5e-4
GA_C = 1.0
GC_ITERS = 5
GC_C = 10.0
GC_POOL = 100
GC_MAX_SHIFT = 0.15
# samples to generate for the figure
N_GEN = 64
# points for the fme curve
N_FME = 30
SEED = 42
DATA_DIR = "./data"
OUT_DIR = Path("./demo_output")

torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}\n")
OUT_DIR.mkdir(parents=True, exist_ok=True)

#data 
tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])
full_ds = datasets.MNIST(DATA_DIR, train=True, download=True, transform=tf)
train_dl = DataLoader(full_ds, batch_size=BATCH, shuffle=True,
                      num_workers=2, pin_memory=True, drop_last=True)

# model
net   = VelocityUNet(in_channels=1, base_ch=BASE_CH, time_dim=TIME_DIM).to(device)
model = RectifiedFlow(net).to(device)
n_p   = sum(p.numel() for p in net.parameters()) / 1e6
print(f"Model: {n_p:.2f}M parameters")

# train
print(f"Training for {EPOCHS} epochs …")
optimizer = optim.AdamW(net.parameters(), lr=LR, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
loss_history = []

for epoch in range(1, EPOCHS + 1):
    net.train()
    running = 0.0
    for x, _ in train_dl:
        x = x.to(device, non_blocking=True)
        loss = model.get_loss(x)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        optimizer.step()
        running += loss.item()
    scheduler.step()
    avg = running / len(train_dl)
    loss_history.append(avg)
    if epoch % 5 == 0 or epoch == EPOCHS:
        print(f"  Epoch {epoch:3d}/{EPOCHS} | loss={avg:.4f}")

# Save the checkpoint
ckpt_path = OUT_DIR / "demo_checkpoint.pt"
torch.save({"epoch": EPOCHS, "state_dict": net.state_dict()}, ckpt_path)
print(f"Checkpoint saved: {ckpt_path}\n")

# measure FME
print("Measuring FME grid …")
net.eval()
# Grab a batch of real data
real_batch, _ = next(iter(train_dl))
real_batch = real_batch.to(device)

t_grid  = [max(0.01, i / (N_FME - 1)) for i in range(N_FME)]
fme_vals = []
for t in t_grid:
    fme = model.compute_flow_matching_error(real_batch, t, n_samples=min(256, BATCH))
    fme_vals.append(fme)
    print(f"  t={t:.3f}  FME={fme:.4f}")


def eps_fn(t: float) -> float:
    if t <= t_grid[0]:  return fme_vals[0]
    if t >= t_grid[-1]: return fme_vals[-1]
    for i in range(len(t_grid) - 1):
        if t_grid[i] <= t <= t_grid[i + 1]:
            a = (t - t_grid[i]) / (t_grid[i + 1] - t_grid[i])
            return fme_vals[i] * (1 - a) + fme_vals[i + 1] * a
    return fme_vals[-1]

# GA algo
print(f"\nRunning GA ({GA_ITERS} iterations) …")
uniform = uniform_schedule(N_STEPS)
ga = GradientAdjustingSchedule(
    init_schedule=uniform[:],
    eps_fn=eps_fn,
    L_fn=L_lipschitz,
    C=GA_C,
    lr=GA_LR,
    n_iters=GA_ITERS,
)
ga_sched, ga_eb_hist = ga.run(verbose=True)
print(f"GA schedule: {[f'{t:.3f}' for t in ga_sched]}")

# GC algo
print(f"\nRunning GC ({GC_ITERS} iterations) …")
pool = [i / (GC_POOL - 1) for i in range(GC_POOL)]
pool[0] = 0.0; pool[-1] = 1.0

pool_set = set(pool)
def snap(pts):
    p_sorted = sorted(pool_set)
    return sorted(list({min(p_sorted, key=lambda q: abs(q - p)) for p in pts}))

gc_init = snap(uniform)
while len(gc_init) < N_STEPS + 1:
    missing = sorted(pool_set - set(gc_init))
    gc_init.append(missing[len(gc_init) // 2])
    gc_init = sorted(gc_init)
gc_init = gc_init[:N_STEPS + 1]

gc = GreedyChoosingSchedule(
    available_points=pool,
    init_schedule=gc_init,
    eps_fn=eps_fn,
    L_fn=L_lipschitz,
    C=GC_C,
    n_iters=GC_ITERS,
    max_shift=GC_MAX_SHIFT,
)
gc_sched, gc_eb_hist = gc.run(verbose=True)
print(f"GC schedule: {[f'{t:.3f}' for t in gc_sched]}")

# generate the samples 
print("\nGenerating samples …")
def gen(sched):
    return model.sample(N_GEN, to_tensor(sched, device), device).cpu()

uniform_eb = compute_EB(uniform, eps_fn, L_lipschitz, GA_C)
ga_eb      = ga_eb_hist[-1]
gc_eb      = gc_eb_hist[-1]

samples_uniform = gen(uniform)
samples_ga      = gen(ga_sched)
samples_gc      = gen(gc_sched)

# figure visualization, graph making bit
def make_grid(samples, nrow=8):
    imgs = [(s.squeeze().numpy() + 1) / 2 for s in samples[:nrow*nrow]]
    rows = []
    for r in range(nrow):
        rows.append(np.concatenate(imgs[r*nrow:(r+1)*nrow], axis=1))
    return np.concatenate(rows, axis=0)

fig = plt.figure(figsize=(18, 12))
fig.suptitle("Adaptive Time-Stepping Schedules for Rectified Flow on MNIST", fontsize=14, fontweight="bold")

# Sample grids
for i, (samps, label, eb) in enumerate([
    (samples_uniform, "Uniform",  uniform_eb),
    (samples_ga,      "GA",       ga_eb),
    (samples_gc,      "GC",       gc_eb),
]):
    ax = fig.add_subplot(3, 3, i + 1)
    grid = make_grid(samps, nrow=8)
    ax.imshow(grid, cmap="gray", vmin=0, vmax=1)
    ax.set_title(f"{label}  (EB={eb:.4f})", fontsize=10)
    ax.axis("off")

# Schedule visualisation
ax_sched = fig.add_subplot(3, 1, 2)
for sched, y, col, lbl in [
    (uniform,   0.8, "gray", f"Uniform  (EB={uniform_eb:.4f})"),
    (ga_sched,  0.5, "C0",   f"GA       (EB={ga_eb:.4f})"),
    (gc_sched,  0.2, "C1",   f"GC       (EB={gc_eb:.4f})"),
]:
    ax_sched.scatter(sched, [y] * len(sched), marker="|", s=400, lw=2.5, color=col, label=lbl)
ax_sched.set_xlim(-0.02, 1.02)
ax_sched.set_ylim(0, 1)
ax_sched.set_yticks([0.8, 0.5, 0.2])
ax_sched.set_yticklabels(["Uniform", "GA", "GC"], fontsize=10)
ax_sched.set_xlabel("t", fontsize=10)
ax_sched.legend(loc="upper right", fontsize=9)
ax_sched.set_title("Time-stepping schedules", fontsize=11)
ax_sched.grid(axis="x", alpha=0.3)

# FME + eb convergence + training loss
ax_fme = fig.add_subplot(3, 3, 7)
ax_fme.plot(t_grid, fme_vals, "steelblue", lw=1.5)
ax_fme.set_xlabel("t"); ax_fme.set_ylabel("FME ε(t)")
ax_fme.set_title("Flow Matching Error"); ax_fme.grid(alpha=0.3)

ax_eb = fig.add_subplot(3, 3, 8)
ax_eb.plot(ga_eb_hist, "C0-o", markersize=3, label=f"GA (C={GA_C})")
ax_eb.plot(gc_eb_hist, "C1-o", markersize=3, label=f"GC (C={GC_C})")
ax_eb.set_xlabel("Iteration"); ax_eb.set_ylabel("EB")
ax_eb.set_title("EB convergence"); ax_eb.legend(fontsize=8); ax_eb.grid(alpha=0.3)

ax_loss = fig.add_subplot(3, 3, 9)
ax_loss.plot(loss_history, "C2", lw=1.5)
ax_loss.set_xlabel("Epoch"); ax_loss.set_ylabel("Loss")
ax_loss.set_title("Training loss"); ax_loss.grid(alpha=0.3)

plt.tight_layout()
fig_path = OUT_DIR / "demo_results.png"
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
print(f"\nFigure saved: {fig_path}")

# Summary
print("\n" + "=" * 50)
print(f"{'Schedule':<12} {'EB':>12}")
print("-" * 50)
for name, eb in [("Uniform", uniform_eb), ("GA", ga_eb), ("GC", gc_eb)]:
    print(f"{name:<12} {eb:>12.6f}")
print("=" * 50)
print(f"\nAll outputs in: {OUT_DIR}/")