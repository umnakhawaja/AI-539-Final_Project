import argparse
import csv
import os
import time
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import RectifiedFlow, VelocityUNet

# CLI
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",     type=int,   default=50)
    p.add_argument("--batch",      type=int,   default=128)
    p.add_argument("--lr",         type=float, default=2e-4)
    p.add_argument("--base_ch",    type=int,   default=32,
    help="Base channel width of U-Net (increase for capacity)")
    
    p.add_argument("--time_dim",   type=int,   default=128)
    p.add_argument("--out",        type=str,   default="./checkpoints")
    p.add_argument("--data",       type=str,   default="./data")
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--save_every", type=int,   default=10)
    return p.parse_args()

# main
def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # data bit
    tf = transforms.Compose([
        transforms.ToTensor(),
        # [0,1] -> [-1,1]
        transforms.Normalize((0.5,), (0.5,)),
    ])
    train_ds = datasets.MNIST(args.data, train=True,  download=True, transform=tf)
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                          num_workers=4, pin_memory=True, drop_last=True)

    # model bit
    net   = VelocityUNet(in_channels=1, base_ch=args.base_ch, time_dim=args.time_dim).to(device)
    model = RectifiedFlow(net).to(device)
    n_params = sum(p.numel() for p in net.parameters()) / 1e6
    print(f"Model parameters: {n_params:.2f}M")

    optim_ = optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched  = optim.lr_scheduler.CosineAnnealingLR(optim_, T_max=args.epochs)

    # output dir bit
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_log.csv"
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "loss", "time_s"])

    # train bit
    print(f"\nTraining for {args.epochs} epochs …\n")
    for epoch in range(1, args.epochs + 1):
        net.train()
        t0 = time.time()
        running = 0.0

        for x, _ in train_dl:
            x = x.to(device, non_blocking=True)
            loss = model.get_loss(x)
            optim_.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optim_.step()
            running += loss.item()

        sched.step()
        elapsed = time.time() - t0
        avg_loss = running / len(train_dl)
        print(f"Epoch {epoch:3d}/{args.epochs} | loss={avg_loss:.4f} | {elapsed:.1f}s")

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, avg_loss, elapsed])

        if epoch % args.save_every == 0 or epoch == args.epochs:
            ckpt_path = out_dir / f"rf_mnist_ep{epoch:04d}.pt"
            torch.save({
                "epoch": epoch,
                "state_dict": net.state_dict(),
                "optimizer": optim_.state_dict(),
                "args": vars(args),
            }, ckpt_path)
            print(f"  Saved checkpoint: {ckpt_path}")

    print("\nTraining complete.")


if __name__ == "__main__":
    main()