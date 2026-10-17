import argparse
import json
import os
import random
 
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
 
from dataset import CLASS_NAMES, get_dataloaders, load_splits, save_splits
from modules import build_model
from utils import DiceCELoss, count_parameters, evaluate
 
PROSTATE = CLASS_NAMES.index("prostate")
 
 
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--out-dir", default="runs/improved")
    p.add_argument("--model", choices=["improved", "baseline"], default="improved")
    p.add_argument("--split-file", default=None, help="re-use an existing splits.json")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--base-channels", type=int, default=16)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--target-size", type=int, nargs=3, default=[128, 128, 64])
    p.add_argument("--slice-axis", type=int, default=2, help="axis along which slices are stacked")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-amp", action="store_true", help="disable mixed precision")
    p.add_argument("--max-train", type=int, default=None, help="limit #train volumes (smoke test)")
    return p.parse_args()
 
 
def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
 
 
def plot_history(hist, path):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(hist["train_loss"], label="train")
    ax[0].plot(hist["val_loss"], label="val")
    ax[0].set(xlabel="epoch", ylabel="loss (Dice+CE)", title="Loss")
    ax[0].legend()
    ax[1].plot(hist["val_fg_dice"], label="val mean foreground Dice")
    ax[1].plot(hist["val_prostate_dice"], label="val prostate Dice")
    ax[1].set(xlabel="epoch", ylabel="Dice", title="Validation Dice")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
 
 
def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = len(CLASS_NAMES)
    use_amp = device.type == "cuda" and not args.no_amp
 
    splits = load_splits(args.split_file) if args.split_file else None
    if splits is not None and args.max_train:
        splits["train"] = splits["train"][:args.max_train]
    loaders, splits = get_dataloaders(
        args.data_root, splits, tuple(args.target_size), args.batch_size,
        args.num_workers, num_classes, args.seed)
    if args.max_train and not args.split_file:
        loaders["train"].dataset.pairs = loaders["train"].dataset.pairs[:args.max_train]
    save_splits(splits, os.path.join(args.out_dir, "splits.json"))
 
    model = build_model(args.model, 1, num_classes, args.base_channels, args.dropout).to(device)
    n_params = count_parameters(model)
    print(f"Model: {args.model} | trainable parameters: {n_params:,} | device: {device}")
 
    criterion = DiceCELoss(num_classes)
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
 
    hist = {"train_loss": [], "val_loss": [], "val_fg_dice": [], "val_prostate_dice": []}
    best, ckpt_path = -1.0, os.path.join(args.out_dir, "best.pt")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()