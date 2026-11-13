"""Step 5a: training loop.

    python -m src.train --smoke      # quick pipeline check (tiny subset, 1 epoch)
    python -m src.train              # full training (needs GPU)
"""
from __future__ import annotations

import argparse
import random
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.config import CFG
from src.data import build_loaders, pos_weight
from src.evaluate import best_threshold, evaluate_all, predict, safe_auc
from src.model import build_model


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None):
    train = optimizer is not None
    model.train(train)
    use_amp = device.type == "cuda"
    total, n = 0.0, 0
    ys, ps = [], []
    with torch.set_grad_enabled(train):
        for x, y, _ in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device).float()
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(x).squeeze(1)
            loss = criterion(logits.float(), y)
            if train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            total += loss.item() * len(y)
            n += len(y)
            ys.append(y.cpu())
            ps.append(torch.sigmoid(logits.detach().float()).cpu())
    y = torch.cat(ys).numpy()
    p = torch.cat(ps).numpy()
    return total / n, safe_auc(y, p), float(((p >= 0.5) == y).mean())


def plot_history(hist: pd.DataFrame, path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    axes[0].plot(hist.epoch, hist.train_loss, label="train")
    axes[0].plot(hist.epoch, hist.val_loss, label="val")
    axes[0].set_title("loss"); axes[0].set_xlabel("epoch"); axes[0].legend()
    axes[1].plot(hist.epoch, hist.train_auc, label="train")
    axes[1].plot(hist.epoch, hist.val_auc, label="val")
    axes[1].set_title("AUROC"); axes[1].set_xlabel("epoch"); axes[1].legend()
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=CFG.epochs)
    ap.add_argument("--batch_size", type=int, default=CFG.batch_size)
    ap.add_argument("--lr", type=float, default=CFG.lr)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--splits", default=str(CFG.paths.processed / "splits.csv"))
    ap.add_argument("--smoke", action="store_true", help="tiny subset, 1 epoch")
    args = ap.parse_args()

    seed_everything(CFG.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    if device.type == "cpu" and not args.smoke:
        print("!!! No GPU. Turn on GPU in Session options (Accelerator), or use --smoke.")

    df = pd.read_csv(args.splits)
    if args.smoke:
        df = df.groupby(["split", "label"]).head(16)
        args.epochs = 1
    loaders = build_loaders(df, batch_size=args.batch_size)
    train_df = df[df.split == "train"]

    CFG.paths.make_dirs()
    ckpt = CFG.paths.models / ("smoke.pt" if args.smoke else "best.pt")

    model = build_model().to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight(train_df).to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=CFG.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    hist, best_auc, bad = [], -1.0, 0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_auc, tr_acc = run_epoch(model, loaders["train"], criterion, device, optimizer, scaler)
        va_loss, va_auc, va_acc = run_epoch(model, loaders["val"], criterion, device)
        scheduler.step()
        hist.append(dict(epoch=epoch, train_loss=tr_loss, train_auc=tr_auc, train_acc=tr_acc,
                         val_loss=va_loss, val_auc=va_auc, val_acc=va_acc))
        improved = np.nan_to_num(va_auc, nan=0.0) > best_auc
        if improved:
            best_auc, bad = float(np.nan_to_num(va_auc, nan=0.0)), 0
            torch.save({"model": model.state_dict(), "backbone": CFG.backbone,
                        "epoch": epoch, "val_auc": va_auc}, ckpt)
        else:
            bad += 1
        print(f"epoch {epoch:02d}/{args.epochs} | train loss {tr_loss:.4f} auc {tr_auc:.4f} | "
              f"val loss {va_loss:.4f} auc {va_auc:.4f} acc {va_acc:.4f} | "
              f"{time.time() - t0:.0f}s {'*best*' if improved else ''}")
        if bad >= args.patience:
            print(f"early stop (no val AUROC improvement for {args.patience} epochs)")
            break

    hist = pd.DataFrame(hist)
    hist.to_csv(CFG.paths.figures / "history.csv", index=False)
    plot_history(hist, CFG.paths.figures / "history.png")

    # ---- final evaluation with the best checkpoint ----
    model.load_state_dict(torch.load(ckpt, map_location=device)["model"])
    yv, pv = predict(model, loaders["val"], device)
    thr = best_threshold(yv, pv)  # chosen on VAL only, then applied to test/external
    table = evaluate_all(model, loaders, device, thr)
    print(f"\nthreshold from val (Youden J): {thr:.4f}\n")
    print(table.to_string())
    print("\nsaved:", ckpt, "and figures in", CFG.paths.figures)


if __name__ == "__main__":
    main()
