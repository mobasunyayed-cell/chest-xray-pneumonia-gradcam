"""Step 5b: evaluation (metrics, threshold selection, ROC / confusion plots)."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score, roc_curve

from src.config import CFG


@torch.no_grad()
def predict(model, loader, device):
    """Return (y_true, prob_pneumonia) as numpy arrays."""
    model.eval()
    ys, ps = [], []
    for x, y, _ in loader:
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(x.to(device, non_blocking=True)).squeeze(1)
        ps.append(torch.sigmoid(logits.float()).cpu())
        ys.append(y)
    return torch.cat(ys).numpy(), torch.cat(ps).numpy()


def safe_auc(y, p) -> float:
    try:
        return float(roc_auc_score(y, p))
    except ValueError:  # only one class present
        return float("nan")


def best_threshold(y, p) -> float:
    """Youden's J (sensitivity + specificity - 1). Choose it on VAL only."""
    fpr, tpr, thr = roc_curve(y, p)
    return float(thr[int(np.argmax(tpr - fpr))])


def compute_metrics(y, p, thr: float = 0.5) -> dict:
    y = np.asarray(y).astype(int)
    pred = (np.asarray(p) >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    prec = tp / max(tp + fp, 1)
    f1 = 2 * prec * sens / max(prec + sens, 1e-9)
    return dict(
        threshold=round(float(thr), 4),
        accuracy=round(float((tp + tn) / len(y)), 4),
        sensitivity=round(float(sens), 4),
        specificity=round(float(spec), 4),
        precision=round(float(prec), 4),
        f1=round(float(f1), 4),
        auroc=round(safe_auc(y, p), 4),
        auprc=round(float(average_precision_score(y, p)), 4),
        tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
    )


def evaluate_all(model, loaders: dict, device, thr: float, out_dir=None) -> pd.DataFrame:
    """Evaluate val/test/external, save metrics.json + roc.png + confusion.png."""
    out_dir = Path(out_dir or CFG.paths.figures)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, preds = {}, {}
    for name in ("val", "test", "external"):
        if name not in loaders:
            continue
        y, p = predict(model, loaders[name], device)
        preds[name] = (y, p)
        rows[name] = compute_metrics(y, p, thr)
    table = pd.DataFrame(rows).T
    (out_dir / "metrics.json").write_text(json.dumps(rows, indent=2))
    table.to_csv(out_dir / "metrics.csv")

    # ROC curves
    fig, ax = plt.subplots(figsize=(5, 5))
    for name, (y, p) in preds.items():
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, label=f"{name} (AUC {rows[name]['auroc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("1 - specificity"); ax.set_ylabel("sensitivity"); ax.legend(); ax.set_title("ROC")
    fig.tight_layout(); fig.savefig(out_dir / "roc.png", dpi=150); plt.close(fig)

    # Confusion matrices
    fig, axes = plt.subplots(1, len(rows), figsize=(4 * len(rows), 3.6))
    axes = np.atleast_1d(axes)
    for ax, (name, m) in zip(axes, rows.items()):
        cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
        ax.imshow(cm, cmap="Blues")
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, v, ha="center", va="center", fontsize=13)
        ax.set_xticks([0, 1]); ax.set_xticklabels(CFG.classes, fontsize=8)
        ax.set_yticks([0, 1]); ax.set_yticklabels(CFG.classes, fontsize=8)
        ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(f"{name} (thr {m['threshold']})")
    fig.tight_layout(); fig.savefig(out_dir / "confusion.png", dpi=150); plt.close(fig)
    return table
