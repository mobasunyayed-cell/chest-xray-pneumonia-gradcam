"""Step 6: Grad-CAM (self-contained, no extra dependency).

    python -m src.gradcam            # galleries for test + external -> outputs/figures/

Binary model: the target is the single "PNEUMONIA" logit, so the heatmap shows
which regions pushed the prediction toward pneumonia.
"""
from __future__ import annotations

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.config import CFG
from src.data import CXRDataset, denormalize, get_transforms
from src.evaluate import predict
from src.model import build_model


class GradCAM:
    """Grad-CAM on the last feature maps of DenseNet121 / EfficientNet-B0.

    We re-run the forward pass by hand (features -> pool -> classifier) so the
    feature maps are a normal tensor we can call retain_grad() on. This avoids
    the in-place ReLU / backward-hook problems of DenseNet.
    """

    def __init__(self, model, backbone: str | None = None):
        self.model = model
        self.backbone = backbone or CFG.backbone

    def _forward(self, x):
        f = self.model.features(x)
        a = F.relu(f) if self.backbone == "densenet121" else f
        a.retain_grad()
        pooled = F.adaptive_avg_pool2d(a, 1).flatten(1)
        return self.model.classifier(pooled), a

    def __call__(self, x):
        """x: (B,3,H,W) normalised. Returns cam (B,H,W) in [0,1], prob (B,)."""
        self.model.eval()
        with torch.enable_grad():
            self.model.zero_grad(set_to_none=True)
            logits, a = self._forward(x)
            logits.squeeze(1).sum().backward()
        w = a.grad.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((w * a).sum(dim=1, keepdim=True)).detach()
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
        mn = cam.amin(dim=(1, 2), keepdim=True)
        mx = cam.amax(dim=(1, 2), keepdim=True)
        cam = (cam - mn) / (mx - mn + 1e-8)
        return cam.cpu(), torch.sigmoid(logits.detach()).squeeze(1).cpu()


def overlay(img: torch.Tensor, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """img: (3,H,W) normalised tensor, cam: (H,W) in [0,1] -> (H,W,3) float image."""
    base = denormalize(img.cpu()).permute(1, 2, 0).numpy()
    heat = plt.get_cmap("jet")(cam)[..., :3]
    return np.clip((1 - alpha) * base + alpha * heat, 0, 1)


def explain_pil(model, pil_img, device, backbone: str | None = None):
    """For the demo app: PIL image -> (probability of pneumonia, overlay array)."""
    x = get_transforms(train=False)(pil_img.convert("RGB"))[None].to(device)
    cam, prob = GradCAM(model, backbone)(x)
    return float(prob[0]), overlay(x[0], cam[0].numpy())


def select_cases(y, p, thr, n, seed=CFG.seed):
    pred = (p >= thr).astype(int)
    rng = np.random.default_rng(seed)
    out = {}
    for name, (yt, yp) in {"TP": (1, 1), "TN": (0, 0), "FP": (0, 1), "FN": (1, 0)}.items():
        idx = np.where((y == yt) & (pred == yp))[0]
        out[name] = list(rng.choice(idx, size=min(n, len(idx)), replace=False)) if len(idx) else []
    return out


def gallery(model, df_split: pd.DataFrame, device, thr: float, n: int, out_path, title: str):
    ds = CXRDataset(df_split, get_transforms(train=False))
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=CFG.num_workers)
    y, p = predict(model, loader, device)  # same order as df_split
    cases = select_cases(y, p, thr, n)
    gc = GradCAM(model)

    fig, axes = plt.subplots(4, n, figsize=(2.6 * n, 2.8 * 4), squeeze=False)
    for r, (name, idxs) in enumerate(cases.items()):
        for c in range(n):
            ax = axes[r][c]
            ax.axis("off")
            if c >= len(idxs):
                continue
            x, _, _ = ds[int(idxs[c])]
            cam, prob = gc(x[None].to(device))
            ax.imshow(overlay(x, cam[0].numpy()))
            ax.set_title(f"{name}  p={float(prob[0]):.2f}", fontsize=9)
    fig.suptitle(f"{title}  (thr={thr:.3f}; TP/TN correct, FP/FN errors)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print("saved", out_path, {k: len(v) for k, v in cases.items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5, help="examples per category")
    ap.add_argument("--splits", default=str(CFG.paths.processed / "splits.csv"))
    ap.add_argument("--ckpt", default=str(CFG.paths.models / "best.pt"))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(pretrained=False).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device)["model"])

    thr = 0.5
    mpath = CFG.paths.figures / "metrics.json"
    if mpath.exists():
        thr = json.loads(mpath.read_text())["val"]["threshold"]

    df = pd.read_csv(args.splits)
    CFG.paths.make_dirs()
    for split in ("test", "external"):
        gallery(model, df[df.split == split], device, thr, args.n,
                CFG.paths.figures / f"gradcam_{split}.png", f"Grad-CAM: {split}")


if __name__ == "__main__":
    main()
