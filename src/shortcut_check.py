"""Step 6b: does the model rely on image corners / borders (markers, text, edges)?

Re-evaluates val/test/external with the same threshold while hiding regions:
  original | corners masked (laterality marker / text) | outer border masked
A big drop when a region is hidden = the model uses it (possible shortcut).

    python -m src.shortcut_check
"""
from __future__ import annotations

import json

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.config import CFG
from src.data import CXRDataset, get_transforms
from src.evaluate import compute_metrics, predict
from src.model import build_model


def mask_corners(x, f=0.22):
    h, w = int(x.shape[-2] * f), int(x.shape[-1] * f)
    x = x.clone()
    x[:, :h, :w] = 0      # top-left (usually the "R" marker)
    x[:, :h, -w:] = 0     # top-right
    x[:, -h:, :w] = 0     # bottom-left
    x[:, -h:, -w:] = 0    # bottom-right (usually text)
    return x


def mask_border(x, f=0.12):
    h, w = int(x.shape[-2] * f), int(x.shape[-1] * f)
    out = torch.zeros_like(x)
    out[:, h:-h, w:-w] = x[:, h:-h, w:-w]
    return out


VARIANTS = {"original": lambda x: x, "corners_masked": mask_corners, "border_masked": mask_border}


class MaskedDataset(CXRDataset):
    def __init__(self, df, fn):
        super().__init__(df, get_transforms(train=False))
        self.fn = fn

    def __getitem__(self, i):
        x, y, p = super().__getitem__(i)
        return self.fn(x), y, p


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(pretrained=False).to(device)
    model.load_state_dict(torch.load(CFG.paths.models / "best.pt", map_location=device)["model"])
    thr = json.loads((CFG.paths.figures / "metrics.json").read_text())["val"]["threshold"]
    df = pd.read_csv(CFG.paths.processed / "splits.csv")

    rows = []
    for split in ("test", "external"):
        d = df[df.split == split]
        for name, fn in VARIANTS.items():
            loader = DataLoader(MaskedDataset(d, fn), batch_size=64, shuffle=False,
                                num_workers=CFG.num_workers)
            y, p = predict(model, loader, device)
            m = compute_metrics(y, p, thr)
            rows.append(dict(split=split, variant=name, accuracy=m["accuracy"],
                             sensitivity=m["sensitivity"], specificity=m["specificity"],
                             auroc=m["auroc"], fp=m["fp"], fn=m["fn"]))
    out = pd.DataFrame(rows)
    out.to_csv(CFG.paths.figures / "shortcut_check.csv", index=False)
    print(f"threshold (from val): {thr:.4f}\n")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
