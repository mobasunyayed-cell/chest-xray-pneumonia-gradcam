"""Step 3: data scanning, patient-aware 70/15/15 split, transforms, loaders.

Usage (CLI):
    python -m src.data --root /kaggle/input/chest-xray-pneumonia/chest_xray

Design notes
- The original `val/` has only 16 images, so we pool original train+val and
  re-split 70/15/15 (train/val/test). The original `test/` is kept untouched as
  an optional *external* test set (it comes from a different distribution).
- Patient leakage: file names encode patient ids (`person12_bacteria_34.jpeg`,
  `IM-0115-0001.jpeg`, `NORMAL2-IM-0373-0001.jpeg`). We split by patient
  (StratifiedGroupKFold) so one patient never lands in two splits.
  Files with an unparseable name are treated as their own patient.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms as T

from src.config import CFG

EXTS = {".jpeg", ".jpg", ".png"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

_PERSON = re.compile(r"person(\d+)_", re.I)
_NORMAL2 = re.compile(r"NORMAL2-IM-(\d+)-", re.I)
_NORMAL = re.compile(r"IM-(\d+)-", re.I)


# ----------------------------------------------------------------- scanning
def find_root(base) -> Path | None:
    """Return the folder that contains train/ and test/."""
    for p in Path(base).rglob("train"):
        if p.is_dir() and (p.parent / "test").exists():
            return p.parent
    return None


def patient_id(path: str | Path) -> str:
    name = Path(path).name
    if m := _PERSON.search(name):
        return f"P{m.group(1)}"
    if m := _NORMAL2.search(name):
        return f"N2-{m.group(1)}"
    if m := _NORMAL.search(name):
        return f"N-{m.group(1)}"
    return f"F-{Path(path).stem}"


def scan_dataset(root, classes=None) -> pd.DataFrame:
    """DataFrame: path, label (str), y (int), orig_split, patient."""
    root = Path(root)
    classes = classes or CFG.classes
    rows = []
    for split in ("train", "val", "test"):
        for y, cls in enumerate(classes):
            d = root / split / cls
            if not d.is_dir():
                continue
            for f in sorted(d.iterdir()):
                if f.suffix.lower() in EXTS:
                    rows.append(dict(path=str(f), label=cls, y=y,
                                     orig_split=split, patient=patient_id(f)))
    if not rows:
        raise FileNotFoundError(f"No images found under {root}")
    return pd.DataFrame(rows)


# ----------------------------------------------------------------- splitting
def make_splits(df: pd.DataFrame, val_frac=None, test_frac=None, seed=None) -> pd.DataFrame:
    """Pool original train+val, split by patient into train/val/test.

    The original test folder is returned with split == "external".
    """
    val_frac = CFG.val_split if val_frac is None else val_frac
    test_frac = CFG.test_split if test_frac is None else test_frac
    seed = CFG.seed if seed is None else seed

    df = df.copy()
    pool = df[df.orig_split.isin(["train", "val"])].copy()

    # 1) carve out test (~test_frac)
    k = max(2, round(1 / test_frac))
    sgkf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
    rest_idx, test_idx = next(sgkf.split(pool, pool.y, pool.patient))
    test, rest = pool.iloc[test_idx], pool.iloc[rest_idx]

    # 2) carve out val from the rest (~val_frac of the whole pool)
    k = max(2, round((1 - test_frac) / val_frac))
    sgkf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
    tr_idx, val_idx = next(sgkf.split(rest, rest.y, rest.patient))
    train, val = rest.iloc[tr_idx], rest.iloc[val_idx]

    df["split"] = "external"
    df.loc[train.index, "split"] = "train"
    df.loc[val.index, "split"] = "val"
    df.loc[test.index, "split"] = "test"
    df.loc[df.orig_split == "test", "split"] = "external"
    return df


def check_no_leakage(df: pd.DataFrame) -> None:
    sets = {s: set(df[df.split == s].patient) for s in ("train", "val", "test")}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = sets[a] & sets[b]
        assert not overlap, f"Patient leakage {a}/{b}: {len(overlap)} patients"


def split_summary(df: pd.DataFrame) -> pd.DataFrame:
    t = df.groupby(["split", "label"]).size().unstack(fill_value=0)
    t["total"] = t.sum(axis=1)
    t["pneumonia_%"] = (100 * t.get("PNEUMONIA", 0) / t["total"]).round(1)
    return t


# ----------------------------------------------------------------- weights
def class_weights(df_train: pd.DataFrame, n_classes: int | None = None) -> torch.Tensor:
    """Inverse-frequency weights, normalised so they average to 1."""
    n_classes = n_classes or len(CFG.classes)
    counts = np.bincount(df_train.y.values, minlength=n_classes).astype(np.float64)
    w = counts.sum() / (n_classes * np.maximum(counts, 1))
    return torch.tensor(w / w.mean(), dtype=torch.float32)


def pos_weight(df_train: pd.DataFrame) -> torch.Tensor:
    """For BCEWithLogitsLoss in the binary setup: n_neg / n_pos."""
    pos = int((df_train.y == 1).sum())
    neg = int((df_train.y == 0).sum())
    return torch.tensor([neg / max(pos, 1)], dtype=torch.float32)


# -------------------------------------------------------------- transforms
def get_transforms(train: bool, size: int | None = None):
    size = size or CFG.img_size
    norm = T.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    if train:
        # Mild, anatomy-safe augmentation. No horizontal flip (heart/liver side).
        return T.Compose([
            T.Resize((size, size)),
            T.RandomAffine(degrees=7, translate=(0.05, 0.05), scale=(0.95, 1.05)),
            T.ColorJitter(brightness=0.15, contrast=0.15),
            T.ToTensor(),
            norm,
        ])
    return T.Compose([T.Resize((size, size)), T.ToTensor(), norm])


def denormalize(x: torch.Tensor) -> torch.Tensor:
    """Undo ImageNet normalisation (for plotting / Grad-CAM overlays)."""
    mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(-1, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device).view(-1, 1, 1)
    return (x * std + mean).clamp(0, 1)


# ----------------------------------------------------------------- dataset
class CXRDataset(Dataset):
    """Returns (3xHxW float tensor, label tensor, path)."""

    def __init__(self, df: pd.DataFrame, transform=None):
        self.paths = df.path.tolist()
        self.y = df.y.values
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")  # grayscale -> 3 channels
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(self.y[i], dtype=torch.long), self.paths[i]


def build_loaders(df: pd.DataFrame, batch_size=None, num_workers=None,
                  balanced_sampler: bool = False) -> dict[str, DataLoader]:
    """Loaders for train / val / test (+ external if present).

    Default is class-weighted loss (see `class_weights`); set
    `balanced_sampler=True` to oversample instead (don't combine both).
    """
    batch_size = batch_size or CFG.batch_size
    num_workers = CFG.num_workers if num_workers is None else num_workers
    pin = torch.cuda.is_available()
    loaders = {}
    for s in ("train", "val", "test", "external"):
        d = df[df.split == s]
        if d.empty:
            continue
        is_train = s == "train"
        ds = CXRDataset(d, get_transforms(train=is_train))
        sampler = None
        if is_train and balanced_sampler:
            w = class_weights(d)[d.y.values].numpy()
            sampler = WeightedRandomSampler(w, num_samples=len(d), replacement=True)
        loaders[s] = DataLoader(ds, batch_size=batch_size,
                                shuffle=is_train and sampler is None,
                                sampler=sampler, num_workers=num_workers,
                                pin_memory=pin, drop_last=False)
    return loaders


# --------------------------------------------------------------------- CLI
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None, help="folder containing train/ val/ test/")
    ap.add_argument("--out", default=None, help="where to write splits.csv")
    args = ap.parse_args()

    root = Path(args.root) if args.root else find_root(CFG.paths.raw) or find_root("/kaggle/input")
    if root is None:
        raise SystemExit("Dataset not found. Pass --root.")
    CFG.paths.make_dirs()
    out = Path(args.out) if args.out else CFG.paths.processed / "splits.csv"

    df = make_splits(scan_dataset(root))
    check_no_leakage(df)
    df.to_csv(out, index=False)

    print(f"root: {root}\nsaved: {out}\n")
    print(split_summary(df))
    print("\nunique patients per split:")
    print(df.groupby("split").patient.nunique())
    print("\nclass weights (train):", class_weights(df[df.split == 'train']).tolist())
    print("pos_weight  (train):", pos_weight(df[df.split == 'train']).tolist())


if __name__ == "__main__":
    main()
