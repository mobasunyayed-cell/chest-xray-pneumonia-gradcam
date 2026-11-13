"""Step 4: model factory (DenseNet121 / EfficientNet-B0) + helpers.

Binary task  -> 1 logit, use BCEWithLogitsLoss(pos_weight=data.pos_weight(...)).
Multi-label  -> len(classes) logits, same loss (pos_weight per class if needed).
"""
from __future__ import annotations

import warnings

import torch
import torch.nn as nn
from torchvision import models

from src.config import CFG


def n_outputs(classes=None, task=None) -> int:
    classes = classes or CFG.classes
    task = task or CFG.task
    return 1 if task == "binary" else len(classes)


def build_model(backbone: str | None = None, pretrained: bool | None = None,
                dropout: float | None = None, num_outputs: int | None = None) -> nn.Module:
    backbone = backbone or CFG.backbone
    pretrained = CFG.pretrained if pretrained is None else pretrained
    dropout = CFG.dropout if dropout is None else dropout
    num_outputs = num_outputs or n_outputs()

    if backbone == "densenet121":
        w = models.DenseNet121_Weights.DEFAULT if pretrained else None
        m = _load(models.densenet121, w)
        m.classifier = nn.Sequential(nn.Dropout(dropout),
                                     nn.Linear(m.classifier.in_features, num_outputs))
    elif backbone == "efficientnet_b0":
        w = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        m = _load(models.efficientnet_b0, w)
        m.classifier = nn.Sequential(nn.Dropout(dropout),
                                     nn.Linear(m.classifier[1].in_features, num_outputs))
    else:
        raise ValueError(f"Unknown backbone: {backbone}")
    return m


def _load(fn, weights):
    """Load with pretrained weights; fall back to random init (loudly) if the
    download fails, e.g. Kaggle notebook with Internet turned off."""
    try:
        return fn(weights=weights)
    except Exception as e:  # noqa: BLE001
        warnings.warn(
            f"\n!!! Could not load pretrained weights ({type(e).__name__}). "
            "Using RANDOM init. Turn Internet ON in the notebook settings.\n")
        return fn(weights=None)


def gradcam_target_layer(model: nn.Module, backbone: str | None = None):
    """Last conv feature block, used by Grad-CAM in step 6."""
    backbone = backbone or CFG.backbone
    if backbone in ("densenet121", "efficientnet_b0"):
        return model.features[-1]
    raise ValueError(f"Unknown backbone: {backbone}")


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    """Freeze/unfreeze everything except the classifier head."""
    for name, p in model.named_parameters():
        if not name.startswith("classifier"):
            p.requires_grad = trainable


def count_params(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, train


def logits_to_prob(logits: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(logits)


if __name__ == "__main__":
    m = build_model()
    x = torch.randn(2, 3, CFG.img_size, CFG.img_size)
    out = m(x)
    print("backbone:", CFG.backbone, "| output shape:", tuple(out.shape))
    print("params (total, trainable):", count_params(m))
    print("gradcam layer:", type(gradcam_target_layer(m)).__name__)
