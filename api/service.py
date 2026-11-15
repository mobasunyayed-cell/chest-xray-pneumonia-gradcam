"""Model loading and inference, kept free of any web code so it is easy to test."""

from __future__ import annotations

import base64
import io
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.config import CFG
from src.data import get_transforms
from src.gradcam import explain_pil
from src.model import build_model

DEFAULT_THRESHOLD = 0.5


class ModelLoadError(RuntimeError):
    """The checkpoint exists but cannot be used."""


@dataclass(frozen=True)
class Prediction:
    probability: float
    label: str
    threshold: float
    heatmap_png_base64: str | None = None


def to_rgb(image: Image.Image) -> Image.Image:
    """Convert any supported mode to 8-bit RGB. 16-bit X-rays are rescaled, not clipped."""
    if image.mode in ("I;16", "I;16L", "I;16B", "I", "F"):
        array = np.asarray(image, dtype=np.float64)
        low, high = float(np.min(array)), float(np.max(array))
        scaled = (array - low) / (high - low) if high > low else np.zeros_like(array)
        image = Image.fromarray((scaled * 255).astype(np.uint8), mode="L")
    return image.convert("RGB")


def png_base64(array: np.ndarray) -> str:
    """Encode an (H, W, 3) float image in [0, 1] as a base64 PNG."""
    buffer = io.BytesIO()
    Image.fromarray((np.clip(array, 0, 1) * 255).astype(np.uint8)).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


class ModelService:
    def __init__(
        self,
        model: torch.nn.Module,
        *,
        backbone: str,
        threshold: float,
        threshold_source: str,
        checkpoint_name: str,
        device: str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.backbone = backbone
        self.threshold = threshold
        self.threshold_source = threshold_source
        self.checkpoint_name = checkpoint_name
        self.classes = list(CFG.classes)
        # Grad-CAM needs gradients and mutates model state, so requests take turns.
        self._lock = threading.Lock()

    @classmethod
    def from_checkpoint(
        cls, path: Path, *, device: str = "cpu", threshold: float | None = None
    ) -> ModelService:
        try:
            # weights_only=True refuses to unpickle arbitrary code from the file.
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            backbone = checkpoint.get("backbone", CFG.backbone)
            model = build_model(backbone=backbone, pretrained=False)
            model.load_state_dict(checkpoint["model"])
        except Exception as error:  # noqa: BLE001 - re-raised with context
            raise ModelLoadError(f"cannot load checkpoint {path}: {error}") from error

        if threshold is not None:
            value, source = threshold, "environment"
        elif "threshold" in checkpoint:
            value, source = float(checkpoint["threshold"]), "checkpoint"
        else:
            value, source = DEFAULT_THRESHOLD, "default"
        if not 0.0 < value < 1.0:
            raise ModelLoadError(f"threshold must be between 0 and 1, got {value}")
        return cls(
            model,
            backbone=backbone,
            threshold=value,
            threshold_source=source,
            checkpoint_name=Path(path).name,
            device=device,
        )

    def predict(
        self, image: Image.Image, *, threshold: float | None = None, explain: bool = False
    ) -> Prediction:
        rgb = to_rgb(image)
        limit = self.threshold if threshold is None else threshold
        with self._lock:
            if explain:
                probability, overlay = explain_pil(self.model, rgb, self.device, self.backbone)
                heatmap = png_base64(overlay)
            else:
                tensor = get_transforms(train=False)(rgb)[None].to(self.device)
                with torch.inference_mode():
                    probability = float(torch.sigmoid(self.model(tensor)).item())
                heatmap = None
        label = self.classes[1] if probability >= limit else self.classes[0]
        return Prediction(probability, label, limit, heatmap)
