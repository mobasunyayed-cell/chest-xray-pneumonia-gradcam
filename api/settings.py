"""Configuration, read from environment variables (all optional)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    checkpoint: Path = ROOT / "outputs" / "models" / "best.pt"
    threshold: float | None = None  # None: use the checkpoint's value, else 0.5
    device: str = "cpu"
    max_upload_mb: int = 10
    max_pixels: int = 50_000_000
    api_key: str | None = None  # None: no authentication (fine on localhost only)

    @classmethod
    def from_env(cls) -> Settings:
        env = os.environ
        threshold = env.get("CXR_THRESHOLD")
        return cls(
            checkpoint=Path(env.get("CXR_CHECKPOINT", str(cls.checkpoint))),
            threshold=float(threshold) if threshold else None,
            device=env.get("CXR_DEVICE", cls.device),
            max_upload_mb=int(env.get("CXR_MAX_UPLOAD_MB", cls.max_upload_mb)),
            api_key=env.get("CXR_API_KEY") or None,
        )
