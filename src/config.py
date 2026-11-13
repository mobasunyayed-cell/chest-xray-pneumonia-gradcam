"""Central configuration. Every other module imports from here."""
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Paths:
    raw: Path = ROOT / "data" / "raw"
    processed: Path = ROOT / "data" / "processed"
    models: Path = ROOT / "outputs" / "models"
    figures: Path = ROOT / "outputs" / "figures"

    def make_dirs(self) -> None:
        for p in (self.raw, self.processed, self.models, self.figures):
            p.mkdir(parents=True, exist_ok=True)


@dataclass
class Config:
    # --- task ---
    # Start with binary pneumonia; extend this list for multi-label later.
    classes: list = field(default_factory=lambda: ["NORMAL", "PNEUMONIA"])
    task: str = "binary"  # "binary" | "multilabel"

    # --- data ---
    img_size: int = 224
    val_split: float = 0.15
    test_split: float = 0.15
    seed: int = 42
    num_workers: int = 2

    # --- model ---
    backbone: str = "densenet121"  # "densenet121" | "efficientnet_b0"
    pretrained: bool = True
    dropout: float = 0.3

    # --- training ---
    batch_size: int = 32
    epochs: int = 15
    lr: float = 1e-4
    weight_decay: float = 1e-5

    paths: Paths = field(default_factory=Paths)


CFG = Config()
