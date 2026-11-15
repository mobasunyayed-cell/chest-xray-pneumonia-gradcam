import pytest

torch = pytest.importorskip("torch")
from api.service import ModelService  # noqa: E402
from src.model import build_model  # noqa: E402
from src.train import attach_metadata  # noqa: E402


def test_threshold_written_by_training_is_used_by_the_api(tmp_path):
    path = tmp_path / "best.pt"
    torch.save(
        {
            "model": build_model(pretrained=False).state_dict(),
            "backbone": "densenet121",
            "epoch": 3,
            "val_auc": 0.99,
        },
        path,
    )
    attach_metadata(path, 0.2901)
    service = ModelService.from_checkpoint(path)  # weights_only=True must still accept the file
    assert service.threshold == pytest.approx(0.2901) and service.threshold_source == "checkpoint"
    assert torch.load(path, weights_only=True)["classes"] == ["NORMAL", "PNEUMONIA"]
