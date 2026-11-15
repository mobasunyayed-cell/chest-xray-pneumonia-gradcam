import base64
import io

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("fastapi")
import torch  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from PIL import Image  # noqa: E402

from api.main import create_app  # noqa: E402
from api.service import DEFAULT_THRESHOLD, ModelLoadError, ModelService, to_rgb  # noqa: E402
from api.settings import Settings  # noqa: E402
from src.model import build_model  # noqa: E402


@pytest.fixture(scope="session")
def checkpoint(tmp_path_factory):
    """A randomly initialised model. Enough to test the plumbing, says nothing about accuracy."""
    path = tmp_path_factory.mktemp("ckpt") / "best.pt"
    model = build_model(backbone="densenet121", pretrained=False)
    torch.save({"model": model.state_dict(), "backbone": "densenet121", "threshold": 0.3}, path)
    return path


@pytest.fixture(scope="session")
def service(checkpoint):
    return ModelService.from_checkpoint(checkpoint)


def png_bytes(size=(64, 48), mode="L", fmt="PNG", color=128):
    buffer = io.BytesIO()
    Image.new(mode, size, color).save(buffer, format=fmt)
    return buffer.getvalue()


def upload(data, name="xray.png"):
    return {"file": (name, data, "application/octet-stream")}


@pytest.fixture
def client(service):
    with TestClient(create_app(Settings(), service=service)) as test_client:
        yield test_client


def test_health_without_a_model_reports_not_loaded(tmp_path):
    settings = Settings(checkpoint=tmp_path / "missing.pt")
    with TestClient(create_app(settings)) as bare:
        assert bare.get("/health").json()["model_loaded"] is False
        assert bare.get("/model").status_code == 503
        assert bare.post("/predict", files=upload(png_bytes())).status_code == 503


def test_model_info_reports_threshold_source(client):
    body = client.get("/model").json()
    assert body["backbone"] == "densenet121" and body["classes"] == ["NORMAL", "PNEUMONIA"]
    assert body["threshold"] == 0.3 and body["threshold_source"] == "checkpoint"
    assert client.get("/health").json()["model_loaded"] is True


def test_predict_returns_probability_and_consistent_label(client):
    body = client.post("/predict", files=upload(png_bytes())).json()
    assert 0.0 <= body["probability"] <= 1.0
    expected = "PNEUMONIA" if body["probability"] >= body["threshold"] else "NORMAL"
    assert body["label"] == expected
    assert body["heatmap_png_base64"] is None and "not a medical device" in body["disclaimer"]


def test_threshold_override_flips_the_label(client):
    low = client.post("/predict?threshold=0.0001", files=upload(png_bytes())).json()
    high = client.post("/predict?threshold=0.9999", files=upload(png_bytes())).json()
    assert low["label"] == "PNEUMONIA" and high["label"] == "NORMAL"
    assert low["threshold"] == 0.0001


def test_explain_returns_a_png_overlay_and_same_probability(client):
    plain = client.post("/predict", files=upload(png_bytes())).json()
    explained = client.post("/predict?explain=true", files=upload(png_bytes())).json()
    overlay = Image.open(io.BytesIO(base64.b64decode(explained["heatmap_png_base64"])))
    assert overlay.format == "PNG" and overlay.size == (224, 224)
    assert explained["probability"] == pytest.approx(plain["probability"], abs=1e-4)


def test_jpeg_and_rgb_uploads_work(client):
    data = png_bytes(mode="RGB", fmt="JPEG", color=(90, 90, 90))
    assert client.post("/predict", files=upload(data, "x.jpg")).status_code == 200


def test_sixteen_bit_png_is_rescaled_not_rejected(client):
    array = np.linspace(0, 65535, 64 * 64, dtype=np.uint16).reshape(64, 64)
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    assert client.post("/predict", files=upload(buffer.getvalue())).status_code == 200


@pytest.mark.parametrize(
    ("data", "status"),
    [
        (b"", 400),
        (b"this is not an image", 400),
        (png_bytes(fmt="GIF", mode="P"), 415),
        (png_bytes(fmt="BMP", mode="RGB", color=(1, 2, 3)), 415),
    ],
)
def test_bad_uploads_are_rejected_with_a_clear_status(client, data, status):
    assert client.post("/predict", files=upload(data)).status_code == status


def test_truncated_png_is_a_400_not_a_crash(client):
    assert client.post("/predict", files=upload(png_bytes()[:30])).status_code == 400


def test_oversized_upload_is_413(service):
    app = create_app(Settings(max_upload_mb=1), service=service)
    with TestClient(app) as small:
        assert small.post("/predict", files=upload(b"x" * (1024 * 1024 + 1))).status_code == 413


def test_decompression_bomb_is_refused(service):
    app = create_app(Settings(max_pixels=1000), service=service)
    with TestClient(app) as strict:
        response = strict.post("/predict", files=upload(png_bytes(size=(200, 200))))
        assert response.status_code == 400


def test_missing_file_and_bad_threshold_are_422(client):
    assert client.post("/predict").status_code == 422
    for value in ("0", "1", "-0.2", "abc"):
        assert (
            client.post(f"/predict?threshold={value}", files=upload(png_bytes())).status_code == 422
        )


def test_api_key_protects_everything_but_health(service):
    app = create_app(Settings(api_key="s3cret"), service=service)
    with TestClient(app) as secured:
        assert secured.get("/health").status_code == 200
        assert secured.get("/model").status_code == 401
        assert secured.post("/predict", files=upload(png_bytes())).status_code == 401
        wrong = {"X-API-Key": "nope"}
        assert secured.post("/predict", files=upload(png_bytes()), headers=wrong).status_code == 401
        right = {"X-API-Key": "s3cret"}
        assert secured.post("/predict", files=upload(png_bytes()), headers=right).status_code == 200


def test_threshold_resolution_order(checkpoint, tmp_path):
    assert ModelService.from_checkpoint(checkpoint, threshold=0.7).threshold_source == "environment"
    assert ModelService.from_checkpoint(checkpoint).threshold == 0.3
    bare = tmp_path / "bare.pt"
    torch.save({"model": build_model(pretrained=False).state_dict()}, bare)
    fallback = ModelService.from_checkpoint(bare)
    assert fallback.threshold == DEFAULT_THRESHOLD and fallback.threshold_source == "default"


def test_corrupt_or_wrong_checkpoint_fails_loudly(tmp_path):
    garbage = tmp_path / "garbage.pt"
    garbage.write_bytes(b"not a checkpoint")
    with pytest.raises(ModelLoadError, match="cannot load checkpoint"):
        ModelService.from_checkpoint(garbage)
    wrong = tmp_path / "wrong.pt"
    torch.save({"model": {"layer.weight": torch.zeros(1)}}, wrong)
    with pytest.raises(ModelLoadError):
        ModelService.from_checkpoint(wrong)


def test_app_refuses_to_start_with_a_corrupt_checkpoint(tmp_path):
    bad = tmp_path / "bad.pt"
    bad.write_bytes(b"junk")
    with pytest.raises(ModelLoadError), TestClient(create_app(Settings(checkpoint=bad))):
        pass


def test_to_rgb_handles_constant_16_bit_images():
    flat = Image.fromarray(np.full((8, 8), 500, dtype=np.uint16))
    assert to_rgb(flat).getextrema() == ((0, 0), (0, 0), (0, 0))
