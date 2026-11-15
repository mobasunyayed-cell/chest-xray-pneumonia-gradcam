"""FastAPI application.

Run from the repository root:
    uvicorn api.main:create_app --factory --port 8000
"""

from __future__ import annotations

import hmac
import io
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, File, Header, HTTPException, Query, Request, UploadFile
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

from api import __version__
from api.service import ModelLoadError, ModelService
from api.settings import Settings

log = logging.getLogger("cxr.api")

DISCLAIMER = (
    "Research demo, not a medical device. Trained on pediatric chest X-rays only; "
    "adult or non-chest images are outside its training distribution."
)
ALLOWED_FORMATS = {"JPEG", "PNG"}


class HealthOut(BaseModel):
    status: str
    model_loaded: bool
    version: str


class ModelOut(BaseModel):
    backbone: str
    checkpoint: str
    classes: list[str]
    threshold: float
    threshold_source: str
    device: str


class PredictionOut(BaseModel):
    label: str
    probability: float
    threshold: float
    heatmap_png_base64: str | None = None
    disclaimer: str = DISCLAIMER


def require_key(request: Request, x_api_key: Annotated[str | None, Header()] = None) -> None:
    key = request.app.state.config.api_key
    if key is None:
        return
    if x_api_key is None or not hmac.compare_digest(x_api_key, key):
        raise HTTPException(401, "missing or invalid X-API-Key")


def loaded_service(request: Request) -> ModelService:
    current: ModelService | None = request.app.state.service
    if current is None:
        raise HTTPException(503, "model not loaded: checkpoint file not found")
    return current


Authenticated = Depends(require_key)
ServiceDep = Annotated[ModelService, Depends(loaded_service)]


def create_app(settings: Settings | None = None, service: ModelService | None = None) -> FastAPI:
    config = settings or Settings.from_env()
    Image.MAX_IMAGE_PIXELS = config.max_pixels

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.service = service
        if service is None:
            if config.checkpoint.exists():
                # A bad checkpoint should stop the server, not hide behind 503s.
                app.state.service = ModelService.from_checkpoint(
                    config.checkpoint, device=config.device, threshold=config.threshold
                )
                log.info("loaded %s", config.checkpoint)
            else:
                log.warning("no checkpoint at %s; /predict will return 503", config.checkpoint)
        yield

    app = FastAPI(title="Chest X-ray pneumonia API", version=__version__, lifespan=lifespan)
    app.state.config = config
    app.state.service = None

    @app.get("/health", response_model=HealthOut)
    def health(request: Request) -> HealthOut:
        return HealthOut(
            status="ok", model_loaded=request.app.state.service is not None, version=__version__
        )

    @app.get("/model", response_model=ModelOut, dependencies=[Authenticated])
    def model_info(model: ServiceDep) -> ModelOut:
        return ModelOut(
            backbone=model.backbone,
            checkpoint=model.checkpoint_name,
            classes=model.classes,
            threshold=model.threshold,
            threshold_source=model.threshold_source,
            device=str(model.device),
        )

    @app.post("/predict", response_model=PredictionOut, dependencies=[Authenticated])
    def predict(
        model: ServiceDep,
        file: Annotated[UploadFile, File(description="A JPEG or PNG chest X-ray")],
        explain: Annotated[bool, Query(description="Include a Grad-CAM overlay")] = False,
        threshold: Annotated[float | None, Query(gt=0.0, lt=1.0)] = None,
    ) -> PredictionOut:
        limit = config.max_upload_mb * 1024 * 1024
        data = file.file.read(limit + 1)
        if len(data) > limit:
            raise HTTPException(413, f"file is larger than {config.max_upload_mb} MB")
        if not data:
            raise HTTPException(400, "the uploaded file is empty")
        try:
            image = Image.open(io.BytesIO(data))
            if image.format not in ALLOWED_FORMATS:
                raise HTTPException(415, "only JPEG and PNG images are accepted")
            image.load()
        except HTTPException:
            raise
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError, SyntaxError):
            raise HTTPException(400, "could not read the file as an image") from None
        result = model.predict(image, threshold=threshold, explain=explain)
        return PredictionOut(
            label=result.label,
            probability=round(result.probability, 6),
            threshold=result.threshold,
            heatmap_png_base64=result.heatmap_png_base64,
        )

    return app


__all__ = ["create_app", "ModelLoadError"]
