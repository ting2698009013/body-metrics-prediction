"""FastAPI application for the local body-composition workbench."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, model_validator

from body_soft_sensor import __version__
from body_soft_sensor.inference import (
    BodyMeasurements,
    DERIVED_LIMITS,
    INPUT_LIMITS,
    ModelRegistry,
)


STATIC_DIR = Path(__file__).resolve().parent / "static"


class PredictionRequest(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    sex: Literal["male", "female"]
    age_years: int = Field(
        ge=int(INPUT_LIMITS["age_years"]["minimum"]),
        le=int(INPUT_LIMITS["age_years"]["maximum"]),
    )
    height_cm: float = Field(
        ge=INPUT_LIMITS["height_cm"]["minimum"],
        le=INPUT_LIMITS["height_cm"]["maximum"],
    )
    weight_kg: float = Field(
        ge=INPUT_LIMITS["weight_kg"]["minimum"],
        le=INPUT_LIMITS["weight_kg"]["maximum"],
    )
    waist_circumference_cm: float = Field(
        ge=INPUT_LIMITS["waist_circumference_cm"]["minimum"],
        le=INPUT_LIMITS["waist_circumference_cm"]["maximum"],
    )

    @model_validator(mode="after")
    def validate_derived_scope(self) -> "PredictionRequest":
        bmi = self.weight_kg / (self.height_cm / 100.0) ** 2
        waist_to_height = self.waist_circumference_cm / self.height_cm
        bmi_limits = DERIVED_LIMITS["bmi"]
        ratio_limits = DERIVED_LIMITS["waist_to_height_ratio"]
        if not bmi_limits["minimum"] <= bmi <= bmi_limits["maximum"]:
            raise ValueError("身高与体重组合超出模型训练范围")
        if not ratio_limits["minimum"] <= waist_to_height <= ratio_limits["maximum"]:
            raise ValueError("腰围与身高组合超出模型训练范围")
        return self


def create_app(registry: ModelRegistry | None = None, model_dir: Path | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(application: FastAPI):
        if registry is not None:
            application.state.registry = registry
            application.state.model_error = None
        else:
            try:
                application.state.registry = ModelRegistry(model_dir) if model_dir else ModelRegistry()
                application.state.model_error = None
            except Exception as error:  # health endpoint exposes a safe summary
                application.state.registry = None
                application.state.model_error = str(error)
        yield

    application = FastAPI(
        title="体成分估算台 API",
        version=__version__,
        description="Local inference API for the body-composition soft sensor.",
        lifespan=lifespan,
        docs_url="/api/docs",
        redoc_url=None,
    )
    application.mount("/assets", StaticFiles(directory=STATIC_DIR), name="assets")

    @application.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @application.get("/api/health")
    async def health(request: Request) -> dict[str, object]:
        active_registry = request.app.state.registry
        if active_registry is None:
            return {
                "status": "degraded",
                "models_loaded": 0,
                "detail": "模型文件缺失或不兼容。",
                "software_version": __version__,
                "model_version": None,
            }
        version = getattr(active_registry, "version", None)
        if version is None:
            version = active_registry.model_info().get("version", "unknown")
        return {
            "status": "ok",
            "models_loaded": len(active_registry.targets),
            "version": version,
            "software_version": __version__,
            "model_version": version,
        }

    @application.get("/api/v1/model-info")
    async def model_info(request: Request) -> dict[str, object]:
        active_registry = request.app.state.registry
        if active_registry is None:
            raise HTTPException(status_code=503, detail="模型尚未加载。")
        return active_registry.model_info()

    @application.post("/api/v1/predict")
    def predict(payload: PredictionRequest, request: Request) -> dict[str, object]:
        active_registry = request.app.state.registry
        if active_registry is None:
            raise HTTPException(status_code=503, detail="模型尚未加载。")
        measurements = BodyMeasurements(**payload.model_dump())
        return active_registry.predict(measurements)

    return application


app = create_app()


def run() -> None:
    uvicorn.run(
        "body_soft_sensor.webapp:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
    )

