from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

MODEL_PATH = Path(os.getenv("MODEL_PATH", "artifacts/model.joblib"))

_state: dict = {}


def load_model(path: Path = MODEL_PATH) -> dict:
    bundle = joblib.load(path)
    _state.update(bundle)
    return bundle


@asynccontextmanager
async def lifespan(_: FastAPI):
    if "pipeline" not in _state:
        load_model()
    yield


app = FastAPI(title="Diabetes Progression Triage", lifespan=lifespan)


# Features are on the dataset's standardized scale (all within ±0.2). The bound rejects
# raw clinical values (e.g. age=45), which would otherwise yield a nonsense prediction.
FEATURE_BOUND = 0.5

# allow_inf_nan=False rejects NaN/Infinity, which would otherwise yield a NaN prediction.
_Feature = Field(..., allow_inf_nan=False, ge=-FEATURE_BOUND, le=FEATURE_BOUND)


class PatientFeatures(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "age": 0.02,
                "sex": -0.044,
                "bmi": 0.06,
                "bp": -0.03,
                "s1": -0.02,
                "s2": 0.03,
                "s3": -0.02,
                "s4": 0.02,
                "s5": 0.02,
                "s6": -0.001,
            }
        }
    )

    age: float = _Feature
    sex: float = _Feature
    bmi: float = _Feature
    bp: float = _Feature
    s1: float = _Feature
    s2: float = _Feature
    s3: float = _Feature
    s4: float = _Feature
    s5: float = _Feature
    s6: float = _Feature


class Prediction(BaseModel):
    prediction: float


def _error_field(err: dict) -> str:
    # For JSON parse errors the loc is ("body", <char offset>), not a field name.
    if err["type"] == "json_invalid":
        return "body"
    return ".".join(str(p) for p in err["loc"] if p != "body") or "body"


@app.exception_handler(RequestValidationError)
async def validation_error_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
    errors = [
        {
            "field": _error_field(err),
            "message": err["msg"],
            "type": err["type"],
        }
        for err in exc.errors()
    ]
    return JSONResponse(
        status_code=422,
        content={"error": "Invalid input", "details": errors},
    )


# Without this, an unexpected server error returns plain text "Internal Server Error".
# The traceback is still logged by uvicorn; the client only gets a generic JSON body.
@app.exception_handler(Exception)
async def unhandled_error_handler(_: Request, __: Exception) -> JSONResponse:
    return JSONResponse(status_code=500, content={"error": "Internal error"})


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_version": _state.get("model_version", "unknown")}


@app.post("/predict", response_model=Prediction)
def predict(features: PatientFeatures) -> Prediction:
    row = np.array([[getattr(features, name) for name in _state["features"]]])
    value = float(_state["pipeline"].predict(row)[0])
    return Prediction(prediction=value)
