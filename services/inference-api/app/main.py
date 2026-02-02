# services/inference-api/app/main.py
from __future__ import annotations

import os
from pathlib import Path

import joblib
import requests
from fastapi import BackgroundTasks, FastAPI, HTTPException

from .feature_build import build_feature_row
from .schemas import PredictRequest, PredictResponse

# In docker-compose we will set MODEL_PATH=/app/model/best_model.joblib
MODEL_PATH = Path(os.getenv("MODEL_PATH", "/app/model/best_model.joblib"))
LOGGER_URL = os.getenv("LOGGER_URL", "http://logger:8001/log")

app = FastAPI(title="Predictive Maintenance Inference API", version="1.0.0")

_model = None


def risk_level(p: float) -> str:
    if p >= 0.70:
        return "high"
    if p >= 0.35:
        return "medium"
    return "low"


def send_log(payload: dict):
    """Fire-and-forget logging. Never crash inference if logger is down."""
    try:
        requests.post(LOGGER_URL, json=payload, timeout=2)
    except Exception:
        pass


def _load_model():
    global _model
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model not found at {MODEL_PATH}")
    _model = joblib.load(MODEL_PATH)


@app.on_event("startup")
def startup():
    _load_model()


@app.get("/")
def root():
    return {"message": "Inference API running. Visit /docs for Swagger UI."}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": _model is not None,
        "model_path": str(MODEL_PATH),
        "logger_url": LOGGER_URL,
    }


@app.post("/reload-model")
def reload_model():
    _load_model()
    return {"status": "reloaded", "model_path": str(MODEL_PATH)}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest, background_tasks: BackgroundTasks):
    if _model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    X = build_feature_row(
        Type=req.Type,
        Air_temperature_C=req.Air_temperature_C,
        Process_temperature_C=req.Process_temperature_C,
        Rotational_speed_rpm=req.Rotational_speed_rpm,
        Torque_Nm=req.Torque_Nm,
        Tool_wear_min=req.Tool_wear_min,
    )

    try:
        if hasattr(_model, "predict_proba"):
            p = float(_model.predict_proba(X)[:, 1][0])
        else:
            pred = int(_model.predict(X)[0])
            p = 1.0 if pred == 1 else 0.0
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")

    label = 1 if p >= 0.5 else 0

    result = PredictResponse(
        failure_probability=round(p, 6),
        predicted_label=label,
        risk_level=risk_level(p),
    )

    background_tasks.add_task(
        send_log,
        {
            "request": req.model_dump(),
            "response": result.model_dump(),
            "model_path": str(MODEL_PATH),
        },
    )

    return result
