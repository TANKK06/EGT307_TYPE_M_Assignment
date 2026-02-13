# services/inference-api/app/main.py
# FastAPI Inference service:
# - Loads a trained sklearn pipeline from MODEL_PATH at startup
# - Exposes /predict for single prediction requests
# - Exposes /health for K8s readiness/liveness checks
# - Sends prediction logs to the logger service in the background (non-blocking)

from __future__ import annotations

import os
from pathlib import Path

import joblib
import requests
from fastapi import BackgroundTasks, FastAPI, HTTPException

from .feature_build import build_feature_row
from .schemas import PredictRequest, PredictResponse

# -----------------------------
# Environment configuration
# -----------------------------
# MODEL_PATH points to the trained model file inside the container.
# In Docker compose / Kubernetes, we set this so inference knows where to load the model from.
MODEL_PATH = Path(os.getenv("MODEL_PATH", "/app/model/best_model.joblib"))

# Logger endpoint used to store request/response into Postgres (via logger service).
LOGGER_URL = os.getenv("LOGGER_URL", "http://logger:8001/log")

# Create the FastAPI app
app = FastAPI(title="Predictive Maintenance Inference API", version="1.0.0")

# Global variable to hold the loaded model in memory
# (keeps predictions fast; we don't load the model on every request)
_model = None


def risk_level(p: float) -> str:
    """
    Convert failure probability into a human-friendly risk bucket.

    Thresholds are design choices (can be tuned later):
      - p >= 0.70  -> high
      - p >= 0.35  -> medium
      - else       -> low
    """
    if p >= 0.70:
        return "high"
    if p >= 0.35:
        return "medium"
    return "low"


def send_log(payload: dict):
    """
    Fire-and-forget logging to the logger service.

    Important design choice:
    - We NEVER want inference to fail just because logging is down.
    - If logger is unreachable, we silently ignore the error.
    """
    try:
        requests.post(LOGGER_URL, json=payload, timeout=2)
    except Exception:
        # Intentionally ignore logging errors
        pass


def _load_model():
    """
    Load the model from disk into memory.
    Called at startup and by /reload-model.
    """
    global _model

    # Fail fast if the model file does not exist
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model not found at {MODEL_PATH}")

    # joblib.load restores the full sklearn pipeline object
    _model = joblib.load(MODEL_PATH)


@app.on_event("startup")
def startup():
    """
    FastAPI startup hook:
    loads the model once when the container starts.
    """
    _load_model()


@app.get("/")
def root():
    """
    Simple root endpoint.
    Useful for quick sanity checks.
    """
    return {"message": "Inference API running. Visit /docs for Swagger UI."}


@app.get("/health")
def health():
    """
    Health endpoint for K8s probes / monitoring.
    Returns whether the model is loaded and what paths/URLs are configured.
    """
    return {
        "status": "ok",
        "model_loaded": _model is not None,
        "model_path": str(MODEL_PATH),
        "logger_url": LOGGER_URL,
    }


@app.post("/reload-model")
def reload_model():
    """
    Reload the model from disk without restarting the container.

    Use case:
    - Trainer writes a new model file into shared storage (PVC/volume).
    - Call this endpoint so inference loads the newest model.
    """
    _load_model()
    return {"status": "reloaded", "model_path": str(MODEL_PATH)}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest, background_tasks: BackgroundTasks):
    """
    Predict endpoint.

    Steps:
      1) Validate request using PredictRequest schema
      2) Convert request -> training feature schema DataFrame (build_feature_row)
      3) Run model prediction (probability if available)
      4) Convert probability to label + risk level
      5) Send request/response to logger in background
      6) Return PredictResponse
    """
    # Ensure model is loaded before prediction
    if _model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    # Build the feature row using training column names + engineered features
    X = build_feature_row(
        Type=req.Type,
        Air_temperature_C=req.Air_temperature_C,
        Process_temperature_C=req.Process_temperature_C,
        Rotational_speed_rpm=req.Rotational_speed_rpm,
        Torque_Nm=req.Torque_Nm,
        Tool_wear_min=req.Tool_wear_min,
    )

    # Run model prediction safely
    try:
        # Best case: model supports predict_proba -> use probability for PR-AUC and risk levels
        if hasattr(_model, "predict_proba"):
            p = float(_model.predict_proba(X)[:, 1][0])
        else:
            # Fallback: model only supports predict() -> convert label to pseudo-prob
            pred = int(_model.predict(X)[0])
            p = 1.0 if pred == 1 else 0.0
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")

    # Convert probability into a binary label (simple 0.5 threshold)
    label = 1 if p >= 0.5 else 0

    # Build response object (FastAPI will serialize it to JSON)
    result = PredictResponse(
        failure_probability=round(p, 6),
        predicted_label=label,
        risk_level=risk_level(p),
    )

    # Log in the background so we don't slow down /predict response time
    background_tasks.add_task(
        send_log,
        {
            "request": req.model_dump(),
            "response": result.model_dump(),
            "model_path": str(MODEL_PATH),
        },
    )

    return result
