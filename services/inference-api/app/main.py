from __future__ import annotations

import os
from pathlib import Path

import joblib
import requests
from fastapi import BackgroundTasks, FastAPI, HTTPException

from .feature_build import build_feature_row
from .schemas import PredictRequest, PredictResponse

# ---------------- Environment / config ----------------
# Path to the trained model artifact inside the container.
# In docker-compose / Kubernetes you typically set: MODEL_PATH=/app/model_store/best_model.joblib
MODEL_PATH = Path(os.getenv("MODEL_PATH", "/app/model/best_model.joblib"))

# Logger service endpoint (fire-and-forget)
LOGGER_URL = os.getenv("LOGGER_URL", "http://logger:8001/log")

# FastAPI app metadata (shows in /docs)
app = FastAPI(title="Predictive Maintenance Inference API", version="1.0.0")

# Global cached model (loaded once at startup)
_model = None


def risk_level(p: float) -> str:
    """
    Convert failure probability into a human-friendly risk band.
    Thresholds can be tuned based on business preference.
    """
    if p >= 0.70:
        return "high"
    if p >= 0.35:
        return "medium"
    return "low"


def send_log(payload: dict):
    """
    Fire-and-forget logging call to logger service.

    IMPORTANT:
    - This should never crash inference if logger is down.
    - We keep timeout short to avoid blocking.
    """
    try:
        requests.post(LOGGER_URL, json=payload, timeout=2)
    except Exception:
        pass  # intentionally ignore errors (best-effort logging)


def _load_model():
    """
    Load the trained model pipeline into memory.

    Raises:
        RuntimeError: If the model file does not exist.
    """
    global _model
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model not found at {MODEL_PATH}")
    _model = joblib.load(MODEL_PATH)


@app.on_event("startup")
def startup():
    """Load model when the service starts."""
    _load_model()


@app.get("/")
def root():
    """Basic root endpoint for quick sanity checks."""
    return {"message": "Inference API running. Visit /docs for Swagger UI."}


@app.get("/health")
def health():
    """
    Health endpoint for Kubernetes probes and dashboard status checks.
    Returns model status + key configuration details.
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
    Reload model from disk without restarting the pod/container.

    Useful when trainer writes a new model into the shared PVC and
    you want inference to use it immediately.
    """
    _load_model()
    return {"status": "reloaded", "model_path": str(MODEL_PATH)}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest, background_tasks: BackgroundTasks):
    """
    Predict machine failure risk for a single input row.

    Flow:
      1) Convert request -> DataFrame with training feature names
      2) Run model to get probability/label
      3) Return PredictResponse
      4) Log request+response asynchronously in the background
    """
    # Ensure model is loaded
    if _model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    # Build a 1-row DataFrame matching the training schema
    X = build_feature_row(
        Type=req.Type,
        Air_temperature_C=req.Air_temperature_C,
        Process_temperature_C=req.Process_temperature_C,
        Rotational_speed_rpm=req.Rotational_speed_rpm,
        Torque_Nm=req.Torque_Nm,
        Tool_wear_min=req.Tool_wear_min,
    )

    # Run inference (probabilities preferred, fallback to hard label)
    try:
        if hasattr(_model, "predict_proba"):
            p = float(_model.predict_proba(X)[:, 1][0])  # probability of failure (class 1)
        else:
            pred = int(_model.predict(X)[0])
            p = 1.0 if pred == 1 else 0.0
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")

    # Convert probability -> predicted class label
    label = 1 if p >= 0.5 else 0

    # Build response object (rounded for neat output)
    result = PredictResponse(
        failure_probability=round(p, 6),
        predicted_label=label,
        risk_level=risk_level(p),
    )

    # Log request/response asynchronously (does not block /predict response)
    background_tasks.add_task(
        send_log,
        {
            "request": req.model_dump(),          # request payload as dict
            "response": result.model_dump(),      # response payload as dict
            "model_path": str(MODEL_PATH),        # which model generated it
        },
    )

    return result
