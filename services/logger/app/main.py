# services/logger/app/main.py
# FastAPI Logger service:
# - Receives prediction logs from inference (/log)
# - Writes them into Postgres (predictions table)
# - Exposes /health for readiness/liveness checks

from __future__ import annotations

import time
from fastapi import FastAPI, HTTPException

from .schemas import LogRequest
from .db import insert_prediction_log, get_conn

# Create the FastAPI app for the logger service
app = FastAPI(title="Prediction Logger Service", version="1.0.0")


@app.on_event("startup")
def startup():
    """
    Startup hook: wait for Postgres to be ready before accepting requests.

    Why we need this:
    - In Docker Compose / Kubernetes, services may start in any order.
    - Logger depends on Postgres.
    - We retry for ~30 seconds so logger doesn't crash immediately if DB is still booting.

    Behavior:
    - Try connecting once per second for 30 attempts.
    - If still failing after 30 seconds, raise an error so the container restarts.
    """
    last_err = None
    for _ in range(30):
        try:
            get_conn()  # will connect if not connected yet
            return
        except Exception as e:
            last_err = e
            time.sleep(1)

    # If we reach here, DB never became ready
    raise RuntimeError(f"DB not ready after waiting 30s: {last_err}")


@app.get("/health")
def health():
    """
    Basic health endpoint.
    (K8s liveness/readiness probes can call this.)
    """
    return {"status": "ok"}


@app.post("/log")
def log_prediction(payload: LogRequest):
    """
    Receive a log payload from inference and store it in Postgres.

    Expected payload structure (LogRequest):
      - request: input JSON that was sent to /predict
      - response: output JSON returned by inference
      - model_path: which model file/version was used

    Returns:
      {"status": "saved"} on success

    Errors:
      - 500 if DB insert fails
    """
    try:
        insert_prediction_log(payload.request, payload.response, payload.model_path)
        return {"status": "saved"}
    except Exception as e:
        # Return a 500 to indicate logger failed (inference ignores logger failures)
        raise HTTPException(status_code=500, detail=f"Failed to save log: {e}")
