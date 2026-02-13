from __future__ import annotations

import time
from fastapi import FastAPI, HTTPException

from .schemas import LogRequest
from .db import insert_prediction_log, get_conn

# FastAPI app metadata (shows in /docs)
app = FastAPI(title="Prediction Logger Service", version="1.0.0")


@app.on_event("startup")
def startup():
    """
    Startup hook: wait for Postgres to be reachable before serving requests.

    Why:
    - In Kubernetes/Docker, logger may start before the DB is ready.
    - We retry for ~30 seconds (30 tries * 1s sleep).
    """
    last_err = None
    for _ in range(30):
        try:
            get_conn()            # attempts DB connection (and caches it globally)
            return                # success -> start the API
        except Exception as e:
            last_err = e          # remember the last error for debugging
            time.sleep(1)         # wait 1s then retry
    raise RuntimeError(f"DB not ready after waiting 30s: {last_err}")


@app.get("/health")
def health():
    """Health endpoint for probes and dashboard checks."""
    return {"status": "ok"}


@app.post("/log")
def log_prediction(payload: LogRequest):
    """
    Store a single prediction log record.

    Expects:
      payload.request   -> original inference request JSON
      payload.response  -> inference response JSON
      payload.model_path-> which model generated the response
    """
    try:
        insert_prediction_log(payload.request, payload.response, payload.model_path)
        return {"status": "saved"}
    except Exception as e:
        # Return 500 if DB insert fails
        raise HTTPException(status_code=500, detail=f"Failed to save log: {e}")
