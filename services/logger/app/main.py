# services/logger/app/main.py
from __future__ import annotations

import time
from fastapi import FastAPI, HTTPException

from .schemas import LogRequest
from .db import insert_prediction_log, get_conn

app = FastAPI(title="Prediction Logger Service", version="1.0.0")


@app.on_event("startup")
def startup():
    # Wait for Postgres to be ready (retry up to ~30s)
    last_err = None
    for _ in range(30):
        try:
            get_conn()
            return
        except Exception as e:
            last_err = e
            time.sleep(1)
    raise RuntimeError(f"DB not ready after waiting 30s: {last_err}")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/log")
def log_prediction(payload: LogRequest):
    try:
        insert_prediction_log(payload.request, payload.response, payload.model_path)
        return {"status": "saved"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save log: {e}")
