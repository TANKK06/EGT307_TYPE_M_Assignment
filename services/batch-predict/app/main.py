from __future__ import annotations

import io
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse

INFERENCE_URL = os.getenv("INFERENCE_URL", "http://inference:8000/predict")

# How many parallel requests to inference (tune 4~16 depending on your PC)
MAX_WORKERS = int(os.getenv("BATCH_MAX_WORKERS", "8"))

app = FastAPI(title="Batch Prediction Service", version="1.0.0")

REQUIRED_COLS = [
    "Type",
    "Air_temperature_C",
    "Process_temperature_C",
    "Rotational_speed_rpm",
    "Torque_Nm",
    "Tool_wear_min",
]


@app.get("/health")
def health():
    return {"status": "ok", "inference_url": INFERENCE_URL, "max_workers": MAX_WORKERS}


def _require_finite_number(x, col: str, row_idx: int) -> float:
    """Convert to float and validate not NaN."""
    try:
        v = float(x)
    except Exception:
        raise ValueError(f"Row {row_idx}: '{col}' is not a number: {x!r}")
    if pd.isna(v):
        raise ValueError(f"Row {row_idx}: '{col}' is NaN/empty")
    return v


def _require_int(x, col: str, row_idx: int) -> int:
    v = _require_finite_number(x, col, row_idx)
    return int(v)


def _build_payload(row: pd.Series, row_idx: int) -> dict:
    t = row.get("Type", "")
    if pd.isna(t) or str(t).strip() == "":
        raise ValueError(f"Row {row_idx}: 'Type' is empty")
    return {
        "Type": str(t).strip(),
        "Air_temperature_C": _require_finite_number(row["Air_temperature_C"], "Air_temperature_C", row_idx),
        "Process_temperature_C": _require_finite_number(row["Process_temperature_C"], "Process_temperature_C", row_idx),
        "Rotational_speed_rpm": _require_finite_number(row["Rotational_speed_rpm"], "Rotational_speed_rpm", row_idx),
        "Torque_Nm": _require_finite_number(row["Torque_Nm"], "Torque_Nm", row_idx),
        "Tool_wear_min": _require_int(row["Tool_wear_min"], "Tool_wear_min", row_idx),
    }


def _predict_one(session: requests.Session, row_idx: int, payload: dict) -> tuple[int, float, int, str]:
    """Return (row_idx, prob, label, risk). Raises if inference fails."""
    r = session.post(INFERENCE_URL, json=payload, timeout=15)
    r.raise_for_status()
    out = r.json()

    # Validate expected keys
    for k in ("failure_probability", "predicted_label", "risk_level"):
        if k not in out:
            raise ValueError(f"Inference response missing key '{k}': {out}")

    return row_idx, float(out["failure_probability"]), int(out["predicted_label"]), str(out["risk_level"])


@app.post("/predict-file")
async def predict_file(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a CSV file")

    content = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read CSV: {e}")

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing columns: {missing}")

    # Build payloads first (so we catch CSV issues early)
    payloads: list[tuple[int, dict]] = []
    try:
        for idx, row in df.iterrows():
            payloads.append((idx, _build_payload(row, idx)))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    probs = [None] * len(df)
    labels = [None] * len(df)
    risks = [None] * len(df)

    # Parallel calls to inference
    try:
        with requests.Session() as session:
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
                futures = [ex.submit(_predict_one, session, idx, payload) for idx, payload in payloads]

                for fut in as_completed(futures):
                    row_idx, p, y, r = fut.result()
                    probs[row_idx] = p
                    labels[row_idx] = y
                    risks[row_idx] = r

    except requests.HTTPError as e:
        # Inference returned non-2xx
        raise HTTPException(status_code=502, detail=f"Inference error: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {e}")

    df["failure_probability"] = probs
    df["predicted_label"] = labels
    df["risk_level"] = risks

    out_buf = io.StringIO()
    df.to_csv(out_buf, index=False)
    out_buf.seek(0)

    return StreamingResponse(
        iter([out_buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=predictions.csv"},
    )