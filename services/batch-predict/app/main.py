from __future__ import annotations

import io
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse

# URL of the inference service (ClusterIP DNS inside Kubernetes)
INFERENCE_URL = os.getenv("INFERENCE_URL", "http://inference:8000/predict")

# Number of parallel requests to the inference service (tune based on CPU/network)
MAX_WORKERS = int(os.getenv("BATCH_MAX_WORKERS", "8"))

# FastAPI app metadata (shows up in /docs)
app = FastAPI(title="Batch Prediction Service", version="1.0.0")

# Columns that MUST exist in the uploaded CSV file
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
    """Health endpoint for Kubernetes probes and quick debugging."""
    return {"status": "ok", "inference_url": INFERENCE_URL, "max_workers": MAX_WORKERS}


def _require_finite_number(x, col: str, row_idx: int) -> float:
    """
    Convert a cell to float and ensure it's a valid number (not NaN/empty).

    Raises:
        ValueError: If value cannot be converted or is NaN.
    """
    try:
        v = float(x)
    except Exception:
        raise ValueError(f"Row {row_idx}: '{col}' is not a number: {x!r}")
    if pd.isna(v):
        raise ValueError(f"Row {row_idx}: '{col}' is NaN/empty")
    return v


def _require_int(x, col: str, row_idx: int) -> int:
    """
    Convert a cell to int safely (via float conversion first).

    Useful for columns that should be integer-like (e.g., tool wear minutes).
    """
    v = _require_finite_number(x, col, row_idx)
    return int(v)


def _build_payload(row: pd.Series, row_idx: int) -> dict:
    """
    Convert one CSV row into the JSON payload expected by the inference API.

    Raises:
        ValueError: If required fields are missing/invalid.
    """
    t = row.get("Type", "")
    if pd.isna(t) or str(t).strip() == "":
        raise ValueError(f"Row {row_idx}: 'Type' is empty")

    return {
        # Categorical feature
        "Type": str(t).strip(),

        # Numeric features (validated + converted)
        "Air_temperature_C": _require_finite_number(row["Air_temperature_C"], "Air_temperature_C", row_idx),
        "Process_temperature_C": _require_finite_number(row["Process_temperature_C"], "Process_temperature_C", row_idx),
        "Rotational_speed_rpm": _require_finite_number(row["Rotational_speed_rpm"], "Rotational_speed_rpm", row_idx),
        "Torque_Nm": _require_finite_number(row["Torque_Nm"], "Torque_Nm", row_idx),

        # Integer-like feature
        "Tool_wear_min": _require_int(row["Tool_wear_min"], "Tool_wear_min", row_idx),
    }


def _predict_one(session: requests.Session, row_idx: int, payload: dict) -> tuple[int, float, int, str]:
    """
    Send one request to inference and return results for a single row.

    Returns:
        (row_idx, failure_probability, predicted_label, risk_level)

    Raises:
        requests.HTTPError: If inference returns non-2xx response.
        ValueError: If inference response is missing expected keys.
    """
    r = session.post(INFERENCE_URL, json=payload, timeout=15)
    r.raise_for_status()
    out = r.json()

    # Validate expected response keys from inference service
    for k in ("failure_probability", "predicted_label", "risk_level"):
        if k not in out:
            raise ValueError(f"Inference response missing key '{k}': {out}")

    return row_idx, float(out["failure_probability"]), int(out["predicted_label"]), str(out["risk_level"])


@app.post("/predict-file")
async def predict_file(file: UploadFile = File(...)):
    """
    Upload a CSV -> call inference in parallel -> return a new CSV with predictions appended.

    Output columns added:
      - failure_probability
      - predicted_label
      - risk_level
    """
    # Basic file type validation
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a CSV file")

    # Read file bytes from upload
    content = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(content))  # parse CSV into DataFrame
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read CSV: {e}")

    # Validate schema: required columns must be present
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing columns: {missing}")

    # Build payloads first so we fail early on bad CSV content
    payloads: list[tuple[int, dict]] = []
    try:
        for idx, row in df.iterrows():
            payloads.append((idx, _build_payload(row, idx)))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Pre-allocate result arrays (same length as df)
    probs = [None] * len(df)
    labels = [None] * len(df)
    risks = [None] * len(df)

    # Parallel calls to inference (one HTTP request per row)
    try:
        with requests.Session() as session:
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
                futures = [ex.submit(_predict_one, session, idx, payload) for idx, payload in payloads]

                # as_completed yields results as soon as each finishes
                for fut in as_completed(futures):
                    row_idx, p, y, r = fut.result()
                    probs[row_idx] = p
                    labels[row_idx] = y
                    risks[row_idx] = r

    except requests.HTTPError as e:
        # Inference returned non-2xx (treat as upstream/bad gateway)
        raise HTTPException(status_code=502, detail=f"Inference error: {e}")
    except Exception as e:
        # Any other unexpected failure
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {e}")

    # Append predictions to the original DataFrame
    df["failure_probability"] = probs
    df["predicted_label"] = labels
    df["risk_level"] = risks

    # Convert result to CSV in-memory
    out_buf = io.StringIO()
    df.to_csv(out_buf, index=False)
    out_buf.seek(0)

    # Stream CSV back to the client as a downloadable file
    return StreamingResponse(
        iter([out_buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=predictions.csv"},
    )
