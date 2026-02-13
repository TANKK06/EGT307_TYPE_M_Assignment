from __future__ import annotations

import io
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse

# URL of the inference service inside Kubernetes (Service DNS name: inference)
# Can be overridden via env var INFERENCE_URL (useful for local dev/testing).
INFERENCE_URL = os.getenv("INFERENCE_URL", "http://inference:8000/predict")

# Number of parallel requests to inference.
# Higher = faster batch processing, but uses more CPU/network and can overload inference.
MAX_WORKERS = int(os.getenv("BATCH_MAX_WORKERS", "8"))

# Create the FastAPI app for batch prediction
app = FastAPI(title="Batch Prediction Service", version="1.0.0")

# CSV schema required by this service (must match what inference expects)
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
    """
    Simple health endpoint for Kubernetes readiness/liveness probes.
    Returns config info so we can debug what URL/workers are being used.
    """
    return {"status": "ok", "inference_url": INFERENCE_URL, "max_workers": MAX_WORKERS}


def _require_finite_number(x, col: str, row_idx: int) -> float:
    """
    Convert a value to float and ensure it's not NaN/empty.

    We validate row-by-row so we can return helpful error messages like:
      "Row 12: Torque_Nm is not a number"

    Args:
        x: raw value from CSV cell
        col: column name (for error messages)
        row_idx: row index (for error messages)

    Returns:
        float value if valid
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
    Convert a value to int with validation.
    We first validate it as a finite number, then cast to int.
    """
    v = _require_finite_number(x, col, row_idx)
    return int(v)


def _build_payload(row: pd.Series, row_idx: int) -> dict:
    """
    Convert one CSV row into the JSON payload format expected by inference.

    This also validates required fields so we fail early with clear messages
    before sending requests to inference.
    """
    # Type must be non-empty
    t = row.get("Type", "")
    if pd.isna(t) or str(t).strip() == "":
        raise ValueError(f"Row {row_idx}: 'Type' is empty")

    # Build the request body exactly matching inference input schema
    return {
        "Type": str(t).strip(),
        "Air_temperature_C": _require_finite_number(row["Air_temperature_C"], "Air_temperature_C", row_idx),
        "Process_temperature_C": _require_finite_number(row["Process_temperature_C"], "Process_temperature_C", row_idx),
        "Rotational_speed_rpm": _require_finite_number(row["Rotational_speed_rpm"], "Rotational_speed_rpm", row_idx),
        "Torque_Nm": _require_finite_number(row["Torque_Nm"], "Torque_Nm", row_idx),
        "Tool_wear_min": _require_int(row["Tool_wear_min"], "Tool_wear_min", row_idx),
    }


def _predict_one(session: requests.Session, row_idx: int, payload: dict) -> tuple[int, float, int, str]:
    """
    Send ONE row to the inference service and return the prediction.

    Args:
        session: requests.Session (reuses TCP connection for performance)
        row_idx: which row this payload belongs to (so we can write results back correctly)
        payload: JSON input for inference

    Returns:
        (row_idx, failure_probability, predicted_label, risk_level)

    Raises:
        requests.HTTPError if inference returns non-2xx status
        ValueError if inference response is missing expected keys
    """
    # POST to inference with a timeout so we don't hang forever
    r = session.post(INFERENCE_URL, json=payload, timeout=15)
    r.raise_for_status()
    out = r.json()

    # Validate the response format (protects us if inference changes)
    for k in ("failure_probability", "predicted_label", "risk_level"):
        if k not in out:
            raise ValueError(f"Inference response missing key '{k}': {out}")

    return row_idx, float(out["failure_probability"]), int(out["predicted_label"]), str(out["risk_level"])


@app.post("/predict-file")
async def predict_file(file: UploadFile = File(...)):
    """
    Upload a CSV file -> run inference for each row -> return a new CSV with predictions appended.

    Output columns added:
      - failure_probability
      - predicted_label
      - risk_level
    """
    # Basic validation: only accept CSV uploads
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a CSV file")

    # Read file contents into memory (bytes)
    content = await file.read()

    # Parse CSV into a DataFrame
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read CSV: {e}")

    # Validate that required columns exist
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing columns: {missing}")

    # Build payloads first (fail early on bad data rather than partially processing)
    payloads: list[tuple[int, dict]] = []
    try:
        for idx, row in df.iterrows():
            payloads.append((idx, _build_payload(row, idx)))
    except Exception as e:
        # Return clear "Row X" errors to the user
        raise HTTPException(status_code=400, detail=str(e))

    # Pre-allocate result arrays so we can fill them in parallel
    probs = [None] * len(df)
    labels = [None] * len(df)
    risks = [None] * len(df)

    # --------------------
    # Parallel calls to inference
    # --------------------
    # We use a single requests.Session for connection pooling.
    # ThreadPoolExecutor speeds up batch inference by sending requests concurrently.
    try:
        with requests.Session() as session:
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
                # Create a future for each row prediction
                futures = [ex.submit(_predict_one, session, idx, payload) for idx, payload in payloads]

                # as_completed returns futures in the order they finish (not submission order)
                for fut in as_completed(futures):
                    row_idx, p, y, r = fut.result()
                    # Write results back into the correct row index position
                    probs[row_idx] = p
                    labels[row_idx] = y
                    risks[row_idx] = r

    except requests.HTTPError as e:
        # Inference service returned an error response (e.g., 500/400)
        raise HTTPException(status_code=502, detail=f"Inference error: {e}")
    except Exception as e:
        # Any other unexpected failure in batch service
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {e}")

    # Append prediction results to the original DataFrame
    df["failure_probability"] = probs
    df["predicted_label"] = labels
    df["risk_level"] = risks

    # Convert DataFrame back to CSV in-memory (so we can stream it back)
    out_buf = io.StringIO()
    df.to_csv(out_buf, index=False)
    out_buf.seek(0)

    # Return as downloadable CSV file
    return StreamingResponse(
        iter([out_buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=predictions.csv"},
    )
