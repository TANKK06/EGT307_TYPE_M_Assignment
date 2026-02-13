from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

# Trainer Dockerfile should set: ENV PYTHONPATH=/app/model/src
# so these imports resolve correctly inside the container.
from config import Config  # type: ignore
from main import main as pipeline_main  # type: ignore

# FastAPI app metadata (shows in /docs)
app = FastAPI(title="Trainer Service", version="1.0.0")

# Directory where the trained model + metrics will be stored permanently (usually a PVC mount)
MODEL_STORE = Path(os.getenv("MODEL_STORE", "/app/model/artifacts"))
MODEL_STORE.mkdir(parents=True, exist_ok=True)

# In docker-compose / Kubernetes, inference is reachable by its service DNS name
INFERENCE_RELOAD_URL = os.getenv("INFERENCE_RELOAD_URL", "http://inference:8000/reload-model")


def _post_no_requests(url: str, timeout: int = 10) -> tuple[bool, str | None]:
    """
    Send a POST request using Python stdlib only (no 'requests' dependency).

    Returns:
        (ok, error_message)
    """
    try:
        req = Request(url, method="POST")
        with urlopen(req, timeout=timeout) as resp:
            # any 2xx response is considered success
            code = getattr(resp, "status", 200)
            if 200 <= code < 300:
                return True, None
            return False, f"HTTP {code}"

    except HTTPError as e:
        # Server responded but with an error status (4xx/5xx)
        return False, f"HTTPError {e.code}: {e.reason}"
    except URLError as e:
        # DNS/connection failure (service down, wrong URL, etc.)
        return False, f"URLError: {e}"
    except Exception as e:
        # Any other unexpected error
        return False, str(e)


@app.get("/health")
def health():
    """Health endpoint for probes and dashboard status checks."""
    return {
        "status": "ok",
        "model_store": str(MODEL_STORE),
        "inference_reload": INFERENCE_RELOAD_URL,
    }


@app.post("/train")
async def train(file: UploadFile = File(...)):
    """
    Train a new model from an uploaded CSV.

    Flow:
      1) Save uploaded CSV to a temporary folder
      2) Run the existing training pipeline (pipeline_main) with CLI-style args
      3) Copy best_model.joblib (+ metrics.json if available) into MODEL_STORE (PVC)
      4) Call inference /reload-model so inference starts using the new model
    """
    # Validate file type
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a .csv file")

    # Use a temp directory so each request is isolated and cleaned up automatically
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)

        # Write uploaded content to disk for the training pipeline to read
        csv_path = td_path / "train.csv"
        csv_path.write_bytes(await file.read())

        # Run your existing training pipeline and write artifacts to a temp output dir
        cfg = Config()
        out_dir = td_path / "artifacts"

        # pipeline_main uses argparse, so we temporarily fake sys.argv
        import sys as _sys
        old_argv = _sys.argv[:]
        _sys.argv = [
            "main.py",
            "--csv", str(csv_path),
            "--out", str(out_dir),
            "--metric", cfg.select_metric,
        ]

        try:
            pipeline_main()  # runs baseline -> tune -> save artifacts
        finally:
            _sys.argv = old_argv  # restore argv even if training fails

        trained_model = out_dir / "best_model.joblib"
        trained_metrics = out_dir / "metrics.json"

        # Ensure training produced a model
        if not trained_model.exists():
            raise HTTPException(
                status_code=500,
                detail="Training finished but best_model.joblib not found",
            )

        # Copy artifacts into the shared model store (mounted volume / PVC)
        shutil.copy2(trained_model, MODEL_STORE / "best_model.joblib")
        if trained_metrics.exists():
            shutil.copy2(trained_metrics, MODEL_STORE / "metrics.json")

    # Ask inference to reload model so new predictions use the latest artifact
    reload_ok, reload_error = _post_no_requests(INFERENCE_RELOAD_URL, timeout=10)

    # Return a structured response for the dashboard
    return JSONResponse(
        {
            "status": "trained",
            "saved_model": str(MODEL_STORE / "best_model.joblib"),
            "saved_metrics": str(MODEL_STORE / "metrics.json"),
            "inference_reload_ok": reload_ok,
            "inference_reload_error": reload_error,
        }
    )
