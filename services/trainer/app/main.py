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
# so imports like "from config import Config" work.
from config import Config  # type: ignore
from main import main as pipeline_main  # type: ignore

# FastAPI app for training service
app = FastAPI(title="Trainer Service", version="1.0.0")

# -----------------------------
# Where trained artifacts are stored
# -----------------------------
# This should be a shared volume/PVC so inference can read the updated model.
# Default matches your project folder: /app/model/artifacts
MODEL_STORE = Path(os.getenv("MODEL_STORE", "/app/model/artifacts"))
MODEL_STORE.mkdir(parents=True, exist_ok=True)

# In docker-compose / Kubernetes, inference is reachable by service name "inference"
# After training, we call this endpoint so inference reloads the new model.
INFERENCE_RELOAD_URL = os.getenv("INFERENCE_RELOAD_URL", "http://inference:8000/reload-model")


def _post_no_requests(url: str, timeout: int = 10) -> tuple[bool, str | None]:
    """
    POST using Python stdlib only (no 'requests' dependency).

    Why:
    - Keeps trainer requirements smaller
    - We only need a simple POST to trigger inference reload

    Returns:
        (ok, error_message)
    """
    try:
        req = Request(url, method="POST")
        with urlopen(req, timeout=timeout) as resp:
            # If status exists and is 2xx, treat as success
            code = getattr(resp, "status", 200)
            if 200 <= code < 300:
                return True, None
            return False, f"HTTP {code}"

    except HTTPError as e:
        # Server returned a valid HTTP response but with an error status code (4xx/5xx)
        return False, f"HTTPError {e.code}: {e.reason}"

    except URLError as e:
        # Network/DNS/connection issue (e.g., inference service down)
        return False, f"URLError: {e}"

    except Exception as e:
        # Any other unexpected error
        return False, str(e)


@app.get("/health")
def health():
    """
    Health endpoint for readiness/liveness checks.
    Includes model store path and reload URL for debugging.
    """
    return {"status": "ok", "model_store": str(MODEL_STORE), "inference_reload": INFERENCE_RELOAD_URL}


@app.post("/train")
async def train(file: UploadFile = File(...)):
    """
    Upload a training CSV -> run training pipeline -> save artifacts -> trigger inference reload.

    Steps:
      1) Validate file is CSV
      2) Save uploaded file to a temp directory
      3) Run existing pipeline_main() with CLI args overridden (csv, out_dir, metric)
      4) Copy best_model.joblib (and metrics.json if present) into MODEL_STORE (shared volume)
      5) POST to inference /reload-model so inference loads the new model

    Returns:
      JSON containing saved paths + reload status.
    """
    # Basic validation: only accept .csv uploads
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a .csv file")

    # Use a temporary directory so we don't pollute the container filesystem
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)

        # Save uploaded CSV to disk so training pipeline can read it
        csv_path = td_path / "train.csv"
        csv_path.write_bytes(await file.read())

        # Prepare to run your existing training pipeline in-process
        cfg = Config()
        out_dir = td_path / "artifacts"  # training outputs will be written here

        # ---------------------------------------------------------
        # Trick: pipeline_main uses argparse, so we override sys.argv
        # This lets us reuse the same training code without rewriting it.
        # ---------------------------------------------------------
        import sys as _sys
        old_argv = _sys.argv[:]
        _sys.argv = [
            "main.py",
            "--csv", str(csv_path),
            "--out", str(out_dir),
            "--metric", cfg.select_metric,
        ]

        try:
            # Run training + tuning pipeline
            pipeline_main()
        finally:
            # Always restore argv even if training fails
            _sys.argv = old_argv

        # Paths produced by save_artifacts()
        trained_model = out_dir / "best_model.joblib"
        trained_metrics = out_dir / "metrics.json"

        # If model not found, training pipeline failed silently or misconfigured output
        if not trained_model.exists():
            raise HTTPException(status_code=500, detail="Training finished but best_model.joblib not found")

        # ---------------------------------------------------------
        # Copy outputs into the shared model store (mounted volume/PVC)
        # Inference reads from this location.
        # ---------------------------------------------------------
        shutil.copy2(trained_model, MODEL_STORE / "best_model.joblib")
        if trained_metrics.exists():
            shutil.copy2(trained_metrics, MODEL_STORE / "metrics.json")

    # After leaving the temp dir, files there are deleted, but MODEL_STORE keeps artifacts.

    # Trigger inference to reload the newly saved model (stdlib POST)
    reload_ok, reload_error = _post_no_requests(INFERENCE_RELOAD_URL, timeout=10)

    # Return training result summary
    return JSONResponse(
        {
            "status": "trained",
            "saved_model": str(MODEL_STORE / "best_model.joblib"),
            "saved_metrics": str(MODEL_STORE / "metrics.json"),
            "inference_reload_ok": reload_ok,
            "inference_reload_error": reload_error,
        }
    )
