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
from config import Config  # type: ignore
from main import main as pipeline_main  # type: ignore

app = FastAPI(title="Trainer Service", version="1.0.0")

MODEL_STORE = Path(os.getenv("MODEL_STORE", "/app/model/artifacts"))
MODEL_STORE.mkdir(parents=True, exist_ok=True)

# In docker-compose, inference is reachable by service name "inference"
INFERENCE_RELOAD_URL = os.getenv("INFERENCE_RELOAD_URL", "http://inference:8000/reload-model")


def _post_no_requests(url: str, timeout: int = 10) -> tuple[bool, str | None]:
    """POST using stdlib only (no 'requests' dependency)."""
    try:
        req = Request(url, method="POST")
        with urlopen(req, timeout=timeout) as resp:
            # any 2xx is ok
            code = getattr(resp, "status", 200)
            if 200 <= code < 300:
                return True, None
            return False, f"HTTP {code}"
    except HTTPError as e:
        return False, f"HTTPError {e.code}: {e.reason}"
    except URLError as e:
        return False, f"URLError: {e}"
    except Exception as e:
        return False, str(e)


@app.get("/health")
def health():
    return {"status": "ok", "model_store": str(MODEL_STORE), "inference_reload": INFERENCE_RELOAD_URL}


@app.post("/train")
async def train(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a .csv file")

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        csv_path = td_path / "train.csv"
        csv_path.write_bytes(await file.read())

        # Run your existing training pipeline
        cfg = Config()
        out_dir = td_path / "artifacts"

        import sys as _sys
        old_argv = _sys.argv[:]
        _sys.argv = [
            "main.py",
            "--csv", str(csv_path),
            "--out", str(out_dir),
            "--metric", cfg.select_metric,
        ]

        try:
            pipeline_main()
        finally:
            _sys.argv = old_argv

        trained_model = out_dir / "best_model.joblib"
        trained_metrics = out_dir / "metrics.json"

        if not trained_model.exists():
            raise HTTPException(status_code=500, detail="Training finished but best_model.joblib not found")

        # Copy outputs into the shared model store (mounted volume)
        shutil.copy2(trained_model, MODEL_STORE / "best_model.joblib")
        if trained_metrics.exists():
            shutil.copy2(trained_metrics, MODEL_STORE / "metrics.json")

    # Ask inference to reload the model (stdlib POST)
    reload_ok, reload_error = _post_no_requests(INFERENCE_RELOAD_URL, timeout=10)

    return JSONResponse(
        {
            "status": "trained",
            "saved_model": str(MODEL_STORE / "best_model.joblib"),
            "saved_metrics": str(MODEL_STORE / "metrics.json"),
            "inference_reload_ok": reload_ok,
            "inference_reload_error": reload_error,
        }
    )