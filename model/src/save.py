from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple
import json
import joblib


def save_finetuned_model(out_dir: str | Path, model) -> Path:
    """
    Backwards-compatible helper: save ONLY the fine-tuned model pipeline.

    Args:
        out_dir: Output directory to write the model into.
        model: Trained model/pipeline object (e.g., sklearn Pipeline).

    Returns:
        Path to the saved model file (best_model.joblib).
    """
    out_dir = Path(out_dir)                   # accept str or Path
    out_dir.mkdir(parents=True, exist_ok=True)  # create folder if missing

    path = out_dir / "best_model.joblib"      # standard filename used across services
    joblib.dump(model, path)                  # serialize model to disk
    return path


def save_artifacts(out_dir: str | Path, model, report: Dict[str, Any]) -> Tuple[Path, Path]:
    """
    Recommended: save BOTH model + metrics/report for deployment + documentation.

    Writes:
      - best_model.joblib  (trained model/pipeline)
      - metrics.json       (evaluation + config details for your report)

    Args:
        out_dir: Output directory for artifacts.
        model: Trained model/pipeline object to save.
        report: Dict containing metrics, parameters, and run metadata.

    Returns:
        (model_path, metrics_path)
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = out_dir / "best_model.joblib"   # model artifact used by inference service
    metrics_path = out_dir / "metrics.json"      # JSON metrics for reporting

    # Save model
    joblib.dump(model, model_path)

    # Save report/metrics as readable JSON (indent makes it easy to inspect)
    metrics_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    return model_path, metrics_path
