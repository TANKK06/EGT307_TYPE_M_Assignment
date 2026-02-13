from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple
import json
import joblib


def save_finetuned_model(out_dir: str | Path, model) -> Path:
    """
    Save ONLY the fine-tuned model pipeline (legacy/backwards-compatible function).

    Why this exists:
    - Some older parts of the project (or teammates' code) may expect only a single output file.
    - This function keeps that behavior stable while the newer save_artifacts() saves more.

    Args:
        out_dir: Folder where the model file should be saved.
        model: Trained sklearn Pipeline (preprocessor + SMOTE + classifier).

    Returns:
        Path to the saved model file: best_model.joblib
    """
    # Ensure out_dir is a Path object and exists
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save the trained pipeline using joblib (recommended for sklearn objects)
    path = out_dir / "best_model.joblib"
    joblib.dump(model, path)

    return path


def save_artifacts(out_dir: str | Path, model, report: Dict[str, Any]) -> Tuple[Path, Path]:
    """
    Save ALL artifacts needed for deployment + reporting.

    What gets written:
      1) best_model.joblib  -> the trained model pipeline used by inference service
      2) metrics.json       -> a detailed report (baseline results, tuned metrics, settings)

    Args:
        out_dir: Folder where artifacts will be saved.
        model: Trained sklearn Pipeline.
        report: Dictionary containing metrics + metadata (will be saved as JSON).

    Returns:
        (model_path, metrics_path)
    """
    # Ensure output folder exists
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # File locations
    model_path = out_dir / "best_model.joblib"
    metrics_path = out_dir / "metrics.json"

    # Save model pipeline (binary)
    joblib.dump(model, model_path)

    # Save report/metrics (human-readable JSON)
    # indent=2 makes it easier for teammates to read / debug
    metrics_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    return model_path, metrics_path
