from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple
import json
import joblib


def save_finetuned_model(out_dir: str | Path, model) -> Path:
    """
    Backwards-compatible: saves ONLY the fine-tuned model pipeline.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    path = out_dir / "best_model.joblib"
    joblib.dump(model, path)
    return path


def save_artifacts(out_dir: str | Path, model, report: Dict[str, Any]) -> Tuple[Path, Path]:
    """
    Recommended: save model + metrics for your report and deployment.

    Writes:
      - best_model.joblib
      - metrics.json
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = out_dir / "best_model.joblib"
    metrics_path = out_dir / "metrics.json"

    joblib.dump(model, model_path)
    metrics_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    return model_path, metrics_path
