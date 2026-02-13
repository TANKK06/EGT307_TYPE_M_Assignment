from __future__ import annotations

from pydantic import BaseModel
from typing import Any, Dict, Optional


class LogRequest(BaseModel):
    """
    Schema for the logger service POST /log endpoint.

    Why we use this model:
    - Validates incoming JSON structure (must contain request + response objects)
    - Provides clear API documentation in FastAPI /docs
    - Keeps inference/logger contract consistent

    Fields:
        request:
            The original input payload sent to the inference API /predict endpoint.
            Example keys: Type, Air_temperature_C, Process_temperature_C, ...

        response:
            The prediction output returned by inference.
            Example keys: failure_probability, predicted_label, risk_level

        model_path:
            Optional string indicating which model file/version was used.
            This is useful for tracking model updates over time (retraining).
    """
    request: Dict[str, Any]
    response: Dict[str, Any]
    model_path: Optional[str] = None
