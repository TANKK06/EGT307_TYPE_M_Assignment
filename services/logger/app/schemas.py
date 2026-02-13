from __future__ import annotations

from typing import Any, Dict, Optional
from pydantic import BaseModel


class LogRequest(BaseModel):
    """
    Request schema for the logger service (/log).

    Fields:
        request: Original inference request payload (input features).
        response: Inference response payload (probability/label/risk).
        model_path: Optional path/name of the model artifact used for prediction.
    """
    request: Dict[str, Any]
    response: Dict[str, Any]
    model_path: Optional[str] = None
