from __future__ import annotations

from pydantic import BaseModel
from typing import Any, Dict, Optional


class LogRequest(BaseModel):
    request: Dict[str, Any]
    response: Dict[str, Any]
    model_path: Optional[str] = None
