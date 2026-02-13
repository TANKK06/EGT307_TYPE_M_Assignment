from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Literal


class PredictRequest(BaseModel):
    """
    Schema for the /predict request body.

    Why we use Pydantic models:
    - Automatic validation (types, required fields, ranges)
    - Clear API documentation (FastAPI uses these models for Swagger docs)
    - Prevents bad inputs from reaching the ML model

    Field naming note:
    - We keep the request fields in a simple format (e.g., Air_temperature_C)
      and later convert them into the exact training column names.
    """

    # Restrict Type to only valid categories to avoid unknown labels at inference time
    Type: Literal["L", "M", "H"] = Field(..., description="Product type/quality: L/M/H")

    # Temperatures (in °C)
    Air_temperature_C: float = Field(..., description="Air temperature in °C")
    Process_temperature_C: float = Field(..., description="Process temperature in °C")

    # Machine sensor readings (non-negative constraints using ge=0)
    Rotational_speed_rpm: float = Field(..., ge=0, description="Rotational speed in rpm")
    Torque_Nm: float = Field(..., ge=0, description="Torque in Nm")
    Tool_wear_min: float = Field(..., ge=0, description="Tool wear in minutes")


class PredictResponse(BaseModel):
    """
    Schema for the /predict response body.

    Fields:
        failure_probability: probability of failure (0..1)
        predicted_label: final predicted class (0 = normal, 1 = failure)
        risk_level: human-friendly category (e.g., low/medium/high)
    """
    failure_probability: float
    predicted_label: int
    risk_level: str
