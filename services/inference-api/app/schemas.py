from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    """
    Request schema for /predict.

    Validates input types and basic constraints before inference runs.
    """
    Type: Literal["L", "M", "H"] = Field(..., description="Product type/quality: L/M/H")

    # Temperatures in Celsius
    Air_temperature_C: float = Field(..., description="Air temperature in °C")
    Process_temperature_C: float = Field(..., description="Process temperature in °C")

    # Basic non-negative constraints for physical quantities
    Rotational_speed_rpm: float = Field(..., ge=0, description="Rotational speed in rpm")
    Torque_Nm: float = Field(..., ge=0, description="Torque in Nm")
    Tool_wear_min: float = Field(..., ge=0, description="Tool wear in minutes")


class PredictResponse(BaseModel):
    """
    Response schema returned by /predict.
    """
    failure_probability: float          # Probability of failure (0.0 - 1.0)
    predicted_label: int                # Predicted class (0 = normal, 1 = failure)
    risk_level: str                     # Human-friendly label (low/medium/high)
