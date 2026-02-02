from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Literal


class PredictRequest(BaseModel):
    Type: Literal["L", "M", "H"] = Field(..., description="Product type/quality: L/M/H")

    Air_temperature_C: float = Field(..., description="Air temperature in °C")
    Process_temperature_C: float = Field(..., description="Process temperature in °C")

    Rotational_speed_rpm: float = Field(..., ge=0, description="Rotational speed in rpm")
    Torque_Nm: float = Field(..., ge=0, description="Torque in Nm")
    Tool_wear_min: float = Field(..., ge=0, description="Tool wear in minutes")


class PredictResponse(BaseModel):
    failure_probability: float
    predicted_label: int
    risk_level: str
