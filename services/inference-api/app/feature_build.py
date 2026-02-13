from __future__ import annotations

import numpy as np
import pandas as pd


def build_feature_row(
    Type: str,
    Air_temperature_C: float,
    Process_temperature_C: float,
    Rotational_speed_rpm: float,
    Torque_Nm: float,
    Tool_wear_min: float,
) -> pd.DataFrame:
    """
    Build a single-row DataFrame for inference that matches the TRAINING feature schema.

    Why this is important:
    - Your trained pipeline expects specific column names (including engineered features).
    - If inference uses different names (or missing engineered columns), prediction can fail
      or give wrong results.

    Inputs:
        Type: Machine type/category (e.g., "L", "M", "H")
        Air_temperature_C: Air temperature in °C
        Process_temperature_C: Process temperature in °C
        Rotational_speed_rpm: Rotational speed in rpm
        Torque_Nm: Torque in Nm
        Tool_wear_min: Tool wear in minutes

    Returns:
        A pandas DataFrame with exactly 1 row and the expected training columns.
    """
    # --------------------
    # Base (raw) feature columns used during training
    # --------------------
    row = {
        # Categorical feature (will be one-hot encoded by the preprocessor)
        "Type": Type,

        # Numeric features (names match your training dataset canonical column names)
        "Air temperature [°C]": float(Air_temperature_C),
        "Process temperature [°C]": float(Process_temperature_C),
        "Rotational speed [rpm]": float(Rotational_speed_rpm),
        "Torque [Nm]": float(Torque_Nm),
        "Tool wear [min]": float(Tool_wear_min),
    }

    # --------------------
    # Engineered features (MUST match training feature_engineering.py)
    # --------------------
    # Temperature difference: Process - Air
    row["Temperature difference [°C]"] = (
        row["Process temperature [°C]"] - row["Air temperature [°C]"]
    )

    # Mechanical Power [W]:
    # Convert RPM -> rad/s using: omega = rpm * 2π / 60
    # Then Power P = torque * omega
    row["Mechanical Power [W]"] = float(
        round((row["Torque [Nm]"] * row["Rotational speed [rpm]"] * 2.0 * np.pi) / 60.0, 4)
    )

    # Return as a 1-row DataFrame (sklearn pipelines expect tabular input)
    return pd.DataFrame([row])
