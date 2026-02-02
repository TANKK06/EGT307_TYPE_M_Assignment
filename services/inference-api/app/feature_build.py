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
    Build a single-row DataFrame with the SAME columns used in training.
    """
    row = {
        "Type": Type,
        "Air temperature [°C]": float(Air_temperature_C),
        "Process temperature [°C]": float(Process_temperature_C),
        "Rotational speed [rpm]": float(Rotational_speed_rpm),
        "Torque [Nm]": float(Torque_Nm),
        "Tool wear [min]": float(Tool_wear_min),
    }

    # Feature engineering (must match your training)
    row["Temperature difference [°C]"] = row["Process temperature [°C]"] - row["Air temperature [°C]"]
    row["Mechanical Power [W]"] = float(
        round((row["Torque [Nm]"] * row["Rotational speed [rpm]"] * 2.0 * np.pi) / 60.0, 4)
    )

    return pd.DataFrame([row])
