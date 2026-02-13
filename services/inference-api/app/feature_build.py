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
    Build a single-row DataFrame that is compatible with models trained on
    either underscore columns or bracket/unit columns.

    This avoids breaking inference when retraining data uses different headers.
    """
    air = float(Air_temperature_C)
    proc = float(Process_temperature_C)
    rpm = float(Rotational_speed_rpm)
    tq = float(Torque_Nm)
    wear = float(Tool_wear_min)

    row = {
        "Type": str(Type),

        # --- Underscore style (common in your API payloads / batch CSV) ---
        "Air_temperature_C": air,
        "Process_temperature_C": proc,
        "Rotational_speed_rpm": rpm,
        "Torque_Nm": tq,
        "Tool_wear_min": wear,

        # --- Bracket/unit style (common in original dataset / some trained pipelines) ---
        "Air temperature [°C]": air,
        "Process temperature [°C]": proc,
        "Rotational speed [rpm]": rpm,
        "Torque [Nm]": tq,
        "Tool wear [min]": wear,

        # --- Engineered features (your training adds these) ---
        "Temperature difference [°C]": proc - air,
        "Mechanical Power [W]": float(round((tq * rpm * 2.0 * np.pi) / 60.0, 4)),
    }

    return pd.DataFrame([row])
