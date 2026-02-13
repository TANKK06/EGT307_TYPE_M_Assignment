from __future__ import annotations

import numpy as np
import pandas as pd


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add extra (engineered) features to improve model performance.

    Features added:
      1) Temperature difference [°C] = Process temperature - Air temperature
         - Can capture overheating / abnormal operating conditions.

      2) Mechanical Power [W] = Torque * Angular velocity
         - Torque is in Nm, rotational speed is in RPM.
         - Convert RPM -> rad/s using: omega = rpm * 2π / 60
         - Power formula: P = τ * ω

    Important:
    - This function returns a NEW DataFrame (it does NOT modify the original input).
    """
    # Make a copy so we don't accidentally change the caller's DataFrame
    df = df.copy()

    # Column names from the predictive maintenance dataset
    air_c = "Air temperature [°C]"
    proc_c = "Process temperature [°C]"
    torque = "Torque [Nm]"
    rpm = "Rotational speed [rpm]"

    # --------------------
    # Feature 1: Temperature difference
    # --------------------
    # Only create the feature if the required columns exist in the input DataFrame
    if air_c in df.columns and proc_c in df.columns:
        # Convert to numeric safely (invalid values become NaN instead of crashing)
        a = pd.to_numeric(df[air_c], errors="coerce")
        p = pd.to_numeric(df[proc_c], errors="coerce")

        # Process temp - Air temp (positive means process is hotter than environment)
        df["Temperature difference [°C]"] = p - a

    # --------------------
    # Feature 2: Mechanical Power
    # --------------------
    # Only create the feature if the required columns exist
    if torque in df.columns and rpm in df.columns:
        # Convert to numeric safely
        tq = pd.to_numeric(df[torque], errors="coerce")
        r = pd.to_numeric(df[rpm], errors="coerce")

        # Convert RPM to rad/s and compute power in Watts:
        # P = torque(Nm) * rpm * 2π / 60
        power = (tq * r * 2.0 * np.pi) / 60.0

        # Round for cleaner logs/CSV output (not required, just nicer)
        df["Mechanical Power [W]"] = np.round(power, 4)

    return df
