from __future__ import annotations

import numpy as np
import pandas as pd


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create additional engineered features for predictive maintenance.

    Adds (when required input columns exist):
      - Temperature difference [°C] = Process temperature - Air temperature
      - Mechanical Power [W] = Torque * AngularVelocity
        where AngularVelocity = 2π * RPM / 60  (rad/s)

    Returns:
        A NEW DataFrame (input is not mutated).
    """
    df = df.copy()  # avoid modifying the caller's DataFrame

    # Expected raw column names from the dataset
    air_c = "Air temperature [°C]"
    proc_c = "Process temperature [°C]"
    torque = "Torque [Nm]"
    rpm = "Rotational speed [rpm]"

    # Feature 1: temperature difference
    if air_c in df.columns and proc_c in df.columns:
        # Convert to numeric safely (invalid values -> NaN)
        a = pd.to_numeric(df[air_c], errors="coerce")
        p = pd.to_numeric(df[proc_c], errors="coerce")
        df["Temperature difference [°C]"] = p - a

    # Feature 2: mechanical power (Watts)
    if torque in df.columns and rpm in df.columns:
        # Convert to numeric safely (invalid values -> NaN)
        tq = pd.to_numeric(df[torque], errors="coerce")  # torque in Newton-meters
        r = pd.to_numeric(df[rpm], errors="coerce")      # rotational speed in RPM

        # Power formula: P = τ * ω, and ω = 2π * RPM / 60
        power = (tq * r * 2.0 * np.pi) / 60.0

        # Round for cleaner logs/exports (keeps numeric type)
        df["Mechanical Power [W]"] = np.round(power, 4)

    return df
