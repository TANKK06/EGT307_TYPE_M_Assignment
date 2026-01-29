from __future__ import annotations

import numpy as np
import pandas as pd


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds:
      - Temperature difference [°C]
      - Mechanical Power [W]
    Returns a NEW df (does not mutate input).
    """
    df = df.copy()

    air_c = "Air temperature [°C]"
    proc_c = "Process temperature [°C]"
    torque = "Torque [Nm]"
    rpm = "Rotational speed [rpm]"

    if air_c in df.columns and proc_c in df.columns:
        a = pd.to_numeric(df[air_c], errors="coerce")
        p = pd.to_numeric(df[proc_c], errors="coerce")
        df["Temperature difference [°C]"] = p - a

    if torque in df.columns and rpm in df.columns:
        tq = pd.to_numeric(df[torque], errors="coerce")
        r = pd.to_numeric(df[rpm], errors="coerce")
        power = (tq * r * 2.0 * np.pi) / 60.0
        df["Mechanical Power [W]"] = np.round(power, 4)

    return df
