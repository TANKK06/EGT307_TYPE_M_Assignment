from __future__ import annotations

import pandas as pd

# Canonical (standard) column names we want the rest of the pipeline to use.
# This prevents bugs when different datasets use slightly different naming/units.
CANON_AIR_C = "Air temperature [°C]"
CANON_PROC_C = "Process temperature [°C]"


def drop_leakage_and_ids(df: pd.DataFrame, drop_cols: list[str] | tuple[str, ...]) -> pd.DataFrame:
    """
    Drop ID / leakage columns if they exist in the dataset.

    Why we do this:
    - ID columns (e.g., Product ID) usually do not help prediction.
    - They can cause "data leakage" (model learns patterns that won't generalize).
    - We only drop columns that are present, so the function works across datasets.

    Returns:
        A NEW DataFrame (does not modify input).
    """
    return df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")


def _to_num(s: pd.Series) -> pd.Series:
    """
    Convert a pandas Series to numeric safely.
    Non-numeric values become NaN instead of crashing.
    """
    return pd.to_numeric(s, errors="coerce")


def normalize_temperature_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize temperature columns so the pipeline always sees the same names + units (°C).

    This function supports multiple dataset formats:

      A) Kelvin input:
         - "Air temperature [K]" and "Process temperature [K]"
         - Converts to °C using: °C = K - 273.15
         - Stores result in canonical °C columns

      B) Underscore input already in °C:
         - "Air_temperature_C" and "Process_temperature_C"
         - Copies/renames them into canonical °C columns

      C) Already canonical:
         - "Air temperature [°C]" and "Process temperature [°C]"
         - Leaves as-is

    Returns:
        A NEW DataFrame with canonical temperature columns available when possible.
    """
    df = df.copy()

    # --------------------
    # Case B: underscore-format temperature columns (already in °C)
    # --------------------
    # Only create canonical column if it doesn't exist already (avoid overwriting).
    if "Air_temperature_C" in df.columns and CANON_AIR_C not in df.columns:
        df[CANON_AIR_C] = _to_num(df["Air_temperature_C"])

    if "Process_temperature_C" in df.columns and CANON_PROC_C not in df.columns:
        df[CANON_PROC_C] = _to_num(df["Process_temperature_C"])

    # --------------------
    # Case A: Kelvin columns (convert K -> °C)
    # --------------------
    air_k = "Air temperature [K]"
    proc_k = "Process temperature [K]"

    if air_k in df.columns and CANON_AIR_C not in df.columns:
        df[CANON_AIR_C] = _to_num(df[air_k]) - 273.15

    if proc_k in df.columns and CANON_PROC_C not in df.columns:
        df[CANON_PROC_C] = _to_num(df[proc_k]) - 273.15

    # --------------------
    # Cleanup: if Kelvin columns exist and we've created canonical °C columns,
    # drop Kelvin columns to avoid duplicates/confusion later.
    # --------------------
    drop_candidates = []
    if air_k in df.columns and CANON_AIR_C in df.columns:
        drop_candidates.append(air_k)
    if proc_k in df.columns and CANON_PROC_C in df.columns:
        drop_candidates.append(proc_k)

    if drop_candidates:
        df = df.drop(columns=drop_candidates, errors="ignore")

    return df


def prepare_dataframe(df: pd.DataFrame, drop_cols: list[str] | tuple[str, ...]) -> pd.DataFrame:
    """
    Standard preprocessing shared by training and evaluation.

    Steps:
      1) Drop leakage/ID columns (safe across datasets)
      2) Normalize temperature columns into canonical °C columns

    Returns:
        A cleaned DataFrame ready for feature engineering + model pipeline.
    """
    df2 = drop_leakage_and_ids(df, drop_cols=drop_cols)
    df2 = normalize_temperature_columns(df2)
    return df2
