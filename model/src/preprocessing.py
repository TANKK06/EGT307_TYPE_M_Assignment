from __future__ import annotations

import pandas as pd

# Canonical column names used throughout the pipeline (feature engineering expects these)
CANON_AIR_C = "Air temperature [°C]"
CANON_PROC_C = "Process temperature [°C]"


def drop_leakage_and_ids(df: pd.DataFrame, drop_cols: list[str] | tuple[str, ...]) -> pd.DataFrame:
    """
    Drop ID / leakage columns if they exist.

    Args:
        df: Input DataFrame.
        drop_cols: Column names to remove (only dropped if present).

    Returns:
        A NEW DataFrame with specified columns removed (input is not mutated).
    """
    return df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")


def _to_num(s: pd.Series) -> pd.Series:
    """Convert a series to numeric safely (invalid values become NaN)."""
    return pd.to_numeric(s, errors="coerce")


def normalize_temperature_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize temperature columns into canonical Celsius names.

    Supported input formats:
      A) Kelvin format:
         - "Air temperature [K]", "Process temperature [K]"
         -> converts to °C and stores in canonical columns

      B) Underscore Celsius format:
         - "Air_temperature_C", "Process_temperature_C"
         -> copies/renames into canonical columns (assumes already °C)

      C) Already canonical Celsius format:
         - "Air temperature [°C]", "Process temperature [°C]"
         -> leave unchanged

    Returns:
        A NEW DataFrame with consistent temperature columns for downstream steps.
    """
    df = df.copy()  # avoid mutating input

    # -------- Case B: underscore format already in °C --------
    # Copy values into canonical columns only if canonical doesn't already exist
    if "Air_temperature_C" in df.columns and CANON_AIR_C not in df.columns:
        df[CANON_AIR_C] = _to_num(df["Air_temperature_C"])

    if "Process_temperature_C" in df.columns and CANON_PROC_C not in df.columns:
        df[CANON_PROC_C] = _to_num(df["Process_temperature_C"])

    # -------- Case A: Kelvin columns --------
    air_k = "Air temperature [K]"
    proc_k = "Process temperature [K]"

    # Convert K -> °C using °C = K - 273.15
    if air_k in df.columns and CANON_AIR_C not in df.columns:
        df[CANON_AIR_C] = _to_num(df[air_k]) - 273.15

    if proc_k in df.columns and CANON_PROC_C not in df.columns:
        df[CANON_PROC_C] = _to_num(df[proc_k]) - 273.15

    # If Kelvin columns exist AND canonical columns exist, drop Kelvin to avoid confusion/duplication
    drop_candidates: list[str] = []
    if air_k in df.columns and CANON_AIR_C in df.columns:
        drop_candidates.append(air_k)
    if proc_k in df.columns and CANON_PROC_C in df.columns:
        drop_candidates.append(proc_k)

    if drop_candidates:
        df = df.drop(columns=drop_candidates, errors="ignore")

    return df


def prepare_dataframe(df: pd.DataFrame, drop_cols: list[str] | tuple[str, ...]) -> pd.DataFrame:
    """
    Standard preprocessing used across training & evaluation.

    Steps:
      1) Drop ID/leakage columns (if present)
      2) Normalize temperature columns into canonical °C columns

    Returns:
        Cleaned DataFrame ready for feature engineering and modeling.
    """
    df2 = drop_leakage_and_ids(df, drop_cols=drop_cols)
    df2 = normalize_temperature_columns(df2)
    return df2
