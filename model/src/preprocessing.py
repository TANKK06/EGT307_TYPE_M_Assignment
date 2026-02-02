from __future__ import annotations

import pandas as pd

CANON_AIR_C = "Air temperature [°C]"
CANON_PROC_C = "Process temperature [°C]"


def drop_leakage_and_ids(df: pd.DataFrame, drop_cols: list[str] | tuple[str, ...]) -> pd.DataFrame:
    """Drop ID/leakage columns if present. Returns a NEW df."""
    return df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")


def _to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def normalize_temperature_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Supports:
      A) Air temperature [K], Process temperature [K]  -> converts to °C and renames
      B) Air_temperature_C, Process_temperature_C      -> renames to canonical °C columns
      C) Already canonical °C columns                 -> leaves as is
    """
    df = df.copy()

    # Case B: underscore format already in °C
    if "Air_temperature_C" in df.columns and CANON_AIR_C not in df.columns:
        df[CANON_AIR_C] = _to_num(df["Air_temperature_C"])

    if "Process_temperature_C" in df.columns and CANON_PROC_C not in df.columns:
        df[CANON_PROC_C] = _to_num(df["Process_temperature_C"])

    # Case A: Kelvin columns
    air_k = "Air temperature [K]"
    proc_k = "Process temperature [K]"

    if air_k in df.columns and CANON_AIR_C not in df.columns:
        df[CANON_AIR_C] = _to_num(df[air_k]) - 273.15

    if proc_k in df.columns and CANON_PROC_C not in df.columns:
        df[CANON_PROC_C] = _to_num(df[proc_k]) - 273.15

    # If Kelvin columns exist, optionally drop them to avoid duplicates/confusion
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
    Standard preprocessing used across training & evaluation:
      1) drop leakage/IDs
      2) normalize temperatures into canonical °C columns
    """
    df2 = drop_leakage_and_ids(df, drop_cols=drop_cols)
    df2 = normalize_temperature_columns(df2)
    return df2