from __future__ import annotations

import pandas as pd


def drop_leakage_and_ids(df: pd.DataFrame, drop_cols: list[str] | tuple[str, ...]) -> pd.DataFrame:
    """Drop ID columns and leakage column(s). Returns a NEW df."""
    return df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")


def convert_temps_to_celsius(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert temperature columns from Kelvin to Celsius and rename columns.
    Returns a NEW df (does not mutate input).
    """
    df = df.copy()

    air_k = "Air temperature [K]"
    proc_k = "Process temperature [K]"

    if air_k in df.columns:
        df[air_k] = pd.to_numeric(df[air_k], errors="coerce") - 273.15
    if proc_k in df.columns:
        df[proc_k] = pd.to_numeric(df[proc_k], errors="coerce") - 273.15

    rename_map: dict[str, str] = {}
    if air_k in df.columns:
        rename_map[air_k] = "Air temperature [°C]"
    if proc_k in df.columns:
        rename_map[proc_k] = "Process temperature [°C]"

    if rename_map:
        df = df.rename(columns=rename_map)

    return df


def prepare_dataframe(
    df: pd.DataFrame,
    drop_cols: list[str] | tuple[str, ...],
) -> pd.DataFrame:
    """
    Standard preprocessing used across training & evaluation:
      1) drop leakage/IDs
      2) Kelvin -> Celsius conversion (+ rename)
    """
    df2 = drop_leakage_and_ids(df, drop_cols=drop_cols)
    df2 = convert_temps_to_celsius(df2)
    return df2
