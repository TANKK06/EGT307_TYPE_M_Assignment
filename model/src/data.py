from __future__ import annotations

from pathlib import Path
import pandas as pd


def load_raw(csv_path: str | Path) -> pd.DataFrame:
    """
    Load the raw predictive maintenance dataset from a CSV file.

    Args:
        csv_path: Path to the CSV file. Accepts either a string or a pathlib.Path.

    Returns:
        A pandas DataFrame containing the raw data (no cleaning/processing done here).

    Raises:
        FileNotFoundError: If the provided CSV path does not exist.
    """
    # Convert input to Path object so we can use Path utilities (exists(), etc.)
    csv_path = Path(csv_path)

    # Fail fast with a clear error message if the file path is wrong
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    # Read the CSV into a DataFrame and return it
    return pd.read_csv(csv_path)
