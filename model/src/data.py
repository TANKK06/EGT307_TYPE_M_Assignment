from __future__ import annotations

from pathlib import Path
import pandas as pd


def load_raw(csv_path: str | Path) -> pd.DataFrame:
    """
    Load the raw predictive maintenance CSV into a pandas DataFrame.

    Args:
        csv_path: Path to the CSV file (string or pathlib.Path).

    Raises:
        FileNotFoundError: If the CSV file does not exist at the given path.

    Returns:
        A pandas DataFrame containing the raw dataset.
    """
    csv_path = Path(csv_path)              # Convert input to Path (handles str or Path)
    if not csv_path.exists():              # Validate the file exists before reading
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    return pd.read_csv(csv_path)           # Read CSV into DataFrame
