from __future__ import annotations

from pathlib import Path
import pandas as pd


def load_raw(csv_path: str | Path) -> pd.DataFrame:
    """Load the raw predictive maintenance CSV."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    return pd.read_csv(csv_path)
