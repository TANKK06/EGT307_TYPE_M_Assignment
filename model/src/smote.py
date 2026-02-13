from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
from imblearn.over_sampling import SMOTE


@dataclass
class SmoteConfig:
    random_state: int = 42
    k_neighbors: int = 5
    sampling_strategy: str | float | dict | None = "auto"  # allow None


def make_smote(cfg: Optional[SmoteConfig] = None) -> SMOTE:
    """Create a SMOTE instance (use inside an imblearn Pipeline)."""
    if cfg is None:
        cfg = SmoteConfig()

    kwargs = {
        "random_state": cfg.random_state,
        "k_neighbors": cfg.k_neighbors,
    }

    # ✅ only include sampling_strategy if it's not None
    if cfg.sampling_strategy is not None:
        kwargs["sampling_strategy"] = cfg.sampling_strategy

    return SMOTE(**kwargs)