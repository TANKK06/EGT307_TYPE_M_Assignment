from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
from imblearn.over_sampling import SMOTE


@dataclass
class SmoteConfig:
    """
    Configuration holder for SMOTE (Synthetic Minority Over-sampling Technique).

    Why we use SMOTE:
    - In predictive maintenance, failures are usually rare (class imbalance).
    - SMOTE generates synthetic minority samples to help the model learn failure patterns better.

    Fields:
        random_state: Makes SMOTE results reproducible across runs.
        k_neighbors: Number of nearest neighbors used to create synthetic samples.
        sampling_strategy: Controls how much oversampling to do.
          Examples:
            - "auto" (default): balance minority up to majority
            - 0.5: minority will be 50% of majority
            - {"0": 1000, "1": 1000}: explicit target counts (dict form depends on labels)
    """
    random_state: int = 42
    k_neighbors: int = 5
    sampling_strategy: str | float | dict = "auto"


def make_smote(cfg: Optional[SmoteConfig] = None) -> SMOTE:
    """
    Create a SMOTE instance from configuration.

    Notes:
    - This SMOTE object is intended to be used inside an *imblearn* Pipeline,
      so oversampling happens ONLY on the training folds during CV/training
      (and does NOT leak into the test set).

    Args:
        cfg: Optional SmoteConfig. If None, uses default SmoteConfig().

    Returns:
        A configured imblearn.over_sampling.SMOTE object.
    """
    # If no config is provided, use default settings
    if cfg is None:
        cfg = SmoteConfig()

    # Create and return the SMOTE oversampler with configured parameters
    return SMOTE(
        random_state=cfg.random_state,
        k_neighbors=cfg.k_neighbors,
        sampling_strategy=cfg.sampling_strategy,
    )
