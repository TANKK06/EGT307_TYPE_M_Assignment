from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Config:
    # ---------- Paths ----------
    # Path to the training dataset CSV (can be overridden by env var DATA_PATH)
    # Using Path makes file handling (join/exists/resolve) easier than plain strings.
    data_path: Path = Path(os.getenv("DATA_PATH", "model/data/Machine_Failure.csv"))

    # Directory where training outputs (models, metrics, plots, etc.) are stored
    # Override with env var ARTIFACT_DIR if needed.
    artifacts_dir: Path = Path(os.getenv("ARTIFACT_DIR", "services/inference-api/model"))

    # ---------- Problem setup ----------
    # Name of the target/label column in the dataset
    target_col: str = os.getenv("TARGET_COL", "Target")

    # Categorical columns that need encoding (e.g., OneHotEncoder)
    # default_factory is required for mutable defaults in dataclasses.
    cat_cols: list[str] = field(default_factory=lambda: ["Type"])

    # Columns to drop before training (IDs / non-predictive identifiers)
    # Includes multiple variants in case dataset uses different naming.
    drop_cols: list[str] = field(
        default_factory=lambda: [
            "UDI",
            "Product ID",
            "Product_ID",
            "product_id",
            "id",
        ]
    )

    # ---------- Split / seed ----------
    # Proportion of data used for testing (e.g., 0.2 = 20%)
    test_size: float = float(os.getenv("TEST_SIZE", "0.2"))

    # Random seed to make splits and model training reproducible
    seed: int = int(os.getenv("SEED", "42"))

    # ---------- SMOTE ----------
    # SMOTE parameters for handling class imbalance
    smote_k_neighbors: int = int(os.getenv("SMOTE_K", "5"))

    # SMOTE strategy (None = use library default; e.g., "auto", "minority", etc.)
    smote_strategy: str | None = os.getenv("SMOTE_STRATEGY", None)

    # ---------- Model selection / tuning ----------
    # Metric used to select the "best" model during evaluation/tuning
    select_metric: str = os.getenv("SELECT_METRIC", "f1_macro")

    # Number of cross-validation folds
    cv_folds: int = int(os.getenv("CV_FOLDS", "5"))

    # Number of random search iterations for hyperparameter tuning
    n_iter_tune: int = int(os.getenv("N_ITER_TUNE", "30"))
