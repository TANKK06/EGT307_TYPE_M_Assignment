# model/src/config.py
# Central configuration for the ML pipeline.
# This file defines default paths + training settings, and allows overriding via environment variables
# (useful for Docker/Kubernetes where you pass configs through env vars).

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Config:
    # --------------------
    # Paths (file locations)
    # --------------------
    # Path to the training CSV dataset.
    # Can be overridden with env var DATA_PATH (e.g., in Docker/K8s).
    data_path: Path = Path(os.getenv("DATA_PATH", "model/data/Machine_Failure.csv"))

    # Directory to store generated artifacts (trained model, encoders, metrics, etc.).
    # Can be overridden with env var ARTIFACT_DIR.
    artifacts_dir: Path = Path(os.getenv("ARTIFACT_DIR", "model/artifacts"))

    # --------------------
    # Problem setup
    # --------------------
    # Target column name (the label we want to predict).
    # Override with TARGET_COL if your dataset uses a different column name.
    target_col: str = os.getenv("TARGET_COL", "Target")

    # Categorical columns (need encoding before training).
    # Use default_factory to avoid the "mutable default" dataclass issue.
    cat_cols: list[str] = field(default_factory=lambda: ["Type"])

    # Columns to drop before training:
    # - IDs are not useful features and may leak information
    drop_cols: list[str] = field(
        default_factory=lambda: [
            "UDI",
            "Product ID",
        ]
    )

    # --------------------
    # Train/test split + reproducibility
    # --------------------
    # Fraction of dataset used for testing/validation.
    test_size: float = float(os.getenv("TEST_SIZE", "0.2"))

    # Random seed so results are repeatable across runs.
    seed: int = int(os.getenv("SEED", "42"))

    # --------------------
    # SMOTE (handling class imbalance)
    # --------------------
    # SMOTE creates synthetic samples for the minority class.
    # k_neighbors controls how SMOTE generates synthetic points.
    smote_k_neighbors: int = int(os.getenv("SMOTE_K", "5"))

    # Strategy for SMOTE (e.g., "auto", "minority", or custom ratios).
    # None means "use library default" / disable custom strategy.
    smote_strategy: str | None = os.getenv("SMOTE_STRATEGY", None)  # e.g. "auto"

    # --------------------
    # Model selection / hyperparameter tuning
    # --------------------
    # Metric used to select the "best" model during tuning/CV.
    # f1_macro is good when classes are imbalanced (treats classes equally).
    select_metric: str = os.getenv("SELECT_METRIC", "f1_macro")

    # Number of folds for cross-validation (higher = more stable, slower).
    cv_folds: int = int(os.getenv("CV_FOLDS", "5"))

    # Number of random search iterations (higher = better chance of good params, slower).
    n_iter_tune: int = int(os.getenv("N_ITER_TUNE", "30"))