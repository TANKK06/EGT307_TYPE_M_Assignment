from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    # Data
    data_path: Path = Path("model/data/Machine_Failure.csv")
    target_col: str = "Target"

    # Split
    test_size: float = 0.2
    seed: int = 42

    # Columns
    cat_cols: tuple[str, ...] = ("Type",)
    # IMPORTANT: drop leakage + identifiers
    drop_cols: tuple[str, ...] = ("UDI", "Product ID", "Failure Type")

    # Model selection metric (imbalance-friendly)
    # options: "pr_auc", "f1_macro", "f1_failure", "recall_failure", "precision_failure"
    select_metric: str = "pr_auc"

    # SMOTE
    smote_k_neighbors: int = 5
    smote_strategy: str | float | dict = "auto"

    # Fine-tuning
    cv_folds: int = 5
    n_iter_tune: int = 25

    # Output
    artifacts_dir: Path = Path("model/artifacts")
