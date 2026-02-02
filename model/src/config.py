# model/src/config.py
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Config:
    # ---------- Paths ----------
    data_path: Path = Path(os.getenv("DATA_PATH", "model/data/Machine_Failure.csv"))
    artifacts_dir: Path = Path(os.getenv("ARTIFACT_DIR", "model/artifacts"))

    # ---------- Problem setup ----------
    target_col: str = os.getenv("TARGET_COL", "Target")

    # Use default_factory for lists (required by dataclasses)
    cat_cols: list[str] = field(default_factory=lambda: ["Type"])
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
    test_size: float = float(os.getenv("TEST_SIZE", "0.2"))
    seed: int = int(os.getenv("SEED", "42"))

    # ---------- SMOTE ----------
    smote_k_neighbors: int = int(os.getenv("SMOTE_K", "5"))
    smote_strategy: str | None = os.getenv("SMOTE_STRATEGY", None)  # e.g. "auto"

    # ---------- Model selection / tuning ----------
    select_metric: str = os.getenv("SELECT_METRIC", "f1_macro")
    cv_folds: int = int(os.getenv("CV_FOLDS", "5"))
    n_iter_tune: int = int(os.getenv("N_ITER_TUNE", "30"))