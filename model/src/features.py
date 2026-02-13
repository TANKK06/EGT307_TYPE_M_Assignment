from __future__ import annotations

from typing import List
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


def _ohe_no_sparse():
    """
    Create a OneHotEncoder that returns a DENSE output.

    Why this exists:
    - Newer scikit-learn versions use `sparse_output=False`
    - Older versions use `sparse=False`
    This helper keeps the code working across different sklearn versions.
    """
    try:
        # sklearn >= 1.2
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        # sklearn < 1.2
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor(cat_cols: List[str], num_cols: List[str]) -> ColumnTransformer:
    """
    Build a preprocessing pipeline for ML training/inference.

    What it does:
    - Categorical columns:
        1) Impute missing values using the most frequent category
        2) One-hot encode to convert categories -> numeric vectors
        * handle_unknown="ignore" prevents crashing if a new category appears at inference time

    - Numerical columns:
        1) Impute missing values using the median (robust to outliers)
        2) Standardize values (mean=0, std=1) to help many models train better

    Returns:
        A ColumnTransformer that applies different pipelines to different column types.

    Important:
    - remainder="drop" means any columns not listed in cat_cols/num_cols are removed.
      (This helps avoid accidentally training on ID columns or unexpected fields.)
    """

    # --------------------
    # Pipeline for categorical features
    # --------------------
    cat_pipe = Pipeline(
        steps=[
            # Fill NaN with the most common category so OHE won't crash
            ("imputer", SimpleImputer(strategy="most_frequent")),
            # Convert categories into one-hot vectors (dense output)
            ("ohe", _ohe_no_sparse()),
        ]
    )

    # --------------------
    # Pipeline for numeric features
    # --------------------
    num_pipe = Pipeline(
        steps=[
            # Fill NaN with median (safer than mean when there are outliers)
            ("imputer", SimpleImputer(strategy="median")),
            # Scale features so they are comparable in magnitude
            ("scaler", StandardScaler()),
        ]
    )

    # ColumnTransformer applies cat_pipe to cat_cols and num_pipe to num_cols
    return ColumnTransformer(
        transformers=[
            ("cat", cat_pipe, cat_cols),  # apply categorical pipeline
            ("num", num_pipe, num_cols),  # apply numeric pipeline
        ],
        remainder="drop",  # drop any other columns not specified
    )
