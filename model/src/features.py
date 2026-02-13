from __future__ import annotations

from typing import List
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


def _ohe_no_sparse():
    """
    Create a OneHotEncoder that outputs a dense array.

    Why this exists:
    - Newer sklearn uses `sparse_output=False`
    - Older sklearn uses `sparse=False`
    This helper keeps your code working across versions.
    """
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor(cat_cols: List[str], num_cols: List[str]) -> ColumnTransformer:
    """
    Build a preprocessing pipeline:

    - Categorical columns:
        1) Impute missing values using most frequent value
        2) One-hot encode (ignore unseen categories at inference time)

    - Numeric columns:
        1) Impute missing values using median
        2) Standardize (zero mean, unit variance)

    Using imputers prevents NaNs from breaking OHE/scaler and makes inference safer.
    """
    # Pipeline for categorical features
    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),  # fill NaNs with mode
            ("ohe", _ohe_no_sparse()),                             # convert categories -> binary columns
        ]
    )

    # Pipeline for numeric features
    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),         # fill NaNs with median
            ("scaler", StandardScaler()),                          # normalize feature scales
        ]
    )

    # Combine both pipelines into a single transformer
    return ColumnTransformer(
        transformers=[
            ("cat", cat_pipe, cat_cols),                           # apply cat_pipe to cat_cols
            ("num", num_pipe, num_cols),                           # apply num_pipe to num_cols
        ],
        remainder="drop",                                          # drop any columns not listed above
    )
