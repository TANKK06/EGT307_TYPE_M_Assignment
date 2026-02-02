from __future__ import annotations

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline


FEATURES = [
    "Type",
    "Air_temperature_C",
    "Process_temperature_C",
    "Rotational_speed_rpm",
    "Torque_Nm",
    "Tool_wear_min",
]


def train_and_save(df: pd.DataFrame, model_path: str, seed: int = 42) -> dict:
    if "Target" not in df.columns:
        raise ValueError("Dataset must contain 'Target' column for training.")
    for c in FEATURES:
        if c not in df.columns:
            raise ValueError(f"Missing required feature column: {c}")

    X = df[FEATURES].copy()
    y = df["Target"].astype(int).copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )

    cat_cols = ["Type"]
    num_cols = [c for c in FEATURES if c not in cat_cols]

    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", StandardScaler(), num_cols),
        ]
    )

    clf = RandomForestClassifier(
        n_estimators=400,
        random_state=seed,
        class_weight="balanced",
        n_jobs=-1,
    )

    pipe = ImbPipeline(steps=[
        ("pre", pre),
        ("smote", SMOTE(random_state=seed)),
        ("model", clf),
    ])

    pipe.fit(X_train, y_train)

    pred = pipe.predict(X_test)
    f1 = f1_score(y_test, pred, average="macro")
    acc = accuracy_score(y_test, pred)

    joblib.dump(pipe, model_path)

    return {"f1_macro": float(f1), "accuracy": float(acc), "model_path": model_path}
