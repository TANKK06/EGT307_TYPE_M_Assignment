from __future__ import annotations

from typing import Any, Dict
import numpy as np

from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    accuracy_score,
    recall_score,
    confusion_matrix,
    classification_report,
)


def get_proba(model, X) -> np.ndarray:
    """Probability for class 1 (failure), with fallbacks."""
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        proba = np.asarray(proba)
        if proba.ndim == 2 and proba.shape[1] >= 2:
            return proba[:, 1].ravel()
        return proba.ravel()

    if hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(X)).ravel()
        return 1.0 / (1.0 + np.exp(-scores))

    return np.asarray(model.predict(X)).astype(float).ravel()


def compute_metrics(y_true, y_pred, y_proba) -> Dict[str, Any]:
    """
    Metrics designed for imbalanced binary classification.
    Keys here MUST match what you use for select_metric in config/main.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    y_proba = np.asarray(y_proba).ravel()

    return {
        # imbalance-friendly
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "f1_failure": float(f1_score(y_true, y_pred, pos_label=1)),
        "recall_failure": float(recall_score(y_true, y_pred, pos_label=1)),
        "precision_failure": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        # helpful for report
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(y_true, y_pred, output_dict=True, zero_division=0),
    }


def metric_to_sklearn_scoring(select_metric: str) -> str:
    """Map our metric names to scikit-learn scoring strings."""
    mapping = {
        "pr_auc": "average_precision",
        "f1_failure": "f1",
        "recall_failure": "recall",
        "precision_failure": "precision",
        "f1_macro": "f1_macro",
    }
    return mapping.get(select_metric, "average_precision")
