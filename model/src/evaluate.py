from __future__ import annotations

from typing import Any, Dict
import numpy as np

from sklearn.metrics import (
    average_precision_score,     # PR-AUC (better than ROC-AUC for heavy imbalance)
    f1_score,                    # F1 score (harmonic mean of precision + recall)
    precision_score,             # Precision = TP / (TP + FP)
    accuracy_score,              # Accuracy = (TP + TN) / total
    recall_score,                # Recall = TP / (TP + FN)
    confusion_matrix,            # Confusion matrix counts
    classification_report,       # Per-class precision/recall/F1 summary
)


def get_proba(model, X) -> np.ndarray:
    """
    Return probability for class 1 (failure), with safe fallbacks.

    Priority:
    1) predict_proba: standard probabilistic models (LogReg, RF, etc.)
    2) decision_function: models like SVM -> convert scores to (0,1) using sigmoid
    3) predict: last resort -> treat predicted labels as "probabilities"
    """
    # Most sklearn classifiers implement predict_proba
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)     # usually shape (n_samples, 2)
        proba = np.asarray(proba)
        if proba.ndim == 2 and proba.shape[1] >= 2:
            return proba[:, 1].ravel()     # probability of class 1
        return proba.ravel()               # fallback if model returns unusual shape

    # Some classifiers (e.g., LinearSVC) provide decision_function instead
    if hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(X)).ravel()
        # Convert raw scores to pseudo-probabilities using a sigmoid
        return 1.0 / (1.0 + np.exp(-scores))

    # Worst-case fallback: use predicted labels (0/1) as float "probabilities"
    return np.asarray(model.predict(X)).astype(float).ravel()


def compute_metrics(y_true, y_pred, y_proba) -> Dict[str, Any]:
    """
    Compute metrics for imbalanced binary classification.

    IMPORTANT: Keys here MUST match what you use for `select_metric`
    (e.g., in config.py / training pipeline).
    """
    # Ensure 1D arrays
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    y_proba = np.asarray(y_proba).ravel()

    return {
        # Core metrics (imbalance-friendly)
        "accuracy": float(accuracy_score(y_true, y_pred)),  # can be misleading under imbalance
        "pr_auc": float(average_precision_score(y_true, y_proba)),  # preferred for imbalance
        "f1_failure": float(f1_score(y_true, y_pred, pos_label=1)),  # F1 for the failure class
        "recall_failure": float(recall_score(y_true, y_pred, pos_label=1)),  # sensitivity to failures
        "precision_failure": float(
            precision_score(y_true, y_pred, pos_label=1, zero_division=0)  # avoid warnings if no positives predicted
        ),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),  # treats both classes equally

        # Diagnostics (useful for debugging + reports)
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),  # JSON-serializable
        "classification_report": classification_report(
            y_true, y_pred, output_dict=True, zero_division=0
        ),  # JSON-style report per class
    }


def metric_to_sklearn_scoring(select_metric: str) -> str:
    """
    Map our internal metric names to scikit-learn scoring strings
    (used by GridSearchCV / RandomizedSearchCV / CrossValidator).
    """
    mapping = {
        "pr_auc": "average_precision",
        "f1_failure": "f1",
        "recall_failure": "recall",
        "precision_failure": "precision",
        "f1_macro": "f1_macro",
    }
    # Default to average_precision if unknown metric is requested
    return mapping.get(select_metric, "average_precision")
