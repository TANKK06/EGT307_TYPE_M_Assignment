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
    """
    Get probability scores for the positive class (class 1 = failure), with fallbacks.

    Why we need this:
    - Different ML models expose "probability" in different ways.
    - Some have predict_proba(), some have decision_function(), and some only predict().

    Returns:
        1D numpy array of "probability-like" scores for class 1.
    """
    # Best case: the model supports predict_proba (most classifiers do)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)        # usually shape = (n_samples, n_classes)
        proba = np.asarray(proba)

        # If output is 2D and has at least 2 classes, take column 1 (positive class)
        if proba.ndim == 2 and proba.shape[1] >= 2:
            return proba[:, 1].ravel()

        # Some models may return a single column; flatten it
        return proba.ravel()

    # Fallback: decision_function gives raw scores (e.g., SVM, linear models)
    # Convert scores -> (0,1) range using sigmoid so we can compute PR-AUC, etc.
    if hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(X)).ravel()
        return 1.0 / (1.0 + np.exp(-scores))  # sigmoid

    # Last fallback: no probability or decision score available
    # Use predicted labels (0/1) as float. Not ideal, but avoids crashing.
    return np.asarray(model.predict(X)).astype(float).ravel()


def compute_metrics(y_true, y_pred, y_proba) -> Dict[str, Any]:
    """
    Compute metrics for imbalanced binary classification.

    Notes:
    - Keys returned here MUST match "select_metric" in your config
      (so the training/tuning code knows what value to optimize).
    - We include both "threshold metrics" (accuracy/f1/precision/recall)
      and "ranking metrics" (PR-AUC based on probabilities).

    Args:
        y_true: Ground truth labels (0/1).
        y_pred: Predicted labels (0/1).
        y_proba: Predicted probability-like scores for class 1.

    Returns:
        Dictionary of metrics + extra report objects.
    """
    # Ensure everything is a clean 1D array (avoids shape bugs)
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    y_proba = np.asarray(y_proba).ravel()

    return {
        # --------------------
        # Core metrics
        # --------------------
        # Accuracy can be misleading with imbalance but still useful for reference
        "accuracy": float(accuracy_score(y_true, y_pred)),

        # PR-AUC (Average Precision): good for imbalanced problems
        # Uses probabilities/scores, not hard labels
        "pr_auc": float(average_precision_score(y_true, y_proba)),

        # Focus on positive class (failure = 1)
        "f1_failure": float(f1_score(y_true, y_pred, pos_label=1)),
        "recall_failure": float(recall_score(y_true, y_pred, pos_label=1)),
        "precision_failure": float(
            precision_score(y_true, y_pred, pos_label=1, zero_division=0)
        ),

        # Macro F1 treats both classes equally (useful under imbalance)
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),

        # --------------------
        # Helpful diagnostics
        # --------------------
        # Confusion matrix: [[TN, FP], [FN, TP]]
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),

        # Full classification report (per-class precision/recall/f1 + averages)
        # output_dict=True returns a dict so you can save as JSON
        "classification_report": classification_report(
            y_true, y_pred, output_dict=True, zero_division=0
        ),
    }


def metric_to_sklearn_scoring(select_metric: str) -> str:
    """
    Map our metric names to scikit-learn scoring strings.

    Why we need this:
    - scikit-learn uses specific scoring names in GridSearchCV / RandomizedSearchCV.
    - Our config uses friendlier names (e.g., "pr_auc"), so we translate them here.

    If the metric name isn't recognized, default to "average_precision" (PR-AUC).
    """
    mapping = {
        "pr_auc": "average_precision",
        "f1_failure": "f1",
        "recall_failure": "recall",
        "precision_failure": "precision",
        "f1_macro": "f1_macro",
    }
    return mapping.get(select_metric, "average_precision")
