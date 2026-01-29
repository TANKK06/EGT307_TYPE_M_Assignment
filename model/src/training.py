from __future__ import annotations

from typing import Any, Dict, Tuple
import numpy as np

from imblearn.pipeline import Pipeline as ImbPipeline

from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, SGDClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    BaggingClassifier,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

# Support both `python -m src.main` (package) and running a file directly
try:
    from .evaluate import compute_metrics, get_proba, metric_to_sklearn_scoring
except Exception:  # pragma: no cover
    from evaluate import compute_metrics, get_proba, metric_to_sklearn_scoring  # type: ignore


def build_models(seed: int = 42) -> Dict[str, Any]:
    """Candidate models for baseline comparison."""
    return {
        "Logistic Regression": LogisticRegression(max_iter=5000, class_weight="balanced"),
        "Logistic Regression CV": LogisticRegressionCV(max_iter=5000, class_weight="balanced", cv=5, l1_ratios=(0.0,), use_legacy_attributes=True),
        "SGD": SGDClassifier(random_state=seed, class_weight="balanced", loss="log_loss"),
        "Random Forest": RandomForestClassifier(random_state=seed, class_weight="balanced", n_estimators=400, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(random_state=seed),
        "AdaBoost": AdaBoostClassifier(random_state=seed),
        "Bagging": BaggingClassifier(random_state=seed),
        "Decision Tree": DecisionTreeClassifier(random_state=seed, class_weight="balanced"),
        "Support Vector Machine": SVC(probability=True, class_weight="balanced", random_state=seed),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=15),
    }


def make_pipeline(preprocessor, smote, model) -> ImbPipeline:
    """
    Full pipeline: preprocess -> SMOTE -> model
    SMOTE is applied ONLY during training fit (and inside CV folds).
    """
    return ImbPipeline(steps=[
        ("pre", preprocessor),
        ("smote", smote),
        ("model", model),
    ])


def evaluate_and_pick_best(
    model_pipes: Dict[str, Any],
    X_train, y_train,
    X_test, y_test,
    select_metric: str = "pr_auc",
) -> Tuple[Dict[str, Any], str, Any]:
    """
    Fit all pipelines, compute metrics, select best by select_metric.
    Returns: results, best_name, best_fitted_pipeline
    """
    results: Dict[str, Any] = {}
    best_name: str | None = None
    best_score = float("-inf")
    best_pipe = None

    for name, pipe in model_pipes.items():
        pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_test)
        y_proba = get_proba(pipe, X_test)

        m = compute_metrics(y_test, y_pred, y_proba)
        results[name] = m

        score = m.get(select_metric)
        if score is None:
            raise ValueError(
                f"Unknown select_metric: {select_metric}. Available: {list(m.keys())}"
            )

        print(
            f"{name:24s} | {select_metric}={m[select_metric]:.4f} "
            f"| acc={m['accuracy']:.4f} "
            f"| f1_fail={m['f1_failure']:.4f} "
            f"| recall_fail={m['recall_failure']:.4f}"
        )

        if score > best_score:
            best_score = score
            best_name = name
            best_pipe = pipe

    assert best_name is not None and best_pipe is not None
    print(f"\nBest baseline: {best_name} ({select_metric}={best_score:.4f})")
    return results, best_name, best_pipe


def _param_space_for(best_name: str) -> dict:
    """
    Search spaces for fine-tuning.
    Keys must be prefixed with 'model__' because model is a step in the pipeline.
    """
    if best_name == "Logistic Regression":
        return {
            "model__C": np.logspace(-3, 2, 30),
            "model__solver": ["lbfgs", "liblinear", "saga"],
        }

    if best_name == "Logistic Regression CV":
        # LogisticRegressionCV already cross-validates C internally; keep tuning light.
        return {
            "model__Cs": [np.logspace(-3, 2, 10), np.logspace(-4, 3, 15)],
            "model__solver": ["lbfgs", "liblinear", "saga"],
        }

    if best_name == "SGD":
        return {
            "model__alpha": np.logspace(-6, -2, 30),
            "model__penalty": ["l2", "l1", "elasticnet"],
            "model__l1_ratio": np.linspace(0.0, 1.0, 6),
        }

    if best_name == "Random Forest":
        return {
            "model__n_estimators": [300, 500, 800, 1000],
            "model__max_depth": [None, 10, 12, 16, 20],
            "model__min_samples_split": [2, 5, 10, 15],
            "model__min_samples_leaf": [1, 2, 3, 5],
            "model__max_features": ["sqrt", "log2", 0.5],
            "model__bootstrap": [True, False],
        }

    if best_name == "Gradient Boosting":
        return {
            "model__n_estimators": [100, 200, 400],
            "model__learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
            "model__max_depth": [2, 3, 4],
        }

    if best_name == "AdaBoost":
        return {
            "model__n_estimators": [100, 200, 400],
            "model__learning_rate": [0.01, 0.05, 0.1, 0.2, 0.5, 1.0],
        }

    if best_name == "Bagging":
        return {
            "model__n_estimators": [10, 30, 50, 100, 200],
            "model__max_samples": [0.5, 0.7, 1.0],
            "model__max_features": [0.5, 0.7, 1.0],
        }

    if best_name == "Decision Tree":
        return {
            "model__max_depth": [None, 4, 6, 8, 10, 12, 16],
            "model__min_samples_split": [2, 5, 10, 15],
            "model__min_samples_leaf": [1, 2, 3, 5],
        }

    if best_name == "Support Vector Machine":
        return {
            "model__C": np.logspace(-2, 2, 30),
            "model__gamma": ["scale", "auto"],
            "model__kernel": ["rbf", "linear"],
        }

    if best_name == "K-Nearest Neighbors":
        return {
            "model__n_neighbors": list(range(3, 31, 2)),
            "model__weights": ["uniform", "distance"],
            "model__p": [1, 2],
        }

    return {}


def fine_tune_best(
    best_name: str,
    best_pipe,
    X_train, y_train,
    seed: int = 42,
    cv_folds: int = 5,
    n_iter: int = 25,
    select_metric: str = "pr_auc",
):
    """
    Fine-tune ONLY the selected best pipeline using RandomizedSearchCV.

    Returns: tuned_estimator, best_cv_score, best_params
    """
    space = _param_space_for(best_name)
    if not space:
        best_pipe.fit(X_train, y_train)
        return best_pipe, None, {}

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    scoring = metric_to_sklearn_scoring(select_metric)

    search = RandomizedSearchCV(
        estimator=best_pipe,
        param_distributions=space,
        n_iter=n_iter,
        scoring=scoring,
        cv=cv,
        random_state=seed,
        n_jobs=-1,
        verbose=0,
    )
    search.fit(X_train, y_train)

    return search.best_estimator_, float(search.best_score_), dict(search.best_params_)
