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

# Support both `python -m src.main` (package import) and running a file directly
try:
    from .evaluate import compute_metrics, get_proba, metric_to_sklearn_scoring
except Exception:  # pragma: no cover
    from evaluate import compute_metrics, get_proba, metric_to_sklearn_scoring  # type: ignore


def build_models(seed: int = 42) -> Dict[str, Any]:
    """
    Define candidate models for baseline comparison.

    Notes:
    - Many models use class_weight="balanced" to handle class imbalance.
    - SVC uses probability=True so we can compute PR-AUC (requires probabilities).
    """
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=5000,
            class_weight="balanced",
        ),
        "Logistic Regression CV": LogisticRegressionCV(
            max_iter=5000,
            class_weight="balanced",
            cv=5,                              # internal CV for selecting regularization strength
            l1_ratios=(0.0,),                  # keep as L2 (0.0) unless you want elasticnet
            use_legacy_attributes=True,        # compatibility across sklearn versions
        ),
        "SGD": SGDClassifier(
            random_state=seed,
            class_weight="balanced",
            loss="log_loss",                   # logistic regression via SGD
        ),
        "Random Forest": RandomForestClassifier(
            random_state=seed,
            class_weight="balanced",
            n_estimators=400,
            n_jobs=-1,                         # use all CPU cores
        ),
        "Gradient Boosting": GradientBoostingClassifier(random_state=seed),
        "AdaBoost": AdaBoostClassifier(random_state=seed),
        "Bagging": BaggingClassifier(random_state=seed),
        "Decision Tree": DecisionTreeClassifier(random_state=seed, class_weight="balanced"),
        "Support Vector Machine": SVC(
            probability=True,                  # needed to compute probability-based metrics
            class_weight="balanced",
            random_state=seed,
        ),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=15),
    }


def make_pipeline(preprocessor, smote, model) -> ImbPipeline:
    """
    Build the full imblearn pipeline:
        preprocess -> SMOTE -> model

    Why ImbPipeline?
    - Ensures SMOTE is applied only during fit (training),
      including inside cross-validation folds.
    """
    return ImbPipeline(
        steps=[
            ("pre", preprocessor),             # OHE + scaling + imputing
            ("smote", smote),                  # handle imbalance (train-time only)
            ("model", model),                  # estimator
        ]
    )


def evaluate_and_pick_best(
    model_pipes: Dict[str, Any],
    X_train,
    y_train,
    X_test,
    y_test,
    select_metric: str = "pr_auc",
) -> Tuple[Dict[str, Any], str, Any]:
    """
    Fit all pipelines, compute metrics on the test split, and select the best model.

    Args:
        model_pipes: Dict of {model_name: pipeline}.
        X_train, y_train: Training data.
        X_test, y_test: Test data.
        select_metric: Metric key used to choose the best model.

    Returns:
        results: Dict of {model_name: metrics_dict}
        best_name: Name of best baseline model
        best_pipe: Fitted pipeline for the best model
    """
    results: Dict[str, Any] = {}
    best_name: str | None = None
    best_score = float("-inf")
    best_pipe = None

    for name, pipe in model_pipes.items():
        # Train on training split (SMOTE runs here)
        pipe.fit(X_train, y_train)

        # Predict labels + probabilities on test split
        y_pred = pipe.predict(X_test)
        y_proba = get_proba(pipe, X_test)

        # Compute evaluation metrics
        m = compute_metrics(y_test, y_pred, y_proba)
        results[name] = m

        # Use selected metric to compare models
        score = m.get(select_metric)
        if score is None:
            raise ValueError(
                f"Unknown select_metric: {select_metric}. Available: {list(m.keys())}"
            )

        # Print a quick comparison row (useful when running via CLI)
        print(
            f"{name:24s} | {select_metric}={m[select_metric]:.4f} "
            f"| acc={m['accuracy']:.4f} "
            f"| f1_fail={m['f1_failure']:.4f} "
            f"| recall_fail={m['recall_failure']:.4f}"
        )

        # Track the best-performing model
        if score > best_score:
            best_score = score
            best_name = name
            best_pipe = pipe

    assert best_name is not None and best_pipe is not None
    print(f"\nBest baseline: {best_name} ({select_metric}={best_score:.4f})")
    return results, best_name, best_pipe


def _param_space_for(best_name: str) -> dict:
    """
    Define hyperparameter search spaces for fine-tuning.

    IMPORTANT:
    - Keys must be prefixed with 'model__' because the estimator step in the pipeline is named "model".
    """
    if best_name == "Logistic Regression":
        return {
            "model__C": np.logspace(-3, 2, 30),           # regularization strength
            "model__solver": ["lbfgs", "liblinear", "saga"],
        }

    if best_name == "Logistic Regression CV":
        # LogisticRegressionCV already cross-validates C internally; keep tuning lighter.
        return {
            "model__Cs": [np.logspace(-3, 2, 10), np.logspace(-4, 3, 15)],
            "model__solver": ["lbfgs", "liblinear", "saga"],
        }

    if best_name == "SGD":
        return {
            "model__alpha": np.logspace(-6, -2, 30),      # regularization
            "model__penalty": ["l2", "l1", "elasticnet"],
            "model__l1_ratio": np.linspace(0.0, 1.0, 6),  # only used for elasticnet
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
            "model__p": [1, 2],  # 1=Manhattan, 2=Euclidean
        }

    # No tuning space defined for this model name
    return {}


def fine_tune_best(
    best_name: str,
    best_pipe,
    X_train,
    y_train,
    seed: int = 42,
    cv_folds: int = 5,
    n_iter: int = 25,
    select_metric: str = "pr_auc",
):
    """
    Fine-tune ONLY the best baseline pipeline using RandomizedSearchCV.

    Args:
        best_name: Name of the best baseline model.
        best_pipe: The fitted/unfitted pipeline for that model.
        X_train, y_train: Training data used for tuning (CV happens inside this).
        seed: Random seed.
        cv_folds: Number of stratified CV folds.
        n_iter: Number of random search iterations.
        select_metric: Metric key used for scoring.

    Returns:
        tuned_estimator: Best estimator found by the search (pipeline)
        best_cv_score: Best mean CV score (float) or None if no tuning space
        best_params: Best hyperparameters found (dict)
    """
    space = _param_space_for(best_name)

    # If no tuning space is defined, just fit and return the existing pipeline
    if not space:
        best_pipe.fit(X_train, y_train)
        return best_pipe, None, {}

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    scoring = metric_to_sklearn_scoring(select_metric)

    # RandomizedSearchCV tries random combinations of hyperparameters from the space
    search = RandomizedSearchCV(
        estimator=best_pipe,
        param_distributions=space,
        n_iter=n_iter,
        scoring=scoring,
        cv=cv,
        random_state=seed,
        n_jobs=-1,      # parallelize search across CPU cores
        verbose=0,
    )

    search.fit(X_train, y_train)

    return search.best_estimator_, float(search.best_score_), dict(search.best_params_)
