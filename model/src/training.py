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

# Support both:
# 1) Running as a package:   python -m src.main
# 2) Running as a script:    python training.py
# This try/except keeps imports working in both cases.
try:
    from .evaluate import compute_metrics, get_proba, metric_to_sklearn_scoring
except Exception:  # pragma: no cover
    from evaluate import compute_metrics, get_proba, metric_to_sklearn_scoring  # type: ignore


def build_models(seed: int = 42) -> Dict[str, Any]:
    """
    Define candidate models for baseline comparison.

    Why multiple models:
    - We don't know which algorithm will perform best on the dataset.
    - We test a mix of linear models, tree models, ensembles, and distance-based models.

    Args:
        seed: Random seed for reproducibility (used in models that involve randomness).

    Returns:
        Dictionary mapping {model_name: sklearn_estimator}
    """
    return {
        # Linear baseline (fast, interpretable). class_weight balances minority class.
        "Logistic Regression": LogisticRegression(max_iter=5000, class_weight="balanced"),

        # LogisticRegressionCV does its own internal CV for C.
        # use_legacy_attributes helps compatibility across some sklearn versions.
        "Logistic Regression CV": LogisticRegressionCV(
            max_iter=5000,
            class_weight="balanced",
            cv=5,
            l1_ratios=(0.0,),
            use_legacy_attributes=True,
        ),

        # SGD = linear model trained with stochastic gradient descent (fast on large data)
        "SGD": SGDClassifier(random_state=seed, class_weight="balanced", loss="log_loss"),

        # Random Forest: strong baseline for tabular data
        "Random Forest": RandomForestClassifier(
            random_state=seed, class_weight="balanced", n_estimators=400, n_jobs=-1
        ),

        # Tree boosting / ensembles
        "Gradient Boosting": GradientBoostingClassifier(random_state=seed),
        "AdaBoost": AdaBoostClassifier(random_state=seed),
        "Bagging": BaggingClassifier(random_state=seed),

        # Single tree baseline
        "Decision Tree": DecisionTreeClassifier(random_state=seed, class_weight="balanced"),

        # SVM: can be strong but slower; probability=True needed for PR-AUC
        "Support Vector Machine": SVC(probability=True, class_weight="balanced", random_state=seed),

        # KNN: simple distance-based model (works best when features are scaled)
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=15),
    }


def make_pipeline(preprocessor, smote, model) -> ImbPipeline:
    """
    Build the full ML pipeline: preprocess -> SMOTE -> model.

    Why imblearn Pipeline:
    - SMOTE must be applied ONLY on training data.
    - With an imblearn Pipeline, SMOTE is executed inside fit() / CV folds properly
      and avoids leaking synthetic samples into the test set.

    Args:
        preprocessor: ColumnTransformer handling encoding/scaling/imputation.
        smote: SMOTE oversampler instance.
        model: sklearn classifier.

    Returns:
        An imblearn Pipeline ready to fit/predict.
    """
    return ImbPipeline(steps=[
        ("pre", preprocessor),  # encodes categorical + scales numeric
        ("smote", smote),       # oversample minority class (training only)
        ("model", model),       # classifier
    ])


def evaluate_and_pick_best(
    model_pipes: Dict[str, Any],
    X_train, y_train,
    X_test, y_test,
    select_metric: str = "pr_auc",
) -> Tuple[Dict[str, Any], str, Any]:
    """
    Train and evaluate all candidate pipelines, then select the best one.

    What happens:
    - Fit each pipeline on training data
    - Predict on test data
    - Compute metrics (accuracy, PR-AUC, f1, recall, precision, etc.)
    - Pick the best pipeline based on `select_metric`

    Args:
        model_pipes: {name: pipeline} dict.
        X_train, y_train: training split.
        X_test, y_test: test split.
        select_metric: metric key from compute_metrics() used to decide the best model.

    Returns:
        results: dict of metrics for each model
        best_name: name of the best model
        best_pipe: fitted pipeline of the best model
    """
    results: Dict[str, Any] = {}
    best_name: str | None = None
    best_score = float("-inf")
    best_pipe = None

    for name, pipe in model_pipes.items():
        # Train model pipeline
        pipe.fit(X_train, y_train)

        # Predict test split
        y_pred = pipe.predict(X_test)
        y_proba = get_proba(pipe, X_test)  # probability-like scores for PR-AUC etc.

        # Compute metrics and store
        m = compute_metrics(y_test, y_pred, y_proba)
        results[name] = m

        # Select the metric we optimize on
        score = m.get(select_metric)
        if score is None:
            raise ValueError(
                f"Unknown select_metric: {select_metric}. Available: {list(m.keys())}"
            )

        # Print a compact summary row for comparison
        print(
            f"{name:24s} | {select_metric}={m[select_metric]:.4f} "
            f"| acc={m['accuracy']:.4f} "
            f"| f1_fail={m['f1_failure']:.4f} "
            f"| recall_fail={m['recall_failure']:.4f}"
        )

        # Track best model so far
        if score > best_score:
            best_score = score
            best_name = name
            best_pipe = pipe

    assert best_name is not None and best_pipe is not None
    print(f"\nBest baseline: {best_name} ({select_metric}={best_score:.4f})")
    return results, best_name, best_pipe


def _param_space_for(best_name: str) -> dict:
    """
    Hyperparameter search space for fine-tuning each model.

    Important:
    - Because the model is stored under the pipeline step name "model",
      hyperparameters must be prefixed with "model__" for RandomizedSearchCV.

    Args:
        best_name: Name of the selected best baseline model.

    Returns:
        Dict of parameter distributions to search.
        Empty dict means "no tuning defined".
    """
    if best_name == "Logistic Regression":
        return {
            "model__C": np.logspace(-3, 2, 30),                # regularization strength
            "model__solver": ["lbfgs", "liblinear", "saga"],   # different optimizers
        }

    if best_name == "Logistic Regression CV":
        # This model already does CV internally, so keep tuning lighter.
        return {
            "model__Cs": [np.logspace(-3, 2, 10), np.logspace(-4, 3, 15)],
            "model__solver": ["lbfgs", "liblinear", "saga"],
        }

    if best_name == "SGD":
        return {
            "model__alpha": np.logspace(-6, -2, 30),          # regularization strength
            "model__penalty": ["l2", "l1", "elasticnet"],
            "model__l1_ratio": np.linspace(0.0, 1.0, 6),      # only used for elasticnet
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

    # No tuning defined for this model
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

    Why only tune best:
    - Tuning every model is very expensive.
    - We first choose the best baseline, then spend compute on improving it.

    Args:
        best_name: The name of the best baseline model.
        best_pipe: The pipeline for that model.
        X_train, y_train: Training data.
        seed: Random seed (reproducibility).
        cv_folds: Number of CV folds for the tuning search.
        n_iter: Number of random parameter samples to try.
        select_metric: Metric to optimize (must map to sklearn scoring).

    Returns:
        tuned_estimator: Best pipeline found.
        best_cv_score: Best cross-validation score achieved (float) or None if no tuning.
        best_params: Dict of best hyperparameters.
    """
    space = _param_space_for(best_name)

    # If we have no search space, just fit the pipeline and return it
    if not space:
        best_pipe.fit(X_train, y_train)
        return best_pipe, None, {}

    # StratifiedKFold keeps class ratios balanced in each fold (important for imbalance)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)

    # Translate our metric name -> sklearn scoring string
    scoring = metric_to_sklearn_scoring(select_metric)

    # RandomizedSearchCV tries random combinations from the parameter space
    search = RandomizedSearchCV(
        estimator=best_pipe,
        param_distributions=space,
        n_iter=n_iter,
        scoring=scoring,
        cv=cv,
        random_state=seed,
        n_jobs=-1,   # use all CPU cores
        verbose=0,
    )

    # Fit the search on training data only (CV is done inside)
    search.fit(X_train, y_train)

    # Return best tuned pipeline + score + params
    return search.best_estimator_, float(search.best_score_), dict(search.best_params_)