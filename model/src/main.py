# src/main.py
# Entry point for training + tuning the predictive maintenance model.
# Flow:
#   Load CSV -> clean/preprocess -> feature engineering -> split -> build pipeline (preprocess + SMOTE + model)
#   Compare baseline models -> fine-tune the best -> evaluate -> save model + metrics report.

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


from config import Config  
from data import load_raw   
from preprocessing import prepare_dataframe  
from feature_engineering import add_engineered_features  
from features import build_preprocessor  
from smote import SmoteConfig, make_smote  
from training import build_models, make_pipeline, evaluate_and_pick_best, fine_tune_best  
from evaluate import compute_metrics, get_proba  
from save import save_artifacts  


def parse_args(cfg: Config):
    """
    Parse CLI arguments so we can run training with different settings without editing code.

    Example:
      python src/main.py --csv data.csv --metric pr_auc --out model/artifacts

    Args:
        cfg: Config object providing default values.
    """
    p = argparse.ArgumentParser(
        description="Predictive Maintenance Pipeline (baseline -> fine-tune -> save)"
    )

    # Dataset + column config
    p.add_argument("--csv", default=str(cfg.data_path), help=f"Path to dataset CSV (default: {cfg.data_path})")
    p.add_argument("--target", default=None, help="Override target column (default from config)")

    # Model selection metric (must match keys returned by compute_metrics)
    p.add_argument(
        "--metric",
        default=None,
        choices=["pr_auc", "f1_macro", "f1_failure", "recall_failure", "precision_failure"],
        help="Metric used to select & tune the best model (default from config)",
    )

    # Output folder for saved artifacts (model + metrics report)
    p.add_argument("--out", default=None, help="Override artifacts dir (default from config)")

    # Train/test split settings
    p.add_argument("--test-size", type=float, default=None, help="Override test split ratio")
    p.add_argument("--seed", type=int, default=None, help="Override seed")

    # SMOTE settings (used to handle imbalanced target labels)
    p.add_argument("--smote-k", type=int, default=None, help="Override SMOTE k_neighbors")
    p.add_argument("--smote-strategy", default=None, help="Override SMOTE sampling_strategy")

    # Hyperparameter tuning settings
    p.add_argument("--cv-folds", type=int, default=None, help="Override tune CV folds")
    p.add_argument("--n-iter", type=int, default=None, help="Override tune iterations")

    return p.parse_args()


def main():
    # Load defaults from Config (can be overridden by CLI args)
    cfg = Config()
    args = parse_args(cfg)

    # Resolve runtime settings (CLI overrides config)
    target = args.target or cfg.target_col
    select_metric = args.metric or cfg.select_metric
    out_dir = Path(args.out or str(cfg.artifacts_dir))

    test_size = args.test_size if args.test_size is not None else cfg.test_size
    seed = args.seed if args.seed is not None else cfg.seed

    smote_k = args.smote_k if args.smote_k is not None else cfg.smote_k_neighbors
    smote_strategy = args.smote_strategy if args.smote_strategy is not None else cfg.smote_strategy

    cv_folds = args.cv_folds if args.cv_folds is not None else cfg.cv_folds
    n_iter = args.n_iter if args.n_iter is not None else cfg.n_iter_tune

    # --------------------
    # 1) Load raw dataset
    # --------------------
    df = load_raw(args.csv)

    # --------------------
    # 2) Preprocess / cleaning
    # - drop ID-like columns (not useful for prediction, can cause leakage)
    # - any dataset cleaning (e.g., units conversion) is done inside prepare_dataframe()
    # --------------------
    df = prepare_dataframe(df, drop_cols=cfg.drop_cols)

    # --------------------
    # 3) Feature engineering
    # Adds extra helpful features (temperature diff, mechanical power, etc.)
    # --------------------
    df = add_engineered_features(df)

    # Validate target exists
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found. Columns: {list(df.columns)}")

    # --------------------
    # 4) Split features (X) and label (y)
    # --------------------
    X = df.drop(columns=[target]).copy()
    y = df[target].astype(int).copy()  # ensure binary int labels

    # --------------------
    # 5) Detect categorical + numeric columns safely
    # - cat_cols is based on cfg.cat_cols but filtered to those that exist in the dataset
    # - num_cols: numeric dtype columns that are not categorical
    # --------------------
    cat_cols = [c for c in cfg.cat_cols if c in X.columns]
    num_cols = [
        c for c in X.columns
        if c not in cat_cols and pd.api.types.is_numeric_dtype(X[c])
    ]

    # Guardrails: pipeline needs at least one cat and one numeric col (in this dataset design)
    if not cat_cols:
        raise ValueError("No categorical columns found. Check cfg.cat_cols and your dataset columns.")
    if not num_cols:
        raise ValueError("No numeric columns found. Check preprocessing/feature engineering outputs.")

    # --------------------
    # 6) Train/test split
    # - stratify keeps class ratio similar in train and test (important for imbalance)
    # --------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=seed,
        stratify=y if y.nunique() > 1 else None
    )

    # --------------------
    # 7) Build preprocessing + SMOTE components
    # pre: OneHotEncode categorical + Scale numeric + Impute missing values
    # smote: Oversample minority class to reduce imbalance during training
    # --------------------
    pre = build_preprocessor(cat_cols=cat_cols, num_cols=num_cols)

    smote = make_smote(SmoteConfig(
        random_state=seed,
        k_neighbors=smote_k,
        sampling_strategy=smote_strategy,
    ))

    # --------------------
    # 8) Baseline comparison
    # - build_models() returns a dict of candidate sklearn models
    # - make_pipeline() wraps: preprocessor -> SMOTE -> classifier
    # - evaluate_and_pick_best compares models on the select_metric and picks the best
    # --------------------
    models = build_models(seed=seed)
    model_pipes = {name: make_pipeline(pre, smote, model) for name, model in models.items()}

    baseline_results, best_name, best_pipe = evaluate_and_pick_best(
        model_pipes,
        X_train, y_train,
        X_test, y_test,
        select_metric=select_metric,
    )

    # --------------------
    # 9) Fine-tune ONLY the best baseline model
    # - runs RandomizedSearchCV (or similar) using select_metric aligned with config
    # --------------------
    tuned_pipe, best_cv_score, best_params = fine_tune_best(
        best_name=best_name,
        best_pipe=best_pipe,
        X_train=X_train,
        y_train=y_train,
        seed=seed,
        cv_folds=cv_folds,
        n_iter=n_iter,
        select_metric=select_metric,
    )

    # --------------------
    # 10) Final evaluation on test set
    # - y_pred: hard labels
    # - y_proba: probability-like scores (needed for PR-AUC)
    # --------------------
    y_pred = tuned_pipe.predict(X_test)
    y_proba = get_proba(tuned_pipe, X_test)
    tuned_metrics = compute_metrics(y_test, y_pred, y_proba)

    # Print human-readable summary
    print("\n=== Fine-tuned result on test ===")
    print(f"Selected metric   : {select_metric}")
    print(f"Best baseline     : {best_name}")
    if best_cv_score is not None:
        print(f"Tune CV score     : {best_cv_score:.4f}")
    if best_params:
        print(f"Best params       : {best_params}")
    print(f"Test PR-AUC       : {tuned_metrics['pr_auc']:.4f}")
    print(f"Test Accuracy     : {tuned_metrics['accuracy']:.4f}")
    print(f"Test F1 (failure) : {tuned_metrics['f1_failure']:.4f}")
    print(f"Test Recall(fail) : {tuned_metrics['recall_failure']:.4f}")
    print(f"Test Precision(f) : {tuned_metrics['precision_failure']:.4f}")
    print(f"Test F1-macro     : {tuned_metrics['f1_macro']:.4f}")
    print(f"Confusion matrix  : {tuned_metrics['confusion_matrix']}")

    # --------------------
    # 11) Save artifacts
    # Saves:
    # - trained pipeline (includes preprocess + SMOTE + model)
    # - metrics/report JSON for transparency + debugging
    # --------------------
    report = {
        "selected_metric": select_metric,
        "best_baseline_model": best_name,
        "best_cv_score": best_cv_score,
        "best_params": best_params,
        "baseline_results": baseline_results,
        "tuned_test_metrics": tuned_metrics,
        "columns": {"categorical": cat_cols, "numeric": num_cols},
        "data": {"csv": str(Path(args.csv).resolve()), "n_rows": int(df.shape[0])},
        "split": {"test_size": float(test_size), "seed": int(seed)},
        "smote": {"k_neighbors": int(smote_k), "sampling_strategy": smote_strategy},
        "tuning": {"cv_folds": int(cv_folds), "n_iter": int(n_iter)},
    }

    model_path, metrics_path = save_artifacts(out_dir, tuned_pipe, report)
    print(f"\nSaved model   to: {model_path.resolve()}")
    print(f"Saved metrics to: {metrics_path.resolve()}")


if __name__ == "__main__":
    # Allows running: python src/main.py
    main()
