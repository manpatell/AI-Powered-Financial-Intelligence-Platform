"""
Model training with MLflow experiment tracking.
Trains XGBoost + LightGBM classifiers, logs params/metrics/artifacts,
and registers the best model to the MLflow Model Registry.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import joblib
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, classification_report,
    f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from finai.config.settings import (
    MLFLOW_EXPERIMENT_NAME, MLFLOW_TRACKING_URI,
    MODELS_DIR, RANDOM_STATE,
)
from finai.utils.logger import get_logger

logger = get_logger(__name__)

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


# ── Default hyperparameters ───────────────────────────────────────────────────
XGB_PARAMS: dict[str, Any] = {
    "n_estimators":     300,
    "max_depth":        6,
    "learning_rate":    0.05,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
    "gamma":            0.1,
    "reg_alpha":        0.1,
    "reg_lambda":       1.0,
    "use_label_encoder": False,
    "eval_metric":      "logloss",
    "random_state":     RANDOM_STATE,
    "n_jobs":           -1,
}

LGBM_PARAMS: dict[str, Any] = {
    "n_estimators":     300,
    "max_depth":        6,
    "learning_rate":    0.05,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "min_child_samples": 20,
    "reg_alpha":        0.1,
    "reg_lambda":       1.0,
    "random_state":     RANDOM_STATE,
    "n_jobs":           -1,
    "verbose":         -1,
}


def _compute_metrics(y_true, y_pred, y_prob) -> dict:
    return {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
        "f1":        f1_score(y_true, y_pred, zero_division=0),
        "roc_auc":   roc_auc_score(y_true, y_prob),
    }


def train_model(
    ticker: str,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    feature_cols: list[str],
    model_type: str = "xgb",
    params: dict | None = None,
    register: bool = True,
) -> tuple[Any, dict]:
    """
    Train one model, track everything in MLflow, optionally register it.

    Returns (fitted_model, metrics_dict)
    """
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    base_params = (XGB_PARAMS if model_type == "xgb" else LGBM_PARAMS).copy()
    if params:
        base_params.update(params)

    model_name = f"{ticker}_{model_type}"

    with mlflow.start_run(run_name=model_name) as run:
        run_id = run.info.run_id
        logger.info(f"MLflow run: {run_id}")

        # Log params
        mlflow.log_params({**base_params, "ticker": ticker, "model_type": model_type})
        mlflow.log_param("n_train", len(X_train))
        mlflow.log_param("n_test",  len(X_test))
        mlflow.log_param("n_features", len(feature_cols))

        # Build model
        if model_type == "xgb":
            model = XGBClassifier(**{k: v for k, v in base_params.items()
                                     if k != "use_label_encoder"})
        else:
            model = LGBMClassifier(**base_params)

        # Cross-validation on training set
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
        mlflow.log_metric("cv_roc_auc_mean", cv_scores.mean())
        mlflow.log_metric("cv_roc_auc_std",  cv_scores.std())
        logger.info(f"CV ROC-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        # Final fit
        t0 = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - t0
        mlflow.log_metric("train_time_s", train_time)

        # Evaluate
        y_pred  = model.predict(X_test)
        y_prob  = model.predict_proba(X_test)[:, 1]
        metrics = _compute_metrics(y_test, y_pred, y_prob)

        mlflow.log_metrics(metrics)
        logger.info(f"Test metrics: {metrics}")

        # Feature importance artifact
        if hasattr(model, "feature_importances_"):
            fi = pd.DataFrame({
                "feature":   feature_cols,
                "importance": model.feature_importances_,
            }).sort_values("importance", ascending=False)
            fi_path = MODELS_DIR / f"{model_name}_feature_importance.csv"
            fi.to_csv(fi_path, index=False)
            mlflow.log_artifact(str(fi_path))

        # Classification report artifact
        report = classification_report(y_test, y_pred, output_dict=True)
        report_path = MODELS_DIR / f"{model_name}_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        mlflow.log_artifact(str(report_path))

        # Log model
        if model_type == "xgb":
            mlflow.xgboost.log_model(model, artifact_path="model")
        else:
            mlflow.sklearn.log_model(model, artifact_path="model")

        # Save locally too
        model_path = MODELS_DIR / f"{model_name}.joblib"
        joblib.dump(model, model_path)

        # Register best model
        if register:
            registry_name = f"finai-{ticker}-{model_type}"
            mv = mlflow.register_model(
                f"runs:/{run_id}/model",
                registry_name,
            )
            logger.info(f"Registered model: {registry_name} v{mv.version}")

        return model, metrics


def train_all_models(ticker: str, data: dict) -> dict[str, tuple]:
    """Train both XGBoost and LightGBM for a ticker; return {model_type: (model, metrics)}."""
    results = {}
    for mtype in ["xgb", "lgbm"]:
        try:
            model, metrics = train_model(
                ticker=ticker,
                X_train=data["X_train"],
                X_test=data["X_test"],
                y_train=data["y_train"],
                y_test=data["y_test"],
                feature_cols=data["feature_cols"],
                model_type=mtype,
            )
            results[mtype] = (model, metrics)
        except Exception as e:
            logger.error(f"Training {mtype} for {ticker} failed: {e}")
    return results
