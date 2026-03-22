"""
Model training with MLflow experiment tracking.

Improvements over v1:
- TimeSeriesSplit CV (no shuffle) — prevents future leakage
- Optuna hyperparameter tuning (30 trials)
- Early stopping on eval set
- class_weight / scale_pos_weight to handle label imbalance
- Soft-voting ensemble of XGB + LGBM saved as third model
"""
from __future__ import annotations

import json
import time
import warnings
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
    average_precision_score,
)
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from finai.config.settings import (
    MLFLOW_EXPERIMENT_NAME, MLFLOW_TRACKING_URI,
    MODELS_DIR, RANDOM_STATE,
)
from finai.utils.logger import get_logger

warnings.filterwarnings("ignore", category=UserWarning)
logger = get_logger(__name__)
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

N_OPTUNA_TRIALS = 30
CV_SPLITS       = 5          # TimeSeriesSplit folds


# ── Metric helper ─────────────────────────────────────────────────────────────

def _compute_metrics(y_true, y_pred, y_prob) -> dict:
    return {
        "accuracy":  round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall":    round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1":        round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "roc_auc":   round(float(roc_auc_score(y_true, y_prob)), 4),
        "avg_precision": round(float(average_precision_score(y_true, y_prob)), 4),
    }


# ── Optuna hyperparameter search ──────────────────────────────────────────────

def _optuna_tune(
    model_type: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_trials: int = N_OPTUNA_TRIALS,
) -> dict:
    """
    Run Optuna to find best hyperparameters using TimeSeriesSplit CV.
    Returns the best params dict.
    """
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        logger.warning("Optuna not installed — skipping hyperparameter tuning.")
        return {}

    tscv = TimeSeriesSplit(n_splits=CV_SPLITS)

    def objective(trial):
        if model_type == "xgb":
            params = {
                "n_estimators":     trial.suggest_int("n_estimators", 200, 600),
                "max_depth":        trial.suggest_int("max_depth", 3, 8),
                "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
                "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "gamma":            trial.suggest_float("gamma", 0.0, 0.5),
                "reg_alpha":        trial.suggest_float("reg_alpha", 1e-4, 5.0, log=True),
                "reg_lambda":       trial.suggest_float("reg_lambda", 1e-4, 5.0, log=True),
                "random_state":     RANDOM_STATE,
                "n_jobs":           -1,
                "eval_metric":      "logloss",
            }
            model = XGBClassifier(**params)
        else:
            params = {
                "n_estimators":      trial.suggest_int("n_estimators", 200, 600),
                "max_depth":         trial.suggest_int("max_depth", 3, 8),
                "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
                "num_leaves":        trial.suggest_int("num_leaves", 20, 80),
                "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "min_child_samples": trial.suggest_int("min_child_samples", 10, 50),
                "reg_alpha":         trial.suggest_float("reg_alpha", 1e-4, 5.0, log=True),
                "reg_lambda":        trial.suggest_float("reg_lambda", 1e-4, 5.0, log=True),
                "random_state":      RANDOM_STATE,
                "n_jobs":            -1,
                "verbose":          -1,
            }
            model = LGBMClassifier(**params)

        scores = []
        for train_idx, val_idx in tscv.split(X_train):
            m = model.__class__(**params)
            m.fit(X_train[train_idx], y_train[train_idx])
            prob = m.predict_proba(X_train[val_idx])[:, 1]
            try:
                scores.append(roc_auc_score(y_train[val_idx], prob))
            except Exception:
                scores.append(0.5)
        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize",
                                 sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    logger.info(f"Optuna best ROC-AUC ({model_type}): {study.best_value:.4f}")
    return study.best_params


# ── Core train function ───────────────────────────────────────────────────────

def train_model(
    ticker: str,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    feature_cols: list[str],
    model_type: str = "xgb",
    tune: bool = True,
    register: bool = True,
) -> tuple[Any, dict]:
    """
    Train one model with optional Optuna tuning, track in MLflow.
    Returns (fitted_model, metrics_dict).
    """
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    model_name = f"{ticker}_{model_type}"

    # Class imbalance weight
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    pos_weight = n_neg / max(n_pos, 1)
    logger.info(f"{ticker} class ratio — pos:{n_pos}  neg:{n_neg}  weight:{pos_weight:.2f}")

    with mlflow.start_run(run_name=model_name) as run:
        run_id = run.info.run_id

        # ── Hyperparameter tuning ──────────────────────────────────────────
        best_params = {}
        if tune:
            logger.info(f"Running Optuna ({N_OPTUNA_TRIALS} trials) for {model_type}…")
            best_params = _optuna_tune(model_type, X_train, y_train, N_OPTUNA_TRIALS)
            mlflow.log_params({f"opt_{k}": v for k, v in best_params.items()})

        # ── Build final model ──────────────────────────────────────────────
        if model_type == "xgb":
            defaults = dict(
                n_estimators=400, max_depth=5, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
                gamma=0.1, reg_alpha=0.2, reg_lambda=1.0,
                scale_pos_weight=pos_weight,
                eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=-1,
            )
            defaults.update(best_params)
            defaults["scale_pos_weight"] = pos_weight   # always set
            model = XGBClassifier(**defaults)
        else:
            defaults = dict(
                n_estimators=400, max_depth=5, learning_rate=0.05,
                num_leaves=40, subsample=0.8, colsample_bytree=0.8,
                min_child_samples=20, reg_alpha=0.2, reg_lambda=1.0,
                class_weight="balanced",
                random_state=RANDOM_STATE, n_jobs=-1, verbose=-1,
            )
            defaults.update(best_params)
            defaults["class_weight"] = "balanced"
            model = LGBMClassifier(**defaults)

        mlflow.log_params({
            "ticker": ticker, "model_type": model_type,
            "n_train": len(X_train), "n_test": len(X_test),
            "n_features": len(feature_cols),
            "tune": tune, "pos_weight": round(pos_weight, 3),
        })

        # ── TimeSeriesSplit CV (no shuffle — no leakage) ───────────────────
        tscv = TimeSeriesSplit(n_splits=CV_SPLITS)
        cv_scores = []
        for tr_idx, val_idx in tscv.split(X_train):
            m = model.__class__(**{k: v for k, v in model.get_params().items()})
            m.fit(X_train[tr_idx], y_train[tr_idx])
            prob = m.predict_proba(X_train[val_idx])[:, 1]
            try:
                cv_scores.append(roc_auc_score(y_train[val_idx], prob))
            except Exception:
                cv_scores.append(0.5)

        cv_mean = float(np.mean(cv_scores))
        cv_std  = float(np.std(cv_scores))
        mlflow.log_metric("cv_roc_auc_mean", round(cv_mean, 4))
        mlflow.log_metric("cv_roc_auc_std",  round(cv_std,  4))
        logger.info(f"TimeSeriesSplit CV ROC-AUC: {cv_mean:.4f} ± {cv_std:.4f}")

        # ── Final fit with early stopping on held-out slice ────────────────
        t0 = time.time()
        val_size = int(len(X_train) * 0.15)
        X_tr, X_val = X_train[:-val_size], X_train[-val_size:]
        y_tr, y_val = y_train[:-val_size], y_train[-val_size:]

        if model_type == "xgb":
            model.set_params(n_estimators=2000, early_stopping_rounds=50)
            model.fit(X_tr, y_tr,
                      eval_set=[(X_val, y_val)],
                      verbose=False)
        else:
            model.set_params(n_estimators=2000)
            model.fit(X_tr, y_tr,
                      eval_set=[(X_val, y_val)],
                      callbacks=[__import__("lightgbm").early_stopping(50, verbose=False),
                                  __import__("lightgbm").log_evaluation(-1)])

        train_time = time.time() - t0
        mlflow.log_metric("train_time_s", round(train_time, 2))
        best_iter = getattr(model, "best_iteration_",
                            getattr(model, "best_ntree_limit", None))
        if best_iter:
            mlflow.log_metric("best_iteration", int(best_iter))
        logger.info(f"Training done in {train_time:.1f}s  best_iter={best_iter}")

        # ── Evaluate on test set ───────────────────────────────────────────
        y_pred  = model.predict(X_test)
        y_prob  = model.predict_proba(X_test)[:, 1]
        metrics = _compute_metrics(y_test, y_pred, y_prob)
        mlflow.log_metrics(metrics)
        logger.info(f"Test metrics: {metrics}")

        # ── Artifacts ─────────────────────────────────────────────────────
        if hasattr(model, "feature_importances_"):
            fi = pd.DataFrame({
                "feature": feature_cols,
                "importance": model.feature_importances_,
            }).sort_values("importance", ascending=False)
            fi_path = MODELS_DIR / f"{model_name}_feature_importance.csv"
            fi.to_csv(fi_path, index=False)
            mlflow.log_artifact(str(fi_path))

        report = classification_report(y_test, y_pred, output_dict=True)
        rpt_path = MODELS_DIR / f"{model_name}_report.json"
        with open(rpt_path, "w") as f:
            json.dump(report, f, indent=2)
        mlflow.log_artifact(str(rpt_path))

        if model_type == "xgb":
            mlflow.xgboost.log_model(model, artifact_path="model")
        else:
            mlflow.sklearn.log_model(model, artifact_path="model")

        model_path = MODELS_DIR / f"{model_name}.joblib"
        joblib.dump(model, model_path)

        if register:
            try:
                mv = mlflow.register_model(f"runs:/{run_id}/model",
                                            f"finai-{ticker}-{model_type}")
                logger.info(f"Registered model v{mv.version}")
            except Exception as e:
                logger.warning(f"Model Registry unavailable: {e}")

        return model, metrics


# ── Ensemble helper ───────────────────────────────────────────────────────────

def build_ensemble(ticker: str, X_test: np.ndarray) -> np.ndarray | None:
    """
    Load saved XGB and LGBM models and return soft-vote probabilities.
    Returns None if either model is missing.
    """
    xgb_path  = MODELS_DIR / f"{ticker}_xgb.joblib"
    lgbm_path = MODELS_DIR / f"{ticker}_lgbm.joblib"
    if not (xgb_path.exists() and lgbm_path.exists()):
        return None
    xgb_model  = joblib.load(xgb_path)
    lgbm_model = joblib.load(lgbm_path)
    prob_xgb   = xgb_model.predict_proba(X_test)[:, 1]
    prob_lgbm  = lgbm_model.predict_proba(X_test)[:, 1]
    return (prob_xgb + prob_lgbm) / 2


def train_all_models(
    ticker: str,
    data: dict,
    tune: bool = True,
) -> dict[str, tuple]:
    """
    Train XGB + LGBM, save an ensemble, return {model_type: (model, metrics)}.
    """
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
                tune=tune,
            )
            results[mtype] = (model, metrics)
        except Exception as e:
            logger.error(f"Training {mtype} for {ticker} failed: {e}")

    # Build and evaluate ensemble if both models trained
    if "xgb" in results and "lgbm" in results:
        try:
            scaler    = data.get("scaler")
            X_test_sc = data["X_test"]
            ens_prob  = build_ensemble(ticker, X_test_sc)
            if ens_prob is not None:
                ens_pred = (ens_prob >= 0.5).astype(int)
                ens_metrics = _compute_metrics(data["y_test"], ens_pred, ens_prob)
                # Save ensemble probs for predictor
                joblib.dump({"xgb_w": 0.5, "lgbm_w": 0.5},
                            MODELS_DIR / f"{ticker}_ensemble_config.joblib")
                logger.info(f"Ensemble metrics: {ens_metrics}")
                results["ensemble"] = (None, ens_metrics)
        except Exception as e:
            logger.warning(f"Ensemble step failed: {e}")

    return results
