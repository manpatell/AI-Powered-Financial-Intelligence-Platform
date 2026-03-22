"""
Inference module — loads a trained model and runs predictions.
Handles scaler loading, feature alignment, and confidence scoring.
"""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from pathlib import Path

import mlflow.sklearn
import mlflow.xgboost

from finai.config.settings import MODELS_DIR, MLFLOW_TRACKING_URI, PREDICTION_HORIZON
from finai.utils.logger import get_logger

logger = get_logger(__name__)
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


def load_local_model(ticker: str, model_type: str = "xgb"):
    """Load model saved locally by trainer.py."""
    path = MODELS_DIR / f"{ticker}_{model_type}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"No model at {path}. Train first.")
    return joblib.load(path)


def load_scaler(ticker: str):
    path = MODELS_DIR / f"{ticker}_scaler.joblib"
    if not path.exists():
        return None
    return joblib.load(path)


def predict(
    ticker: str,
    feature_df: pd.DataFrame,
    feature_cols: list[str],
    model_type: str = "xgb",
) -> pd.DataFrame:
    """
    Run inference on a feature DataFrame.

    Returns a DataFrame with columns:
      Date, close, prediction (0/1), probability, signal
    """
    model  = load_local_model(ticker, model_type)
    scaler = load_scaler(ticker)

    X = feature_df[feature_cols].values
    if scaler is not None:
        X = scaler.transform(X)

    preds = model.predict(X)
    probs = model.predict_proba(X)[:, 1]

    signals = pd.Categorical(
        np.where(probs > 0.65, "STRONG BUY",
        np.where(probs > 0.55, "BUY",
        np.where(probs < 0.35, "STRONG SELL",
        np.where(probs < 0.45, "SELL", "HOLD")))),
        categories=["STRONG SELL", "SELL", "HOLD", "BUY", "STRONG BUY"],
        ordered=True,
    )

    out = pd.DataFrame({
        "Date":        feature_df.index,
        "close":       feature_df["Close"].values,
        "prediction":  preds,
        "probability": probs,
        "signal":      signals,
    })
    return out.set_index("Date")


def get_latest_signal(ticker: str, feature_df: pd.DataFrame, feature_cols: list[str]) -> dict:
    """Return the most recent prediction as a dict."""
    pred_df = predict(ticker, feature_df, feature_cols)
    latest  = pred_df.iloc[-1]
    return {
        "ticker":      ticker,
        "date":        str(pred_df.index[-1].date()),
        "close":       round(float(latest["close"]), 2),
        "signal":      str(latest["signal"]),
        "probability": round(float(latest["probability"]), 4),
        "horizon_days": PREDICTION_HORIZON,
    }
