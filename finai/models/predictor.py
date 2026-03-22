"""
Inference module — loads trained models and runs predictions.
Supports individual models and the XGB+LGBM soft-vote ensemble.
"""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

from finai.config.settings import MODELS_DIR, PREDICTION_HORIZON
from finai.utils.logger import get_logger

logger = get_logger(__name__)


def load_local_model(ticker: str, model_type: str = "xgb"):
    path = MODELS_DIR / f"{ticker}_{model_type}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"No model at {path}. Train first.")
    return joblib.load(path)


def load_scaler(ticker: str):
    path = MODELS_DIR / f"{ticker}_scaler.joblib"
    return joblib.load(path) if path.exists() else None


def _prob_to_signal(prob: float) -> str:
    if prob >= 0.70:  return "STRONG BUY"
    if prob >= 0.58:  return "BUY"
    if prob <= 0.30:  return "STRONG SELL"
    if prob <= 0.42:  return "SELL"
    return "HOLD"


def predict(
    ticker: str,
    feature_df: pd.DataFrame,
    feature_cols: list[str],
    model_type: str = "xgb",
) -> pd.DataFrame:
    """
    Run inference and return a DataFrame with:
    Date, close, prediction, probability, signal
    Supports model_type: 'xgb', 'lgbm', 'ensemble'
    """
    scaler = load_scaler(ticker)

    if model_type == "ensemble":
        xgb_model  = load_local_model(ticker, "xgb")
        lgbm_model = load_local_model(ticker, "lgbm")
        X = feature_df[feature_cols].values
        if scaler is not None:
            X = scaler.transform(X)
        probs = (xgb_model.predict_proba(X)[:, 1] +
                 lgbm_model.predict_proba(X)[:, 1]) / 2
    else:
        model = load_local_model(ticker, model_type)
        X = feature_df[feature_cols].values
        if scaler is not None:
            X = scaler.transform(X)
        probs = model.predict_proba(X)[:, 1]

    preds   = (probs >= 0.5).astype(int)
    signals = [_prob_to_signal(p) for p in probs]

    return pd.DataFrame({
        "Date":        feature_df.index,
        "close":       feature_df["Close"].values,
        "prediction":  preds,
        "probability": probs,
        "signal":      signals,
    }).set_index("Date")


def get_latest_signal(ticker: str, feature_df: pd.DataFrame,
                      feature_cols: list[str], model_type: str = "xgb") -> dict:
    pred_df = predict(ticker, feature_df, feature_cols, model_type)
    latest  = pred_df.iloc[-1]
    return {
        "ticker":       ticker,
        "date":         str(pred_df.index[-1].date()),
        "close":        round(float(latest["close"]), 2),
        "signal":       str(latest["signal"]),
        "probability":  round(float(latest["probability"]), 4),
        "horizon_days": PREDICTION_HORIZON,
    }
