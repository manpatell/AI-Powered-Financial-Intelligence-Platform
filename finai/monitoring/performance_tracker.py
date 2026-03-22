"""
Model performance tracker — logs rolling accuracy/AUC over time
so you can see if the model degrades as market regimes shift.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

from finai.config.settings import MODELS_DIR
from finai.utils.logger import get_logger

logger = get_logger(__name__)


class PerformanceTracker:
    """Tracks rolling model performance over recent prediction windows."""

    def __init__(self, ticker: str, model_type: str = "xgb", window: int = 30):
        self.ticker     = ticker
        self.model_type = model_type
        self.window     = window
        self._log_path  = MODELS_DIR / f"{ticker}_{model_type}_perf_log.json"

    def _load_log(self) -> list[dict]:
        if self._log_path.exists():
            with open(self._log_path) as f:
                return json.load(f)
        return []

    def _save_log(self, log: list[dict]) -> None:
        with open(self._log_path, "w") as f:
            json.dump(log, f, indent=2)

    def record(self, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
        """Compute and persist a performance snapshot."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "n_samples": int(len(y_true)),
            "accuracy":  round(float(accuracy_score(y_true, y_pred)), 4),
        }
        try:
            entry["roc_auc"] = round(float(roc_auc_score(y_true, y_prob)), 4)
        except Exception:
            entry["roc_auc"] = None

        log = self._load_log()
        log.append(entry)
        self._save_log(log)
        logger.info(f"Perf snapshot: acc={entry['accuracy']:.4f} auc={entry.get('roc_auc')}")
        return entry

    def get_history(self) -> pd.DataFrame:
        """Return performance history as a DataFrame."""
        log = self._load_log()
        if not log:
            return pd.DataFrame(columns=["timestamp", "n_samples", "accuracy", "roc_auc"])
        df = pd.DataFrame(log)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df.sort_values("timestamp")

    def rolling_stats(self) -> dict:
        """Return mean/std of last `window` snapshots."""
        df = self.get_history().tail(self.window)
        if df.empty:
            return {}
        return {
            "accuracy_mean": round(float(df["accuracy"].mean()), 4),
            "accuracy_std":  round(float(df["accuracy"].std()),  4),
            "roc_auc_mean":  round(float(df["roc_auc"].dropna().mean()), 4) if df["roc_auc"].notna().any() else None,
            "n_snapshots":   len(df),
        }
