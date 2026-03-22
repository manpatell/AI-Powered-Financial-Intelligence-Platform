"""
Data drift and model performance monitoring using Evidently AI.
Compares reference (train) vs current (recent) data distributions
and flags features that have drifted significantly.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

from finai.config.settings import MODELS_DIR, PROCESSED_DIR
from finai.utils.logger import get_logger

logger = get_logger(__name__)


# ── Statistical drift tests ───────────────────────────────────────────────────

def ks_drift(reference: pd.Series, current: pd.Series, threshold: float = 0.05) -> dict:
    """Kolmogorov-Smirnov test for distribution drift."""
    ref_clean = reference.dropna()
    cur_clean = current.dropna()
    if len(ref_clean) < 10 or len(cur_clean) < 10:
        return {"drifted": False, "p_value": 1.0, "statistic": 0.0, "test": "ks"}
    stat, p_val = stats.ks_2samp(ref_clean, cur_clean)
    return {
        "drifted":   bool(p_val < threshold),
        "p_value":   round(float(p_val), 6),
        "statistic": round(float(stat), 6),
        "test":      "ks",
    }


def psi_score(reference: pd.Series, current: pd.Series, bins: int = 10) -> dict:
    """Population Stability Index — detects distribution shift."""
    ref_clean = reference.dropna().values
    cur_clean = current.dropna().values
    if len(ref_clean) == 0 or len(cur_clean) == 0:
        return {"psi": 0.0, "drifted": False}

    breakpoints = np.percentile(ref_clean, np.linspace(0, 100, bins + 1))
    breakpoints = np.unique(breakpoints)
    if len(breakpoints) < 2:
        return {"psi": 0.0, "drifted": False}

    ref_counts = np.histogram(ref_clean, bins=breakpoints)[0]
    cur_counts = np.histogram(cur_clean, bins=breakpoints)[0]

    ref_pct = ref_counts / len(ref_clean) + 1e-10
    cur_pct = cur_counts / len(cur_clean) + 1e-10

    psi = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))

    # PSI thresholds: <0.1 = stable, 0.1–0.2 = moderate, >0.2 = significant
    return {
        "psi":     round(psi, 6),
        "drifted": psi > 0.2,
        "level":   "stable" if psi < 0.1 else "moderate" if psi < 0.2 else "significant",
    }


# ── Main detector ─────────────────────────────────────────────────────────────

class DriftDetector:
    """Compare reference and current feature DataFrames for drift."""

    def __init__(self, ticker: str):
        self.ticker = ticker
        self._report_path = MODELS_DIR / f"{ticker}_drift_report.json"

    def _load_processed(self) -> Optional[pd.DataFrame]:
        path = PROCESSED_DIR / f"{self.ticker}_features.parquet"
        if not path.exists():
            return None
        return pd.read_parquet(path)

    def run(
        self,
        reference_df: Optional[pd.DataFrame] = None,
        current_df: Optional[pd.DataFrame] = None,
        feature_cols: Optional[list[str]] = None,
        split: float = 0.7,
    ) -> dict:
        """
        Run drift detection.

        If reference_df / current_df not supplied, loads the saved parquet
        and splits it at `split` fraction (train vs recent).

        Returns a dict with per-feature drift stats + summary.
        """
        if reference_df is None or current_df is None:
            full_df = self._load_processed()
            if full_df is None:
                return {"error": f"No processed data for {self.ticker}. Run feature pipeline first."}
            cutoff = int(len(full_df) * split)
            reference_df = full_df.iloc[:cutoff]
            current_df   = full_df.iloc[cutoff:]

        if feature_cols is None:
            exclude = {"Open", "High", "Low", "Close", "Volume", "ticker", "target"}
            feature_cols = [c for c in reference_df.columns if c not in exclude]

        feature_results = {}
        drifted_count = 0

        for feat in feature_cols:
            if feat not in reference_df.columns or feat not in current_df.columns:
                continue
            ks  = ks_drift(reference_df[feat], current_df[feat])
            psi = psi_score(reference_df[feat], current_df[feat])

            drifted = ks["drifted"] or psi["drifted"]
            if drifted:
                drifted_count += 1

            feature_results[feat] = {
                "ks":  ks,
                "psi": psi,
                "drifted": drifted,
                "ref_mean": round(float(reference_df[feat].mean()), 6),
                "cur_mean": round(float(current_df[feat].mean()), 6),
                "ref_std":  round(float(reference_df[feat].std()),  6),
                "cur_std":  round(float(current_df[feat].std()),   6),
            }

        total = len(feature_results)
        report = {
            "ticker":          self.ticker,
            "n_features":      total,
            "n_drifted":       drifted_count,
            "drift_rate":      round(drifted_count / total, 4) if total else 0,
            "ref_rows":        len(reference_df),
            "cur_rows":        len(current_df),
            "features":        feature_results,
        }

        # Persist
        with open(self._report_path, "w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Drift report: {drifted_count}/{total} features drifted for {self.ticker}")
        return report

    def load_report(self) -> Optional[dict]:
        if self._report_path.exists():
            with open(self._report_path) as f:
                return json.load(f)
        return None
