"""
End-to-end feature pipeline.
Orchestrates: fetch → indicators → sentiment → target → scale → persist.
"""
from __future__ import annotations

import joblib
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import RobustScaler

from finai.config.settings import PROCESSED_DIR, MODELS_DIR, PREDICTION_HORIZON
from finai.data.stock_fetcher import fetch_stock_data
from finai.data.news_fetcher import fetch_all_news
from finai.features.technical_indicators import add_technical_indicators, add_target, get_feature_columns
from finai.features.sentiment_features import merge_sentiment_features
from finai.utils.logger import get_logger

logger = get_logger(__name__)

# Map ticker → company name for better news search
TICKER_NAMES = {
    "AAPL": "Apple", "MSFT": "Microsoft", "GOOGL": "Google Alphabet",
    "AMZN": "Amazon", "TSLA": "Tesla", "NVDA": "NVIDIA",
    "META": "Meta Platforms", "JPM": "JPMorgan Chase", "NFLX": "Netflix", "AMD": "AMD",
}


def build_features(
    ticker: str,
    period: str = "2y",
    use_cache: bool = True,
    include_sentiment: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Full feature pipeline for one ticker.

    Returns
    -------
    df          : DataFrame with features + target column
    feature_cols: ordered list of feature column names
    """
    logger.info(f"Building features for {ticker}")

    # 1. OHLCV
    raw = fetch_stock_data(ticker, period=period, use_cache=use_cache)

    # 2. Technical indicators
    featured = add_technical_indicators(raw)

    # 3. Sentiment (optional — skipped if no news available)
    if include_sentiment:
        company = TICKER_NAMES.get(ticker, ticker)
        news_df = fetch_all_news(ticker, company_name=company, use_cache=use_cache)
        featured = merge_sentiment_features(featured, news_df)

    # 4. Target label
    featured = add_target(featured, horizon=PREDICTION_HORIZON)

    # 5. Drop remaining NaNs (from rolling windows)
    feat_cols = get_feature_columns(featured)
    featured = featured.dropna(subset=feat_cols)

    logger.info(f"{ticker}: {len(featured)} rows, {len(feat_cols)} features")

    # 6. Persist processed data
    out_path = PROCESSED_DIR / f"{ticker}_features.parquet"
    featured.to_parquet(out_path)
    logger.debug(f"Saved processed features → {out_path}")

    return featured, feat_cols


def build_train_test(
    ticker: str,
    test_size: float = 0.2,
    scale: bool = True,
    **kwargs,
) -> dict:
    """
    Build train/test splits with optional RobustScaler.

    Returns dict with keys:
      X_train, X_test, y_train, y_test, feature_cols, scaler (or None)
    """
    df, feat_cols = build_features(ticker, **kwargs)

    split = int(len(df) * (1 - test_size))
    train_df = df.iloc[:split]
    test_df  = df.iloc[split:]

    X_train = train_df[feat_cols].values
    X_test  = test_df[feat_cols].values
    y_train = train_df["target"].values
    y_test  = test_df["target"].values

    scaler = None
    if scale:
        scaler = RobustScaler()
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)

        # Save scaler
        scaler_path = MODELS_DIR / f"{ticker}_scaler.joblib"
        joblib.dump(scaler, scaler_path)
        logger.debug(f"Saved scaler → {scaler_path}")

    return {
        "X_train": X_train,
        "X_test":  X_test,
        "y_train": y_train,
        "y_test":  y_test,
        "feature_cols": feat_cols,
        "scaler": scaler,
        "train_df": train_df,
        "test_df":  test_df,
    }
