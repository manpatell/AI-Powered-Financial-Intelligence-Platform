"""
Stock data fetcher using yfinance.
Handles downloading, caching, and validation of OHLCV data.
"""
import hashlib
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf

from finai.config.settings import CACHE_DIR, DEFAULT_INTERVAL, DEFAULT_PERIOD, RAW_DIR
from finai.utils.logger import get_logger

logger = get_logger(__name__)


def _cache_key(ticker: str, period: str, interval: str) -> Path:
    key = hashlib.md5(f"{ticker}-{period}-{interval}".encode()).hexdigest()[:10]
    return CACHE_DIR / f"{ticker}_{key}.pkl"


def _is_cache_fresh(path: Path, max_age_hours: int = 4) -> bool:
    if not path.exists():
        return False
    age = datetime.now() - datetime.fromtimestamp(path.stat().st_mtime)
    return age < timedelta(hours=max_age_hours)


def fetch_stock_data(
    ticker: str,
    period: str = DEFAULT_PERIOD,
    interval: str = DEFAULT_INTERVAL,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Download OHLCV data for a ticker.
    Returns a DataFrame indexed by Date with columns:
    Open, High, Low, Close, Volume, ticker.
    """
    cache_path = _cache_key(ticker, period, interval)

    if use_cache and _is_cache_fresh(cache_path):
        logger.debug(f"Cache hit for {ticker}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    logger.info(f"Fetching {ticker} ({period}, {interval})")
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
        if df.empty:
            raise ValueError(f"No data returned for {ticker}")

        # Flatten MultiIndex columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df.index = pd.to_datetime(df.index)
        df.index.name = "Date"
        df["ticker"] = ticker

        # Save raw copy
        raw_path = RAW_DIR / f"{ticker}_{period}_{interval}.parquet"
        df.to_parquet(raw_path)

        # Cache
        with open(cache_path, "wb") as f:
            pickle.dump(df, f)

        logger.info(f"Downloaded {len(df)} rows for {ticker}")
        return df

    except Exception as e:
        logger.error(f"Failed to fetch {ticker}: {e}")
        raise


def fetch_multiple_tickers(
    tickers: list[str],
    period: str = DEFAULT_PERIOD,
    interval: str = DEFAULT_INTERVAL,
    use_cache: bool = True,
) -> dict[str, pd.DataFrame]:
    """Fetch data for multiple tickers; returns {ticker: df} dict."""
    results = {}
    failed = []
    for ticker in tickers:
        try:
            results[ticker] = fetch_stock_data(ticker, period, interval, use_cache)
        except Exception:
            failed.append(ticker)
    if failed:
        logger.warning(f"Failed tickers: {failed}")
    return results


def get_ticker_info(ticker: str) -> dict:
    """Return basic company metadata from yfinance."""
    try:
        info = yf.Ticker(ticker).info
        return {
            "name": info.get("longName", ticker),
            "sector": info.get("sector", "N/A"),
            "industry": info.get("industry", "N/A"),
            "market_cap": info.get("marketCap", 0),
            "pe_ratio": info.get("trailingPE", None),
            "52w_high": info.get("fiftyTwoWeekHigh", None),
            "52w_low": info.get("fiftyTwoWeekLow", None),
            "description": info.get("longBusinessSummary", ""),
        }
    except Exception as e:
        logger.warning(f"Could not fetch info for {ticker}: {e}")
        return {"name": ticker}
