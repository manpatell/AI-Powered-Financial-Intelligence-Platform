"""
Technical indicator feature engineering.
Uses the `ta` library to compute 25+ indicators across
trend, momentum, volatility, and volume categories.
"""
import pandas as pd
import numpy as np
from ta import add_all_ta_features
from ta.trend import MACD, EMAIndicator, SMAIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator, VolumeWeightedAveragePrice

from finai.utils.logger import get_logger

logger = get_logger(__name__)


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add 25+ technical indicators to an OHLCV DataFrame.

    Expects columns: Open, High, Low, Close, Volume
    Returns the same DataFrame with additional feature columns.
    """
    df = df.copy()

    # Ensure numeric types
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Open", "High", "Low", "Close", "Volume"])

    close = df["Close"]
    high  = df["High"]
    low   = df["Low"]
    vol   = df["Volume"]

    # ── Trend ─────────────────────────────────────────────────────────────────
    df["sma_10"]  = SMAIndicator(close, window=10).sma_indicator()
    df["sma_20"]  = SMAIndicator(close, window=20).sma_indicator()
    df["sma_50"]  = SMAIndicator(close, window=50).sma_indicator()
    df["ema_10"]  = EMAIndicator(close, window=10).ema_indicator()
    df["ema_20"]  = EMAIndicator(close, window=20).ema_indicator()

    macd = MACD(close)
    df["macd"]        = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_diff"]   = macd.macd_diff()

    # Price vs moving averages (normalised)
    df["close_vs_sma20"] = (close - df["sma_20"]) / df["sma_20"]
    df["close_vs_sma50"] = (close - df["sma_50"]) / df["sma_50"]

    # ── Momentum ──────────────────────────────────────────────────────────────
    df["rsi_14"] = RSIIndicator(close, window=14).rsi()
    df["rsi_7"]  = RSIIndicator(close, window=7).rsi()

    stoch = StochasticOscillator(high, low, close, window=14, smooth_window=3)
    df["stoch_k"] = stoch.stoch()
    df["stoch_d"] = stoch.stoch_signal()

    df["roc_10"] = close.pct_change(periods=10) * 100   # Rate of Change

    # ── Volatility ────────────────────────────────────────────────────────────
    bb = BollingerBands(close, window=20, window_dev=2)
    df["bb_upper"]  = bb.bollinger_hband()
    df["bb_lower"]  = bb.bollinger_lband()
    df["bb_width"]  = bb.bollinger_wband()
    df["bb_pct"]    = bb.bollinger_pband()   # 0–1 position within band

    df["atr_14"] = AverageTrueRange(high, low, close, window=14).average_true_range()

    # Historical volatility (20-day rolling std of log returns)
    log_ret = np.log(close / close.shift(1))
    df["hist_vol_20"] = log_ret.rolling(20).std() * np.sqrt(252)

    # ── Volume ────────────────────────────────────────────────────────────────
    df["obv"]        = OnBalanceVolumeIndicator(close, vol).on_balance_volume()
    df["vol_sma_20"] = vol.rolling(20).mean()
    df["vol_ratio"]  = vol / df["vol_sma_20"]   # relative volume

    # VWAP (daily rolling)
    df["vwap"] = VWAP = VolumeWeightedAveragePrice(
        high, low, close, vol, window=14
    ).volume_weighted_average_price()

    # ── Price-derived ─────────────────────────────────────────────────────────
    df["daily_return"]  = close.pct_change()
    df["ret_5d"]        = close.pct_change(5)
    df["ret_10d"]       = close.pct_change(10)
    df["high_low_pct"]  = (high - low) / close
    df["open_close_pct"]= (close - df["Open"]) / df["Open"]

    logger.debug(f"Added {len(df.columns)} columns (including {len(df.columns)-7} features)")
    return df


def add_target(df: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    """
    Add binary classification target:
      1  if Close price is higher `horizon` days from now
      0  otherwise
    Drops the last `horizon` rows (no future available).
    """
    df = df.copy()
    df["future_close"] = df["Close"].shift(-horizon)
    df["target"] = (df["future_close"] > df["Close"]).astype(int)
    df = df.drop(columns=["future_close"]).dropna(subset=["target"])
    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return feature column names (excludes OHLCV, ticker, target, Date)."""
    exclude = {"Open", "High", "Low", "Close", "Volume", "ticker", "target"}
    return [c for c in df.columns if c not in exclude]
