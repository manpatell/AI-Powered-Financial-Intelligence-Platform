"""
Technical indicator feature engineering.
All price-based features are ratio-normalised (divided by Close) so they
generalise across stocks with different price levels and over time.
"""
import pandas as pd
import numpy as np
from ta.trend import MACD, EMAIndicator, SMAIndicator
from ta.momentum import RSIIndicator, StochasticOscillator, ROCIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator, VolumeWeightedAveragePrice

from finai.utils.logger import get_logger

logger = get_logger(__name__)

# Minimum return magnitude to count as a directional signal (0 = no threshold)
RETURN_THRESHOLD = 0.005   # 0.5 % — flat moves excluded from training


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add 40+ normalised technical features to an OHLCV DataFrame.

    Price-based indicators are expressed as ratios to Close so that
    the model sees relative magnitude, not absolute dollar levels.
    """
    df = df.copy()
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Open", "High", "Low", "Close", "Volume"])

    close = df["Close"]
    high  = df["High"]
    low   = df["Low"]
    vol   = df["Volume"]

    # ── Log returns (primary signal source) ──────────────────────────────────
    log_ret = np.log(close / close.shift(1))
    df["ret_1d"]  = log_ret
    df["ret_2d"]  = np.log(close / close.shift(2))
    df["ret_3d"]  = np.log(close / close.shift(3))
    df["ret_5d"]  = np.log(close / close.shift(5))
    df["ret_10d"] = np.log(close / close.shift(10))
    df["ret_20d"] = np.log(close / close.shift(20))

    # ── Lagged returns (momentum memory) ─────────────────────────────────────
    for lag in [1, 2, 3, 5]:
        df[f"ret_1d_lag{lag}"] = df["ret_1d"].shift(lag)

    # ── Price position in recent range ────────────────────────────────────────
    for window in [10, 20, 50]:
        roll_max = close.rolling(window).max()
        roll_min = close.rolling(window).min()
        rng = (roll_max - roll_min).replace(0, np.nan)
        df[f"price_pos_{window}"] = (close - roll_min) / rng   # 0–1

    # ── Trend: moving averages as ratio to Close ──────────────────────────────
    for w in [10, 20, 50]:
        sma = SMAIndicator(close, window=w).sma_indicator()
        df[f"sma{w}_ratio"] = close / sma - 1          # positive = above MA

    for w in [10, 20]:
        ema = EMAIndicator(close, window=w).ema_indicator()
        df[f"ema{w}_ratio"] = close / ema - 1

    # SMA slope (rate of change of MA)
    sma20 = SMAIndicator(close, window=20).sma_indicator()
    df["sma20_slope"] = sma20.pct_change(5)

    # ── MACD (normalised by Close) ────────────────────────────────────────────
    macd_ind = MACD(close, window_slow=26, window_fast=12, window_sign=9)
    df["macd_ratio"]   = macd_ind.macd()        / close
    df["macd_sig_ratio"] = macd_ind.macd_signal() / close
    df["macd_diff_ratio"]= macd_ind.macd_diff()  / close

    # ── Momentum ──────────────────────────────────────────────────────────────
    df["rsi_14"]  = RSIIndicator(close, window=14).rsi() / 100   # normalise 0–1
    df["rsi_7"]   = RSIIndicator(close, window=7).rsi()  / 100
    df["rsi_21"]  = RSIIndicator(close, window=21).rsi() / 100

    # Lagged RSI
    df["rsi_14_lag1"] = df["rsi_14"].shift(1)
    df["rsi_14_lag3"] = df["rsi_14"].shift(3)

    stoch = StochasticOscillator(high, low, close, window=14, smooth_window=3)
    df["stoch_k"] = stoch.stoch()        / 100
    df["stoch_d"] = stoch.stoch_signal() / 100
    df["stoch_diff"] = df["stoch_k"] - df["stoch_d"]

    df["roc_5"]  = ROCIndicator(close, window=5).roc()  / 100
    df["roc_10"] = ROCIndicator(close, window=10).roc() / 100
    df["roc_20"] = ROCIndicator(close, window=20).roc() / 100

    # ── Volatility ────────────────────────────────────────────────────────────
    bb = BollingerBands(close, window=20, window_dev=2)
    df["bb_pct"]   = bb.bollinger_pband()    # position 0–1 within band
    df["bb_width"] = bb.bollinger_wband() / close    # width normalised

    df["atr_ratio"] = (
        AverageTrueRange(high, low, close, window=14).average_true_range() / close
    )

    # Historical volatility regimes
    df["hvol_10"]  = log_ret.rolling(10).std()  * np.sqrt(252)
    df["hvol_20"]  = log_ret.rolling(20).std()  * np.sqrt(252)
    df["hvol_60"]  = log_ret.rolling(60).std()  * np.sqrt(252)
    df["vol_regime"] = df["hvol_10"] / (df["hvol_60"] + 1e-9)   # short/long vol ratio

    # ── Volume ────────────────────────────────────────────────────────────────
    vol_ma20 = vol.rolling(20).mean()
    df["vol_ratio"]  = vol / (vol_ma20 + 1)
    df["vol_ratio_5"]= vol.rolling(5).mean() / (vol_ma20 + 1)

    obv = OnBalanceVolumeIndicator(close, vol).on_balance_volume()
    df["obv_slope"] = obv.pct_change(5)     # 5-day OBV momentum

    vwap = VolumeWeightedAveragePrice(high, low, close, vol, window=14)
    df["vwap_ratio"] = close / vwap.volume_weighted_average_price() - 1

    # ── Candle structure ──────────────────────────────────────────────────────
    df["body_ratio"]    = (close - df["Open"]).abs() / (high - low + 1e-9)
    df["upper_shadow"]  = (high - close.clip(lower=df["Open"])) / (high - low + 1e-9)
    df["lower_shadow"]  = (close.clip(upper=df["Open"]) - low)  / (high - low + 1e-9)
    df["gap"]           = (df["Open"] - close.shift(1)) / (close.shift(1) + 1e-9)

    # ── Calendar ──────────────────────────────────────────────────────────────
    df["day_of_week"]  = pd.to_datetime(df.index).dayofweek / 4    # 0–1
    df["month"]        = pd.to_datetime(df.index).month    / 11    # 0–1

    logger.debug(f"Feature matrix: {len(df.columns)} columns total")
    return df


def add_target(df: pd.DataFrame, horizon: int = 5,
               threshold: float = RETURN_THRESHOLD) -> pd.DataFrame:
    """
    Add a clean binary target using log-return over `horizon` days.

    If threshold > 0: rows with |return| < threshold are dropped —
    only clear directional moves are kept, reducing label noise.

    Label:
      1  →  future return > +threshold
      0  →  future return < -threshold
      (rows in the dead-zone between ±threshold are excluded)
    """
    df = df.copy()
    future_log_ret = np.log(df["Close"].shift(-horizon) / df["Close"])

    if threshold > 0:
        mask = future_log_ret.abs() >= threshold
        df = df[mask]
        future_log_ret = future_log_ret[mask]

    df["target"] = (future_log_ret > 0).astype(int)
    df = df.dropna(subset=["target"])
    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return feature column names (excludes OHLCV, ticker, target)."""
    exclude = {"Open", "High", "Low", "Close", "Volume", "ticker", "target"}
    return [c for c in df.columns if c not in exclude]
