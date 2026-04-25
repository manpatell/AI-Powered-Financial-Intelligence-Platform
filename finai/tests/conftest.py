"""Shared pytest fixtures for FinAI test suite."""
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    """Return a synthetic OHLCV DataFrame with 200 daily rows."""
    np.random.seed(42)
    n = 200
    dates = pd.date_range("2022-01-03", periods=n, freq="B")

    price = 100.0
    closes = []
    for _ in range(n):
        price *= 1 + np.random.normal(0, 0.01)
        closes.append(price)

    closes = np.array(closes)
    highs  = closes * (1 + np.abs(np.random.normal(0, 0.005, n)))
    lows   = closes * (1 - np.abs(np.random.normal(0, 0.005, n)))
    opens  = closes * (1 + np.random.normal(0, 0.003, n))
    volumes = np.random.randint(1_000_000, 10_000_000, n).astype(float)

    df = pd.DataFrame(
        {"Open": opens, "High": highs, "Low": lows, "Close": closes, "Volume": volumes},
        index=dates,
    )
    df.index.name = "Date"
    df["ticker"] = "TEST"
    return df


@pytest.fixture
def sample_feature_df(sample_ohlcv):
    """Return an OHLCV DataFrame with technical indicators added."""
    from finai.features.technical_indicators import add_technical_indicators
    return add_technical_indicators(sample_ohlcv)


@pytest.fixture
def sample_news_df() -> pd.DataFrame:
    """Return a minimal news DataFrame."""
    return pd.DataFrame(
        {
            "ticker":    ["TEST"] * 5,
            "title":     [
                "Test company reports strong earnings",
                "Market rally continues",
                "Analysts downgrade test stock",
                "New product launch announced",
                "Quarterly revenue beats expectations",
            ],
            "summary":   [""] * 5,
            "url":       [f"https://example.com/{i}" for i in range(5)],
            "source":    ["TestNews"] * 5,
            "published": pd.date_range("2024-01-01", periods=5, freq="D"),
        }
    )
