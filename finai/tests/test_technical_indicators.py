"""Unit tests for finai.features.technical_indicators."""
import numpy as np
import pandas as pd
import pytest

from finai.features.technical_indicators import (
    add_technical_indicators,
    add_target,
    get_feature_columns,
)


class TestAddTechnicalIndicators:
    def test_returns_dataframe(self, sample_ohlcv):
        df = add_technical_indicators(sample_ohlcv)
        assert isinstance(df, pd.DataFrame)

    def test_does_not_mutate_input(self, sample_ohlcv):
        original_cols = set(sample_ohlcv.columns)
        add_technical_indicators(sample_ohlcv)
        assert set(sample_ohlcv.columns) == original_cols

    def test_contains_log_return_columns(self, sample_ohlcv):
        df = add_technical_indicators(sample_ohlcv)
        for col in ["ret_1d", "ret_2d", "ret_3d", "ret_5d", "ret_10d", "ret_20d"]:
            assert col in df.columns, f"Missing column: {col}"

    def test_contains_rsi_columns(self, sample_ohlcv):
        df = add_technical_indicators(sample_ohlcv)
        for col in ["rsi_14", "rsi_7", "rsi_21"]:
            assert col in df.columns, f"Missing column: {col}"

    def test_rsi_normalised_0_to_1(self, sample_ohlcv):
        df = add_technical_indicators(sample_ohlcv).dropna(subset=["rsi_14"])
        assert df["rsi_14"].between(0, 1).all(), "rsi_14 should be in [0, 1]"

    def test_contains_sma_ratio_and_chart_columns(self, sample_ohlcv):
        df = add_technical_indicators(sample_ohlcv)
        for col in ["sma10_ratio", "sma20_ratio", "sma50_ratio",
                    "sma_10", "sma_20", "sma_50"]:
            assert col in df.columns, f"Missing column: {col}"

    def test_contains_bollinger_band_columns(self, sample_ohlcv):
        df = add_technical_indicators(sample_ohlcv)
        for col in ["bb_pct", "bb_width", "bb_upper", "bb_lower"]:
            assert col in df.columns, f"Missing column: {col}"

    def test_bb_upper_above_lower(self, sample_ohlcv):
        df = add_technical_indicators(sample_ohlcv).dropna(subset=["bb_upper", "bb_lower"])
        assert (df["bb_upper"] >= df["bb_lower"]).all()

    def test_contains_macd_columns(self, sample_ohlcv):
        df = add_technical_indicators(sample_ohlcv)
        for col in ["macd", "macd_signal", "macd_diff",
                    "macd_ratio", "macd_sig_ratio", "macd_diff_ratio"]:
            assert col in df.columns, f"Missing column: {col}"

    def test_contains_volume_columns(self, sample_ohlcv):
        df = add_technical_indicators(sample_ohlcv)
        for col in ["vol_ratio", "obv_slope", "vwap_ratio"]:
            assert col in df.columns, f"Missing column: {col}"

    def test_contains_calendar_columns(self, sample_ohlcv):
        df = add_technical_indicators(sample_ohlcv)
        assert "day_of_week" in df.columns
        assert "month" in df.columns

    def test_handles_missing_volume(self, sample_ohlcv):
        df_no_vol = sample_ohlcv.copy()
        df_no_vol["Volume"] = np.nan
        # Should either drop rows or raise — not crash silently with wrong output
        result = add_technical_indicators(df_no_vol)
        assert isinstance(result, pd.DataFrame)


class TestAddTarget:
    def test_adds_target_column(self, sample_ohlcv):
        featured = add_technical_indicators(sample_ohlcv)
        result = add_target(featured, horizon=5)
        assert "target" in result.columns

    def test_target_is_binary(self, sample_ohlcv):
        featured = add_technical_indicators(sample_ohlcv)
        result = add_target(featured, horizon=5)
        assert set(result["target"].unique()).issubset({0, 1})

    def test_threshold_filters_small_moves(self, sample_ohlcv):
        featured = add_technical_indicators(sample_ohlcv)
        result_no_thresh = add_target(featured, horizon=5, threshold=0.0)
        result_with_thresh = add_target(featured, horizon=5, threshold=0.005)
        assert len(result_with_thresh) <= len(result_no_thresh)

    def test_drops_nan_targets(self, sample_ohlcv):
        featured = add_technical_indicators(sample_ohlcv)
        result = add_target(featured, horizon=5)
        assert result["target"].isna().sum() == 0


class TestGetFeatureColumns:
    def test_excludes_ohlcv(self, sample_feature_df):
        cols = get_feature_columns(sample_feature_df)
        for excluded in ["Open", "High", "Low", "Close", "Volume"]:
            assert excluded not in cols

    def test_excludes_ticker_and_target(self, sample_feature_df):
        df = sample_feature_df.copy()
        df["target"] = 0
        cols = get_feature_columns(df)
        assert "ticker" not in cols
        assert "target" not in cols

    def test_excludes_chart_only_columns(self, sample_feature_df):
        cols = get_feature_columns(sample_feature_df)
        for chart_col in ["sma_10", "sma_20", "sma_50", "ema_10", "ema_20",
                           "bb_upper", "bb_lower", "macd", "macd_signal", "macd_diff"]:
            assert chart_col not in cols, f"Chart-only column should be excluded: {chart_col}"

    def test_returns_list(self, sample_feature_df):
        cols = get_feature_columns(sample_feature_df)
        assert isinstance(cols, list)
        assert len(cols) > 0
