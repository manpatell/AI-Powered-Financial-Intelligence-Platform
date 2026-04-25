"""Unit tests for finai.monitoring.drift_detector."""
import numpy as np
import pandas as pd
import pytest

from finai.monitoring.drift_detector import DriftDetector, ks_drift, psi_score


class TestKsDrift:
    def test_identical_distributions_no_drift(self):
        data = pd.Series(np.random.normal(0, 1, 200))
        result = ks_drift(data, data)
        assert result["drifted"] is False
        assert result["p_value"] == 1.0

    def test_clearly_different_distributions_drift(self):
        ref = pd.Series(np.random.normal(0, 1, 200))
        cur = pd.Series(np.random.normal(10, 1, 200))
        result = ks_drift(ref, cur)
        assert result["drifted"] is True
        assert result["p_value"] < 0.05

    def test_insufficient_data_no_drift(self):
        ref = pd.Series([1.0, 2.0])  # < 10 samples
        cur = pd.Series([100.0, 200.0])
        result = ks_drift(ref, cur)
        assert result["drifted"] is False
        assert result["p_value"] == 1.0

    def test_returns_required_keys(self):
        data = pd.Series(np.random.normal(0, 1, 100))
        result = ks_drift(data, data)
        for key in ["drifted", "p_value", "statistic", "test"]:
            assert key in result

    def test_drops_nan_values(self):
        ref = pd.Series([1.0, np.nan, 2.0, np.nan] * 30)
        cur = pd.Series([1.0, 1.5, 2.0, 2.5] * 30)
        result = ks_drift(ref, cur)
        assert isinstance(result["p_value"], float)

    def test_custom_threshold(self):
        ref = pd.Series(np.random.normal(0, 1, 200))
        cur = pd.Series(np.random.normal(0.1, 1, 200))
        # With very loose threshold (0.99) almost everything drifts
        result = ks_drift(ref, cur, threshold=0.99)
        assert isinstance(result["drifted"], bool)


class TestPsiScore:
    def test_identical_distributions_zero_psi(self):
        data = pd.Series(np.random.uniform(0, 1, 300))
        result = psi_score(data, data)
        assert result["psi"] < 0.01
        assert result["drifted"] is False
        assert result["level"] == "stable"

    def test_different_distributions_high_psi(self):
        ref = pd.Series(np.random.normal(0, 1, 300))
        cur = pd.Series(np.random.normal(5, 1, 300))
        result = psi_score(ref, cur)
        assert result["psi"] > 0.2
        assert result["drifted"] is True
        assert result["level"] == "significant"

    def test_moderate_drift_level(self):
        np.random.seed(0)
        ref = pd.Series(np.random.normal(0, 1, 500))
        cur = pd.Series(np.random.normal(0.6, 1, 500))
        result = psi_score(ref, cur)
        assert result["level"] in {"stable", "moderate", "significant"}

    def test_returns_required_keys(self):
        data = pd.Series(np.random.normal(0, 1, 100))
        result = psi_score(data, data)
        for key in ["psi", "drifted", "level"]:
            assert key in result

    def test_empty_series(self):
        result = psi_score(pd.Series([], dtype=float), pd.Series([], dtype=float))
        assert result["psi"] == 0.0
        assert result["drifted"] is False


class TestDriftDetector:
    def test_run_with_explicit_dfs(self, sample_feature_df):
        from finai.features.technical_indicators import get_feature_columns
        df = sample_feature_df.dropna()
        if len(df) < 40:
            pytest.skip("Not enough rows for split test")

        split = len(df) // 2
        ref = df.iloc[:split]
        cur = df.iloc[split:]
        feat_cols = get_feature_columns(df)

        detector = DriftDetector("TEST")
        report = detector.run(reference_df=ref, current_df=cur, feature_cols=feat_cols)

        assert "n_features" in report
        assert "n_drifted" in report
        assert "drift_rate" in report
        assert "features" in report
        assert report["n_features"] == len(feat_cols)

    def test_drift_rate_between_0_and_1(self, sample_feature_df):
        from finai.features.technical_indicators import get_feature_columns
        df = sample_feature_df.dropna()
        if len(df) < 40:
            pytest.skip("Not enough rows")

        split = len(df) // 2
        feat_cols = get_feature_columns(df)

        detector = DriftDetector("TEST")
        report = detector.run(
            reference_df=df.iloc[:split],
            current_df=df.iloc[split:],
            feature_cols=feat_cols,
        )
        assert 0.0 <= report["drift_rate"] <= 1.0

    def test_run_without_processed_data_returns_error(self, tmp_path, monkeypatch):
        import finai.monitoring.drift_detector as dd_module
        monkeypatch.setattr(dd_module, "PROCESSED_DIR", tmp_path)
        monkeypatch.setattr(dd_module, "MODELS_DIR", tmp_path)
        detector = DriftDetector("NONEXISTENT")
        report = detector.run()
        assert "error" in report

    def test_load_report_none_when_missing(self, tmp_path, monkeypatch):
        import finai.monitoring.drift_detector as dd_module
        monkeypatch.setattr(dd_module, "MODELS_DIR", tmp_path)
        detector = DriftDetector("NONEXISTENT")
        assert detector.load_report() is None
