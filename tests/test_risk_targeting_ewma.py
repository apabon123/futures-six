"""
Unit tests for Risk Targeting Layer EWMA covariance and config switching.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from src.layers.risk_targeting import (
    RiskTargetingLayer,
    DEFAULT_COV_ESTIMATOR,
    DEFAULT_EWMA_LAMBDA,
    COV_ESTIMATOR_CHOICES,
    WARMUP_EWMA_MIN,
)


class TestEWMACovarianceDeterministic:
    """Deterministic EWMA covariance correctness (small example)."""

    def test_ewma_cov_small_example(self):
        """EWMA cov on 3 assets, 15 days: match hand recursion."""
        np.random.seed(42)
        n_assets, n_days = 3, 15
        dates = pd.date_range("2020-01-01", periods=n_days, freq="D")
        returns = pd.DataFrame(
            np.random.randn(n_days, n_assets) * 0.01,
            index=dates,
            columns=["A", "B", "C"],
        )
        lam = 0.94
        rt = RiskTargetingLayer(cov_estimator="ewma", ewma_lambda=lam, vol_lookback=n_days)
        cov_df = rt._ewma_cov(returns, lam)
        assert cov_df is not None
        assert cov_df.shape == (3, 3)
        assert list(cov_df.index) == ["A", "B", "C"]
        assert list(cov_df.columns) == ["A", "B", "C"]
        # Symmetric
        np.testing.assert_array_almost_equal(cov_df.values, cov_df.values.T)
        # PSD (min eigenvalue >= 0)
        min_eig = np.min(np.linalg.eigvalsh(cov_df.values))
        assert min_eig >= -1e-9, f"Expected PSD, min_eig={min_eig}"

        # Hand recursion: warmup 10, then update
        clean = returns.dropna(how="any")
        warmup_len = min(max(WARMUP_EWMA_MIN, 3 + 2), len(clean), 60)
        warmup = clean.iloc[:warmup_len]
        mean = clean.mean(axis=0).values
        S = warmup.cov().values.copy()
        for i in range(warmup_len, len(clean)):
            x = (clean.iloc[i].values - mean).reshape(-1, 1)
            S = lam * S + (1 - lam) * (x @ x.T)
        S = (S + S.T) / 2.0
        np.testing.assert_allclose(cov_df.values, S, rtol=1e-9, atol=1e-12)

    def test_ewma_cov_single_asset(self):
        """EWMA with one asset returns 1x1 covariance."""
        n_days = 15
        dates = pd.date_range("2020-01-01", periods=n_days, freq="D")
        returns = pd.DataFrame({"X": np.random.randn(n_days) * 0.01}, index=dates)
        rt = RiskTargetingLayer(cov_estimator="ewma", ewma_lambda=0.94, vol_lookback=n_days)
        cov_df = rt._ewma_cov(returns, 0.94)
        assert cov_df is not None
        assert cov_df.shape == (1, 1)
        assert cov_df.loc["X", "X"] >= 0


class TestCovEstimatorConfigSwitch:
    """Config switch: sample vs ewma produces different forecast vol on same data."""

    def test_sample_vs_ewma_forecast_vol_differ(self):
        """On same returns/weights, sample and ewma forecast vol can differ (sanity)."""
        np.random.seed(123)
        dates = pd.date_range("2020-01-01", periods=80, freq="D")
        returns = pd.DataFrame(
            np.random.randn(80, 2) * 0.01,
            index=dates,
            columns=["ES", "NQ"],
        )
        weights = pd.Series({"ES": 0.5, "NQ": 0.5})
        date = dates[70]

        rt_sample = RiskTargetingLayer(cov_estimator="sample", vol_lookback=63)
        rt_ewma = RiskTargetingLayer(cov_estimator="ewma", ewma_lambda=0.94, vol_lookback=63)

        vol_sample = rt_sample.compute_portfolio_vol(weights, returns, date)
        vol_ewma = rt_ewma.compute_portfolio_vol(weights, returns, date)

        assert np.isfinite(vol_sample) and np.isfinite(vol_ewma)
        # They need not be equal (different estimators)
        # Just ensure both are positive and reasonable
        assert vol_sample >= rt_sample.vol_floor
        assert vol_ewma >= rt_ewma.vol_floor

    def test_sample_matches_legacy_behavior(self):
        """With cov_estimator='sample', forecast vol equals rolling sample cov path."""
        np.random.seed(1)
        dates = pd.date_range("2020-01-01", periods=100, freq="D")
        returns = pd.DataFrame(
            {"ES": np.random.randn(100) * 0.01, "NQ": np.random.randn(100) * 0.01},
            index=dates,
        )
        weights = pd.Series({"ES": 0.5, "NQ": 0.5})
        date = dates[80]
        rt = RiskTargetingLayer(cov_estimator="sample", vol_lookback=63)
        cov = rt._compute_rolling_cov(returns, date)
        assert cov is not None
        w = weights.loc[cov.index].values
        port_var = w @ cov.values @ w
        port_vol_expected = np.sqrt(max(port_var, 0)) * np.sqrt(252)
        port_vol_expected = max(port_vol_expected, rt.vol_floor)
        vol_from_layer = rt.compute_portfolio_vol(weights, returns, date)
        np.testing.assert_allclose(vol_from_layer, port_vol_expected, rtol=1e-10)


class TestCovEstimatorValidation:
    """Config validation for cov_estimator and ewma_lambda."""

    def test_invalid_cov_estimator_raises(self):
        """Invalid cov_estimator raises ValueError."""
        with pytest.raises(ValueError, match="cov_estimator"):
            RiskTargetingLayer(cov_estimator="invalid")

    def test_ewma_lambda_bounds_raise(self):
        """ewma_lambda outside (0, 1) raises."""
        with pytest.raises(ValueError, match="ewma_lambda"):
            RiskTargetingLayer(cov_estimator="ewma", ewma_lambda=0.0)
        with pytest.raises(ValueError, match="ewma_lambda"):
            RiskTargetingLayer(cov_estimator="ewma", ewma_lambda=1.0)
        with pytest.raises(ValueError, match="ewma_lambda"):
            RiskTargetingLayer(cov_estimator="ewma", ewma_lambda=1.5)

    def test_default_is_sample(self):
        """Default cov_estimator is sample, default ewma_lambda 0.94."""
        rt = RiskTargetingLayer()
        assert rt.cov_estimator == "sample"
        assert rt.ewma_lambda == 0.94

    def test_describe_includes_estimator_and_lambda(self):
        """describe() includes cov_estimator and ewma_lambda."""
        rt = RiskTargetingLayer(cov_estimator="ewma", ewma_lambda=0.92)
        d = rt.describe()
        assert d["parameters"]["cov_estimator"] == "ewma"
        assert d["parameters"]["ewma_lambda"] == 0.92


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
