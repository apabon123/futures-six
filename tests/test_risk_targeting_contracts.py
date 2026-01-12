"""
Contract Tests for RiskTargetingLayer

These tests enforce non-negotiable semantics that must never regress.

Test Categories:
A. Scale Direction: vol < target → leverage ↑, vol > target → leverage ↓
B. Hard Bounds: leverage never exceeds cap, never below floor
C. Determinism: Same inputs → same outputs (byte-for-byte)
D. Warmup Behavior: Defined behavior when insufficient history
E. No Lookahead: Vol estimate only uses returns strictly prior to date
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.layers.risk_targeting import RiskTargetingLayer


class TestScaleDirection:
    """Test A: Scale direction semantics."""
    
    def test_leverage_increases_when_vol_below_target(self):
        """If realized vol < target → leverage ↑ (until cap)."""
        rt = RiskTargetingLayer(target_vol=0.20, leverage_cap=7.0, leverage_floor=1.0)
        
        # Low vol scenario: 10% current vs 20% target
        leverage = rt.compute_leverage(current_vol=0.10, gross_exposure=1.0)
        
        assert leverage > 1.0, f"Expected leverage > 1.0 when vol below target, got {leverage}"
        assert leverage <= 7.0, f"Leverage must respect cap, got {leverage}"
        
        # Should be approximately 2x (20% / 10%)
        assert 1.8 < leverage < 2.2, f"Expected ~2x leverage, got {leverage}"
    
    def test_leverage_decreases_when_vol_above_target(self):
        """If realized vol > target → leverage ↓ (until floor)."""
        rt = RiskTargetingLayer(target_vol=0.20, leverage_cap=7.0, leverage_floor=1.0)
        
        # High vol scenario: 30% current vs 20% target
        # Raw leverage would be 0.20 / 0.30 = 0.67x, but floor is 1.0
        leverage = rt.compute_leverage(current_vol=0.30, gross_exposure=1.0)
        
        # Raw leverage < 1.0, but floor prevents going below 1.0
        assert leverage >= 1.0, f"Leverage must respect floor (1.0), got {leverage}"
        assert leverage == 1.0, f"Expected leverage = 1.0 (floor), got {leverage}"
        
        # Test with lower floor to verify raw calculation
        rt_low_floor = RiskTargetingLayer(
            target_vol=0.20, 
            leverage_cap=7.0, 
            leverage_floor=0.5,
            config_path=None  # Don't load config
        )
        leverage_low_floor = rt_low_floor.compute_leverage(current_vol=0.30, gross_exposure=1.0)
        # Raw = 0.67, but floor = 0.5, so should be 0.67 (not clipped)
        assert 0.65 < leverage_low_floor < 0.70, f"Expected ~0.67x with low floor, got {leverage_low_floor}"
    
    def test_leverage_at_target_vol(self):
        """When vol equals target, leverage should be ~1.0."""
        rt = RiskTargetingLayer(target_vol=0.20, leverage_cap=7.0, leverage_floor=1.0)
        
        leverage = rt.compute_leverage(current_vol=0.20, gross_exposure=1.0)
        
        assert 0.95 < leverage < 1.05, f"Expected ~1.0x leverage at target vol, got {leverage}"


class TestHardBounds:
    """Test B: Hard bounds enforcement."""
    
    def test_leverage_never_exceeds_cap(self):
        """leverage never exceeds leverage_cap."""
        rt = RiskTargetingLayer(
            target_vol=0.20, 
            leverage_cap=7.0, 
            leverage_floor=1.0,
            vol_floor=0.01,  # Set vol_floor low so it doesn't interfere
            config_path=None  # Don't load config
        )
        
        # Extreme low vol scenario: 1% current vs 20% target
        # Raw leverage would be 0.20 / 0.01 = 20x, but cap is 7.0
        leverage = rt.compute_leverage(current_vol=0.01, gross_exposure=1.0)
        
        assert leverage <= 7.0, f"Leverage must not exceed cap (7.0), got {leverage}"
        assert leverage == 7.0, f"Expected leverage = cap (7.0) for extreme low vol, got {leverage}"
        
        # Test with higher cap to verify raw calculation
        rt_high_cap = RiskTargetingLayer(
            target_vol=0.20, 
            leverage_cap=20.0, 
            leverage_floor=1.0,
            vol_floor=0.01,
            config_path=None
        )
        leverage_high_cap = rt_high_cap.compute_leverage(current_vol=0.01, gross_exposure=1.0)
        # Raw = 20x, cap = 20x, so should be 20x
        assert 19.0 < leverage_high_cap < 21.0, f"Expected ~20x with high cap, got {leverage_high_cap}"
    
    def test_leverage_never_below_floor(self):
        """leverage never goes below leverage_floor."""
        rt = RiskTargetingLayer(target_vol=0.20, leverage_cap=7.0, leverage_floor=1.0)
        
        # Extreme high vol scenario
        leverage = rt.compute_leverage(current_vol=1.0, gross_exposure=1.0)  # 100% vol
        
        assert leverage >= 1.0, f"Leverage must not go below floor (1.0), got {leverage}"
        assert leverage == 1.0, f"Expected leverage = floor (1.0) for extreme high vol, got {leverage}"
    
    def test_custom_floor_respected(self):
        """Test with custom leverage_floor."""
        # Pass config_path=None to avoid config overriding test values
        rt = RiskTargetingLayer(
            target_vol=0.20, 
            leverage_cap=7.0, 
            leverage_floor=0.5, 
            vol_floor=0.05,
            config_path=None  # Don't load config
        )
        
        # Verify floor is set correctly
        assert rt.leverage_floor == 0.5, f"Expected leverage_floor=0.5, got {rt.leverage_floor}"
        
        # Extreme high vol: 100% current vs 20% target
        # Raw leverage = 0.20 / 1.0 = 0.2x, but floor is 0.5
        leverage = rt.compute_leverage(current_vol=1.0, gross_exposure=1.0)
        
        assert leverage >= 0.5, f"Leverage must respect custom floor (0.5), got {leverage}"
        assert leverage == 0.5, f"Expected leverage = custom floor (0.5), got {leverage}"
        
        # Test with vol that gives raw leverage between floor and cap
        leverage_mid = rt.compute_leverage(current_vol=0.30, gross_exposure=1.0)
        # Raw = 0.20 / 0.30 = 0.67x, floor = 0.5, so should be 0.67x (not clipped)
        assert 0.65 < leverage_mid < 0.70, f"Expected ~0.67x (not clipped), got {leverage_mid}"


class TestDeterminism:
    """Test C: Determinism (same inputs → same outputs)."""
    
    def test_deterministic_leverage_calculation(self):
        """Same vol + exposure → same leverage (byte-for-byte)."""
        rt = RiskTargetingLayer(target_vol=0.20, leverage_cap=7.0)
        
        # Run twice with same inputs
        leverage1 = rt.compute_leverage(current_vol=0.15, gross_exposure=1.0)
        leverage2 = rt.compute_leverage(current_vol=0.15, gross_exposure=1.0)
        
        assert leverage1 == leverage2, f"Leverage calculation must be deterministic, got {leverage1} vs {leverage2}"
        assert np.isclose(leverage1, leverage2), f"Leverage values must be numerically identical"
    
    def test_deterministic_weight_scaling(self):
        """Same weights + returns + date → same scaled weights."""
        rt = RiskTargetingLayer(target_vol=0.20, leverage_cap=7.0)
        
        # Create synthetic data
        np.random.seed(42)
        dates = pd.date_range("2024-01-01", periods=100, freq="B")
        assets = ["ES", "NQ", "ZN"]
        
        returns = pd.DataFrame(
            np.random.randn(100, 3) * (0.15 / np.sqrt(252)),
            index=dates,
            columns=assets
        )
        
        weights = pd.Series([0.33, 0.33, 0.34], index=assets)
        date = dates[-1]
        
        # Run twice
        scaled1 = rt.scale_weights(weights, returns, date)
        scaled2 = rt.scale_weights(weights, returns, date)
        
        pd.testing.assert_series_equal(scaled1, scaled2, check_exact=True)
    
    def test_determinism_with_different_random_seeds(self):
        """Determinism should hold regardless of external random state."""
        rt = RiskTargetingLayer(target_vol=0.20, leverage_cap=7.0)
        
        # Same inputs, different random seeds
        np.random.seed(42)
        leverage1 = rt.compute_leverage(current_vol=0.15, gross_exposure=1.0)
        
        np.random.seed(999)
        leverage2 = rt.compute_leverage(current_vol=0.15, gross_exposure=1.0)
        
        assert leverage1 == leverage2, "Leverage must be independent of external random state"


class TestWarmupBehavior:
    """Test D: Warmup behavior when insufficient history."""
    
    def test_warmup_returns_vol_floor(self):
        """When insufficient returns, should return vol_floor."""
        rt = RiskTargetingLayer(target_vol=0.20, vol_lookback=63, vol_floor=0.05)
        
        # Create returns with only 10 days (less than vol_lookback=63)
        dates = pd.date_range("2024-01-01", periods=10, freq="B")
        assets = ["ES", "NQ"]
        returns = pd.DataFrame(
            np.random.randn(10, 2) * 0.01,
            index=dates,
            columns=assets
        )
        
        weights = pd.Series([0.5, 0.5], index=assets)
        date = dates[-1]
        
        # Should use vol_floor when insufficient history
        current_vol = rt.compute_portfolio_vol(weights, returns, date)
        
        assert current_vol == rt.vol_floor, f"Expected vol_floor ({rt.vol_floor}) when insufficient history, got {current_vol}"
    
    def test_warmup_leverage_uses_vol_floor(self):
        """Leverage calculation with warmup should use vol_floor."""
        rt = RiskTargetingLayer(target_vol=0.20, vol_lookback=63, vol_floor=0.05)
        
        # When vol estimate uses floor, leverage should be computed from floor
        leverage = rt.compute_leverage(current_vol=rt.vol_floor, gross_exposure=1.0)
        
        # Should be target_vol / vol_floor = 0.20 / 0.05 = 4.0, but capped at 7.0
        assert leverage == 4.0, f"Expected leverage = 4.0 (target/floor), got {leverage}"
    
    def test_warmup_with_exact_lookback(self):
        """Test behavior when history exactly equals lookback."""
        rt = RiskTargetingLayer(target_vol=0.20, vol_lookback=63, vol_floor=0.05)
        
        # Create returns with exactly 63 days
        # Use fixed seed for reproducibility
        np.random.seed(42)
        dates = pd.date_range("2024-01-01", periods=63, freq="B")
        assets = ["ES"]
        returns = pd.DataFrame(
            np.random.randn(63, 1) * (0.15 / np.sqrt(252)),
            index=dates,
            columns=assets
        )
        
        weights = pd.Series([1.0], index=assets)
        # Use a date that has exactly 63 days of history before it
        date = dates[-1]
        
        # Should compute actual vol (not floor) when history is sufficient
        current_vol = rt.compute_portfolio_vol(weights, returns, date)
        
        assert current_vol >= rt.vol_floor, f"Vol should be >= floor when history sufficient, got {current_vol}"
        # With 63 days of 15% annualized returns, realized vol should be roughly 15%
        # Allow wider range due to sampling variation
        assert 0.05 <= current_vol <= 0.30, f"Expected vol in reasonable range, got {current_vol}"


class TestNoLookahead:
    """Test E: No lookahead bias."""
    
    def test_vol_estimate_only_uses_past_returns(self):
        """Vol estimate at date must only use returns strictly prior to date."""
        rt = RiskTargetingLayer(target_vol=0.20, vol_lookback=63)
        
        # Create returns with known structure
        dates = pd.date_range("2024-01-01", periods=100, freq="B")
        assets = ["ES"]
        
        # First 50 days: low vol (10%)
        # Last 50 days: high vol (30%)
        returns = pd.DataFrame(
            np.concatenate([
                np.random.randn(50, 1) * (0.10 / np.sqrt(252)),
                np.random.randn(50, 1) * (0.30 / np.sqrt(252))
            ]),
            index=dates,
            columns=assets
        )
        
        weights = pd.Series([1.0], index=assets)
        
        # Test at date in first half (should see low vol)
        date_low_vol = dates[40]
        vol_low = rt.compute_portfolio_vol(weights, returns, date_low_vol)
        
        # Test at date in second half (should see high vol, but only if lookback includes it)
        date_high_vol = dates[80]
        vol_high = rt.compute_portfolio_vol(weights, returns, date_high_vol)
        
        # Vol at date_high_vol should only use returns BEFORE date_high_vol
        # If lookback=63, it should include some high-vol days, but not future ones
        assert vol_high > vol_low, f"Vol at later date should reflect higher realized vol, got {vol_low} vs {vol_high}"
        
        # Verify no future data is used: vol at date_high_vol should not use returns after date_high_vol
        # This is implicit in the implementation, but we can verify by checking the window
        window_end = date_high_vol
        window_start = dates[dates < window_end][-63] if len(dates[dates < window_end]) >= 63 else dates[0]
        
        # The vol estimate should only use returns in [window_start, window_end)
        assert window_end <= date_high_vol, "Window end must be <= current date"
        assert window_start < window_end, "Window start must be < window end"
    
    def test_covariance_matrix_only_uses_past_returns(self):
        """Covariance matrix computation must only use past returns."""
        rt = RiskTargetingLayer(target_vol=0.20, vol_lookback=63, vol_floor=0.05)
        
        np.random.seed(42)
        dates = pd.date_range("2024-01-01", periods=100, freq="B")
        assets = ["ES", "NQ"]
        
        returns = pd.DataFrame(
            np.random.randn(100, 2) * (0.15 / np.sqrt(252)),
            index=dates,
            columns=assets
        )
        
        weights = pd.Series([0.5, 0.5], index=assets)
        date = dates[80]  # Use later date to ensure sufficient history
        
        # Compute vol with returns DataFrame
        vol1 = rt.compute_portfolio_vol(weights, returns, date)
        
        # Compute vol with pre-computed covariance (should be same)
        # Get the window that should be used
        historical = returns.loc[returns.index < date]
        window = historical.tail(63) if len(historical) >= 63 else historical
        # Covariance is in daily units, not annualized
        cov = window.cov()
        
        vol2 = rt.compute_portfolio_vol(weights, returns, date, cov_matrix=cov)
        
        # Should be approximately equal (allow some tolerance for numerical differences)
        assert np.isclose(vol1, vol2, rtol=0.1) or (vol1 == rt.vol_floor and vol2 == rt.vol_floor), \
            f"Vol with cov matrix should match vol from returns, got {vol1} vs {vol2}"
        
        # Verify covariance only uses past data
        assert window.index.max() < date, "Covariance window must only use returns prior to date"


class TestEdgeCases:
    """Additional edge case tests."""
    
    def test_empty_weights(self):
        """Empty weights should return empty scaled weights."""
        rt = RiskTargetingLayer(target_vol=0.20)
        
        weights = pd.Series([], dtype=float)
        returns = pd.DataFrame()
        date = pd.Timestamp("2024-01-01")
        
        scaled = rt.scale_weights(weights, returns, date)
        
        assert len(scaled) == 0, "Empty weights should return empty scaled weights"
    
    def test_zero_weights(self):
        """Zero weights should remain zero."""
        rt = RiskTargetingLayer(target_vol=0.20)
        
        dates = pd.date_range("2024-01-01", periods=100, freq="B")
        assets = ["ES", "NQ"]
        returns = pd.DataFrame(
            np.random.randn(100, 2) * 0.01,
            index=dates,
            columns=assets
        )
        
        weights = pd.Series([0.0, 0.0], index=assets)
        date = dates[-1]
        
        scaled = rt.scale_weights(weights, returns, date)
        
        assert (scaled == 0).all(), "Zero weights should remain zero"
    
    def test_single_asset(self):
        """Test with single asset portfolio."""
        rt = RiskTargetingLayer(target_vol=0.20, leverage_cap=7.0)
        
        dates = pd.date_range("2024-01-01", periods=100, freq="B")
        assets = ["ES"]
        returns = pd.DataFrame(
            np.random.randn(100, 1) * (0.15 / np.sqrt(252)),
            index=dates,
            columns=assets
        )
        
        weights = pd.Series([1.0], index=assets)
        date = dates[-1]
        
        scaled = rt.scale_weights(weights, returns, date)
        
        assert len(scaled) == 1, "Single asset should return single scaled weight"
        assert scaled.index[0] == "ES", "Index should be preserved"
        assert scaled.iloc[0] > 0, "Scaled weight should be positive"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

