"""
Tests for VolManagedOverlay agent.

Verifies that:
1. Scaled signals achieve target volatility (ex-ante)
2. Position bounds are respected
3. Leverage cap is enforced
4. Deterministic outputs given same inputs
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock

from src.agents.overlay_volmanaged import VolManagedOverlay


@pytest.fixture
def mock_market():
    """Create a mock MarketData instance."""
    market = Mock()
    market.universe = ['ES', 'NQ', 'GC', 'CL', 'TY']
    return market


@pytest.fixture
def mock_risk_vol():
    """Create a mock RiskVol instance with realistic vol/cov data."""
    risk_vol = Mock()
    
    # Mock volatilities: 15-30% annualized
    def mock_vols(market, date):
        return pd.Series({
            'ES': 0.18,
            'NQ': 0.25,
            'GC': 0.15,
            'CL': 0.30,
            'TY': 0.10
        })
    
    # Mock covariance matrix: realistic correlations
    def mock_covariance(market, date):
        symbols = ['ES', 'NQ', 'GC', 'CL', 'TY']
        vols = np.array([0.18, 0.25, 0.15, 0.30, 0.10])
        
        # Create correlation matrix
        corr = np.array([
            [1.00, 0.85, 0.20, 0.30, -0.10],
            [0.85, 1.00, 0.15, 0.25, -0.15],
            [0.20, 0.15, 1.00, 0.35, 0.05],
            [0.30, 0.25, 0.35, 1.00, 0.10],
            [-0.10, -0.15, 0.05, 0.10, 1.00]
        ])
        
        # Convert to covariance: Σ = diag(σ) * Corr * diag(σ)
        D = np.diag(vols)
        cov_matrix = D @ corr @ D
        
        return pd.DataFrame(cov_matrix, index=symbols, columns=symbols)
    
    risk_vol.vols = mock_vols
    risk_vol.covariance = mock_covariance
    risk_vol.vol_lookback = 63
    
    return risk_vol


@pytest.fixture
def sample_signals():
    """Create sample strategy signals."""
    return pd.Series({
        'ES': 1.5,
        'NQ': -0.8,
        'GC': 0.5,
        'CL': -1.2,
        'TY': 0.3
    })


def test_initialization_default(mock_risk_vol):
    """Test default initialization."""
    overlay = VolManagedOverlay(risk_vol=mock_risk_vol)
    
    assert overlay.target_vol == 0.20
    assert overlay.floor_vol == 0.05
    assert overlay.cap_leverage == 7.0
    assert overlay.leverage_mode == "global"
    assert overlay.position_bounds == (-3.0, 3.0)


def test_initialization_from_config(mock_risk_vol, tmp_path):
    """Test initialization from config file."""
    # Create temp config
    config_path = tmp_path / "test_strategies.yaml"
    config_path.write_text("""
vol_overlay:
  target_vol: 0.15
  lookback_vol: 42
  floor_vol: 0.03
  cap_leverage: 5.0
  leverage_mode: "per-asset"
  position_bounds: [-2.0, 2.0]
""")
    
    overlay = VolManagedOverlay(
        risk_vol=mock_risk_vol,
        config_path=str(config_path)
    )
    
    assert overlay.target_vol == 0.15
    assert overlay.lookback_vol == 42
    assert overlay.floor_vol == 0.03
    assert overlay.cap_leverage == 5.0
    assert overlay.leverage_mode == "per-asset"
    assert overlay.position_bounds == (-2.0, 2.0)


def test_initialization_validation(mock_risk_vol):
    """Test parameter validation."""
    with pytest.raises(ValueError, match="target_vol must be > 0"):
        VolManagedOverlay(risk_vol=mock_risk_vol, target_vol=-0.1)
    
    with pytest.raises(ValueError, match="floor_vol must be > 0"):
        VolManagedOverlay(risk_vol=mock_risk_vol, floor_vol=0)
    
    with pytest.raises(ValueError, match="cap_leverage must be > 0"):
        VolManagedOverlay(risk_vol=mock_risk_vol, cap_leverage=-1)
    
    with pytest.raises(ValueError, match="leverage_mode must be"):
        VolManagedOverlay(risk_vol=mock_risk_vol, leverage_mode="invalid")
    
    with pytest.raises(ValueError, match="position_bounds must be"):
        VolManagedOverlay(risk_vol=mock_risk_vol, position_bounds=[1.0, -1.0])


def test_scales_toward_target_global(mock_market, mock_risk_vol, sample_signals):
    """
    Test that scaling achieves target volatility in global mode.
    
    Verifies that ex-ante portfolio vol ≈ target_vol after scaling.
    """
    target_vol = 0.20
    overlay = VolManagedOverlay(
        risk_vol=mock_risk_vol,
        target_vol=target_vol,
        leverage_mode="global"
    )
    
    date = datetime(2024, 1, 15)
    scaled = overlay.scale(sample_signals, mock_market, date)
    
    # Compute ex-ante portfolio vol with scaled signals
    cov = mock_risk_vol.covariance(mock_market, date)
    common = scaled.index.intersection(cov.index)
    w = scaled.loc[common].values
    cov_mat = cov.loc[common, common].values
    
    port_var = w @ cov_mat @ w
    port_vol = np.sqrt(port_var)
    
    # Should be close to target_vol (within 10% tolerance)
    assert abs(port_vol - target_vol) / target_vol < 0.10, \
        f"Portfolio vol {port_vol:.3f} not close to target {target_vol:.3f}"
    
    print(f"✓ Global mode: target={target_vol:.3f}, achieved={port_vol:.3f}")


def test_scales_toward_target_per_asset(mock_market, mock_risk_vol, sample_signals):
    """
    Test that per-asset scaling normalizes by individual volatilities.
    
    In per-asset mode, each signal should be scaled by target_vol / asset_vol.
    """
    target_vol = 0.15
    overlay = VolManagedOverlay(
        risk_vol=mock_risk_vol,
        target_vol=target_vol,
        leverage_mode="per-asset",
        config_path="nonexistent.yaml"  # Don't load config
    )
    
    date = datetime(2024, 1, 15)
    scaled = overlay.scale(sample_signals, mock_market, date)
    
    # Get vols
    vols = mock_risk_vol.vols(mock_market, date)
    
    # Check that scaling follows target_vol / asset_vol pattern
    for symbol in sample_signals.index:
        if symbol in vols.index and abs(sample_signals[symbol]) > 1e-10:
            expected_scale = target_vol / vols[symbol]
            actual_scale = abs(scaled[symbol] / sample_signals[symbol])
            
            # Should be close (within 20% tolerance for constraints)
            # The actual scale might be smaller or slightly larger due to constraints
            assert abs(actual_scale - expected_scale) / expected_scale < 0.20, \
                f"Scale for {symbol} differs too much: {actual_scale:.3f} vs {expected_scale:.3f}"
    
    print(f"✓ Per-asset mode: target_vol={target_vol:.3f}")


def test_bounds(mock_market, mock_risk_vol):
    """
    Test that position bounds are never exceeded.
    
    All scaled positions must be within position_bounds.
    """
    position_bounds = (-2.0, 2.0)
    overlay = VolManagedOverlay(
        risk_vol=mock_risk_vol,
        target_vol=0.20,
        position_bounds=position_bounds,
        leverage_mode="global"
    )
    
    # Test with various signal strengths
    test_cases = [
        pd.Series({'ES': 5.0, 'NQ': -4.0, 'GC': 3.0, 'CL': -2.0, 'TY': 1.0}),  # Large signals
        pd.Series({'ES': 0.1, 'NQ': -0.1, 'GC': 0.05, 'CL': -0.05, 'TY': 0.02}),  # Small signals
        pd.Series({'ES': 10.0, 'NQ': 0.0, 'GC': 0.0, 'CL': 0.0, 'TY': 0.0}),  # Concentrated
    ]
    
    date = datetime(2024, 1, 15)
    
    for i, signals in enumerate(test_cases):
        scaled = overlay.scale(signals, mock_market, date)
        
        # Check bounds
        assert scaled.min() >= position_bounds[0], \
            f"Case {i}: min position {scaled.min():.3f} < lower bound {position_bounds[0]}"
        assert scaled.max() <= position_bounds[1], \
            f"Case {i}: max position {scaled.max():.3f} > upper bound {position_bounds[1]}"
        
        print(f"✓ Bounds test case {i}: min={scaled.min():.3f}, max={scaled.max():.3f}")


def test_cap_leverage(mock_market, mock_risk_vol):
    """
    Test that leverage cap is enforced.
    
    Sum of absolute weights must be <= cap_leverage.
    """
    cap_leverage = 4.0
    overlay = VolManagedOverlay(
        risk_vol=mock_risk_vol,
        target_vol=0.20,
        cap_leverage=cap_leverage,
        leverage_mode="global"
    )
    
    # Test with signals that would naturally exceed cap
    test_cases = [
        pd.Series({'ES': 2.0, 'NQ': 2.0, 'GC': 2.0, 'CL': 2.0, 'TY': 2.0}),  # High uniform
        pd.Series({'ES': 5.0, 'NQ': -5.0, 'GC': 3.0, 'CL': -3.0, 'TY': 2.0}),  # High opposing
    ]
    
    date = datetime(2024, 1, 15)
    
    for i, signals in enumerate(test_cases):
        scaled = overlay.scale(signals, mock_market, date)
        
        gross_leverage = scaled.abs().sum()
        
        assert gross_leverage <= cap_leverage * 1.001, \
            f"Case {i}: gross leverage {gross_leverage:.3f} > cap {cap_leverage}"
        
        print(f"✓ Leverage cap test case {i}: gross={gross_leverage:.3f} <= cap={cap_leverage:.3f}")


def test_deterministic(mock_market, mock_risk_vol, sample_signals):
    """Test that outputs are deterministic given same inputs."""
    overlay = VolManagedOverlay(
        risk_vol=mock_risk_vol,
        target_vol=0.20,
        leverage_mode="global"
    )
    
    date = datetime(2024, 1, 15)
    
    # Run twice with same inputs
    result1 = overlay.scale(sample_signals, mock_market, date)
    result2 = overlay.scale(sample_signals, mock_market, date)
    
    # Should be identical
    pd.testing.assert_series_equal(result1, result2)
    
    print("✓ Deterministic: outputs are identical for same inputs")


def test_empty_signals(mock_market, mock_risk_vol):
    """Test handling of empty or zero signals."""
    overlay = VolManagedOverlay(
        risk_vol=mock_risk_vol,
        target_vol=0.20
    )
    
    date = datetime(2024, 1, 15)
    
    # Empty signals
    empty_signals = pd.Series(dtype=float)
    result = overlay.scale(empty_signals, mock_market, date)
    assert result.empty
    
    # Zero signals
    zero_signals = pd.Series({'ES': 0.0, 'NQ': 0.0, 'GC': 0.0})
    result = overlay.scale(zero_signals, mock_market, date)
    assert result.abs().sum() == 0
    
    print("✓ Empty/zero signals handled correctly")


def test_floor_vol_prevents_extreme_scaling(mock_market, mock_risk_vol):
    """Test that floor_vol prevents extreme scaling in low-vol environments."""
    # Create risk_vol with very low volatility
    low_vol_risk = Mock()
    low_vol_risk.vols = lambda m, d: pd.Series({
        'ES': 0.02,  # Very low vol
        'NQ': 0.02,
        'GC': 0.02,
        'CL': 0.02,
        'TY': 0.02
    })
    low_vol_risk.covariance = mock_risk_vol.covariance
    low_vol_risk.vol_lookback = 63
    
    floor_vol = 0.05
    overlay = VolManagedOverlay(
        risk_vol=low_vol_risk,
        target_vol=0.20,
        floor_vol=floor_vol,
        leverage_mode="per-asset",
        cap_leverage=10.0  # High cap to test floor effect
    )
    
    signals = pd.Series({'ES': 1.0, 'NQ': 1.0, 'GC': 1.0, 'CL': 1.0, 'TY': 1.0})
    date = datetime(2024, 1, 15)
    
    scaled = overlay.scale(signals, mock_market, date)
    
    # Without floor, scale would be target_vol / 0.02 = 10x
    # With floor, scale should be target_vol / floor_vol = 4x
    expected_max_scale = overlay.target_vol / floor_vol
    
    # Actual scale should not exceed expected (may be lower due to cap)
    actual_scale = scaled.abs().max() / signals.abs().max()
    
    assert actual_scale <= expected_max_scale * 1.01, \
        f"Scale {actual_scale:.3f} exceeds floor-based max {expected_max_scale:.3f}"
    
    print(f"✓ Floor vol prevents extreme scaling: scale={actual_scale:.3f} <= {expected_max_scale:.3f}")


def test_mixed_signal_directions(mock_market, mock_risk_vol):
    """Test scaling with mixed long/short signals."""
    overlay = VolManagedOverlay(
        risk_vol=mock_risk_vol,
        target_vol=0.20,
        leverage_mode="global"
    )
    
    # Mixed signals
    signals = pd.Series({
        'ES': 1.5,    # Long
        'NQ': -1.2,   # Short
        'GC': 0.8,    # Long
        'CL': -0.3,   # Short
        'TY': 0.0     # Neutral
    })
    
    date = datetime(2024, 1, 15)
    scaled = overlay.scale(signals, mock_market, date)
    
    # Signs should be preserved
    for symbol in signals.index:
        if abs(signals[symbol]) > 1e-10:
            assert np.sign(scaled[symbol]) == np.sign(signals[symbol]), \
                f"Sign flip for {symbol}: signal={signals[symbol]:.3f}, scaled={scaled[symbol]:.3f}"
    
    # Should still respect bounds and leverage cap
    assert scaled.min() >= -3.0
    assert scaled.max() <= 3.0
    assert scaled.abs().sum() <= 7.0
    
    print("✓ Mixed signal directions handled correctly")


def test_describe(mock_risk_vol):
    """Test describe method returns correct configuration."""
    overlay = VolManagedOverlay(
        risk_vol=mock_risk_vol,
        target_vol=0.18,
        floor_vol=0.04,
        cap_leverage=6.0,
        leverage_mode="per-asset",
        position_bounds=(-2.5, 2.5),
        config_path="nonexistent.yaml"  # Don't load config
    )
    
    desc = overlay.describe()
    
    assert desc['agent'] == 'VolManagedOverlay'
    assert desc['target_vol'] == 0.18
    assert desc['floor_vol'] == 0.04
    assert desc['cap_leverage'] == 6.0
    assert desc['leverage_mode'] == 'per-asset'
    assert desc['position_bounds'] == (-2.5, 2.5)
    assert 'scale' in desc['outputs'][0]
    
    print("✓ Describe method returns correct configuration")


def test_robustness_to_missing_data(mock_market, sample_signals):
    """Test robustness when RiskVol fails to provide data."""
    # Create risk_vol that raises exceptions
    failing_risk_vol = Mock()
    failing_risk_vol.vols = Mock(side_effect=ValueError("No data"))
    failing_risk_vol.covariance = Mock(side_effect=ValueError("No data"))
    failing_risk_vol.vol_lookback = 63
    
    overlay = VolManagedOverlay(
        risk_vol=failing_risk_vol,
        target_vol=0.20,
        leverage_mode="global"
    )
    
    date = datetime(2024, 1, 15)
    
    # Should handle gracefully (return bounded signals)
    result = overlay.scale(sample_signals, mock_market, date)
    
    # Should still respect bounds
    assert result.min() >= -3.0
    assert result.max() <= 3.0
    
    print("✓ Robust to missing RiskVol data")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

