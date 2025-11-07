"""
Unit tests for Allocator agent.

Tests verify:
- Respects gross/net caps
- All weights within bounds
- Turnover control when weights_prev given
- Positive risk contributions for ERC
- Deterministic outputs given inputs
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.allocator import Allocator


class TestAllocatorInitialization:
    """Test Allocator initialization and configuration."""
    
    def test_init_defaults(self):
        """Test Allocator initializes with default parameters."""
        agent = Allocator()
        
        assert agent.method == "signal-beta"
        assert agent.gross_cap == 7.0
        assert agent.net_cap == 2.0
        assert agent.w_bounds_per_asset == [-1.5, 1.5]
        assert agent.turnover_cap == 0.5
        assert agent.lambda_turnover == 0.001
    
    def test_init_custom_params(self):
        """Test Allocator initializes with custom parameters."""
        agent = Allocator(
            method="erc",
            gross_cap=5.0,
            net_cap=1.0,
            w_bounds_per_asset=[-1.0, 1.0],
            turnover_cap=0.3,
            lambda_turnover=0.01
        )
        
        assert agent.method == "erc"
        assert agent.gross_cap == 5.0
        assert agent.net_cap == 1.0
        assert agent.w_bounds_per_asset == [-1.0, 1.0]
        assert agent.turnover_cap == 0.3
        assert agent.lambda_turnover == 0.01
    
    def test_init_from_config(self):
        """Test Allocator loads configuration from YAML file."""
        agent = Allocator(config_path="configs/strategies.yaml")
        
        # Should load from config
        assert agent.method == "signal-beta"
        assert agent.gross_cap == 7.0
        assert agent.net_cap == 2.0
        assert agent.w_bounds_per_asset == [-1.5, 1.5]
        assert agent.turnover_cap == 0.5
    
    def test_invalid_method(self):
        """Test that invalid method raises error."""
        with pytest.raises(ValueError, match="method must be"):
            Allocator(method="invalid")
    
    def test_invalid_gross_cap(self):
        """Test that invalid gross_cap raises error."""
        with pytest.raises(ValueError, match="gross_cap must be positive"):
            Allocator(gross_cap=-1.0)
    
    def test_invalid_net_vs_gross(self):
        """Test that net_cap cannot exceed gross_cap."""
        with pytest.raises(ValueError, match="net_cap.*cannot exceed gross_cap"):
            Allocator(gross_cap=2.0, net_cap=5.0)
    
    def test_invalid_bounds(self):
        """Test that invalid bounds raise errors."""
        with pytest.raises(ValueError, match="w_bounds_per_asset must have 2 elements"):
            Allocator(w_bounds_per_asset=[1.0])
        
        with pytest.raises(ValueError, match="w_bounds_per_asset min > max"):
            Allocator(w_bounds_per_asset=[1.0, -1.0])
    
    def test_invalid_turnover_cap(self):
        """Test that invalid turnover_cap raises error."""
        with pytest.raises(ValueError, match="turnover_cap must be in"):
            Allocator(turnover_cap=1.5)
        
        with pytest.raises(ValueError, match="turnover_cap must be in"):
            Allocator(turnover_cap=-0.1)
    
    def test_invalid_lambda_turnover(self):
        """Test that invalid lambda_turnover raises error."""
        with pytest.raises(ValueError, match="lambda_turnover must be non-negative"):
            Allocator(lambda_turnover=-0.01)
    
    def test_describe(self):
        """Test describe method returns configuration."""
        agent = Allocator()
        desc = agent.describe()
        
        assert isinstance(desc, dict)
        assert desc['agent'] == 'Allocator'
        assert 'method' in desc
        assert 'gross_cap' in desc
        assert 'net_cap' in desc
        assert 'w_bounds_per_asset' in desc
        assert 'turnover_cap' in desc
        assert 'outputs' in desc


class TestConstraints:
    """Test constraint enforcement."""
    
    @pytest.fixture
    def simple_cov(self):
        """Create a simple covariance matrix for testing."""
        assets = ['A', 'B', 'C', 'D']
        # Identity-like covariance (uncorrelated assets)
        cov = pd.DataFrame(
            np.eye(4) * 0.04,  # 20% vol each
            index=assets,
            columns=assets
        )
        return cov
    
    @pytest.fixture
    def simple_signals(self):
        """Create simple signals for testing."""
        return pd.Series([1.0, 0.5, -0.5, -1.0], index=['A', 'B', 'C', 'D'])
    
    def test_sum_caps_gross(self, simple_signals, simple_cov):
        """Test that gross cap is respected."""
        agent = Allocator(method="signal-beta", gross_cap=3.0)
        
        weights = agent.solve(simple_signals, simple_cov)
        
        gross = weights.abs().sum()
        assert gross <= agent.gross_cap + 1e-6, \
            f"Gross leverage {gross:.4f} exceeds cap {agent.gross_cap}"
    
    def test_sum_caps_net(self, simple_signals, simple_cov):
        """Test that net cap is respected."""
        agent = Allocator(method="signal-beta", net_cap=1.5)
        
        weights = agent.solve(simple_signals, simple_cov)
        
        net = weights.sum()
        assert abs(net) <= abs(agent.net_cap) + 1e-6, \
            f"Net leverage {net:.4f} exceeds cap {agent.net_cap}"
    
    def test_sum_caps_both(self, simple_signals, simple_cov):
        """Test that both gross and net caps are respected simultaneously."""
        agent = Allocator(method="signal-beta", gross_cap=5.0, net_cap=2.0)
        
        weights = agent.solve(simple_signals, simple_cov)
        
        gross = weights.abs().sum()
        net = weights.sum()
        
        assert gross <= agent.gross_cap + 1e-6, \
            f"Gross leverage {gross:.4f} exceeds cap {agent.gross_cap}"
        assert abs(net) <= abs(agent.net_cap) + 1e-6, \
            f"Net leverage {net:.4f} exceeds cap {agent.net_cap}"
    
    def test_bounds(self, simple_signals, simple_cov):
        """Test that all weights within bounds."""
        bounds = [-1.0, 1.0]
        agent = Allocator(method="signal-beta", w_bounds_per_asset=bounds)
        
        weights = agent.solve(simple_signals, simple_cov)
        
        for asset, w in weights.items():
            assert bounds[0] - 1e-6 <= w <= bounds[1] + 1e-6, \
                f"Weight for {asset} ({w:.4f}) outside bounds {bounds}"
    
    def test_bounds_tight(self, simple_signals, simple_cov):
        """Test that tight bounds are respected."""
        bounds = [-0.5, 0.5]
        agent = Allocator(method="signal-beta", w_bounds_per_asset=bounds, gross_cap=10.0)
        
        weights = agent.solve(simple_signals, simple_cov)
        
        for asset, w in weights.items():
            assert bounds[0] - 1e-6 <= w <= bounds[1] + 1e-6, \
                f"Weight for {asset} ({w:.4f}) outside bounds {bounds}"
    
    def test_turnover(self, simple_signals, simple_cov):
        """Test that turnover cap is respected when weights_prev given."""
        agent = Allocator(method="signal-beta", turnover_cap=0.3)
        
        # Previous weights
        weights_prev = pd.Series([0.5, 0.3, -0.2, -0.4], index=['A', 'B', 'C', 'D'])
        
        # Solve with turnover constraint
        weights = agent.solve(simple_signals, simple_cov, weights_prev=weights_prev)
        
        # Calculate one-way turnover
        turnover = (weights - weights_prev).abs().sum()
        
        assert turnover <= agent.turnover_cap + 1e-6, \
            f"Turnover {turnover:.4f} exceeds cap {agent.turnover_cap}"
    
    def test_turnover_zero_prev(self, simple_signals, simple_cov):
        """Test turnover with zero previous weights."""
        agent = Allocator(method="signal-beta", turnover_cap=0.5)
        
        weights_prev = pd.Series(0.0, index=['A', 'B', 'C', 'D'])
        
        weights = agent.solve(simple_signals, simple_cov, weights_prev=weights_prev)
        
        turnover = weights.abs().sum()
        
        # Turnover from zero should be same as gross exposure
        assert turnover <= agent.turnover_cap + 1e-6, \
            f"Turnover {turnover:.4f} exceeds cap {agent.turnover_cap}"


class TestERCMethod:
    """Test Equal Risk Contribution method."""
    
    @pytest.fixture
    def simple_cov(self):
        """Create a simple covariance matrix."""
        assets = ['A', 'B', 'C', 'D']
        # Varying volatilities
        vols = np.array([0.15, 0.20, 0.25, 0.30])
        corr = np.eye(4) * 0.7 + np.ones((4, 4)) * 0.3  # 30% correlation
        cov = np.outer(vols, vols) * corr
        return pd.DataFrame(cov, index=assets, columns=assets)
    
    @pytest.fixture
    def flat_signals(self):
        """Create flat (equal) signals."""
        return pd.Series([1.0, 1.0, 1.0, 1.0], index=['A', 'B', 'C', 'D'])
    
    def test_positive_risk_contribs_erc(self, flat_signals, simple_cov):
        """Test that ERC produces non-negative risk contributions."""
        agent = Allocator(method="erc", gross_cap=5.0)
        
        weights = agent.solve(flat_signals, simple_cov)
        
        # Calculate risk contributions
        cov_mat = simple_cov.loc[weights.index, weights.index].values
        w = weights.values
        
        portfolio_vol = np.sqrt(w @ cov_mat @ w)
        
        if portfolio_vol > 1e-12:
            marginal_contrib = cov_mat @ w / portfolio_vol
            risk_contrib = w * marginal_contrib
            
            # All risk contributions should be non-negative
            assert np.all(risk_contrib >= -1e-6), \
                f"Found negative risk contributions: {risk_contrib}"
    
    def test_erc_approximately_equal(self, flat_signals, simple_cov):
        """Test that ERC produces approximately equal risk contributions."""
        agent = Allocator(method="erc", gross_cap=5.0)
        
        weights = agent.solve(flat_signals, simple_cov)
        
        # Calculate risk contributions
        cov_mat = simple_cov.loc[weights.index, weights.index].values
        w = weights.values
        
        portfolio_var = w @ cov_mat @ w
        
        if portfolio_var > 1e-12:
            risk_contrib = w * (cov_mat @ w)
            
            # Remove zero-weight assets
            nonzero_rc = risk_contrib[np.abs(w) > 1e-6]
            
            if len(nonzero_rc) > 1:
                # Standard deviation of risk contributions should be small relative to mean
                rc_std = np.std(nonzero_rc)
                rc_mean = np.mean(np.abs(nonzero_rc))
                
                # Check that risk contributions are similar
                # (allowing some variation due to constraints)
                assert rc_std / rc_mean < 2.0, \
                    f"Risk contributions not equal: std/mean = {rc_std/rc_mean:.2f}"
    
    def test_erc_with_zero_signals(self, simple_cov):
        """Test ERC with zero signals."""
        agent = Allocator(method="erc")
        
        zero_signals = pd.Series([0.0, 0.0, 0.0, 0.0], index=['A', 'B', 'C', 'D'])
        
        weights = agent.solve(zero_signals, simple_cov)
        
        # Should still produce valid weights
        assert len(weights) > 0
        gross = weights.abs().sum()
        assert gross <= agent.gross_cap + 1e-6


class TestSignalBetaMethod:
    """Test signal-beta allocation method."""
    
    @pytest.fixture
    def simple_cov(self):
        """Create a simple covariance matrix."""
        assets = ['A', 'B', 'C', 'D']
        cov = pd.DataFrame(
            np.eye(4) * 0.04,  # 20% vol each, uncorrelated
            index=assets,
            columns=assets
        )
        return cov
    
    @pytest.fixture
    def simple_signals(self):
        """Create simple signals."""
        return pd.Series([2.0, 1.0, -1.0, -2.0], index=['A', 'B', 'C', 'D'])
    
    def test_signal_beta_direction(self, simple_signals, simple_cov):
        """Test that signal-beta respects signal direction."""
        agent = Allocator(method="signal-beta")
        
        weights = agent.solve(simple_signals, simple_cov)
        
        # Weights should generally align with signal direction
        for asset in weights.index:
            signal = simple_signals[asset]
            weight = weights[asset]
            
            if abs(signal) > 0.1:  # Only check non-trivial signals
                assert np.sign(weight) == np.sign(signal) or abs(weight) < 1e-6, \
                    f"Weight direction for {asset} doesn't match signal"
    
    def test_signal_beta_with_zero_signals(self, simple_cov):
        """Test signal-beta with zero signals."""
        agent = Allocator(method="signal-beta")
        
        zero_signals = pd.Series([0.0, 0.0, 0.0, 0.0], index=['A', 'B', 'C', 'D'])
        
        weights = agent.solve(zero_signals, simple_cov)
        
        # Should return zero or near-zero weights
        assert weights.abs().sum() < 1e-6, "Expected near-zero weights for zero signals"
    
    def test_signal_beta_magnitude(self, simple_signals, simple_cov):
        """Test that signal-beta weights scale with signal magnitude."""
        agent = Allocator(method="signal-beta", gross_cap=10.0)
        
        weights = agent.solve(simple_signals, simple_cov)
        
        # Larger absolute signals should generally get larger absolute weights
        # (when uncorrelated assets)
        w_A = abs(weights['A'])
        w_B = abs(weights['B'])
        w_C = abs(weights['C'])
        w_D = abs(weights['D'])
        
        # A has signal 2.0, B has signal 1.0
        # D has signal -2.0, C has signal -1.0
        # For uncorrelated assets, |w_A| should be >= |w_B| (roughly)
        # Allow some flexibility due to constraints
        assert w_A + w_D >= w_B + w_C - 0.5, \
            "Larger signals should generally get larger weights"


class TestDeterminism:
    """Test deterministic behavior."""
    
    @pytest.fixture
    def simple_cov(self):
        """Create a simple covariance matrix."""
        assets = ['A', 'B', 'C']
        cov = pd.DataFrame(
            [[0.04, 0.01, 0.00],
             [0.01, 0.09, 0.02],
             [0.00, 0.02, 0.16]],
            index=assets,
            columns=assets
        )
        return cov
    
    @pytest.fixture
    def simple_signals(self):
        """Create simple signals."""
        return pd.Series([1.5, 0.5, -1.0], index=['A', 'B', 'C'])
    
    def test_deterministic_signal_beta(self, simple_signals, simple_cov):
        """Test that signal-beta allocation is deterministic."""
        agent = Allocator(method="signal-beta")
        
        weights1 = agent.solve(simple_signals, simple_cov)
        weights2 = agent.solve(simple_signals, simple_cov)
        
        pd.testing.assert_series_equal(weights1, weights2)
    
    def test_deterministic_erc(self, simple_signals, simple_cov):
        """Test that ERC allocation is deterministic."""
        agent = Allocator(method="erc", lambda_turnover=0.0)
        
        # Run multiple times
        weights1 = agent.solve(simple_signals, simple_cov)
        weights2 = agent.solve(simple_signals, simple_cov)
        
        # Should be very close (optimization might have small numerical differences)
        pd.testing.assert_series_equal(weights1, weights2, atol=1e-6)
    
    def test_deterministic_with_prev_weights(self, simple_signals, simple_cov):
        """Test determinism with previous weights."""
        agent = Allocator(method="signal-beta")
        
        weights_prev = pd.Series([0.3, 0.2, -0.1], index=['A', 'B', 'C'])
        
        weights1 = agent.solve(simple_signals, simple_cov, weights_prev)
        weights2 = agent.solve(simple_signals, simple_cov, weights_prev)
        
        pd.testing.assert_series_equal(weights1, weights2)


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_signals(self):
        """Test with empty signals."""
        agent = Allocator()
        
        empty_signals = pd.Series(dtype=float)
        cov = pd.DataFrame(np.eye(2), index=['A', 'B'], columns=['A', 'B'])
        
        weights = agent.solve(empty_signals, cov)
        
        assert len(weights) == 0
    
    def test_no_common_assets(self):
        """Test when signals and cov have no common assets."""
        agent = Allocator()
        
        signals = pd.Series([1.0, 2.0], index=['A', 'B'])
        cov = pd.DataFrame(np.eye(2), index=['C', 'D'], columns=['C', 'D'])
        
        weights = agent.solve(signals, cov)
        
        assert len(weights) == 0
    
    def test_nan_signals(self):
        """Test that NaN signals are handled."""
        agent = Allocator(method="signal-beta")
        
        signals = pd.Series([1.0, np.nan, -1.0], index=['A', 'B', 'C'])
        cov = pd.DataFrame(
            np.eye(3) * 0.04,
            index=['A', 'B', 'C'],
            columns=['A', 'B', 'C']
        )
        
        weights = agent.solve(signals, cov)
        
        # Should only have weights for A and C (not B with NaN signal)
        assert 'B' not in weights.index or abs(weights.get('B', 0.0)) < 1e-6
        assert 'A' in weights.index
        assert 'C' in weights.index
    
    def test_single_asset(self):
        """Test with single asset."""
        agent = Allocator(method="signal-beta")
        
        signals = pd.Series([1.5], index=['A'])
        cov = pd.DataFrame([[0.04]], index=['A'], columns=['A'])
        
        weights = agent.solve(signals, cov)
        
        assert len(weights) == 1
        assert 'A' in weights.index
        # Should respect caps
        assert abs(weights['A']) <= agent.net_cap + 1e-6
    
    def test_singular_covariance(self):
        """Test handling of near-singular covariance matrix."""
        agent = Allocator(method="signal-beta")
        
        signals = pd.Series([1.0, 1.0, 1.0], index=['A', 'B', 'C'])
        
        # Nearly singular covariance (all assets perfectly correlated)
        cov = pd.DataFrame(
            [[0.04, 0.02, 0.02],
             [0.02, 0.04, 0.02],
             [0.02, 0.02, 0.04]],
            index=['A', 'B', 'C'],
            columns=['A', 'B', 'C']
        )
        
        # Should handle gracefully (regularization should help)
        weights = agent.solve(signals, cov)
        
        assert len(weights) > 0
        gross = weights.abs().sum()
        assert gross <= agent.gross_cap + 1e-6


class TestAlignmentAndMisalignment:
    """Test handling of aligned and misaligned inputs."""
    
    def test_partial_overlap(self):
        """Test when signals and cov have partial overlap."""
        agent = Allocator(method="signal-beta")
        
        signals = pd.Series([1.0, 2.0, 3.0], index=['A', 'B', 'C'])
        cov = pd.DataFrame(
            np.eye(3) * 0.04,
            index=['B', 'C', 'D'],
            columns=['B', 'C', 'D']
        )
        
        weights = agent.solve(signals, cov)
        
        # Should only have weights for B and C (common assets)
        assert 'A' not in weights.index
        assert 'D' not in weights.index
        assert 'B' in weights.index
        assert 'C' in weights.index
    
    def test_prev_weights_alignment(self):
        """Test alignment of previous weights."""
        agent = Allocator(method="signal-beta", turnover_cap=0.3)
        
        signals = pd.Series([1.0, 2.0], index=['A', 'B'])
        cov = pd.DataFrame(
            np.eye(2) * 0.04,
            index=['A', 'B'],
            columns=['A', 'B']
        )
        
        # Previous weights with extra asset C
        weights_prev = pd.Series([0.5, 0.3, 0.2], index=['A', 'B', 'C'])
        
        weights = agent.solve(signals, cov, weights_prev)
        
        # Should work, using only A and B from prev weights
        assert 'C' not in weights.index
        assert 'A' in weights.index
        assert 'B' in weights.index


class TestMultipleMethods:
    """Test consistency across different methods."""
    
    @pytest.fixture
    def simple_cov(self):
        """Create a simple covariance matrix."""
        assets = ['A', 'B', 'C']
        cov = pd.DataFrame(
            np.eye(3) * 0.04,
            index=assets,
            columns=assets
        )
        return cov
    
    @pytest.fixture
    def simple_signals(self):
        """Create simple signals."""
        return pd.Series([1.0, 0.5, -0.5], index=['A', 'B', 'C'])
    
    def test_all_methods_respect_caps(self, simple_signals, simple_cov):
        """Test that all methods respect caps."""
        methods = ["erc", "signal-beta", "meanvar"]
        
        for method in methods:
            agent = Allocator(method=method, gross_cap=4.0, net_cap=1.5)
            
            weights = agent.solve(simple_signals, simple_cov)
            
            gross = weights.abs().sum()
            net = weights.sum()
            
            assert gross <= agent.gross_cap + 1e-6, \
                f"Method {method}: gross {gross:.4f} exceeds cap"
            assert abs(net) <= abs(agent.net_cap) + 1e-6, \
                f"Method {method}: net {net:.4f} exceeds cap"
    
    def test_all_methods_respect_bounds(self, simple_signals, simple_cov):
        """Test that all methods respect bounds."""
        methods = ["erc", "signal-beta", "meanvar"]
        bounds = [-0.8, 0.8]
        
        for method in methods:
            agent = Allocator(method=method, w_bounds_per_asset=bounds)
            
            weights = agent.solve(simple_signals, simple_cov)
            
            for asset, w in weights.items():
                assert bounds[0] - 1e-6 <= w <= bounds[1] + 1e-6, \
                    f"Method {method}: weight for {asset} outside bounds"


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "-s"])
