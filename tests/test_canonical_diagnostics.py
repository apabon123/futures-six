"""
Tests for Canonical Diagnostics

Tests validate that diagnostics computation handles missing/empty artifacts gracefully.
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.diagnostics.canonical_diagnostics import compute_constraint_binding


class TestConstraintBindingRobustness:
    """Test that constraint binding handles missing/empty artifacts gracefully."""
    
    def test_constraint_binding_missing_inputs_does_not_crash(self):
        """Test that missing inputs don't cause crashes."""
        # Missing portfolio_returns (required)
        artifacts = {}
        result = compute_constraint_binding(artifacts)
        
        assert 'status' in result
        assert result['status'] == 'missing_artifact'
        assert 'details' in result
        assert 'portfolio_returns' in result['details']
        
        # All metrics should be NaN
        assert np.isnan(result['allocator_active_pct'])
        assert np.isnan(result['allocator_avg_scalar_when_active'])
    
    def test_constraint_binding_empty_dataframes_does_not_crash(self):
        """Test that empty DataFrames don't cause truthiness errors."""
        # Empty portfolio returns
        artifacts = {
            'portfolio_returns': pd.Series(dtype=float),
            'weights': pd.DataFrame(),
            'weights_scaled': pd.DataFrame(),
            'weights_raw': pd.DataFrame(),
            'allocator_scalar': None,
            'engine_policy': None,
            'meta': {}
        }
        
        result = compute_constraint_binding(artifacts)
        
        # Should return valid structure (not crash)
        assert 'allocator_active_pct' in result
        assert 'vol_below_target_pct' in result
        # Metrics should be NaN since data is empty
        assert np.isnan(result['allocator_active_pct'])
    
    def test_constraint_binding_weights_truthiness_handled_correctly(self):
        """Test that weights selection doesn't use DataFrame truthiness."""
        # weights is empty DataFrame, weights_scaled has data
        dates = pd.date_range('2020-01-01', periods=5, freq='D')
        weights_scaled = pd.DataFrame({
            'SYM1': [0.5, 0.6, 0.4, 0.5, 0.5],
            'SYM2': [0.5, 0.4, 0.6, 0.5, 0.5]
        }, index=dates)
        
        artifacts = {
            'portfolio_returns': pd.Series(0.001, index=dates),
            'weights': pd.DataFrame(),  # Empty - should be skipped
            'weights_scaled': weights_scaled,  # Non-empty - should be used
            'weights_raw': None,
            'allocator_scalar': None,
            'engine_policy': None,
            'meta': {}
        }
        
        # Should not raise ValueError about ambiguous truth value
        result = compute_constraint_binding(artifacts)
        
        # Should successfully compute metrics
        assert 'exposure_capped_pct' in result
        assert not np.isnan(result['exposure_capped_pct'])  # Should have a value
    
    def test_constraint_binding_with_valid_data(self):
        """Test that valid data computes correctly."""
        dates = pd.date_range('2020-01-01', periods=10, freq='D')
        
        artifacts = {
            'portfolio_returns': pd.Series(np.random.randn(10) * 0.01, index=dates),
            'weights': pd.DataFrame({
                'SYM1': np.random.uniform(0.4, 0.6, 10),
                'SYM2': np.random.uniform(0.4, 0.6, 10)
            }, index=dates),
            'allocator_scalar': pd.Series(np.random.uniform(0.8, 1.2, 10), index=dates),
            'engine_policy': None,
            'meta': {}
        }
        
        result = compute_constraint_binding(artifacts)
        
        # Should have all keys
        assert 'allocator_active_pct' in result
        assert 'allocator_avg_scalar_when_active' in result
        assert 'policy_gated_trend_pct' in result
        assert 'policy_gated_vrp_pct' in result
        assert 'exposure_capped_pct' in result
        assert 'vol_below_target_pct' in result
        
        # Should not have error status
        assert 'status' not in result or result.get('status') != 'missing_artifact'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
