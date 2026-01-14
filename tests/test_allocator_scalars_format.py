"""
Test: Allocator Scalars CSV Format Invariant

Validates that allocator_scalars_at_rebalances.csv files:
1. Can be loaded (either canonical or legacy format)
2. Have rebalance_date column (or can be normalized to it)
3. Have monotonic dates
4. Have expected scalar columns

This test prevents format drift and ensures backward compatibility.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import tempfile
import shutil

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.diagnostics.canonical_diagnostics import _load_csv_with_date_index


class TestAllocatorScalarsFormat:
    """Test allocator_scalars_at_rebalances.csv format invariants."""
    
    def test_canonical_format_loads_correctly(self, tmp_path):
        """Test that canonical format (with rebalance_date) loads correctly."""
        # Create canonical format CSV
        dates = pd.date_range('2020-01-01', periods=10, freq='W')
        df = pd.DataFrame({
            'risk_scalar_computed': np.random.uniform(0.8, 1.2, 10),
            'risk_scalar_applied': np.random.uniform(0.8, 1.2, 10)
        }, index=dates)
        
        csv_path = tmp_path / 'allocator_scalars_at_rebalances.csv'
        df.to_csv(csv_path, index_label='rebalance_date')
        
        # Load using the helper function
        loaded_df = _load_csv_with_date_index(csv_path)
        
        # Assertions
        assert isinstance(loaded_df.index, pd.DatetimeIndex), "Index should be DatetimeIndex"
        assert len(loaded_df) == 10, "Should have 10 rows"
        assert 'risk_scalar_computed' in loaded_df.columns, "Should have risk_scalar_computed column"
        assert 'risk_scalar_applied' in loaded_df.columns, "Should have risk_scalar_applied column"
        assert loaded_df.index.is_monotonic_increasing, "Dates should be monotonic"
    
    def test_legacy_format_loads_correctly(self, tmp_path):
        """Test that legacy format (Unnamed: 0) loads correctly."""
        # Create legacy format CSV (no index_label)
        dates = pd.date_range('2020-01-01', periods=10, freq='W')
        df = pd.DataFrame({
            'risk_scalar_computed': np.random.uniform(0.8, 1.2, 10),
            'risk_scalar_applied': np.random.uniform(0.8, 1.2, 10)
        }, index=dates)
        
        csv_path = tmp_path / 'allocator_scalars_at_rebalances.csv'
        df.to_csv(csv_path)  # No index_label - creates 'Unnamed: 0'
        
        # Load using the helper function
        loaded_df = _load_csv_with_date_index(csv_path)
        
        # Assertions
        assert isinstance(loaded_df.index, pd.DatetimeIndex), "Index should be DatetimeIndex"
        assert len(loaded_df) == 10, "Should have 10 rows"
        assert 'risk_scalar_computed' in loaded_df.columns, "Should have risk_scalar_computed column"
        assert loaded_df.index.is_monotonic_increasing, "Dates should be monotonic"
    
    def test_dates_are_monotonic(self, tmp_path):
        """Test that dates are monotonic after loading."""
        # Create CSV with dates
        dates = pd.date_range('2020-01-01', periods=20, freq='W')
        df = pd.DataFrame({
            'risk_scalar_computed': np.random.uniform(0.8, 1.2, 20),
            'risk_scalar_applied': np.random.uniform(0.8, 1.2, 20)
        }, index=dates)
        
        csv_path = tmp_path / 'allocator_scalars_at_rebalances.csv'
        df.to_csv(csv_path, index_label='rebalance_date')
        
        # Load
        loaded_df = _load_csv_with_date_index(csv_path)
        
        # Assert monotonic
        assert loaded_df.index.is_monotonic_increasing, "Dates must be monotonic increasing"
        assert not loaded_df.index.duplicated().any(), "Dates must be unique"
    
    def test_has_expected_columns(self, tmp_path):
        """Test that CSV has expected scalar columns."""
        dates = pd.date_range('2020-01-01', periods=5, freq='W')
        df = pd.DataFrame({
            'risk_scalar_computed': [0.9, 0.95, 1.0, 1.05, 1.1],
            'risk_scalar_applied': [0.9, 0.95, 1.0, 1.05, 1.1]
        }, index=dates)
        
        csv_path = tmp_path / 'allocator_scalars_at_rebalances.csv'
        df.to_csv(csv_path, index_label='rebalance_date')
        
        # Load
        loaded_df = _load_csv_with_date_index(csv_path)
        
        # Assert columns
        assert 'risk_scalar_computed' in loaded_df.columns, "Must have risk_scalar_computed"
        assert 'risk_scalar_applied' in loaded_df.columns, "Must have risk_scalar_applied"
    
    def test_integration_with_real_file(self):
        """Test loading a real file if it exists (optional, doesn't fail if missing)."""
        # Try to load the canonical run if it exists
        real_path = Path('reports/runs/canonical_frozen_stack_precomputed_20260113_123354/allocator_scalars_at_rebalances.csv')
        if real_path.exists():
            loaded_df = _load_csv_with_date_index(real_path)
            
            # Basic assertions
            assert isinstance(loaded_df.index, pd.DatetimeIndex), "Should load with DatetimeIndex"
            assert len(loaded_df) > 0, "Should have rows"
            assert loaded_df.index.is_monotonic_increasing, "Dates should be monotonic"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
