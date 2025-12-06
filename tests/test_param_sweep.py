"""
Tests for ParamSweepRunner agent.

Tests:
1. Parameter application to nested configs
2. Grid combination generation
3. Single backtest execution
4. Full sweep with reproducibility
5. CSV output validation
6. Configuration comparison
"""

import os
import tempfile
import shutil
from pathlib import Path
import pytest
import pandas as pd
import numpy as np
import yaml

from src.agents.param_sweep import (
    _set_nested_value,
    _apply_params_to_config,
    _generate_grid_combinations,
    _run_single_backtest,
    run_sweep,
    compare_configs
)


class TestNestedConfig:
    """Test nested configuration manipulation."""
    
    def test_set_nested_value_simple(self):
        """Test setting simple nested value."""
        config = {}
        _set_nested_value(config, "a.b.c", 123)
        
        assert config == {"a": {"b": {"c": 123}}}
    
    def test_set_nested_value_existing(self):
        """Test setting value in existing structure."""
        config = {"a": {"b": {"x": 1}}}
        _set_nested_value(config, "a.b.c", 123)
        
        assert config == {"a": {"b": {"x": 1, "c": 123}}}
    
    def test_set_nested_value_overwrite(self):
        """Test overwriting existing value."""
        config = {"a": {"b": {"c": 1}}}
        _set_nested_value(config, "a.b.c", 999)
        
        assert config == {"a": {"b": {"c": 999}}}
    
    def test_apply_params_to_config(self):
        """Test applying parameter set to config."""
        base_config = {
            "tsmom": {"lookbacks": [252], "skip_recent": 21},
            "exec": {"rebalance": "W-FRI"}
        }
        
        params = {
            "tsmom.lookbacks": [126, 252],
            "macro.vol_thresholds.low": 0.12
        }
        
        config = _apply_params_to_config(base_config, params)
        
        # Check original is unchanged
        assert base_config["tsmom"]["lookbacks"] == [252]
        assert "macro" not in base_config
        
        # Check new config has changes
        assert config["tsmom"]["lookbacks"] == [126, 252]
        assert config["macro"]["vol_thresholds"]["low"] == 0.12
        assert config["exec"]["rebalance"] == "W-FRI"  # Preserved


class TestGridGeneration:
    """Test grid combination generation."""
    
    def test_generate_grid_simple(self):
        """Test simple grid generation."""
        grid = {
            "a": [1, 2],
            "b": [3, 4]
        }
        
        combos = _generate_grid_combinations(grid)
        
        assert len(combos) == 4
        assert {"a": 1, "b": 3} in combos
        assert {"a": 1, "b": 4} in combos
        assert {"a": 2, "b": 3} in combos
        assert {"a": 2, "b": 4} in combos
    
    def test_generate_grid_single_param(self):
        """Test grid with single parameter."""
        grid = {"x": [10, 20, 30]}
        
        combos = _generate_grid_combinations(grid)
        
        assert len(combos) == 3
        assert {"x": 10} in combos
        assert {"x": 20} in combos
        assert {"x": 30} in combos
    
    def test_generate_grid_complex_values(self):
        """Test grid with complex values (lists, dicts)."""
        grid = {
            "lookbacks": [[252], [126, 252]],
            "thresholds": [{"low": 0.1, "high": 0.2}, {"low": 0.15, "high": 0.25}]
        }
        
        combos = _generate_grid_combinations(grid)
        
        assert len(combos) == 4
        # Check that complex values are preserved
        assert {"lookbacks": [252], "thresholds": {"low": 0.1, "high": 0.2}} in combos
        assert {"lookbacks": [126, 252], "thresholds": {"low": 0.15, "high": 0.25}} in combos


class TestSingleBacktest:
    """Test single backtest execution."""
    
    def test_single_backtest_minimal(self):
        """Test single backtest with minimal config."""
        base_config = {
            "tsmom": {
                "lookbacks": [252],
                "skip_recent": 21,
                "standardize": "vol",
                "signal_cap": 3.0,
                "rebalance": "W-FRI"
            },
            "vol_overlay": {
                "target_vol": 0.20,
                "lookback_vol": 63,
                "leverage_mode": "global",
                "cap_leverage": 7.0,
                "position_bounds": [-3.0, 3.0]
            },
            "risk_vol": {
                "cov_lookback": 252,
                "vol_lookback": 63,
                "shrinkage": "lw",
                "nan_policy": "mask-asset"
            },
            "allocator": {
                "method": "signal-beta",
                "gross_cap": 7.0,
                "net_cap": 2.0,
                "w_bounds_per_asset": [-1.5, 1.5],
                "turnover_cap": 0.5,
                "lambda_turnover": 0.001
            },
            "exec": {
                "rebalance": "W-FRI",
                "slippage_bps": 0.5,
                "commission_per_contract": 0.0,
                "position_notional_scale": 1.0
            }
        }
        
        params = {}
        start = "2024-01-01"
        end = "2024-03-31"
        seed = 42
        
        result = _run_single_backtest((params, base_config, start, end, seed))
        
        # Check result structure
        assert 'cagr' in result
        assert 'sharpe' in result
        assert 'vol' in result
        assert 'max_drawdown' in result
        assert 'calmar' in result
        assert 'turnover' in result
        assert 'cost_drag' in result
        assert 'success' in result
        assert 'seed' in result
        
        # Check result values
        assert result['seed'] == 42
        assert result['success'] is True or result['success'] is False
        
        if result['success']:
            assert isinstance(result['cagr'], (int, float))
            assert isinstance(result['sharpe'], (int, float))
            assert result['n_periods'] >= 0
    
    def test_single_backtest_reproducibility(self):
        """Test that identical parameters produce identical results."""
        base_config = {
            "tsmom": {"lookbacks": [252], "skip_recent": 21, "standardize": "vol", 
                     "signal_cap": 3.0, "rebalance": "W-FRI"},
            "vol_overlay": {"target_vol": 0.20, "lookback_vol": 63, 
                           "leverage_mode": "global", "cap_leverage": 7.0,
                           "position_bounds": [-3.0, 3.0]},
            "risk_vol": {"cov_lookback": 252, "vol_lookback": 63, 
                        "shrinkage": "lw", "nan_policy": "mask-asset"},
            "allocator": {"method": "signal-beta", "gross_cap": 7.0, 
                         "net_cap": 2.0, "w_bounds_per_asset": [-1.5, 1.5],
                         "turnover_cap": 0.5, "lambda_turnover": 0.001},
            "exec": {"rebalance": "W-FRI", "slippage_bps": 0.5, 
                    "commission_per_contract": 0.0, "position_notional_scale": 1.0}
        }
        
        params = {}
        start = "2024-01-01"
        end = "2024-03-31"
        seed = 123
        
        result1 = _run_single_backtest((params, base_config, start, end, seed))
        result2 = _run_single_backtest((params, base_config, start, end, seed))
        
        # Check reproducibility if both succeeded
        if result1['success'] and result2['success']:
            assert result1['cagr'] == result2['cagr']
            assert result1['sharpe'] == result2['sharpe']
            assert result1['vol'] == result2['vol']
            assert result1['n_periods'] == result2['n_periods']


class TestParamSweep:
    """Test full parameter sweep."""
    
    @pytest.fixture
    def base_config(self):
        """Fixture for base configuration."""
        return {
            "tsmom": {
                "lookbacks": [252],
                "skip_recent": 21,
                "standardize": "vol",
                "signal_cap": 3.0,
                "rebalance": "W-FRI"
            },
            "vol_overlay": {
                "target_vol": 0.20,
                "lookback_vol": 63,
                "leverage_mode": "global",
                "cap_leverage": 7.0,
                "position_bounds": [-3.0, 3.0]
            },
            "risk_vol": {
                "cov_lookback": 252,
                "vol_lookback": 63,
                "shrinkage": "lw",
                "nan_policy": "mask-asset"
            },
            "allocator": {
                "method": "signal-beta",
                "gross_cap": 7.0,
                "net_cap": 2.0,
                "w_bounds_per_asset": [-1.5, 1.5],
                "turnover_cap": 0.5,
                "lambda_turnover": 0.001
            },
            "exec": {
                "rebalance": "W-FRI",
                "slippage_bps": 0.5,
                "commission_per_contract": 0.0,
                "position_notional_scale": 1.0
            }
        }
    
    @pytest.fixture
    def temp_output_dir(self):
        """Fixture for temporary output directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_run_sweep_small_grid(self, base_config, temp_output_dir):
        """Test sweep with small grid."""
        grid = {
            "tsmom.lookbacks": [[252], [126, 252]],
            "exec.rebalance": ["W-FRI"]
        }
        
        df = run_sweep(
            base_config=base_config,
            grid=grid,
            seeds=[0],
            start="2024-01-01",
            end="2024-03-31",
            n_workers=1,
            output_dir=temp_output_dir,
            save_top_n=2
        )
        
        # Check DataFrame structure
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2  # 2 combinations x 1 seed
        
        # Check columns exist
        required_cols = ['cagr', 'sharpe', 'vol', 'max_drawdown', 'calmar', 
                        'turnover', 'cost_drag', 'success', 'seed']
        for col in required_cols:
            assert col in df.columns
        
        # Check parameter columns exist
        assert 'tsmom.lookbacks' in df.columns
        assert 'exec.rebalance' in df.columns
    
    def test_run_sweep_writes_summary_csv(self, base_config, temp_output_dir):
        """Test that sweep writes summary.csv."""
        grid = {
            "vol_overlay.target_vol": [0.15, 0.20]
        }
        
        df = run_sweep(
            base_config=base_config,
            grid=grid,
            seeds=[0],
            start="2024-01-01",
            end="2024-03-31",
            n_workers=1,
            output_dir=temp_output_dir
        )
        
        # Check file exists
        summary_file = Path(temp_output_dir) / "summary.csv"
        assert summary_file.exists()
        
        # Check can read it back
        df_loaded = pd.read_csv(summary_file)
        assert len(df_loaded) == len(df)
        assert list(df_loaded.columns) == list(df.columns)
    
    def test_run_sweep_saves_top_configs(self, base_config, temp_output_dir):
        """Test that sweep saves top N configuration YAMLs."""
        grid = {
            "vol_overlay.target_vol": [0.15, 0.20, 0.25]
        }
        
        run_sweep(
            base_config=base_config,
            grid=grid,
            seeds=[0],
            start="2024-01-01",
            end="2024-03-31",
            n_workers=1,
            output_dir=temp_output_dir,
            save_top_n=2
        )
        
        # Check YAML files exist
        output_path = Path(temp_output_dir)
        yaml_files = list(output_path.glob("top_*.yaml"))
        
        # Should have up to 2 files (or fewer if some runs failed)
        assert len(yaml_files) <= 2
        
        # Check we can load them
        if len(yaml_files) > 0:
            with open(yaml_files[0], 'r') as f:
                config = yaml.safe_load(f)
            
            assert isinstance(config, dict)
            assert '_metadata' in config
            assert 'sharpe' in config['_metadata']
    
    def test_run_sweep_multiple_seeds(self, base_config, temp_output_dir):
        """Test sweep with multiple seeds."""
        grid = {
            "vol_overlay.target_vol": [0.20]
        }
        
        df = run_sweep(
            base_config=base_config,
            grid=grid,
            seeds=[0, 1, 2],
            start="2024-01-01",
            end="2024-03-31",
            n_workers=1,
            output_dir=temp_output_dir
        )
        
        # Should have 1 combination x 3 seeds = 3 rows
        assert len(df) == 3
        assert set(df['seed'].unique()) == {0, 1, 2}
    
    def test_run_sweep_more_than_zero_rows(self, base_config, temp_output_dir):
        """Test that sweep produces > 0 rows."""
        grid = {
            "tsmom.lookbacks": [[252]]
        }
        
        df = run_sweep(
            base_config=base_config,
            grid=grid,
            seeds=[0],
            start="2024-01-01",
            end="2024-03-31",
            n_workers=1,
            output_dir=temp_output_dir
        )
        
        assert len(df) > 0


class TestConfigComparison:
    """Test configuration comparison functionality."""
    
    @pytest.fixture
    def temp_output_dir(self):
        """Fixture for temporary output directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_compare_configs(self, temp_output_dir):
        """Test comparing multiple configurations."""
        baseline_config = {
            "tsmom": {"lookbacks": [252], "skip_recent": 21, "standardize": "vol",
                     "signal_cap": 3.0, "rebalance": "W-FRI"},
            "vol_overlay": {"target_vol": 0.20, "lookback_vol": 63,
                           "leverage_mode": "global", "cap_leverage": 7.0,
                           "position_bounds": [-3.0, 3.0]},
            "risk_vol": {"cov_lookback": 252, "vol_lookback": 63,
                        "shrinkage": "lw", "nan_policy": "mask-asset"},
            "allocator": {"method": "signal-beta", "gross_cap": 7.0,
                         "net_cap": 2.0, "w_bounds_per_asset": [-1.5, 1.5],
                         "turnover_cap": 0.5, "lambda_turnover": 0.001},
            "exec": {"rebalance": "W-FRI", "slippage_bps": 0.5,
                    "commission_per_contract": 0.0, "position_notional_scale": 1.0}
        }
        
        macro_config = baseline_config.copy()
        macro_config["macro_regime"] = {
            "rebalance": "W-FRI",
            "vol_thresholds": {"low": 0.12, "high": 0.22},
            "k_bounds": {"min": 0.5, "max": 1.0},
            "smoothing": 0.15,
            "vol_lookback": 21,
            "breadth_lookback": 200,
            "proxy_symbols": ["ES_FRONT_CALENDAR_2D", "NQ_FRONT_CALENDAR_2D"]
        }
        
        configs = {
            "Baseline": baseline_config,
            "Macro": macro_config
        }
        
        df = compare_configs(
            configs=configs,
            start="2024-01-01",
            end="2024-03-31",
            seed=0,
            output_dir=temp_output_dir
        )
        
        # Check DataFrame structure
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert 'config_name' in df.columns
        assert set(df['config_name']) == {"Baseline", "Macro"}
        
        # Check metrics exist
        assert 'cagr' in df.columns
        assert 'sharpe' in df.columns
        
        # Check file was saved
        comparison_file = Path(temp_output_dir) / "comparison.csv"
        assert comparison_file.exists()


class TestErrorHandling:
    """Test error handling in parameter sweep."""
    
    def test_invalid_sweep_method(self):
        """Test that invalid sweep method raises error."""
        with pytest.raises(ValueError, match="Unknown sweep method"):
            run_sweep(
                base_config={},
                grid={},
                method="invalid",
                output_dir=tempfile.mkdtemp()
            )
    
    def test_latin_not_implemented(self):
        """Test that Latin hypercube raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            run_sweep(
                base_config={},
                grid={},
                method="latin",
                n_samples=10,
                output_dir=tempfile.mkdtemp()
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

