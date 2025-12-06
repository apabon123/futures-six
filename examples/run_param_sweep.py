"""
Example Parameter Sweep Script

Demonstrates how to use ParamSweepRunner to explore configuration space.

This script shows:
1. Running a grid sweep over multiple parameters
2. Comparing specific configurations (Baseline vs Macro vs Macro+XSec)
3. Analyzing results

To run:
    python examples/run_param_sweep.py --mode grid
    python examples/run_param_sweep.py --mode compare
"""

import sys
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.param_sweep import run_sweep, compare_configs


def load_base_config():
    """Load base configuration."""
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


def run_grid_sweep():
    """Run comprehensive grid sweep over key parameters."""
    
    print("\n" + "=" * 80)
    print("GRID SWEEP: Exploring Configuration Space")
    print("=" * 80)
    
    base_config = load_base_config()
    
    # Define parameter grid
    # Note: Start with a smaller grid for testing, then expand
    grid = {
        # Macro regime filter parameters
        "macro_regime.vol_thresholds.low": [0.10, 0.12, 0.14],
        "macro_regime.vol_thresholds.high": [0.20, 0.22, 0.25],
        "macro_regime.k_bounds.min": [0.5, 0.6, 0.7],
        
        # TSMOM parameters
        "tsmom.lookbacks": [[252], [126, 252], [63, 126, 252]],
        
        # Execution parameters
        "exec.rebalance": ["W-FRI", "M"],
        
        # Sleeve allocations (simplified - just vol overlay target)
        "vol_overlay.target_vol": [0.15, 0.20, 0.25]
    }
    
    # Add macro regime proxy symbols to all configs
    base_config["macro_regime"] = {
        "rebalance": "W-FRI",
        "vol_thresholds": {"low": 0.12, "high": 0.22},
        "k_bounds": {"min": 0.5, "max": 1.0},
        "smoothing": 0.15,
        "vol_lookback": 21,
        "breadth_lookback": 200,
        "proxy_symbols": ["ES_FRONT_CALENDAR_2D", "NQ_FRONT_CALENDAR_2D"]
    }
    
    print(f"\nGrid dimensions:")
    for key, values in grid.items():
        print(f"  {key:40s}: {len(values)} values")
    
    total_combos = 1
    for values in grid.values():
        total_combos *= len(values)
    
    print(f"\nTotal combinations: {total_combos}")
    print("\nNote: This is a LARGE sweep. For testing, consider:")
    print("  1. Reducing grid size")
    print("  2. Using shorter backtest period")
    print("  3. Running with n_workers > 1 for parallelization")
    
    # Run sweep
    results = run_sweep(
        base_config=base_config,
        grid=grid,
        seeds=[0],  # Add more seeds for robustness: [0, 1, 2]
        start="2021-01-01",
        end="2025-11-05",
        n_workers=None,  # Auto-detect CPU count
        save_top_n=10
    )
    
    print(f"\nResults saved to: reports/sweeps/<timestamp>/")
    print(f"  - summary.csv: All results")
    print(f"  - top_*.yaml: Top 10 configurations")
    
    return results


def run_small_test_sweep():
    """Run a small test sweep for quick validation."""
    
    print("\n" + "=" * 80)
    print("TEST SWEEP: Quick Parameter Exploration")
    print("=" * 80)
    
    base_config = load_base_config()
    
    # Small grid for testing
    grid = {
        "tsmom.lookbacks": [[252], [126, 252]],
        "vol_overlay.target_vol": [0.15, 0.20],
        "exec.rebalance": ["W-FRI"]
    }
    
    print(f"\nTest grid: {len(grid)} parameters")
    print(f"Combinations: 2 x 2 x 1 = 4 backtests")
    
    results = run_sweep(
        base_config=base_config,
        grid=grid,
        seeds=[0],
        start="2024-01-01",
        end="2024-12-31",
        n_workers=1,
        save_top_n=2
    )
    
    print("\n" + "=" * 80)
    print("Top Results:")
    print("=" * 80)
    
    if len(results[results['success']]) > 0:
        top_results = results[results['success']].nlargest(3, 'sharpe')
        print(top_results[['tsmom.lookbacks', 'vol_overlay.target_vol', 
                          'sharpe', 'cagr', 'max_drawdown']])
    
    return results


def run_comparison():
    """Compare specific configurations: Baseline vs Macro vs Macro+XSec."""
    
    print("\n" + "=" * 80)
    print("CONFIGURATION COMPARISON")
    print("=" * 80)
    
    # 1. Baseline: No macro filter
    baseline = {
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
    
    # 2. Macro: Add macro regime filter
    macro = baseline.copy()
    macro["macro_regime"] = {
        "rebalance": "W-FRI",
        "vol_thresholds": {"low": 0.12, "high": 0.22},
        "k_bounds": {"min": 0.5, "max": 1.0},
        "smoothing": 0.15,
        "vol_lookback": 21,
        "breadth_lookback": 200,
        "proxy_symbols": ["ES_FRONT_CALENDAR_2D", "NQ_FRONT_CALENDAR_2D"]
    }
    
    # 3. Macro + Multi-lookback (simulate cross-sectional momentum)
    macro_xsec = macro.copy()
    macro_xsec["tsmom"]["lookbacks"] = [63, 126, 252]
    
    # 4. Macro + Tuned thresholds
    macro_tuned = macro.copy()
    macro_tuned["macro_regime"]["vol_thresholds"] = {"low": 0.10, "high": 0.25}
    macro_tuned["macro_regime"]["k_bounds"] = {"min": 0.6, "max": 1.0}
    
    configs = {
        "Baseline": baseline,
        "Macro": macro,
        "Macro+XSec": macro_xsec,
        "Macro+Tuned": macro_tuned
    }
    
    results = compare_configs(
        configs=configs,
        start="2021-01-01",
        end="2025-11-05",
        seed=0
    )
    
    print("\n" + "=" * 80)
    print("Comparison Summary:")
    print("=" * 80)
    
    if len(results[results['success']]) > 0:
        successful = results[results['success']]
        print("\nSharpe Ranking:")
        for i, (_, row) in enumerate(successful.sort_values('sharpe', ascending=False).iterrows(), 1):
            print(f"  {i}. {row['config_name']:20s} - Sharpe: {row['sharpe']:6.2f}")
        
        print("\nCAGR Ranking:")
        for i, (_, row) in enumerate(successful.sort_values('cagr', ascending=False).iterrows(), 1):
            print(f"  {i}. {row['config_name']:20s} - CAGR: {row['cagr']:7.2%}")
        
        print("\nMax Drawdown (best = least negative):")
        for i, (_, row) in enumerate(successful.sort_values('max_drawdown', ascending=False).iterrows(), 1):
            print(f"  {i}. {row['config_name']:20s} - MaxDD: {row['max_drawdown']:7.2%}")
    
    return results


def run_sleeve_budget_sweep():
    """Sweep over sleeve risk budget allocations."""
    
    print("\n" + "=" * 80)
    print("SLEEVE BUDGET SWEEP")
    print("=" * 80)
    print("Note: This is a conceptual example. Actual sleeve implementation")
    print("      would require SleeveAllocator agent.")
    
    base_config = load_base_config()
    
    # Conceptual grid - in practice, you'd use SleeveAllocator
    grid = {
        "vol_overlay.target_vol": [0.15, 0.20, 0.25],  # Simulate total risk budget
        "tsmom.lookbacks": [[252], [126, 252]],  # TSMOM vs TSMOM+XSec
    }
    
    results = run_sweep(
        base_config=base_config,
        grid=grid,
        seeds=[0],
        start="2021-01-01",
        end="2025-11-05",
        n_workers=None,
        save_top_n=5
    )
    
    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run parameter sweeps")
    parser.add_argument(
        "--mode",
        choices=["grid", "test", "compare", "sleeves"],
        default="test",
        help="Sweep mode: grid (full), test (small), compare (specific configs), sleeves (budget allocation)"
    )
    
    args = parser.parse_args()
    
    if args.mode == "grid":
        results = run_grid_sweep()
    elif args.mode == "test":
        results = run_small_test_sweep()
    elif args.mode == "compare":
        results = run_comparison()
    elif args.mode == "sleeves":
        results = run_sleeve_budget_sweep()
    else:
        print(f"Unknown mode: {args.mode}")
        sys.exit(1)
    
    print("\n" + "=" * 80)
    print("SWEEP COMPLETE")
    print("=" * 80)
    print("\nNext steps:")
    print("  1. Review summary.csv for full results")
    print("  2. Check top_*.yaml files for best configurations")
    print("  3. Run specific configs with run_strategy.py")
    print("  4. Analyze equity curves and weights")
    
    return results


if __name__ == "__main__":
    main()

