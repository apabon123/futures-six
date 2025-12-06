# Parameter Sweep Runner

The `ParamSweepRunner` module provides a systematic way to explore the configuration space of the futures-six strategy. It supports grid search and configuration comparison to find optimal parameter combinations.

## Features

- **Grid Search**: Exhaustive exploration of parameter combinations
- **Parallel Execution**: Multi-process execution for faster sweeps
- **Deterministic**: Reproducible results with seed control
- **Comprehensive Metrics**: CAGR, Sharpe, vol, maxDD, Calmar, turnover, cost drag
- **Automatic Saving**: CSV summaries and top-N YAML configurations
- **Configuration Comparison**: Easy A/B testing of specific setups

## Quick Start

### 1. Basic Test Sweep

Run a small sweep to verify functionality:

```python
from src.agents.param_sweep import run_sweep

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

grid = {
    "tsmom.lookbacks": [[252], [126, 252]],
    "vol_overlay.target_vol": [0.15, 0.20]
}

results = run_sweep(
    base_config=base_config,
    grid=grid,
    seeds=[0],
    start="2024-01-01",
    end="2024-12-31"
)
```

### 2. Run Example Script

Use the provided example script:

```bash
# Quick test sweep
python examples/run_param_sweep.py --mode test

# Compare specific configurations
python examples/run_param_sweep.py --mode compare

# Full grid sweep (warning: can be slow!)
python examples/run_param_sweep.py --mode grid
```

## API Reference

### `run_sweep()`

Main function for running parameter sweeps.

**Signature:**
```python
def run_sweep(
    base_config: dict,
    grid: Dict[str, List],
    seeds: List[int] = None,
    start: str = "2021-01-01",
    end: str = "2025-11-05",
    method: str = "grid",
    n_samples: Optional[int] = None,
    n_workers: Optional[int] = None,
    output_dir: Optional[str] = None,
    save_top_n: int = 10
) -> pd.DataFrame
```

**Parameters:**
- `base_config`: Base configuration dictionary (modified by grid params)
- `grid`: Parameter grid with dot-notation keys mapping to value lists
- `seeds`: List of random seeds for reproducibility (default: `[0]`)
- `start`: Backtest start date
- `end`: Backtest end date
- `method`: Sweep method (`"grid"` or `"latin"` - Latin not yet implemented)
- `n_samples`: Number of samples for Latin hypercube (ignored for grid)
- `n_workers`: Number of parallel workers (default: `cpu_count() - 1`)
- `output_dir`: Output directory (default: `reports/sweeps/<timestamp>`)
- `save_top_n`: Number of top configurations to save as YAML

**Returns:**
- DataFrame with columns:
  - All parameter values from grid
  - `cagr`: Compound annual growth rate
  - `vol`: Annualized volatility
  - `sharpe`: Sharpe ratio
  - `max_drawdown`: Maximum drawdown
  - `calmar`: Calmar ratio (CAGR / |maxDD|)
  - `turnover`: Average turnover per rebalance
  - `cost_drag`: Slippage cost (bps * turnover)
  - `hit_rate`: Fraction of positive returns
  - `gross_leverage`: Average gross leverage
  - `net_exposure`: Average net exposure
  - `n_periods`: Number of rebalance periods
  - `seed`: Random seed used
  - `success`: Whether backtest completed successfully
  - `error`: Error message if failed

**Output Files:**
- `summary.csv`: All results in tidy format
- `top_NN_sharpe_X.XX.yaml`: Top N configurations ranked by Sharpe

### `compare_configs()`

Compare multiple named configurations.

**Signature:**
```python
def compare_configs(
    configs: Dict[str, dict],
    start: str = "2021-01-01",
    end: str = "2025-11-05",
    seed: int = 0,
    output_dir: Optional[str] = None
) -> pd.DataFrame
```

**Parameters:**
- `configs`: Dict mapping config names to config dictionaries
- `start`: Backtest start date
- `end`: Backtest end date
- `seed`: Random seed
- `output_dir`: Output directory (default: `reports/sweeps/comparison_<timestamp>`)

**Returns:**
- DataFrame with one row per configuration

**Example:**
```python
from src.agents.param_sweep import compare_configs

configs = {
    "Baseline": baseline_config,
    "Macro": macro_config,
    "Macro+XSec": macro_xsec_config
}

results = compare_configs(configs)
```

## Grid Specification

Use dot-notation to specify nested parameters:

```python
grid = {
    # Macro regime parameters
    "macro_regime.vol_thresholds.low": [0.10, 0.12, 0.14],
    "macro_regime.vol_thresholds.high": [0.20, 0.22, 0.25],
    "macro_regime.k_bounds.min": [0.5, 0.6, 0.7],
    
    # TSMOM parameters
    "tsmom.lookbacks": [[252], [126, 252], [63, 126, 252]],
    "tsmom.skip_recent": [21, 42],
    
    # Execution parameters
    "exec.rebalance": ["W-FRI", "M"],
    "exec.slippage_bps": [0.5, 1.0],
    
    # Vol overlay parameters
    "vol_overlay.target_vol": [0.15, 0.20, 0.25],
    
    # Allocator parameters
    "allocator.gross_cap": [5.0, 7.0, 10.0],
}
```

### Parameter Types

The grid supports any JSON-serializable values:
- **Numbers**: `[0.5, 1.0, 1.5]`
- **Strings**: `["W-FRI", "M"]`
- **Lists**: `[[252], [126, 252]]`
- **Dicts**: `[{"low": 0.1, "high": 0.2}]`

## Example Sweeps

### 1. Macro Regime Tuning

Optimize macro regime filter thresholds:

```python
grid = {
    "macro_regime.vol_thresholds.low": [0.10, 0.12, 0.14],
    "macro_regime.vol_thresholds.high": [0.20, 0.22, 0.25],
    "macro_regime.k_bounds.min": [0.5, 0.6, 0.7],
    "macro_regime.smoothing": [0.10, 0.15, 0.20]
}
# Total: 3 * 3 * 3 * 3 = 81 combinations
```

### 2. TSMOM Lookback Exploration

Test different momentum lookback periods:

```python
grid = {
    "tsmom.lookbacks": [
        [252],           # 12-1 month
        [126, 252],      # 6+12-1 month
        [63, 126, 252],  # 3+6+12-1 month
        [42, 126, 252],  # 2+6+12-1 month
    ],
    "tsmom.skip_recent": [21, 42]
}
# Total: 4 * 2 = 8 combinations
```

### 3. Rebalance Frequency

Compare weekly vs monthly rebalancing:

```python
grid = {
    "exec.rebalance": ["W-FRI", "M"],
    "tsmom.rebalance": ["W-FRI", "M"],  # Must match exec rebalance
}
# Total: 2 combinations (paired)
```

### 4. Risk Budget Allocation

Explore different volatility targets:

```python
grid = {
    "vol_overlay.target_vol": [0.10, 0.15, 0.20, 0.25, 0.30],
    "vol_overlay.leverage_mode": ["global", "asset"],
    "vol_overlay.cap_leverage": [5.0, 7.0, 10.0]
}
# Total: 5 * 2 * 3 = 30 combinations
```

### 5. Comprehensive Sweep

Kitchen sink exploration (warning: expensive!):

```python
grid = {
    "macro_regime.vol_thresholds.low": [0.10, 0.12, 0.14],
    "macro_regime.vol_thresholds.high": [0.20, 0.22, 0.25],
    "macro_regime.k_bounds.min": [0.5, 0.6, 0.7],
    "tsmom.lookbacks": [[252], [126, 252], [63, 126, 252]],
    "exec.rebalance": ["W-FRI", "M"],
    "vol_overlay.target_vol": [0.15, 0.20, 0.25]
}
# Total: 3 * 3 * 3 * 3 * 2 * 3 = 486 combinations
```

## Performance Considerations

### Parallelization

The sweep runner uses multiprocessing to parallelize backtests:

```python
# Auto-detect CPU count (default)
results = run_sweep(..., n_workers=None)

# Explicit worker count
results = run_sweep(..., n_workers=4)

# Serial execution (for debugging)
results = run_sweep(..., n_workers=1)
```

**Recommendations:**
- Use `n_workers = cpu_count() - 1` for production sweeps
- Use `n_workers = 1` for debugging
- Each worker runs independently with no shared state

### Runtime Estimates

Approximate runtime per backtest (2021-2025, weekly rebalance):
- Single run: ~0.5 seconds (fast)
- 100 runs: ~50 seconds (1 worker), ~10 seconds (8 workers)
- 500 runs: ~250 seconds (1 worker), ~50 seconds (8 workers)

**Tips for large sweeps:**
1. Start with a test grid (2-3 params, 2-3 values each)
2. Run on shorter period first (e.g., 2024 only)
3. Use multiple seeds only for final validation
4. Parallelize heavily (8+ workers on capable machines)

## Output Analysis

### 1. Review Summary CSV

Load and analyze results:

```python
import pandas as pd

df = pd.read_csv("reports/sweeps/<timestamp>/summary.csv")

# Filter successful runs
df_ok = df[df['success']]

# Top 10 by Sharpe
top_sharpe = df_ok.nlargest(10, 'sharpe')
print(top_sharpe[['sharpe', 'cagr', 'max_drawdown', 'tsmom.lookbacks', 
                  'macro_regime.vol_thresholds.low']])

# Scatter plot: risk vs return
import matplotlib.pyplot as plt
plt.scatter(df_ok['vol'], df_ok['cagr'], c=df_ok['sharpe'], cmap='viridis')
plt.xlabel('Volatility')
plt.ylabel('CAGR')
plt.colorbar(label='Sharpe')
plt.show()
```

### 2. Parameter Importance

Analyze which parameters matter most:

```python
# Group by parameter and compute mean Sharpe
for param in ['tsmom.lookbacks', 'exec.rebalance', 'vol_overlay.target_vol']:
    print(f"\n{param}:")
    print(df_ok.groupby(param)['sharpe'].agg(['mean', 'std', 'count']))
```

### 3. Load Top Configuration

Use top configuration in production:

```python
import yaml

# Load top config
with open("reports/sweeps/<timestamp>/top_01_sharpe_X.XX.yaml", 'r') as f:
    best_config = yaml.safe_load(f)

# Remove metadata
del best_config['_metadata']

# Use with run_strategy.py or custom script
```

## Best Practices

### 1. Hierarchical Sweeps

Start broad, then narrow:

```python
# Phase 1: Coarse grid
grid_coarse = {
    "macro_regime.vol_thresholds.low": [0.08, 0.12, 0.16],
    "vol_overlay.target_vol": [0.15, 0.20, 0.25]
}

# Phase 2: Fine grid around best region
grid_fine = {
    "macro_regime.vol_thresholds.low": [0.10, 0.11, 0.12, 0.13, 0.14],
    "vol_overlay.target_vol": [0.18, 0.19, 0.20, 0.21, 0.22]
}
```

### 2. Robustness Checks

Use multiple seeds to validate:

```python
# Single seed for exploration
results = run_sweep(..., seeds=[0])

# Multiple seeds for top configs
top_params = results.nlargest(5, 'sharpe')['params']
results_robust = run_sweep(..., grid=top_params, seeds=[0, 1, 2, 3, 4])
```

### 3. Out-of-Sample Testing

Split data for validation:

```python
# In-sample optimization (2021-2023)
results_is = run_sweep(..., start="2021-01-01", end="2023-12-31")

# Out-of-sample test (2024-2025)
best_config = load_top_config(results_is)
results_oos = run_single_backtest(best_config, start="2024-01-01", end="2025-11-05")
```

### 4. Metric Selection

Choose appropriate objective:
- **Sharpe**: Risk-adjusted returns (default)
- **Calmar**: Drawdown-adjusted returns
- **CAGR**: Absolute returns (ignore risk)
- **MaxDD**: Worst-case loss (defensive)

```python
# Rank by Calmar instead of Sharpe
df.nlargest(10, 'calmar')

# Multi-objective: high Sharpe AND low drawdown
df_filtered = df[(df['sharpe'] > 1.5) & (df['max_drawdown'] > -0.15)]
```

## Troubleshooting

### Issue: Sweep too slow

**Solution:**
1. Increase `n_workers`
2. Reduce backtest period
3. Reduce grid size
4. Use coarser rebalance frequency (M instead of W-FRI)

### Issue: All backtests failing

**Solution:**
1. Run single backtest first with `n_workers=1` to see error
2. Check MarketData connection
3. Verify config structure
4. Ensure lookback periods have sufficient data

### Issue: High variance across seeds

**Solution:**
1. Use more seeds (5-10)
2. Longer backtest period
3. Check if parameter is overfitting

### Issue: Top configs look similar

**Solution:**
1. Grid too narrow - expand parameter ranges
2. Parameters might not matter much - this is useful information!
3. Consider different parameter families

## Future Enhancements

Planned features:
- [ ] Latin hypercube sampling for high-dimensional spaces
- [ ] Bayesian optimization for guided search
- [ ] Multi-objective optimization (Pareto frontier)
- [ ] Walk-forward analysis with rolling windows
- [ ] Automated outlier detection
- [ ] Parallel tempering for robustness

## Related Documentation

- [Trend Meta-Sleeve Implementation](META_SLEEVES/TREND_IMPLEMENTATION.md): Current production implementation
- [Legacy TSMOM](legacy/TSMOM_IMPLEMENTATION.md): Legacy single-horizon TSMOM class
- [Macro Regime Filter](MACRO_REGIME_FILTER.md): Regime overlay
- [Strategy Flow](STRATEGY.md): Full system overview
- [README](../README.md): Project overview

## Citation

If you use this parameter sweep framework in your research, please cite:

```
futures-six Parameter Sweep Runner
https://github.com/yourusername/futures-six
```

