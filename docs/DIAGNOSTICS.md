# Performance Diagnostics

This document describes the performance diagnostics framework for analyzing backtest runs.

## Overview

The diagnostics framework provides a systematic way to:
- Load and analyze backtest run artifacts
- Compute core performance metrics (CAGR, Sharpe, MaxDD, etc.)
- Break down performance by year and by asset
- Compare runs against baselines for ablation testing

## Architecture

### Core Module: `src/agents/diagnostics_perf.py`

The diagnostics module provides:

1. **`RunData` dataclass**: Container for backtest run data
   - `run_id`: Run identifier
   - `portfolio_returns`: Daily portfolio returns (simple returns)
   - `equity_curve`: Daily equity curve (cumulative)
   - `asset_returns`: Daily asset returns (simple returns)
   - `weights`: Rebalance-date weights
   - `meta`: Run metadata (config, dates, universe, etc.)

2. **`load_run()`**: Load run artifacts from disk
   - Reads from `reports/runs/{run_id}/`
   - Loads: `portfolio_returns.csv`, `equity_curve.csv`, `asset_returns.csv`, `weights.csv`, `meta.json`
   - Returns a `RunData` object

3. **`compute_core_metrics()`**: Calculate core performance metrics
   - CAGR (Compound Annual Growth Rate)
   - Annualized Volatility
   - Sharpe Ratio (annualized)
   - Maximum Drawdown
   - Hit Rate (fraction of positive return days)

4. **`compute_yearly_stats()`**: Year-by-year performance breakdown
   - CAGR, Volatility, Sharpe, MaxDD per year
   - Helps identify which years drove performance

5. **`compute_per_asset_stats()`**: Per-asset attribution
   - Annualized return contribution
   - Annualized volatility
   - Sharpe ratio per asset
   - Identifies which assets contributed positively/negatively

6. **`compare_to_baseline()`**: Compare two runs
   - Computes deltas in all core metrics
   - Shows equity ratio over time
   - Essential for ablation testing

### CLI Script: `scripts/run_perf_diagnostics.py`

Command-line interface for running diagnostics:

```bash
# Analyze a single run
python scripts/run_perf_diagnostics.py --run_id <run_id>

# Compare against a baseline
python scripts/run_perf_diagnostics.py --run_id <run_id> --baseline_id <baseline_id>

# Specify custom base directory
python scripts/run_perf_diagnostics.py --run_id <run_id> --base_dir <path>
```

## Usage

### Running Diagnostics

**Basic usage:**
```bash
python scripts/run_perf_diagnostics.py --run_id momentum_only_v1
```

**With baseline comparison:**
```bash
python scripts/run_perf_diagnostics.py \
  --run_id momentum_stack_v1 \
  --baseline_id momentum_only_v1
```

### Output Format

The diagnostics script prints:

1. **Core Metrics**: CAGR, Vol, Sharpe, MaxDD, HitRate
2. **Year-by-Year Stats**: Performance breakdown by calendar year
3. **Per-Asset Stats**: All assets sorted by annualized return contribution
4. **Baseline Comparison** (if provided):
   - Current vs baseline metrics
   - Deltas (current - baseline)
   - Equity ratio over time

### Example Output

```
================================================================================
CORE METRICS
================================================================================
  CAGR           :     0.0033 (0.33%)
  Vol            :     0.1645 (16.45%)
  Sharpe         :     0.1023
  MaxDD          :    -0.4062 (-40.62%)
  HitRate        :     0.5003 (50.03%)

================================================================================
YEAR-BY-YEAR STATS
================================================================================
          CAGR       Vol    Sharpe     MaxDD
Year
2021  0.151500  0.122962  1.234567 -0.097163
2022 -0.012500  0.151184 -0.082654 -0.168414
2023 -0.051400  0.126604 -0.405678 -0.174025
2024 -0.080800  0.108487 -0.744444 -0.148823
2025  0.051900  0.146115  0.355123 -0.140045

================================================================================
PER-ASSET STATS (all assets, sorted by AnnRet)
================================================================================
                         AnnRet    AnnVol    Sharpe
symbol
ES_FRONT_CALENDAR_2D   0.015900  0.031549  0.503755
6E_FRONT_CALENDAR      0.003207  0.015480  0.207173
...
Total assets: 13

================================================================================
BASELINE COMPARISON
================================================================================
Current:  momentum_stack_v1
Baseline: momentum_only_v1

Delta (Current - Baseline):
  CAGR_delta          :    -0.0026 (-0.26%)
  Sharpe_delta        :    -0.1346
  MaxDD_delta         :    -8.28% (worse)
  ...
```

## Canonical Medium-Term Diagnostics Examples

### Phase-0: Sign-Only Sanity Check
```bash
python scripts/run_trend_med_canonical_phase0.py --start 2020-01-01 --end 2025-10-31
# Results saved to: reports/sanity_checks/trend/medium_canonical/archive/{timestamp}/
# Phase-0 index: reports/phase_index/trend/medium_canonical/phase0.txt
```

### Phase-1: Standalone Canonical vs Legacy Comparison
```bash
python scripts/run_trend_med_canonical_phase1.py --start 2020-01-01 --end 2025-10-31
# Results saved to: reports/runs/med_canonical_phase1_standalone_{timestamp}/
#                    reports/runs/med_canonical_phase1_legacy_{timestamp}/
# Phase-1 index: reports/phase_index/trend/medium_canonical/phase1.txt

# Compare canonical vs legacy:
python scripts/run_perf_diagnostics.py \
  --run_id med_canonical_phase1_standalone_{timestamp} \
  --baseline_id med_canonical_phase1_legacy_{timestamp}
```

### Phase-2: Integrated A/B Test (Canonical vs Baseline)
```bash
# Run with canonical medium-term variant integrated into Trend Meta-Sleeve:
python run_strategy.py \
  --strategy_profile core_v3_trend_medcanon_no_macro \
  --run_id core_v3_medcanon_v1 \
  --start 2020-01-01 \
  --end 2025-10-31

# Compare to baseline (legacy medium-term):
python scripts/run_perf_diagnostics.py \
  --run_id core_v3_medcanon_v1 \
  --baseline_id core_v3_no_macro

# Phase-2 index: reports/phase_index/trend/medium_canonical/phase2.txt
```

## Canonical Short-Term Diagnostics Examples

### Phase-0: Sign-Only Sanity Check
```bash
python scripts/run_trend_short_canonical_phase0.py --start 2020-01-01 --end 2025-10-31
# Results saved to: reports/sanity_checks/trend/short_canonical/archive/{timestamp}/
# Phase-0 index: reports/phase_index/trend/short_canonical/phase0.txt
```

### Phase-1: Standalone Canonical vs Legacy Comparison
```bash
python scripts/run_trend_short_canonical_phase1.py --start 2020-01-01 --end 2025-10-31
# Results saved to: reports/runs/short_canonical_phase1_standalone_{timestamp}/
#                    reports/runs/short_canonical_phase1_legacy_{timestamp}/
# Phase-1 index: reports/phase_index/trend/short_canonical/phase1.txt

# Compare canonical vs legacy:
python scripts/run_perf_diagnostics.py \
  --run_id short_canonical_phase1_standalone_{timestamp} \
  --baseline_id short_canonical_phase1_legacy_{timestamp}
```

### Phase-2: Integrated A/B Test (Canonical vs Baseline)
```bash
# Run with canonical short-term variant integrated into Trend Meta-Sleeve:
python run_strategy.py \
  --strategy_profile core_v3_trend_shortcanon_no_macro \
  --run_id core_v3_shortcanon_v1 \
  --start 2020-01-01 \
  --end 2025-10-31

# Compare to baseline (legacy short-term):
python scripts/run_perf_diagnostics.py \
  --run_id core_v3_shortcanon_v1 \
  --baseline_id core_v3_no_macro
```

**Note**: These diagnostics were used to evaluate the equal-weight short-term variant (Phase-0/1/2 completed Nov 2025). The variant was **not promoted** to production; legacy weights (0.5, 0.3, 0.2) remain the production standard. The diagnostic commands and canonical variant code (`variant="canonical"`) are preserved for future re-testing if the universe, timeframe, or surrounding architecture changes. See `TREND_RESEARCH.md` and `PROCEDURES.md` for full results and rationale.

# Phase-2 index: reports/phase_index/trend/short_canonical/phase2.txt
```

## Reports Directory Structure

The reports directory uses a canonical structure that makes it easy to find the current Phase-0, Phase-1, and Phase-2 results for each atomic sleeve:

```
reports/
  sanity_checks/              # Phase-0 raw logs (all runs)
    {meta_sleeve}/
      {sleeve_name}/
        archive/
          {timestamp}/        # All historical Phase-0 runs
        latest/               # Canonical Phase-0 results (always current)
        latest_failed/         # Optional: last failing Phase-0 (for parked sleeves)
  
  phase_index/                # Canonical references to current runs
    {meta_sleeve}/
      {sleeve_name}/
        phase0.txt           # Points to latest/ directory
        phase1.txt           # Points to Phase-1 run_id
        phase2.txt           # Points to Phase-2 run_id
        status.txt           # Optional: status for parked sleeves
  
  runs/                       # Phase-1/2 backtest runs
    {run_id}/
      ...
```

### Finding Canonical Results

**Quick Answer**: "What's the current Phase-0/1/2 for Trend → breakout_mid_50_100?"

1. Open `reports/phase_index/trend/breakout_mid_50_100/`
2. Read `phase0.txt`, `phase1.txt`, or `phase2.txt`
3. Navigate to the referenced location

**Using the Helper Module**:
```python
from src.utils.phase_index import get_phase_path

# Get canonical Phase-0 path
phase0_path = get_phase_path("trend", "breakout_mid_50_100", "phase0")
# Returns: Path("reports/sanity_checks/trend/breakout_mid_50_100/latest")

# Get canonical Phase-1 path
phase1_path = get_phase_path("trend", "breakout_mid_50_100", "phase1")
# Returns: Path("reports/runs/breakout_1b_7030")
```

For more details, see `docs/REPORTS_STRUCTURE.md`.

## Run Artifacts

When `ExecSim.run()` is called with a `run_id`, it saves artifacts to `reports/runs/{run_id}/`:

- **`portfolio_returns.csv`**: Daily portfolio returns (simple returns)
  - Columns: `date`, `ret`
  
- **`equity_curve.csv`**: Daily equity curve (cumulative, starts at 1.0)
  - Columns: `date`, `equity`
  
- **`asset_returns.csv`**: Daily asset returns (simple returns)
  - Index: Date
  - Columns: Asset symbols
  
- **`weights.csv`**: Portfolio weights at rebalance dates
  - Index: Rebalance date
  - Columns: Asset symbols
  
- **`meta.json`**: Run metadata
  - `run_id`: Run identifier
  - `start_date`: Backtest start date
  - `end_date`: Backtest end date
  - `strategy_config_name`: Strategy configuration name
  - `universe`: List of assets in universe
  - `rebalance`: Rebalance frequency
  - `slippage_bps`: Slippage in basis points
  - `n_rebalances`: Number of rebalances
  - `n_trading_days`: Number of trading days

## Data Format Notes

### Return Types

All returns in the diagnostics module use **simple returns** (not log returns):
- Portfolio returns: `r_simple = exp(r_log) - 1`
- Asset returns: `r_simple = exp(r_log) - 1`

This ensures consistency across all calculations.

### Equity Curve

The equity curve is computed from daily simple returns:
```python
equity_daily = (1 + portfolio_returns_daily).cumprod()
equity_daily.iloc[0] = 1.0  # Start at 1.0
```

### Per-Asset Attribution

Per-asset stats compute approximate contribution by:
1. Forward-filling weights to daily frequency
2. Computing daily PnL: `asset_pnl = weights * asset_returns`
3. Computing annualized stats from daily PnL series

## Ablation Testing Workflow

The diagnostics framework is designed for systematic ablation testing:

1. **Run baseline**: `python run_strategy.py --strategy_profile <baseline> --run_id <baseline_id>`
2. **Run variant**: `python run_strategy.py --strategy_profile <variant> --run_id <variant_id>`
3. **Compare**: `python scripts/run_perf_diagnostics.py --run_id <variant_id> --baseline_id <baseline_id>`

This workflow allows you to:
- Identify which components add/subtract value
- Understand year-by-year impact
- See per-asset attribution changes
- Make data-driven decisions about strategy components

## Integration with ExecSim

The diagnostics framework integrates with `ExecSim`:

```python
from src.agents.exec_sim import ExecSim

exec_sim = ExecSim(...)

# Run backtest with run_id
results = exec_sim.run(
    market=market,
    start=start_date,
    end=end_date,
    components=components,
    run_id="my_run_v1"  # Artifacts saved to reports/runs/my_run_v1/
)

# Later, analyze the run
from src.agents.diagnostics_perf import load_run, compute_core_metrics

run = load_run("my_run_v1")
metrics = compute_core_metrics(run)
print(f"CAGR: {metrics['cagr']:.2%}")
```

## Best Practices

1. **Use descriptive run_ids**: Include strategy profile and version (e.g., `momentum_only_v1`, `core_v1_no_macro`)

2. **Keep baselines**: Don't delete baseline runs - you'll need them for comparison

3. **Check all assets**: The per-asset stats show ALL assets, not just top 10, to ensure nothing is dropped

4. **Year-by-year analysis**: Look for patterns in bad years - they often reveal component issues

5. **Equity ratio**: When comparing runs, the equity ratio shows cumulative impact over time

## Phase-0 Sanity Checks

Phase-0 sanity checks are diagnostic scripts that validate core economic ideas before adding complexity. They follow a strict "sign-only, no overlays" methodology to verify that the underlying economic edge exists.

### Purpose

Phase-0 sanity checks implement minimal strategies to verify:
- Data correctness (continuous prices, returns, features)
- P&L calculation accuracy
- Signal generation logic
- **Core economic edge**: Does the basic idea have positive alpha?

**Pass Criteria**: Sharpe ≥ 0.2+ over long window. Any sleeve that fails Phase-0 remains disabled until reworked.

### Folder Structure

All sanity checks use a **canonical structure** with archive/ and latest/ directories:

```
reports/sanity_checks/
├── trend/                          # Trend Meta-Sleeve
│   ├── breakout_mid_50_100/        # Atomic Sleeve
│   │   ├── archive/                # All historical runs
│   │   │   └── <timestamp>/
│   │   └── latest/                 # Canonical Phase-0 results
│   ├── residual_trend/             # Atomic Sleeve
│   │   ├── archive/
│   │   │   └── <timestamp>/
│   │   └── latest/
│   └── persistence/                # Atomic Sleeve (parked)
│       ├── archive/
│       │   └── <timestamp>/
│       └── latest_failed/          # Last failing run
│
└── ... (other meta-sleeves)
```

**Phase Index**: Canonical references in `reports/phase_index/{meta_sleeve}/{sleeve_name}/phase0.txt`

This structure makes it easy to:
- Find canonical Phase-0 results in `latest/` directories
- Track historical test runs in `archive/` directories
- Quickly answer "what's the current Phase-0 for this sleeve?" via phase_index
- Clean up old archives without affecting canonical references

For more details, see `docs/REPORTS_STRUCTURE.md`.

## TSMOM Sign-Only Sanity Check (Trend Meta-Sleeve)

A diagnostic script for validating data and P&L machinery using a simplified trend-following strategy.

### Purpose

The TSMOM sign-only sanity check (`scripts/run_tsmom_sanity.py`) implements a minimal trend-following strategy to verify:
- Data correctness (continuous prices, returns)
- P&L calculation accuracy
- Signal generation logic
- **Core momentum edge**: Do simple momentum signals have positive alpha?

If the sign-only strategy produces a Sharpe ratio of ~0.3-0.6, the data and P&L pipeline are likely functioning correctly, and the momentum edge is validated.

### Strategy Logic

**Sign-Only TSMOM:**
1. Compute lookback return: `r_lookback = log(price[t] / price[t-lookback])`
2. Take sign: `signal = sign(r_lookback)` → +1, -1, or 0
3. Equal-weight across assets
4. No magnitude scaling (pure sign-based)

**Default Horizons:**
- `long_252`: 252-day lookback
- `med_84`: 84-day lookback
- `short_21`: 21-day lookback

### Usage

```bash
# Run with default horizons (long_252, med_84, short_21)
# Results saved to: reports/sanity_checks/trend/<atomic_sleeve>/archive/<timestamp>/
# Canonical results automatically updated in: reports/sanity_checks/trend/<atomic_sleeve>/latest/
python scripts/run_tsmom_sanity.py --start 2021-01-01 --end 2025-10-31

# Run with custom horizons
python scripts/run_tsmom_sanity.py --start 2021-01-01 --end 2025-10-31 --horizons long_252:252,med_84:84,short_21:21

# Run single horizon (backward compatibility)
python scripts/run_tsmom_sanity.py --start 2021-01-01 --end 2025-10-31 --lookback 252

# Specify custom output directory
python scripts/run_tsmom_sanity.py --start 2021-01-01 --end 2025-10-31 --output_dir reports/sanity_checks/trend/my_run
```

### Output

The script generates results organized by atomic sleeve:

**Meta-Sleeve Level:**
- `reports/sanity_checks/trend/summary_<timestamp>.csv`: Summary metrics comparing all atomic sleeves

**Atomic Sleeve Level** (`reports/sanity_checks/trend/<atomic_sleeve>/latest/` for canonical, `archive/<timestamp>/` for historical):
- `portfolio_returns.csv`: Daily portfolio returns
- `equity_curve.csv`: Cumulative equity curve
- `asset_strategy_returns.csv`: Per-asset strategy returns
- `per_asset_stats.csv`: Per-asset performance statistics
- `meta.json`: Run metadata (dates, lookback, universe, metrics)
- **Plots**:
  - `equity_curve.png`: Equity curve over time
  - `price_series.png`: Price series for all assets
  - `return_histogram.png`: Return distribution

**Atomic Sleeves Tested:**
- `long_term/`: 252-day lookback momentum
- `medium_term/`: 84-day lookback momentum
- `short_term/`: 21-day lookback momentum

### Example Output

```
================================================================================
SUMMARY: Portfolio Sharpe by Horizon
================================================================================
         horizon  lookback      cagr       vol    sharpe     maxdd  hit_rate
0      long_252       252  -0.012345  0.123456  0.260000 -0.234567  0.512345
1       med_84        84  -0.023456  0.145678  0.180000 -0.345678  0.498765
2      short_21        21  -0.034567  0.167890  0.120000 -0.456789  0.485432

Interpretation:
  - Sharpe ~0.3-0.6: Data and P&L pipeline likely correct
  - Sharpe <0.1: Possible data issues or incorrect P&L calculation
  - Sharpe >1.0: Unusually strong (verify data correctness)
```

### Key Features

- **Read-only**: Uses already-adjusted continuous prices from `MarketData.prices_cont`
- **No look-ahead**: Correctly handles date filtering to preserve historical data for lookback calculations
- **Multi-horizon**: Tests multiple lookback periods simultaneously
- **Comprehensive metrics**: CAGR, Sharpe, MaxDD, Hit Rate, Volatility
- **Visualization**: Equity curves, price series, return histograms

### Implementation

**Core Module:** `src/diagnostics/tsmom_sanity.py`

**Key Functions:**
- `run_sign_only_momentum()`: Main strategy computation function
- `compute_sign_only_tsmom()`: Core sign-only TSMOM logic
- `compute_summary_stats()`: Performance metrics calculation
- `save_results()`: Save artifacts to disk
- `generate_plots()`: Generate visualization plots

**CLI Script:** `scripts/run_tsmom_sanity.py`

## Persistence Sign-Only Sanity Check (Trend Meta-Sleeve)

A Phase-0 diagnostic for validating the persistence (momentum-of-momentum) atomic sleeve idea.

### Purpose

The Persistence sign-only sanity check (`scripts/run_persistence_sanity.py`) implements a minimal persistence strategy to verify:
- Return acceleration: `ret_84[t] - ret_84[t-21]` (rate of change of momentum)
- Slope acceleration: `(EMA20 - EMA84)[t] - (EMA20 - EMA84)[t-21]` (trend velocity)
- **Core persistence edge**: Does momentum-of-momentum have positive alpha?

### Strategy Logic

**Sign-Only Persistence (Three Variants):**

**Variant 1: Return Acceleration**
1. Compute 84-day return: `ret_84[t] = log(price[t] / price[t-84])`
2. Compute acceleration: `persistence_raw = ret_84[t] - ret_84[t-21]`
3. Take sign: `signal = sign(persistence_raw)` → +1, -1, or 0
4. Equal-weight across assets
5. No magnitude scaling (pure sign-based)

**Variant 2: Slope Acceleration**
1. Compute EMAs: `EMA20`, `EMA84`
2. Compute slope now: `slope_now = EMA20[t] - EMA84[t]`
3. Compute slope old: `slope_old = EMA20[t-21] - EMA84[t-21]`
4. Acceleration: `persistence_raw = slope_now - slope_old`
5. Take sign: `signal = sign(persistence_raw)` → +1, -1, or 0
6. Equal-weight across assets

**Variant 3: Breakout Acceleration**
1. Compute 126-day breakout: `breakout_126[t] = (price[t] - min_126) / (max_126 - min_126)`, scaled to [-1, +1]
2. Compute acceleration: `persistence_raw = breakout_126[t] - breakout_126[t-21]`
3. Take sign: `signal = sign(persistence_raw)` → +1, -1, or 0
4. Equal-weight across assets

**Default Parameters:**
- `variant`: "return_accel", "slope_accel", or "breakout_accel" (default: "return_accel")
- `lookback`: 84 days (base lookback for return calculation)
- `acceleration_window`: 21 days (window for acceleration calculation)

### Usage

```bash
# Return acceleration variant (default)
python scripts/run_persistence_sanity.py --start 2021-01-01 --end 2025-10-31

# Slope acceleration variant
python scripts/run_persistence_sanity.py --start 2021-01-01 --end 2025-10-31 --variant slope_accel

# Breakout acceleration variant
python scripts/run_persistence_sanity.py --start 2021-01-01 --end 2025-10-31 --variant breakout_accel

# Custom parameters
python scripts/run_persistence_sanity.py \
  --start 2021-01-01 \
  --end 2025-10-31 \
  --variant return_accel \
  --lookback 84 \
  --acceleration_window 21

# Custom output directory
python scripts/run_persistence_sanity.py \
  --start 2021-01-01 \
  --end 2025-10-31 \
  --output_dir reports/sanity_checks/trend/persistence/my_run
```

### Output

**Phase-0 Results** (`reports/sanity_checks/trend/persistence/<timestamp>/`):
- `portfolio_returns.csv`: Daily portfolio returns
- `equity_curve.csv`: Cumulative equity curve
- `asset_returns.csv`: Daily asset returns
- `asset_strategy_returns.csv`: Daily per-asset strategy returns
- `positions.csv`: Daily portfolio positions (signs)
- `persistence_signals.csv`: Raw persistence signals used
- `per_asset_stats.csv`: Per-asset performance statistics
- `summary_metrics.csv`: Portfolio summary metrics
- `meta.json`: Run metadata (dates, variant, parameters, universe, metrics)
- **Plots**:
  - `equity_curve.png`: Equity curve over time
  - `return_histogram.png`: Return distribution (portfolio and per-asset)

### Key Metrics

- Portfolio Sharpe (all assets combined)
- Per-asset Sharpe
- CAGR, Volatility, Max Drawdown, Hit Rate

### Implementation

**Core Module:** `src/diagnostics/persistence_sanity.py`

**Key Functions:**
- `run_persistence()`: Main function to run persistence strategy
- `compute_persistence()`: Core persistence logic (both variants)
- `compute_summary_stats()`: Performance metrics calculation
- `save_results()`: Save artifacts to disk
- `generate_plots()`: Generate visualization plots

**CLI Script:** `scripts/run_persistence_sanity.py`

## Residual Trend Sign-Only Sanity Check (Trend Meta-Sleeve)

A Phase-0 diagnostic for validating the residual trend atomic sleeve idea.

### Purpose

The Residual Trend sign-only sanity check (`scripts/run_residual_trend_sanity.py`) implements a minimal residual trend strategy to verify:
- Long-horizon trend (e.g., 252 days) minus short-term movement (e.g., 21 days)
- Signal generation logic for residual returns
- **Core residual trend edge**: Does subtracting short-term noise from long-term trends improve alpha?

### Strategy Logic

**Sign-Only Residual Trend:**
1. Compute long-horizon log return: `long_ret = log(price[t] / price[t-L_long])`
2. Compute short-horizon log return: `short_ret = log(price[t] / price[t-L_short])`
3. Residual return: `resid_ret = long_ret - short_ret`
4. Take sign: `signal = sign(resid_ret)` → +1, -1, or 0
5. Equal-weight across assets
6. No magnitude scaling (pure sign-based)

**Default Parameters:**
- `long_lookback`: 252 days
- `short_lookback`: 21 days

### Usage

```bash
# Run with default parameters
# Results saved to: reports/sanity_checks/trend/residual_trend/<timestamp>/
python scripts/run_residual_trend_sanity.py --start 2021-01-01 --end 2025-10-31

# Custom lookback periods
python scripts/run_residual_trend_sanity.py --start 2021-01-01 --end 2025-10-31 --long_lookback 252 --short_lookback 21

# Custom output directory
python scripts/run_residual_trend_sanity.py --start 2021-01-01 --end 2025-10-31 --output_dir reports/sanity_checks/trend/residual_trend/my_run
```

### Output

**Atomic Sleeve Level** (`reports/sanity_checks/trend/residual_trend/<timestamp>/`):
- `portfolio_returns.csv`: Daily portfolio returns
- `equity_curve.csv`: Cumulative equity curve
- `asset_strategy_returns.csv`: Per-asset strategy returns
- `per_asset_stats.csv`: Per-asset performance statistics
- `meta.json`: Run metadata (dates, lookbacks, universe, metrics)
- **Plots**:
  - `equity_curve.png`: Equity curve over time
  - `return_histogram.png`: Return distribution

### Implementation

**Core Module:** `src/diagnostics/residual_trend_sanity.py`

**Key Functions:**
- `run_residual_trend()`: Main strategy computation function
- `compute_residual_trend()`: Core residual trend logic
- `compute_summary_stats()`: Performance metrics calculation
- `save_results()`: Save artifacts to disk
- `generate_plots()`: Generate visualization plots

**CLI Script:** `scripts/run_residual_trend_sanity.py`

**Research Documentation:** See `docs/META_SLEEVES/TREND_RESEARCH.md` for detailed design and hypotheses.

## Rates Curve Sign-Only Sanity Check (Rates Curve RV Meta-Sleeve)

A diagnostic script for validating rates curve trading using a simplified flattener/steepener strategy.

### Purpose

The Rates Curve sign-only sanity check (`scripts/run_rates_curve_sanity.py`) implements a minimal curve trading strategy to verify:
- FRED-anchored yield reconstruction accuracy
- Curve feature computation (2s10s, 5s30s slopes)
- DV01-neutral weighting
- **Core curve edge**: Do simple curve signals have positive alpha?

### Strategy Logic

**Sign-Only Curve Trading:**
1. Get curve features: `curve_2s10s_z`, `curve_5s30s_z`
2. Take sign: `signal = sign(curve_feature)` → +1 (flattener), -1 (steepener), or 0
3. Map to trades:
   - Positive signal → flattener: long front, short back
   - Negative signal → steepener: short front, long back
4. DV01-neutral or equal-notional weighting
5. Combine 2s10s and 5s30s legs 50/50 for portfolio

### Usage

```bash
# Run with DV01-neutral weighting (default)
# Results saved to: reports/sanity_checks/rates_curve_rv/<atomic_sleeve>/<timestamp>/
python scripts/run_rates_curve_sanity.py --start 2021-01-01 --end 2025-10-31

# Run with equal notional weighting
python scripts/run_rates_curve_sanity.py --start 2021-01-01 --end 2025-10-31 --equal_notional

# Custom output directory
python scripts/run_rates_curve_sanity.py --start 2021-01-01 --end 2025-10-31 --output_dir reports/sanity_checks/rates_curve_rv/my_run
```

### Output

**Meta-Sleeve Level:**
- `reports/sanity_checks/rates_curve_rv/portfolio/<timestamp>/`: Combined portfolio results (2s10s + 5s30s)

**Atomic Sleeve Level:**
- `reports/sanity_checks/rates_curve_rv/2s10s/<timestamp>/`: 2s10s leg results
- `reports/sanity_checks/rates_curve_rv/5s30s/<timestamp>/`: 5s30s leg results

Each atomic sleeve folder contains:
- `portfolio_returns.csv`: Daily returns for the leg
- `equity_curve.csv`: Cumulative equity curve
- `meta.json`: Run metadata
- **Plots**: Equity curves, return histograms, subperiod comparison

**Key Metrics:**
- Per-leg Sharpe (2s10s, 5s30s)
- Portfolio Sharpe (combined)
- Subperiod analysis (pre-2022 vs post-2022)

**CLI Script:** `scripts/run_rates_curve_sanity.py`

## FX/Commodity Carry Sign-Only Sanity Check (Carry Meta-Sleeve)

A diagnostic script for validating roll yield carry using a simplified sign-only strategy.

### Purpose

The FX/Commodity Carry sign-only sanity check (`scripts/run_carry_sanity.py`) implements a minimal carry strategy to verify:
- Roll yield calculation accuracy
- Term structure feature computation
- **Core carry edge**: Do simple roll yield signals have positive alpha?

### Strategy Logic

**Sign-Only Carry:**
1. Compute raw roll yield: `roll_yield = -(ln(F1) - ln(F0))`
   - F0 = front contract
   - F1 = next contract
2. Take sign: `signal = sign(roll_yield)` → +1 (backwardation, long), -1 (contango, short)
3. Equal-weight across assets
4. No cross-sectional ranking, no vol targeting

### Usage

```bash
# Run with default universe (CL, GC, 6E, 6B, 6J)
# Results saved to: reports/sanity_checks/carry/fx_commodity/<timestamp>/
python scripts/run_carry_sanity.py --start 2020-01-01 --end 2025-10-31

# Custom universe
python scripts/run_carry_sanity.py --start 2020-01-01 --end 2025-10-31 --universe CL,GC,6E

# Custom output directory
python scripts/run_carry_sanity.py --start 2020-01-01 --end 2025-10-31 --output_dir reports/sanity_checks/carry/fx_commodity/my_run
```

### Output

**Atomic Sleeve Level:**
- `reports/sanity_checks/carry/fx_commodity/<timestamp>/`: FX/Commodity carry results

Contains:
- `portfolio_returns.csv`: Daily portfolio returns
- `equity_curve.csv`: Cumulative equity curve
- `asset_strategy_returns.csv`: Per-asset strategy returns
- `roll_yields.csv`: Raw roll yields by asset
- `meta.json`: Run metadata
- **Plots**: Equity curves, return histograms, subperiod comparison

**Key Metrics:**
- Portfolio Sharpe (all assets combined)
- Per-asset Sharpe
- Subperiod analysis (pre-2022 vs post-2022)

**CLI Script:** `scripts/run_carry_sanity.py`

## CSMOM Sign-Only Sanity Check (Cross-Sectional Momentum Meta-Sleeve)

A diagnostic script for validating cross-sectional momentum using a simplified ranking strategy.

### Purpose

The CSMOM sign-only sanity check (`scripts/run_csmom_sanity.py`) implements a minimal cross-sectional momentum strategy to verify:
- Cross-sectional ranking logic
- Equal-notional long/short construction
- Market-neutral portfolio mechanics
- **Core cross-sectional momentum edge**: Do simple ranking signals have positive alpha?

### Strategy Logic

**Sign-Only Cross-Sectional Momentum:**
1. For each rebalance date, compute k-day return across universe
2. Rank assets by return (ascending: lowest = rank 1, highest = rank N)
3. Long top fraction (e.g., top 33%), short bottom fraction (e.g., bottom 33%)
4. Equal notional within long and within short
5. Net exposure ≈ 0 by construction
6. Daily rebalancing, no vol targeting or overlays

### Usage

```bash
# Run with default parameters (lookback=126, top_frac=0.33, bottom_frac=0.33)
# Results saved to: reports/sanity_checks/csmom/phase0/<timestamp>/
python scripts/run_csmom_sanity.py --start 2020-01-01 --end 2025-10-31

# Custom lookback period
python scripts/run_csmom_sanity.py --start 2020-01-01 --end 2025-10-31 --lookback 126

# Custom fractions
python scripts/run_csmom_sanity.py --start 2020-01-01 --end 2025-10-31 --top_frac 0.30 --bottom_frac 0.30

# Specific universe
python scripts/run_csmom_sanity.py --start 2020-01-01 --end 2025-10-31 --universe ES,NQ,RTY,CL,GC

# Custom output directory
python scripts/run_csmom_sanity.py --start 2020-01-01 --end 2025-10-31 --output_dir reports/sanity_checks/csmom/phase0/my_run
```

### Output

**Phase-0 Results** (`reports/sanity_checks/csmom/phase0/<timestamp>/`):
- `portfolio_returns.csv`: Daily portfolio returns
- `equity_curve.csv`: Cumulative equity curve
- `asset_returns.csv`: Daily asset returns
- `weights.csv`: Daily portfolio weights (long/short positions)
- `per_asset_stats.csv`: Per-asset performance statistics
- `summary_metrics.csv`: Portfolio summary metrics
- `meta.json`: Run metadata (dates, lookback, fractions, universe, metrics)
- **Plots**:
  - `equity_curve.png`: Equity curve over time
  - `return_histogram.png`: Return distribution (portfolio and per-asset)

### Key Metrics

- Portfolio Sharpe (all assets combined)
- Per-asset Sharpe
- Net exposure (should be ≈ 0)

### Implementation

**Core Module:** `src/diagnostics/csmom_sanity.py`

**Key Functions:**
- `compute_sign_only_csmom()`: Core sign-only CSMOM logic
- `compute_summary_stats()`: Performance metrics calculation
- `save_results()`: Save artifacts to disk
- `generate_plots()`: Generate visualization plots

**CLI Script:** `scripts/run_csmom_sanity.py`

## Phase-0 Results Summary

### CSMOM Phase-0 (Sign-Only Cross-Sectional Momentum)

**Script:** `scripts/run_csmom_sanity.py`

**Test window:** 2020-01-02 to 2025-10-31

**Result:** Sharpe 0.236, MaxDD −30.26%, CAGR 2.42%

**Verdict:** PASS (Sharpe ≥ 0.2). Eligible for Phase-1.

## Phase-0 Workflow

The Phase-0 sanity check process follows this workflow:

1. **Run Phase-0 Test**: Execute the appropriate sanity check script
   ```bash
   python scripts/run_<meta_sleeve>_sanity.py --start <start> --end <end>
   ```

2. **Review Results**: Check Sharpe ratios and subperiod analysis
   - **Pass**: Sharpe ≥ 0.2+ → Proceed to Phase-1 (clean implementation)
   - **Fail**: Sharpe < 0.2 → Sleeve remains disabled, redesign required

3. **Compare Atomic Sleeves**: Within a Meta-Sleeve, compare atomic sleeve results
   - Identify which atomic sleeves contribute positively
   - Determine optimal weighting for meta-signal combination

4. **Document Findings**: Update `STRATEGY.md` with Phase-0 results
   - Pass/fail status
   - Key metrics (Sharpe, CAGR, MaxDD)
   - Subperiod analysis insights
   - Redesign plans if failed

## Multi-Sleeve Diagnostics — Trend vs Trend + CSMOM

**Baseline:** `core_v3_baseline_2020_2025` (Trend-only)  
**Variant:** `core_v4_trend_csmom_2020_2025` (Trend + CSMOM, 75/25)

- Sharpe: 0.0294 → 0.0902
- CAGR: -0.38% → +0.35%
- MaxDD: -29.85% → -28.09%
- Vol: 12.19% → 12.29% (unchanged)
- Final equity ratio (variant / baseline): 1.0549

**Verdict:** Phase-2 CSMOM integration PASSED.  
CSMOM is approved for use in the production-style core configuration and will participate in future multi-sleeve tests as additional Meta-Sleeves are added.

- **Baseline run_id:** `core_v3_baseline_2020_2025`
- **Variant run_id:** `core_v4_trend_csmom_2020_2025`
- **Promotion Note:** As of Nov 2025, `core_v4_trend_csmom_no_macro` is the canonical multi-sleeve baseline configuration for future Phase-2 tests. `core_v3_no_macro` remains the Trend-only reference profile.

## Multi-Sleeve Diagnostics — Trend MultiHorizon with vs without Residual

**Purpose**: Compare Trend Meta-Sleeve performance with and without Residual Trend as an internal atomic sleeve.

### Diagnostic Template

**Baseline:** `core_v3_no_macro` (Trend MultiHorizon only, 3 atomic sleeves)  
**Variant (External):** `core_v3_trend_plus_residual_experiment` (80% Trend + 20% Residual, external blending)  
**Variant (Internal):** `core_v3_trend_v2_with_residual` (Trend MultiHorizon with Residual as 4th atomic sleeve, internal blending)

### Comparison Workflow

1. **Run Baseline**:
   ```bash
   python run_strategy.py \
     --strategy_profile core_v3_no_macro \
     --start 2021-01-01 \
     --end 2025-10-31 \
     --run_id trend_v1_baseline
   ```

2. **Run External Blending Variant**:
   ```bash
   python scripts/run_residual_trend_phase1.py \
     --run_id trend_plus_residual_external \
     --start 2021-01-01 \
     --end 2025-10-31
   ```

3. **Run Internal Blending Variant** (after implementation):
   ```bash
   python run_strategy.py \
     --strategy_profile core_v3_trend_v2_with_residual \
     --start 2021-01-01 \
     --end 2025-10-31 \
     --run_id trend_v2_with_residual_internal
   ```

4. **Compare Results**:
   ```bash
   # External vs Baseline
   python scripts/run_perf_diagnostics.py \
     --run_id trend_plus_residual_external \
     --baseline_id trend_v1_baseline
   
   # Internal vs Baseline
   python scripts/run_perf_diagnostics.py \
     --run_id trend_v2_with_residual_internal \
     --baseline_id trend_v1_baseline
   
   # Internal vs External
   python scripts/run_perf_diagnostics.py \
     --run_id trend_v2_with_residual_internal \
     --baseline_id trend_plus_residual_external
   ```

### Expected Metrics to Compare

- **CAGR**: Should improve with residual trend
- **Sharpe Ratio**: Should improve (may turn positive if baseline is negative)
- **Max Drawdown**: Should reduce
- **Volatility**: Should remain similar or slightly increase
- **Hit Rate**: Should improve
- **Equity Ratio**: Should show consistent outperformance over time

### Key Questions

1. **Does internal blending behave similarly to external blending?**
   - Internal: Residual Trend as 4th atomic sleeve with static weights
   - External: Residual Trend as separate sleeve blended at CombinedStrategy level
   - Both should show similar improvements if architecture is sound

2. **Are improvements consistent across years?**
   - Check year-by-year breakdown
   - Residual trend should help especially in choppy markets (2024-2025)

3. **Is the signal economically distinct?**
   - Residual trend should provide diversification vs pure momentum
   - Correlation analysis between residual and other atomic sleeves

### Phase-1 Results (External Blending)

**Baseline:** `core_v3_no_macro`  
**Variant:** `core_v3_trend_plus_residual_experiment` (80% Trend + 20% Residual)

- **CAGR**: -2.59% → -0.49% (+2.10% improvement)
- **Sharpe**: -0.15 → 0.03 (+0.18 improvement) ✨
- **MaxDD**: -39.69% → -36.80% (+2.89% improvement)
- **Vol**: 12.31% → 12.93% (+0.62%)
- **HitRate**: 49.77% → 50.83% (+1.06% improvement)
- **Equity Ratio**: 1.136x (13.6% better over full period)

**Verdict**: ✅ **Phase-1 PASSED** — Residual Trend validated. Ready for internal integration.

## Multi-Sleeve Diagnostics — Trend MultiHorizon with vs without Breakout Mid

**Purpose**: Compare Trend Meta-Sleeve performance with and without Breakout Mid (50-100d) as an internal atomic sleeve.

### Phase-1B Results (Refinement Cycle)

**Baseline:** `core_v3_no_macro_phase1b_baseline`  
**Test Configurations:** 4 variants tested with 3% horizon weight

**Winner: 70/30 Configuration (breakout_50: 70%, breakout_100: 30%)**

- **CAGR**: 0.29% → 0.42% (+0.12% improvement)
- **Sharpe**: 0.0857 → 0.0953 (+0.0095 improvement) ✨
- **MaxDD**: -32.02% → -31.52% (+0.50% improvement)
- **Vol**: 12.44% → 12.35% (-0.09% improvement)
- **HitRate**: 52.11% → 52.37% (+0.26% improvement)
- **Equity Ratio**: 1.0073 (0.73% better over full period)

**Other Test Results:**
- **30/70 (50d/100d)**: ❌ FAILED (Sharpe: -0.1502, MaxDD: -39.00%)
- **Pure 50d (100/0)**: ❌ FAILED (Sharpe: -0.1396, MaxDD: -38.60%)
- **Pure 100d (0/100)**: ❌ FAILED (Sharpe: -0.1547, MaxDD: -39.15%)

**Key Findings:**
- 3% horizon weight optimal (vs 10% in Phase-1 initial test)
- 50d breakout more effective than 100d in this setup
- 70/30 blend outperforms pure configurations
- Lower weight reduces signal conflicts with existing sleeves

**Verdict**: ✅ **Phase-1B PASSED** — 70/30 configuration promoted to Phase-2 validation.

### Phase-2 Validation

**Profile**: `core_v3_trend_breakout_70_30_phase2`  
**Baseline**: `core_v3_no_macro_phase1b_baseline`  
**Configuration**: 3% horizon weight, 70% breakout_50, 30% breakout_100  
**Target Window**: 2021-01-01 to 2025-11-19

**Results Summary**:

| Metric | Baseline | Phase-2 | Delta | Status |
|--------|----------|---------|-------|--------|
| **Sharpe** | 0.0857 | **0.0953** | **+0.0095** | ✅ Better |
| **MaxDD** | -32.02% | **-31.52%** | **+0.50%** | ✅ Better |
| **CAGR** | 0.29% | **0.42%** | **+0.12%** | ✅ Better |
| **Vol** | 12.44% | **12.35%** | **-0.09%** | ✅ Better |
| **HitRate** | 52.11% | **52.37%** | **+0.26%** | ✅ Better |

**Equity Curve Comparison**:
- Final equity ratio (Phase-2 / Baseline): 1.0073 (0.73% outperformance)
- Mean equity ratio: 1.0027
- Consistent outperformance across full period

**Year-by-Year Performance**:
- **2021**: Outperformed baseline
- **2022**: Outperformed baseline
- **2023**: Outperformed baseline
- **2024**: Outperformed baseline
- **2025**: Outperformed baseline

**Promotion Criteria**:
- ✅ **Sharpe ≥ baseline**: 0.0953 ≥ 0.0857 (passes)
- ✅ **MaxDD ≤ baseline**: -31.52% ≤ -32.02% (passes)
- ✅ **Robust across years**: Outperforms in all 5 years
- ✅ **Stable correlations**: Configuration reproducible and stable

**Verdict**: ✅ **Phase-2 PASSED** — Breakout Mid (50-100d) approved for Phase-3 (production monitoring).

---

## Future Enhancements

Potential additions to the diagnostics framework:
- Rolling window metrics (rolling Sharpe, rolling MaxDD)
- Drawdown analysis (drawdown length, recovery time)
- Turnover and cost attribution
- Sleeve-level attribution (if sleeve P&L is tracked)
- Statistical significance testing for metric differences

