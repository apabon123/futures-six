# futures-six: Time-Series Momentum Strategy Framework

A complete, production-ready systematic trading framework for futures momentum strategies with comprehensive backtesting capabilities.

## Overview

**futures-six** is a modular, agent-based framework for building and backtesting systematic trading strategies on continuous futures contracts. The framework implements a **two-layer sleeve architecture** with Meta-Sleeves (economic sources of return) and Atomic Sleeves (implementation variants). Currently, the **Trend Meta-Sleeve** is active in production, combining five atomic sleeves (long-term, medium-term, short-term momentum, residual trend, and breakout) into a unified meta-signal.

### What This Project Does

- 📊 **Reads market data** from a local DuckDB database with strict read-only access
- 📈 **Generates momentum signals** using 12-1 month lookback with customizable parameters
- 🎯 **Targets volatility** to achieve consistent risk exposure (default: 20% annual vol)
- ⚖️ **Optimizes portfolio weights** using signal-beta or equal-risk-contribution methods
- 💰 **Simulates execution** with realistic slippage and transaction costs
- 📉 **Produces comprehensive metrics**: CAGR, Sharpe ratio, max drawdown, turnover, etc.

### Quick Start Results

**Current Production Baseline (core_v3_no_macro - Trend Meta-Sleeve with 5 atomic sleeves):**
- **0.42% CAGR** (2021-2025)
- **0.095 Sharpe Ratio**
- **-31.52% Max Drawdown**
- **12.35% Volatility**
- **52.37% Hit Rate**

**Note**: Performance reflects the challenging 2022-2025 period for trend-following strategies. The Trend Meta-Sleeve includes 5 atomic sleeves (long, medium, short, residual trend, breakout) and is production-ready. Performance will improve as additional meta-sleeves are added.

## Universe

The strategy now trades **thirteen** continuous futures contracts across equities, rates, commodities, and FX:

**Equities**
- **ES_FRONT_CALENDAR_2D** – E-mini S&P 500 (calendar roll, 2 days before expiry)
- **NQ_FRONT_CALENDAR_2D** – E-mini NASDAQ-100 (calendar roll)
- **RTY_FRONT_CALENDAR_2D** – E-mini Russell 2000 (calendar roll)

**Rates**
- **ZN_FRONT_VOLUME** – 10-Year Treasury Note (volume-weighted roll)
- **ZF_FRONT_VOLUME** – 5-Year Treasury Note (volume-weighted roll)
- **ZT_FRONT_VOLUME** – 2-Year Treasury Note (volume-weighted roll)
- **UB_FRONT_VOLUME** – Ultra U.S. Treasury Bond (volume-weighted roll)
- **SR3_FRONT_CALENDAR_2D** – SOFR (Secured Overnight Financing Rate) futures (calendar roll, T-2 offset)

**Commodities**
- **CL_FRONT_VOLUME** – Crude Oil WTI (volume-weighted roll)
- **GC_FRONT_VOLUME** – Gold (volume-weighted roll)

**FX**
- **6E_FRONT_CALENDAR_2D** – Euro FX (calendar roll)
- **6B_FRONT_CALENDAR_2D** – British Pound (calendar roll)
- **6J_FRONT_CALENDAR_2D** – Japanese Yen (calendar roll)

## Key Features

- ✅ **Read-Only Access**: All database connections enforce `read_only=True`
- ✅ **Schema Discovery**: Auto-detects OHLCV tables by required columns
- ✅ **Dual-Price Architecture**: Raw prices (for sizing) and back-adjusted continuous prices (for signals/P&L) - see `docs/DUAL_PRICE_ARCHITECTURE.md`
- ✅ **Back-Adjustment**: Continuous prices built in-memory using backward-panama adjustment (roll jumps removed)
- ✅ **Point-in-Time Snapshots**: `snapshot(asof)` for historical queries
- ✅ **Standardized APIs**: Prices, returns, volatility, covariance
- ✅ **Comprehensive Logging**: All queries logged with `[READ-ONLY]` prefix
- ✅ **Minimal Dependencies**: DuckDB, pandas, numpy, PyYAML

## Project Structure

```
futures-six/
├── configs/
│   ├── data.yaml              # Database connection and universe config (dict format with roll settings)
│   ├── strategies.yaml        # Strategy parameters (TSMOM, vol overlay, allocator)
│   └── fred_series.yaml       # FRED economic indicators configuration
├── scripts/
│   ├── download/
│   │   └── download_fred_series.py  # FRED data downloader script
│   ├── run_tsmom_sanity.py          # TSMOM Phase-0 sanity check
│   ├── run_rates_curve_sanity.py   # Rates Curve Phase-0 sanity check
│   └── run_carry_sanity.py         # FX/Commodity Carry Phase-0 sanity check
├── src/
│   ├── agents/
│   │   ├── data_broker.py     # MarketData: Read-only OHLCV data access
│   │   ├── feature_service.py # FeatureService: Centralized feature computation
│   │   ├── feature_long_momentum.py # Momentum Features: Long/Medium/Short-term momentum features
│   │   ├── feature_sr3_curve.py # SR3 Curve Features: Carry & curve features
│   │   ├── feature_rates_curve.py # Rates Curve Features: FRED-anchored yield curves
│   │   ├── feature_carry_fx_commod.py # FX/Commodity Carry Features: Roll yield features
│   │   ├── strat_momentum.py  # TSMOM: Long-term momentum (multi-feature)
│   │   ├── strat_tsmom_multihorizon.py  # Trend Meta-Sleeve: Unified multi-horizon momentum (long/medium/short)
│   │   ├── strat_momentum_medium.py  # Medium-Term Momentum: Multi-feature medium-horizon momentum
│   │   ├── strat_momentum_short.py  # Short-Term Momentum: Multi-feature short-horizon momentum
│   │   ├── strat_sr3_carry_curve.py  # SR3 Carry/Curve: SOFR carry strategy (parked)
│   │   ├── strat_rates_curve.py  # Rates Curve: Treasury curve trading (parked)
│   │   ├── strat_carry_fx_commod.py  # FX/Commodity Carry: Roll yield strategy (parked)
│   │   ├── strat_combined.py  # CombinedStrategy: Meta-sleeve signal combination
│   │   ├── strat_cross_sectional.py  # Cross-Sectional Momentum strategy
│   │   ├── overlay_volmanaged.py  # VolManaged: Volatility targeting
│   │   ├── overlay_macro_regime.py  # MacroRegime: Regime-based signal scaling
│   │   ├── risk_vol.py        # RiskVol: Volatility & covariance calculations
│   │   ├── allocator.py       # Allocator: Portfolio weight optimization
│   │   ├── exec_sim.py        # ExecSim: Backtest execution engine
│   │   ├── param_sweep.py     # ParamSweepRunner: Parameter optimization
│   │   ├── diagnostics.py     # Diagnostics: Performance metrics & attribution
│   │   └── feature_store.py   # FeatureStore: Optional caching layer
│   └── diagnostics/
│       ├── tsmom_sanity.py    # TSMOM Phase-0 sanity check
│       ├── rates_curve_sanity.py  # Rates Curve Phase-0 sanity check
│       └── carry_sanity.py    # FX/Commodity Carry Phase-0 sanity check
├── tests/
│   ├── test_marketdata.py     # MarketData tests
│   ├── test_strat_momentum.py # TSMOM tests (24 tests)
│   ├── test_strat_cross_sectional.py # Cross-Sectional Momentum tests (22 tests)
│   ├── test_overlay_volmanaged.py # Vol overlay tests
│   ├── test_overlay_macro_regime.py # Macro regime overlay tests (14 tests)
│   ├── test_risk_vol.py       # Risk calculation tests
│   ├── test_allocator.py      # Portfolio optimization tests
│   ├── test_exec_sim.py       # Backtest engine tests
│   ├── test_param_sweep.py    # Parameter sweep tests (17 tests)
│   └── test_diagnostics.py    # Diagnostics & attribution tests
├── docs/                      # Centralized documentation (see docs/README.md)
├── run_strategy.py            # Main entry point - run the complete backtest
├── verify_setup.py            # Quick verification script
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

### Documentation

All component-specific guides live in `docs/`. 

**Essential reading:**
- **`docs/SOTs/STRATEGY.md`** ⭐ – Complete step-by-step strategy execution flow, two-layer sleeve architecture (Meta-Sleeves vs Atomic Sleeves), and current baseline (core_v3_no_macro - Trend Meta-Sleeve with 5 atomic sleeves) (read this first!)
- **`docs/SOTs/DIAGNOSTICS.md`** ⭐ – Performance diagnostics framework and Phase-0 sanity check process
- **`docs/SOTs/PROCEDURES.md`** ⭐ – Step-by-step procedures for adding/changing sleeves, assets, and parameters (when and how to run Phase-0 → Phase-3)
- **`docs/SOTs/ROADMAP.md`** ⭐ – Strategic development roadmap (2026–2028). Long-term sequencing, meta-sleeve expansion plans, production deployment planning, and Sharpe targets

**Component docs:**
- `docs/README.md` – Documentation index
- `docs/SOTs/STRATEGY.md` – Two-layer sleeve architecture, Sleeve Development Lifecycle (Phase-0 through Phase-3), current baseline
- `docs/META_SLEEVES/TREND_IMPLEMENTATION.md` – Trend Meta-Sleeve implementation (current production)
- `docs/META_SLEEVES/TREND_RESEARCH.md` – Trend Meta-Sleeve research notebook (structured research document with Phase-0/1/2 results for all tested sleeves)
- `docs/REPORTS_STRUCTURE.md` – Reports directory structure and phase indexing system (canonical Phase-0/1/2 results)
- `docs/legacy/TSMOM_IMPLEMENTATION.md` – Legacy TSMOM class (not used in production)
- `docs/SOTs/DIAGNOSTICS.md` – Performance diagnostics and Phase-0 sanity check workflow
- `docs/SOTs/PROCEDURES.md` – Procedures and checklists for adding/changing sleeves, assets, parameters, and overlays
- `src/agents/feature_long_momentum.py` – Long/Medium/Short momentum feature computation
- `src/agents/strat_tsmom_multihorizon.py` – Trend Meta-Sleeve implementation (combines 3 atomic sleeves)
- `src/agents/strat_momentum_medium.py` – Medium-term momentum atomic sleeve
- `src/agents/strat_momentum_short.py` – Short-term momentum atomic sleeve
- `docs/SR3_CARRY_CURVE.md` – SR3 carry and curve features (parked - Phase-0 failed)
- Rates Curve: See `src/agents/feature_rates_curve.py` and `src/agents/strat_rates_curve.py` (parked - Phase-0 failed)
- FX/Commodity Carry: See `src/agents/feature_carry_fx_commod.py` and `src/agents/strat_carry_fx_commod.py` (parked - Phase-0 failed)
- `docs/MACRO_REGIME_FILTER.md` – Regime filter with FRED indicators
- `docs/PARAM_SWEEP.md` – Parameter optimization
- `docs/DUAL_PRICE_ARCHITECTURE.md` – Dual-price architecture (raw vs continuous prices)
- And more...

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Database Path

Edit `configs/data.yaml` to point to your database:

```yaml
db:
  path: "path/to/your/database"
  engine: "auto"  # auto-detect: duckdb or sqlite

universe:
  ES:  { roll: calendar }
  NQ:  { roll: calendar }
  # ... see configs/data.yaml for full universe
```

### 3. (Optional) Download FRED Economic Indicators

If you want to use FRED indicators in the macro regime filter:

```bash
# Set your FRED API key
export FRED_API_KEY="your_api_key_here"

# Download FRED series (configurable in configs/fred_series.yaml)
python scripts/download/download_fred_series.py
```

This downloads all configured FRED indicators, dailyizes monthly series, and saves to parquet format.

### 4. Verify Setup

```bash
python verify_setup.py
```

This will test the data broker connection and verify all components are working.

### 5. Run the Strategy

```bash
python run_strategy.py
```

This runs the complete strategy backtest (Trend Meta-Sleeve only) from 2021 to present and displays performance metrics.

**Strategy Profiles:**
- `--strategy_profile core_v3_no_macro` (default) - Trend Meta-Sleeve with 5 atomic sleeves (current production baseline)
- `--strategy_profile core_v2_no_macro` - Multi-horizon TSMOM (intermediate version, deprecated)
- `--strategy_profile core_v1_no_macro` - Long-term TSMOM + FX/Commodity Carry (legacy baseline, deprecated)

### 6. Run Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific component tests
pytest tests/test_strat_momentum.py -v
pytest tests/test_marketdata.py -v
```

## Strategy Components

The framework uses an agent-based architecture where each component has a specific role:

### 1. MarketData (Data Broker)
- **Role**: Read-only access to OHLCV data and FRED economic indicators from DuckDB/SQLite
- **Features**: Schema auto-discovery, column name mapping, point-in-time snapshots, FRED indicator queries
- **Methods**: `get_price_panel()`, `get_returns()`, `get_vol()`, `get_cov()`, `get_fred_indicator()`, `get_fred_indicators()`

### 2. TSMOM (Long-Term Momentum Strategy)
- **Role**: Generate long-term momentum signals using multi-feature approach
- **Features**: 252-day return momentum, 252-day breakout strength, slow trend slope (EMA_63 - EMA_252)
- **Combination**: Weighted combination of features with configurable weights (default: ret_252=0.5, breakout_252=0.3, slope_slow=0.2)
- **Standardization**: Cross-sectional z-scoring and clipping at ±3.0
- **Rebalance**: Weekly on Fridays
- **Output**: Signals capped at ±3.0 standard deviations

### 2a. Trend Meta-Sleeve (Production Baseline)
- **Role**: Generate unified multi-horizon momentum signals combining three atomic sleeves
- **Architecture**: Two-layer model - Meta-Sleeve (Trend) contains Atomic Sleeves (long-term, medium-term, short-term)
- **Atomic Sleeves**: 
  - Long-term momentum (252d): 252-day return, breakout, slow trend slope
  - Medium-term momentum (84/126d): 84-day return, 126-day breakout, medium trend slope, persistence
  - Short-term momentum (21d): 21-day return, breakout, fast trend slope
  - Residual Trend (252d-21d): Long-horizon trend minus short-term movement
  - Breakout (50-100d): Donchian-style range breakouts (70/30 feature blend, 3% horizon weight)
- **Horizon Weights**: Production weights (long=0.485, med=0.291, short=0.194, breakout_mid=0.03)
- **Signal Processing**: Atomic signals → meta-signal blend → cross-sectional z-score → clip ±3.0
- **EWMA Vol Normalization**: Risk-normalize signals by dividing by EWMA vol (63-day half-life, 5% floor)
- **Rebalance**: Weekly on Fridays
- **Output**: Risk-normalized Trend Meta-Sleeve signals for all symbols
- **Configuration**: `tsmom_multihorizon` block in `configs/strategies.yaml`
- **Status**: ✅ Active (core_v3_no_macro)

### 2b. Medium-Term Momentum Strategy
- **Role**: Generate medium-term momentum signals using multi-feature approach
- **Features**: 84-day return momentum, 126-day breakout strength, medium trend slope (EMA_20 - EMA_84), persistence
- **Combination**: Weighted combination of features with configurable weights (default: ret_84=0.4, breakout_126=0.3, slope_med=0.2, persistence=0.1)
- **Standardization**: Cross-sectional z-scoring and clipping at ±3.0
- **Rebalance**: Weekly on Fridays

### 2c. Short-Term Momentum Strategy
- **Role**: Generate short-term momentum signals using multi-feature approach
- **Features**: 21-day return momentum, 21-day breakout strength, fast trend slope (EMA_10 - EMA_40), reversal filter (RSI-like)
- **Combination**: Weighted combination of features with configurable weights (default: ret_21=0.5, breakout_21=0.3, slope_fast=0.2, reversal_filter=0.0)
- **Standardization**: Cross-sectional z-scoring and clipping at ±3.0
- **Rebalance**: Weekly on Fridays

### 2d. Carry Meta-Sleeve (Parked - Phase-0 Failed)
- **Status**: ❌ Parked for redesign
- **Phase-0 Results**: Sign-only roll yield strategy showed negative Sharpe (-0.69) across all assets (2020-2025)
- **Findings**: All assets (CL, GC, 6E, 6B, 6J) showed negative Sharpe; performance degraded post-2022
- **Roadmap**: Redesign with sector-based roll yield, DV01-neutral carry, regime-dependent filters
- **Atomic Sleeves**: FX/Commodity Carry (CL, GC, 6E, 6B, 6J), SR3 Carry/Curve

### 2e. Rates Curve RV Meta-Sleeve (Parked - Phase-0 Failed)
- **Status**: ❌ Parked for redesign
- **Phase-0 Results**: Sign-only curve trading showed near-zero Sharpe (0.002) with negative 2s10s leg (-0.20 Sharpe)
- **Findings**: Initial slope-based approach fragile to post-2022 regime changes (Fed hiking cycle)
- **Roadmap**: Redesign with DV01-neutral flies, pack spreads, macro gating, improved yield reconstruction
- **Atomic Sleeves**: 2s10s flattener/steepener, 5s30s flattener/steepener

### 2f. Cross-Sectional Momentum (Alternative Strategy)
- **Role**: Generate market-neutral momentum signals by ranking assets
- **Method**: 6-1 month lookback (126 days lookback, skip recent 21 days)
- **Ranking**: Ranks assets by simple returns, long top 33%, short bottom 33%
- **Neutralization**: Signals sum to ~0 (dollar-neutral portfolio)
- **Standardization**: Volatility-scaled or z-score
- **Rebalance**: Weekly on Fridays
- **Output**: Signals capped at ±3.0, neutralized across universe
- **Key Difference**: Relative ranking (market-neutral) vs absolute momentum (can be net long/short)

### 3. VolManagedOverlay (Volatility Targeting)
- **Role**: Scale signals to achieve target portfolio volatility
- **Target Vol**: 20% annualized (configurable)
- **Mode**: Global portfolio vol or per-asset vol
- **Constraints**: Leverage cap (7x), position bounds (±3.0)

### 4. MacroRegimeFilter (Regime-Based Signal Scaling)
- **Role**: Apply continuous scaler k ∈ [k_min, k_max] to signals based on macro regime
- **Inputs**: 
  - Realized volatility: 21-day rolling vol of ES+NQ equal-weighted portfolio
  - Market breadth: Fraction of {ES, NQ} above 200-day SMA
  - **FRED economic indicators**: 10 indicators (8 daily + 2 monthly) from `configs/fred_series.yaml`
- **Logic**: 
  - Higher volatility → lower scaler (risk-off)
  - Lower breadth → lower scaler (defensive)
  - FRED indicators normalized and combined to adjust scaler (positive = risk-on, negative = risk-off)
  - EMA smoothing (default: 0.15) for gradual transitions
- **Rebalance**: Weekly on Fridays (configurable)
- **Bounds**: k ∈ [0.5, 1.0] (default), reduces exposure by up to 50% in adverse regimes
- **No Look-Ahead**: All indicators computed using point-in-time data only
- **FRED Integration**: 
  - 10 indicators (8 daily + 2 monthly) configured in `configs/fred_series.yaml`
  - Monthly series (CPI, UNRATE) dailyized before z-scoring
  - Z-score capping (±5.0) prevents single prints from swinging scaler
  - Input smoothing (5-day EMA) prevents stepwise jumps
  - Data freshness checks (warns if stale > 45 days)

### 5. RiskVol (Risk Calculations)
- **Role**: Calculate rolling volatilities and covariance matrix
- **Vol Lookback**: 63 days (3 months)
- **Cov Lookback**: 252 days (1 year)
- **Shrinkage**: Ledoit-Wolf shrinkage for covariance stability
- **Min Vol Floor**: 50 bps annualized minimum to prevent exploding leverage in calm regimes

### 6. Allocator (Portfolio Optimization)
- **Role**: Convert vol-managed signals to final portfolio weights
- **Method**: Signal-beta (align weights with signal direction)
- **Constraints**: Gross cap (7x), net cap (2x), per-asset bounds (±1.5)
- **Turnover**: Penalizes excessive trading (50% cap, 0.001 lambda)
- **Data Validity**: Invalid signals (from mask) are zeroed before allocation

### 7. ExecSim (Backtest Engine)
- **Role**: Orchestrate all components and simulate execution
- **Costs**: Slippage (0.5 bps), commission (configurable)
- **Rebalance**: Weekly on Fridays (with holiday handling: falls back to previous business day)
- **Holding Period Returns**: Correctly computes cumulative period returns (weights fixed over [t, next_t))
- **Diagnostics**: "What-moved" reports with top weight changes, k values, turnover
- **Metrics**: CAGR, Sharpe, max drawdown, hit rate, turnover, leverage

### 8. Diagnostics (Performance Attribution)
- **Role**: Generate comprehensive performance metrics and reports
- **Metrics**: CAGR, volatility, Sharpe, Calmar, hit rate, avg drawdown length, exposure, turnover, cost drag
- **Outputs**: CSV reports (equity, weights, P&L, turnover/costs)
- **Features**: Time-aligned data, backward compatible with ExecSim output

### 9. ParamSweepRunner (Parameter Optimization)
- **Role**: Systematic exploration of configuration space to find optimal parameters
- **Methods**: Grid search (exhaustive), Latin hypercube (coming soon)
- **Parallelization**: Multi-process execution for fast sweeps
- **Outputs**: Tidy CSV summaries with all metrics, top-N YAML configs
- **Features**: Reproducible with seeds, compare specific configurations
- **See**: `docs/PARAM_SWEEP.md` for full guide and examples

## Usage

### Running the Complete Strategy

```python
python run_strategy.py
```

Output example:
```
================================================================================
FUTURES-SIX: TSMOM Strategy Backtest
================================================================================

Backtest Period: 2021-01-01 to 2025-11-05

[1/6] Initializing MarketData broker...
[2/6] Initializing TSMOM strategy...
[3/6] Initializing RiskVol agent...
[4/6] Initializing VolManaged overlay...
[5/6] Initializing Allocator...
[6/6] Running backtest with ExecSim...

================================================================================
BACKTEST RESULTS
================================================================================

Equity Curve:
  Total Return:   152.61%

Performance Metrics:
  cagr                :   20.82%
  vol                 :   0.2239
  sharpe              :   0.8446
  max_drawdown        :  -0.2231
  hit_rate            :   0.5382
  avg_turnover        :   0.1234
  avg_gross           :   3.45x
  avg_net             :   1.23x

Backtest completed successfully!
```

### Basic MarketData Usage

```python
from src.agents import MarketData

# Initialize broker (connects read-only)
md = MarketData()

# Get closing prices (wide format)
prices = md.get_price_panel(
    symbols=("ES", "NQ"),
    fields=("close",),
    start="2020-01-01",
    end="2023-12-31"
)

# Calculate log returns
returns = md.get_returns(
    symbols=("ES", "NQ"),
    method="log",
    start="2020-01-01"
)

# Calculate 63-day rolling volatility (annualized)
vol = md.get_vol(
    symbols=("ES", "NQ"),
    lookback=63
)

# Get covariance matrix with Ledoit-Wolf shrinkage
cov = md.get_cov(
    symbols=("ES", "NQ", "ZN"),
    lookback=252,
    shrink="lw"
)

# Close connection when done
md.close()
```

### Point-in-Time Snapshots

```python
# Create snapshot at specific date
md_snapshot = md.snapshot("2023-06-30")

# All queries now filtered to date <= 2023-06-30
historical_prices = md_snapshot.get_price_panel(("ES", "NQ"))
historical_returns = md_snapshot.get_returns(("ES", "NQ"))

md_snapshot.close()
```

### Context Manager Pattern

```python
with MarketData() as md:
    prices = md.get_price_panel(("ES", "NQ", "CL"))
    returns = md.get_returns(("ES", "NQ", "CL"))
    # Connection automatically closed
```

### Using Cross-Sectional Momentum

```python
from src.agents.data_broker import MarketData
from src.agents.strat_cross_sectional import CrossSectionalMomentum

# Initialize market data
md = MarketData()
symbols = list(md.universe)

# Initialize cross-sectional momentum strategy
cs_mom = CrossSectionalMomentum(
    symbols=symbols,
    lookback=126,       # 6-month lookback
    skip_recent=21,     # Skip last month
    top_frac=0.33,      # Long top 33%
    bottom_frac=0.33,   # Short bottom 33%
    standardize="vol",  # Volatility-scaled signals
    signal_cap=3.0,     # Cap at ±3
    rebalance="W-FRI"   # Weekly rebalance
)

# Generate signals for a date
signals = cs_mom.signals(md, "2024-01-05")

print(f"Signal sum: {signals.sum():.6f}")  # Should be near 0 (market-neutral)
print(signals)
# ES    -0.2341  (SHORT)
# NQ     0.8721  (LONG)
# ZN     0.0123  (NEUTRAL)
# CL    -1.2456  (SHORT)
# GC     0.5432  (LONG)
# 6E     0.0521  (NEUTRAL)

# Strategy description
print(cs_mom.describe())

md.close()
```

### Using MacroRegimeFilter (Regime-Based Scaling with FRED Indicators)

```python
from src.agents.data_broker import MarketData
from src.agents.overlay_macro_regime import MacroRegimeFilter

# Initialize market data
md = MarketData()

# Initialize macro regime filter with FRED indicators
macro_filter = MacroRegimeFilter(
    rebalance="W-FRI",                      # Weekly rebalance
    vol_thresholds={'low': 0.12, 'high': 0.22},  # Vol regime thresholds
    k_bounds={'min': 0.5, 'max': 1.0},      # Scaler bounds
    smoothing=0.15,                         # EMA smoothing (15% new, 85% old)
    vol_lookback=21,                        # 1-month realized vol
    breadth_lookback=200,                   # 200-day SMA for breadth
    proxy_symbols=("ES_FRONT_CALENDAR_2D", "NQ_FRONT_CALENDAR_2D"),  # Symbols for regime detection
    fred_series=("VIXCLS", "VXVCLS", "FEDFUNDS", "DGS2", "DGS10", "BAMLH0A0HYM2", "TEDRATE", "CPIAUCSL", "UNRATE", "DTWEXBGS"),  # FRED indicators (10 total)
    fred_lookback=252,                      # 1-year rolling window for FRED normalization
    fred_weight=0.3                         # Weight of FRED signal in scaler (30%)
)

# Sample strategy signals
signals = pd.Series({'ES': 1.5, 'NQ': -0.8, 'GC': 0.5})

# Get regime scaler for a date (includes FRED signal)
date = "2024-01-05"
k = macro_filter.scaler(md, date)
print(f"Regime scaler: {k:.3f}")  # e.g., 0.750 (25% risk reduction)
# Logs show: vol, breadth, fred_signal, base_k, smoothed_k

# Apply scaler to signals
scaled_signals = macro_filter.apply(signals, md, date)
print(f"Original gross leverage: {signals.abs().sum():.2f}")
print(f"Scaled gross leverage: {scaled_signals.abs().sum():.2f}")

# Filter description
print(macro_filter.describe())

md.close()
```

### Accessing FRED Economic Indicators

```python
from src.agents import MarketData

# Initialize market data
md = MarketData()

# Get single FRED indicator
vix = md.get_fred_indicator("VIXCLS", start="2020-01-01", end="2024-12-31")
print(f"VIX data: {len(vix)} observations")
print(vix.tail())

# Get multiple FRED indicators
fred_data = md.get_fred_indicators(
    series_ids=("VIXCLS", "DGS10", "UNRATE", "FEDFUNDS"),
    start="2020-01-01"
)
print(f"FRED indicators DataFrame:")
print(fred_data.tail())

md.close()
```

**FRED Data Management:**
- Configuration: `configs/fred_series.yaml` (10 indicators: 8 daily + 2 monthly)
- Download script: `scripts/download/download_fred_series.py`
- Monthly series (CPI, UNRATE) are automatically dailyized (forward-filled to business days) before z-scoring
- Data freshness checks: warns if monthly series stale > 45 days

### Parameter Sweeps and Optimization

Run systematic parameter sweeps to find optimal configurations:

```python
from src.agents.param_sweep import run_sweep, compare_configs

# Define base configuration
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

# Define parameter grid (dot-notation for nested params)
grid = {
    "macro_regime.vol_thresholds.low": [0.10, 0.12, 0.14],
    "macro_regime.vol_thresholds.high": [0.20, 0.22, 0.25],
    "tsmom.lookbacks": [[252], [126, 252], [63, 126, 252]],
    "vol_overlay.target_vol": [0.15, 0.20, 0.25],
    "exec.rebalance": ["W-FRI", "M"]
}

# Run sweep (parallelized)
results = run_sweep(
    base_config=base_config,
    grid=grid,
    seeds=[0],  # Add more seeds for robustness
    start="2021-01-01",
    end="2025-11-05",
    n_workers=None,  # Auto-detect CPU count
    save_top_n=10
)

# Analyze results
print(results[results['success']].nlargest(5, 'sharpe'))

# Or compare specific configurations
configs = {
    "Baseline": baseline_config,
    "Macro": macro_config,
    "Macro+XSec": macro_xsec_config
}
comparison = compare_configs(configs)
```

**Example command-line usage:**

```bash
# Quick test sweep
python examples/run_param_sweep.py --mode test

# Compare specific configurations
python examples/run_param_sweep.py --mode compare

# Full grid sweep
python examples/run_param_sweep.py --mode grid
```

**Output:**
- `reports/sweeps/<timestamp>/summary.csv` - All results with metrics
- `reports/sweeps/<timestamp>/top_*.yaml` - Top N configurations

See `docs/PARAM_SWEEP.md` for comprehensive guide and examples.

### Phase-0 Sanity Checks

Before any new sleeve enters production, it must pass Phase-0 sanity check (sign-only, no overlays):

```bash
# TSMOM sanity check (Trend Meta-Sleeve)
python scripts/run_tsmom_sanity.py --start 2021-01-01 --end 2025-10-31

# Rates Curve sanity check
python scripts/run_rates_curve_sanity.py --start 2021-01-01 --end 2025-10-31

# FX/Commodity Carry sanity check
python scripts/run_carry_sanity.py --start 2020-01-01 --end 2025-10-31

# Cross-Sectional Momentum sanity check
python scripts/run_csmom_sanity.py --start 2020-01-01 --end 2025-10-31

# CSMOM Phase-1 diagnostics
python scripts/run_csmom_phase1.py --start 2020-01-01 --end 2025-10-31
```

**Output:** All sanity check results are saved to `reports/sanity_checks/<meta_sleeve>/<phase>/<timestamp>/` with:
- Portfolio metrics (CAGR, Sharpe, MaxDD, HitRate)
- Per-asset/per-leg metrics
- Subperiod analysis (pre-2022 vs post-2022)
- Equity curves and return histograms

**Phase-0 Pass Criteria**: Sharpe ≥ 0.2+ over long window. Any sleeve that fails Phase-0 remains disabled until reworked.

See `docs/SOTs/STRATEGY.md` for the complete Sleeve Development Lifecycle (Phase-0 through Phase-3).

### Using FeatureStore (Caching)

```python
from src.agents import MarketData, FeatureStore

md = MarketData()
fs = FeatureStore(md)

# First call - cache miss
returns1 = fs.get_returns(("ES",))

# Second call - cache hit
returns2 = fs.get_returns(("ES",))

# Check cache stats
print(fs.cache_stats())

# Clear cache if needed
fs.clear_cache()

md.close()
```

### Run Phase-0 Sanity Checks

Validate that core economic ideas have positive alpha before adding complexity:

```bash
# TSMOM (Trend Meta-Sleeve) - validates momentum edge
python scripts/run_tsmom_sanity.py --start 2021-01-01 --end 2025-10-31

# Rates Curve - validates curve trading edge
python scripts/run_rates_curve_sanity.py --start 2021-01-01 --end 2025-10-31 --equal_notional

# FX/Commodity Carry - validates roll yield edge
python scripts/run_carry_sanity.py --start 2020-01-01 --end 2025-10-31 --universe CL,GC,6E,6B,6J
```

Results are saved to `reports/sanity_checks/<type>/<timestamp>/` with comprehensive metrics and plots.

### Generate Diagnostic Reports

```python
from src.agents.diagnostics import make_report

# After running a backtest with ExecSim
results = exec_sim.run(market, start, end, components)

# Generate comprehensive diagnostic report
report = make_report(results, outdir="reports/phase1")

# Access computed metrics
print(f"CAGR: {report['metrics']['cagr']:.2%}")
print(f"Sharpe: {report['metrics']['sharpe']:.2f}")
print(f"Max Drawdown: {report['metrics']['max_drawdown']:.2%}")
print(f"Calmar Ratio: {report['metrics']['calmar']:.2f}")

# Access saved CSV file paths
print(f"Equity curve saved to: {report['files']['equity']}")
print(f"Weights saved to: {report['files']['weights_total']}")

# All metrics available:
# - cagr, vol, sharpe, max_drawdown, calmar
# - hit_rate, avg_drawdown_length
# - avg_gross_exposure, avg_net_exposure
# - avg_turnover, cost_drag
```

### Flag Roll Jumps

```python
# Identify potential roll jumps (large returns > 1%)
jumps = md.flag_roll_jumps(
    symbols=("ES", "NQ", "CL"),
    threshold_bp=100  # 100 basis points = 1%
)

print(jumps)
# date       | symbol | return  | flagged
# 2023-03-15 | CL     | -0.0234 | True
# 2023-09-20 | ES     | 0.0156  | True
```

### Missing Data Report

```python
# Get coverage report
report = md.missing_report(("ES", "NQ", "ZN", "CL", "GC", "6E"))

print(report)
# symbol | total_days | missing_days | coverage_pct
# ES     | 1256       | 0            | 100.0
# NQ     | 1256       | 2            | 99.84
# CL     | 1256       | 15           | 98.81
```

## Public API Reference

### MarketData Class

| Method | Description |
|--------|-------------|
| `get_price_panel(symbols, fields, start, end, tidy)` | Get OHLCV data in tidy or wide format |
| `get_returns(symbols, start, end, method, price)` | Calculate returns (log or simple) |
| `get_vol(symbols, lookback, start, end, returns)` | Calculate rolling volatility (annualized) |
| `get_cov(symbols, lookback, end, shrink)` | Calculate covariance matrix |
| `get_fred_indicator(series_id, start, end)` | Get single FRED economic indicator time series |
| `get_fred_indicators(series_ids, start, end)` | Get multiple FRED indicators as DataFrame |
| `snapshot(asof)` | Create point-in-time snapshot instance |
| `get_meta(symbols)` | Get metadata (multiplier, point_value, etc.) |
| `trading_days(symbols)` | Get union of all trading days |
| `missing_report(symbols)` | Report missing data coverage |
| `flag_roll_jumps(symbols, threshold_bp)` | Flag potential roll jumps |
| `close()` | Close database connection |

### Diagnostics Module

| Function | Description |
|----------|-------------|
| `make_report(results, outdir)` | Generate comprehensive diagnostic report from backtest results |

**Returns:**
- `metrics`: Dict with CAGR, vol, Sharpe, Calmar, max drawdown, hit rate, avg DD length, exposure metrics, turnover, cost drag
- `files`: Dict with paths to saved CSV files (equity, weights, sleeve P&L, asset P&L, turnover/costs)

## Running Tests

Run the complete test suite:

```bash
# Run all tests with verbose output
pytest tests/test_marketdata.py -v

# Run with coverage report
pytest tests/test_marketdata.py --cov=src.agents --cov-report=term-missing

# Run specific test class
pytest tests/test_marketdata.py::TestReadOnlyConnection -v

# Run specific test
pytest tests/test_marketdata.py::TestReadOnlyConnection::test_connect_readonly -v
```

### Test Categories

#### MarketData Tests
- **TestReadOnlyConnection**: Verify read-only enforcement and schema discovery
- **TestMarketDataAPI**: Test core API methods and data shapes
- **TestAdditionalMethods**: Test helper methods (trading_days, missing_report, etc.)
- **TestFeatureStore**: Test caching wrapper
- **TestEdgeCases**: Test error handling and edge cases

#### Diagnostics Tests (22 tests)
- **TestMetricShapes**: Validate output structure and column shapes
- **TestPathsExist**: Verify CSV files are created at correct paths
- **TestSharpeDefinition**: Validate Sharpe = mean(daily)/std(daily)*sqrt(252)
- **TestBackwardCompatibility**: Ensure compatibility with ExecSim output
- **TestMetricCalculations**: Verify individual metric calculations
- **TestEdgeCases**: Handle empty data and edge cases
- **TestTimeAlignment**: Ensure consistent date alignment
- **TestDeterminism**: Verify deterministic outputs

## Safety Guarantees

1. ✅ **No Writes**: All connections use `read_only=True`; write attempts raise errors
2. ✅ **No Mutations**: No CREATE, INSERT, UPDATE, DELETE, DROP, ALTER, or COPY TO operations
3. ✅ **Dual-Price Architecture**: Raw prices from DB are unchanged; continuous prices built in-memory (no DB writes)
4. ✅ **Schema Discovery**: No hardcoded table names; discovers by column requirements
5. ✅ **Validation**: Drops duplicates, coerces numerics, verifies monotonic dates
6. ✅ **Logging**: All queries logged with `[READ-ONLY]` prefix at DEBUG level

## Configuration

Edit `configs/data.yaml`:

```yaml
db:
  path: "path/to/your/database"
  engine: "auto"  # auto-detect: duckdb or sqlite

universe:
  - "ES"
  - "NQ"
  - "ZN"
  - "CL"
  - "GC"
  - "6E"

defaults:
  vol_lookbacks: [21, 63, 252]
  cov_lookback: 252
  return_method: "log"
  roll_jump_threshold_bp: 100

logging:
  level: "DEBUG"
  prefix: "[READ-ONLY]"
```

## Database Requirements

The source database must contain a table with these **required columns** (case-insensitive):

- `date`
- `symbol`
- `open`
- `high`
- `low`
- `close`
- `volume`

**Optional helpful columns:**

- `roll_type`
- `source`
- `currency`
- `multiplier`
- `point_value`

## Limitations

- **No Back-Adjustment**: Prices contain roll jumps; use `flag_roll_jumps()` for diagnostics
- **No Forward Fill**: Missing data returns NaN; no automatic gap filling
- **Small Dataset**: Optimized for ~6 symbols × 5 years; larger datasets may need optimization
- **Daily Data Only**: Currently designed for daily OHLCV; intraday not supported

## Troubleshooting

### "Database path does not exist"

Check that the path in `configs/data.yaml` is correct and accessible.

### "No table found with required columns"

Verify your database contains a table with all required OHLCV columns.

### "Connection is NOT read-only"

The database type couldn't be opened in read-only mode. Check that you're using DuckDB ≥0.9.0 or SQLite with URI support.

## Development

### Adding New Features

1. Follow the "read-only only" principle
2. Add tests in `tests/test_marketdata.py`
3. Log all queries with `[READ-ONLY]` prefix
4. Update this README

### Code Style

- Follow PEP 8
- Use type hints
- Add docstrings to all public methods
- Keep functions focused and testable

## License

[Specify license]

## Contact

[Specify contact information]

