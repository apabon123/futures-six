# futures-six: Time-Series Momentum Strategy Framework

A complete, production-ready systematic trading framework for futures momentum strategies with comprehensive backtesting capabilities.

## Overview

**futures-six** is a modular, agent-based framework for building and backtesting systematic trading strategies on continuous futures contracts. The framework implements a full Time-Series Momentum (TSMOM) strategy with volatility targeting, risk management, and portfolio optimization.

### What This Project Does

- 📊 **Reads market data** from a local DuckDB database with strict read-only access
- 📈 **Generates momentum signals** using 12-1 month lookback with customizable parameters
- 🎯 **Targets volatility** to achieve consistent risk exposure (default: 20% annual vol)
- ⚖️ **Optimizes portfolio weights** using signal-beta or equal-risk-contribution methods
- 💰 **Simulates execution** with realistic slippage and transaction costs
- 📉 **Produces comprehensive metrics**: CAGR, Sharpe ratio, max drawdown, turnover, etc.

### Quick Start Results

Out of the box, the framework achieves:
- **20.8% CAGR** (2021-2025)
- **0.84 Sharpe Ratio**
- **-22.3% Max Drawdown**
- **249 rebalances** over 4.8 years

## Universe

The strategy trades six continuous futures contracts:

- **ES_FRONT_CALENDAR_2D** - E-mini S&P 500 (calendar roll, 2 days before expiry)
- **NQ_FRONT_CALENDAR_2D** - E-mini NASDAQ-100 (calendar roll)
- **ZN_FRONT_VOLUME** - 10-Year Treasury Note (volume-weighted roll)
- **CL_FRONT_VOLUME** - Crude Oil WTI (volume-weighted roll)
- **GC_FRONT_VOLUME** - Gold (volume-weighted roll)
- **6E_FRONT_CALENDAR_2D** - Euro FX (calendar roll)

## Key Features

- ✅ **Read-Only Access**: All database connections enforce `read_only=True`
- ✅ **Schema Discovery**: Auto-detects OHLCV tables by required columns
- ✅ **No Back-Adjustment**: Prices contain roll jumps; jumps are flagged, not adjusted
- ✅ **Point-in-Time Snapshots**: `snapshot(asof)` for historical queries
- ✅ **Standardized APIs**: Prices, returns, volatility, covariance
- ✅ **Comprehensive Logging**: All queries logged with `[READ-ONLY]` prefix
- ✅ **Minimal Dependencies**: DuckDB, pandas, numpy, PyYAML

## Project Structure

```
futures-six/
├── configs/
│   ├── data.yaml              # Database connection and universe config
│   └── strategies.yaml        # Strategy parameters (TSMOM, vol overlay, allocator)
├── src/
│   └── agents/
│       ├── data_broker.py     # MarketData: Read-only OHLCV data access
│       ├── strat_momentum.py  # TSMOM: Time-series momentum strategy
│       ├── strat_cross_sectional.py  # Cross-Sectional Momentum strategy
│       ├── overlay_volmanaged.py  # VolManaged: Volatility targeting
│       ├── overlay_macro_regime.py  # MacroRegime: Regime-based signal scaling
│       ├── risk_vol.py        # RiskVol: Volatility & covariance calculations
│       ├── allocator.py       # Allocator: Portfolio weight optimization
│       ├── exec_sim.py        # ExecSim: Backtest execution engine
│       ├── diagnostics.py     # Diagnostics: Performance metrics & attribution
│       └── feature_store.py   # FeatureStore: Optional caching layer
├── tests/
│   ├── test_marketdata.py     # MarketData tests
│   ├── test_strat_momentum.py # TSMOM tests (24 tests)
│   ├── test_strat_cross_sectional.py # Cross-Sectional Momentum tests (22 tests)
│   ├── test_overlay_volmanaged.py # Vol overlay tests
│   ├── test_overlay_macro_regime.py # Macro regime overlay tests (14 tests)
│   ├── test_risk_vol.py       # Risk calculation tests
│   ├── test_allocator.py      # Portfolio optimization tests
│   ├── test_exec_sim.py       # Backtest engine tests
│   └── test_diagnostics.py    # Diagnostics & attribution tests
├── docs/                      # Centralized documentation (see docs/README.md)
├── run_strategy.py            # Main entry point - run the complete backtest
├── verify_setup.py            # Quick verification script
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

### Documentation

All component-specific guides live in `docs/`. Start with `docs/README.md` for an index of the available references (TSMOM, Cross-Sectional Momentum, MacroRegimeFilter, implementation summaries, etc.).

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
```

### 3. Verify Setup

```bash
python verify_setup.py
```

This will test the data broker connection and verify all components are working.

### 4. Run the Strategy

```bash
python run_strategy.py
```

This runs the complete TSMOM strategy backtest from 2021 to present and displays performance metrics.

### 5. Run Tests

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
- **Role**: Read-only access to OHLCV data from DuckDB/SQLite
- **Features**: Schema auto-discovery, column name mapping, point-in-time snapshots
- **Methods**: `get_price_panel()`, `get_returns()`, `get_vol()`, `get_cov()`

### 2. TSMOM (Strategy Agent)
- **Role**: Generate time-series momentum signals
- **Method**: 12-1 month lookback (252 days lookback, skip recent 21 days)
- **Standardization**: Volatility-scaled signals (return / trailing vol)
- **Rebalance**: Weekly on Fridays
- **Output**: Signals capped at ±3.0 standard deviations

### 2b. Cross-Sectional Momentum (Alternative Strategy)
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
- **Logic**: 
  - Higher volatility → lower scaler (risk-off)
  - Lower breadth → lower scaler (defensive)
  - EMA smoothing (default: 0.2) for gradual transitions
- **Rebalance**: Weekly on Fridays (configurable)
- **Bounds**: k ∈ [0.4, 1.0] (default), reduces exposure by up to 60% in adverse regimes
- **No Look-Ahead**: All indicators computed using point-in-time data only

### 5. RiskVol (Risk Calculations)
- **Role**: Calculate rolling volatilities and covariance matrix
- **Vol Lookback**: 63 days (3 months)
- **Cov Lookback**: 252 days (1 year)
- **Shrinkage**: Ledoit-Wolf shrinkage for covariance stability

### 6. Allocator (Portfolio Optimization)
- **Role**: Convert vol-managed signals to final portfolio weights
- **Method**: Signal-beta (align weights with signal direction)
- **Constraints**: Gross cap (7x), net cap (2x), per-asset bounds (±1.5)
- **Turnover**: Penalizes excessive trading (50% cap, 0.001 lambda)

### 7. ExecSim (Backtest Engine)
- **Role**: Orchestrate all components and simulate execution
- **Costs**: Slippage (0.5 bps), commission (configurable)
- **Rebalance**: Weekly on Fridays
- **Metrics**: CAGR, Sharpe, max drawdown, hit rate, turnover, leverage

### 8. Diagnostics (Performance Attribution)
- **Role**: Generate comprehensive performance metrics and reports
- **Metrics**: CAGR, volatility, Sharpe, Calmar, hit rate, avg drawdown length, exposure, turnover, cost drag
- **Outputs**: CSV reports (equity, weights, P&L, turnover/costs)
- **Features**: Time-aligned data, backward compatible with ExecSim output

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

### Using MacroRegimeFilter (Regime-Based Scaling)

```python
from src.agents.data_broker import MarketData
from src.agents.overlay_macro_regime import MacroRegimeFilter

# Initialize market data
md = MarketData()

# Initialize macro regime filter
macro_filter = MacroRegimeFilter(
    rebalance="W-FRI",                      # Weekly rebalance
    vol_thresholds={'low': 0.15, 'high': 0.30},  # Vol regime thresholds
    k_bounds={'min': 0.4, 'max': 1.0},      # Scaler bounds
    smoothing=0.2,                          # EMA smoothing (20% new, 80% old)
    vol_lookback=21,                        # 1-month realized vol
    breadth_lookback=200,                   # 200-day SMA for breadth
    proxy_symbols=("ES", "NQ")              # Symbols for regime detection
)

# Sample strategy signals
signals = pd.Series({'ES': 1.5, 'NQ': -0.8, 'GC': 0.5})

# Get regime scaler for a date
date = "2024-01-05"
k = macro_filter.scaler(md, date)
print(f"Regime scaler: {k:.3f}")  # e.g., 0.750 (25% risk reduction)

# Apply scaler to signals
scaled_signals = macro_filter.apply(signals, md, date)
print(f"Original gross leverage: {signals.abs().sum():.2f}")
print(f"Scaled gross leverage: {scaled_signals.abs().sum():.2f}")

# Filter description
print(macro_filter.describe())

md.close()
```

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
3. ✅ **No Back-Adjustment**: Prices are not modified; roll jumps are flagged only
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

