# SR3 Carry and Curve Features

## Overview

Phase 2 implementation: Carry and curve features for SOFR (SR3) futures using 12 contract ranks (0-11). These features capture the short-rate curve structure and can be used for carry trading and curve shape analysis.

## Features Computed

### 1. Carver Carry (`sr3_carry_01_z`)
- **Definition**: `carry_01 = r1 - r0` where `r_k = 100 - P_k`
- **Interpretation**: "How much higher is the next 3M rate vs the front 3M rate"
- **Standardization**: Rolling 252-day z-score, clipped at ±3.0
- **Trading Signal**: Positive carry (r1 > r0) suggests upward-sloping front curve

### 2. Curve Shape (`sr3_curve_02_z`)
- **Definition**: `curve_02 = r2 - r0` where `r_k = 100 - P_k`
- **Interpretation**: "How much higher is the 3rd 3M rate vs front 3M rate"
- **Standardization**: Rolling 252-day z-score, clipped at ±3.0
- **Trading Signal**: Captures longer-term curve slope than just 0-1

### 3. Pack Slope (`sr3_pack_slope_fb_z`)
- **Definition**: `pack_slope_fb = pack_back - pack_front` (simple difference in rate space)
- **Packs**:
  - **Front**: Ranks 0-3 (mean of rates)
  - **Belly**: Ranks 4-7 (mean of rates)
  - **Back**: Ranks 8-11 (mean of rates)
- **Interpretation**: Positive = back higher than front (upward sloping curve)
- **Standardization**: Rolling 252-day z-score, clipped at ±3.0
- **Trading Signal**: Steep front-to-back curve suggests term structure shape

### 4. Front-Pack Level (`sr3_front_pack_level_z`)
- **Definition**: `front_pack_level = pack_front = mean(r_0, r_1, r_2, r_3)`
- **Interpretation**: Absolute level of expected policy over ~1 year (0-12M)
- **Standardization**: Rolling 252-day z-score, clipped at ±3.0
- **Trading Signal**: 
  - High front-pack level → policy very tight → less room for hikes, more room for cuts
  - Low front-pack level → policy very easy → more room for hikes
- **Usage**: Usually small weight (0.1-0.2) to bias curve trades that align with extreme policy levels

### 5. Belly Curvature (`sr3_curvature_belly_z`)
- **Definition**: `curvature_belly = belly_pack - (pack_front + pack_back) / 2`
- **Interpretation**: Pure curvature signal measuring hump vs straight term structure
- **Standardization**: Rolling 252-day z-score, clipped at ±3.0
- **Trading Signal**:
  - Positive: belly rates higher than straight line between front and back → "hump-y" curve
  - Negative: belly lower → "U-shaped" or very concave curve
- **Usage**: Modest weight (0.15-0.2) to influence pack-based positioning

## Rate Space vs Price Space

**Key Design Decision**: All features computed in **rate space** (`r_k = 100 - P_k`), not price space.

### Why Rate Space?
- Prices are near 100; small differences are hard to interpret
- Rate differences are directly in basis points of expected future short-rate changes
- Natural roll-down of contracts is "rate moving toward spot" - exactly what carry captures
- More intuitive: positive carry means next period rate is higher than front

### Formulas

```
P_k(t) = close price of SR3 rank k on date t
r_k(t) = 100 - P_k(t)  # Convert to rate space

carry_01(t) = r1(t) - r0(t)        # Next 3M vs front 3M
curve_02(t) = r2(t) - r0(t)        # 3rd 3M vs front 3M

pack_front_r(t) = mean(r_0, r_1, r_2, r_3)
pack_belly_r(t) = mean(r_4, r_5, r_6, r_7)
pack_back_r(t)  = mean(r_8, r_9, r_10, r_11)

pack_slope_fb_raw(t) = pack_back_r(t) - pack_front_r(t)
front_pack_level_raw(t) = pack_front_r(t)
front_back_avg(t) = (pack_front_r(t) + pack_back_r(t)) / 2
curvature_belly_raw(t) = pack_belly_r(t) - front_back_avg(t)
```

## Implementation

### MarketData Extension

Added `get_contracts_by_root()` method to query contracts by root symbol and rank:

```python
# Query SR3 contracts by rank
close_prices = market.get_contracts_by_root(
    root="SR3",
    ranks=list(range(12)),  # 0-11
    fields=("close",),
    start="2020-01-01",
    end="2025-01-01"
)
# Returns: DataFrame [date x rank] with close prices
```

### Feature Service

**Module**: `src/agents/feature_sr3_curve.py`

**Class**: `Sr3CurveFeatures`

```python
from src.agents.feature_sr3_curve import Sr3CurveFeatures

# Initialize
sr3_features = Sr3CurveFeatures(root="SR3", window=252)

# Compute features
features = sr3_features.compute(market, end_date="2025-01-01")

# Returns DataFrame with columns:
# - sr3_carry_01_z
# - sr3_curve_02_z
# - sr3_pack_slope_fb_z
# - sr3_front_pack_level_z
# - sr3_curvature_belly_z
```

### Strategy Sleeve

**Module**: `src/agents/strat_sr3_carry_curve.py`

**Class**: `Sr3CarryCurveStrategy`

```python
from src.agents.strat_sr3_carry_curve import Sr3CarryCurveStrategy

# Initialize
strategy = Sr3CarryCurveStrategy(
    root="SR3",
    w_carry=0.30,      # Weight for carry feature
    w_curve=0.25,      # Weight for curve feature
    w_pack_slope=0.20, # Weight for pack slope feature
    w_front_lvl=0.10,  # Weight for front-pack level feature
    w_curv_belly=0.15, # Weight for belly curvature feature
    cap=3.0            # Signal cap in std devs
)

# Generate signals
signals = strategy.signals(market, date="2025-01-01")
# Returns: pd.Series with signal for SR3_FRONT_CALENDAR
```

## Standardization

All features are standardized using **rolling z-scores**:

```python
μ_252(t) = rolling_mean_252d(x(t))
σ_252(t) = rolling_std_252d(x(t))
z_252(t) = (x(t) - μ_252(t)) / σ_252(t)
z_clipped = clip(z_252(t), -3, 3)
```

- **Window**: 252 trading days (1 year)
- **Min Periods**: 126 days (allows features to start earlier with less stable estimates)
- **Clipping**: ±3.0 standard deviations
- **Purpose**: Make features comparable across time and prevent outliers from dominating

## Usage in Portfolio

The SR3 carry/curve strategy can be used as a separate sleeve in the portfolio allocator:

```python
# In portfolio construction
tsmom_signals = tsmom_strategy.signals(market, date)
sr3_carry_signals = sr3_carry_strategy.signals(market, date)

# Combine in allocator
combined_signals = pd.concat([tsmom_signals, sr3_carry_signals])
weights = allocator.solve(combined_signals, cov, ...)
```

Or as a separate sleeve with its own risk budget:

```python
# Multi-sleeve allocation
sleeves = {
    "TSMOM": tsmom_signals,
    "SR3_Carry": sr3_carry_signals,
    "XSec": xsec_signals
}
risk_budgets = {
    "TSMOM": 0.6,
    "SR3_Carry": 0.2,
    "XSec": 0.2
}
```

## Data Requirements

- **12 SR3 contracts** in database (ranks 0-11)
- Contracts should be named with "SR3" prefix (e.g., "SR3_RANK_0_CALENDAR", "SR3_RANK_1_CALENDAR", etc.)
- Contracts sorted by expiration date to determine rank order
- Minimum 126 days of history to start computing features (min_periods for rolling window)
- Full 252-day window for stable z-score standardization
- **Current availability**: Features computed from 2020-06-07 onwards (data starts 2020-01-01)

## Notes

- **Only rank 0 is tradeable**: Other ranks (1-11) are feature-only
- **No sign convention yet**: SR3 = 100 - rate; can flip signs later if live tests suggest
- **Point-in-time**: All features computed with no look-ahead bias
- **Handles missing data**: 
  - Forward-fill and backward-fill for contract data gaps
  - Returns empty DataFrame if insufficient data or missing ranks
  - Uses most recent available features when rebalance dates don't align (forward-fill)
- **Data availability**: Features available from 2020-06-07 onwards (after 126 days of history)

## Configuration and Usage

### Enable SR3 Carry/Curve Sleeve

Edit `configs/strategies.yaml`:

```yaml
strategies:
  tsmom:
    enabled: true
    weight: 0.6  # 60% weight in combined signal
  
  sr3_carry_curve:
    enabled: true  # Set to true to enable
    weight: 0.15   # 15% weight in combined signal
    params:
      w_carry: 0.30      # Weight for carry feature
      w_curve: 0.25      # Weight for curve feature
      w_pack_slope: 0.20 # Weight for pack slope feature
      w_front_lvl: 0.10  # Weight for front-pack level feature
      w_curv_belly: 0.15 # Weight for belly curvature feature
      cap: 3.0           # Signal cap in standard deviations
      window: 252         # Rolling window for feature standardization

features:
  sr3_curve:
    root: "SR3"
    ranks: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]  # 12 ranks
    window: 252  # Rolling window for z-score standardization
```

### Disable SR3 Sleeve

Simply set `enabled: false`:

```yaml
sr3_carry_curve:
  enabled: false  # Disabled - will not be included in combined signals
```

### Signal Combination

Signals are combined **before** overlays are applied:

```python
# In CombinedStrategy.signals()
combined_signal = (
    weight_tsmom * tsmom_signal +
    weight_sr3 * sr3_signal
)
```

Then the combined signal goes through:
1. **VolManaged overlay**: Volatility targeting
2. **MacroRegime overlay**: Regime-based scaling (if enabled)
3. **Allocator**: Portfolio optimization

### Execution Flow

**Initialization (run_strategy.py)**:
1. MarketData: Connect to database
2. FeatureService: Initialize feature calculators
3. Strategy Sleeves: Initialize TSMOM and SR3 (if enabled)
4. CombinedStrategy: Combine strategies with weights
5. Overlays: Initialize VolManaged and MacroRegime
6. Allocator: Initialize portfolio optimizer

**Per-Rebalance Loop (ExecSim)**:
- Features computed point-in-time (no look-ahead)
- Signals generated for each sleeve
- Combined with configurable weights
- Applied through overlays and allocator

### Troubleshooting

**SR3 Features Not Available**:
- Check: Are 12 SR3 contracts in database? (ranks 0-11)
- Check: Do contracts have sufficient history? (need 252+ days)
- Check: Are contracts named with "SR3" prefix?

```python
# Test SR3 data availability
from src.agents.data_broker import MarketData

market = MarketData()
sr3_data = market.get_contracts_by_root("SR3", ranks=list(range(12)))
print(f"SR3 contracts: {sr3_data.columns.tolist()}")
print(f"Date range: {sr3_data.index.min()} to {sr3_data.index.max()}")
```

**Zero Signals from SR3**:
- Check: Are features computed successfully?
- Check: Are feature values NaN?
- Check: Is the date in the feature index?

```python
# Check features
from src.agents.feature_service import FeatureService

feature_service = FeatureService(market)
features = feature_service.get_features(end_date="2025-01-01")
print(features["SR3_CURVE"].tail())
```

**Note**: SR3 Carry/Curve is currently **disabled** in Core v1 baseline (see `docs/SOTs/STRATEGY.md`). It will be redesigned in a future phase.

## Future Enhancements

- Additional pack slopes (front-belly, belly-back)
- ~~Term structure level features (absolute rate levels)~~ ✅ Implemented: `sr3_front_pack_level_z`
- ~~Curvature features (hump vs straight)~~ ✅ Implemented: `sr3_curvature_belly_z`
- Cross-sectional carry (compare SR3 carry to other rate futures)
- Dynamic weight adjustment based on regime
- Integration with macro regime filter

