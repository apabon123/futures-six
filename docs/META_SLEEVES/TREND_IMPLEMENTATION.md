# Trend Meta-Sleeve — Implementation Reference

## Overview

This document provides a detailed implementation reference for the **Trend Meta-Sleeve**, the current production baseline for the futures-six strategy. The Trend Meta-Sleeve combines multiple atomic sleeves (long-term, medium-term, short-term momentum) into a unified meta-signal.

**Relationship to Other Documentation:**
- **`SOTs/STRATEGY.md`**: High-level architecture and execution flow
- **`TREND_RESEARCH.md`**: Research notebook for experimental atomic sleeve ideas
- **`SOTs/DIAGNOSTICS.md`**: Performance diagnostics and Phase-0 sanity checks

## Production Implementation

### Strategy Class

**`TSMOMMultiHorizon` (v2)** in `src/agents/strat_tsmom_multihorizon.py`

This is the unified Trend Meta-Sleeve implementation that combines three atomic sleeves into a single meta-signal.

### Architecture

The Trend Meta-Sleeve implements a **two-layer architecture**:

1. **Atomic Sleeves** (implementation variants):
   - **Long-term momentum (252d)**: 252-day return, 252-day breakout, slow trend slope (EMA_63 - EMA_252)
   - **Medium-term momentum (84/126d)**: 84-day return, 126-day breakout, medium trend slope (EMA_20 - EMA_84), persistence
   - **Short-term momentum (21d)**: 21-day return, 21-day breakout, fast trend slope (EMA_10 - EMA_40), reversal filter (optional)
   - **Residual Trend (252d-21d)**: Long-horizon trend minus short-term movement (Phase-2 integrated, currently not in production)
   - **Breakout Mid (50-100d)**: 50/100-day breakout-based signals (Phase-2 approved → Phase-3; production)

2. **Meta-Sleeve** (economic idea):
   - Combines atomic sleeves using configurable horizon weights
   - Applies cross-sectional z-scoring and EWMA volatility normalization
   - Produces risk-normalized signals for all assets

### Feature Computation

Features are pre-computed by `FeatureService` using `src/agents/feature_long_momentum.py`:

**Long-Term Features** (252-day horizon — Canonical TSMOM-252):
- `mom_long_ret_252_z_{symbol}`: 252-day return momentum (vol-standardized, z-scored)
- `mom_long_breakout_252_z_{symbol}`: 252-day breakout strength (normalized position in 252-day range)
- `mom_long_slope_slow_z_{symbol}`: Slow trend slope (EMA_63 - EMA_252, vol-standardized)

**Canonical Long-Term Composite** (Production, Nov 2025):
```python
long_trend_score = (1/3) * mom_long_ret_252_z 
                 + (1/3) * mom_long_breakout_252_z 
                 + (1/3) * mom_long_slope_slow_z
```
**Note**: The equal-weight (1/3, 1/3, 1/3) composite replaced the legacy (0.5, 0.3, 0.2) after Phase-0/1/2 validation in Nov 2025. The canonical weighting provides better diversification and is more robust (Sharpe +0.013, CAGR +0.17%, MaxDD +0.45% vs legacy).

**Medium-Term Features** (84/126-day horizon):
- `mom_med_ret_84_z_{symbol}`: 84-day return momentum (vol-standardized, z-scored)
- `mom_med_breakout_126_z_{symbol}`: 126-day breakout strength (normalized position in 126-day range)
- `mom_med_slope_med_z_{symbol}`: Medium trend slope (EMA_20 - EMA_84, vol-standardized)
- `mom_med_persistence_z_{symbol}`: Trend persistence (sign consistency over 20 days)

**Short-Term Features** (21-day horizon):
- `mom_short_ret_21_z_{symbol}`: 21-day return momentum (vol-standardized, z-scored)
- `mom_short_breakout_21_z_{symbol}`: 21-day breakout strength (normalized position in 21-day range)
- `mom_short_slope_fast_z_{symbol}`: Fast trend slope (EMA_10 - EMA_40, vol-standardized)
- `mom_short_reversal_filter_z_{symbol}`: Reversal filter (RSI-like, optional, not used by default)

**Residual Trend Features (4th Atomic Sleeve):**
- `trend_resid_ret_252_21_z_{symbol}`: Residual trend (long-horizon minus short-term, z-scored)

**Breakout Mid Features (5th Atomic Sleeve - Experimental):**
- `mom_breakout_mid_50_z_{symbol}`: 50-day breakout strength (normalized position in 50-day range)
- `mom_breakout_mid_100_z_{symbol}`: 100-day breakout strength (normalized position in 100-day range)

All features are standardized with rolling 252-day z-scores (clipped at ±3.0) per symbol.

### Signal Processing Pipeline

**IMPORTANT: Two-Layer Weight Architecture**

The Trend Meta-Sleeve uses a **two-layer weight structure**:
1. **Feature Weights** (Layer 1): Combine features *inside* each atomic sleeve
2. **Horizon Weights** (Layer 2): Combine atomic sleeves into the meta-signal

**Step 1: Combine Features Within Each Atomic Sleeve** (Feature Weights - Layer 1)

- **Long-term signal** (Canonical, Nov 2025): 
  ```python
  long_signal = (1/3) * ret_252 + (1/3) * breakout_252 + (1/3) * slope_slow
  ```
  *Note*: Legacy weighting was 0.5/0.3/0.2; canonical equal-weight replaced it after Phase-2 validation

- **Medium-term signal**: 
  ```python
  med_signal = 0.4 * ret_84 + 0.3 * breakout_126 + 0.2 * slope_med + 0.1 * persistence
  ```

- **Short-term signal**: 
  ```python
  short_signal = 0.5 * ret_21 + 0.3 * breakout_21 + 0.2 * slope_fast
  ```

**Step 2: Blend Atomic Sleeves Using Horizon Weights** (Horizon Weights - Layer 2)

**Production (5 atomic sleeves):**
```python
trend_meta_signal = 0.485 * long_signal 
                  + 0.291 * med_signal 
                  + 0.194 * short_signal 
                  + 0.03  * breakout_mid_signal
```

Where `breakout_mid_signal = 0.7 * breakout_50 + 0.3 * breakout_100`

**Horizon Weights** (across atomic sleeves): 48.5% / 29.1% / 19.4% / 3.0%  
**Feature Weights** (inside long-term sleeve): 1/3 / 1/3 / 1/3 (canonical)

**Note**: Residual Trend (252d-21d) is currently not included in the production configuration. The production Trend Meta-Sleeve uses long, medium, short, and breakout_mid atomic sleeves.

**Step 3: Cross-Sectional Z-Scoring**

Normalize across assets:
```
z_scored = (trend_meta_signal - mean(trend_meta_signal)) / std(trend_meta_signal)
```

**Step 4: Clip to ±3.0 Standard Deviations**

```
z_clipped = clip(z_scored, -3.0, 3.0)
```

**Step 5: EWMA Volatility Normalization** (if enabled)

- Compute EWMA annualized volatility for each asset: `σ_annual = EWMA(returns², halflife=63) * sqrt(252)`
- Risk-normalize: `s_risk = z_clipped / max(σ_annual, σ_floor)` where `σ_floor = 0.05` (5%)
- Apply global scale: `s_final = risk_scale * s_risk` (default: `risk_scale = 0.2`)

### Configuration

**YAML Configuration** (`configs/strategies.yaml`):

```yaml
tsmom_multihorizon:
  enabled: true
  
  # ===== LAYER 1: Feature Weights (inside each atomic sleeve) =====
  feature_weights:
    long:
      ret_252: 0.333333       # Canonical (1/3) - Nov 2025
      breakout_252: 0.333333  # Canonical (1/3) - Nov 2025
      slope_slow: 0.333334    # Canonical (1/3) - Nov 2025
      # Legacy weighting was 0.5/0.3/0.2
    medium:
      ret_84: 0.4
      breakout_126: 0.3
      slope_med: 0.2
      persistence: 0.1
    short:
      ret_21: 0.5
      breakout_21: 0.3
      slope_fast: 0.2
      reversal: 0.0  # Not used by default
    breakout_mid_50_100:
      breakout_50: 0.7
      breakout_100: 0.3
  
  # ===== LAYER 2: Horizon Weights (across atomic sleeves) =====
  horizon_weights:
    long_252: 0.485          # Long-term (252d) horizon
    med_84: 0.291            # Medium-term (84/126d) horizon
    short_21: 0.194          # Short-term (21d) horizon
    breakout_mid_50_100: 0.03  # Breakout (50-100d) horizon
    # Note: These are horizon/sleeve weights, NOT feature weights
  
  # Signal processing
  signal_cap: 3.0  # Clip at ±3.0 standard deviations
  
  # EWMA volatility normalization
  vol_normalization:
    enabled: true
    halflife_days: 63
    sigma_floor_annual: 0.05
    risk_scale: 0.2
```

**Key Distinction**:
- **Feature Weights** (Layer 1): Combine features *inside* each atomic sleeve (e.g., 1/3 + 1/3 + 1/3 for long-term)
- **Horizon Weights** (Layer 2): Combine atomic sleeves into meta-signal (e.g., 48.5% long + 29.1% med + ...)

### Usage Example

```python
from src.agents.data_broker import MarketData
from src.agents.strat_tsmom_multihorizon import TSMOMMultiHorizon
from src.agents.feature_service import FeatureService

# Initialize market data
market = MarketData()

# Initialize feature service (pre-computes momentum features)
feature_service = FeatureService(market)
feature_service.precompute_features(
    start="2020-01-01",
    end="2025-10-31"
)

# Initialize Trend Meta-Sleeve strategy
strategy = TSMOMMultiHorizon(
    feature_service=feature_service,
    config_path="configs/strategies.yaml"
)

# Generate signals for a specific date
date = "2024-01-15"
signals = strategy.signals(market, date)

print(signals)
# Output:
# ES_FRONT_CALENDAR_2D     0.523
# NQ_FRONT_CALENDAR_2D     1.234
# ZN_FRONT_VOLUME         -0.456
# ...
```

### Integration with Execution Flow

The Trend Meta-Sleeve integrates into the execution flow as follows:

1. **Feature Pre-computation**: `FeatureService` pre-computes all momentum features
2. **Signal Generation**: `TSMOMMultiHorizon.signals()` generates meta-signal for all assets
3. **Combined Strategy**: Signals are combined with other meta-sleeves (e.g., CSMOM) via `CombinedStrategy`
4. **Risk Overlays**: Volatility targeting and macro regime filters operate at Meta-Sleeve layer
5. **Portfolio Allocation**: Allocator optimizes portfolio weights from meta-sleeve signals

### Performance Characteristics

**Current Performance (2021-01-01 to 2025-10-31):**
- **CAGR**: -2.59%
- **Sharpe**: -0.15
- **MaxDD**: -39.69%
- **Vol**: 12.31%
- **HitRate**: 49.77%

**Note**: Performance reflects the challenging 2022-2025 period for trend-following strategies. The architecture is validated and production-ready; performance will improve as additional meta-sleeves are added.

### Key Design Decisions

1. **Multi-Horizon Approach**: Combines long, medium, and short-term horizons to capture trends at different time scales
2. **Industry-Standard Weights**: Uses 0.50/0.30/0.20 horizon weights (long/med/short) based on CTA industry standards
3. **Multi-Feature Per Horizon**: Each horizon uses multiple features (return, breakout, slope) to improve signal quality
4. **EWMA Vol Normalization**: Risk-normalizes signals to ensure same signal magnitude implies same risk across assets
5. **Cross-Sectional Z-Scoring**: Normalizes signals across assets to prevent single asset from dominating

### Dependencies

**Internal Dependencies:**
- `src/agents/feature_long_momentum.py`: Feature computation
- `src/agents/feature_service.py`: Feature pre-computation and caching
- `src/agents/data_broker.py`: MarketData interface
- `configs/strategies.yaml`: Configuration

**Data Requirements:**
- Continuous futures prices (back-adjusted)
- Sufficient history for feature computation (252+ days)
- All 13 assets in production universe

### Troubleshooting

**Issue: All Signals are NaN**
- **Cause**: Insufficient data for feature computation
- **Solution**: Ensure data history ≥ 252 days for all assets

**Issue: Signals Not Changing**
- **Cause**: Features not being recomputed or cached incorrectly
- **Solution**: Check `FeatureService` pre-computation and cache invalidation

**Issue: Volatility Normalization Producing Extreme Signals**
- **Cause**: Volatility floor too low or risk_scale too high
- **Solution**: Adjust `sigma_floor_annual` and `risk_scale` in config

### Comparison to Legacy TSMOM

The current `TSMOMMultiHorizon` implementation replaces the legacy `TSMOM` class (`src/agents/strat_momentum.py`):

**Legacy TSMOM:**
- Single-horizon momentum (e.g., 252 days)
- Single feature (return momentum)
- Optional multi-lookback blend

**Current TSMOMMultiHorizon:**
- Multi-horizon momentum (long/med/short)
- Multi-feature per horizon (return, breakout, slope, persistence)
- Unified meta-sleeve architecture
- EWMA volatility normalization

**Migration Note**: The legacy `TSMOM` class is still available but not used in production. See `docs/legacy/TSMOM_IMPLEMENTATION.md` for legacy implementation details.

---

**Last Updated**: November 2025

**Status**: ✅ Production (core_v3_no_macro baseline)

**Strategy Profile**: `core_v3_no_macro` (Trend Meta-Sleeve only)

