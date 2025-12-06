# Strategy Execution Flow

This document describes the **exact step-by-step process** of how the futures-six strategy is executed. This is the single source of truth for understanding what happens at each stage of the backtest.

## Architecture: Two-Layer Sleeve Model

The strategy implements a **two-layer sleeve architecture** that separates economic ideas from implementation variants:

### Meta-Sleeves (Economic Sources of Return)

**Meta-Sleeves** represent distinct economic sources of return:

- **Trend (TSMOM)**: Time-series momentum across multiple horizons (each asset vs its own past).
- **Cross-Sectional Momentum (CSMOM)**: Relative momentum across assets (rank winners vs losers; typically market-neutral).
- **Carry**: Roll yield and term carry from futures curves (rates, FX, commodities).
- **Value**: Cross-sectional value and mean-reversion opportunities (valuation, spreads, slow fundamentals).
- **Curve RV**: Relative value in yield curve shapes (flatteners/steepeners, flies, pack spreads).
- **Volatility Risk Premium (VRP)**: Systematic volatility selling strategies (multiple atomic sleeves).
- **Seasonality / Flows**: Calendar effects and flow-driven patterns.

### Atomic Sleeves (Implementation Variants)

**Atomic Sleeves** are specific implementations within each Meta-Sleeve:
- **Trend Meta-Sleeve** includes:
  - Long-term momentum (252d) — Canonical (1/3, 1/3, 1/3)
  - Medium-term momentum (84/126d) — Legacy (production)
  - Medium-term momentum (84d) — Canonical (research, Phase-1)
  - Short-term momentum (21d) — Legacy (production: 0.5/0.3/0.2)
  - Short-term momentum (21d) — Canonical (research, Phase-0/1: 1/3, 1/3, 1/3)
  - Residual Trend (252d long - 21d short; validated Phase-1)
  - Breakout Mid (50-100d; production)
- **Volatility Risk Premium Meta-Sleeve** will include multiple atomic sleeves (to be designed):
  - Various volatility selling strategies
  - Different implementation approaches
  - Multiple feature sets and signal processing methods
- Each atomic sleeve can have different feature sets, horizons, or signal processing methods

### Signal Flow

```
Atomic Sleeves → Meta-Sleeve (vol-normalized, combined) → Risk Overlays → Portfolio
```

**Key Principle**: Risk overlays (macro filters, vol targeting, portfolio allocator) operate **only at the Meta-Sleeve layer**, not on individual atomic sleeves. Atomic sleeves feed into meta-sleeves, and meta-sleeves feed into the portfolio.

## Data Consistency and Run Alignment ⚠️

**CRITICAL**: To ensure valid performance comparisons, all runs must follow strict data alignment rules.

### Canonical Start Date Rule

**All runs begin at the first timestamp where ALL enabled atomic sleeves have valid signals for ALL assets.**

**Why This Matters:**

Different sleeves require different warmup periods:
- Long-Term (252d): ~252 trading days
- Medium-Term (84d): ~84 trading days  
- Short-Term (21d): ~21 trading days
- Breakout Mid (100d): ~100 trading days

**Example:**
```
Requested start: 2018-01-01
Longest warmup: 252 trading days
Effective start: 2018-12-15 (approximately)
```

The system automatically determines the effective start date as:
```
effective_start = requested_start + max(all_enabled_sleeve_warmups)
```

### NA Handling Pipeline

To prevent silent misalignments, NAs are dropped at each stage:

1. **FeatureService**: `features.dropna(how="any")` after computation
2. **Atomic Sleeves**: `signals.dropna(how="any")` after sleeve computation
3. **Meta-Sleeve**: `combined_sleeves.dropna(how="any")` before aggregation
4. **Portfolio**: `aligned_signals.dropna(how="any")` before position sizing

This ensures:
- No silent NA propagation
- Consistent row counts across runs
- Valid apples-to-apples comparisons
- Deterministic performance metrics

### Run Validation

Every run logs:
```
Requested start: 2018-01-01
Effective start: 2018-12-15
Valid rows: 1754
Rows dropped: 0
```

**Red flags** (investigate immediately):
- Effective start ≠ expected (requested + warmup)
- Valid rows < 1000 (for 2018-2025 range)
- Rows dropped > 5% of sample
- Row count mismatch between runs being compared

See `PROCEDURES.md` § 2 "Run Consistency Contract" for full details.

## Sleeve Development Lifecycle

All new sleeves must follow a structured development lifecycle:

### Phase-0: Simple Sanity Check
- **Purpose**: Verify that any positive alpha exists in the core economic idea
- **Requirements**: 
  - Sign-only signals (no z-scoring, no overlays)
  - Equal-weight or simple DV01-neutral weighting
  - Daily rebalancing
  - No vol targeting, macro filters, or portfolio optimization
- **Pass Criteria**: Sharpe ≥ 0.2+ over long window
- **Outcome**: If Phase-0 fails, sleeve remains **disabled** until reworked

### Phase-1: Clean Implementation
- Add proper feature engineering (z-scoring, standardization)
- Implement cross-sectional ranking if applicable
- Add basic signal processing

### Phase-2: Overlay Integration
- Integrate with macro regime filters
- Add volatility targeting
- Portfolio allocator integration

### Phase-3: Production + Monitoring
- Full production deployment
- Performance monitoring
- Ongoing refinement

**Any sleeve that fails Phase-0 remains disabled until the underlying economic idea is validated or redesigned.**

## Roadmap

### Next Meta-Sleeves to Test

1. **Cross-Sectional Momentum Meta-Sleeve** → ✅ Phase-2 Complete
   - **Status**: Active Meta-Sleeve in production configuration
   - See "Cross-Sectional Momentum (CSMOM) Meta-Sleeve — Status" section below for full details

2. **Volatility Risk Premium Meta-Sleeve** → Planned
   - Multiple atomic sleeves for volatility selling strategies
   - Various implementation approaches (e.g., VIX futures, volatility spreads, vol-of-vol)
   - Different feature sets and signal processing methods
   - Systematic capture of volatility risk premium across asset classes

3. **Value Meta-Sleeve** → Planned
   - Mean reversion signals
   - Relative value opportunities
   - Term structure value (futures vs spot)

4. **Rates Curve RV Meta-Sleeve** → Redesign
   - DV01-neutral flies (2s5s10s, 5s10s30s)
   - Pack spreads (front/mid/back pack positioning)
   - Macro gating (Fed policy regime filters)
   - Improved yield reconstruction

5. **Carry Meta-Sleeve** → Redesign
   - Sector-based roll yield (commodities vs FX vs rates)
   - DV01-neutral carry for rates
   - Regime-dependent filters (USD strength, commodity cycles)

6. **Seasonality/Flows Meta-Sleeve** → Future
   - Calendar effects (month-of-year, day-of-week)
   - Flow-driven patterns (quarter-end, rebalancing flows)
   - Event-driven signals

## Core Baseline Configurations

**Trend-Only Baseline:** `core_v3_no_macro`  

- Trend Meta-Sleeve only (no other Meta-Sleeves enabled)

- Used as the canonical reference profile for Trend research and A/B tests

- Configuration: `tsmom_multihorizon` with 4 active atomic sleeves (long, med, short, breakout)

- See "Trend Meta-Sleeve Architecture" section for full details.

**Current Multi-Sleeve Baseline (Production-Style):** `core_v4_trend_csmom_no_macro`  

- Trend Meta-Sleeve (75%) + Cross-Sectional Momentum Meta-Sleeve (25%)

- No macro overlay, no vol overlay, no allocator yet

- Used as the default "full strategy" profile for future Meta-Sleeve integration and Phase-2 tests

- Configuration:  

  - `tsmom_multihorizon` weight: 0.75 (Trend)  

  - `csmom_meta` weight: 0.25 (CSMOM)  

  - CSMOM uses 63/126/252-day cross-sectional momentum with horizon weights [0.4, 0.35, 0.25]

### Baseline Performance (Trend vs Trend + CSMOM, 2020-01-02 to 2025-10-31)

**Baseline (Trend-only):** `core_v3_baseline_2020_2025`  

**Variant (Trend + CSMOM 75/25):** `core_v4_trend_csmom_2020_2025`

| Metric   | Trend Only | Trend + CSMOM | Change |
|----------|-----------:|--------------:|-------:|
| Sharpe   | 0.0294     | 0.0902        | +0.0608 |
| CAGR     | -0.38%     | +0.35%        | +0.73% |
| MaxDD    | -29.85%    | -28.09%       | +1.76% |
| Vol      | 12.19%     | 12.29%        | +0.10% |
| Equity Ratio (variant / baseline) | 1.0000 | 1.0549 | +5.49% |

**Verdict:** `core_v4_trend_csmom_no_macro` is promoted to the current multi-sleeve baseline. `core_v3_no_macro` remains the canonical Trend-only baseline for diagnostics and future research.

### Enabled Meta-Sleeves

- ✅ **Trend Meta-Sleeve**: Multi-horizon time-series momentum
  - **Active Atomic Sleeves** (5 production):
    - **Long-term momentum (252d) — Canonical**: Equal-weight (1/3, 1/3, 1/3) of 252d return, 252d breakout, slow trend slope
    - **Medium-term momentum (84/126d) — Legacy**: 84-day return, 126-day breakout, medium trend slope (EMA_20 - EMA_84), persistence
    - **Short-term momentum (21d)**: 21-day return, 21-day breakout, fast trend slope (EMA_10 - EMA_40), reversal filter (optional)
    - **Residual Trend (252d-21d)**: Long-horizon trend minus short-term movement, cross-sectionally z-scored
    - **Breakout (50–100d)**: 50/100-day breakout-based trend sleeve (Production; Donchian-style range breakouts, 70/30 feature weights, 3% horizon weight)
  - **Research Atomic Sleeves** (Phase-1):
    - **Medium-term momentum (84d) — Canonical**: Equal-weight (1/3, 1/3, 1/3) of 84d return, 84d breakout, EMA21-84 slope (skip 10d, 21d vol scaling)
  - **Signal Processing**: 
    - Atomic sleeves are vol-normalized and combined using **frozen horizon weights** (long=0.485, med=0.291, short=0.194, breakout_mid=0.03)
    - Cross-sectional z-scoring across assets
    - EWMA volatility normalization (63-day half-life, 5% floor)
  - **Internal Weights (FROZEN)**: 48.5% long, 29.1% medium, 19.4% short, 3% breakout_mid
    - **⚠️ Weight Freeze Policy**: These weights are frozen during architecture build-out. Optimization belongs in Phase-B with proper OOS controls (see `PROCEDURES.md`).
    - **Note**: Breakout (50-100d) atomic sleeve passed Phase-0/1B/2/3 validation (70/30 config, 3% weight). Integrated into production `core_v3_no_macro` profile.
  - **Implementation**: TSMOMMultiHorizon strategy (v2)

- ✅ **Cross-Sectional Momentum (CSMOM) Meta-Sleeve**

### Cross-Sectional Momentum (CSMOM) Meta-Sleeve

**Status:** Phase-2 complete (validated), not yet in production baseline  

**Implementation:** Single multi-horizon atomic sleeve

The CSMOM Meta-Sleeve is currently implemented as a **single atomic sleeve** that internally blends multiple return horizons. Unlike the Trend Meta-Sleeve (which exposes several atomic sleeves explicitly), CSMOM uses a single engine with multi-horizon features:

- Lookbacks: **63d, 126d, 252d**

- Horizon weights: **[0.4, 0.35, 0.25]** (shorter horizons slightly favored)

- Vol lookback: **63d**

- Rebalance: **Daily ("D")**

- Cross-sectional neutralization: **enabled**

- Clipping: **±3.0** (signals scaled to [-1, 1])

**Signal construction (CSMOMMeta):**

1. Compute k-day log returns for each horizon (63d, 126d, 252d)

2. For each date and horizon, z-score returns across assets (cross-sectional z-score)

3. Form a weighted composite of horizon z-scores using [0.4, 0.35, 0.25]

4. Apply volatility tempering (divide by 63d realized vol) and re-z-score

5. Cross-sectionally neutralize final scores (zero-sum across the universe)

6. Clip to ±3.0 and scale to the [-1, +1] signal range

7. Forward-fill between daily rebalance dates (rebalance = "D" currently, so no gap)

**Lifecycle status:**

- **Phase-0:** PASS (simple sign-only CSMOM shows positive Sharpe on 2020–2025 window)

- **Phase-1:** PASS (multi-horizon, volatility-aware implementation)

- **Phase-2:** PASS (Trend + CSMOM, with 75%/25% sleeve weights, improves Sharpe and reduces drawdown vs Trend-only baseline)

At this stage, CSMOM is treated as a **single canonical atomic sleeve** for the Cross-Sectional Momentum Meta-Sleeve. The three horizons (63d/126d/252d) are treated as internal features of that atomic sleeve, not as separate atomic sleeves.

Future work (Phase-B enhancement) may introduce additional atomic sleeves for CSMOM (e.g., different lookback structures, volatility-adjusted variants, sector-neutral variants) following the same Meta-Sleeve → Atomic Sleeve pattern used in Trend.

### Disabled/Parked Meta-Sleeves

- ❌ **Carry Meta-Sleeve**: **Parked for redesign**
  - **Phase-0 Sanity Check Result**: Negative Sharpe (-0.69) across all assets (2020-2025)
  - **Findings**: Sign-only roll yield strategy showed negative alpha in recent years
  - **Status**: Remains on roadmap for redesign (e.g., sector-based roll yield, DV01-neutral carry, regime-dependent filters)
  - **Atomic Sleeves**: FX/Commodity Carry (CL, GC, 6E, 6B, 6J), SR3 Carry/Curve

- ❌ **Rates Curve RV Meta-Sleeve**: **Parked for redesign**
  - **Phase-0 Sanity Check Result**: Near-zero Sharpe (0.002) with negative 2s10s leg (-0.20 Sharpe)
  - **Findings**: Initial slope-based approach fragile to post-2022 regime changes (Fed hiking cycle)
  - **Status**: Will return with DV01-neutral flies, pack spreads, macro gating, and improved yield reconstruction
  - **Atomic Sleeves**: 2s10s flattener/steepener, 5s30s flattener/steepener

- ❌ **Volatility Risk Premium Meta-Sleeve**: **Planned** (not yet implemented)
  - Will include multiple atomic sleeves for various volatility selling strategies
  - Different implementation approaches and feature sets

- ❌ **Value Meta-Sleeve**: **Planned** (not yet implemented)

- ❌ **Seasonality/Flows Meta-Sleeve**: **Future** (not yet implemented)

- ❌ **Macro Overlay**: Disabled (too aggressive with only 1 meta-sleeve; will revisit when more meta-sleeves are added)

### Baseline Performance

**Trend Meta-Sleeve Only** (`core_v3_no_macro` - with canonical Long-Term and canonical Medium-Term):
- **Production Profile**: Uses canonical Long-Term (1/3, 1/3, 1/3) and canonical Medium-Term (1/3, 1/3, 1/3) as of Nov 2025
- **Phase-2 Results** (canonical vs legacy baseline, 2020-01-02 to 2025-10-31):
  - **CAGR**: +6.83% (vs legacy -2.59%, +9.42% improvement)
  - **Sharpe**: +0.59 (vs legacy -0.15, +0.74 improvement)
  - **MaxDD**: -16.72% (vs legacy -39.69%, +22.97% improvement)
  - **HitRate**: 50.55% (vs legacy 49.77%, +0.78% improvement)

**Legacy Baseline** (`core_v3_no_macro_baseline` - with legacy medium-term, 2021-01-01 to 2025-10-31):
- **CAGR**: -2.59%
- **Sharpe**: -0.15
- **MaxDD**: -39.69%
- **Vol**: 12.31%
- **HitRate**: 49.77%

**Note**: Performance reflects the challenging 2022-2025 period for trend-following strategies. The canonical medium-term sleeve shows significant improvement over the legacy version. The Trend Meta-Sleeve architecture is validated and production-ready; performance will improve as additional meta-sleeves are added.

### Strategy Profile
Use `--strategy_profile core_v3_no_macro` in `run_strategy.py` to run the baseline configuration.

## Core v3: Trend Meta-Sleeve (Phase 4)

**Production Strategy:** `core_v3_no_macro`

Phase 4 introduces the **Trend Meta-Sleeve** architecture, which combines multiple atomic trend sleeves (long-term, medium-term, short-term momentum, and breakout) into a unified meta-signal using industry-standard horizon weights. The production profile (`core_v3_no_macro`) currently uses 4 active atomic sleeves with canonical long-term and canonical medium-term (promoted Nov 2025). This represents the current production baseline.

### Phase 4a: Trend Meta-Sleeve Implementation

**Strategy Profile:** `core_v3_no_macro`

**Trend Meta-Sleeve Architecture (`core_v3_no_macro` Production Profile):**

The `core_v3_no_macro` profile uses **TSMOMMultiHorizon (v2)** with **4 active atomic sleeves**:

1. **Long-term (252d) — Canonical** (48.5% horizon weight)
   - Equal-weight (1/3, 1/3, 1/3) of 252d return, 252d breakout, slow trend slope
   - Promoted Nov 2025

2. **Medium-term (84d) — Canonical** (29.1% horizon weight)
   - Equal-weight (1/3, 1/3, 1/3) of 84d return (skip 10d, vol-scaled), 84d breakout, EMA21-84 slope
   - Promoted Nov 2025

3. **Short-term (21d) — Legacy** (19.4% horizon weight)
   - 21-day return, 21-day breakout, fast trend slope (EMA_10 - EMA_40), reversal filter (optional)
   - Feature weights: 0.5 ret_21 + 0.3 breakout_21 + 0.2 slope_fast (legacy, for comparison)
   - **Canonical variant (Phase-0/1/2)**: Equal-weight (1/3, 1/3, 1/3) composite; reversal filter weight 0

4. **Breakout Mid (50-100d)** (3.0% horizon weight)
   - 70/30 blend of 50d and 100d breakout strength
   - Phase-3 production

**Additional Validated Sleeves (Not Currently Active):**
- **Residual Trend (252d-21d)**: Long-horizon trend minus short-term movement, cross-sectionally z-scored
  - Phase-0/1/2 passed, production-ready but not currently enabled in `core_v3_no_macro` profile
  - Design intent: 4th atomic trend sleeve for internal integration

**Deprecated:**
- **Medium-term (84/126d) — Legacy**: 84-day return, 126-day breakout, medium trend slope (EMA_20 - EMA_84), persistence
  - Deprecated Nov 2025, preserved in `core_v3_no_macro_legacy` for historical comparison
- ❌ **Persistence (Momentum-of-Momentum)**: **PARKED** (Phase-1 failed)
  - Phase-0: Only slope acceleration variant passed minimum Sharpe (0.22)
  - Phase-1: Engineered sleeve did not improve Trend Meta-Sleeve (Sharpe declined from 0.097 to 0.087)
  - Status: Not integrated. Re-test when universe expands (≥25%) or historical sample increases (≥5 years)

**Horizon Weights (Frozen for Architecture Build-Out):**
- Long (252d): **0.45** (FIXED)
- Medium (84/126d): **0.28** (FIXED)
- Short (21d): **0.20** (FIXED)
- Residual (252d-21d): **0.15** (FIXED)

**⚠️ Weight Freeze Policy:**
These internal weights (45/28/20/15) are **frozen during the architecture build-out phase**. Do not sweep or optimize these weights until Phase-B (Optimization & Pruning), where they must be optimized jointly across all sleeves with proper out-of-sample controls. See `PROCEDURES.md` for details.

**Feature Weights (Within Each Horizon):**
- **Long**: ret_252=0.5, breakout_252=0.3, slope_slow=0.2
- **Medium**: ret_84=0.4, breakout_126=0.3, slope_med=0.2, persistence=0.1
- **Short**: ret_21=0.5, breakout_21=0.3, slope_fast=0.2, reversal=0.0

**Signal Processing:**
1. Combine features within each horizon using feature weights
2. Blend horizons using horizon weights: `tsmom_raw = 0.50 * long_signal + 0.30 * med_signal + 0.20 * short_signal`
3. Cross-sectional z-score across assets
4. Clip to ±3.0 standard deviations

**Performance (2021-01-01 to 2025-10-31, without EWMA):**
- **CAGR**: -2.41%
- **Sharpe**: -0.08
- **MaxDD**: -43.80%
- **Vol**: 15.37%
- **HitRate**: 49.57%

### Phase 4b: EWMA Volatility Normalization

**Strategy Profile:** `core_v2_no_macro` (with `vol_normalization.enabled: true`)

**Enhancement:** Added EWMA volatility normalization to the TSMOMMultiHorizon sleeve to ensure:
- Same signal magnitude implies same risk across assets
- Exposure stops jumping wildly when vol spikes/drops
- More stable risk-adjusted signals

**EWMA Vol Calculation:**
- **Half-life**: 63 trading days (standard CTA choice)
- **Formula**: `σ²_t = λ * σ²_{t-1} + (1-λ) * r²_{t-1}` where `λ = 0.5^(1/63)`
- **Annualization**: `σ_annual = σ_daily * sqrt(252)`
- **Floor**: 5% annualized minimum to prevent extreme scaling

**Vol Normalization:**
- After cross-sectional z-scoring and clipping: `s_risk = z_clipped / max(σ_annual, σ_floor)`
- Apply global scale factor: `s_final = risk_scale * s_risk` (default: risk_scale=0.2)
- Final vol targeting overlay still applied on top (unchanged)

**Configuration:**
```yaml
vol_normalization:
  enabled: true
  halflife_days: 63
  sigma_floor_annual: 0.05
  risk_scale: 0.2
```

**Performance (2021-01-01 to 2025-10-31, with EWMA):**
- **CAGR**: -2.59%
- **Sharpe**: -0.15
- **MaxDD**: -39.69% (improved from -43.80%)
- **Vol**: 12.31% (reduced from 15.37%)
- **HitRate**: 49.77%

**Key Improvement:** MaxDD improved by 4.11% and volatility reduced by 3.06%, demonstrating more stable risk-adjusted signals.

### Strategy Profiles
- **`core_v1_no_macro`**: Long-term TSMOM + FX/Commodity Carry (legacy baseline, deprecated)
- **`core_v2_no_macro`**: Multi-Horizon TSMOM (v2) with EWMA normalization (intermediate version, deprecated)
- **`core_v3_no_macro`**: Trend Meta-Sleeve only (current production baseline)

## Overview

The strategy implements a **two-layer sleeve architecture** with Meta-Sleeves representing economic sources of return and Atomic Sleeves representing implementation variants.

**Note:** The strategy uses a **dual-price architecture** (raw vs continuous prices) for all calculations. See `docs/DUAL_PRICE_ARCHITECTURE.md` for details on how prices are back-adjusted and which modules use which price source.

### Current Production Architecture

**Active Meta-Sleeves:**
- **Trend Meta-Sleeve**: Multi-horizon time-series momentum (3 atomic sleeves: long-term, medium-term, short-term)
- **Cross-Sectional Momentum (CSMOM) Meta-Sleeve**: Relative momentum across assets (Phase-2 integrated)

**Disabled/Parked Meta-Sleeves:**
- **Carry Meta-Sleeve**: Parked (Phase-0 failed; negative Sharpe in recent years)
- **Rates Curve RV Meta-Sleeve**: Parked (Phase-0 failed; fragile to regime changes)
- **Persistence Atomic Sleeve (Trend Meta-Sleeve)**: Parked (Phase-1 failed; did not improve Trend Meta-Sleeve performance)
- **Volatility Risk Premium Meta-Sleeve**: Planned (not yet implemented; will include multiple atomic sleeves)
- **Value Meta-Sleeve**: Planned (not yet implemented)
- **Seasonality/Flows Meta-Sleeve**: Future (not yet implemented)

### System Components

- **13 futures contracts** across equities, rates, commodities, and FX
- **Regime-based signal scaling** (macro overlay) - operates at Meta-Sleeve layer
- **Volatility targeting** - operates at Meta-Sleeve layer
- **Portfolio optimization** with constraints - operates at Meta-Sleeve layer
- **Weekly rebalancing** on Fridays

## Universe

**13 Continuous Futures Contracts:**

- **Equities (3)**: ES, NQ, RTY (calendar roll, 2 days before expiry)
- **Rates (5)**: ZN, ZF, ZT, UB (volume-weighted roll), SR3 (SOFR, calendar roll, T-2)
- **Commodities (2)**: CL, GC (volume-weighted roll)
- **FX (3)**: 6E, 6B, 6J (calendar roll, 2 days before expiry)

**Universe Configuration** (`configs/data.yaml`):
- Dictionary format with roll settings per contract
- Calendar roll: rolls 2 business days before expiry
- Volume roll: rolls when volume shifts to next contract
- SOFR (SR3): Calendar roll with `roll_offset_bdays: -2` (IMM convention, roll T-2)
- **Symbol Naming Note**: Equities (ES, NQ, RTY) use `*_FRONT_CALENDAR_2D` suffix in database. FX (6E, 6B, 6J) and SR3 use `*_FRONT_CALENDAR` (no `_2D` suffix) in database. The `MarketData` broker automatically maps universe config to the correct database symbol names.

## Complete Execution Flow

### Initialization Phase

**1. MarketData Broker**
- Connect to DuckDB database (read-only)
- Load universe from `configs/data.yaml`
- Auto-detect OHLCV table schema
- Validate data availability
- **Build continuous prices**: Load raw prices with `contract_id`, build back-adjusted continuous prices using `ContinuousContractBuilder` (backward-panama adjustment)
- **Expose dual prices**: `prices_raw` (raw DB prices), `prices_cont` (back-adjusted), `contract_ids`, `returns_cont` (log returns from continuous prices)

**2. FeatureService**
- Initialize feature computation service
- Configure SR3 curve features (12 ranks, 252-day window)
- Configure Rates curve features (FRED-anchored yields, 252-day window, anchor_lag_days=2)
- Configure FX/Commodity carry features (CL, GC, 6E, 6B, 6J; 3 features per root: time-series, cross-sectional, momentum)
- Configure Long-term momentum features (252-day return, breakout, slow trend slope; 252-day window)
- Configure Medium-term momentum features (84-day return, 126-day breakout, medium trend slope, persistence; 252-day window)
- Configure Short-term momentum features (21-day return, breakout, fast trend slope, reversal filter; 252-day window)
- Pre-compute features if feature-based sleeves are enabled

**3. Strategy Sleeves**
- **TSMOM Strategy (Long-Term Momentum)** (if enabled in config):
  - Uses three long-term momentum features: 252-day return momentum, 252-day breakout strength, slow trend slope
  - Combines features with configurable weights (default: ret_252=0.5, breakout_252=0.3, slope_slow=0.2)
  - Uses pre-computed LONG_MOMENTUM features from FeatureService
  - Cross-sectional z-scoring and clipping at ±3.0
  - Returns: `pd.Series` with signals for all symbols
- **TSMOMMultiHorizon Strategy (v2)** (if enabled in config):
  - **Unified multi-horizon momentum sleeve** combining long, medium, and short-term features
  - **Long-term (252d)**: 252-day return, 252-day breakout, slow trend slope (EMA_63 - EMA_252)
  - **Medium-term (84/126d)**: 84-day return, 126-day breakout, medium trend slope (EMA_20 - EMA_84), persistence
  - **Short-term (21d)**: 21-day return, 21-day breakout, fast trend slope (EMA_10 - EMA_40), reversal filter (optional)
  - **Horizon weights**: long_252=0.50, med_84=0.30, short_21=0.20 (industry-standard)
  - **Feature weights**: Same as individual sleeves within each horizon
  - **Signal processing**: Combine features → blend horizons → cross-sectional z-score → clip ±3.0
  - **EWMA Vol Normalization** (if enabled): Risk-normalize signals by dividing by EWMA vol (63-day half-life, 5% floor)
  - Uses pre-computed LONG_MOMENTUM, MEDIUM_MOMENTUM, SHORT_MOMENTUM features from FeatureService
  - Returns: `pd.Series` with risk-normalized signals for all symbols
- **Medium-Term Momentum Strategy** (if enabled in config):
  - Uses four medium-term momentum features: 84-day return momentum, 126-day breakout strength, medium trend slope, persistence
  - Combines features with configurable weights (default: ret_84=0.4, breakout_126=0.3, slope_med=0.2, persistence=0.1)
  - Uses pre-computed MEDIUM_MOMENTUM features from FeatureService
  - Cross-sectional z-scoring and clipping at ±3.0
  - Returns: `pd.Series` with signals for all symbols
- **Short-Term Momentum Strategy** (if enabled in config):
  - Uses four short-term momentum features: 21-day return momentum, 21-day breakout strength, fast trend slope, reversal filter
  - Combines features with configurable weights (default: ret_21=0.5, breakout_21=0.3, slope_fast=0.2, reversal_filter=0.0)
  - Uses pre-computed SHORT_MOMENTUM features from FeatureService
  - Cross-sectional z-scoring and clipping at ±3.0
  - Returns: `pd.Series` with signals for all symbols
- **SR3 Carry/Curve Strategy** (if enabled in config):
  - Initialize with feature weights (w_carry=0.30, w_curve=0.25, w_pack_slope=0.20, w_front_lvl=0.10, w_curv_belly=0.15)
  - Signal cap: ±3.0 standard deviations
  - Uses pre-computed SR3 curve features from FeatureService (5 features total)
- **Rates Curve Strategy** (if enabled in config):
  - Initialize with feature weights (w_slope_2s10s=0.35, w_slope_5s30s=0.35, w_curv_2s5s10s=0.15, w_curv_5s10s30s=0.15)
  - Signal cap: ±3.0 standard deviations
  - Uses pre-computed Rates curve features from FeatureService (4 features total)
  - Generates flattener/steepener signals for ZT, ZF, ZN, UB
- **FX/Commodity Carry Strategy** (if enabled in config):
  - Initialize with feature weights (w_ts=0.6, w_xs=0.25, w_mom=0.15)
  - Uses roll yield between front and next contracts for CL, GC, 6E, 6B, 6J
  - Defines roll yield in log price space: `roll_yield_raw = -(ln(F1) - ln(F0))`
  - Uses pre-computed FX/Commodity carry features from FeatureService (3 features per root)
  - Returns: pd.Series with signals for CL, GC, 6E, 6B, 6J

**4. CombinedStrategy**
- Combine strategy sleeves with configurable weights
- **Weight Normalization**: Weights are **relative importance weights**, not absolute risk budgets
  - Automatically normalized to sum to 1.0: `normalized_weight_i = weight_i / sum(all_weights)`
  - Example: If config has `tsmom=0.6, sr3=0.15, rates=0.15, fx_commod=0.10`, they become `tsmom=0.60, sr3=0.15, rates=0.15, fx_commod=0.10` (already normalized)
  - Alternative: If config has `tsmom=1.0, sr3=0.3, rates=0.3, fx_commod=0.2`, they become `tsmom=0.556, sr3=0.167, rates=0.167, fx_commod=0.111`
  - This ensures signals are combined proportionally regardless of the absolute weight values
- Handle strategies that require features (e.g., TSMOM, Medium/Short Momentum, SR3, Rates Curve, FX/Commodity Carry)

**5. MacroRegimeFilter**
- Initialize with vol thresholds, breadth lookback, FRED indicators
- Configure FRED indicators from `configs/fred_series.yaml` (macro FRED series):
  - Daily: VIXCLS, VXVCLS, FEDFUNDS, DGS2, DGS10, BAMLH0A0HYM2, TEDRATE, DTWEXBGS
  - Monthly: CPIAUCSL, UNRATE (dailyized before z-scoring)
  - **Note**: DGS2 and DGS10 are shared with Rates Curve features; DGS5 and DGS30 are curve-only
- Set scaler bounds [0.5, 1.0] and smoothing (EMA α=0.15)
- Input smoothing: 5-day EMA on breadth and FRED composite to avoid stepwise jumps

**6. RiskVol Agent**
- Configure volatility lookback (63 days) and covariance lookback (252 days)
- Set Ledoit-Wolf shrinkage for covariance stability
- Apply min vol floor (50 bps annualized) to prevent exploding leverage in calm regimes

**7. VolManagedOverlay**
- Set target volatility (20% annualized)
- Configure leverage cap (7x) and position bounds (±3.0)

**8. Allocator**
- Configure method (signal-beta), gross/net caps, turnover penalty
- Set per-asset bounds (±1.5)

**9. ExecSim**
- Set rebalance frequency (W-FRI), slippage (0.5 bps), commission (0.0)
- Friday holiday handling: if Friday is not a trading day, rebalance on previous business day

---

### Per-Rebalance Loop (Weekly Fridays)

For each rebalance date `t`, the following steps execute **in this exact order**:

#### Step 1: Generate Combined Strategy Signals

```python
signals = combined_strategy.signals(market, date=t)
```

**What happens:**

**1a. TSMOM Signals (Long-Term Momentum)** (if enabled)
- Get pre-computed long-term momentum features up to date `t`:
  - `mom_long_ret_252_z_{symbol}`: 252-day return momentum (vol-standardized, z-scored)
  - `mom_long_breakout_252_z_{symbol}`: 252-day breakout strength (normalized position in 252-day range)
  - `mom_long_slope_slow_z_{symbol}`: Slow trend slope (EMA_63 - EMA_252, vol-standardized)
- Combine features with configurable weights: `signal = w_ret * ret_252_z + w_breakout * breakout_252_z + w_slope * slope_slow_z`
- Cross-sectional z-scoring across assets and clipping at ±3.0
- Returns: `pd.Series` with signals for all symbols
- **Note**: If features unavailable for date `t`, uses most recent available features (forward-fill)

**1a2. Trend Meta-Sleeve Signals** (if enabled)
- **Atomic Sleeve 1: Long-term momentum (252d)**
  - Get pre-computed long-term momentum features: `mom_long_ret_252_z_{symbol}`, `mom_long_breakout_252_z_{symbol}`, `mom_long_slope_slow_z_{symbol}`
  - Combine features: `long_signal = 0.5 * ret_252 + 0.3 * breakout_252 + 0.2 * slope_slow`
- **Atomic Sleeve 2: Medium-term momentum (84/126d)**
  - Get pre-computed medium-term momentum features: `mom_med_ret_84_z_{symbol}`, `mom_med_breakout_126_z_{symbol}`, `mom_med_slope_med_z_{symbol}`, `mom_med_persistence_z_{symbol}`
  - Combine features: `med_signal = 0.4 * ret_84 + 0.3 * breakout_126 + 0.2 * slope_med + 0.1 * persistence`
- **Atomic Sleeve 3: Short-term momentum (21d)**
  - Get pre-computed short-term momentum features: `mom_short_ret_21_z_{symbol}`, `mom_short_breakout_21_z_{symbol}`, `mom_short_slope_fast_z_{symbol}`, `mom_short_reversal_filter_z_{symbol}`
  - Combine features: `short_signal = 0.5 * ret_21 + 0.3 * breakout_21 + 0.2 * slope_fast`
- **Meta-Sleeve Combination**: Blend atomic sleeves using horizon weights
  - `trend_meta_signal = 0.50 * long_signal + 0.30 * med_signal + 0.20 * short_signal`
- **Cross-sectional z-score**: Normalize across assets and clip to ±3.0
- **EWMA Vol Normalization** (if enabled):
  - Compute EWMA annualized volatility for each asset: `σ_annual = EWMA(returns², halflife=63) * sqrt(252)`
  - Risk-normalize: `s_risk = z_clipped / max(σ_annual, σ_floor)` where `σ_floor = 0.05` (5%)
  - Apply global scale: `s_final = risk_scale * s_risk` (default: `risk_scale = 0.2`)
- Returns: `pd.Series` with risk-normalized Trend Meta-Sleeve signals for all symbols
- **Note**: If features unavailable for date `t`, uses most recent available features (forward-fill)

**1b. SR3 Carry/Curve Signals** (if enabled)
- Get pre-computed SR3 curve features up to date `t`:
  - `sr3_carry_01_z`: Carver carry (r1 - r0) standardized
  - `sr3_curve_02_z`: Curve shape (r2 - r0) standardized
  - `sr3_pack_slope_fb_z`: Pack slope (front vs back) standardized
  - `sr3_front_pack_level_z`: Front-pack level (policy expectation) standardized
  - `sr3_curvature_belly_z`: Belly curvature (hump vs straight) standardized
- Compute weighted combination: `signal = w_carry * carry + w_curve * curve + w_pack_slope * pack_slope + w_front_lvl * front_lvl + w_curv_belly * curv_belly`
- Default weights: w_carry=0.30, w_curve=0.25, w_pack_slope=0.20, w_front_lvl=0.10, w_curv_belly=0.15
- Cap at ±3.0 standard deviations
- Returns: `pd.Series` with signal for SR3_FRONT_CALENDAR only
- **Note**: If features unavailable for date `t`, uses most recent available features (forward-fill)

**1c. Rates Curve Signals** (if enabled)
- Get pre-computed Rates curve features up to date `t`:
  - `curve_2s10s_z`: 2s10s curve slope standardized (ZT vs ZN)
  - `curve_5s30s_z`: 5s30s curve slope standardized (ZF vs UB)
  - `curve_2s5s10s_curv_z`: 2s-5s-10s belly curvature standardized
  - `curve_5s10s30s_curv_z`: 5s-10s-30s belly curvature standardized
- For 2s10s: Combine slope and curvature: `signal_2s10s = w_slope_2s10s * slope_z + w_curv_2s5s10s * curv_z`
  - Default weights: w_slope_2s10s=0.35, w_curv_2s5s10s=0.15 (normalized within 2s10s segment to sum to 1.0)
  - After normalization: w_slope_2s10s=0.70, w_curv_2s5s10s=0.30
  - Positive signal → flattener (long ZT, short ZN); Negative signal → steepener (short ZT, long ZN)
  - Final signals: `s_2y = +signal_2s10s`, `s_10y = -signal_2s10s`
- For 5s30s: Combine slope and curvature: `signal_5s30s = w_slope_5s30s * slope_z + w_curv_5s10s30s * curv_z`
  - Default weights: w_slope_5s30s=0.35, w_curv_5s10s30s=0.15 (normalized within 5s30s segment to sum to 1.0)
  - After normalization: w_slope_5s30s=0.70, w_curv_5s10s30s=0.30
  - Positive signal → flattener (long ZF, short UB); Negative signal → steepener (short ZF, long UB)
  - Final signals: `s_5y = +signal_5s30s`, `s_30y = -signal_5s30s`
- Cap combined signals at ±3.0 standard deviations
- Returns: `pd.Series` with signals for ZT, ZF, ZN, UB
- **Note**: If features unavailable for date `t`, uses most recent available features (forward-fill)

**1d. Medium-Term Momentum Signals** (if enabled)
- Get pre-computed medium-term momentum features up to date `t`:
  - `mom_med_ret_84_z_{symbol}`: 84-day return momentum (vol-standardized, z-scored)
  - `mom_med_breakout_126_z_{symbol}`: 126-day breakout strength (normalized position in 126-day range)
  - `mom_med_slope_med_z_{symbol}`: Medium trend slope (EMA_20 - EMA_84, vol-standardized)
  - `mom_med_persistence_z_{symbol}`: Trend persistence (sign consistency over 20 days)
- Combine features with configurable weights: `signal = w_ret * ret_84_z + w_breakout * breakout_126_z + w_slope * slope_med_z + w_persist * persistence_z`
- Cross-sectional z-scoring across assets and clipping at ±3.0
- Returns: `pd.Series` with signals for all symbols
- **Note**: If features unavailable for date `t`, uses most recent available features (forward-fill)

**1e. Short-Term Momentum Signals** (if enabled)
- Get pre-computed short-term momentum features up to date `t`:
  - `mom_short_ret_21_z_{symbol}`: 21-day return momentum (vol-standardized, z-scored)
  - `mom_short_breakout_21_z_{symbol}`: 21-day breakout strength (normalized position in 21-day range)
  - `mom_short_slope_fast_z_{symbol}`: Fast trend slope (EMA_10 - EMA_40, vol-standardized)
  - `mom_short_reversal_filter_z_{symbol}`: Reversal filter (RSI-like, optional, not used by default)
- Combine features with configurable weights: `signal = w_ret * ret_21_z + w_breakout * breakout_21_z + w_slope * slope_fast_z`
- Cross-sectional z-scoring across assets and clipping at ±3.0
- Returns: `pd.Series` with signals for all symbols
- **Note**: If features unavailable for date `t`, uses most recent available features (forward-fill)

**1f. FX/Commodity Carry Signals** (if enabled)
- Get pre-computed FX/Commodity carry features up to date `t` (3 features per root):
  - `carry_ts_z_<root>`: Time-series roll yield standardized per root
  - `carry_xs_z_<root>`: Cross-sectional carry strength (relative to other assets on same day)
  - `carry_mom_63_z_<root>`: 63-day carry momentum (change in roll yield over 3 months)
- For each asset: Combine features with weights: `signal = w_ts * carry_ts_z + w_xs * carry_xs_z + w_mom * carry_mom_63_z`
  - Default weights: w_ts=0.6, w_xs=0.25, w_mom=0.15 (normalized to sum to 1.0)
  - Positive signal → long (backwardation, attractive carry); Negative signal → short (contango, negative carry)
- Cap combined signals at ±3.0 standard deviations
- Returns: `pd.Series` with signals for CL, GC, 6E, 6B, 6J
- **Note**: If features unavailable for date `t`, uses most recent available features (forward-fill)

**1g. Combine Signals**
- Weighted combination: `combined = weight_tsmom * tsmom_signals + weight_tsmom_multihorizon * tsmom_multihorizon_signals + weight_tsmom_med * tsmom_med_signals + weight_tsmom_short * tsmom_short_signals + weight_sr3 * sr3_signals + weight_rates * rates_signals + weight_fx_commod * fx_commod_signals`
- **Weight Normalization**: Weights from `configs/strategies.yaml` are **relative importance weights**
  - Automatically normalized: `normalized_weight = weight / sum(all_weights)`
  - Example: Config `{tsmom: 0.6, sr3: 0.15, rates: 0.15, fx_commod: 0.10}` → Normalized `{tsmom: 0.60, sr3: 0.15, rates: 0.15, fx_commod: 0.10}` (already sums to 1.0)
  - Alternative example: Config `{tsmom_multihorizon: 1.0}` → Normalized `{tsmom_multihorizon: 1.0}` (single sleeve)
  - This means you can use any scale and get the same relative proportions after normalization
- **Note**: When using `tsmom_multihorizon`, typically disable `tsmom`, `tsmom_med`, and `tsmom_short` to avoid double-counting momentum features
- Returns: `pd.Series` with combined signals for all symbols

**Example output:**
```
ES_FRONT_CALENDAR_2D     1.25  (from TSMOM long-term)
NQ_FRONT_CALENDAR_2D    -0.87  (from TSMOM long-term)
RTY_FRONT_CALENDAR_2D    0.45  (from TSMOM long-term + Medium-term)
ZN_FRONT_VOLUME          0.42  (from TSMOM long-term + Rates Curve)
ZT_FRONT_VOLUME          0.18  (from Rates Curve 2s10s)
ZF_FRONT_VOLUME         -0.12  (from Rates Curve 5s30s)
UB_FRONT_VOLUME          0.12  (from Rates Curve 5s30s)
SR3_FRONT_CALENDAR       0.15  (from SR3 carry/curve, weighted)
CL_FRONT_VOLUME         -0.28  (from FX/Commodity Carry)
GC_FRONT_VOLUME          0.19  (from FX/Commodity Carry)
6E_FRONT_CALENDAR        0.55  (from FX/Commodity Carry)
6B_FRONT_CALENDAR       -0.12  (from FX/Commodity Carry)
6J_FRONT_CALENDAR        0.24  (from FX/Commodity Carry)
...
```

**Reference:** 
- `src/agents/strat_combined.py` (CombinedStrategy)
- `src/agents/strat_momentum.py` (TSMOM - Long-Term Momentum)
- `src/agents/strat_tsmom_multihorizon.py` (TSMOMMultiHorizon - v2, unified multi-horizon momentum)
- `src/agents/strat_momentum_medium.py` (Medium-Term Momentum)
- `src/agents/strat_momentum_short.py` (Short-Term Momentum)
- `src/agents/strat_sr3_carry_curve.py` (SR3 Carry/Curve)
- `src/agents/strat_rates_curve.py` (Rates Curve)
- `src/agents/strat_carry_fx_commod.py` (FX/Commodity Carry)
- `src/agents/feature_long_momentum.py` (Long/Medium/Short Momentum Features)
- `src/agents/feature_sr3_curve.py` (SR3 Features)
- `src/agents/feature_rates_curve.py` (Rates Curve Features)
- `src/agents/feature_carry_fx_commod.py` (FX/Commodity Carry Features)
- `docs/legacy/TSMOM_IMPLEMENTATION.md`
- `docs/SR3_CARRY_CURVE.md`

---

#### Step 2: Compute Macro Regime Scaler

```python
macro_k = macro_overlay.scaler(market, date=t)
```

**What happens:**

**2a. Realized Volatility**
- Get ES and NQ returns over last 21 days
- Compute equal-weighted portfolio returns
- Calculate annualized volatility: `vol = std(returns) * sqrt(252)`

**2b. Market Breadth**
- For ES and NQ:
  - Get 200-day SMA as of date `t`
  - Check if price > SMA
- Breadth = fraction of symbols above SMA (0.0, 0.5, or 1.0)

**2c. FRED Economic Indicators**
- Query `f_fred_observations` table for indicators up to date `t` (configurable via `configs/fred_series.yaml`)
- For monthly series (CPI, UNRATE):
  - Forward-fill to business days before z-scoring
  - Resample to daily frequency with `asfreq('B', method='pad')`
- For each indicator:
  - Get last 252 days (or available data, after dailyization)
  - Compute rolling z-score: `z = (value - mean) / std` (63-day window after dailyization)
  - Cap z-score at ±5.0 to avoid single prints swinging the scaler
  - Map to [-1, 1] using `tanh(z / 2.0)`
- Combine all normalized indicators with equal weights
- Result: Combined FRED signal in [-1, 1]
- Data freshness check: warn if monthly series stale > 45 days

**2d. Input Smoothing**
- Apply 5-day EMA to breadth and FRED composite inputs
- Prevents stepwise k-jumps when monthly FRED data updates
- Formula: `smoothed = 0.2 * new_value + 0.8 * previous_smoothed`

**2e. Base Scaler Calculation**
- Map vol linearly: `[vol_low, vol_high] → [k_max, k_min]`
  - Higher vol → lower scaler (risk-off)
- Add breadth adjustment (using smoothed breadth):
  - Breadth = 1.0 → +0.1 (bullish)
  - Breadth = 0.0 → -0.1 (bearish)
  - Breadth = 0.5 → 0.0 (neutral)
- Add FRED adjustment (using smoothed FRED signal): `fred_signal * fred_weight` (default: 0.3)
- Clamp to [k_min, k_max] = [0.5, 1.0]
- Apply EMA smoothing to final scaler: `k_smooth = 0.15 * k_new + 0.85 * k_prev`

**Returns:** Scaler `k` ∈ [0.5, 1.0]

**Example:** `macro_k = 0.837` (16.3% risk reduction)

**Reference:** `src/agents/overlay_macro_regime.py`, `docs/MACRO_REGIME_FILTER.md`

---

#### Step 3: Apply Volatility Targeting

```python
scaled_signals = vol_overlay.scale(signals, market, date=t)
```

**What happens:**
- Get current portfolio volatility (63-day lookback)
- Calculate target leverage: `leverage = target_vol / current_vol`
- Scale all signals: `scaled = signals * leverage`
- Apply leverage cap (7x) and position bounds (±3.0)
- **Then apply macro scaler:** `scaled_signals = scaled_signals * macro_k`

**Returns:** Vol-targeted and regime-scaled signals

**Example:**
```
ES_FRONT_CALENDAR_2D     0.95  (was 1.25, scaled by vol + macro)
NQ_FRONT_CALENDAR_2D    -0.66  (was -0.87)
...
```

**Reference:** `src/agents/overlay_volmanaged.py`

---

#### Step 4: Get Risk Data

```python
cov = risk_vol.covariance(market, date=t)
vols = risk_vol.vols(market, date=t)
mask = risk_vol.mask(market, date=t)
```

**What happens:**
- Compute covariance matrix (252-day lookback, Ledoit-Wolf shrinkage)
- Apply min vol floor: ensure all diagonal elements ≥ (50 bps)² to prevent exploding leverage
- Compute individual volatilities (63-day lookback)
- Create mask of valid symbols (have sufficient data)

**Returns:** Covariance matrix, volatility series, validity mask

**Reference:** `src/agents/risk_vol.py`

---

#### Step 5: Apply Data Validity Mask

```python
mask = risk_vol.mask(market, date=t)
valid_symbols = mask.intersection(scaled_signals.index)
scaled_signals.loc[invalid_symbols] = 0.0  # Zero out invalid signals
```

**What happens:**
- Get validity mask from RiskVol (symbols with sufficient data)
- Zero out signals for assets failing validity mask
- Prevents allocator from wasting budget on NaNs that got imputed

**Returns:** Masked signals (invalid assets set to 0.0)

---

#### Step 6: Optimize Portfolio Weights

```python
weights = allocator.solve(scaled_signals, cov, weights_prev=prev_weights)
```

**What happens:**
- Use signal-beta method: align weights with signal direction
- Enforce constraints:
  - Gross leverage cap: `sum(abs(weights)) <= 7.0`
  - Net leverage cap: `sum(weights) <= 2.0`
  - Per-asset bounds: `-1.5 <= weight <= 1.5`
  - Turnover penalty: minimize `lambda * turnover`
- If constraints violated, use L2 projection to find nearest feasible solution

**Returns:** Final portfolio weights `pd.Series`

**Example:**
```
ES_FRONT_CALENDAR_2D     0.1299  (12.99%)
NQ_FRONT_CALENDAR_2D     0.1415  (14.15%)
ZN_FRONT_VOLUME          0.1562  (15.62%)
...
```

**Reference:** `src/agents/allocator.py`

---

#### Step 7: Calculate Portfolio Return

```python
# Get returns for holding period (t to t+1)
next_date = rebalance_dates[i+1] if i < len(rebalance_dates)-1 else end_date
period_returns = market.get_returns(start=t+1, end=next_date)  # Daily returns

# For log returns: sum is correct (additive)
# For simple returns: would use prod() to compound
holding_returns = period_returns.sum()  # Cumulative period return per asset

# Portfolio return (weights are fixed over [t, next_t))
port_ret = (weights * holding_returns).sum()

# Apply transaction costs (slippage on turnover)
turnover = sum(abs(weights - prev_weights)) if prev_weights is not None else sum(abs(weights))
slippage_cost = turnover * slippage_bps / 10000
net_ret = port_ret - slippage_cost
```

**What happens:**
- Get asset returns from date `t+1` to next rebalance (or end)
- Compute cumulative period return per asset: `sum(daily_returns)` for log returns
- Compute portfolio return: `Σ(weight_i * period_return_i)` where weights are fixed over the holding period
- Calculate turnover: `sum(abs(new_weights - old_weights))`
- Apply slippage: `slippage = turnover * 0.5 bps`
- Net return: `portfolio_return - slippage`

**Returns:** Portfolio return for the holding period

**Reference:** `src/agents/exec_sim.py` (lines 393-417)

---

#### Step 8: Update Equity Curve & Diagnostics

```python
equity[t+1] = equity[t] * (1 + net_ret)

# Diagnostics: what-moved report
if prev_weights is not None:
    weight_changes = (weights - prev_weights).abs().sort_values(ascending=False)
    top_movers = weight_changes.head(5)
    logger.info(f"top_movers={dict(top_movers.head(3))}, k={macro_k:.3f}, turnover={turnover:.3f}")
```

**What happens:**
- Compound returns: `equity_new = equity_old * (1 + return)`
- Track cumulative performance
- Log diagnostics: top weight changes, macro scaler (k), turnover

**Returns:** Updated equity value

---

### Post-Processing Phase

After all rebalance dates complete:

**1. Compute Performance Metrics**
- CAGR: `(equity_end / equity_start) ^ (1/years) - 1`
- Volatility: `std(daily_returns) * sqrt(252)`
- Sharpe: `mean(daily_returns) / std(daily_returns) * sqrt(252)`
- Max Drawdown: Maximum peak-to-trough decline
- Hit Rate: Fraction of positive return periods
- Turnover: Average turnover per rebalance
- Gross/Net Leverage: Average portfolio leverage

**2. Generate Outputs**
- Equity curve: `pd.Series` indexed by date
- Weights panel: `pd.DataFrame` [date × symbol]
- Signals panel: `pd.DataFrame` [date × symbol]
- Performance report: `dict` with all metrics

**Reference:** `src/agents/exec_sim.py` (lines 382-450)

---

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    INITIALIZATION                            │
├─────────────────────────────────────────────────────────────┤
│ MarketData → FeatureService → Strategy Sleeves →            │
│ CombinedStrategy → MacroRegime → RiskVol → VolManaged →     │
│ Allocator → ExecSim                                          │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              PER-REBALANCE LOOP (Weekly Fridays)            │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  t = rebalance_date                                         │
│                                                              │
│  Step 1: combined_strategy.signals(market, t)               │
│           → combined_signals (pd.Series)                     │
│           • TSMOM: long-term momentum signals (multi-feature)│
│           • Medium-Term: medium-term momentum signals       │
│             (if enabled)                                     │
│           • Short-Term: short-term momentum signals         │
│             (if enabled)                                     │
│           • SR3: carry/curve signal for SR3 (if enabled)    │
│           • Rates: curve signals for ZT/ZF/ZN/UB (if enabled)│
│           • FX/Commodity: carry signals for CL/GC/6E/6B/6J  │
│             (if enabled)                                     │
│           • Combine: weighted sum of sleeve signals         │
│                                                              │
│  Step 2: macro_overlay.scaler(market, t)                   │
│           → macro_k (float ∈ [0.5, 1.0])                    │
│           • Compute vol (21-day ES+NQ)                      │
│           • Compute breadth (200-day SMA)                    │
│           • Fetch & normalize FRED indicators              │
│           • Dailyize monthly series (CPI, UNRATE)          │
│           • Cap z-scores (±5.0), smooth inputs (5-day EMA) │
│           • Combine → scaler                                │
│                                                              │
│  Step 3: vol_overlay.scale(combined_signals, market, t)   │
│           → vol_scaled_signals                              │
│           • Scale to target vol (20%)                       │
│           • Apply macro_k: vol_scaled * macro_k             │
│           → final_scaled_signals                            │
│                                                              │
│  Step 4: risk_vol.covariance(market, t)                     │
│           → cov_matrix (pd.DataFrame)                        │
│           • Apply min vol floor (50 bps)                     │
│           mask = risk_vol.mask(market, t)                   │
│                                                              │
│  Step 5: Apply data validity mask                           │
│           • Zero out signals for invalid assets            │
│           → masked_signals                                  │
│                                                              │
│  Step 6: allocator.solve(masked_signals, cov, ...)         │
│           → portfolio_weights (pd.Series)                  │
│           • Enforce constraints (gross/net/bounds)          │
│                                                              │
│  Step 7: Compute portfolio return                           │
│           period_returns = market.get_returns(t+1, next_date)│
│           holding_returns = period_returns.sum()            │
│           port_ret = (weights * holding_returns).sum()     │
│           slippage = turnover * 0.5 bps                     │
│           net_ret = port_ret - slippage                     │
│                                                              │
│  Step 8: Update equity curve & diagnostics                 │
│           equity[t+1] = equity[t] * (1 + net_ret)          │
│           • Log top_movers, k, turnover                     │
│                                                              │
│  prev_weights = portfolio_weights                           │
│  t = next_rebalance_date                                    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    POST-PROCESSING                           │
├─────────────────────────────────────────────────────────────┤
│ Compute metrics (CAGR, Sharpe, MaxDD, etc.)                 │
│ Generate outputs (equity, weights, signals, report)         │
└─────────────────────────────────────────────────────────────┘
```

## Key Implementation Details

### No Look-Ahead Bias
- All data queries use `end=date` (point-in-time)
- Indicators computed only with data available at `date`
- No future information leaks into past decisions

### Deterministic Execution
- Same inputs → same outputs
- No random number generation in signal/weight calculation
- Reproducible results across runs

### Read-Only Database Access
- All connections use `read_only=True`
- No writes, no mutations
- Safe to run on production databases

### Transaction Costs
- Slippage: 0.5 basis points on turnover
- Applied only on rebalance days
- Commission: 0.0 (placeholder for future)
- Turnover calculation: `sum(abs(new_weights - old_weights))` (full roundtrip)

### Rebalance Schedule
- Weekly Fridays (`W-FRI`)
- Holiday handling: if Friday is not a trading day, rebalance on previous business day
- Calendar is explicit to prevent silent skips

### SR3 Feature Computation
- **Data Source**: 12 SR3 contract ranks (0-11) queried by root symbol
- **Rate Space**: All calculations use `r_k = 100 - P_k` (not price space)
- **Features** (5 total):
  - **Carry** (`sr3_carry_01_z`): `r1 - r0` (next 3M rate vs front 3M rate)
  - **Curve** (`sr3_curve_02_z`): `r2 - r0` (3rd 3M rate vs front 3M rate)
  - **Pack Slope** (`sr3_pack_slope_fb_z`): `pack_back - pack_front` (simple difference in rate space)
    - Front pack: ranks 0-3 (mean of rates)
    - Belly pack: ranks 4-7 (mean of rates)
    - Back pack: ranks 8-11 (mean of rates)
  - **Front-Pack Level** (`sr3_front_pack_level_z`): `pack_front = mean(r_0, r_1, r_2, r_3)` (absolute policy level over ~1 year)
  - **Belly Curvature** (`sr3_curvature_belly_z`): `belly_pack - (pack_front + pack_back) / 2` (hump vs straight term structure)
- **Standardization**: Rolling 252-day z-scores with min_periods=126 (clipped at ±3.0)
- **Missing Data**: Forward-fill for contract gaps; features require min_periods (126 days) before use
- **Date Alignment**: If rebalance date doesn't have features, uses most recent available (forward-fill)
- **Availability**: Features available from 2020-06-07 onwards (after 126 days of history)

### Rates Curve Feature Computation
- **Data Source**: Treasury futures (ZT, ZF, ZN, UB) and FRED yields (curve FRED series: DGS2, DGS5, DGS10, DGS30)
- **FRED Series**: Uses curve FRED series from `configs/fred_series.yaml` (separate from macro FRED series used by MacroRegimeFilter)
- **Anchor Method**: FRED yields from 2-3 business days ago used as anchors
- **Yield Calculation**: `y_t = y_anchor - (F_t - F_anchor) / (DV01 * 100)`
  - `y_anchor`: FRED yield at anchor date
  - `F_t`, `F_anchor`: Futures prices at date t and anchor date
  - `DV01`: Dollar value of 01 per $100 notional (from `configs/rates_dv01.yaml`)
- **Features** (4 total):
  - **2s10s Slope** (`curve_2s10s_z`): `y10 - y2` (ZN yield - ZT yield) standardized
  - **5s30s Slope** (`curve_5s30s_z`): `y30 - y5` (UB yield - ZF yield) standardized
  - **2s-5s-10s Curvature** (`curve_2s5s10s_curv_z`): `2 * y5 - y2 - y10` standardized
    - Positive: 5y higher than straight line between 2y and 10y → "hump" in the belly
    - Negative: belly lower → curve is more U-shaped / concave
  - **5s-10s-30s Curvature** (`curve_5s10s30s_curv_z`): `2 * y10 - y5 - y30` standardized
    - Positive: 10y high relative to 5y & 30y (hump at the 10y)
    - Negative: belly low (U-shaped between 5y and 30y)
- **Standardization**: Rolling 252-day z-scores with min_periods=126 (clipped at ±3.0)
- **Anchor Lag**: Default 2 business days (configurable; use 0 for backtests, 2-3 for live)
- **Missing Data**: Gracefully handles missing FRED series (computes available curves only)
- **Availability**: Features available from 2020-01-01 onwards (z-scores from mid-2020 after min_periods)

### FX/Commodity Carry Feature Computation
- **Data Source**: Continuous futures contracts with rank 0 (front) and rank 1 (next) for CL, GC, 6E, 6B, 6J
  - **CL**: `CL_FRONT_VOLUME` (rank 0), `CL_RANK_1_VOLUME` (rank 1) - volume-weighted roll
  - **GC**: `GC_FRONT_VOLUME` (rank 0), `GC_RANK_1_VOLUME` (rank 1) - volume-weighted roll
  - **6E**: `6E_FRONT_CALENDAR` (rank 0), `6E_RANK_1_CALENDAR` (rank 1) - calendar roll
  - **6B**: `6B_FRONT_CALENDAR` (rank 0), `6B_RANK_1_CALENDAR` (rank 1) - calendar roll
  - **6J**: `6J_FRONT_CALENDAR` (rank 0), `6J_RANK_1_CALENDAR` (rank 1) - calendar roll
- **Roll Yield Calculation**: `roll_yield_raw = -(ln(F1) - ln(F0))` where:
  - `F0`: Close price of rank 0 (front contract)
  - `F1`: Close price of rank 1 (next contract)
  - Positive roll_yield_raw = backwardation (F1 < F0) = attractive long carry
  - Negative roll_yield_raw = contango (F1 > F0) = attractive short carry
- **Features** (3 per root):
  - **Time-Series Carry** (`carry_ts_z_<root>`): `roll_yield_raw` standardized with rolling 252-day z-score per root
    - Positive = backwardation (attractive long carry)
    - Negative = contango (attractive short carry)
  - **Cross-Sectional Carry** (`carry_xs_z_<root>`): Daily z-score of `roll_yield_raw` across all 5 assets
    - Compares each asset's carry to the mean across CL, GC, 6E, 6B, 6J on the same day
    - Identifies which assets have relatively best/worst carry among the sleeve
    - Clipped at ±3.0 standard deviations
  - **Carry Momentum** (`carry_mom_63_z_<root>`): 63-day change in `roll_yield_raw`, then standardized
    - `carry_mom_63_raw = roll_yield_raw(t) - roll_yield_raw(t-63)`
    - Standardized with rolling 252-day z-score per root
    - Positive = carry improving over last 3 months; Negative = carry deteriorating
- **Standardization**: 
  - Time-series and momentum: Rolling 252-day z-scores with min_periods=126 (clipped at ±3.0) per root
  - Cross-sectional: Daily z-score across assets (clipped at ±3.0)
- **Missing Data**: Forward-fill and backward-fill for contract data gaps; forward-fill features when rebalance dates don't align
- **Availability**: Features available from 2020-01-01 onwards (z-scores from mid-2020 after min_periods, momentum requires additional 63 days)

### Long-Term Momentum Feature Computation
- **Data Source**: Continuous futures contracts for all symbols in universe
- **Features**:
  - `mom_long_ret_252_z_{symbol}`: 252-day return momentum
    - Lookback: 252 days, skip: 21 days
    - Formula: `r_252 = log(price[t-21] / price[t-21-252])`, vol-standardized with 63-day vol, z-scored
  - `mom_long_breakout_252_z_{symbol}`: 252-day breakout strength
    - Normalized position in 252-day range: `(price - min_252) / (max_252 - min_252)`, rescaled to [-1, 1], z-scored
  - `mom_long_slope_slow_z_{symbol}`: Slow trend slope
    - EMA_63 and EMA_252 of log prices, difference vol-standardized, z-scored
- **Standardization**: Rolling 252-day z-scores with min_periods=126 (clipped at ±3.0) per symbol
- **Missing Data**: Forward-fill and backward-fill for data gaps; forward-fill features when rebalance dates don't align
- **Availability**: Features available from 2020-01-01 onwards (z-scores from mid-2020 after min_periods)

### Medium-Term Momentum Feature Computation
- **Data Source**: Continuous futures contracts for all symbols in universe
- **Features**:
  - `mom_med_ret_84_z_{symbol}`: 84-day return momentum
    - Lookback: 84 days, skip: 10 days
    - Formula: `r_84 = log(price[t-10] / price[t-10-84])`, vol-standardized with 63-day vol, z-scored
  - `mom_med_breakout_126_z_{symbol}`: 126-day breakout strength
    - Normalized position in 126-day range, rescaled to [-1, 1], z-scored
  - `mom_med_slope_med_z_{symbol}`: Medium trend slope
    - EMA_20 and EMA_84 of log prices, difference vol-standardized, z-scored
  - `mom_med_persistence_z_{symbol}`: Trend persistence
    - Rolling mean of return signs over 20 days (measures trend consistency), z-scored
- **Standardization**: Rolling 252-day z-scores with min_periods=126 (clipped at ±3.0) per symbol
- **Missing Data**: Forward-fill and backward-fill for data gaps; forward-fill features when rebalance dates don't align
- **Availability**: Features available from 2020-01-01 onwards (z-scores from mid-2020 after min_periods)

### Short-Term Momentum Feature Computation
- **Data Source**: Continuous futures contracts for all symbols in universe
- **Production Weights**: 0.5 ret_21 / 0.3 breakout_21 / 0.2 slope_fast (empirically validated Nov 2025)
- **Alternatives Tested**:
  - **Equal-Weight Canonical** (1/3, 1/3, 1/3): Tested but not promoted
    - Phase-0: PASSED (sign-only Sharpe 0.31)
    - Phase-1/2: FAILED — Legacy weights consistently outperform equal weights in both standalone and integrated tests
    - Rationale for rejection: 21d return signal is economically more informative than breakout/slope at this horizon; empirical weights preserve the strongest signal
    - Status: Preserved as research variant (`variant="canonical"` in code), not used in production
- **Features**:
  - `mom_short_ret_21_z_{symbol}`: 21-day return momentum
    - Lookback: 21 days, skip: 5 days
    - Formula: `r_21 = log(price[t-5] / price[t-5-21])`, vol-standardized with 20-day vol, z-scored
  - `mom_short_breakout_21_z_{symbol}`: 21-day breakout strength
    - Normalized position in 21-day range, rescaled to [-1, 1], z-scored
  - `mom_short_slope_fast_z_{symbol}`: Fast trend slope
    - EMA_10 and EMA_40 of log prices, difference vol-standardized, z-scored
  - `mom_short_reversal_filter_z_{symbol}`: Reversal filter (RSI-like) - optional, not used in canonical composite
    - 14-day RSI converted to z-score (optional, not used in signal combination by default)
- **Standardization**: Rolling 252-day z-scores with min_periods=126 (clipped at ±3.0) per symbol
- **Missing Data**: Forward-fill and backward-fill for data gaps; forward-fill features when rebalance dates don't align
- **Availability**: Features available from 2020-01-01 onwards (z-scores from mid-2020 after min_periods)

## Configuration Files

- **`configs/data.yaml`**: Universe (dictionary format with roll settings), database path
- **`configs/strategies.yaml`**: Strategy parameters, overlays, allocator settings, strategy sleeves configuration
  - **Important**: Strategy sleeve weights are **relative importance weights** (normalized to sum to 1.0), not absolute risk budgets
- **`configs/fred_series.yaml`**: FRED economic indicators configuration (series list, download settings)
- **`configs/rates_dv01.yaml`**: DV01 values for Treasury futures (ZT, ZF, ZN, UB)

## Related Documentation

- **Trend Meta-Sleeve Implementation**: `docs/META_SLEEVES/TREND_IMPLEMENTATION.md` (current production implementation)
- **Legacy TSMOM**: `docs/legacy/TSMOM_IMPLEMENTATION.md` (legacy single-horizon TSMOM class, not used in production)
- **Medium/Short Momentum**: See `src/agents/feature_long_momentum.py` and `src/agents/strat_momentum_medium.py`, `src/agents/strat_momentum_short.py`
- **SR3 Carry/Curve**: `docs/SR3_CARRY_CURVE.md` (features, implementation, and configuration)
- **Rates Curve**: See `src/agents/feature_rates_curve.py` and `src/agents/strat_rates_curve.py`
- **FX/Commodity Carry**: See `src/agents/feature_carry_fx_commod.py` and `src/agents/strat_carry_fx_commod.py`
- **Macro Regime Filter**: `docs/MACRO_REGIME_FILTER.md`
- **Volatility Targeting**: See `src/agents/overlay_volmanaged.py`
- **Portfolio Allocation**: See `src/agents/allocator.py`
- **Backtest Engine**: See `src/agents/exec_sim.py`

## Code References

- **Main Entry Point**: `run_strategy.py`
- **ExecSim Loop**: `src/agents/exec_sim.py` (lines 307-380)
- **Combined Strategy**: `src/agents/strat_combined.py`
- **TSMOM Signals (Long-Term)**: `src/agents/strat_momentum.py`
- **Medium-Term Momentum**: `src/agents/strat_momentum_medium.py`
- **Short-Term Momentum**: `src/agents/strat_momentum_short.py`
- **SR3 Carry/Curve**: `src/agents/strat_sr3_carry_curve.py`
- **Rates Curve**: `src/agents/strat_rates_curve.py`
- **FX/Commodity Carry**: `src/agents/strat_carry_fx_commod.py`
- **Momentum Features**: `src/agents/feature_long_momentum.py` (Long/Medium/Short)
- **SR3 Features**: `src/agents/feature_sr3_curve.py`
- **Rates Curve Features**: `src/agents/feature_rates_curve.py`
- **FX/Commodity Carry Features**: `src/agents/feature_carry_fx_commod.py`
- **Feature Service**: `src/agents/feature_service.py`
- **Macro Scaler**: `src/agents/overlay_macro_regime.py`
- **Vol Targeting**: `src/agents/overlay_volmanaged.py`
- **Allocation**: `src/agents/allocator.py`

---

**Last Updated**: November 2025

**Current Baseline Performance (core_v3_no_macro)**: Trend Meta-Sleeve only
- **CAGR**: -2.59%
- **Sharpe**: -0.15
- **MaxDD**: -39.69%
- **Vol**: 12.31%
- **HitRate**: 49.77%

**Note**: The Trend Meta-Sleeve combines three atomic sleeves (long-term, medium-term, short-term momentum) into a unified meta-signal using industry-standard horizon weights. EWMA volatility normalization provides more stable risk-adjusted signals. Performance metrics reflect the challenging 2022-2025 period for trend-following strategies. The architecture is validated and production-ready; performance will improve as additional meta-sleeves are added.

## Recent Improvements (November 2025)

### Phase 4: Trend Meta-Sleeve Architecture (v2)

- ✅ **Trend Meta-Sleeve Implementation (Phase 4a)**: Unified meta-sleeve architecture
  - **Two-Layer Architecture**: Introduced Meta-Sleeves (economic ideas) and Atomic Sleeves (implementation variants)
  - **Trend Meta-Sleeve**: Combines three atomic sleeves (long-term, medium-term, short-term momentum) into unified meta-signal
  - **Horizon Weights**: Industry-standard weights (long_252=0.50, med_84=0.30, short_21=0.20)
  - **Feature Preservation**: All existing features from three atomic sleeves preserved and used as subcomponents
  - **Signal Processing**: Atomic signals → meta-signal blend → cross-sectional z-score → clip ±3.0
  - **Configuration**: `tsmom_multihorizon` config block in `configs/strategies.yaml`
  - **Strategy Profile**: `core_v3_no_macro` (current production baseline)
  - **Integration**: Fully integrated with CombinedStrategy; risk overlays operate at Meta-Sleeve layer

- ✅ **EWMA Volatility Normalization (Phase 4b)**: Risk-normalized signals within Trend Meta-Sleeve
  - **EWMA Vol Calculation**: 63-day half-life EWMA volatility (standard CTA choice)
  - **Formula**: `σ²_t = λ * σ²_{t-1} + (1-λ) * r²_{t-1}` where `λ = 0.5^(1/63)`
  - **Annualization**: `σ_annual = σ_daily * sqrt(252)`
  - **Vol Floor**: 5% annualized minimum to prevent extreme scaling
  - **Risk Normalization**: `s_risk = z_clipped / max(σ_annual, σ_floor)`
  - **Global Scale**: `s_final = risk_scale * s_risk` (default: 0.2)
  - **Benefits**: Same signal magnitude implies same risk across assets; prevents exposure jumps during vol spikes
  - **Configuration**: `vol_normalization` section in `tsmom_multihorizon` params
  - **Performance Impact**: Reduced volatility (15.37% → 12.31%) and improved MaxDD (-43.80% → -39.69%)

### Phase 2: SR3 Carry/Curve and Rates Curve Features

- ✅ **SR3 Carry/Curve Strategy Sleeve**: New strategy sleeve for SOFR futures (5-feature implementation)
  - **Carver-style carry**: `r1 - r0` in rate space (next 3M vs front 3M)
  - **Curve shape**: `r2 - r0` in rate space (3rd 3M vs front 3M)
  - **STIR pack slope**: Front pack (ranks 0-3) vs back pack (ranks 8-11)
  - **Front-pack level**: Absolute policy expectation over ~1 year (ranks 0-3 average)
  - **Belly curvature**: Hump vs straight term structure (belly pack vs front/back average)
  - All features standardized with rolling 252-day z-scores (clipped at ±3.0)
  - Default feature weights: w_carry=0.30, w_curve=0.25, w_pack_slope=0.20, w_front_lvl=0.10, w_curv_belly=0.15
  - Features computed from 2020-06-07 onwards (after 126 days of history)

- ❌ **Rates Curve RV Meta-Sleeve**: **Parked for redesign** (Phase-0 sanity check failed)
  - **Phase-0 Results**: Sign-only curve trading showed near-zero Sharpe (0.002) with negative 2s10s leg (-0.20 Sharpe)
  - **Findings**: Initial slope-based approach fragile to post-2022 regime changes (Fed hiking cycle); 2s10s leg particularly weak
  - **Status**: Will return with DV01-neutral flies, pack spreads, macro gating, and improved yield reconstruction
  - **Atomic Sleeves**: 2s10s flattener/steepener, 5s30s flattener/steepener
  - **Implementation Details** (for reference):
    - FRED-anchored yields: Uses DGS2, DGS5, DGS10, DGS30 from 2-3 days ago as anchors
    - DV01-based yield updates: `y_t = y_anchor - (F_t - F_anchor) / (DV01 * 100)`
    - Features: 2s10s slope, 5s30s slope, 2s-5s-10s curvature, 5s-10s-30s curvature
    - Features available from 2020-01-01 onwards

- ✅ **Multi-Sleeve Architecture**: CombinedStrategy wrapper
  - Combines TSMOM (long-term), Medium-Term Momentum, Short-Term Momentum, SR3 carry/curve, Rates curve, and FX/Commodity carry signals with configurable weights
  - Config-driven: Enable/disable sleeves via `configs/strategies.yaml`
  - Handles strategies that require features (e.g., TSMOM, Medium/Short Momentum, SR3, Rates Curve, FX/Commodity Carry)

- ✅ **FeatureService**: Centralized feature computation
  - Manages SR3 curve, Rates curve, FX/Commodity carry, and Momentum (Long/Medium/Short) features with caching
  - Point-in-time feature computation (no look-ahead)
  - Handles missing data with forward-fill/backward-fill
  - Fetches all historical data from beginning (not limited by end_date)

- ✅ **TSMOM Refactor (Long-Term Momentum)**: Multi-feature momentum strategy
  - Replaced single-feature TSMOM with three-feature long-term momentum
  - Features: 252-day return momentum, 252-day breakout strength, slow trend slope (EMA_63 - EMA_252)
  - Combines features with configurable weights (default: ret_252=0.5, breakout_252=0.3, slope_slow=0.2)
  - Uses pre-computed LONG_MOMENTUM features from FeatureService
  - Cross-sectional z-scoring and clipping at ±3.0
  - Integrated as primary momentum sleeve with configurable weight (default: 0.6)

- ✅ **Medium-Term Momentum Sleeve**: Multi-feature momentum for medium horizons
  - Features: 84-day return momentum, 126-day breakout strength, medium trend slope (EMA_20 - EMA_84), persistence
  - Combines features with configurable weights (default: ret_84=0.4, breakout_126=0.3, slope_med=0.2, persistence=0.1)
  - Uses pre-computed MEDIUM_MOMENTUM features from FeatureService
  - Cross-sectional z-scoring and clipping at ±3.0
  - Integrated as a new sleeve in CombinedStrategy with configurable weight (default: 0.15)

- ✅ **Short-Term Momentum Sleeve**: Multi-feature momentum for short horizons
  - Features: 21-day return momentum, 21-day breakout strength, fast trend slope (EMA_10 - EMA_40), reversal filter (RSI-like)
  - Combines features with configurable weights (default: ret_21=0.5, breakout_21=0.3, slope_fast=0.2, reversal_filter=0.0)
  - Uses pre-computed SHORT_MOMENTUM features from FeatureService
  - Cross-sectional z-scoring and clipping at ±3.0
  - Integrated as a new sleeve in CombinedStrategy with configurable weight (default: 0.10)

- ❌ **Carry Meta-Sleeve**: **Parked for redesign** (Phase-0 sanity check failed)
  - **Phase-0 Results**: Sign-only roll yield strategy showed negative Sharpe (-0.69) across all assets (2020-2025)
  - **Findings**: All assets (CL, GC, 6E, 6B, 6J) showed negative Sharpe; performance degraded post-2022
  - **Status**: Remains on roadmap for redesign (e.g., sector-based roll yield, DV01-neutral carry, regime-dependent filters)
  - **Atomic Sleeves**: FX/Commodity Carry (CL, GC, 6E, 6B, 6J), SR3 Carry/Curve
  - **Implementation Details** (for reference):
    - Roll yield: `roll_yield_raw = -(ln(F1) - ln(F0))` where F0=rank 0, F1=rank 1
    - Features: time-series, cross-sectional, momentum (3 per root)
    - Features available from 2020-01-01 onwards

- ✅ **MarketData Extensions**:
  - `get_contracts_by_root()`: Query contracts by root symbol and rank
    - Supports 12 SR3 contract ranks (0-11) for curve analysis
    - Automatic rank assignment based on contract expiration order
  - `get_fred_indicators()`: Query multiple FRED series at once
    - Used for Rates curve features (DGS2, DGS5, DGS10, DGS30)

- ✅ **Missing Data Handling**: Robust gap handling
  - Forward-fill and backward-fill for contract data gaps
  - Forward-fill features when rebalance dates don't align with feature dates
  - Graceful degradation: returns zero signals when features unavailable

### Previous Improvements

- ✅ **SOFR Roll Configuration**: SR3 uses calendar roll with T-2 offset (IMM convention)
- ✅ **FRED Series Configuration**: Config-driven FRED indicators with dailyization for monthly series
- ✅ **MacroRegimeFilter Enhancements**:
  - Monthly series (CPI, UNRATE) dailyized before z-scoring
  - Z-score capping (±5.0) to prevent single prints from swinging scaler
  - 5-day EMA input smoothing for breadth and FRED composite
  - Data freshness checks (warns if monthly series stale > 45 days)
- ✅ **Friday Holiday Handling**: Automatic fallback to previous business day
- ✅ **Covariance Stability**: Min vol floor (50 bps) prevents exploding leverage
- ✅ **Data Validity Mask**: Invalid signals zeroed before allocation
- ✅ **Diagnostics**: "What-moved" reports with top weight changes, k values, turnover

