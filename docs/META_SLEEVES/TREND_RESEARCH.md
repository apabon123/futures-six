# Trend Meta-Sleeve — Research Notebook

## 1. Executive Summary

### What Is the Trend Meta-Sleeve?

The **Trend Meta-Sleeve** is a multi-horizon time-series momentum strategy that combines multiple atomic sleeves (implementation variants) into a unified meta-signal. It captures trend-following alpha across different time horizons, from short-term (21 days) to long-term (252 days), using a combination of return momentum, breakout signals, and trend slope indicators.

**Implementation**: `TSMOMMultiHorizon` strategy (v2) in `src/agents/strat_tsmom_multihorizon.py`

**Signal Processing**:
- Atomic sleeves are vol-normalized and combined using frozen horizon weights
- Cross-sectional z-scoring across assets
- EWMA volatility normalization (63-day half-life, 5% floor)
- Signal clipping at ±3.0 standard deviations

### Production Atomic Sleeves

The Trend Meta-Sleeve has **5 validated production atomic sleeves**. The `core_v3_no_macro` production profile currently uses **4 active atomic sleeves**:

**Active in `core_v3_no_macro` (Production Profile):**

1. **Long-Term (252d) — Canonical**: Equal-weight (1/3, 1/3, 1/3) composite of return, breakout, and slope
   - **Academic Basis**: Moskowitz/Ooi/Pedersen (2012), Hurst/Ooi/Pedersen (2014)
   - **Status**: Production (Phase-0/1/2 passed, promoted Nov 2025)
   - **Horizon Weight**: 48.5%
   - **Feature Weights**: 1/3 ret_252 + 1/3 breakout_252 + 1/3 slope_slow
   - **Phase-0**: Sharpe 0.58, CAGR 3.15%
   - **Phase-1 vs Legacy (0.5/0.3/0.2)**: Sharpe +0.013, CAGR +0.17%, MaxDD +0.45%
   - **Phase-2**: All criteria passed (5/5), years outperformed 3/5, correlation stable 0.996

2. **Medium-Term (84d) — Canonical**: Equal-weight (1/3, 1/3, 1/3) composite of 84d return, 84d breakout, EMA21-84 slope
   - **Academic Basis**: Moskowitz/Ooi/Pedersen (2012) — 3-6m canonical medium-term horizon
   - **Status**: Production (Phase-0/1/2 passed, promoted Nov 2025)
   - **Horizon Weight**: 29.1%
   - **Feature Weights**: 1/3 ret_84 (skip 10d, vol-scaled) + 1/3 breakout_84 + 1/3 slope_21_84
   - **Phase-0**: Sharpe 0.71, CAGR 3.15% (sign-only, 84d lookback, skip 10d)
   - **Phase-1 vs Legacy**: Sharpe -0.39 vs -0.43, CAGR -6.17% vs -7.30% (standalone sleeves, both negative as expected)
   - **Phase-2**: CAGR +9.42%, Sharpe +0.74, MaxDD +22.97% improvement vs baseline (integrated into Trend Meta-Sleeve)
   
   **Medium-Term (84/126d) — Legacy**: 84-day return, 126-day breakout, medium trend slope (EMA_20 - EMA_84), persistence
   - **Academic Basis**: AQR multi-horizon momentum
   - **Status**: DEPRECATED (replaced by canonical in Nov 2025, preserved in core_v3_no_macro_legacy for historical comparison)

3. **Short-Term (21d) — Production**: 21-day return, 21-day breakout, fast trend slope (EMA_10 - EMA_40)
   - **Academic Basis**: Jegadeesh/Titman (1993) — short-term momentum (1-3 months)
   - **Status**: Production (empirically-tuned weights validated Nov 2025)
   - **Horizon Weight**: 19.4%
   - **Feature Weights**: 0.5 ret_21 (skip 5d, vol-scaled with 20d vol) + 0.3 breakout_21 + 0.2 slope_fast
   - **Equal-Weight Canonical Tested**: ❌ FAILED Phase-1/2 (Nov 2025)
     - Canonical 1/3, 1/3, 1/3 consistently underperformed legacy 0.5, 0.3, 0.2 in both standalone and integrated tests
     - Return signal is economically more informative at 21d horizon; empirical weights > equal weights
     - Legacy weights remain production standard

4. **Breakout Mid (50–100d)**: 50/100-day breakout-based trend sleeve using Donchian-style range breakouts
   - **Academic Basis**: Donchian channels, Baz et al. (2015)
   - **Status**: Production (Phase-0/1B/2/3 passed, integrated in core_v3_no_macro)
   - **Horizon Weight**: 3.0%
   - **Feature Weights**: 0.7 breakout_50 + 0.3 breakout_100 (70/30 blend)

**Validated but Not Currently Active in Production Profile:**

5. **Residual Trend (252d-21d)**: Long-horizon trend minus short-term movement, cross-sectionally z-scored
   - **Academic Basis**: Hurst/Ooi/Pedersen extension (noise filtering)
   - **Status**: Production-ready (Phase-0/1/2 passed, validated but not currently enabled in `core_v3_no_macro` profile)
   - **Design Intent**: 4th atomic trend sleeve, designed for internal integration into Trend Meta-Sleeve
   - **Current Status**: Disabled in production profile (can be enabled via configuration)

**Weight Architecture** (Two Layers):

**Layer 1 — Horizon Weights** (across atomic sleeves in `core_v3_no_macro`):
- Long (252d): 48.5%
- Medium (84d): 29.1%
- Short (21d): 19.4%
- Breakout Mid (50-100d): 3.0%
- **Total**: 100.0% (4 active sleeves)

**Layer 2 — Feature Weights** (inside each atomic sleeve):
- **Long (canonical, production)**: 1/3 ret_252 + 1/3 breakout_252 + 1/3 slope_slow (promoted Nov 2025)
- **Medium (canonical, production)**: 1/3 ret_84 + 1/3 breakout_84 + 1/3 slope_21_84 (promoted Nov 2025)
- **Medium (legacy, deprecated)**: 0.4 ret_84 + 0.3 breakout_126 + 0.2 slope_med + 0.1 persistence
- **Short (canonical, Phase-0/1/2)**: 1/3 ret_21 + 1/3 breakout_21 + 1/3 slope_fast (equal-weight, Phase-0/1/2 Nov 2025)
- **Short (legacy, production)**: 0.5 ret_21 + 0.3 breakout_21 + 0.2 slope_fast (baseline for comparison)
- **Breakout Mid (production)**: 0.7 breakout_50 + 0.3 breakout_100 (70/30 blend)

### Research Status

**Parked Sleeves**:
- **Persistence (Momentum-of-Momentum)**: Failed Phase-1
  - **Why Failed**: Weak performance in small universe (13 assets), challenging 2021-2025 regime, insufficient cross-asset diversity
  - **Re-Test Conditions**: Universe expansion (≥25%), historical expansion (≥5 years), architecture changes, or 3+ years of new live data

**Future Research Candidates**:
- Volatility-Adjusted Momentum: Normalize momentum by realized volatility
- Regime-Adaptive Horizons: Adjust lookbacks based on volatility regimes
- Cross-Asset Trend Confirmation: Use correlated asset trends to filter positions
- Trend Strength Scoring: Combine multiple indicators into strength score

### What Passed / Failed and Why

**✅ Long-Term Canonical (252d Equal-Weight)**: **PASSED** (Phase-0/1/2)
- **Phase-0**: Sharpe 0.58, CAGR 3.15% (sign-only, equal-weighted)
- **Phase-1 vs Legacy (0.5/0.3/0.2)**: Sharpe +0.013 (+12.5%), CAGR +0.17%, MaxDD +0.45%, Vol -0.10%
- **Phase-2**: 5/5 criteria passed, outperformed 3/5 years, correlation 0.996 (extremely stable)
- **Why It Works**: Equal weighting provides better diversification across complementary signals (return, breakout, slope), more robust than overfitted legacy weights
- **Status**: Production (promoted Nov 2025, now the canonical Long-Term sleeve)

**✅ Residual Trend (252d-21d)**: **PASSED** (Phase-0/1/2)
- **Phase-0**: Sharpe 0.29 (sign-only, equal-weighted)
- **Phase-1**: +2.10% CAGR improvement, +0.18 Sharpe improvement at 20% weight
- **Why It Works**: Isolates persistent trend by subtracting short-term noise, reduces drawdowns in choppy markets
- **Status**: Production (integrated as 4th atomic sleeve, currently not in production profile)

**✅ Breakout (50-100d)**: **PASSED** (Phase-0/1B/2/3)
- **Phase-0**: Sharpe 0.246 (sign-only)
- **Phase-1**: Initial 10% weight failed; Phase-1B refinement tested 4 variants (70/30, 30/70, 100/0, 0/100), only 70/30 passed
- **Phase-2**: Sharpe 0.0953 vs baseline 0.0857, MaxDD -31.52% vs -32.02%
- **Phase-3**: Promoted to production, integrated into `core_v3_no_macro` profile
- **Why It Works**: Captures volatility expansions and regime shifts, complementary to return-based momentum
- **Status**: Production (70/30 feature blend, 3% horizon weight)

**✅ Canonical Medium-Term (84d Equal-Weight)**: **PASSED** (Phase-0/1/2)
- **Phase-0**: Sharpe 0.71, CAGR 3.15% (sign-only, 84d lookback, skip 10d, 1,729 days of data)
- **Phase-1 vs Legacy**: 
  - Standalone canonical: Sharpe -0.39, CAGR -6.17%, MaxDD -53.07%
  - Standalone legacy: Sharpe -0.43, CAGR -7.30%, MaxDD -54.59%
  - Canonical outperformed legacy in standalone comparison
- **Phase-2**: Integrated into Trend Meta-Sleeve (`core_v3_medcanon_v1` vs `core_v3_no_macro_baseline`)
  - CAGR: +9.42% improvement (from -2.59% to +6.83%)
  - Sharpe: +0.74 improvement (from -0.15 to +0.59)
  - MaxDD: +22.97% improvement (from -39.69% to -16.72%)
  - Hit Rate: +0.78% improvement
- **Academic Justification**: 
  - Moskowitz/Ooi/Pedersen (2012) — TSMOM strongest in 3m-12m window; 3-6m is canonical "intermediate"
  - Hurst/Ooi/Pedersen (2017) — AQR uses 1m, 3m, 12m; 3m = "intermediate trend"
  - 84 days (~4 months) fits neatly between short (21d) and long (252d)
  - Skip 10d to avoid short-term reversal/noise (futures-appropriate, less than equity 21d skip)
  - 21d vol window for internal scaling (standard short-term vol)
  - Equal-weight (1/3, 1/3, 1/3) for balanced diversification
- **Canonical Features**:
  - 84d return momentum (skip 10d, vol-scaled with 21d vol, z-scored 252d, clip ±3)
  - 84d breakout strength (z-scored 252d, clip ±3, complements return momentum)
  - EMA21 vs EMA84 slope (vol-scaled with 21d vol, z-scored 252d, clip ±3, captures acceleration)
- **Status**: Production (promoted Nov 2025, integrated into `core_v3_no_macro` profile)

**✅ Short-Term (21d) — Legacy (Production)**: **0.5 / 0.3 / 0.2 weights** (Nov 2025)
- **Phase-0/1/2**: Legacy weights validated as superior to equal-weight canonical
- **Production Weights**: 0.5 ret_21 / 0.3 breakout_21 / 0.2 slope_fast (reversal filter available but not used)
- **Academic Justification**: 
  - Jegadeesh/Titman (1993) — classic short-term momentum (1-3 months)
  - Moskowitz/Ooi/Pedersen (2012) — TSMOM works across horizons; short-term captures fast trends
  - 21 days (~1 month) is canonical short-term horizon, complements medium (84d) and long (252d)
  - Skip 5d to avoid short-term reversal/noise (JT-style reversal avoidance)
  - 20d vol window for internal scaling (standard short-term vol)
  - **Empirically-tuned weights (0.5/0.3/0.2) outperform equal-weight**: Return signal is economically more informative at 21d horizon
- **Production Features**:
  - 21d return momentum (skip 5d, vol-scaled with 20d vol, z-scored 252d, clip ±3)
  - 21d breakout strength (z-scored 252d, clip ±3, complements return momentum)
  - Fast slope: EMA10 vs EMA40 (vol-scaled with 20d vol, z-scored 252d, clip ±3, captures acceleration)
- **Status**: Production (canonical weights for short-term sleeve, Nov 2025)

**❌ Short-Term (21d) — Equal-Weight Canonical**: **FAILED** (Phase-0/1/2, Nov 2025)
- **Phase-0**: ✅ PASSED — sign-only 21d TS-MOM, Sharpe 0.31, hit rate 52.5%
- **Phase-1**: ❌ FAILED — Legacy (0.5/0.3/0.2) outperforms equal-weight (1/3, 1/3, 1/3)
  - Canonical: CAGR -4.13%, Sharpe -0.324
  - Legacy: CAGR -3.87%, Sharpe -0.308 (better by +0.26% CAGR, +0.016 Sharpe)
- **Phase-2**: ❌ FAILED — In full Trend Meta-Sleeve, legacy again outperforms canonical
  - Canonical: CAGR 6.63%, Sharpe 0.5735, MaxDD -17.24%
  - Legacy: CAGR 6.83%, Sharpe 0.5876, MaxDD -16.72% (better by +0.20% CAGR, +0.014 Sharpe, +0.52% MaxDD)
- **Why It Failed**: 
  - 21d return signal is economically more informative than breakout/slope at this short horizon
  - Equal-weighting dilutes the strongest signal (ret_21) which carries most of the edge
  - Unlike LT/MT, where equal-weighting improves diversification, ST benefits from empirically-tuned weights
  - Breakout and slope act more as "regularizers" or confirmers rather than independent sources of alpha at 21d
- **Key Insight**: Equal-weight symmetry worked for Long-Term and Medium-Term horizons but does not generalize to Short-Term. The framework correctly identifies that different horizons require different weight structures.
- **Status**: Tested but not promoted. Legacy weights (0.5/0.3/0.2) remain production standard. Equal-weight variant preserved as research option for future re-testing if universe/regime changes.

**❌ Persistence (Momentum-of-Momentum)**: **FAILED** (Phase-1)
- **Phase-0**: Only slope acceleration variant passed (Sharpe 0.22)
- **Phase-1**: Standalone Sharpe 0.06, external blend underperformed baseline (0.087 vs 0.097)
- **Why It Failed**: Small universe (13 assets), challenging 2021-2025 regime, weak in volatility clustering periods
- **Status**: Parked (re-test when conditions improve)

### Next Planned Research

1. **Re-test Persistence**: When universe expands or historical depth increases
2. **Volatility-Adjusted Momentum**: Phase-0 candidate for future testing
3. **Regime-Adaptive Features**: Explore volatility regime overlays for existing sleeves
4. **Monitor Production Performance**: Track all production sleeves for stability and performance

---

## 2. Current Production Sleeves (Summary)

**Production Profile (`core_v3_no_macro`) Active Sleeves:**

The `core_v3_no_macro` Trend Meta-Sleeve currently uses **4 active atomic sleeves**:
1. **Long-Term (252d) — Canonical** (48.5% weight)
2. **Medium-Term (84d) — Canonical** (29.1% weight) 
3. **Short-Term (21d) — Legacy** (19.4% weight)
4. **Breakout Mid (50-100d)** (3.0% weight)

**Active in `core_v3_no_macro` Production Profile:**

| Sleeve | Status | Phase | Academic Basis | Summary | Horizon Weight |
|--------|--------|-------|----------------|---------|----------------|
| **Long-Term (252d) Canonical** | Production | Passed Ph0/Ph1/Ph2 | MOP (2012), HOP (2014) | Equal-weight (1/3, 1/3, 1/3) composite: ret_252 + breakout_252 + slope_slow | 48.5% |
| **Medium-Term (84d) Canonical** | Production | Passed Ph0/Ph1/Ph2 | MOP (2012) | Equal-weight (1/3, 1/3, 1/3) composite: ret_84 + breakout_84 + slope_21_84 | 29.1% |
| **Short-Term (21d) Legacy** | Production | N/A (legacy) | JT93 | 21d momentum / breakout / fast slope (0.5/0.3/0.2 weights) | 19.4% |
| **Breakout Mid (50/100d)** | Production | Passed Ph0/Ph1/Ph1B/Ph2/Ph3 | Donchian | Range breakout, 70/30 blend | 3.0% |

**Validated but Not Currently Active:**

| Sleeve | Status | Phase | Academic Basis | Summary |
|--------|--------|-------|----------------|---------|
| **Residual Trend (252d-21d)** | Production-ready | Passed Ph0/Ph1/Ph2 | HOP extension | Long-horizon minus short-term noise (not enabled in core_v3_no_macro) |

**Deprecated:**

| Sleeve | Status | Phase | Academic Basis | Summary |
|--------|--------|-------|----------------|---------|
| **Medium-Term (84/126d) Legacy** | Deprecated | N/A (legacy) | AQR | 84–126d blend, persistence (preserved in core_v3_no_macro_legacy) |

**Parked:**

| Sleeve | Status | Phase | Academic Basis | Summary |
|--------|--------|-------|----------------|---------|
| **Persistence** | Parked | Failed Ph1 | Baz et al. | Momentum-of-momentum (re-test when universe/history expands) |

---

## 3. Sleeves Tested and Their Outcomes

### Residual Trend (252d-21d)

**What It Is**: Long-horizon trend (252d) with short-term movement (21d) subtracted out, isolating persistent trend component.

**Why Tried**: Hypothesis that subtracting short-term noise from long-horizon trends improves signal quality in choppy markets and reduces drawdowns during regime flips.

**Phase-0 Result**: ✅ PASSED (Sharpe 0.29, sign-only, equal-weighted). Strong performance in rates (ZT, ZN, ZF) and SR3.

**Phase-1 Result**: ✅ PASSED (external blend at 20% weight). +2.10% CAGR improvement, +0.18 Sharpe improvement, reduced MaxDD by 2.89%.

**Promotion Reason**: Economically distinct signal (noise-filtered trend), validated performance across multiple years, especially helpful in challenging 2024-2025 period.

**Status**: Production (integrated as 4th atomic sleeve, currently not in production profile)

---

### Breakout (50-100d)

**What It Is**: Donchian-style range breakout signals using 50-day and 100-day lookbacks, capturing volatility expansions and regime shifts.

**Why Tried**: Hypothesis that breakout signals are complementary to return-based momentum, providing diversification and better performance in volatility expansion regimes.

**Phase-0 Result**: ✅ PASSED (Sharpe 0.246, sign-only). Strong performance in rates, moderate in commodities.

**Phase-1 Result**: ❌ FAILED (initial 10% weight), ✅ PASSED (Phase-1B refinement at 3% weight, 70/30 feature blend). Phase-1B tested 4 variants (70/30, 30/70, 100/0, 0/100), only 70/30 passed. Phase-1B winner: Sharpe 0.0953 vs baseline 0.0857, MaxDD -31.52% vs -32.02%.

**Phase-2 Result**: ✅ PASSED. All promotion criteria met: Sharpe ≥ baseline, MaxDD ≤ baseline, robust across all 5 years.

**Phase-3 Result**: ✅ PASSED. Promoted to production, integrated into `core_v3_no_macro` profile.

**Promotion Reason**: 70/30 blend (50d/100d) optimal configuration, 3% horizon weight sufficient for alpha contribution without signal conflicts, outperforms baseline in all metrics.

**Status**: Production (70/30 feature blend, 3% horizon weight, integrated in core_v3_no_macro)

**Re-Test Conditions**: N/A (in production)

---

### Canonical Medium-Term (84d Equal-Weight)

**What It Is**: Canonical medium-term momentum sleeve using equal-weight (1/3, 1/3, 1/3) composite of 84d return momentum, 84d breakout strength, and EMA21-84 slope. Replaces legacy medium-term (0.4/0.3/0.2/0.1) with academically grounded equal-weight approach.

**Why Tried**: Hypothesis that equal-weight composite provides better diversification across complementary signals (return, breakout, slope) and is more robust than legacy weights. Academic basis: Moskowitz/Ooi/Pedersen (2012) identifies 3-6m as canonical "intermediate" horizon for TSMOM.

**Phase-0 Result**: ✅ PASSED (Sharpe 0.71, CAGR 3.15%, sign-only, 84d lookback, skip 10d, 1,729 days of data from 2020-01-02 to 2025-10-31)

**Phase-1 Result**: ✅ PASSED (standalone canonical vs legacy comparison)
- Standalone canonical: Sharpe -0.39, CAGR -6.17%, MaxDD -53.07%
- Standalone legacy: Sharpe -0.43, CAGR -7.30%, MaxDD -54.59%
- Canonical outperformed legacy in standalone comparison (both negative as expected for standalone medium-term sleeves)

**Phase-2 Result**: ✅ PASSED (integrated into Trend Meta-Sleeve)
- Run: `core_v3_medcanon_v1` vs `core_v3_no_macro_baseline` (2020-01-02 to 2025-10-31)
- CAGR: +9.42% improvement (from -2.59% to +6.83%)
- Sharpe: +0.74 improvement (from -0.15 to +0.59)
- MaxDD: +22.97% improvement (from -39.69% to -16.72%)
- Hit Rate: +0.78% improvement (from 49.77% to 50.55%)
- Equity Ratio: Final ratio 1.6276 (canonical outperformed baseline by ~62.76%)

**Phase-3 Result**: ✅ PASSED. Promoted to production, integrated into `core_v3_no_macro` profile (Nov 2025).

**Promotion Reason**: Significant outperformance across all metrics in Phase-2, equal-weight composite provides better diversification, academically grounded specification (84d horizon, 10d skip, 21d vol window, 252d z-score).

**Status**: Production (equal-weight 1/3, 1/3, 1/3 composite, integrated in core_v3_no_macro, promoted Nov 2025)

**Canonical Specification**:
- **Lookback**: 84 trading days (~4 months)
- **Skip Recent**: 10 trading days (avoids short-term reversal/noise)
- **Vol Window**: 21 days (for internal vol scaling)
- **Z-Score Window**: 252 days (rolling standardization)
- **Clip**: ±3 standard deviations
- **Features**: 
  - 84d return momentum (skip 10d, vol-scaled, z-scored)
  - 84d breakout strength (z-scored)
  - EMA21 vs EMA84 slope (vol-scaled, z-scored)
- **Composite**: Equal-weight (1/3, 1/3, 1/3)

**Re-Test Conditions**: N/A (in production)

---

### Persistence (Momentum-of-Momentum)

**What It Is**: "Momentum-of-momentum" captures the rate of change of trend itself. If momentum is strengthening, that is bullish; if fading, that is bearish.

**Why Tried**: Academic evidence (Baz et al. 2015) suggests persistence improves trend stability and helps during strong trend extensions.

**Phase-0 Result**: ⚠️ PARTIAL PASS (only slope acceleration variant passed with Sharpe 0.22). Return and breakout acceleration variants failed.

**Phase-1 Result**: ❌ FAILED. Standalone Sharpe 0.06, external blend underperformed baseline (0.087 vs 0.097). Weak in 2022-2023 volatility clustering periods.

**Failure Reason**: Small universe (13 assets), challenging 2021-2025 regime dominated by reversals and sideways chop, insufficient cross-asset diversity. Persistence effects are stronger in broad universes (35-60 assets) and long sustained trends (2010-2020).

**Status**: Parked (not deleted, documented for future re-test)

**Re-Test Conditions**: 
1. Universe expansion (≥25%, e.g., 13 → 20+ assets)
2. Historical expansion (≥5 years, e.g., extending to 2010 or earlier)
3. Architecture change (related Trend component added/changed)
4. Feature platform upgrades (cross-asset validation, multi-horizon acceleration)
5. Market regime shifts (3+ years of new live data)

---

## 4. Detailed Research Sections

### Sleeve 1 — Residual Trend (252d - 21d)

#### 4.1.1 Economic Idea

**Residual Trend** focuses on **medium-to-long horizon** trend after stripping out short-term noise. The core hypothesis is:

- Long-horizon trends (e.g., 252 days) capture the underlying economic momentum
- Short-term movements (e.g., 21 days) often reflect noise, reversals, or temporary disruptions
- By subtracting short-term returns from long-term returns, we isolate the "residual" trend that persists beyond short-term fluctuations

**Intended Benefits:**
- Better performance in noisy/choppy markets than raw long lookback
- Reduced drawdowns during sharp regime flips (short-term reversals don't trigger position changes)
- Preserves strong long-horizon trends while avoiding chasing very recent short reversals

**Academic Grounding**: Extension of Hurst/Ooi/Pedersen multi-horizon momentum, focusing on noise filtering.

#### 4.1.2 Data Requirements

**Uses only existing continuous prices / returns:**
- No additional data sources required
- No database schema changes
- No FRED or macro data
- No options data
- Works with the same 13-contract universe used in production

#### 4.1.3 Phase-0 Design

The Phase-0 implementation uses a minimal sign-only approach to validate the core economic idea:

**For each asset and date `t`:**

1. **Compute long-horizon log return** over `L_long` days (default 252):
   ```
   long_ret = log(price_t / price_{t-L_long})
   ```

2. **Compute short-horizon log return** over `L_short` days (default 21):
   ```
   short_ret = log(price_t / price_{t-L_short})
   ```

3. **Define residual trend return**:
   ```
   resid_ret = long_ret - short_ret
   ```

4. **Signal generation**:
   - If `resid_ret > epsilon` (e.g., 1e-8): `signal = +1`
   - If `resid_ret < -epsilon`: `signal = -1`
   - If `abs(resid_ret) < epsilon`: `signal = 0` (avoid noise)

5. **Portfolio construction**:
   - Equal-weight across assets each day
   - Daily rebalancing
   - No overlays (no vol targeting, no macro filters, no z-scoring)
   - No leverage constraints (positions are ±1 or 0)

**Default Parameters:**
- `long_lookback`: 252 days
- `short_lookback`: 21 days
- `epsilon`: 1e-8 (threshold for zero signal)

#### 4.1.4 Phase-0 Results

**Status**: ✅ PASS (Sharpe ≥ 0.2)

**Target Window**: 2021-01-01 to 2025-10-31

**Portfolio Metrics:**
- **Sharpe Ratio**: 0.29
- **CAGR**: 1.03%
- **Volatility**: 3.86%
- **Max Drawdown**: -5.75%
- **Hit Rate**: 22.60%
- **Trading Days**: 1,509 (5.99 years)

**Top Performers (by Sharpe):**
- **SR3_FRONT_CALENDAR**: Sharpe 0.58, AnnRet 0.34%
- **CL_FRONT_VOLUME**: Sharpe 0.48, AnnRet 11.67%
- **ZT_FRONT_VOLUME**: Sharpe 0.43, AnnRet 0.63%
- **6J_FRONT_CALENDAR**: Sharpe 0.18, AnnRet 1.12%
- **ZN_FRONT_VOLUME**: Sharpe 0.19, AnnRet 0.77%

**Underperformers:**
- **UB_FRONT_VOLUME**: Sharpe -0.45, AnnRet -4.51%
- **6B_FRONT_CALENDAR**: Sharpe -0.32, AnnRet -1.63%
- **6E_FRONT_CALENDAR**: Sharpe -0.27, AnnRet -1.28%
- **RTY_FRONT_CALENDAR_2D**: Sharpe -0.15, AnnRet -1.80%

**Key Observations:**
- Rates (ZT, ZN, ZF) show positive Sharpe ratios
- SR3 performs exceptionally well (highest Sharpe)
- CL shows strong returns with moderate Sharpe
- UB underperforms significantly (negative Sharpe)
- FX (6B, 6E) underperform; 6J is positive
- Equities mixed (ES, NQ positive; RTY negative)

**Verdict**: Residual trend idea shows positive alpha over full period. Eligible for Phase-1.

#### 4.1.5 Phase-1 Implementation

**Objective**: Convert residual trend into a proper atomic sleeve within the Trend Meta-Sleeve.

**Implementation Steps:**

1. **Feature Computation**:
   - Add `ResidualTrendFeatures` class to compute residual trend features
   - Compute raw residual return: `resid_ret = long_ret - short_ret`
   - Compute cross-sectional z-score: `resid_ret_z` (normalized across assets each day)
   - Register features in `FeatureService` as `RESIDUAL_TREND`

2. **Strategy Class**:
   - Create `ResidualTrendStrategy` class in `src/agents/strat_residual_trend.py`
   - Use z-scored residual trend feature (`trend_resid_ret_252_21_z`)
   - Apply clipped z-score: `signal = clip(z, -3, 3)` (keeps magnitude information)
   - Return signals as `pd.Series` indexed by symbol

3. **Integration**:
   - Wire `ResidualTrendStrategy` into `CombinedStrategy`
   - Add experimental strategy profile: `core_v3_trend_plus_residual_experiment`
   - Start with small weight (0.20) on residual trend, 0.80 on trend_multihorizon
   - Keep production profile (`core_v3_no_macro`) unchanged

4. **Phase-1 Diagnostics**:
   - Create `scripts/run_residual_trend_phase1.py` for Phase-1 testing
   - Run over same window (2021-01-01 to 2025-10-31)
   - Compare against baseline using `run_perf_diagnostics.py`
   - Save artifacts to `reports/runs/<run_id>/`

**Expected Improvements Over Phase-0:**
- Cross-sectional z-scoring should improve signal quality
- Clipped z-scores preserve magnitude information (vs pure sign-only)
- Integration with Trend Meta-Sleeve allows blending with other atomic sleeves

**Phase-1 Profile**: `core_v3_trend_plus_residual_experiment` (experimental, off by default)

#### 4.1.6 Phase-1 Results

**Status**: ✅ PASSED

**Experimental Configuration**: 80% Trend Multi-Horizon + 20% Residual Trend (external blending)

**Performance Metrics:**
- **CAGR**: -0.49% (vs -2.59% baseline) → **+2.10% improvement**
- **Sharpe**: 0.03 (vs -0.15 baseline) → **+0.18 improvement** ✨ **Now positive!**
- **MaxDD**: -36.80% (vs -39.69% baseline) → **+2.89% improvement**
- **Vol**: 12.93% (vs 12.31% baseline) → +0.62% (slight increase)
- **HitRate**: 50.83% (vs 49.77% baseline) → **+1.06% improvement**
- **Equity Ratio**: **1.136x** (13.6% better performance over full period)

**Year-by-Year Performance:**
- **2021**: CAGR 21.5% (vs 18.8% baseline) → +2.7%
- **2022**: CAGR -11.6% (vs -12.0% baseline) → +0.4%
- **2023**: CAGR -6.4% (vs -6.2% baseline) → -0.2%
- **2024**: CAGR -0.7% (vs -5.2% baseline) → **+4.5%** (significant improvement)
- **2025**: CAGR -1.8% (vs -5.6% baseline) → **+3.8%** (significant improvement)

**Key Findings:**
- Residual Trend meaningfully improves strategy performance at 20% weight
- Reduces losses (CAGR closer to breakeven)
- Turns negative Sharpe positive
- Reduces maximum drawdown
- Improves hit rate
- Consistently helps in recent years (2024-2025), where trend-following struggled

**Verdict**: ✅ **Phase-1 PASSED** — Residual Trend validated as a valuable atomic sleeve component.

#### 4.1.7 Phase-2 Results

**Status**: ✅ PASSED

Residual Trend was promoted from experimental external blending to an internal atomic sleeve within the Trend Meta-Sleeve.

**Promotion Rationale:**
1. **Economically Different Signal**: Captures "long-horizon trend minus short-term noise" — distinct from pure momentum
2. **Validated Performance**: Phase-0 Sharpe 0.29, Phase-1 +2.10% CAGR improvement, +0.18 Sharpe improvement
3. **Architectural Fit**: All trend-related atomic ideas should be inside the Trend Meta-Sleeve
4. **Robustness**: Performance improvement consistent across years, reduces drawdowns

**Implementation**: Integrated into `TSMOMMultiHorizonStrategy` as 4th atomic sleeve.

#### 4.1.8 Promotion Status

**Status**: ✅ **PRODUCTION** (integrated in Trend Meta-Sleeve v2)

**Note**: Currently not included in production profile `core_v3_no_macro`, but available as internal atomic sleeve for future use.

---

### Sleeve 2 — Breakout (50-100d)

#### 4.2.1 Economic Idea

**Breakout (50–100d)** captures price breakouts in the medium-term horizon using Donchian-style range analysis. The core hypothesis is:

- **Volatility Expansions**: Breakouts above recent ranges often signal the start of trends or volatility regime shifts
- **Complementary to Return-Based Momentum**: While existing sleeves use returns and slopes, breakouts emphasize position within historical range
- **Regime Shift Detection**: Breakout signals are sensitive to transitions from consolidation to trending markets

**Intended Benefits:**
- Low/moderate correlation to existing long/medium/short return-based sleeves
- Better performance in volatility expansion regimes and breakout scenarios
- Enhanced left-tail protection when markets transition from consolidation to trend

**Academic Grounding:**
- Donchian channel breakouts (traditional trend-following)
- Baz et al. (2015): Breakout-style signals add diversification to return-based momentum
- Range-based indicators capture different market dynamics than pure returns

#### 4.2.2 Data Requirements

**Uses only existing continuous prices:**
- No additional data sources required
- No database schema changes
- No FRED or macro data
- No options data
- Works with the same 13-contract universe used in production

#### 4.2.3 Phase-0 Design

The Phase-0 implementation uses a minimal sign-only approach to validate the core economic idea:

**For each asset and date `t`:**

1. **Compute 100-day breakout score**:
   ```
   high_100 = max(price_{t-99}..price_t)
   low_100 = min(price_{t-99}..price_t)
   breakout_100 = (price_t - low_100) / max(high_100 - low_100, eps)
   ```
   - Result: 0.0 (at low) to 1.0 (at high)

2. **Compute 50-day breakout score** similarly:
   ```
   high_50 = max(price_{t-49}..price_t)
   low_50 = min(price_{t-49}..price_t)
   breakout_50 = (price_t - low_50) / max(high_50 - low_50, eps)
   ```

3. **Combine into single breakout score**:
   ```
   breakout_mid = 0.5 * breakout_50 + 0.5 * breakout_100
   ```

4. **Signal mapping (sign-only)**:
   - If `breakout_mid > 0.55` → `signal = +1` (upper breakout)
   - If `breakout_mid < 0.45` → `signal = -1` (lower breakout)
   - Else → `signal = 0` (neutral band)

5. **Portfolio construction**:
   - Equal-weight across assets each day
   - Daily rebalancing
   - No overlays (no vol targeting, no z-scoring, no macro filters)

**Default Parameters:**
- `lookback_50`: 50 days
- `lookback_100`: 100 days
- `upper_threshold`: 0.55
- `lower_threshold`: 0.45
- `eps`: 1e-8 (avoid division by zero)

#### 4.2.4 Phase-0 Results

**Status**: ✅ PASSED

**Target Window**: 2020-01-01 to 2025-11-19

**Results:**
- **Sharpe**: 0.246 (passes threshold ≥ 0.20)
- **CAGR**: 1.01%
- **Vol**: 4.48%
- **MaxDD**: -10.70%
- **HitRate**: 20.11%

**Key Findings:**
- Strong performance in rates (ZT, ZN, ZF) with Sharpe 0.50-0.62
- Moderate performance in commodities (GC, CL)
- Weak performance in equities (ES, NQ)
- Low hit rate (20%) expected for sign-only with neutral band thresholds

**Verdict**: Breakout mid idea shows positive alpha. Eligible for Phase-1.

#### 4.2.5 Phase-1 Implementation

**Objective**: Convert breakout into a proper atomic sleeve within the Trend Meta-Sleeve.

**Implementation Steps:**

1. **Feature Computation**:
   - Add `mom_breakout_mid_50_z_{symbol}` feature to `feature_long_momentum.py`
   - Add `mom_breakout_mid_100_z_{symbol}` feature to `feature_long_momentum.py`
   - Compute raw breakout scores using 50/100-day lookbacks
   - Apply time-series standardization (252-day rolling z-score, clipped at ±3.0)

2. **Strategy Integration**:
   - Extend `TSMOMMultiHorizon` to load breakout features
   - Create `breakout_mid_50_100` atomic sleeve signal:
     ```
     breakout_mid_signal = w_50 * mom_breakout_mid_50_z + w_100 * mom_breakout_mid_100_z
     ```
   - Add to meta-signal blend with new horizon weight
   - Start with `w_50 = 0.5`, `w_100 = 0.5`

3. **Configuration**:
   - Add `breakout_mid_50_100` to `horizon_weights` in `configs/strategies.yaml`
   - Add `feature_weights.breakout_mid_50_100` config block
   - Create new strategy profile: `core_v3_trend_breakout`

4. **Phase-1 Diagnostics**:
   - Run full backtest with `core_v3_trend_breakout` profile
   - Compare against baseline `core_v3_no_macro` using `run_perf_diagnostics.py`
   - Save artifacts to `reports/runs/<run_id>/`

**Expected Improvements Over Phase-0:**
- Time-series z-scoring should improve signal quality
- Integration with Trend Meta-Sleeve allows blending with other atomic sleeves
- Vol normalization layer provides risk-adjusted signals

**Phase-1 Profile**: `core_v3_trend_breakout` (experimental, off by default)

#### 4.2.6 Phase-1 Results

**Status**: ❌ FAILED (initial integration), ✅ PASSED (Phase-1B refinement)

**Phase-1 Initial Test (10% weight, 50/50 feature weights):**
- **Sharpe**: -0.038 vs baseline 0.086 (delta: -0.124)
- **CAGR**: -1.15% vs baseline 0.29% (delta: -1.45%)
- **MaxDD**: -35.01% vs baseline -32.02% (delta: -2.99%)
- **Verdict**: Integration failure - weight too high, signal style conflicts

**Phase-1B Refinement Tests:**

Tested 4 configurations with 3% horizon weight:

1. **70/30 (50d/100d)**: ✅ **PASSED**
   - Sharpe: 0.0953 vs baseline 0.0857 (+0.0095)
   - MaxDD: -31.52% vs baseline -32.02% (+0.50%)
   - CAGR: 0.42% vs baseline 0.29% (+0.12%)
   - Vol: 12.35% vs baseline 12.44% (-0.09%)
   - **Verdict**: Clear winner, passes all criteria

2. **30/70 (50d/100d)**: ❌ FAILED
   - Sharpe: -0.1502 vs baseline 0.0857
   - MaxDD: -39.00% vs baseline -32.02%
   - Verdict: 100d-heavy blend underperforms

3. **Pure 50d (100/0)**: ❌ FAILED
   - Sharpe: -0.1396 vs baseline 0.0857
   - MaxDD: -38.60% vs baseline -32.02%
   - Verdict: Pure 50d insufficient

4. **Pure 100d (0/100)**: ❌ FAILED
   - Sharpe: -0.1547 vs baseline 0.0857
   - MaxDD: -39.15% vs baseline -32.02%
   - Verdict: Pure 100d worst performer

**Phase-1B Key Findings:**
- 3% horizon weight is appropriate (vs 10% in Phase-1)
- 50d breakout more effective than 100d in this setup
- 70/30 blend (50d/100d) optimal - outperforms pure configurations
- Lower weight reduces signal conflicts with existing sleeves

**Phase-1B Winner**: 70/30 configuration promoted to Phase-2

#### 4.2.7 Phase-2 Results

**Status**: ✅ PASSED

**Target Window**: 2021-01-01 to 2025-11-19

**Configuration:**
- Horizon weight: 3% for `breakout_mid_50_100`
- Feature weights: 70% `breakout_50`, 30% `breakout_100`
- Adjusted horizon weights: `long=0.485, med=0.291, short=0.194, breakout_mid=0.03`
- Profile: `core_v3_trend_breakout_70_30_phase2`

**Portfolio Metrics** (vs Baseline `core_v3_no_macro_phase1b_baseline`):

| Metric | Baseline | Phase-2 | Delta | Status |
|--------|----------|---------|-------|--------|
| **Sharpe** | 0.0857 | **0.0953** | **+0.0095** | ✅ Better |
| **MaxDD** | -32.02% | **-31.52%** | **+0.50%** | ✅ Better |
| **CAGR** | 0.29% | **0.42%** | **+0.12%** | ✅ Better |
| **Vol** | 12.44% | **12.35%** | **-0.09%** | ✅ Better |
| **HitRate** | 52.11% | **52.37%** | **+0.26%** | ✅ Better |

**Equity Curve Comparison:**
- Equity ratio (Current/Baseline): Final = 1.0073 (0.73% outperformance)
- Mean equity ratio: 1.0027
- Consistent outperformance across full period

**Year-by-Year Performance:**
- **2021**: 21.10% CAGR (vs baseline strong performance) - Better
- **2022**: -9.79% CAGR (vs baseline weak) - Better
- **2023**: -5.46% CAGR (vs baseline weak) - Better
- **2024**: 1.36% CAGR (vs baseline weak) - Better
- **2025**: -2.12% CAGR (vs baseline weak) - Better

Phase-2 outperforms baseline in all years.

**Correlation Analysis:**
- Results match Phase-1B Test 1 (70/30) exactly, confirming configuration stability
- No degradation in performance vs Phase-1B results
- Improvements preserved across full backtest period

**Why 70/30 Works:**
1. **50d Dominance**: 50-day breakout captures medium-term volatility expansions more effectively than 100d in this setup
2. **Complementary Blend**: 30% 100d provides stability and reduces whipsaws from pure 50d
3. **Optimal Weight**: 3% horizon weight is sufficient to contribute alpha without conflicting with existing sleeves
4. **Signal Quality**: 70/30 blend outperforms both pure 50d and pure 100d, indicating synergy between horizons

**Promotion Rationale:**
- ✅ **Sharpe ≥ baseline**: 0.0953 ≥ 0.0857 (passes)
- ✅ **MaxDD ≤ baseline**: -31.52% ≤ -32.02% (passes)
- ✅ **Robust across years**: Outperforms in all 5 years
- ✅ **Stable correlations**: Configuration reproducible and stable
- ✅ **Acceptable drawdown shape**: MaxDD improvement maintained

**Verdict**: ✅ **Phase-2 PASSED** — Breakout Mid (50-100d) approved for Phase-3 (production monitoring).

#### 4.2.8 Phase-3 Results

**Status**: ✅ PASSED

**Target Window**: Production integration (2025-11)

**Configuration:**
- Horizon weight: 3% for `breakout_mid_50_100`
- Feature weights: 70% `breakout_50`, 30% `breakout_100`
- Integrated into `core_v3_no_macro` production profile

**Phase-3 Actions:**
- Updated production configuration (`configs/strategies.yaml`)
- Integrated into `core_v3_no_macro` profile
- Documentation updated across all relevant files
- All promotion criteria validated and confirmed

**Verdict**: ✅ **Phase-3 PASSED** — Breakout Mid (50-100d) promoted to production.

#### 4.2.9 Promotion Status

**Status**: ✅ **PRODUCTION** (Phase-0/1B/2/3 passed, integrated in core_v3_no_macro)

**Production Configuration:**
- Horizon weight: 3% for `breakout_mid_50_100`
- Feature weights: 70% `breakout_50`, 30% `breakout_100`
- Profile: `core_v3_no_macro` (production baseline)

---

### Sleeve 3 — Persistence (Momentum-of-Momentum)

#### 4.3.1 Economic Idea

**Persistence (Momentum-of-Momentum)** captures the observation that the rate of change of trend itself is predictive. If momentum is strengthening, that is bullish; if momentum is fading, that is bearish.

**Academic Grounding:**
- Baz et al. (2015), "Dissecting Investment Strategies": persistence improves trend stability
- Hurst/Ooi/Pedersen: multi-horizon slope acceleration matters

**Expected Behavior:**
- Helps during strong trend extensions (2020, early 2021)
- Stabilizes whipsaw periods where simple horizons disagree
- Weakens during flattening momentum, ahead of reversals

#### 4.3.2 Phase-0 Design

**Purpose**: Validate that persistence has any positive economic edge before adding complexity.

**Variants Tested:**
1. **Return Acceleration**: `persistence_raw = ret_84[t] - ret_84[t-21]`, `signal = sign(persistence_raw)`
2. **Slope Acceleration**: `slope_now = EMA20 - EMA84`, `slope_old = EMA20.shift(21) - EMA84.shift(21)`, `signal = sign(slope_now - slope_old)`
3. **Breakout Acceleration**: `breakout_now = breakout_126[t]`, `breakout_old = breakout_126[t-21]`, `signal = sign(breakout_now - breakout_old)`

#### 4.3.3 Phase-0 Results

**Target Window**: 2021-01-01 to 2025-10-31

| Variant | CAGR | Sharpe | Vol | MaxDD | HitRate | Verdict |
|---------|------|--------|-----|-------|---------|---------|
| Return Acceleration | -0.95% | **-0.23** | 4.15% | -12.99% | 20.81% | ❌ FAIL |
| Slope Acceleration | +0.83% | **0.22** | 3.85% | -5.81% | 22.00% | ✅ PASS |
| Breakout Acceleration | -0.54% | **-0.14** | 3.99% | -11.58% | 21.21% | ❌ FAIL |

**Phase-0 Conclusion**: Only slope acceleration variant passed the minimum Sharpe threshold (≥0.2). Return and breakout acceleration variants failed.

#### 4.3.4 Phase-1 Implementation

**Implementation**: Created `MomentumPersistence` strategy class with three standardized components:

1. **Slope Acceleration** (primary, 80% weight): `accel_slope = zscore((EMA20[t] - EMA84[t]) - (EMA20[t-21] - EMA84[t-21]))`
2. **Breakout Acceleration** (10% weight): `accel_breakout = zscore(breakout_126[t] - breakout_126[t-21])`
3. **Return Acceleration** (10% weight): `accel_ret = zscore(ret_84[t] - ret_84[t-21])`

**Signal Construction:**
```
persistence_signal_raw = 0.80 * accel_slope + 0.10 * accel_breakout + 0.10 * accel_ret
persistence_signal_z = cross_sectional_zscore(persistence_signal_raw)
persistence_signal = clip(persistence_signal_z, -3, +3)
```

#### 4.3.5 Phase-1 Results

**Target Window**: 2021-01-01 to 2025-10-31

**Standalone Persistence (100% Persistence):**
- **CAGR**: -0.04%
- **Sharpe**: 0.0645
- **Vol**: 13.38%
- **MaxDD**: -29.55%
- **HitRate**: 51.29%

**External Blend (80% Trend + 20% Persistence):**
- **CAGR**: 0.31% (vs baseline 0.44%)
- **Sharpe**: 0.0871 (vs baseline 0.0973)
- **Vol**: 12.43%
- **MaxDD**: -32.79% (vs baseline -32.02%)
- **HitRate**: 52.75%

**Baseline Trend (core_v3_no_macro):**
- **CAGR**: 0.44%
- **Sharpe**: 0.0973
- **Vol**: 12.46%
- **MaxDD**: -32.02%
- **HitRate**: 52.15%

**Phase-1 Conclusion**: 
- Standalone persistence showed weak performance (Sharpe 0.06)
- External blend **underperformed** baseline Trend (Sharpe decline from 0.097 to 0.087)
- Persistence did not add value when blended externally with Trend Meta-Sleeve

**Year-by-Year Breakdown (External Blend):**
- **2021**: Strong (CAGR 19.65%, Sharpe 1.70) — persistence helped during trend extension
- **2022**: Weak (CAGR -9.58%, Sharpe -0.55) — persistence hurt during volatility clustering
- **2023**: Weak (CAGR -6.94%, Sharpe -0.53) — persistence hurt during whipsaw
- **2024**: Flat (CAGR 1.34%, Sharpe 0.17)
- **2025**: Flat (CAGR 0.38%, Sharpe 0.01)

#### 4.3.6 Why Persistence Failed

**1. Small Universe (13 Assets)**
- Persistence effects are known to be weak in small universes
- Academic papers showing strong persistence use 35-60 futures across diverse asset classes
- Current universe lacks sufficient cross-asset diversity

**2. Challenging Market Regime (2021-2025)**
- Sample period dominated by:
  - COVID dislocation (2021)
  - 2022 inflation shock
  - 2023-24 low-momentum whipsaw regime
- Persistence is notoriously weak in:
  - Sharp reversals
  - Sideways chop
  - Volatility clustering periods (like 2022-23)
- Persistence shines best in:
  - Long sustained trends (2010-2020)
  - Macro cycles (carry/trend coherent periods)

**3. Feature Construction**
- Current implementation may not capture persistence effects optimally
- May require different acceleration windows or normalization approaches
- May need to be combined with other features (e.g., vol-adjusted trend, cross-asset confirmation)

#### 4.3.7 Promotion Status

**Status**: ❌ **PARKED** (not deleted, documented for future re-test)

**Re-Test Conditions**:

Persistence will be automatically re-tested when any of the following occur:

1. **Universe Expansion**: When the traded universe grows by ≥25% (e.g., from 13 to 20+ assets)
   - Persistence effects are stronger in broad, diverse universes
   - More cross-asset validation opportunities

2. **Historical Expansion**: When total backtest length increases by ≥5 years (e.g., extending to 2010 or earlier)
   - Access to longer sustained trend periods (2010-2020)
   - More macro cycle data (carry/trend coherent periods)

3. **Architecture Change**: When a closely related Trend component is added or changed
   - Examples: vol-adjusted trend, cross-asset trend confirmation, short-term reversal filters
   - Persistence may add value only when combined with these enhancements

4. **Feature Platform Upgrades**: When FeatureService adds materially new features relevant to persistence
   - Cross-asset trend validation
   - Multi-horizon acceleration features
   - Volatility-adjusted acceleration

5. **Market Regime Shifts**: When more than 3 years of new live data accumulate
   - Re-test to see if new regime supports persistence effects

**Re-Test Procedure**: If any trigger occurs, repeat Phase-0 and Phase-1. If both pass, Persistence becomes eligible for Phase-2 integration following standard procedures.

**Implementation Notes**:
- **Files Created**: `src/agents/strat_momentum_persistence.py`, `scripts/run_persistence_phase1.py`
- **Files Modified**: `src/agents/feature_service.py`, `run_strategy.py`, `src/agents/strat_combined.py`, `configs/strategies.yaml`
- **Run IDs for Reference**: Phase-0: `reports/sanity_checks/trend/persistence/20251118_180815/` (slope_accel variant)

**Decision**: Do NOT integrate Persistence into Trend Meta-Sleeve. Marked as experimental and parked.

**Rationale**: Academic idea is valid, but current universe/time period does not support it. Revisit when conditions improve (universe expansion, historical depth, or related feature additions).

---

## 5. Appendix

### 5.1 Related Documentation

- **`SOTs/STRATEGY.md`**: Describes the architecture and execution flow of the production Trend Meta-Sleeve
- **`SOTs/DIAGNOSTICS.md`**: Documents the diagnostics framework and Phase-0 sanity check methodology
- **`SOTs/PROCEDURES.md`**: Outlines the sleeve lifecycle (Phase-0 → Phase-3) and development procedures
- **`TREND_IMPLEMENTATION.md`**: Detailed implementation reference for the Trend Meta-Sleeve

### 5.2 Implementation Files

**Strategy Classes:**
- `src/agents/strat_tsmom_multihorizon.py`: Main Trend Meta-Sleeve implementation
- `src/agents/strat_residual_trend.py`: Residual Trend atomic sleeve
- `src/agents/strat_momentum_persistence.py`: Persistence atomic sleeve (parked)

**Feature Computation:**
- `src/agents/feature_long_momentum.py`: Computes momentum and breakout features

**Diagnostic Scripts:**
- `scripts/run_trend_med_canonical_phase0.py`: Phase-0 sanity check for Canonical Medium-Term
- `scripts/run_trend_med_canonical_phase1.py`: Phase-1 diagnostics for Canonical Medium-Term
- `scripts/run_trend_breakout_mid_sanity.py`: Phase-0 sanity check for Breakout
- `scripts/run_residual_trend_sanity.py`: Phase-0 sanity check for Residual Trend
- `scripts/run_persistence_sanity.py`: Phase-0 sanity check for Persistence
- `scripts/run_residual_trend_phase1.py`: Phase-1 diagnostics for Residual Trend
- `scripts/run_persistence_phase1.py`: Phase-1 diagnostics for Persistence
- `scripts/run_breakout_phase1b.py`: Phase-1B diagnostics for Breakout

### 5.3 Configuration

**Strategy Profiles** (in `configs/strategies.yaml`):
- `core_v3_no_macro`: Production baseline (includes canonical Long-Term, canonical Medium-Term, and Breakout Mid)
- `core_v3_no_macro_legacy`: Legacy baseline (preserves legacy Long-Term and legacy Medium-Term for historical comparison)
- `core_v3_trend_medcanon_no_macro`: Phase-2 validation profile for canonical Medium-Term
- `core_v3_trend_breakout_70_30_phase2`: Phase-2 validation profile for Breakout
- `core_v3_trend_plus_residual_experiment`: Residual Trend external blend (experimental)
- `core_v3_trend_v2_with_residual`: Residual Trend internal integration (experimental)

### 5.4 Results Locations

**Phase-0 Sanity Checks:**
- `reports/sanity_checks/trend/medium_canonical/<timestamp>/`
- `reports/sanity_checks/trend/residual_trend/<timestamp>/`
- `reports/sanity_checks/trend/breakout_mid_50_100/<timestamp>/`
- `reports/sanity_checks/trend/persistence/<timestamp>/`

**Phase-1/2 Backtests:**
- `reports/runs/<run_id>/`

---

**Last Updated**: November 2025
