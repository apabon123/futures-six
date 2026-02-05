# Carry Phase-1 Run Memo

**Date**: January 21, 2026  
**Run ID**: `carry_phase1_v1_20260121_155610`  
**Config**: `configs/carry_phase1_v1.yaml`  
**Period**: 2020-01-01 to 2025-10-31

---

## Executive Summary

**STATUS**: ❌ **PHASE-1 FAILED** (Sharpe = -0.248, negative)

Carry Meta-Sleeve v1 Phase-1 backtest completed. The strategy shows **negative Sharpe** (-0.248), worse than Phase-0 (0.181). However, **FX and Rates asset classes are positive**, suggesting the issue is with Equity/Commodity carry or vol normalization.

**Key Finding**: FX carry (Sharpe = 0.686) and Rates carry (Sharpe = 0.244) are working, but Equity carry (Sharpe = -0.537) is strongly negative, dragging down the portfolio.

---

## Phase-1 Compliance

**✅ COMPLIANT**:
- Risk Targeting: `enabled: false` ✅
- Allocator: `enabled: false` ✅
- Engine Policy: `enabled: false` ✅
- Carry Phase: `1` ✅
- Evaluation Layer: Post-Construction (belief layer) ✅

**Requested Start**: 2020-01-01  
**Effective Start**: 2020-03-20 (first rebalance after warmup)  
**Valid Rows**: 1,822 observations  
**Rows Dropped**: Minimal (NA handling at feature level)

---

## Headline Metrics (Post-Construction)

| Metric | Value | vs Phase-0 |
|--------|-------|------------|
| **Observations** | 1,822 trading days | Same |
| **Annualized Return** | **-1.77%** | ⚠️ Negative (Phase-0: +2.01%) |
| **Annualized Volatility** | 7.14% | ✅ Lower (Phase-0: 11.13%) |
| **Sharpe Ratio** | **-0.248** | ❌ Worse (Phase-0: 0.181) |
| **Max Drawdown** | -24.79% | ✅ Similar (Phase-0: -25.81%) |
| **Best Day** | +2.61% | ⚠️ Lower (Phase-0: +4.02%) |
| **Worst Day** | -7.93% | ✅ Better (Phase-0: -8.78%) |
| **Skewness** | -3.516 | ⚠️ Worse (Phase-0: -1.271) |
| **Kurtosis** | 62.205 | ⚠️ Worse (Phase-0: 18.378) |
| **Positive Days** | 881 (48.4%) | ⚠️ Lower (Phase-0: 50.1%) |
| **Final Equity** | $0.86 | ❌ Loss (Phase-0: $1.11) |

**Overall**: ❌ **FAILED** (Negative Sharpe, negative return)

---

## Year-by-Year Breakdown

| Year | Sharpe | Return | Vol | Status |
|------|--------|--------|-----|--------|
| **2020** | 0.031 | 0.10% | 3.32% | ⚠️ Near zero |
| **2021** | -0.857 | -3.25% | 3.79% | ❌ Negative |
| **2022** | 1.429 | +9.18% | 6.42% | ✅ Strong |
| **2023** | -0.229 | -1.56% | 6.80% | ❌ Negative |
| **2024** | 0.310 | +1.31% | 4.22% | ✅ Positive |
| **2025** | -1.366 | -19.26% | 14.11% | ❌ Very negative |

**Key Observations**:
- **2022 was strong** (Sharpe = 1.429, Return = +9.18%) - rates shock period
- **2025 is very negative** (Sharpe = -1.366, Return = -19.26%) - likely data issue or regime change
- **2021 and 2023 are negative** - consistent underperformance

---

## Stress Windows

| Window | Cumulative Return | Acceptable | Status |
|--------|-------------------|------------|--------|
| **2020 Q1** | +1.67% | ✅ Yes (> -20%) | ✅ Excellent |
| **2022** | +11.71% | ✅ Yes (> -30%) | ✅ Excellent |

**Crisis Behavior**: ✅ **EXCELLENT** (both stress windows positive)

---

## Asset-Class Contributions

| Asset Class | Cumulative Return | Sharpe | Symbols | Status |
|-------------|-------------------|--------|---------|--------|
| **Equity** | -1.53% | -0.537 | 3 (ES, NQ, RTY) | ❌ Strongly negative |
| **FX** | +1.85% | 0.686 | 3 (6E, 6B, 6J) | ✅ Strongly positive |
| **Rates** | +0.87% | 0.244 | 4 (ZT, ZF, ZN, UB) | ✅ Positive |
| **Commodity** | -0.41% | -0.094 | 2 (CL, GC) | ⚠️ Slightly negative |

**Key Finding**: 
- ✅ **FX carry is working** (Sharpe = 0.686)
- ✅ **Rates carry is working** (Sharpe = 0.244)
- ❌ **Equity carry is broken** (Sharpe = -0.537)
- ⚠️ **Commodity carry is weak** (Sharpe = -0.094)

**Diversification**: ✅ 2 asset classes positive (FX, Rates)

---

## Dominance Diagnostics

### Average Absolute Weight Per Asset

**Top 5 Assets** (by avg |weight|):
1. ZT_FRONT_VOLUME: 0.2497 (24.97%)
2. ZF_FRONT_VOLUME: 0.2131 (21.31%)
3. ZN_FRONT_VOLUME: 0.1688 (16.88%)
4. 6J_FRONT_CALENDAR: 0.1391 (13.91%)
5. 6E_FRONT_CALENDAR: 0.1233 (12.33%)

**Observation**: ⚠️ **Rates dominate** (ZT, ZF, ZN = 63% combined). This suggests vol normalization is pushing risk toward low-vol rates markets despite vol floor.

### Crisis Window Risk Share (Mar-May 2020)

**Top 5 Assets** (by |w| × vol):
1. CL_FRONT_VOLUME: 0.0319
2. 6B_FRONT_CALENDAR: 0.0287
3. 6J_FRONT_CALENDAR: 0.0148
4. 6E_FRONT_CALENDAR: 0.0111
5. GC_FRONT_VOLUME: 0.0059

**Observation**: During crisis, commodities and FX had higher risk share (vol was elevated), but rates still dominated average weights.

---

## Phase-1 Implementation Notes

### Z-Score Consistency

**Equity/Rates**: Re-z-scored in meta-sleeve (252d rolling window, ±3.0 clip)  
**FX/Commodity**: Use pre-computed `carry_ts_z_{root}` from feature module (252d rolling window, ±3.0 clip)

**Status**: ✅ Consistent (same 252d window, same clipping)

### Vol Normalization

**Method**: `signal = z_score / asset_vol` (carry strength per unit realized vol)

**Vol Floor**: Applied (max of 5th percentile of asset's vol history or 4% minimum)

**Unit Gross Scaling**: Applied (sum of absolute values = 1.0)

**Observation**: ⚠️ Rates still dominate despite vol floor. Possible causes:
1. Vol floor too low (4% may be below actual rates vol)
2. Vol normalization logic inverted (should be `signal = z_score * vol` for equal risk?)
3. Rates z-scores are systematically larger

---

## Root Cause Analysis

### ✅ Hypothesis 1: Equity Carry Sign Error — **RULED OUT**

**Evidence**: Correlation (raw_ES vs signal_ES) = 0.6845 (positive)  
**Same Sign**: 801/1219 (65.7%) — sign logic is correct

**Conclusion**: Sign logic is correct. Issue is elsewhere.

### ✅ Hypothesis 2: Vol Normalization Causing Dominance — **CONFIRMED**

**Evidence**: 
- ES Vol (median): 15.9% (0.1590)
- ZT Vol (median): 1.82% (0.0182)
- Vol ratio: 8.72x (ES vol is 8.72x higher than ZT)

**Impact**: 
- ES signal = z / 0.159 ≈ 6.3 × z
- ZT signal = z / 0.018 ≈ 55 × z (8.7x larger for same z-score)

**Vol Floor Applied**: max(5th percentile, 4%) — helps but doesn't fully solve

**Conclusion**: Vol normalization (`signal = z / vol`) pushes risk toward low-vol assets (rates), causing rates to dominate (63% combined weight). This is expected behavior, but may need adjustment.

### ⚠️ Hypothesis 3: 2025 Data Issue — **NEEDS INVESTIGATION**

**Evidence**: 2025 return = -19.26% (very negative, Sharpe = -1.366)

**Possible Causes**:
1. Data quality issue in 2025
2. Regime change not captured
3. Feature computation error
4. Vol spike in 2025 (14.11% vs 4-7% in other years)

**Action**: Inspect 2025 data quality, check feature values, verify vol calculations

### Hypothesis 4: Equity Signals Too Small

**Evidence**: 
- Equity signals are very small (0.0000, 0.0142 in samples)
- After vol normalization and unit gross scaling, equity contributes little
- When equity does contribute, it may be wrong due to timing/alignment

**Possible Causes**:
1. High equity vol makes signals small after normalization
2. Unit gross scaling further reduces equity weight
3. Timing/alignment issue between features and returns

**Action**: Check if disabling equity carry improves overall Sharpe

---

## Comparison: Phase-0 vs Phase-1

| Aspect | Phase-0 | Phase-1 | Change |
|--------|---------|---------|--------|
| **Sharpe** | 0.181 | -0.248 | ❌ -0.429 worse |
| **Return** | +2.01% | -1.77% | ❌ -3.78% worse |
| **Vol** | 11.13% | 7.14% | ✅ -3.99% better |
| **MaxDD** | -25.81% | -24.79% | ✅ +1.02% better |
| **2020 Q1** | -1.56% | +1.67% | ✅ +3.23% better |
| **2022** | N/A | +11.71% | ✅ Strong |

**Key Insight**: Phase-1 **reduces volatility** (good) but **flips return negative** (bad). This suggests:
- Vol normalization is working (lower vol)
- But signal direction or magnitude is wrong (negative return)

---

## Acceptance Criteria Evaluation

| Criterion | Target | Actual | Status |
|-----------|--------|--------|--------|
| **Sharpe ≥ 0.25** | Recommended | -0.248 | ❌ **FAIL** |
| **Sharpe 0.20-0.25** | Conditional | -0.248 | ❌ **FAIL** |
| **MaxDD ≥ -30%** | Acceptable | -24.79% | ✅ **PASS** |
| **2020 Q1 > -20%** | Crisis behavior | +1.67% | ✅ **PASS** |
| **2022 > -30%** | Stress behavior | +11.71% | ✅ **PASS** |
| **≥2 Asset Classes Positive** | Diversified | 2 (FX, Rates) | ✅ **PASS** |

**Overall**: ❌ **FAILED** (Sharpe < 0.20)

---

## Next Steps (Diagnostic)

### Immediate Actions

1. **Check Equity Carry Sign Logic**
   - Print raw equity carry values for 5-10 dates
   - Verify: positive carry → long, negative carry → short
   - Check SOFR conversion (percentage vs decimal)

2. **Inspect Vol Normalization**
   - Print vol values used per asset (min/median/max)
   - Verify vol floor application
   - Check if normalization should be inverted (`z * vol` vs `z / vol`)

3. **Investigate 2025 Data**
   - Check data quality in 2025
   - Inspect feature values for anomalies
   - Verify no computation errors

4. **Asset-Level Analysis**
   - Compute per-asset Sharpe (ES, NQ, RTY individually)
   - Identify which equity asset is worst
   - Check if issue is systematic or asset-specific

### Possible Fixes

**Option A: Increase Vol Floor** ⭐ **RECOMMENDED**
- Raise vol floor from 4% to 6-8% (or use 10th percentile instead of 5th)
- This reduces the vol ratio between equity and rates
- Re-run Phase-1

**Option B: Alternative Vol Normalization**
- Use `signal = z_score * sqrt(vol)` for more balanced scaling
- Or use `signal = z_score / sqrt(vol)` for less aggressive scaling
- Re-run Phase-1

**Option C: Disable Equity Carry (Diagnostic)**
- Temporarily disable equity carry to isolate the issue
- Run with FX + Rates + Commodity only
- If Sharpe improves significantly, equity carry is the problem
- If Sharpe stays negative, issue is elsewhere

**Option D: Check 2025 Data Quality**
- Inspect 2025 data for anomalies
- Check if 2025 negative return is data-driven or strategy-driven
- Exclude 2025 and re-run if data quality issue found

---

## Vol Normalization Diagnostic

**Units Clarification**:
- Z-score: Standard deviations of carry (dimensionless)
- Vol: Annualized volatility (e.g., 0.15 = 15%)
- Signal: `z_score / vol` = carry strength per unit realized vol

**Vol Floor**: Applied (max of 5th percentile or 4% minimum)

**Risk Share**: `|signal| × vol` should be equal per asset (unit risk)

**Actual Vol Values** (from diagnostic):
- ES Vol: min=10.12%, median=15.90%, max=48.56%
- ZT Vol: min=0.25%, median=1.82%, max=2.94%
- Vol Ratio (ES/ZT): 8.72x

**Impact**: 
- ES signals are 8.7x smaller than ZT signals for same z-score
- After unit gross scaling, rates dominate (63% combined weight)
- Equity signals become very small (often near zero)

**Vol Floor Effect**: 
- ZT vol floor raises from 1.82% to 4% (2.2x increase)
- But ES vol is still 4x higher (15.9% vs 4%)
- So rates still get 4x larger signals than equity

**Recommendation**: Consider higher vol floor (e.g., 6-8%) or alternative normalization

---

## FX/Commodity Z-Score Consistency

**Status**: ✅ **CONSISTENT**

- FX/Commodity z-scores computed in `FxCommodCarryFeatures` with:
  - 252d rolling window
  - ±3.0 clipping
  - Same NA handling rules

- Meta-sleeve consumes `carry_ts_z_{root}` directly (no re-z-scoring)

- Equity/Rates z-scores computed in meta-sleeve with:
  - 252d rolling window
  - ±3.0 clipping
  - Same NA handling rules

**Result**: All asset classes use identical z-score definition (252d, ±3.0 clip)

---

## Conclusion

Phase-1 implementation is **functionally correct** (z-scoring, clipping, vol normalization all working), but the **strategy performance is negative** (Sharpe = -0.248).

**Root Cause**: Equity carry is strongly negative (Sharpe = -0.537), dragging down the portfolio despite positive FX and Rates contributions.

**Recommendation**: 
1. **Immediate**: Investigate equity carry sign logic and vol normalization
2. **Diagnostic**: Print raw equity carry values, vol values, and per-asset Sharpe
3. **Fix**: Correct sign error or vol normalization if found
4. **Re-run**: Phase-1 after fix

**Do NOT proceed to Phase-2** until Phase-1 passes (Sharpe ≥ 0.20).

---

**End of Memo**
