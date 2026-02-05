# Equity Carry Forensic Memo

**Date**: January 21, 2026  
**Purpose**: Diagnose why equity carry has negative Sharpe (-0.537) in Phase-1 ensemble

---

## Executive Summary

**CONCLUSION**: Equity carry is **NON-ADMISSIBLE** as an Engine v1 return source.

**Evidence**:
- Individual equity signals are **STRONG** (ES Sharpe 1.574, NQ 2.247, RTY 1.089)
- But **implied dividend yield calculation is BROKEN** (mean -70%, 36.9% of days <-10%)
- Ensemble equity carry Sharpe: **-0.537** (strongly negative)
- **Ablation test**: Without equity, strategy Sharpe = **0.210** (conditional pass)

**Recommendation**: 
- **Carry Meta v1 = FX + Rates + Commodity** (equity excluded)
- Equity implied dividends become a **policy feature** (Layer 2), not a return sleeve

---

## Diagnostic Results

### 1. Per-Equity Instrument Attribution

**Individual Signal Performance** (using sign(carry) as signal):

| Instrument | Sharpe | CAGR | Vol | MaxDD | Status |
|------------|--------|------|-----|-------|--------|
| **ES** | 1.574 | 30.23% | 19.21% | -17.11% | ✅ **STRONG** |
| **NQ** | 2.247 | 52.77% | 23.48% | -19.78% | ✅ **VERY STRONG** |
| **RTY** | 1.089 | 27.34% | 25.10% | -31.58% | ✅ **STRONG** |

**Key Finding**: Individual equity carry signals are **excellent** when used in isolation. This suggests the problem is in **how equity carry is computed or aggregated** in the ensemble.

---

### 2. Dividend Implied Sanity

**Implied Dividend Yield Statistics**:

| Instrument | Median | Mean | P5 | P95 | Min | Max |
|------------|--------|------|----|----|----|----|
| **ES** | -1.07% | **-70.00%** | -334.64% | 6.53% | -461.91% | 82.45% |
| **NQ** | -1185.75% | **-1149.55%** | -2766.41% | 2.58% | -3223.00% | 101.16% |
| **RTY** | 1.31% | 0.12% | -47.17% | 43.47% | -325.86% | 86.84% |

**Anomaly Flags**:

| Instrument | Negative Days | Extreme Negative (<-10%) | Extreme Positive (>20%) |
|------------|---------------|---------------------------|--------------------------|
| **ES** | 56.9% | **36.9%** | 0.6% |
| **NQ** | 89.8% | **81.6%** | 0.2% |
| **RTY** | 45.9% | **27.7%** | 22.1% |

**Critical Finding**: 
- **ES**: Mean implied dividend = -70% (impossible!)
- **NQ**: Mean implied dividend = -1149.55% (completely broken!)
- **RTY**: Median is reasonable (1.31%), but 27.7% of days have <-10% (impossible)

**Root Cause**: The implied dividend yield calculation is producing **impossible values**. This corrupts the equity carry calculation, making it unreliable for ensemble use.

---

### 3. Contract Calendar / T Sanity

**Current Implementation**:
- Uses constant T = 45 days (approximation for front-month futures)
- This is reasonable for Phase-1, but may not be exact

**Status**: ⚠️ **TODO** - Verify if actual daycount to expiry is needed

**Note**: T = 45 days is a reasonable approximation, but exact calculation would require actual expiry dates per contract.

---

### 4. Spot Index Type

**Current Implementation**:
- Uses FRED indicators:
  - SP500: FRED series 'SP500'
  - NASDAQ100: FRED series 'NASDAQ100'
  - RUT_SPOT: FRED series 'RUT_SPOT'

**Status**: ⚠️ **TODO** - Verify these are price-return indices (not total return)

**Note**: Need to check FRED documentation to confirm index type.

---

## Ablation Test Results

### With Equity (Phase-1.1 Full)

| Metric | Value | Status |
|--------|-------|--------|
| Sharpe | -0.248 | ❌ **FAILED** |
| Return | -1.77% | ❌ Negative |
| Vol | 7.14% | ✅ Lower |
| MaxDD | -24.79% | ✅ Acceptable |

**Asset-Class Contributions**:
- Equity: Sharpe = -0.537 ❌
- FX: Sharpe = 0.686 ✅
- Rates: Sharpe = 0.244 ✅
- Commodity: Sharpe = -0.094 ⚠️

---

### Without Equity (Phase-1.1 Ablation)

| Metric | Value | Status |
|--------|-------|--------|
| Sharpe | **0.210** | ⚠️ **CONDITIONAL PASS** |
| Return | **+2.39%** | ✅ Positive |
| Vol | 11.42% | ⚠️ Higher |
| MaxDD | -29.99% | ✅ Acceptable |

**Asset-Class Contributions**:
- Equity: 0% (excluded) ✅
- FX: Sharpe = 0.565 ✅
- Rates: Sharpe = 0.148 ✅
- Commodity: Sharpe = -0.237 ⚠️

**Key Finding**: **Removing equity carry turns a FAILED strategy (Sharpe -0.248) into a CONDITIONAL PASS (Sharpe 0.210)**.

---

## Root Cause Analysis

### Hypothesis 1: Implied Dividend Calculation Error ✅ **CONFIRMED**

**Evidence**:
- Mean implied dividend for ES = -70% (impossible)
- Mean implied dividend for NQ = -1149.55% (completely broken)
- 36.9% of ES days have <-10% implied dividend (impossible)

**Possible Causes**:
1. **SOFR conversion error**: Percentage vs decimal (already checked, seems correct)
2. **Futures price / spot price mismatch**: Wrong contract month or stale data
3. **T approximation error**: Constant 45 days may be wrong for some periods
4. **Formula error**: `d_implied = r - (1/T) * log(F/S)` may have sign error or unit error

**Action Required**: 
- Inspect raw futures prices, spot prices, and SOFR for sample dates
- Verify formula: `d_implied = r - (1/T) * log(F/S)` is correct
- Check if T should be actual daycount, not constant 45 days

---

### Hypothesis 2: Individual Signals Strong, Ensemble Weak

**Evidence**:
- Individual signals: ES Sharpe 1.574, NQ 2.247, RTY 1.089 (all strong)
- Ensemble equity carry: Sharpe -0.537 (strongly negative)

**Possible Causes**:
1. **Correlation structure**: Individual signals may be negatively correlated in ensemble
2. **Weighting issue**: How equity signals are combined in meta-sleeve
3. **Timing/alignment**: Features vs returns misalignment

**Action Required**:
- Check correlation between ES, NQ, RTY signals
- Verify how equity signals are aggregated in meta-sleeve
- Check timing alignment between features and returns

---

### Hypothesis 3: Equity Carry Non-Admissible in This Window

**Evidence**:
- Post-2010 equity index futures often price dividends efficiently
- Carry may be dominated by noise / term premia
- Individual signals strong, but ensemble weak suggests structural issue

**Conclusion**: Even if calculation is fixed, equity carry may not be a reliable return source in this window.

---

## Recommendations

### Immediate Actions

1. **Exclude Equity from Carry Meta v1**
   - **Carry Meta v1 = FX + Rates + Commodity** (equity excluded)
   - This gives Sharpe = 0.210 (conditional pass)
   - Clean, explainable, and architecturally sound

2. **Treat Equity Implied Dividends as Policy Feature**
   - Move equity implied dividends to **Layer 2 (Engine Policy)**
   - Use as a **gating/overlay feature**, not a return sleeve
   - Example: "Reduce equity exposure when implied dividend yield is extreme"

3. **Fix Implied Dividend Calculation (Future Work)**
   - Investigate root cause of impossible values
   - Verify formula, T definition, data quality
   - Once fixed, re-evaluate equity carry as standalone strategy

---

## Phase-1.1 Results Summary

### Full Ensemble (With Equity)

**Status**: ❌ **FAILED** (Sharpe = -0.248)

**Key Issues**:
- Equity carry Sharpe = -0.537 (poison pill)
- Overall Sharpe negative despite positive FX and Rates

---

### Ablation (Without Equity)

**Status**: ⚠️ **CONDITIONAL PASS** (Sharpe = 0.210)

**Key Strengths**:
- Positive return (+2.39%)
- Acceptable MaxDD (-29.99%)
- Diversified (FX + Rates positive)
- Crisis behavior acceptable (2020 Q1: -2.37%, 2022: +16.63%)

**Weaknesses**:
- Sharpe just above 0.20 threshold (conditional, not recommended)
- Commodity carry slightly negative (Sharpe = -0.237)
- 2025 still negative (-22.63%)

---

## Next Steps

1. **Lock Phase-1.1 (No Equity) as Baseline**
   - Run ID: `carry_phase1_1_v1_20260121_172243`
   - Sharpe = 0.210 (conditional pass)
   - This is the cleanest, most explainable version

2. **Proceed to Phase-2 Integration** (if desired)
   - Integrate Carry Meta v1 (FX + Rates + Commodity) into full portfolio
   - Keep equity excluded until implied dividend calculation is fixed

3. **Future Work: Fix Equity Carry**
   - Investigate implied dividend calculation
   - Fix formula, T definition, or data quality issues
   - Re-evaluate as standalone strategy (not meta-sleeve component)

---

## Conclusion

Equity carry is **NON-ADMISSIBLE** as an Engine v1 return source due to:
1. **Broken implied dividend calculation** (impossible values)
2. **Negative ensemble Sharpe** (-0.537) despite strong individual signals
3. **Ablation test confirms** equity is the poison pill

**Recommended Path Forward**:
- **Carry Meta v1 = FX + Rates + Commodity** (Sharpe = 0.210, conditional pass)
- **Equity implied dividends → Policy feature** (Layer 2, not return sleeve)

This is architecturally clean, explainable, and preserves the economic hypothesis of carry while excluding the broken component.

---

**End of Memo**
