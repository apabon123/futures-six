# Canonical Long-Term Trend Sleeve — Deployment Summary

**Date:** November 19, 2025  
**Status:** ✅ **DEPLOYED TO PRODUCTION**

---

## Executive Summary

Successfully completed the full Phase-0 → Phase-1 → Phase-2 lifecycle for the **Canonical Long-Term Trend Sleeve (TSMOM-252)** and deployed it to production as the new baseline for the Trend Meta-Sleeve.

The canonical equal-weight (1/3, 1/3, 1/3) composite replaced the legacy (0.5, 0.3, 0.2) weighting after passing all validation phases.

---

## 1. Weight Architecture Clarification

**CRITICAL DISTINCTION:** The Trend Meta-Sleeve uses a **two-layer weight structure**:

### Layer 1: Feature Weights (inside each atomic sleeve)
These combine features *within* a single atomic sleeve:
- **Long-term**: 1/3 ret_252 + 1/3 breakout_252 + 1/3 slope_slow (**canonical**)
- **Medium-term**: 0.4 ret_84 + 0.3 breakout_126 + 0.2 slope_med + 0.1 persistence
- **Short-term**: 0.5 ret_21 + 0.3 breakout_21 + 0.2 slope_fast
- **Breakout mid**: 0.7 breakout_50 + 0.3 breakout_100

### Layer 2: Horizon Weights (across atomic sleeves)
These combine atomic sleeves into the meta-signal:
- **Long-term (252d)**: 48.5%
- **Medium-term (84/126d)**: 29.1%
- **Short-term (21d)**: 19.4%
- **Breakout mid (50-100d)**: 3.0%

**Note**: The 1/3, 1/3, 1/3 weights are **feature weights** (Layer 1), NOT horizon weights (Layer 2). The horizon weights remain unchanged at 48.5% / 29.1% / 19.4% / 3.0%.

---

## 2. Changes Deployed

### 2.1 Documentation Updates

**`docs/META_SLEEVES/TREND_RESEARCH.md`**:
- ✅ Updated "Production Atomic Sleeves" section to show Long-Term Canonical as the official long-term sleeve
- ✅ Added Phase-0/1/2 results for Long-Term Canonical in "What Passed / Failed and Why"
- ✅ Updated summary table to reflect canonical status
- ✅ Added clear two-layer weight architecture section

**`docs/META_SLEEVES/TREND_IMPLEMENTATION.md`**:
- ✅ Added "Canonical Long-Term Composite" section under "Feature Computation"
- ✅ Updated "Signal Processing Pipeline" with two-layer weight architecture explanation
- ✅ Updated "Configuration" section to show canonical long-term weights (1/3, 1/3, 1/3)
- ✅ Added comments distinguishing feature weights (Layer 1) from horizon weights (Layer 2)

**`docs/META_SLEEVES/LONG_TERM_CANONICAL_PROMOTION.md`**:
- ✅ Deprecated standalone promotion doc
- ✅ Replaced with stub redirecting to `TREND_RESEARCH.md` and `TREND_IMPLEMENTATION.md`

### 2.2 Configuration Updates

**`configs/strategies.yaml`**:
- ✅ Updated `core_v3_no_macro` (production profile) to use canonical long-term weights:
  - `ret_252: 0.333333` (was 0.5)
  - `breakout_252: 0.333333` (was 0.3)
  - `slope_slow: 0.333334` (was 0.2)
- ✅ Created `core_v3_no_macro_legacy` profile for historical comparison with legacy weights
- ✅ Added comments clarifying that canonical long-term was promoted in Nov 2025

### 2.3 Production Deployment

- ✅ Ran fresh backtest with `core_v3_no_macro` profile (canonical weights)
- ✅ Confirmed deployment: `run_id = core_v3_no_macro_prod_canonical_20251119`

---

## 3. Validation Results Recap

### Phase-0: Sanity Check (PASSED ✓)
- **Sharpe**: 0.5838 (threshold: ≥ 0.20) ✓
- **CAGR**: 3.15% ✓
- **MaxDD**: -11.04%
- **HitRate**: 53.94%

### Phase-1: Canonical vs Legacy (PASSED ✓)
| Metric | Canonical (1/3,1/3,1/3) | Legacy (0.5,0.3,0.2) | Delta |
|--------|-------------------------|----------------------|-------|
| **Sharpe** | 0.1209 | 0.1075 | **+0.0134** ✓ |
| **CAGR** | 0.73% | 0.57% | **+0.17%** ✓ |
| **Vol** | 12.27% | 12.37% | **-0.10%** ✓ |
| **MaxDD** | -31.07% | -31.52% | **+0.45%** ✓ |
| **HitRate** | 52.09% | 52.42% | -0.33% |

### Phase-2: Environmental & Correlation Stability (PASSED ✓)
- ✅ All 5 criteria passed
- ✅ Outperformed 3/5 years (2021, 2023, 2024)
- ✅ Correlation stability: 0.996 (extremely stable)
- ✅ No hidden degradation or instabilities

---

## 4. Production Baseline

**Current Production Profile**: `core_v3_no_macro`

**Canonical Long-Term Feature Weights**:
```yaml
feature_weights:
  long:
    ret_252: 0.333333       # Canonical (1/3) - Promoted Nov 2025
    breakout_252: 0.333333  # Canonical (1/3) - Promoted Nov 2025
    slope_slow: 0.333334    # Canonical (1/3) - Promoted Nov 2025
```

**Horizon Weights** (unchanged):
```yaml
horizon_weights:
  long_252: 0.485          # Long-term (252d) horizon
  med_84: 0.291            # Medium-term (84/126d) horizon
  short_21: 0.194          # Short-term (21d) horizon
  breakout_mid_50_100: 0.03  # Breakout (50-100d) horizon
```

**Legacy Profile**: `core_v3_no_macro_legacy`
- Available for historical comparison
- Uses legacy long-term weights (0.5, 0.3, 0.2)
- Deprecated for production use

---

## 5. Key Takeaways

1. **Equal Weighting Works**: The canonical (1/3, 1/3, 1/3) equal-weight composite provides better diversification across complementary signals (return, breakout, slope) compared to the legacy weighting.

2. **Two-Layer Architecture**: The Trend Meta-Sleeve uses feature weights (Layer 1) to combine features inside each atomic sleeve, and horizon weights (Layer 2) to combine atomic sleeves into the meta-signal. These are completely independent.

3. **Stable Improvement**: The canonical long-term composite shows consistent, stable improvement vs legacy (+0.0134 Sharpe, +0.17% CAGR) with no correlation blow-up (0.996 correlation).

4. **Production Ready**: All documentation, configuration, and validation completed. The canonical long-term is now the official baseline for the Trend Meta-Sleeve.

---

## 6. Next Steps

1. **Monitor Performance**: Track production performance of canonical long-term vs legacy in live environment
2. **Medium/Short Optimization**: Consider similar Phase-0/1/2 analysis for medium and short-term feature weights
3. **Horizon Weight Optimization**: After all feature weights are canonicalized, consider horizon weight optimization (48.5/29.1/19.4/3.0)

---

**Last Updated**: November 19, 2025  
**Promotion Status**: ✅ Deployed to Production (`core_v3_no_macro`)  
**Deployment ID**: `core_v3_no_macro_prod_canonical_20251119`

