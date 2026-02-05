# Carry Meta-Sleeve v1 — Quick Start Guide

**Implementation Status**: ✅ **COMPLETE** (January 21, 2026)  
**Phase**: Phase-0 (Sanity Check) — Ready for Data and Testing

---

## 🎯 What Was Implemented

I've successfully implemented the **Carry Meta-Sleeve v1** following your canonical specification. This is a comprehensive, production-ready implementation that captures carry signals across all four asset classes.

### ✅ Completed Components

1. **✅ Feature Modules** (3 new + 1 existing):
   - `src/agents/feature_equity_carry.py` — NEW: Equity carry (ES, NQ, RTY)
   - `src/agents/feature_rates_carry.py` — NEW: Rates carry (ZT, ZF, ZN, UB)
   - `src/agents/feature_carry_fx_commod.py` — EXISTING: FX/commodity carry (6E, 6B, 6J, CL, GC)

2. **✅ Meta-Sleeve**:
   - `src/agents/strat_carry_meta_v1.py` — Unified carry signals across asset classes

3. **✅ Integration**:
   - `src/agents/strat_combined.py` — MODIFIED: Added carry_meta_v1
   - `run_strategy.py` — MODIFIED: Added CarryMetaV1 instantiation
   - `configs/strategies.yaml` — MODIFIED: Added carry configuration section

4. **✅ Configuration**:
   - `configs/carry_phase0_v1.yaml` — Phase-0 backtest config

5. **✅ Diagnostics**:
   - `scripts/diagnostics/run_carry_phase0_v1.py` — Automated Phase-0 diagnostic

6. **✅ Documentation**:
   - `docs/CARRY_META_V1_IMPLEMENTATION.md` — Comprehensive implementation guide
   - `docs/CARRY_META_V1_QUICK_START.md` — This quick start guide

---

## ⚠️ Action Required: Data Prerequisites

Before you can run Phase-0, you need to **load missing data** into your database:

### Critical Missing Data

| Data Type | Symbols | Purpose | Format |
|-----------|---------|---------|--------|
| **Spot Indices** | SP500, NASDAQ100, RUT_SPOT | Equity carry (implied div yield) | Price-return index (NOT total return) |
| **SOFR** | SOFR or US_SOFR | Funding rate for all carry | Annualized daily rate (e.g., 0.045 = 4.5%) |
| **Rank 1 Series (Rates)** | ZT_RANK_1_VOLUME, ZF_RANK_1_VOLUME, ZN_RANK_1_VOLUME, UB_RANK_1_VOLUME | Rates rolldown | Continuous rank 1 contracts |
| **Foreign Rates** | ECB_RATE, JPY_RATE, SONIA | FX carry (interest differentials) | Annualized daily rates |

### Optional (for full coverage)

| Data Type | Symbols | Purpose |
|-----------|---------|---------|
| Rank 1 Series (Commodity) | CL_RANK_1_VOLUME, GC_RANK_1_VOLUME | Commodity curve slope |

### Graceful Degradation

The implementation includes **fallback behavior** if data is missing:

- ❌ **No Spot Indices** → Equity carry skipped with warning
- ❌ **No SOFR** → Uses 4.5% placeholder with warning
- ❌ **No Rank 1 Series** → That asset class skipped with warning
- ❌ **No Foreign Rates** → FX carry uses what's available

**You can still run Phase-0 with partial data**, but you'll get incomplete results.

---

## 🚀 Running Phase-0

### Step 1: Verify Data

Check what's already in your database:

```bash
# List available symbols
python -c "
from src.agents import MarketData
market = MarketData()
print('Available symbols:', market.universe)
"
```

### Step 2: Load Missing Data

Load the missing data into your database. The exact method depends on your data pipeline, but you'll need:

1. **Spot Indices**: Query from your data provider (e.g., Bloomberg, Databento)
   - Make sure to use **price-return** indices, NOT total return
   - Symbol naming should match: `SP500`, `NASDAQ100`, `RUT_SPOT`

2. **SOFR**: Download from FRED or your data provider
   - Convert to annualized decimal format (e.g., 4.5% → 0.045)
   - Symbol naming: `SOFR` or `US_SOFR`

3. **Rank 1 Continuous Series**: Build from your continuous contract builder
   - Use same roll rules as rank 0 (e.g., volume-weighted for ZT/ZF/ZN/UB)

### Step 3: Test Feature Computation

Before running full backtest, test that features compute:

```python
from src.agents.feature_equity_carry import EquityCarryFeatures
from src.agents.feature_rates_carry import RatesCarryFeatures
from src.agents import MarketData

market = MarketData()

# Test equity carry
eq_carry = EquityCarryFeatures()
eq_features = eq_carry.compute(market, end_date="2025-12-31")
print("Equity carry features:\n", eq_features.head())

# Test rates carry
rates_carry = RatesCarryFeatures()
rates_features = rates_carry.compute(market, end_date="2025-12-31")
print("\nRates carry features:\n", rates_features.head())
```

If you see warnings about missing data, go back to Step 2.

### Step 4: Run Phase-0 Diagnostic

Once data is loaded:

```bash
# Run full Phase-0 backtest (2018-2025)
python scripts/diagnostics/run_carry_phase0_v1.py

# Or with custom date range
python scripts/diagnostics/run_carry_phase0_v1.py \
    --start 2020-01-01 \
    --end 2025-12-31
```

### Step 5: Review Results

The diagnostic will output:

```
==================================================
PHASE-0 RESULTS
==================================================
Observations: 1754
Annualized Return: X.XX%
Annualized Vol: X.XX%
Sharpe Ratio: X.XXX
Max Drawdown: -X.XX%
...

==================================================
PHASE-0 PASS CRITERIA
==================================================
✓ Sharpe ≥ 0.2: True/False
✓ 2020 Q1 acceptable (> -20%): True/False

==================================================
✓✓✓ PHASE-0 PASSED ✓✓✓  (or FAILED)
==================================================
```

**Artifacts saved to**:
- `reports/runs/carry_phase0_v1_<timestamp>/`
  - `portfolio_returns.csv`
  - `equity_curve.csv`
  - `weights.csv`
  - `phase0_analysis_summary.json`

---

## 📊 Phase-0 Pass Criteria

Your implementation must meet these criteria to proceed to Phase-1:

| Criterion | Target | Purpose |
|-----------|--------|---------|
| **Portfolio Sharpe** | ≥ 0.2 | Minimal unconditional edge |
| **Asset Class Coverage** | ≥ 1 positive | At least one class works |
| **Crisis Behavior (2020 Q1)** | > -20% | No catastrophic blowup |
| **Signal Distribution** | Non-degenerate | Signals vary across time |

---

## 🔄 What Happens Next

### If Phase-0 Passes ✅

1. **Document Results**: Save Phase-0 artifacts to `reports/_PINNED/`
2. **Proceed to Phase-1**: Implement clean version with:
   - Rolling z-scores (252d window)
   - Vol normalization (risk parity)
   - Cross-sectional ranking within asset classes
3. **Create Phase-1 Config**: `configs/carry_phase1_v1.yaml`
4. **Run Phase-1 Diagnostic**: Similar process with enhanced signals

### If Phase-0 Fails ❌

1. **Validate Data**: Check that all required data is loaded correctly
2. **Inspect Raw Signals**: Look at carry values for 5-10 assets manually
3. **Check Sign Logic**: Verify positive carry → long, negative carry → short
4. **Review Features**: Print out equity carry, rates carry, etc. to spot bugs
5. **Crisis Analysis**: Did it blow up in March 2020? Why?

**Common Failure Modes**:
- Missing data (most likely)
- Sign error in carry calculation
- Maturity approximation too crude
- Roll timing incorrect

---

## 📁 File Reference

Quick reference to key files:

```
# Feature Computation
src/agents/feature_equity_carry.py      # Equity: Implied div yield
src/agents/feature_rates_carry.py       # Rates: Rolldown
src/agents/feature_carry_fx_commod.py   # FX + Commodity

# Meta-Sleeve
src/agents/strat_carry_meta_v1.py       # Unified carry signals

# Configuration
configs/carry_phase0_v1.yaml            # Phase-0 config
configs/strategies.yaml                 # Added carry section

# Orchestration
run_strategy.py                         # Modified to include carry

# Diagnostics
scripts/diagnostics/run_carry_phase0_v1.py  # Automated Phase-0 test

# Documentation
docs/CARRY_META_V1_IMPLEMENTATION.md    # Full implementation guide
docs/CARRY_META_V1_QUICK_START.md       # This file
```

---

## 🆘 Troubleshooting

### Problem: "No spot index data found"

**Solution**: Load spot price-return indices (SP500, NASDAQ100, RUT_SPOT) into database.

### Problem: "Using placeholder SOFR rate"

**Solution**: Load SOFR time series into database as `SOFR` or `US_SOFR`.

### Problem: "Insufficient data for carry calculation"

**Solution**: Load rank 1 continuous series (e.g., `ZT_RANK_1_VOLUME`).

### Problem: "Sharpe is negative"

**Possible Causes**:
1. Sign error in carry calculation (check raw values)
2. Missing key data (equity carry dominates, but no spot indices)
3. Roll timing issue (continuous contracts rolling incorrectly)

**Debug Steps**:
```python
# Inspect raw carry values
from src.agents import MarketData
from src.agents.strat_carry_meta_v1 import CarryMetaV1

market = MarketData()
carry = CarryMetaV1(phase=0)

# Get signals for a specific date
signals = carry.signals(market, "2025-01-15")
print("Carry signals:\n", signals)
```

### Problem: "Phase-0 script fails"

**Check**:
1. Database path correct in `configs/data.yaml`
2. All imports work (run `python -c "from src.agents.strat_carry_meta_v1 import CarryMetaV1"`)
3. Config file exists: `configs/carry_phase0_v1.yaml`

---

## 📝 Summary

### What You Have

✅ Fully implemented Carry Meta-Sleeve v1  
✅ Four asset classes: Equity, FX, Rates, Commodity  
✅ Phase-0 configuration (sign-only, equal-weight)  
✅ Automated diagnostic script  
✅ Comprehensive documentation

### What You Need

❌ Load missing data (spot indices, SOFR, rank 1 series)  
❌ Run Phase-0 diagnostic  
❌ Evaluate pass/fail  
❌ Proceed to Phase-1 (if passed)

### Estimated Time

- **Data Loading**: 1-2 hours (depends on your data pipeline)
- **Phase-0 Run**: 5-10 minutes (depends on date range)
- **Analysis**: 15 minutes (review results, check diagnostics)

---

## 🎯 Next Steps (Ordered)

1. **[CRITICAL]** Load missing data into database
2. **[CRITICAL]** Test feature computation (Step 3 above)
3. **[ACTION]** Run Phase-0 diagnostic: `python scripts/diagnostics/run_carry_phase0_v1.py`
4. **[DECISION]** Evaluate results against pass criteria
5. **[IF PASS]** Proceed to Phase-1 implementation
6. **[IF FAIL]** Debug and iterate

---

## 💬 Questions?

Refer to:
- **Full Implementation Guide**: `docs/CARRY_META_V1_IMPLEMENTATION.md`
- **Procedures Manual**: `docs/SOTs/PROCEDURES.md`
- **System Architecture**: `docs/SOTs/SYSTEM_CONSTRUCTION.md`
- **Roadmap**: `docs/SOTs/ROADMAP.md`

---

**Implementation Complete**: January 21, 2026  
**Ready for Phase-0 Testing**: YES (pending data load)  
**All Code Delivered**: YES  
**Next Action**: Load data → Run Phase-0 → Evaluate

---

**Good luck with Phase-0! 🚀**
