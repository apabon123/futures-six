# Phase 1C Sanity Check Report

**Date:** 2026-01-09  
**Status:** 🚨 **CRITICAL ISSUES FOUND**

---

## Executive Summary

The A/B backtests ran successfully, but analysis reveals **two critical issues** that must be addressed before declaring Phase 1C complete:

1. **❌ Allocator-H was NOT actually applied in "RT + Alloc-H" scenario**
2. **⚠️  Risk Targeting vol gap needs explanation**

---

## Issue 1: Allocator-H Was Not Applied (CRITICAL)

### What We Discovered

The "RT + Alloc-H" scenario **did not actually apply the allocator**. Returns are 100% identical to "RT only":

```
RT only total return:     -0.008769
RT + Alloc-H total return: -0.008769
Difference: 0.0 (exactly zero)
```

### Root Cause

The terminal logs show:
```
[ExecSim] Allocator v1 is DISABLED (mode='off')
Risk scalars applied: 0/52 rebalances (0.0%)
Average scalar: 1.000
```

**But the allocator artifacts show active intervention:**
- 42.3% of rebalances had multiplier < 0.999
- Min multiplier: 0.678 (very aggressive!)
- Mean multiplier: 0.960

**This means:**
- Allocator artifacts were **computed** correctly
- But the multipliers were **not applied** to weights
- The system ran in "artifacts-only" mode

### Why This Happened

The A/B script sets:
```python
'allocator_v1.enabled': True,
'allocator_v1.mode': 'compute',
```

But the base `configs/strategies.yaml` has:
```yaml
allocator_v1:
  enabled: true
  mode: "precomputed"
  precomputed_run_id: null
```

**The config override is not working correctly.** The system is falling back to the base config, which has `mode: "precomputed"` with `precomputed_run_id: null`, causing it to default to 'off' mode.

### Fix Required

1. **Immediate Fix:** Verify that config overrides in `run_phase1c_ab_backtests.py` are actually being written to the temp config file
2. **Verification:** Re-run just the "RT + Alloc-H" scenario and confirm:
   - Terminal logs show: `Allocator v1 ENABLED (mode='compute')`
   - Returns differ from "RT only"
   - `ExecSim` summary shows scalars were applied

---

## Issue 2: Risk Targeting Vol Gap (Needs Explanation)

### What We Observed

|  | Expected | Actual | Gap |
|---|---|---|---|
| **Target Vol** | 20% | 20% | - |
| **Realized Vol** | ~20%? | **7.28%** | **-12.72%** |
| **Avg Leverage** | ~4.0× | 2.75× | -1.25× |

### Analysis: Why Is Realized Vol So Low?

#### Factor 1: Vol Floor is Binding Early in the Year

```
First 10 rebalances (Jan-Mar 2024):
- Portfolio vol estimate: 5.0% (hitting vol_floor!)
- Leverage: 4.0× (capped)
```

**This is correct behavior** - when portfolio vol < vol_floor (5%), the system uses vol_floor to calculate leverage:
```
leverage = target_vol / vol_floor = 20% / 5% = 4.0×
```

#### Factor 2: Leverage is Applied Only on Rebalance Dates (Weekly)

- RT computes leverage and scales weights **only on Friday rebalances**
- Between rebalances, weights are held constant
- This creates "leverage drift" as asset prices move
- Result: **effective leverage ≠ target leverage** between rebalances

#### Factor 3: Gross Exposure Adjustment

From the realized vol artifact:
```
Portfolio vol estimates range: 5.0% - 11.4%
Median vol: 7.6%
```

With avg leverage 2.75× and median portfolio vol 7.6%:
```
Expected realized vol = 2.75 × 7.6% = 20.9%
```

**But actual realized vol = 7.28%**

This 3× gap suggests:
1. The vol estimate is using **pre-leverage** returns
2. OR the leverage is not being applied correctly to all assets
3. OR there's a gross exposure normalization happening

### What We Need to Check

1. **Weights alignment:**
   ```python
   # For one representative date:
   weights_pre = pd.read_csv('.../weights_pre_risk_targeting.csv')
   weights_post = pd.read_csv('.../weights_post_risk_targeting.csv')
   leverage = pd.read_csv('.../leverage_series.csv')
   
   # Check: weights_post ≈ weights_pre × leverage?
   ```

2. **Gross exposure check:**
   ```python
   # Check if gross is being normalized back down
   gross_pre = weights_pre.abs().sum()
   gross_post = weights_post.abs().sum()
   
   # Expected: gross_post ≈ gross_pre × leverage
   # If gross_post ≈ gross_pre, RT is being undone somewhere
   ```

3. **Vol calculation source:**
   - Is realized_vol computed from pre-RT or post-RT returns?
   - This would explain the 3× gap

### Acceptance Criterion

**We don't need realized vol = 20% exactly.** But we do need to understand:
- Why the gap exists
- Confirm it's not a bug (e.g., RT being overwritten)
- Document the expected behavior

**Likely outcome:** "RT sizing is rebalance-frequency only, so realized vol will be lower than target due to leverage drift between rebalances. This is expected behavior for weekly rebalancing."

---

## Issue 3: Artifact Integrity Check (Recommended)

### What to Verify

Pick **one representative date** (e.g., 2024-11-14, a top drawdown day) and verify:

1. **RT artifacts align:**
   ```
   Date: 2024-11-14
   - leverage_series.csv: leverage = X
   - weights_pre: [asset weights]
   - weights_post: [should equal weights_pre × X]
   ```

2. **Allocator artifacts align:**
   ```
   Date: 2024-11-14
   - allocator_regime_v1.csv: regime = Y
   - allocator_risk_v1.csv: multiplier = Z
   - (When re-run with allocator enabled:)
     weights_final should equal weights_post × Z
   ```

3. **Returns calculation:**
   ```
   portfolio_return[date] = sum(weights_final[asset] × asset_return[date])
   ```

This spot-check confirms the artifact wiring is correct.

---

## Corrected A/B Results (Provisional)

### Current State (❌ INVALID)

| Scenario | CAGR | Vol | Sharpe | MaxDD |
|----------|------|-----|--------|-------|
| Baseline | -3.23% | 10.15% | -0.32 | -9.80% |
| RT only | -0.96% | 7.28% | -0.13 | -7.30% |
| **RT + Alloc-H** | **-0.96%** | **7.28%** | **-0.13** | **-7.30%** |

**❌ RT + Alloc-H is identical to RT only** - allocator was not applied!

### Expected State (After Fix)

| Scenario | CAGR | Vol | Sharpe | MaxDD | Notes |
|----------|------|-----|--------|-------|-------|
| Baseline | -3.23% | 10.15% | -0.32 | -9.80% | ✅ Correct |
| RT only | -0.96% | 7.28% | -0.13 | -7.30% | ✅ Correct |
| **RT + Alloc-H** | **TBD** | **TBD** | **TBD** | **TBD** | ⏳ Needs re-run with allocator actually enabled |

**Expected behavior when fixed:**
- CAGR: Slightly worse than RT only (allocator drag)
- Vol: Slightly lower than RT only (allocator reduces exposure in stress)
- Sharpe: Slightly worse (drag dominates in low-vol year)
- MaxDD: **Should be better** (this is the key test - tail protection)

---

## Action Items (Before Phase 1C Closure)

### 1. Fix A/B Script Config Overrides ⏰ **URGENT**

**Problem:** Config overrides are not being applied correctly.

**Investigation steps:**
1. Check `run_phase1c_ab_backtests.py` lines 80-120 (config override logic)
2. Verify temp config file is being created with correct values
3. Confirm `--config_path` is being passed to `run_strategy.py`
4. Add debug logging to print the actual config being used

**Quick fix:**
```python
# In run_phase1c_ab_backtests.py, after creating temp config:
with open(tmp_config, 'r') as f:
    print(f"DEBUG: Temp config contents:\n{f.read()}")
```

### 2. Re-Run "RT + Alloc-H" Scenario ⏰ **URGENT**

```bash
# Manual test to verify allocator works:
python run_strategy.py \
  --strategy_profile core_v9_... \
  --start 2024-01-01 \
  --end 2024-12-31 \
  --run_id test_rt_alloc_h_manual
  
# Then check terminal logs for:
# "[ExecSim] Allocator v1 ENABLED (mode='compute')"
# "Risk scalars applied: X/52 rebalances"
```

### 3. Verify RT Weights Alignment ⏰ **HIGH**

```python
# Spot-check one date:
import pandas as pd

date = '2024-11-14'
run_dir = 'reports/runs/...rt_only_2024-01-01_2024-12-31/'

weights_pre = pd.read_csv(f'{run_dir}/risk_targeting/weights_pre_risk_targeting.csv', index_col=0)
weights_post = pd.read_csv(f'{run_dir}/risk_targeting/weights_post_risk_targeting.csv', index_col=0)
leverage = pd.read_csv(f'{run_dir}/risk_targeting/leverage_series.csv', index_col=0)

lev = leverage.loc[date, 'leverage']
pre = weights_pre.loc[date]
post = weights_post.loc[date]

print(f"Leverage: {lev:.2f}×")
print(f"Gross pre: {pre.abs().sum():.2f}")
print(f"Gross post: {post.abs().sum():.2f}")
print(f"Expected gross post: {pre.abs().sum() * lev:.2f}")
print(f"\nWeights ratio (post/pre): {(post / pre).dropna().unique()}")
```

**Expected output:**
```
Leverage: 2.64×
Gross pre: 1.45
Gross post: 3.83   # Should equal 1.45 × 2.64
Expected gross post: 3.83
Weights ratio: [2.64, 2.64, 2.64, ...]  # All assets scaled by same factor
```

If the actual gross_post ≠ expected, we have a bug.

### 4. Document Vol Gap Explanation ⏰ **MEDIUM**

Once we understand the vol gap (likely just rebalance-frequency effect), document it in `PHASE_1C_RESULTS_ANALYSIS.md`:

```markdown
## Why Is Realized Vol 7.3% vs Target 20%?

RT targets 20% volatility, but applies leverage only on rebalance dates (weekly).
Between rebalances, weights are held constant, causing "leverage drift" as prices move.

Additionally, the portfolio vol estimate is computed from pre-leverage returns,
so the mapping from "target vol → leverage" is correct, but the realized vol
of the final (post-leverage) portfolio will differ due to:
1. Rebalance frequency (weekly vs daily)
2. Leverage drift between rebalances
3. Asset correlation changes

This is expected behavior for weekly rebalancing with daily return measurement.
```

---

## Phase 1C Completion Checklist (REVISED)

| Item | Status | Notes |
|------|--------|-------|
| RT artifacts exist and line up with weights deltas | ⏳ **NEEDS VERIFICATION** | Must verify weights_post = weights_pre × leverage |
| Allocator profile tests enforce behavior | ✅ DONE | Tests pass |
| Allocator artifacts show regimes/multipliers per day | ✅ DONE | Artifacts are correct |
| **Allocator is actually applied in RT + Alloc-H scenario** | ❌ **FAILED** | **A/B script config override bug** |
| A/B backtests run from one command | ✅ DONE | Script works |
| A/B backtests generate consistent outputs | ⚠️  **PARTIAL** | Outputs generated, but RT + Alloc-H is wrong |
| RT in correct layer order (Layer 5) | ✅ DONE | Confirmed in logs |
| Contract tests prevent RT regressions | ✅ DONE | Tests pass |
| Activation tests prevent allocator regressions | ✅ DONE | Tests pass |
| **Vol gap is understood and documented** | ⏳ **NEEDS INVESTIGATION** | Must verify RT is not being overwritten |

---

## Bottom Line

**Phase 1C is NOT complete yet.** We have:

1. ✅ **Good news:** RT layer works, allocator artifacts are correct, system architecture is sound
2. ❌ **Bad news:** A/B script has a config override bug preventing allocator from being applied
3. ⏰ **Action:** Fix config override, re-run RT + Alloc-H, verify weights alignment, then close Phase 1C

**Estimated time to fix:** 1-2 hours

---

**Next Steps:**
1. Fix A/B script config override logic
2. Re-run RT + Alloc-H scenario
3. Verify weights alignment for one representative date
4. Update PHASE_1C_RESULTS_ANALYSIS.md with corrected results
5. **Then** Phase 1C is done and we can proceed to Phase 2

---

**Signed off by:** AI Agent  
**Date:** 2026-01-09  
**Status:** 🚨 PHASE 1C INCOMPLETE - CONFIG BUG FOUND

