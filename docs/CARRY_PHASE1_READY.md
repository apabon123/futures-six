# Carry Phase-1 Implementation — Ready for Testing

**Date**: January 21, 2026  
**Status**: ✅ **IMPLEMENTED & READY**

---

## Implementation Complete

✅ **Phase-1 transforms implemented**:
- Step B: Rolling z-score (252d window) per instrument
- Step C: Clip at ±3.0
- Step D: Vol-normalize within sleeve (equal risk per asset, unit gross scaling)
- Step E: Cross-sectional ranking (deferred to Phase-1.1)

✅ **Configuration**: `configs/carry_phase1_v1.yaml`  
✅ **Diagnostic Script**: `scripts/diagnostics/run_carry_phase1_v1.py`  
✅ **Code**: `src/agents/strat_carry_meta_v1.py` → `_signals_phase1()`

---

## Signal Validation

**Test Results** (2025-01-15):
- ✅ Signals in reasonable range: [-0.25, +0.25]
- ✅ Unit gross: Sum of absolute values = 1.0
- ✅ All asset classes represented
- ✅ Vol normalization working correctly

**Sample Signals**:
```
ES_FRONT_CALENDAR_2D    -0.183
NQ_FRONT_CALENDAR_2D    -0.123
RTY_FRONT_CALENDAR_2D   -0.030
6E_FRONT_CALENDAR       -0.043
6B_FRONT_CALENDAR        0.022
6J_FRONT_CALENDAR        0.014
ZT_FRONT_VOLUME         -0.252
ZF_FRONT_VOLUME          0.097
ZN_FRONT_VOLUME          0.124
UB_FRONT_VOLUME          0.014
CL_FRONT_VOLUME         -0.085
GC_FRONT_VOLUME         -0.013
```

---

## Running Phase-1

### Quick Test

```bash
python scripts/diagnostics/run_carry_phase1_v1.py \
    --start 2020-01-01 \
    --end 2025-10-31 \
    --config configs/carry_phase1_v1.yaml
```

### Expected Output

- Phase-1 compliance check (RT/Allocator/Policy disabled)
- Performance metrics (Sharpe, MaxDD, etc.)
- Year-by-year breakdown
- Stress window analysis (2020 Q1, 2022)
- Pass/fail determination

---

## Acceptance Criteria

### Pass (Recommended)
- ✅ Sharpe ≥ 0.25
- ✅ MaxDD ≥ -30% (or better than Phase-0)
- ✅ No single asset dominates risk
- ✅ Crisis behavior sane

### Conditional Pass
- ⚠️ Sharpe 0.20-0.25
- ✅ ≥2 asset classes positive
- ✅ Crisis behavior sane

### Fail
- ❌ Sharpe < 0.20
- ❌ Equity beta behavior
- ❌ Single instrument dominates

---

## Commit B Validation (Separate)

**⚠️ IMPORTANT**: Commit B (exec_sim.py RT-disabled fix) must be validated separately:

```bash
# On branch with only Commit B
python scripts/diagnostics/verify_phase3b_baseline_checkpoints.py --both
python scripts/diagnostics/test_decoupling.py
```

**Pass Condition**: All 7 checkpoints green (identity/no-sidecar/allocator coherence/decoupling)

**If Fails**: Revert Commit B before it touches main

**Status**: ⏸️ Pending user validation (not blocking Phase-1)

---

## Phase-1 vs Phase-0 Comparison

| Metric | Phase-0 | Phase-1 Target |
|--------|---------|----------------|
| Sharpe | 0.181 | ≥ 0.25 (recommended) |
| Signal Type | Sign-only | Z-scored + vol-normalized |
| Normalization | None | Rolling z-score (252d) |
| Vol Scaling | None | Equal risk per asset |
| Gross Exposure | Variable | Unit gross (sum abs = 1.0) |

---

## Next Steps

1. **Run Phase-1 Backtest**: Execute diagnostic script
2. **Evaluate Results**: Compare vs Phase-0 and acceptance criteria
3. **If Pass**: Proceed to Phase-2 (Integration)
4. **If Conditional**: Review asset-class contributions
5. **If Fail**: Investigate (sign logic, vol normalization, data quality)

---

## Files Modified (Phase-1 Only)

**Engine Layer (Carry-only)**:
- ✅ `src/agents/strat_carry_meta_v1.py` (Phase-1 implementation)
- ✅ `configs/carry_phase1_v1.yaml` (Phase-1 config)
- ✅ `scripts/diagnostics/run_carry_phase1_v1.py` (Phase-1 diagnostic)

**No frozen stack changes** ✅

---

**Ready to run Phase-1! 🚀**
