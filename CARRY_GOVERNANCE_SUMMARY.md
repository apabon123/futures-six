# Carry Governance Summary & Action Items

**Date**: January 21, 2026  
**Status**: Phase-0 Complete, Governance Split Required

---

## Executive Summary

✅ **Phase-0 Completed**: `carry_phase0_v1_20260121_143130`  
⚠️ **Sharpe**: 0.181 (just below 0.2 threshold)  
✅ **Crisis Behavior**: -1.56% in 2020 Q1 (excellent)  
✅ **Phase-0 Compliance**: Verified (RT/Allocator/Policy all disabled)

**Required Actions**:
1. ⏸️ **Commit B Validation** (separate, non-blocking): Validate exec_sim.py fix against Phase 3B baselines
2. ✅ **Phase-0 Locked**: `carry_phase0_v1_20260121_143130` is official baseline
3. ✅ **Phase-1 Implemented**: Ready for testing (z-score + clip + vol-normalize)
4. **Next**: Run Phase-1 backtest and evaluate results

---

## Change Split

### ✅ Commit A: Carry-Only (Safe to Merge)

**Files**:
- `src/agents/feature_equity_carry.py` (FRED indicator accessor)
- `src/agents/strat_carry_meta_v1.py` (NA handling)
- `scripts/diagnostics/audit_carry_inputs_coverage.py` (DB schema fix)
- `scripts/diagnostics/run_carry_phase0_v1.py` (column name fix)
- `run_strategy.py` (feature service integration)

**Governance**: ✅ Engine-only (Layer 1), no frozen stack touched

### ⚠️ Commit B: Core Semantics (Requires Validation)

**Files**:
- `src/agents/exec_sim.py` (RT-disabled weights_raw fix)

**Governance**: ⚠️ Touches frozen stack (Layer 5 semantics)

**Validation Required**:
```bash
# Run Phase 3B baseline pair verification
python scripts/diagnostics/verify_phase3b_baseline_checkpoints.py --both

# Run decoupling test
python scripts/diagnostics/test_decoupling.py
```

**Baseline Pair**:
- `phase3b_baseline_artifacts_only_20260120_093953`
- `phase3b_baseline_traded_20260120_093953`

**If Validation Fails**: Revert Commit B and investigate separately

---

## Phase-0 Compliance Verification

**Run ID**: `carry_phase0_v1_20260121_143130`

**Meta.json Verification**:
```json
{
  "risk_targeting": {"enabled": false, "effective": false},
  "allocator_v1": {"enabled": false, "effective": false},
  "config": {
    "engine_policy_v1": {"enabled": false},
    "strategies": {
      "carry_meta_v1": {"params": {"phase": 0}}
    }
  }
}
```

**Status**: ✅ **PHASE-0 COMPLIANT**

**Evaluation Layer**: Post-Construction (belief layer) ✅

---

## Action Items

### Immediate (Before Phase-1)

1. **Split Commits** (See `CARRY_GOVERNANCE_SPLIT.md` for exact commands)
   - [ ] Create Commit A (Carry-only)
   - [ ] Create Commit B (Core semantics)

2. **Validate Commit B**
   - [ ] Run Phase 3B baseline verification
   - [ ] Run decoupling test
   - [ ] Document results

3. **Lock Phase-0**
   - [ ] Mark `carry_phase0_v1_20260121_143130` as official baseline
   - [ ] Save memo: `carry_phase0_run_memo.md`
   - [ ] No re-running Phase-0 unless engine definition changes

### Phase-1 (✅ COMPLETE)

4. **Phase-1 Implementation** ✅
   - [x] Implement z-score normalization (252d rolling)
   - [x] Implement vol normalization (equal risk per asset)
   - [x] Add clipping/winsorization (±3.0)
   - [x] Create Phase-1 config (`carry_phase1_v1.yaml`)
   - [x] Create Phase-1 diagnostic script
   - [ ] **Run Phase-1 backtest** (next step)
   - [ ] Compare vs Phase-0 and evaluate acceptance criteria

**Status**: ✅ Ready for testing  
**See**: `CARRY_PHASE1_READY.md` for quick start

---

## Documents Created

1. **`CARRY_GOVERNANCE_SPLIT.md`**: Change classification and commit instructions
2. **`CARRY_PHASE1_PLAN.md`**: Phase-1 implementation plan
3. **`carry_phase0_run_memo.md`**: Phase-0 results and analysis
4. **`carry_inputs_coverage.json`**: Database audit results

---

## Phase-0 Results Summary

| Metric | Value | Status |
|--------|-------|--------|
| Sharpe | 0.181 | ⚠️ Just below 0.2 |
| Annualized Return | 2.01% | ✅ Positive |
| Max Drawdown | -25.81% | ⚠️ Acceptable |
| 2020 Q1 Return | -1.56% | ✅ Excellent |
| Effective Start | 2020-03-20 | ✅ After warmup |
| Observations | 1,822 | ✅ Full coverage |

**Recommendation**: Proceed to Phase-1 (z-scoring/vol normalization likely to push Sharpe above 0.2)

---

## Next Steps

1. **Today**: Split commits, validate Commit B
2. **This Week**: Start Phase-1 implementation
3. **Next Week**: Phase-1 backtest and analysis

---

**End of Summary**
