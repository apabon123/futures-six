# Carry Meta-Sleeve v1 — Next Actions Checklist

**Date**: January 21, 2026  
**Status**: Phase-0 Complete, Phase-1 Implemented

---

## Immediate Actions (In Order)

### 1. ⏸️ Commit B Validation (Separate, Non-Blocking)

**Action**: Validate `exec_sim.py` RT-disabled fix on branch with only Commit B

**Commands**:
```bash
# On branch with Commit B only
python scripts/diagnostics/verify_phase3b_baseline_checkpoints.py --both
python scripts/diagnostics/test_decoupling.py
```

**Pass Condition**: All 7 checkpoints green

**If Fails**: Revert Commit B, investigate separately

**Status**: ⏸️ Pending user validation (not blocking Phase-1)

---

### 2. ✅ Run Phase-1 Backtest

**Action**: Execute Phase-1 diagnostic script

**Command**:
```bash
python scripts/diagnostics/run_carry_phase1_v1.py \
    --start 2020-01-01 \
    --end 2025-10-31 \
    --config configs/carry_phase1_v1.yaml
```

**Expected Output**:
- Phase-1 compliance check
- Performance metrics (Sharpe, MaxDD, etc.)
- Year-by-year breakdown
- Stress window analysis
- Pass/fail determination

**Time Estimate**: 5-10 minutes

---

### 3. 📊 Evaluate Phase-1 Results

**Check Against Acceptance Criteria**:

| Criterion | Target | Action |
|-----------|--------|--------|
| **Sharpe ≥ 0.25** | Recommended pass | ✅ Proceed to Phase-2 |
| **Sharpe 0.20-0.25** | Conditional pass | ⚠️ Review asset-class contributions |
| **Sharpe < 0.20** | Fail | ❌ Investigate root cause |
| **MaxDD ≥ -30%** | Acceptable | ✅ Continue |
| **MaxDD < -30%** | Unacceptable | ❌ Review vol normalization |
| **2020 Q1 > -20%** | Crisis behavior | ✅ Continue |
| **2020 Q1 ≤ -20%** | Crisis failure | ❌ Review signal logic |

**Decision Tree**:
- **Pass (Sharpe ≥ 0.25)**: Proceed to Phase-2 (Integration)
- **Conditional (0.20-0.25)**: Review asset-class breakdown, consider Phase-1.1 with cross-sectional ranking
- **Fail (< 0.20)**: Debug (check sign logic, vol normalization, data quality)

---

### 4. 📝 Document Phase-1 Results

**Create**: `archive/phases/carry_phase1_run_memo.md` (similar to Phase-0 memo)

**Include**:
- Performance metrics vs Phase-0
- Asset-class contribution breakdown
- Stress window analysis
- Pass/fail determination
- Next steps recommendation

---

## Phase-1 Implementation Status

✅ **Complete**:
- Rolling z-score (252d window)
- Clipping (±3.0)
- Vol normalization (equal risk per asset)
- Unit gross scaling (sum abs = 1.0)
- Phase-1 config and diagnostic script

⏸️ **Deferred**:
- Cross-sectional ranking within asset classes (Phase-1.1 if needed)

---

## Files Ready

**Phase-1 Code**:
- ✅ `src/agents/strat_carry_meta_v1.py` (Phase-1 implementation)
- ✅ `configs/carry_phase1_v1.yaml` (Phase-1 config)
- ✅ `scripts/diagnostics/run_carry_phase1_v1.py` (Phase-1 diagnostic)

**Documentation**:
- ✅ `CARRY_PHASE1_IMPLEMENTATION.md` (Implementation details)
- ✅ `CARRY_PHASE1_READY.md` (Quick start guide)
- ✅ `CARRY_PHASE1_PLAN.md` (Original plan)

---

## Governance Status

**Commit A (Carry-only)**: ✅ Safe to merge (engine-only)  
**Commit B (Core semantics)**: ⏸️ Pending Phase 3B validation  
**Phase-0**: ✅ Locked as official baseline  
**Phase-1**: ✅ Implemented and ready for testing

---

## Quick Reference

**Run Phase-1**:
```bash
python scripts/diagnostics/run_carry_phase1_v1.py
```

**Validate Commit B** (separate):
```bash
python scripts/diagnostics/verify_phase3b_baseline_checkpoints.py --both
python scripts/diagnostics/test_decoupling.py
```

**Phase-0 Baseline**: `carry_phase0_v1_20260121_143130`

---

**Ready to proceed! 🚀**
