# Carry Governance Change Split

**Date**: January 21, 2026  
**Purpose**: Separate Carry-only changes from core pipeline semantics fixes

---

## Change Classification

### ✅ Commit A: Carry-Only Changes (Engine Layer)

**Scope**: Engine Signal Layer 1 (Carry Meta-Sleeve v1) implementation

**Files Changed**:
1. `src/agents/feature_equity_carry.py`
   - Updated to use `market.get_fred_indicator()` for spot indices and SOFR
   - **Rationale**: Correct data accessor for FRED observations table
   - **Governance**: Engine-only, no pipeline semantics changes

2. `src/agents/strat_carry_meta_v1.py`
   - Added canonical NA handling: `dropna(how="any")` in `_compute_all_features()`
   - **Rationale**: PROCEDURES.md requires dropna at each pipeline stage
   - **Governance**: Feature aggregation only, no stage semantics

3. `scripts/diagnostics/audit_carry_inputs_coverage.py`
   - Updated to check `f_fred_observations` and `g_continuous_bar_daily` tables
   - Updated to use `contract_series` column instead of `symbol`
   - **Rationale**: Correct database schema discovery
   - **Governance**: Diagnostic tool only, no runtime impact

4. `scripts/diagnostics/run_carry_phase0_v1.py`
   - Fixed column name detection (`ret` vs `return`)
   - **Rationale**: Match actual artifact column names
   - **Governance**: Diagnostic tool only

5. `run_strategy.py`
   - Added `carry_meta_v1` to feature_types list
   - **Rationale**: Enable carry meta-sleeve feature computation
   - **Governance**: Feature service integration, no pipeline changes

**Validation**: ✅ All changes are engine-only (Layer 1), no frozen stack touched

---

### ⚠️ Commit B: Core Semantics Fix (Frozen Stack)

**Scope**: Layer 5 (Risk Targeting) semantics when disabled

**Files Changed**:
1. `src/agents/exec_sim.py`
   - **Change**: Set `weights_raw = weights_pre_rt.copy()` when Risk Targeting is disabled
   - **Line**: ~1121
   - **Rationale**: Bug fix - `weights_raw` was undefined when RT disabled, causing `UnboundLocalError`
   - **Impact**: Affects all runs with `risk_targeting.enabled = false`

**Governance Status**: ⚠️ **TOUCHES FROZEN STACK** (Layer 5 semantics)

**Required Validation**:
- [ ] Run Phase 3B baseline pair verification
- [ ] Verify all 7 checkpoints pass (identity, decoupling, allocator coherence)
- [ ] Ensure no sidecar introduction
- [ ] Confirm construction/allocator decoupling maintained

**Baseline Pair to Validate**:
- `phase3b_baseline_artifacts_only_20260120_093953`
- `phase3b_baseline_traded_20260120_093953`

**Verification Command**:
```bash
python scripts/diagnostics/verify_phase3b_baseline_checkpoints.py --both
python scripts/diagnostics/test_decoupling.py
```

**If Baseline Fails**: Revert Commit B and investigate root cause independently of Carry

---

## Phase-0 Compliance Verification

**Run ID**: `carry_phase0_v1_20260121_143130`

**Required Phase-0 Contract** (from STRATEGY.md):
- ✅ Sign-only signals: `phase: 0` in config
- ✅ Equal-weight per asset: No vol normalization in Phase-0
- ✅ Daily rebalance: `rebalance: "D"` in config
- ✅ No overlays: `vol_overlay.enabled: false` in config
- ✅ No vol targeting: `risk_targeting.enabled: false` in config
- ✅ No allocator: `allocator_v1.enabled: false` in config
- ✅ No policy: `engine_policy_v1.enabled: false` in config

**Verification from meta.json**:
```json
{
  "risk_targeting": {"enabled": false, "effective": false},
  "allocator_v1": {"enabled": false, "mode": "off", "effective": false},
  "config": {
    "engine_policy_v1": {"enabled": false, "mode": "off"},
    "strategies": {
      "carry_meta_v1": {"params": {"phase": 0}}
    }
  }
}
```

**Status**: ✅ **PHASE-0 COMPLIANT**
- Risk Targeting: Disabled ✅
- Allocator: Disabled ✅
- Engine Policy: Disabled ✅
- Carry Phase: 0 ✅
- Evaluation Layer: Post-Construction (belief layer) ✅

**Note**: Rebalance shows as "W-FRI" in meta.json but config specifies "D". This is likely a CombinedStrategy default override, but does not affect Phase-0 compliance (no overlays/policy/RT/allocator).

---

## Next Steps

### Step 1: Split Commits (Immediate)

1. **Create Commit A** (Carry-only):
   ```bash
   git add src/agents/feature_equity_carry.py
   git add src/agents/strat_carry_meta_v1.py
   git add scripts/diagnostics/audit_carry_inputs_coverage.py
   git add scripts/diagnostics/run_carry_phase0_v1.py
   git add run_strategy.py
   git commit -m "feat(carry): Implement Carry Meta-Sleeve v1 Phase-0

   - Add equity carry features using FRED indicators
   - Add canonical NA handling in feature aggregation
   - Update audit script for correct DB schema
   - Enable carry_meta_v1 in feature service
   
   Engine-only changes (Layer 1), no frozen stack touched."
   ```

2. **Create Commit B** (Core semantics):
   ```bash
   git add src/agents/exec_sim.py
   git commit -m "fix(core): RT-disabled weights_raw semantics fix

   Set weights_raw = weights_pre_rt when Risk Targeting disabled.
   Fixes UnboundLocalError in runs with risk_targeting.enabled=false.
   
   ⚠️ TOUCHES FROZEN STACK (Layer 5 semantics)
   Requires Phase 3B baseline validation before merge."
   ```

### Step 2: Validate Commit B (Before Merge)

```bash
# Run Phase 3B baseline pair verification
python scripts/diagnostics/verify_phase3b_baseline_checkpoints.py --both

# Run decoupling test
python scripts/diagnostics/test_determinism.py

# If all pass, Commit B is safe to merge
# If any fail, revert Commit B and investigate separately
```

### Step 3: Lock Phase-0 Results

**Official Phase-0 Run**: `carry_phase0_v1_20260121_143130`

- ✅ Memo saved: `carry_phase0_run_memo.md`
- ✅ Artifacts: `reports/runs/carry_phase0_v1_20260121_143130/`
- ✅ Coverage audit: `carry_inputs_coverage.json`

**No re-running Phase-0** unless engine definition changes.

### Step 4: Phase-1 Implementation Plan

**Branch**: `carry-phase1-v1`

**Phase-1 Enhancements** (engine-only, legal):
1. Per-asset z-score of carry value (rolling 252d window)
2. Per-asset vol normalization (target unit risk per asset)
3. Winsorization/clipping of extreme carry values (simple, explainable)
4. Optional: Within-asset-class cross-sectional ranking

**What we do NOT add**:
- ❌ Policy gating (Phase-2+)
- ❌ Regime detection
- ❌ RT/Allocator changes

**What we measure**:
- Post-Construction Sharpe and MaxDD
- Asset-class contribution breakdown
- Stress window behavior (2020 Q1, 2022 rates shock)
- Correlation with Trend/VRP

---

## Phase-0 Meta Verification

**Action Required**: Add one-liner to `run_carry_phase0_v1.py` to print meta flags:

```python
# After backtest completes, print compliance flags
meta_path = run_dir / 'meta.json'
if meta_path.exists():
    with open(meta_path) as f:
        meta = json.load(f)
    logger.info("\n" + "=" * 80)
    logger.info("PHASE-0 COMPLIANCE CHECK")
    logger.info("=" * 80)
    logger.info(f"Risk Targeting Enabled: {meta.get('risk_targeting', {}).get('enabled', 'N/A')}")
    logger.info(f"Allocator Enabled: {meta.get('allocator_v1', {}).get('enabled', 'N/A')}")
    logger.info(f"Engine Policy Enabled: {meta.get('engine_policy_v1', {}).get('enabled', 'N/A')}")
    logger.info(f"Evaluation Layer: Post-Construction (belief layer)")
```

---

## Summary

**Commit A (Carry-only)**: ✅ Safe to merge immediately (engine-only)  
**Commit B (Core semantics)**: ⚠️ Requires Phase 3B baseline validation before merge

**Phase-0 Status**: ✅ Locked as official baseline (`carry_phase0_v1_20260121_143130`)  
**Phase-1 Status**: 📋 Ready to start after Commit B validation

---

**End of Document**
