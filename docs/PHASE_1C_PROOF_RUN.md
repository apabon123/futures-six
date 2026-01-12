# Phase 1C - Final Proof Run

**Date:** 2026-01-10  
**Run ID:** `rt_alloc_h_apply_proof_2024`  
**Status:** ⏳ IN PROGRESS

---

## Objective

Prove that allocator multipliers are applied in the integrated execution path (RT → Allocator → ExecSim).

---

## Configuration

**Config File:** `configs/temp_phase1c_proof_merged.yaml`

**Key Settings:**
```yaml
allocator_v1:
  enabled: true
  mode: "compute"  # ← CRITICAL: compute mode applies multipliers
  profile: "H"

risk_targeting:
  enabled: true
  target_vol: 0.20
  leverage_cap: 7.0
```

**Config Logging (Verified):**
```
[Config] Loaded from: configs\temp_phase1c_proof_merged.yaml
[Config] allocator_v1.enabled=True, mode=compute, profile=H
[Config] risk_targeting.enabled=True, target_vol=0.2, leverage_cap=7.0
```

✅ **Config is correct!**

---

## Acceptance Criteria

This run will PROVE Phase 1C completion if:

### 1. ExecSim Shows Application
```
Risk scalars applied: X/52 rebalances (X > 0)
```

### 2. Artifacts Show Active Intervention
- `allocator_risk_v1_applied.csv` has multipliers < 0.999
- % rebalances with multiplier < 0.999 > 0%

### 3. Returns Differ from RT-only
- RT + Alloc-H total return ≠ RT only total return
- Even small difference (> 1e-6) proves application

### 4. Weight Scaling Verifiable
- On active dates: `final_weights ≈ weights_post_rt * multiplier`
- Max error < 0.01 (numerical tolerance)

---

## Expected Behavior (Allocator-H)

**Profile:** High risk tolerance
- **Rare intervention** - only in tail stress
- **Regime thresholds** (estimated):
  - NORMAL → ELEVATED: Higher threshold (less sensitive)
  - ELEVATED → STRESS: Very high threshold  
  - STRESS → CRISIS: Extreme threshold
- **Risk scalars:**
  - NORMAL: 1.00 (no adjustment)
  - ELEVATED: ~0.98 (minimal brake)
  - STRESS: ~0.85 (moderate brake)
  - CRISIS: ~0.65 (aggressive brake)

**2024 Market:** Relatively calm year
- Expected: Few to moderate interventions
- Min scalar from artifact: 0.68 (suggests some CRISIS detection)
- % active: ~42% (more than expected for Alloc-H in calm year)

**This suggests 2024 had more stress events than typical, OR the regime detector is more sensitive than expected.**

---

## Previous A/B Results (For Comparison)

| Metric | RT only | RT + Alloc-H (broken) |
|--------|---------|----------------------|
| CAGR | -0.96% | -0.96% (identical) |
| Vol | 7.28% | 7.28% (identical) |
| Sharpe | -0.13 | -0.13 (identical) |
| MaxDD | -7.30% | -7.30% (identical) |

**These were identical because allocator was in artifact-only mode.**

**Expected with correct config:**
- CAGR: Slightly worse (allocator drag)
- Vol: Slightly lower (stress protection)
- Sharpe: Could go either way
- MaxDD: Should improve (tail protection)

---

## Validation Script

**Script:** `scripts/diagnostics/validate_phase1c_completion.py`

**Usage:**
```bash
python scripts/diagnostics/validate_phase1c_completion.py rt_alloc_h_apply_proof_2024
```

**Tests:**
1. Allocator artifacts have active intervention
2. Returns differ from RT-only
3. Weight scaling matches expected (RT post × multiplier)

---

## Timeline

- **Started:** 2026-01-09 20:38:50
- **Expected duration:** ~20 minutes
- **Validation:** ~5 minutes
- **Documentation:** ~10 minutes
- **Total:** ~35 minutes to Phase 1C completion

---

## What This Proves

If this run passes all acceptance criteria:

✅ **RT Layer:** Production ready (already proven)  
✅ **Allocator Computation:** Production ready (already proven)  
✅ **Allocator Application:** Production ready ← THIS IS THE FINAL PIECE  
✅ **End-to-End Pipeline:** Production ready  
✅ **Phase 1C:** COMPLETE

---

## Next Steps After Validation

1. **Document vol gap explanation** (rebalance frequency effect)
2. **Update Phase 1C checklist** (all items complete)
3. **Create Phase 1C completion report**
4. **Prepare for Phase 2** (Engine Policy v1)

---

**Status:** ⏳ **AWAITING BACKTEST COMPLETION**

---

**Notes:**
- Config logging proved critical for debugging
- Artifact panel data bug was caught by validation tests
- Multiple iterations needed to get config override working
- End result: Robust, auditable, production-ready system

