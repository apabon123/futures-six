# Phase 3A Baseline Re-Freeze: Exact Command Sequence

This document provides the exact sequence of commands to re-freeze the canonical baseline with PolicyFeatureBuilder integration and validate Phase 3A acceptance criteria.

## Overview

The re-freeze follows the two-pass proof philosophy:
1. **Compute mode**: Policy enabled, features built in-loop, generates `engine_policy_applied_v1.csv`
2. **Precomputed mode**: Materializes multipliers from compute baseline (self-contained)

Both runs must pass validation before pinning.

## Prerequisites

- PolicyFeatureBuilder integrated into `run_strategy.py`
- Canonical window: 2020-01-06 to 2025-10-31
- Strategy profile: `core_v9_trend_csmom_vrp_core_convergence_vrp_alt_vx_carry_sr3_curverv_no_macro`

## Step-by-Step Sequence

### A) Compute Mode Baseline (Policy Enabled)

**Step A1: Run compute mode baseline**

```bash
python scripts/run_canonical_frozen_stack.py \
    --strategy_profile core_v9_trend_csmom_vrp_core_convergence_vrp_alt_vx_carry_sr3_curverv_no_macro \
    --compute_run_id canonical_frozen_stack_compute_phase3a_<timestamp>
```

This will:
- Build policy features via PolicyFeatureBuilder (upstream of EnginePolicyV1)
- Run with `engine_policy_v1.mode=compute`
- Generate `engine_policy_applied_v1.csv` artifact
- Save run to `reports/runs/canonical_frozen_stack_compute_phase3a_<timestamp>/`

**Step A2: Generate canonical diagnostics**

```bash
python scripts/diagnostics/generate_canonical_diagnostics.py \
    --run_id canonical_frozen_stack_compute_phase3a_<timestamp>
```

This generates:
- `canonical_diagnostics.json`
- `canonical_diagnostics.md`

**Step A3: Validate compute mode baseline**

```bash
python scripts/diagnostics/validate_phase3a_policy_baseline.py \
    canonical_frozen_stack_compute_phase3a_<timestamp>
```

**Expected output:**
- ✅ All hard checks pass (no hard failures)
- ⚠️  Soft warnings (zero gating) are acceptable IF justified
- Check that `policy_inert=false` and `policy_effective=true` in diagnostics

**If validation fails:**
- Fix issues and re-run from Step A1
- Do not proceed to precomputed mode until compute mode passes

---

### B) Precomputed Mode Baseline (Materialized Multipliers)

**Step B1: Run precomputed mode baseline**

```bash
python scripts/run_canonical_frozen_stack.py \
    --strategy_profile core_v9_trend_csmom_vrp_core_convergence_vrp_alt_vx_carry_sr3_curverv_no_macro \
    --skip_compute \
    --existing_compute_run_id canonical_frozen_stack_compute_phase3a_<timestamp> \
    --precomputed_run_id canonical_frozen_stack_precomputed_phase3a_<timestamp>
```

This will:
- Materialize `engine_policy_applied_v1.csv` from compute baseline
- Run with `engine_policy_v1.mode=precomputed`
- Create self-contained baseline run
- Save run to `reports/runs/canonical_frozen_stack_precomputed_phase3a_<timestamp>/`

**Step B2: Generate canonical diagnostics**

```bash
python scripts/diagnostics/generate_canonical_diagnostics.py \
    --run_id canonical_frozen_stack_precomputed_phase3a_<timestamp>
```

**Step B3: Validate precomputed mode baseline**

```bash
python scripts/diagnostics/validate_phase3a_policy_baseline.py \
    canonical_frozen_stack_precomputed_phase3a_<timestamp>
```

**Expected output:**
- ✅ All hard checks pass (no hard failures)
- ⚠️  Soft warnings (zero gating) are acceptable IF justified
- Check that `policy_inert=false` and `policy_effective=true` in diagnostics

**If validation fails:**
- Fix issues and re-run from Step B1
- Do not proceed to pinning until precomputed mode passes

---

### C) Pin Only If Both Pass

**Pinning Criteria (ALL must be true):**

1. ✅ **Both runs validated:** Compute mode AND precomputed mode passed validation
2. ✅ **policy_inert=false:** In canonical diagnostics for both runs
3. ✅ **policy_effective=true:** In canonical diagnostics for both runs
4. ✅ **stress_value not-all-NaN:** For both Trend and VRP engines in `engine_policy_state_v1.csv`
5. ✅ **At least one multiplier=0 OR explicit justification:** 
   - Either: At least one rebalance with `trend_multiplier=0` or `vrp_multiplier=0`
   - Or: Explicit "zero gating justification" note in committee pack explaining why zero gating is legitimate

**Step C1: Verify pinning criteria**

For each run (compute and precomputed), verify:

```bash
# Check canonical diagnostics
cat reports/runs/<run_id>/canonical_diagnostics.json | jq '.constraint_binding | {policy_inert, policy_effective, policy_inputs_present}'

# Check policy state artifact
python -c "
import pandas as pd
state = pd.read_csv('reports/runs/<run_id>/engine_policy_state_v1.csv')
for engine in ['trend', 'vrp']:
    engine_state = state[state['engine'] == engine]
    n_valid = engine_state['stress_value'].notna().sum()
    n_total = len(engine_state)
    print(f'{engine}: {n_valid}/{n_total} valid stress_value entries')
"

# Check policy applied artifact
python -c "
import pandas as pd
applied = pd.read_csv('reports/runs/<run_id>/engine_policy_applied_v1.csv')
for col in ['trend_multiplier', 'vrp_multiplier']:
    if col in applied.columns:
        gated = (applied[col] < 0.999).sum()
        total = len(applied)
        print(f'{col}: {gated}/{total} gated ({gated/total*100:.1f}%)')
"
```

**Step C2: Add to _PINNED if all criteria pass**

Edit `reports/_PINNED/README.md` and add:

```markdown
## Phase 3A Policy Features Baseline (YYYY-MM-DD)

**Compute Mode:**
- Run ID: `canonical_frozen_stack_compute_phase3a_<timestamp>`
- Purpose: Phase 3A baseline with PolicyFeatureBuilder integration (compute mode)
- Status: ✅ Validated - policy_effective=true, policy_inert=false
- Policy Features: ✅ All present (gamma_stress_proxy, vx_backwardation, vrp_stress_proxy)
- Policy Gating: Trend X%, VRP Y% (or: Zero gating - justified: [reason])

**Precomputed Mode:**
- Run ID: `canonical_frozen_stack_precomputed_phase3a_<timestamp>`
- Purpose: Phase 3A baseline with PolicyFeatureBuilder integration (precomputed mode)
- Status: ✅ Validated - policy_effective=true, policy_inert=false
- Source Run: `canonical_frozen_stack_compute_phase3a_<timestamp>`
- Policy Features: ✅ All present (materialized from compute baseline)
- Policy Gating: Trend X%, VRP Y% (or: Zero gating - justified: [reason])

**Acceptance Criteria:**
- ✅ policy_inert=false (both runs)
- ✅ policy_effective=true (both runs)
- ✅ stress_value not-all-NaN for Trend and VRP engines
- ✅ At least one multiplier=0 OR explicit zero gating justification

**Unblocks:** Phase 3A ablations per committee-pack workflow
```

---

## Validation Checklist

Before pinning, verify ALL of the following:

### Compute Mode Run
- [ ] `canonical_diagnostics.json` exists
- [ ] `constraint_binding.policy_inert = false`
- [ ] `constraint_binding.policy_effective = true`
- [ ] `constraint_binding.policy_inputs_present` has all three features with `has_data=true`
- [ ] `engine_policy_state_v1.csv` exists
- [ ] `engine_policy_state_v1.csv`: Trend engine `stress_value` not-all-NaN
- [ ] `engine_policy_state_v1.csv`: VRP engine `stress_value` not-all-NaN
- [ ] `engine_policy_applied_v1.csv` exists
- [ ] At least one `trend_multiplier=0` OR `vrp_multiplier=0` OR explicit zero gating justification

### Precomputed Mode Run
- [ ] `canonical_diagnostics.json` exists
- [ ] `constraint_binding.policy_inert = false`
- [ ] `constraint_binding.policy_effective = true`
- [ ] `constraint_binding.policy_inputs_present` has all three features with `has_data=true`
- [ ] `engine_policy_applied_v1.csv` exists (materialized from compute baseline)
- [ ] `meta.json.engine_policy_source_run_id` points to compute mode run
- [ ] At least one `trend_multiplier=0` OR `vrp_multiplier=0` OR explicit zero gating justification

---

## Once Pinned

After both runs are pinned in `reports/_PINNED/README.md`:

✅ **Phase 3A ablations are unblocked** per the committee-pack workflow (PROCEDURES.md § 13).

You can now:
- Run ablation tests comparing against the pinned baseline
- Use the baseline for attribution analysis
- Generate committee packs for decision runs

---

## Troubleshooting

**If compute mode validation fails:**
- Check that PolicyFeatureBuilder ran successfully (look for log messages)
- Verify `meta.json.policy_features` has all three features with `has_data=true`
- Check that `engine_policy_v1.enabled=true` in config
- Review `engine_policy_state_v1.csv` for data quality issues

**If precomputed mode validation fails:**
- Verify compute mode run ID is correct
- Check that `engine_policy_applied_v1.csv` exists in compute mode run
- Verify `meta.json.engine_policy_source_run_id` is set correctly

**If zero gating (soft warning):**
- This is acceptable IF the canonical window legitimately didn't hit stress thresholds
- Document justification in committee pack or pinned run notes
- Example: "Zero gating justified: VVIX 95th percentile threshold not reached in 2020-2025 window"

---

## Related Documentation

- `docs/SOTs/SYSTEM_CONSTRUCTION.md` - Policy artifact contract
- `docs/SOTs/PROCEDURES.md` - Phase 3A operational procedure
- `docs/SOTs/DIAGNOSTICS.md` - Committee pack generation
- `scripts/diagnostics/validate_phase3a_policy_baseline.py` - Validation script
