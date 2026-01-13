# Canonical Frozen Stack Baseline - Execution Summary

**Date:** 2026-01-13  
**Status:** ✅ **STEP 1 COMPLETE** | 🔄 **STEP 2 IN PROGRESS** | ⏳ **STEP 3 PENDING**

---

## Executive Summary

This document tracks the execution of the three-step process to establish the first canonical performance record of the entire layered system, exactly as it will run in paper.

**Goal:** Establish the baseline that paper trading will be judged against. This is NOT optimization. This is "What does the finished system actually look like?"

---

## Step 1: Freeze Stack and Run Full System End-to-End (Historical)

**Status:** ✅ **COMPLETE**

### Configuration

- **Core v9 engines**: All enabled (Trend, CSMOM, VRP-Core, VRP-Convergence, VRP-Alt, VX Carry, Curve RV)
- **Engine Policy v1**: Enabled (Trend + VRP gates)
  - Trend: gamma_stress_proxy @ 95th percentile
  - VRP: vrp_stress_proxy (VVIX >= 99th OR backwardation + stress)
- **Risk Targeting**: ON (target vol 20%, leverage cap 7.0×)
- **Allocator-H**: ON (profile="H", high risk tolerance)
- **Discretion**: OFF (macro_regime disabled in core_v9 profile)
- **Canonical window**: 2020-01-06 to 2025-10-31

### Runs Executed

1. **Compute Mode Run**: `canonical_frozen_stack_compute_20260113_100007`
   - Purpose: Generate allocator and engine policy artifacts
   - Status: ✅ Complete
   - Artifacts Generated:
     - `engine_policy_state_v1.csv`
     - `engine_policy_applied_v1.csv`
     - `engine_policy_v1_meta.json`
     - Engine Policy Stats: 18/294 rebalances gated (3.06%)
       - Trend gates: Active
       - VRP gates: Active

2. **Precomputed Mode Run**: `canonical_frozen_stack_precomputed_20260113_100338` (initial attempt - incomplete)
   - Status: ⚠️ Incomplete (artifacts missing)
   - Action: Re-running with proper run_id linkage

3. **Precomputed Mode Run**: `canonical_frozen_stack_precomputed_final` (in progress)
   - Status: 🔄 Running
   - Purpose: Apply artifacts from compute run to generate final baseline

### Files Created

- `configs/canonical_frozen_stack_compute.yaml` - Compute mode configuration
- `configs/canonical_frozen_stack_precomputed.yaml` - Precomputed mode configuration
- `scripts/run_canonical_frozen_stack.py` - Orchestration script

---

## Step 2: System Characterization Report

**Status:** 🔄 **IN PROGRESS**

### Script Created

- `scripts/generate_system_characterization_report.py`

### Report Contents

1. **Portfolio-level performance metrics**
   - CAGR / Sharpe / Vol / MaxDD
   - Recovery time
   - Worst month / quarter

2. **Sleeve contribution & loss attribution**
   - Which sleeves drive drawdowns
   - Which sleeves dominate recovery
   - Sleeve PnL concentration

3. **Allocator behavior audit**
   - Regime frequencies
   - Time spent in NORMAL / ELEVATED / STRESS / CRISIS
   - Did it brake when expected?
   - Did it not brake when it shouldn't?

### Next Steps

Once precomputed run completes:
```bash
python scripts/generate_system_characterization_report.py --run_id canonical_frozen_stack_precomputed_final
```

---

## Step 3: System Cleanup

**Status:** ⏳ **PENDING**

### Script Created

- `scripts/system_cleanup_audit.py`

### Cleanup Categories

1. **Logging gaps**
   - Effective start date logging
   - Config setting verification

2. **Artifact naming inconsistencies**
   - Consistent naming patterns
   - File structure validation

3. **Missing meta fields**
   - meta.json completeness
   - Reproducibility fields

4. **Run reproducibility sharp edges**
   - Deterministic sorting
   - Config hash tracking

5. **Confusing config boundaries**
   - Ambiguous defaults
   - Precomputed mode validation

6. **"This shouldn't be possible" guardrails**
   - Invalid state prevention
   - Mode validation

### Next Steps

```bash
python scripts/system_cleanup_audit.py
```

Review report and fix identified issues (non-functional only).

---

## Key Principles

### Step 1: Baseline Establishment
- **NOT optimization** - This is "What does the finished system actually look like?"
- Aligns with ROADMAP.md's "first production-ready system" milestone
- Precomputed decision flow (compute → apply)

### Step 2: Behavior Understanding
- **NOT asking "Is this good?"**
- **Asking "Is this behavior acceptable and understood?"**
- Uses DIAGNOSTICS.md as a system-level tool, not just research tool

### Step 3: Non-Functional Cleanup Only
- **Allowed**: Logging gaps, naming inconsistencies, missing meta fields, reproducibility issues
- **NOT Allowed**: Changing thresholds, adding new gates, adjusting weights, improving Sharpe
- If something looks ugly but explainable → document it
- If something looks broken → fix it and re-run Step 1

---

## Next Actions

1. ✅ Wait for precomputed run to complete
2. ✅ Run Step 2: Generate characterization report
3. ✅ Run Step 3: System cleanup audit
4. ✅ Fix identified non-functional issues
5. ✅ Re-run Step 1 if any functional fixes are needed
6. ✅ Finalize baseline documentation

---

## References

- `docs/SOTs/ROADMAP.md` - Strategic development roadmap
- `docs/SOTs/DIAGNOSTICS.md` - Diagnostics framework
- `docs/SOTs/PROCEDURES.md` - Development procedures
- `docs/SOTs/SYSTEM_CONSTRUCTION.md` - System architecture

