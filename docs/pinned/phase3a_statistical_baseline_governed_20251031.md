# Phase 3A Statistical Baseline (Governed)

**Run ID:** `phase3a_statistical_baseline_governed_20251031`  
**Category:** Engine Quality  
**Window:** 2020-01-06 → 2025-10-31 (eval from 2020-03-20)

## Summary

Phase 3A governed baseline; policy + RT + allocator governance. Allocator mode precomputed, effective; RT effective. Canonical window true.

## Sleeves

Core v9 profile: trend 52.44%, csmom 21.85%, vrp 21.85% (atomic: vrp_core, vrp_convergence, vrp_alt), vx_carry 4.6%, curve_rv 8%.

## Reproduce

```bash
python run_strategy.py --config_path configs/canonical_frozen_stack_precomputed.yaml \
  --strategy_profile core_v9_trend_csmom_vrp_core_convergence_vrp_alt_vx_carry_sr3_curverv_no_macro \
  --run_id phase3a_statistical_baseline_governed_20251031 \
  --start 2020-01-06 --end 2025-10-31
```
