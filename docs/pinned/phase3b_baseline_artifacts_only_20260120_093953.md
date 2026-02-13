# Phase 3B Artifacts-Only Baseline (2026-01-20)

**Run ID:** `phase3b_baseline_artifacts_only_20260120_093953`  
**Category:** Production  
**Window:** 2020-01-06 → 2025-10-31

## Summary

Pre-allocator reference; construction curve without allocator scaling. Use when comparing pre-vs-post allocator behavior.

## Sleeves

Same as traded baseline: trend, csmom, vrp (atomic: vrp_core, vrp_convergence, vrp_alt), vx_carry, curve_rv.

## Reproduce

```bash
python run_strategy.py --config_path configs/phase3b_baseline_artifacts_only.yaml \
  --run_id phase3b_baseline_artifacts_only_20260120_093953 \
  --start 2020-01-06 --end 2025-10-31
```
