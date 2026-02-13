# Phase 3B Traded Baseline (2026-01-20)

**Run ID:** `phase3b_baseline_traded_20260120_093953`  
**Category:** Production  
**Window:** 2020-01-06 → 2025-10-31

## Summary

Current reference baseline for attribution and regression checks. Post-allocator traded curve; RT target_vol 0.42; allocator v1 precomputed, applied. All 7 Phase 3B checkpoints passed.

## Sleeves

| Metasleeve | Weight | Atomic sleeves |
|------------|--------|----------------|
| trend | 52.44% | — |
| csmom | 21.85% | — |
| vrp | 21.85% | vrp_core, vrp_convergence, vrp_alt |
| vx_carry | 4.6% | — |
| curve_rv | 8% | — |

## Reproduce

```bash
python run_strategy.py --config_path configs/phase3b_baseline_traded.yaml \
  --run_id phase3b_baseline_traded_20260120_093953 \
  --start 2020-01-06 --end 2025-10-31
```
