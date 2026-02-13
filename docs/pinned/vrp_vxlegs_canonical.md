# VRP VX Legs Canonical

**Run ID:** `vrp_vxlegs_canonical`  
**Category:** Diagnostic  
**Window:** 2020-01-01 → 2024-10-31

## Summary

VRP-only canonical run with restored VX2–VX1 spread geometry for Convergence sleeve. Validates that VRP Convergence no longer acts like a VX1-only proxy.

## Sleeves

VRP metasleeve only (atomic sleeves: vrp_core, vrp_convergence, vrp_alt).

## Reproduce

```bash
python scripts/runs/run_vrp_canonical_2020_2024.py
# Or:
python run_strategy.py --config_path configs/phase4_vrp_baseline_v1.yaml \
  --run_id vrp_vxlegs_canonical \
  --start 2020-01-01 --end 2024-10-31 --strict_universe
```
