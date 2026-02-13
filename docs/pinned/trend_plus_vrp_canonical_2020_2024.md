# Trend+VRP Canonical (2020–2024)

**Run ID:** `trend_plus_vrp_canonical_2020_2024`  
**Category:** Integration  
**Window:** 2020-01-01 → 2024-10-31

## Summary

Trend + VRP Meta-Sleeves; VRP uses corrected VX2−VX1 convergence geometry. Apples-to-apples comparison with Trend-Only baseline.

## Sleeves

trend metasleeve (60%) + vrp metasleeve (40%, atomic: vrp_core, vrp_convergence, vrp_alt).

## Reproduce

```bash
python run_strategy.py --start 2020-01-01 --end 2024-10-31 \
  --run_id trend_plus_vrp_canonical_2020_2024 \
  --config_path configs/phase4_trend_plus_vrp_canonical_v1.yaml --strict_universe
```
