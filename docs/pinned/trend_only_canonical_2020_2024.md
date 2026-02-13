# Trend-Only Canonical (2020–2024)

**Run ID:** `trend_only_canonical_2020_2024`  
**Category:** Integration  
**Window:** 2020-01-01 → 2024-10-31

## Summary

Trend (TSMOM) Meta-Sleeve only for apples-to-apples comparison vs Trend+VRP. All other sleeves disabled.

## Sleeves

trend metasleeve only (100%).

## Reproduce

```bash
python run_strategy.py --start 2020-01-01 --end 2024-10-31 \
  --run_id trend_only_canonical_2020_2024 \
  --config_path configs/phase4_trend_only_canonical_v1.yaml --strict_universe
```
