# Two-Pass Allocator Comparison

Baseline run: `baseline_20200106_20251031`  
Scaled run: `allocpass2_20200106_20251031`

## Summary Metrics

| Metric | Baseline | Scaled | Δ (Scaled - Baseline) |
|---|---:|---:|---:|
| CAGR | 0.0935 | 0.0642 | -0.0294 |
| Ann Vol | 0.1201 | 0.0988 | -0.0213 |
| Sharpe | 0.6607 | 0.5573 | -0.1034 |
| MaxDD | -0.1532 | -0.1433 | 0.0098 |
| Worst Month | -0.0530 (2023-03-31) | -0.0533 (2023-03-31) | -0.0004 |
| Worst Quarter | -0.0602 (2022-03-31) | -0.0540 (2024-12-31) | 0.0062 |

## Scalar Usage (Scaled Run)

- Rebalances: 263
- % scaled: 73.8%
- Mean scalar: 0.9221
- Min scalar: 0.4247
- Max scalar: 1.0000
- P05/P50/P95: 0.5897 / 0.9940 / 1.0000
