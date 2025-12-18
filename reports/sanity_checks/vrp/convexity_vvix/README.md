# VRP-Convexity (VVIX Threshold) Sleeve

**Status: PARKED**

This sleeve is PARKED as of Phase-0. No Phase-1 or Phase-2 development will proceed under the current specification.

## Phase-0 Results

- **Sharpe**: -0.1288 (FAIL - hard fail; not borderline)
- **CAGR**: -35.35%
- **MaxDD**: -97.73% (catastrophic)
- **HitRate**: 21.23%
- **Signal Distribution**: 52.4% active (non-degenerate)
- **Status**: Economic failure - not proceeding to Phase-1

## What Was Tested

**Phase-0 Rule**: Short VX1 when VVIX > 100, else flat

**Economic Thesis**: Vol-of-vol (VVIX) is structurally overpriced relative to VIX. Elevated VVIX indicates expensive convexity insurance → favorable short-vol carry once shocks stabilize.

**Data**: VVIX data successfully loaded (1467 rows, 2020-01-02 to 2025-10-31). VVIX loader (`load_vvix()`) works correctly.

## Why It Failed

The Phase-0 sign-only test demonstrated that mapping VVIX threshold (VVIX > 100) to directional short VX1 exposure does not produce a positive economic edge.

**Key Findings:**

1. **Negative Sharpe**: -0.1288 (well below 0.10 Phase-0 threshold)
2. **Catastrophic Drawdown**: -97.73% (hard fail, not borderline)
3. **Non-Degenerate Signal**: 52.4% active days (signal works, but strategy fails)
4. **Wrong Expression**: Simple threshold-based directional short is not the right expression of convexity premium

The negative Sharpe plus catastrophic MaxDD indicates a hard fail, not an "engineer it in Phase-1" situation. This is fundamentally different from borderline cases (e.g., VRP-Alt Phase-0) where Phase-1 engineering can rescue the idea.

## Revisit Options

- **Reframe as Conditioning Feature**: Use VVIX as a regime filter or conditioning variable rather than a directional trade signal
- **Spread-Style Trade**: Consider VVIX-VIX spread trades rather than outright VX1 directionality
- **Crisis Meta-Sleeve Integration**: Potential integration into Crisis Meta-Sleeve as volatility regime indicator
- **Volatility Regime Indicator**: Use VVIX to identify high-volatility regimes for other VRP sleeves

## Phase-0 Script

The Phase-0 implementation is available at:
- `scripts/diagnostics/run_vrp_convexity_vvix_phase0.py`
- Results: `reports/sanity_checks/vrp/convexity_vvix/latest/`
- Phase Index: `reports/phase_index/vrp/convexity_vvix_threshold/`

## Data Infrastructure

**VVIX Loader**: `src.market_data.vrp_loaders.load_vvix()`
- Successfully loads VVIX from `market_data_cboe` table (symbol='VVIX')
- Fallback to FRED table (`f_fred_observations`, series_id='VVIXCLS') if CBOE unavailable
- VVIX data infrastructure is now available for future research

