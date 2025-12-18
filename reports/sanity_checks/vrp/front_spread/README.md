# VRP Front-Spread Directional Sleeve

**Status: PARKED**

This sleeve is PARKED as of Phase-0. No Phase-1 or Phase-2 development will proceed under the current specification.

## Phase-0 Results

- **Sharpe**: -0.5846 (FAIL - hard fail; not borderline)
- **CAGR**: -34.31%
- **MaxDD**: -96.78% (catastrophic)
- **HitRate**: 6.05%
- **Signal Distribution**: 15.7% active (non-degenerate)
- **Status**: Economic failure - not proceeding to Phase-1

## Rationale

The Phase-0 sign-only test demonstrated that mapping calendar-richness (VX1 > VX2) to directional short VX1 exposure does not produce a positive economic edge.

**Key Findings:**

1. **Market Structure**: VX term structure is usually in backwardation (VX1 < VX2), so the contango signal (VX1 > VX2) only triggers in a small subset of regimes (~15.7% active days).

2. **Wrong Instrument / Wrong Payoff**: Even when contango exists, a simple directional short is not the right expression of "calendar carry." The results (CAGR -34%, MaxDD -97%, hit rate ~6%) indicate crisis convexity + bad timing, not "no carry exists."

3. **Economic Mapping Failure**: Calendar-richness does not map to profitable outright VX1 short. Curve relationships are better treated as features/regime inputs or spread trades, not outright VX1 directionality.

## Future Revisit Options

- Reframe as calendar-spread trade rather than directional VX1 exposure
- Consider using front-spread as a feature/regime input rather than a directional sleeve
- Potential integration into spread-based VRP research
- Calendar-spread-based research rather than outright VX1 directionality

## Phase-0 Script

The Phase-0 implementation is available at:
- `scripts/diagnostics/run_vrp_front_spread_phase0.py`
- Results: `reports/sanity_checks/vrp/front_spread/latest/`
- Phase Index: `reports/phase_index/vrp/front_spread_directional/`

