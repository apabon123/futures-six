# VRP-TermStructure Sleeve

**Status: PARKED**

This sleeve is PARKED as of Phase-0. No Phase-1 or Phase-2 development will proceed under the current specification.

## Phase-0 Results

- **Sharpe**: -0.63 (FAIL)
- **CAGR**: -52.4%
- **MaxDD**: -98.9%
- **Status**: Economic failure - not proceeding to Phase-1

## Rationale

The Phase-0 sign-only test demonstrated that mapping VX2-VX1 slope to directional short VX1 exposure does not produce a positive economic edge. The strategy suffered large losses during volatility spikes where contango persisted.

## Future Revisit Options

- Use term-structure slope as a regime filter rather than a directional VRP sleeve
- Potential integration into Crisis Meta-Sleeve
- Calendar-spread-based research rather than outright VX1 directionality

## Phase-0 Script

The Phase-0 implementation is available at:
- `scripts/vrp/run_vrp_termstructure_phase0.py`
- Results: `reports/sanity_checks/vrp/vrp_termstructure/latest/`

