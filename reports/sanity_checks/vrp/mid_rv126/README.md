# VRP-Mid (RV126) Phase-0 Signal Test

## Status
**PARKED — Phase-0 FAIL (both variants)**

## Test Specification

### Economic Thesis
Mid-horizon implied vs realized volatility premium: VIX (1-month implied volatility) vs RV126 (126-day realized volatility) should contain a tradable volatility risk premium.

### Phase-0 Signal Definition
- **Signal**: `signal = -1 if VIX > RV126 else 0`
- **Trade Expression**: Short VX when signal = -1, flat otherwise
- **Tested Variants**: VX2, VX3 (back-month futures only)
- **Discipline**: Sign-only, no z-scores, no filters, no vol targeting, constant unit exposure

### Data Requirements
- VIX: 1-month implied volatility index (FRED VIXCLS)
- RV126: 126-day rolling realized volatility from ES futures returns (annualized, vol points)
- VX Curve: VX2, VX3 futures prices
- Window: 2020-01-01 to 2025-10-31 (effective start: 2020-05-29 after RV126 warmup)

## Results

### VX2 Variant
- **Sharpe**: -0.1704
- **CAGR**: -24.27%
- **MaxDD**: -85.68% (catastrophic)
- **Hit Rate**: 33.26%
- **Active Days**: 81.9%
- **Verdict**: FAIL (catastrophic MaxDD, negative Sharpe)

### VX3 Variant
- **Sharpe**: -0.1517
- **CAGR**: -14.90%
- **MaxDD**: -73.88%
- **Hit Rate**: 34.50%
- **Active Days**: 81.9%
- **Verdict**: FAIL (negative Sharpe, large MaxDD)

## Interpretation

Both variants failed Phase-0 criteria (Sharpe < 0.10). The simple directional expression of "short VX when VIX > RV126" does not capture a profitable edge:

1. **Negative Sharpe across both tenors**: The spread (VIX - RV126) is positive ~82% of the time, but shorting VX in these regimes is not profitable.

2. **Crisis convexity risk**: VX2 shows catastrophic drawdown (-86%), indicating vulnerability to volatility spikes when short.

3. **Tenor effect**: VX3 performs better (less negative Sharpe, smaller MaxDD), suggesting back-month futures are less vulnerable to crisis convexity, but still not profitable.

4. **Horizon/instrument mismatch**: Mid-horizon implied vs realized volatility spread (VIX - RV126) may be a structural feature, but it does not translate to profitable directional short-VX trades, especially on back-month futures.

## Verdict

**FAILED Phase-0 (both variants) — No Phase-1**

The simple directional expression of VIX > RV126 → short VX2/VX3 does not contain a tradable edge. This is not an engine in simple form.

## Artifacts

- **Per-variant results**: 
  - `reports/sanity_checks/vrp/mid_rv126_vx2/latest/`
  - `reports/sanity_checks/vrp/mid_rv126_vx3/latest/`
- **Comparison summary**: `reports/sanity_checks/vrp/mid_rv126_compare/latest/summary.json`
- **Phase index**: `reports/phase_index/vrp/mid_rv126/`

## Future Revisit Options

If revisiting this idea, consider:

1. **Conditioning feature**: Use VIX - RV126 spread as a regime/conditioning input rather than a direct signal.

2. **Different instrument**: Test expression on different volatility instruments (e.g., VIX options, volatility ETFs).

3. **Different expression**: Test as a spread trade (long VIX, short realized vol proxy) rather than directional VX short.

4. **Macro filters**: Add macro regime filters (e.g., only trade in low-vol regimes, avoid crisis periods).

5. **Z-scoring/thresholds**: Apply z-scoring or dynamic thresholds rather than simple sign rule.

**Note**: Any revisit should be treated as a new Phase-0 with a different specification, not an engineering rescue of this failed expression.

