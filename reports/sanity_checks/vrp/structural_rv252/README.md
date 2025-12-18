# VRP-Structural (RV252) Phase-0 Signal Test

## Status
**PARKED — Phase-0 FAIL (all variants)**

## Test Specification

### Economic Thesis
Long-horizon implied vs realized volatility premium: VIX (1-month implied volatility) vs RV252 (252-day realized volatility) should contain a tradable volatility risk premium.

### Phase-0 Signal Definition
- **Signal**: `signal = -1 if VIX > RV252 else 0`
- **Trade Expression**: Short VX when signal = -1, flat otherwise
- **Tested Variants**: VX1, VX2, VX3 (all three tenors)
- **Discipline**: Sign-only, no z-scores, no filters, no vol targeting, constant unit exposure

### Data Requirements
- VIX: 1-month implied volatility index (FRED VIXCLS)
- RV252: 252-day rolling realized volatility from ES futures returns (annualized, vol points)
- VX Curve: VX1, VX2, VX3 futures prices
- Window: 2020-01-01 to 2025-10-31 (effective start: 2020-10-23 after RV252 warmup)

## Results

### VX1 Variant
- **Sharpe**: -0.1817
- **CAGR**: -38.92%
- **MaxDD**: -92.90% (catastrophic)
- **Hit Rate**: 30.02%
- **Active Days**: 75.0%
- **Verdict**: FAIL (catastrophic MaxDD, negative Sharpe)

### VX2 Variant
- **Sharpe**: -0.1663
- **CAGR**: -24.14%
- **MaxDD**: -78.28%
- **Hit Rate**: 30.84%
- **Active Days**: 74.9%
- **Verdict**: FAIL (negative Sharpe, large MaxDD)

### VX3 Variant
- **Sharpe**: -0.1481
- **CAGR**: -14.81%
- **MaxDD**: -60.04%
- **Hit Rate**: 31.94%
- **Active Days**: 74.9%
- **Verdict**: FAIL (negative Sharpe, but least bad performer)

## Interpretation

All three variants failed Phase-0 criteria (Sharpe < 0.10). The simple directional expression of "short VX when VIX > RV252" does not capture a profitable edge:

1. **Negative Sharpe across all tenors**: The spread (VIX - RV252) is positive ~75% of the time, but shorting VX in these regimes is not profitable.

2. **Crisis convexity risk**: VX1 shows catastrophic drawdown (-93%), indicating vulnerability to volatility spikes when short.

3. **Tenor effect**: VX3 performs best (least negative Sharpe, smallest MaxDD), suggesting back-month futures are less vulnerable to crisis convexity, but still not profitable.

4. **Economic mismatch**: Long-horizon implied vs realized volatility spread may be a structural feature (VIX typically > RV252 in normal markets), but it does not translate to profitable directional short-VX trades.

## Verdict

**FAILED Phase-0 (all variants) — No Phase-1**

The simple directional expression of VIX > RV252 → short VX does not contain a tradable edge. This is not an engine in simple form.

## Artifacts

- **Per-variant results**: 
  - `reports/sanity_checks/vrp/structural_rv252_vx1/latest/`
  - `reports/sanity_checks/vrp/structural_rv252_vx2/latest/`
  - `reports/sanity_checks/vrp/structural_rv252_vx3/latest/`
- **Comparison summary**: `reports/sanity_checks/vrp/structural_rv252_compare/latest/summary.json`
- **Phase index**: `reports/phase_index/vrp/structural_rv252/`

## Future Revisit Options

If revisiting this idea, consider:

1. **Conditioning feature**: Use VIX - RV252 spread as a regime/conditioning input rather than a direct signal.

2. **Different instrument**: Test expression on different volatility instruments (e.g., VIX options, volatility ETFs).

3. **Different expression**: Test as a spread trade (long VIX, short realized vol proxy) rather than directional VX short.

4. **Macro filters**: Add macro regime filters (e.g., only trade in low-vol regimes, avoid crisis periods).

5. **Z-scoring/thresholds**: Apply z-scoring or dynamic thresholds rather than simple sign rule.

**Note**: Any revisit should be treated as a new Phase-0 with a different specification, not an engineering rescue of this failed expression.

