# Carry Meta-Sleeve v1 — Phase-1 Implementation

**Date**: January 21, 2026  
**Status**: ✅ **IMPLEMENTED**  
**Phase**: Phase-1 (Clean Implementation)

---

## Implementation Summary

Phase-1 transforms Phase-0 sign-only signals into a clean, standardized signal with:
- ✅ Rolling z-score normalization (252d window)
- ✅ Clipping at ±3.0
- ✅ Vol normalization (equal risk per asset within sleeve)
- ⏸️ Cross-sectional ranking (optional, not yet implemented)

---

## Phase-1 Feature Contract

### Step A: Raw Carry (Already Done)

**Equity**: `r - d` via implied dividends  
**FX**: Rate differentials (via roll yield)  
**Rates**: Rolldown slope (near-far mapping)  
**Commodity**: `log(F2/F1)` (roll yield)

### Step B: Standardize Per Instrument (Rolling Z-Score)

**Window**: 252 trading days  
**Method**: Plain mean/std (robust z-score optional for v1.1)

**Implementation**:
- Equity: Re-z-score `equity_carry_raw_{symbol}` (252d rolling)
- Rates: Re-z-score `rates_carry_raw_{symbol}` (252d rolling)
- FX/Commodity: Use pre-computed `carry_ts_z_{root}` (already z-scored in feature module)

**Output**: `carry_z_i(t)` per instrument

### Step C: Clip / Winsorize

**Bounds**: ±3.0 (configurable via `clip` parameter)

**Implementation**: `np.clip(carry_z, -clip, clip)`

**Output**: `carry_z_clip_i(t)`

### Step D: Vol-Normalize Sleeve Exposures

**Target**: Equal risk per asset within the Carry sleeve

**Method**:
1. Compute rolling volatility (252d annualized) for each asset
2. Normalize: `signal_normalized = signal_z / asset_vol`
3. This makes each asset contribute equal risk to the sleeve

**Scope**: Local to Carry sleeve (not portfolio RT)

**Output**: Vol-normalized signals

### Step E: Cross-Sectional Ranking (Optional, Not Yet Implemented)

**Scope**: Within asset classes only
- Rates vs rates (ZT, ZF, ZN, UB)
- FX vs FX (6E, 6B, 6J)
- Commodities vs commodities (CL, GC)
- Equities vs equities (ES, NQ, RTY)

**Status**: ⏸️ Deferred to Phase-1.1 if needed

---

## Code Implementation

### File: `src/agents/strat_carry_meta_v1.py`

**Method**: `_signals_phase1(features, market, date)`

**Key Steps**:
1. Extract raw carry values from features DataFrame
2. Compute rolling z-score (252d window) for equity/rates
3. Use pre-computed z-scores for FX/commodity
4. Clip at ±3.0
5. Vol-normalize using market returns (252d rolling vol)
6. Return normalized signals

**NA Handling**: Canonical `dropna(how="any")` at feature aggregation stage

---

## Configuration

### File: `configs/carry_phase1_v1.yaml`

**Key Settings**:
```yaml
carry_meta_v1:
  enabled: true
  weight: 1.0
  params:
    phase: 1  # Phase-1
    window: 252  # Rolling window for z-score
    clip: 3.0    # Z-score clipping bounds
    enabled_asset_classes: ["equity", "fx", "rates", "commodity"]
```

**Disabled**:
- ✅ Policy: `enabled: false`
- ✅ RT: `enabled: false`
- ✅ Allocator: `enabled: false`
- ✅ Vol Overlay: `enabled: false`

**Evaluation**: Post-Construction (belief layer)

---

## Running Phase-1

### Diagnostic Script

```bash
python scripts/diagnostics/run_carry_phase1_v1.py \
    --start 2020-01-01 \
    --end 2025-10-31 \
    --config configs/carry_phase1_v1.yaml
```

### Direct Backtest

```bash
python run_strategy.py \
    --start 2020-01-01 \
    --end 2025-10-31 \
    --config_path configs/carry_phase1_v1.yaml \
    --run_id carry_phase1_v1_$(date +%Y%m%d_%H%M%S)
```

---

## Acceptance Criteria

### Pass (Recommended)

- ✅ Sharpe ≥ 0.25 at Post-Construction
- ✅ MaxDD improves or does not worsen materially vs Phase-0
- ✅ No single asset dominates risk in stress windows
- ✅ Crisis behavior remains sane (2020 Q1, 2022)

### Conditional Pass

- ⚠️ Sharpe 0.20-0.25 but:
  - Clear positive contribution from ≥2 asset classes
  - Crisis behavior remains sane

### Fail

- ❌ Sharpe < 0.20
- ❌ Behavior is basically equity beta
- ❌ Single instrument drives everything

---

## Diagnostics Generated

**Minimum Pack**:
1. ✅ Canonical metrics (CAGR/Vol/Sharpe/MaxDD)
2. ✅ Year-by-year stats
3. ✅ Per-asset contribution (to be added)
4. ✅ Stress window slices: 2020 Q1 and 2022
5. ⏸️ Correlation vs Trend/VRP (Phase-2)

**Output**: `reports/runs/carry_phase1_v1_*/phase1_analysis_summary.json`

---

## Comparison to Phase-0

| Aspect | Phase-0 | Phase-1 |
|--------|---------|---------|
| **Signal Type** | Sign-only | Z-scored |
| **Normalization** | None | Rolling z-score (252d) |
| **Clipping** | None | ±3.0 |
| **Vol Normalization** | None | Equal risk per asset |
| **Cross-Sectional** | None | Optional (deferred) |
| **Sharpe Target** | ≥ 0.2 | ≥ 0.25 (recommended) |

---

## Next Steps

1. **Run Phase-1 Backtest**: Execute diagnostic script
2. **Evaluate Results**: Compare vs Phase-0 and acceptance criteria
3. **If Pass**: Proceed to Phase-2 (Integration)
4. **If Conditional**: Review asset-class contributions, consider Phase-1.1 with cross-sectional ranking
5. **If Fail**: Investigate root cause (sign logic, vol normalization, data quality)

---

## Governance

**Engine-Only**: ✅ All changes are Layer 1 (Engine Signals)  
**Frozen Stack**: ✅ No changes to RT, Allocator, Construction, Policy  
**NA Handling**: ✅ Canonical `dropna(how="any")` at feature aggregation  
**Single-Change Discipline**: ✅ Only adds z+clip+volnorm; nothing else

---

**End of Document**
