# Carry Meta-Sleeve v1 — Phase-1 Implementation Plan

**Date**: January 21, 2026  
**Status**: Ready to start after Commit B validation  
**Branch**: `carry-phase1-v1`

---

## Phase-1 Objective

Transform Phase-0 sign-only signals into a "real signal" with proper normalization, risk scaling, and cross-sectional ranking.

**Goal**: Achieve Post-Construction Sharpe ≥ 0.2 (Phase-0 was 0.181, just below threshold)

---

## Phase-1 Enhancements (Engine-Only, Legal)

### 1. Per-Asset Z-Score Normalization

**Implementation**: Rolling 252-day z-score of raw carry values

**Code Location**: `src/agents/strat_carry_meta_v1.py` → `_signals_phase1()`

**Formula**:
```python
carry_z = (carry_raw - rolling_mean) / rolling_std
carry_z_clipped = np.clip(carry_z, -clip, clip)  # clip = 3.0
```

**Rationale**: Standardize carry signals across assets with different scales (equity vs rates vs FX)

---

### 2. Per-Asset Vol Normalization

**Implementation**: Target unit risk per asset within the sleeve

**Code Location**: `src/agents/strat_carry_meta_v1.py` → `_signals_phase1()`

**Formula**:
```python
# Compute rolling volatility (252-day) for each asset
asset_vol = returns.rolling(252).std() * np.sqrt(252)

# Normalize signals to target unit risk
signal_normalized = signal_z / asset_vol
```

**Rationale**: Risk parity approach - each asset contributes equal risk to the portfolio

---

### 3. Winsorization / Clipping

**Implementation**: Clip extreme carry values to ±3.0 (already in z-score step)

**Code Location**: Same as z-score normalization

**Rationale**: Prevent extreme signals from dominating portfolio

---

### 4. Within-Asset-Class Cross-Sectional Ranking (Optional)

**Implementation**: Rank assets within each asset class (equity vs equity, rates vs rates, etc.)

**Code Location**: `src/agents/strat_carry_meta_v1.py` → `_signals_phase1()`

**Formula**:
```python
# Within asset class, rank by carry strength
for asset_class in ["equity", "fx", "rates", "commodity"]:
    class_signals = signals[asset_class_symbols]
    class_ranks = class_signals.rank(pct=True)  # 0-1 percentile rank
    signals[asset_class_symbols] = class_ranks * 2 - 1  # Convert to -1 to +1
```

**Rationale**: Emphasize relative carry strength within asset classes

**Status**: Optional enhancement (can be Phase-1.1 if needed)

---

## What We Do NOT Add (Phase-2+)

- ❌ **Policy gating**: Keep in Phase-2+ if ever
- ❌ **Regime detection**: Not in Phase-1 scope
- ❌ **RT/Allocator changes**: Frozen stack, no modifications
- ❌ **Cross-asset-class ranking**: Keep equal-weight across classes for Phase-1

---

## What We Measure

### Primary Metrics

1. **Post-Construction Sharpe**: Target ≥ 0.2 (Phase-0 was 0.181)
2. **Post-Construction MaxDD**: Monitor for degradation vs Phase-0 (-25.81%)

### Secondary Metrics

3. **Asset-Class Contribution Breakdown**:
   - Equity carry contribution
   - FX carry contribution
   - Rates carry contribution
   - Commodity carry contribution

4. **Stress Window Behavior**:
   - 2020 Q1 (COVID crash): Phase-0 was -1.56% ✅
   - 2022 rates shock: Monitor for acceptable behavior

5. **Correlation Analysis**:
   - Correlation with Trend (TSMOM)
   - Correlation with VRP
   - Correlation with Curve RV

---

## Implementation Steps

### Step 1: Create Phase-1 Branch

```bash
git checkout -b carry-phase1-v1
```

### Step 2: Update `strat_carry_meta_v1.py`

**Add `_signals_phase1()` method**:

```python
def _signals_phase1(self, features_row: pd.Series, market, date_dt) -> pd.Series:
    """
    Phase-1: Z-scored, vol-normalized signals with clipping.
    
    Steps:
    1. Get raw carry values
    2. Compute rolling z-scores (252d window)
    3. Clip to ±3.0
    4. Vol normalize (target unit risk per asset)
    5. Optional: Cross-sectional ranking within asset classes
    """
    # Implementation here
    pass
```

**Update `signals()` method** to route to `_signals_phase1()` when `phase == 1`

### Step 3: Update Config

**Create `configs/carry_phase1_v1.yaml`**:

```yaml
carry_meta_v1:
  enabled: true
  weight: 1.0
  params:
    phase: 1  # Phase-1: Z-scored, vol-normalized
    enabled_asset_classes: ["equity", "fx", "rates", "commodity"]
    equity_symbols: ["ES", "NQ", "RTY"]
    fx_symbols: ["6E", "6B", "6J"]
    rates_symbols: ["ZT", "ZF", "ZN", "UB"]
    commodity_symbols: ["CL", "GC"]
    window: 252
    clip: 3.0
    rebalance: "D"
    cross_sectional_ranking: false  # Optional, start with false
```

### Step 4: Run Phase-1 Backtest

```bash
python run_strategy.py \
  --start 2020-01-01 \
  --end 2025-10-31 \
  --config_path configs/carry_phase1_v1.yaml \
  --run_id carry_phase1_v1_$(date +%Y%m%d_%H%M%S)
```

### Step 5: Compare vs Phase-0

**Metrics to Compare**:
- Sharpe ratio (target: ≥ 0.2)
- Max drawdown (monitor degradation)
- 2020 Q1 return (maintain acceptable behavior)
- Asset-class contributions (identify strong/weak classes)

---

## Success Criteria

**Phase-1 Pass**:
- ✅ Post-Construction Sharpe ≥ 0.2
- ✅ MaxDD acceptable (≤ -30% or better than Phase-0)
- ✅ 2020 Q1 return > -20%
- ✅ At least 2 asset classes positive contribution

**Phase-1 Fail → Investigate**:
- Sharpe < 0.2: Check z-score/vol normalization implementation
- MaxDD degradation: Review clipping bounds
- Asset-class breakdown: Identify weak classes

---

## Rollback Plan

If Phase-1 degrades vs Phase-0:
1. Revert to Phase-0 implementation
2. Investigate per-asset-class performance
3. Consider Phase-1.1 with cross-sectional ranking only
4. Document findings in Phase-1 memo

---

## Documentation

**Phase-1 Artifacts**:
- `carry_phase1_run_memo.md` (similar to Phase-0 memo)
- `reports/runs/carry_phase1_v1_*/` (full run artifacts)
- Asset-class contribution analysis
- Correlation analysis vs Trend/VRP

---

**End of Plan**
