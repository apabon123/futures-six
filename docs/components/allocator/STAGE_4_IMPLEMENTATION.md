# Stage 4 Implementation Summary

## Stage 4A: Complete the 10-Feature State ✅

### Implementation

**Files Modified:**
- `src/agents/exec_sim.py`
- `scripts/diagnostics/run_allocator_state_v1.py`

### 4A.1: Trend Unit Returns Export ✅

**Artifact:** `trend_unit_returns.csv`
- Shape: `index=date, columns=assets`
- Computation: `weight[t-1] * return[t]` (lagged weights × asset returns)
- Aligned to `asset_returns.csv` date index and asset universe

**Implementation:**
```python
# In ExecSim._save_run_artifacts():
weights_lagged = weights_daily[common_symbols].shift(1).fillna(0.0)
returns_simple = np.exp(returns_aligned) - 1.0
trend_unit_returns_df = weights_lagged * returns_simple
trend_unit_returns_df.to_csv(run_dir / 'trend_unit_returns.csv')
```

### 4A.2: Sleeve Returns Export ✅

**Artifact:** `sleeve_returns.csv`
- Shape: `index=date, columns=sleeves`
- Computation: Per-sleeve signal contribution to portfolio returns
- Sleeves captured: `tsmom_multihorizon`, `csmom_meta`, `vrp_core`, `vrp_convergence`, `vrp_alt`, `sr3_curve_rv`, etc.

**Implementation:**
```python
# Capture per-sleeve signals during backtest loop:
for sleeve_name, sleeve_strategy in strategy.strategies.items():
    sleeve_sigs = sleeve_strategy.signals(market, date, features=features_dict)
    sleeve_signals_history[sleeve_name].append((date, sleeve_sigs * sleeve_weight))

# Compute sleeve returns in _save_run_artifacts():
sleeve_sigs_lagged = sleeve_sigs_daily.shift(1).fillna(0.0)
rets_simple = np.exp(rets_aligned) - 1.0
sleeve_returns_data[sleeve_name] = (sigs_aligned * rets_simple).sum(axis=1)
```

### 4A.3: Acceptance Criteria ✅

**Test Run:** `test_stage4a_all10`

**Results:**
```json
{
  "features_present": [
    "port_rvol_20d", "port_rvol_60d", "vol_accel",
    "dd_level", "dd_slope_10d",
    "corr_20d", "corr_60d", "corr_shock",
    "trend_breadth_20d",      // ✅ NEW
    "sleeve_concentration_60d" // ✅ NEW
  ],
  "features_missing": []  // ✅ Empty!
}
```

**Artifacts Generated:**
- ✅ `trend_unit_returns.csv` (76 rows × 13 assets)
- ✅ `sleeve_returns.csv` (76 rows × 2 sleeves: tsmom_multihorizon, csmom_meta)
- ✅ `allocator_state_v1.csv` with all 10 features

**Validator:** Passed with normal row-drop levels (59 rows for 60d warmup)

---

## Stage 4B: RegimeClassifierV1 (In Progress)

### Goal
Consume `allocator_state_v1.csv` and output a sticky, descriptive regime series:
- **NORMAL**: Typical market conditions
- **ELEVATED**: Increased volatility or correlation
- **STRESS**: Significant drawdown or volatility spike
- **CRISIS**: Extreme conditions requiring defensive positioning

### API (Locked)
```python
from src.allocator.regime_v1 import RegimeClassifierV1

classifier = RegimeClassifierV1()
regime = classifier.classify(state_df)
# Returns: pd.Series with values: 'NORMAL', 'ELEVATED', 'STRESS', 'CRISIS'
```

### Implementation Requirements

1. **Deterministic, Rule-Based**
   - No ML, no optimization
   - Pure threshold-based classification
   - Reproducible across runs

2. **Hysteresis / Stickiness**
   - Avoid thrash between regimes
   - Require sustained conditions to change regime
   - Implement minimum hold periods or confirmation windows

3. **Uses Core Features Initially**
   - Focus on 8 required features
   - Optionally incorporate trend_breadth_20d and sleeve_concentration_60d once stable

4. **Outputs**
   - `allocator_regime_v1.csv`: Daily regime series
   - `allocator_regime_v1_meta.json`: Rules version + transition counts

### Threshold Design (Proposed)

**Volatility Regime:**
- `vol_accel > 1.5`: Elevated (short-term vol >> long-term vol)
- `port_rvol_20d > 0.20`: Stress (20% annualized vol)
- `port_rvol_20d > 0.30`: Crisis (30% annualized vol)

**Drawdown Regime:**
- `dd_level < -0.05`: Elevated (5% drawdown)
- `dd_level < -0.10`: Stress (10% drawdown)
- `dd_level < -0.15`: Crisis (15% drawdown)
- `dd_slope_10d < -0.05`: Stress (rapid deterioration)

**Correlation Regime:**
- `corr_shock > 0.15`: Elevated (correlation spike)
- `corr_20d > 0.40`: Stress (high cross-asset correlation)
- `corr_20d > 0.50`: Crisis (extreme correlation)

**Combined Logic:**
- CRISIS: Any crisis-level threshold breached
- STRESS: Any stress-level threshold breached (and not crisis)
- ELEVATED: Any elevated-level threshold breached (and not stress/crisis)
- NORMAL: None of the above

**Hysteresis:**
- Regime changes require 3 consecutive days of new regime conditions
- Once in STRESS/CRISIS, require 5 consecutive days of lower regime to downgrade

---

## Stage 4C: RiskTransformerV1 (Pending)

### Goal
Map regimes to portfolio-level risk scalars:
- **NORMAL**: `risk_scalar = 1.0` (no adjustment)
- **ELEVATED**: `risk_scalar = 0.7-0.9` (moderate reduction)
- **STRESS**: `risk_scalar = 0.4-0.7` (significant reduction)
- **CRISIS**: `risk_scalar = 0.0-0.4` (defensive positioning)

### API (Locked)
```python
from src.allocator.risk_v1 import RiskTransformerV1

transformer = RiskTransformerV1()
risk_scalars = transformer.transform(state_df, regime)
# Returns: pd.DataFrame with 'risk_scalar' column
```

### Output Schema
```python
pd.DataFrame(index=date, columns=["risk_scalar"])
# risk_scalar: float in [0.0, 1.0]
```

---

## Stage 4D: No-Rewire Integration Point (Pending)

### Integration Design

**Single Integration Point:**
```python
# In ExecSim.run(), just before position sizing:

# Load risk_scalar (if available)
risk_scalar = 1.0  # Default: no adjustment
if risk_scalar_df is not None and date in risk_scalar_df.index:
    risk_scalar = risk_scalar_df.loc[date, 'risk_scalar']

# Apply scalar to weights
weights = allocator.solve(scaled_signals, cov, weights_prev)
weights = weights * risk_scalar  # Simple multiplicative scaling
```

**No Other Changes Required:**
- Signals remain unchanged
- Vol targeting remains unchanged
- Allocator logic remains unchanged
- Only final weights are scaled

---

## Testing Summary

### Stage 4A Tests ✅

**Test 1: Short Window (2024-09-01 to 2024-12-31)**
- Run: `test_stage4a_complete`
- Result: 9/10 features (missing `sleeve_concentration_60d`)
- Issue: `sleeve_signals_history` not passed to `_save_run_artifacts`

**Test 2: Short Window (2024-10-01 to 2024-12-31)**
- Run: `test_stage4a_all10`
- Result: ✅ 10/10 features present
- Artifacts: `trend_unit_returns.csv`, `sleeve_returns.csv`, `allocator_state_v1.csv`
- Validator: Passed

---

## Key Design Decisions

### 1. Trend Unit Returns = Lagged Weights × Returns

**Rationale:**
- Represents the per-asset contribution to portfolio P&L
- Lagged weights (t-1) ensure no lookahead
- Aligned to asset universe and date index

### 2. Sleeve Returns = Sleeve Signals × Asset Returns

**Rationale:**
- Approximates each sleeve's contribution to portfolio returns
- Captures sleeve-level P&L attribution
- Enables sleeve concentration monitoring (Herfindahl index)

### 3. Sleeve Signals Captured During Backtest Loop

**Rationale:**
- Avoids recomputing signals in `_save_run_artifacts`
- Ensures consistency with actual trading signals
- Minimal performance overhead (signals already computed)

### 4. Simple Returns for Interpretability

**Rationale:**
- Trend unit returns and sleeve returns use simple returns (not log)
- More interpretable for contribution attribution
- Consistent with allocator state expectations

---

## Files Modified

### Stage 4A:
1. `src/agents/exec_sim.py`
   - Added `sleeve_signals_history` tracking
   - Added `trend_unit_returns` computation in `_save_run_artifacts`
   - Added `sleeve_returns` computation in `_save_run_artifacts`
   - Wired both into `AllocatorStateV1.compute()`

2. `scripts/diagnostics/run_allocator_state_v1.py`
   - Enhanced loading of optional artifacts with try/except
   - Added informative messages for Stage 4A artifacts

### Stage 4B (In Progress):
1. `src/allocator/regime_v1.py` (to be implemented)

### Stage 4C (Pending):
1. `src/allocator/risk_v1.py` (to be implemented)

### Stage 4D (Pending):
1. `src/agents/exec_sim.py` (integration point)

---

**Stage 4A Complete: December 18, 2025**
**Stage 4B In Progress**

