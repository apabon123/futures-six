# Dual-Price Architecture: Raw vs Continuous Prices

This document describes the **dual-price architecture** that separates raw contract prices (from the database) from back-adjusted continuous prices (computed in-memory) for all futures trading calculations.

## Overview

The system maintains **two separate price series** for each symbol:

1. **Raw prices** (`prices_raw`): Actual settlement prices from the database, with roll jumps intact
   - Used for: Contract sizing, notional calculations, margin requirements
   - Source: Direct from database
   - Format: `pd.DataFrame` indexed by date, columns = symbols

2. **Continuous prices** (`prices_cont`): Back-adjusted prices with roll jumps removed
   - Used for: Returns, features, volatility, covariance, P&L calculations
   - Source: Built in-memory using `ContinuousContractBuilder`
   - Format: `pd.DataFrame` indexed by date, columns = symbols

## The Rule of Thumb

**"What did the market do?" → Use continuous prices**  
**"How many contracts/$ risk?" → Use raw prices**

## Architecture Components

### 1. ContinuousContractBuilder Service

**Location:** `src/services/continuous_contract_builder.py`

**Algorithm:** Backward-Panama back-adjustment

```python
# At each roll point (contract_id changes):
1. Compute gap: gap = raw_new - raw_old
2. Accumulate adjustment: adj += gap
3. Apply adjustment: cont_price = raw_price - adj
```

**Input:** DataFrame with columns `['close', 'contract_id']`, indexed by date  
**Output:** Back-adjusted `close` price Series

**Properties:**
- No jumps at roll points (continuous series)
- Level differences between contracts preserved as history
- First price is unadjusted (no roll has occurred yet)

### 2. MarketData Properties

**Location:** `src/agents/data_broker.py`

**New Properties:**
- `market.prices_raw`: Raw close prices [date × symbol]
- `market.prices_cont`: Back-adjusted close prices [date × symbol]
- `market.contract_ids`: Contract IDs per [date × symbol]
- `market.returns_cont`: Log returns from `prices_cont` [date × symbol]

**Implementation:**
- All properties are **lazy-loaded** and cached
- Built on first access via `_build_continuous_prices()`
- Checks for `contract_id` column in database (optional)
- Falls back to using `symbol` as `contract_id` if not available

### 3. Module Usage Map

#### ✅ Use Continuous Prices/Returns

| Module | Uses | Purpose |
|--------|------|---------|
| **Features** | `prices_cont`, `returns_cont` | Momentum, trend, breakout calculations |
| **Rates Curve** | `prices_cont` | Yield curve construction |
| **Macro Regime** | `prices_cont`, `returns_cont` | Realized vol, breadth (200d SMA) |
| **RiskVol** | `returns_cont` | Covariance matrix, volatility |
| **Allocator** | `returns_cont` | Portfolio optimization inputs |
| **ExecSim** | `returns_cont` | P&L calculation |

#### 📦 Use Raw Prices (Future)

| Module | Uses | Purpose |
|--------|------|---------|
| **ExecSim** (sizing) | `prices_raw` | Contract quantity: `qty = notional / (price_raw * multiplier)` |

**Note:** Currently in dimensionless weights mode, so raw prices are not yet used for sizing.

### 4. Roll Jump Filter Removal

**Before:** `ExecSim` filtered roll jumps using `RollJumpFilter` (100 bp threshold)

**After:** Roll jump filtering removed because:
- Continuous returns have no jumps (back-adjusted)
- Backward-panama adjustment removes price gaps at roll points
- P&L is computed correctly from continuous returns

**RollJumpFilter:** Kept in codebase for diagnostics/debugging, but no longer used in production P&L.

## Data Flow Diagram

```
Database
  │
  ├─► Raw Prices (with roll jumps)
  │   │
  │   └─► market.prices_raw
  │       └─► Future: Contract sizing
  │
  ├─► Contract IDs
  │   │
  │   └─► market.contract_ids
  │
  └─► ContinuousContractBuilder
        │
        ├─► Detect rolls (contract_id changes)
        ├─► Compute gaps at roll points
        ├─► Accumulate adjustments
        └─► Build back-adjusted series
            │
            └─► market.prices_cont
                │
                ├─► market.returns_cont
                │
                └─► Used by:
                    ├─► Feature calculations (momentum, trends, breakouts)
                    ├─► Volatility & covariance (RiskVol)
                    ├─► Macro regime overlay (realized vol, breadth)
                    └─► P&L calculation (ExecSim)
```

## Validation & Diagnostics

### Continuous Price Validation

**Location:** `src/diagnostics/continuous_price_validation.py`

**Purpose:** Ensure `ContinuousContractBuilder` is working correctly

**Checks:**
1. **Shape/NaN validation**: Raw and continuous prices have matching shapes, no unexpected NaNs
2. **Non-roll day matching**: On non-roll days, raw and continuous returns should match exactly (diff ≈ 0)
3. **Roll day gap removal**: On roll days, raw has jumps but continuous doesn't (gaps vanish)
4. **Summary statistics**: Number of symbols, days, roll days, percentage of roll days

**Integration:** Runs automatically in Phase 3 validation (`scripts/run_phase3_stabilization.py`)

**Expected Output:**
```
✓ Non-roll daily returns match as expected (max diff≈0)
✓ Roll jumps successfully removed from continuous series
  Max raw return on roll days: [shows large jumps in bps]
  Max cont return on roll days: [shows normal daily moves]
```

## Migration Summary

### What Changed

1. **Created `ContinuousContractBuilder`** service for back-adjustment
2. **Updated `MarketData`** to expose `prices_raw`, `prices_cont`, `contract_ids`, `returns_cont`
3. **Migrated all features** to use continuous prices/returns
4. **Migrated overlays** to use continuous prices/returns
5. **Migrated RiskVol/Allocator** to use continuous returns
6. **Migrated ExecSim** to use continuous returns for P&L
7. **Removed roll-jump filtering** from ExecSim (no longer needed)
8. **Added validation** to ensure builder works correctly

### What Stays the Same

- Database structure unchanged (no writes)
- Raw prices available via `prices_raw` property
- RollJumpFilter still exists (for diagnostics)
- All existing APIs still work (backward compatible)

## Future Work

### Contract Sizing (When Needed)

When implementing real contract sizing:

```python
# In ExecSim or sizing layer:
price_raw = market.prices_raw.loc[date, symbol]
qty = target_notional / (price_raw * multiplier)

# P&L still uses continuous:
ret_cont = market.returns_cont.loc[date, symbol]
pnl = qty_prev * ret_cont * (price_cont_prev * multiplier)
```

**Key Point:** Sizing uses raw prices (actual contract prices), P&L uses continuous returns (smooth series).

## Files Modified

### New Files
- `src/services/continuous_contract_builder.py` - Back-adjustment service
- `src/services/__init__.py` - Services module init
- `src/diagnostics/continuous_price_validation.py` - Validation diagnostics
- `docs/DUAL_PRICE_ARCHITECTURE.md` - This document

### Modified Files
- `src/agents/data_broker.py` - Added dual-price properties
- `src/agents/feature_long_momentum.py` - Uses continuous prices/returns
- `src/agents/feature_rates_curve.py` - Uses continuous prices
- `src/agents/risk_vol.py` - Uses continuous returns
- `src/agents/overlay_macro_regime.py` - Uses continuous prices/returns
- `src/agents/exec_sim.py` - Uses continuous returns, removed roll-jump filtering
- `src/diagnostics/__init__.py` - Exports validation functions
- `scripts/run_phase3_stabilization.py` - Added continuous price validation

## References

- **Strategy Execution Flow**: See `docs/STRATEGY.md` for how continuous prices are used in the execution pipeline
- **Implementation Details**: See `src/services/continuous_contract_builder.py` for back-adjustment algorithm
- **Validation**: See `src/diagnostics/continuous_price_validation.py` for validation logic

