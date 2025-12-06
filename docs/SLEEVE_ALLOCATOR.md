# PortfolioSleeveAllocator

## Overview

The `PortfolioSleeveAllocator` combines multiple strategy sleeves (TSMOM, XSec, Carry, etc.) into unified portfolio weights using explicit risk budgets. It operates on scaled signals that have already passed through MacroRegime and VolManaged overlays.

## Key Features

- **Risk Budget Allocation**: Combines sleeves using explicit risk weights (e.g., 60% TSMOM, 40% XSec)
- **Portfolio Constraints**: Enforces gross/net caps, per-asset bounds, and turnover limits
- **Deterministic**: No randomness, same inputs always produce same outputs
- **No Database Writes**: Pure computation, no side effects
- **L2 Projection**: Uses iterative projection to find nearest feasible solution when constraints are violated

## Architecture

### Workflow Integration

```
┌─────────────────────────────────────────────────────────────┐
│ Multi-Sleeve Portfolio Construction                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  For Each Sleeve (TSMOM, XSec, Carry, ...):                │
│    1. Generate raw signals                                  │
│    2. Apply MacroRegime overlay (optional)                  │
│    3. Apply VolManaged overlay                              │
│    → Result: Scaled positions in risk units                 │
│                                                              │
│  PortfolioSleeveAllocator:                                  │
│    1. Combine sleeves with risk budgets                     │
│    2. Aggregate: total = Σ(w_sleeve * pos_sleeve)          │
│    3. Enforce constraints (gross/net/bounds/turnover)       │
│    → Result: Final portfolio weights                        │
│                                                              │
│  ExecSim:                                                    │
│    1. Apply weights to market returns                       │
│    2. Calculate transaction costs                           │
│    3. Track performance metrics                             │
│    → Result: Equity curve and diagnostics                   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## API Reference

### AllocationConstraints

Configuration dataclass for portfolio constraints.

```python
@dataclass
class AllocationConstraints:
    bounds_per_asset: Tuple[float, float] = (-1.5, 1.5)  # Min/max per asset
    gross_cap: float = 7.0                                 # Max sum(abs(weights))
    net_cap: float = 2.0                                   # Max abs(sum(weights))
    turnover_cap: Optional[float] = None                   # Max turnover vs prev
```

### PortfolioSleeveAllocator

Main allocator class.

#### Constructor

```python
def __init__(self, constraints: Optional[AllocationConstraints] = None)
```

**Args:**
- `constraints`: Portfolio constraints. Uses defaults if None.

#### combine()

Main method to combine sleeves into portfolio weights.

```python
def combine(
    self,
    sleeve_positions: Dict[str, pd.Series],
    risk_budgets: Dict[str, float],
    prev_weights: Optional[pd.Series] = None
) -> Dict[str, pd.Series]
```

**Args:**
- `sleeve_positions`: Dict mapping sleeve name to position Series (in risk units, post-overlay)
- `risk_budgets`: Dict mapping sleeve name to risk budget weight (must sum to 1.0)
- `prev_weights`: Previous total weights for turnover constraint (optional)

**Returns:**

Dict with keys:
- `'total_weights'`: pd.Series of combined portfolio weights
- `'per_sleeve_contrib'`: pd.DataFrame of per-sleeve contributions (assets × sleeves)
- `'feasibility'`: Dict with constraint violation info

**Raises:**
- `ValueError`: If inputs are invalid (empty, budgets don't sum to 1.0, etc.)

## Usage Examples

### Basic Two-Sleeve Combination

```python
from src.agents.sleeve_allocator import (
    PortfolioSleeveAllocator,
    AllocationConstraints
)

# Create allocator
constraints = AllocationConstraints(
    bounds_per_asset=(-1.5, 1.5),
    gross_cap=7.0,
    net_cap=2.0
)
allocator = PortfolioSleeveAllocator(constraints)

# Sleeve positions (post-overlay, in risk units)
sleeve_positions = {
    'tsmom': pd.Series({'ES': 0.8, 'GC': -0.4, 'CL': 0.6}),
    'xsec': pd.Series({'ES': -0.3, 'GC': 0.7, 'SI': 0.5})
}

# Risk budgets (must sum to 1.0)
risk_budgets = {'tsmom': 0.6, 'xsec': 0.4}

# Combine
result = allocator.combine(sleeve_positions, risk_budgets)

# Extract results
total_weights = result['total_weights']
per_sleeve = result['per_sleeve_contrib']
```

### Three-Sleeve Portfolio

```python
sleeve_positions = {
    'tsmom': pd.Series({'ES': 0.5, 'GC': -0.3, 'CL': 0.4}),
    'xsec': pd.Series({'ES': -0.2, 'GC': 0.4, 'SI': 0.3}),
    'carry': pd.Series({'GC': 0.6, 'CL': -0.2, 'SI': 0.1})
}

risk_budgets = {'tsmom': 0.5, 'xsec': 0.3, 'carry': 0.2}

result = allocator.combine(sleeve_positions, risk_budgets)
```

### With Turnover Constraint

```python
constraints = AllocationConstraints(
    bounds_per_asset=(-1.5, 1.5),
    gross_cap=7.0,
    net_cap=2.0,
    turnover_cap=2.0  # Max 2.0 turnover
)

allocator = PortfolioSleeveAllocator(constraints)

# Previous weights from last rebalance
prev_weights = pd.Series({'ES': 0.5, 'GC': -0.2, 'CL': 0.3})

result = allocator.combine(
    sleeve_positions,
    risk_budgets,
    prev_weights=prev_weights
)
```

## Integration with ExecSim

The sleeve allocator integrates into the backtest workflow at the allocation stage. Here's how to modify your backtest:

### Traditional Single-Sleeve Workflow

```python
# Traditional (single strategy)
for date in rebalance_dates:
    signals = strategy.signals(market, date)
    scaled_signals = overlay.scale(signals, market, date)
    cov = risk_vol.covariance(market, date)
    weights = allocator.solve(scaled_signals, cov, prev_weights)
    # ... apply returns ...
```

### Multi-Sleeve Workflow

```python
# Multi-sleeve
for date in rebalance_dates:
    # Step 1: Generate and scale each sleeve
    sleeve_positions = {}
    
    for sleeve_name, strategy in strategies.items():
        signals = strategy.signals(market, date)
        
        # Apply macro regime overlay (optional)
        if macro_overlay:
            macro_scaler = macro_overlay.scaler(market, date)
            signals = signals * macro_scaler
        
        # Apply vol-managed overlay
        scaled = vol_overlay.scale(signals, market, date)
        sleeve_positions[sleeve_name] = scaled
    
    # Step 2: Combine sleeves with risk budgets
    result = sleeve_allocator.combine(
        sleeve_positions,
        risk_budgets={'tsmom': 0.6, 'xsec': 0.4},
        prev_weights=prev_weights
    )
    
    weights = result['total_weights']
    per_sleeve = result['per_sleeve_contrib']
    
    # Store diagnostics
    diagnostics[date] = {
        'per_sleeve': per_sleeve,
        'feasibility': result['feasibility']
    }
    
    # ... apply returns ...
```

## Constraint Enforcement

When constraints are violated, the allocator uses iterative L2 projection to find the nearest feasible solution.

### Projection Algorithm

1. **Per-asset bounds**: Clip each weight to [min, max]
2. **Turnover cap**: Scale trade to respect turnover limit
3. **Gross cap**: Scale all positions proportionally
4. **Net cap**: Shift all positions equally
5. **Repeat** until convergence

The algorithm converges quickly (typically < 100 iterations) and produces deterministic results.

### Feasibility Information

The `feasibility` dict in the result contains:

```python
{
    'initial_violations': {
        'bounds': bool,      # Any asset outside bounds?
        'gross_cap': bool,   # Gross > cap?
        'net_cap': bool,     # |Net| > cap?
        'turnover_cap': bool # Turnover > cap?
    },
    'final_violations': {...},  # Same structure, after projection
    'projection_applied': bool   # Was projection needed?
}
```

## Diagnostics and Reporting

### Per-Sleeve Attribution

The `per_sleeve_contrib` DataFrame shows how each sleeve contributes to each asset:

```python
result = allocator.combine(sleeve_positions, risk_budgets)
per_sleeve = result['per_sleeve_contrib']

# Example output:
#         tsmom   xsec  carry
# ES      0.48  -0.12   0.00
# GC     -0.24   0.28   0.12
# CL      0.36  -0.08  -0.04
# SI      0.18   0.20   0.02

# Verify: sum across sleeves = total weight
assert np.allclose(per_sleeve.sum(axis=1), total_weights)
```

### Portfolio Metrics

```python
total_weights = result['total_weights']

# Leverage metrics
gross = total_weights.abs().sum()
net = total_weights.sum()
n_positions = (total_weights.abs() > 1e-6).sum()

# Turnover (if prev_weights provided)
if prev_weights is not None:
    turnover = (total_weights - prev_weights).abs().sum()
```

## Performance Characteristics

- **Memory**: O(n * s) where n = assets, s = sleeves
- **Time**: O(n * s + k * n) where k = projection iterations (typically k < 100)
- **Deterministic**: Always produces same output for same input
- **No side effects**: Pure computation, no DB writes or global state

## Testing

Comprehensive test suite in `tests/test_sleeve_allocator.py`:

- Basic aggregation and contribution tracking
- All constraint types (bounds, gross, net, turnover)
- Edge cases (single asset, no overlap, zero positions)
- Input validation
- Real-world multi-sleeve scenarios

Run tests:

```bash
pytest tests/test_sleeve_allocator.py -v
```

## Examples

See `examples/multi_sleeve_example.py` for working examples:

```bash
python examples/multi_sleeve_example.py
```

## Comparison with Single-Sleeve Allocator

| Feature | Single-Sleeve (`Allocator`) | Multi-Sleeve (`PortfolioSleeveAllocator`) |
|---------|----------------------------|------------------------------------------|
| **Input** | Signals (one strategy) | Positions (multiple sleeves) |
| **Risk Budgeting** | N/A (single strategy) | Explicit weights per sleeve |
| **Optimization** | Signal-beta / ERC / MeanVar | Linear aggregation + constraints |
| **Use Case** | Single strategy portfolio | Multi-strategy portfolio |
| **Covariance** | Used in ERC/MeanVar | Not used (pre-scaled) |

## Best Practices

1. **Risk Budgets**: Ensure risk budgets sum to exactly 1.0
2. **Pre-Scaling**: Apply all overlays (macro, vol) before passing to sleeve allocator
3. **Turnover**: Set reasonable turnover caps (typically 1.0-3.0) to avoid excessive trading
4. **Diagnostics**: Store `per_sleeve_contrib` for attribution analysis
5. **Constraints**: Set bounds/caps that align with your risk management rules

## Future Enhancements

Potential improvements for future versions:

- Risk-parity weighting across sleeves (instead of fixed budgets)
- Dynamic risk budgets based on sleeve performance
- Correlation-aware sleeve combination
- Transaction cost optimization
- Parallel constraint solving for very large portfolios

## References

- See `docs/STRATEGY.md` for overall system architecture
- See `docs/META_SLEEVES/TREND_IMPLEMENTATION.md` for Trend Meta-Sleeve implementation details
- See `docs/legacy/TSMOM_IMPLEMENTATION.md` for legacy TSMOM class (not used in production)
- See `docs/CROSS_SECTIONAL_MOMENTUM.md` for XSec sleeve details

