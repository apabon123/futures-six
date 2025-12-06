"""
Example: Multi-Sleeve Portfolio with PortfolioSleeveAllocator

Demonstrates how to combine multiple strategy sleeves (TSMOM, XSec, etc.)
into a single portfolio using risk budgets.

Workflow:
1. For each sleeve: Generate signals → Apply overlays → Get positions
2. Combine sleeves using PortfolioSleeveAllocator with risk budgets
3. Run backtest with combined weights
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from datetime import datetime

from src.agents.sleeve_allocator import (
    PortfolioSleeveAllocator,
    AllocationConstraints
)


def example_basic_sleeve_combination():
    """Basic example: Combine TSMOM and XSec sleeves."""
    
    print("=" * 60)
    print("Example 1: Basic Sleeve Combination (60/40 TSMOM/XSec)")
    print("=" * 60)
    
    # Create sleeve allocator with portfolio constraints
    constraints = AllocationConstraints(
        bounds_per_asset=(-1.5, 1.5),
        gross_cap=7.0,
        net_cap=2.0
    )
    
    allocator = PortfolioSleeveAllocator(constraints)
    
    # Simulated sleeve positions (post-overlay exposures in risk units)
    # These would come from: strategy.signals() → macro_overlay.scale() → vol_overlay.scale()
    sleeve_positions = {
        'tsmom': pd.Series({
            'ES': 0.8,   # Long equities
            'GC': -0.4,  # Short gold
            'CL': 0.6,   # Long crude
            'SI': 0.3,   # Long silver
            'NG': -0.5   # Short nat gas
        }),
        'xsec': pd.Series({
            'ES': -0.3,  # Short equities
            'GC': 0.7,   # Long gold
            'CL': -0.2,  # Short crude
            'SI': 0.5,   # Long silver
            'HG': 0.4    # Long copper
        })
    }
    
    # Risk budgets (must sum to 1.0)
    risk_budgets = {
        'tsmom': 0.6,  # 60% to TSMOM
        'xsec': 0.4    # 40% to XSec
    }
    
    # Combine sleeves
    result = allocator.combine(sleeve_positions, risk_budgets)
    
    # Extract results
    total_weights = result['total_weights']
    per_sleeve_contrib = result['per_sleeve_contrib']
    feasibility = result['feasibility']
    
    # Display results
    print("\nTotal Portfolio Weights:")
    print(total_weights.sort_values(ascending=False))
    
    print("\nPer-Sleeve Contributions:")
    print(per_sleeve_contrib)
    
    print("\nPortfolio Statistics:")
    print(f"  Gross Exposure: {total_weights.abs().sum():.3f}")
    print(f"  Net Exposure: {total_weights.sum():.3f}")
    print(f"  Number of Positions: {(total_weights.abs() > 0.001).sum()}")
    
    print("\nConstraint Violations:")
    print(f"  Initial Violations: {any(feasibility['initial_violations'].values())}")
    print(f"  Projection Applied: {feasibility['projection_applied']}")
    
    return result


def example_three_sleeve_portfolio():
    """Example with three sleeves: TSMOM, XSec, and Carry."""
    
    print("\n" + "=" * 60)
    print("Example 2: Three-Sleeve Portfolio (50/30/20)")
    print("=" * 60)
    
    constraints = AllocationConstraints(
        bounds_per_asset=(-1.5, 1.5),
        gross_cap=7.0,
        net_cap=2.0
    )
    
    allocator = PortfolioSleeveAllocator(constraints)
    
    sleeve_positions = {
        'tsmom': pd.Series({
            'ES': 0.5, 'GC': -0.3, 'CL': 0.4, 'TY': 0.6
        }),
        'xsec': pd.Series({
            'ES': -0.2, 'GC': 0.4, 'CL': -0.1, 'SI': 0.3
        }),
        'carry': pd.Series({
            'GC': 0.6, 'CL': -0.2, 'SI': 0.1, 'TY': -0.4
        })
    }
    
    risk_budgets = {
        'tsmom': 0.5,   # 50%
        'xsec': 0.3,    # 30%
        'carry': 0.2    # 20%
    }
    
    result = allocator.combine(sleeve_positions, risk_budgets)
    total_weights = result['total_weights']
    per_sleeve_contrib = result['per_sleeve_contrib']
    
    print("\nTotal Portfolio Weights:")
    print(total_weights.sort_values(ascending=False))
    
    print("\nPer-Sleeve Contributions:")
    print(per_sleeve_contrib)
    
    print("\nDecomposition for ES (Equities):")
    if 'ES' in per_sleeve_contrib.index:
        for sleeve in risk_budgets.keys():
            contrib = per_sleeve_contrib.loc['ES', sleeve]
            print(f"  {sleeve}: {contrib:+.4f}")
        print(f"  Total: {total_weights['ES']:+.4f}")
    
    return result


def example_with_turnover_constraint():
    """Example with turnover constraint (rebalancing scenario)."""
    
    print("\n" + "=" * 60)
    print("Example 3: Rebalancing with Turnover Constraint")
    print("=" * 60)
    
    constraints = AllocationConstraints(
        bounds_per_asset=(-1.5, 1.5),
        gross_cap=7.0,
        net_cap=2.0,
        turnover_cap=2.0  # Max 2.0 turnover
    )
    
    allocator = PortfolioSleeveAllocator(constraints)
    
    # Previous portfolio weights
    prev_weights = pd.Series({
        'ES': 0.5, 'GC': -0.2, 'CL': 0.3, 'SI': 0.1
    })
    
    print("\nPrevious Weights:")
    print(prev_weights)
    
    # New target positions from sleeves
    sleeve_positions = {
        'tsmom': pd.Series({
            'ES': 1.0, 'GC': 0.5, 'CL': -0.8, 'SI': 0.6
        })
    }
    
    risk_budgets = {'tsmom': 1.0}
    
    # Combine with turnover constraint
    result = allocator.combine(
        sleeve_positions,
        risk_budgets,
        prev_weights=prev_weights
    )
    
    new_weights = result['total_weights']
    
    print("\nNew Weights (with turnover limit):")
    print(new_weights)
    
    # Calculate actual turnover
    turnover = (new_weights - prev_weights).abs().sum()
    print(f"\nActual Turnover: {turnover:.3f} (cap: {constraints.turnover_cap})")
    
    print("\nTrades:")
    trades = new_weights - prev_weights
    print(trades.sort_values(key=abs, ascending=False))
    
    return result


def example_constraint_violations():
    """Example showing constraint enforcement."""
    
    print("\n" + "=" * 60)
    print("Example 4: Constraint Violation Handling")
    print("=" * 60)
    
    # Tight constraints
    constraints = AllocationConstraints(
        bounds_per_asset=(-0.5, 0.5),  # Tight bounds
        gross_cap=1.5,                  # Low gross cap
        net_cap=0.8                     # Low net cap
    )
    
    allocator = PortfolioSleeveAllocator(constraints)
    
    # Aggressive positions that violate constraints
    sleeve_positions = {
        'tsmom': pd.Series({
            'ES': 2.0, 'GC': 1.5, 'CL': 1.0, 'SI': 1.2
        })
    }
    
    risk_budgets = {'tsmom': 1.0}
    
    print("\nTarget Positions (violate constraints):")
    print(sleeve_positions['tsmom'])
    print(f"  Target Gross: {sleeve_positions['tsmom'].abs().sum():.3f}")
    print(f"  Target Net: {sleeve_positions['tsmom'].sum():.3f}")
    
    result = allocator.combine(sleeve_positions, risk_budgets)
    
    final_weights = result['total_weights']
    feasibility = result['feasibility']
    
    print("\nFinal Weights (after constraint enforcement):")
    print(final_weights.sort_values(ascending=False))
    print(f"  Final Gross: {final_weights.abs().sum():.3f} (cap: {constraints.gross_cap})")
    print(f"  Final Net: {final_weights.sum():.3f} (cap: {constraints.net_cap})")
    print(f"  Max Position: {final_weights.abs().max():.3f} (cap: {constraints.bounds_per_asset[1]})")
    
    print("\nViolations Detected:")
    for constraint, violated in feasibility['initial_violations'].items():
        if violated:
            print(f"  [X] {constraint}")
        else:
            print(f"  [OK] {constraint}")
    
    print(f"\nProjection Applied: {feasibility['projection_applied']}")
    
    return result


def example_integration_workflow():
    """
    Example showing complete workflow from signals to combined portfolio.
    
    This demonstrates how PortfolioSleeveAllocator fits into the full system.
    """
    
    print("\n" + "=" * 60)
    print("Example 5: Complete Integration Workflow")
    print("=" * 60)
    
    print("""
Workflow:
---------
1. For each sleeve (TSMOM, XSec, etc.):
   a. Generate raw signals from strategy
   b. Apply macro regime overlay (if enabled)
   c. Apply vol-managed overlay for risk targeting
   d. Result: scaled positions in risk units

2. Combine sleeves using PortfolioSleeveAllocator:
   a. Pass all sleeve positions (post-overlay)
   b. Specify risk budgets (e.g., 60% TSMOM, 40% XSec)
   c. Enforce portfolio constraints (gross/net/bounds/turnover)
   d. Result: final portfolio weights

3. Execute and compute P&L:
   a. Apply weights to market returns
   b. Calculate transaction costs
   c. Track performance metrics
    """)
    
    # Simulated workflow
    print("Step 1: Generate and scale sleeve positions...")
    
    # TSMOM sleeve
    tsmom_raw_signals = pd.Series({'ES': 1.5, 'GC': -0.8, 'CL': 1.2})
    tsmom_macro_scaler = 0.8  # Regime overlay: scale down 20%
    tsmom_vol_scale = 0.7     # Vol overlay: target volatility
    tsmom_positions = tsmom_raw_signals * tsmom_macro_scaler * tsmom_vol_scale
    
    print(f"  TSMOM: raw -> macro(*{tsmom_macro_scaler}) -> vol(*{tsmom_vol_scale})")
    print(f"    Final: {tsmom_positions.to_dict()}")
    
    # XSec sleeve
    xsec_raw_signals = pd.Series({'ES': -0.5, 'GC': 1.0, 'SI': 0.8})
    xsec_macro_scaler = 0.8
    xsec_vol_scale = 0.6
    xsec_positions = xsec_raw_signals * xsec_macro_scaler * xsec_vol_scale
    
    print(f"  XSec: raw -> macro(*{xsec_macro_scaler}) -> vol(*{xsec_vol_scale})")
    print(f"    Final: {xsec_positions.to_dict()}")
    
    print("\nStep 2: Combine sleeves with risk budgets...")
    
    sleeve_positions = {
        'tsmom': tsmom_positions,
        'xsec': xsec_positions
    }
    
    risk_budgets = {'tsmom': 0.6, 'xsec': 0.4}
    
    constraints = AllocationConstraints(
        bounds_per_asset=(-1.5, 1.5),
        gross_cap=7.0,
        net_cap=2.0
    )
    
    allocator = PortfolioSleeveAllocator(constraints)
    result = allocator.combine(sleeve_positions, risk_budgets)
    
    final_weights = result['total_weights']
    per_sleeve = result['per_sleeve_contrib']
    
    print(f"  Risk budgets: {risk_budgets}")
    print(f"  Final weights: {final_weights.to_dict()}")
    
    print("\nStep 3: Ready for execution...")
    print(f"  Portfolio gross exposure: {final_weights.abs().sum():.3f}")
    print(f"  Portfolio net exposure: {final_weights.sum():.3f}")
    print(f"  Number of positions: {(final_weights.abs() > 0.001).sum()}")
    
    print("\n  -> Pass to ExecSim for P&L calculation")
    
    return result


if __name__ == "__main__":
    # Run all examples
    example_basic_sleeve_combination()
    example_three_sleeve_portfolio()
    example_with_turnover_constraint()
    example_constraint_violations()
    example_integration_workflow()
    
    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)

