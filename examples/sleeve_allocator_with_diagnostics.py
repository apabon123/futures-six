"""
Integration Example: PortfolioSleeveAllocator with Diagnostics

Shows how to use the sleeve allocator in a backtest and generate
comprehensive diagnostics including per-sleeve attribution.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.agents.sleeve_allocator import (
    PortfolioSleeveAllocator,
    AllocationConstraints
)
from src.agents.diagnostics import make_report


def simulate_multi_sleeve_backtest():
    """
    Simulates a multi-sleeve backtest with the sleeve allocator.
    
    Returns results dict compatible with diagnostics.make_report().
    """
    print("=" * 70)
    print("Multi-Sleeve Backtest with Diagnostics")
    print("=" * 70)
    
    # Setup
    constraints = AllocationConstraints(
        bounds_per_asset=(-1.5, 1.5),
        gross_cap=7.0,
        net_cap=2.0,
        turnover_cap=2.0
    )
    
    allocator = PortfolioSleeveAllocator(constraints)
    risk_budgets = {'tsmom': 0.6, 'xsec': 0.4}
    
    # Simulate 100 days of trading
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    
    # Track results
    equity_values = [1.0]  # Start at 1.0
    weights_history = []
    per_sleeve_history = {'tsmom': [], 'xsec': []}
    turnover_history = []
    
    prev_weights = None
    
    for i, date in enumerate(dates):
        # Simulate sleeve positions (would come from strategies in real backtest)
        sleeve_positions = {
            'tsmom': pd.Series({
                'ES': 0.8 * np.sin(i / 10),
                'GC': -0.4 * np.cos(i / 15),
                'CL': 0.6 * np.sin(i / 12)
            }),
            'xsec': pd.Series({
                'ES': -0.3 * np.cos(i / 8),
                'GC': 0.7 * np.sin(i / 10),
                'SI': 0.5 * np.cos(i / 14)
            })
        }
        
        # Combine sleeves
        result = allocator.combine(
            sleeve_positions,
            risk_budgets,
            prev_weights=prev_weights
        )
        
        weights = result['total_weights']
        per_sleeve = result['per_sleeve_contrib']
        
        # Calculate turnover
        if prev_weights is not None:
            prev_aligned = prev_weights.reindex(weights.index, fill_value=0)
            turnover = (weights - prev_aligned).abs().sum()
        else:
            turnover = weights.abs().sum()
        
        turnover_history.append(turnover)
        
        # Simulate returns (would come from market data in real backtest)
        returns = pd.Series({
            'ES': np.random.normal(0.0005, 0.01),
            'GC': np.random.normal(0.0003, 0.008),
            'CL': np.random.normal(0.0004, 0.012),
            'SI': np.random.normal(0.0002, 0.009)
        })
        
        # Portfolio return
        common = weights.index.intersection(returns.index)
        port_return = (weights.loc[common] * returns.loc[common]).sum()
        
        # Update equity
        equity_values.append(equity_values[-1] * (1 + port_return))
        
        # Store weights with date
        weights_history.append(weights)
        
        # Store per-sleeve contributions
        for sleeve_name in risk_budgets.keys():
            per_sleeve_history[sleeve_name].append(
                per_sleeve[sleeve_name] if sleeve_name in per_sleeve.columns else pd.Series()
            )
        
        prev_weights = weights.copy()
    
    # Build results dict for diagnostics
    equity_curve = pd.Series(equity_values[1:], index=dates)
    weights_df = pd.DataFrame(weights_history, index=dates)
    turnover_series = pd.Series(turnover_history, index=dates)
    
    # Per-sleeve weights
    sleeve_weights = {}
    for sleeve_name, history in per_sleeve_history.items():
        sleeve_weights[sleeve_name] = pd.DataFrame(history, index=dates)
    
    results = {
        'equity_curve': equity_curve,
        'weights': sleeve_weights,  # Per-sleeve weights for attribution
        'weights_panel': weights_df,  # Total weights (legacy key)
        'turnover': turnover_series,
        'risk_budgets': risk_budgets,  # Meta info
    }
    
    return results


def main():
    """Run backtest and generate diagnostics."""
    
    # Run backtest
    print("\nRunning multi-sleeve backtest...")
    results = simulate_multi_sleeve_backtest()
    
    # Summary statistics
    equity = results['equity_curve']
    total_return = (equity.iloc[-1] / equity.iloc[0]) - 1
    vol = equity.pct_change().std() * np.sqrt(252)
    sharpe = (equity.pct_change().mean() * 252) / vol if vol > 0 else 0
    
    print(f"\nBacktest Summary:")
    print(f"  Start: {equity.index[0].date()}")
    print(f"  End: {equity.index[-1].date()}")
    print(f"  Total Return: {total_return:.2%}")
    print(f"  Annualized Vol: {vol:.2%}")
    print(f"  Sharpe Ratio: {sharpe:.2f}")
    
    print(f"\nPortfolio Statistics:")
    weights_panel = results['weights_panel']
    print(f"  Avg Gross: {weights_panel.abs().sum(axis=1).mean():.3f}")
    print(f"  Avg Net: {weights_panel.sum(axis=1).abs().mean():.3f}")
    print(f"  Avg Turnover: {results['turnover'].mean():.3f}")
    
    print(f"\nPer-Sleeve Statistics:")
    for sleeve_name, sleeve_weights_df in results['weights'].items():
        avg_gross = sleeve_weights_df.abs().sum(axis=1).mean()
        avg_net = sleeve_weights_df.sum(axis=1).abs().mean()
        print(f"  {sleeve_name}:")
        print(f"    Avg Gross: {avg_gross:.3f}")
        print(f"    Avg Net: {avg_net:.3f}")
    
    # Generate diagnostics
    print("\n" + "=" * 70)
    print("Generating Diagnostics Report...")
    print("=" * 70)
    
    try:
        report = make_report(results, outdir="reports/sleeve_allocator_demo")
        
        print("\nDiagnostics Generated:")
        print(f"  Files saved to: reports/sleeve_allocator_demo/")
        
        metrics = report['metrics']
        print(f"\nPerformance Metrics:")
        print(f"  CAGR: {metrics.get('cagr', 0):.2%}")
        print(f"  Sharpe: {metrics.get('sharpe', 0):.2f}")
        print(f"  Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")
        print(f"  Calmar Ratio: {metrics.get('calmar', 0):.2f}")
        
        if 'files' in report:
            print(f"\nFiles Created:")
            for file_type, file_path in report['files'].items():
                print(f"  - {file_type}: {file_path}")
    
    except Exception as e:
        print(f"\nNote: Diagnostics generation skipped (may need data dependencies)")
        print(f"  Error: {e}")
    
    print("\n" + "=" * 70)
    print("Integration Example Complete!")
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    results = main()

