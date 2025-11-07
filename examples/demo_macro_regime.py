"""
Demo: MacroRegimeFilter Integration

Shows how the MacroRegimeFilter can be integrated into the backtest pipeline
to apply regime-based scaling to strategy signals before volatility targeting.

The typical flow:
1. Strategy generates raw signals
2. MacroRegimeFilter applies regime scaler k ∈ [0.4, 1.0]
3. VolManagedOverlay scales to target volatility
4. Allocator converts to final weights
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
from datetime import datetime

from src.agents.data_broker import MarketData
from src.agents.strat_momentum import TSMOM
from src.agents.overlay_macro_regime import MacroRegimeFilter


def main():
    """Demonstrate MacroRegimeFilter usage."""
    
    print("=" * 80)
    print("MacroRegimeFilter Demo")
    print("=" * 80)
    print()
    
    # Initialize market data
    print("[1/3] Initializing MarketData...")
    with MarketData() as market:
        
        # Initialize strategy
        print("[2/3] Initializing TSMOM strategy...")
        strategy = TSMOM(
            lookbacks=[252, 126, 63],  # 12m, 6m, 3m
            skip_recent=21
        )
        
        # Initialize macro regime filter
        print("[3/3] Initializing MacroRegimeFilter...")
        macro_filter = MacroRegimeFilter(
            rebalance="W-FRI",
            vol_thresholds={'low': 0.15, 'high': 0.30},
            k_bounds={'min': 0.4, 'max': 1.0},
            smoothing=0.2
        )
        
        print()
        print("-" * 80)
        print("Filter Configuration:")
        print("-" * 80)
        desc = macro_filter.describe()
        for key, value in desc.items():
            if key not in ['outputs']:
                print(f"  {key:20s}: {value}")
        print()
        
        # Get some example dates
        trading_days = market.trading_days()
        test_dates = list(trading_days[-20:])  # Last 20 trading days
        
        print("-" * 80)
        print("Regime Analysis (Last 20 Trading Days)")
        print("-" * 80)
        print(f"{'Date':<12} {'Scaler k':>10} {'Regime':<15}")
        print("-" * 80)
        
        for date in test_dates:
            # Generate strategy signals
            signals = strategy.signals(market, date)
            
            if signals.empty or signals.abs().sum() < 0.01:
                continue
            
            # Get regime scaler
            k = macro_filter.scaler(market, date)
            
            # Classify regime
            if k >= 0.9:
                regime = "FAVORABLE"
            elif k >= 0.7:
                regime = "NEUTRAL"
            elif k >= 0.5:
                regime = "CAUTIOUS"
            else:
                regime = "DEFENSIVE"
            
            print(f"{str(date)[:10]:<12} {k:>10.3f} {regime:<15}")
        
        print("-" * 80)
        
        # Show detailed example for last date
        print()
        print("=" * 80)
        print("Detailed Example (Last Date)")
        print("=" * 80)
        
        last_date = test_dates[-1]
        signals = strategy.signals(market, last_date)
        
        # Apply regime filter
        k = macro_filter.scaler(market, last_date)
        scaled_signals = macro_filter.apply(signals, market, last_date)
        
        print(f"\nDate: {last_date}")
        print(f"Regime Scaler: {k:.3f}")
        print()
        
        # Show signal comparison
        print(f"{'Symbol':<10} {'Raw Signal':>12} {'Regime-Scaled':>15} {'Change':>10}")
        print("-" * 50)
        for symbol in signals.index:
            raw = signals[symbol]
            scaled = scaled_signals[symbol]
            change = scaled - raw
            print(f"{symbol:<10} {raw:>12.3f} {scaled:>15.3f} {change:>10.3f}")
        
        print()
        print(f"Raw Gross Leverage:    {signals.abs().sum():.3f}")
        print(f"Scaled Gross Leverage: {scaled_signals.abs().sum():.3f}")
        print(f"Reduction:             {(1 - scaled_signals.abs().sum() / signals.abs().sum()) * 100:.1f}%")
        print()
        
        print("=" * 80)
        print("Integration Notes")
        print("=" * 80)
        print()
        print("In a full backtest pipeline, the MacroRegimeFilter would be applied:")
        print("  1. BEFORE VolManagedOverlay (to reduce exposure in bad regimes)")
        print("  2. AFTER strategy signal generation")
        print()
        print("Example flow:")
        print("  raw_signals = strategy.signals(market, date)")
        print("  regime_signals = macro_filter.apply(raw_signals, market, date)")
        print("  vol_signals = vol_overlay.scale(regime_signals, market, date)")
        print("  weights = allocator.allocate(vol_signals, market, date)")
        print()
        print("This ensures that:")
        print("  - High volatility environments reduce overall exposure")
        print("  - Poor breadth (downtrends) trigger defensive positioning")
        print("  - Changes happen gradually (EMA smoothing prevents whipsaws)")
        print("  - No look-ahead bias (all data point-in-time)")
        print()
        print("=" * 80)


if __name__ == "__main__":
    main()

