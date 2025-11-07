"""
Demo script for Cross-Sectional Momentum Strategy

Demonstrates integration with MarketData and typical usage patterns.
Run this after setting up the database with market data.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agents.data_broker import MarketData
from agents.strat_cross_sectional import CrossSectionalMomentum
import pandas as pd


def demo_basic_usage():
    """Demonstrate basic usage of CrossSectionalMomentum."""
    print("=" * 60)
    print("Cross-Sectional Momentum Strategy Demo")
    print("=" * 60)
    
    # Initialize MarketData
    try:
        market = MarketData(config_path="configs/data.yaml")
        print(f"\n✓ Connected to database: {market.db_path}")
        print(f"✓ Universe: {market.universe}")
    except Exception as e:
        print(f"\n✗ Could not connect to database: {e}")
        print("  This demo requires a populated database.")
        print("  Run data ingestion first, then try again.")
        return
    
    # Initialize strategy
    symbols = list(market.universe)
    strategy = CrossSectionalMomentum(
        symbols=symbols,
        lookback=126,       # 6 months
        skip_recent=21,     # Skip last month
        top_frac=0.33,      # Long top 33%
        bottom_frac=0.33,   # Short bottom 33%
        standardize="vol",  # Volatility-scaled signals
        signal_cap=3.0,     # Cap at ±3
        rebalance="W-FRI"   # Rebalance weekly on Friday
    )
    
    print(f"\n✓ Strategy initialized:")
    desc = strategy.describe()
    for key, value in desc.items():
        print(f"  {key}: {value}")
    
    # Get available dates
    dates = market.trading_days()
    if len(dates) < 200:
        print(f"\n✗ Insufficient data: only {len(dates)} trading days")
        return
    
    print(f"\n✓ Data available: {len(dates)} trading days")
    print(f"  Range: {dates[0].date()} to {dates[-1].date()}")
    
    # Generate signals for a recent date
    test_date = dates[-50]  # 50 days ago
    print(f"\n→ Generating signals for {test_date.date()}...")
    
    signals = strategy.signals(market, test_date)
    
    print("\n✓ Signals generated:")
    print(f"  Sum: {signals.sum():.6f} (should be near 0)")
    print(f"  Mean: {signals.mean():.4f}")
    print(f"  Std: {signals.std():.4f}")
    print(f"  Min: {signals.min():.4f}")
    print(f"  Max: {signals.max():.4f}")
    
    print("\n  Individual signals:")
    for symbol in signals.index:
        signal = signals[symbol]
        direction = "LONG" if signal > 0.1 else ("SHORT" if signal < -0.1 else "NEUTRAL")
        print(f"    {symbol:6s}: {signal:+7.4f}  ({direction})")
    
    # Test rebalance behavior
    print(f"\n→ Testing rebalance behavior...")
    next_day = dates[dates.get_loc(test_date) + 1]
    signals_next = strategy.signals(market, next_day)
    
    if (signals == signals_next).all():
        print("  ✓ Signals held constant on non-rebalance date")
    else:
        print("  ✗ Signals changed on non-rebalance date (unexpected)")
    
    # Generate signals over time
    print(f"\n→ Generating signals for last 10 Fridays...")
    fridays = [d for d in dates[-100:] if d.dayofweek == 4][-10:]
    
    strategy.reset_state()  # Reset for clean run
    
    results = []
    for friday in fridays:
        sigs = strategy.signals(market, friday)
        results.append({
            'date': friday.date(),
            'sum': sigs.sum(),
            'long_count': (sigs > 0.1).sum(),
            'short_count': (sigs < -0.1).sum(),
            'neutral_count': ((sigs >= -0.1) & (sigs <= 0.1)).sum()
        })
    
    print("\n  Date         Sum      Long  Short  Neutral")
    print("  " + "-" * 50)
    for r in results:
        print(f"  {r['date']}  {r['sum']:+7.4f}    {r['long_count']}     {r['short_count']}      {r['neutral_count']}")
    
    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)
    
    market.close()


def demo_comparison_with_tsmom():
    """Compare cross-sectional vs time-series momentum."""
    print("\n" + "=" * 60)
    print("Comparing Cross-Sectional vs Time-Series Momentum")
    print("=" * 60)
    
    try:
        from agents.strat_momentum import TSMOM
        market = MarketData(config_path="configs/data.yaml")
    except Exception as e:
        print(f"\n✗ Could not initialize: {e}")
        return
    
    symbols = list(market.universe)
    dates = market.trading_days()
    test_date = dates[-50]
    
    # Initialize both strategies
    cs_mom = CrossSectionalMomentum(
        symbols=symbols,
        lookback=126,
        skip_recent=21,
        standardize="vol",
        rebalance="W-FRI"
    )
    
    ts_mom = TSMOM(
        lookbacks=[126],
        skip_recent=21,
        standardize="vol",
        rebalance="W-FRI"
    )
    
    # Generate signals
    cs_signals = cs_mom.signals(market, test_date)
    ts_signals = ts_mom.signals(market, test_date)
    
    print(f"\nSignals on {test_date.date()}:")
    print("\n  Symbol   Cross-Sect  Time-Series  Difference")
    print("  " + "-" * 50)
    
    for symbol in symbols:
        cs_sig = cs_signals[symbol]
        ts_sig = ts_signals[symbol]
        diff = cs_sig - ts_sig
        print(f"  {symbol:6s}  {cs_sig:+10.4f}  {ts_sig:+11.4f}  {diff:+10.4f}")
    
    print(f"\n  Sum:     {cs_signals.sum():+10.4f}  {ts_signals.sum():+11.4f}")
    
    print("\nKey differences:")
    print("  • Cross-sectional: Market-neutral (sum ≈ 0), relative ranking")
    print("  • Time-series: Can be net long/short, absolute momentum")
    
    market.close()


if __name__ == "__main__":
    demo_basic_usage()
    demo_comparison_with_tsmom()

