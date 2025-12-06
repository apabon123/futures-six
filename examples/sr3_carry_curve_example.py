"""
Example: SR3 Carry and Curve Features

Demonstrates how to compute and use SR3 carry/curve features for trading signals.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.data_broker import MarketData
from src.agents.feature_sr3_curve import Sr3CurveFeatures
from src.agents.strat_sr3_carry_curve import Sr3CarryCurveStrategy
import pandas as pd


def main():
    """Example usage of SR3 carry/curve features."""
    
    print("=" * 60)
    print("SR3 Carry and Curve Features Example")
    print("=" * 60)
    
    # 1. Initialize MarketData
    print("\n1. Initializing MarketData...")
    market = MarketData()
    print(f"   ✓ Connected to database")
    print(f"   ✓ Universe: {len(market.universe)} symbols")
    
    # 2. Check SR3 contracts availability
    print("\n2. Checking SR3 contracts...")
    try:
        # Query SR3 contracts
        sr3_contracts = market.get_contracts_by_root(
            root="SR3",
            ranks=list(range(12)),
            fields=("close",),
            end="2025-01-01"
        )
        
        if sr3_contracts.empty:
            print("   ✗ No SR3 contracts found in database")
            print("   → Make sure 12 SR3 contracts are loaded (ranks 0-11)")
            return
        
        print(f"   ✓ Found {len(sr3_contracts.columns)} SR3 contract ranks")
        print(f"   ✓ Date range: {sr3_contracts.index.min()} to {sr3_contracts.index.max()}")
        print(f"   ✓ Available ranks: {sorted(sr3_contracts.columns.tolist())}")
        
    except Exception as e:
        print(f"   ✗ Error querying SR3 contracts: {e}")
        return
    
    # 3. Compute features
    print("\n3. Computing SR3 features...")
    try:
        feature_calc = Sr3CurveFeatures(root="SR3", window=252)
        features = feature_calc.compute(market, end_date="2025-01-01")
        
        if features.empty:
            print("   ✗ No features computed (insufficient data or missing ranks)")
            return
        
        print(f"   ✓ Computed features for {len(features)} dates")
        print(f"   ✓ Features: {list(features.columns)}")
        
        # Show sample
        print("\n   Sample features (last 5 dates):")
        print(features.tail().to_string())
        
        # Show statistics
        print("\n   Feature statistics:")
        print(features.describe().to_string())
        
    except Exception as e:
        print(f"   ✗ Error computing features: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 4. Generate strategy signals
    print("\n4. Generating strategy signals...")
    try:
        strategy = Sr3CarryCurveStrategy(
            root="SR3",
            w_carry=0.30,
            w_curve=0.25,
            w_pack_slope=0.20,
            w_front_lvl=0.10,
            w_curv_belly=0.15,
            cap=3.0
        )
        
        # Get signals for a few dates
        test_dates = features.index[-10:]  # Last 10 dates
        
        signals_list = []
        for date in test_dates:
            sig = strategy.signals(market, date, features=features)
            signals_list.append(sig)
        
        signals_df = pd.concat(signals_list, axis=1).T
        signals_df.index = test_dates
        
        print(f"   ✓ Generated signals for {len(test_dates)} dates")
        print("\n   Sample signals:")
        print(signals_df.to_string())
        
        # Show signal statistics
        print("\n   Signal statistics:")
        print(signals_df.describe().to_string())
        
    except Exception as e:
        print(f"   ✗ Error generating signals: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 5. Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print("✓ SR3 contracts queried successfully")
    print("✓ Features computed and standardized")
    print("✓ Strategy signals generated")
    print("\nNext steps:")
    print("  - Integrate SR3 carry/curve strategy into portfolio allocator")
    print("  - Test with different weight combinations (w_carry, w_curve, w_pack_slope, w_front_lvl, w_curv_belly)")
    print("  - Combine with TSMOM and other strategy sleeves")
    
    # Cleanup
    market.close()


if __name__ == "__main__":
    main()

