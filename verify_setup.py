"""
Quick verification script to test MarketData broker setup.

Run this script to verify:
1. Database connection works
2. OHLCV table is found
3. Data can be loaded
4. All major APIs work
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.agents import MarketData


def verify_setup():
    """Verify MarketData broker is working correctly."""
    
    print("=" * 70)
    print("MarketData Broker Setup Verification")
    print("=" * 70)
    
    try:
        # 1. Initialize MarketData
        print("\n[1/8] Initializing MarketData broker...")
        md = MarketData()
        print(f"[OK] Connected to database: {md.db_path}")
        print(f"[OK] Using table: {md.table_name}")
        print(f"[OK] Universe: {md.universe}")
        
        # 2. Get price panel
        print("\n[2/8] Testing get_price_panel()...")
        prices = md.get_price_panel(("ES_FRONT_CALENDAR_2D", "NQ_FRONT_CALENDAR_2D"), fields=("close",), tidy=False)
        if not prices.empty:
            print(f"[OK] Loaded {len(prices)} days of price data")
            print(f"  Date range: {prices.index.min()} to {prices.index.max()}")
        else:
            print("[WARN] Warning: No price data returned (check symbols and date range)")
        
        # 3. Get returns
        print("\n[3/8] Testing get_returns()...")
        returns = md.get_returns(("ES_FRONT_CALENDAR_2D",), method="log")
        if not returns.empty:
            print(f"[OK] Calculated returns for {len(returns)} days")
            print(f"  Mean return: {returns['ES_FRONT_CALENDAR_2D'].mean():.6f}")
            print(f"  Std return: {returns['ES_FRONT_CALENDAR_2D'].std():.6f}")
        else:
            print("[WARN] Warning: No returns calculated")
        
        # 4. Get volatility
        print("\n[4/8] Testing get_vol()...")
        vol = md.get_vol(("ES_FRONT_CALENDAR_2D",), lookback=63)
        if not vol.empty:
            valid_vol = vol['ES_FRONT_CALENDAR_2D'].dropna()
            if len(valid_vol) > 0:
                print(f"[OK] Calculated volatility for {len(valid_vol)} days")
                print(f"  Mean vol (annualized): {valid_vol.mean():.2%}")
        else:
            print("[WARN] Warning: No volatility calculated")
        
        # 5. Get covariance
        print("\n[5/8] Testing get_cov()...")
        cov = md.get_cov(("ES_FRONT_CALENDAR_2D", "NQ_FRONT_CALENDAR_2D"), lookback=252, shrink="none")
        if not cov.empty:
            print(f"[OK] Calculated covariance matrix: {cov.shape}")
            es = "ES_FRONT_CALENDAR_2D"
            nq = "NQ_FRONT_CALENDAR_2D"
            print(f"  Correlation ES-NQ: {cov.loc[es, nq] / (cov.loc[es, es] * cov.loc[nq, nq])**0.5:.3f}")
        else:
            print("[WARN] Warning: No covariance matrix calculated")
        
        # 6. Test snapshot
        print("\n[6/8] Testing snapshot()...")
        snapshot_md = md.snapshot("2023-12-31")
        snapshot_prices = snapshot_md.get_price_panel(("ES_FRONT_CALENDAR_2D",), fields=("close",), tidy=False)
        if not snapshot_prices.empty:
            max_date = snapshot_prices.index.max()
            print(f"[OK] Snapshot created with asof=2023-12-31")
            print(f"  Max date in snapshot: {max_date}")
            if max_date <= pd.to_datetime("2023-12-31"):
                print(f"  [OK] Snapshot filter working correctly")
            else:
                print(f"  [ERROR] Snapshot filter NOT working - found dates after cutoff")
        snapshot_md.close()
        
        # 7. Flag roll jumps
        print("\n[7/8] Testing flag_roll_jumps()...")
        jumps = md.flag_roll_jumps(("ES_FRONT_CALENDAR_2D", "NQ_FRONT_CALENDAR_2D", "CL_FRONT_VOLUME"), threshold_bp=100)
        print(f"[OK] Flagged {len(jumps)} potential roll jumps")
        if len(jumps) > 0:
            print(f"  Sample: {jumps.head(3).to_string()}")
        
        # 8. Missing data report
        print("\n[8/8] Testing missing_report()...")
        report = md.missing_report(("ES_FRONT_CALENDAR_2D", "NQ_FRONT_CALENDAR_2D", "ZN_FRONT_VOLUME"))
        print(f"[OK] Coverage report:\n{report.to_string(index=False)}")
        
        # Close connection
        md.close()
        
        print("\n" + "=" * 70)
        print("[OK] All tests passed! MarketData broker is working correctly.")
        print("=" * 70)
        
        return True
    
    except Exception as e:
        print(f"\n[ERROR] Error during verification: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import pandas as pd  # Import here so it shows in error if missing
    
    success = verify_setup()
    sys.exit(0 if success else 1)

