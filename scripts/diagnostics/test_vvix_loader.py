#!/usr/bin/env python3
"""
Smoke test for VVIX loader.

Quick verification that VVIX data is available in the canonical database.
"""

import sys
from pathlib import Path
import yaml
import duckdb
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.agents.utils_db import open_readonly_connection
from src.market_data.vrp_loaders import load_vvix

def main():
    # Load config
    config_path = Path("configs/data.yaml")
    if not config_path.exists():
        print("ERROR: configs/data.yaml not found")
        sys.exit(1)
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    db_path = config['db']['path']
    print(f"Database: {db_path}")
    
    # Connect to database
    con = open_readonly_connection(db_path)
    
    try:
        # Test date range
        start = "2020-01-01"
        end = "2025-10-31"
        
        print("\n" + "=" * 80)
        print("VVIX LOADER SMOKE TEST")
        print("=" * 80)
        print(f"Date range: {start} to {end}")
        print()
        
        # Load VVIX data
        print("Loading VVIX data...")
        df = load_vvix(con, start, end)
        
        print(f"\nRow count: {len(df)}")
        
        if len(df) == 0:
            print("\n" + "=" * 80)
            print("WARNING: No VVIX data found!")
            print("=" * 80)
            print("\nChecking available sources...")
            
            # Check CBOE table
            cboe_check = con.execute(
                """
                SELECT COUNT(*) as count
                FROM market_data_cboe
                WHERE symbol = 'VVIX'
                """
            ).df()
            print(f"  CBOE (market_data_cboe, symbol='VVIX'): {cboe_check['count'].iloc[0]} rows")
            
            # Check FRED table
            fred_check = con.execute(
                """
                SELECT COUNT(*) as count
                FROM f_fred_observations
                WHERE series_id = 'VVIXCLS'
                """
            ).df()
            print(f"  FRED (f_fred_observations, series_id='VVIXCLS'): {fred_check['count'].iloc[0]} rows")
            
            # List available CBOE symbols
            cboe_symbols = con.execute(
                """
                SELECT DISTINCT symbol
                FROM market_data_cboe
                WHERE symbol LIKE '%VIX%'
                ORDER BY symbol
                """
            ).df()
            if len(cboe_symbols) > 0:
                print(f"\n  Available VIX-related symbols in CBOE table:")
                for symbol in cboe_symbols['symbol']:
                    print(f"    - {symbol}")
            
            # List available FRED series
            fred_series = con.execute(
                """
                SELECT DISTINCT series_id
                FROM f_fred_observations
                WHERE series_id LIKE '%VVIX%'
                ORDER BY series_id
                """
            ).df()
            if len(fred_series) > 0:
                print(f"\n  Available VVIX-related series in FRED table:")
                for series_id in fred_series['series_id']:
                    print(f"    - {series_id}")
            
            print("\n" + "=" * 80)
            print("ACTION REQUIRED: VVIX data must be added to the database")
            print("=" * 80)
            sys.exit(1)
        
        # Show date range
        print(f"Date range: {df['date'].min()} to {df['date'].max()}")
        
        # Show basic stats
        print(f"\nVVIX Statistics:")
        print(f"  Mean:   {df['vvix'].mean():.2f}")
        print(f"  Median: {df['vvix'].median():.2f}")
        print(f"  Std:    {df['vvix'].std():.2f}")
        print(f"  Min:    {df['vvix'].min():.2f}")
        print(f"  Max:    {df['vvix'].max():.2f}")
        
        # Show last 5 values
        print(f"\nLast 5 values:")
        print(df.tail(5).to_string(index=False))
        
        print("\n" + "=" * 80)
        print("[PASS] VVIX loader smoke test PASSED")
        print("=" * 80)
        
    finally:
        con.close()

if __name__ == "__main__":
    main()

