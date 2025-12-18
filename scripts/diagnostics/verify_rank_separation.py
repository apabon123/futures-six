"""
Rank Separation Verification Diagnostic

Verifies that contract ranks are correctly separated for SR3 and VX contracts.
This is a critical data integrity check to ensure the rank mapping fix is working.

Tests:
1. SR3: Rank 0 ≠ Rank 1 ≠ Rank 2 over time
2. VX: VX1 ≠ VX2 ≠ VX3 over time
3. Optional: corr(returns VX1, VX2) ≠ 1.0

If all tests pass → close the book permanently on the rank mapping bug.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.agents.data_broker import MarketData
from src.market_data.vrp_loaders import load_vx_curve
import duckdb


def verify_sr3_rank_separation(
    market: MarketData,
    start: str = "2020-01-02",
    end: str = "2025-10-31",
    tolerance: float = 1e-6,
    min_pct_different: float = 95.0,
    **kwargs
) -> dict:
    """
    Verify SR3 ranks 0, 1, 2 are different over time.
    
    Args:
        market: MarketData instance
        start: Start date
        end: End date
        tolerance: Floating point tolerance for equality
        min_pct_different: Minimum percentage of overlapping days that must be different
        
    Returns:
        Dictionary with test results
    """
    print("\n" + "="*70)
    print("SR3 RANK SEPARATION TEST")
    print("="*70)
    
    # Query SR3 data directly from database to bypass assertion
    # This allows us to see the actual data separation
    from src.data.contracts.rank_mapping import parse_sr3_calendar_rank, map_contracts_to_ranks
    from src.agents.utils_db import open_readonly_connection
    import yaml
    from pathlib import Path
    
    # Get database connection
    config_path = Path("configs/data.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    db_path = config['db']['path']
    conn = open_readonly_connection(db_path)
    
    # Get table name and column mappings
    date_col = market._column_map['date']
    symbol_col = market._column_map['symbol']
    
    # Query all SR3 contracts
    query = f"""
        SELECT {date_col} as date, {symbol_col} as symbol, close
        FROM {market.table_name}
        WHERE {symbol_col} LIKE 'SR3%'
          AND {date_col} >= '{start}'
          AND {date_col} <= '{end}'
        ORDER BY {symbol_col}, {date_col}
    """
    
    try:
        df_raw = market._execute_query(query)
    except Exception as e:
        import traceback
        print(f"\n[ERROR] Failed to query SR3 data: {e}")
        print(traceback.format_exc())
        conn.close()
        return {
            "status": "FAILED",
            "error": f"Failed to query SR3 data: {e}",
            "details": {}
        }
    
    if df_raw.empty:
        conn.close()
        return {
            "status": "FAILED",
            "error": "No SR3 data returned",
            "details": {}
        }
    
    # Parse ranks
    unique_symbols = sorted(df_raw['symbol'].unique())
    try:
        symbol_to_rank = map_contracts_to_ranks(unique_symbols, root="SR3")
    except Exception as e:
        conn.close()
        return {
            "status": "FAILED",
            "error": f"Failed to parse SR3 ranks: {e}",
            "details": {}
        }
    
    # Filter to ranks 0, 1, 2
    df_raw = df_raw[df_raw['symbol'].isin([s for s, r in symbol_to_rank.items() if r in [0, 1, 2]])].copy()
    df_raw['rank'] = df_raw['symbol'].map(symbol_to_rank)
    
    # Parse dates
    df_raw['date'] = pd.to_datetime(df_raw['date'])
    
    # Pivot to wide format
    df = df_raw.pivot_table(
        index='date',
        columns='rank',
        values='close',
        aggfunc='first'
    )
    
    conn.close()
    
    if df.empty:
        return {
            "status": "FAILED",
            "error": "No SR3 data returned",
            "details": {}
        }
    
    # Check that all ranks are present
    available_ranks = [col for col in df.columns if col in [0, 1, 2]]
    if len(available_ranks) < 3:
        return {
            "status": "FAILED",
            "error": f"Missing ranks. Available: {available_ranks}, Expected: [0, 1, 2]",
            "details": {}
        }
    
    results = {
        "status": "PASSED",
        "ranks_tested": available_ranks,
        "n_days": len(df),
        "pairwise_comparisons": {}
    }
    
    # Compare each pair
    pairs = [(0, 1), (0, 2), (1, 2)]
    all_passed = True
    
    for rank_a, rank_b in pairs:
        # Find overlapping dates
        overlap_mask = df[rank_a].notna() & df[rank_b].notna()
        n_overlap = overlap_mask.sum()
        
        if n_overlap == 0:
            results["pairwise_comparisons"][f"R{rank_a}_vs_R{rank_b}"] = {
                "status": "WARNING",
                "n_overlap": 0,
                "pct_different": None,
                "message": "No overlapping dates"
            }
            all_passed = False
            continue
        
        # Count identical values (within tolerance)
        # For SR3 (interest rate futures), prices are typically stored to 4 decimal places
        # During low-rate periods (2020-2021), prices near 100 can appear identical due to rounding
        # Use a tolerance that accounts for price precision: 0.0001 (1 tick for many IR futures)
        # But also check with stricter tolerance for diagnostic
        strict_identical = overlap_mask & (df[rank_a] - df[rank_b]).abs() < 1e-6
        n_strict_identical = strict_identical.sum()
        
        # For interest rate futures, use a more reasonable tolerance
        # 0.0001 = 1 basis point in price space (0.01% in rate space)
        price_tolerance = max(tolerance, 0.0001)  # At least 0.0001 for IR futures price precision
        identical_mask = overlap_mask & (df[rank_a] - df[rank_b]).abs() < price_tolerance
        n_identical = identical_mask.sum()
        pct_different = (1.0 - (n_identical / n_overlap)) * 100.0
        
        # Check if identical dates are concentrated in low-rate period (2020-2021)
        if n_strict_identical > 0:
            identical_dates = df.index[strict_identical]
            low_rate_period = (identical_dates >= '2020-01-01') & (identical_dates < '2022-01-01')
            n_identical_low_rate = low_rate_period.sum()
            pct_identical_in_low_rate = (n_identical_low_rate / n_strict_identical * 100.0) if n_strict_identical > 0 else 0.0
        else:
            n_identical_low_rate = 0
            pct_identical_in_low_rate = 0.0
        
        passed = pct_different >= min_pct_different
        
        # Get statistics on differences
        diff_series = (df[rank_a] - df[rank_b]).abs()
        diff_overlap = diff_series[overlap_mask]
        
        results["pairwise_comparisons"][f"R{rank_a}_vs_R{rank_b}"] = {
            "status": "PASSED" if passed else "FAILED",
            "n_overlap": n_overlap,
            "n_identical": n_identical,
            "n_strict_identical": n_strict_identical,
            "n_identical_low_rate": n_identical_low_rate,
            "pct_identical_in_low_rate": round(pct_identical_in_low_rate, 1) if n_strict_identical > 0 else None,
            "pct_different": round(pct_different, 2),
            "mean_abs_diff": round(diff_overlap.mean(), 6) if n_overlap > 0 else None,
            "median_abs_diff": round(diff_overlap.median(), 6) if n_overlap > 0 else None,
            "min_abs_diff": round(diff_overlap.min(), 6) if n_overlap > 0 else None,
            "max_abs_diff": round(diff_overlap.max(), 6) if n_overlap > 0 else None
        }
        
        if not passed:
            all_passed = False
    
    results["status"] = "PASSED" if all_passed else "FAILED"
    
    # Print results
    print(f"\nSR3 Rank Separation Results:")
    print(f"  Status: {results['status']}")
    print(f"  Ranks tested: {results['ranks_tested']}")
    print(f"  Total days: {results['n_days']}")
    print(f"\n  Pairwise Comparisons:")
    for pair_name, pair_results in results["pairwise_comparisons"].items():
        status = pair_results["status"]
        pct = pair_results["pct_different"]
        n_overlap = pair_results["n_overlap"]
        mean_diff = pair_results.get("mean_abs_diff", "N/A")
        print(f"    {pair_name}: {status}")
        print(f"      Overlapping days: {n_overlap}")
        if pct is not None:
            print(f"      % Different: {pct:.2f}% (required: >= {min_pct_different}%)")
            print(f"      Mean abs difference: {mean_diff}")
            if 'n_strict_identical' in pair_results and pair_results['n_strict_identical'] > 0:
                print(f"      Strictly identical (within 1e-6): {pair_results['n_strict_identical']}/{pair_results['n_overlap']}")
                if 'pct_identical_in_low_rate' in pair_results and pair_results['pct_identical_in_low_rate'] is not None:
                    print(f"      Of identical dates, {pair_results['pct_identical_in_low_rate']:.1f}% are in 2020-2021 (low-rate period)")
    
    return results


def verify_vx_rank_separation(
    db_path: str,
    start: str = "2020-01-02",
    end: str = "2025-10-31",
    tolerance: float = 1e-6,
    min_pct_different: float = 95.0,
    check_correlation: bool = True
) -> dict:
    """
    Verify VX1, VX2, VX3 are different over time.
    
    Args:
        db_path: Path to canonical database
        start: Start date
        end: End date
        tolerance: Floating point tolerance for equality
        min_pct_different: Minimum percentage of overlapping days that must be different
        check_correlation: If True, also check return correlations
        
    Returns:
        Dictionary with test results
    """
    print("\n" + "="*70)
    print("VX RANK SEPARATION TEST")
    print("="*70)
    
    # Load VX curve data
    try:
        from src.agents.utils_db import open_readonly_connection
        con = open_readonly_connection(db_path)
        df = load_vx_curve(con, start=start, end=end)
        con.close()
    except Exception as e:
        import traceback
        print(f"\n[ERROR] Failed to load VX data: {e}")
        print(traceback.format_exc())
        return {
            "status": "FAILED",
            "error": f"Failed to load VX data: {e}",
            "details": {}
        }
    
    if df.empty:
        return {
            "status": "FAILED",
            "error": "No VX data returned",
            "details": {}
        }
    
    # Set date as index if it's a column
    if 'date' in df.columns:
        df = df.set_index('date')
    
    # Check that all VX contracts are present
    required_cols = ['vx1', 'vx2', 'vx3']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        return {
            "status": "FAILED",
            "error": f"Missing columns: {missing_cols}",
            "details": {}
        }
    
    results = {
        "status": "PASSED",
        "n_days": len(df),
        "pairwise_comparisons": {},
        "correlations": {}
    }
    
    # Compare each pair
    pairs = [('vx1', 'vx2'), ('vx1', 'vx3'), ('vx2', 'vx3')]
    all_passed = True
    
    for vx_a, vx_b in pairs:
        # Find overlapping dates
        overlap_mask = df[vx_a].notna() & df[vx_b].notna()
        n_overlap = overlap_mask.sum()
        
        if n_overlap == 0:
            results["pairwise_comparisons"][f"{vx_a}_vs_{vx_b}"] = {
                "status": "WARNING",
                "n_overlap": 0,
                "pct_different": None,
                "message": "No overlapping dates"
            }
            all_passed = False
            continue
        
        # Count identical values (within tolerance)
        identical_mask = overlap_mask & (df[vx_a] - df[vx_b]).abs() < tolerance
        n_identical = identical_mask.sum()
        pct_different = (1.0 - (n_identical / n_overlap)) * 100.0
        
        passed = pct_different >= min_pct_different
        
        results["pairwise_comparisons"][f"{vx_a}_vs_{vx_b}"] = {
            "status": "PASSED" if passed else "FAILED",
            "n_overlap": n_overlap,
            "n_identical": n_identical,
            "pct_different": round(pct_different, 2),
            "mean_abs_diff": round((df[vx_a] - df[vx_b]).abs().mean(), 6) if n_overlap > 0 else None
        }
        
        if not passed:
            all_passed = False
        
        # Compute return correlations if requested
        if check_correlation:
            returns_a = df[vx_a].pct_change().dropna()
            returns_b = df[vx_b].pct_change().dropna()
            
            # Align returns
            common_dates = returns_a.index.intersection(returns_b.index)
            if len(common_dates) > 1:
                returns_a_aligned = returns_a.loc[common_dates]
                returns_b_aligned = returns_b.loc[common_dates]
                
                # Check for zero variance
                if returns_a_aligned.std() > 1e-10 and returns_b_aligned.std() > 1e-10:
                    corr = returns_a_aligned.corr(returns_b_aligned)
                    results["correlations"][f"{vx_a}_vs_{vx_b}"] = {
                        "correlation": round(corr, 4),
                        "n_observations": len(common_dates),
                        "status": "PASSED" if abs(corr) < 0.99 else "WARNING"
                    }
                else:
                    results["correlations"][f"{vx_a}_vs_{vx_b}"] = {
                        "correlation": None,
                        "n_observations": len(common_dates),
                        "status": "WARNING",
                        "message": "Zero variance in returns"
                    }
            else:
                results["correlations"][f"{vx_a}_vs_{vx_b}"] = {
                    "correlation": None,
                    "n_observations": len(common_dates),
                    "status": "WARNING",
                    "message": "Insufficient overlapping returns"
                }
    
    results["status"] = "PASSED" if all_passed else "FAILED"
    
    # Print results
    print(f"\nVX Rank Separation Results:")
    print(f"  Status: {results['status']}")
    print(f"  Total days: {results['n_days']}")
    print(f"\n  Pairwise Comparisons:")
    for pair_name, pair_results in results["pairwise_comparisons"].items():
        status = pair_results["status"]
        pct = pair_results["pct_different"]
        n_overlap = pair_results["n_overlap"]
        mean_diff = pair_results.get("mean_abs_diff", "N/A")
        print(f"    {pair_name}: {status}")
        print(f"      Overlapping days: {n_overlap}")
        if pct is not None:
            print(f"      % Different: {pct:.2f}% (required: >= {min_pct_different}%)")
            print(f"      Mean abs difference: {mean_diff}")
    
    if check_correlation and results["correlations"]:
        print(f"\n  Return Correlations:")
        for pair_name, corr_results in results["correlations"].items():
            corr = corr_results.get("correlation")
            status = corr_results["status"]
            n_obs = corr_results["n_observations"]
            print(f"    {pair_name}: {status}")
            if corr is not None:
                print(f"      Correlation: {corr:.4f} (required: < 0.99)")
            print(f"      Observations: {n_obs}")
    
    return results


def main():
    """Run rank separation verification for SR3 and VX."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Verify rank separation for SR3 and VX contracts")
    parser.add_argument("--start", type=str, default="2020-01-02", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default="2025-10-31", help="End date (YYYY-MM-DD)")
    parser.add_argument("--db-path", type=str, default=None, help="Path to canonical database (default: from config)")
    parser.add_argument("--config", type=str, default="configs/data.yaml", help="Path to data config")
    parser.add_argument("--no-vx-corr", action="store_true", help="Skip VX return correlation check")
    parser.add_argument("--bypass-assertion", action="store_true", help="Bypass runtime assertion to see full diagnostic results")
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("RANK SEPARATION VERIFICATION DIAGNOSTIC")
    print("="*70)
    print(f"Start date: {args.start}")
    print(f"End date: {args.end}")
    print(f"Database: {args.db_path}")
    
    # Initialize MarketData
    try:
        market = MarketData(config_path=args.config)
    except Exception as e:
        print(f"\n[ERROR] Failed to initialize MarketData: {e}")
        return 1
    
    # Get database path for VX (use config if not provided)
    if args.db_path is None:
        import yaml
        from pathlib import Path
        config_path = Path(args.config)
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            args.db_path = config['db']['path']
        else:
            print(f"\n[ERROR] Config file not found: {args.config}")
            return 1
    
    # Test SR3
    sr3_results = verify_sr3_rank_separation(
        market,
        start=args.start,
        end=args.end,
        bypass_assertion=args.bypass_assertion
    )
    
    # Test VX
    vx_results = verify_vx_rank_separation(
        args.db_path,
        start=args.start,
        end=args.end,
        check_correlation=not args.no_vx_corr
    )
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"SR3 Rank Separation: {sr3_results['status']}")
    print(f"VX Rank Separation: {vx_results['status']}")
    
    all_passed = (
        sr3_results.get("status") == "PASSED" and
        vx_results.get("status") == "PASSED"
    )
    
    if all_passed:
        print("\n[SUCCESS] All rank separation tests PASSED.")
        print("The rank mapping bug is permanently closed.")
    else:
        print("\n[FAILURE] One or more rank separation tests FAILED.")
        print("Data integrity issues detected. Review results above.")
        return 1
    
    # Close MarketData
    market.close()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

