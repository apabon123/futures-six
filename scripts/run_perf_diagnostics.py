"""
CLI script for running performance diagnostics on backtest runs.

Usage:
    python scripts/run_perf_diagnostics.py --run_id <run_id> [--baseline_id <baseline_id>] [--base_dir <base_dir>]
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.diagnostics_perf import (
    load_run,
    compute_core_metrics,
    compute_yearly_stats,
    compute_per_asset_stats,
    compare_to_baseline
)


def main():
    parser = argparse.ArgumentParser(
        description="Run performance diagnostics on a backtest run"
    )
    parser.add_argument(
        "--run_id",
        required=True,
        help="ID of the run to analyze"
    )
    parser.add_argument(
        "--baseline_id",
        required=False,
        help="Optional baseline run_id to compare against"
    )
    parser.add_argument(
        "--base_dir",
        default="reports/runs",
        help="Base directory containing run folders (default: reports/runs)"
    )
    
    args = parser.parse_args()
    
    try:
        # Load run
        print(f"Loading run: {args.run_id}")
        run = load_run(args.run_id, base_dir=args.base_dir)
        
        # Compute metrics
        print("\n" + "=" * 80)
        print("CORE METRICS")
        print("=" * 80)
        metrics = compute_core_metrics(run)
        for k, v in metrics.items():
            if isinstance(v, float) and not (v != v):  # Check for NaN
                if k in ["CAGR", "Vol", "MaxDD"]:
                    print(f"  {k:15}: {v:10.4f} ({v*100:.2f}%)")
                elif k == "Sharpe":
                    print(f"  {k:15}: {v:10.4f}")
                elif k == "HitRate":
                    print(f"  {k:15}: {v:10.4f} ({v*100:.2f}%)")
                else:
                    print(f"  {k:15}: {v:10.4f}")
            else:
                print(f"  {k:15}: NaN")
        
        # Yearly stats
        print("\n" + "=" * 80)
        print("YEAR-BY-YEAR STATS")
        print("=" * 80)
        yearly = compute_yearly_stats(run)
        if not yearly.empty:
            print(yearly.to_string())
        else:
            print("  No yearly data available")
        
        # Per-asset stats
        print("\n" + "=" * 80)
        print("PER-ASSET STATS (all assets, sorted by AnnRet)")
        print("=" * 80)
        per_asset = compute_per_asset_stats(run)
        if not per_asset.empty:
            all_assets = per_asset.sort_values("AnnRet", ascending=False)
            print(all_assets.to_string())
            print(f"\nTotal assets: {len(all_assets)}")
        else:
            print("  No per-asset data available")
        
        # Baseline comparison
        if args.baseline_id:
            print("\n" + "=" * 80)
            print("BASELINE COMPARISON")
            print("=" * 80)
            print(f"Current:  {args.run_id}")
            print(f"Baseline: {args.baseline_id}")
            
            baseline = load_run(args.baseline_id, base_dir=args.base_dir)
            comp = compare_to_baseline(run, baseline)
            
            print("\nCurrent Metrics:")
            for k, v in comp["metrics_current"].items():
                if isinstance(v, float) and not (v != v):
                    if k in ["CAGR", "Vol", "MaxDD"]:
                        print(f"  {k:15}: {v:10.4f} ({v*100:.2f}%)")
                    elif k == "Sharpe":
                        print(f"  {k:15}: {v:10.4f}")
                    elif k == "HitRate":
                        print(f"  {k:15}: {v:10.4f} ({v*100:.2f}%)")
                    else:
                        print(f"  {k:15}: {v:10.4f}")
                else:
                    print(f"  {k:15}: NaN")
            
            print("\nBaseline Metrics:")
            for k, v in comp["metrics_baseline"].items():
                if isinstance(v, float) and not (v != v):
                    if k in ["CAGR", "Vol", "MaxDD"]:
                        print(f"  {k:15}: {v:10.4f} ({v*100:.2f}%)")
                    elif k == "Sharpe":
                        print(f"  {k:15}: {v:10.4f}")
                    elif k == "HitRate":
                        print(f"  {k:15}: {v:10.4f} ({v*100:.2f}%)")
                    else:
                        print(f"  {k:15}: {v:10.4f}")
                else:
                    print(f"  {k:15}: NaN")
            
            print("\nDelta (Current - Baseline):")
            for k, v in comp["metrics_delta"].items():
                if isinstance(v, float) and not (v != v):
                    if "CAGR" in k or "Vol" in k or "MaxDD" in k:
                        print(f"  {k:20}: {v:10.4f} ({v*100:.2f}%)")
                    elif "Sharpe" in k:
                        print(f"  {k:20}: {v:10.4f}")
                    elif "HitRate" in k:
                        print(f"  {k:20}: {v:10.4f} ({v*100:.2f}%)")
                    else:
                        print(f"  {k:20}: {v:10.4f}")
                else:
                    print(f"  {k:20}: NaN")
            
            if not comp["equity_ratio"].empty:
                print(f"\nEquity Ratio (Current/Baseline) over {len(comp['equity_ratio'])} overlapping days:")
                print(f"  Mean: {comp['equity_ratio'].mean():.4f}")
                print(f"  Final: {comp['equity_ratio'].iloc[-1]:.4f}")
        
        print("\n" + "=" * 80)
        print("Diagnostics complete!")
        print("=" * 80)
        
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

