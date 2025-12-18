#!/usr/bin/env python3
"""
VRP-Alt Scaling Analysis: Phase-2 diagnostics across multiple weights.

Tests VRP-Alt at different weights (5%, 7.5%, 10%, 15%) to determine optimal allocation
by analyzing marginal contribution curves.

Usage:
    python scripts/diagnostics/run_vrp_alt_scaling_analysis.py
    python scripts/diagnostics/run_vrp_alt_scaling_analysis.py --start 2020-01-01 --end 2025-10-31
"""

import sys
import argparse
from pathlib import Path
import logging
from datetime import datetime
from typing import Dict, List
import json
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config.backtest_window import CANONICAL_START, CANONICAL_END
from run_strategy import main as run_strategy_main
from src.diagnostics.tsmom_sanity import compute_summary_stats

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_strategy_profile(profile_name: str, run_id: str, start_date: str, end_date: str):
    """Run a strategy profile and save results."""
    logger.info("=" * 80)
    logger.info(f"Running strategy profile: {profile_name}")
    logger.info(f"Run ID: {run_id}")
    logger.info(f"Date Range: {start_date} to {end_date}")
    logger.info("=" * 80)
    
    original_argv = sys.argv.copy()
    sys.argv = [
        "run_strategy.py",
        "--strategy_profile", profile_name,
        "--run_id", run_id,
        "--start", start_date,
        "--end", end_date
    ]
    
    try:
        run_strategy_main()
        logger.info(f"✓ Completed run: {run_id}")
    except Exception as e:
        logger.error(f"✗ Failed run: {run_id}")
        logger.error(f"Error: {e}")
        raise
    finally:
        sys.argv = original_argv


def load_run_returns(run_id: str) -> pd.Series:
    """Load portfolio returns from a run."""
    run_dir = Path(f"reports/runs/{run_id}")
    if not run_dir.exists():
        run_dir = Path(f"data/runs/{run_id}")
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: reports/runs/{run_id} or data/runs/{run_id}")
    
    returns_file = run_dir / "portfolio_returns.csv"
    if not returns_file.exists():
        raise FileNotFoundError(f"Portfolio returns file not found: {returns_file}")
    
    df = pd.read_csv(returns_file, index_col=0, parse_dates=True)
    if 'portfolio_return' in df.columns:
        returns = df['portfolio_return']
    elif 'ret' in df.columns:
        returns = df['ret']
    else:
        returns = df.iloc[:, 0]
    
    returns.name = 'portfolio_return'
    return returns


def compute_crisis_periods(returns: pd.Series) -> dict:
    """Compute performance metrics for crisis periods."""
    crisis_periods = {
        "2020_Q1": ("2020-01-01", "2020-03-31"),
        "2020_Q2": ("2020-04-01", "2020-06-30"),
        "2022": ("2022-01-01", "2022-12-31"),
    }
    
    crisis_metrics = {}
    
    for period_name, (start, end) in crisis_periods.items():
        period_returns = returns[(returns.index >= start) & (returns.index <= end)]
        
        if len(period_returns) == 0:
            continue
        
        equity = (1 + period_returns).cumprod()
        asset_strategy_returns = pd.DataFrame({'Portfolio': period_returns})
        
        stats = compute_summary_stats(
            portfolio_returns=period_returns,
            equity_curve=equity,
            asset_strategy_returns=asset_strategy_returns
        )
        
        crisis_metrics[period_name] = stats['portfolio']
    
    return crisis_metrics


def main():
    parser = argparse.ArgumentParser(
        description="VRP-Alt Scaling Analysis: Phase-2 diagnostics across multiple weights",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with canonical dates
  python scripts/diagnostics/run_vrp_alt_scaling_analysis.py
  
  # Run with custom dates
  python scripts/diagnostics/run_vrp_alt_scaling_analysis.py --start 2020-01-01 --end 2025-10-31
        """
    )
    
    parser.add_argument(
        "--start",
        type=str,
        default=CANONICAL_START,
        help=f"Start date (YYYY-MM-DD), default: {CANONICAL_START}"
    )
    parser.add_argument(
        "--end",
        type=str,
        default=CANONICAL_END,
        help=f"End date (YYYY-MM-DD), default: {CANONICAL_END}"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: data/diagnostics/phase2/vrp_alt_scaling)"
    )
    
    args = parser.parse_args()
    
    try:
        logger.info("=" * 80)
        logger.info("VRP-ALT SCALING ANALYSIS")
        logger.info("=" * 80)
        logger.info(f"Start date: {args.start}")
        logger.info(f"End date: {args.end}")
        
        # Determine output directory
        if args.output_dir:
            output_dir = Path(args.output_dir)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = Path(f"data/diagnostics/phase2/vrp_alt_scaling/{timestamp}")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {output_dir}")
        
        # Define test weights and profiles
        test_configs = [
            {"weight": 0.05, "profile": "core_v6_trend_csmom_vrp_core_convergence_vrp_alt_no_macro", "name": "5%"},
            {"weight": 0.075, "profile": "core_v6_trend_csmom_vrp_core_convergence_vrp_alt_7p5_no_macro", "name": "7.5%"},
            {"weight": 0.10, "profile": "core_v6_trend_csmom_vrp_core_convergence_vrp_alt_10_no_macro", "name": "10%"},
            {"weight": 0.15, "profile": "core_v6_trend_csmom_vrp_core_convergence_vrp_alt_15_no_macro", "name": "15%"},
        ]
        
        baseline_profile = "core_v6_trend_csmom_vrp_core_convergence_no_macro"
        
        # 1) Run baseline
        logger.info("\n" + "=" * 80)
        logger.info("[1/5] Running baseline: core_v6")
        logger.info("=" * 80)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        baseline_run_id = f"core_v6_baseline_scaling_{timestamp}"
        run_strategy_profile(
            profile_name=baseline_profile,
            run_id=baseline_run_id,
            start_date=args.start,
            end_date=args.end
        )
        
        baseline_returns = load_run_returns(baseline_run_id)
        baseline_equity = (1 + baseline_returns).cumprod()
        baseline_asset_returns = pd.DataFrame({'Portfolio': baseline_returns})
        baseline_stats = compute_summary_stats(
            portfolio_returns=baseline_returns,
            equity_curve=baseline_equity,
            asset_strategy_returns=baseline_asset_returns
        )
        baseline_metrics = baseline_stats['portfolio']
        
        # 2) Run each weight variant
        results = []
        results.append({
            "weight": 0.0,
            "weight_pct": 0.0,
            "name": "Baseline",
            "profile": baseline_profile,
            "run_id": baseline_run_id,
            "metrics": baseline_metrics,
            "crisis": compute_crisis_periods(baseline_returns)
        })
        
        for i, config in enumerate(test_configs, start=2):
            logger.info("\n" + "=" * 80)
            logger.info(f"[{i}/5] Running VRP-Alt at {config['name']} weight")
            logger.info("=" * 80)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            variant_run_id = f"core_v6_vrp_alt_{config['name'].replace('%', 'pct')}_scaling_{timestamp}"
            
            run_strategy_profile(
                profile_name=config['profile'],
                run_id=variant_run_id,
                start_date=args.start,
                end_date=args.end
            )
            
            variant_returns = load_run_returns(variant_run_id)
            
            # Align with baseline
            common_dates = baseline_returns.index.intersection(variant_returns.index)
            variant_returns = variant_returns.loc[common_dates]
            
            variant_equity = (1 + variant_returns).cumprod()
            variant_asset_returns = pd.DataFrame({'Portfolio': variant_returns})
            variant_stats = compute_summary_stats(
                portfolio_returns=variant_returns,
                equity_curve=variant_equity,
                asset_strategy_returns=variant_asset_returns
            )
            variant_metrics = variant_stats['portfolio']
            
            # Compute differences vs baseline
            diff_metrics = {
                'sharpe_diff': variant_metrics.get('Sharpe', 0) - baseline_metrics.get('Sharpe', 0),
                'cagr_diff': variant_metrics.get('CAGR', 0) - baseline_metrics.get('CAGR', 0),
                'vol_diff': variant_metrics.get('Vol', 0) - baseline_metrics.get('Vol', 0),
                'maxdd_diff': variant_metrics.get('MaxDD', 0) - baseline_metrics.get('MaxDD', 0),
                'hitrate_diff': variant_metrics.get('HitRate', 0) - baseline_metrics.get('HitRate', 0),
            }
            
            results.append({
                "weight": config['weight'],
                "weight_pct": config['weight'] * 100,
                "name": config['name'],
                "profile": config['profile'],
                "run_id": variant_run_id,
                "metrics": variant_metrics,
                "diff_metrics": diff_metrics,
                "crisis": compute_crisis_periods(variant_returns)
            })
        
        # 3) Analyze scaling curves
        logger.info("\n" + "=" * 80)
        logger.info("[5/5] Analyzing scaling curves")
        logger.info("=" * 80)
        
        # Build scaling DataFrame
        scaling_data = []
        for r in results:
            scaling_data.append({
                "weight_pct": r["weight_pct"],
                "sharpe": r["metrics"].get("Sharpe", float('nan')),
                "cagr": r["metrics"].get("CAGR", float('nan')),
                "vol": r["metrics"].get("Vol", float('nan')),
                "maxdd": r["metrics"].get("MaxDD", float('nan')),
                "hitrate": r["metrics"].get("HitRate", float('nan')),
                "sharpe_diff": r.get("diff_metrics", {}).get("sharpe_diff", 0.0),
                "cagr_diff": r.get("diff_metrics", {}).get("cagr_diff", 0.0),
                "maxdd_diff": r.get("diff_metrics", {}).get("maxdd_diff", 0.0),
            })
        
        scaling_df = pd.DataFrame(scaling_data)
        scaling_df = scaling_df.sort_values("weight_pct")
        
        # Compute marginal contributions (change per 1% weight increase)
        scaling_df['sharpe_marginal'] = scaling_df['sharpe_diff'].diff() / scaling_df['weight_pct'].diff()
        scaling_df['cagr_marginal'] = scaling_df['cagr_diff'].diff() / scaling_df['weight_pct'].diff()
        scaling_df['maxdd_marginal'] = scaling_df['maxdd_diff'].diff() / scaling_df['weight_pct'].diff()
        
        # 4) Save results
        logger.info("Saving scaling analysis results...")
        
        # Save scaling DataFrame
        scaling_df.to_csv(output_dir / 'scaling_analysis.csv', index=False)
        scaling_df.to_parquet(output_dir / 'scaling_analysis.parquet', index=False)
        
        # Save full results JSON
        results_json = []
        for r in results:
            results_json.append({
                "weight_pct": r["weight_pct"],
                "name": r["name"],
                "profile": r["profile"],
                "run_id": r["run_id"],
                "metrics": r["metrics"],
                "diff_metrics": r.get("diff_metrics", {}),
                "crisis": r.get("crisis", {})
            })
        
        with open(output_dir / 'scaling_results.json', 'w') as f:
            json.dump(results_json, f, indent=2, default=str)
        
        # Generate scaling plots
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            
            # Plot 1: Sharpe vs Weight
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            # Sharpe
            axes[0, 0].plot(scaling_df['weight_pct'], scaling_df['sharpe'], marker='o', linewidth=2, markersize=8)
            axes[0, 0].set_xlabel('VRP-Alt Weight (%)')
            axes[0, 0].set_ylabel('Sharpe Ratio')
            axes[0, 0].set_title('Sharpe Ratio vs VRP-Alt Weight')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Sharpe Difference
            axes[0, 1].plot(scaling_df['weight_pct'], scaling_df['sharpe_diff'], marker='o', linewidth=2, markersize=8, color='green')
            axes[0, 1].axhline(0, color='black', linestyle='--', alpha=0.3)
            axes[0, 1].set_xlabel('VRP-Alt Weight (%)')
            axes[0, 1].set_ylabel('Sharpe Difference vs Baseline')
            axes[0, 1].set_title('Sharpe Improvement vs Baseline')
            axes[0, 1].grid(True, alpha=0.3)
            
            # MaxDD
            axes[1, 0].plot(scaling_df['weight_pct'], scaling_df['maxdd'] * 100, marker='o', linewidth=2, markersize=8, color='red')
            axes[1, 0].set_xlabel('VRP-Alt Weight (%)')
            axes[1, 0].set_ylabel('MaxDD (%)')
            axes[1, 0].set_title('MaxDD vs VRP-Alt Weight')
            axes[1, 0].grid(True, alpha=0.3)
            
            # MaxDD Difference
            axes[1, 1].plot(scaling_df['weight_pct'], scaling_df['maxdd_diff'] * 100, marker='o', linewidth=2, markersize=8, color='orange')
            axes[1, 1].axhline(0, color='black', linestyle='--', alpha=0.3)
            axes[1, 1].set_xlabel('VRP-Alt Weight (%)')
            axes[1, 1].set_ylabel('MaxDD Difference vs Baseline (%)')
            axes[1, 1].set_title('MaxDD Change vs Baseline')
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(output_dir / 'scaling_curves.png', dpi=150)
            plt.close()
            
            # Plot 2: Marginal Contributions
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            
            # Marginal Sharpe (per 1% weight)
            axes[0].plot(scaling_df['weight_pct'].iloc[1:], scaling_df['sharpe_marginal'].iloc[1:], marker='o', linewidth=2, markersize=8, color='blue')
            axes[0].axhline(0, color='black', linestyle='--', alpha=0.3)
            axes[0].set_xlabel('VRP-Alt Weight (%)')
            axes[0].set_ylabel('Marginal Sharpe (per 1% weight)')
            axes[0].set_title('Marginal Sharpe Contribution')
            axes[0].grid(True, alpha=0.3)
            
            # Marginal CAGR
            axes[1].plot(scaling_df['weight_pct'].iloc[1:], scaling_df['cagr_marginal'].iloc[1:] * 100, marker='o', linewidth=2, markersize=8, color='green')
            axes[1].axhline(0, color='black', linestyle='--', alpha=0.3)
            axes[1].set_xlabel('VRP-Alt Weight (%)')
            axes[1].set_ylabel('Marginal CAGR (per 1% weight, %)')
            axes[1].set_title('Marginal CAGR Contribution')
            axes[1].grid(True, alpha=0.3)
            
            # Marginal MaxDD
            axes[2].plot(scaling_df['weight_pct'].iloc[1:], scaling_df['maxdd_marginal'].iloc[1:] * 100, marker='o', linewidth=2, markersize=8, color='red')
            axes[2].axhline(0, color='black', linestyle='--', alpha=0.3)
            axes[2].set_xlabel('VRP-Alt Weight (%)')
            axes[2].set_ylabel('Marginal MaxDD (per 1% weight, %)')
            axes[2].set_title('Marginal MaxDD Contribution')
            axes[2].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(output_dir / 'marginal_contributions.png', dpi=150)
            plt.close()
            
            logger.info(f"  Generated scaling plots in {output_dir}")
        except ImportError:
            logger.warning("matplotlib not available, skipping plots")
        
        # 5) Print summary
        print("\n" + "=" * 80)
        print("VRP-ALT SCALING ANALYSIS SUMMARY")
        print("=" * 80)
        print(f"\nBaseline (core_v6):")
        print(f"  Sharpe:  {baseline_metrics.get('Sharpe', float('nan')):8.4f}")
        print(f"  CAGR:   {baseline_metrics.get('CAGR', float('nan')):8.4f} ({baseline_metrics.get('CAGR', 0)*100:6.2f}%)")
        print(f"  MaxDD:  {baseline_metrics.get('MaxDD', float('nan')):8.4f} ({baseline_metrics.get('MaxDD', 0)*100:6.2f}%)")
        
        print(f"\nScaling Results:")
        print(f"{'Weight':<8} {'Sharpe':<10} {'Sharpe Δ':<12} {'CAGR Δ':<12} {'MaxDD Δ':<12} {'Marg. Sharpe':<15}")
        print("-" * 80)
        for r in results[1:]:  # Skip baseline
            weight_pct = r["weight_pct"]
            sharpe = r["metrics"].get("Sharpe", float('nan'))
            sharpe_diff = r.get("diff_metrics", {}).get("sharpe_diff", 0.0)
            cagr_diff = r.get("diff_metrics", {}).get("cagr_diff", 0.0)
            maxdd_diff = r.get("diff_metrics", {}).get("maxdd_diff", 0.0)
            
            # Find marginal contribution
            row = scaling_df[scaling_df['weight_pct'] == weight_pct]
            marg_sharpe = row['sharpe_marginal'].iloc[0] if not row.empty else float('nan')
            
            print(f"{weight_pct:>6.1f}%  {sharpe:>9.4f}  {sharpe_diff:>+11.4f}  {cagr_diff*100:>+10.2f}%  {maxdd_diff*100:>+10.2f}%  {marg_sharpe:>+14.4f}")
        
        # Find optimal weight (highest Sharpe with acceptable MaxDD)
        print(f"\nOptimal Weight Analysis:")
        best_sharpe_idx = scaling_df['sharpe'].idxmax()
        best_sharpe_weight = scaling_df.loc[best_sharpe_idx, 'weight_pct']
        best_sharpe_value = scaling_df.loc[best_sharpe_idx, 'sharpe']
        
        print(f"  Highest Sharpe: {best_sharpe_value:.4f} at {best_sharpe_weight:.1f}% weight")
        
        # Check MaxDD at best Sharpe weight
        best_sharpe_maxdd = scaling_df.loc[best_sharpe_idx, 'maxdd']
        baseline_maxdd = baseline_metrics.get('MaxDD', 0)
        maxdd_increase = (best_sharpe_maxdd - baseline_maxdd) * 100
        
        if maxdd_increase <= 1.0:
            print(f"  MaxDD at {best_sharpe_weight:.1f}%: {best_sharpe_maxdd*100:.2f}% (increase: {maxdd_increase:.2f}% - ACCEPTABLE)")
        else:
            print(f"  MaxDD at {best_sharpe_weight:.1f}%: {best_sharpe_maxdd*100:.2f}% (increase: {maxdd_increase:.2f}% - WARNING)")
        
        # Check marginal contributions
        print(f"\nMarginal Contributions (per 1% weight increase):")
        for r in results[1:]:
            weight_pct = r["weight_pct"]
            row = scaling_df[scaling_df['weight_pct'] == weight_pct]
            if not row.empty:
                marg_sharpe = row['sharpe_marginal'].iloc[0]
                marg_cagr = row['cagr_marginal'].iloc[0]
                marg_maxdd = row['maxdd_marginal'].iloc[0]
                
                print(f"  {weight_pct:>5.1f}%: Sharpe={marg_sharpe:>+7.4f}, CAGR={marg_cagr*100:>+6.2f}%, MaxDD={marg_maxdd*100:>+6.2f}%")
        
        print(f"\nResults saved to: {output_dir}")
        print("=" * 80)
        
    except Exception as e:
        logger.error(f"Scaling analysis failed: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()

