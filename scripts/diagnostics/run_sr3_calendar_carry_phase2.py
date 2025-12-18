#!/usr/bin/env python3
"""
Phase-2 Diagnostics Script for SR3 Calendar Carry Integration.

Compares baseline (Core v7) vs SR3 Calendar Carry-enhanced portfolio.

This script:
1. Loads Core v7 baseline returns (or runs it if needed)
2. Loads SR3 Calendar Carry Phase-1 returns
3. Combines them with fixed weight (5-10% for carry)
4. Compares portfolio metrics (Sharpe, CAGR, Vol, MaxDD, HitRate)
5. Analyzes crisis-period performance
6. Computes correlation between Core v7 and Carry
7. Saves comparison outputs and registers in phase index

Usage:
    python scripts/diagnostics/run_sr3_calendar_carry_phase2.py
    python scripts/diagnostics/run_sr3_calendar_carry_phase2.py --start 2020-01-02 --end 2025-10-31 --carry-weight 0.05
"""

import sys
import argparse
from pathlib import Path
import logging
from datetime import datetime
from typing import Dict, Optional
import json
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config.backtest_window import CANONICAL_START, CANONICAL_END
from run_strategy import main as run_strategy_main
from src.diagnostics.tsmom_sanity import compute_summary_stats
from src.utils.phase_index import get_phase_path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_run_returns(run_id: str) -> pd.Series:
    """
    Load portfolio returns from a run.
    
    Args:
        run_id: Run identifier or path to run directory
        
    Returns:
        Series of portfolio returns indexed by date
    """
    # Check if run_id is a path
    if Path(run_id).exists() and Path(run_id).is_dir():
        run_dir = Path(run_id)
    else:
        # Find the run directory (check both data/runs and reports/runs)
        run_dir = Path(f"reports/runs/{run_id}")
        if not run_dir.exists():
            run_dir = Path(f"data/runs/{run_id}")
        if not run_dir.exists():
            raise FileNotFoundError(f"Run directory not found: reports/runs/{run_id} or data/runs/{run_id}")
    
    # Load portfolio returns
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


def load_carry_phase1_returns() -> pd.Series:
    """
    Load SR3 Calendar Carry Phase-1 returns from the latest run.
    
    Returns:
        Series of portfolio returns indexed by date
    """
    # Get Phase-1 path from phase index
    phase1_path = get_phase_path("carry", "sr3_calendar_carry", "phase1")
    
    if phase1_path is None:
        raise FileNotFoundError(
            "SR3 Calendar Carry Phase-1 not found. "
            "Please run Phase-1 first: python scripts/run_sr3_calendar_carry_phase1.py"
        )
    
    # Phase-1 run_id is just the timestamp, but the actual directory is nested
    # Try the direct path first, then try the nested path
    if phase1_path.exists():
        logger.info(f"Loading Phase-1 returns from: {phase1_path}")
        return load_run_returns(str(phase1_path))
    else:
        # Try nested path: reports/runs/carry/sr3_calendar_carry_phase1/{run_id}/
        nested_path = Path("reports/runs/carry/sr3_calendar_carry_phase1") / phase1_path.name
        if nested_path.exists():
            logger.info(f"Loading Phase-1 returns from: {nested_path}")
            return load_run_returns(str(nested_path))
        else:
            raise FileNotFoundError(
                f"SR3 Calendar Carry Phase-1 run not found at {phase1_path} or {nested_path}. "
                "Please run Phase-1 first: python scripts/run_sr3_calendar_carry_phase1.py"
            )


def run_core_v7_if_needed(start_date: str, end_date: str) -> str:
    """
    Run Core v7 baseline if needed, or return existing run_id.
    
    Args:
        start_date: Start date
        end_date: End date
        
    Returns:
        Run ID for Core v7
    """
    # For now, we'll run Core v7 each time to ensure consistency
    # In production, you might want to cache this
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"core_v7_baseline_phase2_{timestamp}"
    
    logger.info("=" * 80)
    logger.info("Running Core v7 baseline")
    logger.info("=" * 80)
    
    # Build command-line arguments for run_strategy.py
    original_argv = sys.argv.copy()
    sys.argv = [
        "run_strategy.py",
        "--strategy_profile", "core_v7_trend_csmom_vrp_core_convergence_vrp_alt_no_macro",
        "--run_id", run_id,
        "--start", start_date,
        "--end", end_date
    ]
    
    try:
        run_strategy_main()
        logger.info(f"✓ Completed Core v7 run: {run_id}")
        return run_id
    except Exception as e:
        logger.error(f"✗ Failed Core v7 run: {run_id}")
        logger.error(f"Error: {e}")
        raise
    finally:
        sys.argv = original_argv


def compute_crisis_periods(returns: pd.Series) -> dict:
    """
    Compute performance metrics for crisis periods.
    
    Args:
        returns: Portfolio returns series
        
    Returns:
        Dict with crisis period metrics
    """
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
        description="Phase-2 Diagnostics: SR3 Calendar Carry Integration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with canonical dates and default 5% carry weight
  python scripts/diagnostics/run_sr3_calendar_carry_phase2.py
  
  # Run with custom dates and 10% carry weight
  python scripts/diagnostics/run_sr3_calendar_carry_phase2.py --start 2020-01-02 --end 2025-10-31 --carry-weight 0.10
  
  # Use existing Core v7 run
  python scripts/diagnostics/run_sr3_calendar_carry_phase2.py --core-v7-run-id <run_id>
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
        "--carry-weight",
        type=float,
        default=0.05,
        help="Weight for SR3 Calendar Carry (default: 0.05 = 5%%)"
    )
    parser.add_argument(
        "--core-v7-run-id",
        type=str,
        default=None,
        help="Existing Core v7 run ID to use (if not provided, will run Core v7)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: reports/runs/carry/sr3_calendar_carry_phase2/{timestamp})"
    )
    
    args = parser.parse_args()
    
    # Validate carry weight
    if args.carry_weight <= 0 or args.carry_weight >= 1:
        raise ValueError(f"carry-weight must be between 0 and 1, got {args.carry_weight}")
    
    try:
        logger.info("=" * 80)
        logger.info("SR3 CALENDAR CARRY PHASE-2 DIAGNOSTICS")
        logger.info("=" * 80)
        logger.info(f"Start date: {args.start}")
        logger.info(f"End date: {args.end}")
        logger.info(f"Carry weight: {args.carry_weight:.1%}")
        
        # Determine output directory
        if args.output_dir:
            output_dir = Path(args.output_dir)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = Path(f"reports/runs/carry/sr3_calendar_carry_phase2/{timestamp}")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {output_dir}")
        
        # 1) Load or run Core v7 baseline
        logger.info("\n" + "=" * 80)
        logger.info("[1/4] Loading Core v7 baseline")
        logger.info("=" * 80)
        
        if args.core_v7_run_id:
            core_v7_run_id = args.core_v7_run_id
            logger.info(f"Using existing Core v7 run: {core_v7_run_id}")
        else:
            core_v7_run_id = run_core_v7_if_needed(args.start, args.end)
        
        baseline_returns = load_run_returns(core_v7_run_id)
        
        # 2) Load SR3 Calendar Carry Phase-1 returns
        logger.info("\n" + "=" * 80)
        logger.info("[2/4] Loading SR3 Calendar Carry Phase-1 returns")
        logger.info("=" * 80)
        
        carry_returns = load_carry_phase1_returns()
        
        # 3) Combine returns with fixed weight
        logger.info("\n" + "=" * 80)
        logger.info("[3/4] Combining returns")
        logger.info("=" * 80)
        
        # Align returns
        common_dates = baseline_returns.index.intersection(carry_returns.index)
        baseline_returns = baseline_returns.loc[common_dates]
        carry_returns = carry_returns.loc[common_dates]
        
        logger.info(f"Aligned returns: {len(baseline_returns)} days")
        logger.info(f"Baseline date range: {baseline_returns.index.min()} to {baseline_returns.index.max()}")
        logger.info(f"Carry date range: {carry_returns.index.min()} to {carry_returns.index.max()}")
        
        # Combine: (1 - w) * baseline + w * carry
        combined_returns = (1 - args.carry_weight) * baseline_returns + args.carry_weight * carry_returns
        
        # 4) Compute metrics
        logger.info("\n" + "=" * 80)
        logger.info("[4/4] Computing comparison metrics")
        logger.info("=" * 80)
        
        # Compute baseline metrics
        baseline_equity = (1 + baseline_returns).cumprod()
        baseline_asset_returns = pd.DataFrame({'Portfolio': baseline_returns})
        baseline_stats = compute_summary_stats(
            portfolio_returns=baseline_returns,
            equity_curve=baseline_equity,
            asset_strategy_returns=baseline_asset_returns
        )
        baseline_metrics = baseline_stats['portfolio']
        
        # Compute combined metrics
        combined_equity = (1 + combined_returns).cumprod()
        combined_asset_returns = pd.DataFrame({'Portfolio': combined_returns})
        combined_stats = compute_summary_stats(
            portfolio_returns=combined_returns,
            equity_curve=combined_equity,
            asset_strategy_returns=combined_asset_returns
        )
        combined_metrics = combined_stats['portfolio']
        
        # Compute carry-only metrics (for reference)
        carry_equity = (1 + carry_returns).cumprod()
        carry_asset_returns = pd.DataFrame({'Portfolio': carry_returns})
        carry_stats = compute_summary_stats(
            portfolio_returns=carry_returns,
            equity_curve=carry_equity,
            asset_strategy_returns=carry_asset_returns
        )
        carry_metrics = carry_stats['portfolio']
        
        # Compute crisis period metrics
        logger.info("Computing crisis period metrics...")
        baseline_crisis = compute_crisis_periods(baseline_returns)
        combined_crisis = compute_crisis_periods(combined_returns)
        
        # Compute correlation
        correlation = baseline_returns.corr(carry_returns)
        
        # Compute difference metrics
        diff_metrics = {
            'sharpe_diff': combined_metrics.get('Sharpe', 0) - baseline_metrics.get('Sharpe', 0),
            'cagr_diff': combined_metrics.get('CAGR', 0) - baseline_metrics.get('CAGR', 0),
            'vol_diff': combined_metrics.get('Vol', 0) - baseline_metrics.get('Vol', 0),
            'maxdd_diff': combined_metrics.get('MaxDD', 0) - baseline_metrics.get('MaxDD', 0),
            'hitrate_diff': combined_metrics.get('HitRate', 0) - baseline_metrics.get('HitRate', 0),
        }
        
        # Save outputs
        logger.info("\n" + "=" * 80)
        logger.info("Saving comparison outputs")
        logger.info("=" * 80)
        
        # Save returns
        returns_df = pd.DataFrame({
            'baseline': baseline_returns,
            'carry': carry_returns,
            'combined': combined_returns
        })
        returns_df.to_csv(output_dir / 'comparison_returns.csv')
        returns_df.to_parquet(output_dir / 'comparison_returns.parquet')
        
        # Save equity curves
        equity_df = pd.DataFrame({
            'baseline': baseline_equity,
            'carry': carry_equity,
            'combined': combined_equity
        })
        equity_df.to_csv(output_dir / 'comparison_equity.csv')
        
        # Save metrics
        comparison_summary = {
            'run_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'start_date': args.start,
            'end_date': args.end,
            'carry_weight': args.carry_weight,
            'core_v7_run_id': core_v7_run_id,
            'baseline_metrics': baseline_metrics,
            'carry_metrics': carry_metrics,
            'combined_metrics': combined_metrics,
            'diff_metrics': diff_metrics,
            'correlation_core_v7_vs_carry': float(correlation),
            'baseline_crisis': baseline_crisis,
            'combined_crisis': combined_crisis,
        }
        
        with open(output_dir / 'comparison_summary.json', 'w') as f:
            json.dump(comparison_summary, f, indent=2, default=str)
        
        # Save diff metrics separately
        with open(output_dir / 'diff_metrics.json', 'w') as f:
            json.dump(diff_metrics, f, indent=2)
        
        # Generate plots
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            
            # Equity curves
            fig, ax = plt.subplots(figsize=(14, 6))
            ax.plot(baseline_equity.index, baseline_equity.values, label='Baseline (Core v7)', linewidth=1.5)
            ax.plot(combined_equity.index, combined_equity.values, label=f'Combined (Core v7 + {args.carry_weight:.1%} Carry)', linewidth=1.5)
            ax.plot(carry_equity.index, carry_equity.values, label='Carry Only', linewidth=1.5, alpha=0.5)
            ax.set_xlabel('Date')
            ax.set_ylabel('Equity')
            ax.set_title('Phase-2: Baseline vs Combined Equity Curves')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_dir / 'equity_curves.png', dpi=150)
            plt.close()
            
            # Drawdown curves
            baseline_dd = (baseline_equity / baseline_equity.expanding().max() - 1) * 100
            combined_dd = (combined_equity / combined_equity.expanding().max() - 1) * 100
            fig, ax = plt.subplots(figsize=(14, 6))
            ax.plot(baseline_dd.index, baseline_dd.values, label='Baseline (Core v7)', linewidth=1.5)
            ax.plot(combined_dd.index, combined_dd.values, label=f'Combined (Core v7 + {args.carry_weight:.1%} Carry)', linewidth=1.5)
            ax.set_xlabel('Date')
            ax.set_ylabel('Drawdown (%)')
            ax.set_title('Phase-2: Baseline vs Combined Drawdown Curves')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_dir / 'drawdown_curves.png', dpi=150)
            plt.close()
            
            logger.info(f"  Generated plots in {output_dir}")
        except ImportError:
            logger.warning("matplotlib not available, skipping plots")
        
        # Register in phase index
        phase_index_dir = Path("reports/phase_index/carry/sr3_calendar_carry")
        phase_index_dir.mkdir(parents=True, exist_ok=True)
        
        phase2_file = phase_index_dir / "phase2.txt"
        with open(phase2_file, 'w') as f:
            f.write(f"# Phase-2: SR3 Calendar Carry Portfolio Integration\n")
            f.write(f"# Registered: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"baseline_profile: core_v7_trend_csmom_vrp_core_convergence_vrp_alt_no_macro\n")
            f.write(f"carry_weight: {args.carry_weight}\n")
            f.write(f"start_date: {args.start}\n")
            f.write(f"end_date: {args.end}\n")
            f.write(f"\n# Baseline Metrics (Core v7)\n")
            f.write(f"baseline_sharpe: {baseline_metrics.get('Sharpe', float('nan')):.4f}\n")
            f.write(f"baseline_cagr: {baseline_metrics.get('CAGR', float('nan')):.4f}\n")
            f.write(f"baseline_maxdd: {baseline_metrics.get('MaxDD', float('nan')):.4f}\n")
            f.write(f"\n# Combined Metrics (Core v7 + Carry)\n")
            f.write(f"combined_sharpe: {combined_metrics.get('Sharpe', float('nan')):.4f}\n")
            f.write(f"combined_cagr: {combined_metrics.get('CAGR', float('nan')):.4f}\n")
            f.write(f"combined_maxdd: {combined_metrics.get('MaxDD', float('nan')):.4f}\n")
            f.write(f"\n# Differences\n")
            f.write(f"sharpe_diff: {diff_metrics.get('sharpe_diff', float('nan')):.4f}\n")
            f.write(f"cagr_diff: {diff_metrics.get('cagr_diff', float('nan')):.4f}\n")
            f.write(f"maxdd_diff: {diff_metrics.get('maxdd_diff', float('nan')):.4f}\n")
            f.write(f"\n# Correlation\n")
            f.write(f"corr_core_v7_vs_carry: {correlation:.4f}\n")
            f.write(f"\npath: {output_dir}\n")
            
            # Pass criteria check
            sharpe_diff = diff_metrics.get('sharpe_diff', float('-inf'))
            maxdd_diff = diff_metrics.get('maxdd_diff', float('inf'))
            
            f.write(f"\n# Phase-2 Pass Criteria\n")
            f.write(f"sharpe_pass: {'PASS' if sharpe_diff >= -0.01 else 'FAIL'} (combined >= baseline - 0.01)\n")
            f.write(f"maxdd_pass: {'PASS' if maxdd_diff >= 0 else 'FAIL'} (combined >= baseline)\n")
            
            verdict = "PASS" if (sharpe_diff >= -0.01 and maxdd_diff >= 0) else "FAIL"
            f.write(f"\nverdict: {verdict}\n")
        
        logger.info(f"  Registered in: {phase2_file}")
        
        # Print summary
        print("\n" + "=" * 80)
        print("SR3 CALENDAR CARRY PHASE-2 SUMMARY")
        print("=" * 80)
        print(f"\nBaseline (Core v7):")
        print(f"  Sharpe:  {baseline_metrics.get('Sharpe', float('nan')):8.4f}")
        print(f"  CAGR:   {baseline_metrics.get('CAGR', float('nan')):8.4f} ({baseline_metrics.get('CAGR', 0)*100:6.2f}%)")
        print(f"  MaxDD:  {baseline_metrics.get('MaxDD', float('nan')):8.4f} ({baseline_metrics.get('MaxDD', 0)*100:6.2f}%)")
        print(f"  Vol:    {baseline_metrics.get('Vol', float('nan')):8.4f} ({baseline_metrics.get('Vol', 0)*100:6.2f}%)")
        print(f"\nCarry Only (Phase-1):")
        print(f"  Sharpe:  {carry_metrics.get('Sharpe', float('nan')):8.4f}")
        print(f"  CAGR:   {carry_metrics.get('CAGR', float('nan')):8.4f} ({carry_metrics.get('CAGR', 0)*100:6.2f}%)")
        print(f"  MaxDD:  {carry_metrics.get('MaxDD', float('nan')):8.4f} ({carry_metrics.get('MaxDD', 0)*100:6.2f}%)")
        print(f"  Vol:    {carry_metrics.get('Vol', float('nan')):8.4f} ({carry_metrics.get('Vol', 0)*100:6.2f}%)")
        print(f"\nCombined (Core v7 + {args.carry_weight:.1%} Carry):")
        print(f"  Sharpe:  {combined_metrics.get('Sharpe', float('nan')):8.4f}")
        print(f"  CAGR:   {combined_metrics.get('CAGR', float('nan')):8.4f} ({combined_metrics.get('CAGR', 0)*100:6.2f}%)")
        print(f"  MaxDD:  {combined_metrics.get('MaxDD', float('nan')):8.4f} ({combined_metrics.get('MaxDD', 0)*100:6.2f}%)")
        print(f"  Vol:    {combined_metrics.get('Vol', float('nan')):8.4f} ({combined_metrics.get('Vol', 0)*100:6.2f}%)")
        print(f"\nDifferences:")
        print(f"  Sharpe:  {diff_metrics.get('sharpe_diff', float('nan')):8.4f}")
        print(f"  CAGR:   {diff_metrics.get('cagr_diff', float('nan')):8.4f} ({diff_metrics.get('cagr_diff', 0)*100:6.2f}%)")
        print(f"  MaxDD:  {diff_metrics.get('maxdd_diff', float('nan')):8.4f} ({diff_metrics.get('maxdd_diff', 0)*100:6.2f}%)")
        print(f"\nCorrelation (Core v7 vs Carry): {correlation:.4f}")
        print(f"\nResults saved to: {output_dir}")
        print("=" * 80)
        
    except Exception as e:
        logger.error(f"Phase-2 diagnostics failed: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()

