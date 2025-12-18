#!/usr/bin/env python3
"""
Phase-2 Diagnostics Script for VX Calendar Carry Integration.

Compares baseline (Core v7) vs VX Calendar Carry-enhanced portfolio.

This script:
1. Loads Core v7 baseline returns (or runs it if needed)
2. Loads VX Calendar Carry Phase-1 returns (for specified variant)
3. Combines them with fixed weight (5% for carry)
4. Compares portfolio metrics (Sharpe, CAGR, Vol, MaxDD, HitRate)
5. Analyzes crisis-period performance
6. Computes correlation between Core v7, VX Carry, and Core v7 sleeves
7. Saves comparison outputs and registers in phase index

Usage:
    python scripts/diagnostics/run_vx_calendar_carry_phase2.py --variant vx2_vx1_short
    python scripts/diagnostics/run_vx_calendar_carry_phase2.py --variant vx2_vx1_short --start 2020-01-02 --end 2025-10-31 --carry-weight 0.05
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
from src.agents import MarketData
from scripts.diagnostics.run_core_v6_trend_csmom_vrp_core_convergence_phase2 import compute_sleeve_returns

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# VX Carry variants
VARIANT_VX2_VX1_SHORT = "vx2_vx1_short"
VARIANT_VX3_VX2_SHORT = "vx3_vx2_short"


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


def load_vx_carry_phase1_returns(variant: str) -> pd.Series:
    """
    Load VX Calendar Carry Phase-1 returns for the specified variant.
    
    Args:
        variant: Variant name (vx2_vx1_short or vx3_vx2_short)
        
    Returns:
        Series of portfolio returns indexed by date
    """
    # Get Phase-1 path from phase index
    atomic_sleeve = f"vx_calendar_carry/{variant}"
    phase1_path_obj = get_phase_path("carry", atomic_sleeve, "phase1")
    
    if phase1_path_obj is None:
        raise FileNotFoundError(
            f"VX Calendar Carry Phase-1 ({variant}) not found. "
            f"Please run Phase-1 first: python scripts/run_vx_calendar_carry_phase1.py --variant {variant}"
        )
    
    # Convert to string and normalize path separators
    phase1_path_str = str(phase1_path_obj).replace("\\", "/")
    
    # Extract relative path from phase index (remove "reports/runs/" prefix if present)
    if phase1_path_str.startswith("reports/runs/"):
        relative_path = phase1_path_str[len("reports/runs/"):]
    else:
        relative_path = phase1_path_str
    
    # Phase index may contain:
    # 1. New format: vxcarry_phase1_{variant}_{timestamp} (top-level run_id)
    # 2. Old format: {timestamp}/{variant} (nested path)
    
    # Try parsing as run_id first (new format)
    if "/" not in relative_path:
        # New format: direct run_id
        run_id = relative_path
        run_dir = Path(f"reports/runs/carry/vx_calendar_carry_phase1/{run_id}")
        if run_dir.exists():
            logger.info(f"Loading Phase-1 returns from: {run_dir}")
            return load_run_returns(str(run_dir))
    
    # Old format: {timestamp}/{variant}
    parts = relative_path.split("/")
    if len(parts) == 2:
        timestamp, variant_name = parts
        # Try nested path: reports/runs/carry/vx_calendar_carry_phase1/{timestamp}/{variant}/
        nested_path = Path(f"reports/runs/carry/vx_calendar_carry_phase1/{timestamp}/{variant_name}")
        if nested_path.exists():
            logger.info(f"Loading Phase-1 returns from: {nested_path}")
            return load_run_returns(str(nested_path))
    
    # Try as direct path (full path from phase index)
    phase1_path = Path(phase1_path_str)
    if phase1_path.exists():
        logger.info(f"Loading Phase-1 returns from: {phase1_path}")
        return load_run_returns(str(phase1_path))
    
    # Try reports/runs/{relative_path} format
    run_dir = Path(f"reports/runs/{relative_path}")
    if run_dir.exists():
        logger.info(f"Loading Phase-1 returns from: {run_dir}")
        return load_run_returns(str(run_dir))
    
    raise FileNotFoundError(
        f"VX Calendar Carry Phase-1 run not found for variant {variant}. "
        f"Phase index points to: {phase1_path_str}. "
        f"Please run Phase-1 first: python scripts/run_vx_calendar_carry_phase1.py --variant {variant}"
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
        logger.info(f"[PASS] Completed Core v7 run: {run_id}")
        return run_id
    except Exception as e:
        logger.error(f"[FAIL] Failed Core v7 run: {run_id}")
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
        description="Phase-2 Diagnostics: VX Calendar Carry Integration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with canonical dates, default 5% carry weight, VX2-VX1_short variant
  python scripts/diagnostics/run_vx_calendar_carry_phase2.py --variant vx2_vx1_short
  
  # Run with custom dates and 10% carry weight
  python scripts/diagnostics/run_vx_calendar_carry_phase2.py --variant vx2_vx1_short --start 2020-01-02 --end 2025-10-31 --carry-weight 0.10
  
  # Use existing Core v7 run
  python scripts/diagnostics/run_vx_calendar_carry_phase2.py --variant vx2_vx1_short --core-v7-run-id <run_id>
        """
    )
    
    parser.add_argument(
        "--variant",
        type=str,
        required=True,
        choices=[VARIANT_VX2_VX1_SHORT, VARIANT_VX3_VX2_SHORT],
        help="VX Carry variant: vx2_vx1_short (primary) or vx3_vx2_short (optional)"
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
        help="Weight for VX Calendar Carry (default: 0.05 = 5%%)"
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
        help="Output directory (default: reports/runs/carry/vx_calendar_carry_phase2/{variant}/{timestamp})"
    )
    
    args = parser.parse_args()
    
    # Validate carry weight
    if args.carry_weight <= 0 or args.carry_weight >= 1:
        raise ValueError(f"carry-weight must be between 0 and 1, got {args.carry_weight}")
    
    try:
        logger.info("=" * 80)
        logger.info("VX CALENDAR CARRY PHASE-2 DIAGNOSTICS")
        logger.info("=" * 80)
        logger.info(f"Variant: {args.variant}")
        logger.info(f"Start date: {args.start}")
        logger.info(f"End date: {args.end}")
        logger.info(f"Carry weight: {args.carry_weight:.1%}")
        
        # Determine output directory
        if args.output_dir:
            output_dir = Path(args.output_dir)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = Path(f"reports/runs/carry/vx_calendar_carry_phase2/{args.variant}/{timestamp}")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {output_dir}")
        
        # 1) Load or run Core v7 baseline
        logger.info("\n" + "=" * 80)
        logger.info("[1/5] Loading Core v7 baseline")
        logger.info("=" * 80)
        
        if args.core_v7_run_id:
            core_v7_run_id = args.core_v7_run_id
            logger.info(f"Using existing Core v7 run: {core_v7_run_id}")
        else:
            core_v7_run_id = run_core_v7_if_needed(args.start, args.end)
        
        baseline_returns = load_run_returns(core_v7_run_id)
        
        # 2) Load VX Calendar Carry Phase-1 returns
        logger.info("\n" + "=" * 80)
        logger.info(f"[2/5] Loading VX Calendar Carry Phase-1 returns ({args.variant})")
        logger.info("=" * 80)
        
        carry_returns = load_vx_carry_phase1_returns(args.variant)
        
        # 3) Combine returns with fixed weight
        logger.info("\n" + "=" * 80)
        logger.info("[3/5] Combining returns")
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
        logger.info("[4/5] Computing comparison metrics")
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
        
        # 5) Compute sleeve-level correlations
        logger.info("\n" + "=" * 80)
        logger.info("[5/5] Computing sleeve-level correlations")
        logger.info("=" * 80)
        
        sleeve_corr_summary = {}
        corr_matrix = pd.DataFrame()
        
        try:
            market = MarketData()
            sleeve_returns = compute_sleeve_returns(
                profile_name="core_v7_trend_csmom_vrp_core_convergence_vrp_alt_no_macro",
                start_date=args.start,
                end_date=args.end,
                market=market
            )
            
            # Align all return series
            all_dates = common_dates.copy()
            for sleeve_name, sleeve_ret in sleeve_returns.items():
                all_dates = all_dates.intersection(sleeve_ret.index)
            
            # Build correlation DataFrame
            corr_data = {}
            corr_data['baseline_portfolio'] = baseline_returns.loc[all_dates]
            corr_data['vx_carry'] = carry_returns.loc[all_dates]
            
            # Sleeve returns
            for sleeve_name, sleeve_ret in sleeve_returns.items():
                corr_data[sleeve_name] = sleeve_ret.loc[all_dates]
            
            # Build DataFrame and compute correlation matrix
            if len(corr_data) > 1:
                corr_df = pd.DataFrame(corr_data)
                corr_matrix = corr_df.corr()
                
                # Extract correlations with VX carry
                if 'vx_carry' in corr_matrix.index:
                    for col in corr_matrix.columns:
                        if col != 'vx_carry':
                            sleeve_corr_summary[col] = float(corr_matrix.loc['vx_carry', col])
                
                logger.info("Sleeve correlation matrix computed")
                logger.info(f"\n{corr_matrix}")
            else:
                logger.warning("Insufficient data for sleeve correlation matrix")
        except Exception as e:
            logger.warning(f"Failed to compute sleeve correlations: {e}")
            import traceback
            traceback.print_exc()
        
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
            'vx_carry': carry_returns,
            'combined': combined_returns
        })
        returns_df.to_csv(output_dir / 'comparison_returns.csv')
        returns_df.to_parquet(output_dir / 'comparison_returns.parquet')
        
        # Save equity curves
        equity_df = pd.DataFrame({
            'baseline': baseline_equity,
            'vx_carry': carry_equity,
            'combined': combined_equity
        })
        equity_df.to_csv(output_dir / 'comparison_equity.csv')
        
        # Save correlation matrix if available
        if not corr_matrix.empty:
            corr_matrix.to_csv(output_dir / 'sleeve_correlation_matrix.csv')
        
        # Save metrics
        comparison_summary = {
            'run_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'variant': args.variant,
            'start_date': args.start,
            'end_date': args.end,
            'carry_weight': args.carry_weight,
            'core_v7_run_id': core_v7_run_id,
            'baseline_metrics': baseline_metrics,
            'carry_metrics': carry_metrics,
            'combined_metrics': combined_metrics,
            'diff_metrics': diff_metrics,
            'correlation_core_v7_vs_carry': float(correlation),
            'sleeve_correlations': sleeve_corr_summary,
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
            ax.plot(combined_equity.index, combined_equity.values, label=f'Combined (Core v7 + {args.carry_weight:.1%} VX Carry)', linewidth=1.5)
            ax.plot(carry_equity.index, carry_equity.values, label='VX Carry Only', linewidth=1.5, alpha=0.5)
            ax.set_xlabel('Date')
            ax.set_ylabel('Equity')
            ax.set_title(f'Phase-2: Baseline vs Combined Equity Curves ({args.variant})')
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
            ax.plot(combined_dd.index, combined_dd.values, label=f'Combined (Core v7 + {args.carry_weight:.1%} VX Carry)', linewidth=1.5)
            ax.set_xlabel('Date')
            ax.set_ylabel('Drawdown (%)')
            ax.set_title(f'Phase-2: Baseline vs Combined Drawdown Curves ({args.variant})')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_dir / 'drawdown_curves.png', dpi=150)
            plt.close()
            
            logger.info(f"  Generated plots in {output_dir}")
        except ImportError:
            logger.warning("matplotlib not available, skipping plots")
        
        # Register in phase index
        phase_index_dir = Path(f"reports/phase_index/carry/vx_calendar_carry/{args.variant}")
        phase_index_dir.mkdir(parents=True, exist_ok=True)
        
        phase2_file = phase_index_dir / "phase2.txt"
        with open(phase2_file, 'w') as f:
            f.write(f"# Phase-2: VX Calendar Carry Portfolio Integration ({args.variant})\n")
            f.write(f"# Registered: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"variant: {args.variant}\n")
            f.write(f"baseline_profile: core_v7_trend_csmom_vrp_core_convergence_vrp_alt_no_macro\n")
            f.write(f"carry_weight: {args.carry_weight}\n")
            f.write(f"start_date: {args.start}\n")
            f.write(f"end_date: {args.end}\n")
            f.write(f"\n# Baseline Metrics (Core v7)\n")
            f.write(f"baseline_sharpe: {baseline_metrics.get('Sharpe', float('nan')):.4f}\n")
            f.write(f"baseline_cagr: {baseline_metrics.get('CAGR', float('nan')):.4f}\n")
            f.write(f"baseline_maxdd: {baseline_metrics.get('MaxDD', float('nan')):.4f}\n")
            f.write(f"\n# VX Carry Metrics (Phase-1)\n")
            f.write(f"carry_sharpe: {carry_metrics.get('Sharpe', float('nan')):.4f}\n")
            f.write(f"carry_cagr: {carry_metrics.get('CAGR', float('nan')):.4f}\n")
            f.write(f"carry_maxdd: {carry_metrics.get('MaxDD', float('nan')):.4f}\n")
            f.write(f"\n# Combined Metrics (Core v7 + VX Carry)\n")
            f.write(f"combined_sharpe: {combined_metrics.get('Sharpe', float('nan')):.4f}\n")
            f.write(f"combined_cagr: {combined_metrics.get('CAGR', float('nan')):.4f}\n")
            f.write(f"combined_maxdd: {combined_metrics.get('MaxDD', float('nan')):.4f}\n")
            f.write(f"\n# Differences\n")
            f.write(f"sharpe_diff: {diff_metrics.get('sharpe_diff', float('nan')):.4f}\n")
            f.write(f"cagr_diff: {diff_metrics.get('cagr_diff', float('nan')):.4f}\n")
            f.write(f"maxdd_diff: {diff_metrics.get('maxdd_diff', float('nan')):.4f}\n")
            f.write(f"vol_diff: {diff_metrics.get('vol_diff', float('nan')):.4f}\n")
            f.write(f"\n# Correlation\n")
            f.write(f"corr_core_v7_vs_carry: {correlation:.4f}\n")
            
            # Sleeve correlations
            if sleeve_corr_summary:
                f.write(f"\n# Sleeve Correlations (vs VX Carry)\n")
                for sleeve_name, corr_val in sleeve_corr_summary.items():
                    f.write(f"corr_{sleeve_name}_vs_carry: {corr_val:.4f}\n")
            
            f.write(f"\npath: {output_dir}\n")
            
            # Pass criteria check (carry-style: Sharpe preservation + at least one improvement)
            sharpe_diff = diff_metrics.get('sharpe_diff', float('-inf'))
            maxdd_diff = diff_metrics.get('maxdd_diff', float('inf'))
            vol_diff = diff_metrics.get('vol_diff', float('inf'))
            
            f.write(f"\n# Phase-2 Pass Criteria (Carry-style)\n")
            f.write(f"sharpe_pass: {'PASS' if sharpe_diff >= -0.01 else 'FAIL'} (combined >= baseline - 0.01)\n")
            f.write(f"maxdd_pass: {'PASS' if maxdd_diff >= 0 else 'FAIL'} (combined >= baseline)\n")
            f.write(f"vol_pass: {'PASS' if vol_diff <= 0 else 'FAIL'} (combined <= baseline)\n")
            
            # Pass if Sharpe preserved (small negative allowed) AND at least one improvement
            passes_sharpe = sharpe_diff >= -0.01
            has_improvement = (maxdd_diff >= 0) or (vol_diff <= 0)
            verdict = "PASS" if (passes_sharpe and has_improvement) else "FAIL"
            f.write(f"\nverdict: {verdict}\n")
        
        logger.info(f"  Registered in: {phase2_file}")
        
        # Print summary
        print("\n" + "=" * 80)
        print(f"VX CALENDAR CARRY PHASE-2 SUMMARY ({args.variant})")
        print("=" * 80)
        print(f"\nBaseline (Core v7):")
        print(f"  Sharpe:  {baseline_metrics.get('Sharpe', float('nan')):8.4f}")
        print(f"  CAGR:   {baseline_metrics.get('CAGR', float('nan')):8.4f} ({baseline_metrics.get('CAGR', 0)*100:6.2f}%)")
        print(f"  MaxDD:  {baseline_metrics.get('MaxDD', float('nan')):8.4f} ({baseline_metrics.get('MaxDD', 0)*100:6.2f}%)")
        print(f"  Vol:    {baseline_metrics.get('Vol', float('nan')):8.4f} ({baseline_metrics.get('Vol', 0)*100:6.2f}%)")
        print(f"\nVX Carry Only (Phase-1):")
        print(f"  Sharpe:  {carry_metrics.get('Sharpe', float('nan')):8.4f}")
        print(f"  CAGR:   {carry_metrics.get('CAGR', float('nan')):8.4f} ({carry_metrics.get('CAGR', 0)*100:6.2f}%)")
        print(f"  MaxDD:  {carry_metrics.get('MaxDD', float('nan')):8.4f} ({carry_metrics.get('MaxDD', 0)*100:6.2f}%)")
        print(f"  Vol:    {carry_metrics.get('Vol', float('nan')):8.4f} ({carry_metrics.get('Vol', 0)*100:6.2f}%)")
        print(f"\nCombined (Core v7 + {args.carry_weight:.1%} VX Carry):")
        print(f"  Sharpe:  {combined_metrics.get('Sharpe', float('nan')):8.4f}")
        print(f"  CAGR:   {combined_metrics.get('CAGR', float('nan')):8.4f} ({combined_metrics.get('CAGR', 0)*100:6.2f}%)")
        print(f"  MaxDD:  {combined_metrics.get('MaxDD', float('nan')):8.4f} ({combined_metrics.get('MaxDD', 0)*100:6.2f}%)")
        print(f"  Vol:    {combined_metrics.get('Vol', float('nan')):8.4f} ({combined_metrics.get('Vol', 0)*100:6.2f}%)")
        print(f"\nDifferences:")
        print(f"  Sharpe:  {diff_metrics.get('sharpe_diff', float('nan')):8.4f}")
        print(f"  CAGR:   {diff_metrics.get('cagr_diff', float('nan')):8.4f} ({diff_metrics.get('cagr_diff', 0)*100:6.2f}%)")
        print(f"  MaxDD:  {diff_metrics.get('maxdd_diff', float('nan')):8.4f} ({diff_metrics.get('maxdd_diff', 0)*100:6.2f}%)")
        print(f"  Vol:    {diff_metrics.get('vol_diff', float('nan')):8.4f} ({diff_metrics.get('vol_diff', 0)*100:6.2f}%)")
        print(f"\nCorrelation (Core v7 vs VX Carry): {correlation:.4f}")
        
        if sleeve_corr_summary:
            print(f"\nSleeve Correlations (vs VX Carry):")
            for sleeve_name, corr_val in sleeve_corr_summary.items():
                print(f"  {sleeve_name}: {corr_val:.4f}")
        
        print(f"\nResults saved to: {output_dir}")
        print("=" * 80)
        
    except Exception as e:
        logger.error(f"Phase-2 diagnostics failed: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()

