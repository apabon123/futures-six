#!/usr/bin/env python3
"""
Phase-2 Diagnostics Script for SR3 Curve RV Pack Slope Momentum Integration.

Compares baseline (Core v8) vs Pack Slope Momentum-enhanced portfolio (Core v9).

This script:
1. Runs baseline strategy (Core v8) via run_strategy.py
2. Runs variant strategy (Core v9 + Pack Slope) via run_strategy.py
3. Compares portfolio metrics (Sharpe, CAGR, Vol, MaxDD, HitRate)
4. Analyzes crisis-period performance
5. Computes correlation between baseline and variant
6. Saves comparison outputs and registers in phase index

Usage:
    python scripts/diagnostics/run_sr3_curve_rv_pack_slope_momentum_phase2.py
    python scripts/diagnostics/run_sr3_curve_rv_pack_slope_momentum_phase2.py --start 2020-01-01 --end 2025-10-31 --core-v8-run-id <id>
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

from src.utils.canonical_window import load_canonical_window
from run_strategy import main as run_strategy_main
from src.utils.phase_index import update_phase_index

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Strategy profiles
BASELINE_PROFILE = "core_v8_trend_csmom_vrp_core_convergence_vrp_alt_vx_carry_no_macro"
VARIANT_PROFILE = "core_v9_trend_csmom_vrp_core_convergence_vrp_alt_vx_carry_sr3_curverv_packslope_no_macro"


def run_strategy_profile(profile_name: str, run_id: str, start_date: str, end_date: str):
    """
    Run a strategy profile and save results.
    
    Args:
        profile_name: Strategy profile name from configs/strategies.yaml
        run_id: Run identifier for saving artifacts
        start_date: Start date for backtest
        end_date: End date for backtest
    """
    logger.info("=" * 80)
    logger.info(f"Running strategy profile: {profile_name}")
    logger.info(f"Run ID: {run_id}")
    logger.info(f"Date Range: {start_date} to {end_date}")
    logger.info("=" * 80)
    
    # Build command-line arguments for run_strategy.py
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
        # Find the run directory
        run_dir = Path(f"reports/runs/{run_id}")
        if not run_dir.exists():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")
    
    # Load portfolio returns
    returns_file = run_dir / "portfolio_returns.csv"
    if not returns_file.exists():
        raise FileNotFoundError(f"Portfolio returns file not found: {returns_file}")
    
    df = pd.read_csv(returns_file, index_col=0, parse_dates=True)
    if 'ret' in df.columns:
        returns = df['ret']
    else:
        returns = df.iloc[:, 0]
    
    returns.name = 'portfolio_return'
    return returns


def compute_crisis_periods(returns: pd.Series) -> Dict[str, Dict]:
    """
    Compute performance metrics for predefined crisis periods.
    
    Args:
        returns: Portfolio returns Series
        
    Returns:
        Dict with crisis period names as keys and metrics dicts as values
    """
    crisis_periods = {
        "COVID-19": ("2020-02-01", "2020-04-30"),
        "2022 Rate Hikes": ("2022-01-01", "2022-12-31"),
        "2023 Banking Crisis": ("2023-03-01", "2023-05-31"),
    }
    
    results = {}
    for name, (start, end) in crisis_periods.items():
        period_returns = returns[(returns.index >= start) & (returns.index <= end)]
        if len(period_returns) > 0:
            equity = (1 + period_returns).cumprod()
            n_days = len(period_returns)
            n_years = n_days / 252.0
            total_ret = equity.iloc[-1] / equity.iloc[0] - 1.0
            cagr = (1 + total_ret) ** (1 / n_years) - 1 if n_years > 0 else 0.0
            vol = period_returns.std() * np.sqrt(252)
            sharpe = (period_returns.mean() * 252) / vol if vol > 0 else 0.0
            running_max = equity.expanding().max()
            drawdown = (equity - running_max) / running_max
            max_drawdown = drawdown.min()
            hit_rate = (period_returns > 0).mean()
            results[name] = {
                'cagr': cagr,
                'vol': vol,
                'sharpe': sharpe,
                'max_drawdown': max_drawdown,
                'hit_rate': hit_rate,
                'n_days': n_days
            }
        else:
            results[name] = {"n_days": 0}
    
    return results


def main():
    # Load canonical window as default
    CANONICAL_START, CANONICAL_END = load_canonical_window()
    
    parser = argparse.ArgumentParser(description="SR3 Curve RV Pack Slope Momentum Phase-2 Diagnostics")
    parser.add_argument("--start", type=str, default=CANONICAL_START, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default=CANONICAL_END, help="End date (YYYY-MM-DD)")
    parser.add_argument("--core-v8-run-id", type=str, default=None, 
                       help="Existing Core v8 run ID (if not provided, will run baseline)")
    parser.add_argument("--variant-run-id", type=str, default=None,
                       help="Existing variant run ID (if not provided, will run variant)")
    
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("SR3 Curve RV Pack Slope Momentum Phase-2 Diagnostics")
    logger.info("=" * 80)
    logger.info(f"Date Range: {args.start} to {args.end}")
    logger.info(f"Baseline Profile: {BASELINE_PROFILE}")
    logger.info(f"Variant Profile: {VARIANT_PROFILE}")
    
    # Step 1: Run or load baseline
    if args.core_v8_run_id:
        logger.info(f"Loading baseline from run ID: {args.core_v8_run_id}")
        baseline_returns = load_run_returns(args.core_v8_run_id)
        baseline_run_id = args.core_v8_run_id
    else:
        logger.info("Running baseline strategy...")
        baseline_run_id = f"core_v8_phase2_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        run_strategy_profile(BASELINE_PROFILE, baseline_run_id, args.start, args.end)
        baseline_returns = load_run_returns(baseline_run_id)
    
    # Step 2: Run or load variant
    if args.variant_run_id:
        logger.info(f"Loading variant from run ID: {args.variant_run_id}")
        variant_returns = load_run_returns(args.variant_run_id)
        variant_run_id = args.variant_run_id
    else:
        logger.info("Running variant strategy...")
        variant_run_id = f"core_v9_packslope_phase2_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        run_strategy_profile(VARIANT_PROFILE, variant_run_id, args.start, args.end)
        variant_returns = load_run_returns(variant_run_id)
    
    # Step 3: Align returns and enforce canonical window
    common_dates = baseline_returns.index.intersection(variant_returns.index)
    baseline_aligned = baseline_returns.loc[common_dates]
    variant_aligned = variant_returns.loc[common_dates]
    
    # Apply canonical window filter
    CANONICAL_START, CANONICAL_END = load_canonical_window()
    baseline_aligned = baseline_aligned.loc[CANONICAL_START:CANONICAL_END]
    variant_aligned = variant_aligned.loc[CANONICAL_START:CANONICAL_END]
    
    # Hard assertion: both must have same start and end dates
    assert baseline_aligned.index[0] == variant_aligned.index[0], \
        f"Start date mismatch: baseline={baseline_aligned.index[0]}, variant={variant_aligned.index[0]}"
    assert baseline_aligned.index[-1] == variant_aligned.index[-1], \
        f"End date mismatch: baseline={baseline_aligned.index[-1]}, variant={variant_aligned.index[-1]}"
    
    logger.info(f"Aligned returns (canonical window): {len(baseline_aligned)} days")
    logger.info(f"  Baseline: {baseline_aligned.index[0]} to {baseline_aligned.index[-1]}")
    logger.info(f"  Variant: {variant_aligned.index[0]} to {variant_aligned.index[-1]}")
    
    # Step 4: Compute metrics
    baseline_equity = (1 + baseline_aligned).cumprod()
    variant_equity = (1 + variant_aligned).cumprod()
    
    # Compute metrics manually (simpler than compute_summary_stats which requires asset-level returns)
    def compute_metrics(returns: pd.Series, equity: pd.Series) -> Dict:
        """Compute basic portfolio metrics."""
        n_days = len(returns)
        n_years = n_days / 252.0
        total_ret = equity.iloc[-1] / equity.iloc[0] - 1.0
        cagr = (1 + total_ret) ** (1 / n_years) - 1 if n_years > 0 else 0.0
        vol = returns.std() * np.sqrt(252)
        sharpe = (returns.mean() * 252) / vol if vol > 0 else 0.0
        running_max = equity.expanding().max()
        drawdown = (equity - running_max) / running_max
        max_drawdown = drawdown.min()
        hit_rate = (returns > 0).mean()
        return {
            'cagr': cagr,
            'vol': vol,
            'sharpe': sharpe,
            'max_drawdown': max_drawdown,
            'hit_rate': hit_rate,
            'n_days': n_days
        }
    
    baseline_stats = compute_metrics(baseline_aligned, baseline_equity)
    variant_stats = compute_metrics(variant_aligned, variant_equity)
    
    # Step 5: Crisis period analysis
    baseline_crisis = compute_crisis_periods(baseline_aligned)
    variant_crisis = compute_crisis_periods(variant_aligned)
    
    # Step 6: Correlation
    correlation = baseline_aligned.corr(variant_aligned)
    
    # Step 7: Pass criteria
    sharpe_pass = variant_stats['sharpe'] >= baseline_stats['sharpe'] - 0.01
    maxdd_pass = variant_stats['max_drawdown'] >= baseline_stats['max_drawdown']
    
    logger.info("=" * 80)
    logger.info("Phase-2 Results")
    logger.info("=" * 80)
    logger.info(f"\nBaseline Metrics:")
    logger.info(f"  Sharpe: {baseline_stats['sharpe']:.4f}")
    logger.info(f"  CAGR: {baseline_stats['cagr']:.2%}")
    logger.info(f"  Vol: {baseline_stats['vol']:.2%}")
    logger.info(f"  MaxDD: {baseline_stats['max_drawdown']:.2%}")
    logger.info(f"  HitRate: {baseline_stats['hit_rate']:.2%}")
    
    logger.info(f"\nVariant Metrics:")
    logger.info(f"  Sharpe: {variant_stats['sharpe']:.4f} ({'+' if variant_stats['sharpe'] >= baseline_stats['sharpe'] else ''}{variant_stats['sharpe'] - baseline_stats['sharpe']:.4f})")
    logger.info(f"  CAGR: {variant_stats['cagr']:.2%} ({'+' if variant_stats['cagr'] >= baseline_stats['cagr'] else ''}{variant_stats['cagr'] - baseline_stats['cagr']:.2%})")
    logger.info(f"  Vol: {variant_stats['vol']:.2%} ({'+' if variant_stats['vol'] >= baseline_stats['vol'] else ''}{variant_stats['vol'] - baseline_stats['vol']:.2%})")
    logger.info(f"  MaxDD: {variant_stats['max_drawdown']:.2%} ({'+' if variant_stats['max_drawdown'] >= baseline_stats['max_drawdown'] else ''}{variant_stats['max_drawdown'] - baseline_stats['max_drawdown']:.2%})")
    logger.info(f"  HitRate: {variant_stats['hit_rate']:.2%}")
    
    logger.info(f"\nCorrelation: {correlation:.4f}")
    
    logger.info(f"\nPass Criteria:")
    logger.info(f"  Sharpe >= baseline - 0.01: {'[PASS]' if sharpe_pass else '[FAIL]'}")
    logger.info(f"  MaxDD >= baseline: {'[PASS]' if maxdd_pass else '[FAIL]'}")
    
    overall_pass = sharpe_pass and maxdd_pass
    logger.info(f"\nOverall Phase-2 Result: {'[PASS]' if overall_pass else '[FAIL]'}")
    
    # Step 8: Save results
    output_dir = Path("reports/phase_index/rates_curve_rv/sr3_curve_rv_pack_slope_momentum/phase2")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save comparison JSON
    comparison = {
        "baseline": {
            "profile": BASELINE_PROFILE,
            "run_id": baseline_run_id,
            "stats": baseline_stats,
            "crisis_periods": baseline_crisis
        },
        "variant": {
            "profile": VARIANT_PROFILE,
            "run_id": variant_run_id,
            "stats": variant_stats,
            "crisis_periods": variant_crisis
        },
        "comparison": {
            "correlation": float(correlation),
            "sharpe_diff": float(variant_stats['sharpe'] - baseline_stats['sharpe']),
            "cagr_diff": float(variant_stats['cagr'] - baseline_stats['cagr']),
            "maxdd_diff": float(variant_stats['max_drawdown'] - baseline_stats['max_drawdown']),
            "pass_criteria": {
                "sharpe_pass": bool(sharpe_pass),
                "maxdd_pass": bool(maxdd_pass),
                "overall_pass": bool(overall_pass)
            }
        },
        "date_range": {
            "requested_start": args.start,
            "requested_end": args.end,
            "canonical_start": CANONICAL_START,
            "canonical_end": CANONICAL_END,
            "effective_start": str(baseline_aligned.index[0].date()),
            "effective_end": str(baseline_aligned.index[-1].date()),
            "n_days": len(baseline_aligned)
        },
        "evaluation_context": "diagnostic_comparison",
        "canonical_window_enforced": True
    }
    
    with open(output_dir / "comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)
    
    # Save returns
    returns_df = pd.DataFrame({
        "baseline": baseline_aligned,
        "variant": variant_aligned
    })
    returns_df.to_csv(output_dir / "returns.csv")
    
    # Save equity curves
    equity_df = pd.DataFrame({
        "baseline": baseline_equity,
        "variant": variant_equity
    })
    equity_df.to_csv(output_dir / "equity.csv")
    
    # Update phase index
    update_phase_index(
        meta_sleeve="rates_curve_rv",
        sleeve_name="sr3_curve_rv_pack_slope_momentum",
        phase="phase2",
        run_id=variant_run_id
    )
    
    logger.info(f"\nResults saved to: {output_dir}")
    logger.info("=" * 80)
    
    return comparison


if __name__ == "__main__":
    main()
