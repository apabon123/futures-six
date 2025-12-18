#!/usr/bin/env python3
"""
Crisis Meta-Sleeve Phase-1 Diagnostics Script.

Tests VX-based convexity optimization variants:
1. Long VX2 (benchmark / convexity ceiling)
2. Long VX3 (reduced carry, weaker convexity)
3. Long VX2 - VX1 spread (primary candidate)
4. Long VX3 - VX2 spread (exploratory)

Each variant is tested at 5% weight against Core v9 baseline.
Phase-1 focuses on cost reduction while preserving tail protection.

Phase-1 Evaluation Criteria:
- Tail Preservation: ≥70% of VX2 MaxDD improvement OR match/improve worst-month vs VX2
- Cost Reduction: Improve CAGR vs VX2 OR reduce carry bleed
- Stability: No new left-tail events, no volatility amplification

Usage:
    python scripts/diagnostics/run_crisis_phase1.py
    python scripts/diagnostics/run_crisis_phase1.py --start 2020-01-06 --end 2025-10-31
"""

import sys
import argparse
from pathlib import Path
import logging
from datetime import datetime
from typing import Dict, Optional, Tuple, List
import json
import pandas as pd
import numpy as np
import duckdb

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.canonical_window import load_canonical_window
from src.agents.utils_db import open_readonly_connection, find_ohlcv_table
from src.market_data.vrp_loaders import (
    load_vx_curve, 
    VX_FRONT_SYMBOL, 
    VX_SECOND_SYMBOL,
    VX_THIRD_SYMBOL
)
from run_strategy import main as run_strategy_main
from src.utils.phase_index import update_phase_index

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Canonical evaluation window
CANONICAL_START = "2020-01-06"
CANONICAL_END = "2025-10-31"

# Crisis sleeve weight (fixed)
CRISIS_SLEEVE_WEIGHT = 0.05

# Core v9 profile name
CORE_V9_PROFILE = "core_v9_trend_csmom_vrp_core_convergence_vrp_alt_vx_carry_sr3_curverv_no_macro"


def run_strategy_profile(profile_name: str, run_id: str, start_date: str, end_date: str):
    """Run a strategy profile and save results."""
    logger.info("=" * 80)
    logger.info(f"Running strategy profile: {profile_name}")
    logger.info("=" * 80)
    
    import sys
    original_argv = sys.argv
    try:
        sys.argv = [
            'run_strategy.py',
            '--strategy_profile', profile_name,
            '--run_id', run_id,
            '--start', start_date,
            '--end', end_date
        ]
        run_strategy_main()
    finally:
        sys.argv = original_argv


def load_run_returns(run_id: str) -> pd.Series:
    """
    Load portfolio returns from a backtest run.
    
    Args:
        run_id: Run identifier (e.g., 'core_v9_baseline_phase0_20251217_193451')
    
    Returns:
        Series of daily returns indexed by date
    """
    # Find the run directory
    runs_dir = Path("reports/runs")
    run_dir = None
    for d in runs_dir.iterdir():
        if d.is_dir() and run_id in d.name:
            run_dir = d
            break
    
    if run_dir is None:
        raise FileNotFoundError(f"Run directory not found for {run_id}")
    
    # Sanity check: Confirm run directory exists
    if not run_dir.exists() or not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory is invalid: {run_dir}")
    
    # Load returns CSV
    returns_file = run_dir / "portfolio_returns.csv"
    if not returns_file.exists():
        raise FileNotFoundError(f"Portfolio returns file not found: {returns_file}")
    
    df = pd.read_csv(returns_file, index_col=0, parse_dates=True)
    if 'ret' in df.columns:
        returns = df['ret']
    else:
        returns = df.iloc[:, 0]
    
    # Sanity check: Validate row count matches canonical window expectations
    # Canonical window: 2020-01-06 to 2025-10-31
    # Expected: ~1,800-1,900 trading days (252 trading days/year × ~5.8 years)
    canonical_start = pd.Timestamp("2020-01-06")
    canonical_end = pd.Timestamp("2025-10-31")
    
    # Filter to canonical window
    returns_canonical = returns[(returns.index >= canonical_start) & (returns.index <= canonical_end)]
    
    if len(returns_canonical) < 1500:
        logger.warning(
            f"⚠️  Run {run_id}: Only {len(returns_canonical)} days in canonical window "
            f"(expected ~1800). This may indicate truncated data or missing dates."
        )
    elif len(returns_canonical) > 2000:
        logger.warning(
            f"⚠️  Run {run_id}: {len(returns_canonical)} days in canonical window "
            f"(expected ~1800). This may indicate data outside expected range."
        )
    else:
        logger.debug(f"✓ Run {run_id}: {len(returns_canonical)} days in canonical window (valid)")
    
    returns.name = 'portfolio_return'
    return returns


def load_vx_instruments(
    con: duckdb.DuckDBPyConnection,
    start: str,
    end: str
) -> Dict[str, pd.Series]:
    """
    Load returns for VX instruments: VX1, VX2, VX3.
    
    Returns:
        Dict with keys: 'vx1', 'vx2', 'vx3'
        Each value is a Series of daily log returns indexed by date
    """
    # Load VX curve (includes VX3)
    vx_df = load_vx_curve(con, start, end, VX_FRONT_SYMBOL, VX_SECOND_SYMBOL, VX_THIRD_SYMBOL)
    vx_df = vx_df.set_index('date')
    
    # Compute VX returns
    vx1_returns = np.log(vx_df['vx1']).diff().dropna()
    vx2_returns = np.log(vx_df['vx2']).diff().dropna()
    vx3_returns = None
    if 'vx3' in vx_df.columns:
        vx3_returns = np.log(vx_df['vx3']).diff().dropna()
    
    return {
        'vx1': vx1_returns,
        'vx2': vx2_returns,
        'vx3': vx3_returns
    }


def compute_crisis_sleeve_returns(
    instruments: Dict[str, pd.Series],
    sleeve_type: str
) -> pd.Series:
    """
    Compute crisis sleeve returns for a given sleeve type.
    
    Args:
        instruments: Dict of instrument returns (from load_vx_instruments)
        sleeve_type: 'vx2_long', 'vx3_long', 'vx_spread', or 'vx3_spread'
    
    Returns:
        Series of daily returns for the crisis sleeve
    """
    if sleeve_type == 'vx2_long':
        # Long VX2: constant long position
        return instruments['vx2'].copy()
    
    elif sleeve_type == 'vx3_long':
        # Long VX3: constant long position
        if instruments['vx3'] is None or instruments['vx3'].empty:
            raise ValueError("VX3 data not available")
        return instruments['vx3'].copy()
    
    elif sleeve_type == 'vx_spread':
        # Long VX2 - Short VX1: dollar-neutral spread
        common_dates = instruments['vx2'].index.intersection(instruments['vx1'].index)
        vx2_aligned = instruments['vx2'].loc[common_dates]
        vx1_aligned = instruments['vx1'].loc[common_dates]
        # Spread return = VX2 return - VX1 return
        return (vx2_aligned - vx1_aligned).dropna()
    
    elif sleeve_type == 'vx3_spread':
        # Long VX3 - Short VX2: dollar-neutral spread
        if instruments['vx3'] is None or instruments['vx3'].empty:
            raise ValueError("VX3 data not available")
        common_dates = instruments['vx3'].index.intersection(instruments['vx2'].index)
        vx3_aligned = instruments['vx3'].loc[common_dates]
        vx2_aligned = instruments['vx2'].loc[common_dates]
        # Spread return = VX3 return - VX2 return
        return (vx3_aligned - vx2_aligned).dropna()
    
    else:
        raise ValueError(f"Unknown sleeve type: {sleeve_type}")


def compute_crisis_metrics(returns: pd.Series) -> Dict:
    """
    Compute crisis-specific metrics.
    
    Args:
        returns: Portfolio returns Series
    
    Returns:
        Dict with crisis metrics
    """
    if returns.empty:
        return {}
    
    equity = (1 + returns).cumprod()
    n_days = len(returns)
    n_years = n_days / 252.0
    
    # Basic metrics
    total_ret = equity.iloc[-1] / equity.iloc[0] - 1.0 if len(equity) > 0 else 0.0
    cagr = (1 + total_ret) ** (1.0 / n_years) - 1.0 if n_years > 0 else 0.0
    vol = returns.std() * np.sqrt(252)
    sharpe = (cagr / vol) if vol > 0 else 0.0
    
    # Drawdown
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max
    max_dd = drawdown.min()
    
    # Worst periods
    monthly_returns = returns.resample('ME').apply(lambda x: (1 + x).prod() - 1)
    worst_month = monthly_returns.min()
    
    quarterly_returns = returns.resample('QE').apply(lambda x: (1 + x).prod() - 1)
    worst_quarter = quarterly_returns.min()
    
    # Worst 10-day window
    rolling_10d = returns.rolling(10).apply(lambda x: (1 + x).prod() - 1)
    worst_10d = rolling_10d.min()
    
    # Crisis period attribution
    crisis_periods = {
        '2020_q1': ('2020-01-01', '2020-03-31'),
        '2022_drawdown': ('2022-01-01', '2022-12-31'),
        '2023_2024_vol': ('2023-01-01', '2024-12-31')
    }
    
    crisis_attribution = {}
    for period_name, (start, end) in crisis_periods.items():
        period_returns = returns.loc[(returns.index >= start) & (returns.index <= end)]
        if not period_returns.empty:
            period_equity = (1 + period_returns).cumprod()
            period_total = period_equity.iloc[-1] / period_equity.iloc[0] - 1.0 if len(period_equity) > 0 else 0.0
            crisis_attribution[period_name] = period_total
        else:
            crisis_attribution[period_name] = None
    
    return {
        'cagr': cagr,
        'vol': vol,
        'sharpe': sharpe,
        'max_dd': max_dd,
        'worst_month': worst_month,
        'worst_quarter': worst_quarter,
        'worst_10d': worst_10d,
        'crisis_attribution': crisis_attribution,
        'n_days': n_days,
        'total_return': total_ret
    }


def test_crisis_variant(
    variant_name: str,
    variant_type: str,
    core_v9_returns: pd.Series,
    vx2_benchmark_metrics: Dict,
    instruments: Dict[str, pd.Series],
    start: str,
    end: str,
    output_base: Path
) -> Dict:
    """
    Test a single crisis variant and compute Phase-1 metrics.
    
    Args:
        variant_name: Short name (e.g., 'vx2', 'vx3', 'vx_spread', 'vx3_spread')
        variant_type: Sleeve type for compute_crisis_sleeve_returns
        core_v9_returns: Core v9 baseline returns
        vx2_benchmark_metrics: Metrics from VX2 benchmark (for comparison)
        instruments: VX instrument returns
        start: Start date
        end: End date
        output_base: Base output directory
    
    Returns:
        Dict with test results and pass/fail status
    """
    logger.info(f"\nTesting variant: {variant_name}")
    
    # Compute crisis sleeve returns
    try:
        crisis_returns = compute_crisis_sleeve_returns(instruments, variant_type)
    except Exception as e:
        logger.error(f"Failed to compute {variant_name} returns: {e}")
        return {
            'variant': variant_name,
            'error': str(e),
            'pass_criteria': {'overall_pass': False}
        }
    
    # Align dates with Core v9
    common_dates = core_v9_returns.index.intersection(crisis_returns.index)
    if len(common_dates) == 0:
        logger.error(f"No overlapping dates for {variant_name}")
        return {
            'variant': variant_name,
            'error': 'No overlapping dates',
            'pass_criteria': {'overall_pass': False}
        }
    
    core_aligned = core_v9_returns.loc[common_dates]
    crisis_aligned = crisis_returns.loc[common_dates]
    
    # Combine: Core v9 + Crisis Sleeve (5% weight)
    combined_returns = core_aligned + (CRISIS_SLEEVE_WEIGHT * crisis_aligned)
    
    # Compute metrics
    baseline_metrics = compute_crisis_metrics(core_aligned)
    combined_metrics = compute_crisis_metrics(combined_returns)
    
    # Differences vs baseline
    differences = {
        'maxdd_diff': combined_metrics['max_dd'] - baseline_metrics['max_dd'],
        'worst_month_diff': combined_metrics['worst_month'] - baseline_metrics['worst_month'],
        'worst_quarter_diff': combined_metrics['worst_quarter'] - baseline_metrics['worst_quarter'],
        'worst_10d_diff': combined_metrics['worst_10d'] - baseline_metrics['worst_10d'],
        'cagr_diff': combined_metrics['cagr'] - baseline_metrics['cagr'],
        'vol_diff': combined_metrics['vol'] - baseline_metrics['vol']
    }
    
    # Phase-1 Pass Criteria
    # 1. Tail Preservation: ≥70% of VX2 MaxDD improvement OR match/improve worst-month vs VX2
    vx2_maxdd_improvement = vx2_benchmark_metrics['differences']['maxdd_diff']
    required_maxdd_improvement = 0.70 * vx2_maxdd_improvement if vx2_maxdd_improvement > 0 else 0.0
    
    tail_preservation = (
        differences['maxdd_diff'] >= required_maxdd_improvement
        or differences['worst_month_diff'] >= vx2_benchmark_metrics['differences']['worst_month_diff']
    )
    
    # 2. Cost Reduction: Improve CAGR vs VX2 OR reduce carry bleed
    vx2_cagr_diff = vx2_benchmark_metrics['differences']['cagr_diff']
    cost_reduction = differences['cagr_diff'] > vx2_cagr_diff
    
    # 3. Stability: No new left-tail events
    # Check if worst-month or MaxDD is worse than baseline
    stability = (
        differences['worst_month_diff'] >= 0  # Not worse than baseline
        and differences['maxdd_diff'] >= -0.005  # MaxDD not materially worse (>0.5%)
    )
    
    # Automatic FAIL conditions
    auto_fail = (
        differences['worst_month_diff'] < 0  # Worsens worst-month vs Core v9
        or differences['maxdd_diff'] < -0.005  # Materially worsens MaxDD (>0.5%)
    )
    
    overall_pass = not auto_fail and tail_preservation and (cost_reduction or stability)
    
    pass_criteria = {
        'tail_preservation': tail_preservation,
        'cost_reduction': cost_reduction,
        'stability': stability,
        'auto_fail': auto_fail,
        'overall_pass': overall_pass,
        'required_maxdd_improvement': required_maxdd_improvement,
        'vx2_maxdd_improvement': vx2_maxdd_improvement
    }
    
    # Save results
    output_dir = output_base / f"core_v9_crisis_{variant_name}_phase1"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save metrics
    results = {
        'variant': variant_name,
        'baseline_metrics': baseline_metrics,
        'combined_metrics': combined_metrics,
        'differences': differences,
        'pass_criteria': pass_criteria,
        'start': start,
        'end': end,
        'sleeve_weight': CRISIS_SLEEVE_WEIGHT
    }
    
    with open(output_dir / "metrics.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Save returns
    returns_df = pd.DataFrame({
        'core_v9': core_aligned,
        'crisis_sleeve': crisis_aligned,
        'combined': combined_returns
    })
    returns_df.to_csv(output_dir / "returns.csv")
    
    # Save summary
    summary = f"""Crisis Meta-Sleeve Phase-1: {variant_name}

Evaluation Period: {start} to {end}
Sleeve Weight: {CRISIS_SLEEVE_WEIGHT*100:.1f}%

Baseline (Core v9) Metrics:
  MaxDD: {baseline_metrics['max_dd']:.4f}
  Worst Month: {baseline_metrics['worst_month']:.4f}
  CAGR: {baseline_metrics['cagr']:.4f}

Combined (Core v9 + {variant_name}) Metrics:
  MaxDD: {combined_metrics['max_dd']:.4f} (diff: {differences['maxdd_diff']:+.4f})
  Worst Month: {combined_metrics['worst_month']:.4f} (diff: {differences['worst_month_diff']:+.4f})
  CAGR: {combined_metrics['cagr']:.4f} (diff: {differences['cagr_diff']:+.4f})

Phase-1 Pass Criteria:
  Tail Preservation: {tail_preservation} (required: {required_maxdd_improvement:.4f}, achieved: {differences['maxdd_diff']:.4f})
  Cost Reduction: {cost_reduction} (CAGR diff: {differences['cagr_diff']:+.4f} vs VX2: {vx2_cagr_diff:+.4f})
  Stability: {stability}
  Overall Pass: {overall_pass}
"""
    
    with open(output_dir / "summary.txt", 'w') as f:
        f.write(summary)
    
    logger.info(summary)
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Crisis Meta-Sleeve Phase-1 Diagnostics")
    parser.add_argument('--start', type=str, default=CANONICAL_START,
                       help=f"Start date (default: {CANONICAL_START})")
    parser.add_argument('--end', type=str, default=CANONICAL_END,
                       help=f"End date (default: {CANONICAL_END})")
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("Crisis Meta-Sleeve Phase-1 Diagnostics")
    logger.info("=" * 80)
    logger.info(f"Evaluation Window: {args.start} to {args.end}")
    logger.info(f"Sleeve Weight: {CRISIS_SLEEVE_WEIGHT*100:.1f}%")
    
    # Step 1: Run Core v9 baseline (if not already run)
    logger.info("\nStep 1: Running Core v9 baseline...")
    baseline_run_id = f"core_v9_baseline_phase1_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Run baseline backtest
    run_strategy_profile(CORE_V9_PROFILE, baseline_run_id, args.start, args.end)
    
    # Load Core v9 returns
    core_v9_returns = load_run_returns(baseline_run_id)
    logger.info(f"✓ Loaded Core v9 returns: {len(core_v9_returns)} days")
    
    # Sanity check: Validate baseline returns match canonical window
    canonical_start = pd.Timestamp(args.start)
    canonical_end = pd.Timestamp(args.end)
    baseline_canonical = core_v9_returns[
        (core_v9_returns.index >= canonical_start) & 
        (core_v9_returns.index <= canonical_end)
    ]
    if len(baseline_canonical) < 1500:
        raise ValueError(
            f"Baseline run {baseline_run_id} has insufficient data: "
            f"{len(baseline_canonical)} days in canonical window (expected ~1800)"
        )
    logger.info(f"✓ Baseline canonical window: {len(baseline_canonical)} days ({args.start} to {args.end})")
    
    # Step 2: Load VX instruments
    logger.info("\nLoading VX instruments (VX1, VX2, VX3)...")
    import yaml
    config_path = Path("configs/data.yaml")
    db_path = None
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            data_config = yaml.safe_load(f)
        db_path = data_config.get('db', {}).get('path')
    
    if not db_path:
        raise ValueError("Database path not found in configs/data.yaml")
    
    con = open_readonly_connection(db_path)
    try:
        instruments = load_vx_instruments(con, args.start, args.end)
        logger.info(f"  VX1: {len(instruments['vx1'])} days")
        logger.info(f"  VX2: {len(instruments['vx2'])} days")
        logger.info(f"  VX3: {len(instruments['vx3']) if instruments['vx3'] is not None else 0} days")
    finally:
        con.close()
    
    # Step 3: Test VX2 benchmark first (needed for comparisons)
    logger.info("\n" + "=" * 80)
    logger.info("Testing VX2 Benchmark (for Phase-1 comparisons)")
    logger.info("=" * 80)
    
    output_base = Path("reports/diagnostics/crisis_phase1")
    output_base.mkdir(parents=True, exist_ok=True)
    
    # Compute VX2 benchmark metrics
    vx2_crisis_returns = compute_crisis_sleeve_returns(instruments, 'vx2_long')
    common_dates = core_v9_returns.index.intersection(vx2_crisis_returns.index)
    core_aligned = core_v9_returns.loc[common_dates]
    vx2_aligned = vx2_crisis_returns.loc[common_dates]
    vx2_combined = core_aligned + (CRISIS_SLEEVE_WEIGHT * vx2_aligned)
    
    baseline_metrics = compute_crisis_metrics(core_aligned)
    vx2_combined_metrics = compute_crisis_metrics(vx2_combined)
    
    vx2_benchmark_metrics = {
        'baseline_metrics': baseline_metrics,
        'combined_metrics': vx2_combined_metrics,
        'differences': {
            'maxdd_diff': vx2_combined_metrics['max_dd'] - baseline_metrics['max_dd'],
            'worst_month_diff': vx2_combined_metrics['worst_month'] - baseline_metrics['worst_month'],
            'cagr_diff': vx2_combined_metrics['cagr'] - baseline_metrics['cagr']
        }
    }
    
    logger.info(f"VX2 Benchmark MaxDD improvement: {vx2_benchmark_metrics['differences']['maxdd_diff']:+.4f}")
    logger.info(f"VX2 Benchmark Worst-month improvement: {vx2_benchmark_metrics['differences']['worst_month_diff']:+.4f}")
    
    # Step 4: Test all variants
    variants = [
        ('vx2', 'vx2_long', 'Long VX2 (Benchmark)'),
        ('vx3', 'vx3_long', 'Long VX3'),
        ('vx_spread', 'vx_spread', 'Long VX2 - VX1 Spread'),
        ('vx3_spread', 'vx3_spread', 'Long VX3 - VX2 Spread')
    ]
    
    all_results = {}
    for variant_name, variant_type, description in variants:
        try:
            results = test_crisis_variant(
                variant_name=variant_name,
                variant_type=variant_type,
                core_v9_returns=core_v9_returns,
                vx2_benchmark_metrics=vx2_benchmark_metrics,
                instruments=instruments,
                start=args.start,
                end=args.end,
                output_base=output_base
            )
            all_results[variant_name] = results
            
            # Update phase index
            update_phase_index(
                meta_sleeve="crisis",
                sleeve_name=variant_name,
                phase="phase1",
                run_id=f"core_v9_crisis_{variant_name}_phase1",
            )
            logger.info(f"✓ Registered phase index for {variant_name}")
            
        except Exception as e:
            logger.error(f"Failed to test {variant_name}: {e}")
            import traceback
            traceback.print_exc()
            all_results[variant_name] = {'variant': variant_name, 'error': str(e)}
    
    # Step 5: Generate consolidated summary
    logger.info("\n" + "=" * 80)
    logger.info("Phase-1 Summary")
    logger.info("=" * 80)
    
    summary_data = []
    for variant_name, results in all_results.items():
        if 'error' in results:
            summary_data.append({
                'variant': variant_name,
                'status': 'ERROR',
                'error': results['error']
            })
        else:
            pass_status = "PASS" if results.get('pass_criteria', {}).get('overall_pass', False) else "FAIL"
            differences = results.get('differences', {})
            summary_data.append({
                'variant': variant_name,
                'status': pass_status,
                'maxdd_diff_vs_core': f"{differences.get('maxdd_diff', 0):+.4f}",
                'worst_month_diff_vs_core': f"{differences.get('worst_month_diff', 0):+.4f}",
                'cagr_diff_vs_core': f"{differences.get('cagr_diff', 0):+.4f}",
                'tail_preservation': results.get('pass_criteria', {}).get('tail_preservation', False),
                'cost_reduction': results.get('pass_criteria', {}).get('cost_reduction', False),
                'stability': results.get('pass_criteria', {}).get('stability', False)
            })
            logger.info(f"{variant_name:15s}: {pass_status:4s} | "
                       f"MaxDD diff: {differences.get('maxdd_diff', 0):+.4f} | "
                       f"CAGR diff: {differences.get('cagr_diff', 0):+.4f}")
    
    # Save consolidated summary CSV
    summary_df = pd.DataFrame(summary_data)
    summary_csv = output_base / "crisis_phase1_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    logger.info(f"\n✓ Consolidated summary saved to: {summary_csv}")
    logger.info(f"✓ All results saved to: {output_base}")


if __name__ == "__main__":
    main()

