#!/usr/bin/env python3
"""
Crisis Meta-Sleeve Phase-2 Diagnostics Script.

Validates portfolio integration of Long VX3 (5%) with Core v9.

Phase-2 Objective:
Validate that Core v9 + Crisis (Long VX3, 5%) does not worsen portfolio tail risk,
behaves correctly in known crisis windows, and introduces no new pathologies.

Phase-2 PASS Criteria (Crisis-specific):
- MaxDD ≤ Core v9 baseline
- Worst-month and worst-quarter losses ≤ baseline
- Crisis windows (2020 Q1, 2022) are neutral or improved
- No volatility amplification in calm regimes
- Sharpe and CAGR are reported only (not gating)

Usage:
    python scripts/diagnostics/run_crisis_phase2.py
    python scripts/diagnostics/run_crisis_phase2.py --start 2020-01-06 --end 2025-10-31
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
import yaml

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.canonical_window import load_canonical_window
from src.agents.utils_db import open_readonly_connection
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
CRISIS_SLEEVE_WEIGHT = 0.05  # 5% of portfolio capital

# Core v9 profile name
CORE_V9_PROFILE = "core_v9_trend_csmom_vrp_core_convergence_vrp_alt_vx_carry_sr3_curverv_no_macro"


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
    """
    Load portfolio returns from a backtest run.
    
    Args:
        run_id: Run identifier (e.g., 'core_v9_baseline_phase2_20251217_193451')
    
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
    canonical_start = pd.Timestamp(CANONICAL_START)
    canonical_end = pd.Timestamp(CANONICAL_END)
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


def load_vx3_returns(con, start: str, end: str) -> pd.Series:
    """
    Load VX3 returns for Long VX3 crisis sleeve.
    
    Returns:
        Series of daily log returns indexed by date
    """
    # Load VX curve (includes VX3)
    vx_df = load_vx_curve(con, start, end, VX_FRONT_SYMBOL, VX_SECOND_SYMBOL, VX_THIRD_SYMBOL)
    vx_df = vx_df.set_index('date')
    
    if 'vx3' not in vx_df.columns:
        raise ValueError("VX3 data not available in VX curve")
    
    # Compute VX3 returns
    vx3_returns = np.log(vx_df['vx3']).diff().dropna()
    return vx3_returns


def compute_portfolio_metrics(returns: pd.Series) -> Dict:
    """
    Compute portfolio metrics for Phase-2 evaluation.
    
    Args:
        returns: Portfolio returns Series
    
    Returns:
        Dict with portfolio metrics
    """
    if returns.empty:
        return {}
    
    equity = (1 + returns).cumprod()
    n_days = len(returns)
    n_years = n_days / 252.0
    
    # Basic metrics
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / n_years) - 1.0 if n_years > 0 else 0.0
    vol = returns.std() * np.sqrt(252)
    sharpe = (returns.mean() * 252) / vol if vol > 0 else 0.0
    
    # Drawdown metrics
    running_max = equity.expanding().max()
    drawdown = (equity - running_max) / running_max
    max_drawdown = drawdown.min()
    
    # Worst month and quarter
    monthly_returns = returns.resample('ME').apply(lambda x: (1 + x).prod() - 1)
    worst_month = monthly_returns.min()
    
    quarterly_returns = returns.resample('QE').apply(lambda x: (1 + x).prod() - 1)
    worst_quarter = quarterly_returns.min()
    
    return {
        'cagr': cagr,
        'vol': vol,
        'sharpe': sharpe,
        'max_drawdown': max_drawdown,
        'worst_month': worst_month,
        'worst_quarter': worst_quarter,
        'n_days': n_days,
        'n_years': n_years
    }


def compute_crisis_periods(returns: pd.Series) -> Dict[str, Dict]:
    """
    Compute performance metrics for predefined crisis periods.
    
    Args:
        returns: Portfolio returns Series
    
    Returns:
        Dict with crisis period names as keys and metrics dicts as values
    """
    crisis_periods = {
        "2020_Q1": ("2020-01-01", "2020-03-31"),
        "2022_Drawdown": ("2022-01-01", "2022-12-31"),
    }
    
    results = {}
    for name, (start, end) in crisis_periods.items():
        period_returns = returns[(returns.index >= start) & (returns.index <= end)]
        if len(period_returns) > 0:
            equity = (1 + period_returns).cumprod()
            total_ret = equity.iloc[-1] / equity.iloc[0] - 1.0
            running_max = equity.expanding().max()
            drawdown = (equity - running_max) / running_max
            max_drawdown = drawdown.min()
            vol = period_returns.std() * np.sqrt(252)
            results[name] = {
                'total_return': total_ret,
                'max_drawdown': max_drawdown,
                'vol': vol,
                'n_days': len(period_returns)
            }
        else:
            results[name] = {'n_days': 0}
    
    return results


def check_volatility_amplification(baseline_returns: pd.Series, variant_returns: pd.Series) -> Dict:
    """
    Check for volatility amplification in calm regimes.
    
    Calm regime: periods where baseline volatility < median baseline volatility.
    
    Returns:
        Dict with calm regime analysis
    """
    # Identify calm periods (below median volatility)
    baseline_vol_rolling = baseline_returns.rolling(63).std() * np.sqrt(252)
    median_vol = baseline_vol_rolling.median()
    calm_mask = baseline_vol_rolling < median_vol
    
    if calm_mask.sum() == 0:
        return {'calm_days': 0, 'vol_amplification': False}
    
    baseline_calm_vol = baseline_returns[calm_mask].std() * np.sqrt(252)
    variant_calm_vol = variant_returns[calm_mask].std() * np.sqrt(252)
    
    vol_ratio = variant_calm_vol / baseline_calm_vol if baseline_calm_vol > 0 else 1.0
    
    # Volatility amplification if variant vol > 1.1x baseline vol in calm periods
    amplification = vol_ratio > 1.1
    
    return {
        'calm_days': calm_mask.sum(),
        'baseline_calm_vol': baseline_calm_vol,
        'variant_calm_vol': variant_calm_vol,
        'vol_ratio': vol_ratio,
        'vol_amplification': amplification
    }


def main():
    parser = argparse.ArgumentParser(description="Crisis Meta-Sleeve Phase-2 Diagnostics")
    parser.add_argument('--start', type=str, default=CANONICAL_START,
                       help=f"Start date (default: {CANONICAL_START})")
    parser.add_argument('--end', type=str, default=CANONICAL_END,
                       help=f"End date (default: {CANONICAL_END})")
    parser.add_argument('--baseline-run-id', type=str, default=None,
                       help="Optional: Use existing baseline run instead of running new one")
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("Crisis Meta-Sleeve Phase-2 Diagnostics")
    logger.info("=" * 80)
    logger.info(f"Evaluation Window: {args.start} to {args.end}")
    logger.info(f"Sleeve Weight: {CRISIS_SLEEVE_WEIGHT*100:.1f}%")
    logger.info("")
    
    # Step 1: Run or load Core v9 baseline
    if args.baseline_run_id:
        logger.info(f"Loading Core v9 baseline from run ID: {args.baseline_run_id}")
        baseline_run_id = args.baseline_run_id
        baseline_returns = load_run_returns(baseline_run_id)
    else:
        logger.info("Step 1: Running Core v9 baseline...")
        baseline_run_id = f"core_v9_baseline_phase2_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        run_strategy_profile(CORE_V9_PROFILE, baseline_run_id, args.start, args.end)
        baseline_returns = load_run_returns(baseline_run_id)
    
    logger.info(f"✓ Loaded Core v9 returns: {len(baseline_returns)} days")
    
    # Sanity check: Validate baseline returns match canonical window
    canonical_start = pd.Timestamp(args.start)
    canonical_end = pd.Timestamp(args.end)
    baseline_canonical = baseline_returns[
        (baseline_returns.index >= canonical_start) & 
        (baseline_returns.index <= canonical_end)
    ]
    if len(baseline_canonical) < 1500:
        raise ValueError(
            f"Baseline run {baseline_run_id} has insufficient data: "
            f"{len(baseline_canonical)} days in canonical window (expected ~1800)"
        )
    logger.info(f"✓ Baseline canonical window: {len(baseline_canonical)} days ({args.start} to {args.end})")
    
    # Step 2: Load VX3 returns
    logger.info("\nStep 2: Loading VX3 returns...")
    config_path = Path("configs/data.yaml")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    db_path = config['db']['path']
    
    con = open_readonly_connection(db_path)
    try:
        vx3_returns = load_vx3_returns(con, args.start, args.end)
        logger.info(f"  VX3: {len(vx3_returns)} days")
    finally:
        con.close()
    
    # Step 3: Compute variant returns (Core v9 95% + Long VX3 5%)
    logger.info("\nStep 3: Computing variant returns (Core v9 95% + Long VX3 5%)...")
    
    # Align dates
    common_dates = baseline_returns.index.intersection(vx3_returns.index)
    if len(common_dates) == 0:
        raise ValueError("No overlapping dates between Core v9 and VX3")
    
    baseline_aligned = baseline_returns.loc[common_dates]
    vx3_aligned = vx3_returns.loc[common_dates]
    
    # Combine: Core v9 (95%) + Long VX3 (5%)
    variant_returns = (1 - CRISIS_SLEEVE_WEIGHT) * baseline_aligned + CRISIS_SLEEVE_WEIGHT * vx3_aligned
    
    logger.info(f"✓ Computed variant returns: {len(variant_returns)} days")
    
    # Step 4: Compute metrics
    logger.info("\nStep 4: Computing portfolio metrics...")
    baseline_metrics = compute_portfolio_metrics(baseline_aligned)
    variant_metrics = compute_portfolio_metrics(variant_returns)
    
    # Step 5: Analyze crisis periods
    logger.info("\nStep 5: Analyzing crisis periods...")
    baseline_crisis = compute_crisis_periods(baseline_aligned)
    variant_crisis = compute_crisis_periods(variant_returns)
    
    # Step 6: Check volatility amplification
    logger.info("\nStep 6: Checking volatility amplification in calm regimes...")
    vol_check = check_volatility_amplification(baseline_aligned, variant_returns)
    
    # Step 7: Evaluate Phase-2 pass criteria
    logger.info("\nStep 7: Evaluating Phase-2 pass criteria...")
    
    # Crisis-specific pass criteria
    # For negative metrics (drawdowns), "better" means less negative (closer to zero)
    # So variant >= baseline means variant is better (less negative)
    maxdd_pass = variant_metrics['max_drawdown'] >= baseline_metrics['max_drawdown']
    worst_month_pass = variant_metrics['worst_month'] >= baseline_metrics['worst_month']
    worst_quarter_pass = variant_metrics['worst_quarter'] >= baseline_metrics['worst_quarter']
    
    # Crisis window checks (neutral or improved)
    # For negative metrics, "better" means less negative
    crisis_2020_q1_pass = (
        variant_crisis['2020_Q1']['max_drawdown'] >= baseline_crisis['2020_Q1']['max_drawdown']
        if variant_crisis['2020_Q1'].get('n_days', 0) > 0 else True
    )
    crisis_2022_pass = (
        variant_crisis['2022_Drawdown']['max_drawdown'] >= baseline_crisis['2022_Drawdown']['max_drawdown']
        if variant_crisis['2022_Drawdown'].get('n_days', 0) > 0 else True
    )
    
    # No volatility amplification
    vol_amplification_pass = not vol_check['vol_amplification']
    
    overall_pass = (
        maxdd_pass and
        worst_month_pass and
        worst_quarter_pass and
        crisis_2020_q1_pass and
        crisis_2022_pass and
        vol_amplification_pass
    )
    
    # Step 8: Generate summary and save outputs
    logger.info("\nStep 8: Generating summary and saving outputs...")
    
    output_dir = Path("reports/diagnostics/crisis_phase2")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    summary = f"""
================================================================================
Crisis Meta-Sleeve Phase-2 Diagnostics: Long VX3 Integration

Evaluation Period: {args.start} to {args.end}
Sleeve Weight: {CRISIS_SLEEVE_WEIGHT*100:.1f}%

Baseline (Core v9) Metrics:
  MaxDD: {baseline_metrics['max_drawdown']:.4f}
  Worst Month: {baseline_metrics['worst_month']:.4f}
  Worst Quarter: {baseline_metrics['worst_quarter']:.4f}
  CAGR: {baseline_metrics['cagr']:.4f} (informational)
  Sharpe: {baseline_metrics['sharpe']:.4f} (informational)
  Vol: {baseline_metrics['vol']:.4f}

Variant (Core v9 + Long VX3) Metrics:
  MaxDD: {variant_metrics['max_drawdown']:.4f} (diff: {variant_metrics['max_drawdown'] - baseline_metrics['max_drawdown']:+.4f})
  Worst Month: {variant_metrics['worst_month']:.4f} (diff: {variant_metrics['worst_month'] - baseline_metrics['worst_month']:+.4f})
  Worst Quarter: {variant_metrics['worst_quarter']:.4f} (diff: {variant_metrics['worst_quarter'] - baseline_metrics['worst_quarter']:+.4f})
  CAGR: {variant_metrics['cagr']:.4f} (diff: {variant_metrics['cagr'] - baseline_metrics['cagr']:+.4f}) (informational)
  Sharpe: {variant_metrics['sharpe']:.4f} (diff: {variant_metrics['sharpe'] - baseline_metrics['sharpe']:+.4f}) (informational)
  Vol: {variant_metrics['vol']:.4f} (diff: {variant_metrics['vol'] - baseline_metrics['vol']:+.4f})

Crisis Period Analysis:
  2020 Q1:
    Baseline MaxDD: {baseline_crisis['2020_Q1'].get('max_drawdown', 0):.4f}
    Variant MaxDD: {variant_crisis['2020_Q1'].get('max_drawdown', 0):.4f}
    Status: {'✓ Neutral or Improved' if crisis_2020_q1_pass else '✗ Worsened'}
  
  2022 Drawdown:
    Baseline MaxDD: {baseline_crisis['2022_Drawdown'].get('max_drawdown', 0):.4f}
    Variant MaxDD: {variant_crisis['2022_Drawdown'].get('max_drawdown', 0):.4f}
    Status: {'✓ Neutral or Improved' if crisis_2022_pass else '✗ Worsened'}

Volatility Amplification Check:
  Calm Regime Days: {vol_check['calm_days']}
  Baseline Calm Vol: {vol_check.get('baseline_calm_vol', 0):.4f}
  Variant Calm Vol: {vol_check.get('variant_calm_vol', 0):.4f}
  Vol Ratio: {vol_check.get('vol_ratio', 1.0):.4f}
  Status: {'✓ No Amplification' if vol_amplification_pass else '✗ Amplification Detected'}

Phase-2 Pass Criteria (Crisis-specific):
  MaxDD ≥ Baseline (less negative): {'✓ PASS' if maxdd_pass else '✗ FAIL'} ({variant_metrics['max_drawdown']:.4f} vs {baseline_metrics['max_drawdown']:.4f})
  Worst Month ≥ Baseline (less negative): {'✓ PASS' if worst_month_pass else '✗ FAIL'} ({variant_metrics['worst_month']:.4f} vs {baseline_metrics['worst_month']:.4f})
  Worst Quarter ≥ Baseline (less negative): {'✓ PASS' if worst_quarter_pass else '✗ FAIL'} ({variant_metrics['worst_quarter']:.4f} vs {baseline_metrics['worst_quarter']:.4f})
  2020 Q1 Neutral/Improved: {'✓ PASS' if crisis_2020_q1_pass else '✗ FAIL'}
  2022 Neutral/Improved: {'✓ PASS' if crisis_2022_pass else '✗ FAIL'}
  No Vol Amplification: {'✓ PASS' if vol_amplification_pass else '✗ FAIL'}
  
  Overall Pass: {'✓ PASS' if overall_pass else '✗ FAIL'}
================================================================================
"""
    
    with open(output_dir / "summary.txt", 'w', encoding='utf-8') as f:
        f.write(summary)
    
    logger.info(summary)
    
    # Helper to convert numpy types to native Python types for JSON serialization
    def to_json_serializable(obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [to_json_serializable(item) for item in obj]
        else:
            return obj
    
    # Save comparison JSON
    comparison = {
        "baseline": {
            "profile": CORE_V9_PROFILE,
            "run_id": baseline_run_id,
            "metrics": to_json_serializable(baseline_metrics),
            "crisis_periods": to_json_serializable(baseline_crisis)
        },
        "variant": {
            "sleeve": "Long VX3",
            "weight": CRISIS_SLEEVE_WEIGHT,
            "metrics": to_json_serializable(variant_metrics),
            "crisis_periods": to_json_serializable(variant_crisis)
        },
        "comparison": {
            "maxdd_diff": float(variant_metrics['max_drawdown'] - baseline_metrics['max_drawdown']),
            "worst_month_diff": float(variant_metrics['worst_month'] - baseline_metrics['worst_month']),
            "worst_quarter_diff": float(variant_metrics['worst_quarter'] - baseline_metrics['worst_quarter']),
            "cagr_diff": float(variant_metrics['cagr'] - baseline_metrics['cagr']),
            "sharpe_diff": float(variant_metrics['sharpe'] - baseline_metrics['sharpe']),
            "vol_diff": float(variant_metrics['vol'] - baseline_metrics['vol']),
            "vol_amplification": to_json_serializable(vol_check),
            "pass_criteria": {
                "maxdd_pass": bool(maxdd_pass),
                "worst_month_pass": bool(worst_month_pass),
                "worst_quarter_pass": bool(worst_quarter_pass),
                "crisis_2020_q1_pass": bool(crisis_2020_q1_pass),
                "crisis_2022_pass": bool(crisis_2022_pass),
                "vol_amplification_pass": bool(vol_amplification_pass),
                "overall_pass": bool(overall_pass)
            }
        },
        "date_range": {
            "start": args.start,
            "end": args.end,
            "n_days": int(len(variant_returns))
        }
    }
    
    with open(output_dir / "comparison.json", 'w') as f:
        json.dump(comparison, f, indent=2)
    
    # Save returns
    returns_df = pd.DataFrame({
        "baseline": baseline_aligned,
        "variant": variant_returns
    })
    returns_df.to_csv(output_dir / "returns.csv")
    
    # Save equity curves
    equity_df = pd.DataFrame({
        "baseline": (1 + baseline_aligned).cumprod(),
        "variant": (1 + variant_returns).cumprod()
    })
    equity_df.to_csv(output_dir / "equity.csv")
    
    # Update phase index
    update_phase_index(
        meta_sleeve="crisis",
        sleeve_name="vx3",
        phase="phase2",
        run_id=f"core_v9_crisis_vx3_phase2",
    )
    logger.info(f"✓ Registered phase index for vx3")
    
    logger.info(f"\n✓ All results saved to: {output_dir}")
    
    if overall_pass:
        logger.info("\n✅ Phase-2 PASS: Long VX3 integration validated")
    else:
        logger.info("\n❌ Phase-2 FAIL: Long VX3 integration did not meet criteria")
    
    return overall_pass


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

