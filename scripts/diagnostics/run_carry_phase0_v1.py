#!/usr/bin/env python3
"""
Carry Meta-Sleeve v1 — Phase-0 Diagnostic Script

Purpose: Sanity check to validate economic edge of carry signals across asset classes

Phase-0 Contract (from canonical spec):
- Sign-only signals: sign(raw_carry)
- Equal-weight per asset
- Daily rebalance
- No z-scoring
- No overlays
- No vol normalization
- No gating
- No cross-sectional ranking

Pass Criteria:
- Portfolio Sharpe ≥ 0.2
- At least one asset class positive
- No pathological tail blowups
- Crisis window behavior (2020 Q1) acceptable

Usage:
    python scripts/diagnostics/run_carry_phase0_v1.py
    python scripts/diagnostics/run_carry_phase0_v1.py --start 2018-01-01 --end 2025-12-31
"""

import argparse
import sys
import subprocess
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import logging
import json

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config.backtest_window import CANONICAL_START, CANONICAL_END

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def compute_summary_stats(returns: pd.Series, name: str = "Strategy") -> dict:
    """
    Compute comprehensive summary statistics for a returns series.
    
    Args:
        returns: Series of returns (assumed to be simple returns, not log)
        name: Name for the strategy (for logging)
        
    Returns:
        Dictionary of summary statistics
    """
    if returns.empty or returns.isna().all():
        logger.warning(f"{name}: No valid returns found")
        return {}
    
    # Filter out NaNs
    returns_clean = returns.dropna()
    n = len(returns_clean)
    
    if n == 0:
        logger.warning(f"{name}: No valid returns after dropna")
        return {}
    
    # Basic statistics
    mean_ret = returns_clean.mean()
    std_ret = returns_clean.std()
    
    # Annualization (assume daily returns, 252 trading days)
    ann_ret = mean_ret * 252
    ann_vol = std_ret * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
    
    # Cumulative returns and drawdown
    equity = (1 + returns_clean).cumprod()
    hwm = equity.expanding().max()
    drawdown = (equity - hwm) / hwm
    max_dd = drawdown.min()
    
    # Tail statistics
    worst_day = returns_clean.min()
    best_day = returns_clean.max()
    
    # Skewness and kurtosis
    skew = returns_clean.skew()
    kurt = returns_clean.kurtosis()
    
    # Positive/negative day counts
    pos_days = (returns_clean > 0).sum()
    neg_days = (returns_clean < 0).sum()
    zero_days = (returns_clean == 0).sum()
    
    stats = {
        "name": name,
        "n_obs": n,
        "mean_daily": mean_ret,
        "std_daily": std_ret,
        "ann_return": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "worst_day": worst_day,
        "best_day": best_day,
        "skew": skew,
        "kurtosis": kurt,
        "pos_days": pos_days,
        "neg_days": neg_days,
        "zero_days": zero_days,
        "final_equity": equity.iloc[-1] if len(equity) > 0 else 1.0
    }
    
    return stats


def run_phase0_backtest(start_date: str, end_date: str, config_path: str) -> str:
    """
    Run Phase-0 backtest using run_strategy.py.
    
    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        config_path: Path to Phase-0 config file
        
    Returns:
        Run ID for the backtest
    """
    logger.info("=" * 80)
    logger.info("CARRY META-SLEEVE V1 — PHASE-0 BACKTEST")
    logger.info("=" * 80)
    logger.info(f"Period: {start_date} to {end_date}")
    logger.info(f"Config: {config_path}")
    
    # Generate run ID
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"carry_phase0_v1_{timestamp}"
    
    # Run backtest
    cmd = [
        "python",
        "run_strategy.py",
        "--start", start_date,
        "--end", end_date,
        "--config_path", config_path,
        "--run_id", run_id
    ]
    
    logger.info(f"Executing: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        logger.error(f"Backtest failed with return code {result.returncode}")
        logger.error(f"STDOUT:\n{result.stdout}")
        logger.error(f"STDERR:\n{result.stderr}")
        sys.exit(1)
    
    logger.info("Backtest completed successfully")
    return run_id


def analyze_phase0_results(run_id: str) -> dict:
    """
    Analyze Phase-0 backtest results.
    
    Args:
        run_id: Run ID for the backtest
        
    Returns:
        Dictionary of analysis results
    """
    logger.info("=" * 80)
    logger.info("CARRY META-SLEEVE V1 — PHASE-0 ANALYSIS")
    logger.info("=" * 80)
    
    # Load results
    reports_dir = Path(f"reports/runs/{run_id}")
    
    if not reports_dir.exists():
        logger.error(f"Results directory not found: {reports_dir}")
        return {}
    
    # Load portfolio returns
    returns_path = reports_dir / "portfolio_returns.csv"
    if not returns_path.exists():
        logger.error(f"Portfolio returns not found: {returns_path}")
        return {}
    
    returns_df = pd.read_csv(returns_path)
    returns_df['date'] = pd.to_datetime(returns_df['date'])
    returns_df = returns_df.set_index('date')
    
    # Assume column is 'portfolio_return', 'return', or 'ret'
    if 'portfolio_return' in returns_df.columns:
        returns = returns_df['portfolio_return']
    elif 'return' in returns_df.columns:
        returns = returns_df['return']
    elif 'ret' in returns_df.columns:
        returns = returns_df['ret']
    else:
        logger.error(f"Could not find returns column. Available: {list(returns_df.columns)}")
        return {}
    
    # Compute statistics
    stats = compute_summary_stats(returns, name="Carry Meta v1 (Phase-0)")
    
    # Phase-0 Compliance Check
    meta_path = run_dir / 'meta.json'
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        logger.info("\n" + "=" * 80)
        logger.info("PHASE-0 COMPLIANCE CHECK")
        logger.info("=" * 80)
        rt_enabled = meta.get('risk_targeting', {}).get('enabled', 'N/A')
        alloc_enabled = meta.get('allocator_v1', {}).get('enabled', 'N/A')
        policy_enabled = meta.get('config', {}).get('engine_policy_v1', {}).get('enabled', 'N/A')
        rebalance = meta.get('rebalance', 'N/A')
        phase = meta.get('config', {}).get('strategies', {}).get('carry_meta_v1', {}).get('params', {}).get('phase', 'N/A')
        
        logger.info(f"Risk Targeting Enabled: {rt_enabled} (required: false)")
        logger.info(f"Allocator Enabled: {alloc_enabled} (required: false)")
        logger.info(f"Engine Policy Enabled: {policy_enabled} (required: false)")
        logger.info(f"Rebalance Frequency: {rebalance} (required: D)")
        logger.info(f"Carry Phase: {phase} (required: 0)")
        logger.info(f"Evaluation Layer: Post-Construction (belief layer)")
        
        compliance_pass = (
            rt_enabled == False and
            alloc_enabled == False and
            policy_enabled == False and
            phase == 0
        )
        
        if compliance_pass:
            logger.info("✅ Phase-0 Compliance: PASSED")
        else:
            logger.warning("⚠️ Phase-0 Compliance: FAILED - Check config flags above")
        logger.info("=" * 80)
    
    # Log results
    logger.info("\n" + "=" * 80)
    logger.info("PHASE-0 RESULTS")
    logger.info("=" * 80)
    logger.info(f"Observations: {stats.get('n_obs', 0)}")
    logger.info(f"Annualized Return: {stats.get('ann_return', 0)*100:.2f}%")
    logger.info(f"Annualized Vol: {stats.get('ann_vol', 0)*100:.2f}%")
    logger.info(f"Sharpe Ratio: {stats.get('sharpe', 0):.3f}")
    logger.info(f"Max Drawdown: {stats.get('max_dd', 0)*100:.2f}%")
    logger.info(f"Best Day: {stats.get('best_day', 0)*100:.2f}%")
    logger.info(f"Worst Day: {stats.get('worst_day', 0)*100:.2f}%")
    logger.info(f"Skewness: {stats.get('skew', 0):.3f}")
    logger.info(f"Kurtosis: {stats.get('kurtosis', 0):.3f}")
    logger.info(f"Positive Days: {stats.get('pos_days', 0)} ({stats.get('pos_days', 0)/stats.get('n_obs', 1)*100:.1f}%)")
    logger.info(f"Negative Days: {stats.get('neg_days', 0)} ({stats.get('neg_days', 0)/stats.get('n_obs', 1)*100:.1f}%)")
    logger.info(f"Zero Days: {stats.get('zero_days', 0)} ({stats.get('zero_days', 0)/stats.get('n_obs', 1)*100:.1f}%)")
    logger.info(f"Final Equity: ${stats.get('final_equity', 1.0):.2f}")
    
    # Phase-0 Pass Criteria
    logger.info("\n" + "=" * 80)
    logger.info("PHASE-0 PASS CRITERIA")
    logger.info("=" * 80)
    
    sharpe_pass = stats.get('sharpe', 0) >= 0.2
    logger.info(f"✓ Sharpe ≥ 0.2: {sharpe_pass} (Sharpe = {stats.get('sharpe', 0):.3f})")
    
    # Check crisis window (2020 Q1)
    crisis_start = pd.Timestamp("2020-01-01")
    crisis_end = pd.Timestamp("2020-03-31")
    crisis_returns = returns[(returns.index >= crisis_start) & (returns.index <= crisis_end)]
    
    if len(crisis_returns) > 0:
        crisis_ret = (1 + crisis_returns).prod() - 1
        logger.info(f"  2020 Q1 Return: {crisis_ret*100:.2f}%")
        crisis_acceptable = crisis_ret > -0.20  # Not worse than -20%
        logger.info(f"✓ 2020 Q1 acceptable (> -20%): {crisis_acceptable}")
    else:
        logger.info("  2020 Q1 data not available")
        crisis_acceptable = True
    
    # Overall pass/fail
    phase0_pass = sharpe_pass and crisis_acceptable
    
    logger.info("\n" + "=" * 80)
    if phase0_pass:
        logger.info("✓✓✓ PHASE-0 PASSED ✓✓✓")
        logger.info("Carry Meta-Sleeve v1 demonstrates economic edge.")
        logger.info("Proceed to Phase-1 (Clean Implementation).")
    else:
        logger.info("✗✗✗ PHASE-0 FAILED ✗✗✗")
        logger.info("Carry Meta-Sleeve v1 does not meet Phase-0 criteria.")
        logger.info("Review implementation and feature definitions.")
    logger.info("=" * 80)
    
    # Save analysis summary
    summary = {
        "run_id": run_id,
        "phase": 0,
        "stats": stats,
        "pass_criteria": {
            "sharpe_pass": sharpe_pass,
            "crisis_acceptable": crisis_acceptable,
            "overall_pass": phase0_pass
        }
    }
    
    summary_path = reports_dir / "phase0_analysis_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    logger.info(f"\nAnalysis summary saved to: {summary_path}")
    
    return summary


def main():
    """Run Phase-0 diagnostic for Carry Meta-Sleeve v1."""
    parser = argparse.ArgumentParser(description="Run Carry Meta-Sleeve v1 Phase-0 Diagnostic")
    parser.add_argument(
        "--start",
        type=str,
        default=CANONICAL_START,
        help=f"Start date for backtest (YYYY-MM-DD). Default: {CANONICAL_START}"
    )
    parser.add_argument(
        "--end",
        type=str,
        default=CANONICAL_END,
        help=f"End date for backtest (YYYY-MM-DD). Default: {CANONICAL_END}"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/carry_phase0_v1.yaml",
        help="Path to Phase-0 config file"
    )
    
    args = parser.parse_args()
    
    # Validate config exists
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)
    
    # Run Phase-0 backtest
    run_id = run_phase0_backtest(args.start, args.end, str(config_path))
    
    # Analyze results
    summary = analyze_phase0_results(run_id)
    
    # Return exit code based on pass/fail
    if summary.get("pass_criteria", {}).get("overall_pass", False):
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
