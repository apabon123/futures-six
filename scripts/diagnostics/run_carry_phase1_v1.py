#!/usr/bin/env python3
"""
Carry Meta-Sleeve v1 — Phase-1 Diagnostic Script

Purpose: Evaluate clean implementation with z-scoring, vol normalization, clipping

Phase-1 Contract:
- Z-scored signals (252d rolling window)
- Vol-normalized (equal risk per asset)
- Clipped at ±3.0
- Daily rebalance
- No overlays, RT, allocator, policy

Pass Criteria:
- Sharpe ≥ 0.25 (recommended) or 0.20-0.25 (conditional)
- MaxDD improves or does not worsen materially
- No single asset dominates risk
- Crisis behavior remains sane

Usage:
    python scripts/diagnostics/run_carry_phase1_v1.py
    python scripts/diagnostics/run_carry_phase1_v1.py --start 2020-01-01 --end 2025-10-31
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
    """Compute comprehensive summary statistics."""
    if returns.empty or returns.isna().all():
        logger.warning(f"{name}: No valid returns found")
        return {}
    
    returns_clean = returns.dropna()
    n = len(returns_clean)
    
    if n == 0:
        return {}
    
    mean_ret = returns_clean.mean()
    std_ret = returns_clean.std()
    ann_ret = mean_ret * 252
    ann_vol = std_ret * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
    
    equity = (1 + returns_clean).cumprod()
    hwm = equity.expanding().max()
    drawdown = (equity - hwm) / hwm
    max_dd = drawdown.min()
    
    worst_day = returns_clean.min()
    best_day = returns_clean.max()
    skew = returns_clean.skew()
    kurt = returns_clean.kurtosis()
    
    pos_days = (returns_clean > 0).sum()
    neg_days = (returns_clean < 0).sum()
    zero_days = (returns_clean == 0).sum()
    
    final_equity = equity.iloc[-1] if len(equity) > 0 else 1.0
    
    return {
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
        "pos_days": str(pos_days),
        "neg_days": str(neg_days),
        "zero_days": str(zero_days),
        "final_equity": final_equity
    }


def analyze_phase1_results(run_id: str) -> dict:
    """Analyze Phase-1 backtest results with asset-class contributions and dominance diagnostics."""
    run_dir = Path("reports/runs") / run_id
    
    if not run_dir.exists():
        logger.error(f"Run directory not found: {run_dir}")
        return {}
    
    # Load returns
    returns_path = run_dir / "portfolio_ret.csv"
    if not returns_path.exists():
        returns_path = run_dir / "portfolio_returns.csv"
    
    if not returns_path.exists():
        logger.error(f"Returns file not found in {run_dir}")
        return {}
    
    returns_df = pd.read_csv(returns_path, index_col=0, parse_dates=True)
    
    # Find returns column
    if 'portfolio_return' in returns_df.columns:
        returns = returns_df['portfolio_return']
    elif 'return' in returns_df.columns:
        returns = returns_df['return']
    elif 'ret' in returns_df.columns:
        returns = returns_df['ret']
    else:
        logger.error(f"Could not find returns column. Available: {list(returns_df.columns)}")
        return {}
    
    # Load weights for asset-class analysis
    weights_path = run_dir / "weights.csv"
    weights_df = None
    if weights_path.exists():
        weights_df = pd.read_csv(weights_path, index_col=0, parse_dates=True)
    
    # Load asset returns for contribution analysis
    asset_returns_path = run_dir / "asset_returns.csv"
    asset_returns_df = None
    if asset_returns_path.exists():
        asset_returns_df = pd.read_csv(asset_returns_path, index_col=0, parse_dates=True)
    
    # Compute statistics
    stats = compute_summary_stats(returns, name="Carry Meta v1 (Phase-1)")
    
    # Year-by-year breakdown
    yearly_stats = {}
    for year in range(2020, 2026):
        year_returns = returns.loc[returns.index.year == year]
        if len(year_returns) > 0:
            yearly_stats[str(year)] = compute_summary_stats(year_returns, name=f"{year}")
    
    # Stress windows
    stress_windows = {}
    
    # 2020 Q1
    q1_2020 = returns.loc[(returns.index >= '2020-01-01') & (returns.index < '2020-04-01')]
    if len(q1_2020) > 0:
        q1_cumret = (1 + q1_2020).prod() - 1
        stress_windows['2020_Q1'] = {
            'cumulative_return': q1_cumret,
            'acceptable': q1_cumret > -0.20
        }
    
    # 2022 (rates shock)
    y2022 = returns.loc[returns.index.year == 2022]
    if len(y2022) > 0:
        y2022_cumret = (1 + y2022).prod() - 1
        stress_windows['2022'] = {
            'cumulative_return': y2022_cumret,
            'acceptable': y2022_cumret > -0.30
        }
    
    # Asset-class contributions
    asset_class_contributions = {}
    if weights_df is not None and asset_returns_df is not None:
        # Map symbols to asset classes
        asset_class_map = {
            'ES_FRONT_CALENDAR_2D': 'equity',
            'NQ_FRONT_CALENDAR_2D': 'equity',
            'RTY_FRONT_CALENDAR_2D': 'equity',
            '6E_FRONT_CALENDAR': 'fx',
            '6B_FRONT_CALENDAR': 'fx',
            '6J_FRONT_CALENDAR': 'fx',
            'ZT_FRONT_VOLUME': 'rates',
            'ZF_FRONT_VOLUME': 'rates',
            'ZN_FRONT_VOLUME': 'rates',
            'UB_FRONT_VOLUME': 'rates',
            'CL_FRONT_VOLUME': 'commodity',
            'GC_FRONT_VOLUME': 'commodity'
        }
        
        # Align weights and returns
        common_dates = weights_df.index.intersection(asset_returns_df.index)
        if len(common_dates) > 0:
            weights_aligned = weights_df.loc[common_dates]
            returns_aligned = asset_returns_df.loc[common_dates]
            
            # Compute contribution per asset class
            for asset_class in ['equity', 'fx', 'rates', 'commodity']:
                class_symbols = [s for s, ac in asset_class_map.items() if ac == asset_class]
                class_symbols = [s for s in class_symbols if s in weights_aligned.columns and s in returns_aligned.columns]
                
                if len(class_symbols) > 0:
                    class_weights = weights_aligned[class_symbols]
                    class_returns = returns_aligned[class_symbols]
                    
                    # Contribution = sum(weight * return) per date
                    class_contribution = (class_weights * class_returns).sum(axis=1)
                    class_cumret = (1 + class_contribution).prod() - 1
                    class_sharpe = compute_summary_stats(class_contribution, name=asset_class).get('sharpe', 0.0)
                    
                    asset_class_contributions[asset_class] = {
                        'cumulative_return': class_cumret,
                        'sharpe': class_sharpe,
                        'symbols': class_symbols
                    }
    
    # Dominance diagnostics
    dominance_diagnostics = {}
    if weights_df is not None:
        # Average absolute weight per asset
        avg_abs_weights = weights_df.abs().mean()
        dominance_diagnostics['avg_abs_weight_per_asset'] = avg_abs_weights.to_dict()
        
        # Crisis window risk share (Mar-May 2020)
        crisis_window = weights_df.loc[(weights_df.index >= '2020-03-01') & (weights_df.index < '2020-06-01')]
        if len(crisis_window) > 0 and asset_returns_df is not None:
            # Compute vol during crisis window
            crisis_returns = asset_returns_df.loc[crisis_window.index]
            crisis_vol = crisis_returns.std() * np.sqrt(252)
            
            # Risk share proxy: mean(|w| × vol) per asset
            crisis_risk_share = (crisis_window.abs() * crisis_vol).mean()
            dominance_diagnostics['crisis_risk_share'] = crisis_risk_share.to_dict()
    
    # Pass criteria evaluation
    sharpe = stats.get('sharpe', 0.0)
    max_dd = stats.get('max_dd', 0.0)
    
    # Check asset-class diversification
    asset_classes_positive = sum(1 for ac in asset_class_contributions.values() if ac.get('cumulative_return', 0) > 0)
    diversified = asset_classes_positive >= 2
    
    pass_criteria = {
        "sharpe_pass_recommended": sharpe >= 0.25,
        "sharpe_pass_conditional": 0.20 <= sharpe < 0.25,
        "maxdd_acceptable": max_dd >= -0.30,
        "crisis_2020_q1_acceptable": stress_windows.get('2020_Q1', {}).get('acceptable', False),
        "stress_2022_acceptable": stress_windows.get('2022', {}).get('acceptable', False),
        "diversified": diversified,
        "asset_classes_positive": asset_classes_positive,
        "overall_pass": sharpe >= 0.25 and max_dd >= -0.30,
        "conditional_pass": 0.20 <= sharpe < 0.25 and max_dd >= -0.30 and diversified
    }
    
    return {
        "run_id": run_id,
        "phase": 1,
        "stats": stats,
        "yearly_stats": yearly_stats,
        "stress_windows": stress_windows,
        "asset_class_contributions": asset_class_contributions,
        "dominance_diagnostics": dominance_diagnostics,
        "pass_criteria": pass_criteria
    }


def run_phase1_backtest(start: str, end: str, config_path: str) -> str:
    """Run Phase-1 backtest and return run_id."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"carry_phase1_v1_{timestamp}"
    
    cmd = [
        "python", "run_strategy.py",
        "--start", start,
        "--end", end,
        "--config_path", config_path,
        "--run_id", run_id
    ]
    
    logger.info("=" * 80)
    logger.info("CARRY META-SLEEVE V1 — PHASE-1 BACKTEST")
    logger.info("=" * 80)
    logger.info(f"Period: {start} to {end}")
    logger.info(f"Config: {config_path}")
    logger.info(f"Executing: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        logger.error(f"Backtest failed:\n{result.stderr}")
        raise RuntimeError(f"Backtest failed with return code {result.returncode}")
    
    logger.info("Backtest completed successfully")
    return run_id


def main():
    """Run Phase-1 diagnostic."""
    parser = argparse.ArgumentParser(description="Carry Phase-1 Diagnostic")
    parser.add_argument(
        "--start",
        type=str,
        default=CANONICAL_START,
        help="Start date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end",
        type=str,
        default=CANONICAL_END,
        help="End date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/carry_phase1_v1.yaml",
        help="Config file path"
    )
    
    args = parser.parse_args()
    
    # Preflight: Save input coverage JSON
    coverage_json_path = Path("carry_inputs_coverage.json")
    if coverage_json_path.exists():
        run_dir = Path("reports/runs") / f"carry_phase1_v1_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        # We'll copy it after run_id is known
        logger.info(f"Input coverage JSON found: {coverage_json_path}")
    
    # Run Phase-1 backtest
    run_id = run_phase1_backtest(args.start, args.end, args.config)
    
    # Copy coverage JSON to run directory
    run_dir = Path("reports/runs") / run_id
    if coverage_json_path.exists() and run_dir.exists():
        import shutil
        shutil.copy(coverage_json_path, run_dir / "carry_inputs_coverage.json")
        logger.info(f"Copied input coverage JSON to run directory")
    
    # Phase-1 Compliance Check
    run_dir = Path("reports/runs") / run_id
    meta_path = run_dir / 'meta.json'
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        logger.info("\n" + "=" * 80)
        logger.info("PHASE-1 COMPLIANCE CHECK")
        logger.info("=" * 80)
        rt_enabled = meta.get('risk_targeting', {}).get('enabled', 'N/A')
        alloc_enabled = meta.get('allocator_v1', {}).get('enabled', 'N/A')
        policy_enabled = meta.get('config', {}).get('engine_policy_v1', {}).get('enabled', 'N/A')
        rebalance = meta.get('rebalance', 'N/A')
        phase = meta.get('config', {}).get('strategies', {}).get('carry_meta_v1', {}).get('params', {}).get('phase', 'N/A')
        effective_start = meta.get('effective_start_date', 'N/A')
        n_trading_days = meta.get('n_trading_days', 'N/A')
        requested_start = meta.get('start_date', 'N/A')
        requested_end = meta.get('end_date', 'N/A')
        
        logger.info(f"Requested Start: {requested_start}")
        logger.info(f"Requested End: {requested_end}")
        logger.info(f"Effective Start Date: {effective_start}")
        logger.info(f"Valid Rows: {n_trading_days}")
        logger.info(f"Risk Targeting Enabled: {rt_enabled} (required: false)")
        logger.info(f"Allocator Enabled: {alloc_enabled} (required: false)")
        logger.info(f"Engine Policy Enabled: {policy_enabled} (required: false)")
        logger.info(f"Rebalance Frequency: {rebalance} (required: D)")
        logger.info(f"Carry Phase: {phase} (required: 1)")
        logger.info(f"Evaluation Layer: Post-Construction (belief layer)")
        
        compliance_pass = (
            rt_enabled == False and
            alloc_enabled == False and
            policy_enabled == False and
            phase == 1
        )
        
        if compliance_pass:
            logger.info("✅ Phase-1 Compliance: PASSED")
        else:
            logger.warning("⚠️ Phase-1 Compliance: FAILED - Check config flags above")
        logger.info("=" * 80)
    
    # Analyze results
    logger.info("\n" + "=" * 80)
    logger.info("CARRY META-SLEEVE V1 — PHASE-1 ANALYSIS")
    logger.info("=" * 80)
    
    summary = analyze_phase1_results(run_id)
    
    if not summary:
        logger.error("Failed to analyze results")
        sys.exit(1)
    
    # Log results
    stats = summary['stats']
    logger.info("\n" + "=" * 80)
    logger.info("PHASE-1 RESULTS")
    logger.info("=" * 80)
    logger.info(f"Observations: {stats['n_obs']}")
    logger.info(f"Annualized Return: {stats['ann_return']:.2%}")
    logger.info(f"Annualized Vol: {stats['ann_vol']:.2%}")
    logger.info(f"Sharpe Ratio: {stats['sharpe']:.3f}")
    logger.info(f"Max Drawdown: {stats['max_dd']:.2%}")
    logger.info(f"Best Day: {stats['best_day']:.2%}")
    logger.info(f"Worst Day: {stats['worst_day']:.2%}")
    logger.info(f"Skewness: {stats['skew']:.3f}")
    logger.info(f"Kurtosis: {stats['kurtosis']:.3f}")
    logger.info(f"Positive Days: {stats['pos_days']} ({int(stats['pos_days'])/stats['n_obs']*100:.1f}%)")
    logger.info(f"Negative Days: {stats['neg_days']} ({int(stats['neg_days'])/stats['n_obs']*100:.1f}%)")
    logger.info(f"Final Equity: ${stats['final_equity']:.2f}")
    
    # Year-by-year breakdown
    logger.info("\n" + "=" * 80)
    logger.info("YEAR-BY-YEAR BREAKDOWN")
    logger.info("=" * 80)
    for year, year_stats in summary.get('yearly_stats', {}).items():
        logger.info(f"{year}: Sharpe={year_stats.get('sharpe', 0):.3f}, "
                   f"Return={year_stats.get('ann_return', 0):.2%}, "
                   f"Vol={year_stats.get('ann_vol', 0):.2%}")
    
    # Stress windows
    logger.info("\n" + "=" * 80)
    logger.info("STRESS WINDOWS")
    logger.info("=" * 80)
    for window_name, window_data in summary.get('stress_windows', {}).items():
        cumret = window_data['cumulative_return']
        acceptable = window_data['acceptable']
        logger.info(f"{window_name}: {cumret:.2%} {'✅' if acceptable else '❌'}")
    
    # Asset-class contributions
    logger.info("\n" + "=" * 80)
    logger.info("ASSET-CLASS CONTRIBUTIONS")
    logger.info("=" * 80)
    for asset_class, contrib in summary.get('asset_class_contributions', {}).items():
        cumret = contrib.get('cumulative_return', 0)
        sharpe = contrib.get('sharpe', 0)
        symbols = contrib.get('symbols', [])
        logger.info(f"{asset_class.upper()}: "
                   f"CumRet={cumret:.2%}, Sharpe={sharpe:.3f}, "
                   f"Symbols={len(symbols)}")
    
    # Dominance diagnostics
    logger.info("\n" + "=" * 80)
    logger.info("DOMINANCE DIAGNOSTICS")
    logger.info("=" * 80)
    dom_diag = summary.get('dominance_diagnostics', {})
    if 'avg_abs_weight_per_asset' in dom_diag:
        logger.info("Average absolute weight per asset:")
        for sym, weight in sorted(dom_diag['avg_abs_weight_per_asset'].items(), 
                                  key=lambda x: abs(x[1]), reverse=True)[:5]:
            logger.info(f"  {sym}: {weight:.4f}")
    
    if 'crisis_risk_share' in dom_diag:
        logger.info("Crisis window (Mar-May 2020) risk share (|w| × vol):")
        for sym, risk in sorted(dom_diag['crisis_risk_share'].items(), 
                               key=lambda x: abs(x[1]), reverse=True)[:5]:
            logger.info(f"  {sym}: {risk:.4f}")
    
    # Pass criteria
    logger.info("\n" + "=" * 80)
    logger.info("PHASE-1 PASS CRITERIA")
    logger.info("=" * 80)
    pc = summary['pass_criteria']
    logger.info(f"✓ Sharpe ≥ 0.25 (recommended): {pc['sharpe_pass_recommended']}")
    logger.info(f"✓ Sharpe 0.20-0.25 (conditional): {pc['sharpe_pass_conditional']}")
    logger.info(f"✓ MaxDD acceptable (≥ -30%): {pc['maxdd_acceptable']}")
    logger.info(f"✓ 2020 Q1 acceptable: {pc['crisis_2020_q1_acceptable']}")
    logger.info(f"✓ 2022 acceptable: {pc['stress_2022_acceptable']}")
    logger.info(f"✓ Diversified (≥2 asset classes positive): {pc['diversified']} "
               f"({pc.get('asset_classes_positive', 0)} classes positive)")
    
    if pc['overall_pass']:
        logger.info("\n" + "=" * 80)
        logger.info("✓✓✓ PHASE-1 PASSED (RECOMMENDED) ✓✓✓")
        logger.info("=" * 80)
    elif pc['conditional_pass']:
        logger.info("\n" + "=" * 80)
        logger.info("⚠️ PHASE-1 CONDITIONAL PASS ⚠️")
        logger.info("=" * 80)
        logger.info("Sharpe is 0.20-0.25. Review asset-class contributions.")
    else:
        logger.info("\n" + "=" * 80)
        logger.info("✗✗✗ PHASE-1 FAILED ✗✗✗")
        logger.info("=" * 80)
        logger.info("Carry Meta-Sleeve v1 does not meet Phase-1 criteria.")
        logger.info("Review implementation and feature definitions.")
    
    # Save summary
    run_dir = Path("reports/runs") / run_id
    summary_path = run_dir / "phase1_analysis_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    logger.info(f"\nAnalysis summary saved to: {summary_path}")
    
    # Exit code
    if pc['overall_pass']:
        sys.exit(0)
    elif pc['conditional_pass']:
        sys.exit(0)  # Conditional pass is still a pass
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
