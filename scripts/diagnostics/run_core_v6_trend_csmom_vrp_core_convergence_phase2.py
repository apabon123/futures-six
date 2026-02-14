#!/usr/bin/env python3
"""
Phase-2 Diagnostics Script for VRP-Convergence Integration.

Compares baseline (core_v5_trend_csmom_vrp_core_no_macro) vs VRP-Convergence-enhanced (core_v6_trend_csmom_vrp_core_convergence_no_macro).

This script:
1. Runs baseline strategy (Trend + CSMOM + VRP-Core)
2. Runs VRP-Convergence-enhanced strategy (Trend + CSMOM + VRP-Core + VRP-Convergence)
3. Compares portfolio metrics (Sharpe, CAGR, Vol, MaxDD, HitRate)
4. Analyzes crisis-period performance
5. Computes sleeve-level correlations including VRP-Convergence
6. Saves comparison outputs and registers in phase index

Usage:
    python scripts/diagnostics/run_core_v6_trend_csmom_vrp_core_convergence_phase2.py
    python scripts/diagnostics/run_core_v6_trend_csmom_vrp_core_convergence_phase2.py --start 2020-01-01 --end 2025-10-31
"""

import sys
import argparse
from pathlib import Path
import logging
from datetime import datetime
from typing import Dict
import json
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config.backtest_window import CANONICAL_START, CANONICAL_END
from run_strategy import main as run_strategy_main
from src.diagnostics.tsmom_sanity import compute_summary_stats
from src.agents import MarketData
from src.agents.strat_tsmom_multihorizon import TSMOMMultiHorizonStrategy
from src.agents.strat_cross_sectional import CSMOMMeta
from src.agents.strat_vrp_core import VRPCoreMeta
from src.agents.strat_vrp_convergence import VRPConvergenceMeta
from src.agents.strat_combined import CombinedStrategy
from src.agents.feature_service import FeatureService
from src.agents.overlay_volmanaged import VolManagedOverlay
from src.agents.risk_vol import RiskVol
from src.agents.allocator import Allocator
from src.agents.exec_sim import ExecSim
import yaml

from scripts.diagnostics.phase2_vrp_sleeve_io import write_vrp_sleeve_returns_csv

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


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
        run_id: Run identifier
        
    Returns:
        Series of portfolio returns indexed by date
    """
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


def compute_sleeve_returns(profile_name: str, start_date: str, end_date: str, market: MarketData) -> Dict[str, pd.Series]:
    """
    Compute individual sleeve returns for correlation analysis.
    
    Args:
        profile_name: Strategy profile name
        start_date: Start date
        end_date: End date
        market: MarketData instance
        
    Returns:
        Dict mapping sleeve names to return series
    """
    # Load config
    config_path = Path("configs/strategies.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    profile_config = config.get("strategy_profiles", {}).get(profile_name, {})
    if not profile_config:
        # Try top-level profiles
        profile_config = config.get(profile_name, {})
    if not profile_config:
        raise ValueError(f"Profile {profile_name} not found in config")
    
    strategies_cfg = profile_config.get("strategies", {})
    
    # Initialize components
    feature_service = FeatureService(market, config=config.get("features", {}))
    risk = RiskVol(
        cov_lookback=config.get("risk_vol", {}).get("cov_lookback", 252),
        vol_lookback=config.get("risk_vol", {}).get("vol_lookback", 63)
    )
    overlay = VolManagedOverlay(
        risk_vol=risk,
        target_vol=config.get("vol_overlay", {}).get("target_vol", 0.20),
        lookback_vol=config.get("vol_overlay", {}).get("lookback_vol", 63)
    )
    allocator = Allocator(
        method=config.get("allocator", {}).get("method", "signal-beta"),
        gross_cap=config.get("allocator", {}).get("gross_cap", 7.0),
        net_cap=config.get("allocator", {}).get("net_cap", 2.0)
    )
    exec_sim = ExecSim(rebalance="W-FRI")
    
    sleeve_returns = {}
    
    # Compute Trend sleeve returns
    if strategies_cfg.get("tsmom_multihorizon", {}).get("enabled", False):
        logger.info("  Computing Trend sleeve returns...")
        tsmom_params = strategies_cfg.get("tsmom_multihorizon", {}).get("params", {})
        tsmom = TSMOMMultiHorizonStrategy(
            horizon_weights=tsmom_params.get("horizon_weights", {}),
            feature_weights=tsmom_params.get("feature_weights", {}),
            signal_cap=tsmom_params.get("signal_cap", 3.0),
            rebalance=tsmom_params.get("rebalance", "W-FRI")
        )
        tsmom.fit_in_sample(market, start=start_date, end=end_date)
        
        # Run backtest for Trend sleeve only
        combined = CombinedStrategy(
            strategies={"tsmom_multihorizon": tsmom},
            weights={"tsmom_multihorizon": 1.0},
            feature_service=feature_service
        )
        components = {
            'strategy': combined,
            'overlay': overlay,
            'risk_vol': risk,
            'allocator': allocator
        }
        results = exec_sim.run(market, start_date, end_date, components)
        trend_equity = results['equity_curve']
        trend_returns = trend_equity.pct_change().dropna()
        sleeve_returns['trend'] = trend_returns
    
    # Compute CSMOM sleeve returns
    if strategies_cfg.get("csmom_meta", {}).get("enabled", False):
        logger.info("  Computing CSMOM sleeve returns...")
        csmom_params = strategies_cfg.get("csmom_meta", {}).get("params", {})
        csmom = CSMOMMeta(
            symbols=None,
            lookbacks=csmom_params.get("lookbacks", [63, 126, 252]),
            weights=csmom_params.get("horizon_weights", [0.4, 0.35, 0.25]),
            vol_lookback=csmom_params.get("vol_lookback", 63),
            rebalance_freq=csmom_params.get("rebalance", "D"),
            neutralize_cross_section=csmom_params.get("neutralize_cross_section", True),
            clip_score=csmom_params.get("clip", 3.0)
        )
        
        # Run backtest for CSMOM sleeve only
        combined = CombinedStrategy(
            strategies={"csmom_meta": csmom},
            weights={"csmom_meta": 1.0},
            feature_service=feature_service
        )
        components = {
            'strategy': combined,
            'overlay': overlay,
            'risk_vol': risk,
            'allocator': allocator
        }
        results = exec_sim.run(market, start_date, end_date, components)
        csmom_equity = results['equity_curve']
        csmom_returns = csmom_equity.pct_change().dropna()
        sleeve_returns['csmom'] = csmom_returns
    
    # Compute VRP-Core sleeve returns
    if strategies_cfg.get("vrp_core_meta", {}).get("enabled", False):
        logger.info("  Computing VRP-Core sleeve returns...")
        vrp_params = strategies_cfg.get("vrp_core_meta", {}).get("params", {})
        # Get DB path
        data_config_path = Path("configs/data.yaml")
        db_path = None
        if data_config_path.exists():
            with open(data_config_path, 'r') as f:
                data_config = yaml.safe_load(f)
            db_path = data_config.get('db', {}).get('path')
        
        vrp = VRPCoreMeta(
            rv_lookback=vrp_params.get("rv_lookback", 21),
            zscore_window=vrp_params.get("zscore_window", 252),
            clip=vrp_params.get("clip", 3.0),
            signal_mode=vrp_params.get("signal_mode", "zscore"),
            db_path=db_path
        )
        
        # Run backtest for VRP-Core sleeve only
        combined = CombinedStrategy(
            strategies={"vrp_core_meta": vrp},
            weights={"vrp_core_meta": 1.0},
            feature_service=feature_service
        )
        components = {
            'strategy': combined,
            'overlay': overlay,
            'risk_vol': risk,
            'allocator': allocator
        }
        results = exec_sim.run(market, start_date, end_date, components)
        vrp_equity = results['equity_curve']
        vrp_returns = vrp_equity.pct_change().dropna()
        sleeve_returns['vrp_core'] = vrp_returns
    
    # Compute VRP-Convergence sleeve returns
    if strategies_cfg.get("vrp_convergence_meta", {}).get("enabled", False):
        logger.info("  Computing VRP-Convergence sleeve returns...")
        vrp_conv_params = strategies_cfg.get("vrp_convergence_meta", {}).get("params", {})
        # Get DB path
        data_config_path = Path("configs/data.yaml")
        db_path = None
        if data_config_path.exists():
            with open(data_config_path, 'r') as f:
                data_config = yaml.safe_load(f)
            db_path = data_config.get('db', {}).get('path')
        
        vrp_conv = VRPConvergenceMeta(
            zscore_window=vrp_conv_params.get("zscore_window", 252),
            clip=vrp_conv_params.get("clip", 3.0),
            signal_mode=vrp_conv_params.get("signal_mode", "zscore"),
            target_vol=vrp_conv_params.get("target_vol", 0.10),
            vol_lookback=vrp_conv_params.get("vol_lookback", 63),
            vol_floor=vrp_conv_params.get("vol_floor", 0.05),
            db_path=db_path
        )
        
        # Run backtest for VRP-Convergence sleeve only
        combined = CombinedStrategy(
            strategies={"vrp_convergence_meta": vrp_conv},
            weights={"vrp_convergence_meta": 1.0},
            feature_service=feature_service
        )
        components = {
            'strategy': combined,
            'overlay': overlay,
            'risk_vol': risk,
            'allocator': allocator
        }
        results = exec_sim.run(market, start_date, end_date, components)
        vrp_conv_equity = results['equity_curve']
        vrp_conv_returns = vrp_conv_equity.pct_change().dropna()
        sleeve_returns['vrp_convergence'] = vrp_conv_returns
    
    return sleeve_returns


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
        description="Phase-2 Diagnostics: VRP-Convergence Integration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with canonical dates
  python scripts/diagnostics/run_core_v6_trend_csmom_vrp_core_convergence_phase2.py
  
  # Run with custom dates
  python scripts/diagnostics/run_core_v6_trend_csmom_vrp_core_convergence_phase2.py --start 2020-01-01 --end 2025-10-31
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
        help="Output directory (default: data/diagnostics/phase2/core_v6_trend_csmom_vrp_core_convergence)"
    )
    
    args = parser.parse_args()
    
    try:
        logger.info("=" * 80)
        logger.info("VRP-CONVERGENCE PHASE-2 DIAGNOSTICS")
        logger.info("=" * 80)
        logger.info(f"Start date: {args.start}")
        logger.info(f"End date: {args.end}")
        
        # Determine output directory
        if args.output_dir:
            output_dir = Path(args.output_dir)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = Path(f"data/diagnostics/phase2/core_v6_trend_csmom_vrp_core_convergence/{timestamp}")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {output_dir}")
        
        # Generate run IDs
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        baseline_run_id = f"core_v5_baseline_phase2_{timestamp}"
        variant_run_id = f"core_v6_vrp_convergence_phase2_{timestamp}"
        
        # 1) Run baseline (core_v5)
        logger.info("\n" + "=" * 80)
        logger.info("[1/4] Running baseline: core_v5_trend_csmom_vrp_core_no_macro")
        logger.info("=" * 80)
        run_strategy_profile(
            profile_name="core_v5_trend_csmom_vrp_core_no_macro",
            run_id=baseline_run_id,
            start_date=args.start,
            end_date=args.end
        )
        
        # 2) Run VRP-Convergence-enhanced (core_v6)
        logger.info("\n" + "=" * 80)
        logger.info("[2/4] Running VRP-Convergence-enhanced: core_v6_trend_csmom_vrp_core_convergence_no_macro")
        logger.info("=" * 80)
        run_strategy_profile(
            profile_name="core_v6_trend_csmom_vrp_core_convergence_no_macro",
            run_id=variant_run_id,
            start_date=args.start,
            end_date=args.end
        )
        
        # 3) Load returns and compute metrics
        logger.info("\n" + "=" * 80)
        logger.info("[3/4] Computing comparison metrics")
        logger.info("=" * 80)
        
        baseline_returns = load_run_returns(baseline_run_id)
        variant_returns = load_run_returns(variant_run_id)
        
        # Align returns
        common_dates = baseline_returns.index.intersection(variant_returns.index)
        baseline_returns = baseline_returns.loc[common_dates]
        variant_returns = variant_returns.loc[common_dates]
        
        logger.info(f"Aligned returns: {len(baseline_returns)} days")
        
        # Compute baseline metrics
        baseline_equity = (1 + baseline_returns).cumprod()
        baseline_asset_returns = pd.DataFrame({'Portfolio': baseline_returns})
        baseline_stats = compute_summary_stats(
            portfolio_returns=baseline_returns,
            equity_curve=baseline_equity,
            asset_strategy_returns=baseline_asset_returns
        )
        baseline_metrics = baseline_stats['portfolio']
        
        # Compute variant metrics
        variant_equity = (1 + variant_returns).cumprod()
        variant_asset_returns = pd.DataFrame({'Portfolio': variant_returns})
        variant_stats = compute_summary_stats(
            portfolio_returns=variant_returns,
            equity_curve=variant_equity,
            asset_strategy_returns=variant_asset_returns
        )
        variant_metrics = variant_stats['portfolio']
        
        # Compute crisis period metrics
        logger.info("Computing crisis period metrics...")
        baseline_crisis = compute_crisis_periods(baseline_returns)
        variant_crisis = compute_crisis_periods(variant_returns)
        
        # Compute portfolio correlation
        correlation = baseline_returns.corr(variant_returns)
        
        # Initialize sleeve correlation variables
        sleeve_corr_summary = {}
        corr_matrix = pd.DataFrame()
        
        # Compute sleeve-level returns and correlations
        logger.info("Computing sleeve-level returns...")
        market = MarketData()
        try:
            # Compute sleeve returns for variant profile (core_v6)
            variant_sleeve_returns = compute_sleeve_returns(
                profile_name="core_v6_trend_csmom_vrp_core_convergence_no_macro",
                start_date=args.start,
                end_date=args.end,
                market=market
            )
            
            # Align all return series
            all_dates = common_dates.copy()
            for sleeve_name, sleeve_ret in variant_sleeve_returns.items():
                all_dates = all_dates.intersection(sleeve_ret.index)
            
            # Emit Phase-2 atomic sleeve returns (Phase-4 compatible) to variant run dir
            variant_run_dir = Path("reports/runs") / variant_run_id
            write_vrp_sleeve_returns_csv(variant_run_dir, variant_sleeve_returns, all_dates)
            logger.info(f"  Wrote {variant_run_dir / 'sleeve_returns.csv'}")
            
            # Build correlation DataFrame
            corr_data = {}
            
            # Portfolio returns
            corr_data['baseline_portfolio'] = baseline_returns.loc[all_dates]
            corr_data['variant_portfolio'] = variant_returns.loc[all_dates]
            
            # Sleeve returns (from variant profile)
            for sleeve_name, sleeve_ret in variant_sleeve_returns.items():
                corr_data[sleeve_name] = sleeve_ret.loc[all_dates]
            
            # Build DataFrame and compute correlation matrix
            if len(corr_data) > 0:
                corr_df = pd.DataFrame(corr_data).dropna()
                if len(corr_df) > 0:
                    corr_matrix = corr_df.corr()
                    
                    # Extract VRP-Convergence correlations
                    if 'vrp_convergence' in corr_df.columns:
                        if 'trend' in corr_df.columns:
                            sleeve_corr_summary['corr_vrp_convergence_vs_trend'] = float(corr_matrix.loc['vrp_convergence', 'trend'])
                        if 'csmom' in corr_df.columns:
                            sleeve_corr_summary['corr_vrp_convergence_vs_csmom'] = float(corr_matrix.loc['vrp_convergence', 'csmom'])
                        if 'vrp_core' in corr_df.columns:
                            sleeve_corr_summary['corr_vrp_convergence_vs_vrp_core'] = float(corr_matrix.loc['vrp_convergence', 'vrp_core'])
                        if 'baseline_portfolio' in corr_df.columns:
                            sleeve_corr_summary['corr_vrp_convergence_vs_baseline_portfolio'] = float(corr_matrix.loc['vrp_convergence', 'baseline_portfolio'])
                        if 'variant_portfolio' in corr_df.columns:
                            sleeve_corr_summary['corr_vrp_convergence_vs_variant_portfolio'] = float(corr_matrix.loc['vrp_convergence', 'variant_portfolio'])
            
            # Log sleeve correlations
            if sleeve_corr_summary:
                logger.info("\n=== Sleeve-Level Correlations (VRP-Convergence Phase-2) ===")
                if 'corr_vrp_convergence_vs_trend' in sleeve_corr_summary:
                    logger.info("corr(VRP-Convergence, Trend): %.4f", sleeve_corr_summary['corr_vrp_convergence_vs_trend'])
                if 'corr_vrp_convergence_vs_csmom' in sleeve_corr_summary:
                    logger.info("corr(VRP-Convergence, CSMOM): %.4f", sleeve_corr_summary['corr_vrp_convergence_vs_csmom'])
                if 'corr_vrp_convergence_vs_vrp_core' in sleeve_corr_summary:
                    logger.info("corr(VRP-Convergence, VRP-Core): %.4f", sleeve_corr_summary['corr_vrp_convergence_vs_vrp_core'])
                if 'corr_vrp_convergence_vs_baseline_portfolio' in sleeve_corr_summary:
                    logger.info("corr(VRP-Convergence, Baseline Portfolio): %.4f", sleeve_corr_summary['corr_vrp_convergence_vs_baseline_portfolio'])
                if 'corr_vrp_convergence_vs_variant_portfolio' in sleeve_corr_summary:
                    logger.info("corr(VRP-Convergence, Variant Portfolio): %.4f", sleeve_corr_summary['corr_vrp_convergence_vs_variant_portfolio'])
            
        except Exception as e:
            logger.warning(f"Failed to compute sleeve-level correlations: {e}")
            logger.warning("Continuing with portfolio-level metrics only")
            import traceback
            traceback.print_exc()
        finally:
            market.close()
        
        # Compute difference metrics
        diff_metrics = {
            'sharpe_diff': variant_metrics.get('Sharpe', 0) - baseline_metrics.get('Sharpe', 0),
            'cagr_diff': variant_metrics.get('CAGR', 0) - baseline_metrics.get('CAGR', 0),
            'vol_diff': variant_metrics.get('Vol', 0) - baseline_metrics.get('Vol', 0),
            'maxdd_diff': variant_metrics.get('MaxDD', 0) - baseline_metrics.get('MaxDD', 0),
            'hitrate_diff': variant_metrics.get('HitRate', 0) - baseline_metrics.get('HitRate', 0),
        }
        
        # 4) Save outputs
        logger.info("\n" + "=" * 80)
        logger.info("[4/4] Saving comparison outputs")
        logger.info("=" * 80)
        
        # Save returns
        returns_df = pd.DataFrame({
            'baseline': baseline_returns,
            'variant': variant_returns
        })
        returns_df.to_csv(output_dir / 'comparison_returns.csv')
        returns_df.to_parquet(output_dir / 'comparison_returns.parquet')
        
        # Save metrics
        comparison_summary = {
            'run_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'start_date': args.start,
            'end_date': args.end,
            'baseline_run_id': baseline_run_id,
            'variant_run_id': variant_run_id,
            'baseline_metrics': baseline_metrics,
            'variant_metrics': variant_metrics,
            'diff_metrics': diff_metrics,
            'correlation': float(correlation),
            'baseline_crisis': baseline_crisis,
            'variant_crisis': variant_crisis,
            'sleeve_correlations': sleeve_corr_summary if sleeve_corr_summary else None,
        }
        
        with open(output_dir / 'comparison_summary.json', 'w') as f:
            json.dump(comparison_summary, f, indent=2, default=str)
        
        # Save diff metrics separately
        with open(output_dir / 'diff_metrics.json', 'w') as f:
            json.dump(diff_metrics, f, indent=2)
        
        # Save sleeve correlations
        if sleeve_corr_summary:
            with open(output_dir / 'sleeve_correlations.json', 'w') as f:
                json.dump(sleeve_corr_summary, f, indent=2)
            
            # Save full correlation matrix
            if not corr_matrix.empty:
                corr_matrix.to_csv(output_dir / 'sleeve_correlation_matrix.csv')
                logger.info(f"  Saved sleeve correlations to {output_dir}")
        
        # Generate plots
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            
            # Equity curves
            fig, ax = plt.subplots(figsize=(14, 6))
            ax.plot(baseline_equity.index, baseline_equity.values, label='Baseline (core_v5)', linewidth=1.5)
            ax.plot(variant_equity.index, variant_equity.values, label='Variant (core_v6)', linewidth=1.5)
            ax.set_xlabel('Date')
            ax.set_ylabel('Equity')
            ax.set_title('Phase-2: Baseline vs Variant Equity Curves')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_dir / 'equity_curves.png', dpi=150)
            plt.close()
            
            # Drawdown curves
            baseline_dd = (baseline_equity / baseline_equity.expanding().max() - 1) * 100
            variant_dd = (variant_equity / variant_equity.expanding().max() - 1) * 100
            fig, ax = plt.subplots(figsize=(14, 6))
            ax.plot(baseline_dd.index, baseline_dd.values, label='Baseline (core_v5)', linewidth=1.5)
            ax.plot(variant_dd.index, variant_dd.values, label='Variant (core_v6)', linewidth=1.5)
            ax.set_xlabel('Date')
            ax.set_ylabel('Drawdown (%)')
            ax.set_title('Phase-2: Baseline vs Variant Drawdown Curves')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_dir / 'drawdown_curves.png', dpi=150)
            plt.close()
            
            logger.info(f"  Generated plots in {output_dir}")
        except ImportError:
            logger.warning("matplotlib not available, skipping plots")
        
        # Register in phase index
        phase_index_dir = Path("reports/phase_index/vrp")
        phase_index_dir.mkdir(parents=True, exist_ok=True)
        
        phase2_file = phase_index_dir / "phase2_core_v6_trend_csmom_vrp_core_convergence.txt"
        with open(phase2_file, 'w') as f:
            f.write(f"# Phase-2: VRP-Convergence Portfolio Integration\n")
            f.write(f"# Registered: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"baseline_profile: core_v5_trend_csmom_vrp_core_no_macro\n")
            f.write(f"variant_profile: core_v6_trend_csmom_vrp_core_convergence_no_macro\n")
            f.write(f"start_date: {args.start}\n")
            f.write(f"end_date: {args.end}\n")
            f.write(f"\n# Baseline Metrics\n")
            f.write(f"baseline_sharpe: {baseline_metrics.get('Sharpe', float('nan')):.4f}\n")
            f.write(f"baseline_cagr: {baseline_metrics.get('CAGR', float('nan')):.4f}\n")
            f.write(f"baseline_maxdd: {baseline_metrics.get('MaxDD', float('nan')):.4f}\n")
            f.write(f"\n# Variant Metrics\n")
            f.write(f"variant_sharpe: {variant_metrics.get('Sharpe', float('nan')):.4f}\n")
            f.write(f"variant_cagr: {variant_metrics.get('CAGR', float('nan')):.4f}\n")
            f.write(f"variant_maxdd: {variant_metrics.get('MaxDD', float('nan')):.4f}\n")
            f.write(f"\n# Differences\n")
            f.write(f"sharpe_diff: {diff_metrics.get('sharpe_diff', float('nan')):.4f}\n")
            f.write(f"cagr_diff: {diff_metrics.get('cagr_diff', float('nan')):.4f}\n")
            f.write(f"maxdd_diff: {diff_metrics.get('maxdd_diff', float('nan')):.4f}\n")
            f.write(f"\n# Sleeve Correlations\n")
            if sleeve_corr_summary:
                if 'corr_vrp_convergence_vs_vrp_core' in sleeve_corr_summary:
                    f.write(f"corr_vrp_convergence_vs_vrp_core: {sleeve_corr_summary['corr_vrp_convergence_vs_vrp_core']:.4f}\n")
                if 'corr_vrp_convergence_vs_trend' in sleeve_corr_summary:
                    f.write(f"corr_vrp_convergence_vs_trend: {sleeve_corr_summary['corr_vrp_convergence_vs_trend']:.4f}\n")
                if 'corr_vrp_convergence_vs_csmom' in sleeve_corr_summary:
                    f.write(f"corr_vrp_convergence_vs_csmom: {sleeve_corr_summary['corr_vrp_convergence_vs_csmom']:.4f}\n")
            f.write(f"\npath: {output_dir}\n")
            
            # Pass criteria check
            sharpe_diff = diff_metrics.get('sharpe_diff', float('-inf'))
            maxdd_diff = diff_metrics.get('maxdd_diff', float('inf'))
            corr_vrp_core = sleeve_corr_summary.get('corr_vrp_convergence_vs_vrp_core', 1.0) if sleeve_corr_summary else 1.0
            
            f.write(f"\n# Phase-2 Pass Criteria\n")
            f.write(f"sharpe_pass: {'PASS' if sharpe_diff >= -0.01 else 'FAIL'} (variant >= baseline - 0.01)\n")
            f.write(f"maxdd_pass: {'PASS' if maxdd_diff >= 0 else 'FAIL'} (variant >= baseline)\n")
            f.write(f"corr_pass: {'PASS' if corr_vrp_core < 0.9 else 'FAIL'} (corr(VRP-Conv, VRP-Core) < 0.9)\n")
            
            verdict = "PASS" if (sharpe_diff >= -0.01 and maxdd_diff >= 0 and corr_vrp_core < 0.9) else "FAIL"
            f.write(f"\nverdict: {verdict}\n")
        
        logger.info(f"  Registered in: {phase2_file}")
        
        # Print summary
        print("\n" + "=" * 80)
        print("VRP-CONVERGENCE PHASE-2 SUMMARY")
        print("=" * 80)
        print(f"\nBaseline (core_v5):")
        print(f"  Sharpe:  {baseline_metrics.get('Sharpe', float('nan')):8.4f}")
        print(f"  CAGR:   {baseline_metrics.get('CAGR', float('nan')):8.4f} ({baseline_metrics.get('CAGR', 0)*100:6.2f}%)")
        print(f"  MaxDD:  {baseline_metrics.get('MaxDD', float('nan')):8.4f} ({baseline_metrics.get('MaxDD', 0)*100:6.2f}%)")
        print(f"\nVariant (core_v6):")
        print(f"  Sharpe:  {variant_metrics.get('Sharpe', float('nan')):8.4f}")
        print(f"  CAGR:   {variant_metrics.get('CAGR', float('nan')):8.4f} ({variant_metrics.get('CAGR', 0)*100:6.2f}%)")
        print(f"  MaxDD:  {variant_metrics.get('MaxDD', float('nan')):8.4f} ({variant_metrics.get('MaxDD', 0)*100:6.2f}%)")
        print(f"\nDifferences:")
        print(f"  Sharpe:  {diff_metrics.get('sharpe_diff', float('nan')):8.4f}")
        print(f"  CAGR:   {diff_metrics.get('cagr_diff', float('nan')):8.4f} ({diff_metrics.get('cagr_diff', 0)*100:6.2f}%)")
        print(f"  MaxDD:  {diff_metrics.get('maxdd_diff', float('nan')):8.4f} ({diff_metrics.get('maxdd_diff', 0)*100:6.2f}%)")
        if sleeve_corr_summary:
            print(f"\nSleeve Correlations:")
            if 'corr_vrp_convergence_vs_vrp_core' in sleeve_corr_summary:
                print(f"  corr(VRP-Conv, VRP-Core): {sleeve_corr_summary['corr_vrp_convergence_vs_vrp_core']:.4f}")
            if 'corr_vrp_convergence_vs_trend' in sleeve_corr_summary:
                print(f"  corr(VRP-Conv, Trend): {sleeve_corr_summary['corr_vrp_convergence_vs_trend']:.4f}")
            if 'corr_vrp_convergence_vs_csmom' in sleeve_corr_summary:
                print(f"  corr(VRP-Conv, CSMOM): {sleeve_corr_summary['corr_vrp_convergence_vs_csmom']:.4f}")
        
        print(f"\nResults saved to: {output_dir}")
        print("=" * 80)
        
    except Exception as e:
        logger.error(f"Phase-2 diagnostics failed: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
