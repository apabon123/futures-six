"""
SR3 Curve RV Momentum Phase-1 Runner

Engineered, tradable implementation of SR3 curve shape momentum.
Upgrades Phase-0 sign-only sanity check with:
- Z-scored signal normalization
- Volatility targeting
- Signal smoothing
- Redundancy analysis between the three atomic sleeves

Three atomic sleeves:
1. Pack Slope Momentum
2. Pack Curvature Momentum
3. Rank Fly Momentum
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
import pandas as pd
import numpy as np
import json
import logging

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.backtest_window import CANONICAL_START, CANONICAL_END
from src.agents import MarketData
from src.strategies.rates_curve_rv.sr3_curve_rv_momentum import (
    compute_pack_slope_momentum_phase1,
    compute_pack_curvature_momentum_phase1,
    compute_rank_fly_momentum_phase1
)
from src.utils.phase_index import update_phase_index

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Meta-sleeve and atomic sleeve names
META_SLEEVE = "rates_curve_rv"
ATOMIC_SLEEVES = {
    "pack_slope": "sr3_curve_rv_pack_slope_momentum",
    "pack_curvature": "sr3_curve_rv_pack_curvature_momentum",
    "rank_fly": "sr3_curve_rv_rank_fly_2_6_10_momentum"
}


def compute_summary_stats(
    portfolio_returns: pd.Series,
    equity_curve: pd.Series
) -> dict:
    """Compute summary statistics for portfolio returns."""
    if portfolio_returns.empty:
        return {
            'CAGR': 0.0,
            'Vol': 0.0,
            'Sharpe': 0.0,
            'MaxDD': 0.0,
            'HitRate': 0.0,
            'n_days': 0,
            'years': 0.0
        }
    
    n_days = len(portfolio_returns)
    years = n_days / 252.0
    
    total_return = equity_curve.iloc[-1] / equity_curve.iloc[0] - 1.0 if len(equity_curve) > 0 else 0.0
    cagr = (1.0 + total_return) ** (1.0 / years) - 1.0 if years > 0 else 0.0
    
    vol = portfolio_returns.std() * np.sqrt(252)
    sharpe = (cagr / vol) if vol > 0 else 0.0
    
    running_max = equity_curve.expanding().max()
    drawdown = (equity_curve - running_max) / running_max
    maxdd = drawdown.min()
    
    hit_rate = (portfolio_returns > 0).sum() / len(portfolio_returns) if len(portfolio_returns) > 0 else 0.0
    
    return {
        'CAGR': cagr,
        'Vol': vol,
        'Sharpe': sharpe,
        'MaxDD': maxdd,
        'HitRate': hit_rate,
        'n_days': n_days,
        'years': years
    }


def compute_redundancy_analysis(
    results: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Compute redundancy analysis between the three sleeves.
    
    Analyzes:
    - Correlation between portfolio returns
    - Correlation between signals
    - Whether rank fly subsumes pack slope/curvature
    - Whether curvature adds orthogonal information
    """
    redundancy = {}
    
    # Extract portfolio returns for each sleeve
    pack_slope_rets = results.get('pack_slope', {}).get('portfolio_returns', pd.Series())
    pack_curv_rets = results.get('pack_curvature', {}).get('portfolio_returns', pd.Series())
    rank_fly_rets = results.get('rank_fly', {}).get('portfolio_returns', pd.Series())
    
    # Extract signals
    pack_slope_sig = results.get('pack_slope', {}).get('signals', pd.Series())
    pack_curv_sig = results.get('pack_curvature', {}).get('signals', pd.Series())
    rank_fly_sig = results.get('rank_fly', {}).get('signals', pd.Series())
    
    # Portfolio return correlations
    if not pack_slope_rets.empty and not pack_curv_rets.empty:
        common_dates = pack_slope_rets.index.intersection(pack_curv_rets.index)
        if len(common_dates) > 10:
            corr = pack_slope_rets.loc[common_dates].corr(pack_curv_rets.loc[common_dates])
            redundancy['corr_pack_slope_vs_curvature_returns'] = float(corr) if not pd.isna(corr) else None
    
    if not pack_slope_rets.empty and not rank_fly_rets.empty:
        common_dates = pack_slope_rets.index.intersection(rank_fly_rets.index)
        if len(common_dates) > 10:
            corr = pack_slope_rets.loc[common_dates].corr(rank_fly_rets.loc[common_dates])
            redundancy['corr_pack_slope_vs_rank_fly_returns'] = float(corr) if not pd.isna(corr) else None
    
    if not pack_curv_rets.empty and not rank_fly_rets.empty:
        common_dates = pack_curv_rets.index.intersection(rank_fly_rets.index)
        if len(common_dates) > 10:
            corr = pack_curv_rets.loc[common_dates].corr(rank_fly_rets.loc[common_dates])
            redundancy['corr_pack_curvature_vs_rank_fly_returns'] = float(corr) if not pd.isna(corr) else None
    
    # Signal correlations
    if not pack_slope_sig.empty and not pack_curv_sig.empty:
        common_dates = pack_slope_sig.index.intersection(pack_curv_sig.index)
        if len(common_dates) > 10:
            corr = pack_slope_sig.loc[common_dates].corr(pack_curv_sig.loc[common_dates])
            redundancy['corr_pack_slope_vs_curvature_signals'] = float(corr) if not pd.isna(corr) else None
    
    if not pack_slope_sig.empty and not rank_fly_sig.empty:
        common_dates = pack_slope_sig.index.intersection(rank_fly_sig.index)
        if len(common_dates) > 10:
            corr = pack_slope_sig.loc[common_dates].corr(rank_fly_sig.loc[common_dates])
            redundancy['corr_pack_slope_vs_rank_fly_signals'] = float(corr) if not pd.isna(corr) else None
    
    if not pack_curv_sig.empty and not rank_fly_sig.empty:
        common_dates = pack_curv_sig.index.intersection(rank_fly_sig.index)
        if len(common_dates) > 10:
            corr = pack_curv_sig.loc[common_dates].corr(rank_fly_sig.loc[common_dates])
            redundancy['corr_pack_curvature_vs_rank_fly_signals'] = float(corr) if not pd.isna(corr) else None
    
    # Summary interpretation
    redundancy['interpretation'] = {
        'high_correlation_threshold': 0.7,
        'moderate_correlation_threshold': 0.4,
        'note': 'High correlation (>0.7) suggests redundancy. Moderate (0.4-0.7) suggests related but distinct signals. Low (<0.4) suggests orthogonal information.'
    }
    
    return redundancy


def save_results(
    results: Dict[str, Dict[str, Any]],
    stats: Dict[str, dict],
    redundancy: Dict[str, Any],
    output_dir: Path,
    start_date: str,
    end_date: str,
    config: dict
):
    """Save all results to output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save portfolio returns for each sleeve
    for sleeve_key, sleeve_name in ATOMIC_SLEEVES.items():
        if sleeve_key in results:
            result = results[sleeve_key]
            if 'portfolio_returns' in result:
                rets_df = pd.DataFrame({
                    'date': result['portfolio_returns'].index,
                    'ret': result['portfolio_returns'].values
                })
                rets_df.to_csv(output_dir / f'{sleeve_key}_portfolio_returns.csv', index=False)
            
            if 'equity_curve' in result:
                equity_df = pd.DataFrame({
                    'date': result['equity_curve'].index,
                    'equity': result['equity_curve'].values
                })
                equity_df.to_csv(output_dir / f'{sleeve_key}_equity_curve.csv', index=False)
            
            if 'signals' in result:
                sig_df = pd.DataFrame({
                    'date': result['signals'].index,
                    'signal': result['signals'].values
                })
                sig_df.to_csv(output_dir / f'{sleeve_key}_signals.csv', index=False)
            
            if 'positions' in result:
                pos_df = pd.DataFrame({
                    'date': result['positions'].index,
                    'position': result['positions'].values
                })
                pos_df.to_csv(output_dir / f'{sleeve_key}_positions.csv', index=False)
    
    # Save summary stats
    stats_df = pd.DataFrame(stats).T
    stats_df.to_csv(output_dir / 'summary_stats.csv')
    
    # Save redundancy analysis
    with open(output_dir / 'redundancy_analysis.json', 'w') as f:
        json.dump(redundancy, f, indent=2, default=str)
    
    # Save metadata
    meta = {
        'start_date': start_date,
        'end_date': end_date,
        'meta_sleeve': META_SLEEVE,
        'atomic_sleeves': ATOMIC_SLEEVES,
        'config': config,
        'stats': stats,
        'redundancy_analysis': redundancy
    }
    
    with open(output_dir / 'meta.json', 'w') as f:
        json.dump(meta, f, indent=2, default=str)
    
    logger.info(f"Results saved to {output_dir}")


def generate_plots(
    results: Dict[str, Dict[str, Any]],
    stats: Dict[str, dict],
    output_dir: Path
):
    """Generate diagnostic plots."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available, skipping plots")
        return
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Equity curves comparison
    fig, ax = plt.subplots(figsize=(14, 8))
    
    for sleeve_key, sleeve_name in ATOMIC_SLEEVES.items():
        if sleeve_key in results and 'equity_curve' in results[sleeve_key]:
            equity = results[sleeve_key]['equity_curve']
            sharpe = stats.get(sleeve_key, {}).get('Sharpe', 0)
            ax.plot(equity.index, equity.values, 
                   label=f'{sleeve_key} (Sharpe={sharpe:.2f})', linewidth=2)
    
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    ax.set_title('SR3 Curve RV Momentum Phase-1: Equity Curves Comparison')
    ax.set_xlabel('Date')
    ax.set_ylabel('Cumulative Equity')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'equity_curves_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved equity_curves_comparison.png")
    
    # 2. Return histograms
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, (sleeve_key, sleeve_name) in enumerate(ATOMIC_SLEEVES.items()):
        if sleeve_key in results and 'portfolio_returns' in results[sleeve_key]:
            rets = results[sleeve_key]['portfolio_returns'].dropna()
            rets_clean = rets[np.isfinite(rets)]
            if len(rets_clean) > 0:
                axes[idx].hist(rets_clean.values, bins=50, alpha=0.7, edgecolor='black')
                axes[idx].axvline(x=0, color='red', linestyle='--', alpha=0.5)
                axes[idx].set_title(f'{sleeve_key}\n(Mean={rets_clean.mean():.4f}, Std={rets_clean.std():.4f})')
                axes[idx].set_xlabel('Daily Return')
                axes[idx].set_ylabel('Frequency')
                axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'return_histograms.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved return_histograms.png")


def main():
    parser = argparse.ArgumentParser(
        description="Run SR3 Curve RV Momentum Phase-1",
        formatter_class=argparse.RawDescriptionHelpFormatter
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
        "--zscore-window",
        type=int,
        default=252,
        help="Rolling window for z-score normalization (default: 252 days)"
    )
    parser.add_argument(
        "--clip",
        type=float,
        default=3.0,
        help="Symmetric clipping bounds for normalized signal (default: 3.0)"
    )
    parser.add_argument(
        "--target-vol",
        type=float,
        default=0.10,
        help="Target annualized volatility (default: 0.10 = 10%)"
    )
    parser.add_argument(
        "--vol-lookback",
        type=int,
        default=63,
        help="Rolling window for realized vol calculation (default: 63 days)"
    )
    parser.add_argument(
        "--min-vol-floor",
        type=float,
        default=0.01,
        help="Minimum annualized vol floor (default: 0.01 = 1%)"
    )
    parser.add_argument(
        "--max-leverage",
        type=float,
        default=10.0,
        help="Maximum leverage cap (default: 10.0)"
    )
    parser.add_argument(
        "--lag",
        type=int,
        default=1,
        help="Execution lag in days (default: 1)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: reports/runs/rates_curve_rv/sr3_curve_rv_momentum_phase1/<timestamp>)"
    )
    
    args = parser.parse_args()
    
    try:
        logger.info("=" * 80)
        logger.info("SR3 CURVE RV MOMENTUM PHASE-1")
        logger.info("=" * 80)
        logger.info(f"Start date: {args.start}")
        logger.info(f"End date: {args.end}")
        logger.info(f"Z-score window: {args.zscore_window} days")
        logger.info(f"Clip: ±{args.clip}")
        logger.info(f"Target vol: {args.target_vol*100:.1f}%")
        logger.info(f"Vol lookback: {args.vol_lookback} days")
        
        # Initialize MarketData
        logger.info("\n[1/4] Initializing MarketData broker...")
        market = MarketData()
        logger.info("  MarketData initialized")
        
        # Determine output directory
        if args.output_dir:
            output_dir = Path(args.output_dir)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = Path(f"reports/runs/{META_SLEEVE}/sr3_curve_rv_momentum_phase1/{timestamp}")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {output_dir}")
        
        # Run all three sleeves
        logger.info("\n[2/4] Running Phase-1 for all three sleeves...")
        results = {}
        stats = {}
        
        config = {
            'zscore_window': args.zscore_window,
            'clip': args.clip,
            'target_vol': args.target_vol,
            'vol_lookback': args.vol_lookback,
            'min_vol_floor': args.min_vol_floor,
            'max_leverage': args.max_leverage,
            'lag': args.lag
        }
        
        # Pack Slope Momentum
        logger.info("\n  Computing Pack Slope Momentum...")
        try:
            pack_slope_result = compute_pack_slope_momentum_phase1(
                market=market,
                start_date=args.start,
                end_date=args.end,
                **config
            )
            results['pack_slope'] = pack_slope_result
            stats['pack_slope'] = compute_summary_stats(
                pack_slope_result['portfolio_returns'],
                pack_slope_result['equity_curve']
            )
            logger.info(f"    Sharpe: {stats['pack_slope']['Sharpe']:.4f}")
        except Exception as e:
            logger.error(f"    Error computing pack slope: {e}", exc_info=True)
        
        # Pack Curvature Momentum
        logger.info("\n  Computing Pack Curvature Momentum...")
        try:
            pack_curv_result = compute_pack_curvature_momentum_phase1(
                market=market,
                start_date=args.start,
                end_date=args.end,
                **config
            )
            results['pack_curvature'] = pack_curv_result
            stats['pack_curvature'] = compute_summary_stats(
                pack_curv_result['portfolio_returns'],
                pack_curv_result['equity_curve']
            )
            logger.info(f"    Sharpe: {stats['pack_curvature']['Sharpe']:.4f}")
        except Exception as e:
            logger.error(f"    Error computing pack curvature: {e}", exc_info=True)
        
        # Rank Fly Momentum
        logger.info("\n  Computing Rank Fly Momentum...")
        try:
            rank_fly_result = compute_rank_fly_momentum_phase1(
                market=market,
                start_date=args.start,
                end_date=args.end,
                **config
            )
            results['rank_fly'] = rank_fly_result
            stats['rank_fly'] = compute_summary_stats(
                rank_fly_result['portfolio_returns'],
                rank_fly_result['equity_curve']
            )
            logger.info(f"    Sharpe: {stats['rank_fly']['Sharpe']:.4f}")
        except Exception as e:
            logger.error(f"    Error computing rank fly: {e}", exc_info=True)
        
        # Redundancy analysis
        logger.info("\n[3/4] Computing redundancy analysis...")
        redundancy = compute_redundancy_analysis(results)
        
        logger.info("  Return correlations:")
        for key, value in redundancy.items():
            if 'corr' in key and 'returns' in key and value is not None:
                logger.info(f"    {key}: {value:.4f}")
        
        logger.info("  Signal correlations:")
        for key, value in redundancy.items():
            if 'corr' in key and 'signals' in key and value is not None:
                logger.info(f"    {key}: {value:.4f}")
        
        # Save results
        logger.info("\n[4/4] Saving results...")
        save_results(results, stats, redundancy, output_dir, args.start, args.end, config)
        
        # Generate plots
        generate_plots(results, stats, output_dir)
        
        # Update phase index for each sleeve
        for sleeve_key, sleeve_name in ATOMIC_SLEEVES.items():
            if sleeve_key in results:
                run_id = output_dir.name
                update_phase_index(META_SLEEVE, sleeve_name, "phase1", run_id=run_id)
                logger.info(f"  Updated phase_index/{META_SLEEVE}/{sleeve_name}/phase1.txt -> {run_id}")
        
        # Print summary
        print("\n" + "=" * 80)
        print("SR3 CURVE RV MOMENTUM PHASE-1 SUMMARY")
        print("=" * 80)
        
        for sleeve_key, sleeve_name in ATOMIC_SLEEVES.items():
            if sleeve_key in stats:
                s = stats[sleeve_key]
                print(f"\n{sleeve_name}:")
                print(f"  Sharpe:  {s.get('Sharpe', 0):8.4f}")
                print(f"  CAGR:    {s.get('CAGR', 0)*100:8.2f}%")
                print(f"  Vol:     {s.get('Vol', 0)*100:8.2f}%")
                print(f"  MaxDD:   {s.get('MaxDD', 0)*100:8.2f}%")
                print(f"  HitRate: {s.get('HitRate', 0):8.4f}")
        
        print("\n" + "=" * 80)
        print("REDUNDANCY ANALYSIS")
        print("=" * 80)
        print("Return Correlations:")
        for key, value in redundancy.items():
            if 'corr' in key and 'returns' in key and value is not None:
                print(f"  {key}: {value:.4f}")
        print("\nSignal Correlations:")
        for key, value in redundancy.items():
            if 'corr' in key and 'signals' in key and value is not None:
                print(f"  {key}: {value:.4f}")
        
        print(f"\nResults saved to: {output_dir}")
        print("=" * 80)
        
        # Close market connection
        market.close()
        
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

