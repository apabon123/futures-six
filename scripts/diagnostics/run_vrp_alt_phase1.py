#!/usr/bin/env python3
"""
VRP Alt Phase-1 Diagnostics (SHORT-ONLY)

Diagnostics for Phase-1 VRP Alt strategy with z-scored Alt-VRP spread (VIX - RV5).
This implementation enforces short-only signals: all long signals are clipped to 0.
This preserves VRP Meta-Sleeve conceptual purity: all long-vol behavior belongs in Crisis Meta-Sleeve.

Note on Phase-0 MaxDD:
VRP-Alt Phase-0 showed Sharpe ≈ 0.10 with catastrophic MaxDD (~–94%).
This mirrors VRP-Core, where Phase-0 also had severe drawdowns (≈–87%) but a clearly positive Sharpe.
In this framework, Phase-0's primary pass criterion is economic edge (Sharpe ≥ 0.1);
MaxDD is expected to be extreme for raw, unscaled short-vol signals and is addressed in Phase-1
via z-scoring and vol targeting. VRP-Alt is therefore treated as a Phase-0 economic PASS
and advanced to Phase-1 engineering.
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Optional, Dict
from datetime import datetime
import json
import logging
import duckdb

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config.backtest_window import CANONICAL_START, CANONICAL_END
from src.agents.data_broker import MarketData
from src.agents.strat_vrp_alt import VRPAltPhase1, VRPAltConfig
from src.agents.utils_db import open_readonly_connection
from src.diagnostics.tsmom_sanity import compute_summary_stats

logger = logging.getLogger(__name__)


def load_vx1_returns(
    con: duckdb.DuckDBPyConnection,
    start: str,
    end: str,
    symbol: str = "@VX=101XN"
) -> pd.Series:
    """
    Load VX1 returns from canonical DB.
    
    Args:
        con: DuckDB connection
        start: Start date (YYYY-MM-DD)
        end: End date (YYYY-MM-DD)
        symbol: VX1 symbol (default: @VX=101XN)
        
    Returns:
        Series of daily log returns indexed by date
    """
    result = con.execute(
        """
        SELECT
            timestamp::DATE AS date,
            close::DOUBLE AS close
        FROM market_data
        WHERE symbol = ?
          AND timestamp::DATE BETWEEN ? AND ?
        ORDER BY timestamp
        """,
        [symbol, start, end]
    ).df()
    
    if result.empty:
        return pd.Series(dtype=float, name='vx1_ret')
    
    # Compute log returns
    result = result.set_index('date')
    result['vx1_ret'] = np.log(result['close']).diff()
    
    return result['vx1_ret'].dropna()


def run_vrp_alt_phase1(
    start: str,
    end: str,
    outdir: str | None = None,
    zscore_window: int = 252,
    clip: float = 3.0,
    signal_mode: str = "zscore",
    target_vol: float = 0.10,
    vol_lookback: int = 63,
    vol_floor: float = 0.05,
    db_path: Optional[str] = None
) -> Dict:
    """
    Run Phase-1 VRP Alt diagnostics.
    
    Args:
        start: Start date (YYYY-MM-DD)
        end: End date (YYYY-MM-DD)
        outdir: Output directory (None = auto-generate timestamp)
        zscore_window: Z-score rolling window (days)
        clip: Z-score clipping bounds
        signal_mode: "zscore" or "tanh"
        target_vol: Target annualized volatility (default: 0.10 = 10%)
        vol_lookback: Volatility lookback for vol targeting (days)
        vol_floor: Minimum volatility floor (default: 0.05 = 5%)
        db_path: Path to canonical DuckDB (None = from config)
        
    Returns:
        Dict with summary metrics and results
    """
    md = MarketData()
    
    try:
        logger.info(f"[VRPAltPhase1] Running diagnostics from {start} to {end}")
        
        # Determine DB path
        if db_path is None:
            import yaml
            config_path = Path("configs/data.yaml")
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                db_path = config['db']['path']
            else:
                raise ValueError("configs/data.yaml not found and db_path not specified")
        
        # Create config
        cfg = VRPAltConfig(
            zscore_window=zscore_window,
            clip=clip,
            signal_mode=signal_mode,
            target_vol=target_vol,
            vol_lookback=vol_lookback,
            vol_floor=vol_floor,
            db_path=db_path
        )
        
        # Create strategy
        strat = VRPAltPhase1(cfg)
        
        # Compute signals
        signals = strat.compute_signals(md, start=start, end=end)
        
        if signals.empty:
            raise ValueError("No signals generated")
        
        logger.info(f"[VRPAltPhase1] Generated {len(signals)} signals")
        
        # Log signal distribution (short-only: should have no long signals)
        pct_short = (signals < -0.01).sum() / len(signals) * 100
        pct_flat = ((signals >= -0.01) & (signals <= 0.01)).sum() / len(signals) * 100
        pct_long = (signals > 0.01).sum() / len(signals) * 100
        logger.info(f"[VRPAltPhase1] Signal distribution (short-only): {pct_short:.1f}% short, {pct_flat:.1f}% flat, {pct_long:.1f}% long (should be ~0%)")
        
        if pct_long > 0.1:
            logger.warning(f"[VRPAltPhase1] WARNING: {pct_long:.2f}% long signals detected (should be ~0% for short-only strategy)")
        else:
            logger.info(f"[VRPAltPhase1] ✓ Short-only constraint verified: {pct_long:.2f}% long signals (within tolerance)")
        
        # Load VX1 returns
        logger.info("[VRPAltPhase1] Loading VX1 returns...")
        con = open_readonly_connection(db_path)
        try:
            vx1_rets = load_vx1_returns(con, start, end)
        finally:
            con.close()
        
        if vx1_rets.empty:
            raise ValueError("No VX1 returns available")
        
        logger.info(f"[VRPAltPhase1] Loaded {len(vx1_rets)} VX1 returns")
        
        # Align signals with returns
        common_idx = signals.index.intersection(vx1_rets.index)
        if len(common_idx) == 0:
            raise ValueError("No overlapping dates between signals and VX1 returns")
        
        signals = signals.loc[common_idx]
        vx1_rets = vx1_rets.loc[common_idx]
        
        logger.info(f"[VRPAltPhase1] Aligned data: {len(common_idx)} days")
        
        # Compute volatility-targeted positions
        logger.info("[VRPAltPhase1] Computing vol-targeted positions...")
        positions = strat.compute_positions(signals, vx1_rets)
        
        # Align positions with returns (positions may have fewer dates due to vol lookback)
        common_idx = positions.index.intersection(vx1_rets.index)
        positions = positions.loc[common_idx]
        vx1_rets = vx1_rets.loc[common_idx]
        
        # Apply signals with 1-day lag (rebalance at close, apply next day)
        positions_shifted = positions.shift(1).fillna(0.0)
        
        # Align again after shift
        common_idx = positions_shifted.index.intersection(vx1_rets.index)
        positions_shifted = positions_shifted.loc[common_idx]
        vx1_rets = vx1_rets.loc[common_idx]
        
        # Portfolio returns (directional: position * return)
        portfolio_rets = positions_shifted * vx1_rets
        
        # Equity curve
        equity = (1 + portfolio_rets).cumprod()
        if len(equity) > 0:
            equity.iloc[0] = 1.0
        
        # Compute stats
        asset_strategy_returns = pd.DataFrame({
            'VX1': portfolio_rets
        })
        
        stats = compute_summary_stats(
            portfolio_returns=portfolio_rets,
            equity_curve=equity,
            asset_strategy_returns=asset_strategy_returns
        )
        
        # Load features for timeseries output
        from src.agents.feature_vrp_alt import VRPAltFeatures
        features_calc = VRPAltFeatures(
            zscore_window=zscore_window,
            clip=clip,
            db_path=db_path
        )
        features = features_calc.compute(md, start_date=start, end_date=end)
        
        # Generate output directory
        if outdir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            outdir = f"data/diagnostics/vrp_alt_phase1/{timestamp}"
        
        outdir_path = Path(outdir)
        outdir_path.mkdir(parents=True, exist_ok=True)
        
        # Save results
        save_results(
            portfolio_rets=portfolio_rets,
            equity=equity,
            vx1_rets=vx1_rets,
            signals=signals,
            positions=positions,
            features=features,
            stats=stats,
            config=cfg,
            start=start,
            end=end,
            outdir=outdir_path
        )
        
        # Generate plots
        generate_plots(
            portfolio_rets=portfolio_rets,
            equity=equity,
            signals=signals,
            positions=positions,
            features=features,
            outdir=outdir_path
        )
        
        # Register in phase index
        register_phase_index(
            start=start,
            end=end,
            outdir=outdir_path,
            stats=stats
        )
        
        # Print summary
        print_summary(stats['portfolio'], pct_short, pct_flat, pct_long)
        
        return {
            'portfolio_returns': portfolio_rets,
            'equity': equity,
            'signals': signals,
            'positions': positions,
            'vx1_returns': vx1_rets,
            'summary': stats['portfolio'],
            'outdir': str(outdir_path)
        }
        
    finally:
        md.close()


def save_results(
    portfolio_rets: pd.Series,
    equity: pd.Series,
    vx1_rets: pd.Series,
    signals: pd.Series,
    positions: pd.Series,
    features: pd.DataFrame,
    stats: Dict,
    config: VRPAltConfig,
    start: str,
    end: str,
    outdir: Path
):
    """Save all results to output directory."""
    # Portfolio returns
    portfolio_rets.to_frame('ret').to_csv(outdir / 'portfolio_returns.csv')
    
    # Equity curve
    equity.to_frame('equity').to_csv(outdir / 'equity_curve.csv')
    
    # VX1 returns
    vx1_rets.to_frame('vx1_ret').to_csv(outdir / 'vx1_returns.csv')
    
    # Signals
    signals.to_frame('signal').to_csv(outdir / 'signals.csv')
    
    # Positions
    positions.to_frame('position').to_csv(outdir / 'positions.csv')
    
    # Alt-VRP timeseries
    if not features.empty:
        alt_vrp_df = features[['vix', 'rv5', 'vx1', 'alt_vrp', 'alt_vrp_z']].copy()
        alt_vrp_df['signal'] = signals.reindex(alt_vrp_df.index)
        alt_vrp_df['position'] = positions.reindex(alt_vrp_df.index)
        alt_vrp_df.to_csv(outdir / 'vrp_alt_phase1_timeseries.csv')
        alt_vrp_df.to_parquet(outdir / 'vrp_alt_phase1_timeseries.parquet')
    
    # Summary stats
    pd.DataFrame([stats['portfolio']]).to_csv(outdir / 'summary_metrics.csv', index=False)
    
    # Signal distribution diagnostics
    pct_short = (signals < -0.01).sum() / len(signals) * 100
    pct_flat = ((signals >= -0.01) & (signals <= 0.01)).sum() / len(signals) * 100
    pct_long = (signals > 0.01).sum() / len(signals) * 100
    
    # Meta
    meta = {
        'run_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'start_date': start,
        'end_date': end,
        'zscore_window': config.zscore_window,
        'clip': config.clip,
        'signal_mode': config.signal_mode,
        'target_vol': config.target_vol,
        'vol_lookback': config.vol_lookback,
        'vol_floor': config.vol_floor,
        'signal_distribution': {
            'pct_short': pct_short,
            'pct_flat': pct_flat,
            'pct_long': pct_long
        },
        'metrics': stats['portfolio']
    }
    
    with open(outdir / 'vrp_alt_phase1_metrics.json', 'w') as f:
        json.dump(meta, f, indent=2)
    
    logger.info(f"[VRPAltPhase1] Saved results to {outdir}")


def generate_plots(
    portfolio_rets: pd.Series,
    equity: pd.Series,
    signals: pd.Series,
    positions: pd.Series,
    features: pd.DataFrame,
    outdir: Path
):
    """Generate diagnostic plots."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available, skipping plots")
        return
    
    # Plot 1: Equity curve
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(equity.index, equity.values, label='VRP Alt Equity', linewidth=1.5)
    ax.set_xlabel('Date')
    ax.set_ylabel('Equity')
    ax.set_title('VRP Alt Phase-1: Equity Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outdir / 'equity_curve.png', dpi=150)
    plt.close()
    
    # Plot 2: Alt-VRP z-score vs signal (matching VRP-Core diagnostics)
    if not features.empty:
        fig, axes = plt.subplots(3, 1, figsize=(14, 12))
        
        # Alt-VRP spread over time
        common_idx = features.index.intersection(signals.index)
        if len(common_idx) > 0:
            axes[0].plot(common_idx, features.loc[common_idx, 'alt_vrp'], 
                        label='Alt-VRP Spread (VIX - RV5)', alpha=0.7, color='blue')
            axes[0].axhline(0, color='black', linestyle='--', alpha=0.3)
            axes[0].set_xlabel('Date')
            axes[0].set_ylabel('Spread (vol points)')
            axes[0].set_title('Alt-VRP Spread Over Time')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # Z-scored Alt-VRP
            axes[1].plot(common_idx, features.loc[common_idx, 'alt_vrp_z'], 
                         label='Z-Scored Alt-VRP (alt_vrp_z)', alpha=0.7, color='green')
            axes[1].axhline(0, color='black', linestyle='--', alpha=0.3)
            axes[1].set_xlabel('Date')
            axes[1].set_ylabel('Z-Score')
            axes[1].set_title('Z-Scored Alt-VRP Spread')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            
            # Signals (timeseries)
            axes[2].plot(common_idx, signals.loc[common_idx], 
                         label='Signal', alpha=0.7, color='purple')
            axes[2].axhline(0, color='black', linestyle='--', alpha=0.3)
            axes[2].set_xlabel('Date')
            axes[2].set_ylabel('Signal')
            axes[2].set_title('VRP Alt Signals Over Time')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(outdir / 'alt_vrp_z_and_signals.png', dpi=150)
        plt.close()
    
    # Plot 3: PnL histogram (matching VRP-Core)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(portfolio_rets.dropna(), bins=100, alpha=0.7, edgecolor='black')
    ax.axvline(portfolio_rets.mean(), color='red', linestyle='--', 
               label=f'Mean: {portfolio_rets.mean():.4f}')
    ax.set_xlabel('Daily Return')
    ax.set_ylabel('Frequency')
    ax.set_title('VRP Alt: PnL Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outdir / 'pnl_histogram.png', dpi=150)
    plt.close()
    
    logger.info(f"[VRPAltPhase1] Generated plots in {outdir}")


def register_phase_index(
    start: str,
    end: str,
    outdir: Path,
    stats: Dict
):
    """Register Phase-1 run in phase_index."""
    phase_index_dir = Path("reports/phase_index/vrp/vrp_alt")
    phase_index_dir.mkdir(parents=True, exist_ok=True)
    
    phase1_file = phase_index_dir / "phase1.txt"
    with open(phase1_file, 'w') as f:
        f.write(f"# Phase-1: VRP Alt Z-Scored Strategy\n")
        f.write(f"# Registered: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"start_date: {start}\n")
        f.write(f"end_date: {end}\n")
        f.write(f"sharpe: {stats['portfolio'].get('Sharpe', float('nan')):.4f}\n")
        f.write(f"cagr: {stats['portfolio'].get('CAGR', float('nan')):.4f}\n")
        f.write(f"max_dd: {stats['portfolio'].get('MaxDD', float('nan')):.4f}\n")
        f.write(f"path: {outdir}\n")
    
    logger.info(f"[VRPAltPhase1] Registered in: {phase1_file}")


def print_summary(metrics: Dict, pct_short: float, pct_flat: float, pct_long: float):
    """Print Phase-1 summary and pass criteria."""
    print("\n" + "=" * 80)
    print("VRP-ALT PHASE-1 DIAGNOSTICS SUMMARY")
    print("=" * 80)
    print(f"\nMetrics:")
    print(f"  CAGR:      {metrics.get('CAGR', float('nan')):8.4f} ({metrics.get('CAGR', 0)*100:6.2f}%)")
    print(f"  Vol:       {metrics.get('Vol', float('nan')):8.4f} ({metrics.get('Vol', 0)*100:6.2f}%)")
    print(f"  Sharpe:    {metrics.get('Sharpe', float('nan')):8.4f}")
    print(f"  MaxDD:     {metrics.get('MaxDD', float('nan')):8.4f} ({metrics.get('MaxDD', 0)*100:6.2f}%)")
    print(f"  HitRate:   {metrics.get('HitRate', float('nan')):8.4f} ({metrics.get('HitRate', 0)*100:6.2f}%)")
    print(f"  n_days:    {metrics.get('n_days', 0):8d}")
    print(f"  years:     {metrics.get('years', float('nan')):8.2f}")
    print(f"\nSignal Distribution:")
    print(f"  Short:     {pct_short:6.1f}%")
    print(f"  Flat:      {pct_flat:6.1f}%")
    print(f"  Long:      {pct_long:6.1f}%")
    
    # Phase-1 Pass Criteria
    sharpe = metrics.get('Sharpe', float('nan'))
    maxdd = metrics.get('MaxDD', float('nan'))
    
    print(f"\nPhase-1 Pass Criteria:")
    if not pd.isna(sharpe):
        if sharpe >= 0.20:
            print(f"  ✓ Sharpe ≥ 0.20: {sharpe:.4f} (PASS)")
        else:
            print(f"  ✗ Sharpe < 0.20: {sharpe:.4f} (FAIL)")
    else:
        print(f"  ✗ Sharpe could not be computed (FAIL)")
    
    if not pd.isna(maxdd):
        if maxdd > -0.50:
            print(f"  ✓ MaxDD < 50%: {maxdd:.4f} (PASS)")
        else:
            print(f"  ✗ MaxDD ≥ 50%: {maxdd:.4f} (FAIL)")
    else:
        print(f"  ✗ MaxDD could not be computed (FAIL)")
    
    # Check signal distribution (non-degenerate)
    if pct_short + pct_long > 20:
        print(f"  ✓ Signal distribution non-degenerate (PASS)")
    else:
        print(f"  ✗ Signal distribution degenerate (FAIL)")
    
    # Compare to Phase-0
    print(f"\nPhase-0 vs Phase-1 Comparison:")
    print(f"  Phase-0 Sharpe: 0.1008")
    print(f"  Phase-1 Sharpe: {sharpe:.4f}")
    if not pd.isna(sharpe) and sharpe > 0.1008:
        print(f"  ✓ Phase-1 Sharpe improved vs Phase-0")
    else:
        print(f"  ✗ Phase-1 Sharpe did not improve vs Phase-0")
    
    print(f"  Phase-0 MaxDD: -94.25%")
    print(f"  Phase-1 MaxDD: {maxdd*100:.2f}%")
    if not pd.isna(maxdd) and maxdd > -0.9425:
        print(f"  ✓ Phase-1 MaxDD greatly improved vs Phase-0")
    else:
        print(f"  ✗ Phase-1 MaxDD did not improve vs Phase-0")


def main():
    parser = argparse.ArgumentParser(
        description="VRP Alt Phase-1 Diagnostics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with canonical dates and default parameters
  python scripts/diagnostics/run_vrp_alt_phase1.py --start 2020-01-01 --end 2025-10-31
  
  # Run with custom parameters
  python scripts/diagnostics/run_vrp_alt_phase1.py --start 2020-01-01 --end 2025-10-31 --signal_mode tanh --target_vol 0.12
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
        "--outdir",
        type=str,
        default=None,
        help="Output directory (None = auto-generate timestamp)"
    )
    parser.add_argument(
        "--zscore_window",
        type=int,
        default=252,
        help="Z-score rolling window (days), default: 252"
    )
    parser.add_argument(
        "--clip",
        type=float,
        default=3.0,
        help="Z-score clipping bounds, default: 3.0"
    )
    parser.add_argument(
        "--signal_mode",
        type=str,
        default="zscore",
        choices=["zscore", "tanh"],
        help="Signal transformation mode, default: zscore"
    )
    parser.add_argument(
        "--target_vol",
        type=float,
        default=0.10,
        help="Target annualized volatility, default: 0.10 (10%)"
    )
    parser.add_argument(
        "--vol_lookback",
        type=int,
        default=63,
        help="Volatility lookback for vol targeting (days), default: 63"
    )
    parser.add_argument(
        "--vol_floor",
        type=float,
        default=0.05,
        help="Minimum volatility floor, default: 0.05 (5%)"
    )
    parser.add_argument(
        "--db_path",
        type=str,
        default=None,
        help="Path to canonical DuckDB (None = from config)"
    )
    
    args = parser.parse_args()
    
    try:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        logger.info("=" * 80)
        logger.info("VRP-ALT PHASE-1 DIAGNOSTICS")
        logger.info("=" * 80)
        logger.info(f"Start date: {args.start}")
        logger.info(f"End date: {args.end}")
        logger.info(f"Z-score window: {args.zscore_window}")
        logger.info(f"Signal mode: {args.signal_mode}")
        logger.info(f"Target vol: {args.target_vol}")
        
        results = run_vrp_alt_phase1(
            start=args.start,
            end=args.end,
            outdir=args.outdir,
            zscore_window=args.zscore_window,
            clip=args.clip,
            signal_mode=args.signal_mode,
            target_vol=args.target_vol,
            vol_lookback=args.vol_lookback,
            vol_floor=args.vol_floor,
            db_path=args.db_path
        )
        
        print(f"\nResults saved to: {results['outdir']}")
        print(f"Phase index: reports/phase_index/vrp/vrp_alt/phase1.txt")
        
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

