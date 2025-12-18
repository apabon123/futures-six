"""
VRP Convergence Phase-1 Diagnostics

Diagnostics for Phase-1 VRP Convergence strategy with z-scored convergence spread.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict
from pathlib import Path
from datetime import datetime
import json
import logging
import duckdb

from src.agents.data_broker import MarketData
from src.agents.strat_vrp_convergence import VRPConvergencePhase1, VRPConvergenceConfig
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


def run_vrp_convergence_phase1(
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
    Run Phase-1 VRP Convergence diagnostics.
    
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
        logger.info(f"[VRPConvergencePhase1] Running diagnostics from {start} to {end}")
        
        # Determine DB path
        if db_path is None:
            import yaml
            config_path = Path("configs/data.yaml")
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                db_path = config['db']['path']
            else:
                raise ValueError("configs/data.yaml not found and db_path not specified")
        
        # Create config
        cfg = VRPConvergenceConfig(
            zscore_window=zscore_window,
            clip=clip,
            signal_mode=signal_mode,
            target_vol=target_vol,
            vol_lookback=vol_lookback,
            vol_floor=vol_floor,
            db_path=db_path
        )
        
        # Create strategy
        strat = VRPConvergencePhase1(cfg)
        
        # Compute signals
        signals = strat.compute_signals(md, start=start, end=end)
        
        if signals.empty:
            raise ValueError("No signals generated")
        
        logger.info(f"[VRPConvergencePhase1] Generated {len(signals)} signals")
        
        # Verify short-only constraint
        pct_short = (signals < -0.01).sum() / len(signals) * 100
        pct_flat = ((signals >= -0.01) & (signals <= 0.01)).sum() / len(signals) * 100
        pct_long = (signals > 0.01).sum() / len(signals) * 100
        logger.info(f"[VRPConvergencePhase1] Signal distribution: {pct_short:.1f}% short, {pct_flat:.1f}% flat, {pct_long:.1f}% long (should be ~0%)")
        
        if pct_long > 0.1:
            logger.warning(f"[VRPConvergencePhase1] WARNING: {pct_long:.2f}% long signals detected (should be ~0% for short-only strategy)")
        else:
            logger.info(f"[VRPConvergencePhase1] ✓ Short-only constraint verified: {pct_long:.2f}% long signals (within tolerance)")
        
        # Load VX1 returns
        logger.info("[VRPConvergencePhase1] Loading VX1 returns...")
        con = open_readonly_connection(db_path)
        try:
            vx1_rets = load_vx1_returns(con, start, end)
        finally:
            con.close()
        
        if vx1_rets.empty:
            raise ValueError("No VX1 returns available")
        
        logger.info(f"[VRPConvergencePhase1] Loaded {len(vx1_rets)} VX1 returns")
        
        # Align signals with returns
        common_idx = signals.index.intersection(vx1_rets.index)
        if len(common_idx) == 0:
            raise ValueError("No overlapping dates between signals and VX1 returns")
        
        signals = signals.loc[common_idx]
        vx1_rets = vx1_rets.loc[common_idx]
        
        logger.info(f"[VRPConvergencePhase1] Aligned data: {len(common_idx)} days")
        
        # Compute volatility-targeted positions
        logger.info("[VRPConvergencePhase1] Computing vol-targeted positions...")
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
        from src.agents.feature_vrp_convergence import VRPConvergenceFeatures
        features_calc = VRPConvergenceFeatures(
            zscore_window=zscore_window,
            clip=clip,
            db_path=db_path
        )
        features = features_calc.compute(md, start_date=start, end_date=end)
        
        # Generate output directory
        if outdir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            outdir = f"data/diagnostics/vrp_convergence_phase1/{timestamp}"
        
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
    config: VRPConvergenceConfig,
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
    
    # Spread timeseries
    if not features.empty:
        spread_df = features[['vix', 'vx1', 'spread_conv', 'conv_z']].copy()
        spread_df['signal'] = signals.reindex(spread_df.index)
        spread_df['position'] = positions.reindex(spread_df.index)
        spread_df.to_csv(outdir / 'spread_conv_timeseries.csv')
        spread_df.to_parquet(outdir / 'spread_conv_timeseries.parquet')
    
    # Summary stats
    pd.DataFrame([stats['portfolio']]).to_csv(outdir / 'summary_metrics.csv', index=False)
    
    # Signal distribution diagnostics (short-only verification)
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
        'short_only_verified': bool(pct_long < 0.1),  # Should be ~0% for short-only
        'metrics': stats['portfolio']
    }
    
    with open(outdir / 'vrp_convergence_phase1_metrics.json', 'w') as f:
        json.dump(meta, f, indent=2)
    
    logger.info(f"[VRPConvergencePhase1] Saved results to {outdir}")


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
    ax.plot(equity.index, equity.values, label='VRP Convergence Equity', linewidth=1.5)
    ax.set_xlabel('Date')
    ax.set_ylabel('Equity')
    ax.set_title('VRP Convergence Phase-1: Equity Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outdir / 'equity_curve.png', dpi=150)
    plt.close()
    
    # Plot 2: Z-score vs signal (matching VRP-Core diagnostics)
    if not features.empty:
        fig, axes = plt.subplots(3, 1, figsize=(14, 12))
        
        # Spread over time
        common_idx = features.index.intersection(signals.index)
        if len(common_idx) > 0:
            axes[0].plot(common_idx, features.loc[common_idx, 'spread_conv'], 
                        label='Convergence Spread (VIX - VX1)', alpha=0.7, color='blue')
            axes[0].axhline(0, color='black', linestyle='--', alpha=0.3)
            axes[0].set_xlabel('Date')
            axes[0].set_ylabel('Spread (vol points)')
            axes[0].set_title('Convergence Spread Over Time')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # Z-scored spread
            axes[1].plot(common_idx, features.loc[common_idx, 'conv_z'], 
                         label='Z-Scored Spread (conv_z)', alpha=0.7, color='green')
            axes[1].axhline(0, color='black', linestyle='--', alpha=0.3)
            axes[1].set_xlabel('Date')
            axes[1].set_ylabel('Z-Score')
            axes[1].set_title('Z-Scored Convergence Spread')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            
            # Signals (timeseries)
            axes[2].plot(common_idx, signals.loc[common_idx], 
                         label='Signal (tanh(-conv_z / 2.0))', alpha=0.7, color='purple')
            axes[2].axhline(0, color='black', linestyle='--', alpha=0.3)
            axes[2].set_xlabel('Date')
            axes[2].set_ylabel('Signal')
            axes[2].set_title('VRP Convergence Signals Over Time')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(outdir / 'spread_z_and_signals.png', dpi=150)
        plt.close()
    
    # Plot 3: PnL histogram (matching VRP-Core)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(portfolio_rets.dropna(), bins=100, alpha=0.7, edgecolor='black')
    ax.axvline(portfolio_rets.mean(), color='red', linestyle='--', 
               label=f'Mean: {portfolio_rets.mean():.4f}')
    ax.set_xlabel('Daily Return')
    ax.set_ylabel('Frequency')
    ax.set_title('VRP Convergence: PnL Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outdir / 'pnl_histogram.png', dpi=150)
    plt.close()
    
    logger.info(f"[VRPConvergencePhase1] Generated plots in {outdir}")


def register_phase_index(
    start: str,
    end: str,
    outdir: Path,
    stats: Dict
):
    """Register Phase-1 run in phase_index."""
    phase_index_dir = Path("reports/phase_index/vrp")
    phase_index_dir.mkdir(parents=True, exist_ok=True)
    
    phase1_file = phase_index_dir / "vrp_convergence_phase1.txt"
    with open(phase1_file, 'w') as f:
        f.write(f"# Phase-1: VRP Convergence Z-Scored Strategy\n")
        f.write(f"# Registered: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"start_date: {start}\n")
        f.write(f"end_date: {end}\n")
        f.write(f"sharpe: {stats['portfolio'].get('Sharpe', float('nan')):.4f}\n")
        f.write(f"cagr: {stats['portfolio'].get('CAGR', float('nan')):.4f}\n")
        f.write(f"max_dd: {stats['portfolio'].get('MaxDD', float('nan')):.4f}\n")
        f.write(f"path: {outdir}\n")
    
    logger.info(f"[VRPConvergencePhase1] Registered in: {phase1_file}")

