"""
VRP Core Phase-1 Diagnostics

Diagnostics for Phase-1 VRP Core strategy with z-scored VRP spread.
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
from src.agents.strat_vrp_core import VRPCorePhase1, VRPCoreConfig
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


def run_vrp_core_phase1(
    start: str,
    end: str,
    outdir: str | None = None,
    rv_lookback: int = 21,
    zscore_window: int = 252,
    clip: float = 3.0,
    signal_mode: str = "zscore",
    db_path: Optional[str] = None
) -> Dict:
    """
    Run Phase-1 VRP Core diagnostics.
    
    Args:
        start: Start date (YYYY-MM-DD)
        end: End date (YYYY-MM-DD)
        outdir: Output directory (None = auto-generate timestamp)
        rv_lookback: Realized vol lookback (days)
        zscore_window: Z-score rolling window (days)
        clip: Z-score clipping bounds
        signal_mode: "zscore" or "tanh"
        db_path: Path to canonical DuckDB (None = from config)
        
    Returns:
        Dict with summary metrics and results
    """
    md = MarketData()
    
    try:
        logger.info(f"[VRPCorePhase1] Running diagnostics from {start} to {end}")
        
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
        cfg = VRPCoreConfig(
            rv_lookback=rv_lookback,
            zscore_window=zscore_window,
            clip=clip,
            signal_mode=signal_mode,
            db_path=db_path
        )
        
        # Create strategy
        strat = VRPCorePhase1(cfg)
        
        # Compute signals
        signals = strat.compute_signals(md, start=start, end=end)
        
        if signals.empty:
            raise ValueError("No signals generated")
        
        logger.info(f"[VRPCorePhase1] Generated {len(signals)} signals")
        
        # Load VX1 returns
        logger.info("[VRPCorePhase1] Loading VX1 returns...")
        con = open_readonly_connection(db_path)
        try:
            vx1_rets = load_vx1_returns(con, start, end)
        finally:
            con.close()
        
        if vx1_rets.empty:
            raise ValueError("No VX1 returns available")
        
        logger.info(f"[VRPCorePhase1] Loaded {len(vx1_rets)} VX1 returns")
        
        # Align signals with returns
        common_idx = signals.index.intersection(vx1_rets.index)
        if len(common_idx) == 0:
            raise ValueError("No overlapping dates between signals and VX1 returns")
        
        signals = signals.loc[common_idx]
        vx1_rets = vx1_rets.loc[common_idx]
        
        logger.info(f"[VRPCorePhase1] Aligned data: {len(common_idx)} days")
        
        # Apply signals with 1-day lag (rebalance at close, apply next day)
        signals_shifted = signals.shift(1).fillna(0.0)
        
        # Align again after shift
        common_idx = signals_shifted.index.intersection(vx1_rets.index)
        signals_shifted = signals_shifted.loc[common_idx]
        vx1_rets = vx1_rets.loc[common_idx]
        
        # Portfolio returns (directional: signal * return)
        portfolio_rets = signals_shifted * vx1_rets
        
        # Equity curve
        equity = (1 + portfolio_rets).cumprod()
        if len(equity) > 0:
            equity.iloc[0] = 1.0
        
        # Compute stats
        # For single-asset strategy, asset_strategy_returns is just portfolio_rets as DataFrame
        asset_strategy_returns = pd.DataFrame({
            'VX1': portfolio_rets
        })
        
        stats = compute_summary_stats(
            portfolio_returns=portfolio_rets,
            equity_curve=equity,
            asset_strategy_returns=asset_strategy_returns
        )
        
        # Generate output directory
        if outdir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            outdir = f"data/diagnostics/vrp_core_phase1/{timestamp}"
        
        outdir_path = Path(outdir)
        outdir_path.mkdir(parents=True, exist_ok=True)
        
        # Save results
        save_results(
            portfolio_rets=portfolio_rets,
            equity=equity,
            vx1_rets=vx1_rets,
            signals=signals,
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
    stats: Dict,
    config: VRPCoreConfig,
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
    
    # Summary stats
    pd.DataFrame([stats['portfolio']]).to_csv(outdir / 'summary_metrics.csv', index=False)
    
    # Meta
    meta = {
        'run_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'start_date': start,
        'end_date': end,
        'rv_lookback': config.rv_lookback,
        'zscore_window': config.zscore_window,
        'clip': config.clip,
        'signal_mode': config.signal_mode,
        'metrics': stats['portfolio']
    }
    
    with open(outdir / 'meta.json', 'w') as f:
        json.dump(meta, f, indent=2)
    
    logger.info(f"[VRPCorePhase1] Saved results to {outdir}")


def generate_plots(
    portfolio_rets: pd.Series,
    equity: pd.Series,
    signals: pd.Series,
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
    ax.plot(equity.index, equity.values, label='VRP Core Equity', linewidth=1.5)
    ax.set_xlabel('Date')
    ax.set_ylabel('Equity')
    ax.set_title('VRP Core Phase-1: Equity Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outdir / 'equity_curve.png', dpi=150)
    plt.close()
    
    # Plot 2: Return histogram
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Portfolio returns histogram
    axes[0].hist(portfolio_rets.dropna(), bins=100, alpha=0.7, edgecolor='black')
    axes[0].axvline(portfolio_rets.mean(), color='red', linestyle='--', 
                    label=f'Mean: {portfolio_rets.mean():.4f}')
    axes[0].set_xlabel('Daily Return')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('VRP Core: Portfolio Return Distribution')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Signals histogram
    axes[1].hist(signals.dropna(), bins=50, alpha=0.7, edgecolor='black')
    axes[1].axvline(signals.mean(), color='red', linestyle='--',
                    label=f'Mean: {signals.mean():.4f}')
    axes[1].set_xlabel('Signal')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('VRP Core: Signal Distribution')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(outdir / 'distributions.png', dpi=150)
    plt.close()
    
    # Plot 3: Signals over time
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(signals.index, signals.values, label='VRP Signal', linewidth=1, alpha=0.7)
    ax.axhline(0, color='black', linestyle='--', alpha=0.5)
    ax.fill_between(signals.index, 0, signals.values, alpha=0.3)
    ax.set_xlabel('Date')
    ax.set_ylabel('Signal')
    ax.set_title('VRP Core: Signal Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outdir / 'signals_timeseries.png', dpi=150)
    plt.close()
    
    logger.info(f"[VRPCorePhase1] Generated plots in {outdir}")


def register_phase_index(
    start: str,
    end: str,
    outdir: Path,
    stats: Dict
):
    """Register Phase-1 run in phase_index."""
    phase_index_dir = Path("reports/phase_index/vrp")
    phase_index_dir.mkdir(parents=True, exist_ok=True)
    
    phase1_file = phase_index_dir / "vrp_core_phase1.txt"
    with open(phase1_file, 'w') as f:
        f.write(f"# Phase-1: VRP Core Z-Scored Strategy\n")
        f.write(f"# Registered: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"start_date: {start}\n")
        f.write(f"end_date: {end}\n")
        f.write(f"sharpe: {stats['portfolio'].get('Sharpe', float('nan')):.4f}\n")
        f.write(f"cagr: {stats['portfolio'].get('CAGR', float('nan')):.4f}\n")
        f.write(f"max_dd: {stats['portfolio'].get('MaxDD', float('nan')):.4f}\n")
        f.write(f"path: {outdir}\n")
    
    logger.info(f"[VRPCorePhase1] Registered in: {phase1_file}")

