"""
CSMOM Phase-1 Diagnostics

Diagnostics for Phase-1 Cross-Sectional Momentum with multi-horizon z-scored momentum
and volatility-aware cross-sectional ranking.
"""

import pandas as pd
import numpy as np
from typing import Sequence, Optional, Dict
from pathlib import Path
from datetime import datetime
import json
import logging

from src.agents.data_broker import MarketData
from src.agents.strat_cross_sectional import CSMOMPhase1, CSMOMConfig
from src.diagnostics.csmom_sanity import compute_summary_stats, generate_plots

logger = logging.getLogger(__name__)


def run_csmom_phase1(
    start: str,
    end: str,
    universe: Sequence[str] | None = None,
    outdir: str | None = None,
    lookbacks: Sequence[int] = (63, 126, 252),
    weights: Sequence[float] = (0.4, 0.35, 0.25),
    vol_lookback: int = 63,
    rebalance_freq: str = "D",
    neutralize_cross_section: bool = True,
    clip_score: float = 3.0
) -> Dict:
    """
    Run Phase-1 CSMOM diagnostics.
    
    Args:
        start: Start date (YYYY-MM-DD)
        end: End date (YYYY-MM-DD)
        universe: List of symbols (None = use MarketData default)
        outdir: Output directory (None = auto-generate timestamp)
        lookbacks: Lookback periods for multi-horizon momentum
        weights: Weights for each horizon (will be normalized)
        vol_lookback: Lookback for volatility calculation
        rebalance_freq: Rebalance frequency ("D" for daily, "W" for weekly)
        neutralize_cross_section: Whether to z-score cross-sectionally
        clip_score: Z-score clipping threshold
        
    Returns:
        Dict with summary metrics and results
    """
    md = MarketData()
    try:
        if universe is None:
            symbols = list(md.universe)
        else:
            symbols = list(universe)
        
        logger.info(f"[CSMOMPhase1] Using universe: {symbols}")
        
        # Create config
        cfg = CSMOMConfig(
            lookbacks=lookbacks,
            weights=weights,
            vol_lookback=vol_lookback,
            rebalance_freq=rebalance_freq,
            neutralize_cross_section=neutralize_cross_section,
            clip_score=clip_score,
        )
        
        # Create strategy
        strat = CSMOMPhase1(cfg)
        
        # Get returns (simple returns for P&L calculation)
        rets = md.get_returns(symbols, start=start, end=end, method="simple")
        
        if rets.empty:
            raise ValueError("No returns data available")
        
        # Compute signals
        signals = strat.compute_signals(md, start=start, end=end, universe=symbols)
        
        # Align signals with returns
        common_idx = signals.index.intersection(rets.index)
        signals = signals.loc[common_idx]
        rets = rets.loc[common_idx]
        
        # Convert signals to market-neutral weights using cross-sectional scores
        # Normalize by sum of absolute values to ensure dollar neutrality
        weights = signals.div(signals.abs().sum(axis=1), axis=0).fillna(0.0)
        
        # Apply weights with 1-day lag (rebalance at close, apply next day)
        weights_shifted = weights.shift(1).fillna(0.0)
        
        # Align again after shift
        common_idx = weights_shifted.index.intersection(rets.index)
        weights_shifted = weights_shifted.loc[common_idx]
        rets = rets.loc[common_idx]
        
        # Portfolio returns
        portfolio_rets = (weights_shifted * rets).sum(axis=1)
        
        # Equity curve
        equity = (1 + portfolio_rets).cumprod()
        if len(equity) > 0:
            equity.iloc[0] = 1.0
        
        # Per-asset strategy returns
        weights_ffill = weights_shifted.ffill().fillna(0.0)
        asset_strategy_returns = weights_ffill * rets
        
        # Compute stats
        stats = compute_summary_stats(
            portfolio_returns=portfolio_rets,
            equity_curve=equity,
            asset_strategy_returns=asset_strategy_returns
        )
        
        # Generate output directory
        if outdir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            outdir = f"reports/sanity_checks/csmom/phase1/{timestamp}"
        
        outdir_path = Path(outdir)
        outdir_path.mkdir(parents=True, exist_ok=True)
        
        # Save results
        save_results(
            portfolio_rets=portfolio_rets,
            equity=equity,
            asset_returns=rets,
            asset_strategy_returns=asset_strategy_returns,
            weights=weights,
            signals=signals,
            stats=stats,
            config=cfg,
            start=start,
            end=end,
            universe=symbols,
            outdir=outdir_path
        )
        
        # Generate plots
        from src.diagnostics.csmom_sanity import CSMOMResults
        results_obj = CSMOMResults(
            config=None,  # Not used in plots
            portfolio_returns=portfolio_rets,
            equity_curve=equity,
            asset_returns=rets,
            weights=weights,
            asset_strategy_returns=asset_strategy_returns,
            summary=stats['portfolio'],
            per_asset=stats['per_asset']
        )
        generate_plots(results_obj, outdir_path)
        
        logger.info(f"[CSMOMPhase1] Results saved to {outdir_path}")
        
        return {
            'summary': stats['portfolio'],
            'per_asset': stats['per_asset'],
            'outdir': str(outdir_path)
        }
        
    finally:
        md.close()


def save_results(
    portfolio_rets: pd.Series,
    equity: pd.Series,
    asset_returns: pd.DataFrame,
    asset_strategy_returns: pd.DataFrame,
    weights: pd.DataFrame,
    signals: pd.DataFrame,
    stats: Dict,
    config: CSMOMConfig,
    start: str,
    end: str,
    universe: Sequence[str],
    outdir: Path
):
    """Save Phase-1 results to disk."""
    # Portfolio returns
    portfolio_rets_df = pd.DataFrame({
        'date': portfolio_rets.index,
        'ret': portfolio_rets.values
    })
    portfolio_rets_df.to_csv(outdir / "portfolio_returns.csv", index=False)
    
    # Equity curve
    equity_df = pd.DataFrame({
        'date': equity.index,
        'equity': equity.values
    })
    equity_df.to_csv(outdir / "equity_curve.csv", index=False)
    
    # Asset returns
    asset_returns.to_csv(outdir / "asset_returns.csv")
    
    # Asset strategy returns
    asset_strategy_returns.to_csv(outdir / "asset_strategy_returns.csv")
    
    # Weights
    weights.to_csv(outdir / "weights.csv")
    
    # Signals
    signals.to_csv(outdir / "signals.csv")
    
    # Per-asset stats
    stats['per_asset'].to_csv(outdir / "per_asset_stats.csv")
    
    # Summary metrics
    summary_df = pd.DataFrame([stats['portfolio']])
    summary_df.to_csv(outdir / "summary_metrics.csv", index=False)
    
    # Meta.json
    meta = {
        'type': 'csmom_phase1',
        'lookbacks': list(config.lookbacks),
        'weights': list(config.weights),
        'vol_lookback': config.vol_lookback,
        'rebalance_freq': config.rebalance_freq,
        'neutralize_cross_section': config.neutralize_cross_section,
        'clip_score': config.clip_score,
        'universe': list(universe),
        'n_assets': len(universe),
        'n_days': len(portfolio_rets),
        'date_range': {
            'start': str(portfolio_rets.index.min()),
            'end': str(portfolio_rets.index.max())
        },
        'portfolio_metrics': stats['portfolio']
    }
    
    with open(outdir / 'meta.json', 'w') as f:
        json.dump(meta, f, indent=2, default=str)
    
    logger.info(f"[CSMOMPhase1] Saved results to {outdir}")

