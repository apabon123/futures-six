"""
CSMOM Sign-Only Sanity Check

A deliberately simple, academic-style cross-sectional momentum strategy to verify that
the data and P&L machinery are working correctly for cross-sectional ranking.

Strategy:
- For each rebalance date, compute k-day return across universe
- Rank assets by return
- Long top fraction (e.g., top 33%), short bottom fraction (e.g., bottom 33%)
- Equal notional within long and within short
- Net exposure ≈ 0
- Daily rebalancing, no vol targeting or overlays

This is a diagnostic tool to answer: "Does a very simple cross-sectional momentum strategy
show reasonable positive Sharpe on our DataBento-driven continuous futures, or is
the alpha gone because our data / roll / P&L pipeline is wrong?"

If sign-only CSMOM shows Sharpe ≥ 0.2, then the cross-sectional momentum edge is validated.
If it shows negative or near-zero Sharpe, we likely have data/roll/P&L issues or the edge is gone.
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple, List, Dict, Any, Sequence
from pathlib import Path
from dataclasses import dataclass
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class CSMOMConfig:
    """Configuration for sign-only cross-sectional momentum."""
    lookback: int              # e.g. 126
    top_frac: float            # e.g. 0.33
    bottom_frac: float         # e.g. 0.33
    rebalance: str             # "D" for daily
    universe: Tuple[str, ...]  # symbols to trade


@dataclass
class CSMOMResults:
    """Results from sign-only cross-sectional momentum."""
    config: CSMOMConfig
    portfolio_returns: pd.Series
    equity_curve: pd.Series
    asset_returns: pd.DataFrame
    weights: pd.DataFrame
    asset_strategy_returns: pd.DataFrame
    summary: Dict[str, Any]
    per_asset: pd.DataFrame


def compute_sign_only_csmom(
    md,
    config: CSMOMConfig,
    start: str,
    end: str,
) -> CSMOMResults:
    """
    Sign-only cross-sectional momentum:
      - For each rebalance date, compute k-day log return across universe
      - Rank assets by return
      - Long top_frac, short bottom_frac, equal weights
      - Daily rebalancing, no vol targeting or overlays
    
    Args:
        md: MarketData instance
        config: CSMOMConfig with lookback, top_frac, bottom_frac, universe
        start: Start date (YYYY-MM-DD)
        end: End date (YYYY-MM-DD)
        
    Returns:
        CSMOMResults with portfolio returns, equity curve, weights, stats
    """
    logger.info(f"Computing sign-only CSMOM: lookback={config.lookback}, top_frac={config.top_frac}, bottom_frac={config.bottom_frac}")
    
    # 1) Get log returns panel for the universe
    # We need data BEFORE start_date to compute lookback returns, so get all available data first
    # Then filter dates at the end
    rets = md.get_returns(config.universe, method="log")
    
    if rets.empty:
        raise ValueError("No returns data available")
    
    logger.info(f"Returns data shape: {rets.shape}")
    logger.info(f"Date range: {rets.index.min()} to {rets.index.max()}")
    
    # 2) Compute k-day rolling returns (cumulative log returns)
    # For log returns, k-day return = sum of last k daily log returns
    kret = rets.rolling(window=config.lookback, min_periods=1).sum()
    
    # Shift by 1 day to avoid look-ahead bias (use yesterday's k-day return for today's position)
    kret_shifted = kret.shift(1)
    
    # 3) Build weights: for each date, rank assets and assign long/short positions
    weights = pd.DataFrame(index=rets.index, columns=rets.columns, dtype=float)
    weights[:] = 0.0
    
    # Process each date
    for date in rets.index:
        # Get k-day returns for this date (already shifted, so this is yesterday's return)
        kret_date = kret_shifted.loc[date]
        
        # Drop NaNs for ranking
        valid_kret = kret_date.dropna()
        
        if len(valid_kret) < 2:
            # Need at least 2 assets to rank
            continue
        
        # Rank assets by return (ascending: lowest return = rank 1, highest = rank N)
        ranks = valid_kret.rank(method='average', ascending=True)
        n_valid = len(valid_kret)
        
        # Calculate cutoff positions
        n_long = max(1, int(np.ceil(n_valid * config.top_frac)))
        n_short = max(1, int(np.ceil(n_valid * config.bottom_frac)))
        
        # Determine quantile cutoffs
        long_threshold = n_valid - n_long + 1  # Top performers
        short_threshold = n_short  # Bottom performers
        
        # Assign positions
        long_mask = ranks >= long_threshold
        short_mask = ranks <= short_threshold
        
        # Equal weights: each long = +1/n_long, each short = -1/n_short
        n_long_actual = long_mask.sum()
        n_short_actual = short_mask.sum()
        
        if n_long_actual > 0 and n_short_actual > 0:
            # Equal notional long and short
            weights.loc[date, valid_kret[long_mask].index] = 1.0 / n_long_actual
            weights.loc[date, valid_kret[short_mask].index] = -1.0 / n_short_actual
        
        # Assets not in long or short get 0 (already set)
    
    # 4) Filter by date range (after computing weights)
    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end)
    mask = (weights.index >= start_dt) & (weights.index <= end_dt)
    weights = weights.loc[mask]
    rets = rets.loc[mask]
    
    if len(weights) == 0:
        raise ValueError("No data available after date filtering")
    
    logger.info(f"Filtered to {len(weights)} days: {weights.index.min()} to {weights.index.max()}")
    
    # 5) Compute portfolio returns: shift weights by 1 day (rebalance at close, apply next day)
    weights_shifted = weights.shift(1).fillna(0.0)
    
    # Align indices
    common_idx = weights_shifted.index.intersection(rets.index)
    weights_shifted = weights_shifted.loc[common_idx]
    rets_aligned = rets.loc[common_idx]
    
    # Portfolio daily returns: (weights * returns).sum(axis=1)
    # For log returns, we need to convert to simple returns first
    # Or we can work with log returns directly if we're careful
    # Actually, let's use simple returns for P&L calculation to be consistent with TSMOM
    rets_simple = md.get_returns(config.universe, start=start, end=end, method="simple")
    rets_simple = rets_simple.loc[common_idx]
    
    portfolio_returns = (weights_shifted * rets_simple).sum(axis=1)
    
    # 6) Compute equity curve
    if len(portfolio_returns) > 0:
        equity_curve = (1 + portfolio_returns).cumprod()
        # Ensure first value is 1.0 (or close to it if we have returns on first day)
        if len(equity_curve) > 0:
            equity_curve.iloc[0] = 1.0
    else:
        equity_curve = pd.Series(dtype=float)
    
    # 7) Compute per-asset strategy returns (for attribution)
    # Forward fill weights to daily frequency
    weights_ffill = weights_shifted.ffill().fillna(0.0)
    asset_strategy_returns = weights_ffill * rets_simple
    
    # 8) Compute summary stats
    stats = compute_summary_stats(
        portfolio_returns=portfolio_returns,
        equity_curve=equity_curve,
        asset_strategy_returns=asset_strategy_returns
    )
    
    return CSMOMResults(
        config=config,
        portfolio_returns=portfolio_returns,
        equity_curve=equity_curve,
        asset_returns=rets_simple,
        weights=weights,
        asset_strategy_returns=asset_strategy_returns,
        summary=stats['portfolio'],
        per_asset=stats['per_asset']
    )


def compute_summary_stats(
    portfolio_returns: pd.Series,
    equity_curve: pd.Series,
    asset_strategy_returns: pd.DataFrame
) -> Dict:
    """
    Compute summary statistics for the sign-only CSMOM strategy.
    
    Returns:
        Dict with portfolio metrics and per-asset stats
    """
    if len(portfolio_returns) == 0:
        return {
            'portfolio': {},
            'per_asset': pd.DataFrame()
        }
    
    # Portfolio metrics
    n_days = len(portfolio_returns)
    years = n_days / 252.0
    
    # CAGR
    if len(equity_curve) >= 2 and years > 0:
        equity_start = equity_curve.iloc[0]
        equity_end = equity_curve.iloc[-1]
        if equity_start > 0:
            cagr = (equity_end / equity_start) ** (1 / years) - 1
        else:
            cagr = float('nan')
    else:
        cagr = float('nan')
    
    # Annualized volatility
    vol = portfolio_returns.std() * (252 ** 0.5)
    
    # Sharpe ratio (assuming 0% risk-free rate)
    if portfolio_returns.std() != 0:
        sharpe = portfolio_returns.mean() / portfolio_returns.std() * (252 ** 0.5)
    else:
        sharpe = float('nan')
    
    # Max drawdown
    if len(equity_curve) >= 2:
        running_max = equity_curve.cummax()
        dd = (equity_curve / running_max) - 1.0
        max_dd = dd.min()
    else:
        max_dd = float('nan')
    
    # Hit rate
    hit_rate = (portfolio_returns > 0).sum() / len(portfolio_returns) if len(portfolio_returns) > 0 else float('nan')
    
    portfolio_metrics = {
        'CAGR': cagr,
        'Vol': vol,
        'Sharpe': sharpe,
        'MaxDD': max_dd,
        'HitRate': hit_rate,
        'n_days': n_days,
        'years': years
    }
    
    # Per-asset stats
    per_asset_rows = []
    for sym in asset_strategy_returns.columns:
        asset_ret = asset_strategy_returns[sym].dropna()
        
        if len(asset_ret) == 0:
            continue
        
        # Annualized return
        ann_ret = asset_ret.mean() * 252
        
        # Annualized vol
        ann_vol = asset_ret.std() * (252 ** 0.5)
        
        # Sharpe
        if ann_vol != 0:
            sharpe = ann_ret / ann_vol
        else:
            sharpe = float('nan')
        
        per_asset_rows.append({
            'symbol': sym,
            'AnnRet': ann_ret,
            'AnnVol': ann_vol,
            'Sharpe': sharpe
        })
    
    per_asset_df = pd.DataFrame(per_asset_rows)
    if not per_asset_df.empty:
        per_asset_df = per_asset_df.set_index('symbol')
    
    return {
        'portfolio': portfolio_metrics,
        'per_asset': per_asset_df
    }


def save_results(results: CSMOMResults, outdir: str) -> Dict[str, str]:
    """
    Save CSVs + meta.json + basic plots, mirroring tsmom_sanity.save_results.
    
    Args:
        results: CSMOMResults object
        outdir: Output directory path
        
    Returns:
        Dict mapping file type to path
    """
    outdir_path = Path(outdir)
    outdir_path.mkdir(parents=True, exist_ok=True)
    
    saved_files = {}
    
    # Save portfolio returns
    portfolio_returns_df = pd.DataFrame({
        'date': results.portfolio_returns.index,
        'ret': results.portfolio_returns.values
    })
    portfolio_path = outdir_path / 'portfolio_returns.csv'
    portfolio_returns_df.to_csv(portfolio_path, index=False)
    saved_files['portfolio_returns'] = str(portfolio_path)
    
    # Save equity curve
    equity_curve_df = pd.DataFrame({
        'date': results.equity_curve.index,
        'equity': results.equity_curve.values
    })
    equity_path = outdir_path / 'equity_curve.csv'
    equity_curve_df.to_csv(equity_path, index=False)
    saved_files['equity_curve'] = str(equity_path)
    
    # Save asset returns
    asset_returns_path = outdir_path / 'asset_returns.csv'
    results.asset_returns.to_csv(asset_returns_path)
    saved_files['asset_returns'] = str(asset_returns_path)
    
    # Save asset strategy returns
    asset_strategy_returns_path = outdir_path / 'asset_strategy_returns.csv'
    results.asset_strategy_returns.to_csv(asset_strategy_returns_path)
    saved_files['asset_strategy_returns'] = str(asset_strategy_returns_path)
    
    # Save weights
    weights_path = outdir_path / 'weights.csv'
    results.weights.to_csv(weights_path)
    saved_files['weights'] = str(weights_path)
    
    # Save per-asset stats
    per_asset_path = outdir_path / 'per_asset_stats.csv'
    results.per_asset.to_csv(per_asset_path)
    saved_files['per_asset_stats'] = str(per_asset_path)
    
    # Save summary metrics
    summary_path = outdir_path / 'summary_metrics.csv'
    summary_df = pd.DataFrame([results.summary])
    summary_df.to_csv(summary_path, index=False)
    saved_files['summary_metrics'] = str(summary_path)
    
    # Save meta.json
    meta = {
        'type': 'csmom_sign_only',
        'lookback': results.config.lookback,
        'top_frac': results.config.top_frac,
        'bottom_frac': results.config.bottom_frac,
        'rebalance': results.config.rebalance,
        'universe': list(results.config.universe),
        'n_assets': len(results.config.universe),
        'n_days': len(results.portfolio_returns),
        'date_range': {
            'start': str(results.portfolio_returns.index.min()),
            'end': str(results.portfolio_returns.index.max())
        },
        'portfolio_metrics': results.summary
    }
    meta_path = outdir_path / 'meta.json'
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2, default=str)
    saved_files['meta'] = str(meta_path)
    
    logger.info(f"Results saved to {outdir_path}")
    
    return saved_files


def generate_plots(
    results: CSMOMResults,
    output_dir: Path
):
    """
    Generate diagnostic plots.
    
    Plots:
    1. Equity curve
    2. Return histograms
    """
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available, skipping plots")
        return
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Equity curve
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(results.equity_curve.index, results.equity_curve.values, 
            label='Portfolio', linewidth=2, color='black')
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    ax.set_title('Sign-Only CSMOM: Cumulative Equity Curve')
    ax.set_xlabel('Date')
    ax.set_ylabel('Cumulative Equity')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'equity_curve.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved equity_curve.png")
    
    # 2. Return histograms
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Portfolio returns histogram
    portfolio_ret = results.portfolio_returns.dropna()
    axes[0].hist(portfolio_ret.values, bins=50, alpha=0.7, edgecolor='black')
    axes[0].axvline(x=0, color='red', linestyle='--', alpha=0.5)
    axes[0].set_title(f'Portfolio Returns Distribution\n(Mean={portfolio_ret.mean():.4f}, Std={portfolio_ret.std():.4f})')
    axes[0].set_xlabel('Daily Return')
    axes[0].set_ylabel('Frequency')
    axes[0].grid(True, alpha=0.3)
    
    # Per-asset returns histogram (aggregate)
    # Use pre-computed asset strategy returns
    asset_strategy_ret = results.asset_strategy_returns.values.flatten()
    asset_strategy_ret = asset_strategy_ret[~np.isnan(asset_strategy_ret)]
    
    axes[1].hist(asset_strategy_ret, bins=50, alpha=0.7, edgecolor='black', color='orange')
    axes[1].axvline(x=0, color='red', linestyle='--', alpha=0.5)
    axes[1].set_title(f'Per-Asset Strategy Returns Distribution\n(Mean={np.mean(asset_strategy_ret):.4f}, Std={np.std(asset_strategy_ret):.4f})')
    axes[1].set_xlabel('Daily Return')
    axes[1].set_ylabel('Frequency')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'return_histogram.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved return_histogram.png")


def run_sign_only_csmom(
    lookback: int,
    top_frac: float,
    bottom_frac: float,
    start: str,
    end: str,
    universe: Sequence[str] | None,
    outdir: str | None = None,
) -> CSMOMResults:
    """
    Convenience wrapper used by scripts/run_csmom_sanity.py
    
    Args:
        lookback: Lookback period in days
        top_frac: Top fraction to long (0-1)
        bottom_frac: Bottom fraction to short (0-1)
        start: Start date (YYYY-MM-DD)
        end: End date (YYYY-MM-DD)
        universe: List of symbols (None = use default from MarketData)
        outdir: Output directory (None = auto-generate timestamp)
        
    Returns:
        CSMOMResults object
    """
    from src.agents import MarketData
    
    md = MarketData()
    try:
        if universe is None:
            # Default: use MarketData universe
            symbols = tuple(md.universe)
        else:
            symbols = tuple(universe)
        
        logger.info(f"Using universe: {symbols}")
        
        cfg = CSMOMConfig(
            lookback=lookback,
            top_frac=top_frac,
            bottom_frac=bottom_frac,
            rebalance="D",
            universe=symbols,
        )
        
        results = compute_sign_only_csmom(md, cfg, start=start, end=end)
        
        # Generate output directory if not provided
        if outdir is None:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            outdir = f"reports/sanity_checks/csmom/phase0/{timestamp}"
        
        # Save results
        save_results(results, outdir=outdir)
        
        # Generate plots
        generate_plots(results, output_dir=Path(outdir))
        
        return results
    finally:
        md.close()

