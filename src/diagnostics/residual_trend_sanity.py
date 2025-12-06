"""
Residual Trend Sign-Only Sanity Check

A Phase-0 diagnostic for validating the residual trend idea:
- Long-horizon trend (e.g., 252 days) minus short-term movement (e.g., 21 days)
- Sign-only signals, equal-weighted across assets
- No overlays, no vol targeting, no z-scoring

Strategy:
- For each asset and date t:
  - Compute long-horizon log return: long_ret = log(price_t / price_{t-L_long})
  - Compute short-horizon log return: short_ret = log(price_t / price_{t-L_short})
  - Residual return: resid_ret = long_ret - short_ret
  - Signal: sign(resid_ret) → +1, 0, or -1
- Equal-weight portfolio, daily rebalancing

This is a diagnostic tool to validate that the residual trend idea has positive alpha
before adding complexity (z-scoring, vol targeting, etc.).
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)


def run_residual_trend(
    prices: pd.DataFrame,
    long_lookback: int = 252,
    short_lookback: int = 21,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    universe: Optional[List[str]] = None,
    zero_threshold: float = 1e-8
) -> Dict[str, Any]:
    """
    Run residual trend strategy for given lookback periods.
    
    Args:
        prices: Wide DataFrame [date x symbol] of continuous adjusted closes
        long_lookback: Long-horizon lookback window in trading days (default: 252)
        short_lookback: Short-horizon lookback window in trading days (default: 21)
        start_date: Start date for backtest (YYYY-MM-DD)
        end_date: End date for backtest (YYYY-MM-DD)
        universe: List of symbols to include (if None, uses all columns)
        zero_threshold: Threshold for treating residual return as zero
        
    Returns:
        Dict with:
        - 'portfolio_returns': pd.Series of daily portfolio returns
        - 'equity_curve': pd.Series of cumulative equity
        - 'asset_returns': pd.DataFrame of daily asset returns
        - 'asset_strategy_returns': pd.DataFrame of per-asset strategy returns
        - 'positions': pd.DataFrame of daily positions (signs)
        - 'residual_returns': pd.DataFrame of residual returns used for signals
        - 'metrics': Dict with {cagr, vol, sharpe, maxdd, hit_rate, n_days, years}
    """
    results = compute_residual_trend(
        prices=prices,
        long_lookback=long_lookback,
        short_lookback=short_lookback,
        start_date=start_date,
        end_date=end_date,
        universe=universe,
        zero_threshold=zero_threshold
    )
    
    # Compute metrics
    stats = compute_summary_stats(
        portfolio_returns=results['portfolio_returns'],
        equity_curve=results['equity_curve'],
        asset_strategy_returns=results['asset_strategy_returns']
    )
    
    # Add metrics to results
    results['metrics'] = stats['portfolio']
    results['per_asset_stats'] = stats['per_asset']
    
    return results


def compute_residual_trend(
    prices: pd.DataFrame,
    long_lookback: int = 252,
    short_lookback: int = 21,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    universe: Optional[List[str]] = None,
    zero_threshold: float = 1e-8
) -> Dict:
    """
    Compute residual trend strategy returns.
    
    Args:
        prices: DataFrame of continuous prices [date x symbol]
        long_lookback: Long-horizon lookback period in days (default: 252)
        short_lookback: Short-horizon lookback period in days (default: 21)
        start_date: Start date for backtest (YYYY-MM-DD)
        end_date: End date for backtest (YYYY-MM-DD)
        universe: List of symbols to include (if None, uses all columns)
        zero_threshold: Threshold for treating residual return as zero
        
    Returns:
        Dict with:
        - 'portfolio_returns': pd.Series of daily portfolio returns
        - 'equity_curve': pd.Series of cumulative equity (starts at 1.0)
        - 'asset_returns': pd.DataFrame of daily asset returns
        - 'asset_strategy_returns': pd.DataFrame of per-asset strategy returns
        - 'positions': pd.DataFrame of daily positions (signs)
        - 'residual_returns': pd.DataFrame of residual returns used for signals
    """
    # Filter by universe if specified (do this BEFORE date filtering to preserve history)
    if universe:
        # Check which symbols are available
        available = [s for s in universe if s in prices.columns]
        if not available:
            raise ValueError(f"None of the specified universe symbols are available in prices")
        prices = prices[available]
        logger.info(f"Using {len(available)} assets from universe: {available}")
    else:
        logger.info(f"Using all {len(prices.columns)} assets from prices")
    
    # Align all assets on common date index (inner join)
    prices = prices.dropna(how='all')  # Remove rows where all assets are NaN
    logger.info(f"Price data shape after alignment: {prices.shape}")
    
    # IMPORTANT: We need to keep prices BEFORE start_date to compute lookback returns
    # But we'll filter the final returns to only include dates >= start_date
    # So we compute everything first, then filter at the end
    
    # Convert to log prices for log return calculation
    log_prices = np.log(prices)
    
    # Compute daily simple returns (using all available data for now)
    # r_t = price_t / price_{t-1} - 1
    daily_returns = prices.pct_change(fill_method=None).dropna()
    
    if daily_returns.empty:
        raise ValueError("No valid daily returns computed")
    
    logger.info(f"Daily returns computed: {daily_returns.shape}")
    
    # Compute long-horizon log returns
    # long_ret_t = log(price_t / price_{t-L_long}) = log(price_t) - log(price_{t-L_long})
    long_log_returns = log_prices - log_prices.shift(long_lookback)
    
    # Compute short-horizon log returns
    # short_ret_t = log(price_t / price_{t-L_short}) = log(price_t) - log(price_{t-L_short})
    short_log_returns = log_prices - log_prices.shift(short_lookback)
    
    # Compute residual returns: resid_ret = long_ret - short_ret
    residual_returns = long_log_returns - short_log_returns
    
    # Don't drop NaN rows here - we'll handle NaN per asset later
    # This preserves all dates where at least some assets have data
    logger.info(f"Residual returns computed: {residual_returns.shape}")
    
    # Shift residual returns by 1 day so we don't use same-day info for trading
    # signal_basis_t = resid_ret_{t-1}
    signal_basis = residual_returns.shift(1)
    
    # Get all dates from prices DataFrame (this is our source of truth for trading dates)
    # We want to use ALL trading days from the price data, not just days where we have returns
    all_trading_days = prices.index
    
    # Filter by date range FIRST (before alignment)
    if start_date:
        start_dt = pd.to_datetime(start_date)
        all_trading_days = all_trading_days[all_trading_days >= start_dt]
    
    if end_date:
        end_dt = pd.to_datetime(end_date)
        all_trading_days = all_trading_days[all_trading_days <= end_dt]
    
    if len(all_trading_days) == 0:
        raise ValueError("No data available after date filtering")
    
    # Align signal_basis and daily_returns to all_trading_days
    # Use reindex to include all dates, filling NaN where signals don't exist
    signal_basis = signal_basis.reindex(all_trading_days)
    daily_returns = daily_returns.reindex(all_trading_days)
    
    # For dates/assets where signal_basis is NaN, set to 0 (no position)
    # This allows us to trade on all dates, even if some assets don't have signals
    signal_basis = signal_basis.fillna(0.0)
    
    # For dates/assets where daily_returns is NaN, set to 0 (no return)
    # This handles cases where we have a price but no return (e.g., first day of data)
    daily_returns = daily_returns.fillna(0.0)
    
    logger.info(f"Aligned data after date filtering: {len(all_trading_days)} days")
    
    # Generate sign-only positions
    # position_t = sign(signal_basis_t)
    # +1 if > 0, -1 if < 0, 0 if abs < threshold
    positions = signal_basis.copy()
    positions[positions.abs() < zero_threshold] = 0.0
    positions[positions > zero_threshold] = 1.0
    positions[positions < -zero_threshold] = -1.0
    
    # Compute per-asset strategy returns
    # strategy_ret_asset_t = position_t * daily_return_t
    asset_strategy_returns = positions * daily_returns
    
    # Aggregate to portfolio (equal-weight across assets each day)
    # For each date, count active assets (non-zero positions)
    # Each active asset gets weight 1 / n_active
    active_counts = (positions != 0).sum(axis=1)
    
    # Portfolio return = sum(weight * asset_return) where weight = 1/n_active if position != 0, else 0
    portfolio_returns = asset_strategy_returns.sum(axis=1) / active_counts.replace(0, 1)  # Avoid division by zero
    
    # If no active assets on a day, portfolio return is 0
    portfolio_returns[active_counts == 0] = 0.0
    
    # Compute equity curve (cumulative)
    # Standard formula: equity_t = equity_{t-1} * (1 + return_t)
    if len(portfolio_returns) > 0:
        equity_curve = (1 + portfolio_returns).cumprod()
    else:
        equity_curve = pd.Series(dtype=float)
    
    # Get residual returns for the final date range (for reference)
    residual_returns_final = residual_returns.reindex(all_trading_days)
    
    return {
        'portfolio_returns': portfolio_returns,
        'equity_curve': equity_curve,
        'asset_returns': daily_returns,
        'asset_strategy_returns': asset_strategy_returns,
        'positions': positions,
        'residual_returns': residual_returns_final
    }


def compute_summary_stats(
    portfolio_returns: pd.Series,
    equity_curve: pd.Series,
    asset_strategy_returns: pd.DataFrame
) -> Dict:
    """
    Compute summary statistics for the residual trend strategy.
    
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


def save_results(
    results: Dict,
    stats: Dict,
    output_dir: Path,
    start_date: Optional[str],
    end_date: Optional[str],
    long_lookback: int,
    short_lookback: int,
    universe: Optional[List[str]]
):
    """
    Save all results to output directory.
    
    Saves:
    - portfolio_returns.csv
    - equity_curve.csv
    - asset_strategy_returns.csv
    - per_asset_stats.csv
    - meta.json
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save portfolio returns
    portfolio_returns_df = pd.DataFrame({
        'date': results['portfolio_returns'].index,
        'ret': results['portfolio_returns'].values
    })
    portfolio_returns_df.to_csv(output_dir / 'portfolio_returns.csv', index=False)
    
    # Save equity curve
    equity_curve_df = pd.DataFrame({
        'date': results['equity_curve'].index,
        'equity': results['equity_curve'].values
    })
    equity_curve_df.to_csv(output_dir / 'equity_curve.csv', index=False)
    
    # Save asset strategy returns
    results['asset_strategy_returns'].to_csv(output_dir / 'asset_strategy_returns.csv')
    
    # Save per-asset stats
    if not stats['per_asset'].empty:
        stats['per_asset'].to_csv(output_dir / 'per_asset_stats.csv')
    
    # Save meta
    meta = {
        'start_date': start_date,
        'end_date': end_date,
        'long_lookback': long_lookback,
        'short_lookback': short_lookback,
        'universe': universe if universe else 'all',
        'n_assets': len(results['asset_strategy_returns'].columns),
        'n_days': len(results['portfolio_returns']),
        'portfolio_metrics': stats['portfolio']
    }
    
    with open(output_dir / 'meta.json', 'w') as f:
        json.dump(meta, f, indent=2, default=str)
    
    logger.info(f"Results saved to {output_dir}")


def generate_plots(
    results: Dict,
    stats: Dict,
    output_dir: Path,
    prices: pd.DataFrame
):
    """
    Generate diagnostic plots.
    
    Plots:
    1. Equity curve
    2. Return histogram
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
    
    # Portfolio equity curve
    ax.plot(results['equity_curve'].index, results['equity_curve'].values, 
            label='Portfolio', linewidth=2, color='black')
    
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    ax.set_title('Residual Trend: Cumulative Equity Curve')
    ax.set_xlabel('Date')
    ax.set_ylabel('Cumulative Equity')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'equity_curve.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved equity_curve.png")
    
    # 2. Return histogram
    fig, ax = plt.subplots(figsize=(10, 5))
    
    portfolio_ret = results['portfolio_returns'].dropna()
    ax.hist(portfolio_ret.values, bins=50, alpha=0.7, edgecolor='black')
    ax.axvline(x=0, color='red', linestyle='--', alpha=0.5)
    ax.set_title(f'Portfolio Returns Distribution\n(Mean={portfolio_ret.mean():.4f}, Std={portfolio_ret.std():.4f})')
    ax.set_xlabel('Daily Return')
    ax.set_ylabel('Frequency')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'return_histogram.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved return_histogram.png")

