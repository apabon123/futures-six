"""
Breakout Mid (50-100d) Sign-Only Sanity Check

A Phase-0 diagnostic for validating the breakout mid idea:
- 50-day and 100-day breakout strength combined
- Donchian-style range breakouts
- Sign-only signals, equal-weighted across assets
- No overlays, no vol targeting, no z-scoring

Strategy:
- For each asset and date t:
  - Compute 50-day breakout: (price - low_50) / (high_50 - low_50)
  - Compute 100-day breakout: (price - low_100) / (high_100 - low_100)
  - Combine: breakout_mid = 0.5 * breakout_50 + 0.5 * breakout_100
  - Signal: +1 if breakout_mid > 0.55, -1 if < 0.45, else 0
- Equal-weight portfolio, daily rebalancing

This is a diagnostic tool to validate that the breakout mid idea has positive alpha
before adding complexity (z-scoring, vol targeting, etc.).
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List
from pathlib import Path
import json
import logging
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def run_breakout_mid(
    prices: pd.DataFrame,
    lookback_50: int = 50,
    lookback_100: int = 100,
    upper_threshold: float = 0.55,
    lower_threshold: float = 0.45,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    universe: Optional[List[str]] = None,
    eps: float = 1e-8
) -> Dict[str, Any]:
    """
    Run breakout mid strategy for given lookback periods.
    
    Args:
        prices: Wide DataFrame [date x symbol] of continuous adjusted closes
        lookback_50: 50-day lookback window in trading days (default: 50)
        lookback_100: 100-day lookback window in trading days (default: 100)
        upper_threshold: Threshold for upper breakout signal (default: 0.55)
        lower_threshold: Threshold for lower breakout signal (default: 0.45)
        start_date: Start date for backtest (YYYY-MM-DD)
        end_date: End date for backtest (YYYY-MM-DD)
        universe: List of symbols to include (if None, uses all columns)
        eps: Small constant to avoid division by zero (default: 1e-8)
        
    Returns:
        Dict with:
        - 'portfolio_returns': pd.Series of daily portfolio returns
        - 'equity_curve': pd.Series of cumulative equity
        - 'asset_returns': pd.DataFrame of daily asset returns
        - 'asset_strategy_returns': pd.DataFrame of per-asset strategy returns
        - 'positions': pd.DataFrame of daily positions (signs)
        - 'breakout_50': pd.DataFrame of 50-day breakout scores
        - 'breakout_100': pd.DataFrame of 100-day breakout scores
        - 'breakout_mid': pd.DataFrame of combined breakout scores
        - 'metrics': Dict with {cagr, vol, sharpe, maxdd, hit_rate, n_days, years}
        - 'per_asset_stats': pd.DataFrame of per-asset metrics
    """
    results = compute_breakout_mid(
        prices=prices,
        lookback_50=lookback_50,
        lookback_100=lookback_100,
        upper_threshold=upper_threshold,
        lower_threshold=lower_threshold,
        start_date=start_date,
        end_date=end_date,
        universe=universe,
        eps=eps
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


def compute_breakout_mid(
    prices: pd.DataFrame,
    lookback_50: int = 50,
    lookback_100: int = 100,
    upper_threshold: float = 0.55,
    lower_threshold: float = 0.45,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    universe: Optional[List[str]] = None,
    eps: float = 1e-8
) -> Dict:
    """
    Compute breakout mid strategy returns.
    
    Args:
        prices: DataFrame of continuous prices [date x symbol]
        lookback_50: 50-day lookback period (default: 50)
        lookback_100: 100-day lookback period (default: 100)
        upper_threshold: Threshold for upper breakout signal (default: 0.55)
        lower_threshold: Threshold for lower breakout signal (default: 0.45)
        start_date: Start date for backtest (YYYY-MM-DD)
        end_date: End date for backtest (YYYY-MM-DD)
        universe: List of symbols to include (if None, uses all columns)
        eps: Small constant to avoid division by zero (default: 1e-8)
        
    Returns:
        Dict with portfolio returns, equity curve, positions, etc.
    """
    # Filter by universe if specified
    if universe:
        available = [s for s in universe if s in prices.columns]
        if not available:
            raise ValueError(f"None of the specified universe symbols are available in prices")
        prices = prices[available]
        logger.info(f"Using {len(available)} assets from universe: {available}")
    else:
        logger.info(f"Using all {len(prices.columns)} assets from prices")
    
    # Align all assets on common date index
    prices = prices.dropna(how='all')
    logger.info(f"Price data shape after alignment: {prices.shape}")
    
    # Compute daily returns
    daily_returns = prices.pct_change(fill_method=None).dropna()
    
    if daily_returns.empty:
        raise ValueError("No valid daily returns computed")
    
    logger.info(f"Daily returns computed: {daily_returns.shape}")
    
    # Compute 50-day breakout scores
    # breakout_50 = (price - low_50) / (high_50 - low_50)
    high_50 = prices.rolling(window=lookback_50, min_periods=lookback_50).max()
    low_50 = prices.rolling(window=lookback_50, min_periods=lookback_50).min()
    range_50 = high_50 - low_50
    breakout_50 = (prices - low_50) / (range_50 + eps)
    breakout_50 = breakout_50.clip(0.0, 1.0)  # Ensure [0, 1]
    
    # Compute 100-day breakout scores
    # breakout_100 = (price - low_100) / (high_100 - low_100)
    high_100 = prices.rolling(window=lookback_100, min_periods=lookback_100).max()
    low_100 = prices.rolling(window=lookback_100, min_periods=lookback_100).min()
    range_100 = high_100 - low_100
    breakout_100 = (prices - low_100) / (range_100 + eps)
    breakout_100 = breakout_100.clip(0.0, 1.0)  # Ensure [0, 1]
    
    # Combine: breakout_mid = 0.5 * breakout_50 + 0.5 * breakout_100
    breakout_mid = 0.5 * breakout_50 + 0.5 * breakout_100
    
    logger.info(f"Breakout scores computed: {breakout_mid.shape}")
    
    # Shift breakout scores by 1 day so we don't use same-day info for trading
    signal_basis = breakout_mid.shift(1)
    
    # Get all trading days from prices
    all_trading_days = prices.index
    
    # Filter by date range
    if start_date:
        start_dt = pd.to_datetime(start_date)
        all_trading_days = all_trading_days[all_trading_days >= start_dt]
    
    if end_date:
        end_dt = pd.to_datetime(end_date)
        all_trading_days = all_trading_days[all_trading_days <= end_dt]
    
    if len(all_trading_days) == 0:
        raise ValueError("No data available after date filtering")
    
    # Align signal_basis and daily_returns to all_trading_days
    signal_basis = signal_basis.reindex(all_trading_days)
    daily_returns = daily_returns.reindex(all_trading_days)
    
    # For dates/assets where signal_basis is NaN, set to neutral (0.5 midpoint)
    signal_basis = signal_basis.fillna(0.5)
    
    # For dates/assets where daily_returns is NaN, set to 0
    daily_returns = daily_returns.fillna(0.0)
    
    logger.info(f"Aligned data after date filtering: {len(all_trading_days)} days")
    
    # Generate sign-only positions based on thresholds
    # +1 if breakout_mid > upper_threshold (0.55)
    # -1 if breakout_mid < lower_threshold (0.45)
    # 0 if in neutral band [0.45, 0.55]
    positions = pd.DataFrame(0.0, index=signal_basis.index, columns=signal_basis.columns)
    positions[signal_basis > upper_threshold] = 1.0
    positions[signal_basis < lower_threshold] = -1.0
    
    # Compute per-asset strategy returns
    asset_strategy_returns = positions * daily_returns
    
    # Aggregate to portfolio (equal-weight across assets each day)
    active_counts = (positions != 0).sum(axis=1)
    
    # Portfolio return = sum(asset_return) / n_active (equal-weight)
    portfolio_returns = asset_strategy_returns.sum(axis=1) / active_counts.replace(0, 1)
    
    # If no active assets on a day, portfolio return is 0
    portfolio_returns[active_counts == 0] = 0.0
    
    # Compute equity curve
    if len(portfolio_returns) > 0:
        equity_curve = (1 + portfolio_returns).cumprod()
    else:
        equity_curve = pd.Series(dtype=float)
    
    # Get breakout scores for the final date range (for reference)
    breakout_50_final = breakout_50.reindex(all_trading_days)
    breakout_100_final = breakout_100.reindex(all_trading_days)
    breakout_mid_final = breakout_mid.reindex(all_trading_days)
    
    return {
        'portfolio_returns': portfolio_returns,
        'equity_curve': equity_curve,
        'asset_returns': daily_returns,
        'asset_strategy_returns': asset_strategy_returns,
        'positions': positions,
        'breakout_50': breakout_50_final,
        'breakout_100': breakout_100_final,
        'breakout_mid': breakout_mid_final
    }


def compute_summary_stats(
    portfolio_returns: pd.Series,
    equity_curve: pd.Series,
    asset_strategy_returns: pd.DataFrame
) -> Dict:
    """
    Compute summary statistics for the breakout mid strategy.
    
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
    per_asset_stats = []
    
    for symbol in asset_strategy_returns.columns:
        asset_rets = asset_strategy_returns[symbol].dropna()
        
        if len(asset_rets) == 0:
            continue
        
        # Annualized return
        asset_years = len(asset_rets) / 252.0
        if asset_years > 0:
            cum_ret = (1 + asset_rets).prod() - 1
            ann_ret = (1 + cum_ret) ** (1 / asset_years) - 1
        else:
            ann_ret = float('nan')
        
        # Annualized volatility
        ann_vol = asset_rets.std() * (252 ** 0.5)
        
        # Sharpe ratio
        if asset_rets.std() != 0:
            sharpe_asset = asset_rets.mean() / asset_rets.std() * (252 ** 0.5)
        else:
            sharpe_asset = float('nan')
        
        # Max drawdown
        asset_equity = (1 + asset_rets).cumprod()
        if len(asset_equity) >= 2:
            running_max = asset_equity.cummax()
            dd_asset = (asset_equity / running_max) - 1.0
            max_dd_asset = dd_asset.min()
        else:
            max_dd_asset = float('nan')
        
        per_asset_stats.append({
            'Symbol': symbol,
            'AnnRet': ann_ret,
            'AnnVol': ann_vol,
            'Sharpe': sharpe_asset,
            'MaxDD': max_dd_asset
        })
    
    per_asset_df = pd.DataFrame(per_asset_stats)
    
    return {
        'portfolio': portfolio_metrics,
        'per_asset': per_asset_df
    }


def save_results(
    results: Dict,
    stats: Dict,
    output_dir: Path,
    start_date: str,
    end_date: str,
    lookback_50: int,
    lookback_100: int,
    upper_threshold: float,
    lower_threshold: float,
    universe: List[str]
):
    """
    Save results to output directory.
    
    Args:
        results: Results dict from run_breakout_mid()
        stats: Stats dict with portfolio and per-asset metrics
        output_dir: Path to output directory
        start_date: Start date of backtest
        end_date: End date of backtest
        lookback_50: 50-day lookback parameter
        lookback_100: 100-day lookback parameter
        upper_threshold: Upper breakout threshold
        lower_threshold: Lower breakout threshold
        universe: List of symbols used
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving results to {output_dir}")
    
    # Save portfolio returns
    results['portfolio_returns'].to_csv(output_dir / 'portfolio_returns.csv', header=True)
    
    # Save equity curve
    results['equity_curve'].to_csv(output_dir / 'equity_curve.csv', header=True)
    
    # Save asset strategy returns
    results['asset_strategy_returns'].to_csv(output_dir / 'asset_strategy_returns.csv')
    
    # Save per-asset stats
    stats['per_asset'].to_csv(output_dir / 'per_asset_stats.csv', index=False)
    
    # Save meta.json with run parameters and metrics
    meta = {
        'start_date': start_date,
        'end_date': end_date,
        'lookback_50': lookback_50,
        'lookback_100': lookback_100,
        'upper_threshold': upper_threshold,
        'lower_threshold': lower_threshold,
        'universe': universe,
        'metrics': stats['portfolio']
    }
    
    with open(output_dir / 'meta.json', 'w') as f:
        json.dump(meta, f, indent=2, default=str)
    
    logger.info("Results saved successfully")


def generate_plots(
    results: Dict,
    stats: Dict,
    output_dir: Path,
    prices: pd.DataFrame
):
    """
    Generate diagnostic plots.
    
    Args:
        results: Results dict from run_breakout_mid()
        stats: Stats dict with portfolio and per-asset metrics
        output_dir: Path to output directory
        prices: Original price data
    """
    output_dir = Path(output_dir)
    
    # Plot 1: Equity curve
    fig, ax = plt.subplots(figsize=(12, 6))
    results['equity_curve'].plot(ax=ax, linewidth=2, color='#2E86AB')
    ax.set_title('Breakout Mid (50-100d) Sign-Only Strategy - Equity Curve', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Equity (Starting = 1.0)', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Add metrics text box
    sharpe = stats['portfolio'].get('Sharpe', float('nan'))
    cagr = stats['portfolio'].get('CAGR', float('nan'))
    maxdd = stats['portfolio'].get('MaxDD', float('nan'))
    vol = stats['portfolio'].get('Vol', float('nan'))
    
    textstr = f'Sharpe: {sharpe:.3f}\nCAGR: {cagr*100:.2f}%\nMaxDD: {maxdd*100:.2f}%\nVol: {vol*100:.2f}%'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'equity_curve.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Return histogram
    fig, ax = plt.subplots(figsize=(10, 6))
    results['portfolio_returns'].hist(bins=50, ax=ax, edgecolor='black', alpha=0.7)
    ax.set_title('Breakout Mid (50-100d) Sign-Only Strategy - Return Distribution', fontsize=14, fontweight='bold')
    ax.set_xlabel('Daily Return', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.axvline(0, color='red', linestyle='--', linewidth=1.5, label='Zero Return')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'return_histogram.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info("Plots generated successfully")

