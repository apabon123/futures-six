"""
Persistence Sign-Only Sanity Check

A Phase-0 diagnostic for validating the persistence (momentum-of-momentum) idea:
- Return acceleration: ret_84[t] - ret_84[t-21]
- Slope acceleration: (EMA20 - EMA84)[t] - (EMA20 - EMA84)[t-21]
- Breakout acceleration: breakout_126[t] - breakout_126[t-21]
- Sign-only signals, equal-weighted across assets
- No overlays, no vol targeting, no z-scoring

Strategy:
- For each asset and date t:
  - Variant 1 (Return Acceleration): persistence_raw = ret_84[t] - ret_84[t-21], signal = sign(persistence_raw)
  - Variant 2 (Slope Acceleration): slope_now = EMA20 - EMA84, slope_old = EMA20.shift(21) - EMA84.shift(21), signal = sign(slope_now - slope_old)
  - Variant 3 (Breakout Acceleration): breakout_now = breakout_126[t], breakout_old = breakout_126[t-21], signal = sign(breakout_now - breakout_old)
- Equal-weight portfolio, daily rebalancing

This is a diagnostic tool to validate that the persistence idea has positive alpha
before adding complexity (z-scoring, vol targeting, etc.).
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)


def run_persistence(
    prices: pd.DataFrame,
    variant: str = "return_accel",
    lookback: int = 84,
    acceleration_window: int = 21,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    universe: Optional[List[str]] = None,
    zero_threshold: float = 1e-8
) -> Dict[str, Any]:
    """
    Run persistence strategy for given variant and parameters.
    
    Args:
        prices: Wide DataFrame [date x symbol] of continuous adjusted closes
        variant: "return_accel" or "slope_accel"
        lookback: Base lookback window in trading days (default: 84 for ret_84)
        acceleration_window: Window for acceleration calculation (default: 21)
        start_date: Start date for backtest (YYYY-MM-DD)
        end_date: End date for backtest (YYYY-MM-DD)
        universe: List of symbols to include (if None, uses all columns)
        zero_threshold: Threshold for treating persistence signal as zero
        
    Returns:
        Dict with:
        - 'portfolio_returns': pd.Series of daily portfolio returns
        - 'equity_curve': pd.Series of cumulative equity
        - 'asset_returns': pd.DataFrame of daily asset returns
        - 'asset_strategy_returns': pd.DataFrame of per-asset strategy returns
        - 'positions': pd.DataFrame of daily positions (signs)
        - 'persistence_signals': pd.DataFrame of persistence signals used
        - 'metrics': Dict with {cagr, vol, sharpe, maxdd, hit_rate, n_days, years}
    """
    results = compute_persistence(
        prices=prices,
        variant=variant,
        lookback=lookback,
        acceleration_window=acceleration_window,
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


def compute_persistence(
    prices: pd.DataFrame,
    variant: str = "return_accel",
    lookback: int = 84,
    acceleration_window: int = 21,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    universe: Optional[List[str]] = None,
    zero_threshold: float = 1e-8
) -> Dict:
    """
    Compute persistence strategy returns.
    
    Args:
        prices: DataFrame of continuous prices [date x symbol]
        variant: "return_accel" or "slope_accel"
        lookback: Base lookback period in days (default: 84)
        acceleration_window: Window for acceleration (default: 21)
        start_date: Start date for backtest
        end_date: End date for backtest
        universe: List of symbols to include
        zero_threshold: Threshold for zero signal
        
    Returns:
        Dict with portfolio_returns, equity_curve, asset_returns, etc.
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
    
    # Convert to log prices for log return calculation
    log_prices = np.log(prices)
    
    # Compute daily simple returns
    daily_returns = prices.pct_change(fill_method=None).dropna()
    
    if daily_returns.empty:
        raise ValueError("No valid daily returns computed")
    
    logger.info(f"Daily returns computed: {daily_returns.shape}")
    
    # Compute persistence signals based on variant
    if variant == "return_accel":
        # Return acceleration: ret_84[t] - ret_84[t-21]
        # ret_84[t] = log(price_t / price_{t-84})
        ret_base = log_prices - log_prices.shift(lookback)
        ret_base_shifted = ret_base.shift(acceleration_window)
        persistence_signals = ret_base - ret_base_shifted
        
    elif variant == "slope_accel":
        # Slope acceleration: (EMA20 - EMA84)[t] - (EMA20 - EMA84)[t-21]
        # Compute EMAs
        ema20 = prices.ewm(span=20, adjust=False).mean()
        ema84 = prices.ewm(span=84, adjust=False).mean()
        
        # Slope now
        slope_now = ema20 - ema84
        
        # Slope old (shifted by acceleration_window)
        slope_old = slope_now.shift(acceleration_window)
        
        # Persistence = slope acceleration
        persistence_signals = slope_now - slope_old
        
    elif variant == "breakout_accel":
        # Breakout acceleration: breakout_126[t] - breakout_126[t-21]
        # Compute 126-day breakout strength
        breakout_window = 126  # Standard medium-term breakout window
        rolling_min_126 = prices.rolling(window=breakout_window, min_periods=breakout_window // 2).min()
        rolling_max_126 = prices.rolling(window=breakout_window, min_periods=breakout_window // 2).max()
        
        # Compute raw breakout
        eps = 1e-6
        range_126 = rolling_max_126 - rolling_min_126 + eps
        raw_breakout_126 = (prices - rolling_min_126) / range_126
        
        # Map to roughly [-1, +1] by rescaling: (x - 0.5) * 2
        breakout_scaled = (raw_breakout_126 - 0.5) * 2.0
        
        # Breakout now
        breakout_now = breakout_scaled
        
        # Breakout old (shifted by acceleration_window)
        breakout_old = breakout_now.shift(acceleration_window)
        
        # Persistence = breakout acceleration
        persistence_signals = breakout_now - breakout_old
        
    else:
        raise ValueError(f"Unknown variant: {variant}. Must be 'return_accel', 'slope_accel', or 'breakout_accel'")
    
    logger.info(f"Persistence signals computed ({variant}): {persistence_signals.shape}")
    
    # Shift signals by 1 day so we don't use same-day info for trading
    signal_basis = persistence_signals.shift(1)
    
    # Get all dates from prices DataFrame
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
    
    # Fill NaN with 0.0
    signal_basis = signal_basis.fillna(0.0)
    daily_returns = daily_returns.fillna(0.0)
    
    logger.info(f"Aligned data after date filtering: {len(all_trading_days)} days")
    
    # Generate sign-only positions
    positions = signal_basis.copy()
    positions[positions.abs() < zero_threshold] = 0.0
    positions[positions > zero_threshold] = 1.0
    positions[positions < -zero_threshold] = -1.0
    
    # Compute per-asset strategy returns
    asset_strategy_returns = positions * daily_returns
    
    # Aggregate to portfolio (equal-weight across assets each day)
    active_counts = (positions != 0).sum(axis=1)
    portfolio_returns = asset_strategy_returns.sum(axis=1) / active_counts.replace(0, 1)
    portfolio_returns[active_counts == 0] = 0.0
    
    # Compute equity curve
    if len(portfolio_returns) > 0:
        equity_curve = (1 + portfolio_returns).cumprod()
    else:
        equity_curve = pd.Series(dtype=float)
    
    # Get persistence signals for the final date range
    persistence_signals_final = persistence_signals.reindex(all_trading_days)
    
    return {
        'portfolio_returns': portfolio_returns,
        'equity_curve': equity_curve,
        'asset_returns': daily_returns,
        'asset_strategy_returns': asset_strategy_returns,
        'positions': positions,
        'persistence_signals': persistence_signals_final
    }


def compute_summary_stats(
    portfolio_returns: pd.Series,
    equity_curve: pd.Series,
    asset_strategy_returns: pd.DataFrame
) -> Dict:
    """
    Compute summary statistics for the persistence strategy.
    
    Returns:
        Dict with portfolio metrics and per-asset stats
    """
    if len(portfolio_returns) == 0:
        return {
            'portfolio': {
                'cagr': 0.0,
                'vol': 0.0,
                'sharpe': 0.0,
                'maxdd': 0.0,
                'hit_rate': 0.0,
                'n_days': 0,
                'years': 0.0
            },
            'per_asset': pd.DataFrame()
        }
    
    # Portfolio metrics
    n_days = len(portfolio_returns)
    years = n_days / 252.0
    
    # CAGR
    if len(equity_curve) > 0 and equity_curve.iloc[-1] > 0:
        total_return = equity_curve.iloc[-1] - 1.0
        cagr = (1 + total_return) ** (1.0 / years) - 1.0 if years > 0 else 0.0
    else:
        cagr = 0.0
    
    # Volatility (annualized)
    vol = portfolio_returns.std() * np.sqrt(252)
    
    # Sharpe ratio (annualized)
    sharpe = (cagr / vol) if vol > 0 else 0.0
    
    # Maximum drawdown
    running_max = equity_curve.expanding().max()
    drawdown = (equity_curve - running_max) / running_max
    maxdd = drawdown.min()
    
    # Hit rate
    hit_rate = (portfolio_returns > 0).sum() / len(portfolio_returns) if len(portfolio_returns) > 0 else 0.0
    
    portfolio_metrics = {
        'cagr': cagr,
        'vol': vol,
        'sharpe': sharpe,
        'maxdd': maxdd,
        'hit_rate': hit_rate,
        'n_days': n_days,
        'years': years
    }
    
    # Per-asset stats
    per_asset_stats = []
    for symbol in asset_strategy_returns.columns:
        asset_ret = asset_strategy_returns[symbol]
        if len(asset_ret) > 0:
            asset_years = len(asset_ret) / 252.0
            asset_total_ret = (1 + asset_ret).prod() - 1.0
            asset_cagr = (1 + asset_total_ret) ** (1.0 / asset_years) - 1.0 if asset_years > 0 else 0.0
            asset_vol = asset_ret.std() * np.sqrt(252)
            asset_sharpe = (asset_cagr / asset_vol) if asset_vol > 0 else 0.0
            
            per_asset_stats.append({
                'symbol': symbol,
                'cagr': asset_cagr,
                'vol': asset_vol,
                'sharpe': asset_sharpe
            })
    
    per_asset_df = pd.DataFrame(per_asset_stats)
    if not per_asset_df.empty:
        per_asset_df = per_asset_df.sort_values('cagr', ascending=False)
    
    return {
        'portfolio': portfolio_metrics,
        'per_asset': per_asset_df
    }


def save_results(
    results: Dict[str, Any],
    output_dir: Path,
    variant: str,
    lookback: int,
    acceleration_window: int
):
    """
    Save persistence sanity check results to disk.
    
    Args:
        results: Results dict from run_persistence
        output_dir: Directory to save results
        variant: Variant name ("return_accel" or "slope_accel")
        lookback: Lookback parameter
        acceleration_window: Acceleration window parameter
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save CSV files
    results['portfolio_returns'].to_csv(output_dir / 'portfolio_returns.csv', header=True)
    results['equity_curve'].to_csv(output_dir / 'equity_curve.csv', header=True)
    results['asset_returns'].to_csv(output_dir / 'asset_returns.csv')
    results['asset_strategy_returns'].to_csv(output_dir / 'asset_strategy_returns.csv')
    results['positions'].to_csv(output_dir / 'positions.csv')
    results['persistence_signals'].to_csv(output_dir / 'persistence_signals.csv')
    
    # Save per-asset stats
    if not results['per_asset_stats'].empty:
        results['per_asset_stats'].to_csv(output_dir / 'per_asset_stats.csv', index=False)
    
    # Save summary metrics
    summary_df = pd.DataFrame([results['metrics']])
    summary_df.to_csv(output_dir / 'summary_metrics.csv', index=False)
    
    # Save metadata
    meta = {
        'variant': variant,
        'lookback': lookback,
        'acceleration_window': acceleration_window,
        'start_date': str(results['portfolio_returns'].index[0]) if len(results['portfolio_returns']) > 0 else None,
        'end_date': str(results['portfolio_returns'].index[-1]) if len(results['portfolio_returns']) > 0 else None,
        'universe': list(results['asset_returns'].columns),
        'metrics': results['metrics']
    }
    
    with open(output_dir / 'meta.json', 'w') as f:
        json.dump(meta, f, indent=2, default=str)
    
    logger.info(f"Saved results to {output_dir}")


def generate_plots(
    results: Dict[str, Any],
    output_dir: Path,
    variant: str
):
    """
    Generate plots for persistence sanity check.
    
    Args:
        results: Results dict from run_persistence
        output_dir: Directory to save plots
        variant: Variant name for plot titles
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
    except ImportError:
        logger.warning("matplotlib not available, skipping plots")
        return
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Equity curve
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(results['equity_curve'].index, results['equity_curve'].values, linewidth=2)
    ax.set_title(f'Persistence Strategy Equity Curve ({variant})', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Cumulative Equity')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=1.0, color='k', linestyle='--', linewidth=1)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / 'equity_curve.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Return histogram
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Portfolio returns
    ax1.hist(results['portfolio_returns'].values, bins=50, alpha=0.7, edgecolor='black')
    ax1.set_title('Portfolio Returns Distribution', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Daily Return')
    ax1.set_ylabel('Frequency')
    ax1.axvline(x=0, color='r', linestyle='--', linewidth=1)
    ax1.grid(True, alpha=0.3)
    
    # Per-asset returns (sample)
    if not results['asset_strategy_returns'].empty:
        sample_asset = results['asset_strategy_returns'].columns[0]
        ax2.hist(results['asset_strategy_returns'][sample_asset].dropna().values, bins=50, alpha=0.7, edgecolor='black')
        ax2.set_title(f'Asset Returns Distribution ({sample_asset})', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Daily Return')
        ax2.set_ylabel('Frequency')
        ax2.axvline(x=0, color='r', linestyle='--', linewidth=1)
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'return_histogram.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Generated plots in {output_dir}")

