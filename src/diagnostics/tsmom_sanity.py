"""
TSMOM Sign-Only Sanity Check

A deliberately simple, academic-style trend-following strategy to verify that
the data and P&L machinery are working correctly.

Strategy:
- For each asset, compute lookback return over N days (default 252)
- Take the sign of that lookback return (+1 if > 0, -1 if < 0, 0 if ≈ 0)
- Use that sign as the position for the next day
- Daily strategy return = sign * daily_return
- Equal-weight portfolio across assets

This is a diagnostic tool to answer: "Does a very simple trend-following strategy
show reasonable positive Sharpe on our DataBento-driven continuous futures, or is
the alpha gone because our data / roll / P&L pipeline is wrong?"

If sign-only TSMOM shows Sharpe ~0.3-0.6, then data & P&L pipeline are probably fine.
If it shows negative or near-zero Sharpe, we likely have data/roll/P&L issues.

Extended to test multiple horizons (long 252d, medium 84d, short 21d) to identify
which horizons have edge in the universe.
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple, List, Dict, Any
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)

# Default horizons matching sleeve definitions
HORIZONS = {
    "long_252": 252,
    "med_84": 84,
    "short_21": 21,
}


def run_sign_only_momentum(
    prices: pd.DataFrame,
    lookback: int,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    universe: Optional[List[str]] = None,
    zero_threshold: float = 1e-8
) -> Dict[str, Any]:
    """
    Run sign-only momentum strategy for a given lookback period.
    
    Args:
        prices: Wide DataFrame [date x symbol] of continuous adjusted closes
        lookback: Lookback window in trading days (e.g. 252, 84, 21)
        start_date: Start date for backtest (YYYY-MM-DD)
        end_date: End date for backtest (YYYY-MM-DD)
        universe: List of symbols to include (if None, uses all columns)
        zero_threshold: Threshold for treating lookback return as zero
        
    Returns:
        Dict with:
        - 'portfolio_returns': pd.Series of daily portfolio returns
        - 'equity_curve': pd.Series of cumulative equity
        - 'asset_returns': pd.DataFrame of daily asset returns
        - 'asset_strategy_returns': pd.DataFrame of per-asset strategy returns
        - 'positions': pd.DataFrame of daily positions (signs)
        - 'lookback_returns': pd.DataFrame of lookback returns used for signals
        - 'metrics': Dict with {cagr, vol, sharpe, maxdd, hit_rate, n_days, years}
    """
    results = compute_sign_only_tsmom(
        prices=prices,
        lookback=lookback,
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


def compute_sign_only_tsmom(
    prices: pd.DataFrame,
    lookback: int = 252,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    universe: Optional[List[str]] = None,
    zero_threshold: float = 1e-8
) -> Dict:
    """
    Compute sign-only TSMOM strategy returns.
    
    Args:
        prices: DataFrame of continuous prices [date x symbol]
        lookback: Lookback period in days (default: 252)
        start_date: Start date for backtest (YYYY-MM-DD)
        end_date: End date for backtest (YYYY-MM-DD)
        universe: List of symbols to include (if None, uses all columns)
        zero_threshold: Threshold for treating lookback return as zero
        
    Returns:
        Dict with:
        - 'portfolio_returns': pd.Series of daily portfolio returns
        - 'equity_curve': pd.Series of cumulative equity (starts at 1.0)
        - 'asset_returns': pd.DataFrame of daily asset returns
        - 'asset_strategy_returns': pd.DataFrame of per-asset strategy returns
        - 'positions': pd.DataFrame of daily positions (signs)
        - 'lookback_returns': pd.DataFrame of lookback returns used for signals
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
    
    # Compute daily simple returns (using all available data for now)
    # r_t = price_t / price_{t-1} - 1
    daily_returns = prices.pct_change(fill_method=None).dropna()
    
    if daily_returns.empty:
        raise ValueError("No valid daily returns computed")
    
    logger.info(f"Daily returns computed: {daily_returns.shape}")
    
    # Compute lookback returns (using all available data - need history before start_date)
    # lookback_ret_t = price_t / price_{t-LB} - 1
    lookback_returns = prices.pct_change(periods=lookback, fill_method=None).dropna()
    
    if lookback_returns.empty:
        raise ValueError(f"No valid lookback returns computed (need at least {lookback} days of data)")
    
    logger.info(f"Lookback returns computed: {lookback_returns.shape}")
    
    # Shift lookback returns by 1 day so we don't use same-day info for trading
    # signal_basis_t = lookback_ret_{t-1}
    signal_basis = lookback_returns.shift(1).dropna()
    
    # Align daily_returns with signal_basis (they should have same index after shift)
    common_idx = signal_basis.index.intersection(daily_returns.index)
    signal_basis = signal_basis.loc[common_idx]
    daily_returns = daily_returns.loc[common_idx]
    
    # NOW filter by date range (after computing signals, but before generating positions)
    if start_date:
        start_dt = pd.to_datetime(start_date)
        mask = common_idx >= start_dt
        common_idx = common_idx[mask]
        signal_basis = signal_basis.loc[common_idx]
        daily_returns = daily_returns.loc[common_idx]
    
    if end_date:
        end_dt = pd.to_datetime(end_date)
        mask = common_idx <= end_dt
        common_idx = common_idx[mask]
        signal_basis = signal_basis.loc[common_idx]
        daily_returns = daily_returns.loc[common_idx]
    
    if len(common_idx) == 0:
        raise ValueError("No data available after date filtering")
    
    logger.info(f"Aligned data after date filtering: {len(common_idx)} days")
    
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
    # portfolio_ret_t = mean(strategy_ret_asset_t across assets for that day)
    portfolio_returns = asset_strategy_returns.mean(axis=1)
    
    # Compute equity curve (cumulative)
    # Standard formula: equity_t = equity_{t-1} * (1 + return_t)
    # Starting with initial equity = 1.0, if portfolio_returns = [r1, r2, r3]:
    # equity after r1 = 1.0 * (1 + r1)
    # equity after r2 = 1.0 * (1 + r1) * (1 + r2)
    # equity after r3 = 1.0 * (1 + r1) * (1 + r2) * (1 + r3)
    # The cumprod of (1 + returns) gives us exactly this
    # Note: The first value will be (1 + r1), not 1.0, which is correct
    # (it represents equity after the first return)
    if len(portfolio_returns) > 0:
        equity_curve = (1 + portfolio_returns).cumprod()
    else:
        equity_curve = pd.Series(dtype=float)
    
    return {
        'portfolio_returns': portfolio_returns,
        'equity_curve': equity_curve,
        'asset_returns': daily_returns,
        'asset_strategy_returns': asset_strategy_returns,
        'positions': positions,
        'lookback_returns': lookback_returns.loc[common_idx] if common_idx[0] in lookback_returns.index else pd.DataFrame()
    }


def compute_summary_stats(
    portfolio_returns: pd.Series,
    equity_curve: pd.Series,
    asset_strategy_returns: pd.DataFrame
) -> Dict:
    """
    Compute summary statistics for the sign-only TSMOM strategy.
    
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
    lookback: int,
    universe: Optional[List[str]],
    horizon_name: Optional[str] = None
):
    """
    Save all results to output directory.
    
    Saves:
    - portfolio_returns.csv
    - equity_curve.csv
    - asset_strategy_returns.csv
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
    
    # Save meta
    meta = {
        'start_date': start_date,
        'end_date': end_date,
        'lookback': lookback,
        'horizon_name': horizon_name,
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
    1. Price series for key assets (ES, NQ, CL, GC)
    2. Cumulative equity curves (portfolio and per-asset)
    3. Return histograms
    """
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available, skipping plots")
        return
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Price series for key assets
    key_assets = ['ES', 'NQ', 'CL', 'GC']
    available_key = [s for s in prices.columns if any(ka in s for ka in key_assets)]
    
    if available_key:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        for i, sym in enumerate(available_key[:4]):
            if sym in prices.columns:
                price_series = prices[sym].dropna()
                if not price_series.empty:
                    axes[i].plot(price_series.index, price_series.values)
                    axes[i].set_title(f'{sym} Price Series')
                    axes[i].set_xlabel('Date')
                    axes[i].set_ylabel('Price')
                    axes[i].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(len(available_key), 4):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'price_series.png', dpi=150, bbox_inches='tight')
        plt.close()
        logger.info("Saved price_series.png")
    
    # 2. Cumulative equity curves
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Portfolio equity curve
    ax.plot(results['equity_curve'].index, results['equity_curve'].values, 
            label='Portfolio', linewidth=2, color='black')
    
    # Per-asset equity curves (top 5 by Sharpe if available)
    if not stats['per_asset'].empty:
        top_assets = stats['per_asset'].sort_values('Sharpe', ascending=False).head(5)
        colors = plt.cm.tab10(np.linspace(0, 1, len(top_assets)))
        
        for (sym, row), color in zip(top_assets.iterrows(), colors):
            asset_ret = results['asset_strategy_returns'][sym].dropna()
            if not asset_ret.empty:
                asset_equity = (1 + asset_ret).cumprod()
                asset_equity.iloc[0] = 1.0
                # Align index
                common_idx = asset_equity.index.intersection(results['equity_curve'].index)
                if len(common_idx) > 0:
                    ax.plot(common_idx, asset_equity.loc[common_idx].values,
                           label=f'{sym} (Sharpe={row["Sharpe"]:.2f})', 
                           alpha=0.6, color=color)
    
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    # Get horizon name from output_dir if available (for multi-horizon mode)
    horizon_name = output_dir.name if 'horizon' in str(output_dir) or any(h in str(output_dir) for h in ['long_252', 'med_84', 'short_21']) else None
    if horizon_name:
        ax.set_title(f'Sign-Only TSMOM: Cumulative Equity Curves ({horizon_name})')
    else:
        ax.set_title('Sign-Only TSMOM: Cumulative Equity Curves')
    ax.set_xlabel('Date')
    ax.set_ylabel('Cumulative Equity')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'equity_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved equity_curves.png")
    
    # 3. Return histograms
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Portfolio returns histogram
    portfolio_ret = results['portfolio_returns'].dropna()
    axes[0].hist(portfolio_ret.values, bins=50, alpha=0.7, edgecolor='black')
    axes[0].axvline(x=0, color='red', linestyle='--', alpha=0.5)
    axes[0].set_title(f'Portfolio Returns Distribution\n(Mean={portfolio_ret.mean():.4f}, Std={portfolio_ret.std():.4f})')
    axes[0].set_xlabel('Daily Return')
    axes[0].set_ylabel('Frequency')
    axes[0].grid(True, alpha=0.3)
    
    # Per-asset returns histogram (aggregate)
    all_asset_ret = results['asset_strategy_returns'].values.flatten()
    all_asset_ret = all_asset_ret[~np.isnan(all_asset_ret)]
    axes[1].hist(all_asset_ret, bins=50, alpha=0.7, edgecolor='black', color='orange')
    axes[1].axvline(x=0, color='red', linestyle='--', alpha=0.5)
    axes[1].set_title(f'Per-Asset Strategy Returns Distribution\n(Mean={np.mean(all_asset_ret):.4f}, Std={np.std(all_asset_ret):.4f})')
    axes[1].set_xlabel('Daily Return')
    axes[1].set_ylabel('Frequency')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'return_histograms.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved return_histograms.png")

