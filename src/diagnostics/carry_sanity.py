"""
FX/Commodity Carry Sign-Only Sanity Check

A deliberately simple, academic-style carry trading strategy to verify that
the roll yield idea and P&L machinery are working correctly.

Strategy:
- For each asset, compute roll yield: carry_raw = -(ln(F1) - ln(F0))
- Take the sign of roll yield (+1 if > 0, -1 if < 0, 0 if ≈ 0)
- Use that sign as the position for the next day
- Daily strategy return = sign * daily_return
- Equal-weight portfolio across assets

This is a diagnostic tool to answer: "Does a very simple carry strategy
show reasonable positive Sharpe on our roll yield features, or is the alpha
gone because our data / roll / P&L pipeline is wrong?"

If sign-only carry shows Sharpe ~0.2-0.5, then data & P&L pipeline are probably fine.
If it shows negative or near-zero Sharpe, we should question:
- The roll yield calculation
- The data quality (rank 0 vs rank 1 prices)
- The hypothesis that carry is still a tradable edge
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple, List, Dict, Any
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)

# Carry universe
CARRY_UNIVERSE = {
    "CL": {"rank0": "CL_FRONT_VOLUME", "rank1": "CL_RANK_1_VOLUME"},
    "GC": {"rank0": "GC_FRONT_VOLUME", "rank1": "GC_RANK_1_VOLUME"},
    "6E": {"rank0": "6E_FRONT_CALENDAR", "rank1": "6E_RANK_1_CALENDAR"},
    "6B": {"rank0": "6B_FRONT_CALENDAR", "rank1": "6B_RANK_1_CALENDAR"},
    "6J": {"rank0": "6J_FRONT_CALENDAR", "rank1": "6J_RANK_1_CALENDAR"},
}


def run_sign_only_carry(
    market,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    universe: Optional[List[str]] = None,
    zero_threshold: float = 1e-8
) -> Dict[str, Any]:
    """
    Run sign-only carry strategy.
    
    Args:
        market: MarketData instance
        start_date: Start date for backtest (YYYY-MM-DD)
        end_date: End date for backtest (YYYY-MM-DD)
        universe: List of root symbols to include (default: all in CARRY_UNIVERSE)
        zero_threshold: Threshold for treating roll yield as zero
        
    Returns:
        Dict with:
        - 'portfolio_returns': pd.Series of daily portfolio returns
        - 'equity_curve': pd.Series of cumulative equity
        - 'asset_returns': pd.DataFrame of daily asset returns
        - 'asset_strategy_returns': pd.DataFrame of per-asset strategy returns
        - 'positions': pd.DataFrame of daily positions (signs)
        - 'roll_yields': pd.DataFrame of raw roll yields
        - 'metrics': Dict with {cagr, vol, sharpe, maxdd, hit_rate, n_days, years}
    """
    results = compute_sign_only_carry(
        market=market,
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


def compute_sign_only_carry(
    market,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    universe: Optional[List[str]] = None,
    zero_threshold: float = 1e-8
) -> Dict:
    """
    Compute sign-only carry strategy returns.
    
    Args:
        market: MarketData instance
        start_date: Start date for backtest (YYYY-MM-DD)
        end_date: End date for backtest (YYYY-MM-DD)
        universe: List of root symbols to include (default: all in CARRY_UNIVERSE)
        zero_threshold: Threshold for treating roll yield as zero
        
    Returns:
        Dict with returns, positions, and equity curves
    """
    if universe is None:
        universe = list(CARRY_UNIVERSE.keys())
    
    # Filter to available roots
    available_roots = [r for r in universe if r in CARRY_UNIVERSE]
    if not available_roots:
        raise ValueError(f"None of the specified universe roots are available: {universe}")
    
    logger.info(f"Using {len(available_roots)} assets from universe: {available_roots}")
    
    # Load rank 0 and rank 1 prices for each root
    roll_yields_dict = {}
    prices_dict = {}
    
    for root in available_roots:
        symbol_info = CARRY_UNIVERSE[root]
        rank0_symbol = symbol_info["rank0"]
        rank1_symbol = symbol_info["rank1"]
        
        # Get rank 0 and rank 1 prices using get_contracts_by_root
        # This returns a DataFrame with columns = ranks (0, 1)
        try:
            close = market.get_contracts_by_root(
                root=root,
                ranks=[0, 1],
                fields=("close",),
                start=None,  # Get all available data
                end=end_date
            )
            
            if close.empty:
                logger.warning(f"[CarrySanity] No data found for root {root}")
                continue
            
            # Ensure we have both ranks
            if 0 not in close.columns or 1 not in close.columns:
                logger.warning(
                    f"[CarrySanity] Missing required ranks for {root}. "
                    f"Available: {list(close.columns)}"
                )
                continue
            
            # Extract F0 and F1 (front and next contracts)
            F0 = close[0]
            F1 = close[1]
            
            # Handle missing data: forward-fill and backward-fill
            F0_filled = F0.ffill().bfill()
            F1_filled = F1.ffill().bfill()
            
            # Compute roll yield: carry_raw = -(ln(F1) - ln(F0))
            # Positive = backwardation (F1 < F0) = attractive long carry
            # Negative = contango (F1 > F0) = attractive short carry
            roll_yield_raw = -(np.log(F1_filled) - np.log(F0_filled))
            
            roll_yields_dict[root] = roll_yield_raw
            
            # Store rank 0 prices for computing returns
            # Use continuous prices for returns calculation
            prices_cont = market.prices_cont
            if rank0_symbol in prices_cont.columns:
                prices_dict[root] = prices_cont[rank0_symbol]
            else:
                logger.warning(f"[CarrySanity] Rank 0 symbol {rank0_symbol} not in prices_cont, using F0")
                prices_dict[root] = F0_filled
            
        except Exception as e:
            logger.warning(f"[CarrySanity] Error loading data for {root}: {e}")
            continue
    
    if not roll_yields_dict:
        raise ValueError("No roll yield data could be computed for any assets")
    
    # Combine roll yields into DataFrame
    roll_yields_df = pd.DataFrame(roll_yields_dict)
    
    # Combine prices into DataFrame
    prices_df = pd.DataFrame(prices_dict)
    
    # Align on common dates
    common_dates = roll_yields_df.index.intersection(prices_df.index)
    roll_yields_df = roll_yields_df.loc[common_dates]
    prices_df = prices_df.loc[common_dates]
    
    if len(common_dates) == 0:
        raise ValueError("No common dates between roll yields and prices")
    
    logger.info(f"Aligned data: {len(common_dates)} days")
    
    # Compute daily returns (simple returns)
    daily_returns = prices_df.pct_change(fill_method=None).dropna()
    
    if daily_returns.empty:
        raise ValueError("No valid daily returns computed")
    
    logger.info(f"Daily returns computed: {daily_returns.shape}")
    
    # Shift roll yields by 1 day so we don't use same-day info for trading
    # signal_basis_t = roll_yield_{t-1}
    signal_basis = roll_yields_df.shift(1).dropna()
    
    # Align daily_returns with signal_basis
    common_idx = signal_basis.index.intersection(daily_returns.index)
    signal_basis = signal_basis.loc[common_idx]
    daily_returns = daily_returns.loc[common_idx]
    
    # Filter by date range (after computing signals, but before generating positions)
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
    
    logger.info(f"Final aligned data: {len(common_idx)} days")
    
    # Generate sign-only positions
    # position_t = sign(signal_basis_t)
    # +1 if > 0 (backwardation, long), -1 if < 0 (contango, short), 0 if abs < threshold
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
        'roll_yields': roll_yields_df.loc[common_idx] if common_idx[0] in roll_yields_df.index else pd.DataFrame()
    }


def compute_summary_stats(
    portfolio_returns: pd.Series,
    equity_curve: pd.Series,
    asset_strategy_returns: pd.DataFrame
) -> Dict:
    """
    Compute summary statistics for the sign-only carry strategy.
    
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


def compute_subperiod_stats(
    portfolio_returns: pd.Series,
    equity_curve: pd.Series,
    break_date: str = "2022-01-01"
) -> Dict[str, Dict]:
    """
    Compute statistics for subperiods (pre-break vs post-break).
    
    Args:
        portfolio_returns: Daily portfolio returns
        equity_curve: Cumulative equity curve
        break_date: Date to split periods (YYYY-MM-DD)
        
    Returns:
        Dict with 'pre' and 'post' period metrics
    """
    break_dt = pd.to_datetime(break_date)
    
    pre_mask = portfolio_returns.index < break_dt
    post_mask = portfolio_returns.index >= break_dt
    
    pre_returns = portfolio_returns[pre_mask]
    post_returns = portfolio_returns[post_mask]
    
    pre_equity = equity_curve[pre_mask] if len(equity_curve[pre_mask]) > 0 else pd.Series(dtype=float)
    post_equity = equity_curve[post_mask] if len(equity_curve[post_mask]) > 0 else pd.Series(dtype=float)
    
    # Normalize post_equity to start at 1.0 (relative to break date)
    if len(post_equity) > 0 and len(pre_equity) > 0:
        post_equity = post_equity / post_equity.iloc[0]
    
    subperiods = {}
    
    for period_name, rets, eq in [('pre', pre_returns, pre_equity), ('post', post_returns, post_equity)]:
        if len(rets) == 0:
            subperiods[period_name] = {}
            continue
        
        n_days = len(rets)
        years = n_days / 252.0
        
        # CAGR
        if len(eq) >= 2 and years > 0:
            equity_start = eq.iloc[0]
            equity_end = eq.iloc[-1]
            if equity_start > 0:
                cagr = (equity_end / equity_start) ** (1 / years) - 1
            else:
                cagr = float('nan')
        else:
            cagr = float('nan')
        
        # Vol
        vol = rets.std() * (252 ** 0.5)
        
        # Sharpe
        if rets.std() != 0:
            sharpe = rets.mean() / rets.std() * (252 ** 0.5)
        else:
            sharpe = float('nan')
        
        # MaxDD
        if len(eq) >= 2:
            running_max = eq.cummax()
            dd = (eq / running_max) - 1.0
            max_dd = dd.min()
        else:
            max_dd = float('nan')
        
        # Hit rate
        hit_rate = (rets > 0).sum() / len(rets) if len(rets) > 0 else float('nan')
        
        subperiods[period_name] = {
            'CAGR': cagr,
            'Vol': vol,
            'Sharpe': sharpe,
            'MaxDD': max_dd,
            'HitRate': hit_rate,
            'n_days': n_days,
            'years': years
        }
    
    return subperiods


def save_results(
    results: Dict,
    stats: Dict,
    subperiod_stats: Dict,
    output_dir: Path,
    start_date: Optional[str],
    end_date: Optional[str]
):
    """
    Save all results to output directory.
    
    Saves:
    - portfolio_returns.csv
    - equity_curve.csv
    - asset_strategy_returns.csv
    - roll_yields.csv
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
    
    # Save roll yields
    if not results['roll_yields'].empty:
        results['roll_yields'].to_csv(output_dir / 'roll_yields.csv')
    
    # Save meta
    meta = {
        'start_date': start_date,
        'end_date': end_date,
        'n_days': len(results['portfolio_returns']),
        'portfolio_metrics': stats['portfolio'],
        'per_asset_stats': stats['per_asset'].to_dict('index') if not stats['per_asset'].empty else {},
        'subperiod_stats': subperiod_stats
    }
    
    with open(output_dir / 'meta.json', 'w') as f:
        json.dump(meta, f, indent=2, default=str)
    
    logger.info(f"Results saved to {output_dir}")


def generate_plots(
    results: Dict,
    stats: Dict,
    subperiod_stats: Dict,
    output_dir: Path
):
    """
    Generate diagnostic plots.
    
    Plots:
    1. Cumulative equity curves (portfolio and per-asset)
    2. Return histograms
    3. Subperiod comparison
    """
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available, skipping plots")
        return
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Cumulative equity curves
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Portfolio equity curve
    ax.plot(results['equity_curve'].index, results['equity_curve'].values, 
            label='Portfolio (Equal-Weight)', linewidth=2, color='black')
    
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
    ax.set_title(f'Sign-Only Carry: Cumulative Equity Curves\n'
                 f"Portfolio Sharpe={stats['portfolio'].get('Sharpe', 0):.2f}, "
                 f"CAGR={stats['portfolio'].get('CAGR', 0)*100:.2f}%")
    ax.set_xlabel('Date')
    ax.set_ylabel('Cumulative Equity')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'equity_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved equity_curves.png")
    
    # 2. Return histograms
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
    
    # 3. Subperiod comparison (if available)
    if 'pre' in subperiod_stats and 'post' in subperiod_stats:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        periods = ['Pre-2022', 'Post-2022']
        sharpe_values = [
            subperiod_stats['pre'].get('Sharpe', 0),
            subperiod_stats['post'].get('Sharpe', 0)
        ]
        cagr_values = [
            subperiod_stats['pre'].get('CAGR', 0) * 100,
            subperiod_stats['post'].get('CAGR', 0) * 100
        ]
        
        x = np.arange(len(periods))
        width = 0.35
        
        ax2 = ax.twinx()
        bars1 = ax.bar(x - width/2, sharpe_values, width, label='Sharpe', alpha=0.7, color='blue')
        bars2 = ax2.bar(x + width/2, cagr_values, width, label='CAGR (%)', alpha=0.7, color='orange')
        
        ax.set_xlabel('Period')
        ax.set_ylabel('Sharpe Ratio', color='blue')
        ax2.set_ylabel('CAGR (%)', color='orange')
        ax.set_xticks(x)
        ax.set_xticklabels(periods)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}', ha='center', va='bottom' if height >= 0 else 'top')
        
        ax.set_title('Subperiod Performance Comparison')
        plt.tight_layout()
        plt.savefig(output_dir / 'subperiod_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
        logger.info("Saved subperiod_comparison.png")

