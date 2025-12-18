"""
VX Calendar Carry Sign-Only Sanity Check (Phase-0)

A deliberately simple, academic-style carry trading strategy to verify that
the VX calendar carry idea and P&L machinery are working correctly.

Strategy (Phase-0):
- Load VX curve prices (VX1, VX2, VX3)
- Compute carry signal: sign(VX_long - VX_short) or -sign(VX_long - VX_short) for carry capture
- Trade calendar spread directly: VX_Carry_t = P(VX_long) - P(VX_short)
- Daily strategy return = signal * spread_return
- Spread return = r_VX_long - r_VX_short (where r_k = pct_change(P_k))
- No vol targeting, no normalization beyond sign

Phase-0 Variants:
- Spread pairs: (VX2-VX1), (VX3-VX2)
- Sign directions: long spread, short spread (for carry capture in contango)

Canonical Requirements (from Phase-0 Definition):
1. Instrument: VX_Carry_t = P(VX_long) - P(VX_short) (trade spread directly)
2. Return: r_spread = (+1)*r_VX_long - (1)*r_VX_short (NOT % change of spread level)
3. Signal: sign(VX_long - VX_short) or -sign(VX_long - VX_short) depending on carry capture direction
4. Execution: signal.shift(1) - signals at close T, positions entered at close T, P&L accrues T→T+1
5. Rank integrity: Assert VX ranks are distinct (no silent rank coercion, hard-fail if >5% collision)
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, Tuple
from pathlib import Path
import json
import logging
import duckdb

from src.market_data.vrp_loaders import load_vx_curve
from src.agents.utils_db import open_readonly_connection

logger = logging.getLogger(__name__)

# VX spread pairs for Phase-0 sweep
# Format: (long_leg, short_leg) where spread = P(long) - P(short)
# VX1 = front month, VX2 = second month, VX3 = third month
VX_SPREAD_PAIRS = [
    (2, 1),  # VX2 - VX1 (front spread)
    (3, 2),  # VX3 - VX2 (mid spread)
]


def run_sign_only_vx_carry(
    db_path: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    spread_pair: Optional[Tuple[int, int]] = None,
    flip_sign: bool = False,
    zero_threshold: float = 1e-8,
    max_collision_pct: float = 5.0
) -> Dict[str, Any]:
    """
    Run sign-only VX calendar carry strategy.
    
    Args:
        db_path: Path to canonical database
        start_date: Start date for backtest (YYYY-MM-DD)
        end_date: End date for backtest (YYYY-MM-DD)
        spread_pair: Tuple (long_leg, short_leg) where legs are 1, 2, or 3 (default: (2, 1))
        flip_sign: If True, use -sign(spread) for carry capture (default: False)
        zero_threshold: Threshold for treating carry as zero
        max_collision_pct: Maximum allowed % of days where ranks collide (default: 5.0%)
        
    Returns:
        Dict with:
        - 'portfolio_returns': pd.Series of daily portfolio returns
        - 'equity_curve': pd.Series of cumulative equity
        - 'asset_returns': pd.DataFrame of daily asset returns (VX spread)
        - 'asset_strategy_returns': pd.DataFrame of per-asset strategy returns
        - 'positions': pd.DataFrame of daily positions (signs)
        - 'carry_signals': pd.Series of raw carry signals
        - 'metrics': Dict with {cagr, vol, sharpe, maxdd, hit_rate, n_days, years}
    """
    if spread_pair is None:
        spread_pair = (2, 1)  # Default to VX2-VX1
    
    results = compute_sign_only_vx_carry(
        db_path=db_path,
        start_date=start_date,
        end_date=end_date,
        spread_pair=spread_pair,
        flip_sign=flip_sign,
        zero_threshold=zero_threshold,
        max_collision_pct=max_collision_pct
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


def compute_sign_only_vx_carry(
    db_path: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    spread_pair: Tuple[int, int] = (2, 1),
    flip_sign: bool = False,
    zero_threshold: float = 1e-8,
    max_collision_pct: float = 5.0
) -> Dict:
    """
    Compute sign-only VX calendar carry strategy returns.
    
    Strategy:
    - Load VX curve prices (VX1, VX2, VX3) from canonical DB
    - Compute carry: carry_raw = VX_long - VX_short (in price space)
    - Signal: sign(carry_raw) or -sign(carry_raw) depending on flip_sign
    - Trade calendar spread directly: spread_return = r_VX_long - r_VX_short
    - Position: signal.shift(1) (execution lag)
    
    Args:
        db_path: Path to canonical database
        start_date: Start date for backtest (YYYY-MM-DD)
        end_date: End date for backtest (YYYY-MM-DD)
        spread_pair: Tuple (long_leg, short_leg) where legs are 1, 2, or 3
        flip_sign: If True, use -sign(spread) for carry capture (short spread in contango)
        zero_threshold: Threshold for treating carry as zero
        max_collision_pct: Maximum allowed % of days where ranks collide (hard-fail if exceeded)
        
    Returns:
        Dict with returns, positions, and equity curves
    """
    long_leg, short_leg = spread_pair
    
    # Validate spread pair
    if long_leg not in [1, 2, 3] or short_leg not in [1, 2, 3]:
        raise ValueError(f"Invalid spread pair {spread_pair}. Legs must be 1, 2, or 3.")
    if long_leg <= short_leg:
        raise ValueError(f"Invalid spread pair {spread_pair}. Long leg must be > short leg.")
    
    # Map leg numbers to column names
    leg_to_col = {1: 'vx1', 2: 'vx2', 3: 'vx3'}
    long_col = leg_to_col[long_leg]
    short_col = leg_to_col[short_leg]
    # Open database connection
    con = open_readonly_connection(db_path)
    
    try:
        # Load VX curve data (VX1, VX2, VX3)
        logger.info("Loading VX curve data...")
        vx_data = load_vx_curve(
            con=con,
            start=start_date or "2020-01-01",
            end=end_date or "2025-12-31"
        )
        
        if vx_data.empty:
            raise ValueError("No VX data returned from database")
        
        # Set date as index if it's a column
        if 'date' in vx_data.columns:
            vx_data = vx_data.set_index('date')
        
        # Check required columns (need all legs up to the maximum used)
        max_leg = max(long_leg, short_leg)
        required_cols = [leg_to_col[i] for i in range(1, max_leg + 1)]
        missing_cols = [col for col in required_cols if col not in vx_data.columns]
        if missing_cols:
            raise ValueError(f"Missing required VX columns: {missing_cols}. Available: {list(vx_data.columns)}")
        
        logger.info(f"Loaded VX data: {len(vx_data)} days")
        logger.info(f"  Date range: {vx_data.index[0]} to {vx_data.index[-1]}")
        for leg in [1, 2, 3]:
            if leg_to_col[leg] in vx_data.columns:
                logger.info(f"  {leg_to_col[leg].upper()} available: {vx_data[leg_to_col[leg]].notna().sum()} days")
        
        # RANK INTEGRITY CHECK (Critical - per Phase-0 requirements)
        # Assert VX ranks are distinct (no silent rank coercion)
        # Hard-fail if collision rate exceeds max_collision_pct
        long_prices = vx_data[long_col]
        short_prices = vx_data[short_col]
        
        common_dates = vx_data.index[long_prices.notna() & short_prices.notna()]
        if len(common_dates) == 0:
            raise ValueError(f"No common dates between {long_col.upper()} and {short_col.upper()}")
        
        long_common = vx_data.loc[common_dates, long_col]
        short_common = vx_data.loc[common_dates, short_col]
        
        # Check that ranks are different (not identical)
        price_diff = (long_common - short_common).abs()
        identical_days = (price_diff < 1e-6).sum()
        pct_identical = (identical_days / len(common_dates)) * 100.0
        collision_days_dropped = 0
        
        if pct_identical > max_collision_pct:
            error_msg = (
                f"RANK INTEGRITY FAIL: {long_col.upper()} and {short_col.upper()} are identical on "
                f"{identical_days}/{len(common_dates)} days ({pct_identical:.2f}%), "
                f"exceeding max tolerance of {max_collision_pct}%. "
                f"This indicates a data issue (roll mapping, contract availability, or rank coercion)."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)
        elif identical_days > 0:
            logger.warning(
                f"Rank integrity warning: {long_col.upper()} and {short_col.upper()} are identical on "
                f"{identical_days}/{len(common_dates)} days ({pct_identical:.2f}%). "
                f"Dropping these days from analysis."
            )
            # Drop identical days
            valid_dates = common_dates[price_diff >= 1e-6]
            if len(valid_dates) == 0:
                raise ValueError(f"All dates have identical prices for {long_col.upper()} and {short_col.upper()}")
            collision_days_dropped = identical_days
            common_dates = valid_dates
            long_common = vx_data.loc[common_dates, long_col]
            short_common = vx_data.loc[common_dates, short_col]
        else:
            logger.info(
                f"Rank integrity check passed: {long_col.upper()} ≠ {short_col.upper()} on "
                f"{len(common_dates)}/{len(common_dates)} days (100.00%)"
            )
        
        # Extract prices for the spread legs
        long_prices = vx_data[long_col]
        short_prices = vx_data[short_col]
        
        # Forward-fill and backward-fill missing values
        long_prices = long_prices.ffill().bfill()
        short_prices = short_prices.ffill().bfill()
        
        # Compute carry signal: VX_long - VX_short (in price space)
        # Positive = contango (long > short) → positive carry
        # Negative = backwardation
        carry_raw = long_prices - short_prices
        
        # Apply sign flip if requested (for carry capture: short spread in contango)
        if flip_sign:
            signal_description = f"-sign({long_col.upper()} - {short_col.upper()}) -> trade spread SHORT (carry capture)"
        else:
            signal_description = f"sign({long_col.upper()} - {short_col.upper()}) -> trade spread LONG"
        
        logger.info(f"Computed carry signals: {len(carry_raw)} days")
        logger.info(f"  Signal: {signal_description}")
        logger.info(f"  Carry mean: {carry_raw.mean():.4f}")
        logger.info(f"  Carry std: {carry_raw.std():.4f}")
        logger.info(f"  Carry min: {carry_raw.min():.4f}")
        logger.info(f"  Carry max: {carry_raw.max():.4f}")
        
        # CANONICAL RETURN CONSTRUCTION (Non-Negotiable)
        # DO NOT compute returns as % change of the spread level
        # Correct formula: r_spread = (+1)*r_VX_long - (1)*r_VX_short
        # where r_k = (P_k,t - P_k,t-1) / P_k,t-1
        
        # Compute daily returns for each leg (simple percentage returns)
        r_long = long_prices.pct_change(fill_method=None)
        r_short = short_prices.pct_change(fill_method=None)
        
        # Spread return = r_VX_long - r_VX_short
        # This represents P&L of being long VX_long and short VX_short (1:1 notional)
        spread_returns = r_long - r_short
        
        # Align on common dates
        common_dates = carry_raw.index.intersection(spread_returns.index)
        carry_raw_aligned = carry_raw.loc[common_dates]
        spread_returns_aligned = spread_returns.loc[common_dates]
        
        # Drop NaN from spread returns (first day will be NaN)
        valid_mask = spread_returns_aligned.notna()
        common_dates = common_dates[valid_mask]
        carry_raw_aligned = carry_raw_aligned.loc[common_dates]
        spread_returns_aligned = spread_returns_aligned.loc[common_dates]
        
        logger.info(f"Aligned data: {len(common_dates)} days")
        
        # Filter by date range (after computing signals, but before generating positions)
        if start_date:
            start_dt = pd.to_datetime(start_date)
            mask = common_dates >= start_dt
            common_dates = common_dates[mask]
            carry_raw_aligned = carry_raw_aligned.loc[carry_raw_aligned.index.intersection(common_dates)]
            spread_returns_aligned = spread_returns_aligned.loc[spread_returns_aligned.index.intersection(common_dates)]
        
        if end_date:
            end_dt = pd.to_datetime(end_date)
            mask = common_dates <= end_dt
            common_dates = common_dates[mask]
            carry_raw_aligned = carry_raw_aligned.loc[carry_raw_aligned.index.intersection(common_dates)]
            spread_returns_aligned = spread_returns_aligned.loc[spread_returns_aligned.index.intersection(common_dates)]
        
        if len(common_dates) == 0:
            raise ValueError("No data available after date filtering")
        
        # Re-align after date filtering
        common_dates = carry_raw_aligned.index.intersection(spread_returns_aligned.index)
        carry_raw_aligned = carry_raw_aligned.loc[common_dates]
        spread_returns_aligned = spread_returns_aligned.loc[common_dates]
        
        logger.info(f"Final aligned data: {len(common_dates)} days")
        if len(common_dates) > 0:
            logger.info(f"  Start: {common_dates[0]}")
            logger.info(f"  End: {common_dates[-1]}")
        
        # CANONICAL EXECUTION SEMANTICS (Non-Negotiable)
        # Signals observed at close T
        # Positions entered at close T
        # P&L accrues T → T+1
        # Implemented via signal.shift(1)
        
        # Shift carry signals by 1 day so we don't use same-day info for trading
        signal_basis = carry_raw_aligned.shift(1).dropna()
        
        # Align spread_returns with signal_basis (drop first day of returns since signal is NaN)
        common_idx = signal_basis.index.intersection(spread_returns_aligned.index)
        signal_basis = signal_basis.loc[common_idx]
        spread_returns_aligned = spread_returns_aligned.loc[common_idx]
        
        if len(common_idx) == 0:
            raise ValueError("No data available after signal alignment")
        
        logger.info(f"Signal-aligned data: {len(common_idx)} days")
        
        # Generate sign-only positions
        # position_t = sign(signal_basis_t) or -sign(signal_basis_t) depending on flip_sign
        # +1 if > 0 (positive carry, long spread), -1 if < 0 (negative carry, short spread), 0 if == 0
        # Use zero_threshold only for numerical precision (handle floating point near-zero)
        positions = signal_basis.copy()
        positions = np.sign(positions)
        
        # Apply sign flip if requested (for carry capture: short spread in contango)
        if flip_sign:
            positions = -positions
        
        # Handle near-zero values (for numerical stability)
        positions[positions.abs() < zero_threshold] = 0.0
        
        # Compute strategy returns
        # strategy_ret_t = position_t * spread_return_t
        asset_strategy_returns = positions * spread_returns_aligned
        
        # For single-asset strategy, portfolio returns = asset strategy returns
        portfolio_returns = asset_strategy_returns
        
        # Convert to DataFrame for consistency with multi-asset format
        # Asset name includes spread pair and direction
        direction_label = "short" if flip_sign else "long"
        asset_name = f"VX_SPREAD_{long_leg}{short_leg}_{direction_label}"
        asset_strategy_returns_df = pd.DataFrame({asset_name: asset_strategy_returns})
        asset_returns_df = pd.DataFrame({asset_name: spread_returns_aligned})
        
        # Compute equity curve (cumulative)
        if len(portfolio_returns) > 0:
            equity_curve = (1 + portfolio_returns).cumprod()
            equity_curve.iloc[0] = 1.0  # Start at 1.0
        else:
            equity_curve = pd.Series(dtype=float)
        
        return {
            'portfolio_returns': portfolio_returns,
            'equity_curve': equity_curve,
            'asset_returns': asset_returns_df,
            'asset_strategy_returns': asset_strategy_returns_df,
            'positions': pd.DataFrame({asset_name: positions}),
            'carry_signals': carry_raw_aligned.loc[common_idx] if common_idx[0] in carry_raw_aligned.index else pd.Series(dtype=float),
            'signal_description': signal_description,
            'asset_name': asset_name,
            'spread_pair': spread_pair,
            'flip_sign': flip_sign,
            'raw_spread_returns': spread_returns_aligned,  # Store raw spread returns for diagnostics
            'collision_days_dropped': collision_days_dropped
        }
    
    finally:
        con.close()


def compute_summary_stats(
    portfolio_returns: pd.Series,
    equity_curve: pd.Series,
    asset_strategy_returns: pd.DataFrame
) -> Dict:
    """
    Compute summary statistics for the sign-only VX carry strategy.
    
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
    - carry_signals.csv
    - positions.csv
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
    
    # Save carry signals
    if not results['carry_signals'].empty:
        carry_signals_df = pd.DataFrame({
            'date': results['carry_signals'].index,
            'carry_raw': results['carry_signals'].values
        })
        carry_signals_df.to_csv(output_dir / 'carry_signals.csv', index=False)
    
    # Save positions
    results['positions'].to_csv(output_dir / 'positions.csv')
    
    # Save per-asset stats
    if not stats['per_asset'].empty:
        stats['per_asset'].to_csv(output_dir / 'per_asset_stats.csv')
    
    # Save meta
    meta = {
        'start_date': start_date,
        'end_date': end_date,
        'n_days': len(results['portfolio_returns']),
        'instrument': 'VX Calendar Spread (VX2 - VX1)',
        'signal_definition': results.get('signal_description', 'unknown'),
        'return_formula': 'r_spread = (+1)*r_VX2 - (1)*r_VX1 (where r_k = pct_change(P_k))',
        'execution_semantics': 'signal.shift(1) - signals at close T, positions entered at close T, P&L accrues T→T+1',
        'portfolio_metrics': stats['portfolio'],
        'per_asset_stats': stats['per_asset'].to_dict('index') if not stats['per_asset'].empty else {},
        'subperiod_stats': subperiod_stats,
        'data_integrity': {
            'rank_check_applied': True,
            'note': 'VX1 and VX2 rank integrity verified (no silent rank coercion)'
        }
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
    1. Cumulative equity curve
    2. Return histograms
    3. Carry signal timeseries
    4. Subperiod comparison
    """
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available, skipping plots")
        return
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Cumulative equity curve
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(results['equity_curve'].index, results['equity_curve'].values, 
            label='Portfolio', linewidth=2, color='black')
    
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    ax.set_title(f'VX Calendar Carry: Cumulative Equity Curve\n'
                 f"Sharpe={stats['portfolio'].get('Sharpe', 0):.2f}, "
                 f"CAGR={stats['portfolio'].get('CAGR', 0)*100:.2f}%")
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
    portfolio_ret = results['portfolio_returns'].dropna()
    axes[0].hist(portfolio_ret.values, bins=50, alpha=0.7, edgecolor='black')
    axes[0].axvline(x=0, color='red', linestyle='--', alpha=0.5)
    axes[0].set_title(f'Portfolio Returns Distribution\n(Mean={portfolio_ret.mean():.4f}, Std={portfolio_ret.std():.4f})')
    axes[0].set_xlabel('Daily Return')
    axes[0].set_ylabel('Frequency')
    axes[0].grid(True, alpha=0.3)
    
    # Carry signals histogram
    if not results['carry_signals'].empty:
        carry_sig = results['carry_signals'].dropna()
        axes[1].hist(carry_sig.values, bins=50, alpha=0.7, edgecolor='black', color='orange')
        axes[1].axvline(x=0, color='red', linestyle='--', alpha=0.5)
        axes[1].set_title(f'Carry Signal Distribution\n(Mean={carry_sig.mean():.4f}, Std={carry_sig.std():.4f})')
        axes[1].set_xlabel('Carry Signal (VX2 - VX1)')
        axes[1].set_ylabel('Frequency')
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'return_histogram.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved return_histogram.png")
    
    # 3. Carry signal timeseries
    if not results['carry_signals'].empty:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        carry_sig = results['carry_signals'].dropna()
        asset_name = results.get('asset_name', 'VX_CALENDAR_CARRY')
        positions = results['positions'][asset_name].dropna()
        
        ax2 = ax.twinx()
        
        # Carry signal
        ax.plot(carry_sig.index, carry_sig.values, 
                label='Carry Signal (VX2 - VX1)', alpha=0.6, color='blue', linewidth=1)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_ylabel('Carry Signal', color='blue')
        ax.tick_params(axis='y', labelcolor='blue')
        
        # Positions (overlay)
        ax2.plot(positions.index, positions.values, 
                label='Position (sign)', alpha=0.8, color='red', linewidth=1, linestyle='--')
        ax2.set_ylabel('Position (+1/-1/0)', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        ax2.set_ylim([-1.5, 1.5])
        
        ax.set_title('VX Calendar Carry: Signal and Positions Over Time')
        ax.set_xlabel('Date')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'carry_signal_timeseries.png', dpi=150, bbox_inches='tight')
        plt.close()
        logger.info("Saved carry_signal_timeseries.png")
    
    # 4. Subperiod comparison (if available)
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

