"""
CLI script for running SR3 Curve RV Phase-0 Sanity Check.

This script implements three atomic Curve RV expressions to verify that
curve shape mean reversion or momentum (not level) shows positive Sharpe before
proceeding with full implementation.

Strategy (Phase-0):
- Sign-only, no vol targeting, no z-scoring, daily rebalance
- Execution lag: compute signal on close T, hold position from T→T+1 via signal.shift(1)
- Rate space: r_k = 100 - P_k (for signals only)
- Price returns: pct_change(P_k) for leg returns (canonical P&L construction)
- Spread/fly constructions only (no outright directional exposure)

Modes:
- mean_reversion (default): signal = -sign(feature) - fade steepness/inversion/hump
- momentum: signal = sign(feature) - go with steepening/flattening/hump formation

Three Atomic Expressions:
A) Pack Slope RV (front vs back)
   - pack_front = mean(r0,r1,r2,r3)
   - pack_back = mean(r8,r9,r10,r11)
   - spread_ret = mean(ret_back_8_11) - mean(ret_front_0_3)
   - signal = -sign(pack_back - pack_front) [mean_reversion]
   - signal = sign(pack_back - pack_front) [momentum]

B) Belly Curvature RV (hump vs straight line)
   - curv = belly_pack - (front_pack + back_pack)/2
   - fly_ret = mean(ret_belly_4_7) - 0.5*mean(ret_front_0_3) - 0.5*mean(ret_back_8_11)
   - signal = -sign(curv) [mean_reversion]
   - signal = sign(curv) [momentum]

C) Simple Rank Fly (front–belly–back using single ranks)
   - fly_lvl = 2*r6 - r2 - r10
   - fly_ret = 2*ret6 - ret2 - ret10
   - signal = -sign(fly_lvl) [mean_reversion]
   - signal = sign(fly_lvl) [momentum]

Pass Criteria:
- Sharpe >= 0.2 over full window
- Subperiod check (pre/post 2022)

Usage:
    # Mean reversion (default)
    python scripts/run_sr3_curve_rv_phase0.py --start 2020-01-01 --end 2025-10-31
    
    # Momentum mode
    python scripts/run_sr3_curve_rv_phase0.py --start 2020-01-01 --end 2025-10-31 --mode momentum
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, Tuple
import pandas as pd
import numpy as np
import logging
import json

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.backtest_window import CANONICAL_START, CANONICAL_END
from src.agents import MarketData
from src.utils.phase_index import get_sleeve_dirs, copy_to_latest, update_phase_index

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Meta-sleeve and atomic sleeve names
META_SLEEVE = "rates_curve_rv"
ATOMIC_SLEEVES = {
    "pack_slope": {
        "mean_reversion": "sr3_curve_rv_pack_slope_fade",
        "momentum": "sr3_curve_rv_pack_slope_momentum"
    },
    "pack_curvature": {
        "mean_reversion": "sr3_curve_rv_pack_curvature_fade",
        "momentum": "sr3_curve_rv_pack_curvature_momentum"
    },
    "rank_fly": {
        "mean_reversion": "sr3_curve_rv_rank_fly_2_6_10_fade",
        "momentum": "sr3_curve_rv_rank_fly_2_6_10_momentum"
    }
}

# Pack definitions
FRONT_RANKS = [0, 1, 2, 3]
BELLY_RANKS = [4, 5, 6, 7]
BACK_RANKS = [8, 9, 10, 11]

# Rank fly definition
RANK_FLY_RANKS = (2, 6, 10)  # (front, belly, back)


def compute_summary_stats(
    portfolio_returns: pd.Series,
    equity_curve: pd.Series,
    asset_strategy_returns: pd.DataFrame
) -> Dict[str, Any]:
    """
    Compute summary statistics for Phase-0 results.
    
    Returns:
        Dict with 'portfolio' and 'per_asset' stats
    """
    if len(portfolio_returns) == 0:
        return {
            'portfolio': {},
            'per_asset': pd.DataFrame()
        }
    
    # Portfolio stats
    n_days = len(portfolio_returns)
    years = n_days / 252.0
    
    if n_days < 2:
        return {
            'portfolio': {
                'n_days': n_days,
                'years': years
            },
            'per_asset': pd.DataFrame()
        }
    
    # Annualized metrics
    mean_ret = portfolio_returns.mean()
    std_ret = portfolio_returns.std()
    
    cagr = (1 + mean_ret) ** 252 - 1 if mean_ret > -1 else np.nan
    vol = std_ret * np.sqrt(252) if std_ret > 0 else np.nan
    sharpe = (mean_ret / std_ret * np.sqrt(252)) if std_ret > 0 else np.nan
    
    # Max drawdown
    equity_array = equity_curve.values
    # Filter out NaN and inf values
    valid_mask = np.isfinite(equity_array)
    if valid_mask.sum() > 0:
        equity_valid = equity_array[valid_mask]
        running_max = np.maximum.accumulate(equity_valid)
        # Avoid division by zero
        running_max = np.maximum(running_max, 1e-10)
        drawdown = (equity_valid - running_max) / running_max
        maxdd = drawdown.min() if len(drawdown) > 0 else np.nan
    else:
        maxdd = np.nan
    
    # Hit rate
    hit_rate = (portfolio_returns > 0).sum() / n_days if n_days > 0 else np.nan
    
    portfolio_stats = {
        'CAGR': cagr,
        'Vol': vol,
        'Sharpe': sharpe,
        'MaxDD': maxdd,
        'HitRate': hit_rate,
        'n_days': n_days,
        'years': years
    }
    
    # Per-asset stats
    per_asset_stats = pd.DataFrame()
    if not asset_strategy_returns.empty:
        per_asset_list = []
        for col in asset_strategy_returns.columns:
            asset_ret = asset_strategy_returns[col].dropna()
            if len(asset_ret) > 0:
                asset_mean = asset_ret.mean()
                asset_std = asset_ret.std()
                asset_cagr = (1 + asset_mean) ** 252 - 1 if asset_mean > -1 else np.nan
                asset_vol = asset_std * np.sqrt(252) if asset_std > 0 else np.nan
                asset_sharpe = (asset_mean / asset_std * np.sqrt(252)) if asset_std > 0 else np.nan
                asset_hitrate = (asset_ret > 0).sum() / len(asset_ret) if len(asset_ret) > 0 else np.nan
                
                per_asset_list.append({
                    'asset': col,
                    'CAGR': asset_cagr,
                    'Vol': asset_vol,
                    'Sharpe': asset_sharpe,
                    'HitRate': asset_hitrate,
                    'n_days': len(asset_ret)
                })
        
        if per_asset_list:
            per_asset_stats = pd.DataFrame(per_asset_list)
    
    return {
        'portfolio': portfolio_stats,
        'per_asset': per_asset_stats
    }


def compute_subperiod_stats(
    portfolio_returns: pd.Series,
    equity_curve: pd.Series,
    break_date: str = "2022-01-01"
) -> Dict[str, Dict]:
    """
    Compute statistics for pre- and post-break_date subperiods.
    
    Returns:
        Dict with 'pre' and 'post' subperiod stats
    """
    break_dt = pd.to_datetime(break_date)
    
    pre_mask = portfolio_returns.index < break_dt
    post_mask = portfolio_returns.index >= break_dt
    
    pre_returns = portfolio_returns[pre_mask]
    post_returns = portfolio_returns[post_mask]
    
    pre_equity = equity_curve[pre_mask]
    post_equity = equity_curve[post_mask]
    
    subperiods = {}
    
    # Pre-period
    if len(pre_returns) > 0:
        pre_stats = compute_summary_stats(pre_returns, pre_equity, pd.DataFrame())
        subperiods['pre'] = pre_stats['portfolio']
        subperiods['pre']['start_date'] = str(pre_returns.index[0])
        subperiods['pre']['end_date'] = str(pre_returns.index[-1])
    else:
        subperiods['pre'] = {}
    
    # Post-period
    if len(post_returns) > 0:
        post_stats = compute_summary_stats(post_returns, post_equity, pd.DataFrame())
        subperiods['post'] = post_stats['portfolio']
        subperiods['post']['start_date'] = str(post_returns.index[0])
        subperiods['post']['end_date'] = str(post_returns.index[-1])
    else:
        subperiods['post'] = {}
    
    return subperiods


def compute_pack_slope_rv(
    market: MarketData,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    zero_threshold: float = 1e-8,
    mode: str = "mean_reversion"
) -> Dict[str, Any]:
    """
    Compute Pack Slope RV strategy (front vs back).
    
    Strategy:
    - pack_front = mean(r0,r1,r2,r3)
    - pack_back = mean(r8,r9,r10,r11)
    - spread_ret = mean(ret_back_8_11) - mean(ret_front_0_3)
    - signal = -sign(pack_back - pack_front)
    - position = signal.shift(1)
    """
    logger.info("Computing Pack Slope RV strategy...")
    
    # Load all required ranks
    required_ranks = FRONT_RANKS + BACK_RANKS
    close = market.get_contracts_by_root(
        root="SR3",
        ranks=required_ranks,
        fields=("close",),
        start=None,
        end=end_date
    )
    
    if close.empty:
        raise ValueError(f"No SR3 contract data found for ranks {required_ranks}")
    
    # Check required ranks
    missing_front = set(FRONT_RANKS) - set(close.columns)
    missing_back = set(BACK_RANKS) - set(close.columns)
    if missing_front or missing_back:
        raise ValueError(
            f"Missing required ranks. Front missing: {missing_front}, Back missing: {missing_back}. "
            f"Available: {list(close.columns)}"
        )
    
    logger.info(f"Loaded SR3 data: {len(close)} days, ranks: {sorted(close.columns)}")
    
    # Handle missing data: forward-fill and backward-fill
    close_filled = close.ffill().bfill()
    
    # Convert prices to rates: r_k = 100 - P_k
    rates = 100.0 - close_filled
    
    # Compute packs in rate space (for signal only)
    pack_front = rates[FRONT_RANKS].mean(axis=1)
    pack_back = rates[BACK_RANKS].mean(axis=1)
    
    # Compute pack slope (level-neutral slope spread)
    pack_slope = pack_back - pack_front
    
    # CANONICAL RETURN CONSTRUCTION: Use price returns, not rate returns
    # Compute price returns for each rank: r_k = pct_change(P_k)
    price_rets = close_filled.pct_change()
    
    # Compute pack returns (mean of rank price returns)
    ret_front = price_rets[FRONT_RANKS].mean(axis=1)
    ret_back = price_rets[BACK_RANKS].mean(axis=1)
    
    # Spread return: back - front (leg-return construction)
    spread_ret = ret_back - ret_front
    
    # Align dates
    common_idx = pack_slope.index.intersection(spread_ret.index)
    pack_slope = pack_slope.loc[common_idx]
    spread_ret = spread_ret.loc[common_idx]
    
    # Filter by date range
    if start_date:
        start_dt = pd.to_datetime(start_date)
        mask = common_idx >= start_dt
        common_idx = common_idx[mask]
        pack_slope = pack_slope.loc[common_idx]
        spread_ret = spread_ret.loc[common_idx]
    
    if end_date:
        end_dt = pd.to_datetime(end_date)
        mask = common_idx <= end_dt
        common_idx = common_idx[mask]
        pack_slope = pack_slope.loc[common_idx]
        spread_ret = spread_ret.loc[common_idx]
    
    # Generate signal based on mode
    if mode == "momentum":
        # Momentum: go with steepening/flattening
        # If back > front (steep), go long spread (expect more steepening)
        # If back < front (inverted), go short spread (expect more inversion)
        signal = np.sign(pack_slope)
    else:
        # Mean reversion (default): fade steepness/inversion
        # If back > front (steep), fade it (short spread)
        # If back < front (inverted), fade inversion (long spread)
        signal = -np.sign(pack_slope)
    signal[signal.abs() < zero_threshold] = 0.0
    
    # Execution lag: signal computed on close T, position held from T→T+1
    # Position at T+1 uses signal from T, and earns return from T→T+1
    # So we shift signal by 1, and use spread_ret at T+1 (which is return from T→T+1)
    position = signal.shift(1).dropna()
    
    # Align spread_ret with position (both indexed at T+1)
    common_idx = position.index.intersection(spread_ret.index)
    position = position.loc[common_idx]
    spread_ret = spread_ret.loc[common_idx]
    
    # Filter out NaN/inf values in spread_ret
    valid_mask = np.isfinite(spread_ret)
    position = position[valid_mask]
    spread_ret = spread_ret[valid_mask]
    
    # Strategy return: position * spread_return
    portfolio_returns = position * spread_ret
    
    # Compute equity curve
    if len(portfolio_returns) > 0:
        equity_curve = (1 + portfolio_returns).cumprod()
        equity_curve.iloc[0] = 1.0
    else:
        equity_curve = pd.Series(dtype=float)
    
    # Phase-0 validity checks
    max_daily_ret = portfolio_returns.abs().max() if len(portfolio_returns) > 0 else np.nan
    logger.info(f"  Phase-0 validity: max |daily return| = {max_daily_ret:.6f}")
    logger.info(f"  Leg return primitive: price pct_change (✓ correct)")
    logger.info(f"  Gross exposure: 1 long pack + 1 short pack = 2 (sensible)")
    
    # Format for consistency
    asset_strategy_returns = pd.DataFrame({'pack_slope_spread': portfolio_returns})
    asset_returns = pd.DataFrame({'pack_slope_spread': spread_ret})
    
    return {
        'portfolio_returns': portfolio_returns,
        'equity_curve': equity_curve,
        'asset_returns': asset_returns,
        'asset_strategy_returns': asset_strategy_returns,
        'positions': pd.DataFrame({'position': position}),
        'signals': pd.Series(signal, name='signal'),
        'pack_slope': pack_slope,
        'max_daily_ret': max_daily_ret
    }


def compute_pack_curvature_rv(
    market: MarketData,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    zero_threshold: float = 1e-8,
    mode: str = "mean_reversion"
) -> Dict[str, Any]:
    """
    Compute Belly Curvature RV strategy (hump vs straight line).
    
    Strategy:
    - curv = belly_pack - (front_pack + back_pack)/2
    - fly_ret = mean(ret_belly_4_7) - 0.5*mean(ret_front_0_3) - 0.5*mean(ret_back_8_11)
    - signal = -sign(curv)
    - position = signal.shift(1)
    """
    logger.info("Computing Pack Curvature RV strategy...")
    
    # Load all required ranks
    required_ranks = FRONT_RANKS + BELLY_RANKS + BACK_RANKS
    close = market.get_contracts_by_root(
        root="SR3",
        ranks=required_ranks,
        fields=("close",),
        start=None,
        end=end_date
    )
    
    if close.empty:
        raise ValueError(f"No SR3 contract data found for ranks {required_ranks}")
    
    # Check required ranks
    missing_front = set(FRONT_RANKS) - set(close.columns)
    missing_belly = set(BELLY_RANKS) - set(close.columns)
    missing_back = set(BACK_RANKS) - set(close.columns)
    if missing_front or missing_belly or missing_back:
        raise ValueError(
            f"Missing required ranks. Front missing: {missing_front}, "
            f"Belly missing: {missing_belly}, Back missing: {missing_back}. "
            f"Available: {list(close.columns)}"
        )
    
    logger.info(f"Loaded SR3 data: {len(close)} days, ranks: {sorted(close.columns)}")
    
    # Handle missing data: forward-fill and backward-fill
    close_filled = close.ffill().bfill()
    
    # Convert prices to rates: r_k = 100 - P_k
    rates = 100.0 - close_filled
    
    # Compute packs in rate space (for signal only)
    pack_front = rates[FRONT_RANKS].mean(axis=1)
    pack_belly = rates[BELLY_RANKS].mean(axis=1)
    pack_back = rates[BACK_RANKS].mean(axis=1)
    
    # Compute curvature: belly - (front + back)/2
    curvature = pack_belly - (pack_front + pack_back) / 2.0
    
    # CANONICAL RETURN CONSTRUCTION: Use price returns, not rate returns
    # Compute price returns for each rank: r_k = pct_change(P_k)
    price_rets = close_filled.pct_change()
    
    # Compute pack returns (mean of rank price returns)
    ret_front = price_rets[FRONT_RANKS].mean(axis=1)
    ret_belly = price_rets[BELLY_RANKS].mean(axis=1)
    ret_back = price_rets[BACK_RANKS].mean(axis=1)
    
    # Fly return: belly - 0.5*front - 0.5*back (leg-return construction)
    fly_ret = ret_belly - 0.5 * ret_front - 0.5 * ret_back
    
    # Align dates
    common_idx = curvature.index.intersection(fly_ret.index)
    curvature = curvature.loc[common_idx]
    fly_ret = fly_ret.loc[common_idx]
    
    # Filter by date range
    if start_date:
        start_dt = pd.to_datetime(start_date)
        mask = common_idx >= start_dt
        common_idx = common_idx[mask]
        curvature = curvature.loc[common_idx]
        fly_ret = fly_ret.loc[common_idx]
    
    if end_date:
        end_dt = pd.to_datetime(end_date)
        mask = common_idx <= end_dt
        common_idx = common_idx[mask]
        curvature = curvature.loc[common_idx]
        fly_ret = fly_ret.loc[common_idx]
    
    # Generate signal based on mode
    if mode == "momentum":
        # Momentum: go with hump formation/collapse
        # If curvature > 0 (hump), go long fly (expect more hump)
        # If curvature < 0 (U-shaped), go short fly (expect more U-shape)
        signal = np.sign(curvature)
    else:
        # Mean reversion (default): fade hump/U-shape
        # If curvature > 0 (hump), fade it (short fly)
        # If curvature < 0 (U-shaped), fade it (long fly)
        signal = -np.sign(curvature)
    signal[signal.abs() < zero_threshold] = 0.0
    
    # Execution lag: signal computed on close T, position held from T→T+1
    # Position at T+1 uses signal from T, and earns return from T→T+1
    position = signal.shift(1).dropna()
    
    # Align fly_ret with position (both indexed at T+1)
    common_idx = position.index.intersection(fly_ret.index)
    position = position.loc[common_idx]
    fly_ret = fly_ret.loc[common_idx]
    
    # Filter out NaN/inf values in fly_ret
    valid_mask = np.isfinite(fly_ret)
    position = position[valid_mask]
    fly_ret = fly_ret[valid_mask]
    
    # Strategy return: position * fly_return
    portfolio_returns = position * fly_ret
    
    # Compute equity curve
    if len(portfolio_returns) > 0:
        equity_curve = (1 + portfolio_returns).cumprod()
        equity_curve.iloc[0] = 1.0
    else:
        equity_curve = pd.Series(dtype=float)
    
    # Phase-0 validity checks
    max_daily_ret = portfolio_returns.abs().max() if len(portfolio_returns) > 0 else np.nan
    logger.info(f"  Phase-0 validity: max |daily return| = {max_daily_ret:.6f}")
    logger.info(f"  Leg return primitive: price pct_change (✓ correct)")
    logger.info(f"  Gross exposure: 1 belly + 0.5 front + 0.5 back = 2 (sensible)")
    
    # Format for consistency
    asset_strategy_returns = pd.DataFrame({'pack_curvature_fly': portfolio_returns})
    asset_returns = pd.DataFrame({'pack_curvature_fly': fly_ret})
    
    return {
        'portfolio_returns': portfolio_returns,
        'equity_curve': equity_curve,
        'asset_returns': asset_returns,
        'asset_strategy_returns': asset_strategy_returns,
        'positions': pd.DataFrame({'position': position}),
        'signals': pd.Series(signal, name='signal'),
        'curvature': curvature,
        'max_daily_ret': max_daily_ret
    }


def compute_rank_fly_rv(
    market: MarketData,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    zero_threshold: float = 1e-8,
    mode: str = "mean_reversion"
) -> Dict[str, Any]:
    """
    Compute Simple Rank Fly RV strategy (front–belly–back using single ranks).
    
    Strategy:
    - fly_lvl = 2*r6 - r2 - r10
    - fly_ret = 2*ret6 - ret2 - ret10
    - signal = -sign(fly_lvl)
    - position = signal.shift(1)
    """
    logger.info("Computing Rank Fly RV strategy...")
    
    # Load required ranks
    front_rank, belly_rank, back_rank = RANK_FLY_RANKS
    required_ranks = [front_rank, belly_rank, back_rank]
    
    close = market.get_contracts_by_root(
        root="SR3",
        ranks=required_ranks,
        fields=("close",),
        start=None,
        end=end_date
    )
    
    if close.empty:
        raise ValueError(f"No SR3 contract data found for ranks {required_ranks}")
    
    # Check required ranks
    missing = set(required_ranks) - set(close.columns)
    if missing:
        raise ValueError(
            f"Missing required ranks: {missing}. Available: {list(close.columns)}"
        )
    
    logger.info(f"Loaded SR3 data: {len(close)} days, ranks: {sorted(close.columns)}")
    
    # Handle missing data: forward-fill and backward-fill
    close_filled = close.ffill().bfill()
    
    # Convert prices to rates: r_k = 100 - P_k
    rates = 100.0 - close_filled
    
    # Extract individual ranks (for signal in rate space)
    r_front = rates[front_rank]
    r_belly = rates[belly_rank]
    r_back = rates[back_rank]
    
    # Compute fly level: 2*r6 - r2 - r10 (rate space, for signal only)
    fly_lvl = 2.0 * r_belly - r_front - r_back
    
    # CANONICAL RETURN CONSTRUCTION: Use price returns, not rate returns
    # Compute price returns for each rank: r_k = pct_change(P_k)
    price_rets = close_filled.pct_change()
    
    ret_front = price_rets[front_rank]
    ret_belly = price_rets[belly_rank]
    ret_back = price_rets[back_rank]
    
    # Fly return: 2*ret6 - ret2 - ret10 (leg-return construction)
    fly_ret = 2.0 * ret_belly - ret_front - ret_back
    
    # Align dates
    common_idx = fly_lvl.index.intersection(fly_ret.index)
    fly_lvl = fly_lvl.loc[common_idx]
    fly_ret = fly_ret.loc[common_idx]
    
    # Filter by date range
    if start_date:
        start_dt = pd.to_datetime(start_date)
        mask = common_idx >= start_dt
        common_idx = common_idx[mask]
        fly_lvl = fly_lvl.loc[common_idx]
        fly_ret = fly_ret.loc[common_idx]
    
    if end_date:
        end_dt = pd.to_datetime(end_date)
        mask = common_idx <= end_dt
        common_idx = common_idx[mask]
        fly_lvl = fly_lvl.loc[common_idx]
        fly_ret = fly_ret.loc[common_idx]
    
    # Generate signal based on mode
    if mode == "momentum":
        # Momentum: go with hump formation/collapse
        # If fly_lvl > 0 (hump), go long fly (expect more hump)
        # If fly_lvl < 0 (U-shaped), go short fly (expect more U-shape)
        signal = np.sign(fly_lvl)
    else:
        # Mean reversion (default): fade hump/U-shape
        # If fly_lvl > 0 (hump), fade it (short fly)
        # If fly_lvl < 0 (U-shaped), fade it (long fly)
        signal = -np.sign(fly_lvl)
    signal[signal.abs() < zero_threshold] = 0.0
    
    # Execution lag: signal computed on close T, position held from T→T+1
    # Position at T+1 uses signal from T, and earns return from T→T+1
    position = signal.shift(1).dropna()
    
    # Align fly_ret with position (both indexed at T+1)
    common_idx = position.index.intersection(fly_ret.index)
    position = position.loc[common_idx]
    fly_ret = fly_ret.loc[common_idx]
    
    # Filter out NaN/inf values in fly_ret
    valid_mask = np.isfinite(fly_ret)
    position = position[valid_mask]
    fly_ret = fly_ret[valid_mask]
    
    # Strategy return: position * fly_return
    portfolio_returns = position * fly_ret
    
    # Compute equity curve
    if len(portfolio_returns) > 0:
        equity_curve = (1 + portfolio_returns).cumprod()
        equity_curve.iloc[0] = 1.0
    else:
        equity_curve = pd.Series(dtype=float)
    
    # Phase-0 validity checks
    max_daily_ret = portfolio_returns.abs().max() if len(portfolio_returns) > 0 else np.nan
    logger.info(f"  Phase-0 validity: max |daily return| = {max_daily_ret:.6f}")
    logger.info(f"  Leg return primitive: price pct_change (✓ correct)")
    logger.info(f"  Gross exposure: 2 belly + 1 front + 1 back = 4 (sensible)")
    
    # Format for consistency
    asset_strategy_returns = pd.DataFrame({'rank_fly': portfolio_returns})
    asset_returns = pd.DataFrame({'rank_fly': fly_ret})
    
    return {
        'portfolio_returns': portfolio_returns,
        'equity_curve': equity_curve,
        'asset_returns': asset_returns,
        'asset_strategy_returns': asset_strategy_returns,
        'positions': pd.DataFrame({'position': position}),
        'signals': pd.Series(signal, name='signal'),
        'fly_lvl': fly_lvl,
        'max_daily_ret': max_daily_ret
    }


def save_results(
    results: Dict,
    stats: Dict,
    subperiod_stats: Dict,
    output_dir: Path,
    start_date: Optional[str],
    end_date: Optional[str],
    sleeve_name: str,
    mode: str = "mean_reversion"
):
    """Save all results to output directory."""
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
    
    # Save positions
    results['positions'].to_csv(output_dir / 'positions.csv')
    
    # Save signals
    if 'signals' in results and not results['signals'].empty:
        signals_df = pd.DataFrame({
            'date': results['signals'].index,
            'signal': results['signals'].values
        })
        signals_df.to_csv(output_dir / 'signals.csv', index=False)
    
    # Save per-asset stats
    if not stats['per_asset'].empty:
        stats['per_asset'].to_csv(output_dir / 'per_asset_stats.csv')
    
    # Phase-0 validity checklist
    max_daily_ret = results.get('max_daily_ret', np.nan)
    
    # Save meta
    meta = {
        'start_date': start_date,
        'end_date': end_date,
        'sleeve_name': sleeve_name,
        'meta_sleeve': META_SLEEVE,
        'strategy_type': 'Phase-0 (sign-only, no vol targeting)',
        'mode': mode,
        'execution_lag': 'signal.shift(1)',
        'rate_space': 'r_k = 100 - P_k (for signals only)',
        'return_primitive': 'price pct_change (leg returns, not rate returns)',
        'metrics': stats['portfolio'],
        'subperiod_stats': subperiod_stats,
        'phase0_validity': {
            'leg_return_primitive': 'price pct_change (✓ correct)',
            'max_daily_return': float(max_daily_ret) if not pd.isna(max_daily_ret) else None,
            'gross_exposure': 'sensible (pack slope: 2, pack curvature: 2, rank fly: 4)'
        }
    }
    
    with open(output_dir / 'meta.json', 'w') as f:
        json.dump(meta, f, indent=2, default=str)
    
    logger.info(f"Results saved to {output_dir}")


def generate_plots(
    results: Dict,
    stats: Dict,
    subperiod_stats: Dict,
    output_dir: Path,
    sleeve_name: str
):
    """Generate diagnostic plots."""
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
    ax.set_title(f'SR3 Curve RV ({sleeve_name}): Cumulative Equity Curve\n'
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
    
    # 2. Return histogram
    fig, ax = plt.subplots(figsize=(10, 5))
    
    portfolio_ret = results['portfolio_returns'].dropna()
    # Filter out inf and NaN values
    portfolio_ret_clean = portfolio_ret[np.isfinite(portfolio_ret)]
    
    if len(portfolio_ret_clean) > 0:
        ax.hist(portfolio_ret_clean.values, bins=50, alpha=0.7, edgecolor='black')
        ax.axvline(x=0, color='red', linestyle='--', alpha=0.5)
        ax.set_title(f'Portfolio Returns Distribution\n(Mean={portfolio_ret_clean.mean():.4f}, Std={portfolio_ret_clean.std():.4f})')
        ax.set_xlabel('Daily Return')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No valid returns to plot', 
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes)
        ax.set_title('Portfolio Returns Distribution (No Data)')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'return_histogram.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved return_histogram.png")
    
    # 3. Signal timeseries (if available)
    if 'signals' in results and not results['signals'].empty:
        fig, ax = plt.subplots(figsize=(12, 4))
        
        ax.plot(results['signals'].index, results['signals'].values, 
                linewidth=1, alpha=0.7, color='blue')
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_title('Signal Timeseries')
        ax.set_xlabel('Date')
        ax.set_ylabel('Signal')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'signals.png', dpi=150, bbox_inches='tight')
        plt.close()
        logger.info("Saved signals.png")


def run_single_sleeve(
    market: MarketData,
    sleeve_key: str,
    start_date: Optional[str],
    end_date: Optional[str],
    break_date: str = "2022-01-01",
    mode: str = "mean_reversion"
) -> Dict[str, Any]:
    """Run Phase-0 for a single atomic sleeve."""
    sleeve_name = ATOMIC_SLEEVES[sleeve_key][mode]
    
    logger.info(f"\n{'='*80}")
    logger.info(f"Running Phase-0 for: {sleeve_name} (mode: {mode})")
    logger.info(f"{'='*80}")
    
    # Compute strategy based on sleeve
    if sleeve_key == "pack_slope":
        result = compute_pack_slope_rv(market, start_date, end_date, mode=mode)
    elif sleeve_key == "pack_curvature":
        result = compute_pack_curvature_rv(market, start_date, end_date, mode=mode)
    elif sleeve_key == "rank_fly":
        result = compute_rank_fly_rv(market, start_date, end_date, mode=mode)
    else:
        raise ValueError(f"Unknown sleeve key: {sleeve_key}")
    
    # Compute stats
    stats = compute_summary_stats(
        portfolio_returns=result['portfolio_returns'],
        equity_curve=result['equity_curve'],
        asset_strategy_returns=result['asset_strategy_returns']
    )
    
    result['metrics'] = stats['portfolio']
    result['per_asset_stats'] = stats['per_asset']
    
    # Compute subperiod stats
    subperiod_stats = compute_subperiod_stats(
        portfolio_returns=result['portfolio_returns'],
        equity_curve=result['equity_curve'],
        break_date=break_date
    )
    
    # Determine output directory
    sleeve_dirs = get_sleeve_dirs(META_SLEEVE, sleeve_name)
    archive_dir = sleeve_dirs["archive"]
    latest_dir = sleeve_dirs["latest"]
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_dir.mkdir(parents=True, exist_ok=True)
    timestamp_dir = archive_dir / timestamp
    timestamp_dir.mkdir(parents=True, exist_ok=True)
    
    # Save results
    save_results(
        results=result,
        stats=stats,
        subperiod_stats=subperiod_stats,
        output_dir=timestamp_dir,
        start_date=start_date,
        end_date=end_date,
        sleeve_name=sleeve_name,
        mode=mode
    )
    
    # Generate plots
    generate_plots(
        results=result,
        stats=stats,
        subperiod_stats=subperiod_stats,
        output_dir=timestamp_dir,
        sleeve_name=sleeve_name
    )
    
    # Copy to latest
    copy_to_latest(timestamp_dir, latest_dir)
    
    # Update phase index
    update_phase_index(META_SLEEVE, sleeve_name, "phase0")
    
    # Print summary
    metrics = result['metrics']
    max_daily_ret = result.get('max_daily_ret', np.nan)
    print("\n" + "=" * 80)
    print(f"SR3 CURVE RV PHASE-0 RESULTS: {sleeve_name}")
    print("=" * 80)
    print(f"CAGR:      {metrics.get('CAGR', float('nan')):8.4f} ({metrics.get('CAGR', 0)*100:6.2f}%)")
    print(f"Vol:       {metrics.get('Vol', float('nan')):8.4f} ({metrics.get('Vol', 0)*100:6.2f}%)")
    print(f"Sharpe:    {metrics.get('Sharpe', float('nan')):8.4f}")
    print(f"MaxDD:     {metrics.get('MaxDD', float('nan')):8.4f} ({metrics.get('MaxDD', 0)*100:6.2f}%)")
    print(f"HitRate:   {metrics.get('HitRate', float('nan')):8.4f}")
    print(f"N Days:    {metrics.get('n_days', 0):8d}")
    print(f"Years:     {metrics.get('years', float('nan')):8.2f}")
    max_ret_status = 'OK' if max_daily_ret < 0.1 else 'OUTLIER'
    print(f"Max |Ret|: {max_daily_ret:8.6f} ({max_ret_status})")
    
    # Subperiod stats
    if subperiod_stats.get('pre'):
        pre_sharpe = subperiod_stats['pre'].get('Sharpe', float('nan'))
        print(f"\nPre-2022 Sharpe:  {pre_sharpe:8.4f}")
    
    if subperiod_stats.get('post'):
        post_sharpe = subperiod_stats['post'].get('Sharpe', float('nan'))
        print(f"Post-2022 Sharpe:  {post_sharpe:8.4f}")
    
    # Phase-0 pass criteria
    print("\n" + "=" * 80)
    print("PHASE-0 PASS CRITERIA EVALUATION")
    print("=" * 80)
    
    sharpe = metrics.get('Sharpe', float('nan'))
    if not pd.isna(sharpe):
        if sharpe >= 0.2:
            print(f"[PASS] Sharpe >= 0.2: {sharpe:.4f}")
        else:
            print(f"[FAIL] Sharpe < 0.2: {sharpe:.4f}")
    else:
        print("[FAIL] Sharpe could not be computed")
    
    print(f"\nResults saved to: {timestamp_dir}")
    print(f"Phase-0 registered in: reports/phase_index/{META_SLEEVE}/{sleeve_name}/phase0.txt")
    
    return {
        'result': result,
        'stats': stats,
        'subperiod_stats': subperiod_stats,
        'timestamp_dir': timestamp_dir,
        'latest_dir': latest_dir
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run SR3 Curve RV Phase-0 Sanity Check",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all three atomic sleeves
  python scripts/run_sr3_curve_rv_phase0.py --start 2020-01-01 --end 2025-10-31
  
  # Run specific sleeve
  python scripts/run_sr3_curve_rv_phase0.py --start 2020-01-01 --end 2025-10-31 --sleeve pack_slope
        """
    )
    
    parser.add_argument(
        "--start",
        type=str,
        default=CANONICAL_START,
        help=f"Start date for backtest (YYYY-MM-DD), default: {CANONICAL_START}"
    )
    parser.add_argument(
        "--end",
        type=str,
        default=CANONICAL_END,
        help=f"End date for backtest (YYYY-MM-DD), default: {CANONICAL_END}"
    )
    parser.add_argument(
        "--sleeve",
        type=str,
        choices=list(ATOMIC_SLEEVES.keys()) + ['all'],
        default='all',
        help="Which atomic sleeve to run (default: all)"
    )
    parser.add_argument(
        "--break-date",
        type=str,
        default="2022-01-01",
        help="Break date for subperiod analysis (default: 2022-01-01)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=['mean_reversion', 'momentum'],
        default='mean_reversion',
        help="Signal mode: 'mean_reversion' (fade) or 'momentum' (go with) (default: mean_reversion)"
    )
    
    args = parser.parse_args()
    
    try:
        logger.info("=" * 80)
        logger.info("SR3 CURVE RV PHASE-0 SANITY CHECK")
        logger.info("=" * 80)
        logger.info(f"Start date: {args.start}")
        logger.info(f"End date: {args.end}")
        logger.info(f"Break date: {args.break_date}")
        
        # Initialize MarketData
        logger.info("\n[1/3] Initializing MarketData broker...")
        market = MarketData()
        logger.info(f"  MarketData initialized")
        
        # Determine which sleeves to run
        if args.sleeve == 'all':
            sleeves_to_run = list(ATOMIC_SLEEVES.keys())
        else:
            sleeves_to_run = [args.sleeve]
        
        logger.info(f"\n[2/3] Running Phase-0 for {len(sleeves_to_run)} sleeve(s)...")
        
        # Run each sleeve
        results = {}
        for sleeve_key in sleeves_to_run:
            try:
                sleeve_result = run_single_sleeve(
                    market=market,
                    sleeve_key=sleeve_key,
                    start_date=args.start,
                    end_date=args.end,
                    break_date=args.break_date,
                    mode=args.mode
                )
                results[sleeve_key] = sleeve_result
            except Exception as e:
                logger.error(f"Error running {sleeve_key}: {e}", exc_info=True)
                continue
        
        logger.info("\n[3/3] Phase-0 diagnostics complete!")
        logger.info("=" * 80)
        
        # Close market connection
        market.close()
        
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

