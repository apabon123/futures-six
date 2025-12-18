"""
SR3 Curve RV Momentum Phase-1 Strategy

Engineered, tradable implementation of SR3 curve shape momentum.
Upgrades Phase-0 sign-only sanity check with:
- Z-scored signal normalization
- Volatility targeting
- Signal smoothing
- Proper risk scaling

Three atomic sleeves:
1. Pack Slope Momentum: momentum on front vs back pack slope
2. Pack Curvature Momentum: momentum on belly curvature
3. Rank Fly Momentum: momentum on rank fly (2,6,10)
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)

# Pack definitions
FRONT_RANKS = [0, 1, 2, 3]
BELLY_RANKS = [4, 5, 6, 7]
BACK_RANKS = [8, 9, 10, 11]

# Rank fly definition
RANK_FLY_RANKS = (2, 6, 10)  # (front, belly, back)


def compute_pack_slope_momentum_phase1(
    market,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    zscore_window: int = 252,
    clip: float = 3.0,
    target_vol: float = 0.10,
    vol_lookback: int = 63,
    min_vol_floor: float = 0.01,
    max_leverage: float = 10.0,
    lag: int = 1
) -> Dict[str, Any]:
    """
    Compute Pack Slope Momentum Phase-1 signals and positions.
    
    Strategy:
    - pack_front = mean(r0,r1,r2,r3)
    - pack_back = mean(r8,r9,r10,r11)
    - pack_slope = pack_back - pack_front (rate space, for signal)
    - spread_ret = mean(ret_back) - mean(ret_front) (price returns, for P&L)
    - signal = sign(pack_slope) [momentum: go with steepening/flattening]
    - Z-score and vol-target the signal
    
    Args:
        market: MarketData instance
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        zscore_window: Rolling window for z-score normalization (default: 252 days)
        clip: Symmetric clipping bounds for normalized signal (default: ±3.0)
        target_vol: Target annualized volatility (default: 0.10 = 10%)
        vol_lookback: Rolling window for realized vol calculation (default: 63 days)
        min_vol_floor: Minimum annualized vol floor (default: 0.01 = 1%)
        max_leverage: Maximum leverage cap (default: 10.0)
        lag: Execution lag in days (default: 1)
        
    Returns:
        Dict with signals, positions, returns, equity curve, and metadata
    """
    logger.info("[SR3CurveRVMomentumPhase1] Computing Pack Slope Momentum...")
    
    # Load required ranks
    required_ranks = FRONT_RANKS + BACK_RANKS
    close = market.get_contracts_by_root(
        root="SR3",
        ranks=required_ranks,
        fields=("close",),
        start=None,  # Get all data for rolling windows
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
    
    # Handle missing data
    close_filled = close.ffill().bfill()
    
    # Convert prices to rates: r_k = 100 - P_k (for signals only)
    rates = 100.0 - close_filled
    
    # Compute packs in rate space
    pack_front = rates[FRONT_RANKS].mean(axis=1)
    pack_back = rates[BACK_RANKS].mean(axis=1)
    
    # Compute pack slope (level-neutral slope spread)
    pack_slope = pack_back - pack_front
    
    # CANONICAL RETURN CONSTRUCTION: Use price returns, not rate returns
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
    
    # Filter out NaN/inf values
    valid_mask = np.isfinite(spread_ret) & np.isfinite(pack_slope)
    pack_slope = pack_slope[valid_mask]
    spread_ret = spread_ret[valid_mask]
    
    logger.info(f"[SR3CurveRVMomentumPhase1] Pack Slope: {len(pack_slope)} days")
    logger.info(f"  Effective start: {pack_slope.index[0]}")
    logger.info(f"  Effective end: {pack_slope.index[-1]}")
    
    # Base signal: momentum (go with steepening/flattening)
    base_signal = pack_slope  # Positive = steep, negative = inverted
    
    # Z-score normalization
    rolling_mean = base_signal.rolling(
        window=zscore_window,
        min_periods=max(2, zscore_window // 2)
    ).mean()
    
    rolling_std = base_signal.rolling(
        window=zscore_window,
        min_periods=max(2, zscore_window // 2)
    ).std()
    
    # Compute z-scores
    z_scores = (base_signal - rolling_mean) / rolling_std
    z_scores = z_scores.fillna(0.0)
    
    # Clip to symmetric bounds
    normalized_signals = z_scores.clip(-clip, clip)
    
    logger.info(f"[SR3CurveRVMomentumPhase1] Normalized signal stats:")
    logger.info(f"  Mean: {normalized_signals.mean():.4f}")
    logger.info(f"  Std: {normalized_signals.std():.4f}")
    logger.info(f"  Min: {normalized_signals.min():.4f}")
    logger.info(f"  Max: {normalized_signals.max():.4f}")
    
    # Vol targeting
    rolling_vol = spread_ret.rolling(
        window=vol_lookback,
        min_periods=vol_lookback
    ).std() * np.sqrt(252)  # Annualize
    
    rolling_vol = rolling_vol.clip(lower=min_vol_floor)
    
    vol_scalar = target_vol / rolling_vol
    vol_scalar = vol_scalar.fillna(1.0)
    vol_scalar = vol_scalar.clip(0.0, max_leverage)
    
    # Apply vol targeting and lag
    positions = normalized_signals * vol_scalar
    positions = positions.shift(lag)
    
    # Compute portfolio returns
    portfolio_dates = positions.index.intersection(spread_ret.index)
    portfolio_returns = positions.loc[portfolio_dates] * spread_ret.loc[portfolio_dates]
    portfolio_returns = portfolio_returns.dropna()
    
    # Compute equity curve
    equity_curve = (1 + portfolio_returns).cumprod()
    
    metadata = {
        'sleeve': 'pack_slope_momentum',
        'phase': 'Phase-1',
        'zscore_window': zscore_window,
        'clip': clip,
        'target_vol': target_vol,
        'vol_lookback': vol_lookback,
        'min_vol_floor': min_vol_floor,
        'max_leverage': max_leverage,
        'lag': lag,
        'effective_start_date': str(pack_slope.index[0].date()),
        'effective_end_date': str(pack_slope.index[-1].date()),
        'n_days': len(pack_slope)
    }
    
    return {
        'signals': normalized_signals,
        'positions': positions,
        'spread_returns': spread_ret,
        'portfolio_returns': portfolio_returns,
        'equity_curve': equity_curve,
        'metadata': metadata
    }


def compute_pack_curvature_momentum_phase1(
    market,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    zscore_window: int = 252,
    clip: float = 3.0,
    target_vol: float = 0.10,
    vol_lookback: int = 63,
    min_vol_floor: float = 0.01,
    max_leverage: float = 10.0,
    lag: int = 1
) -> Dict[str, Any]:
    """
    Compute Pack Curvature Momentum Phase-1 signals and positions.
    
    Strategy:
    - curv = belly_pack - (front_pack + back_pack)/2 (rate space, for signal)
    - fly_ret = mean(ret_belly) - 0.5*mean(ret_front) - 0.5*mean(ret_back) (price returns, for P&L)
    - signal = sign(curv) [momentum: go with hump formation/collapse]
    - Z-score and vol-target the signal
    
    Args: Same as compute_pack_slope_momentum_phase1
    
    Returns: Same structure as compute_pack_slope_momentum_phase1
    """
    logger.info("[SR3CurveRVMomentumPhase1] Computing Pack Curvature Momentum...")
    
    # Load required ranks
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
    
    # Handle missing data
    close_filled = close.ffill().bfill()
    
    # Convert prices to rates: r_k = 100 - P_k (for signals only)
    rates = 100.0 - close_filled
    
    # Compute packs in rate space
    pack_front = rates[FRONT_RANKS].mean(axis=1)
    pack_belly = rates[BELLY_RANKS].mean(axis=1)
    pack_back = rates[BACK_RANKS].mean(axis=1)
    
    # Compute curvature: belly - (front + back)/2
    curvature = pack_belly - (pack_front + pack_back) / 2.0
    
    # CANONICAL RETURN CONSTRUCTION: Use price returns, not rate returns
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
    
    # Filter out NaN/inf values
    valid_mask = np.isfinite(fly_ret) & np.isfinite(curvature)
    curvature = curvature[valid_mask]
    fly_ret = fly_ret[valid_mask]
    
    logger.info(f"[SR3CurveRVMomentumPhase1] Pack Curvature: {len(curvature)} days")
    logger.info(f"  Effective start: {curvature.index[0]}")
    logger.info(f"  Effective end: {curvature.index[-1]}")
    
    # Base signal: momentum (go with hump formation/collapse)
    base_signal = curvature  # Positive = hump, negative = U-shaped
    
    # Z-score normalization
    rolling_mean = base_signal.rolling(
        window=zscore_window,
        min_periods=max(2, zscore_window // 2)
    ).mean()
    
    rolling_std = base_signal.rolling(
        window=zscore_window,
        min_periods=max(2, zscore_window // 2)
    ).std()
    
    # Compute z-scores
    z_scores = (base_signal - rolling_mean) / rolling_std
    z_scores = z_scores.fillna(0.0)
    
    # Clip to symmetric bounds
    normalized_signals = z_scores.clip(-clip, clip)
    
    logger.info(f"[SR3CurveRVMomentumPhase1] Normalized signal stats:")
    logger.info(f"  Mean: {normalized_signals.mean():.4f}")
    logger.info(f"  Std: {normalized_signals.std():.4f}")
    logger.info(f"  Min: {normalized_signals.min():.4f}")
    logger.info(f"  Max: {normalized_signals.max():.4f}")
    
    # Vol targeting
    rolling_vol = fly_ret.rolling(
        window=vol_lookback,
        min_periods=vol_lookback
    ).std() * np.sqrt(252)  # Annualize
    
    rolling_vol = rolling_vol.clip(lower=min_vol_floor)
    
    vol_scalar = target_vol / rolling_vol
    vol_scalar = vol_scalar.fillna(1.0)
    vol_scalar = vol_scalar.clip(0.0, max_leverage)
    
    # Apply vol targeting and lag
    positions = normalized_signals * vol_scalar
    positions = positions.shift(lag)
    
    # Compute portfolio returns
    portfolio_dates = positions.index.intersection(fly_ret.index)
    portfolio_returns = positions.loc[portfolio_dates] * fly_ret.loc[portfolio_dates]
    portfolio_returns = portfolio_returns.dropna()
    
    # Compute equity curve
    equity_curve = (1 + portfolio_returns).cumprod()
    
    metadata = {
        'sleeve': 'pack_curvature_momentum',
        'phase': 'Phase-1',
        'zscore_window': zscore_window,
        'clip': clip,
        'target_vol': target_vol,
        'vol_lookback': vol_lookback,
        'min_vol_floor': min_vol_floor,
        'max_leverage': max_leverage,
        'lag': lag,
        'effective_start_date': str(curvature.index[0].date()),
        'effective_end_date': str(curvature.index[-1].date()),
        'n_days': len(curvature)
    }
    
    return {
        'signals': normalized_signals,
        'positions': positions,
        'spread_returns': fly_ret,
        'portfolio_returns': portfolio_returns,
        'equity_curve': equity_curve,
        'metadata': metadata
    }


def compute_rank_fly_momentum_phase1(
    market,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    zscore_window: int = 252,
    clip: float = 3.0,
    target_vol: float = 0.10,
    vol_lookback: int = 63,
    min_vol_floor: float = 0.01,
    max_leverage: float = 10.0,
    lag: int = 1
) -> Dict[str, Any]:
    """
    Compute Rank Fly Momentum Phase-1 signals and positions.
    
    Strategy:
    - fly_lvl = 2*r6 - r2 - r10 (rate space, for signal)
    - fly_ret = 2*ret6 - ret2 - ret10 (price returns, for P&L)
    - signal = sign(fly_lvl) [momentum: go with hump formation/collapse]
    - Z-score and vol-target the signal
    
    Args: Same as compute_pack_slope_momentum_phase1
    
    Returns: Same structure as compute_pack_slope_momentum_phase1
    """
    logger.info("[SR3CurveRVMomentumPhase1] Computing Rank Fly Momentum...")
    
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
    
    # Handle missing data
    close_filled = close.ffill().bfill()
    
    # Convert prices to rates: r_k = 100 - P_k (for signals only)
    rates = 100.0 - close_filled
    
    # Extract individual ranks
    r_front = rates[front_rank]
    r_belly = rates[belly_rank]
    r_back = rates[back_rank]
    
    # Compute fly level: 2*r6 - r2 - r10 (rate space, for signal only)
    fly_lvl = 2.0 * r_belly - r_front - r_back
    
    # CANONICAL RETURN CONSTRUCTION: Use price returns, not rate returns
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
    
    # Filter out NaN/inf values
    valid_mask = np.isfinite(fly_ret) & np.isfinite(fly_lvl)
    fly_lvl = fly_lvl[valid_mask]
    fly_ret = fly_ret[valid_mask]
    
    logger.info(f"[SR3CurveRVMomentumPhase1] Rank Fly: {len(fly_lvl)} days")
    logger.info(f"  Effective start: {fly_lvl.index[0]}")
    logger.info(f"  Effective end: {fly_lvl.index[-1]}")
    
    # Base signal: momentum (go with hump formation/collapse)
    base_signal = fly_lvl  # Positive = hump, negative = U-shaped
    
    # Z-score normalization
    rolling_mean = base_signal.rolling(
        window=zscore_window,
        min_periods=max(2, zscore_window // 2)
    ).mean()
    
    rolling_std = base_signal.rolling(
        window=zscore_window,
        min_periods=max(2, zscore_window // 2)
    ).std()
    
    # Compute z-scores
    z_scores = (base_signal - rolling_mean) / rolling_std
    z_scores = z_scores.fillna(0.0)
    
    # Clip to symmetric bounds
    normalized_signals = z_scores.clip(-clip, clip)
    
    logger.info(f"[SR3CurveRVMomentumPhase1] Normalized signal stats:")
    logger.info(f"  Mean: {normalized_signals.mean():.4f}")
    logger.info(f"  Std: {normalized_signals.std():.4f}")
    logger.info(f"  Min: {normalized_signals.min():.4f}")
    logger.info(f"  Max: {normalized_signals.max():.4f}")
    
    # Vol targeting
    rolling_vol = fly_ret.rolling(
        window=vol_lookback,
        min_periods=vol_lookback
    ).std() * np.sqrt(252)  # Annualize
    
    rolling_vol = rolling_vol.clip(lower=min_vol_floor)
    
    vol_scalar = target_vol / rolling_vol
    vol_scalar = vol_scalar.fillna(1.0)
    vol_scalar = vol_scalar.clip(0.0, max_leverage)
    
    # Apply vol targeting and lag
    positions = normalized_signals * vol_scalar
    positions = positions.shift(lag)
    
    # Compute portfolio returns
    portfolio_dates = positions.index.intersection(fly_ret.index)
    portfolio_returns = positions.loc[portfolio_dates] * fly_ret.loc[portfolio_dates]
    portfolio_returns = portfolio_returns.dropna()
    
    # Compute equity curve
    equity_curve = (1 + portfolio_returns).cumprod()
    
    metadata = {
        'sleeve': 'rank_fly_momentum',
        'phase': 'Phase-1',
        'zscore_window': zscore_window,
        'clip': clip,
        'target_vol': target_vol,
        'vol_lookback': vol_lookback,
        'min_vol_floor': min_vol_floor,
        'max_leverage': max_leverage,
        'lag': lag,
        'effective_start_date': str(fly_lvl.index[0].date()),
        'effective_end_date': str(fly_lvl.index[-1].date()),
        'n_days': len(fly_lvl)
    }
    
    return {
        'signals': normalized_signals,
        'positions': positions,
        'spread_returns': fly_ret,
        'portfolio_returns': portfolio_returns,
        'equity_curve': equity_curve,
        'metadata': metadata
    }

