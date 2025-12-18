"""
SR3 Calendar Carry Phase-1 Strategy

Engineered, tradable implementation of SR3 calendar carry.
Upgrades Phase-0 sign-only sanity check with:
- Z-scored signal normalization
- DV01-neutral (or equal-notional proxy) weighting
- Vol targeting
- Proper risk scaling

Canonical pair: R2-R1 (Rank 2 - Rank 1)
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)

# Canonical pair for Phase-1
CANONICAL_LONG_RANK = 2
CANONICAL_SHORT_RANK = 1
CANONICAL_PAIR = (CANONICAL_LONG_RANK, CANONICAL_SHORT_RANK)


def compute_sr3_calendar_carry_phase1(
    market,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    mode: str = "phase1",
    zscore_window: int = 90,
    clip: float = 2.0,
    target_vol: float = 0.10,
    vol_lookback: int = 60,
    use_dv01: bool = False,
    dv01_long: Optional[float] = None,
    dv01_short: Optional[float] = None,
    flip_sign: bool = False,
    lag: int = 1
) -> Dict[str, Any]:
    """
    Compute SR3 Calendar Carry Phase-1 signals and positions.
    
    Args:
        market: MarketData instance
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        mode: Strategy mode - "phase1" (default) or "phase0_equiv" (degenerate mode)
        zscore_window: Rolling window for z-score normalization (default: 90 days)
        clip: Symmetric clipping bounds for normalized signal (default: ±2.0)
        target_vol: Target annualized volatility (default: 0.10 = 10%)
        vol_lookback: Rolling window for realized vol calculation (default: 60 days)
        use_dv01: If True, use DV01-neutral weighting (requires dv01_long and dv01_short)
        dv01_long: DV01 for long leg (rank 2) - required if use_dv01=True
        dv01_short: DV01 for short leg (rank 1) - required if use_dv01=True
        flip_sign: If True, flip the signal sign (for sign convention testing)
        lag: Execution lag in days (default: 1, for carry typically correct)
        
    Returns:
        Dict with:
        - 'signals': pd.Series of normalized signals (index=date)
        - 'positions': pd.Series of vol-targeted positions (index=date)
        - 'spread_prices': pd.Series of spread prices (rank 2 - rank 1) in price space
        - 'spread_returns': pd.Series of spread returns (index=date)
        - 'portfolio_returns': pd.Series of portfolio returns (index=date)
        - 'equity_curve': pd.Series of cumulative equity (index=date)
        - 'metadata': Dict with strategy parameters and diagnostics
    """
    # Guardrail: Assert canonical pair
    long_rank = CANONICAL_LONG_RANK
    short_rank = CANONICAL_SHORT_RANK
    logger.info(f"[SR3CalendarCarryPhase1] Using canonical pair: R{long_rank}-R{short_rank}")
    
    # Load SR3 contract prices
    close = market.get_contracts_by_root(
        root="SR3",
        ranks=[long_rank, short_rank],
        fields=("close",),
        start=start_date,
        end=end_date
    )
    
    if close.empty:
        raise ValueError("No SR3 data returned for canonical pair")
    
    # Guardrail: Assert ranks are present
    if long_rank not in close.columns or short_rank not in close.columns:
        raise ValueError(
            f"Canonical ranks {long_rank} or {short_rank} not found. "
            f"Available ranks: {list(close.columns)}"
        )
    
    # Compute spread returns from leg returns (canonical fix)
    # Load prices for both legs
    p_long = close[long_rank]
    p_short = close[short_rank]
    
    # Compute leg returns
    r_long = p_long.pct_change(fill_method=None)
    r_short = p_short.pct_change(fill_method=None)
    
    # Define spread returns as leg-weighted portfolio return
    # Equal-notional default: w_long = 1.0, w_short = 1.0
    # For spread: long leg - short leg
    w_long = 1.0
    w_short = 1.0
    
    # Apply DV01-neutral weighting if requested
    if use_dv01:
        if dv01_long is None or dv01_short is None:
            raise ValueError("use_dv01=True requires dv01_long and dv01_short")
        # DV01-neutral: w_long / dv01_long = w_short / dv01_short
        # Normalize so weights are reasonable (e.g., sum of abs weights = 2.0 for equal-notional equivalent)
        dv01_ratio = dv01_long / dv01_short
        w_long = dv01_short / (dv01_long + dv01_short) * 2.0
        w_short = dv01_long / (dv01_long + dv01_short) * 2.0
        logger.info(f"[SR3CalendarCarryPhase1] DV01-neutral weights: w_long={w_long:.4f}, w_short={w_short:.4f}")
    
    # Compute spread returns: w_long * r_long - w_short * r_short
    spread_returns = w_long * r_long - w_short * r_short
    
    # Align and drop NA (strict)
    spread_returns = spread_returns.dropna()
    
    if len(spread_returns) < 2:
        raise ValueError("Insufficient data for spread returns calculation")
    
    # Guardrail: Assert spread returns variance > 0 and finite
    spread_returns_std = spread_returns.std()
    if not np.isfinite(spread_returns_std):
        raise RuntimeError(
            f"[DATA INTEGRITY] SR3 spread returns std is not finite "
            f"(std={spread_returns_std}). This indicates data quality issues."
        )
    
    if spread_returns_std <= 0:
        raise RuntimeError(
            f"[DATA INTEGRITY] SR3 spread returns have zero variance "
            f"(std={spread_returns_std:.6f}). This indicates rank mapping failure."
        )
    
    # Guardrail: Assert no Inf in spread returns
    inf_count = np.isinf(spread_returns).sum()
    if inf_count > 0:
        raise RuntimeError(
            f"[DATA INTEGRITY] SR3 spread returns contain {inf_count} Inf values. "
            f"This indicates data quality issues."
        )
    
    logger.info(f"[SR3CalendarCarryPhase1] Spread returns std: {spread_returns_std:.6f}")
    
    # Sanity check: log distribution statistics
    spread_desc = spread_returns.describe(percentiles=[0.01, 0.05, 0.95, 0.99])
    logger.info(f"[SR3CalendarCarryPhase1] Spread returns distribution:")
    logger.info(f"  Mean: {spread_desc['mean']:.6f}")
    logger.info(f"  Std: {spread_desc['std']:.6f}")
    logger.info(f"  1st percentile: {spread_desc['1%']:.6f}")
    logger.info(f"  5th percentile: {spread_desc['5%']:.6f}")
    logger.info(f"  95th percentile: {spread_desc['95%']:.6f}")
    logger.info(f"  99th percentile: {spread_desc['99%']:.6f}")
    
    # Warn if 99th percentile is huge (e.g., > 5-10% daily)
    if abs(spread_desc['99%']) > 0.10:  # 10% daily return threshold
        logger.warning(
            f"[SR3CalendarCarryPhase1] 99th percentile return is {spread_desc['99%']:.4f} "
            f"(>{0.10:.2f}). A SOFR calendar spread should not have equity-like daily returns. "
            f"Something may still be wrong."
        )
    
    # Compute spread prices for reference (not used for returns)
    spread_prices = p_long - p_short
    
    # Convert to rate space for signal: r_k = 100 - P_k
    rates = 100.0 - close
    r_long = rates[long_rank]
    r_short = rates[short_rank]
    
    # Base signal: spread level in rate space (r_long - r_short)
    # This preserves directional meaning: positive = upward sloping curve (positive carry)
    base_signal = r_long - r_short
    
    # Align dates (drop NaN)
    common_dates = base_signal.dropna().index.intersection(spread_returns.index)
    if len(common_dates) == 0:
        raise ValueError("No common dates between base signal and spread returns")
    
    base_signal_aligned = base_signal.loc[common_dates]
    spread_returns_aligned = spread_returns.loc[common_dates]
    
    logger.info(f"[SR3CalendarCarryPhase1] Effective start date: {common_dates.min()}")
    logger.info(f"[SR3CalendarCarryPhase1] Effective end date: {common_dates.max()}")
    logger.info(f"[SR3CalendarCarryPhase1] Total days: {len(common_dates)}")
    logger.info(f"[SR3CalendarCarryPhase1] Mode: {mode}")
    logger.info(f"[SR3CalendarCarryPhase1] Flip sign: {flip_sign}, Lag: {lag}")
    
    # Mode: phase0_equiv (degenerate mode - should match Phase-0)
    if mode == "phase0_equiv":
        # Signal = sign(spread_level) - same as Phase-0C for R2-R1
        normalized_signals = np.sign(base_signal_aligned)
        normalized_signals = normalized_signals.replace(0.0, 0.0)  # Keep zeros as zeros
        
        # Apply sign flip if requested
        if flip_sign:
            normalized_signals = -normalized_signals
            logger.info(f"[SR3CalendarCarryPhase1] Applied sign flip")
        
        # Positions = signal.shift(lag) - no z-score, no clip, no vol targeting
        positions = normalized_signals.shift(lag)
        
        logger.info(f"[SR3CalendarCarryPhase1] Phase-0 equivalent mode:")
        logger.info(f"  Signal stats: mean={normalized_signals.mean():.4f}, std={normalized_signals.std():.4f}")
        logger.info(f"  Signal min: {normalized_signals.min():.4f}, max: {normalized_signals.max():.4f}")
        logger.info(f"  Position stats: mean={positions.mean():.4f}, std={positions.std():.4f}")
    
    # Mode: phase1 (default - with z-score, clip, vol targeting)
    else:
        # Normalize signal with rolling z-score
        # z = (x - mean(window)) / std(window)
        # Use pandas rolling for efficiency
        rolling_mean = base_signal_aligned.rolling(
            window=zscore_window,
            min_periods=min(2, zscore_window)  # Need at least 2 for std
        ).mean()
        
        rolling_std = base_signal_aligned.rolling(
            window=zscore_window,
            min_periods=min(2, zscore_window)
        ).std()
        
        # Compute z-scores
        normalized_signals = (base_signal_aligned - rolling_mean) / rolling_std
        
        # Fill NaN values (early dates with insufficient data) with 0
        normalized_signals = normalized_signals.fillna(0.0)
        
        # Clip to symmetric bounds
        normalized_signals = normalized_signals.clip(-clip, clip)
        
        # Apply sign flip if requested
        if flip_sign:
            normalized_signals = -normalized_signals
            logger.info(f"[SR3CalendarCarryPhase1] Applied sign flip")
        
        logger.info(f"[SR3CalendarCarryPhase1] Normalized signal stats:")
        logger.info(f"  Mean: {normalized_signals.mean():.4f}")
        logger.info(f"  Std: {normalized_signals.std():.4f}")
        logger.info(f"  Min: {normalized_signals.min():.4f}")
        logger.info(f"  Max: {normalized_signals.max():.4f}")
        
        # Vol targeting: scale positions to target volatility
        # Compute rolling realized vol of spread returns
        rolling_vol = spread_returns_aligned.rolling(
            window=vol_lookback,
            min_periods=vol_lookback
        ).std() * np.sqrt(252)  # Annualize
        
        # Apply minimum vol floor to prevent division by very small numbers
        # This prevents extreme leverage when spread vol is very low
        min_vol_floor = 0.01  # 1% minimum annualized vol
        rolling_vol = rolling_vol.clip(lower=min_vol_floor)
        
        # Compute vol scalar: target_vol / realized_vol
        # This scales positions so that: position_vol * spread_vol ≈ target_vol
        # Since portfolio_return = position * spread_return, we want:
        # std(portfolio_return) = std(position * spread_return) ≈ target_vol
        # If positions are constant, then: position * std(spread_return) ≈ target_vol
        # So: position ≈ target_vol / std(spread_return)
        vol_scalar = target_vol / rolling_vol
        vol_scalar = vol_scalar.fillna(1.0)  # Use 1.0 if vol not available yet
        
        # Cap vol scalar to prevent extreme leverage (e.g., max 10x for 10% target / 1% floor)
        max_leverage = 10.0
        vol_scalar = vol_scalar.clip(0.0, max_leverage)
        
        # Apply vol targeting: position = normalized_signal * vol_scalar
        # Then apply lag
        positions = normalized_signals * vol_scalar
        positions = positions.shift(lag)
    
    # Apply DV01-neutral weighting if requested
    if use_dv01:
        if dv01_long is None or dv01_short is None:
            raise ValueError("use_dv01=True requires dv01_long and dv01_short")
        
        # DV01-neutral: position_long / dv01_long = position_short / dv01_short
        # For spread: position_long = position, position_short = -position
        # So: position / dv01_long = -position / dv01_short
        # This simplifies to: position needs no adjustment (spread is already DV01-neutral if legs are equal)
        # Actually, for a spread, we need to adjust the position size based on DV01 ratio
        # If DV01s are different, we scale the position
        dv01_ratio = dv01_long / dv01_short
        # For equal DV01, ratio = 1.0, no adjustment
        # For different DV01, we adjust position to maintain DV01 neutrality
        positions = positions * (2.0 / (1.0 + dv01_ratio))  # Normalize to average DV01
        logger.info(f"[SR3CalendarCarryPhase1] Applied DV01-neutral weighting: ratio={dv01_ratio:.4f}")
    else:
        # Equal-notional proxy (default)
        logger.info(f"[SR3CalendarCarryPhase1] Using equal-notional proxy (DV01-neutral not applied)")
    
    # Compute portfolio returns: position * spread_return
    # Note: lag is already applied to positions above (in phase0_equiv or phase1 mode)
    # So we don't apply an additional shift here
    
    # Align positions with spread_returns_aligned
    portfolio_dates = positions.index.intersection(spread_returns_aligned.index)
    portfolio_returns = positions.loc[portfolio_dates] * spread_returns_aligned.loc[portfolio_dates]
    portfolio_returns = portfolio_returns.dropna()
    
    # Compute equity curve
    # Align equity curve dates with portfolio returns
    equity_dates = portfolio_returns.index
    equity_curve = (1 + portfolio_returns).cumprod()
    equity_curve.index = equity_dates
    
    # Prepare metadata
    metadata = {
        'canonical_pair': f"R{long_rank}-R{short_rank}",
        'phase': 'Phase-1',
        'long_rank': long_rank,
        'short_rank': short_rank,
        'zscore_window': zscore_window,
        'clip': clip,
        'target_vol': target_vol,
        'vol_lookback': vol_lookback,
        'dv01_method': 'true' if use_dv01 else 'proxy',
        'dv01_long': dv01_long if use_dv01 else None,
        'dv01_short': dv01_short if use_dv01 else None,
        'signal_description': (
            f"sign(R{long_rank} - R{short_rank}) in rate space" if mode == "phase0_equiv"
            else f"z-scored(R{long_rank} - R{short_rank}) in rate space, clipped to ±{clip}, spread returns from leg returns"
        ),
        'normalization_method': (
            "sign-only (phase0_equiv mode)" if mode == "phase0_equiv"
            else f"rolling z-score ({zscore_window}d window)"
        ),
        'risk_target': (
            "none (phase0_equiv mode)" if mode == "phase0_equiv"
            else f"{target_vol*100:.1f}% annualized volatility"
        ),
        'mode': mode,
        'flip_sign': flip_sign,
        'lag': lag,
        'effective_start_date': str(common_dates.min().date()),
        'effective_end_date': str(common_dates.max().date()),
        'n_days': len(common_dates),
        'rank_mapping_fix_applied': True
    }
    
    return {
        'signals': normalized_signals,
        'positions': positions,
        'spread_prices': spread_prices.loc[common_dates],
        'spread_returns': spread_returns_aligned,
        'portfolio_returns': portfolio_returns,
        'equity_curve': equity_curve,
        'metadata': metadata
    }

