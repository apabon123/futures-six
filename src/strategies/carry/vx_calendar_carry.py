"""
VX Calendar Carry Phase-1 Strategy

Engineered, tradable implementation of VX calendar carry.
Upgrades Phase-0 sign-only sanity check with:
- Z-scored signal normalization
- Vol targeting
- Proper risk scaling

Phase-1 variants:
- VX2-VX1_short (Front Carry)
- VX3-VX2_short (Mid Carry)

Both represent the same economic idea (volatility term carry) with different curve locations.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, Tuple
import logging
import duckdb

from src.market_data.vrp_loaders import load_vx_curve
from src.agents.utils_db import open_readonly_connection

logger = logging.getLogger(__name__)

# Variant definitions
VARIANT_VX2_VX1_SHORT = "vx2_vx1_short"
VARIANT_VX3_VX2_SHORT = "vx3_vx2_short"

# Variant to spread pair mapping
VARIANT_TO_PAIR = {
    VARIANT_VX2_VX1_SHORT: (2, 1),  # VX2 - VX1
    VARIANT_VX3_VX2_SHORT: (3, 2),  # VX3 - VX2
}


def compute_vx_calendar_carry_phase1(
    db_path: str,
    variant: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    mode: str = "phase1",
    zscore_window: int = 90,
    clip: float = 2.0,
    target_vol: float = 0.10,
    vol_lookback: int = 60,
    min_vol_floor: float = 0.01,
    max_leverage: float = 10.0,
    lag: int = 1,
    max_collision_pct: float = 5.0
) -> Dict[str, Any]:
    """
    Compute VX Calendar Carry Phase-1 signals and positions.
    
    Args:
        db_path: Path to canonical database
        variant: Strategy variant - "vx2_vx1_short" or "vx3_vx2_short"
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        mode: Strategy mode - "phase1" (default) or "phase0_equiv" (degenerate mode)
        zscore_window: Rolling window for z-score normalization (default: 90 days)
        clip: Symmetric clipping bounds for normalized signal (default: ±2.0)
        target_vol: Target annualized volatility (default: 0.10 = 10%)
        vol_lookback: Rolling window for realized vol calculation (default: 60 days)
        min_vol_floor: Minimum vol floor for vol targeting (default: 0.01 = 1%)
        max_leverage: Maximum leverage cap (default: 10.0)
        lag: Execution lag in days (default: 1, for carry typically correct)
        max_collision_pct: Maximum allowed % of days where ranks collide (default: 5.0%)
        
    Returns:
        Dict with:
        - 'signals': pd.Series of normalized signals (index=date)
        - 'positions': pd.Series of vol-targeted positions (index=date)
        - 'spread_prices': pd.Series of spread prices (VX_long - VX_short) in price space
        - 'spread_returns': pd.Series of spread returns (index=date)
        - 'portfolio_returns': pd.Series of portfolio returns (index=date)
        - 'equity_curve': pd.Series of cumulative equity (index=date)
        - 'metadata': Dict with strategy parameters and diagnostics
    """
    # Validate variant
    if variant not in VARIANT_TO_PAIR:
        raise ValueError(f"Invalid variant: {variant}. Must be one of {list(VARIANT_TO_PAIR.keys())}")
    
    # Get spread pair
    long_leg, short_leg = VARIANT_TO_PAIR[variant]
    logger.info(f"[VXCalendarCarryPhase1] Using variant: {variant} (VX{long_leg}-VX{short_leg})")
    
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
        
        # Check required columns
        max_leg = max(long_leg, short_leg)
        required_cols = [leg_to_col[i] for i in range(1, max_leg + 1)]
        missing_cols = [col for col in required_cols if col not in vx_data.columns]
        if missing_cols:
            raise ValueError(f"Missing required VX columns: {missing_cols}. Available: {list(vx_data.columns)}")
        
        logger.info(f"Loaded VX data: {len(vx_data)} days")
        logger.info(f"  Date range: {vx_data.index[0]} to {vx_data.index[-1]}")
        
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
        
        # CANONICAL RETURN CONSTRUCTION (Non-Negotiable)
        # DO NOT compute returns as % change of the spread level
        # Correct formula: r_spread = (+1)*r_VX_long - (1)*r_VX_short
        # where r_k = (P_k,t - P_k,t-1) / P_k,t-1
        
        # Compute leg returns (simple percentage returns)
        r_long = long_prices.pct_change(fill_method=None)
        r_short = short_prices.pct_change(fill_method=None)
        
        # Spread return = r_VX_long - r_VX_short
        # This represents P&L of being long VX_long and short VX_short (1:1 notional)
        spread_returns = r_long - r_short
        
        # Align and drop NA (strict)
        spread_returns = spread_returns.dropna()
        
        if len(spread_returns) < 2:
            raise ValueError("Insufficient data for spread returns calculation")
        
        # Guardrail: Assert spread returns variance > 0 and finite
        spread_returns_std = spread_returns.std()
        if not np.isfinite(spread_returns_std):
            raise RuntimeError(
                f"[DATA INTEGRITY] VX spread returns std is not finite "
                f"(std={spread_returns_std}). This indicates data quality issues."
            )
        
        if spread_returns_std <= 0:
            raise RuntimeError(
                f"[DATA INTEGRITY] VX spread returns have zero variance "
                f"(std={spread_returns_std:.6f}). This indicates rank mapping failure."
            )
        
        # Guardrail: Assert no Inf in spread returns
        inf_count = np.isinf(spread_returns).sum()
        if inf_count > 0:
            raise RuntimeError(
                f"[DATA INTEGRITY] VX spread returns contain {inf_count} Inf values. "
                f"This indicates data quality issues."
            )
        
        logger.info(f"[VXCalendarCarryPhase1] Spread returns std: {spread_returns_std:.6f}")
        
        # Sanity check: log distribution statistics
        spread_desc = spread_returns.describe(percentiles=[0.01, 0.05, 0.95, 0.99])
        logger.info(f"[VXCalendarCarryPhase1] Spread returns distribution:")
        logger.info(f"  Mean: {spread_desc['mean']:.6f}")
        logger.info(f"  Std: {spread_desc['std']:.6f}")
        logger.info(f"  1st percentile: {spread_desc['1%']:.6f}")
        logger.info(f"  5th percentile: {spread_desc['5%']:.6f}")
        logger.info(f"  95th percentile: {spread_desc['95%']:.6f}")
        logger.info(f"  99th percentile: {spread_desc['99%']:.6f}")
        
        # Compute spread prices for signal (in price space, not rate space)
        spread_prices = long_prices - short_prices
        
        # Base signal: spread level in price space
        # Positive = contango (long > short)
        # Negative = backwardation
        base_signal = spread_prices
        
        # Align dates (drop NaN)
        common_dates = base_signal.dropna().index.intersection(spread_returns.index)
        if len(common_dates) == 0:
            raise ValueError("No common dates between base signal and spread returns")
        
        # Filter by date range if specified
        if start_date:
            start_dt = pd.to_datetime(start_date)
            common_dates = common_dates[common_dates >= start_dt]
        if end_date:
            end_dt = pd.to_datetime(end_date)
            common_dates = common_dates[common_dates <= end_dt]
        
        if len(common_dates) == 0:
            raise ValueError("No common dates after date filtering")
        
        base_signal_aligned = base_signal.loc[common_dates]
        spread_returns_aligned = spread_returns.loc[common_dates]
        
        logger.info(f"[VXCalendarCarryPhase1] Effective start date: {common_dates.min()}")
        logger.info(f"[VXCalendarCarryPhase1] Effective end date: {common_dates.max()}")
        logger.info(f"[VXCalendarCarryPhase1] Total days: {len(common_dates)}")
        logger.info(f"[VXCalendarCarryPhase1] Mode: {mode}")
        logger.info(f"[VXCalendarCarryPhase1] Lag: {lag}")
        logger.info(f"[VXCalendarCarryPhase1] Collision days dropped: {collision_days_dropped}")
        
        # Mode: phase0_equiv (degenerate mode - should match Phase-0)
        if mode == "phase0_equiv":
            # Signal = -sign(spread_level) for carry capture (same as Phase-0)
            normalized_signals = -np.sign(base_signal_aligned)
            normalized_signals = normalized_signals.replace(0.0, 0.0)  # Keep zeros as zeros
            
            logger.info(f"[VXCalendarCarryPhase1] Phase-0 equivalent mode:")
            logger.info(f"  Signal stats: mean={normalized_signals.mean():.4f}, std={normalized_signals.std():.4f}")
            logger.info(f"  Signal min: {normalized_signals.min():.4f}, max: {normalized_signals.max():.4f}")
            
            # Positions = signal.shift(lag) - no z-score, no clip, no vol targeting
            positions = normalized_signals.shift(lag)
            
            logger.info(f"  Position stats: mean={positions.mean():.4f}, std={positions.std():.4f}")
        
        # Mode: phase1 (default - with z-score, clip, vol targeting)
        else:
            # Normalize signal with rolling z-score
            # z = (x - mean(window)) / std(window)
            rolling_mean = base_signal_aligned.rolling(
                window=zscore_window,
                min_periods=min(2, zscore_window)  # Need at least 2 for std
            ).mean()
            
            rolling_std = base_signal_aligned.rolling(
                window=zscore_window,
                min_periods=min(2, zscore_window)
            ).std()
            
            # Compute z-scores
            z_scores = (base_signal_aligned - rolling_mean) / rolling_std
            
            # Fill NaN values (early dates with insufficient data) with 0
            z_scores = z_scores.fillna(0.0)
            
            # Clip to symmetric bounds
            z_clipped = z_scores.clip(-clip, clip)
            
            # Direction (carry capture) — frozen for VX
            # Carry capture is short spread in contango, so: raw_signal = -z_clipped
            raw_signal = -z_clipped
            
            # Map to [-1, +1] (since clip is ±2, divide by 2)
            normalized_signals = raw_signal / 2.0
            
            logger.info(f"[VXCalendarCarryPhase1] Normalized signal stats:")
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
            rolling_vol = rolling_vol.clip(lower=min_vol_floor)
            
            # Compute vol scalar: target_vol / realized_vol
            vol_scalar = target_vol / rolling_vol
            vol_scalar = vol_scalar.fillna(1.0)  # Use 1.0 if vol not available yet
            
            # Cap vol scalar to prevent extreme leverage
            vol_scalar = vol_scalar.clip(0.0, max_leverage)
            
            # Apply vol targeting: position = normalized_signal * vol_scalar
            # Then apply lag
            positions = normalized_signals * vol_scalar
            positions = positions.shift(lag)
        
        # Compute portfolio returns: position * spread_return
        # Note: lag is already applied to positions above
        # Align positions with spread_returns_aligned
        portfolio_dates = positions.index.intersection(spread_returns_aligned.index)
        portfolio_returns = positions.loc[portfolio_dates] * spread_returns_aligned.loc[portfolio_dates]
        portfolio_returns = portfolio_returns.dropna()
        
        # Compute equity curve
        equity_dates = portfolio_returns.index
        equity_curve = (1 + portfolio_returns).cumprod()
        equity_curve.index = equity_dates
        
        # Prepare metadata
        metadata = {
            'variant': variant,
            'spread_pair': f"VX{long_leg}-VX{short_leg}",
            'long_leg': long_leg,
            'short_leg': short_leg,
            'phase': 'Phase-1',
            'zscore_window': zscore_window,
            'clip': clip,
            'target_vol': target_vol,
            'vol_lookback': vol_lookback,
            'min_vol_floor': min_vol_floor,
            'max_leverage': max_leverage,
            'signal_description': (
                f"-sign(VX{long_leg} - VX{short_leg})" if mode == "phase0_equiv"
                else f"-z-scored(VX{long_leg} - VX{short_leg}) / 2.0, clipped to ±{clip}, spread returns from leg returns"
            ),
            'normalization_method': (
                "sign-only (phase0_equiv mode)" if mode == "phase0_equiv"
                else f"rolling z-score ({zscore_window}d window), then -z/2 for carry capture"
            ),
            'risk_target': (
                "none (phase0_equiv mode)" if mode == "phase0_equiv"
                else f"{target_vol*100:.1f}% annualized volatility"
            ),
            'mode': mode,
            'lag': lag,
            'effective_start_date': str(common_dates.min().date()),
            'effective_end_date': str(common_dates.max().date()),
            'n_days': len(common_dates),
            'collision_days_dropped': collision_days_dropped,
            'collision_rate_pct': pct_identical
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
    
    finally:
        con.close()

