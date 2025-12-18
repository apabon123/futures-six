"""
SR3 Calendar Carry Sign-Only Sanity Check

A deliberately simple, academic-style carry trading strategy to verify that
the SR3 calendar carry idea and P&L machinery are working correctly.

Strategy:
- Use ranks 1-4 only (avoid rank 0 noise and back-month NA gaps)
- Compute carry signal: sign(RANK_2 - RANK_1) in rate space
  - r_k = 100 - P_k (convert prices to rates)
  - carry_raw = r2 - r1
  - signal = sign(carry_raw) → +1 (positive carry, long), -1 (negative carry, short), 0 (flat)
- Trade SR3_FRONT_CALENDAR (rank 0) based on this signal
- Daily strategy return = signal * daily_return
- No vol targeting, no normalization beyond sign

This is a diagnostic tool to answer: "Does a very simple SR3 calendar carry strategy
show reasonable positive Sharpe, or is the alpha gone because our data / roll / P&L pipeline is wrong?"

If sign-only carry shows Sharpe ~0.2-0.5, then data & P&L pipeline are probably fine.
If it shows negative or near-zero Sharpe, we should question:
- The carry calculation (rate space conversion)
- The data quality (ranks 1-4 availability)
- The hypothesis that SR3 calendar carry is still a tradable edge
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)

# Canonical ranks for Phase-0: 1-4 only
# All start at 2020-01-02, full coverage, liquid, front pack where carry lives
CANONICAL_RANKS = [1, 2, 3, 4]

# Tradeable contract: SR3_FRONT_CALENDAR (rank 0 in database)
TRADEABLE_SYMBOL = "SR3_FRONT_CALENDAR"

# Phase-0 variants
VARIANT_OPTION_A = "option_a"  # sign(RANK_2 - RANK_1)
VARIANT_OPTION_B = "option_b"  # sign(mean(RANK_3, RANK_4) - mean(RANK_1, RANK_2))
VARIANT_SPREAD = "spread"  # Trade spread directly: P&L of (RANK_2 - RANK_1) with sign-only positioning

# Adjacent rank pairs for Phase-0 variant sweep
# Format: (long_rank, short_rank) for spread = long - short
ADJACENT_PAIRS = [
    (1, 0),  # Rank1 - Rank0
    (2, 1),  # Rank2 - Rank1
    (3, 2),  # Rank3 - Rank2
    (4, 3),  # Rank4 - Rank3
    (5, 4),  # Rank5 - Rank4 (optional, check data availability)
]


def run_sign_only_sr3_carry(
    market,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    variant: str = VARIANT_OPTION_A,
    rank_pair: Optional[tuple] = None,
    zero_threshold: float = 1e-8
) -> Dict[str, Any]:
    """
    Run sign-only SR3 calendar carry strategy.
    
    Args:
        market: MarketData instance
        start_date: Start date for backtest (YYYY-MM-DD)
        end_date: End date for backtest (YYYY-MM-DD)
        zero_threshold: Threshold for treating carry as zero
        
    Returns:
        Dict with:
        - 'portfolio_returns': pd.Series of daily portfolio returns
        - 'equity_curve': pd.Series of cumulative equity
        - 'asset_returns': pd.DataFrame of daily asset returns (SR3 only)
        - 'asset_strategy_returns': pd.DataFrame of per-asset strategy returns
        - 'positions': pd.DataFrame of daily positions (signs)
        - 'carry_signals': pd.Series of raw carry signals (r2 - r1)
        - 'metrics': Dict with {cagr, vol, sharpe, maxdd, hit_rate, n_days, years}
    """
    results = compute_sign_only_sr3_carry(
        market=market,
        start_date=start_date,
        end_date=end_date,
        variant=variant,
        rank_pair=rank_pair,
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


def compute_sign_only_sr3_carry(
    market,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    variant: str = VARIANT_OPTION_A,
    rank_pair: Optional[tuple] = None,
    zero_threshold: float = 1e-8
) -> Dict:
    """
    Compute sign-only SR3 calendar carry strategy returns.
    
    Strategy:
    - Load SR3 ranks 1-4 (canonical Phase-0 ranks)
    - Convert to rate space: r_k = 100 - P_k
    - Compute carry: carry_raw = r2 - r1
    - Signal: sign(carry_raw)
    - Trade SR3_FRONT_CALENDAR based on signal
    
    Args:
        market: MarketData instance
        start_date: Start date for backtest (YYYY-MM-DD)
        end_date: End date for backtest (YYYY-MM-DD)
        zero_threshold: Threshold for treating carry as zero
        
    Returns:
        Dict with returns, positions, and equity curves
    """
    # Determine which ranks to load based on variant and rank_pair
    if rank_pair is not None:
        # For adjacent pair variants, load the specific ranks needed
        long_rank, short_rank = rank_pair
        # Include rank 0 if needed (for R1-R0 pair)
        if short_rank == 0:
            required_ranks = sorted(set([long_rank] + CANONICAL_RANKS[:4]))
        else:
            required_ranks = sorted(set([long_rank, short_rank] + CANONICAL_RANKS[:4]))  # Include canonical ranks for safety
        pair_label = f"R{long_rank}-R{short_rank}"
        logger.info(f"Loading SR3 ranks for adjacent pair {pair_label}: {required_ranks}")
    else:
        required_ranks = CANONICAL_RANKS
        pair_label = None
        logger.info(f"Loading SR3 ranks {CANONICAL_RANKS} for carry signal...")
    
    # Load SR3 ranks
    # get_contracts_by_root now uses canonical rank parsing for SR3,
    # so we can directly request the ranks we need
    try:
        close = market.get_contracts_by_root(
            root="SR3",
            ranks=required_ranks,
            fields=("close",),
            start=None,  # Get all available data
            end=end_date
        )
        
        if close.empty:
            raise ValueError(f"No SR3 contract data found for ranks {required_ranks}")
        
        # For spread variants with rank_pair, ensure we have the required ranks
        if rank_pair is not None:
            long_rank, short_rank = rank_pair
            # For R1-R0, rank 0 is not in close.columns (it's SR3_FRONT_CALENDAR in prices_cont)
            if short_rank == 0:
                # Only check that long_rank (rank 1) is available
                if long_rank not in close.columns:
                    raise ValueError(
                        f"Missing required rank {long_rank} for pair {pair_label}. "
                        f"Available: {list(close.columns)}"
                    )
            else:
                # For other pairs, both ranks should be in close.columns
                if long_rank not in close.columns or short_rank not in close.columns:
                    raise ValueError(
                        f"Missing required ranks for pair {pair_label}. "
                        f"Required: [{long_rank}, {short_rank}], Available: {list(close.columns)}"
                    )
        else:
            # Ensure we have ranks 1 and 2 (required for default carry signal)
            if 1 not in close.columns or 2 not in close.columns:
                raise ValueError(
                    f"Missing required ranks 1 or 2. Available: {list(close.columns)}"
                )
        
        logger.info(f"Loaded SR3 data: {len(close)} days, ranks: {sorted(close.columns)}")
        
    except Exception as e:
        logger.error(f"Error loading SR3 contracts: {e}")
        raise
    
    # Handle missing data: forward-fill and backward-fill
    close_filled = close.ffill().bfill()
    
    # Convert prices to rates: r_k = 100 - P_k
    rates = 100.0 - close_filled
    
    # Compute carry signal based on variant and rank_pair
    if rank_pair is not None:
        # Adjacent pair variant: use specified ranks
        long_rank, short_rank = rank_pair
        # For R1-R0, rank 0 is not in rates.columns (it's SR3_FRONT_CALENDAR in prices_cont)
        if short_rank == 0:
            # Only check long_rank (rank 1)
            if long_rank not in rates.columns:
                raise ValueError(
                    f"Rank {long_rank} not available for pair R{long_rank}-R0. "
                    f"Available ranks: {list(rates.columns)}"
                )
            # For R1-R0, get rank 0 from get_contracts_by_root (now uses canonical parsing)
            r_long = rates[long_rank]
            close_rank0 = market.get_contracts_by_root(
                root="SR3",
                ranks=[0],  # Explicitly request rank 0
                fields=("close",),
                start=None,
                end=None
            )
            if 0 not in close_rank0.columns:
                raise ValueError(
                    f"Rank 0 not found for pair R{long_rank}-R0. "
                    f"Available ranks: {list(close_rank0.columns)}"
                )
            p0 = close_rank0[0]
            
            # Debug: Check what contract rank 1 and rank 0 actually represent
            # Get the actual symbol name for rank 1 from the contract data
            close_debug = market.get_contracts_by_root(
                root="SR3",
                ranks=[long_rank],
                fields=("close",),
                start=None,
                end=None
            )
            if long_rank in close_debug.columns:
                # Check a sample of prices to see if they match
                common_debug = r_long.index.intersection(p0.index)
                if len(common_debug) > 0:
                    r1_sample = r_long.loc[common_debug[:5]] if len(common_debug) >= 5 else r_long.loc[common_debug]
                    p0_sample = p0.loc[common_debug[:5]] if len(common_debug) >= 5 else p0.loc[common_debug]
                    r0_sample = 100.0 - p0_sample
                    p1_sample = 100.0 - r1_sample
                    price_diff = p1_sample - p0_sample
                    logger.info(f"R1-R0 Debug: r1 sample (from rank {long_rank}): {r1_sample.values}, "
                              f"r0 sample (from rank 0): {r0_sample.values}, "
                              f"p1 prices: {p1_sample.values}, p0 prices: {p0_sample.values}, "
                              f"price diff (p1-p0): {price_diff.values}")
            
            # Align dates - use full intersection for signal calculation
            common_dates_signal = r_long.index.intersection(p0.index)
            if len(common_dates_signal) == 0:
                raise ValueError(f"No common dates between rank {long_rank} and rank 0 for pair R{long_rank}-R0")
            r_long_aligned = r_long.loc[common_dates_signal]
            p0_aligned = p0.loc[common_dates_signal]
            # Convert rank 0 price to rate
            r0_aligned = 100.0 - p0_aligned
            # Compute carry signal: r1 - r0
            # Store on common_dates_signal index (will be aligned with spread_returns later)
            carry_raw = r_long_aligned - r0_aligned
            # Debug: check if carry_raw is all zeros
            if len(carry_raw) > 0 and carry_raw.abs().max() < 1e-10:
                logger.warning(f"Carry signal for R{long_rank}-R0 is essentially zero. "
                             f"r1 sample: {r_long_aligned.iloc[:5].values if len(r_long_aligned) > 5 else r_long_aligned.values}, "
                             f"r0 sample: {r0_aligned.iloc[:5].values if len(r0_aligned) > 5 else r0_aligned.values}")
            signal_description = f"sign(RANK_{long_rank} - RANK_0) -> trade spread (RANK_{long_rank} - RANK_0) directly"
            pair_label = f"R{long_rank}-R{short_rank}"
        else:
            # For other pairs, both ranks should be in rates.columns
            if long_rank not in rates.columns or short_rank not in rates.columns:
                raise ValueError(
                    f"Ranks {long_rank} or {short_rank} not available. "
                    f"Available ranks: {list(rates.columns)}"
                )
            r_long = rates[long_rank]
            r_short = rates[short_rank]
            carry_raw = r_long - r_short
            signal_description = f"sign(RANK_{long_rank} - RANK_{short_rank}) -> trade spread (RANK_{long_rank} - RANK_{short_rank}) directly"
            pair_label = f"R{long_rank}-R{short_rank}"
    elif variant == VARIANT_OPTION_A:
        # Option A: sign(RANK_2 - RANK_1)
        r1 = rates[1]
        r2 = rates[2]
        carry_raw = r2 - r1
        signal_description = "sign(RANK_2 - RANK_1)"
        pair_label = None
    elif variant == VARIANT_OPTION_B:
        # Option B: sign(mean(RANK_3, RANK_4) - mean(RANK_1, RANK_2))
        r1 = rates[1]
        r2 = rates[2]
        r3 = rates[3]
        r4 = rates[4]
        mean_front = (r1 + r2) / 2.0
        mean_back = (r3 + r4) / 2.0
        carry_raw = mean_back - mean_front
        signal_description = "sign(mean(RANK_3, RANK_4) - mean(RANK_1, RANK_2))"
        pair_label = None
    elif variant == VARIANT_SPREAD:
        # Phase-0C: Trade spread directly - signal is still sign(RANK_2 - RANK_1)
        # but P&L will be computed from spread returns, not rank 0 returns
        r1 = rates[1]
        r2 = rates[2]
        carry_raw = r2 - r1
        signal_description = "sign(RANK_2 - RANK_1) -> trade spread (RANK_2 - RANK_1) directly"
        pair_label = "R2-R1"
    else:
        raise ValueError(f"Unknown variant: {variant}. Use '{VARIANT_OPTION_A}', '{VARIANT_OPTION_B}', or '{VARIANT_SPREAD}'")
    
    # Positive = upward sloping curve (positive carry)
    # Negative = downward sloping curve (negative carry)
    
    # Store signal description for later use
    signal_desc = signal_description
    
    logger.info(f"Computed carry signals ({variant}): {len(carry_raw)} days")
    logger.info(f"  Signal: {signal_desc}")
    logger.info(f"  Carry mean: {carry_raw.mean():.4f} bps")
    logger.info(f"  Carry std: {carry_raw.std():.4f} bps")
    logger.info(f"  Carry min: {carry_raw.min():.4f} bps")
    logger.info(f"  Carry max: {carry_raw.max():.4f} bps")
    
    # For spread variant or rank_pair, compute spread returns directly
    # For other variants, use rank 0 returns
    if variant == VARIANT_SPREAD or rank_pair is not None:
        # Trade the calendar spread directly
        if rank_pair is not None:
            long_rank, short_rank = rank_pair
            
            # Handle R1-R0 case: rank 0 from get_contracts_by_root (now uses canonical parsing)
            if short_rank == 0:
                p_long = close_filled[long_rank]
                # Get rank 0 from get_contracts_by_root
                close_rank0 = market.get_contracts_by_root(
                    root="SR3",
                    ranks=[0],  # Explicitly request rank 0
                    fields=("close",),
                    start=None,
                    end=None
                )
                if 0 not in close_rank0.columns:
                    raise ValueError(
                        f"Rank 0 not found for pair {pair_label}. "
                        f"Available ranks: {list(close_rank0.columns)}"
                    )
                p_short = close_rank0[0]
                # Align p_long and p_short before computing returns
                common_prices = p_long.index.intersection(p_short.index)
                if len(common_prices) < 100:
                    logger.warning(f"Limited overlap between rank {long_rank} and rank 0: {len(common_prices)} days")
                p_long = p_long.loc[common_prices]
                p_short = p_short.loc[common_prices]
                # Also align carry_raw to the same dates
                if len(common_prices) > 0:
                    carry_raw = carry_raw.loc[carry_raw.index.intersection(common_prices)]
                logger.info(f"Phase-0 Adjacent Pair: Using spread returns (RANK_{long_rank} - RANK_0) directly")
                logger.info(f"  Price overlap: {len(common_prices)} days")
            else:
                p_long = close_filled[long_rank]
                p_short = close_filled[short_rank]
                logger.info(f"Phase-0 Adjacent Pair: Using spread returns (RANK_{long_rank} - RANK_{short_rank}) directly")
        else:
            # Default to R2-R1 for VARIANT_SPREAD
            long_rank, short_rank = 2, 1
            p_long = close_filled[2]
            p_short = close_filled[1]
            logger.info(f"Phase-0C: Using spread returns (RANK_2 - RANK_1) directly")
        
        # Compute daily returns for each rank (simple returns)
        r_long_daily = p_long.pct_change(fill_method=None)
        r_short_daily = p_short.pct_change(fill_method=None)
        
        # Spread return = R_long return - R_short return
        # This represents P&L of being long R_long and short R_short (1:1 notional)
        spread_returns = r_long_daily - r_short_daily
        
        # Align on common dates
        common_dates = carry_raw.index.intersection(spread_returns.index)
        carry_raw_aligned = carry_raw.loc[common_dates]
        spread_returns_aligned = spread_returns.loc[common_dates]
        
        # Drop NaN from spread returns (first day will be NaN)
        valid_mask = spread_returns_aligned.notna()
        common_dates = common_dates[valid_mask]
        carry_raw_aligned = carry_raw_aligned.loc[common_dates]
        spread_returns_aligned = spread_returns_aligned.loc[common_dates]
        
        # Use spread returns as the "daily returns" for strategy
        daily_returns = spread_returns_aligned
        carry_raw = carry_raw_aligned
        
        logger.info(f"Aligned data: {len(common_dates)} days")
    else:
        # Load tradeable contract (SR3_FRONT_CALENDAR) for returns
        # This is rank 0 in the database, but we use it for trading
        prices_cont = market.prices_cont
        if TRADEABLE_SYMBOL not in prices_cont.columns:
            raise ValueError(
                f"Tradeable symbol {TRADEABLE_SYMBOL} not found in prices_cont. "
                f"Available symbols: {list(prices_cont.columns)[:10]}..."
            )
        
        tradeable_prices = prices_cont[TRADEABLE_SYMBOL]
        
        # Align on common dates
        common_dates = carry_raw.index.intersection(tradeable_prices.index)
        carry_raw = carry_raw.loc[common_dates]
        tradeable_prices = tradeable_prices.loc[common_dates]
        
        if len(common_dates) == 0:
            raise ValueError("No common dates between carry signals and tradeable prices")
        
        logger.info(f"Aligned data: {len(common_dates)} days")
        
        # Compute daily returns (simple returns)
        daily_returns = tradeable_prices.pct_change(fill_method=None).dropna()
    
    if daily_returns.empty:
        raise ValueError("No valid daily returns computed")
    
    logger.info(f"Daily returns computed: {len(daily_returns)} days")
    
    # Shift carry signals by 1 day so we don't use same-day info for trading
    # signal_basis_t = carry_raw_{t-1}
    # For spread variant, we already aligned above, so we need to handle this carefully
    if variant == VARIANT_SPREAD:
        # For spread, carry_raw and daily_returns are already aligned
        # Shift carry_raw by 1 day
        signal_basis = carry_raw.shift(1).dropna()
        # Align daily_returns with signal_basis (drop first day of returns since signal is NaN)
        common_idx = signal_basis.index.intersection(daily_returns.index)
        signal_basis = signal_basis.loc[common_idx]
        daily_returns = daily_returns.loc[common_idx]
    else:
        signal_basis = carry_raw.shift(1).dropna()
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
    logger.info(f"  Start: {common_idx[0]}")
    logger.info(f"  End: {common_idx[-1]}")
    
    # Generate sign-only positions
    # position_t = sign(signal_basis_t)
    # +1 if > 0 (positive carry, long), -1 if < 0 (negative carry, short), 0 if abs < threshold
    positions = signal_basis.copy()
    positions[positions.abs() < zero_threshold] = 0.0
    positions[positions > zero_threshold] = 1.0
    positions[positions < -zero_threshold] = -1.0
    
    # Compute strategy returns
    # strategy_ret_t = position_t * daily_return_t
    asset_strategy_returns = positions * daily_returns
    
    # For single-asset strategy, portfolio returns = asset strategy returns
    portfolio_returns = asset_strategy_returns
    
    # Convert to DataFrame for consistency with multi-asset format
    if rank_pair is not None:
        long_rank, short_rank = rank_pair
        asset_name = f"SR3_SPREAD_R{long_rank}_R{short_rank}"
    elif variant == VARIANT_SPREAD:
        asset_name = "SR3_SPREAD_R2_R1"
    else:
        asset_name = TRADEABLE_SYMBOL
    
    asset_strategy_returns_df = pd.DataFrame({asset_name: asset_strategy_returns})
    asset_returns_df = pd.DataFrame({asset_name: daily_returns})
    
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
        'carry_signals': carry_raw.loc[common_idx] if common_idx[0] in carry_raw.index else pd.Series(dtype=float),
        'variant': variant,
        'signal_description': signal_desc,
        'asset_name': asset_name,
        'rank_pair': rank_pair,
        'pair_label': pair_label if rank_pair is not None else (pair_label if 'pair_label' in locals() else None),
        'raw_spread_returns': daily_returns if (variant == VARIANT_SPREAD or rank_pair is not None) else None  # Store raw spread returns for correlation calculation
    }


def compute_summary_stats(
    portfolio_returns: pd.Series,
    equity_curve: pd.Series,
    asset_strategy_returns: pd.DataFrame
) -> Dict:
    """
    Compute summary statistics for the sign-only SR3 carry strategy.
    
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
    end_date: Optional[str],
    variant: Optional[str] = None
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
        'ranks_used': CANONICAL_RANKS,
        'tradeable_symbol': TRADEABLE_SYMBOL,
        'variant': variant or results.get('variant', 'unknown'),
        'signal_definition': results.get('signal_description', 'unknown'),
        'portfolio_metrics': stats['portfolio'],
        'per_asset_stats': stats['per_asset'].to_dict('index') if not stats['per_asset'].empty else {},
        'subperiod_stats': subperiod_stats,
        'data_integrity': {
            'rank_mapping_fix_applied': True,
            'fix_date': '2025-12-16',
            'note': 'Results generated after canonical rank parsing fix. Rank mapping uses parse_sr3_calendar_rank() instead of alphabetical sorting.'
        }
    }
    
    # Add rank_pair info if present
    if results.get('rank_pair') is not None:
        meta['rank_pair'] = results['rank_pair']
        meta['pair_label'] = results.get('pair_label')
    
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
    ax.set_title(f'SR3 Calendar Carry: Cumulative Equity Curve\n'
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
        axes[1].set_xlabel('Carry Signal (r2 - r1, bps)')
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
        asset_name = results.get('asset_name', TRADEABLE_SYMBOL)
        positions = results['positions'][asset_name].dropna()
        
        ax2 = ax.twinx()
        
        # Carry signal
        ax.plot(carry_sig.index, carry_sig.values, 
                label='Carry Signal (r2 - r1)', alpha=0.6, color='blue', linewidth=1)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_ylabel('Carry Signal (bps)', color='blue')
        ax.tick_params(axis='y', labelcolor='blue')
        
        # Positions (overlay)
        ax2.plot(positions.index, positions.values, 
                label='Position (sign)', alpha=0.8, color='red', linewidth=1, linestyle='--')
        ax2.set_ylabel('Position (+1/-1/0)', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        ax2.set_ylim([-1.5, 1.5])
        
        ax.set_title('SR3 Calendar Carry: Signal and Positions Over Time')
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

