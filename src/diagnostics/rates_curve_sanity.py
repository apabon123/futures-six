"""
Rates Curve Sign-Only Sanity Check

A deliberately simple, academic-style curve trading strategy to verify that
the rates curve features and P&L machinery are working correctly.

Strategy:
- For each date, get curve features (curve_2s10s_z, curve_5s30s_z)
- Take the sign of each curve feature (+1 if > 0, -1 if < 0, 0 if ≈ 0)
- Map to flattener/steepener trades:
  - Positive signal → flattener: long front, short back
  - Negative signal → steepener: short front, long back
- Compute daily returns for 2s10s and 5s30s legs
- Combine 50/50 for overall curve sleeve return

This is a diagnostic tool to answer: "Does a very simple curve trading strategy
show reasonable positive Sharpe on our FRED-anchored yield curve features, or is
the alpha gone because our yield reconstruction / feature computation / P&L pipeline is wrong?"

If sign-only curve trading shows Sharpe ~0.2-0.5, then features & P&L pipeline are probably fine.
If it shows negative or near-zero Sharpe, we should question:
- The yield reconstruction (FRED + futures)
- DV01 inputs
- The hypothesis that "steep vs flat vs hump" is still a tradable edge
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple, List, Dict, Any
from pathlib import Path
import json
import logging
import yaml
from datetime import datetime

logger = logging.getLogger(__name__)

# Treasury futures symbols
RATES_SYMBOLS = {
    "2y": "ZT_FRONT_VOLUME",
    "5y": "ZF_FRONT_VOLUME",
    "10y": "ZN_FRONT_VOLUME",
    "30y": "UB_FRONT_VOLUME",
}


def load_dv01_config(dv01_config_path: str = "configs/rates_dv01.yaml") -> Dict[str, float]:
    """
    Load DV01 values from config file.
    
    Args:
        dv01_config_path: Path to DV01 config YAML file
        
    Returns:
        Dict mapping symbol root to DV01 value
    """
    config_path = Path(dv01_config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"DV01 config not found: {dv01_config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        dv01_yaml = yaml.safe_load(f)
    
    return {
        "ZT": dv01_yaml["ZT"]["dv01"],
        "ZF": dv01_yaml["ZF"]["dv01"],
        "ZN": dv01_yaml["ZN"]["dv01"],
        "UB": dv01_yaml["UB"]["dv01"],
    }


def compute_dv01_neutral_weights(
    dv01_2y: float,
    dv01_10y: float
) -> Tuple[float, float]:
    """
    Compute DV01-neutral weights for 2s10s trade.
    
    For a flattener (long 2y, short 10y), we want:
    w_2y * DV01_2y + w_10y * DV01_10y = 0
    
    With w_2y = -w_10y (opposite positions), we get:
    w_2y * DV01_2y - w_2y * DV01_10y = 0
    w_2y * (DV01_2y - DV01_10y) = 0
    
    Actually, for DV01-neutral, we want:
    |w_2y| * DV01_2y = |w_10y| * DV01_10y
    
    If we set w_2y = +1 (long), then w_10y = -DV01_2y / DV01_10y (short)
    But we want to normalize so that the total notional is reasonable.
    
    Standard approach: normalize so that |w_2y| + |w_10y| = 2.0 (equal gross exposure)
    But we want DV01-neutral, so:
    w_2y * DV01_2y = -w_10y * DV01_10y
    
    If w_2y = +1, then w_10y = -DV01_2y / DV01_10y
    Then normalize: total = |w_2y| + |w_10y| = 1 + DV01_2y / DV01_10y
    w_2y_norm = 1 / (1 + DV01_2y / DV01_10y) = DV01_10y / (DV01_2y + DV01_10y)
    w_10y_norm = -(DV01_2y / DV01_10y) / (1 + DV01_2y / DV01_10y) = -DV01_2y / (DV01_2y + DV01_10y)
    
    Args:
        dv01_2y: DV01 for 2-year futures
        dv01_10y: DV01 for 10-year futures
        
    Returns:
        Tuple of (w_2y, w_10y) weights
    """
    w_2y = dv01_10y / (dv01_2y + dv01_10y)
    w_10y = -dv01_2y / (dv01_2y + dv01_10y)
    return w_2y, w_10y


def run_sign_only_curve(
    prices: pd.DataFrame,
    curve_features: pd.DataFrame,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    use_dv01_neutral: bool = True,
    dv01_config_path: str = "configs/rates_dv01.yaml",
    zero_threshold: float = 1e-8
) -> Dict[str, Any]:
    """
    Run sign-only curve trading strategy.
    
    Args:
        prices: Wide DataFrame [date x symbol] of continuous adjusted closes
                Must include ZT_FRONT_VOLUME, ZF_FRONT_VOLUME, ZN_FRONT_VOLUME, UB_FRONT_VOLUME
        curve_features: DataFrame [date x feature] with curve_2s10s_z and curve_5s30s_z columns
        start_date: Start date for backtest (YYYY-MM-DD)
        end_date: End date for backtest (YYYY-MM-DD)
        use_dv01_neutral: If True, use DV01-neutral weighting; if False, use equal notional
        dv01_config_path: Path to DV01 config file
        zero_threshold: Threshold for treating curve feature as zero
        
    Returns:
        Dict with:
        - 'portfolio_returns': pd.Series of daily curve sleeve returns
        - 'equity_curve': pd.Series of cumulative equity
        - 'leg_2s10s_returns': pd.Series of 2s10s leg returns
        - 'leg_5s30s_returns': pd.Series of 5s30s leg returns
        - 'positions_2s10s': pd.DataFrame of 2s10s positions [date x (ZT, ZN)]
        - 'positions_5s30s': pd.DataFrame of 5s30s positions [date x (ZF, UB)]
        - 'metrics': Dict with {cagr, vol, sharpe, maxdd, hit_rate, n_days, years}
        - 'leg_metrics': Dict with metrics for each leg
    """
    results = compute_sign_only_curve(
        prices=prices,
        curve_features=curve_features,
        start_date=start_date,
        end_date=end_date,
        use_dv01_neutral=use_dv01_neutral,
        dv01_config_path=dv01_config_path,
        zero_threshold=zero_threshold
    )
    
    # Compute metrics
    stats = compute_summary_stats(
        portfolio_returns=results['portfolio_returns'],
        equity_curve=results['equity_curve'],
        leg_2s10s_returns=results['leg_2s10s_returns'],
        leg_5s30s_returns=results['leg_5s30s_returns']
    )
    
    # Add metrics to results
    results['metrics'] = stats['portfolio']
    results['leg_metrics'] = stats['legs']
    
    return results


def compute_sign_only_curve(
    prices: pd.DataFrame,
    curve_features: pd.DataFrame,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    use_dv01_neutral: bool = True,
    dv01_config_path: str = "configs/rates_dv01.yaml",
    zero_threshold: float = 1e-8
) -> Dict:
    """
    Compute sign-only curve trading strategy returns.
    
    Args:
        prices: DataFrame of continuous prices [date x symbol]
        curve_features: DataFrame [date x feature] with curve_2s10s_z and curve_5s30s_z
        start_date: Start date for backtest (YYYY-MM-DD)
        end_date: End date for backtest (YYYY-MM-DD)
        use_dv01_neutral: If True, use DV01-neutral weighting
        dv01_config_path: Path to DV01 config file
        zero_threshold: Threshold for treating curve feature as zero
        
    Returns:
        Dict with returns, positions, and equity curves
    """
    # Check required symbols
    required_symbols = list(RATES_SYMBOLS.values())
    missing = [s for s in required_symbols if s not in prices.columns]
    if missing:
        raise ValueError(f"Missing required symbols in prices: {missing}")
    
    # Check required features
    required_features = ['curve_2s10s_z', 'curve_5s30s_z']
    missing_features = [f for f in required_features if f not in curve_features.columns]
    if missing_features:
        raise ValueError(f"Missing required features: {missing_features}")
    
    logger.info(f"Using {len(required_symbols)} rates symbols: {required_symbols}")
    
    # Load DV01 if needed
    if use_dv01_neutral:
        dv01 = load_dv01_config(dv01_config_path)
        logger.info(f"Using DV01-neutral weighting: {dv01}")
    else:
        dv01 = None
        logger.info("Using equal notional weighting")
    
    # Extract prices for rates symbols
    rates_prices = prices[required_symbols].copy()
    
    # Compute daily returns (simple returns)
    daily_returns = rates_prices.pct_change(fill_method=None).dropna()
    
    if daily_returns.empty:
        raise ValueError("No valid daily returns computed")
    
    logger.info(f"Daily returns computed: {daily_returns.shape}")
    
    # Align curve features with prices
    # Ensure both have DatetimeIndex
    if not isinstance(curve_features.index, pd.DatetimeIndex):
        curve_features.index = pd.to_datetime(curve_features.index)
    if not isinstance(daily_returns.index, pd.DatetimeIndex):
        daily_returns.index = pd.to_datetime(daily_returns.index)
    
    # Get common dates (inner join)
    common_dates = daily_returns.index.intersection(curve_features.index)
    
    if len(common_dates) == 0:
        raise ValueError("No common dates between prices and curve features")
    
    # Filter by date range
    if start_date:
        start_dt = pd.to_datetime(start_date)
        common_dates = common_dates[common_dates >= start_dt]
    
    if end_date:
        end_dt = pd.to_datetime(end_date)
        common_dates = common_dates[common_dates <= end_dt]
    
    if len(common_dates) == 0:
        raise ValueError("No data available after date filtering")
    
    # Sort dates
    common_dates = common_dates.sort_values()
    
    logger.info(f"Aligned data: {len(common_dates)} days")
    
    # Extract curve features for common dates
    curve_2s10s = curve_features.loc[common_dates, 'curve_2s10s_z']
    curve_5s30s = curve_features.loc[common_dates, 'curve_5s30s_z']
    
    # Forward-fill missing values (use most recent available feature)
    curve_2s10s = curve_2s10s.ffill()
    curve_5s30s = curve_5s30s.ffill()
    
    # Shift features by 1 day so we don't use same-day info for trading
    # signal_basis_t = curve_feature_{t-1}
    curve_2s10s_signal = curve_2s10s.shift(1).dropna()
    curve_5s30s_signal = curve_5s30s.shift(1).dropna()
    
    # Align daily returns with signals
    common_idx = curve_2s10s_signal.index.intersection(
        curve_5s30s_signal.index
    ).intersection(daily_returns.index)
    
    curve_2s10s_signal = curve_2s10s_signal.loc[common_idx]
    curve_5s30s_signal = curve_5s30s_signal.loc[common_idx]
    daily_returns = daily_returns.loc[common_idx]
    
    if len(common_idx) == 0:
        raise ValueError("No aligned data after signal shift")
    
    logger.info(f"Final aligned data: {len(common_idx)} days")
    
    # Generate sign-only positions
    # For 2s10s: positive signal → flattener (long ZT, short ZN)
    #            negative signal → steepener (short ZT, long ZN)
    pos_2s10s = curve_2s10s_signal.copy()
    pos_2s10s[pos_2s10s.abs() < zero_threshold] = 0.0
    pos_2s10s[pos_2s10s > zero_threshold] = 1.0
    pos_2s10s[pos_2s10s < -zero_threshold] = -1.0
    
    # For 5s30s: positive signal → flattener (long ZF, short UB)
    #            negative signal → steepener (short ZF, long UB)
    pos_5s30s = curve_5s30s_signal.copy()
    pos_5s30s[pos_5s30s.abs() < zero_threshold] = 0.0
    pos_5s30s[pos_5s30s > zero_threshold] = 1.0
    pos_5s30s[pos_5s30s < -zero_threshold] = -1.0
    
    # Compute weights for each leg
    if use_dv01_neutral:
        # 2s10s: DV01-neutral weights
        w_2y, w_10y = compute_dv01_neutral_weights(dv01["ZT"], dv01["ZN"])
        # 5s30s: DV01-neutral weights
        w_5y, w_30y = compute_dv01_neutral_weights(dv01["ZF"], dv01["UB"])
        logger.info(f"2s10s weights: ZT={w_2y:.4f}, ZN={w_10y:.4f}")
        logger.info(f"5s30s weights: ZF={w_5y:.4f}, UB={w_30y:.4f}")
    else:
        # Equal notional: +0.5 and -0.5
        w_2y, w_10y = 0.5, -0.5
        w_5y, w_30y = 0.5, -0.5
    
    # Build position DataFrames
    positions_2s10s = pd.DataFrame(index=common_idx)
    positions_2s10s[RATES_SYMBOLS["2y"]] = pos_2s10s * w_2y
    positions_2s10s[RATES_SYMBOLS["10y"]] = pos_2s10s * w_10y
    
    positions_5s30s = pd.DataFrame(index=common_idx)
    positions_5s30s[RATES_SYMBOLS["5y"]] = pos_5s30s * w_5y
    positions_5s30s[RATES_SYMBOLS["30y"]] = pos_5s30s * w_30y
    
    # Compute per-leg returns
    # 2s10s leg: w_ZT * r_ZT + w_ZN * r_ZN
    leg_2s10s_returns = (
        positions_2s10s[RATES_SYMBOLS["2y"]] * daily_returns[RATES_SYMBOLS["2y"]] +
        positions_2s10s[RATES_SYMBOLS["10y"]] * daily_returns[RATES_SYMBOLS["10y"]]
    )
    
    # 5s30s leg: w_ZF * r_ZF + w_UB * r_UB
    leg_5s30s_returns = (
        positions_5s30s[RATES_SYMBOLS["5y"]] * daily_returns[RATES_SYMBOLS["5y"]] +
        positions_5s30s[RATES_SYMBOLS["30y"]] * daily_returns[RATES_SYMBOLS["30y"]]
    )
    
    # Combine legs 50/50 for overall curve sleeve return
    portfolio_returns = 0.5 * leg_2s10s_returns + 0.5 * leg_5s30s_returns
    
    # Compute equity curves
    if len(portfolio_returns) > 0:
        equity_curve = (1 + portfolio_returns).cumprod()
        equity_2s10s = (1 + leg_2s10s_returns).cumprod()
        equity_5s30s = (1 + leg_5s30s_returns).cumprod()
    else:
        equity_curve = pd.Series(dtype=float)
        equity_2s10s = pd.Series(dtype=float)
        equity_5s30s = pd.Series(dtype=float)
    
    return {
        'portfolio_returns': portfolio_returns,
        'equity_curve': equity_curve,
        'leg_2s10s_returns': leg_2s10s_returns,
        'leg_5s30s_returns': leg_5s30s_returns,
        'equity_2s10s': equity_2s10s,
        'equity_5s30s': equity_5s30s,
        'positions_2s10s': positions_2s10s,
        'positions_5s30s': positions_5s30s,
        'curve_2s10s_signal': curve_2s10s_signal,
        'curve_5s30s_signal': curve_5s30s_signal,
    }


def compute_summary_stats(
    portfolio_returns: pd.Series,
    equity_curve: pd.Series,
    leg_2s10s_returns: pd.Series,
    leg_5s30s_returns: pd.Series
) -> Dict:
    """
    Compute summary statistics for the sign-only curve strategy.
    
    Returns:
        Dict with portfolio metrics and per-leg stats
    """
    if len(portfolio_returns) == 0:
        return {
            'portfolio': {},
            'legs': {}
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
    
    # Per-leg stats
    leg_metrics = {}
    
    for leg_name, leg_returns in [('2s10s', leg_2s10s_returns), ('5s30s', leg_5s30s_returns)]:
        if len(leg_returns) == 0:
            continue
        
        # Annualized return
        ann_ret = leg_returns.mean() * 252
        
        # Annualized vol
        ann_vol = leg_returns.std() * (252 ** 0.5)
        
        # Sharpe
        if ann_vol != 0:
            sharpe = ann_ret / ann_vol
        else:
            sharpe = float('nan')
        
        leg_metrics[leg_name] = {
            'AnnRet': ann_ret,
            'AnnVol': ann_vol,
            'Sharpe': sharpe
        }
    
    return {
        'portfolio': portfolio_metrics,
        'legs': leg_metrics
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
    use_dv01_neutral: bool
):
    """
    Save all results to output directory.
    
    Saves:
    - portfolio_returns.csv
    - equity_curve.csv
    - leg_returns.csv
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
    
    # Save leg returns
    leg_returns_df = pd.DataFrame({
        'date': results['leg_2s10s_returns'].index,
        '2s10s': results['leg_2s10s_returns'].values,
        '5s30s': results['leg_5s30s_returns'].values
    })
    leg_returns_df.to_csv(output_dir / 'leg_returns.csv', index=False)
    
    # Save meta
    meta = {
        'start_date': start_date,
        'end_date': end_date,
        'use_dv01_neutral': use_dv01_neutral,
        'n_days': len(results['portfolio_returns']),
        'portfolio_metrics': stats['portfolio'],
        'leg_metrics': stats['legs'],
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
    1. Cumulative equity curves (portfolio and per-leg)
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
            label='Curve Sleeve (50/50)', linewidth=2, color='black')
    
    # Per-leg equity curves
    ax.plot(results['equity_2s10s'].index, results['equity_2s10s'].values,
            label=f"2s10s (Sharpe={stats['legs'].get('2s10s', {}).get('Sharpe', 0):.2f})",
            alpha=0.7, linestyle='--')
    
    ax.plot(results['equity_5s30s'].index, results['equity_5s30s'].values,
            label=f"5s30s (Sharpe={stats['legs'].get('5s30s', {}).get('Sharpe', 0):.2f})",
            alpha=0.7, linestyle='--')
    
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    ax.set_title(f'Sign-Only Rates Curve: Cumulative Equity Curves\n'
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
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Portfolio returns
    portfolio_ret = results['portfolio_returns'].dropna()
    axes[0].hist(portfolio_ret.values, bins=50, alpha=0.7, edgecolor='black')
    axes[0].axvline(x=0, color='red', linestyle='--', alpha=0.5)
    axes[0].set_title(f'Portfolio Returns\n(Mean={portfolio_ret.mean():.4f}, Std={portfolio_ret.std():.4f})')
    axes[0].set_xlabel('Daily Return')
    axes[0].set_ylabel('Frequency')
    axes[0].grid(True, alpha=0.3)
    
    # 2s10s returns
    leg_2s10s_ret = results['leg_2s10s_returns'].dropna()
    axes[1].hist(leg_2s10s_ret.values, bins=50, alpha=0.7, edgecolor='black', color='orange')
    axes[1].axvline(x=0, color='red', linestyle='--', alpha=0.5)
    axes[1].set_title(f'2s10s Leg Returns\n(Mean={leg_2s10s_ret.mean():.4f}, Std={leg_2s10s_ret.std():.4f})')
    axes[1].set_xlabel('Daily Return')
    axes[1].set_ylabel('Frequency')
    axes[1].grid(True, alpha=0.3)
    
    # 5s30s returns
    leg_5s30s_ret = results['leg_5s30s_returns'].dropna()
    axes[2].hist(leg_5s30s_ret.values, bins=50, alpha=0.7, edgecolor='black', color='green')
    axes[2].axvline(x=0, color='red', linestyle='--', alpha=0.5)
    axes[2].set_title(f'5s30s Leg Returns\n(Mean={leg_5s30s_ret.mean():.4f}, Std={leg_5s30s_ret.std():.4f})')
    axes[2].set_xlabel('Daily Return')
    axes[2].set_ylabel('Frequency')
    axes[2].grid(True, alpha=0.3)
    
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

