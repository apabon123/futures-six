"""
Diagnostics & Attribution Utility

Consumes run results from ExecSim and produces standardized diagnostic outputs.
Read-only utility that generates performance metrics, attribution tables, and CSV reports.

No external writes beyond the reports folder. All outputs time-aligned on intersection of dates.
"""

import logging
from typing import Dict, Optional, Tuple
from pathlib import Path
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def make_report(results: dict, outdir: str = "reports/phase1") -> dict:
    """
    Generate diagnostic report from backtest results.
    
    Computes comprehensive performance metrics and saves standardized CSV outputs.
    All data is time-aligned on the intersection of available dates.
    
    Args:
        results: Dict containing run results with keys:
            - equity_curve: pd.Series (required)
            - weights: dict[str->pd.DataFrame] or pd.DataFrame (optional)
            - signals: dict[str->pd.DataFrame] or pd.DataFrame (optional)
            - pnl: dict[str->pd.Series] (optional per-sleeve P&L)
            - asset_pnl: pd.DataFrame (optional, columns=symbols)
            - turnover: pd.Series (optional)
            - costs: pd.Series (optional)
            
            Note: Also accepts legacy keys (weights_panel, signals_panel) for backward compatibility
            
        outdir: Output directory for CSV reports (default: "reports/phase1")
        
    Returns:
        Dict containing:
            - metrics: Dict of performance metrics (cagr, sharpe, max_drawdown, etc.)
            - files: Dict of saved file paths
            
    Raises:
        ValueError: If equity_curve is missing or empty
    """
    # Validate required inputs
    if 'equity_curve' not in results:
        raise ValueError("results must contain 'equity_curve'")
    
    equity_curve = results['equity_curve']
    
    if equity_curve.empty:
        logger.warning("[Diagnostics] Empty equity curve, returning empty report")
        return {
            'metrics': _empty_metrics(),
            'files': {}
        }
    
    # Create output directory
    outpath = Path(outdir)
    outpath.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"[Diagnostics] Generating report for {len(equity_curve)} periods")
    
    # Extract components (handle both new dict format and legacy panel format)
    weights_dict = _extract_weights(results)
    signals_dict = _extract_signals(results)
    pnl_dict = results.get('pnl', {})
    asset_pnl = results.get('asset_pnl', pd.DataFrame())
    turnover = results.get('turnover', pd.Series(dtype=float))
    costs = results.get('costs', pd.Series(dtype=float))
    
    # Compute metrics
    metrics = _compute_metrics(
        equity_curve=equity_curve,
        weights_dict=weights_dict,
        pnl_dict=pnl_dict,
        turnover=turnover,
        costs=costs
    )
    
    # Save CSV files
    files = {}
    
    # 1. Equity curve
    equity_path = outpath / "equity.csv"
    equity_df = equity_curve.to_frame('equity')
    equity_df.index.name = 'date'
    equity_df.to_csv(equity_path)
    files['equity'] = str(equity_path)
    logger.info(f"[Diagnostics] Saved equity curve: {equity_path}")
    
    # 2. Sleeve P&L (if available)
    if pnl_dict:
        sleeve_pnl_df = pd.DataFrame(pnl_dict)
        sleeve_pnl_df.index.name = 'date'
        sleeve_pnl_path = outpath / "sleeve_pnl.csv"
        sleeve_pnl_df.to_csv(sleeve_pnl_path)
        files['sleeve_pnl'] = str(sleeve_pnl_path)
        logger.info(f"[Diagnostics] Saved sleeve P&L: {sleeve_pnl_path}")
    
    # 3. Asset P&L (if available)
    if not asset_pnl.empty:
        asset_pnl.index.name = 'date'
        asset_pnl_path = outpath / "asset_pnl.csv"
        asset_pnl.to_csv(asset_pnl_path)
        files['asset_pnl'] = str(asset_pnl_path)
        logger.info(f"[Diagnostics] Saved asset P&L: {asset_pnl_path}")
    
    # 4. Total weights (prefer 'total' key, fallback to first available)
    if weights_dict:
        if 'total' in weights_dict:
            weights_total = weights_dict['total']
        else:
            # Use first available weights
            weights_total = next(iter(weights_dict.values()))
        
        weights_total.index.name = 'date'
        weights_path = outpath / "weights_total.csv"
        weights_total.to_csv(weights_path)
        files['weights_total'] = str(weights_path)
        logger.info(f"[Diagnostics] Saved total weights: {weights_path}")
    
    # 5. Turnover and costs (combined)
    if not turnover.empty or not costs.empty:
        turnover_costs_df = pd.DataFrame({
            'turnover': turnover,
            'costs': costs
        })
        turnover_costs_df.index.name = 'date'
        turnover_costs_path = outpath / "turnover_costs.csv"
        turnover_costs_df.to_csv(turnover_costs_path)
        files['turnover_costs'] = str(turnover_costs_path)
        logger.info(f"[Diagnostics] Saved turnover & costs: {turnover_costs_path}")
    
    logger.info(
        f"[Diagnostics] Report complete: "
        f"CAGR={metrics['cagr']:.2%}, Sharpe={metrics['sharpe']:.2f}, "
        f"MaxDD={metrics['max_drawdown']:.2%}"
    )
    
    return {
        'metrics': metrics,
        'files': files
    }


def _extract_weights(results: dict) -> Dict[str, pd.DataFrame]:
    """Extract weights from results, handling both dict and DataFrame formats."""
    # Check for dict format first
    if 'weights' in results and isinstance(results['weights'], dict):
        return results['weights']
    
    # Check for legacy DataFrame format
    if 'weights_panel' in results and isinstance(results['weights_panel'], pd.DataFrame):
        if not results['weights_panel'].empty:
            return {'total': results['weights_panel']}
    
    # Check for single DataFrame under 'weights'
    if 'weights' in results and isinstance(results['weights'], pd.DataFrame):
        if not results['weights'].empty:
            return {'total': results['weights']}
    
    return {}


def _extract_signals(results: dict) -> Dict[str, pd.DataFrame]:
    """Extract signals from results, handling both dict and DataFrame formats."""
    # Check for dict format first
    if 'signals' in results and isinstance(results['signals'], dict):
        return results['signals']
    
    # Check for legacy DataFrame format
    if 'signals_panel' in results and isinstance(results['signals_panel'], pd.DataFrame):
        if not results['signals_panel'].empty:
            return {'total': results['signals_panel']}
    
    # Check for single DataFrame under 'signals'
    if 'signals' in results and isinstance(results['signals'], pd.DataFrame):
        if not results['signals'].empty:
            return {'total': results['signals']}
    
    return {}


def _compute_metrics(
    equity_curve: pd.Series,
    weights_dict: Dict[str, pd.DataFrame],
    pnl_dict: Dict[str, pd.Series],
    turnover: pd.Series,
    costs: pd.Series
) -> dict:
    """
    Compute comprehensive performance metrics.
    
    Returns dict with:
        - cagr: Compound annual growth rate
        - vol: Annualized volatility
        - sharpe: Sharpe ratio (mean/std * sqrt(252) on daily returns)
        - max_drawdown: Maximum drawdown
        - calmar: Calmar ratio (CAGR / abs(max_drawdown))
        - hit_rate: Percentage of positive returns
        - avg_drawdown_length: Average length of drawdowns in days
        - avg_gross_exposure: Average gross leverage
        - avg_net_exposure: Average net exposure
        - avg_turnover: Average turnover per period
        - cost_drag: Average cost per period (annualized)
    """
    if equity_curve.empty:
        return _empty_metrics()
    
    # Compute daily returns from equity curve
    daily_returns = _compute_daily_returns(equity_curve)
    
    if daily_returns.empty:
        return _empty_metrics()
    
    # Basic metrics
    metrics = {}
    
    # CAGR
    total_return = equity_curve.iloc[-1] / equity_curve.iloc[0] - 1.0
    n_days = (equity_curve.index[-1] - equity_curve.index[0]).days
    n_years = n_days / 365.25
    
    if n_years > 0:
        metrics['cagr'] = (1 + total_return) ** (1 / n_years) - 1
    else:
        metrics['cagr'] = 0.0
    
    # Volatility (annualized from daily returns)
    metrics['vol'] = daily_returns.std() * np.sqrt(252)
    
    # Sharpe ratio: mean(daily) / std(daily) * sqrt(252)
    if metrics['vol'] > 0:
        metrics['sharpe'] = (daily_returns.mean() * 252) / (daily_returns.std() * np.sqrt(252))
    else:
        metrics['sharpe'] = 0.0
    
    # Maximum drawdown
    metrics['max_drawdown'] = _compute_max_drawdown(equity_curve)
    
    # Calmar ratio
    if abs(metrics['max_drawdown']) > 1e-6:
        metrics['calmar'] = metrics['cagr'] / abs(metrics['max_drawdown'])
    else:
        metrics['calmar'] = 0.0
    
    # Hit rate
    metrics['hit_rate'] = (daily_returns > 0).mean()
    
    # Average drawdown length
    metrics['avg_drawdown_length'] = _compute_avg_drawdown_length(equity_curve)
    
    # Exposure metrics
    if weights_dict:
        # Use total weights if available, otherwise first available
        if 'total' in weights_dict:
            weights = weights_dict['total']
        else:
            weights = next(iter(weights_dict.values()))
        
        metrics['avg_gross_exposure'] = weights.abs().sum(axis=1).mean()
        metrics['avg_net_exposure'] = weights.sum(axis=1).abs().mean()
    else:
        metrics['avg_gross_exposure'] = 0.0
        metrics['avg_net_exposure'] = 0.0
    
    # Turnover
    if not turnover.empty:
        metrics['avg_turnover'] = turnover.mean()
    else:
        metrics['avg_turnover'] = 0.0
    
    # Cost drag (annualized)
    if not costs.empty and len(costs) > 0:
        # Average cost per period, annualized
        avg_cost_per_period = costs.mean()
        # Estimate periods per year (from equity curve frequency)
        periods_per_year = _estimate_periods_per_year(equity_curve)
        metrics['cost_drag'] = avg_cost_per_period * periods_per_year
    else:
        metrics['cost_drag'] = 0.0
    
    return metrics


def _compute_daily_returns(equity_curve: pd.Series) -> pd.Series:
    """
    Compute daily returns from equity curve.
    
    Returns simple returns (pct_change) aligned to equity curve dates.
    """
    if equity_curve.empty or len(equity_curve) < 2:
        return pd.Series(dtype=float)
    
    # Simple returns
    returns = equity_curve.pct_change().dropna()
    
    return returns


def _compute_max_drawdown(equity_curve: pd.Series) -> float:
    """Compute maximum drawdown from equity curve."""
    if equity_curve.empty or len(equity_curve) < 2:
        return 0.0
    
    running_max = equity_curve.expanding().max()
    drawdown = (equity_curve - running_max) / running_max
    max_dd = drawdown.min()
    
    return max_dd


def _compute_avg_drawdown_length(equity_curve: pd.Series) -> float:
    """
    Compute average drawdown length in days.
    
    A drawdown is defined as a period where equity is below its running maximum.
    """
    if equity_curve.empty or len(equity_curve) < 2:
        return 0.0
    
    running_max = equity_curve.expanding().max()
    in_drawdown = equity_curve < running_max
    
    if not in_drawdown.any():
        return 0.0
    
    # Identify drawdown periods (consecutive True values)
    drawdown_periods = []
    current_length = 0
    
    for is_dd in in_drawdown:
        if is_dd:
            current_length += 1
        else:
            if current_length > 0:
                drawdown_periods.append(current_length)
            current_length = 0
    
    # Add final drawdown if still ongoing
    if current_length > 0:
        drawdown_periods.append(current_length)
    
    if not drawdown_periods:
        return 0.0
    
    return np.mean(drawdown_periods)


def _estimate_periods_per_year(equity_curve: pd.Series) -> float:
    """
    Estimate number of periods per year from equity curve frequency.
    
    Returns approximate scaling factor for annualization.
    """
    if equity_curve.empty or len(equity_curve) < 2:
        return 252.0  # Default to daily
    
    # Compute average days between observations
    time_diffs = equity_curve.index.to_series().diff().dropna()
    
    if time_diffs.empty:
        return 252.0
    
    avg_days = time_diffs.mean().days
    
    if avg_days == 0:
        return 252.0
    
    periods_per_year = 365.25 / avg_days
    
    return periods_per_year


def _empty_metrics() -> dict:
    """Return empty metrics dict for edge cases."""
    return {
        'cagr': 0.0,
        'vol': 0.0,
        'sharpe': 0.0,
        'max_drawdown': 0.0,
        'calmar': 0.0,
        'hit_rate': 0.0,
        'avg_drawdown_length': 0.0,
        'avg_gross_exposure': 0.0,
        'avg_net_exposure': 0.0,
        'avg_turnover': 0.0,
        'cost_drag': 0.0
    }

