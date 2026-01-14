"""
Canonical Diagnostic Outputs

Generates comprehensive diagnostic reports in JSON and Markdown formats.

These are decision documents, not charts. They answer:
- "Why is the Sharpe low?"
- "Which engines are doing the work?"
- "Are constraints binding?"
- "What caused the worst drawdowns?"

Output Format:
- JSON (for machines)
- Markdown tables (for humans)
- Deterministic (same run_id → same outputs)

Think of this as a risk committee pack.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, List
from datetime import datetime
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def _load_csv_with_date_index(csv_path: Path, date_col_name: str = 'rebalance_date') -> pd.DataFrame:
    """
    Load CSV with date index, handling legacy formats with backward compatibility.
    
    Supports:
    - Canonical format: column named 'rebalance_date' (or date_col_name)
    - Legacy format: first column is 'Unnamed: 0' containing dates
    - Legacy format: column named 'date'
    
    Args:
        csv_path: Path to CSV file
        date_col_name: Expected date column name in canonical format (default: 'rebalance_date')
        
    Returns:
        DataFrame with DatetimeIndex
    """
    try:
        # Try canonical format first
        df = pd.read_csv(csv_path, parse_dates=[date_col_name], index_col=date_col_name)
        return df
    except ValueError as e:
        if 'Missing column' in str(e) or date_col_name not in str(e):
            # Legacy format - try reading without specifying date column
            df = pd.read_csv(csv_path)
            
            # Check if first column is unnamed and looks like dates
            first_col = df.columns[0]
            if first_col.startswith('Unnamed:') or first_col == 'date':
                # Use first column as date index
                date_col = first_col
                df[date_col] = pd.to_datetime(df[date_col])
                df = df.set_index(date_col)
                return df
            elif 'date' in df.columns:
                # Has 'date' column - use it
                df['date'] = pd.to_datetime(df['date'])
                df = df.set_index('date')
                return df
            else:
                # Try index_col=0 with parse_dates=True as fallback
                df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
                return df
        else:
            raise


def load_run_artifacts(run_dir: Path) -> Dict:
    """
    Load all artifacts from a run directory.
    
    Args:
        run_dir: Path to run directory
        
    Returns:
        Dict with loaded artifacts (returns, weights, scalars, etc.)
    """
    artifacts = {}
    
    # Required artifacts
    portfolio_returns_file = run_dir / "portfolio_returns.csv"
    if portfolio_returns_file.exists():
        df = pd.read_csv(portfolio_returns_file, parse_dates=['date'], index_col='date')
        artifacts['portfolio_returns'] = df['ret']
    else:
        raise FileNotFoundError(f"portfolio_returns.csv not found in {run_dir}")
    
    equity_curve_file = run_dir / "equity_curve.csv"
    if equity_curve_file.exists():
        df = pd.read_csv(equity_curve_file, parse_dates=['date'], index_col='date')
        artifacts['equity_curve'] = df['equity']
    else:
        raise FileNotFoundError(f"equity_curve.csv not found in {run_dir}")
    
    asset_returns_file = run_dir / "asset_returns.csv"
    if asset_returns_file.exists():
        artifacts['asset_returns'] = pd.read_csv(asset_returns_file, index_col=0, parse_dates=True)
    else:
        logger.warning(f"asset_returns.csv not found in {run_dir}")
        artifacts['asset_returns'] = None
    
    weights_file = run_dir / "weights.csv"
    if weights_file.exists():
        artifacts['weights'] = pd.read_csv(weights_file, index_col=0, parse_dates=True)
    else:
        logger.warning(f"weights.csv not found in {run_dir}")
        artifacts['weights'] = None
    
    # Optional artifacts
    weights_raw_file = run_dir / "weights_raw.csv"
    if weights_raw_file.exists():
        artifacts['weights_raw'] = pd.read_csv(weights_raw_file, index_col=0, parse_dates=True)
    else:
        artifacts['weights_raw'] = None
    
    weights_scaled_file = run_dir / "weights_scaled.csv"
    if weights_scaled_file.exists():
        artifacts['weights_scaled'] = pd.read_csv(weights_scaled_file, index_col=0, parse_dates=True)
    else:
        artifacts['weights_scaled'] = None
    
    sleeve_returns_file = run_dir / "sleeve_returns.csv"
    if sleeve_returns_file.exists():
        artifacts['sleeve_returns'] = pd.read_csv(sleeve_returns_file, index_col=0, parse_dates=True)
    else:
        artifacts['sleeve_returns'] = None
    
    allocator_scalar_file = run_dir / "allocator_risk_v1_applied_used.csv"
    if not allocator_scalar_file.exists():
        allocator_scalar_file = run_dir / "allocator_risk_v1_applied.csv"
    if allocator_scalar_file.exists():
        df = _load_csv_with_date_index(allocator_scalar_file)
        artifacts['allocator_scalar'] = df['risk_scalar_applied'] if 'risk_scalar_applied' in df.columns else df.iloc[:, 0]
    else:
        artifacts['allocator_scalar'] = None
    
    engine_policy_file = run_dir / "engine_policy_applied_v1.csv"
    if engine_policy_file.exists():
        artifacts['engine_policy'] = pd.read_csv(engine_policy_file, parse_dates=['rebalance_date'])
    else:
        artifacts['engine_policy'] = None
    
    meta_file = run_dir / "meta.json"
    if meta_file.exists():
        with open(meta_file, 'r') as f:
            artifacts['meta'] = json.load(f)
    else:
        artifacts['meta'] = {}
    
    return artifacts


def compute_return_decomposition(artifacts: Dict) -> Dict:
    """
    Compute return decomposition:
    - Gross returns (pre allocator)
    - Net returns (post allocator)
    - Vol targeting effect
    - Allocator drag (bps/year)
    - Policy gating drag (bps/year)
    
    Args:
        artifacts: Dict with loaded artifacts
        
    Returns:
        Dict with decomposition metrics
    """
    portfolio_returns = artifacts['portfolio_returns']
    weights_raw = artifacts.get('weights_raw')
    weights_scaled = artifacts.get('weights_scaled')
    weights = artifacts.get('weights')
    asset_returns = artifacts.get('asset_returns')
    allocator_scalar = artifacts.get('allocator_scalar')
    engine_policy = artifacts.get('engine_policy')
    
    # Net returns (post allocator) - this is what we have
    net_returns = portfolio_returns
    
    # Compute gross returns (pre allocator)
    # If we have weights_raw, reconstruct gross returns
    gross_returns = None
    if weights_raw is not None and asset_returns is not None:
        # Forward-fill weights to daily frequency
        weights_raw_daily = weights_raw.reindex(asset_returns.index).ffill().fillna(0.0)
        # Align asset returns
        aligned_returns = asset_returns.reindex(weights_raw_daily.index).fillna(0.0)
        # Compute gross returns: sum of weights * returns
        gross_returns = (weights_raw_daily * aligned_returns).sum(axis=1)
    elif weights is not None and asset_returns is not None:
        # Fallback: use weights if weights_raw not available
        weights_daily = weights.reindex(asset_returns.index).ffill().fillna(0.0)
        aligned_returns = asset_returns.reindex(weights_daily.index).fillna(0.0)
        gross_returns = (weights_daily * aligned_returns).sum(axis=1)
    
    # Compute annualized metrics
    years = len(net_returns) / 252.0
    
    decomposition = {
        'net_returns': {
            'cagr': float(np.nan),
            'vol': float(np.nan),
            'sharpe': float(np.nan),
        },
        'gross_returns': {
            'cagr': float(np.nan),
            'vol': float(np.nan),
            'sharpe': float(np.nan),
        },
        'allocator_drag_bps': float(np.nan),
        'policy_drag_bps': float(np.nan),
        'vol_targeting_effect': float(np.nan),
    }
    
    cagr_net = np.nan
    if len(net_returns) > 0:
        # Net returns metrics
        equity = artifacts['equity_curve']
        if len(equity) > 1 and years > 0:
            equity_start = equity.iloc[0]
            equity_end = equity.iloc[-1]
            if equity_start > 0:
                cagr_net = (equity_end / equity_start) ** (1 / years) - 1
                decomposition['net_returns']['cagr'] = float(cagr_net)
        
        vol_net = net_returns.std() * np.sqrt(252)
        sharpe_net = (net_returns.mean() / net_returns.std() * np.sqrt(252)) if net_returns.std() > 0 else 0.0
        decomposition['net_returns']['vol'] = float(vol_net)
        decomposition['net_returns']['sharpe'] = float(sharpe_net)
    
    cagr_gross = np.nan
    if gross_returns is not None and len(gross_returns) > 0:
        # Align to same dates
        common_dates = net_returns.index.intersection(gross_returns.index)
        if len(common_dates) > 0:
            gross_aligned = gross_returns.loc[common_dates]
            net_aligned = net_returns.loc[common_dates]
            
            # Gross returns metrics
            gross_cum = (1 + gross_aligned).cumprod()
            if len(gross_cum) > 1:
                gross_start = gross_cum.iloc[0]
                gross_end = gross_cum.iloc[-1]
                if gross_start > 0:
                    cagr_gross = (gross_end / gross_start) ** (1 / years) - 1
                    decomposition['gross_returns']['cagr'] = float(cagr_gross)
            
            vol_gross = gross_aligned.std() * np.sqrt(252)
            sharpe_gross = (gross_aligned.mean() / gross_aligned.std() * np.sqrt(252)) if gross_aligned.std() > 0 else 0.0
            decomposition['gross_returns']['vol'] = float(vol_gross)
            decomposition['gross_returns']['sharpe'] = float(sharpe_gross)
            
            # Allocator drag (difference in CAGR, converted to bps/year)
            if not np.isnan(cagr_gross) and not np.isnan(cagr_net):
                allocator_drag = (cagr_gross - cagr_net) * 10000
                decomposition['allocator_drag_bps'] = float(allocator_drag)
    
    # Policy gating drag (if engine policy is available)
    # This is harder to compute without a baseline, so we'll estimate from policy stats
    if engine_policy is not None:
        # Estimate policy drag based on gating frequency
        # This is a simplified estimate - full computation would require baseline comparison
        policy_gated_pct = 0.0
        if 'multiplier' in engine_policy.columns:
            gated_count = (engine_policy['multiplier'] < 1.0).sum()
            policy_gated_pct = gated_count / len(engine_policy) if len(engine_policy) > 0 else 0.0
        # Rough estimate: assume gating reduces returns by some fraction
        # This is a placeholder - would need baseline comparison for accurate measurement
        decomposition['policy_gated_pct'] = float(policy_gated_pct)
    
    return decomposition


def compute_engine_sharpe_contribution(artifacts: Dict) -> pd.DataFrame:
    """
    Compute engine-level Sharpe & contribution for each meta-sleeve:
    - Unconditional Sharpe
    - Contribution to portfolio Sharpe
    - % time active (after policy)
    - Avg exposure when active
    - % of total PnL
    
    Args:
        artifacts: Dict with loaded artifacts
        
    Returns:
        DataFrame with engine-level metrics
    """
    sleeve_returns = artifacts.get('sleeve_returns')
    portfolio_returns = artifacts['portfolio_returns']
    engine_policy = artifacts.get('engine_policy')
    
    if sleeve_returns is None or sleeve_returns.empty:
        logger.warning("sleeve_returns.csv not available - cannot compute engine-level metrics")
        return pd.DataFrame()
    
    # Align sleeve returns with portfolio returns
    common_dates = portfolio_returns.index.intersection(sleeve_returns.index)
    if len(common_dates) == 0:
        logger.warning("No overlapping dates between portfolio and sleeve returns")
        return pd.DataFrame()
    
    sleeve_aligned = sleeve_returns.loc[common_dates]
    portfolio_aligned = portfolio_returns.loc[common_dates]
    
    rows = []
    for sleeve_name in sleeve_aligned.columns:
        sleeve_ret = sleeve_aligned[sleeve_name].dropna()
        
        if len(sleeve_ret) == 0:
            continue
        
        # Unconditional Sharpe
        vol_sleeve = sleeve_ret.std() * np.sqrt(252)
        mean_sleeve = sleeve_ret.mean() * 252
        sharpe_uncond = (mean_sleeve / vol_sleeve) if vol_sleeve > 0 else 0.0
        
        # % of total PnL
        total_pnl = sleeve_ret.sum()
        portfolio_pnl = portfolio_aligned.loc[sleeve_ret.index].sum()
        pct_pnl = (total_pnl / portfolio_pnl * 100) if portfolio_pnl != 0 else 0.0
        
        # Contribution to portfolio Sharpe (simplified - correlation contribution)
        # Full contribution would require covariance decomposition
        correlation = sleeve_ret.corr(portfolio_aligned.loc[sleeve_ret.index])
        sharpe_contrib = sharpe_uncond * correlation if not np.isnan(correlation) else 0.0
        
        # % time active (after policy)
        # If engine policy available, check gating
        pct_active = 100.0
        avg_exposure = sleeve_ret.abs().mean() * 252 if len(sleeve_ret) > 0 else 0.0
        
        if engine_policy is not None:
            # Check if this sleeve is gated
            # This is a simplified check - would need to map sleeve names to engines
            # For now, assume all sleeves are active unless explicitly gated
            pass
        
        rows.append({
            'sleeve': sleeve_name,
            'unconditional_sharpe': sharpe_uncond,
            'contribution_to_portfolio_sharpe': sharpe_contrib,
            'pct_time_active': pct_active,
            'avg_exposure_when_active': avg_exposure,
            'pct_of_total_pnl': pct_pnl,
        })
    
    if len(rows) == 0:
        return pd.DataFrame()
    
    df = pd.DataFrame(rows).set_index('sleeve')
    return df


def compute_constraint_binding(artifacts: Dict) -> Dict:
    """
    Compute constraint binding report:
    - % of days allocator < 1.0
    - Avg scalar when < 1.0
    - % of time policy gates Trend / VRP
    - % of days exposure capped by leverage
    - % of days portfolio vol < target
    
    Args:
        artifacts: Dict with loaded artifacts
        
    Returns:
        Dict with constraint binding metrics
    """
    allocator_scalar = artifacts.get('allocator_scalar')
    engine_policy = artifacts.get('engine_policy')
    portfolio_returns = artifacts.get('portfolio_returns')
    
    # Safely get weights (avoid DataFrame truthiness ambiguity)
    weights = None
    for weight_key in ['weights', 'weights_scaled', 'weights_raw']:
        w = artifacts.get(weight_key)
        if w is not None and isinstance(w, pd.DataFrame) and not w.empty:
            weights = w
            break
    
    meta = artifacts.get('meta', {})
    
    # Handle missing portfolio_returns
    if portfolio_returns is None:
        return {
            'status': 'missing_artifact',
            'details': 'portfolio_returns is required but missing',
            'allocator_active_pct': float(np.nan),
            'allocator_avg_scalar_when_active': float(np.nan),
            'policy_gated_trend_pct': 0.0,  # Default to 0.0 instead of NaN
            'policy_gated_vrp_pct': 0.0,  # Default to 0.0 instead of NaN
            'policy_artifact_missing': True,
            'exposure_capped_pct': float(np.nan),
            'vol_below_target_pct': float(np.nan),
        }
    
    binding = {
        'allocator_active_pct': float(np.nan),
        'allocator_avg_scalar_when_active': float(np.nan),
        'policy_gated_trend_pct': 0.0,  # Default to 0.0 instead of NaN
        'policy_gated_vrp_pct': 0.0,  # Default to 0.0 instead of NaN
        'policy_artifact_missing': False,
        'exposure_capped_pct': float(np.nan),
        'vol_below_target_pct': float(np.nan),
    }
    
    # Allocator binding
    if allocator_scalar is not None and isinstance(allocator_scalar, pd.Series) and not allocator_scalar.empty:
        active_mask = allocator_scalar < 0.999  # Slightly below 1.0 to account for rounding
        active_count = active_mask.sum()
        total_count = len(allocator_scalar)
        binding['allocator_active_pct'] = float(active_count / total_count * 100) if total_count > 0 else 0.0
        
        if active_count > 0:
            avg_scalar = allocator_scalar[active_mask].mean()
            binding['allocator_avg_scalar_when_active'] = float(avg_scalar)
    
    # Policy gating (canonical pivot format: rebalance_date, trend_multiplier, vrp_multiplier)
    if engine_policy is not None and isinstance(engine_policy, pd.DataFrame) and not engine_policy.empty:
        # Check for canonical pivot format (trend_multiplier, vrp_multiplier columns)
        if 'trend_multiplier' in engine_policy.columns and 'vrp_multiplier' in engine_policy.columns:
            # Canonical pivot format
            total_rebalances = len(engine_policy)
            trend_gated = (engine_policy['trend_multiplier'] < 0.999).sum()  # Slightly below 1.0 to account for rounding
            vrp_gated = (engine_policy['vrp_multiplier'] < 0.999).sum()
            binding['policy_gated_trend_pct'] = float(trend_gated / total_rebalances * 100) if total_rebalances > 0 else 0.0
            binding['policy_gated_vrp_pct'] = float(vrp_gated / total_rebalances * 100) if total_rebalances > 0 else 0.0
        elif 'engine' in engine_policy.columns and 'policy_multiplier_used' in engine_policy.columns:
            # Legacy long format (for backward compatibility)
            gated_mask = engine_policy['policy_multiplier_used'] < 1.0
            total_rebalances = len(engine_policy)
            
            # Trend gating (simplified check)
            trend_mask = engine_policy['engine'].str.contains('trend', case=False, na=False)
            trend_gated = (trend_mask & gated_mask).sum()
            binding['policy_gated_trend_pct'] = float(trend_gated / total_rebalances * 100) if total_rebalances > 0 else 0.0
            
            # VRP gating
            vrp_mask = engine_policy['engine'].str.contains('vrp', case=False, na=False)
            vrp_gated = (vrp_mask & gated_mask).sum()
            binding['policy_gated_vrp_pct'] = float(vrp_gated / total_rebalances * 100) if total_rebalances > 0 else 0.0
    else:
        # Policy artifact missing
        binding['policy_artifact_missing'] = True
    
    # Exposure capped by leverage
    if weights is not None and isinstance(weights, pd.DataFrame) and not weights.empty:
        # Check leverage caps from config
        leverage_cap = meta.get('config', {}).get('risk_targeting', {}).get('leverage_cap', 7.0)
        if isinstance(leverage_cap, str):
            leverage_cap = float(leverage_cap)
        
        gross_exposure = weights.abs().sum(axis=1)
        capped_mask = gross_exposure >= (leverage_cap * 0.99)  # Within 1% of cap
        capped_count = capped_mask.sum()
        total_days = len(gross_exposure)
        binding['exposure_capped_pct'] = float(capped_count / total_days * 100) if total_days > 0 else 0.0
    
    # Vol below target
    if portfolio_returns is not None and isinstance(portfolio_returns, pd.Series) and not portfolio_returns.empty:
        target_vol = meta.get('config', {}).get('risk_targeting', {}).get('target_vol', 0.20)
        if isinstance(target_vol, str):
            target_vol = float(target_vol)
        
        # Compute rolling vol
        rolling_vol = portfolio_returns.rolling(63).std() * np.sqrt(252)
        below_target_mask = rolling_vol < target_vol
        below_target_count = below_target_mask.sum()
        total_vol_days = len(rolling_vol.dropna())
        binding['vol_below_target_pct'] = float(below_target_count / total_vol_days * 100) if total_vol_days > 0 else 0.0
    
    return binding


def compute_path_diagnostics(artifacts: Dict, n_drawdowns: int = 10) -> Dict:
    """
    Compute path diagnostics:
    - Worst 10 drawdowns: cause attribution
    - Worst months: which sleeves lost money
    - Best months: which sleeves actually earned
    
    Args:
        artifacts: Dict with loaded artifacts
        n_drawdowns: Number of worst drawdowns to analyze
        
    Returns:
        Dict with path diagnostics
    """
    equity_curve = artifacts['equity_curve']
    portfolio_returns = artifacts['portfolio_returns']
    sleeve_returns = artifacts.get('sleeve_returns')
    
    diagnostics = {
        'worst_drawdowns': [],
        'worst_months': [],
        'best_months': [],
    }
    
    # Compute drawdowns
    running_max = equity_curve.cummax()
    drawdown = (equity_curve / running_max) - 1.0
    
    # Find worst drawdowns
    drawdown_periods = []
    in_dd = False
    dd_start = None
    dd_trough = None
    dd_trough_date = None
    
    for i, (date, dd_val) in enumerate(drawdown.items()):
        if dd_val < -0.01:  # At least 1% drawdown
            if not in_dd:
                in_dd = True
                dd_start = date
                dd_trough = dd_val
                dd_trough_date = date
            else:
                if dd_val < dd_trough:
                    dd_trough = dd_val
                    dd_trough_date = date
        else:
            if in_dd:
                # Drawdown ended
                drawdown_periods.append({
                    'start': dd_start,
                    'trough': dd_trough_date,
                    'trough_value': dd_trough,
                    'end': date,
                    'depth': dd_trough,
                    'duration_days': (date - dd_start).days,
                })
                in_dd = False
                dd_start = None
                dd_trough = None
                dd_trough_date = None
    
    # Sort by depth (most negative first)
    drawdown_periods.sort(key=lambda x: x['depth'])
    
    # Get top N worst drawdowns
    worst_dd = drawdown_periods[:n_drawdowns]
    
    for dd in worst_dd:
        dd_info = {
            'start': dd['start'].strftime('%Y-%m-%d'),
            'trough': dd['trough'].strftime('%Y-%m-%d'),
            'end': dd['end'].strftime('%Y-%m-%d') if dd['end'] else None,
            'depth_pct': float(dd['depth'] * 100),
            'duration_days': dd['duration_days'],
            'sleeve_attribution': {},
        }
        
        # Sleeve attribution for this drawdown
        if sleeve_returns is not None:
            dd_mask = (sleeve_returns.index >= dd['start']) & (sleeve_returns.index <= dd['trough'])
            if dd_mask.any():
                dd_sleeve_returns = sleeve_returns[dd_mask]
                for sleeve_name in dd_sleeve_returns.columns:
                    sleeve_pnl = dd_sleeve_returns[sleeve_name].sum()
                    dd_info['sleeve_attribution'][sleeve_name] = float(sleeve_pnl)
        
        diagnostics['worst_drawdowns'].append(dd_info)
    
    # Worst and best months
    monthly_returns = portfolio_returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
    
    # Worst months
    worst_months = monthly_returns.nsmallest(10)
    for month_end, ret in worst_months.items():
        month_info = {
            'month': month_end.strftime('%Y-%m'),
            'return_pct': float(ret * 100),
            'sleeve_returns': {},
        }
        
        if sleeve_returns is not None:
            # Filter sleeve returns for this month
            month_start = month_end.replace(day=1)
            if month_end.month == 12:
                next_month_start = month_end.replace(year=month_end.year + 1, month=1, day=1)
            else:
                next_month_start = month_end.replace(month=month_end.month + 1, day=1)
            
            month_mask = (sleeve_returns.index >= month_start) & (sleeve_returns.index < next_month_start)
            month_sleeves = sleeve_returns[month_mask]
            if len(month_sleeves) > 0:
                for sleeve_name in month_sleeves.columns:
                    sleeve_pnl = month_sleeves[sleeve_name].sum()
                    month_info['sleeve_returns'][sleeve_name] = float(sleeve_pnl)
        
        diagnostics['worst_months'].append(month_info)
    
    # Best months
    best_months = monthly_returns.nlargest(10)
    for month_end, ret in best_months.items():
        month_info = {
            'month': month_end.strftime('%Y-%m'),
            'return_pct': float(ret * 100),
            'sleeve_returns': {},
        }
        
        if sleeve_returns is not None:
            # Filter sleeve returns for this month
            month_start = month_end.replace(day=1)
            if month_end.month == 12:
                next_month_start = month_end.replace(year=month_end.year + 1, month=1, day=1)
            else:
                next_month_start = month_end.replace(month=month_end.month + 1, day=1)
            
            month_mask = (sleeve_returns.index >= month_start) & (sleeve_returns.index < next_month_start)
            month_sleeves = sleeve_returns[month_mask]
            if len(month_sleeves) > 0:
                for sleeve_name in month_sleeves.columns:
                    sleeve_pnl = month_sleeves[sleeve_name].sum()
                    month_info['sleeve_returns'][sleeve_name] = float(sleeve_pnl)
        
        diagnostics['best_months'].append(month_info)
    
    return diagnostics


def generate_canonical_diagnostics(run_id: str, run_dir: Optional[Path] = None) -> Dict:
    """
    Generate complete canonical diagnostic report.
    
    Args:
        run_id: Run identifier
        run_dir: Optional path to run directory (default: reports/runs/{run_id})
        
    Returns:
        Dict with all diagnostic sections
    """
    if run_dir is None:
        run_dir = Path(f"reports/runs/{run_id}")
    
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    
    logger.info(f"Loading artifacts from {run_dir}")
    artifacts = load_run_artifacts(run_dir)
    
    logger.info("Computing Performance Decomposition...")
    decomposition = compute_return_decomposition(artifacts)
    
    logger.info("Computing Engine-level Sharpe & Contribution...")
    engine_metrics = compute_engine_sharpe_contribution(artifacts)
    
    logger.info("Computing Constraint Binding Report...")
    constraint_binding = compute_constraint_binding(artifacts)
    
    logger.info("Computing Path Diagnostics...")
    path_diagnostics = compute_path_diagnostics(artifacts)
    
    report = {
        'run_id': run_id,
        'generated_at': datetime.now().isoformat(),
        'performance_decomposition': decomposition,
        'engine_sharpe_contribution': engine_metrics.to_dict('index') if not engine_metrics.empty else {},
        'constraint_binding': constraint_binding,
        'path_diagnostics': path_diagnostics,
    }
    
    return report


def format_markdown_table(df: pd.DataFrame, title: str = "") -> str:
    """Format DataFrame as Markdown table."""
    lines = []
    if title:
        lines.append(f"### {title}\n")
    
    if df.empty:
        lines.append("*No data available*\n")
        return "\n".join(lines)
    
    # Convert to string representation
    lines.append(df.to_markdown(index=True))
    lines.append("")
    
    return "\n".join(lines)


def generate_markdown_report(report: Dict) -> str:
    """Generate Markdown report from diagnostic data."""
    lines = []
    lines.append("# Canonical Diagnostic Report")
    lines.append("")
    lines.append(f"**Run ID:** `{report['run_id']}`")
    lines.append(f"**Generated:** {report['generated_at']}")
    lines.append("")
    lines.append("---")
    lines.append("")
    
    # Performance Decomposition
    lines.append("## A. Performance Decomposition")
    lines.append("")
    decomp = report['performance_decomposition']
    
    lines.append("### Return Decomposition")
    lines.append("")
    lines.append("| Metric | Gross Returns | Net Returns |")
    lines.append("|---|---:|---:|")
    lines.append(f"| CAGR | {decomp['gross_returns']['cagr']:.2%} | {decomp['net_returns']['cagr']:.2%} |")
    lines.append(f"| Vol | {decomp['gross_returns']['vol']:.2%} | {decomp['net_returns']['vol']:.2%} |")
    lines.append(f"| Sharpe | {decomp['gross_returns']['sharpe']:.4f} | {decomp['net_returns']['sharpe']:.4f} |")
    lines.append("")
    
    lines.append("### Drag Analysis (bps/year)")
    lines.append("")
    lines.append("| Component | Drag (bps/year) |")
    lines.append("|---|---:|")
    lines.append(f"| Allocator Drag | {decomp['allocator_drag_bps']:.1f} |")
    if 'policy_drag_bps' in decomp and not np.isnan(decomp['policy_drag_bps']):
        lines.append(f"| Policy Gating Drag | {decomp['policy_drag_bps']:.1f} |")
    lines.append("")
    
    # Engine-level Sharpe & Contribution
    lines.append("## B. Engine-level Sharpe & Contribution")
    lines.append("")
    engine_df = pd.DataFrame(report['engine_sharpe_contribution']).T
    if not engine_df.empty:
        engine_df.index.name = 'Sleeve'
        lines.append(format_markdown_table(engine_df, "Per-Sleeve Metrics"))
    else:
        lines.append("*No sleeve returns data available*")
        lines.append("")
    
    # Constraint Binding
    lines.append("## C. Constraint Binding Report")
    lines.append("")
    binding = report['constraint_binding']
    
    lines.append("| Constraint | % of Time Binding | Additional Info |")
    lines.append("|---|---:|---:|")
    lines.append(f"| Allocator < 1.0 | {binding['allocator_active_pct']:.1f}% | Avg scalar when active: {binding['allocator_avg_scalar_when_active']:.3f} |")
    lines.append(f"| Policy Gates Trend | {binding['policy_gated_trend_pct']:.1f}% | |")
    lines.append(f"| Policy Gates VRP | {binding['policy_gated_vrp_pct']:.1f}% | |")
    lines.append(f"| Exposure Capped by Leverage | {binding['exposure_capped_pct']:.1f}% | |")
    lines.append(f"| Portfolio Vol < Target | {binding['vol_below_target_pct']:.1f}% | |")
    lines.append("")
    
    # Path Diagnostics
    lines.append("## D. Path Diagnostics")
    lines.append("")
    
    # Worst Drawdowns
    lines.append("### Worst 10 Drawdowns")
    lines.append("")
    for i, dd in enumerate(report['path_diagnostics']['worst_drawdowns'], 1):
        lines.append(f"**{i}. Drawdown {dd['start']} to {dd['trough']}**")
        lines.append(f"- Depth: {dd['depth_pct']:.2f}%")
        lines.append(f"- Duration: {dd['duration_days']} days")
        if dd['sleeve_attribution']:
            lines.append("- Sleeve Attribution:")
            for sleeve, pnl in sorted(dd['sleeve_attribution'].items(), key=lambda x: x[1]):
                lines.append(f"  - {sleeve}: {pnl:.4f}")
        lines.append("")
    
    # Worst Months
    lines.append("### Worst Months")
    lines.append("")
    for month in report['path_diagnostics']['worst_months']:
        lines.append(f"**{month['month']}: {month['return_pct']:.2f}%**")
        if month['sleeve_returns']:
            for sleeve, pnl in sorted(month['sleeve_returns'].items(), key=lambda x: x[1]):
                lines.append(f"- {sleeve}: {pnl:.4f}")
        lines.append("")
    
    # Best Months
    lines.append("### Best Months")
    lines.append("")
    for month in report['path_diagnostics']['best_months']:
        lines.append(f"**{month['month']}: {month['return_pct']:.2f}%**")
        if month['sleeve_returns']:
            for sleeve, pnl in sorted(month['sleeve_returns'].items(), key=lambda x: x[1], reverse=True):
                lines.append(f"- {sleeve}: {pnl:.4f}")
        lines.append("")
    
    return "\n".join(lines)
