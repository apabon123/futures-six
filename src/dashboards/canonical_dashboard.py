"""
Canonical Dashboard

Interactive dashboard for human sanity checks - reading artifacts only.

This dashboard answers: "Is this thing behaving like I think it is?"

Views:
0. Run Overview (artifact completeness, run metadata)
1. Equity + Drawdown
2. Exposure by sleeve over time (raw/post-policy/post-allocator)
3. Position-level view (per asset for any date)
4. Allocator state timeline (regime/scalars/drawdown overlay)

Key Principle: Dashboard reads artifacts. It never computes strategy logic.
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging

logger = logging.getLogger(__name__)

try:
    from src.utils.canonical_window import load_canonical_window
except ImportError:
    # Fallback if canonical_window not available
    def load_canonical_window() -> Tuple[str, str]:
        return None, None


def check_artifact_completeness(run_dir: Path) -> Dict[str, Dict[str, bool]]:
    """
    Check artifact completeness for a run directory.
    
    Returns:
        Dict with 'required', 'optional', 'required_diagnostics' keys, each containing artifact name -> exists mapping
    """
    completeness = {
        'required': {},
        'optional': {},
        'required_diagnostics': {}
    }
    
    # Required artifacts (hard gate)
    required_artifacts = {
        'portfolio_returns.csv': 'Portfolio returns',
        'equity_curve.csv': 'Equity curve',
        'weights.csv': 'Weights',
        'meta.json': 'Metadata'
    }
    
    for artifact_file, description in required_artifacts.items():
        completeness['required'][artifact_file] = (run_dir / artifact_file).exists()
    
    # Required diagnostics (for diagnostics views)
    required_diagnostics = {
        'canonical_diagnostics.json': 'Canonical diagnostics',
        'asset_returns.csv': 'Asset returns'
    }
    
    for artifact_file, description in required_diagnostics.items():
        completeness['required_diagnostics'][artifact_file] = (run_dir / artifact_file).exists()
    
    # Optional artifacts (nice to have)
    optional_artifacts = {
        'allocator_state_v1.csv': 'Allocator state',
        'allocator_regime_v1.csv': 'Allocator regime',
        'allocator_risk_v1_applied_used.csv': 'Allocator risk scalars (applied)',
        'allocator_risk_v1_applied.csv': 'Allocator risk scalars',
        'engine_policy_applied_v1.csv': 'Engine policy applied',
        'sleeve_returns.csv': 'Sleeve returns',
    }
    
    for artifact_file, description in optional_artifacts.items():
        completeness['optional'][artifact_file] = (run_dir / artifact_file).exists()
    
    # Check for run_error.json or allocator_state_v1_error.json
    error_file = run_dir / 'run_error.json'
    if not error_file.exists():
        error_file = run_dir / 'allocator_state_v1_error.json'
    completeness['optional']['run_error.json'] = error_file.exists()
    
    return completeness


def detect_run_warnings(artifacts: Dict, completeness: Dict[str, Dict[str, bool]]) -> List[Dict[str, str]]:
    """
    Detect known issues and warnings from artifacts.
    
    Args:
        artifacts: Dict with loaded artifacts
        completeness: Completeness dict from check_artifact_completeness
        
    Returns:
        List of warning dicts with 'severity', 'title', 'message'
    """
    warnings = []
    
    # Error severity warnings
    if artifacts.get('run_error') is not None:
        warnings.append({
            'severity': 'error',
            'title': 'Run Error Detected',
            'message': 'run_error.json exists - this run encountered an error during execution.'
        })
    
    # Check for NaNs in critical data
    portfolio_returns = artifacts.get('portfolio_returns')
    if portfolio_returns is not None:
        nan_count = portfolio_returns.isna().sum()
        if nan_count > 0:
            warnings.append({
                'severity': 'warning',
                'title': 'NaN Values in Portfolio Returns',
                'message': f'{nan_count} NaN values detected in portfolio_returns.csv'
            })
    
    weights = artifacts.get('weights_scaled') or artifacts.get('weights')
    if weights is not None:
        nan_count = weights.isna().sum().sum()
        if nan_count > 0:
            warnings.append({
                'severity': 'warning',
                'title': 'NaN Values in Weights',
                'message': f'{nan_count} NaN values detected in weights data'
            })
    
    # Check for missing sleeve_returns if views depend on it
    if completeness['optional'].get('sleeve_returns.csv', False) == False:
        warnings.append({
            'severity': 'info',
            'title': 'Sleeve Returns Missing',
            'message': 'sleeve_returns.csv not available - Sleeve Concentration Timeline view will be unavailable'
        })
    
    # Check for extreme turnover spikes
    weights = artifacts.get('weights_scaled') or artifacts.get('weights')
    if weights is not None:
        turnover = compute_turnover(weights)
        if not turnover.empty:
            avg_turnover = turnover.mean()
            max_turnover = turnover.max()
            # Warn if max turnover is > 3x average
            if avg_turnover > 0 and max_turnover > 3 * avg_turnover:
                warnings.append({
                    'severity': 'warning',
                    'title': 'Extreme Turnover Spike Detected',
                    'message': f'Max turnover ({max_turnover:.3f}) is {max_turnover/avg_turnover:.1f}x average ({avg_turnover:.3f})'
                })
    
    # Check for leverage cap binding (if allocator scalars available)
    allocator_scalar = artifacts.get('allocator_scalar')
    if allocator_scalar is not None and not allocator_scalar.empty:
        scalar_below_one_pct = (allocator_scalar < 1.0).sum() / len(allocator_scalar) * 100
        if scalar_below_one_pct > 50:
            warnings.append({
                'severity': 'warning',
                'title': 'Leverage Cap Frequently Binding',
                'message': f'Allocator scalar < 1.0 on {scalar_below_one_pct:.1f}% of rebalance dates - leverage cap may be too restrictive'
            })
    
    return warnings


def compute_completeness_score(completeness: Dict[str, Dict[str, bool]]) -> Dict[str, any]:
    """
    Compute completeness score and determine if views should be blocked.
    
    Returns:
        Dict with:
        - 'required_score': "X/4" string
        - 'required_diagnostics_score': "X/2" string
        - 'optional_score': "X/Y" string
        - 'all_required_present': bool
        - 'block_views': bool (True if required artifacts missing)
    """
    required_count = sum(1 for exists in completeness['required'].values() if exists)
    required_total = len(completeness['required'])
    
    required_diagnostics_count = sum(1 for exists in completeness['required_diagnostics'].values() if exists)
    required_diagnostics_total = len(completeness['required_diagnostics'])
    
    optional_count = sum(1 for exists in completeness['optional'].values() if exists)
    optional_total = len(completeness['optional'])
    
    all_required_present = required_count == required_total
    block_views = not all_required_present
    
    return {
        'required_score': f"{required_count}/{required_total}",
        'required_diagnostics_score': f"{required_diagnostics_count}/{required_diagnostics_total}",
        'optional_score': f"{optional_count}/{optional_total}",
        'all_required_present': all_required_present,
        'block_views': block_views
    }


@st.cache_data(ttl=3600)
def load_run_artifacts_cached(_run_dir_str: str) -> Dict:
    """
    Cached version of load_run_artifacts for performance.
    
    Args:
        _run_dir_str: String path to run directory (must be hashable for cache)
        
    Returns:
        Dict with loaded artifacts
    """
    run_dir = Path(_run_dir_str)
    return load_run_artifacts(run_dir)


def load_run_artifacts(run_dir: Path) -> Dict:
    """
    Load all artifacts from a run directory.
    
    Args:
        run_dir: Path to run directory
        
    Returns:
        Dict with loaded artifacts
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
    
    # Optional artifacts
    asset_returns_file = run_dir / "asset_returns.csv"
    if asset_returns_file.exists():
        artifacts['asset_returns'] = pd.read_csv(asset_returns_file, index_col=0, parse_dates=True)
    else:
        artifacts['asset_returns'] = None
    
    weights_file = run_dir / "weights.csv"
    if weights_file.exists():
        artifacts['weights'] = pd.read_csv(weights_file, index_col=0, parse_dates=True)
    else:
        artifacts['weights'] = None
    
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
    
    allocator_regime_file = run_dir / "allocator_regime_v1.csv"
    if allocator_regime_file.exists():
        df = pd.read_csv(allocator_regime_file, parse_dates=['date'], index_col='date')
        artifacts['allocator_regime'] = df['regime']
    else:
        artifacts['allocator_regime'] = None
    
    # Load allocator regime meta (for transition counts and stats)
    allocator_regime_meta_file = run_dir / "allocator_regime_v1_meta.json"
    if allocator_regime_meta_file.exists():
        with open(allocator_regime_meta_file, 'r') as f:
            artifacts['allocator_regime_meta'] = json.load(f)
    else:
        artifacts['allocator_regime_meta'] = None
    
    allocator_scalar_file = run_dir / "allocator_risk_v1_applied_used.csv"
    if not allocator_scalar_file.exists():
        allocator_scalar_file = run_dir / "allocator_risk_v1_applied.csv"
    if allocator_scalar_file.exists():
        df = pd.read_csv(allocator_scalar_file, parse_dates=['rebalance_date'], index_col='rebalance_date')
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
    
    # Load additional optional artifacts for completeness checking
    allocator_state_file = run_dir / "allocator_state_v1.csv"
    if allocator_state_file.exists():
        artifacts['allocator_state'] = pd.read_csv(allocator_state_file, parse_dates=['date'], index_col='date')
    else:
        artifacts['allocator_state'] = None
    
    canonical_diagnostics_file = run_dir / "canonical_diagnostics.json"
    if canonical_diagnostics_file.exists():
        with open(canonical_diagnostics_file, 'r') as f:
            artifacts['canonical_diagnostics'] = json.load(f)
    else:
        artifacts['canonical_diagnostics'] = None
    
    error_file = run_dir / "run_error.json"
    if not error_file.exists():
        error_file = run_dir / "allocator_state_v1_error.json"
    if error_file.exists():
        with open(error_file, 'r') as f:
            artifacts['run_error'] = json.load(f)
    else:
        artifacts['run_error'] = None
    
    return artifacts


def downsample_series(series: pd.Series, max_points: int = 1000) -> pd.Series:
    """
    Downsample a time series for faster plotting.
    
    Args:
        series: Time series to downsample
        max_points: Maximum number of points to keep
        
    Returns:
        Downsampled series
    """
    if len(series) <= max_points:
        return series
    
    # Use every Nth point
    step = len(series) // max_points
    return series.iloc[::step]


def get_top_asset_contributors(artifacts: Dict, days: int = 30) -> Optional[pd.DataFrame]:
    """
    Get top N assets by PnL contribution over last N days.
    
    Args:
        artifacts: Dict with loaded artifacts
        days: Number of days to look back
        
    Returns:
        DataFrame with top contributors or None
    """
    weights = artifacts.get('weights_scaled') or artifacts.get('weights')
    asset_returns = artifacts.get('asset_returns')
    
    if weights is None or asset_returns is None:
        return None
    
    # Get last N days
    if len(asset_returns) < days:
        days = len(asset_returns)
    
    recent_returns = asset_returns.iloc[-days:]
    
    # Forward-fill weights to daily frequency
    weights_daily = weights.reindex(asset_returns.index).ffill().fillna(0.0)
    recent_weights = weights_daily.iloc[-days:]
    
    # Compute PnL contribution
    pnl_contrib = (recent_weights * recent_returns).sum()
    
    # Sort by absolute contribution
    pnl_abs = pnl_contrib.abs().sort_values(ascending=False)
    
    # Create DataFrame
    top_df = pd.DataFrame({
        'Asset': pnl_abs.index,
        'PnL Contribution': pnl_contrib.loc[pnl_abs.index],
        'Abs PnL Contribution': pnl_abs.values
    }).head(10)
    
    return top_df


def get_top_sleeve_contributors(artifacts: Dict, days: int = 60) -> Optional[pd.DataFrame]:
    """
    Get top N sleeves by PnL contribution over last N days.
    
    Args:
        artifacts: Dict with loaded artifacts
        days: Number of days to look back
        
    Returns:
        DataFrame with top sleeve contributors or None
    """
    sleeve_returns = artifacts.get('sleeve_returns')
    
    if sleeve_returns is None or sleeve_returns.empty:
        return None
    
    # Get last N days
    if len(sleeve_returns) < days:
        days = len(sleeve_returns)
    
    recent_sleeve_returns = sleeve_returns.iloc[-days:]
    
    # Compute total PnL contribution per sleeve
    sleeve_pnl = recent_sleeve_returns.sum()
    
    # Sort by absolute contribution
    pnl_abs = sleeve_pnl.abs().sort_values(ascending=False)
    
    # Create DataFrame
    top_df = pd.DataFrame({
        'Sleeve': pnl_abs.index,
        'PnL Contribution': sleeve_pnl.loc[pnl_abs.index],
        'Abs PnL Contribution': pnl_abs.values
    }).head(5)
    
    return top_df


def get_top_turnover_events(artifacts: Dict, top_n: int = 10) -> Optional[pd.DataFrame]:
    """
    Get top N turnover events (rebalance dates with highest turnover).
    
    Args:
        artifacts: Dict with loaded artifacts
        top_n: Number of top events to return
        
    Returns:
        DataFrame with top turnover events or None
    """
    weights = artifacts.get('weights_scaled') or artifacts.get('weights')
    
    if weights is None or weights.empty:
        return None
    
    turnover = compute_turnover(weights)
    
    if turnover.empty:
        return None
    
    # Sort by turnover (descending)
    top_turnover = turnover.sort_values(ascending=False).head(top_n)
    
    # Create DataFrame
    top_df = pd.DataFrame({
        'Date': top_turnover.index,
        'Turnover': top_turnover.values
    })
    
    return top_df


def plot_equity_drawdown(equity_curve: pd.Series, downsample: bool = False) -> go.Figure:
    """
    Plot equity curve and drawdown.
    
    Args:
        equity_curve: Equity curve Series
        downsample: Whether to downsample for faster rendering
        
    Returns:
        Plotly figure
    """
    # Downsample if requested
    if downsample and len(equity_curve) > 1000:
        equity_curve = downsample_series(equity_curve, max_points=1000)
    
    running_max = equity_curve.cummax()
    drawdown = (equity_curve / running_max - 1.0) * 100
    
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        row_heights=[0.7, 0.3],
        subplot_titles=('Equity Curve', 'Drawdown (%)')
    )
    
    # Equity curve
    fig.add_trace(
        go.Scatter(
            x=equity_curve.index,
            y=equity_curve.values,
            mode='lines',
            name='Equity',
            line=dict(color='blue', width=1.5)
        ),
        row=1, col=1
    )
    
    # Drawdown
    fig.add_trace(
        go.Scatter(
            x=drawdown.index,
            y=drawdown.values,
            mode='lines',
            name='Drawdown',
            fill='tozeroy',
            line=dict(color='red', width=1),
            fillcolor='rgba(255, 0, 0, 0.2)'
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        height=600,
        showlegend=True,
        hovermode='x unified'
    )
    
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Equity", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
    
    return fig


def compute_exposure_by_sleeve(artifacts: Dict) -> Optional[pd.DataFrame]:
    """
    Compute exposure over time (pre-allocator, post-allocator).
    
    Computes gross exposure (sum of abs weights) for pre-allocator and post-allocator weights.
    Note: weights_raw is post-policy but pre-allocator (policy gates affect signals upstream).
    For policy gating visualization, we show policy gating markers separately.
    
    Args:
        artifacts: Dict with loaded artifacts
        
    Returns:
        DataFrame with columns: pre_allocator_exposure, post_allocator_exposure, indexed by date
        Also returns policy_gating_info if engine_policy is available
    """
    weights_raw = artifacts.get('weights_raw')
    weights_scaled = artifacts.get('weights_scaled') or artifacts.get('weights')
    engine_policy = artifacts.get('engine_policy')
    
    if weights_raw is None:
        return None
    
    # Compute gross exposure (sum of abs weights)
    exposure_raw = weights_raw.abs().sum(axis=1)
    
    # Post-allocator exposure
    if weights_scaled is not None:
        exposure_post_allocator = weights_scaled.abs().sum(axis=1)
        # Align to common dates
        common_dates = exposure_raw.index.intersection(exposure_post_allocator.index)
        exposure_post_allocator = exposure_post_allocator.loc[common_dates]
    else:
        exposure_post_allocator = exposure_raw
        common_dates = exposure_raw.index
    
    exposure_raw = exposure_raw.loc[common_dates]
    
    result = pd.DataFrame({
        'pre_allocator_exposure': exposure_raw,  # Pre-allocator (post-policy, before allocator scaling)
        'post_allocator_exposure': exposure_post_allocator  # Post-allocator (final exposure after allocator scaling)
    })
    
    # Get policy gating info (which engines gated at each date)
    if engine_policy is not None and not engine_policy.empty:
        policy_gating_info = {}  # {rebal_date: {engines: [list], both: bool}}
        for rebal_date in engine_policy['rebalance_date'].unique():
            rebal_policy = engine_policy[engine_policy['rebalance_date'] == rebal_date]
            gated_engines = rebal_policy[rebal_policy['policy_multiplier_used'] == 0]['engine'].tolist()
            if gated_engines:
                policy_gating_info[rebal_date] = {
                    'engines': gated_engines,
                    'both': len(gated_engines) > 1
                }
        result.attrs['policy_gating_info'] = policy_gating_info
    
    return result


def plot_exposure_by_sleeve(artifacts: Dict, downsample: bool = False) -> Optional[go.Figure]:
    """
    Plot exposure by sleeve over time.
    
    Shows: pre-allocator exposure (post-policy, before allocator scaling),
    post-allocator exposure (final), and policy gating markers.
    
    Args:
        artifacts: Dict with loaded artifacts
        downsample: Whether to downsample for faster rendering
        
    Returns:
        Plotly figure or None if data unavailable
    """
    exposure_df = compute_exposure_by_sleeve(artifacts)
    
    if exposure_df is None or exposure_df.empty:
        return None
    
    fig = go.Figure()
    
    # Pre-allocator exposure (from weights_raw, which is post-policy but pre-allocator scaling)
    fig.add_trace(go.Scatter(
        x=exposure_df.index,
        y=exposure_df['pre_allocator_exposure'],
        mode='lines',
        name='Pre-Allocator Exposure',
        line=dict(color='blue', width=1.5),
        fill='tozeroy',
        fillcolor='rgba(0, 0, 255, 0.15)',
        hovertemplate='Pre-Allocator: %{y:.2f}x<extra></extra>'
    ))
    
    # Post-allocator exposure (from weights_scaled, which is post-allocator scaling)
    fig.add_trace(go.Scatter(
        x=exposure_df.index,
        y=exposure_df['post_allocator_exposure'],
        mode='lines',
        name='Post-Allocator Exposure',
        line=dict(color='red', width=1.5),
        fill='tozeroy',
        fillcolor='rgba(255, 0, 0, 0.2)',
        hovertemplate='Post-Allocator: %{y:.2f}x<extra></extra>'
    ))
    
    # Policy gating markers (vertical lines with engine-specific colors)
    engine_policy = artifacts.get('engine_policy')
    if hasattr(exposure_df, 'attrs') and 'policy_gating_info' in exposure_df.attrs:
        policy_gating_info = exposure_df.attrs['policy_gating_info']
        if policy_gating_info:
            max_exposure = max(exposure_df['pre_allocator_exposure'].max(), exposure_df['post_allocator_exposure'].max())
            
            # Track which engines gated for legend
            trend_gates = []
            vrp_gates = []
            both_gates = []
            
            for gate_date, info in policy_gating_info.items():
                engines = info['engines']
                engines_lower = [e.lower() for e in engines]
                gate_text = ", ".join(engines).upper()
                
                # Determine color based on which engines gated
                has_trend = any('trend' in e for e in engines_lower)
                has_vrp = any('vrp' in e for e in engines_lower)
                
                if has_trend and has_vrp:
                    color = 'red'
                    both_gates.append(gate_date)
                    gate_label = 'Trend + VRP'
                elif has_trend:
                    color = 'orange'
                    trend_gates.append(gate_date)
                    gate_label = 'Trend'
                elif has_vrp:
                    color = 'purple'
                    vrp_gates.append(gate_date)
                    gate_label = 'VRP'
                else:
                    color = 'gray'
                    gate_label = gate_text
                
                fig.add_vline(
                    x=gate_date,
                    line_dash="dash",
                    line_color=color,
                    opacity=0.7,
                    annotation_text=gate_label,
                    annotation_position="top",
                    annotation_font_size=10
                )
            
            # Add legend note if we have gates
            if trend_gates or vrp_gates or both_gates:
                legend_text = []
                if trend_gates:
                    legend_text.append(f"Trend: {len(trend_gates)} gates")
                if vrp_gates:
                    legend_text.append(f"VRP: {len(vrp_gates)} gates")
                if both_gates:
                    legend_text.append(f"Both: {len(both_gates)} gates")
                if legend_text:
                    fig.add_annotation(
                        x=0.02, y=0.98,
                        xref="paper", yref="paper",
                        text="<b>Policy Gates:</b><br>" + "<br>".join(legend_text),
                        showarrow=False,
                        align="left",
                        bgcolor="rgba(255, 255, 255, 0.8)",
                        bordercolor="gray",
                        borderwidth=1,
                        font=dict(size=10)
                    )
    
    fig.update_layout(
        title='Exposure Over Time',
        xaxis_title='Date',
        yaxis_title='Gross Exposure',
        height=400,
        hovermode='x unified',
        showlegend=True,
        annotations=[
            dict(
                x=0.02, y=0.02,
                xref="paper", yref="paper",
                text="<b>Note:</b> Policy gates affect signals before allocator. 'Pre-Allocator' exposure is post-policy.",
                showarrow=False,
                align="left",
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor="gray",
                borderwidth=1,
                font=dict(size=9)
            )
        ]
    )
    
    return fig


def compute_turnover(weights: pd.DataFrame) -> pd.Series:
    """
    Compute turnover time series from weights DataFrame.
    
    Turnover = sum(abs(weights[t] - weights[t-1])) for each rebalance date.
    
    Args:
        weights: DataFrame with weights (indexed by date, columns = assets)
        
    Returns:
        Series with turnover values (indexed by date)
    """
    if weights is None or weights.empty:
        return pd.Series(dtype=float)
    
    turnover_values = []
    turnover_dates = []
    
    for i, date in enumerate(weights.index):
        if i == 0:
            # First date: turnover = gross exposure
            turnover = weights.loc[date].abs().sum()
        else:
            # Turnover = sum of absolute changes
            prev_weights = weights.iloc[i-1]
            curr_weights = weights.loc[date]
            
            # Align indices
            all_symbols = prev_weights.index.union(curr_weights.index)
            prev_aligned = prev_weights.reindex(all_symbols, fill_value=0)
            curr_aligned = curr_weights.reindex(all_symbols, fill_value=0)
            
            turnover = (curr_aligned - prev_aligned).abs().sum()
        
        turnover_values.append(turnover)
        turnover_dates.append(date)
    
    return pd.Series(turnover_values, index=pd.DatetimeIndex(turnover_dates))


def get_position_view(artifacts: Dict, date: pd.Timestamp) -> Optional[Dict]:
    """
    Get position-level view for a specific date.
    
    Returns holdings snapshot and PnL contribution over last N days.
    
    Args:
        artifacts: Dict with loaded artifacts
        date: Date to view
        
    Returns:
        Dict with 'holdings' (DataFrame) and 'pnl_recent' (DataFrame)
    """
    weights_raw = artifacts.get('weights_raw')
    weights_scaled = artifacts.get('weights_scaled') or artifacts.get('weights')
    asset_returns = artifacts.get('asset_returns')
    
    weights = weights_scaled or weights or weights_raw
    
    if weights is None:
        return None
    
    # Find closest rebalance date
    rebalance_dates = weights.index
    if date not in rebalance_dates:
        closest_idx = rebalance_dates.get_indexer([date], method='nearest')[0]
        if closest_idx >= 0:
            date = rebalance_dates[closest_idx]
        else:
            return None
    
    if date not in weights.index:
        return None
    
    # Get weights for this date
    positions_scaled = (weights_scaled or weights).loc[date]  # Post-allocator
    positions_pre_allocator = weights_raw.loc[date] if weights_raw is not None else positions_scaled  # Pre-allocator (post-policy)
    
    # Compute exposure (abs of position)
    exposure = positions_scaled.abs()
    
    # Get returns for this date (if available)
    pnl_contribution = pd.Series(0.0, index=positions_scaled.index)
    if asset_returns is not None and date in asset_returns.index:
        daily_returns = asset_returns.loc[date]
        pnl_contribution = positions_scaled * daily_returns
    
    # Holdings snapshot
    holdings = pd.DataFrame({
        'weight_pre_allocator': positions_pre_allocator,  # Pre-allocator (post-policy, before allocator scaling)
        'weight_post_allocator': positions_scaled,  # Post-allocator (final weights after allocator scaling)
        'position_direction': ['Long' if w > 0 else ('Short' if w < 0 else 'Flat') for w in positions_scaled],
        'exposure': exposure,
        'pnl_contribution': pnl_contribution
    })
    
    # Sort by absolute exposure
    holdings = holdings.sort_values('exposure', ascending=False)
    
    # PnL contribution over last N days (if asset_returns available)
    pnl_recent = None
    if asset_returns is not None:
        # Get last 30 days of returns
        date_idx = asset_returns.index.get_indexer([date], method='nearest')[0]
        if date_idx >= 0:
            start_idx = max(0, date_idx - 29)  # Last 30 days
            recent_returns = asset_returns.iloc[start_idx:date_idx+1]
            recent_positions = positions_scaled
            
            # Compute cumulative PnL for each asset
            pnl_cumulative = (recent_returns * recent_positions).sum()
            pnl_recent = pd.DataFrame({
                'asset': pnl_cumulative.index,
                'pnl_contribution': pnl_cumulative.values
            }).sort_values('pnl_contribution', ascending=False)
    
    return {
        'holdings': holdings,
        'pnl_recent': pnl_recent
    }


def plot_allocator_timeline(artifacts: Dict, downsample: bool = False) -> Optional[go.Figure]:
    """
    Plot allocator state timeline with regime labels, scalars, drawdown overlay,
    scalar histogram, and regime statistics.
    
    Args:
        artifacts: Dict with loaded artifacts
        
    Returns:
        Plotly figure or None if data unavailable
    """
    allocator_regime = artifacts.get('allocator_regime')
    allocator_scalar = artifacts.get('allocator_scalar')
    allocator_regime_meta = artifacts.get('allocator_regime_meta')
    equity_curve = artifacts.get('equity_curve')
    
    if allocator_regime is None and allocator_scalar is None:
        return None
    
    # Create subplots: timeline (3 rows) + histogram (1 row) = 4 rows
    fig = make_subplots(
        rows=4, cols=2,
        shared_xaxes=True,
        vertical_spacing=0.08,
        horizontal_spacing=0.1,
        row_heights=[0.3, 0.25, 0.25, 0.2],
        column_widths=[0.7, 0.3],
        subplot_titles=(
            'Allocator Regime', 'Scalar Histogram',
            'Risk Scalar', 'Regime Stats',
            'Drawdown Overlay', '',
            '', ''
        )
    )
    
    # Regime timeline (row 1, col 1)
    if allocator_regime is not None:
        regime_colors = {
            'NORMAL': 'green',
            'ELEVATED': 'yellow',
            'STRESS': 'orange',
            'CRISIS': 'red'
        }
        
        for regime in ['NORMAL', 'ELEVATED', 'STRESS', 'CRISIS']:
            mask = allocator_regime == regime
            if mask.any():
                fig.add_trace(
                    go.Scatter(
                        x=allocator_regime[mask].index,
                        y=[regime] * mask.sum(),
                        mode='markers',
                        name=regime,
                        marker=dict(color=regime_colors.get(regime, 'gray'), size=8),
                        showlegend=False
                    ),
                    row=1, col=1
                )
    
    # Risk scalar (row 2, col 1)
    if allocator_scalar is not None:
        fig.add_trace(
            go.Scatter(
                x=allocator_scalar.index,
                y=allocator_scalar.values,
                mode='lines',
                name='Risk Scalar',
                line=dict(color='blue', width=1.5),
                showlegend=False
            ),
            row=2, col=1
        )
        fig.add_hline(y=1.0, line_dash="dash", line_color="gray", row=2, col=1)
    
    # Drawdown overlay (row 3, col 1) - downsample if requested
    if equity_curve is not None:
        equity_plot = equity_curve
        if downsample and len(equity_curve) > 1000:
            equity_plot = downsample_series(equity_curve, max_points=1000)
        
        running_max = equity_plot.cummax()
        drawdown = (equity_plot / running_max - 1.0) * 100
        
        fig.add_trace(
            go.Scatter(
                x=drawdown.index,
                y=drawdown.values,
                mode='lines',
                name='Drawdown',
                line=dict(color='red', width=1),
                fill='tozeroy',
                fillcolor='rgba(255, 0, 0, 0.2)',
                showlegend=False
            ),
            row=3, col=1
        )
    
    # Scalar histogram (row 1, col 2)
    if allocator_scalar is not None:
        fig.add_trace(
            go.Histogram(
                x=allocator_scalar.values,
                nbinsx=30,
                name='Scalar Distribution',
                showlegend=False,
                marker_color='blue',
                opacity=0.6
            ),
            row=1, col=2
        )
    
    # Regime stats (row 2, col 2) - text annotations
    if allocator_regime_meta is not None:
        regime_stats = []
        regime_day_counts = allocator_regime_meta.get('regime_day_counts', {})
        regime_percentages = allocator_regime_meta.get('regime_percentages', {})
        transition_counts = allocator_regime_meta.get('transition_counts', {})
        max_consecutive = allocator_regime_meta.get('max_consecutive_days', {})
        
        # Compute total transitions (excluding same-regime transitions)
        total_transitions = sum(v for k, v in transition_counts.items() if '->' in k and not k.startswith(k.split('->')[0] + '->' + k.split('->')[0]))
        
        regime_stats.append(f"<b>Regime Statistics</b><br>")
        regime_stats.append(f"Total Transitions: {total_transitions}<br>")
        for regime in ['NORMAL', 'ELEVATED', 'STRESS', 'CRISIS']:
            if regime in regime_day_counts:
                days = regime_day_counts[regime]
                pct = regime_percentages.get(regime, 0)
                max_cons = max_consecutive.get(regime, 0)
                regime_stats.append(f"{regime}: {pct:.1f}% ({days}d, max {max_cons}d)<br>")
        
        # Add as text annotation
        fig.add_annotation(
            x=0.5, y=0.5,
            xref='x2 domain', yref='y2 domain',
            text=''.join(regime_stats),
            showarrow=False,
            align='left',
            font=dict(size=10),
            row=2, col=2
        )
    
    # Update axes
    fig.update_xaxes(title_text="Date", row=3, col=1)
    fig.update_yaxes(title_text="Regime", row=1, col=1)
    fig.update_yaxes(title_text="Scalar", row=2, col=1)
    fig.update_yaxes(title_text="Drawdown (%)", row=3, col=1)
    fig.update_xaxes(title_text="Scalar Value", row=1, col=2)
    fig.update_yaxes(title_text="Frequency", row=1, col=2)
    
    # Hide empty subplots
    fig.update_xaxes(showticklabels=False, showgrid=False, row=2, col=2)
    fig.update_yaxes(showticklabels=False, showgrid=False, row=2, col=2)
    fig.update_xaxes(showticklabels=False, showgrid=False, row=4, col=1)
    fig.update_yaxes(showticklabels=False, showgrid=False, row=4, col=1)
    fig.update_xaxes(showticklabels=False, showgrid=False, row=4, col=2)
    fig.update_yaxes(showticklabels=False, showgrid=False, row=4, col=2)
    
    fig.update_layout(
        height=900,
        showlegend=False,
        hovermode='x unified'
    )
    
    return fig


def plot_drag_waterfall(artifacts: Dict) -> Optional[go.Figure]:
    """
    Plot drag waterfall showing gross CAGR -> policy drag -> allocator drag -> net CAGR.
    
    Args:
        artifacts: Dict with loaded artifacts
        
    Returns:
        Plotly waterfall figure or None if data unavailable
    """
    canonical_diagnostics = artifacts.get('canonical_diagnostics')
    if canonical_diagnostics is None:
        return None
    
    decomp = canonical_diagnostics.get('performance_decomposition', {})
    gross_cagr = decomp.get('gross_returns', {}).get('cagr', np.nan)
    net_cagr = decomp.get('net_returns', {}).get('cagr', np.nan)
    allocator_drag_bps = decomp.get('allocator_drag_bps', np.nan)
    policy_drag_bps = decomp.get('policy_drag_bps', np.nan)
    
    if np.isnan(net_cagr):
        return None
    
    # Convert to percentages for display
    gross_cagr_pct = gross_cagr * 100 if not np.isnan(gross_cagr) else 0.0
    allocator_drag_pct = allocator_drag_bps / 100.0 if not np.isnan(allocator_drag_bps) else 0.0
    policy_drag_pct = policy_drag_bps / 100.0 if not np.isnan(policy_drag_bps) else 0.0
    net_cagr_pct = net_cagr * 100
    
    # Compute intermediate values for waterfall
    # Gross -> after policy -> after allocator -> net
    after_policy_cagr = gross_cagr_pct - policy_drag_pct if not np.isnan(gross_cagr) else net_cagr_pct
    after_allocator_cagr = after_policy_cagr - allocator_drag_pct
    
    # Build waterfall data
    measures = []
    x_labels = []
    y_values = []
    text_labels = []
    
    if not np.isnan(gross_cagr):
        # Start with gross
        measures.append("absolute")
        x_labels.append("Gross CAGR")
        y_values.append(gross_cagr_pct)
        text_labels.append(f"{gross_cagr_pct:.2f}%")
        
        # Policy drag
        if not np.isnan(policy_drag_bps) and policy_drag_bps != 0:
            measures.append("relative")
            x_labels.append("Policy Drag")
            y_values.append(-policy_drag_pct)
            text_labels.append(f"-{policy_drag_pct:.2f}%")
        
        # Allocator drag
        if not np.isnan(allocator_drag_bps) and allocator_drag_bps != 0:
            measures.append("relative")
            x_labels.append("Allocator Drag")
            y_values.append(-allocator_drag_pct)
            text_labels.append(f"-{allocator_drag_pct:.2f}%")
    else:
        # Fallback: show net only
        measures.append("absolute")
        x_labels.append("Net CAGR")
        y_values.append(net_cagr_pct)
        text_labels.append(f"{net_cagr_pct:.2f}%")
    
    # Final net value
    measures.append("total")
    x_labels.append("Net CAGR")
    y_values.append(net_cagr_pct)
    text_labels.append(f"{net_cagr_pct:.2f}%")
    
    fig = go.Figure(go.Waterfall(
        name="Drag Waterfall",
        orientation="v",
        measure=measures,
        x=x_labels,
        textposition="outside",
        text=text_labels,
        y=y_values,
        connector={"line": {"color": "rgb(63, 63, 63)"}},
    ))
    
    fig.update_layout(
        title="Drag Waterfall: Gross CAGR → Policy Drag → Allocator Drag → Net CAGR",
        xaxis_title="Component",
        yaxis_title="CAGR (%)",
        height=400,
        showlegend=False
    )
    
    return fig


def compute_rolling_pairwise_correlation(asset_returns: pd.DataFrame, window: int) -> pd.Series:
    """
    Compute rolling average pairwise correlation for asset returns.
    
    Args:
        asset_returns: DataFrame with asset returns (index=date, columns=assets)
        window: Rolling window size (days)
        
    Returns:
        Series of average pairwise correlations (indexed by date)
    """
    if asset_returns is None or asset_returns.empty or len(asset_returns.columns) < 2:
        return pd.Series(dtype=float)
    
    corr_series = pd.Series(index=asset_returns.index, dtype=float)
    
    for i in range(window - 1, len(asset_returns)):
        window_data = asset_returns.iloc[i - window + 1:i + 1]
        
        if len(window_data) < window:
            corr_series.iloc[i] = np.nan
            continue
        
        # Drop columns with insufficient variance
        valid_cols = []
        for col in window_data.columns:
            col_data = window_data[col].dropna()
            if len(col_data) >= 2 and col_data.std() > 1e-10:
                valid_cols.append(col)
        
        if len(valid_cols) < 2:
            corr_series.iloc[i] = np.nan
            continue
        
        returns_valid = window_data[valid_cols]
        corr_matrix = returns_valid.corr()
        
        # Extract off-diagonal elements (upper triangle)
        n = len(corr_matrix)
        if n < 2:
            corr_series.iloc[i] = np.nan
            continue
        
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        off_diag = corr_matrix.where(mask).stack()
        
        if len(off_diag) == 0:
            corr_series.iloc[i] = np.nan
        else:
            corr_series.iloc[i] = off_diag.mean()
    
    return corr_series


def plot_correlation_health(artifacts: Dict, downsample: bool = False) -> Optional[go.Figure]:
    """
    Plot correlation & diversification health: rolling correlation, vol, with drawdown markers.
    
    Args:
        artifacts: Dict with loaded artifacts
        
    Returns:
        Plotly figure or None if data unavailable
    """
    asset_returns = artifacts.get('asset_returns')
    equity_curve = artifacts.get('equity_curve')
    portfolio_returns = artifacts.get('portfolio_returns')
    
    if asset_returns is None or asset_returns.empty:
        return None
    
    # Compute rolling correlations (20d and 60d)
    corr_20d = compute_rolling_pairwise_correlation(asset_returns, window=20)
    corr_60d = compute_rolling_pairwise_correlation(asset_returns, window=60)
    
    # Compute rolling volatility (20d)
    if portfolio_returns is not None and not portfolio_returns.empty:
        vol_20d = portfolio_returns.rolling(window=20).std() * np.sqrt(252) * 100
    else:
        vol_20d = pd.Series(dtype=float)
    
    # Downsample if requested
    if downsample and not hhi_series.empty and len(hhi_series) > 1000:
        hhi_series = downsample_series(hhi_series, max_points=1000)
    
    # Compute drawdown periods for markers
    drawdown_dates = []
    if equity_curve is not None and not equity_curve.empty:
        # Downsample equity for drawdown computation if needed (but keep full series for markers)
        equity_plot = equity_curve
        if downsample and len(equity_curve) > 1000:
            equity_plot = downsample_series(equity_curve, max_points=1000)
        
        running_max = equity_plot.cummax()
        drawdown = (equity_plot / running_max - 1.0)
        # Find dates where drawdown < -0.10 (10% drawdown)
        drawdown_mask = drawdown < -0.10
        if drawdown_mask.any():
            drawdown_dates = drawdown[drawdown_mask].index.tolist()
    
    # Create subplots: correlation (top) and vol (bottom)
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        row_heights=[0.6, 0.4],
        subplot_titles=('Rolling Average Pairwise Correlation', 'Rolling Portfolio Volatility (20d)')
    )
    
    # Downsample if requested
    if downsample:
        if not corr_20d.empty and len(corr_20d) > 1000:
            corr_20d = downsample_series(corr_20d, max_points=1000)
        if not corr_60d.empty and len(corr_60d) > 1000:
            corr_60d = downsample_series(corr_60d, max_points=1000)
        if not vol_20d.empty and len(vol_20d) > 1000:
            vol_20d = downsample_series(vol_20d, max_points=1000)
    
    # Correlation plot (row 1)
    if not corr_20d.empty:
        fig.add_trace(
            go.Scatter(
                x=corr_20d.index,
                y=corr_20d.values,
                mode='lines',
                name='Corr 20d',
                line=dict(color='blue', width=1.5)
            ),
            row=1, col=1
        )
    
    if not corr_60d.empty:
        fig.add_trace(
            go.Scatter(
                x=corr_60d.index,
                y=corr_60d.values,
                mode='lines',
                name='Corr 60d',
                line=dict(color='orange', width=1.5, dash='dash')
            ),
            row=1, col=1
        )
    
    # Vol plot (row 2)
    if not vol_20d.empty:
        fig.add_trace(
            go.Scatter(
                x=vol_20d.index,
                y=vol_20d.values,
                mode='lines',
                name='Vol 20d',
                line=dict(color='green', width=1.5),
                fill='tozeroy',
                fillcolor='rgba(0, 255, 0, 0.1)'
            ),
            row=2, col=1
        )
    
    # Add drawdown markers (vertical lines)
    if drawdown_dates:
        for dd_date in drawdown_dates[:20]:  # Limit to first 20 to avoid clutter
            fig.add_vline(
                x=dd_date,
                line_dash="dot",
                line_color="red",
                opacity=0.3,
                line_width=1,
                row=1, col=1
            )
            fig.add_vline(
                x=dd_date,
                line_dash="dot",
                line_color="red",
                opacity=0.3,
                line_width=1,
                row=2, col=1
            )
    
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Correlation", row=1, col=1)
    fig.update_yaxes(title_text="Volatility (%)", row=2, col=1)
    
    fig.update_layout(
        height=600,
        showlegend=True,
        hovermode='x unified'
    )
    
    return fig


def compute_rolling_sleeve_herfindahl(artifacts: Dict, window: int = 60) -> Optional[pd.Series]:
    """
    Compute rolling Herfindahl index of sleeve concentration over time.
    
    Uses sleeve exposure if sleeve returns are not available.
    Preference: sleeve returns (PnL contribution) > sleeve exposure.
    
    Args:
        artifacts: Dict with loaded artifacts
        window: Rolling window size (days)
        
    Returns:
        Series of Herfindahl index values (indexed by date) or None
    """
    sleeve_returns = artifacts.get('sleeve_returns')
    weights_scaled = artifacts.get('weights_scaled') or artifacts.get('weights')
    
    # Prefer sleeve returns for Herfindahl (based on PnL contribution)
    if sleeve_returns is not None and not sleeve_returns.empty:
        # Use the same logic as AllocatorStateV1._compute_sleeve_concentration
        # Compute rolling sum of sleeve returns (PnL contributions)
        sleeve_pnl = sleeve_returns.rolling(window).sum()
        
        # Compute absolute contributions
        abs_contrib = sleeve_pnl.abs()
        
        # Compute shares (normalize by total absolute contribution)
        total_abs = abs_contrib.sum(axis=1)
        
        # Avoid division by zero
        shares = abs_contrib.div(total_abs, axis=0).fillna(0.0)
        
        # Herfindahl index: sum of squared shares
        hhi = (shares ** 2).sum(axis=1)
        
        return hhi
    
    # Fallback: use exposure concentration from weights
    # This requires grouping weights by meta-sleeve, which is complex
    # For now, return None if sleeve returns are not available
    # (Future: could group by asset prefix or use meta-sleeve mapping)
    
    return None


def plot_sleeve_concentration_timeline(artifacts: Dict, downsample: bool = False) -> Optional[go.Figure]:
    """
    Plot rolling Herfindahl index of sleeve concentration over time.
    
    Args:
        artifacts: Dict with loaded artifacts
        
    Returns:
        Plotly figure or None if data unavailable
    """
    hhi_series = compute_rolling_sleeve_herfindahl(artifacts, window=60)
    
    if hhi_series is None or hhi_series.empty:
        return None
    
    equity_curve = artifacts.get('equity_curve')
    
    # Create subplots: Herfindahl + Drawdown overlay
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        row_heights=[0.7, 0.3],
        subplot_titles=('Rolling Herfindahl Index (60d)', 'Drawdown Overlay')
    )
    
    # Herfindahl plot (row 1)
    fig.add_trace(
        go.Scatter(
            x=hhi_series.index,
            y=hhi_series.values,
            mode='lines',
            name='Herfindahl Index',
            line=dict(color='purple', width=2),
            fill='tozeroy',
            fillcolor='rgba(128, 0, 128, 0.1)'
        ),
        row=1, col=1
    )
    
    # Add reference lines (1.0 = perfect concentration, 1/N = perfect diversification)
    if not hhi_series.empty:
        max_hhi = hhi_series.max()
        min_hhi = hhi_series.min()
        
        # Perfect diversification line (1/N sleeves - approximate with 1/7 for typical 7 sleeves)
        n_sleeves_approx = 7
        perfect_div_line = 1.0 / n_sleeves_approx
        fig.add_hline(
            y=perfect_div_line,
            line_dash="dash",
            line_color="green",
            opacity=0.5,
            annotation_text=f"Perfect Diversification (1/{n_sleeves_approx})",
            annotation_position="right",
            row=1, col=1
        )
    
    # Drawdown overlay (row 2)
    if equity_curve is not None and not equity_curve.empty:
        running_max = equity_curve.cummax()
        drawdown = (equity_curve / running_max - 1.0) * 100
        
        # Align to common dates
        common_dates = hhi_series.index.intersection(drawdown.index)
        if len(common_dates) > 0:
            drawdown_aligned = drawdown.loc[common_dates]
            
            fig.add_trace(
                go.Scatter(
                    x=drawdown_aligned.index,
                    y=drawdown_aligned.values,
                    mode='lines',
                    name='Drawdown',
                    line=dict(color='red', width=1.5),
                    fill='tozeroy',
                    fillcolor='rgba(255, 0, 0, 0.2)'
                ),
                row=2, col=1
            )
    
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Herfindahl Index", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
    
    fig.update_layout(
        height=600,
        showlegend=False,
        hovermode='x unified'
    )
    
    return fig


def compute_core_metrics_from_artifacts(artifacts: Dict) -> Dict[str, float]:
    """
    Compute core metrics (CAGR, Sharpe, MaxDD, Worst month) from artifacts.
    
    Args:
        artifacts: Dict with loaded artifacts
        
    Returns:
        Dict with metrics: CAGR, Sharpe, MaxDD, WorstMonth
    """
    portfolio_returns = artifacts.get('portfolio_returns')
    equity_curve = artifacts.get('equity_curve')
    
    if portfolio_returns is None or equity_curve is None:
        return {
            'CAGR': np.nan,
            'Sharpe': np.nan,
            'MaxDD': np.nan,
            'WorstMonth': np.nan
        }
    
    # Years
    n_days = len(portfolio_returns)
    years = n_days / 252.0
    
    # CAGR
    if len(equity_curve) >= 2 and years > 0:
        equity_start = equity_curve.iloc[0]
        equity_end = equity_curve.iloc[-1]
        if equity_start > 0:
            cagr = (equity_end / equity_start) ** (1 / years) - 1
        else:
            cagr = np.nan
    else:
        cagr = np.nan
    
    # Sharpe
    if portfolio_returns.std() > 0:
        sharpe = portfolio_returns.mean() / portfolio_returns.std() * np.sqrt(252)
    else:
        sharpe = np.nan
    
    # MaxDD
    if len(equity_curve) >= 2:
        running_max = equity_curve.cummax()
        drawdown = (equity_curve / running_max) - 1.0
        max_dd = drawdown.min()
    else:
        max_dd = np.nan
    
    # Worst month
    try:
        monthly_returns = portfolio_returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
        worst_month = monthly_returns.min() if not monthly_returns.empty else np.nan
    except Exception:
        worst_month = np.nan
    
    return {
        'CAGR': cagr,
        'Sharpe': sharpe,
        'MaxDD': max_dd,
        'WorstMonth': worst_month
    }


def plot_baseline_comparison(artifacts: Dict, baseline_artifacts: Dict) -> Optional[go.Figure]:
    """
    Plot baseline comparison: equity ratio and delta table.
    
    Args:
        artifacts: Dict with loaded artifacts (variant run)
        baseline_artifacts: Dict with loaded artifacts (baseline run)
        
    Returns:
        Plotly figure with subplots or None if data unavailable
    """
    equity_curve = artifacts.get('equity_curve')
    baseline_equity = baseline_artifacts.get('equity_curve')
    
    if equity_curve is None or baseline_equity is None:
        return None
    
    # Compute equity ratio
    common_dates = equity_curve.index.intersection(baseline_equity.index)
    if len(common_dates) == 0:
        return None
    
    equity_variant = equity_curve.loc[common_dates]
    equity_base = baseline_equity.loc[common_dates]
    
    # Normalize both to start at 1.0 for fair comparison
    equity_variant_norm = equity_variant / equity_variant.iloc[0]
    equity_base_norm = equity_base / equity_base.iloc[0]
    equity_ratio = equity_variant_norm / equity_base_norm
    
    # Create figure
    fig = go.Figure()
    
    # Equity ratio plot
    fig.add_trace(go.Scatter(
        x=equity_ratio.index,
        y=equity_ratio.values,
        mode='lines',
        name='Variant / Baseline',
        line=dict(color='blue', width=2)
    ))
    
    # Add horizontal line at 1.0
    fig.add_hline(
        y=1.0,
        line_dash="dash",
        line_color="gray",
        opacity=0.5,
        annotation_text="Baseline",
        annotation_position="right"
    )
    
    fig.update_layout(
        title="Equity Ratio: Variant / Baseline",
        xaxis_title="Date",
        yaxis_title="Ratio",
        height=400,
        showlegend=True,
        hovermode='x unified'
    )
    
    return fig


def get_baseline_comparison_table(artifacts: Dict, baseline_artifacts: Dict) -> Optional[pd.DataFrame]:
    """
    Compute delta table of core metrics between variant and baseline.
    
    Args:
        artifacts: Dict with loaded artifacts (variant run)
        baseline_artifacts: Dict with loaded artifacts (baseline run)
        
    Returns:
        DataFrame with metrics comparison or None
    """
    variant_metrics = compute_core_metrics_from_artifacts(artifacts)
    baseline_metrics = compute_core_metrics_from_artifacts(baseline_artifacts)
    
    # Compute deltas
    comparison = {
        'Metric': ['CAGR', 'Sharpe', 'MaxDD', 'Worst Month'],
        'Variant': [
            variant_metrics.get('CAGR', np.nan),
            variant_metrics.get('Sharpe', np.nan),
            variant_metrics.get('MaxDD', np.nan),
            variant_metrics.get('WorstMonth', np.nan)
        ],
        'Baseline': [
            baseline_metrics.get('CAGR', np.nan),
            baseline_metrics.get('Sharpe', np.nan),
            baseline_metrics.get('MaxDD', np.nan),
            baseline_metrics.get('WorstMonth', np.nan)
        ]
    }
    
    # Compute deltas
    comparison['Delta'] = [
        variant_metrics.get('CAGR', np.nan) - baseline_metrics.get('CAGR', np.nan),
        variant_metrics.get('Sharpe', np.nan) - baseline_metrics.get('Sharpe', np.nan),
        variant_metrics.get('MaxDD', np.nan) - baseline_metrics.get('MaxDD', np.nan),
        variant_metrics.get('WorstMonth', np.nan) - baseline_metrics.get('WorstMonth', np.nan)
    ]
    
    df = pd.DataFrame(comparison)
    return df


def main():
    """Main Streamlit app."""
    st.set_page_config(
        page_title="Canonical Dashboard",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("📊 Canonical Dashboard")
    st.markdown("**Interactive dashboard for human sanity checks**")
    st.markdown("---")
    
    # Sidebar: Run selection
    st.sidebar.header("Configuration")
    
    # Get list of available runs
    runs_dir = Path("reports/runs")
    if not runs_dir.exists():
        st.error(f"Reports directory not found: {runs_dir}")
        st.stop()
    
    available_runs = [d.name for d in runs_dir.iterdir() if d.is_dir()]
    available_runs.sort(reverse=True)  # Most recent first
    
    if not available_runs:
        st.error("No runs found in reports/runs/")
        st.stop()
    
    selected_run = st.sidebar.selectbox(
        "Select Run",
        available_runs,
        index=0
    )
    
    # Optional baseline run for comparison
    baseline_run_options = ["None"] + available_runs
    baseline_run = st.sidebar.selectbox(
        "Baseline Run (optional)",
        baseline_run_options,
        index=0,
        help="Optional baseline run for comparison"
    )
    
    if baseline_run == "None":
        baseline_run = None
    
    run_dir = runs_dir / selected_run
    
    # Load artifacts (with caching)
    try:
        with st.spinner("Loading artifacts..."):
            artifacts = load_run_artifacts_cached(str(run_dir))
        st.sidebar.success(f"✓ Loaded: {selected_run}")
    except Exception as e:
        st.error(f"Error loading artifacts: {e}")
        st.stop()
    
    # Display run info
    meta = artifacts.get('meta', {})
    if meta:
        st.sidebar.markdown("---")
        st.sidebar.markdown("**Run Info**")
        if 'strategy_profile' in meta:
            st.sidebar.text(f"Profile: {meta['strategy_profile']}")
        if 'start_date' in meta:
            st.sidebar.text(f"Start: {meta['start_date']}")
        if 'end_date' in meta:
            st.sidebar.text(f"End: {meta['end_date']}")
    
    # =====================================================================
    # VIEW 0: Run Overview (Artifact Completeness + Run Metadata)
    # =====================================================================
    st.header("0️⃣ Run Overview")
    
    # Artifact completeness
    completeness = check_artifact_completeness(run_dir)
    completeness_score = compute_completeness_score(completeness)
    
    # Run Completeness Score
    st.subheader("Run Completeness Score")
    score_cols = st.columns(4)
    with score_cols[0]:
        st.metric("Required Artifacts", completeness_score['required_score'])
    with score_cols[1]:
        st.metric("Required Diagnostics", completeness_score['required_diagnostics_score'])
    with score_cols[2]:
        st.metric("Optional Artifacts", completeness_score['optional_score'])
    with score_cols[3]:
        if completeness_score['all_required_present']:
            st.metric("Status", "✅ Complete", delta="Ready")
        else:
            st.metric("Status", "❌ Incomplete", delta="Blocked", delta_color="inverse")
    
    # Block views if required artifacts missing
    if completeness_score['block_views']:
        st.error("⚠️ **REQUIRED ARTIFACTS MISSING** - Downstream views are disabled until all required artifacts are present.")
        st.markdown("**Missing artifacts:**")
        for artifact_file, exists in completeness['required'].items():
            if not exists:
                st.markdown(f"- ❌ {artifact_file}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Required Artifacts")
        for artifact_file, exists in completeness['required'].items():
            status = "✅" if exists else "❌"
            st.markdown(f"{status} {artifact_file}")
    
    with col2:
        st.subheader("Optional Artifacts")
        for artifact_file, exists in completeness['optional'].items():
            status = "✅" if exists else "⚪"
            st.markdown(f"{status} {artifact_file}")
        
        st.markdown("---")
        st.subheader("Required Diagnostics")
        for artifact_file, exists in completeness['required_diagnostics'].items():
            status = "✅" if exists else "⚪"
            st.markdown(f"{status} {artifact_file}")
    
    # Run metadata
    st.markdown("---")
    st.subheader("Run Metadata")
    
    metadata_cols = st.columns(4)
    with metadata_cols[0]:
        st.markdown(f"**Run ID:**<br>{meta.get('run_id', 'N/A')}", unsafe_allow_html=True)
    with metadata_cols[1]:
        st.markdown(f"**Strategy Profile:**<br>{meta.get('strategy_profile', 'N/A')}", unsafe_allow_html=True)
    with metadata_cols[2]:
        canonical_window_str = "Yes" if meta.get('canonical_window', False) else "No"
        try:
            canonical_start, canonical_end = load_canonical_window()
            if canonical_start and canonical_end:
                canonical_window_str += f" ({canonical_start} to {canonical_end})"
        except:
            pass
        st.markdown(f"**Canonical Window:**<br>{canonical_window_str}", unsafe_allow_html=True)
    with metadata_cols[3]:
        st.markdown(f"**Config Hash:**<br>{meta.get('config_hash', 'N/A')}", unsafe_allow_html=True)
    
    metadata_cols2 = st.columns(4)
    with metadata_cols2[0]:
        st.markdown(f"**Start Date:**<br>{meta.get('start_date', 'N/A')}", unsafe_allow_html=True)
    with metadata_cols2[1]:
        st.markdown(f"**Effective Start:**<br>{meta.get('effective_start_date', 'N/A')}", unsafe_allow_html=True)
    with metadata_cols2[2]:
        st.markdown(f"**End Date:**<br>{meta.get('end_date', 'N/A')}", unsafe_allow_html=True)
    with metadata_cols2[3]:
        allocator_source = meta.get('allocator_source_run_id', 'N/A')
        engine_policy_source = meta.get('engine_policy_source_run_id', 'N/A')
        st.markdown(f"**Source Runs:**<br>Allocator: {allocator_source}<br>Policy: {engine_policy_source}", unsafe_allow_html=True)
    
    # Known Issues / Warnings Panel
    warnings = detect_run_warnings(artifacts, completeness)
    if warnings:
        st.markdown("---")
        st.subheader("⚠️ Known Issues / Warnings")
        for warning in warnings:
            if warning['severity'] == 'error':
                st.error(f"🔴 **{warning['title']}**: {warning['message']}")
            elif warning['severity'] == 'warning':
                st.warning(f"🟡 **{warning['title']}**: {warning['message']}")
            else:
                st.info(f"ℹ️ **{warning['title']}**: {warning['message']}")
    
    # Run Notes
    st.markdown("---")
    st.subheader("Run Notes")
    run_notes_file = run_dir / "run_notes.md"
    
    # Load existing notes
    existing_notes = ""
    if run_notes_file.exists():
        try:
            with open(run_notes_file, 'r', encoding='utf-8') as f:
                existing_notes = f.read()
        except Exception as e:
            st.warning(f"Could not load existing notes: {e}")
    
    # Initialize session state for this run
    notes_state_key = f"run_notes_{selected_run}"
    notes_saved_key = f"run_notes_saved_{selected_run}"
    
    # Initialize state if this is first time viewing this run
    if notes_state_key not in st.session_state:
        st.session_state[notes_state_key] = existing_notes
        st.session_state[notes_saved_key] = True  # No unsaved changes
    
    # Text area for notes (bound to session state)
    notes = st.text_area(
        "Notes",
        value=st.session_state[notes_state_key],
        height=150,
        help="Add notes about this run. Click 'Save Notes' to save to run_notes.md",
        key=f"notes_textarea_{selected_run}"
    )
    
    # Update session state when notes change
    if notes != st.session_state[notes_state_key]:
        st.session_state[notes_state_key] = notes
        st.session_state[notes_saved_key] = False
    
    # Save button
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("💾 Save Notes", key=f"save_notes_{selected_run}"):
            try:
                with open(run_notes_file, 'w', encoding='utf-8') as f:
                    f.write(notes)
                st.session_state[notes_saved_key] = True
                st.success("✓ Notes saved!")
            except Exception as e:
                st.error(f"Error saving notes: {e}")
    
    with col2:
        if not st.session_state[notes_saved_key]:
            st.info("⚠️ You have unsaved changes. Click 'Save Notes' to save.")
    
    st.markdown("---")
    
    # Block views if required artifacts missing
    if completeness_score['block_views']:
        st.stop()
    
    # Main content
    equity_curve = artifacts.get('equity_curve')
    portfolio_returns = artifacts.get('portfolio_returns')
    
    # Performance settings (for long series)
    with st.sidebar:
        st.markdown("---")
        st.subheader("Performance Settings")
        downsample_plots = st.checkbox("Downsample for plotting", value=False, help="Reduce data points for faster rendering on long series")
        max_assets_display = st.slider("Max assets in position view", min_value=10, max_value=100, value=50, step=10)
    
    # View 1: Equity + Drawdown
    st.header("1️⃣ Equity + Drawdown")
    if equity_curve is not None and not equity_curve.empty:
        fig_equity = plot_equity_drawdown(equity_curve, downsample=downsample_plots)
        st.plotly_chart(fig_equity, use_container_width=True)
        
        # Top Contributors: Top 10 assets by PnL (last 30d)
        top_assets = get_top_asset_contributors(artifacts, days=30)
        if top_assets is not None and not top_assets.empty:
            with st.expander("📊 Top 10 Asset Contributors (Last 30 Days)", expanded=False):
                st.dataframe(
                    top_assets.style.format({
                        'PnL Contribution': '{:.6f}',
                        'Abs PnL Contribution': '{:.6f}'
                    }),
                    use_container_width=True,
                    height=300
                )
        
        # Summary stats
        if len(equity_curve) > 0:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Current Equity", f"{equity_curve.iloc[-1]:.4f}")
            with col2:
                total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0] - 1.0) * 100
                st.metric("Total Return", f"{total_return:.2f}%")
            with col3:
                if portfolio_returns is not None and len(portfolio_returns) > 0:
                    vol = portfolio_returns.std() * np.sqrt(252) * 100
                    st.metric("Annualized Vol", f"{vol:.2f}%")
            with col4:
                running_max = equity_curve.cummax()
                drawdown = (equity_curve / running_max - 1.0).min() * 100
                st.metric("Max Drawdown", f"{drawdown:.2f}%")
    else:
        st.warning("Equity curve data not available")
    
    st.markdown("---")
    
    # View 2: Exposure by Sleeve
    st.header("2️⃣ Exposure Over Time")
    fig_exposure = plot_exposure_by_sleeve(artifacts, downsample=downsample_plots)
    if fig_exposure:
        st.plotly_chart(fig_exposure, use_container_width=True)
        
        # Top Contributors: Top 5 sleeves by PnL (last 60d)
        top_sleeves = get_top_sleeve_contributors(artifacts, days=60)
        if top_sleeves is not None and not top_sleeves.empty:
            with st.expander("📊 Top 5 Sleeve Contributors (Last 60 Days)", expanded=False):
                st.dataframe(
                    top_sleeves.style.format({
                        'PnL Contribution': '{:.6f}',
                        'Abs PnL Contribution': '{:.6f}'
                    }),
                    use_container_width=True,
                    height=200
                )
    else:
        st.warning("Exposure data not available (weights_raw.csv required)")
    
    st.markdown("---")
    
    # View 3: Position-level View
    st.header("3️⃣ Position-Level View")
    
    if equity_curve is not None and not equity_curve.empty:
        date_range = (equity_curve.index[0], equity_curve.index[-1])
        
        col1, col2 = st.columns([3, 1])
        with col1:
            selected_date = st.date_input(
                "Select Date",
                value=equity_curve.index[-1].date(),
                min_value=date_range[0].date(),
                max_value=date_range[1].date()
            )
        
        selected_timestamp = pd.Timestamp(selected_date)
        
        position_view = get_position_view(artifacts, selected_timestamp)
        if position_view is not None:
            holdings = position_view['holdings']
            pnl_recent = position_view['pnl_recent']
            
            # Holdings snapshot (limit to top N assets for performance)
            st.subheader("Holdings Snapshot")
            
            # Filter to top N assets by exposure
            holdings_sorted = holdings.sort_values('exposure', ascending=False)
            holdings_display = holdings_sorted.head(max_assets_display)
            
            if len(holdings) > max_assets_display:
                st.info(f"Showing top {max_assets_display} assets by exposure (out of {len(holdings)} total). Use sidebar slider to adjust.")
            
            st.dataframe(
                holdings_display.style.format({
                    'weight_pre_allocator': '{:.4f}',
                    'weight_post_allocator': '{:.4f}',
                    'exposure': '{:.4f}',
                    'pnl_contribution': '{:.6f}'
                }),
                use_container_width=True,
                height=400
            )
            
            # Add note about naming
            st.caption("💡 **Note**: 'Pre-Allocator' weights are post-policy (policy gates affect signals upstream). 'Post-Allocator' weights are final (after allocator scaling).")
            
            # PnL contribution over last 30 days
            if pnl_recent is not None and not pnl_recent.empty:
                st.subheader("PnL Contribution (Last 30 Days)")
                st.dataframe(
                    pnl_recent.style.format({'pnl_contribution': '{:.6f}'}),
                    use_container_width=True,
                    height=200
                )
            
            # Turnover proxy
            st.subheader("Turnover Proxy")
            weights = artifacts.get('weights_scaled') or artifacts.get('weights')
            if weights is not None:
                turnover_series = compute_turnover(weights)
                if not turnover_series.empty:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Average Turnover", f"{turnover_series.mean():.3f}")
                    with col2:
                        st.metric("Max Turnover", f"{turnover_series.max():.3f}")
                    with col3:
                        st.metric("Latest Turnover", f"{turnover_series.iloc[-1]:.3f}")
                    
                    # Downsample if requested
                    turnover_plot = turnover_series
                    if downsample_plots and len(turnover_series) > 1000:
                        turnover_plot = downsample_series(turnover_series, max_points=1000)
                    
                    # Turnover plot
                    fig_turnover = go.Figure()
                    fig_turnover.add_trace(go.Scatter(
                        x=turnover_plot.index,
                        y=turnover_plot.values,
                        mode='lines',
                        name='Turnover',
                        line=dict(color='purple', width=1)
                    ))
                    fig_turnover.update_layout(
                        title='Turnover Over Time',
                        xaxis_title='Date',
                        yaxis_title='Turnover',
                        height=300
                    )
                    st.plotly_chart(fig_turnover, use_container_width=True)
                    
                    # Top Contributors: Top 10 turnover events
                    top_turnover = get_top_turnover_events(artifacts, top_n=10)
                    if top_turnover is not None and not top_turnover.empty:
                        with st.expander("📊 Top 10 Turnover Events", expanded=False):
                            st.dataframe(
                                top_turnover.style.format({
                                    'Turnover': '{:.3f}'
                                }),
                                use_container_width=True,
                                height=300
                            )
        else:
            st.warning(f"Position data not available for {selected_date}")
    else:
        st.warning("Equity curve data not available for date selection")
    
    st.markdown("---")
    
    # View 4: Allocator State Timeline
    st.header("4️⃣ Allocator State Timeline")
    fig_allocator = plot_allocator_timeline(artifacts, downsample=downsample_plots)
    if fig_allocator:
        st.plotly_chart(fig_allocator, use_container_width=True)
    else:
        st.warning("Allocator data not available (allocator_regime_v1.csv or allocator_risk_v1_applied.csv required)")
    
    st.markdown("---")
    
    # View 5: Drag Waterfall
    st.header("5️⃣ Drag Waterfall")
    fig_drag = plot_drag_waterfall(artifacts)
    if fig_drag:
        st.plotly_chart(fig_drag, use_container_width=True)
    else:
        st.info("Drag waterfall data not available (canonical_diagnostics.json required)")
    
    st.markdown("---")
    
    # View 6: Correlation & Diversification Health
    st.header("6️⃣ Correlation & Diversification Health")
    fig_corr = plot_correlation_health(artifacts, downsample=downsample_plots)
    if fig_corr:
        st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.info("Correlation health data not available (asset_returns.csv required)")
    
    st.markdown("---")
    
    # View 7: Sleeve Concentration Timeline
    st.header("7️⃣ Sleeve Concentration Timeline")
    fig_sleeve_conc = plot_sleeve_concentration_timeline(artifacts, downsample=downsample_plots)
    if fig_sleeve_conc:
        st.plotly_chart(fig_sleeve_conc, use_container_width=True)
    else:
        st.info("Sleeve concentration data not available (sleeve_returns.csv required)")
    
    st.markdown("---")
    
    # View 8: Baseline Comparison (if baseline selected)
    if baseline_run is not None:
        st.header("8️⃣ Baseline Comparison")
        
        # Load baseline artifacts
        baseline_run_dir = runs_dir / baseline_run
        try:
            baseline_artifacts = load_run_artifacts_cached(str(baseline_run_dir))
            
            # Equity ratio plot
            fig_baseline = plot_baseline_comparison(artifacts, baseline_artifacts)
            if fig_baseline:
                st.plotly_chart(fig_baseline, use_container_width=True)
            
            # Delta table
            comparison_table = get_baseline_comparison_table(artifacts, baseline_artifacts)
            if comparison_table is not None:
                st.subheader("Metrics Comparison")
                
                # Format the table
                format_dict = {
                    'Variant': '{:.4f}',
                    'Baseline': '{:.4f}',
                    'Delta': '{:.4f}'
                }
                
                # Special formatting for percentages
                comparison_display = comparison_table.copy()
                comparison_display['Variant'] = comparison_display.apply(
                    lambda row: f"{row['Variant']:.2%}" if row['Metric'] in ['CAGR', 'MaxDD', 'Worst Month'] else f"{row['Variant']:.4f}",
                    axis=1
                )
                comparison_display['Baseline'] = comparison_display.apply(
                    lambda row: f"{row['Baseline']:.2%}" if row['Metric'] in ['CAGR', 'MaxDD', 'Worst Month'] else f"{row['Baseline']:.4f}",
                    axis=1
                )
                comparison_display['Delta'] = comparison_display.apply(
                    lambda row: f"{row['Delta']:.2%}" if row['Metric'] in ['CAGR', 'MaxDD', 'Worst Month'] else f"{row['Delta']:.4f}",
                    axis=1
                )
                
                st.dataframe(comparison_display, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error loading baseline run: {e}")
    
    st.markdown("---")
    
    # View 9: Diagnostics Summary
    st.header("9️⃣ Diagnostics Summary")
    canonical_diagnostics = artifacts.get('canonical_diagnostics')
    
    if canonical_diagnostics is not None:
        # Diagnostics Summary Expander
        with st.expander("📊 Diagnostics Summary", expanded=True):
            decomp = canonical_diagnostics.get('performance_decomposition', {})
            engine_sharpe = canonical_diagnostics.get('engine_sharpe_contribution', {})
            binding = canonical_diagnostics.get('constraint_binding', {})
            path_diagnostics = canonical_diagnostics.get('path_diagnostics', {})
            
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                allocator_drag = decomp.get('allocator_drag_bps', np.nan)
                if not np.isnan(allocator_drag):
                    st.metric("Allocator Drag", f"{allocator_drag:.1f} bps/yr")
                else:
                    st.metric("Allocator Drag", "N/A")
            
            with col2:
                policy_drag = decomp.get('policy_drag_bps', np.nan)
                if not np.isnan(policy_drag):
                    st.metric("Policy Drag", f"{policy_drag:.1f} bps/yr")
                else:
                    st.metric("Policy Drag", "N/A")
            
            with col3:
                binding_pct = binding.get('allocator_active_pct', np.nan)
                if not np.isnan(binding_pct):
                    st.metric("Allocator Active", f"{binding_pct:.1f}%")
                else:
                    st.metric("Allocator Active", "N/A")
            
            with col4:
                worst_drawdowns = path_diagnostics.get('worst_drawdowns', [])
                if worst_drawdowns:
                    worst_dd_depth = worst_drawdowns[0].get('depth_pct', np.nan)
                    if not np.isnan(worst_dd_depth):
                        st.metric("Worst Drawdown", f"{worst_dd_depth:.2f}%")
                    else:
                        st.metric("Worst Drawdown", "N/A")
                else:
                    st.metric("Worst Drawdown", "N/A")
            
            # Sleeve Sharpe Table
            if engine_sharpe:
                st.subheader("Sleeve Sharpe & Contribution")
                engine_df = pd.DataFrame(engine_sharpe).T
                if not engine_df.empty:
                    # Select key columns for display
                    display_cols = []
                    if 'unconditional_sharpe' in engine_df.columns:
                        display_cols.append('unconditional_sharpe')
                    if 'pct_of_total_pnl' in engine_df.columns:
                        display_cols.append('pct_of_total_pnl')
                    if 'contribution_to_portfolio_sharpe' in engine_df.columns:
                        display_cols.append('contribution_to_portfolio_sharpe')
                    
                    if display_cols:
                        display_df = engine_df[display_cols].copy()
                        # Format columns
                        format_dict = {}
                        if 'unconditional_sharpe' in display_df.columns:
                            format_dict['unconditional_sharpe'] = '{:.4f}'
                        if 'pct_of_total_pnl' in display_df.columns:
                            format_dict['pct_of_total_pnl'] = '{:.1f}%'
                        if 'contribution_to_portfolio_sharpe' in display_df.columns:
                            format_dict['contribution_to_portfolio_sharpe'] = '{:.4f}'
                        
                        display_df = display_df.rename(columns={
                            'unconditional_sharpe': 'Sharpe',
                            'pct_of_total_pnl': '% PnL',
                            'contribution_to_portfolio_sharpe': 'Sharpe Contrib'
                        })
                        
                        st.dataframe(
                            display_df.style.format(format_dict),
                            use_container_width=True,
                            height=200
                        )
                
                # Worst 10 Drawdowns (top 5 for summary)
                worst_drawdowns = path_diagnostics.get('worst_drawdowns', [])
                if worst_drawdowns:
                    st.subheader("Worst 5 Drawdowns")
                    dd_display = []
                    for i, dd in enumerate(worst_drawdowns[:5], 1):
                        dd_display.append({
                            'Rank': i,
                            'Start': dd.get('start', 'N/A'),
                            'Trough': dd.get('trough', 'N/A'),
                            'Depth (%)': dd.get('depth_pct', np.nan),
                            'Duration (days)': dd.get('duration_days', np.nan)
                        })
                    
                    if dd_display:
                        dd_df = pd.DataFrame(dd_display)
                        st.dataframe(
                            dd_df.style.format({
                                'Depth (%)': '{:.2f}',
                                'Duration (days)': '{:.0f}'
                            }),
                            use_container_width=True,
                            height=200
                        )
        
        # Download/Copy Section
        with st.expander("📥 Download / Copy Markdown Report", expanded=False):
            try:
                from src.diagnostics.canonical_diagnostics import generate_markdown_report
                markdown_report = generate_markdown_report(canonical_diagnostics)
                
                st.text_area(
                    "Markdown Report",
                    markdown_report,
                    height=400,
                    help="Copy this markdown report to paste into documents"
                )
                
                st.download_button(
                    label="📥 Download Markdown Report",
                    data=markdown_report,
                    file_name=f"{selected_run}_canonical_diagnostics.md",
                    mime="text/markdown"
                )
            except ImportError:
                st.warning("Cannot generate markdown report: canonical_diagnostics module not available")
            except Exception as e:
                st.warning(f"Error generating markdown report: {e}")
    else:
        st.info("Canonical diagnostics not available. Run diagnostics generation script to create canonical_diagnostics.json")
    
    # Footer
    st.markdown("---")
    st.markdown("*Dashboard reads artifacts only. No strategy logic computation.*")


if __name__ == "__main__":
    main()
