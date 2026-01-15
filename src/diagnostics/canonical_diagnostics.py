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
    
    # Store run_dir in artifacts for use in compute_constraint_binding
    artifacts['_run_dir'] = run_dir
    
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


def compute_constraint_binding(artifacts: Dict, run_dir: Optional[Path] = None) -> Dict:
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
    
    # System Level Evaluation Window (Phase 3A)
    binding['evaluation_start_date'] = meta.get('evaluation_start_date')
    binding['effective_start_date'] = meta.get('effective_start_date')
    
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
    # Phase 3A: Policy-Inert classification governance hooks
    meta = artifacts.get('meta', {})
    engine_policy_config = meta.get('config', {}).get('engine_policy_v1', {})
    engine_policy_enabled = engine_policy_config.get('enabled', False)
    
    # Check for required policy features in meta (if available)
    # Phase 3A: Policy-Inert classification is data-driven (market object / meta), NOT file-based
    # This is the "don't regress" principle: check data presence, not optional file existence
    policy_features_meta = meta.get('policy_features', {})
    
    # Required policy features for EnginePolicyV1
    required_policy_features = {
        'gamma_stress_proxy': False,
        'vx_backwardation': False,
        'vrp_stress_proxy': False
    }
    
    # Check if features are present (from meta - data-driven check)
    if policy_features_meta:
        for feature_name in required_policy_features.keys():
            # Check if feature is documented in meta with has_data=True
            if feature_name in policy_features_meta:
                feature_info = policy_features_meta[feature_name]
                # Feature is present if documented AND has data (not all NaN)
                if isinstance(feature_info, dict):
                    required_policy_features[feature_name] = feature_info.get('present', False) and feature_info.get('has_data', False)
                elif isinstance(feature_info, bool):
                    # Legacy format: just bool
                    required_policy_features[feature_name] = feature_info
    
    # Fallback: If policy state artifact exists, check for stress_value data
    # This is a secondary check - primary is meta.policy_features (data-driven)
    if run_dir is None:
        # Try to infer run_dir from artifacts if available
        run_dir = artifacts.get('_run_dir')
    
    if run_dir is not None:
        engine_policy_state_file = Path(run_dir) / "engine_policy_state_v1.csv"
        if engine_policy_state_file.exists() and engine_policy_enabled:
            try:
                state_df = pd.read_csv(engine_policy_state_file, parse_dates=['date'])
                # Check if state has non-NaN stress_value for Trend and VRP engines
                for engine_name in ['trend', 'vrp']:
                    engine_state = state_df[state_df['engine'] == engine_name] if 'engine' in state_df.columns else pd.DataFrame()
                    if not engine_state.empty and 'stress_value' in engine_state.columns:
                        has_stress_data = engine_state['stress_value'].notna().any()
                        # If we have state data, infer feature was present (conservative)
                        if has_stress_data and engine_name == 'trend' and not required_policy_features['gamma_stress_proxy']:
                            required_policy_features['gamma_stress_proxy'] = True
                        if has_stress_data and engine_name == 'vrp' and not required_policy_features['vrp_stress_proxy']:
                            required_policy_features['vrp_stress_proxy'] = True
                        # VX backwardation is used by VRP, so if VRP has data, backwardation was likely present
                        if has_stress_data and engine_name == 'vrp' and not required_policy_features['vx_backwardation']:
                            required_policy_features['vx_backwardation'] = True
            except Exception as e:
                logger.warning(f"[Diagnostics] Could not check policy state artifact for feature presence: {e}")
    
    # Compute policy inputs status
    policy_inputs_present = required_policy_features.copy()
    policy_inputs_missing = not all(required_policy_features.values())
    policy_effective = engine_policy_enabled and not policy_inputs_missing
    
    # Determine policy_inert_reason
    policy_inert_reason = None
    if engine_policy_enabled:
        if policy_inputs_missing:
            missing_features = [name for name, present in required_policy_features.items() if not present]
            policy_inert_reason = f"Missing policy features: {', '.join(missing_features)}"
        elif engine_policy is None or (isinstance(engine_policy, pd.DataFrame) and engine_policy.empty):
            policy_inert_reason = "Policy enabled but artifact missing"
        else:
            # Policy is effective - check if it has teeth
            if 'trend_multiplier' in engine_policy.columns and 'vrp_multiplier' in engine_policy.columns:
                total_rebalances = len(engine_policy)
                trend_gated = (engine_policy['trend_multiplier'] < 0.999).sum()
                vrp_gated = (engine_policy['vrp_multiplier'] < 0.999).sum()
                if trend_gated == 0 and vrp_gated == 0:
                    policy_inert_reason = "Policy enabled but never gated (no teeth)"
    
    # Add governance hooks to binding dict
    binding['policy_enabled'] = engine_policy_enabled
    binding['policy_inputs_present'] = policy_inputs_present
    binding['policy_inputs_missing'] = policy_inputs_missing
    binding['policy_effective'] = policy_effective
    if policy_inert_reason:
        binding['policy_inert_reason'] = policy_inert_reason
        binding['policy_inert'] = True
    else:
        binding['policy_inert'] = False
    
    # Risk Targeting governance (Layer 5)
    risk_targeting_meta = meta.get('risk_targeting', {})
    rt_enabled = risk_targeting_meta.get('enabled', False)
    rt_inputs_present = risk_targeting_meta.get('inputs_present', {})
    rt_inputs_missing = risk_targeting_meta.get('inputs_missing', False)
    rt_effective = risk_targeting_meta.get('effective', False)
    rt_has_teeth = risk_targeting_meta.get('has_teeth', False)
    
    rt_inert_reason = None
    if rt_enabled:
        if rt_inputs_missing:
            missing_inputs = [k for k, v in rt_inputs_present.items() if isinstance(v, dict) and not v.get('has_data', False)]
            rt_inert_reason = f"Missing RT inputs: {', '.join(missing_inputs)}" if missing_inputs else "RT inputs missing"
        elif not rt_has_teeth:
            rt_inert_reason = "RT enabled but leverage always 1.0 (no teeth)"
    
    rt_inert = rt_enabled and (not rt_effective or rt_inert_reason is not None)
    
    binding['rt_enabled'] = rt_enabled
    binding['rt_inputs_present'] = rt_inputs_present
    binding['rt_inputs_missing'] = rt_inputs_missing
    binding['rt_effective'] = rt_effective
    binding['rt_has_teeth'] = rt_has_teeth
    binding['rt_inert'] = rt_inert
    if rt_inert_reason:
        binding['rt_inert_reason'] = rt_inert_reason
    
    # Add RT stats if available
    rt_multiplier_stats = risk_targeting_meta.get('multiplier_stats', {})
    if rt_multiplier_stats:
        binding['rt_multiplier_p50'] = rt_multiplier_stats.get('p50')
        binding['rt_multiplier_p95'] = rt_multiplier_stats.get('p95')
        binding['rt_multiplier_at_cap_pct'] = rt_multiplier_stats.get('at_cap', 0.0)
        binding['rt_multiplier_at_floor_pct'] = rt_multiplier_stats.get('at_floor', 0.0)
    
    rt_vol_stats = risk_targeting_meta.get('vol_stats', {})
    if rt_vol_stats:
        binding['rt_current_vol_p50'] = rt_vol_stats.get('p50')
        binding['rt_current_vol_p95'] = rt_vol_stats.get('p95')
    
    # Allocator v1 governance (Layer 6)
    allocator_v1_meta = meta.get('allocator_v1', {})
    alloc_v1_enabled = allocator_v1_meta.get('enabled', False)
    alloc_v1_inputs_present = allocator_v1_meta.get('inputs_present', {})
    alloc_v1_inputs_missing = allocator_v1_meta.get('inputs_missing', False)
    alloc_v1_state_computed = allocator_v1_meta.get('state_computed', False)
    alloc_v1_effective = allocator_v1_meta.get('effective', False)
    alloc_v1_has_teeth = allocator_v1_meta.get('has_teeth', False)
    
    alloc_v1_inert_reason = None
    if alloc_v1_enabled:
        if alloc_v1_inputs_missing:
            missing_inputs = [k for k, v in alloc_v1_inputs_present.items() if isinstance(v, dict) and not v.get('has_data', False)]
            alloc_v1_inert_reason = f"Missing Allocator v1 inputs: {', '.join(missing_inputs)}" if missing_inputs else "Allocator v1 inputs missing"
        elif not alloc_v1_state_computed:
            alloc_v1_inert_reason = "Allocator v1 state computation failed"
        elif not alloc_v1_has_teeth:
            alloc_v1_inert_reason = "Allocator v1 enabled but scalar always 1.0 (no teeth)"
    
    alloc_v1_inert = alloc_v1_enabled and (not alloc_v1_effective or alloc_v1_inert_reason is not None)
    
    binding['alloc_v1_enabled'] = alloc_v1_enabled
    binding['alloc_v1_inputs_present'] = alloc_v1_inputs_present
    binding['alloc_v1_inputs_missing'] = alloc_v1_inputs_missing
    binding['alloc_v1_state_computed'] = alloc_v1_state_computed
    binding['alloc_v1_effective'] = alloc_v1_effective
    binding['alloc_v1_has_teeth'] = alloc_v1_has_teeth
    binding['alloc_v1_inert'] = alloc_v1_inert
    if alloc_v1_inert_reason:
        binding['alloc_v1_inert_reason'] = alloc_v1_inert_reason
    
    # Add Allocator v1 stats if available
    alloc_v1_regime_dist = allocator_v1_meta.get('regime_distribution', {})
    if alloc_v1_regime_dist:
        binding['alloc_v1_regime_normal_pct'] = alloc_v1_regime_dist.get('NORMAL', 0.0)
        binding['alloc_v1_regime_elevated_pct'] = alloc_v1_regime_dist.get('ELEVATED', 0.0)
        binding['alloc_v1_regime_stress_pct'] = alloc_v1_regime_dist.get('STRESS', 0.0)
        binding['alloc_v1_regime_crisis_pct'] = alloc_v1_regime_dist.get('CRISIS', 0.0)
    
    alloc_v1_scalar_stats = allocator_v1_meta.get('scalar_stats', {})
    if alloc_v1_scalar_stats:
        binding['alloc_v1_scalar_p50'] = alloc_v1_scalar_stats.get('p50')
        binding['alloc_v1_scalar_p95'] = alloc_v1_scalar_stats.get('p95')
        binding['alloc_v1_scalar_at_min_pct'] = alloc_v1_scalar_stats.get('at_min', 0.0)
    
    # Policy gating percentages (must be numeric, 0.0 not NaN per SYSTEM_CONSTRUCTION)
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
            # Policy artifact exists but format unknown
            binding['policy_gated_trend_pct'] = 0.0
            binding['policy_gated_vrp_pct'] = 0.0
    else:
        # Policy artifact missing
        binding['policy_artifact_missing'] = True
        binding['policy_gated_trend_pct'] = 0.0
        binding['policy_gated_vrp_pct'] = 0.0
    
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
    constraint_binding = compute_constraint_binding(artifacts, run_dir=run_dir)
    
    logger.info("Computing Path Diagnostics...")
    path_diagnostics = compute_path_diagnostics(artifacts)
    
    
    # Load meta.json for metrics (Phase 3A dual metrics)
    meta = {}
    meta_file = run_dir / 'meta.json'
    if meta_file.exists():
        with open(meta_file, 'r') as f:
            meta = json.load(f)
    
    report = {
        'run_id': run_id,
        'generated_at': datetime.now().isoformat(),
        'metrics_eval': meta.get('metrics_eval', {}),
        'metrics_full': meta.get('metrics_full', {}),
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
    
    # Performance Metrics (Dual Reporting)
    if 'metrics_eval' in report or 'metrics_full' in report:
        lines.append("## Performance Metrics")
        lines.append("")
        lines.append("> **Evaluation Window Metrics**: Authoritative performance computed over the evaluation window only.")
        lines.append("> **Full Run Metrics**: Risk context metrics computed over the entire run (including warmup).")
        lines.append("")
        
        # Metrics Table
        lines.append("| Metric | Evaluation Window | Full Run |")
        lines.append("|---|---:|---:|")
        
        metrics_eval = report.get('metrics_eval', {})
        metrics_full = report.get('metrics_full', {})
        
        if metrics_eval and metrics_full:
            lines.append(f"| CAGR | {metrics_eval.get('cagr', 0):.2%} | {metrics_full.get('cagr', 0):.2%} |")
            lines.append(f"| Volatility | {metrics_eval.get('vol', 0):.2%} | {metrics_full.get('vol', 0):.2%} |")
            lines.append(f"| Sharpe | {metrics_eval.get('sharpe', 0):.2f} | {metrics_full.get('sharpe', 0):.2f} |")
            lines.append(f"| Max DD | {metrics_eval.get('max_drawdown', 0):.2%} | {metrics_full.get('max_drawdown', 0):.2%} |")
            lines.append(f"| Hit Rate | {metrics_eval.get('hit_rate', 0):.2%} | {metrics_full.get('hit_rate', 0):.2%} |")
            lines.append(f"| Avg Turnover | {metrics_eval.get('avg_turnover', 0):.2f} | {metrics_full.get('avg_turnover', 0):.2f} |")
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
    
    # Phase 3A: Policy-Inert Classification
    if 'policy_enabled' in binding:
        lines.append("### Policy Status (Phase 3A Governance)")
        lines.append("")
        policy_enabled = binding.get('policy_enabled', False)
        policy_effective = binding.get('policy_effective', False)
        policy_inert = binding.get('policy_inert', False)
        
        lines.append(f"- **Policy Enabled:** {policy_enabled}")
        lines.append(f"- **Policy Effective:** {policy_effective}")
        lines.append(f"- **Policy Inert:** {policy_inert}")
        
        if policy_inert:
            policy_inert_reason = binding.get('policy_inert_reason', 'Unknown reason')
            lines.append(f"- **⚠️ Policy-Inert Reason:** {policy_inert_reason}")
            lines.append("")
            lines.append("**⚠️ WARNING: This run is Policy-Inert and cannot be used for attribution/ablations.**")
            lines.append("")
        
        if 'policy_inputs_present' in binding:
            lines.append("**Policy Inputs Present:**")
            for feature_name, present in binding['policy_inputs_present'].items():
                # Use text status for Windows compatibility (avoid Unicode emoji encoding issues)
                status = "[OK]" if present else "[MISSING]"
                lines.append(f"  - {status} `{feature_name}`")
            lines.append("")
        
        if policy_enabled and not policy_effective:
            lines.append("**⚠️ Policy is enabled but not effective. Check policy inputs.**")
            lines.append("")
    
    # Risk Targeting governance (Layer 5)
    if 'rt_enabled' in binding:
        lines.append("### Risk Targeting Status (Layer 5 Governance)")
        lines.append("")
        rt_enabled = binding.get('rt_enabled', False)
        rt_effective = binding.get('rt_effective', False)
        rt_inert = binding.get('rt_inert', False)
        rt_has_teeth = binding.get('rt_has_teeth', False)
        
        lines.append(f"- **Risk Targeting Enabled:** {rt_enabled}")
        lines.append(f"- **Risk Targeting Effective:** {rt_effective}")
        lines.append(f"- **Risk Targeting Inert:** {rt_inert}")
        lines.append(f"- **Risk Targeting Has Teeth:** {rt_has_teeth}")
        
        if rt_inert:
            rt_inert_reason = binding.get('rt_inert_reason', 'Unknown reason')
            lines.append(f"- **[WARN] Risk Targeting Inert Reason:** {rt_inert_reason}")
            lines.append("")
            lines.append("**[WARN] WARNING: Risk Targeting is inert and may not be functioning correctly.**")
            lines.append("")
        
        if 'rt_inputs_present' in binding:
            lines.append("**Risk Targeting Inputs Present:**")
            for input_name, input_info in binding['rt_inputs_present'].items():
                if isinstance(input_info, dict):
                    has_data = input_info.get('has_data', False)
                    status = "[OK]" if has_data else "[MISSING]"
                    lines.append(f"  - {status} `{input_name}`")
            lines.append("")
        
        if rt_enabled and rt_has_teeth:
            if 'rt_multiplier_p50' in binding:
                lines.append(f"- **RT Leverage Multiplier (p50):** {binding['rt_multiplier_p50']:.3f}")
            if 'rt_multiplier_at_cap_pct' in binding:
                lines.append(f"- **RT Multiplier at Cap:** {binding['rt_multiplier_at_cap_pct']:.1f}%")
            if 'rt_multiplier_at_floor_pct' in binding:
                lines.append(f"- **RT Multiplier at Floor:** {binding['rt_multiplier_at_floor_pct']:.1f}%")
            lines.append("")
    
    # Allocator v1 governance (Layer 6)
    if 'alloc_v1_enabled' in binding:
        lines.append("### Allocator v1 Status (Layer 6 Governance)")
        lines.append("")
        alloc_v1_enabled = binding.get('alloc_v1_enabled', False)
        alloc_v1_effective = binding.get('alloc_v1_effective', False)
        alloc_v1_inert = binding.get('alloc_v1_inert', False)
        alloc_v1_has_teeth = binding.get('alloc_v1_has_teeth', False)
        
        lines.append(f"- **Allocator v1 Enabled:** {alloc_v1_enabled}")
        lines.append(f"- **Allocator v1 Effective:** {alloc_v1_effective}")
        lines.append(f"- **Allocator v1 Inert:** {alloc_v1_inert}")
        lines.append(f"- **Allocator v1 Has Teeth:** {alloc_v1_has_teeth}")
        
        if alloc_v1_inert:
            alloc_v1_inert_reason = binding.get('alloc_v1_inert_reason', 'Unknown reason')
            lines.append(f"- **[WARN] Allocator v1 Inert Reason:** {alloc_v1_inert_reason}")
            lines.append("")
            lines.append("**[WARN] WARNING: Allocator v1 is inert and may not be functioning correctly.**")
            lines.append("")
        
        if 'alloc_v1_inputs_present' in binding:
            lines.append("**Allocator v1 Inputs Present:**")
            for input_name, input_info in binding['alloc_v1_inputs_present'].items():
                if isinstance(input_info, dict):
                    has_data = input_info.get('has_data', False)
                    status = "[OK]" if has_data else "[MISSING]"
                    lines.append(f"  - {status} `{input_name}`")
            lines.append("")
        
        if alloc_v1_enabled and alloc_v1_has_teeth:
            if 'alloc_v1_regime_normal_pct' in binding:
                lines.append("**Regime Distribution:**")
                lines.append(f"  - NORMAL: {binding.get('alloc_v1_regime_normal_pct', 0.0):.1f}%")
                lines.append(f"  - ELEVATED: {binding.get('alloc_v1_regime_elevated_pct', 0.0):.1f}%")
                lines.append(f"  - STRESS: {binding.get('alloc_v1_regime_stress_pct', 0.0):.1f}%")
                lines.append(f"  - CRISIS: {binding.get('alloc_v1_regime_crisis_pct', 0.0):.1f}%")
            if 'alloc_v1_scalar_p50' in binding:
                lines.append(f"- **Allocator v1 Scalar (p50):** {binding['alloc_v1_scalar_p50']:.3f}")
            if 'alloc_v1_scalar_at_min_pct' in binding:
                lines.append(f"- **Allocator v1 Scalar at Min (0.25):** {binding['alloc_v1_scalar_at_min_pct']:.1f}%")
            lines.append("")
    
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
