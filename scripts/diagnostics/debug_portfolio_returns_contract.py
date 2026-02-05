"""
Debug Portfolio Returns Contract Failure

Diagnostic script to identify root cause of portfolio_returns_base != sum(weights × returns) mismatch.

Tests 6 ranked hypotheses:
1. One-day lag/shift mismatch (t vs t-1)
2. Return definition mismatch (log vs arithmetic)
3. Weights normalization differences
4. Instrument universe mismatch
5. Calendar alignment issues
6. Rounding/clip differences

Also provides per-asset contribution analysis on mismatch date.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def load_artifacts(run_dir: Path) -> Dict:
    """Load all required artifacts for debugging."""
    artifacts = {}
    
    # Load portfolio returns base (the exact vector used before any transforms)
    portfolio_returns_base_path = run_dir / 'portfolio_returns_base.csv'
    if portfolio_returns_base_path.exists():
        artifacts['portfolio_returns_base'] = pd.read_csv(
            portfolio_returns_base_path, index_col=0, parse_dates=True
        )['ret']
    else:
        # Fallback to portfolio_returns if base doesn't exist
        artifacts['portfolio_returns_base'] = pd.read_csv(
            run_dir / 'portfolio_returns.csv', index_col=0, parse_dates=True
        )['ret']
    
    # Load weights used for portfolio returns
    artifacts['weights_used'] = pd.read_csv(
        run_dir / 'weights_used_for_portfolio_returns.csv', index_col=0, parse_dates=True
    )
    
    # Load asset returns
    artifacts['asset_returns'] = pd.read_csv(
        run_dir / 'asset_returns.csv', index_col=0, parse_dates=True
    )
    
    return artifacts


def find_first_mismatch(computed: pd.Series, actual: pd.Series, tolerance_abs: float = 1e-6, tolerance_rel: float = 0.01) -> Optional[pd.Timestamp]:
    """Find the first date where computed and actual differ beyond tolerance."""
    common_dates = computed.index.intersection(actual.index)
    if len(common_dates) == 0:
        return None
    
    computed_aligned = computed.loc[common_dates]
    actual_aligned = actual.loc[common_dates]
    
    diff = computed_aligned - actual_aligned
    abs_diff = diff.abs()
    rel_diff = (abs_diff / (actual_aligned.abs() + 1e-8))
    
    # Find first date exceeding tolerance
    mask = (abs_diff > tolerance_abs) | (rel_diff > tolerance_rel)
    if mask.any():
        return mask.idxmax()
    return None


def test_hypothesis_1_lag_mismatch(artifacts: Dict, mismatch_date: pd.Timestamp) -> Dict:
    """
    Hypothesis 1: One-day lag/shift mismatch (t vs t-1)
    
    Test: sum(weights[t] * returns[t]) vs sum(weights[t-1] * returns[t])
    """
    weights = artifacts['weights_used']
    returns = artifacts['asset_returns']
    actual = artifacts['portfolio_returns_base']
    
    # Align weights and returns
    weights_daily = weights.reindex(returns.index).ffill().fillna(0.0)
    common_symbols = weights_daily.columns.intersection(returns.columns)
    
    if len(common_symbols) == 0:
        return {'tested': False, 'error': 'No common symbols'}
    
    weights_aligned = weights_daily[common_symbols]
    returns_aligned = returns[common_symbols]
    
    # Test 1a: weights[t] * returns[t] (current)
    portfolio_t = (weights_aligned * returns_aligned).sum(axis=1)
    portfolio_t = np.exp(portfolio_t) - 1.0  # Convert log to simple
    
    # Test 1b: weights[t-1] * returns[t] (lagged)
    weights_lagged = weights_aligned.shift(1).fillna(0.0)
    portfolio_t_lagged = (weights_lagged * returns_aligned).sum(axis=1)
    portfolio_t_lagged = np.exp(portfolio_t_lagged) - 1.0
    
    # Compare on mismatch date
    if mismatch_date in portfolio_t.index and mismatch_date in actual.index:
        computed_t = portfolio_t.loc[mismatch_date]
        computed_t_lagged = portfolio_t_lagged.loc[mismatch_date]
        actual_val = actual.loc[mismatch_date]
        
        diff_t = abs(computed_t - actual_val)
        diff_t_lagged = abs(computed_t_lagged - actual_val)
        
        # Find first mismatch for both
        first_mismatch_t = find_first_mismatch(portfolio_t, actual)
        first_mismatch_t_lagged = find_first_mismatch(portfolio_t_lagged, actual)
        
        return {
            'tested': True,
            'hypothesis': 'One-day lag mismatch',
            'mismatch_date': str(mismatch_date),
            'computed_t': float(computed_t),
            'computed_t_lagged': float(computed_t_lagged),
            'actual': float(actual_val),
            'diff_t': float(diff_t),
            'diff_t_lagged': float(diff_t_lagged),
            'first_mismatch_t': str(first_mismatch_t) if first_mismatch_t else None,
            'first_mismatch_t_lagged': str(first_mismatch_t_lagged) if first_mismatch_t_lagged else None,
            'likely_cause': 'lagged_weights' if diff_t_lagged < diff_t else 'current_weights',
            'match_found': diff_t_lagged < 1e-6 if diff_t_lagged < diff_t else diff_t < 1e-6
        }
    
    return {'tested': False, 'error': 'Mismatch date not in computed series'}


def test_hypothesis_2_return_definition(artifacts: Dict, mismatch_date: pd.Timestamp) -> Dict:
    """
    Hypothesis 2: Return definition mismatch (log vs arithmetic)
    
    asset_returns.csv contains SIMPLE returns (converted from log in exec_sim.py).
    Runtime computes: portfolio_log = sum(weights * log_returns), then converts to simple: exp(portfolio_log) - 1
    To reconstruct correctly: convert simple back to log, compute portfolio_log, then convert to simple.
    """
    weights = artifacts['weights_used']
    returns = artifacts['asset_returns']
    actual = artifacts['portfolio_returns_base']
    
    # Align weights and returns
    weights_daily = weights.reindex(returns.index).ffill().fillna(0.0)
    common_symbols = weights_daily.columns.intersection(returns.columns)
    
    if len(common_symbols) == 0:
        return {'tested': False, 'error': 'No common symbols'}
    
    weights_aligned = weights_daily[common_symbols]
    returns_aligned = returns[common_symbols]
    
    # Test 2a: WRONG - Treating simple returns as log returns (incorrect)
    portfolio_log_wrong = (weights_aligned * returns_aligned).sum(axis=1)
    portfolio_log_to_simple_wrong = np.exp(portfolio_log_wrong) - 1.0
    
    # Test 2b: WRONG - Direct sum of simple returns (incorrect for large returns)
    portfolio_simple_wrong = (weights_aligned * returns_aligned).sum(axis=1)
    
    # Test 2c: CORRECT - Convert simple back to log, compute portfolio_log, then convert to simple
    returns_log = np.log(1 + returns_aligned)  # Convert simple returns back to log
    portfolio_log_correct = (weights_aligned * returns_log).sum(axis=1)
    portfolio_simple_correct = np.exp(portfolio_log_correct) - 1.0
    
    # Compare on mismatch date
    if mismatch_date in portfolio_simple_correct.index and mismatch_date in actual.index:
        computed_log_wrong = portfolio_log_to_simple_wrong.loc[mismatch_date]
        computed_simple_wrong = portfolio_simple_wrong.loc[mismatch_date]
        computed_correct = portfolio_simple_correct.loc[mismatch_date]
        actual_val = actual.loc[mismatch_date]
        
        diff_log_wrong = abs(computed_log_wrong - actual_val)
        diff_simple_wrong = abs(computed_simple_wrong - actual_val)
        diff_correct = abs(computed_correct - actual_val)
        
        return {
            'tested': True,
            'hypothesis': 'Return definition mismatch (log vs arithmetic)',
            'mismatch_date': str(mismatch_date),
            'computed_log_wrong': float(computed_log_wrong),
            'computed_simple_wrong': float(computed_simple_wrong),
            'computed_correct': float(computed_correct),
            'actual': float(actual_val),
            'diff_log_wrong': float(diff_log_wrong),
            'diff_simple_wrong': float(diff_simple_wrong),
            'diff_correct': float(diff_correct),
            'likely_cause': 'correct_method' if diff_correct < 1e-6 else ('simple_sum' if diff_simple_wrong < diff_log_wrong else 'log_then_exp'),
            'match_found': diff_correct < 1e-6
        }
    
    return {'tested': False, 'error': 'Mismatch date not in computed series'}


def test_hypothesis_3_weights_normalization(artifacts: Dict, mismatch_date: pd.Timestamp) -> Dict:
    """
    Hypothesis 3: Weights normalization differences
    
    Check if weights are normalized differently (gross exposure, sum to 1, etc.)
    """
    weights = artifacts['weights_used']
    returns = artifacts['asset_returns']
    actual = artifacts['portfolio_returns_base']
    
    # Align weights and returns
    weights_daily = weights.reindex(returns.index).ffill().fillna(0.0)
    common_symbols = weights_daily.columns.intersection(returns.columns)
    
    if len(common_symbols) == 0:
        return {'tested': False, 'error': 'No common symbols'}
    
    weights_aligned = weights_daily[common_symbols]
    returns_aligned = returns[common_symbols]
    
    # Check weights properties on mismatch date
    if mismatch_date in weights_aligned.index:
        weights_row = weights_aligned.loc[mismatch_date]
        
        gross_exposure = weights_row.abs().sum()
        net_exposure = weights_row.sum()
        max_weight = weights_row.abs().max()
        min_weight = weights_row.abs().min()
        
        # Test with normalized weights (sum of abs = 1)
        if gross_exposure > 1e-6:
            weights_normalized = weights_row / gross_exposure
            portfolio_normalized = (weights_normalized * returns_aligned.loc[mismatch_date]).sum()
            portfolio_normalized = np.exp(portfolio_normalized) - 1.0 if portfolio_normalized < 10 else portfolio_normalized
        else:
            portfolio_normalized = None
        
        # Current computation
        portfolio_current = (weights_row * returns_aligned.loc[mismatch_date]).sum()
        portfolio_current = np.exp(portfolio_current) - 1.0
        
        actual_val = actual.loc[mismatch_date] if mismatch_date in actual.index else None
        
        return {
            'tested': True,
            'hypothesis': 'Weights normalization differences',
            'mismatch_date': str(mismatch_date),
            'gross_exposure': float(gross_exposure),
            'net_exposure': float(net_exposure),
            'max_weight': float(max_weight),
            'min_weight': float(min_weight),
            'computed_current': float(portfolio_current),
            'computed_normalized': float(portfolio_normalized) if portfolio_normalized is not None else None,
            'actual': float(actual_val) if actual_val is not None else None,
            'diff_current': float(abs(portfolio_current - actual_val)) if actual_val is not None else None,
            'diff_normalized': float(abs(portfolio_normalized - actual_val)) if (portfolio_normalized is not None and actual_val is not None) else None
        }
    
    return {'tested': False, 'error': 'Mismatch date not in weights'}


def test_hypothesis_4_universe_mismatch(artifacts: Dict, mismatch_date: pd.Timestamp) -> Dict:
    """
    Hypothesis 4: Instrument universe mismatch (missing/extra columns)
    """
    weights = artifacts['weights_used']
    returns = artifacts['asset_returns']
    
    # Align weights and returns
    weights_daily = weights.reindex(returns.index).ffill().fillna(0.0)
    
    weights_symbols = set(weights_daily.columns)
    returns_symbols = set(returns.columns)
    
    common_symbols = weights_symbols.intersection(returns_symbols)
    weights_only = weights_symbols - returns_symbols
    returns_only = returns_symbols - weights_symbols
    
    # Check on mismatch date
    if mismatch_date in weights_daily.index and mismatch_date in returns.index:
        weights_row = weights_daily.loc[mismatch_date]
        returns_row = returns.loc[mismatch_date]
        
        # Check for non-zero weights in symbols not in returns
        missing_contributions = {}
        for sym in weights_only:
            if abs(weights_row[sym]) > 1e-6:
                missing_contributions[sym] = float(weights_row[sym])
        
        # Check for non-zero returns in symbols not in weights
        extra_returns = {}
        for sym in returns_only:
            if abs(returns_row[sym]) > 1e-6:
                extra_returns[sym] = float(returns_row[sym])
        
        return {
            'tested': True,
            'hypothesis': 'Instrument universe mismatch',
            'mismatch_date': str(mismatch_date),
            'weights_symbols_count': len(weights_symbols),
            'returns_symbols_count': len(returns_symbols),
            'common_symbols_count': len(common_symbols),
            'weights_only': list(weights_only),
            'returns_only': list(returns_only),
            'missing_contributions': missing_contributions,
            'extra_returns': extra_returns,
            'has_mismatch': len(weights_only) > 0 or len(returns_only) > 0
        }
    
    return {'tested': False, 'error': 'Mismatch date not in data'}


def compute_per_asset_contribution(artifacts: Dict, mismatch_date: pd.Timestamp, use_lagged_weights: bool = False) -> pd.DataFrame:
    """
    Compute per-asset contribution to portfolio return on mismatch date.
    
    Returns DataFrame with columns: symbol, weight, return, contribution, abs_contribution
    """
    weights = artifacts['weights_used']
    returns = artifacts['asset_returns']
    
    # Align weights and returns
    weights_daily = weights.reindex(returns.index).ffill().fillna(0.0)
    common_symbols = weights_daily.columns.intersection(returns.columns)
    
    if len(common_symbols) == 0:
        return pd.DataFrame()
    
    weights_aligned = weights_daily[common_symbols]
    returns_aligned = returns[common_symbols]
    
    if mismatch_date not in weights_aligned.index or mismatch_date not in returns_aligned.index:
        return pd.DataFrame()
    
    if use_lagged_weights:
        weights_row = weights_aligned.shift(1).loc[mismatch_date].fillna(0.0)
    else:
        weights_row = weights_aligned.loc[mismatch_date]
    
    returns_row = returns_aligned.loc[mismatch_date]
    
    # Compute contributions (assuming log returns)
    contributions = weights_row * returns_row
    
    # Convert to simple returns for contribution
    portfolio_log = contributions.sum()
    portfolio_simple = np.exp(portfolio_log) - 1.0
    
    # Individual contributions (approximate - for relative ranking)
    contrib_df = pd.DataFrame({
        'symbol': common_symbols,
        'weight': weights_row.values,
        'return': returns_row.values,
        'contribution_log': contributions.values,
        'abs_contribution': contributions.abs().values
    }).set_index('symbol')
    
    # Sort by absolute contribution
    contrib_df = contrib_df.sort_values('abs_contribution', ascending=False)
    
    return contrib_df


def main():
    """Run all diagnostic tests."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Debug portfolio returns contract failure")
    parser.add_argument('--run_id', type=str, required=True, help='Run ID to debug')
    parser.add_argument('--mismatch_date', type=str, default=None, help='Specific date to analyze (YYYY-MM-DD). If not provided, finds first mismatch.')
    parser.add_argument('--base_dir', type=str, default='reports/runs', help='Base directory for runs')
    
    args = parser.parse_args()
    
    run_dir = Path(args.base_dir) / args.run_id
    
    if not run_dir.exists():
        print(f"Error: Run directory not found: {run_dir}")
        return 1
    
    print("=" * 80)
    print(f"Portfolio Returns Contract Debug: {args.run_id}")
    print("=" * 80)
    print()
    
    # Load artifacts
    print("Loading artifacts...")
    artifacts = load_artifacts(run_dir)
    
    # Find first mismatch if not specified
    weights = artifacts['weights_used']
    returns = artifacts['asset_returns']
    actual = artifacts['portfolio_returns_base']
    
    weights_daily = weights.reindex(returns.index).ffill().fillna(0.0)
    common_symbols = weights_daily.columns.intersection(returns.columns)
    weights_aligned = weights_daily[common_symbols]
    returns_aligned = returns[common_symbols]
    
    portfolio_computed = (weights_aligned * returns_aligned).sum(axis=1)
    portfolio_computed = np.exp(portfolio_computed) - 1.0
    
    if args.mismatch_date:
        mismatch_date = pd.Timestamp(args.mismatch_date)
    else:
        mismatch_date = find_first_mismatch(portfolio_computed, actual)
        if mismatch_date is None:
            print("No mismatch found! Contract passes.")
            return 0
        print(f"Found first mismatch on: {mismatch_date}")
    
    print(f"\nAnalyzing mismatch on: {mismatch_date}")
    print("=" * 80)
    print()
    
    # Run all hypothesis tests
    print("Testing Hypothesis 1: One-day lag/shift mismatch...")
    result_1 = test_hypothesis_1_lag_mismatch(artifacts, mismatch_date)
    print_result(result_1)
    print()
    
    print("Testing Hypothesis 2: Return definition mismatch (log vs arithmetic)...")
    result_2 = test_hypothesis_2_return_definition(artifacts, mismatch_date)
    print_result(result_2)
    print()
    
    print("Testing Hypothesis 3: Weights normalization differences...")
    result_3 = test_hypothesis_3_weights_normalization(artifacts, mismatch_date)
    print_result(result_3)
    print()
    
    print("Testing Hypothesis 4: Instrument universe mismatch...")
    result_4 = test_hypothesis_4_universe_mismatch(artifacts, mismatch_date)
    print_result(result_4)
    print()
    
    # Per-asset contribution analysis
    print("=" * 80)
    print("Per-Asset Contribution Analysis")
    print("=" * 80)
    print()
    
    print("Top 10 contributions (current weights):")
    contrib_current = compute_per_asset_contribution(artifacts, mismatch_date, use_lagged_weights=False)
    if not contrib_current.empty:
        print(contrib_current.head(10).to_string())
    print()
    
    print("Top 10 contributions (lagged weights t-1):")
    contrib_lagged = compute_per_asset_contribution(artifacts, mismatch_date, use_lagged_weights=True)
    if not contrib_lagged.empty:
        print(contrib_lagged.head(10).to_string())
    print()
    
    # Summary
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    
    if result_1.get('match_found'):
        print("[MATCH] Hypothesis 1 (lag mismatch) likely explains the issue")
        print(f"  Use {'lagged weights (t-1)' if result_1['likely_cause'] == 'lagged_weights' else 'current weights (t)'}")
    elif result_2.get('match_found'):
        print("[MATCH] Hypothesis 2 (return definition) likely explains the issue")
        print(f"  Returns are {'simple' if result_2['likely_cause'] == 'simple_returns' else 'log'}")
    elif result_3.get('diff_normalized') and result_3['diff_normalized'] < result_3.get('diff_current', float('inf')):
        print("[MATCH] Hypothesis 3 (weights normalization) likely explains the issue")
    elif result_4.get('has_mismatch'):
        print("[MATCH] Hypothesis 4 (universe mismatch) likely explains the issue")
        if result_4.get('missing_contributions'):
            print(f"  Missing contributions from: {list(result_4['missing_contributions'].keys())}")
        if result_4.get('extra_returns'):
            print(f"  Extra returns from: {list(result_4['extra_returns'].keys())}")
    else:
        print("[WARN] No single hypothesis fully explains the mismatch")
        print("  Review per-asset contributions above for patterns")
    
    return 0


def print_result(result: Dict):
    """Print test result in readable format."""
    if not result.get('tested'):
        print(f"  [WARN] Not tested: {result.get('error', 'Unknown error')}")
        return
    
    print(f"  Hypothesis: {result.get('hypothesis', 'Unknown')}")
    if 'likely_cause' in result:
        print(f"  Likely cause: {result['likely_cause']}")
    if 'match_found' in result:
        status = "[MATCH FOUND]" if result['match_found'] else "[No match]"
        print(f"  {status}")
    
    # Print key metrics
    for key in ['computed_t', 'computed_t_lagged', 'computed_log', 'computed_simple', 
                'computed_current', 'computed_normalized', 'actual', 
                'diff_t', 'diff_t_lagged', 'diff_log', 'diff_simple', 
                'diff_current', 'diff_normalized']:
        if key in result and result[key] is not None:
            print(f"  {key}: {result[key]:.8f}")


if __name__ == "__main__":
    sys.exit(main())
