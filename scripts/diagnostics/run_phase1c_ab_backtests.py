#!/usr/bin/env python3
"""
Phase 1C: Canonical A/B Backtests

Runs the 3 canonical scenarios to validate Risk Targeting and Allocator behavior:

1. Baseline: Core v9, no RT, no allocator
2. RT only: Core v9 + Risk Targeting
3. RT + Alloc-H: Core v9 + Risk Targeting + Allocator-H
4. (Optional) RT + Alloc-M: Core v9 + Risk Targeting + Allocator-M
5. (Optional) RT + Alloc-L: Core v9 + Risk Targeting + Allocator-L

Generates a comparison report with:
- Annualized return / vol / Sharpe
- Max drawdown
- Worst month
- % days allocator not-1.0 (for H/M/L)
- Avg leverage + 95th percentile leverage
- Event table of top 10 days by drawdown

Usage:
    python scripts/diagnostics/run_phase1c_ab_backtests.py \
        --strategy_profile core_v9 \
        --start 2020-01-01 \
        --end 2025-10-31
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.agents.diagnostics_perf import load_run, compute_yearly_stats


def run_backtest(
    strategy_profile: str,
    start: str,
    end: str,
    run_id: str,
    config_overrides: Optional[Dict] = None
) -> str:
    """
    Run a backtest with given configuration.
    
    Args:
        strategy_profile: Strategy profile name
        start: Start date (YYYY-MM-DD)
        end: End date (YYYY-MM-DD)
        run_id: Run ID for this backtest
        config_overrides: Optional dict of config overrides
    
    Returns:
        Run ID (for artifact loading)
    """
    print(f"\n{'='*80}")
    print(f"Running backtest: {run_id}")
    print(f"{'='*80}")
    
    # Build command
    cmd = [
        "python", "run_strategy.py",
        "--strategy_profile", strategy_profile,
        "--start", start,
        "--end", end,
        "--run_id", run_id,
    ]
    
    # Add config overrides if provided
    if config_overrides:
        # Create temp config file with overrides
        import tempfile
        import yaml
        
        # Load base config
        base_config = Path("configs/strategies.yaml")
        with open(base_config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # Apply overrides
        for key, value in config_overrides.items():
            keys = key.split('.')
            d = config
            for k in keys[:-1]:
                d = d.setdefault(k, {})
            d[keys[-1]] = value
        
        # Write temp config
        tmp_config = Path(tempfile.mkdtemp()) / "strategies.yaml"
        with open(tmp_config, 'w', encoding='utf-8') as f:
            yaml.dump(config, f)
        
        # DEBUG: Print relevant sections of overridden config
        print(f"\nDEBUG: Config overrides applied:")
        for key, value in config_overrides.items():
            print(f"  {key}: {value}")
        
        if 'allocator_v1' in config:
            print(f"\nDEBUG: allocator_v1 config after overrides:")
            print(f"  enabled: {config['allocator_v1'].get('enabled')}")
            print(f"  mode: {config['allocator_v1'].get('mode')}")
            print(f"  profile: {config['allocator_v1'].get('profile')}")
        
        if 'risk_targeting' in config:
            print(f"\nDEBUG: risk_targeting config after overrides:")
            print(f"  enabled: {config['risk_targeting'].get('enabled')}")
        
        cmd.extend(["--config_path", str(tmp_config)])
    
    print(f"Command: {' '.join(cmd)}")
    
    # Run backtest - stream output, handle Windows encoding
    import os
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    result = subprocess.run(cmd, env=env)
    
    if result.returncode != 0:
        print(f"ERROR: Backtest failed for {run_id}")
        raise RuntimeError(f"Backtest failed: {run_id}")
    
    print(f"✓ Backtest completed: {run_id}")
    return run_id


def load_artifacts(run_id: str) -> Dict:
    """Load artifacts from a run."""
    run_dir = Path(f"reports/runs/{run_id}")
    
    artifacts = {
        'run_id': run_id,
        'run_dir': run_dir,
    }
    
    # Load portfolio returns
    returns_file = run_dir / "portfolio_returns.csv"
    if returns_file.exists():
        df = pd.read_csv(returns_file, index_col=0, parse_dates=True)
        # Handle different column names
        col = 'ret' if 'ret' in df.columns else 'portfolio_return'
        artifacts['returns'] = df[col]
    
    # Load equity curve
    equity_file = run_dir / "equity_curve.csv"
    if equity_file.exists():
        artifacts['equity'] = pd.read_csv(equity_file, index_col=0, parse_dates=True)['equity']
    
    # Load Risk Targeting artifacts
    rt_dir = run_dir / "risk_targeting"
    if rt_dir.exists():
        leverage_file = rt_dir / "leverage_series.csv"
        if leverage_file.exists():
            artifacts['leverage'] = pd.read_csv(leverage_file, parse_dates=['date'], index_col='date')['leverage']
        
        vol_file = rt_dir / "realized_vol.csv"
        if vol_file.exists():
            artifacts['realized_vol'] = pd.read_csv(vol_file, parse_dates=['date'], index_col='date')['realized_vol']
    
    # Load Allocator artifacts
    alloc_dir = run_dir / "allocator"
    if alloc_dir.exists():
        regime_file = alloc_dir / "regime_series.csv"
        if regime_file.exists():
            artifacts['regime'] = pd.read_csv(regime_file, parse_dates=['date'], index_col='date')['regime']
        
        multiplier_file = alloc_dir / "multiplier_series.csv"
        if multiplier_file.exists():
            artifacts['multiplier'] = pd.read_csv(multiplier_file, parse_dates=['date'], index_col='date')['multiplier']
    
    return artifacts


def compute_report_metrics(artifacts: Dict) -> Dict:
    """Compute metrics for the report."""
    returns = artifacts.get('returns')
    equity = artifacts.get('equity')
    leverage = artifacts.get('leverage')
    multiplier = artifacts.get('multiplier')
    
    metrics = {}
    
    if returns is not None and len(returns) > 0:
        # Core metrics - compute inline instead of using compute_core_metrics
        n_days = len(returns)
        years = n_days / 252.0
        
        # CAGR from equity
        if equity is not None and len(equity) >= 2:
            cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1.0 / years) - 1 if years > 0 else 0.0
        else:
            cagr = ((1 + returns).prod()) ** (1.0 / years) - 1 if years > 0 else 0.0
        
        # Volatility
        vol = returns.std() * np.sqrt(252)
        
        # Sharpe
        sharpe = (cagr / vol) if vol > 0 else 0.0
        
        # Max Drawdown
        if equity is not None:
            running_max = equity.expanding().max()
            drawdown = (equity - running_max) / running_max
            max_dd = drawdown.min()
        else:
            cum_returns = (1 + returns).cumprod()
            running_max = cum_returns.expanding().max()
            drawdown = (cum_returns - running_max) / running_max
            max_dd = drawdown.min()
        
        # Hit rate
        hit_rate = (returns > 0).mean()
        
        metrics['cagr'] = cagr
        metrics['vol'] = vol
        metrics['sharpe'] = sharpe
        metrics['max_dd'] = max_dd
        metrics['hit_rate'] = hit_rate
        
        # Worst month
        try:
            monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
            metrics['worst_month'] = monthly_returns.min()
        except Exception:
            metrics['worst_month'] = 0.0
        
        # Drawdown analysis - already computed above
        if equity is not None:
            metrics['drawdown_series'] = drawdown
            top_dd_days = drawdown.nsmallest(10).index
            metrics['top_dd_days'] = top_dd_days
    
    if leverage is not None:
        metrics['avg_leverage'] = leverage.mean()
        metrics['leverage_95pct'] = leverage.quantile(0.95)
        metrics['leverage_max'] = leverage.max()
        metrics['leverage_min'] = leverage.min()
    
    if multiplier is not None:
        # % days allocator not-1.0
        not_one = (multiplier != 1.0).sum()
        total = len(multiplier)
        metrics['pct_days_allocator_active'] = (not_one / total * 100) if total > 0 else 0.0
        metrics['avg_multiplier'] = multiplier.mean()
        metrics['min_multiplier'] = multiplier.min()
    
    return metrics


def generate_event_table(artifacts: Dict, top_dd_days: pd.DatetimeIndex) -> pd.DataFrame:
    """Generate event table for top drawdown days."""
    events = []
    
    for date in top_dd_days:
        event = {'date': date.strftime('%Y-%m-%d')}
        
        # Get leverage if available
        leverage = artifacts.get('leverage')
        if leverage is not None and date in leverage.index:
            event['leverage'] = leverage.loc[date]
        
        # Get regime if available
        regime = artifacts.get('regime')
        if regime is not None and date in regime.index:
            event['regime'] = regime.loc[date]
        
        # Get multiplier if available
        multiplier = artifacts.get('multiplier')
        if multiplier is not None and date in multiplier.index:
            event['multiplier'] = multiplier.loc[date]
        
        # Get drawdown
        equity = artifacts.get('equity')
        if equity is not None and date in equity.index:
            running_max = equity.loc[:date].max()
            dd = (equity.loc[date] - running_max) / running_max
            event['drawdown'] = dd
        
        events.append(event)
    
    return pd.DataFrame(events)


def generate_report(scenarios: List[Dict], output_dir: Path) -> None:
    """Generate comparison report."""
    print(f"\n{'='*80}")
    print("GENERATING COMPARISON REPORT")
    print(f"{'='*80}")
    
    # Compute metrics for each scenario
    all_metrics = []
    for scenario in scenarios:
        artifacts = load_artifacts(scenario['run_id'])
        metrics = compute_report_metrics(artifacts)
        metrics['scenario'] = scenario['name']
        all_metrics.append(metrics)
    
    # Create comparison DataFrame
    comparison_data = []
    for m in all_metrics:
        comparison_data.append({
            'Scenario': m['scenario'],
            'CAGR (%)': m.get('cagr', 0) * 100,
            'Vol (%)': m.get('vol', 0) * 100,
            'Sharpe': m.get('sharpe', 0),
            'MaxDD (%)': m.get('max_dd', 0) * 100,
            'Worst Month (%)': m.get('worst_month', 0) * 100,
            'Avg Leverage': m.get('avg_leverage', 1.0),
            'Leverage 95pct': m.get('leverage_95pct', 1.0),
            '% Days Alloc Active': m.get('pct_days_allocator_active', 0.0),
            'Avg Multiplier': m.get('avg_multiplier', 1.0),
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Print report
    print("\n" + "="*80)
    print("PHASE 1C A/B BACKTEST COMPARISON REPORT")
    print("="*80)
    print("\nCore Metrics:")
    print(comparison_df.to_string(index=False))
    
    # Event table for baseline (top 10 drawdown days)
    baseline_artifacts = load_artifacts(scenarios[0]['run_id'])
    baseline_metrics = compute_report_metrics(baseline_artifacts)
    
    if 'top_dd_days' in baseline_metrics:
        event_table = generate_event_table(baseline_artifacts, baseline_metrics['top_dd_days'])
        print("\n" + "="*80)
        print("TOP 10 DRAWDOWN DAYS (Baseline)")
        print("="*80)
        print(event_table.to_string(index=False))
    
    # Save report
    report_file = output_dir / "phase1c_ab_comparison.md"
    with open(report_file, 'w') as f:
        f.write("# Phase 1C A/B Backtest Comparison Report\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## Core Metrics Comparison\n\n")
        f.write(comparison_df.to_markdown(index=False))
        f.write("\n\n")
        
        if 'top_dd_days' in baseline_metrics:
            f.write("## Top 10 Drawdown Days (Baseline)\n\n")
            f.write(event_table.to_markdown(index=False))
    
    print(f"\n✓ Report saved to: {report_file}")
    
    # Save JSON for programmatic access
    json_file = output_dir / "phase1c_ab_comparison.json"
    with open(json_file, 'w') as f:
        json.dump({
            'comparison': comparison_data,
            'scenarios': [s['name'] for s in scenarios],
        }, f, indent=2, default=str)
    
    print(f"✓ JSON saved to: {json_file}")


def main():
    ap = argparse.ArgumentParser(
        description="Phase 1C: Canonical A/B backtests for Risk Targeting + Allocator validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--strategy_profile", required=True, help="Strategy profile (e.g., core_v9)")
    ap.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    ap.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    ap.add_argument("--include_alloc_m_l", action="store_true", help="Include Alloc-M and Alloc-L scenarios")
    ap.add_argument("--output_dir", default="reports/diagnostics/phase1c_ab", help="Output directory for report")
    ap.add_argument(
        "--allocator_mode",
        choices=["compute", "precomputed"],
        default="precomputed",
        help="Allocator mode: 'compute' (live-like, has warmup issues) or 'precomputed' (proven path, requires precomputed_run_id)"
    )
    ap.add_argument(
        "--precomputed_run_id",
        default=None,
        help="Precomputed run ID for allocator scalars (required if allocator_mode='precomputed')"
    )
    
    args = ap.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define scenarios
    scenarios = [
        {
            'name': 'Baseline (no RT, no allocator)',
            'run_id': f"{args.strategy_profile}_baseline_{args.start}_{args.end}",
            'config_overrides': {
                'risk_targeting.enabled': False,
                'allocator_v1.enabled': False,
                'allocator_v1.mode': 'off',
            }
        },
        {
            'name': 'RT only',
            'run_id': f"{args.strategy_profile}_rt_only_{args.start}_{args.end}",
            'config_overrides': {
                'risk_targeting.enabled': True,
                'allocator_v1.enabled': False,
                'allocator_v1.mode': 'off',
            }
        },
        {
            'name': 'RT + Alloc-H',
            'run_id': f"{args.strategy_profile}_rt_alloc_h_{args.start}_{args.end}",
            'config_overrides': {
                'risk_targeting.enabled': True,
                'allocator_v1.enabled': True,
                'allocator_v1.mode': args.allocator_mode,
                'allocator_v1.profile': 'H',
                # If precomputed mode, require precomputed_run_id
                **({'allocator_v1.precomputed_run_id': args.precomputed_run_id} if args.allocator_mode == 'precomputed' and args.precomputed_run_id else {}),
            }
        },
    ]
    
    if args.include_alloc_m_l:
        scenarios.extend([
            {
                'name': 'RT + Alloc-M',
                'run_id': f"{args.strategy_profile}_rt_alloc_m_{args.start}_{args.end}",
                'config_overrides': {
                    'risk_targeting.enabled': True,
                    'allocator_v1.enabled': True,
                    'allocator_v1.mode': 'compute',
                    'allocator_v1.profile': 'M',
                }
            },
            {
                'name': 'RT + Alloc-L',
                'run_id': f"{args.strategy_profile}_rt_alloc_l_{args.start}_{args.end}",
                'config_overrides': {
                    'risk_targeting.enabled': True,
                    'allocator_v1.enabled': True,
                    'allocator_v1.mode': 'compute',
                    'allocator_v1.profile': 'L',
                }
            },
        ])
    
    # Run all scenarios
    failed_scenarios = []
    successful_scenarios = []
    for scenario in scenarios:
        try:
            run_backtest(
                args.strategy_profile,
                args.start,
                args.end,
                scenario['run_id'],
                scenario.get('config_overrides')
            )
            successful_scenarios.append(scenario)
        except Exception as e:
            print(f"ERROR: Failed to run {scenario['name']}: {e}")
            failed_scenarios.append((scenario['name'], str(e)))
            # Continue with other scenarios instead of failing immediately
    
    if not successful_scenarios:
        print("ERROR: All scenarios failed!")
        return 1
    
    # Use only successful scenarios for report
    scenarios = successful_scenarios
    
    # Generate report
    try:
        generate_report(scenarios, output_dir)
    except Exception as e:
        print(f"ERROR: Failed to generate report: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print(f"\n{'='*80}")
    print("PHASE 1C A/B BACKTESTS COMPLETE")
    print(f"{'='*80}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

