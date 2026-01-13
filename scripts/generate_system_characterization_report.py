"""
Generate System Characterization Report - Step 2
================================================

This script generates a comprehensive System Characterization Report for the
canonical frozen stack baseline run.

The report includes:
1. Portfolio-level performance metrics
   - CAGR / Sharpe / Vol / MaxDD
   - Recovery time
   - Worst month / quarter
2. Sleeve contribution & loss attribution
   - Which sleeves drive drawdowns
   - Which sleeves dominate recovery
   - Sleeve PnL concentration
3. Allocator behavior audit
   - Regime frequencies
   - Time spent in NORMAL / ELEVATED / STRESS / CRISIS
   - Did it brake when expected?
   - Did it not brake when it shouldn't?

This is NOT asking "Is this good?" but rather "Is this behavior acceptable and understood?"
"""

import sys
import argparse
import json
from pathlib import Path
from typing import Dict, Optional
import pandas as pd
import numpy as np
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.agents.diagnostics_perf import load_run, compute_core_metrics, compute_yearly_stats
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_allocator_artifacts(run_dir: Path) -> Dict:
    """Load allocator artifacts from run directory."""
    artifacts = {}
    
    # Load regime series (from allocator_regime_v1.csv)
    regime_file = run_dir / "allocator_regime_v1.csv"
    if regime_file.exists():
        artifacts['regime'] = pd.read_csv(regime_file, parse_dates=True, index_col=0)['regime']
    else:
        logger.warning(f"Regime series not found: {regime_file}")
    
    # Load risk scalars (from allocator_risk_v1.csv)
    risk_scalar_file = run_dir / "allocator_risk_v1.csv"
    if risk_scalar_file.exists():
        artifacts['risk_scalar'] = pd.read_csv(risk_scalar_file, parse_dates=True, index_col=0)['risk_scalar']
    else:
        logger.warning(f"Risk scalars not found: {risk_scalar_file}")
    
    # Load applied scalars (from allocator_risk_v1_applied_used.csv)
    applied_file = run_dir / "allocator_risk_v1_applied_used.csv"
    if applied_file.exists():
        artifacts['applied'] = pd.read_csv(applied_file, parse_dates=True, index_col=0)
    else:
        # Fallback to allocator_risk_v1_applied.csv
        applied_file_fallback = run_dir / "allocator_risk_v1_applied.csv"
        if applied_file_fallback.exists():
            artifacts['applied'] = pd.read_csv(applied_file_fallback, parse_dates=True, index_col=0)
        else:
            logger.warning(f"Applied scalars not found: {applied_file} or {applied_file_fallback}")
    
    return artifacts


def load_engine_policy_artifacts(run_dir: Path) -> Dict:
    """Load engine policy artifacts from run directory."""
    artifacts = {}
    
    # Load applied multipliers
    applied_file = run_dir / "engine_policy_applied_v1.csv"
    if applied_file.exists():
        df = pd.read_csv(applied_file, parse_dates=['rebalance_date'])
        artifacts['applied'] = df
    else:
        logger.warning(f"Engine policy applied not found: {applied_file}")
    
    # Load state
    state_file = run_dir / "engine_policy_state_v1.csv"
    if state_file.exists():
        artifacts['state'] = pd.read_csv(state_file, parse_dates=['date'], index_col='date')
    else:
        logger.warning(f"Engine policy state not found: {state_file}")
    
    return artifacts


def compute_recovery_time(equity_curve: pd.Series) -> Dict:
    """Compute recovery time metrics."""
    if len(equity_curve) < 2:
        return {}
    
    running_max = equity_curve.cummax()
    drawdown = (equity_curve / running_max) - 1.0
    
    # Find all drawdown periods
    in_dd = drawdown < -0.01  # At least 1% drawdown
    dd_periods = []
    
    current_dd_start = None
    current_dd_trough = None
    current_dd_trough_date = None
    
    for i, (date, dd_val) in enumerate(drawdown.items()):
        if in_dd.iloc[i]:
            if current_dd_start is None:
                current_dd_start = date
                current_dd_trough = dd_val
                current_dd_trough_date = date
            else:
                if dd_val < current_dd_trough:
                    current_dd_trough = dd_val
                    current_dd_trough_date = date
        else:
            if current_dd_start is not None:
                # Drawdown ended, find recovery date
                recovery_date = date
                recovery_days = (recovery_date - current_dd_trough_date).days
                dd_periods.append({
                    'start': current_dd_start,
                    'trough': current_dd_trough_date,
                    'trough_value': current_dd_trough,
                    'recovery': recovery_date,
                    'recovery_days': recovery_days,
                    'depth': current_dd_trough
                })
                current_dd_start = None
                current_dd_trough = None
                current_dd_trough_date = None
    
    if not dd_periods:
        return {
            'max_recovery_days': 0,
            'avg_recovery_days': 0,
            'n_drawdowns': 0
        }
    
    recovery_days_list = [p['recovery_days'] for p in dd_periods]
    max_dd_period = max(dd_periods, key=lambda x: abs(x['depth']))
    
    return {
        'max_recovery_days': max(recovery_days_list) if recovery_days_list else 0,
        'avg_recovery_days': np.mean(recovery_days_list) if recovery_days_list else 0,
        'n_drawdowns': len(dd_periods),
        'worst_drawdown_recovery_days': max_dd_period['recovery_days'],
        'worst_drawdown_trough': max_dd_period['trough'],
        'worst_drawdown_recovery': max_dd_period['recovery']
    }


def compute_worst_periods(returns: pd.Series) -> Dict:
    """Compute worst month and quarter."""
    if len(returns) == 0:
        return {}
    
    # Resample to monthly and quarterly
    monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
    quarterly_returns = returns.resample('Q').apply(lambda x: (1 + x).prod() - 1)
    
    worst_month = monthly_returns.min()
    worst_month_date = monthly_returns.idxmin()
    worst_quarter = quarterly_returns.min()
    worst_quarter_date = quarterly_returns.idxmin()
    
    # Format quarter date (e.g., "2020-Q1")
    worst_quarter_str = None
    if pd.notna(worst_quarter_date):
        quarter_num = (worst_quarter_date.month - 1) // 3 + 1
        worst_quarter_str = f"{worst_quarter_date.year}-Q{quarter_num}"
    
    return {
        'worst_month': worst_month,
        'worst_month_date': worst_month_date.strftime('%Y-%m') if pd.notna(worst_month_date) else None,
        'worst_quarter': worst_quarter,
        'worst_quarter_date': worst_quarter_str
    }


def compute_sleeve_attribution(run_dir: Path, equity_curve: pd.Series, returns: pd.Series) -> Dict:
    """Compute sleeve contribution and loss attribution."""
    attribution = {}
    
    # Try to load sleeve returns
    sleeve_returns_file = run_dir / "sleeve_returns.csv"
    if sleeve_returns_file.exists():
        sleeve_returns = pd.read_csv(sleeve_returns_file, index_col=0, parse_dates=True)
        
        # Align with equity curve dates
        sleeve_returns = sleeve_returns.reindex(equity_curve.index).fillna(0.0)
        
        # Compute cumulative contribution
        sleeve_cumulative = (1 + sleeve_returns).cumprod() - 1
        
        # Find worst drawdown period
        running_max = equity_curve.cummax()
        drawdown = (equity_curve / running_max) - 1.0
        worst_dd_period_start = drawdown.idxmin()
        worst_dd_period_end = None
        
        # Find when drawdown started and ended
        for i in range(drawdown.index.get_loc(worst_dd_period_start), len(drawdown)):
            if drawdown.iloc[i] >= -0.001:  # Recovered to within 0.1%
                worst_dd_period_end = drawdown.index[i]
                break
        
        if worst_dd_period_end is None:
            worst_dd_period_end = drawdown.index[-1]
        
        # Compute sleeve contribution during worst drawdown
        if worst_dd_period_start in sleeve_returns.index and worst_dd_period_end in sleeve_returns.index:
            dd_mask = (sleeve_returns.index >= worst_dd_period_start) & (sleeve_returns.index <= worst_dd_period_end)
            sleeve_dd_contrib = sleeve_returns[dd_mask].sum()
            
            attribution['sleeve_drawdown_contribution'] = sleeve_dd_contrib.to_dict()
            attribution['worst_drawdown_period'] = {
                'start': worst_dd_period_start.strftime('%Y-%m-%d'),
                'end': worst_dd_period_end.strftime('%Y-%m-%d')
            }
        
        # Overall sleeve contribution
        total_sleeve_contrib = sleeve_returns.sum()
        attribution['sleeve_total_contribution'] = total_sleeve_contrib.to_dict()
        
        # Sleeve PnL concentration (Herfindahl index)
        sleeve_abs_contrib = sleeve_returns.abs().sum()
        total_abs = sleeve_abs_contrib.sum()
        if total_abs > 0:
            concentration = ((sleeve_abs_contrib / total_abs) ** 2).sum()
            attribution['sleeve_concentration'] = concentration
        else:
            attribution['sleeve_concentration'] = 0.0
        
    else:
        logger.warning(f"Sleeve returns not found: {sleeve_returns_file}")
        attribution['sleeve_returns_available'] = False
    
    return attribution


def compute_allocator_audit(allocator_artifacts: Dict) -> Dict:
    """Compute allocator behavior audit."""
    audit = {}
    
    if 'regime' in allocator_artifacts and len(allocator_artifacts['regime']) > 0:
        regime = allocator_artifacts['regime']
        
        # Regime frequencies
        regime_counts = regime.value_counts()
        total_days = len(regime)
        
        audit['regime_frequencies'] = {
            regime: {
                'count': int(count),
                'pct': float(count / total_days * 100)
            }
            for regime, count in regime_counts.items()
        }
        
        # Time in each regime
        audit['total_days'] = total_days
        audit['regime_distribution'] = {
            regime: float(count / total_days * 100)
            for regime, count in regime_counts.items()
        }
    
    if 'multiplier' in allocator_artifacts and len(allocator_artifacts['multiplier']) > 0:
        multiplier = allocator_artifacts['multiplier']
        
        # Multiplier statistics
        audit['multiplier_stats'] = {
            'mean': float(multiplier.mean()),
            'min': float(multiplier.min()),
            'max': float(multiplier.max()),
            'std': float(multiplier.std()),
            'pct_below_1.0': float((multiplier < 1.0).sum() / len(multiplier) * 100),
            'pct_below_0.9': float((multiplier < 0.9).sum() / len(multiplier) * 100),
            'pct_below_0.8': float((multiplier < 0.8).sum() / len(multiplier) * 100)
        }
    
    if 'applied' in allocator_artifacts and len(allocator_artifacts['applied']) > 0:
        applied = allocator_artifacts['applied']
        
        if 'risk_scalar_applied' in applied.columns:
            scalar = applied['risk_scalar_applied']
            audit['applied_scalar_stats'] = {
                'mean': float(scalar.mean()),
                'min': float(scalar.min()),
                'max': float(scalar.max()),
                'pct_active': float((scalar < 0.999).sum() / len(scalar) * 100),
                'n_rebalances': len(scalar)
            }
    
    return audit


def generate_report(run_id: str, output_path: Optional[Path] = None) -> Dict:
    """Generate complete system characterization report."""
    logger.info("=" * 80)
    logger.info("SYSTEM CHARACTERIZATION REPORT - Step 2")
    logger.info("=" * 80)
    logger.info(f"Run ID: {run_id}")
    
    # Load run data
    run_dir = Path(f"reports/runs/{run_id}")
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    
    logger.info(f"Loading run data from: {run_dir}")
    run = load_run(run_id)
    
    # 1. Portfolio-level performance metrics
    logger.info("\n[1/3] Computing portfolio-level performance metrics...")
    core_metrics = compute_core_metrics(run)
    yearly_stats = compute_yearly_stats(run)
    recovery_metrics = compute_recovery_time(run.equity_curve)
    worst_periods = compute_worst_periods(run.portfolio_returns)
    
    # 2. Sleeve contribution & loss attribution
    logger.info("\n[2/3] Computing sleeve contribution & loss attribution...")
    sleeve_attribution = compute_sleeve_attribution(run_dir, run.equity_curve, run.portfolio_returns)
    
    # 3. Allocator behavior audit
    logger.info("\n[3/3] Computing allocator behavior audit...")
    allocator_artifacts = load_allocator_artifacts(run_dir)
    allocator_audit = compute_allocator_audit(allocator_artifacts)
    
    # Engine Policy audit
    engine_policy_artifacts = load_engine_policy_artifacts(run_dir)
    
    # Compile report
    report = {
        'run_id': run_id,
        'generated_at': datetime.now().isoformat(),
        'portfolio_metrics': {
            'core_metrics': core_metrics,
            'yearly_stats': yearly_stats.to_dict('index') if not yearly_stats.empty else {},
            'recovery_metrics': recovery_metrics,
            'worst_periods': worst_periods
        },
        'sleeve_attribution': sleeve_attribution,
        'allocator_audit': allocator_audit,
        'engine_policy_audit': {
            'artifacts_available': len(engine_policy_artifacts) > 0,
            'applied_multipliers': engine_policy_artifacts.get('applied', pd.DataFrame()).to_dict('records') if 'applied' in engine_policy_artifacts else []
        }
    }
    
    # Write report
    if output_path is None:
        output_path = run_dir / "system_characterization_report.json"
    
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    logger.info(f"\nReport written to: {output_path}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("SYSTEM CHARACTERIZATION REPORT SUMMARY")
    print("=" * 80)
    print(f"\nPortfolio Performance:")
    print(f"  CAGR:     {core_metrics.get('CAGR', 0):.2%}")
    print(f"  Sharpe:   {core_metrics.get('Sharpe', 0):.4f}")
    print(f"  Vol:      {core_metrics.get('Vol', 0):.2%}")
    print(f"  MaxDD:    {core_metrics.get('MaxDD', 0):.2%}")
    
    if recovery_metrics:
        print(f"\nRecovery Metrics:")
        print(f"  Max Recovery Days: {recovery_metrics.get('max_recovery_days', 0)}")
        print(f"  Avg Recovery Days: {recovery_metrics.get('avg_recovery_days', 0):.1f}")
        print(f"  Number of Drawdowns: {recovery_metrics.get('n_drawdowns', 0)}")
    
    if worst_periods:
        print(f"\nWorst Periods:")
        if worst_periods.get('worst_month_date'):
            print(f"  Worst Month: {worst_periods['worst_month']:.2%} ({worst_periods['worst_month_date']})")
        if worst_periods.get('worst_quarter_date'):
            print(f"  Worst Quarter: {worst_periods['worst_quarter']:.2%} ({worst_periods['worst_quarter_date']})")
    
    if allocator_audit:
        print(f"\nAllocator Behavior:")
        if 'regime_frequencies' in allocator_audit:
            print(f"  Regime Distribution:")
            for regime, data in allocator_audit['regime_frequencies'].items():
                print(f"    {regime}: {data['pct']:.1f}% ({data['count']} days)")
        
        if 'multiplier_stats' in allocator_audit:
            stats = allocator_audit['multiplier_stats']
            print(f"  Multiplier Stats:")
            print(f"    Mean: {stats['mean']:.4f}")
            print(f"    Min: {stats['min']:.4f}")
            print(f"    % Active (< 1.0): {stats['pct_below_1.0']:.1f}%")
    
    if sleeve_attribution:
        print(f"\nSleeve Attribution:")
        if 'sleeve_total_contribution' in sleeve_attribution:
            print(f"  Total Contribution by Sleeve:")
            for sleeve, contrib in sleeve_attribution['sleeve_total_contribution'].items():
                print(f"    {sleeve}: {contrib:.4f}")
        if 'sleeve_concentration' in sleeve_attribution:
            print(f"  PnL Concentration (Herfindahl): {sleeve_attribution['sleeve_concentration']:.4f}")
    
    print("\n" + "=" * 80)
    print(f"Full report saved to: {output_path}")
    print("=" * 80)
    
    return report


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate System Characterization Report - Step 2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script generates a comprehensive system characterization report including:
1. Portfolio-level performance metrics
2. Sleeve contribution & loss attribution
3. Allocator behavior audit

Examples:
  # Generate report for a run
  python scripts/generate_system_characterization_report.py --run_id canonical_frozen_stack_precomputed_20260113_100007
        """
    )
    
    parser.add_argument(
        '--run_id',
        type=str,
        required=True,
        help='Run ID to analyze'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output path for report JSON (default: reports/runs/{run_id}/system_characterization_report.json)'
    )
    
    args = parser.parse_args()
    
    try:
        output_path = Path(args.output) if args.output else None
        report = generate_report(args.run_id, output_path)
        return 0
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())

