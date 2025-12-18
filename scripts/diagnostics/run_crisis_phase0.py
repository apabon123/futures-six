#!/usr/bin/env python3
"""
Crisis Meta-Sleeve Phase-0 Diagnostics Script.

Tests three always-on crisis hedges:
1. Long VX2 (always-on)
2. Long VX2 - VX1 spread (always-on, dollar-neutral)
3. Long Duration (UB or ZN, always-on)

Each sleeve is tested at 5% weight against Core v9 baseline.

Phase-0 Evaluation Criteria:
- MaxDD improvement ≥ 1.0% vs Core v9 OR worst-month improvement
- No catastrophic bleed (> -2% CAGR at 5% weight)
- Crisis period attribution (2020 Q1, 2022 drawdown, vol spikes)

Usage:
    python scripts/diagnostics/run_crisis_phase0.py
    python scripts/diagnostics/run_crisis_phase0.py --start 2020-01-06 --end 2025-10-31
"""

import sys
import argparse
from pathlib import Path
import logging
from datetime import datetime
from typing import Dict, Optional, Tuple
import json
import pandas as pd
import numpy as np
import duckdb

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.canonical_window import load_canonical_window
from src.agents.utils_db import open_readonly_connection
from src.market_data.vrp_loaders import load_vx_curve, VX_FRONT_SYMBOL, VX_SECOND_SYMBOL
from run_strategy import main as run_strategy_main

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Canonical evaluation window
CANONICAL_START = "2020-01-06"
CANONICAL_END = "2025-10-31"

# Crisis sleeve weight
CRISIS_SLEEVE_WEIGHT = 0.05  # 5% of portfolio capital

# Core v9 baseline profile
CORE_V9_PROFILE = "core_v9_trend_csmom_vrp_core_convergence_vrp_alt_vx_carry_sr3_curverv_no_macro"


def run_strategy_profile(profile_name: str, run_id: str, start_date: str, end_date: str):
    """Run a strategy profile and save results."""
    logger.info("=" * 80)
    logger.info(f"Running strategy profile: {profile_name}")
    logger.info(f"Run ID: {run_id}")
    logger.info(f"Date Range: {start_date} to {end_date}")
    logger.info("=" * 80)
    
    original_argv = sys.argv.copy()
    sys.argv = [
        "run_strategy.py",
        "--strategy_profile", profile_name,
        "--run_id", run_id,
        "--start", start_date,
        "--end", end_date
    ]
    
    try:
        run_strategy_main()
        logger.info(f"✓ Completed run: {run_id}")
    except Exception as e:
        logger.error(f"✗ Failed run: {run_id}")
        logger.error(f"Error: {e}")
        raise
    finally:
        sys.argv = original_argv


def load_run_returns(run_id: str) -> pd.Series:
    """Load portfolio returns from a run."""
    if Path(run_id).exists() and Path(run_id).is_dir():
        run_dir = Path(run_id)
    else:
        run_dir = Path(f"reports/runs/{run_id}")
        if not run_dir.exists():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")
    
    returns_file = run_dir / "portfolio_returns.csv"
    if not returns_file.exists():
        raise FileNotFoundError(f"Portfolio returns file not found: {returns_file}")
    
    df = pd.read_csv(returns_file, index_col=0, parse_dates=True)
    if 'ret' in df.columns:
        returns = df['ret']
    else:
        returns = df.iloc[:, 0]
    
    returns.name = 'portfolio_return'
    return returns


def load_crisis_instruments(
    con: duckdb.DuckDBPyConnection,
    start: str,
    end: str
) -> Dict[str, pd.Series]:
    """
    Load returns for crisis instruments: VX1, VX2, UB, ZN.
    
    Returns:
        Dict with keys: 'vx1', 'vx2', 'ub', 'zn'
        Each value is a Series of daily log returns indexed by date
    """
    # Load VX curve
    vx_df = load_vx_curve(con, start, end, VX_FRONT_SYMBOL, VX_SECOND_SYMBOL)
    vx_df = vx_df.set_index('date')
    
    # Compute VX returns
    vx1_returns = np.log(vx_df['vx1']).diff().dropna()
    vx2_returns = np.log(vx_df['vx2']).diff().dropna()
    
    # Discover the OHLCV table name (same as MarketData broker)
    from src.agents.utils_db import find_ohlcv_table
    table_name = find_ohlcv_table(con)
    
    # Detect date column name
    conn_type = type(con).__module__
    if 'duckdb' in conn_type:
        result = con.execute(f"DESCRIBE {table_name}").fetchall()
        columns = {row[0].lower(): row[0] for row in result}
    else:
        cursor = con.cursor()
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = {row[1].lower(): row[1] for row in cursor.fetchall()}
    
    # Map date column
    if 'date' in columns:
        date_col = columns['date']
    elif 'trading_date' in columns:
        date_col = columns['trading_date']
    elif 'timestamp' in columns:
        date_col = columns['timestamp']
    else:
        raise ValueError(f"Could not find date column in {table_name}")
    
    # Map symbol column
    if 'symbol' in columns:
        symbol_col = columns['symbol']
    elif 'contract_series' in columns:
        symbol_col = columns['contract_series']
    else:
        raise ValueError(f"Could not find symbol column in {table_name}")
    
    # Load UB (30-year) and ZN (10-year) returns
    ub_result = con.execute(
        f"""
        SELECT {date_col}::DATE AS date, close::DOUBLE AS close
        FROM {table_name}
        WHERE {symbol_col} = 'UB_FRONT_VOLUME'
          AND {date_col}::DATE BETWEEN ? AND ?
        ORDER BY {date_col}
        """,
        [start, end]
    ).df()
    
    zn_result = con.execute(
        f"""
        SELECT {date_col}::DATE AS date, close::DOUBLE AS close
        FROM {table_name}
        WHERE {symbol_col} = 'ZN_FRONT_VOLUME'
          AND {date_col}::DATE BETWEEN ? AND ?
        ORDER BY {date_col}
        """,
        [start, end]
    ).df()
    
    ub_returns = None
    zn_returns = None
    
    if not ub_result.empty:
        ub_result = ub_result.set_index('date')
        ub_returns = np.log(ub_result['close']).diff().dropna()
    
    if not zn_result.empty:
        zn_result = zn_result.set_index('date')
        zn_returns = np.log(zn_result['close']).diff().dropna()
    
    return {
        'vx1': vx1_returns,
        'vx2': vx2_returns,
        'ub': ub_returns,
        'zn': zn_returns
    }


def compute_crisis_sleeve_returns(
    instruments: Dict[str, pd.Series],
    sleeve_type: str
) -> pd.Series:
    """
    Compute crisis sleeve returns for a given sleeve type.
    
    Args:
        instruments: Dict of instrument returns (from load_crisis_instruments)
        sleeve_type: 'vx2_long', 'vx_spread', or 'duration'
    
    Returns:
        Series of daily returns for the crisis sleeve
    """
    if sleeve_type == 'vx2_long':
        # Long VX2: constant long position
        return instruments['vx2'].copy()
    
    elif sleeve_type == 'vx_spread':
        # Long VX2 - Short VX1: dollar-neutral spread
        # Align dates
        common_dates = instruments['vx2'].index.intersection(instruments['vx1'].index)
        vx2_aligned = instruments['vx2'].loc[common_dates]
        vx1_aligned = instruments['vx1'].loc[common_dates]
        # Spread return = VX2 return - VX1 return
        return (vx2_aligned - vx1_aligned).dropna()
    
    elif sleeve_type == 'duration':
        # Long Duration: prefer UB, fallback to ZN
        if instruments['ub'] is not None and not instruments['ub'].empty:
            return instruments['ub'].copy()
        elif instruments['zn'] is not None and not instruments['zn'].empty:
            return instruments['zn'].copy()
        else:
            raise ValueError("Neither UB nor ZN data available for duration sleeve")
    
    else:
        raise ValueError(f"Unknown sleeve type: {sleeve_type}")


def compute_crisis_metrics(returns: pd.Series) -> Dict:
    """
    Compute crisis-specific metrics.
    
    Args:
        returns: Portfolio returns Series
    
    Returns:
        Dict with crisis metrics
    """
    if returns.empty:
        return {}
    
    equity = (1 + returns).cumprod()
    n_days = len(returns)
    n_years = n_days / 252.0
    
    # Basic metrics
    total_ret = equity.iloc[-1] / equity.iloc[0] - 1.0 if len(equity) > 0 else 0.0
    cagr = (1 + total_ret) ** (1 / n_years) - 1 if n_years > 0 else 0.0
    vol = returns.std() * np.sqrt(252)
    sharpe = (returns.mean() * 252) / vol if vol > 0 else 0.0
    
    # Drawdown metrics
    running_max = equity.expanding().max()
    drawdown = (equity - running_max) / running_max
    max_drawdown = drawdown.min()
    
    # Worst periods
    monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
    worst_month = monthly_returns.min()
    
    quarterly_returns = returns.resample('Q').apply(lambda x: (1 + x).prod() - 1)
    worst_quarter = quarterly_returns.min()
    
    # Worst 10-day window
    rolling_10d = returns.rolling(10).apply(lambda x: (1 + x).prod() - 1)
    worst_10d = rolling_10d.min()
    
    # Crisis period attribution
    crisis_periods = {
        "2020_Q1": ("2020-01-01", "2020-03-31"),
        "2022_drawdown": ("2022-01-01", "2022-12-31"),
        "2023_vol_spike": ("2023-01-01", "2023-12-31"),
        "2024_vol_spike": ("2024-01-01", "2024-12-31"),
    }
    
    crisis_attribution = {}
    for name, (start, end) in crisis_periods.items():
        period_returns = returns[(returns.index >= start) & (returns.index <= end)]
        if len(period_returns) > 0:
            period_equity = (1 + period_returns).cumprod()
            period_total_ret = period_equity.iloc[-1] / period_equity.iloc[0] - 1.0
            period_dd = ((period_equity / period_equity.expanding().max()) - 1).min()
            crisis_attribution[name] = {
                'total_return': float(period_total_ret),
                'max_drawdown': float(period_dd),
                'n_days': len(period_returns)
            }
        else:
            crisis_attribution[name] = {'n_days': 0}
    
    return {
        'cagr': float(cagr),
        'sharpe': float(sharpe),
        'volatility': float(vol),
        'max_drawdown': float(max_drawdown),
        'worst_month': float(worst_month),
        'worst_quarter': float(worst_quarter),
        'worst_10d': float(worst_10d),
        'crisis_attribution': crisis_attribution,
        'n_days': n_days,
        'n_years': n_years
    }


def test_crisis_sleeve(
    sleeve_name: str,
    sleeve_type: str,
    core_v9_returns: pd.Series,
    instruments: Dict[str, pd.Series],
    start: str,
    end: str,
    output_base: Path
) -> Dict:
    """
    Test a single crisis sleeve against Core v9 baseline.
    
    Args:
        sleeve_name: Name of the sleeve (e.g., 'vx2', 'vx_spread', 'duration')
        sleeve_type: Type identifier ('vx2_long', 'vx_spread', 'duration')
        core_v9_returns: Core v9 baseline returns
        instruments: Dict of instrument returns
        start: Start date
        end: End date
        output_base: Base output directory
    
    Returns:
        Dict with test results
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"Testing Crisis Sleeve: {sleeve_name}")
    logger.info(f"{'='*80}")
    
    # Compute crisis sleeve returns
    crisis_returns = compute_crisis_sleeve_returns(instruments, sleeve_type)
    
    # Align dates with Core v9
    common_dates = core_v9_returns.index.intersection(crisis_returns.index)
    if len(common_dates) == 0:
        raise ValueError(f"No overlapping dates between Core v9 and {sleeve_name} sleeve")
    
    core_aligned = core_v9_returns.loc[common_dates]
    crisis_aligned = crisis_returns.loc[common_dates]
    
    # Combine: Core v9 (95%) + Crisis Sleeve (5%)
    combined_returns = (1 - CRISIS_SLEEVE_WEIGHT) * core_aligned + CRISIS_SLEEVE_WEIGHT * crisis_aligned
    
    # Compute metrics
    baseline_metrics = compute_crisis_metrics(core_aligned)
    combined_metrics = compute_crisis_metrics(combined_returns)
    
    # Compute differences
    maxdd_diff = combined_metrics['max_drawdown'] - baseline_metrics['max_drawdown']
    worst_month_diff = combined_metrics['worst_month'] - baseline_metrics['worst_month']
    worst_quarter_diff = combined_metrics['worst_quarter'] - baseline_metrics['worst_quarter']
    worst_10d_diff = combined_metrics['worst_10d'] - baseline_metrics['worst_10d']
    cagr_diff = combined_metrics['cagr'] - baseline_metrics['cagr']
    
    # Phase-0 pass criteria
    maxdd_pass = maxdd_diff >= 0.01  # ≥ 1.0% improvement
    worst_month_pass = worst_month_diff > 0  # Improvement
    pass_criteria = maxdd_pass or worst_month_pass
    no_catastrophic_bleed = combined_metrics['cagr'] > -0.02  # > -2% CAGR
    
    overall_pass = pass_criteria and no_catastrophic_bleed
    
    logger.info(f"\nBaseline (Core v9) Metrics:")
    logger.info(f"  MaxDD: {baseline_metrics['max_drawdown']:.4f}")
    logger.info(f"  Worst Month: {baseline_metrics['worst_month']:.4f}")
    logger.info(f"  CAGR: {baseline_metrics['cagr']:.4f}")
    
    logger.info(f"\nCombined (Core v9 + {sleeve_name}) Metrics:")
    logger.info(f"  MaxDD: {combined_metrics['max_drawdown']:.4f} (diff: {maxdd_diff:+.4f})")
    logger.info(f"  Worst Month: {combined_metrics['worst_month']:.4f} (diff: {worst_month_diff:+.4f})")
    logger.info(f"  CAGR: {combined_metrics['cagr']:.4f} (diff: {cagr_diff:+.4f})")
    
    logger.info(f"\nPhase-0 Pass Criteria:")
    logger.info(f"  MaxDD improvement ≥ 1.0%: {maxdd_pass} ({maxdd_diff:+.4f})")
    logger.info(f"  Worst-month improvement: {worst_month_pass} ({worst_month_diff:+.4f})")
    logger.info(f"  No catastrophic bleed: {no_catastrophic_bleed} (CAGR: {combined_metrics['cagr']:.4f})")
    logger.info(f"  Overall Pass: {overall_pass}")
    
    # Save results
    run_id = f"core_v9_crisis_{sleeve_name}_phase0"
    output_dir = output_base / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save returns
    returns_df = pd.DataFrame({
        'baseline': core_aligned,
        'crisis_sleeve': crisis_aligned,
        'combined': combined_returns
    })
    returns_df.to_csv(output_dir / "returns.csv")
    
    # Save equity curves
    equity_df = pd.DataFrame({
        'baseline': (1 + core_aligned).cumprod(),
        'combined': (1 + combined_returns).cumprod()
    })
    equity_df.to_csv(output_dir / "equity.csv")
    
    # Save metrics
    results = {
        'run_id': run_id,
        'sleeve_name': sleeve_name,
        'sleeve_type': sleeve_type,
        'crisis_weight': CRISIS_SLEEVE_WEIGHT,
        'start_date': start,
        'end_date': end,
        'baseline_metrics': baseline_metrics,
        'combined_metrics': combined_metrics,
        'differences': {
            'maxdd_diff': float(maxdd_diff),
            'worst_month_diff': float(worst_month_diff),
            'worst_quarter_diff': float(worst_quarter_diff),
            'worst_10d_diff': float(worst_10d_diff),
            'cagr_diff': float(cagr_diff)
        },
        'pass_criteria': {
            'maxdd_pass': bool(maxdd_pass),
            'worst_month_pass': bool(worst_month_pass),
            'no_catastrophic_bleed': bool(no_catastrophic_bleed),
            'overall_pass': bool(overall_pass)
        }
    }
    
    with open(output_dir / "metrics.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # Register in phase index
    phase_index_dir = Path("reports/phase_index/crisis") / sleeve_name
    phase_index_dir.mkdir(parents=True, exist_ok=True)
    
    phase0_file = phase_index_dir / "phase0.txt"
    with open(phase0_file, 'w') as f:
        f.write(f"# Phase-0: Crisis Sleeve {sleeve_name}\n")
        f.write(f"# Registered: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"start_date: {start}\n")
        f.write(f"end_date: {end}\n")
        f.write(f"crisis_weight: {CRISIS_SLEEVE_WEIGHT}\n")
        f.write(f"maxdd_diff: {maxdd_diff:.4f}\n")
        f.write(f"worst_month_diff: {worst_month_diff:.4f}\n")
        f.write(f"cagr_diff: {cagr_diff:.4f}\n")
        f.write(f"overall_pass: {overall_pass}\n")
        f.write(f"path: {output_dir}\n")
    
    logger.info(f"\n✓ Saved results to: {output_dir}")
    logger.info(f"✓ Registered in: {phase0_file}")
    
    return results


def main():
    # Load canonical window as default
    CANONICAL_START_DEFAULT, CANONICAL_END_DEFAULT = load_canonical_window()
    
    parser = argparse.ArgumentParser(description="Crisis Meta-Sleeve Phase-0 Diagnostics")
    parser.add_argument("--start", type=str, default=CANONICAL_START,
                       help=f"Start date (YYYY-MM-DD). Default: {CANONICAL_START}")
    parser.add_argument("--end", type=str, default=CANONICAL_END,
                       help=f"End date (YYYY-MM-DD). Default: {CANONICAL_END}")
    parser.add_argument("--core-v9-run-id", type=str, default=None,
                       help="Existing Core v9 run ID (if not provided, will run baseline)")
    
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("Crisis Meta-Sleeve Phase-0 Diagnostics")
    logger.info("=" * 80)
    logger.info(f"Date Range: {args.start} to {args.end}")
    logger.info(f"Crisis Sleeve Weight: {CRISIS_SLEEVE_WEIGHT*100:.1f}%")
    
    # Step 1: Run or load Core v9 baseline
    if args.core_v9_run_id:
        logger.info(f"\nLoading Core v9 baseline from run ID: {args.core_v9_run_id}")
        core_v9_returns = load_run_returns(args.core_v9_run_id)
    else:
        logger.info(f"\nRunning Core v9 baseline...")
        core_v9_run_id = f"core_v9_baseline_phase0_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        run_strategy_profile(CORE_V9_PROFILE, core_v9_run_id, args.start, args.end)
        core_v9_returns = load_run_returns(core_v9_run_id)
    
    # Step 2: Load crisis instruments
    logger.info("\nLoading crisis instruments (VX1, VX2, UB, ZN)...")
    # Get database path from config
    import yaml
    config_path = Path("configs/data.yaml")
    db_path = None
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            data_config = yaml.safe_load(f)
        db_path = data_config.get('db', {}).get('path')
    
    if not db_path:
        raise ValueError("Database path not found in configs/data.yaml")
    
    con = open_readonly_connection(db_path)
    try:
        instruments = load_crisis_instruments(con, args.start, args.end)
        logger.info(f"  VX1: {len(instruments['vx1'])} days")
        logger.info(f"  VX2: {len(instruments['vx2'])} days")
        logger.info(f"  UB: {len(instruments['ub']) if instruments['ub'] is not None else 0} days")
        logger.info(f"  ZN: {len(instruments['zn']) if instruments['zn'] is not None else 0} days")
    finally:
        con.close()
    
    # Step 3: Test each crisis sleeve
    output_base = Path("reports/diagnostics/crisis_phase0")
    output_base.mkdir(parents=True, exist_ok=True)
    
    sleeves = [
        ('vx2', 'vx2_long', 'Long VX2'),
        ('vx_spread', 'vx_spread', 'Long VX2 - VX1 Spread'),
        ('duration', 'duration', 'Long Duration (UB/ZN)')
    ]
    
    all_results = {}
    for sleeve_name, sleeve_type, description in sleeves:
        try:
            results = test_crisis_sleeve(
                sleeve_name=sleeve_name,
                sleeve_type=sleeve_type,
                core_v9_returns=core_v9_returns,
                instruments=instruments,
                start=args.start,
                end=args.end,
                output_base=output_base
            )
            all_results[sleeve_name] = results
        except Exception as e:
            logger.error(f"Failed to test {sleeve_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("Phase-0 Summary")
    logger.info("=" * 80)
    for sleeve_name, results in all_results.items():
        pass_status = "PASS" if results['pass_criteria']['overall_pass'] else "FAIL"
        logger.info(f"{sleeve_name:15s}: {pass_status:4s} | "
                   f"MaxDD diff: {results['differences']['maxdd_diff']:+.4f} | "
                   f"CAGR diff: {results['differences']['cagr_diff']:+.4f}")
    
    logger.info(f"\n✓ All results saved to: {output_base}")


if __name__ == "__main__":
    main()

