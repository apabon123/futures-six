#!/usr/bin/env python3
"""
Regenerate attribution artifacts from existing run artifacts.

For runs that already completed (before attribution was integrated into
ExecSim), this script reconstructs sleeve weight decomposition from
available artifacts using the sleeve-universe orthogonality property:

- Non-VRP sleeves (tsmom, carry, etc.) trade ONLY non-VX instruments.
- VRP sleeves (vrp_core_meta, vrp_convergence_meta, vrp_alt_meta) trade
  ONLY VX1, VX2, VX3.
- This is enforced in CombinedStrategy.

Because these universes are disjoint, the weight decomposition is exact:
  - Non-VRP sleeve weight on non-VX asset = 100% of that asset's final weight
  - VRP sleeves share VX weight proportionally (estimated from sleeve_returns.csv
    or from configured sleeve weights).

Usage:
    python scripts/analysis/regenerate_attribution.py --run_id <run_id>
    python scripts/analysis/regenerate_attribution.py --run_id vrp_canonical_2020_2024_20260212_180537
"""

import sys
import json
import logging
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure project root on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.attribution.core import CONSISTENCY_TOL, compute_attribution
from src.attribution.artifacts import generate_attribution_artifacts

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

VX_SYMBOLS = {"VX1", "VX2", "VX3"}
VRP_SLEEVES = {"vrp_core_meta", "vrp_convergence_meta", "vrp_alt_meta"}


def reconstruct_sleeve_signals_from_artifacts(
    run_dir: Path,
    weights_panel: pd.DataFrame,
    asset_returns_simple: pd.DataFrame,
    universe: list,
) -> dict:
    """
    Reconstruct sleeve_signals_history from existing artifacts using
    the disjoint-universe property.

    Returns:
        sleeve_signals_history dict compatible with compute_attribution().
    """
    # Read sleeve_returns.csv to identify which sleeves exist
    sleeve_returns_path = run_dir / "sleeve_returns.csv"
    if not sleeve_returns_path.exists():
        logger.error(f"sleeve_returns.csv not found in {run_dir}")
        return {}

    sleeve_returns = pd.read_csv(sleeve_returns_path, index_col=0, parse_dates=True)
    sleeve_names = list(sleeve_returns.columns)
    logger.info(f"Found sleeves: {sleeve_names}")

    # Classify sleeves
    vrp_sleeves = [s for s in sleeve_names if s in VRP_SLEEVES]
    non_vrp_sleeves = [s for s in sleeve_names if s not in VRP_SLEEVES]
    vx_cols = [c for c in universe if c in VX_SYMBOLS]
    non_vx_cols = [c for c in universe if c not in VX_SYMBOLS]

    logger.info(f"VRP sleeves: {vrp_sleeves}")
    logger.info(f"Non-VRP sleeves: {non_vrp_sleeves}")
    logger.info(f"VX instruments: {vx_cols}")
    logger.info(f"Non-VX instruments ({len(non_vx_cols)}): {non_vx_cols[:5]}...")

    # Build sleeve signals history
    # For each rebalance date, construct synthetic weighted signals that produce
    # the correct weight decomposition.
    rebalance_dates = weights_panel.index
    sleeve_signals_history = {s: [] for s in sleeve_names}

    # For VRP weight splitting: use sleeve_returns magnitude as proxy
    # If sleeve_returns are all zero for VRP, split by equal weight
    vrp_daily_abs = {}
    for s in vrp_sleeves:
        vrp_daily_abs[s] = sleeve_returns[s].abs()

    for date in rebalance_dates:
        final_w = weights_panel.loc[date].reindex(universe, fill_value=0.0).fillna(0.0)

        # Non-VRP sleeves: each gets full weight on non-VX instruments
        # If there's only one non-VRP sleeve, it gets everything.
        # If multiple, we'd need to split. For now, sum them equally
        # (in the canonical run there's only tsmom).
        if len(non_vrp_sleeves) == 1:
            sleeve_name = non_vrp_sleeves[0]
            sig = pd.Series(0.0, index=universe)
            for col in non_vx_cols:
                w_val = final_w.get(col, 0.0)
                if abs(w_val) > 1e-15:
                    # Set signal such that sleeve_weight * signal = w_val (as combined signal)
                    # But since this is a synthetic signal for decomposition, we just need
                    # the proportions to be correct. Set signal = final_weight directly.
                    sig[col] = w_val
            sleeve_signals_history[sleeve_name].append((date, sig))
        else:
            # Multiple non-VRP sleeves: split proportionally by sleeve_returns
            for sleeve_name in non_vrp_sleeves:
                sig = pd.Series(0.0, index=universe)
                # Use sleeve_returns as proxy for weight fraction
                # Find nearest earlier date in sleeve_returns
                valid_dates = sleeve_returns.index[sleeve_returns.index <= date]
                if len(valid_dates) > 0:
                    nearest = valid_dates[-1]
                    total_abs = sum(
                        sleeve_returns.loc[nearest, s2] ** 2
                        for s2 in non_vrp_sleeves
                        if s2 in sleeve_returns.columns
                    )
                    if total_abs > 1e-18:
                        frac = sleeve_returns.loc[nearest, sleeve_name] ** 2 / total_abs
                    else:
                        frac = 1.0 / len(non_vrp_sleeves)
                else:
                    frac = 1.0 / len(non_vrp_sleeves)

                for col in non_vx_cols:
                    sig[col] = final_w.get(col, 0.0) * frac
                sleeve_signals_history[sleeve_name].append((date, sig))

        # VRP sleeves: share VX weight
        if vrp_sleeves:
            # Try to use sleeve_returns around this date for proportional split
            valid_dates = sleeve_returns.index[sleeve_returns.index <= date]
            # Look at recent window for activity
            if len(valid_dates) >= 5:
                recent = sleeve_returns.loc[valid_dates[-20:], vrp_sleeves]
                total_abs = recent.abs().sum()
            else:
                total_abs = pd.Series(0.0, index=vrp_sleeves)

            total_vrp_activity = total_abs.sum()
            if total_vrp_activity > 1e-15:
                vrp_fractions = total_abs / total_vrp_activity
            else:
                # Equal split if no recent activity
                vrp_fractions = pd.Series(1.0 / len(vrp_sleeves), index=vrp_sleeves)

            for sleeve_name in vrp_sleeves:
                sig = pd.Series(0.0, index=universe)
                frac = vrp_fractions.get(sleeve_name, 0.0)
                for col in vx_cols:
                    sig[col] = final_w.get(col, 0.0) * frac
                sleeve_signals_history[sleeve_name].append((date, sig))

    return sleeve_signals_history


def main():
    parser = argparse.ArgumentParser(description="Regenerate attribution from existing run artifacts")
    parser.add_argument("--run_id", required=True, help="Run ID to regenerate attribution for")
    parser.add_argument("--run_dir_base", default="reports/runs", help="Base directory for run artifacts")
    args = parser.parse_args()

    run_dir = PROJECT_ROOT / args.run_dir_base / args.run_id

    if not run_dir.exists():
        logger.error(f"Run directory not found: {run_dir}")
        return 1

    logger.info(f"Regenerating attribution for run: {args.run_id}")
    logger.info(f"Run directory: {run_dir}")

    # Load artifacts
    weights_panel = pd.read_csv(run_dir / "weights.csv", index_col=0, parse_dates=True)
    asset_returns_simple = pd.read_csv(run_dir / "asset_returns.csv", index_col=0, parse_dates=True)
    portfolio_returns = pd.read_csv(run_dir / "portfolio_returns.csv", index_col=0, parse_dates=True)
    portfolio_simple = portfolio_returns.iloc[:, 0]  # First column is 'ret'

    # Fill NaN weights with 0
    weights_panel = weights_panel.fillna(0.0)

    # Read meta for universe
    with open(run_dir / "meta.json") as f:
        meta = json.load(f)
    universe = meta.get("universe", list(weights_panel.columns))

    logger.info(f"Universe: {universe}")
    logger.info(f"Weights: {weights_panel.shape}, Asset returns: {asset_returns_simple.shape}")

    # Reconstruct sleeve signals from artifacts
    sleeve_signals_history = reconstruct_sleeve_signals_from_artifacts(
        run_dir=run_dir,
        weights_panel=weights_panel,
        asset_returns_simple=asset_returns_simple,
        universe=universe,
    )

    if not sleeve_signals_history:
        logger.error("Failed to reconstruct sleeve signals")
        return 1

    # Build metasleeve mapping
    metasleeve_map = {}
    for sname in sleeve_signals_history.keys():
        if sname in VRP_SLEEVES:
            metasleeve_map[sname] = "vrp_combined"
        else:
            metasleeve_map[sname] = sname

    # Compute attribution
    result = compute_attribution(
        weights_panel=weights_panel,
        asset_returns_simple=asset_returns_simple,
        portfolio_returns_simple=portfolio_simple,
        sleeve_signals_history=sleeve_signals_history,
        universe=universe,
        metasleeve_mapping=metasleeve_map,
    )

    diag = result["diagnostics"]
    logger.info(f"Consistency pass: {diag['consistency_pass']}")
    logger.info(f"Max daily residual: {diag['max_abs_daily_residual_active']:.2e}")
    logger.info(f"Cumulative residual: {diag['cum_residual']:.2e}")

    # Generate artifacts
    out_dir = generate_attribution_artifacts(
        attribution_result=result,
        portfolio_returns_simple=portfolio_simple,
        run_dir=run_dir,
        run_id=args.run_id,
        metasleeve_mapping=metasleeve_map,
    )

    # Save decomposition
    decomp = result.get("sleeve_weight_decomposition")
    if decomp is not None and not decomp.empty:
        decomp.to_csv(run_dir / "sleeve_weight_decomposition.csv")

    logger.info(f"\nAttribution artifacts generated in: {out_dir}")
    logger.info(f"  attribution_by_metasleeve.csv")
    logger.info(f"  attribution_by_atomic_sleeve.csv")
    logger.info(f"  attribution_summary.json")
    logger.info(f"  ATTRIBUTION_SUMMARY.md")

    # Print summary
    with open(out_dir / "attribution_summary.json") as f:
        summary = json.load(f)

    print("\n" + "=" * 70)
    print("ATTRIBUTION SUMMARY")
    print("=" * 70)
    print(f"Portfolio cumulative return: {summary['total_portfolio_cum_return']:.4%}")
    print(f"Sum sleeve contributions:    {summary['sum_sleeve_cum_return']:.4%}")
    print(f"Residual:                    {summary['residual_cum_return']:.2e}")
    print(f"Consistency:                 {'PASS' if summary['consistency_check']['passed'] else 'FAIL'}")
    print()

    for sleeve, metrics in sorted(summary["per_sleeve"].items(), key=lambda x: x[1]["cum_return"], reverse=True):
        print(f"  {sleeve:30s}: cum={metrics['cum_return']:+.4%}  "
              f"vol={metrics['vol_of_contrib']:.4%}  "
              f"hit={metrics['hit_rate']:.1%}  "
              f"active={metrics['active_days']}/{metrics['active_days'] + metrics['no_signal_days']}")
    print("=" * 70)

    return 0 if diag["consistency_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
