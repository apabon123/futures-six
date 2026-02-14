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
# Sleeves that trade only VX instruments (get VX weight in decomposition, not non-VX)
VX_ONLY_SLEEVES = {"vx_calendar_carry"}


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
    # Canonical source: sleeve_returns.csv. If present, use its columns as the sleeve list.
    sleeve_returns_path = run_dir / "sleeve_returns.csv"
    if not sleeve_returns_path.exists():
        logger.error(f"sleeve_returns.csv not found in {run_dir}")
        return {}

    sleeve_returns = pd.read_csv(sleeve_returns_path, index_col=0, parse_dates=True)
    # Normalize column names (e.g. date might be index; ensure we have sleeve columns only)
    sleeve_names = [c for c in sleeve_returns.columns if c and str(c).strip() and str(c) != "date"]
    if not sleeve_names:
        sleeve_names = list(sleeve_returns.columns)
    logger.info(f"Canonical sleeves from sleeve_returns.csv: {sleeve_names}")

    vx_cols = [c for c in universe if c in VX_SYMBOLS]
    non_vx_cols = [c for c in universe if c not in VX_SYMBOLS]

    # Classify: VRP (VX weight), VX-only non-VRP (vx_calendar_carry -> VX weight), rest (non-VX weight)
    vrp_sleeves = [s for s in sleeve_names if s in VRP_SLEEVES]
    vx_only_sleeves = [s for s in sleeve_names if s in VX_ONLY_SLEEVES]
    non_vx_sleeves = [s for s in sleeve_names if s not in VRP_SLEEVES and s not in VX_ONLY_SLEEVES]

    logger.info(f"VRP sleeves (VX weight): {vrp_sleeves}")
    logger.info(f"VX-only sleeves (VX weight): {vx_only_sleeves}")
    logger.info(f"Non-VX sleeves (non-VX weight): {non_vx_sleeves}")
    logger.info(f"VX instruments: {vx_cols}, Non-VX count: {len(non_vx_cols)}")

    rebalance_dates = weights_panel.index
    sleeve_signals_history = {s: [] for s in sleeve_names}

    def _sleeve_frac_at_date(sleeves_subset: list, sleeve_name: str, date) -> float:
        """Fraction for one sleeve among sleeves_subset using sleeve_returns at date."""
        valid_dates = sleeve_returns.index[sleeve_returns.index <= date]
        if len(valid_dates) == 0 or sleeve_name not in sleeve_returns.columns:
            return 1.0 / len(sleeves_subset) if sleeves_subset else 0.0
        nearest = valid_dates[-1]
        total_abs = sum(
            sleeve_returns.loc[nearest, s2] ** 2
            for s2 in sleeves_subset
            if s2 in sleeve_returns.columns
        )
        if total_abs > 1e-18:
            return sleeve_returns.loc[nearest, sleeve_name] ** 2 / total_abs
        return 1.0 / len(sleeves_subset) if sleeves_subset else 0.0

    for date in rebalance_dates:
        final_w = weights_panel.loc[date].reindex(universe, fill_value=0.0).fillna(0.0)

        # Non-VX weight: split among non_vx_sleeves only (tsmom, csmom, sr3_curve_rv_meta, etc.)
        if len(non_vx_sleeves) == 1:
            sleeve_name = non_vx_sleeves[0]
            sig = pd.Series(0.0, index=universe)
            for col in non_vx_cols:
                w_val = final_w.get(col, 0.0)
                if abs(w_val) > 1e-15:
                    sig[col] = w_val
            sleeve_signals_history[sleeve_name].append((date, sig))
        elif non_vx_sleeves:
            for sleeve_name in non_vx_sleeves:
                sig = pd.Series(0.0, index=universe)
                frac = _sleeve_frac_at_date(non_vx_sleeves, sleeve_name, date)
                for col in non_vx_cols:
                    sig[col] = final_w.get(col, 0.0) * frac
                sleeve_signals_history[sleeve_name].append((date, sig))

        # VX weight: split among vrp_sleeves + vx_only_sleeves
        vx_sleeves = vrp_sleeves + vx_only_sleeves
        if vx_sleeves:
            for sleeve_name in vx_sleeves:
                sig = pd.Series(0.0, index=universe)
                frac = _sleeve_frac_at_date(vx_sleeves, sleeve_name, date)
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
