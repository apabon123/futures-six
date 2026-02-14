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


def load_sleeve_weights(run_dir: Path) -> dict:
    """Load sleeve_weights from run_dir/analysis/sleeve_weights.json if present."""
    path = run_dir / "analysis" / "sleeve_weights.json"
    if not path.exists():
        return {}
    with open(path) as f:
        data = json.load(f)
    return data.get("sleeve_weights", data) if isinstance(data.get("sleeve_weights"), dict) else (data if isinstance(data, dict) else {})


def ensure_sleeve_weights_artifact(run_dir: Path, sleeve_names: list) -> dict:
    """
    Ensure run_dir/analysis/sleeve_weights.json exists. Build from meta.json if missing.
    Returns sleeve_weights dict (sleeve_name -> weight) for sleeves in sleeve_names only.
    """
    path = run_dir / "analysis" / "sleeve_weights.json"
    existing = load_sleeve_weights(run_dir)
    if existing:
        return {k: v for k, v in existing.items() if k in sleeve_names}

    meta_path = run_dir / "meta.json"
    if not meta_path.exists():
        return {}

    with open(meta_path) as f:
        meta = json.load(f)
    strategies = meta.get("config", {}).get("strategies", {})
    sleeve_weights = {}
    for name, cfg in strategies.items():
        if not isinstance(cfg, dict):
            continue
        if cfg.get("enabled") and name in sleeve_names:
            w = cfg.get("weight", 0)
            if w is not None and float(w) > 0:
                sleeve_weights[name] = float(w)

    if not sleeve_weights:
        return {}

    (run_dir / "analysis").mkdir(parents=True, exist_ok=True)
    payload = {"sleeve_weights": sleeve_weights, "source": "meta.json"}
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info(f"Wrote {path} with sleeves: {list(sleeve_weights.keys())}")
    return sleeve_weights


def compute_attribution_return_based(
    sleeve_weights: dict,
    sleeve_returns: pd.DataFrame,
    portfolio_simple: pd.Series,
    metasleeve_mapping: dict,
) -> dict:
    """
    Attribution from config sleeve weights and sleeve returns: contrib_s = w_s * r_s, scaled so sum = portfolio.
    Returns same shape as compute_attribution() for generate_attribution_artifacts().
    """
    # Align to common index
    all_dates = portfolio_simple.index.intersection(sleeve_returns.index).sort_values()
    port = portfolio_simple.reindex(all_dates).fillna(0.0)
    sr = sleeve_returns.reindex(all_dates).fillna(0.0)

    sleeve_names = [s for s in sleeve_returns.columns if s in sleeve_weights]
    if not sleeve_names:
        sleeve_names = list(sleeve_returns.columns)
        # Use equal weights if no config weights
        total = len(sleeve_names)
        sleeve_weights = {s: 1.0 / total for s in sleeve_names}

    # Raw contribution: w_s * r_s
    raw = pd.DataFrame(index=all_dates, columns=sleeve_names, dtype=float)
    for s in sleeve_names:
        w = sleeve_weights.get(s, 0.0)
        raw[s] = w * sr[s]

    denom = raw.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = np.where(np.abs(denom.values) > 1e-18, port.values / denom.values, 1.0)
    atomic_contributions = raw.mul(scale, axis=0)

    # Metasleeve
    metasleeve_names = sorted(set(metasleeve_mapping.get(s, s) for s in sleeve_names))
    meta_contribs = pd.DataFrame(0.0, index=all_dates, columns=metasleeve_names)
    for s in sleeve_names:
        m = metasleeve_mapping.get(s, s)
        if m in meta_contribs.columns:
            meta_contribs[m] = meta_contribs[m].add(atomic_contributions[s], fill_value=0.0)

    residual = atomic_contributions.sum(axis=1) - port
    max_residual = residual.abs().max()
    first_rebal = all_dates[0]
    active_mask = all_dates >= first_rebal
    no_signal_info = {}
    for s in sleeve_names:
        c = atomic_contributions[s]
        zero_days = (c[active_mask].abs() < 1e-15).sum()
        no_signal_info[s] = {"active_days": int(active_mask.sum() - zero_days), "no_signal_days": int(zero_days), "total_days": int(active_mask.sum())}

    diagnostics = {
        "consistency_pass": max_residual < 1e-8,
        "tolerance": 1e-8,
        "max_abs_daily_residual": max_residual,
        "max_abs_daily_residual_active": float(residual[active_mask].abs().max()) if active_mask.any() else 0.0,
        "cum_residual": float(residual.sum()),
        "n_active_days": int(active_mask.sum()),
        "sleeves_captured": list(sleeve_names),
        "no_signal_info": no_signal_info,
        "partial_attribution": False,
        "first_rebalance_date": str(first_rebal),
    }

    return {
        "atomic_contributions": atomic_contributions,
        "metasleeve_contributions": meta_contribs,
        "sleeve_weight_decomposition": pd.DataFrame(),
        "diagnostics": diagnostics,
    }


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

    # Sleeve list and metasleeve mapping (from sleeve_returns if present)
    sleeve_returns_path = run_dir / "sleeve_returns.csv"
    if sleeve_returns_path.exists():
        sleeve_returns_df = pd.read_csv(sleeve_returns_path, index_col=0, parse_dates=True)
        sleeve_names_from_csv = [c for c in sleeve_returns_df.columns if c and str(c).strip() and str(c) != "date"]
        if not sleeve_names_from_csv:
            sleeve_names_from_csv = list(sleeve_returns_df.columns)
    else:
        sleeve_returns_df = pd.DataFrame()
        sleeve_names_from_csv = []

    metasleeve_map = {}
    for sname in sleeve_names_from_csv or ["unknown"]:
        if sname in VRP_SLEEVES:
            metasleeve_map[sname] = "vrp_combined"
        else:
            metasleeve_map[sname] = sname

    # Prefer return-based attribution when sleeve_weights artifact exists or can be built from meta
    sleeve_weights = load_sleeve_weights(run_dir)
    if not sleeve_weights and sleeve_names_from_csv:
        sleeve_weights = ensure_sleeve_weights_artifact(run_dir, sleeve_names_from_csv)

    if sleeve_weights and not sleeve_returns_df.empty and set(sleeve_weights.keys()) >= set(sleeve_names_from_csv):
        logger.info("Using sleeve_weights artifact for return-based attribution")
        result = compute_attribution_return_based(
            sleeve_weights=sleeve_weights,
            sleeve_returns=sleeve_returns_df,
            portfolio_simple=portfolio_simple,
            metasleeve_mapping=metasleeve_map,
        )
    else:
        # Fallback: weight decomposition from weights.csv
        sleeve_signals_history = reconstruct_sleeve_signals_from_artifacts(
            run_dir=run_dir,
            weights_panel=weights_panel,
            asset_returns_simple=asset_returns_simple,
            universe=universe,
        )
        if not sleeve_signals_history:
            logger.error("Failed to reconstruct sleeve signals")
            return 1
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
    print(f"Consistency:                 {summary.get('status', 'PASS' if summary['consistency_check']['passed'] else 'FAIL')}")
    print()

    for sleeve, metrics in sorted(summary["per_sleeve"].items(), key=lambda x: x[1]["cum_return"], reverse=True):
        print(f"  {sleeve:30s}: cum={metrics['cum_return']:+.4%}  "
              f"vol={metrics['vol_of_contrib']:.4%}  "
              f"hit={metrics['hit_rate']:.1%}  "
              f"active={metrics['active_days']}/{metrics['active_days'] + metrics['no_signal_days']}")
    print("=" * 70)

    status = summary.get("status", "FAIL")
    exit_ok = status in ("PASS", "WARN")
    return 0 if exit_ok else 1


if __name__ == "__main__":
    sys.exit(main())
