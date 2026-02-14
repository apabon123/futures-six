#!/usr/bin/env python3
"""
Extract headline metrics and attribution summaries from run artifacts into tracked JSON files.

Reads from reports/runs/<run_id>/ (if present locally) and writes:
- docs/pinned/<run_id>.metrics.json
- docs/pinned/<run_id>.attribution.json (reduced summary, no large time series)

Run:
    python scripts/site/extract_metrics.py [--run_id RUN_ID] [--all]
    python scripts/site/extract_metrics.py --all

If run artifacts are missing, writes empty or minimal JSON (site still builds).
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


def _load_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def extract_metrics_for_run(run_id: str) -> dict:
    """Extract headline metrics from run artifacts. Returns dict for metrics.json."""
    run_dir = PROJECT_ROOT / "reports" / "runs" / run_id
    out: dict = {
        "run_id": run_id,
        "cagr": None,
        "vol": None,
        "sharpe": None,
        "maxdd": None,
        "cum_return": None,
        "turnover": None,
        "hit_rate": None,
    }

    # meta.json (exec_sim writes max_drawdown, we normalize to maxdd)
    meta = _load_json(run_dir / "meta.json")
    if meta:
        metrics_eval = meta.get("metrics_eval", {})
        if metrics_eval:
            out["cagr"] = metrics_eval.get("cagr")
            out["vol"] = metrics_eval.get("vol")
            out["sharpe"] = metrics_eval.get("sharpe")
            out["maxdd"] = metrics_eval.get("maxdd") or metrics_eval.get("max_drawdown")
            out["cum_return"] = metrics_eval.get("cum_return")
            out["turnover"] = metrics_eval.get("turnover_avg") or metrics_eval.get("turnover")
            out["hit_rate"] = metrics_eval.get("hit_rate")

    # canonical_diagnostics.json (fallback)
    if out["cagr"] is None:
        diag = _load_json(run_dir / "analysis" / "canonical_diagnostics.json")
        if diag:
            me = diag.get("metrics_eval", {})
            out["cagr"] = me.get("cagr")
            out["vol"] = me.get("vol")
            out["sharpe"] = me.get("sharpe")
            out["maxdd"] = me.get("maxdd") or me.get("max_drawdown")
            out["cum_return"] = me.get("cum_return")
            out["turnover"] = me.get("turnover_avg") or me.get("turnover")
            out["hit_rate"] = me.get("hit_rate")

    return out


def extract_attribution_for_run(run_id: str) -> Optional[dict]:
    """
    Copy reduced attribution summary from reports/runs/<run_id>/analysis/attribution/
    to docs/pinned/<run_id>.attribution.json. No large time series.
    """
    attr_dir = PROJECT_ROOT / "reports" / "runs" / run_id / "analysis" / "attribution"
    summary_path = attr_dir / "attribution_summary.json"
    if not summary_path.exists():
        return None

    raw = _load_json(summary_path)
    if not raw:
        return None

    out = {
        "run_id": run_id,
        "consistency_check": raw.get("consistency_check", {}),
        "residual_value": raw.get("residual_value"),
        "status": raw.get("status"),
        "tolerance_thresholds": raw.get("tolerance_thresholds"),
        "atomic_summary": raw.get("per_sleeve", {}),
        "metasleeve_summary": [],
        "correlation_matrix_path": None,
    }

    # Build metasleeve summary from attribution_by_metasleeve.csv (last cum_return per metasleeve)
    meta_csv = attr_dir / "attribution_by_metasleeve.csv"
    if meta_csv.exists() and HAS_PANDAS:
        try:
            df = pd.read_csv(meta_csv)
            if "metasleeve" in df.columns and "cumulative_contribution_return" in df.columns:
                last = df.groupby("metasleeve").last().reset_index()
                out["metasleeve_summary"] = [
                    {
                        "metasleeve": row["metasleeve"],
                        "cum_return": float(row["cumulative_contribution_return"]),
                    }
                    for _, row in last.iterrows()
                ]
        except Exception:
            pass

    if (attr_dir / "sleeve_contribution_correlation.csv").exists():
        out["correlation_matrix_path"] = "sleeve_contribution_correlation.csv"

    return out


def extract_leverage_for_run(run_id: str) -> Optional[dict]:
    """
    Copy leverage_summary from reports/runs/<run_id>/analysis/leverage_summary.json
    to docs/pinned/<run_id>.leverage.json.
    """
    path = PROJECT_ROOT / "reports" / "runs" / run_id / "analysis" / "leverage_summary.json"
    if not path.exists():
        return None
    raw = _load_json(path)
    return raw


def main():
    parser = argparse.ArgumentParser(description="Extract run metrics to docs/pinned/*.metrics.json")
    parser.add_argument("--run_id", type=str, help="Single run_id to extract")
    parser.add_argument("--all", action="store_true", help="Extract for all runs in pinned_runs.yaml")
    args = parser.parse_args()

    import yaml
    pinned_path = PROJECT_ROOT / "configs" / "pinned_runs.yaml"
    if not pinned_path.exists():
        print("configs/pinned_runs.yaml not found")
        sys.exit(1)

    with open(pinned_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    runs = data.get("runs", [])

    if args.run_id:
        run_ids = [r["run_id"] for r in runs if r["run_id"] == args.run_id]
        if not run_ids:
            print(f"Run {args.run_id} not in pinned_runs.yaml")
            sys.exit(1)
    elif args.all:
        run_ids = [r["run_id"] for r in runs]
    else:
        print("Use --run_id RUN_ID or --all")
        sys.exit(1)

    pinned_dir = PROJECT_ROOT / "docs" / "pinned"
    pinned_dir.mkdir(parents=True, exist_ok=True)

    for rid in run_ids:
        metrics = extract_metrics_for_run(rid)
        out_path = pinned_dir / f"{rid}.metrics.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"Wrote {out_path}")

        attr = extract_attribution_for_run(rid)
        if attr:
            attr_path = pinned_dir / f"{rid}.attribution.json"
            with open(attr_path, "w", encoding="utf-8") as f:
                json.dump(attr, f, indent=2, default=str)
            print(f"Wrote {attr_path}")

        lev = extract_leverage_for_run(rid)
        if lev:
            lev_path = pinned_dir / f"{rid}.leverage.json"
            with open(lev_path, "w", encoding="utf-8") as f:
                json.dump(lev, f, indent=2, default=str)
            print(f"Wrote {lev_path}")


if __name__ == "__main__":
    main()
