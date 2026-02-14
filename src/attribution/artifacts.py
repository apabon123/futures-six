"""
Attribution artifact generation for Futures-Six.

Generates CSV, JSON, and Markdown attribution artifacts from computed
attribution data.
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.attribution.governance import (
    ATTR_TOL_PASS,
    ATTR_TOL_WARN,
    attribution_status_from_residual,
)

logger = logging.getLogger(__name__)


def _compute_sleeve_metrics(
    contrib: pd.Series,
    portfolio_simple: pd.Series,
    no_signal_info: Dict,
    sleeve_name: str,
) -> Dict:
    """Compute summary metrics for a single sleeve."""
    active_mask = contrib.index >= contrib.first_valid_index() if contrib.first_valid_index() else pd.Series(False, index=contrib.index)

    cum_return = float(contrib.sum())
    mean_daily = float(contrib.mean())
    mean_abs_daily = float(contrib.abs().mean())
    vol_of_contrib = float(contrib.std()) if len(contrib) > 1 else 0.0

    # Max drawdown of cumulative contribution
    cum = contrib.cumsum()
    running_max = cum.cummax()
    drawdown = cum - running_max
    max_dd = float(drawdown.min()) if len(drawdown) > 0 else 0.0

    # Hit rate: fraction of days with positive contribution
    nonzero_days = contrib[contrib.abs() > 1e-15]
    hit_rate = float((nonzero_days > 0).mean()) if len(nonzero_days) > 0 else 0.0

    ns_info = no_signal_info.get(sleeve_name, {})

    return {
        "cum_return": cum_return,
        "mean_daily_contrib": mean_daily,
        "mean_abs_daily_contrib": mean_abs_daily,
        "vol_of_contrib": vol_of_contrib,
        "max_dd_of_contrib": max_dd,
        "hit_rate": hit_rate,
        "active_days": ns_info.get("active_days", 0),
        "no_signal_days": ns_info.get("no_signal_days", 0),
    }


def _compute_correlation_matrix(
    contributions: pd.DataFrame,
) -> pd.DataFrame:
    """Compute correlation matrix of sleeve daily contributions."""
    # Filter to periods with non-trivial activity
    active = contributions.loc[(contributions.abs() > 1e-15).any(axis=1)]
    if len(active) < 10:
        return pd.DataFrame()
    return active.corr()


def _format_pct(val: float, decimals: int = 2) -> str:
    """Format a decimal as percentage string."""
    return f"{val * 100:.{decimals}f}%"


def _format_bps(val: float) -> str:
    """Format a decimal as basis points."""
    return f"{val * 10000:.1f} bps"


def generate_attribution_artifacts(
    attribution_result: Dict,
    portfolio_returns_simple: pd.Series,
    run_dir: Path,
    run_id: str,
    metasleeve_mapping: Optional[Dict[str, str]] = None,
) -> Path:
    """
    Generate all attribution artifacts and write to disk.

    Args:
        attribution_result: Output of compute_attribution().
        portfolio_returns_simple: Daily portfolio simple returns.
        run_dir: Path to the run directory (e.g., reports/runs/<run_id>/).
        run_id: Run identifier string.
        metasleeve_mapping: Optional sleeve-to-metasleeve mapping.

    Returns:
        Path to the attribution output directory.
    """
    atomic_contribs = attribution_result["atomic_contributions"]
    meta_contribs = attribution_result["metasleeve_contributions"]
    diagnostics = attribution_result["diagnostics"]

    if atomic_contribs.empty:
        logger.error("[Attribution] No contribution data; skipping artifact generation.")
        return run_dir

    # Create output directory
    out_dir = run_dir / "analysis" / "attribution"
    os.makedirs(out_dir, exist_ok=True)

    portfolio_simple = portfolio_returns_simple.reindex(atomic_contribs.index).fillna(0.0)

    # Determine active window (from first rebalance date)
    first_rebal_str = diagnostics.get("first_rebalance_date", str(atomic_contribs.index[0]))
    first_rebal = pd.Timestamp(first_rebal_str)
    active_mask = atomic_contribs.index >= first_rebal

    # ========================================================================
    # 1. attribution_by_metasleeve.csv
    # ========================================================================
    meta_rows = []
    for date in meta_contribs.index:
        for sleeve in meta_contribs.columns:
            row = {
                "date": date.strftime("%Y-%m-%d") if hasattr(date, "strftime") else str(date),
                "metasleeve": sleeve,
                "daily_contribution_return": meta_contribs.loc[date, sleeve],
            }
            meta_rows.append(row)

    meta_df = pd.DataFrame(meta_rows)

    # Add cumulative contribution per metasleeve
    for sleeve in meta_contribs.columns:
        mask = meta_df["metasleeve"] == sleeve
        meta_df.loc[mask, "cumulative_contribution_return"] = (
            meta_df.loc[mask, "daily_contribution_return"].cumsum().values
        )

    meta_df.to_csv(out_dir / "attribution_by_metasleeve.csv", index=False)
    logger.info(f"[Attribution] Saved attribution_by_metasleeve.csv ({len(meta_df)} rows)")

    # ========================================================================
    # 2. attribution_by_atomic_sleeve.csv
    # ========================================================================
    atomic_rows = []
    for date in atomic_contribs.index:
        for sleeve in atomic_contribs.columns:
            ns_info = diagnostics.get("no_signal_info", {}).get(sleeve, {})
            row = {
                "date": date.strftime("%Y-%m-%d") if hasattr(date, "strftime") else str(date),
                "atomic_sleeve": sleeve,
                "daily_contribution_return": atomic_contribs.loc[date, sleeve],
            }
            atomic_rows.append(row)

    atomic_df = pd.DataFrame(atomic_rows)

    # Add cumulative contribution per atomic sleeve
    for sleeve in atomic_contribs.columns:
        mask = atomic_df["atomic_sleeve"] == sleeve
        atomic_df.loc[mask, "cumulative_contribution_return"] = (
            atomic_df.loc[mask, "daily_contribution_return"].cumsum().values
        )

    # Add no_signal_flag: 1 if the sleeve has zero weight allocation on that date
    if not attribution_result["sleeve_weight_decomposition"].empty:
        decomp = attribution_result["sleeve_weight_decomposition"]
        for sleeve in atomic_contribs.columns:
            sleeve_mask = atomic_df["atomic_sleeve"] == sleeve
            # Build daily no-signal flags from contribution magnitude
            daily_contrib = atomic_contribs[sleeve]
            no_sig_series = (daily_contrib.abs() < 1e-15).astype(int)
            # Map dates
            date_to_flag = dict(zip(
                [d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d) for d in no_sig_series.index],
                no_sig_series.values
            ))
            atomic_df.loc[sleeve_mask, "no_signal_flag"] = (
                atomic_df.loc[sleeve_mask, "date"].map(date_to_flag).values
            )

    atomic_df.to_csv(out_dir / "attribution_by_atomic_sleeve.csv", index=False)
    logger.info(f"[Attribution] Saved attribution_by_atomic_sleeve.csv ({len(atomic_df)} rows)")

    # ========================================================================
    # 3. attribution_summary.json
    # ========================================================================
    no_signal_info = diagnostics.get("no_signal_info", {})

    # Per-sleeve metrics
    per_sleeve_metrics = {}
    for sleeve in atomic_contribs.columns:
        contrib_series = atomic_contribs[sleeve][active_mask]
        per_sleeve_metrics[sleeve] = _compute_sleeve_metrics(
            contrib_series, portfolio_simple[active_mask], no_signal_info, sleeve
        )

    # Portfolio-level
    total_port_cum = float(portfolio_simple[active_mask].sum())
    sum_sleeve_cum = float(atomic_contribs[active_mask].sum().sum())
    residual_cum = total_port_cum - sum_sleeve_cum

    # Correlation matrix
    corr_matrix = _compute_correlation_matrix(atomic_contribs[active_mask])

    max_residual = diagnostics["max_abs_daily_residual_active"]
    status = attribution_status_from_residual(max_residual)

    summary = {
        "run_id": run_id,
        "generated_at": datetime.now().isoformat(),
        "total_portfolio_cum_return": total_port_cum,
        "sum_sleeve_cum_return": sum_sleeve_cum,
        "residual_cum_return": residual_cum,
        "residual_pct_of_portfolio": (
            residual_cum / abs(total_port_cum) if abs(total_port_cum) > 1e-15 else 0.0
        ),
        "residual_value": max_residual,
        "status": status,
        "tolerance_thresholds": {
            "pass": ATTR_TOL_PASS,
            "warn": ATTR_TOL_WARN,
        },
        "consistency_check": {
            "passed": diagnostics["consistency_pass"],
            "tolerance": diagnostics["tolerance"],
            "max_abs_daily_residual": diagnostics["max_abs_daily_residual_active"],
            "cum_residual": diagnostics["cum_residual"],
        },
        "per_sleeve": per_sleeve_metrics,
        "n_active_days": diagnostics["n_active_days"],
        "sleeves_captured": diagnostics["sleeves_captured"],
        "partial_attribution": diagnostics["partial_attribution"],
    }

    with open(out_dir / "attribution_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info("[Attribution] Saved attribution_summary.json")

    # Save correlation matrix as separate CSV
    if not corr_matrix.empty:
        corr_matrix.to_csv(out_dir / "sleeve_contribution_correlation.csv")
        logger.info("[Attribution] Saved sleeve_contribution_correlation.csv")

    # ========================================================================
    # 4. ATTRIBUTION_SUMMARY.md
    # ========================================================================
    md = _build_attribution_markdown(
        summary=summary,
        atomic_contribs=atomic_contribs[active_mask],
        meta_contribs=meta_contribs[active_mask],
        portfolio_simple=portfolio_simple[active_mask],
        corr_matrix=corr_matrix,
        diagnostics=diagnostics,
        run_id=run_id,
        metasleeve_mapping=metasleeve_mapping,
    )

    md_path = out_dir / "ATTRIBUTION_SUMMARY.md"
    md_path.write_text(md, encoding="utf-8")
    logger.info(f"[Attribution] Saved ATTRIBUTION_SUMMARY.md")

    return out_dir


def _build_attribution_markdown(
    summary: Dict,
    atomic_contribs: pd.DataFrame,
    meta_contribs: pd.DataFrame,
    portfolio_simple: pd.Series,
    corr_matrix: pd.DataFrame,
    diagnostics: Dict,
    run_id: str,
    metasleeve_mapping: Optional[Dict[str, str]] = None,
) -> str:
    """Build the human-readable attribution summary markdown."""
    lines = []
    lines.append("# Attribution Summary")
    lines.append("")
    lines.append(f"**Run ID**: `{run_id}`")
    lines.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Active Window**: {atomic_contribs.index[0].strftime('%Y-%m-%d')} to "
                 f"{atomic_contribs.index[-1].strftime('%Y-%m-%d')}")
    lines.append(f"**Active Trading Days**: {len(atomic_contribs)}")
    lines.append("")

    # ---- What this is measuring ----
    lines.append("## What This Is Measuring")
    lines.append("")
    lines.append("This report decomposes the portfolio's daily simple return into additive")
    lines.append("sleeve-level contributions. The decomposition method:")
    lines.append("")
    lines.append("1. **Weight decomposition**: Final portfolio weights are allocated to each")
    lines.append("   sleeve proportionally to that sleeve's weighted signal contribution")
    lines.append("   to the combined signal for each asset.")
    lines.append("2. **Log-return attribution**: Each sleeve's daily contribution is first")
    lines.append("   computed in log-return space (where weighted sums are exact), then")
    lines.append("   scaled proportionally to match the portfolio simple return.")
    lines.append("3. **Exact additivity**: `sum(sleeve contributions) == portfolio return`")
    lines.append("   by construction (within floating-point tolerance).")
    lines.append("")
    lines.append("**Units**: All values are in DECIMAL form (e.g., 0.01 = 1%). No currency")
    lines.append("PnL is reported. `daily_contribution_return` is the portion of the")
    lines.append("portfolio's daily simple return attributable to that sleeve.")
    lines.append("")
    lines.append("**Formulas**:")
    lines.append("- `sleeve_fraction[s,i] = sleeve_weighted_signal[s,i] / combined_signal[i]`")
    lines.append("- `sleeve_weight[s,i] = sleeve_fraction[s,i] * final_portfolio_weight[i]`")
    lines.append("- `sleeve_log_contrib[s] = sum_i(sleeve_weight[s,i] * log_return[i])`")
    lines.append("- `daily_contribution_return[s] = sleeve_log_contrib[s] / port_log * port_simple`")
    lines.append("")

    # ---- Consistency Checks ----
    lines.append("## Consistency Checks")
    lines.append("")
    check = summary["consistency_check"]
    passed = check["passed"]
    gov_status = summary.get("status", "PASS" if passed else "FAIL")
    lines.append(f"- **Status**: {gov_status} (governance: PASS ≤1e-5, WARN ≤1e-4, FAIL >1e-4)")
    lines.append(f"- **Strict pass (legacy)**: {'PASSED' if passed else 'FAILED'} (tolerance {check['tolerance']:.0e})")
    lines.append(f"- **Max |daily residual|**: {check['max_abs_daily_residual']:.2e}")
    lines.append(f"- **Cumulative residual**: {check['cum_residual']:.2e}")
    lines.append(f"- **Portfolio cumulative return**: {_format_pct(summary['total_portfolio_cum_return'])}")
    lines.append(f"- **Sum of sleeve cumulative returns**: {_format_pct(summary['sum_sleeve_cum_return'])}")
    lines.append(f"- **Residual cumulative**: {summary['residual_cum_return']:.2e} "
                 f"({_format_pct(summary['residual_pct_of_portfolio'])} of portfolio)")
    if summary["partial_attribution"]:
        lines.append(f"- **WARNING: PARTIAL ATTRIBUTION** - not all weight is accounted for.")
    lines.append("")

    # ---- Top Contributors and Detractors ----
    lines.append("## Top Contributors and Detractors")
    lines.append("")

    per_sleeve = summary.get("per_sleeve", {})
    if per_sleeve:
        sorted_sleeves = sorted(per_sleeve.items(), key=lambda x: x[1]["cum_return"], reverse=True)

        lines.append("### By Cumulative Contribution")
        lines.append("")
        lines.append("| Sleeve | Cum Return | Mean Daily | Vol of Contrib | Max DD | Hit Rate | Active Days | No-Signal Days |")
        lines.append("|--------|-----------|------------|---------------|--------|----------|-------------|----------------|")
        for sleeve_name, metrics in sorted_sleeves:
            lines.append(
                f"| {sleeve_name} "
                f"| {_format_pct(metrics['cum_return'])} "
                f"| {_format_bps(metrics['mean_daily_contrib'])} "
                f"| {_format_bps(metrics['vol_of_contrib'])} "
                f"| {_format_pct(metrics['max_dd_of_contrib'])} "
                f"| {metrics['hit_rate']:.1%} "
                f"| {metrics['active_days']} "
                f"| {metrics['no_signal_days']} |"
            )
        lines.append("")

        # Top contributors
        contributors = [s for s, m in sorted_sleeves if m["cum_return"] > 0]
        detractors = [s for s, m in sorted_sleeves if m["cum_return"] < 0]

        if contributors:
            lines.append(f"**Top contributors**: {', '.join(contributors)}")
        if detractors:
            lines.append(f"**Detractors**: {', '.join(detractors)}")
        lines.append("")

    # ---- Regime Narrative ----
    lines.append("## Regime Narrative")
    lines.append("")

    # Last 6 months
    six_months_ago = atomic_contribs.index[-1] - pd.DateOffset(months=6)
    twelve_months_ago = atomic_contribs.index[-1] - pd.DateOffset(months=12)

    for label, start_date in [
        ("Last 6 Months", six_months_ago),
        ("Last 12 Months", twelve_months_ago),
        ("Full Window", atomic_contribs.index[0]),
    ]:
        period_mask = atomic_contribs.index >= start_date
        if period_mask.sum() == 0:
            continue

        period_contribs = atomic_contribs[period_mask]
        period_port = portfolio_simple[period_mask]

        lines.append(f"### {label}")
        lines.append(f"({start_date.strftime('%Y-%m-%d')} to {atomic_contribs.index[-1].strftime('%Y-%m-%d')}, "
                     f"{period_mask.sum()} days)")
        lines.append("")

        port_cum = period_port.sum()
        lines.append(f"- **Portfolio cumulative return**: {_format_pct(port_cum)}")

        for sleeve in period_contribs.columns:
            sleeve_cum = period_contribs[sleeve].sum()
            lines.append(f"- **{sleeve}**: {_format_pct(sleeve_cum)}")
        lines.append("")

    # ---- Correlation Matrix ----
    if not corr_matrix.empty:
        lines.append("## Sleeve Contribution Correlations")
        lines.append("")
        header = "| | " + " | ".join(corr_matrix.columns) + " |"
        separator = "|" + "|".join(["---"] * (len(corr_matrix.columns) + 1)) + "|"
        lines.append(header)
        lines.append(separator)
        for idx in corr_matrix.index:
            row_vals = " | ".join(f"{corr_matrix.loc[idx, c]:.3f}" for c in corr_matrix.columns)
            lines.append(f"| {idx} | {row_vals} |")
        lines.append("")

    # ---- Alerts / Anomalies ----
    lines.append("## Alerts and Anomalies")
    lines.append("")

    alerts_found = False

    # Sudden drop to zero contribution
    for sleeve in atomic_contribs.columns:
        contrib = atomic_contribs[sleeve]
        rolling_mean = contrib.abs().rolling(20, min_periods=10).mean()
        if len(rolling_mean.dropna()) > 20:
            recent = rolling_mean.iloc[-5:]
            earlier = rolling_mean.iloc[-25:-5]
            if earlier.mean() > 1e-6 and recent.mean() < 1e-8:
                lines.append(f"- **ALERT**: {sleeve} contribution dropped to near-zero in recent period.")
                alerts_found = True

    # Persistent no-signal
    no_signal_info = diagnostics.get("no_signal_info", {})
    for sleeve, ns in no_signal_info.items():
        no_sig_pct = ns["no_signal_days"] / ns["total_days"] * 100 if ns["total_days"] > 0 else 0
        if no_sig_pct > 50:
            lines.append(f"- **ALERT**: {sleeve} has no signal {no_sig_pct:.0f}% of days ({ns['no_signal_days']}/{ns['total_days']}).")
            alerts_found = True

    # Extreme contribution spikes
    for sleeve in atomic_contribs.columns:
        contrib = atomic_contribs[sleeve]
        if contrib.std() > 0:
            z_scores = (contrib - contrib.mean()) / contrib.std()
            extreme_days = (z_scores.abs() > 5).sum()
            if extreme_days > 0:
                lines.append(f"- **INFO**: {sleeve} had {extreme_days} days with |z-score| > 5 contribution spikes.")
                alerts_found = True

    # Correlation crowding
    if not corr_matrix.empty:
        for i, s1 in enumerate(corr_matrix.columns):
            for j, s2 in enumerate(corr_matrix.columns):
                if i < j and abs(corr_matrix.loc[s1, s2]) > 0.8:
                    lines.append(
                        f"- **INFO**: High correlation ({corr_matrix.loc[s1, s2]:.3f}) "
                        f"between {s1} and {s2} contributions."
                    )
                    alerts_found = True

    if not alerts_found:
        lines.append("No alerts or anomalies detected.")
    lines.append("")

    # ---- If Something Is Failing ----
    lines.append("## If Something Is Failing")
    lines.append("")
    lines.append("1. **Consistency check failed**: The sum of sleeve contributions does not")
    lines.append("   match portfolio return. Check that all enabled sleeves are captured in")
    lines.append("   `sleeve_signals_history` during ExecSim. Look for sleeves with weight > 0")
    lines.append("   that are missing from attribution.")
    lines.append("2. **A sleeve shows persistent no-signal**: Check the sleeve's feature")
    lines.append("   pipeline. VRP sleeves require VIX/VX data from DuckDB; verify with")
    lines.append("   `scripts/preflight/check_window_coverage.py`.")
    lines.append("3. **Extreme contribution spikes**: Check for rebalance dates near market")
    lines.append("   events (VIX spikes, flash crashes). Verify that weight caps are binding.")
    lines.append("4. **Correlation crowding**: Multiple sleeves trading the same instruments")
    lines.append("   with correlated signals. Review sleeve universe separation.")
    lines.append("5. **Partial attribution warning**: Some final weights are not explained by")
    lines.append("   captured sleeve signals. This typically occurs when a new sleeve is added")
    lines.append("   but its signal capture is not wired in ExecSim.")
    lines.append("")

    return "\n".join(lines)
