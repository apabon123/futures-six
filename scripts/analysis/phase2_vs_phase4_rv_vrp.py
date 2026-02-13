#!/usr/bin/env python3
"""
Phase-2 vs Phase-4 atomic return stream consistency check for SR3 curve RV and VRP.

Reads existing artifacts only. Writes to reports/runs/<phase4_run_id>/analysis/phase2_vs_phase4_rv_vrp/.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _annualize_return(cum_return: float, n_days: int) -> float:
    if n_days <= 0:
        return np.nan
    return (1 + cum_return) ** (252 / n_days) - 1


def _annualize_vol(daily_vol: float) -> float:
    return daily_vol * np.sqrt(252)


def _sharpe(mean_daily: float, daily_vol: float) -> float:
    if daily_vol <= 0:
        return np.nan
    return (mean_daily / daily_vol) * np.sqrt(252)


def series_metrics(s: pd.Series, label: str) -> dict:
    s = s.dropna()
    if s.empty:
        return {"series": label, "start_date": "", "end_date": "", "n_days": 0, "mean_daily_return": np.nan,
                "daily_vol": np.nan, "ann_return": np.nan, "ann_vol": np.nan, "sharpe": np.nan,
                "cum_return": np.nan, "hit_rate": np.nan, "pct_zero": np.nan}
    n = len(s)
    mean_d = s.mean()
    vol_d = s.std()
    cum = (1 + s).prod() - 1
    hit = (s > 0).mean()
    zeros = (s == 0).mean()
    return {
        "series": label,
        "start_date": str(s.index.min().date()),
        "end_date": str(s.index.max().date()),
        "n_days": n,
        "mean_daily_return": mean_d,
        "daily_vol": vol_d if not np.isnan(vol_d) and vol_d > 0 else np.nan,
        "ann_return": _annualize_return(cum, n),
        "ann_vol": _annualize_vol(vol_d) if vol_d > 0 else np.nan,
        "sharpe": _sharpe(mean_d, vol_d) if vol_d > 0 else np.nan,
        "cum_return": cum,
        "hit_rate": hit,
        "pct_zero": zeros,
    }


def _parse_phase2_vrp_index() -> dict:
    """Read Phase-2 VRP run id and dates from phase index. Returns dict with vrp_run_id, start_date, end_date or empty."""
    path = PROJECT_ROOT / "reports" / "phase_index" / "vrp" / "phase2_core_v5_trend_csmom_vrp_core.txt"
    if not path.exists():
        return {}
    out = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("vrp_run_id:"):
                out["vrp_run_id"] = line.split(":", 1)[1].strip()
            elif line.startswith("start_date:"):
                out["start_date"] = line.split(":", 1)[1].strip()
            elif line.startswith("end_date:"):
                out["end_date"] = line.split(":", 1)[1].strip()
    return out


def main():
    ap = argparse.ArgumentParser(description="Phase-2 vs Phase-4 RV/VRP return stream comparison")
    ap.add_argument("--phase4_run_id", default="carry_int_v1stack_no_vrp_plus_vx_and_sr3spread_20200106_20251031_20260213_152417",
                    help="Phase-4 run ID (integration with sr3_curve_rv_meta)")
    ap.add_argument("--vrp_phase4_run_id", type=str, default="vrp_int_trend_plus_vrp_20200106_20251031_20260213_132033",
                    help="Phase-4 VRP run ID for VRP sleeve comparison (default: vrp_int_trend_plus_vrp_...)")
    ap.add_argument("--out_dir", type=str, default=None, help="Override output dir (default: reports/runs/<phase4_run_id>/analysis/phase2_vs_phase4_rv_vrp)")
    args = ap.parse_args()

    phase4_run_id = args.phase4_run_id
    run_dir = PROJECT_ROOT / "reports" / "runs" / phase4_run_id
    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "analysis" / "phase2_vs_phase4_rv_vrp"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- SR3 curve RV: Phase-2 proxy from phase_index (variant - baseline) ---
    phase2_rv_path = PROJECT_ROOT / "reports" / "phase_index" / "rates_curve_rv" / "sr3_curve_rv_rank_fly_2_6_10_momentum" / "phase2" / "returns.csv"
    phase4_sleeve_path = run_dir / "sleeve_returns.csv"

    rows = []
    notes_lines = [
        "# Phase-2 vs Phase-4 atomic return stream comparison — notes",
        "",
        "## Phase-2 run IDs and evidence",
        "",
        "- **SR3 curve RV (Rank Fly) Phase-2**: run_id `core_v9_rankfly_phase2_20251217_145301` (from reports/phase_index/rates_curve_rv/sr3_curve_rv_rank_fly_2_6_10_momentum/phase2.txt). Run dir exists but has no sleeve_returns.csv; Phase-2 comparison uses portfolio-level baseline/variant from phase_index phase2/returns.csv.",
        "- **SR3 curve RV (Pack Slope) Phase-2**: run_id `core_v9_packslope_phase2_20251217_145414` (from phase2.txt). Same: no sleeve_returns in run dir.",
        "- **VRP Phase-2**: run_id `core_v5_vrp_core_phase2_20251209_220603` (from reports/phase_index/vrp/phase2_core_v5_trend_csmom_vrp_core.txt). Run dir has portfolio_returns.csv only; no sleeve_returns.csv. Path data/diagnostics/phase2/core_v5_trend_csmom_vrp_core/20251209_220603 not present; no comparison_returns.csv for atomic VRP.",
        "",
        "## Phase-4 reference runs used",
        "",
        f"- **sr3_curve_rv_meta**: run_id `{phase4_run_id}` (sleeve_returns.csv column sr3_curve_rv_meta).",
        "- **VRP atomics**: run_id `vrp_int_trend_plus_vrp_20200106_20251031_20260213_132033` (sleeve_returns.csv columns vrp_core_meta, vrp_convergence_meta, vrp_alt_meta).",
        "",
        "## Column mappings",
        "",
        "- SR3 curve RV Phase-2: proxy = (variant - baseline) from reports/phase_index/rates_curve_rv/sr3_curve_rv_rank_fly_2_6_10_momentum/phase2/returns.csv (portfolio incremental, not atomic sleeve).",
        "- SR3 curve RV Phase-4: sleeve_returns.csv column `sr3_curve_rv_meta`.",
        "- VRP: Phase-2 atomic series not available; comparison not performed.",
        "",
    ]

    if phase2_rv_path.exists() and phase4_sleeve_path.exists():
        df_p2 = pd.read_csv(phase2_rv_path, index_col=0, parse_dates=True)
        df_p4 = pd.read_csv(phase4_sleeve_path, index_col=0, parse_dates=True)
        # Proxy: incremental return from adding Rank Fly = variant - baseline
        p2_proxy = (df_p2["variant"] - df_p2["baseline"]).dropna()
        p4_rv = df_p4["sr3_curve_rv_meta"].dropna()
        common = p2_proxy.index.intersection(p4_rv.index)
        if len(common) > 0:
            p2_aligned = p2_proxy.reindex(common).dropna()
            p4_aligned = p4_rv.reindex(common).dropna()
            common = p2_aligned.index.intersection(p4_aligned.index)
            p2_aligned = p2_aligned.loc[common]
            p4_aligned = p4_aligned.loc[common]
            corr = p2_aligned.corr(p4_aligned)
            m2 = series_metrics(p2_aligned, "sr3_curve_rv_meta Phase-2 (proxy)")
            m2["corr_phase2_vs_phase4"] = corr
            m4 = series_metrics(p4_aligned, "sr3_curve_rv_meta Phase-4")
            m4["corr_phase2_vs_phase4"] = None
            rows.append(m2)
            rows.append(m4)
            notes_lines.append("## SR3 curve RV comparison")
            notes_lines.append("")
            notes_lines.append("- Phase-2 proxy (variant - baseline) is portfolio incremental return from adding Rank Fly to Core v8, not atomic sleeve return.")
            notes_lines.append(f"- Correlation(Phase-2 proxy, Phase-4 sr3_curve_rv_meta): {corr:.4f}.")
            notes_lines.append("")
    else:
        notes_lines.append("## SR3 curve RV")
        notes_lines.append("")
        if not phase2_rv_path.exists():
            notes_lines.append(f"- Phase-2 returns file missing: {phase2_rv_path}")
        if not phase4_sleeve_path.exists():
            notes_lines.append(f"- Phase-4 sleeve_returns missing: {phase4_sleeve_path}")
        notes_lines.append("")

    notes_lines.append("## VRP")
    notes_lines.append("")
    # --- VRP: Phase-2 vs Phase-4 atomic sleeve comparison ---
    vrp_phase4_run_id = getattr(args, "vrp_phase4_run_id", None) or "vrp_int_trend_plus_vrp_20200106_20251031_20260213_132033"
    phase2_index = _parse_phase2_vrp_index()
    vrp_out_dir = PROJECT_ROOT / "reports" / "runs" / vrp_phase4_run_id / "analysis" / "phase2_vs_phase4_rv_vrp"
    phase2_sleeve_path = None
    if phase2_index:
        phase2_sleeve_path = PROJECT_ROOT / "reports" / "runs" / phase2_index["vrp_run_id"] / "sleeve_returns.csv"
    phase4_vrp_sleeve_path = PROJECT_ROOT / "reports" / "runs" / vrp_phase4_run_id / "sleeve_returns.csv"
    vrp_summary_lines = []
    identity_consistent = True
    if phase2_index and phase2_sleeve_path and phase2_sleeve_path.exists() and phase4_vrp_sleeve_path.exists():
        vrp_out_dir.mkdir(parents=True, exist_ok=True)
        df_p2 = pd.read_csv(phase2_sleeve_path, parse_dates=["date"], index_col="date")
        df_p4 = pd.read_csv(phase4_vrp_sleeve_path, parse_dates=["date"], index_col="date")
        vrp_cols = ["vrp_core_meta", "vrp_convergence_meta", "vrp_alt_meta"]
        results = []
        for col in vrp_cols:
            if col not in df_p2.columns or col not in df_p4.columns:
                continue
            p2_s = df_p2[col].dropna()
            p4_s = df_p4[col].dropna()
            common = p2_s.index.intersection(p4_s.index)
            if len(common) < 10:
                continue
            p2_a = p2_s.reindex(common).dropna()
            p4_a = p4_s.reindex(common).dropna()
            common = p2_a.index.intersection(p4_a.index)
            p2_a = p2_a.loc[common]
            p4_a = p4_a.loc[common]
            if len(p2_a) < 10:
                continue
            corr_raw = p2_a.corr(p4_a)
            corr = float(corr_raw) if pd.notna(corr_raw) else np.nan
            m2 = series_metrics(p2_a, f"{col} Phase-2")
            m4 = series_metrics(p4_a, f"{col} Phase-4")
            results.append({
                "sleeve": col,
                "corr": corr,
                "phase2_mean_daily": m2["mean_daily_return"],
                "phase4_mean_daily": m4["mean_daily_return"],
                "phase2_daily_vol": m2["daily_vol"],
                "phase4_daily_vol": m4["daily_vol"],
                "phase2_ann_return": m2["ann_return"],
                "phase4_ann_return": m4["ann_return"],
                "phase2_ann_vol": m2["ann_vol"],
                "phase4_ann_vol": m4["ann_vol"],
            })
            if np.isfinite(corr) and corr < 0.95:
                identity_consistent = False
        vrp_summary_lines = [
            "# Phase-2 vs Phase-4 VRP atomic sleeve comparison",
            "",
            "- **Phase-2 run id**: " + phase2_index.get("vrp_run_id", ""),
            "- **Phase-4 run id**: " + vrp_phase4_run_id,
            "",
            "## Per-sleeve correlation",
            "",
        ]
        for r in results:
            corr_s = f"{r['corr']:.4f}" if np.isfinite(r["corr"]) else "N/A (Phase-2 sleeve not present or constant)"
            vrp_summary_lines.append(f"- **{r['sleeve']}**: corr = {corr_s}")
        vrp_summary_lines.append("")
        vrp_summary_lines.append("## Per-sleeve mean / vol comparison")
        vrp_summary_lines.append("")
        for r in results:
            vrp_summary_lines.append(f"### {r['sleeve']}")
            vol2 = r['phase2_daily_vol']
            vol4 = r['phase4_daily_vol']
            ann_vol2 = r['phase2_ann_vol']
            ann_vol4 = r['phase4_ann_vol']
            v2 = f"{vol2:.6f}" if np.isfinite(vol2) else "nan"
            v4 = f"{vol4:.6f}" if np.isfinite(vol4) else "nan"
            a2 = f"{ann_vol2:.4f}" if np.isfinite(ann_vol2) else "nan"
            a4 = f"{ann_vol4:.4f}" if np.isfinite(ann_vol4) else "nan"
            vrp_summary_lines.append(f"- Phase-2: mean_daily = {r['phase2_mean_daily']:.6f}, daily_vol = {v2}, ann_return = {r['phase2_ann_return']:.4f}, ann_vol = {a2}")
            vrp_summary_lines.append(f"- Phase-4: mean_daily = {r['phase4_mean_daily']:.6f}, daily_vol = {v4}, ann_return = {r['phase4_ann_return']:.4f}, ann_vol = {a4}")
            vrp_summary_lines.append("")
        vrp_summary_lines.append("## Identity consistency")
        vrp_summary_lines.append("")
        vrp_summary_lines.append("Identity looks **consistent**." if identity_consistent else "Identity looks **not fully consistent** (some sleeve correlations < 0.95).")
        (vrp_out_dir / "vrp_summary.md").write_text("\n".join(vrp_summary_lines), encoding="utf-8")
        notes_lines.append("- Phase-2 atomic sleeve_returns.csv present; VRP comparison performed.")
        notes_lines.append(f"- Results in `reports/runs/{vrp_phase4_run_id}/analysis/phase2_vs_phase4_rv_vrp/vrp_summary.md`.")
    else:
        if not phase2_index:
            notes_lines.append("- Phase-2 VRP phase index not found.")
        elif not phase2_sleeve_path or not phase2_sleeve_path.exists():
            notes_lines.append("- Phase-2 run dir does not contain sleeve_returns.csv.")
        elif not phase4_vrp_sleeve_path.exists():
            notes_lines.append("- Phase-4 VRP run sleeve_returns.csv not found.")
        notes_lines.append("")
    notes_lines.append("")

    # Write compare_table.csv
    if rows:
        df_out = pd.DataFrame(rows)
        csv_path = out_dir / "compare_table.csv"
        df_out.to_csv(csv_path, index=False)
        # Markdown table
        md_lines = [
            "## Phase-2 vs Phase-4 atomic return stream comparison",
            "",
            "Aligned on intersection of dates. SR3 curve RV Phase-2 uses proxy (variant - baseline). VRP: Phase-2 atomic series missing.",
            "",
            "| series | start_date | end_date | n_days | mean_daily_return | daily_vol | ann_return | ann_vol | sharpe | cum_return | hit_rate | pct_zero | corr_phase2_vs_phase4 |",
            "|--------|------------|----------|--------|-------------------|-----------|------------|---------|--------|------------|----------|----------|------------------------|",
        ]
        for _, r in df_out.iterrows():
            corr = r.get("corr_phase2_vs_phase4")
            corr_s = f"{corr:.4f}" if pd.notna(corr) and corr is not None else ""
            md_lines.append(
                f"| {r['series']} | {r['start_date']} | {r['end_date']} | {int(r['n_days'])} | "
                f"{r['mean_daily_return']:.6f} | {r['daily_vol']:.6f} | {r['ann_return']:.4f} | {r['ann_vol']:.4f} | "
                f"{r['sharpe']:.4f} | {r['cum_return']:.4f} | {r['hit_rate']:.2%} | {r['pct_zero']:.2%} | {corr_s} |"
            )
        (out_dir / "compare_table.md").write_text("\n".join(md_lines), encoding="utf-8")
    else:
        (out_dir / "compare_table.csv").write_text("series,start_date,end_date,n_days,mean_daily_return,daily_vol,ann_return,ann_vol,sharpe,cum_return,hit_rate,pct_zero,corr_phase2_vs_phase4\n", encoding="utf-8")
        (out_dir / "compare_table.md").write_text("## Phase-2 vs Phase-4\n\nNo comparable series (Phase-2 data missing or no alignment).\n", encoding="utf-8")

    (out_dir / "notes.md").write_text("\n".join(notes_lines), encoding="utf-8")
    printed = [f"Wrote {out_dir}/compare_table.csv, compare_table.md, notes.md"]
    if vrp_out_dir.exists() and (vrp_out_dir / "vrp_summary.md").exists():
        printed.append(f"Wrote {vrp_out_dir}/vrp_summary.md")
    print("; ".join(printed))
    return 0


if __name__ == "__main__":
    sys.exit(main())
