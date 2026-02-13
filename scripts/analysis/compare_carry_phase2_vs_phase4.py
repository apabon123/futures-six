#!/usr/bin/env python3
"""
Phase-2 vs Phase-4 atomic carry return comparison (analysis-only, no backtests).

Loads:
- Phase-2 SR3: reports/runs/carry/sr3_calendar_carry_phase2/20251216_150004/comparison_returns.csv (column: carry)
- Phase-2 VX:  reports/runs/carry/vx_calendar_carry_phase2/vx2_vx1_short/20251217_125425/comparison_returns.csv (column: vx_carry)
- Phase-4:     reports/runs/<phase4_run_id>/sleeve_returns.csv (columns: sr3_carry_curve, vx_calendar_carry)

Aligns on common dates, computes metrics and Phase-2 vs Phase-4 correlation per sleeve,
writes compare_table.md, compare_table.csv, notes.md under the Phase-4 run's analysis dir.
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np

# Phase-2 paths from phase_index (DIAGNOSTICS.md)
PHASE2_SR3_DIR = Path("reports/runs/carry/sr3_calendar_carry_phase2/20251216_150004")
PHASE2_VX_DIR = Path("reports/runs/carry/vx_calendar_carry_phase2/vx2_vx1_short/20251217_125425")
PHASE2_SR3_RUN_ID = "carry/sr3_calendar_carry_phase2/20251216_150004"
PHASE2_VX_RUN_ID = "carry/vx_calendar_carry_phase2/vx2_vx1_short/20251217_125425"


def load_series(path: Path, column: str, name: str) -> pd.Series:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    if column not in df.columns:
        raise KeyError(f"Column '{column}' not in {path}. Columns: {list(df.columns)}")
    s = df[column].astype(float)
    s.name = name
    return s


def metrics(series: pd.Series) -> dict:
    s = series.dropna()
    s = s[s.index.notna()]
    if len(s) == 0:
        return {
            "start_date": "",
            "end_date": "",
            "n_days": 0,
            "mean_daily_ret": np.nan,
            "daily_vol": np.nan,
            "ann_return": np.nan,
            "ann_vol": np.nan,
            "sharpe": np.nan,
            "cum_return": np.nan,
            "hit_rate": np.nan,
            "pct_zero": np.nan,
        }
    mean_ret = s.mean()
    vol = s.std()
    n = len(s)
    ann_ret = mean_ret * 252
    ann_vol = vol * np.sqrt(252) if vol > 0 else np.nan
    sharpe = ann_ret / ann_vol if ann_vol and ann_vol > 0 else (np.nan if ann_vol == 0 else (np.inf if ann_ret > 0 else -np.inf))
    cum = (1 + s).prod() - 1
    hit = (s > 0).mean()
    pct_zero = (s.abs() < 1e-12).mean()
    return {
        "start_date": str(s.index.min().date()) if hasattr(s.index.min(), "date") else str(s.index.min()),
        "end_date": str(s.index.max().date()) if hasattr(s.index.max(), "date") else str(s.index.max()),
        "n_days": n,
        "mean_daily_ret": mean_ret,
        "daily_vol": vol,
        "ann_return": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "cum_return": cum,
        "hit_rate": hit,
        "pct_zero": pct_zero,
    }


def main():
    ap = argparse.ArgumentParser(description="Compare Phase-2 vs Phase-4 atomic carry returns")
    ap.add_argument("--phase4_run_id", default="carry_eq_trend_plus_carry_20200106_20251031_20260213_145229", help="Phase-4 reference run_id")
    ap.add_argument("--out_dir", default=None, help="Override output dir (default: reports/runs/<phase4_run_id>/analysis/carry_phase2_vs_phase4)")
    args = ap.parse_args()

    phase4_run_dir = Path("reports/runs") / args.phase4_run_id
    if not phase4_run_dir.exists():
        raise FileNotFoundError(f"Phase-4 run dir not found: {phase4_run_dir}")

    # ----- Step 1: Phase-2 run IDs and evidence -----
    print("Step 1 — Phase-2 run IDs (evidence)")
    print("  SR3 Phase-2 run_id:", PHASE2_SR3_RUN_ID)
    print("    Evidence: reports/phase_index/carry/sr3_calendar_carry/phase2.txt → path: reports\\runs\\carry\\sr3_calendar_carry_phase2\\20251216_150004")
    print("  VX Phase-2 run_id:", PHASE2_VX_RUN_ID)
    print("    Evidence: reports/phase_index/carry/vx_calendar_carry/vx2_vx1_short/phase2.txt → path: reports\\runs\\carry\\vx_calendar_carry_phase2\\vx2_vx1_short\\20251217_125425")
    print("")

    if not PHASE2_SR3_DIR.exists():
        raise FileNotFoundError(f"Phase-2 SR3 dir not found: {PHASE2_SR3_DIR}")
    if not PHASE2_VX_DIR.exists():
        raise FileNotFoundError(f"Phase-2 VX dir not found: {PHASE2_VX_DIR}")

    sr3_comp = PHASE2_SR3_DIR / "comparison_returns.csv"
    vx_comp = PHASE2_VX_DIR / "comparison_returns.csv"
    if not sr3_comp.exists():
        raise FileNotFoundError(f"Phase-2 SR3 comparison_returns.csv not found: {sr3_comp}")
    if not vx_comp.exists():
        raise FileNotFoundError(f"Phase-2 VX comparison_returns.csv not found: {vx_comp}")

    phase4_sleeve = phase4_run_dir / "sleeve_returns.csv"
    if not phase4_sleeve.exists():
        raise FileNotFoundError(f"Phase-4 sleeve_returns.csv not found: {phase4_sleeve}")

    # ----- Step 2: Load and name-map -----
    # Phase-2: "carry" -> sr3_carry_curve (same sleeve); "vx_carry" -> vx_calendar_carry
    p2_sr3 = load_series(sr3_comp, "carry", "sr3_carry_curve")
    p2_vx = load_series(vx_comp, "vx_carry", "vx_calendar_carry")

    phase4_df = pd.read_csv(phase4_sleeve, index_col=0, parse_dates=True)
    p4_sr3 = phase4_df["sr3_carry_curve"].astype(float)
    p4_sr3.name = "sr3_carry_curve"
    p4_vx = phase4_df["vx_calendar_carry"].astype(float)
    p4_vx.name = "vx_calendar_carry"

    # Align on common index (intersection of dates)
    common = p2_sr3.index.intersection(p2_vx.index).intersection(p4_sr3.index).intersection(p4_vx.index)
    common = common.sort_values()
    p2_sr3_a = p2_sr3.reindex(common).fillna(0.0)
    p2_vx_a = p2_vx.reindex(common).fillna(0.0)
    p4_sr3_a = p4_sr3.reindex(common).fillna(0.0)
    p4_vx_a = p4_vx.reindex(common).fillna(0.0)

    # ----- Step 3: Metrics per series -----
    m_p2_sr3 = metrics(p2_sr3_a)
    m_p2_vx = metrics(p2_vx_a)
    m_p4_sr3 = metrics(p4_sr3_a)
    m_p4_vx = metrics(p4_vx_a)

    corr_sr3 = p2_sr3_a.corr(p4_sr3_a)
    corr_vx = p2_vx_a.corr(p4_vx_a)
    corr_p2_sr3_vx = p2_sr3_a.corr(p2_vx_a)
    corr_p4_sr3_vx = p4_sr3_a.corr(p4_vx_a)

    # ----- Step 4: Build table rows -----
    def row(label: str, m: dict, corr_p2_p4: float = np.nan) -> dict:
        r = {"series": label, **m}
        if not np.isnan(corr_p2_p4):
            r["corr_phase2_vs_phase4"] = corr_p2_p4
        return r

    rows = [
        row("sr3_carry_curve Phase-2", m_p2_sr3, corr_sr3),
        row("sr3_carry_curve Phase-4", m_p4_sr3),
        row("vx_calendar_carry Phase-2", m_p2_vx, corr_vx),
        row("vx_calendar_carry Phase-4", m_p4_vx),
    ]

    # Add correlation between sleeves (optional)
    notes_extra = [
        f"Correlation sr3_carry_curve Phase-2 vs Phase-4 (daily): {corr_sr3:.6f}",
        f"Correlation vx_calendar_carry Phase-2 vs Phase-4 (daily): {corr_vx:.6f}",
        f"Correlation sr3 vs vx within Phase-2: {corr_p2_sr3_vx:.6f}",
        f"Correlation sr3 vs vx within Phase-4: {corr_p4_sr3_vx:.6f}",
    ]

    table_df = pd.DataFrame(rows)
    out_dir = Path(args.out_dir) if args.out_dir else phase4_run_dir / "analysis" / "carry_phase2_vs_phase4"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ----- Write compare_table.csv -----
    table_df.to_csv(out_dir / "compare_table.csv", index=False)

    # ----- Write compare_table.md -----
    md_lines = [
        "## Phase-2 vs Phase-4 atomic carry return comparison",
        "",
        "Aligned daily index: intersection of Phase-2 SR3, Phase-2 VX, Phase-4 sleeve_returns.",
        "",
        "| series | start_date | end_date | n_days | mean_daily_ret | daily_vol | ann_return | ann_vol | sharpe | cum_return | hit_rate | pct_zero | corr_phase2_vs_phase4 |",
        "|--------|------------|----------|--------|----------------|-----------|------------|---------|--------|------------|----------|----------|------------------------|",
    ]
    for _, r in table_df.iterrows():
        corr_val = r.get("corr_phase2_vs_phase4", "")
        if pd.isna(corr_val):
            corr_val = ""
        else:
            corr_val = f"{float(corr_val):.4f}"
        md_lines.append(
            "| {} | {} | {} | {} | {:.6f} | {:.6f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.2%} | {:.2%} | {} |".format(
                r["series"],
                r["start_date"],
                r["end_date"],
                int(r["n_days"]),
                r["mean_daily_ret"],
                r["daily_vol"],
                r["ann_return"],
                r["ann_vol"],
                r["sharpe"] if np.isfinite(r["sharpe"]) else np.nan,
                r["cum_return"],
                r["hit_rate"],
                r["pct_zero"],
                corr_val,
            )
        )
    with open(out_dir / "compare_table.md", "w") as f:
        f.write("\n".join(md_lines))

    # ----- Write notes.md -----
    notes = [
        "# Carry Phase-2 vs Phase-4 comparison — notes",
        "",
        "## Run IDs used",
        "",
        "- **SR3 Phase-2**: path `" + str(PHASE2_SR3_DIR) + "`",
        "  - Source: `reports/phase_index/carry/sr3_calendar_carry/phase2.txt` (path field)",
        "- **VX Phase-2**: path `" + str(PHASE2_VX_DIR) + "`",
        "  - Source: `reports/phase_index/carry/vx_calendar_carry/vx2_vx1_short/phase2.txt` (path field)",
        "- **Phase-4 reference**: run_id `" + args.phase4_run_id + "`",
        "",
        "## Where each series was loaded from",
        "",
        "- **sr3_carry_curve Phase-2**: `" + str(sr3_comp) + "` column `carry`",
        "- **vx_calendar_carry Phase-2**: `" + str(vx_comp) + "` column `vx_carry`",
        "- **sr3_carry_curve Phase-4**: `" + str(phase4_sleeve) + "` column `sr3_carry_curve`",
        "- **vx_calendar_carry Phase-4**: `" + str(phase4_sleeve) + "` column `vx_calendar_carry`",
        "",
        "## Name mapping",
        "",
        "- Phase-2 SR3 column `carry` → sleeve `sr3_carry_curve`",
        "- Phase-2 VX column `vx_carry` → sleeve `vx_calendar_carry`",
        "",
        "## Correlations",
        "",
    ]
    notes.extend([f"- {line}" for line in notes_extra])
    notes.append("")
    notes.append("## Largest discrepancies (summary)")
    notes.append("")
    vol_ratio_sr3 = m_p4_sr3["daily_vol"] / m_p2_sr3["daily_vol"] if m_p2_sr3["daily_vol"] and m_p2_sr3["daily_vol"] > 0 else np.nan
    vol_ratio_vx = m_p4_vx["daily_vol"] / m_p2_vx["daily_vol"] if m_p2_vx["daily_vol"] and m_p2_vx["daily_vol"] > 0 else np.nan
    if not np.isnan(vol_ratio_sr3):
        notes.append(f"- sr3_carry_curve: Phase-4 daily vol / Phase-2 daily vol = {vol_ratio_sr3:.4f}")
    if not np.isnan(vol_ratio_vx):
        notes.append(f"- vx_calendar_carry: Phase-4 daily vol / Phase-2 daily vol = {vol_ratio_vx:.4f}")
    notes.append(f"- sr3_carry_curve Phase-2 vs Phase-4 daily return correlation: {corr_sr3:.4f}")
    notes.append(f"- vx_calendar_carry Phase-2 vs Phase-4 daily return correlation: {corr_vx:.4f}")
    with open(out_dir / "notes.md", "w") as f:
        f.write("\n".join(notes))

    # ----- Print table and paths -----
    print("\n" + "\n".join(md_lines))
    print("\nOutput files:")
    print(f"  {out_dir / 'compare_table.md'}")
    print(f"  {out_dir / 'compare_table.csv'}")
    print(f"  {out_dir / 'notes.md'}")


if __name__ == "__main__":
    main()
