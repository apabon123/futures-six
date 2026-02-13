#!/usr/bin/env python3
"""
VRP "No signal available" diagnostic: check presence of required series on specific rebalance dates.

Run from repo root:
    uv run python scripts/diagnostics/vrp_no_signal_dates_diagnostic.py

Uses same DuckDB as futures-six: configs/data.yaml -> db.path (../databento-es-options/data/silver).
No code changes to application; diagnostics only.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import yaml
import duckdb
import pandas as pd

# Rebalance dates where VRP reported "No signal available"
# 2024-03-29 = Good Friday (US market closed; no CBOE/VX/ES data expected)
DATES = ["2024-03-29", "2024-04-05", "2024-04-12", "2024-04-19"]


def get_db_path():
    root = Path(__file__).resolve().parents[2]
    cfg = root / "configs" / "data.yaml"
    if not cfg.exists():
        raise FileNotFoundError(f"Config not found: {cfg}")
    with open(cfg, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    path = Path(data["db"]["path"])
    if not path.is_absolute():
        path = (root / path).resolve()
    return path


def run_diagnostic(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    rows = []
    for d in DATES:
        row = {"date": d}

        # f_fred_observations: VIXCLS, VXVCLS (VIX3M)
        for sid in ["VIXCLS", "VXVCLS"]:
            try:
                r = conn.execute(
                    "SELECT 1 FROM f_fred_observations WHERE series_id = ? AND date = ?",
                    [sid, d],
                ).fetchone()
                row[f"fred_{sid}"] = "Y" if r else "N"
            except Exception:
                row[f"fred_{sid}"] = "err"

        # market_data_cboe: VVIX, VIX3M (timestamp may be date or datetime)
        for sym in ["VVIX", "VIX3M"]:
            try:
                r = conn.execute(
                    """
                    SELECT 1 FROM market_data_cboe
                    WHERE symbol = ? AND CAST(timestamp AS DATE) = ?
                    LIMIT 1
                    """,
                    [sym, d],
                ).fetchone()
                row[f"cboe_{sym}"] = "Y" if r else "N"
            except Exception:
                row[f"cboe_{sym}"] = "err"

        # market_data: @VX=101XN, 201XN, 301XN
        for sym in ["@VX=101XN", "@VX=201XN", "@VX=301XN"]:
            try:
                r = conn.execute(
                    """
                    SELECT 1 FROM market_data
                    WHERE symbol = ? AND CAST(timestamp AS DATE) = ?
                      AND interval_value = 1 AND interval_unit = 'daily'
                    LIMIT 1
                    """,
                    [sym, d],
                ).fetchone()
                key = f"md_{sym.replace('@VX=', 'VX')}"
                row[key] = "Y" if r else "N"
            except Exception:
                row[f"md_{sym.replace('@VX=', 'VX')}"] = "err"

        # g_continuous_bar_daily: ES_FRONT_CALENDAR_2D (RV)
        try:
            r = conn.execute(
                """
                SELECT 1 FROM g_continuous_bar_daily
                WHERE contract_series = ? AND trading_date = ?
                LIMIT 1
                """,
                ["ES_FRONT_CALENDAR_2D", d],
            ).fetchone()
            row["ohlcv_ES"] = "Y" if r else "N"
        except Exception:
            row["ohlcv_ES"] = "err"

        rows.append(row)

    return pd.DataFrame(rows)


def main():
    db_path = get_db_path()
    if not db_path.exists():
        print(f"DB path does not exist: {db_path}")
        print("Resolve db.path in configs/data.yaml or ensure databento-es-options silver DB is present.")
        return 1

    # Directory: use same logic as utils_db (largest .duckdb)
    if db_path.is_dir():
        duckdb_files = list(db_path.glob("*.duckdb"))
        if not duckdb_files:
            print(f"No .duckdb file in: {db_path}")
            return 1
        db_file = max(duckdb_files, key=lambda p: p.stat().st_size)
    else:
        db_file = db_path

    conn = duckdb.connect(str(db_file), read_only=True)
    try:
        df = run_diagnostic(conn)
        print("VRP required series on 'No signal' rebalance dates")
        print("=" * 80)
        print(df.to_string(index=False))
        print()
        # Which dates have any missing series
        data_cols = [c for c in df.columns if c != "date"]
        dates_with_gaps = [
            r["date"] for _, r in df.iterrows()
            if any(r[c] in ("N", "err") for c in data_cols)
        ]
        missing = (df == "N").any(axis=0)
        err = (df == "err").any(axis=0)
        missing_cols = [c for c in data_cols if missing.get(c, False) or err.get(c, False)]
        if missing_cols:
            print("Dates with missing data:", ", ".join(dates_with_gaps) or "none")
            if dates_with_gaps == ["2024-03-29"]:
                print("(2024-03-29 is Good Friday — US market closed; no CBOE/VX/ES bars expected.)")
            print()
            print("Recommendation:")
            if dates_with_gaps == ["2024-03-29"]:
                print("  - No backfill needed for 2024-03-29: no trading that day. Use a holiday calendar so rebalance is skipped on Good Friday, or accept no VRP signal on that date.")
            else:
                if any("fred_" in c for c in missing_cols):
                    print("  - Backfill FRED (VIXCLS, VXVCLS) for the missing dates in f_fred_observations.")
                if any("cboe_" in c for c in missing_cols):
                    print("  - Backfill CBOE (VVIX, VIX3M) for the missing dates in market_data_cboe.")
                if any("md_" in c for c in missing_cols):
                    print("  - Rebuild continuous VX (build_vx_continuous) so market_data has rows for those dates.")
                if "ohlcv_ES" in missing_cols:
                    print("  - Ensure g_continuous_bar_daily has ES_FRONT_CALENDAR_2D for those dates (calendar/source).")
                print("  - If data exists under a different date (e.g. weekend), check calendar/join logic.")
        else:
            print("All series present on these dates. If VRP still shows 'No signal', check join logic or feature pipeline.")
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
