#!/usr/bin/env python3
"""
Preflight Coverage Check

Validates that the canonical DuckDB contains data for all required series
over a target date window. Exits non-zero if any series is missing or
its coverage is insufficient.

Usage:
    python scripts/preflight/check_window_coverage.py --start 2024-03-01 --end 2024-04-30
    python scripts/preflight/check_window_coverage.py --start 2020-01-01 --end 2024-10-31

Required series (VRP / VIX meta-sleeves):
    f_fred_observations : VIXCLS
    market_data_cboe    : VIX3M, VVIX
    market_data         : @VX=101XN, @VX=201XN, @VX=301XN
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Tuple

import yaml

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Series definitions
# ---------------------------------------------------------------------------
# Each tuple: (table, filter_column, filter_value, date_column, display_name)
REQUIRED_SERIES: List[Tuple[str, str, str, str, str]] = [
    # FRED VIX
    ("f_fred_observations", "series_id", "VIXCLS", "date", "FRED VIXCLS (VIX)"),
    # CBOE indices
    ("market_data_cboe", "symbol", "VIX3M", "timestamp::DATE", "CBOE VIX3M"),
    ("market_data_cboe", "symbol", "VVIX", "timestamp::DATE", "CBOE VVIX"),
    # VX futures
    ("market_data", "symbol", "@VX=101XN", "timestamp::DATE", "VX Front (@VX=101XN)"),
    ("market_data", "symbol", "@VX=201XN", "timestamp::DATE", "VX Second (@VX=201XN)"),
    ("market_data", "symbol", "@VX=301XN", "timestamp::DATE", "VX Third (@VX=301XN)"),
]


def _resolve_db_path() -> Path:
    """Resolve canonical DuckDB path from configs/data.yaml."""
    data_yaml = PROJECT_ROOT / "configs" / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"configs/data.yaml not found at {data_yaml}")
    with open(data_yaml, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    raw = cfg.get("db", {}).get("path", "")
    return (PROJECT_ROOT / raw).resolve()


def _find_duckdb(db_dir: Path) -> Path:
    """Find the .duckdb file in the given directory."""
    if db_dir.is_file() and db_dir.suffix == ".duckdb":
        return db_dir
    candidates = sorted(db_dir.glob("*.duckdb"))
    if not candidates:
        raise FileNotFoundError(f"No .duckdb file found in {db_dir}")
    return candidates[0]


def check_coverage(start: str, end: str) -> bool:
    """
    Check data coverage for all required series.

    Returns True if all series pass, False otherwise.
    """
    import duckdb

    db_dir = _resolve_db_path()
    db_file = _find_duckdb(db_dir)

    print(f"[PREFLIGHT] DB: {db_file}")
    stat = db_file.stat()
    print(f"[PREFLIGHT] DB size: {stat.st_size / (1024 * 1024):.1f} MB")
    print(f"[PREFLIGHT] DB mtime: {datetime.fromtimestamp(stat.st_mtime).isoformat()}")
    print(f"[PREFLIGHT] Target window: {start} → {end}\n")

    con = duckdb.connect(str(db_file), read_only=True)

    all_pass = True
    results = []

    for table, filter_col, filter_val, date_col, display in REQUIRED_SERIES:
        try:
            # Check if table exists
            tables = [r[0] for r in con.execute("SHOW TABLES").fetchall()]
            if table not in tables:
                results.append((display, "FAIL", f"Table '{table}' does not exist"))
                all_pass = False
                continue

            query = f"""
                SELECT
                    MIN({date_col}) AS min_date,
                    MAX({date_col}) AS max_date,
                    COUNT(*)        AS row_count
                FROM {table}
                WHERE {filter_col} = ?
            """
            row = con.execute(query, [filter_val]).fetchone()
            min_date, max_date, count = row

            if count == 0:
                results.append((display, "FAIL", f"No rows found for {filter_col}={filter_val}"))
                all_pass = False
                continue

            min_str = str(min_date)[:10]
            max_str = str(max_date)[:10]

            # Check coverage bounds
            covers_start = min_str <= start
            covers_end = max_str >= end
            status = "PASS" if (covers_start and covers_end) else "FAIL"
            if status == "FAIL":
                all_pass = False

            detail = (
                f"rows={count:,}  range=[{min_str} → {max_str}]  "
                f"covers_start={'✓' if covers_start else '✗'}  "
                f"covers_end={'✓' if covers_end else '✗'}"
            )
            results.append((display, status, detail))

        except Exception as e:
            results.append((display, "FAIL", f"Error: {e}"))
            all_pass = False

    con.close()

    # Print report
    print(f"{'Series':<30} {'Status':<8} {'Detail'}")
    print("-" * 100)
    for display, status, detail in results:
        marker = "✓" if status == "PASS" else "✗"
        print(f"{display:<30} {marker} {status:<6} {detail}")

    print()
    if all_pass:
        print("✓ PREFLIGHT PASS: All series cover the target window.")
    else:
        print("✗ PREFLIGHT FAIL: One or more series do not cover the target window.")

    return all_pass


def main():
    parser = argparse.ArgumentParser(description="Preflight coverage check for target window")
    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument(
        "--profile", type=str, default=None,
        help="(Reserved for future use) Strategy profile name.",
    )
    args = parser.parse_args()

    passed = check_coverage(args.start, args.end)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
