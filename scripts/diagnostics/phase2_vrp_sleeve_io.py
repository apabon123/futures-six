"""
Emit Phase-2 VRP atomic sleeve return CSV for Phase-2 vs Phase-4 identity checks.

Writes sleeve_returns.csv with columns: date, vrp_core_meta, vrp_convergence_meta, vrp_alt_meta
to the given run directory. Uses same column naming as Phase-4.

Only fills with 0: (1) columns for sleeves not present in this run; (2) gaps/after-range
within an aligned series. Does not pre-fill leading warmup days with 0; those remain NaN
so they are distinguishable from true zero returns.
"""
from pathlib import Path
from typing import Dict

import pandas as pd


VRP_SLEEVE_COLUMNS = ["vrp_core_meta", "vrp_convergence_meta", "vrp_alt_meta"]
INTERNAL_TO_COLUMN = {"vrp_core": "vrp_core_meta", "vrp_convergence": "vrp_convergence_meta", "vrp_alt": "vrp_alt_meta"}


def write_vrp_sleeve_returns_csv(
    run_dir: Path,
    sleeve_returns: Dict[str, pd.Series],
    date_index: pd.DatetimeIndex,
) -> None:
    """
    Write sleeve_returns.csv (Phase-4 compatible) to run_dir.

    sleeve_returns: dict with optional keys vrp_core, vrp_convergence, vrp_alt (Series).
    date_index: index to align all series to (portfolio trading days).
    Only missing columns are filled with 0. For present series: align to date_index,
    leave leading warmup (dates before first valid value) as NaN; fill gaps and
    post-range with 0 so the written series is comparable.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    data = {}
    for col in VRP_SLEEVE_COLUMNS:
        internal_key = next((k for k, v in INTERNAL_TO_COLUMN.items() if v == col), None)
        s = sleeve_returns.get(internal_key) if internal_key else None
        if s is not None and len(s) > 0:
            aligned = s.reindex(date_index)
            first_valid = s.dropna().index.min() if s.notna().any() else None
            if first_valid is not None:
                # From first valid date onward: fill NaN (gaps or after end) with 0
                mask = aligned.index >= first_valid
                aligned = aligned.copy()
                aligned.loc[mask] = aligned.loc[mask].fillna(0.0)
            # Leading dates (before first_valid) stay NaN
            data[col] = aligned
        else:
            data[col] = pd.Series(0.0, index=date_index)
    df = pd.DataFrame(data, index=date_index)
    df.index.name = "date"
    out_path = run_dir / "sleeve_returns.csv"
    df.to_csv(out_path)
    return None
