"""
Emit Phase-2 VRP atomic sleeve return CSV for Phase-2 vs Phase-4 identity checks.

Writes sleeve_returns.csv with columns: date, vrp_core_meta, vrp_convergence_meta, vrp_alt_meta
to the given run directory. Uses same column naming as Phase-4.
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
    date_index: index to align all series to (e.g. common_dates from portfolio returns).
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    data = {}
    for col in VRP_SLEEVE_COLUMNS:
        # Find internal key (e.g. vrp_core -> vrp_core_meta)
        internal_key = next((k for k, v in INTERNAL_TO_COLUMN.items() if v == col), None)
        s = sleeve_returns.get(internal_key) if internal_key else None
        if s is not None and len(s) > 0:
            data[col] = s.reindex(date_index).fillna(0.0)
        else:
            data[col] = pd.Series(0.0, index=date_index)
    df = pd.DataFrame(data, index=date_index)
    df.index.name = "date"
    out_path = run_dir / "sleeve_returns.csv"
    df.to_csv(out_path)
    return None
