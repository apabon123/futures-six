#!/usr/bin/env python3
"""
FRED Series Downloader

Downloads FRED economic indicators and saves to parquet format.
Config-driven via configs/fred_series.yaml.
"""

import os
import json
import pathlib
import pandas as pd
from datetime import datetime
from fredapi import Fred
import yaml

CFG = yaml.safe_load(open("configs/fred_series.yaml"))

FRED_KEY = os.getenv("FRED_API_KEY")
fred = Fred(api_key=FRED_KEY)

OUT_DIR = pathlib.Path(CFG["options"]["parquet_dir"])
OUT_DIR.mkdir(parents=True, exist_ok=True)


def dailyize(df, freq="B"):
    """
    Convert series to daily frequency with forward-fill.
    
    Args:
        df: DataFrame with index=date (datetime), column='value'
        freq: Target frequency (default: 'B' for business days)
    
    Returns:
        DataFrame with dailyized values
    """
    s = df["value"].asfreq(freq, method="pad")  # forward-fill to biz days
    return s.to_frame("value")


def fetch_series(sid, start):
    """
    Fetch a FRED series and dailyize it.
    
    Args:
        sid: FRED series ID
        start: Start date (string or datetime)
    
    Returns:
        DataFrame with columns: date, value, series_id, source, last_updated
    """
    obs = fred.get_series(sid, observation_start=start)  # pandas Series
    df = pd.DataFrame({"date": pd.to_datetime(obs.index), "value": obs.values})
    df = df.set_index("date").sort_index()
    df = dailyize(df, "B")
    df["series_id"] = sid
    df["source"] = "FRED"
    df["last_updated"] = pd.Timestamp.utcnow()
    return df.reset_index()


def main():
    """Main download function."""
    start = CFG["options"]["start"]
    
    for s in CFG["series"]:
        sid = s["id"]
        df = fetch_series(sid, start)
        outp = OUT_DIR / f"{sid}.parquet"
        df.to_parquet(outp, index=False)
        print(f"[OK] {sid}: {len(df)} rows → {outp}")
    
    # (optional) also write a union file for DuckDB scans
    union = []
    for p in OUT_DIR.glob("*.parquet"):
        if p.name != "_all_fred.parquet":  # Skip the union file itself
            union.append(pd.read_parquet(p))
    
    if union:
        all_df = pd.concat(union, ignore_index=True)
        all_df.to_parquet(OUT_DIR / "_all_fred.parquet", index=False)
        print("[OK] wrote _all_fred.parquet")


if __name__ == "__main__":
    main()

