from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Helpers
# -----------------------------

def _find_price_col(df: pd.DataFrame) -> str:
    candidates = [
        "settle", "Settlement", "SETTLE",
        "close", "Close", "CLOSE",
        "adj_close", "Adj Close", "adjClose",
        "price", "Price"
    ]
    for c in candidates:
        if c in df.columns:
            return c
    # fallback: last numeric column
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if not num_cols:
        raise ValueError(f"No numeric price column found. Columns: {list(df.columns)}")
    return num_cols[-1]

def _find_date_col(df: pd.DataFrame) -> str:
    candidates = ["date", "Date", "DATE", "timestamp", "time", "Time"]
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"No date column found. Columns: {list(df.columns)}")

def load_series_csv(path: Path) -> pd.Series:
    df = pd.read_csv(path)
    dcol = _find_date_col(df)
    pcol = _find_price_col(df)

    df[dcol] = pd.to_datetime(df[dcol])
    df = df.sort_values(dcol).dropna(subset=[pcol])
    s = df.set_index(dcol)[pcol].astype(float)
    s.name = path.stem
    return s

def to_returns(price: pd.Series) -> pd.Series:
    r = price.pct_change().fillna(0.0)
    r.name = price.name
    return r

def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min())

@dataclass
class InstrumentResult:
    name: str
    start_price: float
    end_price: float
    total_return: float
    max_dd: float
    vol_ann: float

def summarize_returns(r: pd.Series) -> InstrumentResult:
    eq = (1.0 + r).cumprod()
    start_price = float(eq.iloc[0])
    end_price = float(eq.iloc[-1])
    total_return = end_price / start_price - 1.0
    mdd = max_drawdown(eq)
    vol_ann = float(r.std(ddof=0) * np.sqrt(252))
    return InstrumentResult(
        name=r.name or "series",
        start_price=start_price,
        end_price=end_price,
        total_return=total_return,
        max_dd=mdd,
        vol_ann=vol_ann
    )

# -----------------------------
# Main
# -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vx1", type=str, required=True, help="Path to VX1 continuous CSV")
    ap.add_argument("--vx2", type=str, required=True, help="Path to VX2 continuous CSV")
    ap.add_argument("--vx3", type=str, required=True, help="Path to VX3 continuous CSV")
    ap.add_argument("--es",  type=str, required=True, help="Path to ES continuous CSV")
    ap.add_argument("--start", type=str, default="2020-01-02")
    ap.add_argument("--end",   type=str, default="2020-03-31")
    ap.add_argument("--outdir", type=str, default="reports/diagnostics/2020q1_check")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    series = {
        "VX1": load_series_csv(Path(args.vx1)),
        "VX2": load_series_csv(Path(args.vx2)),
        "VX3": load_series_csv(Path(args.vx3)),
        "ES":  load_series_csv(Path(args.es)),
    }

    # Align prices on common dates
    prices = pd.concat(series.values(), axis=1).sort_index()
    start = pd.to_datetime(args.start)
    end = pd.to_datetime(args.end)

    prices = prices.loc[(prices.index >= start) & (prices.index <= end)].dropna(how="any")
    if prices.empty:
        raise RuntimeError("No overlapping data in the requested window after alignment.")

    rets = prices.pct_change().fillna(0.0)
    rets.columns = ["VX1", "VX2", "VX3", "ES"]

    # Summaries
    rows = []
    for c in rets.columns:
        res = summarize_returns(rets[c])
        rows.append({
            "instrument": c,
            "total_return": res.total_return,
            "max_drawdown": res.max_dd,
            "ann_vol": res.vol_ann,
        })

    summary = pd.DataFrame(rows).set_index("instrument").sort_index()
    summary.to_csv(outdir / "summary.csv")

    # Correlation (returns)
    corr = rets.corr()
    corr.to_csv(outdir / "corr.csv")

    # Equity curves plot
    eq = (1.0 + rets).cumprod()
    plt.figure()
    for c in eq.columns:
        plt.plot(eq.index, eq[c], label=c)
    plt.legend()
    plt.title(f"Equity Curves (Normalized) {args.start} → {args.end}")
    plt.tight_layout()
    plt.savefig(outdir / "equity_curves.png", dpi=200)

    # Price level plot (optional sanity)
    plt.figure()
    for c in prices.columns:
        plt.plot(prices.index, prices[c] / prices[c].iloc[0], label=c)
    plt.legend()
    plt.title(f"Price Index (Normalized) {args.start} → {args.end}")
    plt.tight_layout()
    plt.savefig(outdir / "price_index.png", dpi=200)

    print("Saved:")
    print(f"  {outdir / 'summary.csv'}")
    print(f"  {outdir / 'corr.csv'}")
    print(f"  {outdir / 'equity_curves.png'}")
    print(f"  {outdir / 'price_index.png'}")
    print("\nSummary:")
    print(summary)
    print("\nCorrelation:")
    print(corr)

if __name__ == "__main__":
    main()

