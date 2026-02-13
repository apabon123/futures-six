#!/usr/bin/env python3
"""
Check that the rebalance schedule excludes Good Friday (uses US trading calendar).

Run from repo root:
    python scripts/diagnostics/check_rebalance_good_friday.py

Expect: 2024-03-29 is NOT in rebalance_dates; 2024-03-28 (Thu) is the rebalance for that week.
"""

import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.agents.exec_sim import ExecSim
from src.config.trading_calendar import is_us_trading_day


class MockMarketData:
    """Minimal mock: trading_days() returns business days in range (no holiday awareness)."""
    def __init__(self, start, end):
        self._dates = pd.date_range(start=start, end=end, freq="B")
    def trading_days(self, symbols=None):
        return self._dates


class MockRiskVol:
    def mask(self, market, date):
        return pd.Series(True, index=["ES", "NQ"])  # arbitrary


def main():
    start = "2024-03-01"
    end = "2024-04-30"
    market = MockMarketData(start, end)
    risk_vol = MockRiskVol()
    exec_sim = ExecSim(rebalance="W-FRI")
    rebalance_dates = exec_sim._build_rebalance_dates(market, risk_vol, start, end)

    good_friday = pd.Timestamp("2024-03-29")
    thursday_before = pd.Timestamp("2024-03-28")

    in_list = good_friday in rebalance_dates
    thursday_in_list = thursday_before in rebalance_dates

    print("Rebalance dates 2024-03-01 .. 2024-04-30 (W-FRI with US calendar):")
    for d in rebalance_dates:
        mark = "  <-- week of Good Friday" if d == thursday_before else ""
        print(f"  {d.date()}{mark}")
    print()
    print(f"2024-03-29 (Good Friday) in rebalance_dates: {in_list}")
    print(f"2024-03-28 (Thursday) in rebalance_dates: {thursday_in_list}")
    print(f"is_us_trading_day(2024-03-29): {is_us_trading_day(good_friday.date())}")

    if in_list:
        print("\nFAIL: Good Friday should not be a rebalance date.")
        return 1
    if not thursday_in_list:
        print("\nFAIL: Thursday 2024-03-28 should be the rebalance for that week.")
        return 1
    print("\nPASS: Good Friday excluded; rebalance for that week is Thursday 2024-03-28.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
