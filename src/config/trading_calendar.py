"""
Minimal US exchange (NYSE) trading calendar for rebalance schedule hardening.

Used so rebalance dates never fall on market holidays (e.g. Good Friday),
avoiding "No signal available" when VRP/other features have no data.

No external dependencies; uses a small set of NYSE holidays.
"""

from datetime import date, timedelta
from typing import Set


def _easter(year: int) -> date:
    """Anonymous Gregorian Easter Sunday. Returns Easter date for given year."""
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    g = (8 * b + 13) // 25
    h = (19 * a + b - d - g + 15) % 30
    j = c // 4
    k = c % 4
    m = (a + 11 * h) // 319
    r = (2 * e + 2 * j - k - h + m + 32) % 7
    n = (h - m + r + 90) // 25
    p = (h - m + r + n + 19) % 32
    return date(year, n, p)


def _good_friday(year: int) -> date:
    """Good Friday = Easter Sunday - 2 days."""
    return _easter(year) - timedelta(days=2)


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    """nth weekday of month (weekday 0=Mon, 6=Sun). n=1 is first, n=-1 is last."""
    if n >= 1:
        first = date(year, month, 1)
        off = (weekday - first.weekday()) % 7
        if n > 1:
            off += 7 * (n - 1)
        return first + timedelta(days=off)
    else:
        last_day = date(year, month + 1, 1) - timedelta(days=1)
        # last occurrence of weekday in month
        back = (last_day.weekday() - weekday) % 7
        return last_day - timedelta(days=back)


def _us_holidays_for_year(year: int) -> Set[date]:
    """NYSE-style holidays for one year (set of dates)."""
    h = set()
    # New Year (Jan 1, or Mon if weekend)
    d = date(year, 1, 1)
    if d.weekday() == 5:  # Saturday -> Friday
        d -= timedelta(days=1)
    elif d.weekday() == 6:  # Sunday -> Monday
        d += timedelta(days=1)
    h.add(d)
    # MLK Jr (3rd Mon Jan)
    h.add(_nth_weekday(year, 1, 0, 3))
    # Presidents (3rd Mon Feb)
    h.add(_nth_weekday(year, 2, 0, 3))
    # Good Friday
    h.add(_good_friday(year))
    # Memorial (last Mon May)
    h.add(_nth_weekday(year, 5, 0, -1))
    # Juneteenth (June 19, or next weekday if weekend)
    d = date(year, 6, 19)
    if d.weekday() == 5:
        d -= timedelta(days=1)
    elif d.weekday() == 6:
        d += timedelta(days=1)
    h.add(d)
    # Independence (July 4)
    d = date(year, 7, 4)
    if d.weekday() == 5:
        d -= timedelta(days=1)
    elif d.weekday() == 6:
        d += timedelta(days=1)
    h.add(d)
    # Labor (1st Mon Sep)
    h.add(_nth_weekday(year, 9, 0, 1))
    # Thanksgiving (4th Thu Nov)
    h.add(_nth_weekday(year, 11, 3, 4))
    # Christmas (Dec 25, or Fri/Mon if weekend)
    d = date(year, 12, 25)
    if d.weekday() == 5:
        d -= timedelta(days=1)
    elif d.weekday() == 6:
        d += timedelta(days=1)
    h.add(d)
    return h


# Cache holidays for a range of years (backtest window + buffer)
_HOLIDAYS_CACHE: Set[date] = set()


def _ensure_holidays_cache(start: date, end: date) -> None:
    global _HOLIDAYS_CACHE
    for y in range(start.year - 1, end.year + 2):
        _HOLIDAYS_CACHE.update(_us_holidays_for_year(y))


def is_us_trading_day(d: date) -> bool:
    """
    True if date is a US exchange (NYSE) trading day.
    Weekend and NYSE holidays return False.
    """
    if d.weekday() >= 5:  # Saturday=5, Sunday=6
        return False
    _ensure_holidays_cache(d, d)
    return d not in _HOLIDAYS_CACHE


def previous_us_trading_day(d: date) -> date:
    """Previous US trading day (does not include d)."""
    out = d
    while True:
        out -= timedelta(days=1)
        if is_us_trading_day(out):
            return out
