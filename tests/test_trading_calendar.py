"""
Tests for US exchange trading calendar (rebalance schedule hardening).
"""

import pytest
import sys
from pathlib import Path
from datetime import date

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config.trading_calendar import (
    is_us_trading_day,
    previous_us_trading_day,
    _good_friday,
)


class TestTradingCalendar:
    """Good Friday and rebalance hardening."""

    def test_good_friday_2024_not_trading_day(self):
        """2024-03-29 (Good Friday) must not be a US trading day."""
        good_friday_2024 = date(2024, 3, 29)
        assert not is_us_trading_day(good_friday_2024)

    def test_week_of_good_friday_rebalance_is_thursday(self):
        """For week of Good Friday 2024, previous US trading day is Thursday 2024-03-28."""
        good_friday_2024 = date(2024, 3, 29)
        prev = previous_us_trading_day(good_friday_2024)
        assert prev == date(2024, 3, 28)
        assert prev.weekday() == 3  # Thursday

    def test_good_friday_dates(self):
        """Good Friday computation matches known dates."""
        assert _good_friday(2024) == date(2024, 3, 29)
        assert _good_friday(2025) == date(2025, 4, 18)
        assert _good_friday(2023) == date(2023, 4, 7)

    def test_regular_friday_is_trading_day(self):
        """A normal Friday is a trading day."""
        assert is_us_trading_day(date(2024, 3, 22))  # Fri
        assert is_us_trading_day(date(2024, 4, 5))   # Fri

    def test_weekend_not_trading_day(self):
        """Saturday and Sunday are not trading days."""
        assert not is_us_trading_day(date(2024, 3, 30))  # Sat
        assert not is_us_trading_day(date(2024, 3, 31))  # Sun
