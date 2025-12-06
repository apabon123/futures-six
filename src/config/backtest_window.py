"""
Canonical Backtest Window Configuration

Single source of truth for all backtest date ranges across the project.
All runs, diagnostics, and Phase scripts must use these dates to ensure
apples-to-apples comparisons per the Run Consistency Contract.

See: docs/PROCEDURES.md § 2 "Run Consistency Contract"
"""

# Canonical start date for all backtests
# Rationale: 2020-01-01 provides ~5 years of data after warmup,
#            includes COVID stress test, and ensures all sleeves
#            (including LT 252d lookback) are fully warmed up.
CANONICAL_START = "2020-01-01"

# Canonical end date for all backtests
# Set to None to use latest available data, or specify explicit date
CANONICAL_END = "2025-10-31"

# Effective start after warmup (approximate)
# For Trend Meta-Sleeve with 252d max lookback:
# 2020-01-01 + 252 trading days ≈ 2020-12-15
# This is logged automatically by run_strategy.py

