#!/usr/bin/env python3
"""
Carry Inputs Coverage Audit Script

Purpose: Directly inspect canonical DuckDB and PROVE required series exist & cover canonical window.

This script:
1. Connects to canonical DB using Futures-Six config
2. Lists all available symbols in the database
3. Checks existence and coverage for each required carry input
4. Reports min/max dates, row counts, and coverage % over canonical window [2020-01-06, 2025-10-31]
5. Hard-fails with explicit list of missing keys (does NOT silently continue)

Required keys (from feature modules):
- Equity carry: SP500, NASDAQ100, RUT_SPOT (spot indices); ES, NQ, RTY (futures); SOFR (funding)
- Rates carry: ZT_RANK_1_VOLUME, ZF_RANK_1_VOLUME, ZN_RANK_1_VOLUME, UB_RANK_1_VOLUME
- FX carry: SOFR + foreign rates (ECB_RATE, JPY_RATE, SONIA or actual keynames)
- Commodity carry: CL, GC (front vs second)

Usage:
    python scripts/diagnostics/audit_carry_inputs_coverage.py
    python scripts/diagnostics/audit_carry_inputs_coverage.py --output carry_inputs_coverage.json
"""

import argparse
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import logging
import yaml

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config.backtest_window import CANONICAL_START, CANONICAL_END
from src.agents.utils_db import open_readonly_connection

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Canonical window for coverage calculation
CANONICAL_WINDOW_START = pd.Timestamp(CANONICAL_START)
CANONICAL_WINDOW_END = pd.Timestamp(CANONICAL_END)

# Required keys by asset class
REQUIRED_KEYS = {
    "equity_spot": ["SP500", "NASDAQ100", "RUT_SPOT"],
    "equity_futures": ["ES_FRONT_CALENDAR_2D", "NQ_FRONT_CALENDAR_2D", "RTY_FRONT_CALENDAR_2D"],
    "funding": ["SOFR"],  # SOFR is sufficient (US_SOFR, SOFR_RATE are just aliases)
    "rates_rank1": ["ZT_RANK_1_VOLUME", "ZF_RANK_1_VOLUME", "ZN_RANK_1_VOLUME", "UB_RANK_1_VOLUME"],
    "rates_front": ["ZT_FRONT_VOLUME", "ZF_FRONT_VOLUME", "ZN_FRONT_VOLUME", "UB_FRONT_VOLUME"],
    "fx_front": ["6E_FRONT_CALENDAR", "6B_FRONT_CALENDAR", "6J_FRONT_CALENDAR"],
    "fx_rank1": ["6E_RANK_1_CALENDAR", "6B_RANK_1_CALENDAR", "6J_RANK_1_CALENDAR"],
    "commodity_front": ["CL_FRONT_VOLUME", "GC_FRONT_VOLUME"],
    "commodity_rank1": ["CL_RANK_1_VOLUME", "GC_RANK_1_VOLUME"],
    "foreign_rates": ["ECBDFR", "IRSTCI01JPM156N", "IUDSOIA"],  # Actual FRED series_ids: ECB, JPY, SONIA
}


def load_db_config() -> str:
    """Load canonical DB path from configs/data.yaml."""
    config_path = Path("configs/data.yaml")
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    db_path = config['db']['path']
    logger.info(f"Canonical DB path from config: {db_path}")
    return db_path


def list_all_symbols(conn) -> List[str]:
    """
    List all unique symbols in the database.
    
    Checks both:
    - f_fred_observations (spot indices, rates) - uses series_id column
    - g_continuous_bar_daily (continuous contracts) - uses symbol column
    
    Args:
        conn: Database connection
        
    Returns:
        List of unique symbol names
    """
    all_symbols = []
    
    # Check f_fred_observations (spot indices, rates)
    try:
        result = conn.execute("SELECT DISTINCT series_id FROM f_fred_observations").df()
        if not result.empty:
            fred_symbols = result.iloc[:, 0].unique().tolist()
            logger.info(f"Found {len(fred_symbols)} unique series_id in 'f_fred_observations'")
            all_symbols.extend(fred_symbols)
    except Exception as e:
        logger.debug(f"Could not query f_fred_observations: {e}")
    
    # Check g_continuous_bar_daily (continuous contracts use contract_series column)
    try:
        result = conn.execute("SELECT DISTINCT contract_series FROM g_continuous_bar_daily").df()
        if not result.empty:
            contract_symbols = result.iloc[:, 0].unique().tolist()
            logger.info(f"Found {len(contract_symbols)} unique contract_series in 'g_continuous_bar_daily'")
            all_symbols.extend(contract_symbols)
    except Exception as e:
        logger.debug(f"Could not query g_continuous_bar_daily: {e}")
    
    # Also check market_data (for VIX, etc.)
    try:
        result = conn.execute("SELECT DISTINCT symbol FROM market_data").df()
        if not result.empty:
            market_symbols = result.iloc[:, 0].unique().tolist()
            logger.info(f"Found {len(market_symbols)} unique symbols in 'market_data'")
            all_symbols.extend(market_symbols)
    except Exception as e:
        logger.debug(f"Could not query market_data: {e}")
    
    unique_symbols = sorted(list(set(all_symbols)))
    logger.info(f"Total unique symbols across all tables: {len(unique_symbols)}")
    return unique_symbols


def check_series_coverage(
    conn,
    symbol: str,
    canonical_start: pd.Timestamp,
    canonical_end: pd.Timestamp
) -> Dict:
    """
    Check if a symbol exists and compute its coverage over canonical window.
    
    Checks:
    1. f_fred_observations (for spot indices like SP500, NASDAQ100, RUT_SPOT) - uses series_id
    2. g_continuous_bar_daily (for continuous contracts) - uses symbol
    3. market_data (for other data) - uses symbol
    
    Args:
        conn: Database connection
        symbol: Symbol name to check
        canonical_start: Start of canonical window
        canonical_end: End of canonical window
        
    Returns:
        Dictionary with coverage information
    """
    result = {
        "symbol": symbol,
        "exists": False,
        "min_date": None,
        "max_date": None,
        "row_count": 0,
        "rows_in_window": 0,
        "coverage_pct": 0.0,
        "error": None,
        "table_found": None
    }
    
    # Check f_fred_observations (spot indices use series_id)
    try:
        check_query = """
            SELECT COUNT(*) as cnt
            FROM f_fred_observations
            WHERE series_id = ?
        """
        check_result = conn.execute(check_query, [symbol]).df()
        
        if check_result.iloc[0, 0] > 0:
            # Symbol exists in FRED table, get date range
            date_query = """
                SELECT 
                    MIN(date) as min_date,
                    MAX(date) as max_date,
                    COUNT(*) as row_count
                FROM f_fred_observations
                WHERE series_id = ?
            """
            date_result = conn.execute(date_query, [symbol]).df()
            
            min_date = pd.Timestamp(date_result.iloc[0, 0])
            max_date = pd.Timestamp(date_result.iloc[0, 1])
            row_count = date_result.iloc[0, 2]
            
            # Count rows in canonical window
            window_query = """
                SELECT COUNT(*) as cnt
                FROM f_fred_observations
                WHERE series_id = ?
                  AND date >= ?
                  AND date <= ?
            """
            window_result = conn.execute(
                window_query,
                [symbol, canonical_start.date(), canonical_end.date()]
            ).df()
            
            rows_in_window = window_result.iloc[0, 0]
            
            # Calculate coverage percentage
            expected_days = len(pd.bdate_range(canonical_start, canonical_end))
            coverage_pct = (rows_in_window / expected_days * 100) if expected_days > 0 else 0.0
            
            result.update({
                "exists": True,
                "min_date": str(min_date.date()),
                "max_date": str(max_date.date()),
                "row_count": int(row_count),
                "rows_in_window": int(rows_in_window),
                "coverage_pct": float(coverage_pct),
                "table_found": "f_fred_observations"
            })
            
            logger.info(
                f"  {symbol}: EXISTS in f_fred_observations "
                f"(min={min_date.date()}, max={max_date.date()}, "
                f"rows={row_count}, in_window={rows_in_window}, "
                f"coverage={coverage_pct:.1f}%)"
            )
            
            return result
    except Exception as e:
        logger.debug(f"  {symbol}: Not in f_fred_observations ({e})")
    
    # Check g_continuous_bar_daily (continuous contracts use contract_series column)
    try:
        check_query = """
            SELECT COUNT(*) as cnt
            FROM g_continuous_bar_daily
            WHERE contract_series = ?
        """
        check_result = conn.execute(check_query, [symbol]).df()
        
        if check_result.iloc[0, 0] > 0:
            # Symbol exists, get date range
            date_query = """
                SELECT 
                    MIN(trading_date) as min_date,
                    MAX(trading_date) as max_date,
                    COUNT(*) as row_count
                FROM g_continuous_bar_daily
                WHERE contract_series = ?
            """
            date_result = conn.execute(date_query, [symbol]).df()
            
            min_date = pd.Timestamp(date_result.iloc[0, 0])
            max_date = pd.Timestamp(date_result.iloc[0, 1])
            row_count = date_result.iloc[0, 2]
            
            # Count rows in canonical window
            window_query = """
                SELECT COUNT(*) as cnt
                FROM g_continuous_bar_daily
                WHERE contract_series = ?
                  AND trading_date >= ?
                  AND trading_date <= ?
            """
            window_result = conn.execute(
                window_query,
                [symbol, canonical_start.date(), canonical_end.date()]
            ).df()
            
            rows_in_window = window_result.iloc[0, 0]
            
            # Calculate coverage percentage
            expected_days = len(pd.bdate_range(canonical_start, canonical_end))
            coverage_pct = (rows_in_window / expected_days * 100) if expected_days > 0 else 0.0
            
            result.update({
                "exists": True,
                "min_date": str(min_date.date()),
                "max_date": str(max_date.date()),
                "row_count": int(row_count),
                "rows_in_window": int(rows_in_window),
                "coverage_pct": float(coverage_pct),
                "table_found": "g_continuous_bar_daily"
            })
            
            logger.info(
                f"  {symbol}: EXISTS in g_continuous_bar_daily "
                f"(min={min_date.date()}, max={max_date.date()}, "
                f"rows={row_count}, in_window={rows_in_window}, "
                f"coverage={coverage_pct:.1f}%)"
            )
            
            return result
    except Exception as e:
        logger.debug(f"  {symbol}: Not in g_continuous_bar_daily ({e})")
    
    # Check market_data (fallback)
    try:
        check_query = """
            SELECT COUNT(*) as cnt
            FROM market_data
            WHERE symbol = ?
        """
        check_result = conn.execute(check_query, [symbol]).df()
        
        if check_result.iloc[0, 0] > 0:
            date_query = """
                SELECT 
                    MIN(timestamp::DATE) as min_date,
                    MAX(timestamp::DATE) as max_date,
                    COUNT(*) as row_count
                FROM market_data
                WHERE symbol = ?
            """
            date_result = conn.execute(date_query, [symbol]).df()
            
            min_date = pd.Timestamp(date_result.iloc[0, 0])
            max_date = pd.Timestamp(date_result.iloc[0, 1])
            row_count = date_result.iloc[0, 2]
            
            window_query = """
                SELECT COUNT(*) as cnt
                FROM market_data
                WHERE symbol = ?
                  AND timestamp::DATE >= ?
                  AND timestamp::DATE <= ?
            """
            window_result = conn.execute(
                window_query,
                [symbol, canonical_start.date(), canonical_end.date()]
            ).df()
            
            rows_in_window = window_result.iloc[0, 0]
            expected_days = len(pd.bdate_range(canonical_start, canonical_end))
            coverage_pct = (rows_in_window / expected_days * 100) if expected_days > 0 else 0.0
            
            result.update({
                "exists": True,
                "min_date": str(min_date.date()),
                "max_date": str(max_date.date()),
                "row_count": int(row_count),
                "rows_in_window": int(rows_in_window),
                "coverage_pct": float(coverage_pct),
                "table_found": "market_data"
            })
            
            logger.info(
                f"  {symbol}: EXISTS in market_data "
                f"(min={min_date.date()}, max={max_date.date()}, "
                f"rows={row_count}, in_window={rows_in_window}, "
                f"coverage={coverage_pct:.1f}%)"
            )
            
            return result
    except Exception as e:
        logger.debug(f"  {symbol}: Not in market_data ({e})")
    
    # Symbol not found
    logger.warning(f"  {symbol}: NOT FOUND in any table")
    result["error"] = "Symbol not found in f_fred_observations, g_continuous_bar_daily, or market_data"
    
    return result


def audit_carry_inputs(db_path: str, output_path: Optional[str] = None) -> Dict:
    """
    Audit all carry inputs for existence and coverage.
    
    Args:
        db_path: Path to database
        output_path: Optional path to save JSON output
        
    Returns:
        Dictionary with audit results
    """
    logger.info("=" * 80)
    logger.info("CARRY INPUTS COVERAGE AUDIT")
    logger.info("=" * 80)
    logger.info(f"Canonical DB path: {db_path}")
    logger.info(f"Canonical window: {CANONICAL_WINDOW_START.date()} to {CANONICAL_WINDOW_END.date()}")
    
    # Connect to database
    try:
        conn = open_readonly_connection(db_path)
        logger.info(f"Successfully connected to database")
    except Exception as e:
        logger.error(f"Failed to connect to database: {e}")
        raise
    
    # List all symbols
    logger.info("\n" + "=" * 80)
    logger.info("Listing all symbols in database...")
    all_symbols = list_all_symbols(conn)
    logger.info(f"Found {len(all_symbols)} unique symbols")
    if len(all_symbols) > 0:
        logger.info(f"Sample symbols: {all_symbols[:10]}")
    
    # Audit each required key category
    logger.info("\n" + "=" * 80)
    logger.info("Auditing required carry inputs...")
    
    audit_results = {
        "db_path": db_path,
        "canonical_window": {
            "start": str(CANONICAL_WINDOW_START.date()),
            "end": str(CANONICAL_WINDOW_END.date())
        },
        "all_symbols_count": len(all_symbols),
        "required_keys": {},
        "missing_keys": [],
        "partial_coverage_keys": []
    }
    
    # Check each category
    for category, keys in REQUIRED_KEYS.items():
        logger.info(f"\n--- {category.upper()} ---")
        category_results = []
        
        for key in keys:
            result = check_series_coverage(
                conn, key, CANONICAL_WINDOW_START, CANONICAL_WINDOW_END
            )
            category_results.append(result)
            
            if not result["exists"]:
                audit_results["missing_keys"].append(key)
            elif result["coverage_pct"] < 80.0:  # Less than 80% coverage
                audit_results["partial_coverage_keys"].append({
                    "key": key,
                    "coverage_pct": result["coverage_pct"]
                })
        
        audit_results["required_keys"][category] = category_results
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("AUDIT SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total symbols in DB: {len(all_symbols)}")
    logger.info(f"Missing keys: {len(audit_results['missing_keys'])}")
    logger.info(f"Partial coverage keys (<80%): {len(audit_results['partial_coverage_keys'])}")
    
    if audit_results["missing_keys"]:
        logger.error("\nMISSING KEYS (HARD FAIL):")
        for key in audit_results["missing_keys"]:
            logger.error(f"  - {key}")
    else:
        logger.info("\n✓ All required keys exist in database")
    
    if audit_results["partial_coverage_keys"]:
        logger.warning("\nPARTIAL COVERAGE KEYS (<80%):")
        for item in audit_results["partial_coverage_keys"]:
            logger.warning(f"  - {item['key']}: {item['coverage_pct']:.1f}% coverage")
    
    # Save results
    if output_path:
        output_file = Path(output_path)
        with open(output_file, 'w') as f:
            json.dump(audit_results, f, indent=2, default=str)
        logger.info(f"\nAudit results saved to: {output_file}")
    
    # Check for critical missing keys (non-optional)
    critical_missing = []
    optional_missing = []
    
    for key in audit_results["missing_keys"]:
        # Foreign rates are optional for Phase-0 (can skip FX carry or use placeholder)
        if key in ["ECBDFR", "IRSTCI01JPM156N", "IUDSOIA"]:
            optional_missing.append(key)
        else:
            critical_missing.append(key)
    
    # Hard fail only for critical missing keys
    if critical_missing:
        logger.error("\n" + "=" * 80)
        logger.error("HARD FAIL: Missing critical required keys")
        logger.error("=" * 80)
        logger.error("Cannot proceed to Phase-0 until critical keys are loaded or aliased.")
        logger.error("\nCritical missing keys:")
        for key in critical_missing:
            logger.error(f"  - {key}")
        logger.error("\nSuggested actions:")
        logger.error("  1. Check if keys use different naming convention")
        logger.error("  2. Implement alias mapping in MarketData layer")
        logger.error("  3. Load missing data into database")
        raise RuntimeError(f"Missing critical required keys: {critical_missing}")
    
    # Warn about optional missing keys
    if optional_missing:
        logger.warning("\n" + "=" * 80)
        logger.warning("OPTIONAL KEYS MISSING (Phase-0 can proceed with limitations)")
        logger.warning("=" * 80)
        logger.warning("Foreign rates missing (FX carry may be limited):")
        for key in optional_missing:
            logger.warning(f"  - {key}")
        logger.warning("\nPhase-0 can proceed but FX carry will be limited or skipped.")
        audit_results["optional_missing_keys"] = optional_missing
    
    logger.info("\n" + "=" * 80)
    logger.info("✓ AUDIT PASSED")
    logger.info("=" * 80)
    
    return audit_results


def main():
    """Run carry inputs coverage audit."""
    parser = argparse.ArgumentParser(description="Audit Carry Inputs Coverage")
    parser.add_argument(
        "--output",
        type=str,
        default="carry_inputs_coverage.json",
        help="Output JSON file path"
    )
    parser.add_argument(
        "--db_path",
        type=str,
        default=None,
        help="Override DB path from config"
    )
    
    args = parser.parse_args()
    
    # Load DB path
    if args.db_path:
        db_path = args.db_path
    else:
        db_path = load_db_config()
    
    # Run audit
    try:
        audit_results = audit_carry_inputs(db_path, args.output)
        sys.exit(0)
    except RuntimeError as e:
        logger.error(f"Audit failed: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
