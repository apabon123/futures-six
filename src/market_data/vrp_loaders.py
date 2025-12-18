"""
Volatility Risk Premium (VRP) Data Loaders

This module provides clean loader functions to pull VIX, VIX3M, and VX futures
data from the canonical research database (databento-es-options).

Data Sources:
- VIX (1M): FRED series VIXCLS in f_fred_observations
- VIX3M (3M): CBOE index in market_data_cboe (symbol='VIX3M')
- VX1/2/3: VX futures from market_data (@VX=101XN, @VX=201XN, @VX=301XN)

Usage:
    import duckdb
    from src.market_data.vrp_loaders import load_vrp_inputs
    
    con = duckdb.connect("path/to/canonical.duckdb", read_only=True)
    df = load_vrp_inputs(con, start="2020-01-01", end="2025-12-31")
    con.close()

See Also:
- financial-data-system: handles CBOE ingestion (VIX3M, VX futures)
- databento-es-options: canonical research DB with synced VRP data
"""

from typing import Optional
import duckdb
import pandas as pd
import numpy as np


# VX continuous futures symbols (1-day roll, unadjusted)
VX_FRONT_SYMBOL = "@VX=101XN"
VX_SECOND_SYMBOL = "@VX=201XN"
VX_THIRD_SYMBOL = "@VX=301XN"


def load_vix(
    con: duckdb.DuckDBPyConnection,
    start: str,
    end: str,
    series_id: str = "VIXCLS",
) -> pd.DataFrame:
    """
    Load spot VIX (1-month implied volatility index) from FRED mirror in canonical DB.
    
    Data Source:
        Table: f_fred_observations
        Series: VIXCLS (VIX Close)
        Provider: Federal Reserve Economic Data (FRED)
    
    Args:
        con: DuckDB connection to canonical database
        start: Start date (YYYY-MM-DD)
        end: End date (YYYY-MM-DD)
        series_id: FRED series ID (default: "VIXCLS")
    
    Returns:
        DataFrame with columns:
            - date (DATE): Trading date
            - vix (FLOAT): VIX closing value
    
    Example:
        >>> con = duckdb.connect("canonical.duckdb", read_only=True)
        >>> vix = load_vix(con, "2020-01-01", "2025-12-31")
        >>> print(vix.head())
                date    vix
        0 2020-01-02  13.78
        1 2020-01-03  12.24
        ...
    """
    return con.execute(
        """
        SELECT
            date AS date,
            value::DOUBLE AS vix
        FROM f_fred_observations
        WHERE series_id = ?
          AND date BETWEEN ? AND ?
        ORDER BY date
        """,
        [series_id, start, end],
    ).df()


def load_vix3m(
    con: duckdb.DuckDBPyConnection,
    start: str,
    end: str,
    symbol: str = "VIX3M",
) -> pd.DataFrame:
    """
    Load VIX3M (3-month implied volatility index) from canonical DB.
    
    Data Source:
        Table: market_data_cboe
        Symbol: VIX3M
        Provider: CBOE via financial-data-system ETL
    
    Args:
        con: DuckDB connection to canonical database
        start: Start date (YYYY-MM-DD)
        end: End date (YYYY-MM-DD)
        symbol: CBOE symbol (default: "VIX3M")
    
    Returns:
        DataFrame with columns:
            - date (DATE): Trading date
            - vix3m (FLOAT): VIX3M closing value
    
    Example:
        >>> con = duckdb.connect("canonical.duckdb", read_only=True)
        >>> vix3m = load_vix3m(con, "2020-01-01", "2025-12-31")
        >>> print(vix3m.head())
                date  vix3m
        0 2020-01-02  14.52
        1 2020-01-03  13.89
        ...
    """
    return con.execute(
        """
        SELECT
            timestamp::DATE AS date,
            settle::DOUBLE AS vix3m
        FROM market_data_cboe
        WHERE symbol = ?
          AND timestamp::DATE BETWEEN ? AND ?
          AND settle IS NOT NULL
        ORDER BY timestamp
        """,
        [symbol, start, end],
    ).df()


def load_vvix(
    con: duckdb.DuckDBPyConnection,
    start: str,
    end: str,
    symbol: str = "VVIX",
    series_id: str = "VVIXCLS",
) -> pd.DataFrame:
    """
    Load VVIX (volatility of volatility index) from canonical DB.
    
    Tries multiple sources in order:
    1. CBOE table (market_data_cboe) with symbol='VVIX'
    2. FRED table (f_fred_observations) with series_id='VVIXCLS'
    
    Data Sources:
        - Table: market_data_cboe (Symbol: VVIX) - CBOE via financial-data-system ETL
        - Table: f_fred_observations (Series: VVIXCLS) - FRED API
    
    Args:
        con: DuckDB connection to canonical database
        start: Start date (YYYY-MM-DD)
        end: End date (YYYY-MM-DD)
        symbol: CBOE symbol (default: "VVIX")
        series_id: FRED series ID (default: "VVIXCLS")
        
    Returns:
        DataFrame with columns:
            - date (DATE): Trading date
            - vvix (FLOAT): VVIX closing value
    
    Example:
        >>> con = duckdb.connect("canonical.duckdb", read_only=True)
        >>> vvix = load_vvix(con, "2020-01-01", "2025-12-31")
        >>> print(vvix.head())
                date   vvix
        0 2020-01-02  85.23
        1 2020-01-03  82.15
        ...
    """
    # Try CBOE table first
    result = con.execute(
        """
        SELECT
            timestamp::DATE AS date,
            settle::DOUBLE AS vvix
        FROM market_data_cboe
        WHERE symbol = ?
          AND timestamp::DATE BETWEEN ? AND ?
          AND settle IS NOT NULL
        ORDER BY timestamp
        """,
        [symbol, start, end],
    ).df()
    
    # If no data from CBOE, try FRED
    if len(result) == 0:
        result = con.execute(
            """
            SELECT
                date AS date,
                value::DOUBLE AS vvix
            FROM f_fred_observations
            WHERE series_id = ?
              AND date BETWEEN ? AND ?
            ORDER BY date
            """,
            [series_id, start, end],
        ).df()
    
    return result


def load_vx_curve(
    con: duckdb.DuckDBPyConnection,
    start: str,
    end: str,
    front_symbol: str = VX_FRONT_SYMBOL,
    second_symbol: str = VX_SECOND_SYMBOL,
    third_symbol: str = VX_THIRD_SYMBOL,
) -> pd.DataFrame:
    """
    Load VX1/2/3 continuous futures (unadjusted, 1-day roll) from canonical DB.
    
    Data Source:
        Table: market_data
        Symbols:
            - @VX=101XN: VX front month (VX1)
            - @VX=201XN: VX second month (VX2)
            - @VX=301XN: VX third month (VX3)
        Provider: CBOE via financial-data-system ETL
    
    Args:
        con: DuckDB connection to canonical database
        start: Start date (YYYY-MM-DD)
        end: End date (YYYY-MM-DD)
        front_symbol: VX1 symbol (default: "@VX=101XN")
        second_symbol: VX2 symbol (default: "@VX=201XN")
        third_symbol: VX3 symbol (default: "@VX=301XN")
    
    Returns:
        DataFrame with columns:
            - date (DATE): Trading date
            - vx1 (FLOAT): VX front month closing price
            - vx2 (FLOAT): VX second month closing price
            - vx3 (FLOAT): VX third month closing price
    
    Example:
        >>> con = duckdb.connect("canonical.duckdb", read_only=True)
        >>> vx = load_vx_curve(con, "2020-01-01", "2025-12-31")
        >>> print(vx.head())
                date    vx1    vx2    vx3
        0 2020-01-02  13.85  14.20  14.50
        1 2020-01-03  12.95  13.45  13.80
        ...
    """
    return con.execute(
        """
        WITH vx1 AS (
            SELECT timestamp::DATE AS date, close::DOUBLE AS vx1
            FROM market_data
            WHERE symbol = ?
              AND timestamp::DATE BETWEEN ? AND ?
        ),
        vx2 AS (
            SELECT timestamp::DATE AS date, close::DOUBLE AS vx2
            FROM market_data
            WHERE symbol = ?
              AND timestamp::DATE BETWEEN ? AND ?
        ),
        vx3 AS (
            SELECT timestamp::DATE AS date, close::DOUBLE AS vx3
            FROM market_data
            WHERE symbol = ?
              AND timestamp::DATE BETWEEN ? AND ?
        )
        SELECT
            COALESCE(vx1.date, vx2.date, vx3.date) AS date,
            vx1.vx1,
            vx2.vx2,
            vx3.vx3
        FROM vx1
        FULL OUTER JOIN vx2 USING(date)
        FULL OUTER JOIN vx3 USING(date)
        ORDER BY date
        """,
        [front_symbol, start, end,
         second_symbol, start, end,
         third_symbol, start, end],
    ).df()


def load_vrp_inputs(
    con: duckdb.DuckDBPyConnection,
    start: str,
    end: str,
) -> pd.DataFrame:
    """
    Convenience loader returning all VRP inputs in a single DataFrame.
    
    Combines VIX (1M), VIX3M (3M), and VX1/2/3 futures data.
    
    Args:
        con: DuckDB connection to canonical database
        start: Start date (YYYY-MM-DD)
        end: End date (YYYY-MM-DD)
    
    Returns:
        DataFrame with columns:
            - date (DATE): Trading date
            - vix (FLOAT): VIX 1-month implied vol
            - vix3m (FLOAT): VIX 3-month implied vol
            - vx1 (FLOAT): VX front month futures price
            - vx2 (FLOAT): VX second month futures price
            - vx3 (FLOAT): VX third month futures price
    
    Notes:
        - Performs left join from VIX (most complete series)
        - VIX3M starts 2009-09-18 (first available date)
        - VX futures may have missing data on some dates
    
    Example:
        >>> con = duckdb.connect("canonical.duckdb", read_only=True)
        >>> vrp = load_vrp_inputs(con, "2020-01-01", "2025-12-31")
        >>> print(vrp.head())
                date    vix  vix3m    vx1    vx2    vx3
        0 2020-01-02  13.78  14.52  13.85  14.20  14.50
        1 2020-01-03  12.24  13.89  12.95  13.45  13.80
        ...
        
        >>> # Compute VRP spreads
        >>> vrp['vrp_vix_vx1'] = vrp['vix'] - vrp['vx1']
        >>> vrp['term_vix3m_vix'] = vrp['vix3m'] - vrp['vix']
        >>> vrp['slope_vx2_vx1'] = vrp['vx2'] - vrp['vx1']
    """
    vix = load_vix(con, start, end)
    vix3m = load_vix3m(con, start, end)
    vx = load_vx_curve(con, start, end)

    df = (
        vix
        .merge(vix3m, on="date", how="left")
        .merge(vx, on="date", how="left")
        .sort_values("date")
    )
    return df


def load_rv(
    con: duckdb.DuckDBPyConnection,
    start: str,
    end: str,
    symbol: str = "ES_FRONT_CALENDAR_2D",
    lookbacks: list[int] = [5, 10, 21]
) -> pd.DataFrame:
    """
    Load realized volatility (RV) for ES futures computed from daily returns.
    
    Computes rolling realized volatility for specified lookback windows.
    All RVs are expressed in vol points (not decimals).
    
    Data Source:
        Table: market_data
        Symbol: ES_FRONT_CALENDAR_2D (ES continuous contract)
        Provider: CBOE via financial-data-system ETL
    
    Args:
        con: DuckDB connection to canonical database
        start: Start date (YYYY-MM-DD)
        end: End date (YYYY-MM-DD)
        symbol: ES symbol (default: "ES_FRONT_CALENDAR_2D")
        lookbacks: List of lookback windows in days (default: [5, 10, 21])
        
    Returns:
        DataFrame with columns:
            - date (DATE): Trading date
            - rv5 (FLOAT): 5-day realized vol in vol points
            - rv10 (FLOAT): 10-day realized vol in vol points
            - rv21 (FLOAT): 21-day realized vol in vol points
            (only columns for requested lookbacks are included)
    
    Example:
        >>> con = duckdb.connect("canonical.duckdb", read_only=True)
        >>> rv = load_rv(con, "2020-01-01", "2025-12-31", lookbacks=[5, 21])
        >>> print(rv.head())
                date    rv5   rv21
        0 2020-01-02  15.2   18.5
        1 2020-01-03  14.8   18.2
        ...
    """
    # Load ES prices
    result = con.execute(
        """
        SELECT
            timestamp::DATE AS date,
            close::DOUBLE AS close
        FROM market_data
        WHERE symbol = ?
          AND timestamp::DATE BETWEEN ? AND ?
        ORDER BY timestamp
        """,
        [symbol, start, end]
    ).df()
    
    if result.empty:
        # Return empty DataFrame with date column
        return pd.DataFrame(columns=['date'] + [f'rv{lb}' for lb in lookbacks])
    
    # Convert date to datetime if needed
    result['date'] = pd.to_datetime(result['date'])
    result = result.set_index('date').sort_index()
    
    # Compute log returns
    result['log_return'] = np.log(result['close']).diff()
    
    # Compute realized volatility for each lookback
    rv_df = pd.DataFrame({'date': result.index})
    
    for lookback in lookbacks:
        # Compute rolling std of log returns
        # RV = std(daily_log_returns) * sqrt(252) * 100
        # sqrt(252) annualizes, *100 converts to vol points
        rv = result['log_return'].rolling(
            window=lookback,
            min_periods=lookback
        ).std() * np.sqrt(252) * 100.0
        
        # Add to DataFrame
        col_name = f'rv{lookback}'
        rv_df[col_name] = rv.values
    
    # Drop rows with NaN (before we have enough data for rolling window)
    rv_df = rv_df.dropna().reset_index(drop=True)
    
    return rv_df

