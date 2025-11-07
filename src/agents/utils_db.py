"""
Read-only database connection utilities for MarketData broker.

This module provides safe, read-only connections to DuckDB or SQLite databases
and schema discovery functions to locate OHLCV tables without hardcoding names.
"""

import logging
from pathlib import Path
from typing import Tuple, Optional, List
import glob

logger = logging.getLogger(__name__)


def open_readonly_connection(db_path: str):
    """
    Open a read-only connection to a DuckDB or SQLite database.
    
    Args:
        db_path: Path to database file or directory containing database
        
    Returns:
        Database connection object (DuckDB or SQLite)
        
    Raises:
        FileNotFoundError: If database path doesn't exist
        ValueError: If database type cannot be determined or read-only mode unavailable
        RuntimeError: If connection cannot be established
    """
    path = Path(db_path)
    
    # Check if path exists
    if not path.exists():
        raise FileNotFoundError(f"Database path does not exist: {db_path}")
    
    # If directory, search for .duckdb files
    if path.is_dir():
        duckdb_files = glob.glob(str(path / "*.duckdb"))
        if not duckdb_files:
            raise FileNotFoundError(f"No .duckdb files found in directory: {db_path}")
        
        # Pick the largest file (most likely to be the main database)
        db_file = max(duckdb_files, key=lambda f: Path(f).stat().st_size)
        logger.info(f"Found DuckDB file in directory: {db_file}")
        path = Path(db_file)
    
    # Determine database type
    db_type = None
    
    if path.suffix == '.duckdb' or str(path).endswith('.duckdb'):
        db_type = 'duckdb'
    elif path.suffix in ['.db', '.sqlite', '.sqlite3']:
        db_type = 'sqlite'
    else:
        # Try to detect by attempting connections
        logger.warning(f"Cannot determine DB type from extension: {path.suffix}. Attempting auto-detection.")
    
    # Try DuckDB first (preferred)
    if db_type in ['duckdb', None]:
        try:
            import duckdb
            conn = duckdb.connect(str(path), read_only=True)
            logger.info(f"[READ-ONLY] Connected to DuckDB: {path}")
            return conn
        except ImportError:
            if db_type == 'duckdb':
                raise ValueError("DuckDB file detected but duckdb package not installed")
            logger.debug("DuckDB not available, trying SQLite")
        except Exception as e:
            if db_type == 'duckdb':
                raise RuntimeError(f"Failed to connect to DuckDB: {e}")
            logger.debug(f"DuckDB connection failed: {e}, trying SQLite")
    
    # Try SQLite
    if db_type in ['sqlite', None]:
        try:
            import sqlite3
            # SQLite read-only URI mode
            uri = f"file:{path}?mode=ro"
            conn = sqlite3.connect(uri, uri=True)
            logger.info(f"[READ-ONLY] Connected to SQLite: {path}")
            return conn
        except Exception as e:
            raise RuntimeError(f"Failed to connect to SQLite in read-only mode: {e}")
    
    raise ValueError(f"Could not establish read-only connection to: {db_path}")


def find_ohlcv_table(conn) -> str:
    """
    Discover the OHLCV table by scanning for required columns.
    
    Required columns (case-insensitive):
        - date (or trading_date), symbol (or contract_series), open, high, low, close, volume
    
    Args:
        conn: Database connection (DuckDB or SQLite)
        
    Returns:
        Name of the table containing OHLCV data
        
    Raises:
        ValueError: If no suitable table is found
    """
    # Required columns with alternatives
    required_cols = {'open', 'high', 'low', 'close', 'volume'}
    date_cols = {'date', 'trading_date'}
    symbol_cols = {'symbol', 'contract_series'}
    
    try:
        # Get connection type
        conn_type = type(conn).__module__
        
        if 'duckdb' in conn_type:
            # DuckDB: query information schema
            tables_query = """
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'main'
            """
            tables_result = conn.execute(tables_query).fetchall()
            table_names = [row[0] for row in tables_result]
            
        elif 'sqlite' in conn_type:
            # SQLite: query sqlite_master
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            table_names = [row[0] for row in cursor.fetchall()]
        else:
            raise ValueError(f"Unsupported connection type: {conn_type}")
        
        logger.debug(f"[READ-ONLY] Found tables: {table_names}")
        
        # Check each table for required columns
        candidates = []
        
        for table in table_names:
            try:
                # Query to get column names
                if 'duckdb' in conn_type:
                    cols_query = f"DESCRIBE {table}"
                    cols_result = conn.execute(cols_query).fetchall()
                    columns = {row[0].lower() for row in cols_result}
                else:
                    cursor = conn.cursor()
                    cursor.execute(f"PRAGMA table_info({table})")
                    columns = {row[1].lower() for row in cursor.fetchall()}
                
                # Check if all required columns are present
                has_required = required_cols.issubset(columns)
                has_date = bool(date_cols.intersection(columns))
                has_symbol = bool(symbol_cols.intersection(columns))
                
                if has_required and has_date and has_symbol:
                    # Get row count
                    if 'duckdb' in conn_type:
                        count_result = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
                        row_count = count_result[0]
                    else:
                        cursor = conn.cursor()
                        cursor.execute(f"SELECT COUNT(*) FROM {table}")
                        row_count = cursor.fetchone()[0]
                    
                    candidates.append((table, row_count))
                    logger.debug(f"[READ-ONLY] Table '{table}' matches OHLCV schema with {row_count} rows")
            
            except Exception as e:
                logger.debug(f"Error inspecting table {table}: {e}")
                continue
        
        if not candidates:
            raise ValueError(
                f"No table found with required columns: {required_cols}. "
                f"Available tables: {table_names}"
            )
        
        # Return table with most rows
        best_table = max(candidates, key=lambda x: x[1])
        logger.info(f"[READ-ONLY] Selected OHLCV table: '{best_table[0]}' with {best_table[1]} rows")
        
        return best_table[0]
    
    except Exception as e:
        raise ValueError(f"Failed to discover OHLCV table: {e}")


def validate_readonly(conn) -> bool:
    """
    Validate that connection is truly read-only by attempting a write.
    
    Args:
        conn: Database connection
        
    Returns:
        True if connection is read-only (write attempt fails)
        
    Raises:
        RuntimeError: If write attempt succeeds (connection is NOT read-only)
    """
    try:
        conn_type = type(conn).__module__
        
        if 'duckdb' in conn_type:
            # Try to create a temp table
            conn.execute("CREATE TEMP TABLE _readonly_test (id INTEGER)")
            raise RuntimeError("Connection is NOT read-only - write succeeded!")
        else:
            cursor = conn.cursor()
            cursor.execute("CREATE TEMP TABLE _readonly_test (id INTEGER)")
            raise RuntimeError("Connection is NOT read-only - write succeeded!")
    
    except Exception as e:
        error_msg = str(e).lower()
        if 'read-only' in error_msg or 'readonly' in error_msg or 'cannot' in error_msg:
            logger.debug("[READ-ONLY] Validated: connection is read-only")
            return True
        else:
            # Re-raise if it's our custom error
            if "NOT read-only" in str(e):
                raise
            logger.warning(f"Unexpected error during read-only validation: {e}")
            return True


def get_column_names(conn, table_name: str) -> List[str]:
    """
    Get column names for a table.
    
    Args:
        conn: Database connection
        table_name: Name of table
        
    Returns:
        List of column names (original case preserved)
    """
    conn_type = type(conn).__module__
    
    if 'duckdb' in conn_type:
        result = conn.execute(f"DESCRIBE {table_name}").fetchall()
        return [row[0] for row in result]
    else:
        cursor = conn.cursor()
        cursor.execute(f"PRAGMA table_info({table_name})")
        return [row[1] for row in cursor.fetchall()]

