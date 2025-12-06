"""
MarketData: Read-only data broker for continuous futures OHLCV.

This module provides a safe, read-only interface to query continuous futures
market data from a local database. Prices are NOT back-adjusted; roll jumps
are flagged but not modified.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple, Union, List, Dict
import yaml
import pandas as pd
import numpy as np
from datetime import datetime

from .utils_db import open_readonly_connection, find_ohlcv_table, validate_readonly

logger = logging.getLogger(__name__)

# Import ContinuousContractBuilder
try:
    from ..services.continuous_contract_builder import ContinuousContractBuilder
except ImportError:
    # Handle absolute import fallback
    try:
        from src.services.continuous_contract_builder import ContinuousContractBuilder
    except ImportError:
        # Handle relative import fallback
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from services.continuous_contract_builder import ContinuousContractBuilder


class MarketData:
    """
    Read-only broker for continuous futures OHLCV data.
    
    Provides standardized access to prices, returns, volatility, and covariance
    without mutating the source database. Supports point-in-time snapshots.
    """
    
    def __init__(
        self,
        db_path: Optional[str] = None,
        universe: Optional[Tuple[str, ...]] = None,
        asof: Optional[Union[str, datetime]] = None,
        config_path: str = "configs/data.yaml"
    ):
        """
        Initialize MarketData broker.
        
        Args:
            db_path: Path to database (overrides config)
            universe: Tuple of symbols to track (overrides config)
            asof: Optional snapshot date for point-in-time queries
            config_path: Path to configuration YAML file
        """
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Override with constructor arguments
        self.db_path = db_path or self.config['db']['path']
        
        # Handle universe config: can be list (old format) or dict (new format)
        if universe is not None:
            self.universe = universe
        else:
            universe_cfg = self.config['universe']
            if isinstance(universe_cfg, dict):
                # New format: dict with roll settings
                # Map short names to database symbols based on roll type
                # Note: FX (6E, 6B, 6J) and SR3 use FRONT_CALENDAR (no _2D suffix)
                # Equities (ES, NQ, RTY) use FRONT_CALENDAR_2D
                db_symbols = []
                fx_symbols = {'6E', '6B', '6J'}
                for short_name, roll_cfg in universe_cfg.items():
                    if isinstance(roll_cfg, dict):
                        roll_type = roll_cfg.get('roll', 'calendar')
                        if roll_type == 'calendar':
                            # FX and SR3 don't have _2D suffix in database
                            if short_name in fx_symbols or short_name == 'SR3':
                                db_symbols.append(f"{short_name}_FRONT_CALENDAR")
                            else:
                                db_symbols.append(f"{short_name}_FRONT_CALENDAR_2D")
                        elif roll_type == 'volume':
                            db_symbols.append(f"{short_name}_FRONT_VOLUME")
                        else:
                            raise ValueError(f"Unknown roll type: {roll_type} for {short_name}")
                    else:
                        # Fallback: assume calendar roll with _2D for non-FX/SR3
                        if short_name in fx_symbols or short_name == 'SR3':
                            db_symbols.append(f"{short_name}_FRONT_CALENDAR")
                        else:
                            db_symbols.append(f"{short_name}_FRONT_CALENDAR_2D")
                self.universe = tuple(db_symbols)
            else:
                # Old format: list of database symbols
                self.universe = tuple(universe_cfg)
        
        self.asof = pd.to_datetime(asof) if asof else None
        
        # Establish read-only connection
        logger.info(f"[READ-ONLY] Initializing MarketData for {len(self.universe)} symbols")
        self.conn = open_readonly_connection(self.db_path)
        
        # Validate read-only connection
        validate_readonly(self.conn)
        
        # Discover OHLCV table
        self.table_name = find_ohlcv_table(self.conn)
        logger.info(f"[READ-ONLY] Using table: {self.table_name}")
        
        # Detect and map column names
        self._column_map = self._detect_column_names()
        logger.info(f"[READ-ONLY] Column mapping: {self._column_map}")
        
        # Check if contract_id column exists (optional)
        self._has_contract_id = self._check_contract_id_column()
        
        # Initialize ContinuousContractBuilder
        self._builder = ContinuousContractBuilder()
        
        # Cache for loaded data
        self._cache = {}
        self._cache_key = None
        
        # Cache for continuous prices (lazy-loaded)
        self._prices_raw_cache = None
        self._prices_cont_cache = None
        self._contract_ids_cache = None
        self._returns_cont_cache = None
    
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(path, 'r') as f:
            config = yaml.safe_load(f)
        
        return config
    
    def _detect_column_names(self) -> Dict[str, str]:
        """
        Detect actual column names in the database table.
        
        Maps standard names to actual column names:
        - 'date' -> 'date' or 'trading_date'
        - 'symbol' -> 'symbol' or 'contract_series'
        
        Returns:
            Dictionary mapping standard names to actual column names
        """
        conn_type = type(self.conn).__module__
        
        # Get actual column names from table
        if 'duckdb' in conn_type:
            result = self.conn.execute(f"DESCRIBE {self.table_name}").fetchall()
            columns = {row[0].lower(): row[0] for row in result}
        else:
            cursor = self.conn.cursor()
            cursor.execute(f"PRAGMA table_info({self.table_name})")
            columns = {row[1].lower(): row[1] for row in cursor.fetchall()}
        
        # Map standard names to actual names
        column_map = {}
        
        # Date column
        if 'date' in columns:
            column_map['date'] = columns['date']
        elif 'trading_date' in columns:
            column_map['date'] = columns['trading_date']
        else:
            raise ValueError("Could not find date or trading_date column")
        
        # Symbol column
        if 'symbol' in columns:
            column_map['symbol'] = columns['symbol']
        elif 'contract_series' in columns:
            column_map['symbol'] = columns['contract_series']
        else:
            raise ValueError("Could not find symbol or contract_series column")
        
        return column_map
    
    def _check_contract_id_column(self) -> bool:
        """
        Check if contract_id column exists in the database table.
        
        Returns:
            True if contract_id column exists, False otherwise
        """
        conn_type = type(self.conn).__module__
        
        # Get actual column names from table
        if 'duckdb' in conn_type:
            result = self.conn.execute(f"DESCRIBE {self.table_name}").fetchall()
            columns = {row[0].lower() for row in result}
        else:
            cursor = self.conn.cursor()
            cursor.execute(f"PRAGMA table_info({self.table_name})")
            columns = {row[1].lower() for row in cursor.fetchall()}
        
        # Check for contract_id (case-insensitive)
        return 'contract_id' in columns or 'contractid' in columns
    
    def _execute_query(self, query: str) -> pd.DataFrame:
        """
        Execute a SQL query and return results as DataFrame.
        
        Args:
            query: SQL query string
            
        Returns:
            DataFrame with query results
        """
        logger.debug(f"[READ-ONLY] {query}")
        
        conn_type = type(self.conn).__module__
        
        if 'duckdb' in conn_type:
            return self.conn.execute(query).df()
        else:
            return pd.read_sql_query(query, self.conn)
    
    def _load_raw_ohlcv(
        self,
        symbols: Optional[Tuple[str, ...]] = None,
        fields: Tuple[str, ...] = ("open", "high", "low", "close", "volume"),
        start: Optional[Union[str, datetime]] = None,
        end: Optional[Union[str, datetime]] = None,
        include_contract_id: bool = False
    ) -> pd.DataFrame:
        """
        Load raw OHLCV data from database with validation.
        
        Args:
            symbols: Symbols to query
            fields: OHLCV fields to load
            start: Start date filter
            end: End date filter
            include_contract_id: If True, include contract_id in query (if available)
        
        Returns:
            Tidy DataFrame with columns: date, symbol, <fields>, contract_id (if available)
        """
        symbols = symbols or self.universe
        
        # Get mapped column names
        date_col = self._column_map['date']
        symbol_col = self._column_map['symbol']
        
        # Build query fields
        fields_list = list(fields)
        if include_contract_id and self._has_contract_id:
            fields_list.append("contract_id")
        fields_str = ", ".join(fields_list)
        
        symbols_str = ", ".join([f"'{s}'" for s in symbols])
        
        query = f"""
            SELECT {date_col} as date, {symbol_col} as symbol, {fields_str}
            FROM {self.table_name}
            WHERE {symbol_col} IN ({symbols_str})
        """
        
        # Add date filters
        if start:
            start_dt = pd.to_datetime(start)
            query += f" AND {date_col} >= '{start_dt.strftime('%Y-%m-%d')}'"
        
        if end:
            end_dt = pd.to_datetime(end)
            query += f" AND {date_col} <= '{end_dt.strftime('%Y-%m-%d')}'"
        
        # Apply asof filter if snapshot is active
        if self.asof:
            query += f" AND {date_col} <= '{self.asof.strftime('%Y-%m-%d')}'"
        
        query += " ORDER BY symbol, date"
        
        # Execute query
        df = self._execute_query(query)
        
        if df.empty:
            logger.warning(f"[READ-ONLY] No data returned for symbols: {symbols}")
            return df
        
        # Validation and cleaning
        df = self._validate_and_clean(df, fields)
        
        # If contract_id not in database, use symbol as contract_id
        if include_contract_id and 'contract_id' not in df.columns:
            df['contract_id'] = df['symbol']
            logger.debug("[READ-ONLY] contract_id column not found, using symbol as contract_id")
        
        return df
    
    def _validate_and_clean(self, df: pd.DataFrame, fields: Tuple[str, ...]) -> pd.DataFrame:
        """
        Validate and clean OHLCV data.
        
        - Parse dates
        - Drop duplicates
        - Coerce numeric columns
        - Check for monotonic dates per symbol
        """
        # Parse date column
        df['date'] = pd.to_datetime(df['date'])
        
        # Drop duplicates
        n_before = len(df)
        df = df.drop_duplicates(subset=['date', 'symbol'], keep='first')
        n_after = len(df)
        if n_before != n_after:
            logger.warning(f"[READ-ONLY] Dropped {n_before - n_after} duplicate rows")
        
        # Coerce numeric columns
        for field in fields:
            if field in df.columns:
                df[field] = pd.to_numeric(df[field], errors='coerce')
                n_na = df[field].isna().sum()
                if n_na > 0:
                    logger.warning(f"[READ-ONLY] Field '{field}' has {n_na} non-numeric values converted to NaN")
        
        # Check monotonic dates per symbol
        for symbol in df['symbol'].unique():
            symbol_df = df[df['symbol'] == symbol]
            if not symbol_df['date'].is_monotonic_increasing:
                raise ValueError(f"Dates not monotonic for symbol '{symbol}'")
        
        return df
    
    def _build_continuous_prices(
        self,
        symbols: Optional[Tuple[str, ...]] = None,
        start: Optional[Union[str, datetime]] = None,
        end: Optional[Union[str, datetime]] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Build continuous (back-adjusted) prices from raw prices.
        
        Args:
            symbols: Symbols to process (default: all in universe)
            start: Start date filter
            end: End date filter
            
        Returns:
            Tuple of (prices_raw, prices_cont, contract_ids) DataFrames
            All are wide format [date x symbols]
        """
        symbols = symbols or self.universe
        
        # Load raw prices with contract_id
        df_raw = self._load_raw_ohlcv(
            symbols=symbols,
            fields=("close",),
            start=start,
            end=end,
            include_contract_id=True
        )
        
        if df_raw.empty:
            empty_df = pd.DataFrame()
            return empty_df, empty_df, empty_df
        
        # Ensure contract_id exists (use symbol as fallback)
        if 'contract_id' not in df_raw.columns:
            df_raw['contract_id'] = df_raw['symbol']
            logger.debug("[READ-ONLY] Using symbol as contract_id")
        
        # Build continuous prices per symbol
        prices_raw_dict = {}
        prices_cont_dict = {}
        contract_ids_dict = {}
        
        for symbol in df_raw['symbol'].unique():
            symbol_df = df_raw[df_raw['symbol'] == symbol].copy()
            symbol_df = symbol_df.set_index('date').sort_index()
            
            # Extract required columns
            if 'close' not in symbol_df.columns or 'contract_id' not in symbol_df.columns:
                logger.warning(f"[READ-ONLY] Missing columns for {symbol}, skipping")
                continue
            
            # Build back-adjusted series
            try:
                cont_series = self._builder.build_back_adjusted(
                    symbol_df[['close', 'contract_id']]
                )
                
                prices_raw_dict[symbol] = symbol_df['close']
                prices_cont_dict[symbol] = cont_series
                contract_ids_dict[symbol] = symbol_df['contract_id']
            except Exception as e:
                logger.error(f"[READ-ONLY] Error building continuous prices for {symbol}: {e}")
                # Fallback to raw prices
                prices_raw_dict[symbol] = symbol_df['close']
                prices_cont_dict[symbol] = symbol_df['close']
                contract_ids_dict[symbol] = symbol_df['contract_id']
        
        # Combine into DataFrames
        prices_raw = pd.DataFrame(prices_raw_dict).sort_index()
        prices_cont = pd.DataFrame(prices_cont_dict).sort_index()
        contract_ids = pd.DataFrame(contract_ids_dict).sort_index()
        
        return prices_raw, prices_cont, contract_ids
    
    @property
    def prices_raw(self) -> pd.DataFrame:
        """
        Raw close prices by [date x symbol] from DB.
        
        Returns:
            Wide DataFrame of raw close prices [date x symbols]
        """
        if self._prices_raw_cache is None:
            logger.info("[READ-ONLY] Building continuous prices (first access)...")
            self._prices_raw_cache, self._prices_cont_cache, self._contract_ids_cache = \
                self._build_continuous_prices()
        return self._prices_raw_cache
    
    @property
    def prices_cont(self) -> pd.DataFrame:
        """
        Back-adjusted close prices by [date x symbol].
        
        Returns:
            Wide DataFrame of back-adjusted close prices [date x symbols]
        """
        if self._prices_cont_cache is None:
            logger.info("[READ-ONLY] Building continuous prices (first access)...")
            self._prices_raw_cache, self._prices_cont_cache, self._contract_ids_cache = \
                self._build_continuous_prices()
        return self._prices_cont_cache
    
    @property
    def contract_ids(self) -> pd.DataFrame:
        """
        Contract id per [date x symbol].
        
        Returns:
            Wide DataFrame of contract IDs [date x symbols]
        """
        if self._contract_ids_cache is None:
            logger.info("[READ-ONLY] Building continuous prices (first access)...")
            self._prices_raw_cache, self._prices_cont_cache, self._contract_ids_cache = \
                self._build_continuous_prices()
        return self._contract_ids_cache
    
    @property
    def returns_cont(self) -> pd.DataFrame:
        """
        Continuous returns (log returns) from prices_cont.
        
        Returns:
            Wide DataFrame of log returns [date x symbols]
        """
        if self._returns_cont_cache is None:
            prices_cont = self.prices_cont
            if prices_cont.empty:
                self._returns_cont_cache = pd.DataFrame()
            else:
                # Calculate log returns
                self._returns_cont_cache = np.log(prices_cont).diff()
                self._returns_cont_cache = self._returns_cont_cache.dropna(how='all')
        return self._returns_cont_cache
    
    def get_price_panel(
        self,
        symbols: Optional[Tuple[str, ...]] = None,
        fields: Tuple[str, ...] = ("close",),
        start: Optional[Union[str, datetime]] = None,
        end: Optional[Union[str, datetime]] = None,
        tidy: bool = False
    ) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        Get OHLCV price panel.
        
        Args:
            symbols: Symbols to query (default: all in universe)
            fields: OHLCV fields to return
            start: Start date filter
            end: End date filter
            tidy: If True, return long format; else wide format
            
        Returns:
            - If tidy=True: long DataFrame with columns [date, symbol, <fields>]
            - If tidy=False and one field: wide DataFrame [date x symbols]
            - If tidy=False and multiple fields: dict {field: wide_df}
        """
        df = self._load_raw_ohlcv(symbols, fields, start, end)
        
        if df.empty:
            if tidy:
                return df
            elif len(fields) == 1:
                return pd.DataFrame()
            else:
                return {field: pd.DataFrame() for field in fields}
        
        if tidy:
            return df
        
        # Wide format
        if len(fields) == 1:
            field = fields[0]
            wide = df.pivot(index='date', columns='symbol', values=field)
            wide = wide.sort_index()
            return wide
        else:
            result = {}
            for field in fields:
                wide = df.pivot(index='date', columns='symbol', values=field)
                wide = wide.sort_index()
                result[field] = wide
            return result
    
    def get_returns(
        self,
        symbols: Optional[Tuple[str, ...]] = None,
        start: Optional[Union[str, datetime]] = None,
        end: Optional[Union[str, datetime]] = None,
        method: str = "log",
        price: str = "close"
    ) -> pd.DataFrame:
        """
        Calculate returns from prices.
        
        Args:
            symbols: Symbols to query
            start: Start date
            end: End date
            method: "log" or "simple"
            price: Price field to use (default: "close")
            
        Returns:
            Wide DataFrame of returns [date x symbols]
        """
        # Get prices
        prices = self.get_price_panel(symbols, fields=(price,), start=start, end=end, tidy=False)
        
        if prices.empty:
            return prices
        
        # Calculate returns
        if method == "log":
            returns = np.log(prices).diff()
        elif method == "simple":
            returns = prices.pct_change()
        else:
            raise ValueError(f"Unknown return method: {method}. Use 'log' or 'simple'")
        
        # Drop all-NaN rows
        returns = returns.dropna(how='all')
        
        return returns
    
    def get_vol(
        self,
        symbols: Optional[Tuple[str, ...]] = None,
        lookback: int = 63,
        start: Optional[Union[str, datetime]] = None,
        end: Optional[Union[str, datetime]] = None,
        returns: str = "log"
    ) -> pd.DataFrame:
        """
        Calculate rolling volatility (annualized).
        
        Args:
            symbols: Symbols to query
            lookback: Rolling window in days
            start: Start date
            end: End date
            returns: Return method ("log" or "simple")
            
        Returns:
            Wide DataFrame of annualized volatility [date x symbols]
        """
        # Get returns
        ret = self.get_returns(symbols, start=start, end=end, method=returns)
        
        if ret.empty:
            return ret
        
        # Calculate rolling std and annualize
        vol = ret.rolling(window=lookback, min_periods=lookback).std() * np.sqrt(252)
        
        return vol
    
    def get_cov(
        self,
        symbols: Optional[Tuple[str, ...]] = None,
        lookback: int = 252,
        end: Optional[Union[str, datetime]] = None,
        shrink: str = "lw"
    ) -> pd.DataFrame:
        """
        Calculate covariance matrix for the last available window.
        
        Args:
            symbols: Symbols to query
            lookback: Window size in days
            end: End date (default: last available)
            shrink: Shrinkage method ("none" or "lw" for Ledoit-Wolf)
            
        Returns:
            Covariance matrix as DataFrame [symbols x symbols]
        """
        symbols = symbols or self.universe
        
        # Get returns
        ret = self.get_returns(symbols, end=end, method="log")
        
        if ret.empty:
            return pd.DataFrame()
        
        # Get last lookback days
        ret_window = ret.iloc[-lookback:]
        
        if len(ret_window) < lookback:
            logger.warning(
                f"[READ-ONLY] Insufficient data for covariance: "
                f"got {len(ret_window)} days, need {lookback}"
            )
        
        # Drop any symbols with all NaN in window
        ret_window = ret_window.dropna(axis=1, how='all')
        
        if shrink == "none":
            # Sample covariance
            cov = ret_window.cov() * 252  # Annualize
        elif shrink == "lw":
            # Ledoit-Wolf shrinkage
            try:
                from sklearn.covariance import LedoitWolf
                lw = LedoitWolf()
                cov_matrix = lw.fit(ret_window.dropna()).covariance_ * 252
                cov = pd.DataFrame(cov_matrix, index=ret_window.columns, columns=ret_window.columns)
            except ImportError:
                logger.warning("[READ-ONLY] sklearn not available, using sample covariance")
                cov = ret_window.cov() * 252
        else:
            raise ValueError(f"Unknown shrinkage method: {shrink}")
        
        return cov
    
    def snapshot(self, asof: Union[str, datetime]) -> "MarketData":
        """
        Create a snapshot instance that filters all queries to date <= asof.
        
        Args:
            asof: Snapshot date
            
        Returns:
            New MarketData instance with asof filter applied
        """
        return MarketData(
            db_path=self.db_path,
            universe=self.universe,
            asof=asof,
            config_path="configs/data.yaml"  # Reuse config
        )
    
    def get_meta(
        self,
        symbols: Optional[Tuple[str, ...]] = None
    ) -> pd.DataFrame:
        """
        Get metadata for symbols (if available in database).
        
        Args:
            symbols: Symbols to query
            
        Returns:
            DataFrame with available metadata columns
        """
        symbols = symbols or self.universe
        symbols_str = ", ".join([f"'{s}'" for s in symbols])
        
        # Get mapped column names
        symbol_col = self._column_map['symbol']
        
        # Check for optional metadata columns
        optional_cols = ['roll_type', 'source', 'currency', 'multiplier', 'point_value']
        
        query = f"SELECT * FROM {self.table_name} WHERE {symbol_col} IN ({symbols_str}) LIMIT 1"
        sample = self._execute_query(query)
        
        if sample.empty:
            return pd.DataFrame()
        
        # Rename symbol column if needed
        if symbol_col != 'symbol':
            sample = sample.rename(columns={symbol_col: 'symbol'})
        
        # Find which optional columns exist
        available_meta = [col for col in optional_cols if col in sample.columns]
        
        if not available_meta:
            logger.info("[READ-ONLY] No metadata columns found in table")
            return pd.DataFrame(index=symbols)
        
        # Query metadata (distinct per symbol)
        meta_cols = ", ".join([f"{symbol_col} as symbol"] + available_meta)
        query = f"""
            SELECT DISTINCT {meta_cols}
            FROM {self.table_name}
            WHERE {symbol_col} IN ({symbols_str})
        """
        
        meta = self._execute_query(query)
        return meta.set_index('symbol')
    
    def trading_days(
        self,
        symbols: Optional[Tuple[str, ...]] = None
    ) -> pd.DatetimeIndex:
        """
        Get union of all trading days across symbols.
        
        Args:
            symbols: Symbols to query
            
        Returns:
            Sorted DatetimeIndex of unique dates
        """
        symbols = symbols or self.universe
        symbols_str = ", ".join([f"'{s}'" for s in symbols])
        
        # Get mapped column names
        date_col = self._column_map['date']
        symbol_col = self._column_map['symbol']
        
        query = f"""
            SELECT DISTINCT {date_col} as date
            FROM {self.table_name}
            WHERE {symbol_col} IN ({symbols_str})
        """
        
        if self.asof:
            query += f" AND {date_col} <= '{self.asof.strftime('%Y-%m-%d')}'"
        
        query += " ORDER BY date"
        
        dates_df = self._execute_query(query)
        
        if dates_df.empty:
            return pd.DatetimeIndex([])
        
        return pd.to_datetime(dates_df['date'])
    
    def missing_report(
        self,
        symbols: Optional[Tuple[str, ...]] = None
    ) -> pd.DataFrame:
        """
        Report missing dates per symbol vs. union trading calendar.
        
        Args:
            symbols: Symbols to analyze
            
        Returns:
            DataFrame with columns [symbol, total_days, missing_days, coverage_pct]
        """
        symbols = symbols or self.universe
        
        # Get union calendar
        all_days = self.trading_days(symbols)
        total_days = len(all_days)
        
        if total_days == 0:
            return pd.DataFrame(columns=['symbol', 'total_days', 'missing_days', 'coverage_pct'])
        
        # Count days per symbol
        report = []
        for symbol in symbols:
            prices = self.get_price_panel((symbol,), fields=('close',), tidy=False)
            
            if prices.empty:
                actual_days = 0
            else:
                actual_days = prices[symbol].notna().sum()
            
            missing_days = total_days - actual_days
            coverage_pct = (actual_days / total_days) * 100
            
            report.append({
                'symbol': symbol,
                'total_days': total_days,
                'missing_days': missing_days,
                'coverage_pct': coverage_pct
            })
        
        return pd.DataFrame(report)
    
    def flag_roll_jumps(
        self,
        symbols: Optional[Tuple[str, ...]] = None,
        threshold_bp: int = 100
    ) -> pd.DataFrame:
        """
        Flag potential roll jumps based on large returns.
        
        This is diagnostic only - prices are NOT adjusted.
        
        Args:
            symbols: Symbols to analyze
            threshold_bp: Threshold in basis points (default: 100 = 1%)
            
        Returns:
            DataFrame with columns [date, symbol, return, flagged]
            Only rows with flagged=True are returned
        """
        # Get simple returns
        returns = self.get_returns(symbols, method="simple")
        
        if returns.empty:
            return pd.DataFrame(columns=['date', 'symbol', 'return', 'flagged'])
        
        # Convert threshold to decimal
        threshold = threshold_bp / 10000
        
        # Find large moves
        flagged_list = []
        
        for symbol in returns.columns:
            symbol_ret = returns[symbol].dropna()
            large_moves = symbol_ret[symbol_ret.abs() > threshold]
            
            for date, ret_val in large_moves.items():
                flagged_list.append({
                    'date': date,
                    'symbol': symbol,
                    'return': ret_val,
                    'flagged': True
                })
        
        result = pd.DataFrame(flagged_list)
        
        if not result.empty:
            result = result.sort_values(['symbol', 'date'])
            logger.info(f"[READ-ONLY] Flagged {len(result)} potential roll jumps across {symbols}")
        
        return result
    
    def get_fred_indicator(
        self,
        series_id: str,
        start: Optional[Union[str, datetime]] = None,
        end: Optional[Union[str, datetime]] = None
    ) -> pd.Series:
        """
        Get FRED economic indicator time series.
        
        Args:
            series_id: FRED series ID (e.g., 'VIXCLS', 'DGS10', 'UNRATE')
            start: Start date filter
            end: End date filter
            
        Returns:
            Series with date index and value column
        """
        # Build query
        query = f"""
            SELECT date, value
            FROM f_fred_observations
            WHERE series_id = '{series_id}'
        """
        
        # Add date filters
        if start:
            start_dt = pd.to_datetime(start)
            query += f" AND date >= '{start_dt.strftime('%Y-%m-%d')}'"
        
        if end:
            end_dt = pd.to_datetime(end)
            query += f" AND date <= '{end_dt.strftime('%Y-%m-%d')}'"
        
        # Apply asof filter if snapshot is active
        if self.asof:
            query += f" AND date <= '{self.asof.strftime('%Y-%m-%d')}'"
        
        query += " ORDER BY date"
        
        # Execute query
        df = self._execute_query(query)
        
        if df.empty:
            logger.warning(f"[READ-ONLY] No FRED data found for series_id: {series_id}")
            return pd.Series(dtype=float)
        
        # Parse dates and set as index
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        
        # Return as Series
        series = df['value'].copy()
        series.name = series_id
        
        return series
    
    def get_fred_indicators(
        self,
        series_ids: Tuple[str, ...],
        start: Optional[Union[str, datetime]] = None,
        end: Optional[Union[str, datetime]] = None
    ) -> pd.DataFrame:
        """
        Get multiple FRED economic indicators as a DataFrame.
        
        Args:
            series_ids: Tuple of FRED series IDs
            start: Start date filter
            end: End date filter
            
        Returns:
            DataFrame with date index and one column per series_id
        """
        # Get each indicator
        indicators = {}
        for series_id in series_ids:
            series = self.get_fred_indicator(series_id, start=start, end=end)
            if not series.empty:
                indicators[series_id] = series
        
        if not indicators:
            return pd.DataFrame()
        
        # Combine into DataFrame
        df = pd.DataFrame(indicators)
        df = df.sort_index()
        
        return df
    
    def get_contracts_by_root(
        self,
        root: str,
        ranks: Optional[List[int]] = None,
        fields: Tuple[str, ...] = ("close",),
        start: Optional[Union[str, datetime]] = None,
        end: Optional[Union[str, datetime]] = None
    ) -> pd.DataFrame:
        """
        Get OHLCV data for contracts by root symbol and rank.
        
        Queries all contracts matching the root prefix (e.g., "SR3") and returns
        data organized by rank. Assumes contracts are named with root prefix
        and can be sorted by expiration or have rank in the name.
        
        Args:
            root: Root symbol (e.g., "SR3")
            ranks: Optional list of ranks to include (0-11 for 12 contracts)
            fields: OHLCV fields to return
            start: Start date filter
            end: End date filter
            
        Returns:
            DataFrame with MultiIndex columns: (rank, field) or (symbol, field)
            Index: date
        """
        # Get mapped column names
        date_col = self._column_map['date']
        symbol_col = self._column_map['symbol']
        
        # Query all contracts matching root prefix
        fields_str = ", ".join(fields)
        
        query = f"""
            SELECT {date_col} as date, {symbol_col} as symbol, {fields_str}
            FROM {self.table_name}
            WHERE {symbol_col} LIKE '{root}%'
        """
        
        # Add date filters
        if start:
            start_dt = pd.to_datetime(start)
            query += f" AND {date_col} >= '{start_dt.strftime('%Y-%m-%d')}'"
        
        if end:
            end_dt = pd.to_datetime(end)
            query += f" AND {date_col} <= '{end_dt.strftime('%Y-%m-%d')}'"
        
        # Apply asof filter if snapshot is active
        if self.asof:
            query += f" AND {date_col} <= '{self.asof.strftime('%Y-%m-%d')}'"
        
        query += f" ORDER BY {symbol_col}, {date_col}"
        
        # Execute query
        df = self._execute_query(query)
        
        if df.empty:
            logger.warning(f"[READ-ONLY] No contracts found for root: {root}")
            return pd.DataFrame()
        
        # Validate and clean
        df = self._validate_and_clean(df, fields)
        
        # Get unique symbols and sort them (assumes they're in expiration order or have rank info)
        unique_symbols = sorted(df['symbol'].unique())
        
        # If ranks specified, map symbols to ranks
        # Otherwise, assume symbols are already in rank order (0, 1, 2, ...)
        if ranks is not None:
            if len(unique_symbols) < len(ranks):
                logger.warning(
                    f"[READ-ONLY] Only {len(unique_symbols)} contracts found for root {root}, "
                    f"but {len(ranks)} ranks requested"
                )
            # Map first N symbols to requested ranks
            symbol_to_rank = {sym: rank for sym, rank in zip(unique_symbols[:len(ranks)], ranks)}
            df = df[df['symbol'].isin(symbol_to_rank.keys())].copy()
            df['rank'] = df['symbol'].map(symbol_to_rank)
        else:
            # Assign ranks 0, 1, 2, ... based on symbol order
            symbol_to_rank = {sym: i for i, sym in enumerate(unique_symbols)}
            df['rank'] = df['symbol'].map(symbol_to_rank)
        
        # Pivot to wide format with rank as columns
        if len(fields) == 1:
            field = fields[0]
            wide = df.pivot(
                index='date',
                columns='rank',
                values=field
            )
            wide = wide.sort_index().sort_index(axis=1)
            return wide
        else:
            # Multiple fields: stack and pivot
            df_melted = df.melt(
                id_vars=['date', 'rank'],
                value_vars=[f for f in fields if f in df.columns],
                var_name='field',
                value_name='value'
            )
            wide = df_melted.pivot_table(
                index='date',
                columns=['rank', 'field'],
                values='value',
                aggfunc='first'
            )
            wide = wide.sort_index().sort_index(axis=1)
            return wide
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            logger.info("[READ-ONLY] Connection closed")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

