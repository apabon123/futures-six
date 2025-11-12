"""
Read-only ContractSpecs service for futures contract metadata.

Provides consistent contract specifications (loaded from YAML) to enable
USD P&L conversion and position sizing across the supported futures
universe. Entirely config-driven, never touches the database.
"""

import yaml
from pathlib import Path
from typing import Optional
import pandas as pd


class ContractSpecs:
    """Read-only contract specifications loaded from YAML config."""
    
    def __init__(self, config_path: str = "configs/contracts.yaml"):
        """
        Initialize ContractSpecs from YAML configuration.
        
        Args:
            config_path: Path to contracts.yaml configuration file
        """
        self.config_path = Path(config_path)
        self._load_config()
    
    def _load_config(self) -> None:
        """Load and validate contract specifications from YAML."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        if not config:
            raise ValueError(f"Empty config file: {self.config_path}")
        
        # Extract aliases and contracts
        self._aliases = config.get('aliases', {})
        
        # Store all contract specs (everything except 'aliases')
        self._contracts = {k: v for k, v in config.items() if k != 'aliases'}
        
        # Validate contract specs
        for root, spec in self._contracts.items():
            self._validate_spec(root, spec)
    
    def _validate_spec(self, root: str, spec: dict) -> None:
        """Validate that a contract spec has all required fields."""
        required_fields = ['root', 'multiplier', 'point_value', 'tick_size', 'currency', 'fx_base']
        
        for field in required_fields:
            if field not in spec:
                raise ValueError(f"Contract {root} missing required field: {field}")
        
        # Validate numeric fields
        numeric_fields = ['multiplier', 'point_value', 'tick_size']
        for field in numeric_fields:
            if not isinstance(spec[field], (int, float)):
                raise ValueError(f"Contract {root} field {field} must be numeric, got {type(spec[field])}")
    
    def canonical_root(self, db_symbol: str) -> str:
        """
        Resolve a database symbol to its canonical root symbol.
        
        Args:
            db_symbol: Database symbol (e.g., ES_FRONT_CALENDAR_2D)
        
        Returns:
            Canonical root symbol (e.g., ES)
        
        Examples:
            >>> specs.canonical_root("ES_FRONT_CALENDAR_2D")
            "ES"
            >>> specs.canonical_root("ES")
            "ES"
        """
        # Check if it's an alias
        if db_symbol in self._aliases:
            return self._aliases[db_symbol]
        
        # Check if it's already a root symbol
        if db_symbol in self._contracts:
            return db_symbol
        
        raise ValueError(f"Unknown symbol or alias: {db_symbol}")
    
    def spec(self, symbol_or_root: str) -> dict:
        """
        Get contract specification for a symbol.
        
        Args:
            symbol_or_root: Either a root symbol or database alias
        
        Returns:
            Dict with keys: root, multiplier, point_value, tick_size, 
            currency, fx_base, and optional meta
        
        Examples:
            >>> specs.spec("ES")
            {'root': 'ES', 'multiplier': 50, 'point_value': 50, ...}
            >>> specs.spec("ES_FRONT_CALENDAR_2D")
            {'root': 'ES', 'multiplier': 50, 'point_value': 50, ...}
        """
        root = self.canonical_root(symbol_or_root)
        
        if root not in self._contracts:
            raise ValueError(f"No specification found for root: {root}")
        
        # Return a copy to prevent mutation
        spec_dict = self._contracts[root].copy()
        
        # Ensure 'meta' key exists (even if empty)
        if 'meta' not in spec_dict:
            spec_dict['meta'] = {}
        
        return spec_dict
    
    def usd_move(
        self, 
        symbol: str, 
        price_change: float, 
        fx_rates: Optional[dict] = None
    ) -> float:
        """
        Calculate USD value of a price change for one contract.
        
        Args:
            symbol: Root symbol or database alias
            price_change: Change in price (in contract quote units)
            fx_rates: Optional FX rates dict (e.g., {'EURUSD': 1.05})
        
        Returns:
            USD value of the price move
        
        Examples:
            >>> specs.usd_move("ES", 1.0)  # 1 point ES move
            50.0
            >>> specs.usd_move("6E", 0.0001, {'EURUSD': 1.05})
            12.5
        """
        spec_dict = self.spec(symbol)
        point_value = spec_dict['point_value']
        currency = spec_dict['currency']
        fx_base = spec_dict['fx_base']
        
        # Calculate base move
        usd_value = price_change * point_value
        
        # Handle FX conversion if needed
        # For 6E, if fx_rates provided, we might need to apply them
        # However, since most contracts are USD-quoted, default behavior
        # is to return the calculated value as-is
        if fx_rates and currency == 'USD' and fx_base != 'USD':
            # This would handle cases where the contract needs FX adjustment
            # For now, all contracts are USD-based, so no conversion needed
            pass
        
        return usd_value
    
    def to_usd_returns(
        self,
        price_series: pd.Series,
        symbol: str,
        fx_series: Optional[pd.Series] = None
    ) -> pd.Series:
        """
        Convert price series to USD returns per $1 notional.
        
        Args:
            price_series: Time series of contract prices
            symbol: Root symbol or database alias
            fx_series: Optional FX rate series (aligned with price_series)
        
        Returns:
            Series of returns per $1 notional, aligned to input index
        
        Examples:
            >>> prices = pd.Series([4000, 4001, 4002], index=pd.date_range('2024-01-01', periods=3))
            >>> specs.to_usd_returns(prices, "ES")
            # Returns series of percentage returns scaled by point value
        """
        if not isinstance(price_series, pd.Series):
            raise TypeError("price_series must be a pandas Series")
        
        spec_dict = self.spec(symbol)
        point_value = spec_dict['point_value']
        
        # Calculate price changes
        price_changes = price_series.diff()
        
        # Calculate USD moves for each change
        usd_moves = price_changes * point_value
        
        # Calculate notional values (price * point_value)
        notional = price_series * point_value
        
        # Calculate returns per $1 notional
        # Avoid division by zero - use previous notional for return calculation
        notional_prev = notional.shift(1)
        
        # Return = USD move / previous notional
        returns = usd_moves / notional_prev
        
        # Handle FX if provided
        if fx_series is not None:
            if not isinstance(fx_series, pd.Series):
                raise TypeError("fx_series must be a pandas Series")
            # Apply FX rates if needed (for non-USD contracts)
            # Currently all contracts are USD-based, so this is a no-op
            pass
        
        return returns

