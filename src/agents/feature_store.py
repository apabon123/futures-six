"""
FeatureStore: Optional caching wrapper for MarketData.

Provides memoization of commonly-used features like returns and volatility
to avoid redundant database queries for small datasets.
"""

import logging
from typing import Optional, Tuple, Union
from datetime import datetime
import pandas as pd
from functools import wraps

logger = logging.getLogger(__name__)


class FeatureStore:
    """
    Caching wrapper for MarketData broker.
    
    Memoizes computed features (returns, volatility) to reduce redundant
    queries for small datasets (~6 symbols × 5 years).
    """
    
    def __init__(self, market_data):
        """
        Initialize feature store.
        
        Args:
            market_data: MarketData instance to wrap
        """
        self.market_data = market_data
        self._cache = {
            'returns': {},
            'vol': {},
            'prices': {}
        }
        logger.debug("[CACHE] FeatureStore initialized")
    
    def _make_key(self, *args, **kwargs) -> str:
        """Generate cache key from arguments."""
        key_parts = [str(arg) for arg in args]
        key_parts.extend([f"{k}={v}" for k, v in sorted(kwargs.items())])
        return "|".join(key_parts)
    
    def get_returns(
        self,
        symbols: Optional[Tuple[str, ...]] = None,
        start: Optional[Union[str, datetime]] = None,
        end: Optional[Union[str, datetime]] = None,
        method: str = "log",
        price: str = "close"
    ) -> pd.DataFrame:
        """
        Get returns with caching.
        
        Args: Same as MarketData.get_returns
        Returns: Wide DataFrame of returns
        """
        cache_key = self._make_key(symbols, start, end, method, price)
        
        if cache_key in self._cache['returns']:
            logger.debug(f"[CACHE] Hit: returns {cache_key[:50]}...")
            return self._cache['returns'][cache_key]
        
        logger.debug(f"[CACHE] Miss: returns {cache_key[:50]}...")
        result = self.market_data.get_returns(symbols, start, end, method, price)
        self._cache['returns'][cache_key] = result
        
        return result
    
    def get_vol(
        self,
        symbols: Optional[Tuple[str, ...]] = None,
        lookback: int = 63,
        start: Optional[Union[str, datetime]] = None,
        end: Optional[Union[str, datetime]] = None,
        returns: str = "log"
    ) -> pd.DataFrame:
        """
        Get volatility with caching.
        
        Args: Same as MarketData.get_vol
        Returns: Wide DataFrame of volatility
        """
        cache_key = self._make_key(symbols, lookback, start, end, returns)
        
        if cache_key in self._cache['vol']:
            logger.debug(f"[CACHE] Hit: vol {cache_key[:50]}...")
            return self._cache['vol'][cache_key]
        
        logger.debug(f"[CACHE] Miss: vol {cache_key[:50]}...")
        result = self.market_data.get_vol(symbols, lookback, start, end, returns)
        self._cache['vol'][cache_key] = result
        
        return result
    
    def get_price_panel(
        self,
        symbols: Optional[Tuple[str, ...]] = None,
        fields: Tuple[str, ...] = ("close",),
        start: Optional[Union[str, datetime]] = None,
        end: Optional[Union[str, datetime]] = None,
        tidy: bool = False
    ):
        """
        Get price panel with caching.
        
        Args: Same as MarketData.get_price_panel
        Returns: DataFrame or dict of DataFrames
        """
        cache_key = self._make_key(symbols, fields, start, end, tidy)
        
        if cache_key in self._cache['prices']:
            logger.debug(f"[CACHE] Hit: prices {cache_key[:50]}...")
            return self._cache['prices'][cache_key]
        
        logger.debug(f"[CACHE] Miss: prices {cache_key[:50]}...")
        result = self.market_data.get_price_panel(symbols, fields, start, end, tidy)
        self._cache['prices'][cache_key] = result
        
        return result
    
    def clear_cache(self):
        """Clear all cached features."""
        self._cache = {
            'returns': {},
            'vol': {},
            'prices': {}
        }
        logger.info("[CACHE] Cleared all cached features")
    
    def cache_stats(self) -> dict:
        """Get cache statistics."""
        stats = {
            'returns_cached': len(self._cache['returns']),
            'vol_cached': len(self._cache['vol']),
            'prices_cached': len(self._cache['prices'])
        }
        return stats
    
    def __getattr__(self, name):
        """
        Delegate non-cached methods to underlying MarketData instance.
        """
        return getattr(self.market_data, name)

