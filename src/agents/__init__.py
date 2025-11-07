"""
Agents package for futures-six backtesting project.

This package contains data brokers and feature stores for managing
market data in a read-only, safe manner.
"""

from .data_broker import MarketData
from .feature_store import FeatureStore
from .utils_db import open_readonly_connection, find_ohlcv_table
from .risk_vol import RiskVol
from .strat_momentum import TSMOM
from .overlay_volmanaged import VolManagedOverlay
from .overlay_roll_jump import RollJumpFilter

__all__ = [
    'MarketData',
    'FeatureStore',
    'RiskVol',
    'TSMOM',
    'VolManagedOverlay',
    'RollJumpFilter',
    'open_readonly_connection',
    'find_ohlcv_table'
]

__version__ = '0.1.0'

