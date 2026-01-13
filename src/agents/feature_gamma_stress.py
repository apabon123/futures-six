"""
Gamma Stress Proxy Feature

Computes a simple binary proxy for gamma/vol stress regimes.

Implementation:
- If VVIX is available: binary indicator when VVIX > 95th percentile (threshold=1)
- Else: binary indicator when VIX change variance (21d rolling std) > 95th percentile

This is a minimal v1 implementation - can be refined later with more sophisticated
gamma imbalance metrics.
"""

import logging
from typing import Optional, Union
from datetime import datetime
import pandas as pd
import numpy as np
import duckdb

from src.market_data.vrp_loaders import load_vvix, load_vix
from src.agents.utils_db import open_readonly_connection

logger = logging.getLogger(__name__)


def compute_gamma_stress_proxy(
    market,
    start_date: Optional[Union[str, datetime]] = None,
    end_date: Optional[Union[str, datetime]] = None,
    percentile_threshold: float = 0.95,
    window: int = 252
) -> pd.Series:
    """
    Compute gamma stress proxy as a binary indicator (0 or 1).
    
    Strategy:
    - Primary: VVIX percentile (if available)
    - Fallback: VIX change variance (21d rolling std of VIX changes)
    
    Binary output:
    - 1 = stress regime (VVIX or VIX change variance above threshold percentile)
    - 0 = normal regime
    
    Args:
        market: MarketData instance (for DB connection)
        start_date: Start date for feature computation
        end_date: End date for feature computation
        percentile_threshold: Percentile threshold for stress (default: 0.95 = 95th percentile)
        window: Rolling window for percentile calculation (default: 252 days)
    
    Returns:
        Series indexed by date with binary values (0 or 1)
    """
    if start_date is None:
        # Use market data range if available
        start_date = "2020-01-01"
    
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")
    
    start_str = pd.to_datetime(start_date).strftime("%Y-%m-%d")
    end_str = pd.to_datetime(end_date).strftime("%Y-%m-%d")
    
    # Get DB connection from market
    close_conn = False
    if hasattr(market, 'conn'):
        con = market.conn
    else:
        # Fallback: open connection (shouldn't happen in normal usage)
        db_path = getattr(market, 'db_path', None)
        if db_path is None:
            logger.warning("[GammaStressProxy] No DB connection available from market")
            return pd.Series(dtype=float, name='gamma_stress_proxy')
        con = open_readonly_connection(db_path)
        close_conn = True
    
    try:
        # Try VVIX first (preferred)
        vvix_df = load_vvix(con, start_str, end_str)
        
        if not vvix_df.empty and 'vvix' in vvix_df.columns:
            vvix_series = vvix_df.set_index('date')['vvix']
            vvix_series.index = pd.to_datetime(vvix_series.index)
            
            # Compute rolling percentile
            rolling_percentile = vvix_series.rolling(
                window=window,
                min_periods=max(window // 2, 63)
            ).quantile(percentile_threshold)
            
            # Binary indicator: 1 when VVIX > threshold percentile
            stress_proxy = (vvix_series > rolling_percentile).astype(int)
            
            logger.info(
                f"[GammaStressProxy] Using VVIX: {len(stress_proxy)} days, "
                f"{stress_proxy.sum()} stress days ({stress_proxy.mean() * 100:.1f}%)"
            )
            
            return stress_proxy
        
        # Fallback: VIX change variance
        logger.info("[GammaStressProxy] VVIX not available, using VIX change variance")
        vix_df = load_vix(con, start_str, end_str)
        
        if vix_df.empty or 'vix' not in vix_df.columns:
            logger.warning("[GammaStressProxy] VIX data not available, returning empty series")
            return pd.Series(dtype=float, name='gamma_stress_proxy')
        
        vix_series = vix_df.set_index('date')['vix']
        vix_series.index = pd.to_datetime(vix_series.index)
        
        # Compute VIX changes (daily changes)
        vix_changes = vix_series.diff()
        
        # Compute rolling std of VIX changes (21-day window)
        vix_change_vol = vix_changes.rolling(
            window=21,
            min_periods=10
        ).std()
        
        # Compute rolling percentile
        rolling_percentile = vix_change_vol.rolling(
            window=window,
            min_periods=max(window // 2, 63)
        ).quantile(percentile_threshold)
        
        # Binary indicator: 1 when VIX change vol > threshold percentile
        stress_proxy = (vix_change_vol > rolling_percentile).astype(int)
        
        logger.info(
            f"[GammaStressProxy] Using VIX change variance: {len(stress_proxy)} days, "
            f"{stress_proxy.sum()} stress days ({stress_proxy.mean() * 100:.1f}%)"
        )
        
        return stress_proxy
    
    except Exception as e:
        logger.error(f"[GammaStressProxy] Error computing feature: {e}")
        return pd.Series(dtype=float, name='gamma_stress_proxy')
    
    finally:
        # Don't close connection if it came from market (market owns it)
        if close_conn:
            con.close()

