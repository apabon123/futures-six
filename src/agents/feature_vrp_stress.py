"""
VRP Stress Proxy Feature

Computes a binary indicator for VRP engine gating based on extreme stress conditions.

Rule: OFF if (VVIX percentile >= 99) OR (gamma_stress_proxy == 1 AND vx_backwardation == 1)

This is a composite feature that combines:
- VVIX extreme stress (99th percentile)
- Gamma stress + VX backwardation (structurally unsafe for short vol)
"""

import logging
from typing import Optional, Union
from datetime import datetime
import pandas as pd
import numpy as np

from src.market_data.vrp_loaders import load_vvix
from src.agents.utils_db import open_readonly_connection

logger = logging.getLogger(__name__)


def compute_vrp_stress_proxy(
    market,
    start_date: Optional[Union[str, datetime]] = None,
    end_date: Optional[Union[str, datetime]] = None,
    vvix_percentile_threshold: float = 0.99,
    window: int = 252
) -> pd.Series:
    """
    Compute VRP stress proxy as a binary indicator (0 or 1).
    
    Rule: OFF (1 = stress) if:
    - VVIX percentile >= 99th percentile, OR
    - (gamma_stress_proxy == 1 AND vx_backwardation == 1)
    
    Args:
        market: MarketData instance (for DB connection and features)
        start_date: Start date for feature computation
        end_date: End date for feature computation
        vvix_percentile_threshold: VVIX percentile threshold (default: 0.99 = 99th percentile)
        window: Rolling window for percentile calculation (default: 252 days)
    
    Returns:
        Series indexed by date with binary values (0 or 1)
        - 1 = extreme stress (gate VRP OFF)
        - 0 = normal (allow VRP)
    """
    if start_date is None:
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
        db_path = getattr(market, 'db_path', None)
        if db_path is None:
            logger.warning("[VRPStressProxy] No DB connection available from market")
            return pd.Series(dtype=float, name='vrp_stress_proxy')
        con = open_readonly_connection(db_path)
        close_conn = True
    
    try:
        # Component 1: VVIX extreme stress (99th percentile)
        vvix_stress = pd.Series(dtype=float, index=pd.DatetimeIndex([]))
        try:
            from src.market_data.vrp_loaders import load_vvix
            vvix_df = load_vvix(con, start_str, end_str)
            
            if not vvix_df.empty and 'vvix' in vvix_df.columns:
                vvix_series = vvix_df.set_index('date')['vvix']
                vvix_series.index = pd.to_datetime(vvix_series.index)
                
                # Compute rolling percentile
                rolling_percentile = vvix_series.rolling(
                    window=window,
                    min_periods=max(window // 2, 63)
                ).quantile(vvix_percentile_threshold)
                
                # Binary indicator: 1 when VVIX >= 99th percentile
                vvix_stress = (vvix_series >= rolling_percentile).astype(int)
        except Exception as e:
            logger.warning(f"[VRPStressProxy] Failed to compute VVIX stress: {e}")
        
        # Component 2: Get gamma_stress_proxy and vx_backwardation from market.features
        gamma_stress = pd.Series(dtype=float, index=pd.DatetimeIndex([]))
        vx_backward = pd.Series(dtype=float, index=pd.DatetimeIndex([]))
        
        if hasattr(market, 'features') and isinstance(market.features, dict):
            if 'gamma_stress_proxy' in market.features:
                gamma_stress = market.features['gamma_stress_proxy']
            if 'vx_backwardation' in market.features:
                vx_backward = market.features['vx_backwardation']
        
        # Combine: vrp_stress = vvix_stress OR (gamma_stress AND vx_backwardation)
        # Align all series to common index
        all_dates = set()
        if not vvix_stress.empty:
            all_dates.update(vvix_stress.index)
        if not gamma_stress.empty:
            all_dates.update(gamma_stress.index)
        if not vx_backward.empty:
            all_dates.update(vx_backward.index)
        
        if not all_dates:
            logger.warning("[VRPStressProxy] No data available")
            return pd.Series(dtype=float, name='vrp_stress_proxy')
        
        all_dates = pd.DatetimeIndex(sorted(all_dates))
        
        # Reindex all series to common index, fill missing with 0
        vvix_stress_aligned = vvix_stress.reindex(all_dates, fill_value=0) if not vvix_stress.empty else pd.Series(0, index=all_dates)
        gamma_stress_aligned = gamma_stress.reindex(all_dates, fill_value=0) if not gamma_stress.empty else pd.Series(0, index=all_dates)
        vx_backward_aligned = vx_backward.reindex(all_dates, fill_value=0) if not vx_backward.empty else pd.Series(0, index=all_dates)
        
        # Composite rule: vvix_stress OR (gamma_stress AND vx_backwardation)
        backwardation_stress = (gamma_stress_aligned == 1) & (vx_backward_aligned == 1)
        vrp_stress = (vvix_stress_aligned == 1) | backwardation_stress
        vrp_stress = vrp_stress.astype(int)
        
        logger.info(
            f"[VRPStressProxy] Computed: {len(vrp_stress)} days, "
            f"{vrp_stress.sum()} stress days ({vrp_stress.mean() * 100:.1f}%)"
        )
        
        return vrp_stress
    
    except Exception as e:
        logger.error(f"[VRPStressProxy] Error computing feature: {e}")
        return pd.Series(dtype=float, name='vrp_stress_proxy')
    
    finally:
        if close_conn:
            con.close()

