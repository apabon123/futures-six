"""
VX Curve Backwardation Feature

Computes a binary indicator for VX curve backwardation (VX1 > VX2).

Binary output:
- 1 = backwardated (VX1 > VX2)
- 0 = contango (VX1 <= VX2) or missing data
"""

import logging
from typing import Optional, Union
from datetime import datetime
import pandas as pd
import numpy as np

from src.market_data.vrp_loaders import load_vx_curve
from src.agents.utils_db import open_readonly_connection

logger = logging.getLogger(__name__)


def compute_vx_backwardation(
    market,
    start_date: Optional[Union[str, datetime]] = None,
    end_date: Optional[Union[str, datetime]] = None
) -> pd.Series:
    """
    Compute VX curve backwardation indicator (binary: 1 = backwardated, 0 = contango).
    
    Args:
        market: MarketData instance (for DB connection)
        start_date: Start date for feature computation
        end_date: End date for feature computation
    
    Returns:
        Series indexed by date with binary values (0 or 1)
        - 1 when VX1 > VX2 (backwardated)
        - 0 when VX1 <= VX2 (contango) or missing data
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
            logger.warning("[VXBackwardation] No DB connection available from market")
            return pd.Series(dtype=float, name='vx_backwardation')
        con = open_readonly_connection(db_path)
        close_conn = True
    
    try:
        # Load VX curve
        vx_df = load_vx_curve(con, start_str, end_str)
        
        if vx_df.empty or 'vx1' not in vx_df.columns or 'vx2' not in vx_df.columns:
            logger.warning("[VXBackwardation] VX curve data not available")
            return pd.Series(dtype=float, name='vx_backwardation')
        
        vx_series = vx_df.set_index('date')
        vx_series.index = pd.to_datetime(vx_series.index)
        
        # Binary indicator: 1 when VX1 > VX2 (backwardated)
        backwardation = (vx_series['vx1'] > vx_series['vx2']).astype(int)
        
        # Fill missing values with 0 (assume contango if data missing)
        backwardation = backwardation.fillna(0)
        
        logger.info(
            f"[VXBackwardation] Computed: {len(backwardation)} days, "
            f"{backwardation.sum()} backwardated days ({backwardation.mean() * 100:.1f}%)"
        )
        
        return backwardation
    
    except Exception as e:
        logger.error(f"[VXBackwardation] Error computing feature: {e}")
        return pd.Series(dtype=float, name='vx_backwardation')
    
    finally:
        if close_conn:
            con.close()

