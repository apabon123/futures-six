"""
VRP Core Features: Volatility Risk Premium signals.

Features:
- vrp_core_spread: VIX - Realized ES Vol (21d annualized)
- vrp_core_z: Z-scored VRP spread (252-day rolling)

Strategy:
- VRP spread = VIX (implied vol) - realized ES volatility
- Positive VRP = volatility risk premium exists (typical)
- Signal = z-scored VRP spread, clipped to ±3σ
- Trade VX1 (front month futures) directionally based on VRP

Data Sources:
- VIX: FRED VIXCLS via canonical DB (f_fred_observations)
- ES returns: MarketData continuous returns for realized vol calculation
"""

import logging
from typing import Optional, Union
from datetime import datetime
import pandas as pd
import numpy as np
import duckdb

from src.market_data.vrp_loaders import load_vrp_inputs
from src.agents.utils_db import open_readonly_connection

logger = logging.getLogger(__name__)


def _zscore_rolling(s: pd.Series, window: int = 252, clip: float = 3.0, min_periods: Optional[int] = None) -> pd.Series:
    """
    Compute rolling z-score with clipping.
    
    Args:
        s: Input series
        window: Rolling window size in days
        clip: Z-score clipping bounds
        min_periods: Minimum periods for rolling calculation (default: window // 2)
        
    Returns:
        Z-scored and clipped series
    """
    if min_periods is None:
        min_periods = max(window // 2, 63)  # At least 63 days (1 quarter)
    
    mu = s.rolling(window=window, min_periods=min_periods).mean()
    sigma = s.rolling(window=window, min_periods=min_periods).std()
    z = (s - mu) / sigma
    return z.clip(-clip, clip)


class VRPCoreFeatures:
    """
    Computes VRP core features.
    
    Features:
    - vrp_core_spread: VIX - Realized ES Vol (21d annualized)
    - vrp_core_z: Z-scored VRP spread
    """
    
    def __init__(
        self,
        rv_lookback: int = 21,
        zscore_window: int = 252,
        clip: float = 3.0,
        db_path: Optional[str] = None
    ):
        """
        Initialize VRP core features calculator.
        
        Args:
            rv_lookback: Lookback period for realized vol calculation (default: 21 days)
            zscore_window: Rolling window for z-score standardization (default: 252 days)
            clip: Z-score clipping bounds (default: 3.0)
            db_path: Path to canonical DuckDB (default: from configs/data.yaml)
        """
        self.rv_lookback = rv_lookback
        self.zscore_window = zscore_window
        self.clip = clip
        self.db_path = db_path
    
    def compute(
        self,
        market,
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None
    ) -> pd.DataFrame:
        """
        Compute VRP core features.
        
        Args:
            market: MarketData instance
            start_date: Start date for feature computation (default: earliest available)
            end_date: End date for feature computation (default: last available)
            
        Returns:
            DataFrame indexed by date with columns:
            - vix: VIX level
            - rv_es_21: 21-day realized ES volatility (annualized)
            - vrp_spread: VIX - rv_es_21
            - vrp_z: Z-scored VRP spread
            - vix3m: VIX3M level (passthrough)
            - vx1: VX1 front month price (passthrough)
            - vx2: VX2 second month price (passthrough)
            - vx3: VX3 third month price (passthrough)
        """
        # Get ES returns for realized vol calculation
        # Use ES_FRONT_CALENDAR_2D (ES continuous contract)
        es_symbol = "ES_FRONT_CALENDAR_2D"
        
        if es_symbol not in market.returns_cont.columns:
            logger.error(f"[VRPCore] {es_symbol} not found in market data")
            return pd.DataFrame()
        
        # Get continuous returns (log returns)
        returns_cont = market.returns_cont[[es_symbol]].copy()
        
        if returns_cont.empty:
            logger.warning("[VRPCore] No ES returns data available")
            return pd.DataFrame()
        
        # Filter by date range
        if start_date is not None:
            start_dt = pd.to_datetime(start_date)
            returns_cont = returns_cont[returns_cont.index >= start_dt]
        
        if end_date is not None:
            end_dt = pd.to_datetime(end_date)
            returns_cont = returns_cont[returns_cont.index <= end_dt]
        
        if returns_cont.empty:
            logger.warning("[VRPCore] No ES returns data after filtering")
            return pd.DataFrame()
        
        # Compute 21-day realized volatility (annualized)
        # RV = rolling std of daily returns * sqrt(252)
        # Note: This gives decimal form (e.g., 0.18 for 18%)
        rv_es = returns_cont[es_symbol].rolling(
            window=self.rv_lookback,
            min_periods=self.rv_lookback
        ).std() * np.sqrt(252)
        
        # Convert to DataFrame
        rv_df = pd.DataFrame({
            'date': rv_es.index,
            'rv_es_21': rv_es.values  # In decimal form (0.18 = 18%)
        })
        
        # Load VRP inputs (VIX, VIX3M, VX1/2/3) from canonical DB
        # Determine DB path
        if self.db_path:
            db_path = self.db_path
        else:
            # Load from config
            import yaml
            from pathlib import Path
            config_path = Path("configs/data.yaml")
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                db_path = config['db']['path']
            else:
                logger.error("[VRPCore] configs/data.yaml not found and db_path not specified")
                return pd.DataFrame()
        
        # Connect to database
        try:
            con = open_readonly_connection(db_path)
        except Exception as e:
            logger.error(f"[VRPCore] Failed to connect to database: {e}")
            return pd.DataFrame()
        
        # Determine date range for VRP data
        vrp_start = str(rv_df['date'].min().date()) if start_date is None else start_date
        vrp_end = str(rv_df['date'].max().date()) if end_date is None else end_date
        
        try:
            # Load VRP inputs
            vrp_data = load_vrp_inputs(con, vrp_start, vrp_end)
        except Exception as e:
            logger.error(f"[VRPCore] Failed to load VRP inputs: {e}")
            con.close()
            return pd.DataFrame()
        finally:
            con.close()
        
        if vrp_data.empty:
            logger.warning("[VRPCore] No VRP data available")
            return pd.DataFrame()
        
        # Merge VRP data with realized vol
        # Use inner join to ensure we have both VIX and realized vol
        df = vrp_data.merge(rv_df, on='date', how='inner')
        
        if df.empty:
            logger.warning("[VRPCore] No overlapping data between VRP and ES returns")
            return pd.DataFrame()
        
        # Compute VRP spread: VIX - Realized Vol
        # VIX is in vol points (20 = 20%), realized vol is in decimals (0.20 = 20%)
        # Convert realized vol to vol points by multiplying by 100
        df['vrp_spread'] = df['vix'] - (df['rv_es_21'] * 100.0)
        
        # Compute z-scored VRP spread
        df['vrp_z'] = _zscore_rolling(
            df['vrp_spread'],
            window=self.zscore_window,
            clip=self.clip
        )
        
        # Drop rows with NaN in key features
        # Keep only rows where we have valid VRP signal
        df = df.dropna(subset=['vrp_z'])
        
        if df.empty:
            logger.warning("[VRPCore] No valid features after dropna")
            return pd.DataFrame()
        
        # Set date as index
        df = df.set_index('date')
        
        logger.info(f"[VRPCore] Computed features for {len(df)} days")
        logger.info(f"[VRPCore] VRP spread: mean={df['vrp_spread'].mean():.2f}, "
                   f"std={df['vrp_spread'].std():.2f}")
        
        return df


def build_vrp_core_features(
    market,
    start: Optional[str] = None,
    end: Optional[str] = None,
    rv_lookback: int = 21,
    zscore_window: int = 252,
    clip: float = 3.0,
    db_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Convenience function to build VRP core features.
    
    Args:
        market: MarketData instance
        start: Start date (YYYY-MM-DD)
        end: End date (YYYY-MM-DD)
        rv_lookback: Lookback period for realized vol (default: 21)
        zscore_window: Window for z-score standardization (default: 252)
        clip: Z-score clipping bounds (default: 3.0)
        db_path: Path to canonical DuckDB (default: from config)
        
    Returns:
        DataFrame with VRP core features
    """
    features = VRPCoreFeatures(
        rv_lookback=rv_lookback,
        zscore_window=zscore_window,
        clip=clip,
        db_path=db_path
    )
    
    return features.compute(market, start_date=start, end_date=end)

