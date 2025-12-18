"""
VRP Convergence Features: VIX vs VX1 convergence signals.

Features:
- spread_conv: VIX - VX1 (spot vs front-month futures)
- Optional: curve_slope_vx = VX2 - VX1 (for diagnostics/plots only)

Strategy:
- Convergence spread = VIX (spot) - VX1 (front-month futures)
- When VX1 is too high vs VIX, expect convergence lower in VX1 → short VX1
- When VX1 is too low vs VIX, expect convergence higher in VX1 → long VX1
- Z-scoring provides consistent signal strength across regimes

Data Sources:
- VIX: FRED VIXCLS via canonical DB (f_fred_observations)
- VX1/2/3: VX futures from canonical DB (market_data, symbols @VX=101XN, @VX=201XN, @VX=301XN)
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


class VRPConvergenceFeatures:
    """
    Computes VRP convergence features.
    
    Features:
    - spread_conv: VIX - VX1 (spot vs front-month futures)
    - conv_z: Z-scored convergence spread
    - Optional: curve_slope_vx = VX2 - VX1 (for diagnostics only)
    """
    
    def __init__(
        self,
        zscore_window: int = 252,
        clip: float = 3.0,
        db_path: Optional[str] = None
    ):
        """
        Initialize VRP convergence features calculator.
        
        Args:
            zscore_window: Rolling window for z-score standardization (default: 252 days)
            clip: Z-score clipping bounds (default: 3.0)
            db_path: Path to canonical DuckDB (default: from configs/data.yaml)
        """
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
        Compute VRP convergence features.
        
        Args:
            market: MarketData instance (not used for VRP convergence, but kept for API consistency)
            start_date: Start date for feature computation (default: earliest available)
            end_date: End date for feature computation (default: last available)
            
        Returns:
            DataFrame indexed by date with columns:
            - vix: VIX level (vol points)
            - vx1: VX1 front month price (vol points)
            - vx2: VX2 second month price (vol points, optional)
            - spread_conv: VIX - VX1 (convergence spread in vol points)
            - conv_z: Z-scored convergence spread
            - curve_slope_vx: VX2 - VX1 (optional, for diagnostics only)
        """
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
                logger.error("[VRPConvergence] configs/data.yaml not found and db_path not specified")
                return pd.DataFrame()
        
        # Connect to database
        try:
            con = open_readonly_connection(db_path)
        except Exception as e:
            logger.error(f"[VRPConvergence] Failed to connect to database: {e}")
            return pd.DataFrame()
        
        # Determine date range for VRP data
        vrp_start = start_date if start_date else "2009-09-18"  # VIX3M start
        vrp_end = end_date if end_date else pd.Timestamp.today().strftime('%Y-%m-%d')
        
        try:
            # Load VRP inputs (VIX, VIX3M, VX1/2/3)
            vrp_data = load_vrp_inputs(con, vrp_start, vrp_end)
        except Exception as e:
            logger.error(f"[VRPConvergence] Failed to load VRP inputs: {e}")
            con.close()
            return pd.DataFrame()
        finally:
            con.close()
        
        if vrp_data.empty:
            logger.warning("[VRPConvergence] No VRP data available")
            return pd.DataFrame()
        
        # Filter by date range if specified
        if start_date is not None:
            start_dt = pd.to_datetime(start_date)
            vrp_data = vrp_data[vrp_data['date'] >= start_dt]
        
        if end_date is not None:
            end_dt = pd.to_datetime(end_date)
            vrp_data = vrp_data[vrp_data['date'] <= end_dt]
        
        # Drop rows with any NAs in key columns (respect canonical NA-handling rules)
        df = vrp_data.dropna(subset=['vix', 'vx1']).copy()
        
        if df.empty:
            logger.warning("[VRPConvergence] No valid data after dropna")
            return pd.DataFrame()
        
        # Compute convergence spread: VIX - VX1
        # Both VIX and VX1 are already in vol points (same units)
        df['spread_conv'] = df['vix'] - df['vx1']
        
        # Optional: curve slope (VX2 - VX1) for diagnostics/plots
        if 'vx2' in df.columns:
            df['curve_slope_vx'] = df['vx2'] - df['vx1']
        else:
            df['curve_slope_vx'] = np.nan
        
        # Compute z-scored convergence spread
        df['conv_z'] = _zscore_rolling(
            df['spread_conv'],
            window=self.zscore_window,
            clip=self.clip
        )
        
        # Drop rows with NaN in key features (after z-scoring)
        df = df.dropna(subset=['conv_z'])
        
        if df.empty:
            logger.warning("[VRPConvergence] No valid features after z-scoring")
            return pd.DataFrame()
        
        # Set date as index
        df = df.set_index('date')
        
        logger.info(f"[VRPConvergence] Computed features for {len(df)} days")
        logger.info(f"[VRPConvergence] Convergence spread: mean={df['spread_conv'].mean():.2f}, "
                   f"std={df['spread_conv'].std():.2f}")
        
        return df


def build_vrp_convergence_features(
    market,
    start: Optional[str] = None,
    end: Optional[str] = None,
    zscore_window: int = 252,
    clip: float = 3.0,
    db_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Convenience function to build VRP convergence features.
    
    Args:
        market: MarketData instance
        start: Start date (YYYY-MM-DD)
        end: End date (YYYY-MM-DD)
        zscore_window: Window for z-score standardization (default: 252)
        clip: Z-score clipping bounds (default: 3.0)
        db_path: Path to canonical DuckDB (default: from config)
        
    Returns:
        DataFrame with VRP convergence features
    """
    features = VRPConvergenceFeatures(
        zscore_window=zscore_window,
        clip=clip,
        db_path=db_path
    )
    
    return features.compute(market, start_date=start, end_date=end)

