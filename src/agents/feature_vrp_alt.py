"""
VRP Alt Features: VIX vs short-term realized volatility (RV5) signals.

Features:
- alt_vrp: VIX - RV5 (spot implied vol vs 5-day realized vol)
- alt_vrp_z: Z-scored Alt-VRP spread

Strategy:
- Alt-VRP spread = VIX (implied vol) - RV5 (5-day realized vol)
- Positive Alt-VRP = volatility risk premium exists (typical)
- Signal = z-scored Alt-VRP spread, clipped to ±3σ
- Trade VX1 (front month futures) directionally based on Alt-VRP

Data Sources:
- VIX: FRED VIXCLS via canonical DB (f_fred_observations)
- RV5: Computed from ES returns (5-day rolling std * sqrt(252) * 100)
- VX1: VX futures from canonical DB (market_data, symbol @VX=101XN)

Note on Phase-0 MaxDD:
VRP-Alt Phase-0 showed Sharpe ≈ 0.10 with catastrophic MaxDD (~–94%).
This mirrors VRP-Core, where Phase-0 also had severe drawdowns (≈–87%) but a clearly positive Sharpe.
In this framework, Phase-0's primary pass criterion is economic edge (Sharpe ≥ 0.1);
MaxDD is expected to be extreme for raw, unscaled short-vol signals and is addressed in Phase-1
via z-scoring and vol targeting. VRP-Alt is therefore treated as a Phase-0 economic PASS
and advanced to Phase-1 engineering.
"""

import logging
from typing import Optional, Union
from datetime import datetime
import pandas as pd
import numpy as np

from src.market_data.vrp_loaders import load_vix, load_vx_curve
from src.agents.utils_db import open_readonly_connection
from src.agents.data_broker import MarketData

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


class VRPAltFeatures:
    """
    Computes VRP Alt features (VIX vs RV5).
    
    Features:
    - alt_vrp: VIX - RV5 (spot implied vol vs 5-day realized vol)
    - alt_vrp_z: Z-scored Alt-VRP spread
    """
    
    def __init__(
        self,
        zscore_window: int = 252,
        clip: float = 3.0,
        db_path: Optional[str] = None
    ):
        """
        Initialize VRP Alt features calculator.
        
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
        Compute VRP Alt features.
        
        Args:
            market: MarketData instance (used for ES returns to compute RV5)
            start_date: Start date for feature computation (default: earliest available)
            end_date: End date for feature computation (default: last available)
            
        Returns:
            DataFrame indexed by date with columns:
            - vix: VIX level (vol points)
            - rv5: 5-day realized ES volatility (vol points)
            - vx1: VX1 front month price (vol points)
            - alt_vrp: VIX - RV5 (Alt-VRP spread in vol points)
            - alt_vrp_z: Z-scored Alt-VRP spread
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
                logger.error("[VRPAlt] configs/data.yaml not found and db_path not specified")
                return pd.DataFrame()
        
        # Determine date range
        vrp_start = start_date if start_date else "2004-01-01"  # VX futures start
        vrp_end = end_date if end_date else pd.Timestamp.today().strftime('%Y-%m-%d')
        
        # Connect to database for VIX and VX1
        try:
            con = open_readonly_connection(db_path)
        except Exception as e:
            logger.error(f"[VRPAlt] Failed to connect to database: {e}")
            return pd.DataFrame()
        
        try:
            # Load VIX and VX1
            df_vix = load_vix(con, vrp_start, vrp_end)
            df_vx = load_vx_curve(con, vrp_start, vrp_end)
        except Exception as e:
            logger.error(f"[VRPAlt] Failed to load VRP inputs: {e}")
            con.close()
            return pd.DataFrame()
        finally:
            con.close()
        
        if df_vix.empty or df_vx.empty:
            logger.warning("[VRPAlt] No VIX or VX data available")
            return pd.DataFrame()
        
        # Merge VIX and VX1
        df = df_vix.merge(df_vx[['date', 'vx1']], on="date", how="inner")
        df = df.dropna(subset=['vix', 'vx1']).copy()
        
        if df.empty:
            logger.warning("[VRPAlt] No valid data after merging VIX and VX1")
            return pd.DataFrame()
        
        # Convert date to datetime if needed
        if not isinstance(df['date'].iloc[0], pd.Timestamp):
            df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        
        # Filter by date range if specified
        if start_date is not None:
            start_dt = pd.to_datetime(start_date)
            df = df[df['date'] >= start_dt]
        
        if end_date is not None:
            end_dt = pd.to_datetime(end_date)
            df = df[df['date'] <= end_dt]
        
        if df.empty:
            logger.warning("[VRPAlt] No data after date filtering")
            return pd.DataFrame()
        
        # Compute RV5 from ES returns using MarketData
        es_symbol = "ES_FRONT_CALENDAR_2D"
        
        if es_symbol not in market.returns_cont.columns:
            logger.error(f"[VRPAlt] {es_symbol} not found in market data")
            return pd.DataFrame()
        
        # Get continuous returns (log returns)
        returns_cont = market.returns_cont[[es_symbol]].copy()
        
        if returns_cont.empty:
            logger.warning("[VRPAlt] No ES returns data available")
            return pd.DataFrame()
        
        # Filter by date range
        if start_date is not None:
            start_dt = pd.to_datetime(start_date)
            returns_cont = returns_cont[returns_cont.index >= start_dt]
        
        if end_date is not None:
            end_dt = pd.to_datetime(end_date)
            returns_cont = returns_cont[returns_cont.index <= end_dt]
        
        if returns_cont.empty:
            logger.warning("[VRPAlt] No ES returns data after filtering")
            return pd.DataFrame()
        
        # Compute 5-day realized volatility (annualized, in vol points)
        # RV5 = rolling std of daily returns * sqrt(252) * 100
        # sqrt(252) annualizes, *100 converts to vol points
        rv5 = returns_cont[es_symbol].rolling(
            window=5,
            min_periods=5
        ).std() * np.sqrt(252) * 100.0
        
        # Convert RV5 to DataFrame and merge with VIX/VX1 data
        rv5_df = pd.DataFrame({
            'date': rv5.index,
            'rv5': rv5.values
        })
        rv5_df = rv5_df.dropna()
        
        # Merge RV5 with VIX/VX1 data
        df = df.merge(rv5_df, on="date", how="inner")
        df = df.dropna(subset=['vix', 'rv5', 'vx1']).copy()
        
        if df.empty:
            logger.warning("[VRPAlt] No valid data after merging RV5")
            return pd.DataFrame()
        
        # Compute Alt-VRP spread: VIX - RV5
        # Both VIX and RV5 are in vol points (same units)
        df['alt_vrp'] = df['vix'] - df['rv5']
        
        # Compute z-scored Alt-VRP spread
        df['alt_vrp_z'] = _zscore_rolling(
            df['alt_vrp'],
            window=self.zscore_window,
            clip=self.clip
        )
        
        # Drop rows with NaN in key features (after z-scoring)
        df = df.dropna(subset=['alt_vrp_z'])
        
        if df.empty:
            logger.warning("[VRPAlt] No valid features after z-scoring")
            return pd.DataFrame()
        
        # Set date as index
        df = df.set_index('date')
        
        logger.info(f"[VRPAlt] Computed features for {len(df)} days")
        logger.info(f"[VRPAlt] Alt-VRP spread: mean={df['alt_vrp'].mean():.2f}, "
                   f"std={df['alt_vrp'].std():.2f}")
        
        return df


def build_vrp_alt_features(
    market,
    start: Optional[str] = None,
    end: Optional[str] = None,
    zscore_window: int = 252,
    clip: float = 3.0,
    db_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Convenience function to build VRP Alt features.
    
    Args:
        market: MarketData instance
        start: Start date (YYYY-MM-DD)
        end: End date (YYYY-MM-DD)
        zscore_window: Window for z-score standardization (default: 252)
        clip: Z-score clipping bounds (default: 3.0)
        db_path: Path to canonical DuckDB (default: from config)
        
    Returns:
        DataFrame with VRP Alt features
    """
    features = VRPAltFeatures(
        zscore_window=zscore_window,
        clip=clip,
        db_path=db_path
    )
    
    return features.compute(market, start_date=start, end_date=end)

