"""
Rates Curve Features: Implied Yields and Curve Features from Treasury Futures.

Computes implied yields using FRED yields as anchors (a few days back) and
futures price changes (via DV01) to update yields forward. Then constructs
curve features (2s10s, 5s30s) standardized with rolling z-scores.
"""

import logging
from typing import Optional, Union, Dict
from datetime import datetime
import pandas as pd
import numpy as np
import yaml
from pathlib import Path

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


class RatesCurveFeatures:
    """
    Builds implied yields and curve features from Treasury futures using
    FRED yields as an anchor a few days back, and futures price changes
    plus DV01 to update yields forward.
    
    Features computed:
    - curve_2s10s_z: 2s10s curve slope standardized
    - curve_5s30s_z: 5s30s curve slope standardized
    - curve_2s5s10s_curv_z: 2s-5s-10s belly curvature standardized
    - curve_5s10s30s_curv_z: 5s-10s-30s belly curvature standardized
    """
    
    def __init__(
        self,
        market,
        dv01_cfg: Optional[Dict] = None,
        dv01_config_path: str = "configs/rates_dv01.yaml",
        window: int = 252,
        anchor_lag_days: int = 2
    ):
        """
        Initialize RatesCurveFeatures.
        
        Args:
            market: MarketData instance
            dv01_cfg: Optional dict with DV01 values {"ZT": dv01, "ZF": dv01, ...}
                     If None, loads from dv01_config_path
            dv01_config_path: Path to DV01 config YAML file
            window: Rolling window for z-score standardization
            anchor_lag_days: Number of business days to lag FRED anchor (default: 2)
        """
        self.market = market
        self.window = window
        self.anchor_lag_days = anchor_lag_days
        
        # Load DV01 config
        if dv01_cfg is None:
            config_path = Path(dv01_config_path)
            if not config_path.exists():
                raise FileNotFoundError(f"DV01 config not found: {dv01_config_path}")
            with open(config_path, 'r', encoding='utf-8') as f:
                dv01_yaml = yaml.safe_load(f)
            self.dv01 = {
                "ZT": dv01_yaml["ZT"]["dv01"],
                "ZF": dv01_yaml["ZF"]["dv01"],
                "ZN": dv01_yaml["ZN"]["dv01"],
                "UB": dv01_yaml["UB"]["dv01"],
            }
        else:
            self.dv01 = dv01_cfg
        
        # Map futures symbols to FRED series
        self.fred_map = {
            "ZT": "DGS2",
            "ZF": "DGS5",
            "ZN": "DGS10",
            "UB": "DGS30",
        }
        
        # Map futures symbols to database symbols (continuous rank 0)
        self.symbol_map = {
            "ZT": "ZT_FRONT_VOLUME",
            "ZF": "ZF_FRONT_VOLUME",
            "ZN": "ZN_FRONT_VOLUME",
            "UB": "UB_FRONT_VOLUME",
        }
    
    def _get_anchor_date(self, end_date: Union[str, datetime]) -> pd.Timestamp:
        """
        Get anchor date (end_date - anchor_lag_days business days).
        
        Args:
            end_date: End date for feature computation
            
        Returns:
            Anchor date (business day)
        """
        end_dt = pd.to_datetime(end_date)
        # Subtract business days
        from pandas.tseries.offsets import BDay
        anchor = end_dt - BDay(self.anchor_lag_days)
        return anchor
    
    def compute(
        self,
        end_date: Optional[Union[str, datetime]] = None
    ) -> pd.DataFrame:
        """
        Compute rates curve features up to end_date.
        
        Args:
            end_date: End date for feature computation (default: last available)
            
        Returns:
            DataFrame indexed by date with columns:
            - curve_2s10s_z: 2s10s curve slope standardized
            - curve_5s30s_z: 5s30s curve slope standardized
            - curve_2s5s10s_curv_z: 2s-5s-10s belly curvature standardized
            - curve_5s10s30s_curv_z: 5s-10s-30s belly curvature standardized
        """
        # 1) Get anchor date
        if end_date is None:
            # Use last available date (approximate)
            end_date = pd.Timestamp.today()
        else:
            end_date = pd.to_datetime(end_date)
        
        anchor_date = self._get_anchor_date(end_date)
        
        # 2) Get futures prices - fetch ALL available data from the beginning
        # Similar to SR3: get all data, let rolling window handle the lookback requirement
        # Only limit the end_date, not the start_date
        
        px = {}
        for sym in ["ZT", "ZF", "ZN", "UB"]:
            db_symbol = self.symbol_map[sym]
            # Get continuous (back-adjusted) futures prices
            # Use prices_cont for all "what did the market do?" calculations
            prices_cont = self.market.prices_cont
            
            if prices_cont.empty or db_symbol not in prices_cont.columns:
                logger.warning(f"[RatesCurve] No continuous price data for {db_symbol}")
                return pd.DataFrame()
            
            # Extract close prices as Series
            price_series = prices_cont[db_symbol].copy()
            
            # Filter by end_date if provided
            if end_date is not None:
                end_dt = pd.to_datetime(end_date)
                price_series = price_series[price_series.index <= end_dt]
            
            if price_series.empty:
                logger.warning(f"[RatesCurve] No price data for {db_symbol} after filtering")
                return pd.DataFrame()
            
            px[sym] = price_series
        
        # Combine into DataFrame
        px_df = pd.DataFrame(px).sort_index()
        
        if px_df.empty:
            logger.warning("[RatesCurve] No futures price data available")
            return pd.DataFrame()
        
        # Ensure anchor_date is in the futures index (or use closest before)
        if anchor_date not in px_df.index:
            available_before = px_df.index[px_df.index <= anchor_date]
            if len(available_before) == 0:
                logger.warning(f"[RatesCurve] No futures data before anchor_date {anchor_date}")
                return pd.DataFrame()
            anchor_date = available_before.max()
        
        # 3) Get FRED yields up to anchor_date
        # Get ALL available FRED data from the beginning (similar to futures prices)
        fred_series_ids = tuple(self.fred_map.values())
        fred_df = self.market.get_fred_indicators(
            series_ids=fred_series_ids,
            start=None,  # Get all available data from the beginning
            end=anchor_date
        )
        
        if fred_df.empty:
            logger.warning(f"[RatesCurve] No FRED data available up to {anchor_date}")
            return pd.DataFrame()
        
        # Get row at anchor_date (or closest before)
        if anchor_date not in fred_df.index:
            available_before = fred_df.index[fred_df.index <= anchor_date]
            if len(available_before) == 0:
                logger.warning(f"[RatesCurve] No FRED data before anchor_date {anchor_date}")
                return pd.DataFrame()
            anchor_date_fred = available_before.max()
        else:
            anchor_date_fred = anchor_date
        
        fred_anchor_row = fred_df.loc[anchor_date_fred]
        
        # 4) Build implied yields from anchor + futures prices
        # Only compute yields for symbols where we have both FRED and futures data
        yields = pd.DataFrame(index=px_df.index, dtype=float)
        
        for sym in px_df.columns:
            fred_series = self.fred_map[sym]
            
            # Check if FRED data is available
            if fred_series not in fred_anchor_row.index:
                logger.warning(f"[RatesCurve] FRED series {fred_series} not available, skipping {sym}")
                continue
            
            y_anchor = fred_anchor_row[fred_series]
            
            if pd.isna(y_anchor):
                logger.warning(f"[RatesCurve] Anchor yield is NaN for {fred_series} at {anchor_date_fred}, skipping {sym}")
                continue
            
            # Get futures prices
            F = px_df[sym]
            F_anchor = F.loc[anchor_date]
            
            if pd.isna(F_anchor):
                logger.warning(f"[RatesCurve] Anchor futures price is NaN for {sym} at {anchor_date}, skipping")
                continue
            
            # Get DV01
            dv = self.dv01[sym]  # per $100 notional
            
            # y_t = y_anchor - (F_t - F_anchor) / (DV01 * 100)
            # Formula: ΔYield ≈ -ΔFuturesPrice / (DV01_fut * 100)
            y_t = y_anchor - (F - F_anchor) / (dv * 100.0)
            yields[sym] = y_t
        
        # 5) Construct curve slopes and curvatures from these yields (only if we have the required data)
        features_dict = {}
        
        # 2s10s curve (requires ZT and ZN)
        if "ZT" in yields.columns and "ZN" in yields.columns:
            y2 = yields["ZT"]
            y10 = yields["ZN"]
            curve_2s10s = y10 - y2
            features_dict["curve_2s10s_z"] = _zscore_rolling(curve_2s10s, window=self.window)
        else:
            logger.warning("[RatesCurve] Missing data for 2s10s curve (need ZT and ZN)")
        
        # 5s30s curve (requires ZF and UB)
        if "ZF" in yields.columns and "UB" in yields.columns:
            y5 = yields["ZF"]
            y30 = yields["UB"]
            curve_5s30s = y30 - y5
            features_dict["curve_5s30s_z"] = _zscore_rolling(curve_5s30s, window=self.window)
        else:
            logger.warning("[RatesCurve] Missing data for 5s30s curve (need ZF and UB)")
        
        # 2s-5s-10s curvature (requires ZT, ZF, ZN)
        # curv_2s5s10s = 2 * y5 - y2 - y10
        # Positive: 5y higher than straight line between 2y and 10y → "hump" in the belly
        # Negative: belly lower → curve is more U-shaped / concave
        if "ZT" in yields.columns and "ZF" in yields.columns and "ZN" in yields.columns:
            y2 = yields["ZT"]
            y5 = yields["ZF"]
            y10 = yields["ZN"]
            curv_2s5s10s = 2.0 * y5 - y2 - y10
            features_dict["curve_2s5s10s_curv_z"] = _zscore_rolling(curv_2s5s10s, window=self.window)
        else:
            logger.warning("[RatesCurve] Missing data for 2s5s10s curvature (need ZT, ZF, ZN)")
        
        # 5s-10s-30s curvature (requires ZF, ZN, UB)
        # curv_5s10s30s = 2 * y10 - y5 - y30
        # Positive: 10y high relative to 5y & 30y (hump at the 10y)
        # Negative: belly low (U-shaped between 5y and 30y)
        if "ZF" in yields.columns and "ZN" in yields.columns and "UB" in yields.columns:
            y5 = yields["ZF"]
            y10 = yields["ZN"]
            y30 = yields["UB"]
            curv_5s10s30s = 2.0 * y10 - y5 - y30
            features_dict["curve_5s10s30s_curv_z"] = _zscore_rolling(curv_5s10s30s, window=self.window)
        else:
            logger.warning("[RatesCurve] Missing data for 5s10s30s curvature (need ZF, ZN, UB)")
        
        if not features_dict:
            logger.warning("[RatesCurve] No curve features could be computed")
            return pd.DataFrame()
        
        # 6) Z-score over time (already done above)
        features = pd.DataFrame(features_dict, index=yields.index)
        
        return features

