"""
Policy Feature Builder

This module implements the PolicyFeatureBuilder layer that sits between MarketData
(raw inputs) and EnginePolicyV1 in the canonical execution stack.

Responsibilities:
- Load: VIX (FRED), VIX3M, VVIX, VX1/VX2
- Compute: gamma_stress_proxy, vx_backwardation, vrp_stress_proxy
- Attach to: market.policy_features or market.features["policy.*"]
- Enforce: aligned dates, no silent dropna, explicit NaN diagnostics

This keeps policy upstream of sleeves, exactly as SYSTEM_CONSTRUCTION demands.
"""

import logging
from typing import Optional, Union, Dict
from datetime import datetime
import pandas as pd
import numpy as np

from src.market_data.vrp_loaders import (
    load_vix,
    load_vix3m,
    load_vvix,
    load_vx_curve,
)
from src.agents.feature_gamma_stress import compute_gamma_stress_proxy
from src.agents.feature_vx_backwardation import compute_vx_backwardation
from src.agents.feature_vrp_stress import compute_vrp_stress_proxy

logger = logging.getLogger(__name__)


class PolicyFeatureBuilder:
    """
    Policy Feature Builder: Loads and computes policy features for EnginePolicyV1.
    
    This class sits between MarketData (raw inputs) and EnginePolicyV1 in the
    canonical execution stack. It ensures policy features are available upstream
    of sleeves, exactly as SYSTEM_CONSTRUCTION demands.
    
    Features computed:
    - gamma_stress_proxy: Binary indicator for gamma/vol stress (VVIX 95th percentile)
    - vx_backwardation: Binary indicator for VX curve backwardation (VX1 > VX2)
    - vrp_stress_proxy: Composite stress indicator (VVIX >= 99th OR gamma_stress + backwardation)
    
    Raw data loaded:
    - VIX (FRED)
    - VIX3M
    - VVIX
    - VX1/VX2 (from VX curve)
    """
    
    def __init__(
        self,
        market,
        gamma_percentile_threshold: float = 0.95,
        gamma_window: int = 252,
        vrp_percentile_threshold: float = 0.99,
        vrp_window: int = 252
    ):
        """
        Initialize PolicyFeatureBuilder.
        
        Args:
            market: MarketData instance (for DB connection)
            gamma_percentile_threshold: Percentile threshold for gamma stress (default: 0.95)
            gamma_window: Rolling window for gamma percentile (default: 252)
            vrp_percentile_threshold: Percentile threshold for VRP stress (default: 0.99)
            vrp_window: Rolling window for VRP percentile (default: 252)
        """
        self.market = market
        self.gamma_percentile_threshold = gamma_percentile_threshold
        self.gamma_window = gamma_window
        self.vrp_percentile_threshold = vrp_percentile_threshold
        self.vrp_window = vrp_window
        
        # Get DB connection from market
        if hasattr(market, 'conn'):
            self.con = market.conn
            self.close_conn = False
        else:
            # Fallback: open connection (shouldn't happen in normal usage)
            from src.agents.utils_db import open_readonly_connection
            db_path = getattr(market, 'db_path', None)
            if db_path is None:
                raise ValueError(
                    "[PolicyFeatureBuilder] No DB connection available from market. "
                    "MarketData must have a valid connection."
                )
            self.con = open_readonly_connection(db_path)
            self.close_conn = True
        
        logger.info(
            f"[PolicyFeatureBuilder] Initialized with "
            f"gamma_threshold={gamma_percentile_threshold}, "
            f"gamma_window={gamma_window}, "
            f"vrp_threshold={vrp_percentile_threshold}, "
            f"vrp_window={vrp_window}"
        )
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - close connection if we opened it."""
        if self.close_conn and hasattr(self, 'con'):
            self.con.close()
    
    def load_raw_data(
        self,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime]
    ) -> Dict[str, pd.DataFrame]:
        """
        Load raw policy data: VIX, VIX3M, VVIX, VX1/VX2.
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
        
        Returns:
            Dict with keys: 'vix', 'vix3m', 'vvix', 'vx_curve'
            Each value is a DataFrame with 'date' column and data column(s)
        """
        start_str = pd.to_datetime(start_date).strftime("%Y-%m-%d")
        end_str = pd.to_datetime(end_date).strftime("%Y-%m-%d")
        
        logger.info(f"[PolicyFeatureBuilder] Loading raw data from {start_str} to {end_str}")
        
        raw_data = {}
        
        # Load VIX (FRED)
        try:
            vix_df = load_vix(self.con, start_str, end_str)
            if not vix_df.empty:
                vix_df['date'] = pd.to_datetime(vix_df['date'])
                raw_data['vix'] = vix_df
                logger.info(f"[PolicyFeatureBuilder] Loaded VIX: {len(vix_df)} rows")
            else:
                logger.warning("[PolicyFeatureBuilder] VIX data is empty")
                raw_data['vix'] = pd.DataFrame(columns=['date', 'vix'])
        except Exception as e:
            logger.error(f"[PolicyFeatureBuilder] Error loading VIX: {e}")
            raw_data['vix'] = pd.DataFrame(columns=['date', 'vix'])
        
        # Load VIX3M
        try:
            vix3m_df = load_vix3m(self.con, start_str, end_str)
            if not vix3m_df.empty:
                vix3m_df['date'] = pd.to_datetime(vix3m_df['date'])
                raw_data['vix3m'] = vix3m_df
                logger.info(f"[PolicyFeatureBuilder] Loaded VIX3M: {len(vix3m_df)} rows")
            else:
                logger.warning("[PolicyFeatureBuilder] VIX3M data is empty")
                raw_data['vix3m'] = pd.DataFrame(columns=['date', 'vix3m'])
        except Exception as e:
            logger.error(f"[PolicyFeatureBuilder] Error loading VIX3M: {e}")
            raw_data['vix3m'] = pd.DataFrame(columns=['date', 'vix3m'])
        
        # Load VVIX
        try:
            vvix_df = load_vvix(self.con, start_str, end_str)
            if not vvix_df.empty:
                vvix_df['date'] = pd.to_datetime(vvix_df['date'])
                raw_data['vvix'] = vvix_df
                logger.info(f"[PolicyFeatureBuilder] Loaded VVIX: {len(vvix_df)} rows")
            else:
                logger.warning("[PolicyFeatureBuilder] VVIX data is empty")
                raw_data['vvix'] = pd.DataFrame(columns=['date', 'vvix'])
        except Exception as e:
            logger.error(f"[PolicyFeatureBuilder] Error loading VVIX: {e}")
            raw_data['vvix'] = pd.DataFrame(columns=['date', 'vvix'])
        
        # Load VX curve (VX1/VX2)
        try:
            vx_df = load_vx_curve(self.con, start_str, end_str)
            if not vx_df.empty:
                vx_df['date'] = pd.to_datetime(vx_df['date'])
                raw_data['vx_curve'] = vx_df
                logger.info(f"[PolicyFeatureBuilder] Loaded VX curve: {len(vx_df)} rows")
            else:
                logger.warning("[PolicyFeatureBuilder] VX curve data is empty")
                raw_data['vx_curve'] = pd.DataFrame(columns=['date', 'vx1', 'vx2', 'vx3'])
        except Exception as e:
            logger.error(f"[PolicyFeatureBuilder] Error loading VX curve: {e}")
            raw_data['vx_curve'] = pd.DataFrame(columns=['date', 'vx1', 'vx2', 'vx3'])
        
        return raw_data
    
    def compute_features(
        self,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime]
    ) -> Dict[str, pd.Series]:
        """
        Compute policy features: gamma_stress_proxy, vx_backwardation, vrp_stress_proxy.
        
        Note: vrp_stress_proxy depends on gamma_stress_proxy and vx_backwardation
        being available in market.features, so we attach the first two before computing vrp_stress_proxy.
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
        
        Returns:
            Dict with keys: 'gamma_stress_proxy', 'vx_backwardation', 'vrp_stress_proxy'
            Each value is a Series indexed by date
        """
        logger.info(f"[PolicyFeatureBuilder] Computing features from {start_date} to {end_date}")
        
        features = {}
        
        # Compute gamma_stress_proxy
        try:
            gamma_stress = compute_gamma_stress_proxy(
                self.market,
                start_date=start_date,
                end_date=end_date,
                percentile_threshold=self.gamma_percentile_threshold,
                window=self.gamma_window
            )
            if not gamma_stress.empty:
                features['gamma_stress_proxy'] = gamma_stress
                logger.info(
                    f"[PolicyFeatureBuilder] Computed gamma_stress_proxy: {len(gamma_stress)} days, "
                    f"{gamma_stress.sum()} stress days ({gamma_stress.mean() * 100:.1f}%)"
                )
            else:
                logger.warning("[PolicyFeatureBuilder] gamma_stress_proxy is empty")
                features['gamma_stress_proxy'] = pd.Series(dtype=float, name='gamma_stress_proxy')
        except Exception as e:
            logger.error(f"[PolicyFeatureBuilder] Error computing gamma_stress_proxy: {e}")
            features['gamma_stress_proxy'] = pd.Series(dtype=float, name='gamma_stress_proxy')
        
        # Compute vx_backwardation
        try:
            vx_backward = compute_vx_backwardation(
                self.market,
                start_date=start_date,
                end_date=end_date
            )
            if not vx_backward.empty:
                features['vx_backwardation'] = vx_backward
                logger.info(
                    f"[PolicyFeatureBuilder] Computed vx_backwardation: {len(vx_backward)} days, "
                    f"{vx_backward.sum()} backwardated days ({vx_backward.mean() * 100:.1f}%)"
                )
            else:
                logger.warning("[PolicyFeatureBuilder] vx_backwardation is empty")
                features['vx_backwardation'] = pd.Series(dtype=float, name='vx_backwardation')
        except Exception as e:
            logger.error(f"[PolicyFeatureBuilder] Error computing vx_backwardation: {e}")
            features['vx_backwardation'] = pd.Series(dtype=float, name='vx_backwardation')
        
        # Attach gamma_stress_proxy and vx_backwardation to market.features
        # BEFORE computing vrp_stress_proxy (which depends on them)
        if not hasattr(self.market, 'features'):
            self.market.features = {}
        if 'gamma_stress_proxy' in features:
            self.market.features['gamma_stress_proxy'] = features['gamma_stress_proxy']
        if 'vx_backwardation' in features:
            self.market.features['vx_backwardation'] = features['vx_backwardation']
        
        # Compute vrp_stress_proxy (depends on gamma_stress_proxy and vx_backwardation in market.features)
        try:
            vrp_stress = compute_vrp_stress_proxy(
                self.market,
                start_date=start_date,
                end_date=end_date,
                vvix_percentile_threshold=self.vrp_percentile_threshold,
                window=self.vrp_window
            )
            if not vrp_stress.empty:
                features['vrp_stress_proxy'] = vrp_stress
                logger.info(
                    f"[PolicyFeatureBuilder] Computed vrp_stress_proxy: {len(vrp_stress)} days, "
                    f"{vrp_stress.sum()} stress days ({vrp_stress.mean() * 100:.1f}%)"
                )
            else:
                logger.warning("[PolicyFeatureBuilder] vrp_stress_proxy is empty")
                features['vrp_stress_proxy'] = pd.Series(dtype=float, name='vrp_stress_proxy')
        except Exception as e:
            logger.error(f"[PolicyFeatureBuilder] Error computing vrp_stress_proxy: {e}")
            features['vrp_stress_proxy'] = pd.Series(dtype=float, name='vrp_stress_proxy')
        
        return features
    
    def align_dates(
        self,
        features: Dict[str, pd.Series],
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None
    ) -> Dict[str, pd.Series]:
        """
        Align all feature series to a common date index.
        
        Enforces aligned dates and explicit NaN diagnostics (no silent dropna).
        
        Args:
            features: Dict of feature Series
            start_date: Optional start date to filter (default: use min date from features)
            end_date: Optional end date to filter (default: use max date from features)
        
        Returns:
            Dict of aligned feature Series with common date index
        """
        if not features:
            logger.warning("[PolicyFeatureBuilder] No features to align")
            return {}
        
        # Collect all dates from all features
        all_dates = set()
        for feature_name, feature_series in features.items():
            if not feature_series.empty:
                all_dates.update(feature_series.index)
        
        if not all_dates:
            logger.warning("[PolicyFeatureBuilder] No dates found in features")
            return features
        
        # Create common date index
        common_index = pd.DatetimeIndex(sorted(all_dates))
        
        # Apply date filters if provided
        if start_date:
            start_dt = pd.to_datetime(start_date)
            common_index = common_index[common_index >= start_dt]
        if end_date:
            end_dt = pd.to_datetime(end_date)
            common_index = common_index[common_index <= end_dt]
        
        logger.info(
            f"[PolicyFeatureBuilder] Aligning features to common index: "
            f"{len(common_index)} dates from {common_index.min()} to {common_index.max()}"
        )
        
        # Align all features to common index
        aligned_features = {}
        nan_diagnostics = {}
        
        for feature_name, feature_series in features.items():
            if feature_series.empty:
                # Create empty series with common index
                aligned_features[feature_name] = pd.Series(
                    dtype=float,
                    index=common_index,
                    name=feature_name
                )
                nan_diagnostics[feature_name] = {
                    'total_dates': len(common_index),
                    'nan_count': len(common_index),
                    'nan_pct': 100.0,
                    'has_data': False
                }
            else:
                # Reindex to common index (forward-fill missing dates, but don't dropna)
                aligned = feature_series.reindex(common_index)
                
                # Explicit NaN diagnostics (no silent dropna)
                nan_count = aligned.isna().sum()
                nan_pct = (nan_count / len(aligned)) * 100.0 if len(aligned) > 0 else 0.0
                
                nan_diagnostics[feature_name] = {
                    'total_dates': len(aligned),
                    'nan_count': int(nan_count),
                    'nan_pct': float(nan_pct),
                    'has_data': True,
                    'data_dates': len(feature_series),
                    'aligned_dates': len(aligned)
                }
                
                aligned_features[feature_name] = aligned
        
        # Log NaN diagnostics
        logger.info("[PolicyFeatureBuilder] NaN Diagnostics:")
        for feature_name, diag in nan_diagnostics.items():
            logger.info(
                f"  {feature_name}: {diag['nan_count']}/{diag['total_dates']} NaN "
                f"({diag['nan_pct']:.1f}%)"
            )
            if diag.get('has_data'):
                logger.info(
                    f"    Original: {diag['data_dates']} dates, "
                    f"Aligned: {diag['aligned_dates']} dates"
                )
        
        return aligned_features
    
    def build(
        self,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        attach_to_market: bool = True
    ) -> Dict[str, pd.Series]:
        """
        Build all policy features and attach to market object.
        
        This is the main entry point that:
        1. Loads raw data (VIX, VIX3M, VVIX, VX1/VX2)
        2. Computes features (gamma_stress_proxy, vx_backwardation, vrp_stress_proxy)
        3. Aligns dates
        4. Attaches to market.policy_features or market.features["policy.*"]
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            attach_to_market: If True, attach features to market object (default: True)
        
        Returns:
            Dict of aligned feature Series
        """
        logger.info("=" * 80)
        logger.info("[PolicyFeatureBuilder] Building policy features")
        logger.info("=" * 80)
        
        # Step 1: Load raw data (for validation/logging, not strictly required for features)
        raw_data = self.load_raw_data(start_date, end_date)
        
        # Step 2: Compute features
        features = self.compute_features(start_date, end_date)
        
        # Step 3: Align dates
        aligned_features = self.align_dates(features, start_date, end_date)
        
        # Step 4: Attach to market object
        if attach_to_market:
            self.attach_to_market(aligned_features)
        
        logger.info("=" * 80)
        logger.info("[PolicyFeatureBuilder] Policy features built successfully")
        logger.info(f"  Features: {list(aligned_features.keys())}")
        logger.info("=" * 80)
        
        return aligned_features
    
    def attach_to_market(self, features: Dict[str, pd.Series]):
        """
        Attach features to market object.
        
        Attaches to:
        - market.policy_features (preferred)
        - market.features["policy.*"] (fallback)
        
        Args:
            features: Dict of feature Series
        """
        if not features:
            logger.warning("[PolicyFeatureBuilder] No features to attach")
            return
        
        # Try to attach to market.policy_features first
        if not hasattr(self.market, 'policy_features'):
            self.market.policy_features = {}
        
        # Attach each feature
        for feature_name, feature_series in features.items():
            # Attach to policy_features dict
            self.market.policy_features[feature_name] = feature_series
            
            # Also attach to market.features["policy.*"] for backward compatibility
            if not hasattr(self.market, 'features'):
                self.market.features = {}
            
            # Use "policy.*" prefix for namespacing
            policy_feature_name = f"policy.{feature_name}"
            self.market.features[policy_feature_name] = feature_series
            
            # Also attach without prefix for direct access (backward compatibility)
            self.market.features[feature_name] = feature_series
        
        logger.info(
            f"[PolicyFeatureBuilder] Attached {len(features)} features to market object: "
            f"{list(features.keys())}"
        )
