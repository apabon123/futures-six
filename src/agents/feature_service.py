"""
FeatureService: Centralized feature computation service.

Manages computation of all features (TSMOM, SR3 curve, etc.) and provides
a unified interface for strategies to access features.
"""

import logging
from typing import Optional, Union, Dict
from datetime import datetime
import pandas as pd

from .feature_sr3_curve import Sr3CurveFeatures
from .feature_rates_curve import RatesCurveFeatures
from .feature_carry_fx_commod import FxCommodCarryFeatures
from .feature_long_momentum import (
    LongMomentumFeatures, 
    MediumMomentumFeatures, 
    CanonicalMediumMomentumFeatures,
    ShortMomentumFeatures, 
    ResidualTrendFeatures, 
    PersistenceFeatures
)

logger = logging.getLogger(__name__)


class FeatureService:
    """
    Centralized feature computation service.
    
    Manages all feature calculators and provides a unified interface
    for strategies to access pre-computed features.
    """
    
    def __init__(self, market, config: Optional[Dict] = None):
        """
        Initialize FeatureService.
        
        Args:
            market: MarketData instance
            config: Optional configuration dict for feature parameters
        """
        self.market = market
        self.config = config or {}
        
        # Initialize feature calculators
        sr3_cfg = self.config.get("sr3_curve", {})
        self.sr3_curve = Sr3CurveFeatures(
            root=sr3_cfg.get("root", "SR3"),
            ranks=sr3_cfg.get("ranks", list(range(12))),
            window=sr3_cfg.get("window", 252)
        )
        
        rates_cfg = self.config.get("rates_curve", {})
        self.rates_curve = RatesCurveFeatures(
            market=self.market,
            dv01_cfg=None,  # Load from config file
            dv01_config_path=rates_cfg.get("dv01_config_path", "configs/rates_dv01.yaml"),
            window=rates_cfg.get("window", 252),
            anchor_lag_days=rates_cfg.get("anchor_lag_days", 2)
        )
        
        fx_commod_cfg = self.config.get("fx_commod_carry", {})
        self.fx_commod_carry = FxCommodCarryFeatures(
            roots=fx_commod_cfg.get("roots", ["CL", "GC", "6E", "6B", "6J"]),
            window=fx_commod_cfg.get("window", 252),
            clip=fx_commod_cfg.get("clip", 3.0)
        )
        
        long_mom_cfg = self.config.get("long_momentum", {})
        self.long_momentum = LongMomentumFeatures(
            symbols=None,  # Will use market.universe
            lookback=long_mom_cfg.get("lookback", 252),
            skip_recent=long_mom_cfg.get("skip_recent", 21),
            vol_window=long_mom_cfg.get("vol_window", 63),
            window=long_mom_cfg.get("window", 252),
            clip=long_mom_cfg.get("clip", 3.0)
        )
        
        med_mom_cfg = self.config.get("medium_momentum", {})
        self.medium_momentum = MediumMomentumFeatures(
            symbols=None,  # Will use market.universe
            lookback=med_mom_cfg.get("lookback", 84),
            skip_recent=med_mom_cfg.get("skip_recent", 10),
            vol_window=med_mom_cfg.get("vol_window", 63),
            window=med_mom_cfg.get("window", 252),
            clip=med_mom_cfg.get("clip", 3.0)
        )
        
        canonical_med_cfg = self.config.get("canonical_medium_momentum", {})
        self.canonical_medium_momentum = CanonicalMediumMomentumFeatures(
            symbols=None,  # Will use market.universe
            lookback=canonical_med_cfg.get("lookback", 84),
            skip_recent=canonical_med_cfg.get("skip_recent", 10),
            vol_window=canonical_med_cfg.get("vol_window", 21),  # Canonical: 21-day vol
            window=canonical_med_cfg.get("window", 252),
            clip=canonical_med_cfg.get("clip", 3.0)
        )
        
        short_mom_cfg = self.config.get("short_momentum", {})
        self.short_momentum = ShortMomentumFeatures(
            symbols=None,  # Will use market.universe
            lookback=short_mom_cfg.get("lookback", 21),
            skip_recent=short_mom_cfg.get("skip_recent", 5),
            vol_window=short_mom_cfg.get("vol_window", 20),
            window=short_mom_cfg.get("window", 252),
            clip=short_mom_cfg.get("clip", 3.0)
        )
        
        residual_trend_cfg = self.config.get("residual_trend", {})
        self.residual_trend = ResidualTrendFeatures(
            symbols=None,  # Will use market.universe
            long_lookback=residual_trend_cfg.get("long_lookback", 252),
            short_lookback=residual_trend_cfg.get("short_lookback", 21),
            clip=residual_trend_cfg.get("clip", 3.0)
        )
        
        persistence_cfg = self.config.get("persistence", {})
        self.persistence = PersistenceFeatures(
            symbols=None,  # Will use market.universe
            acceleration_window=persistence_cfg.get("acceleration_window", 21),
            clip=persistence_cfg.get("clip", 3.0)
        )
        
        # Cache for computed features
        self._features_cache = {}
        self._cache_end_date = None
    
    def get_features(
        self,
        end_date: Optional[Union[str, datetime]] = None,
        feature_types: Optional[list] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Get all features up to end_date.
        
        Args:
            end_date: End date for feature computation (default: last available)
            feature_types: Optional list of feature types to compute
                          (default: all available)
        
        Returns:
            Dictionary of feature DataFrames keyed by feature type:
            - "SR3_CURVE": SR3 carry/curve features
            - "RATES_CURVE": Rates curve features (2s10s, 5s30s)
            - "CARRY_FX_COMMOD": FX/Commodity carry features
            - "LONG_MOMENTUM": Long-term momentum features (ret, breakout, slope)
            - "MEDIUM_MOMENTUM": Medium-term momentum features (ret, breakout, slope, persistence)
            - "CANONICAL_MEDIUM_MOMENTUM": Canonical medium-term features (84d ret, 84d breakout, EMA21-84 slope, composite)
            - "SHORT_MOMENTUM": Short-term momentum features (ret, breakout, slope, reversal)
            - "RESIDUAL_TREND": Residual trend features (raw and z-scored residual returns)
            - "PERSISTENCE": Persistence features (slope, breakout, return acceleration)
        """
        if feature_types is None:
            feature_types = ["SR3_CURVE", "RATES_CURVE", "CARRY_FX_COMMOD", "LONG_MOMENTUM", "MEDIUM_MOMENTUM", "CANONICAL_MEDIUM_MOMENTUM", "SHORT_MOMENTUM", "RESIDUAL_TREND", "PERSISTENCE"]
        
        features = {}
        
        # Compute SR3 curve features if requested
        if "SR3_CURVE" in feature_types:
            # Check cache
            cache_key = "SR3_CURVE"
            if (cache_key not in self._features_cache or 
                self._cache_end_date is None or 
                (end_date and pd.to_datetime(end_date) > self._cache_end_date)):
                # Compute features
                sr3_features = self.sr3_curve.compute(self.market, end_date=end_date)
                self._features_cache[cache_key] = sr3_features
                if end_date:
                    self._cache_end_date = pd.to_datetime(end_date)
            features["SR3_CURVE"] = self._features_cache[cache_key]
        
        # Compute rates curve features if requested
        if "RATES_CURVE" in feature_types:
            # Check cache
            cache_key = "RATES_CURVE"
            if (cache_key not in self._features_cache or 
                self._cache_end_date is None or 
                (end_date and pd.to_datetime(end_date) > self._cache_end_date)):
                # Compute features
                rates_features = self.rates_curve.compute(end_date=end_date)
                self._features_cache[cache_key] = rates_features
                if end_date:
                    self._cache_end_date = pd.to_datetime(end_date)
            features["RATES_CURVE"] = self._features_cache[cache_key]
        
        # Compute FX/Commodity carry features if requested
        if "CARRY_FX_COMMOD" in feature_types:
            # Check cache
            cache_key = "CARRY_FX_COMMOD"
            if (cache_key not in self._features_cache or 
                self._cache_end_date is None or 
                (end_date and pd.to_datetime(end_date) > self._cache_end_date)):
                # Compute features
                fx_commod_features = self.fx_commod_carry.compute(self.market, end_date=end_date)
                self._features_cache[cache_key] = fx_commod_features
                if end_date:
                    self._cache_end_date = pd.to_datetime(end_date)
            features["CARRY_FX_COMMOD"] = self._features_cache[cache_key]
        
        # Compute long-term momentum features if requested
        if "LONG_MOMENTUM" in feature_types:
            # Check cache
            cache_key = "LONG_MOMENTUM"
            if (cache_key not in self._features_cache or 
                self._cache_end_date is None or 
                (end_date and pd.to_datetime(end_date) > self._cache_end_date)):
                # Compute features
                long_mom_features = self.long_momentum.compute(self.market, end_date=end_date)
                self._features_cache[cache_key] = long_mom_features
                if end_date:
                    self._cache_end_date = pd.to_datetime(end_date)
            features["LONG_MOMENTUM"] = self._features_cache[cache_key]
        
        # Compute medium-term momentum features if requested
        if "MEDIUM_MOMENTUM" in feature_types:
            cache_key = "MEDIUM_MOMENTUM"
            if (cache_key not in self._features_cache or 
                self._cache_end_date is None or 
                (end_date and pd.to_datetime(end_date) > self._cache_end_date)):
                med_mom_features = self.medium_momentum.compute(self.market, end_date=end_date)
                self._features_cache[cache_key] = med_mom_features
                if end_date:
                    self._cache_end_date = pd.to_datetime(end_date)
            features["MEDIUM_MOMENTUM"] = self._features_cache[cache_key]
        
        # Compute canonical medium-term momentum features if requested
        if "CANONICAL_MEDIUM_MOMENTUM" in feature_types:
            cache_key = "CANONICAL_MEDIUM_MOMENTUM"
            if (cache_key not in self._features_cache or 
                self._cache_end_date is None or 
                (end_date and pd.to_datetime(end_date) > self._cache_end_date)):
                canonical_med_features = self.canonical_medium_momentum.compute(self.market, end_date=end_date)
                self._features_cache[cache_key] = canonical_med_features
                if end_date:
                    self._cache_end_date = pd.to_datetime(end_date)
            features["CANONICAL_MEDIUM_MOMENTUM"] = self._features_cache[cache_key]
        
        # Compute short-term momentum features if requested
        if "SHORT_MOMENTUM" in feature_types:
            cache_key = "SHORT_MOMENTUM"
            if (cache_key not in self._features_cache or 
                self._cache_end_date is None or 
                (end_date and pd.to_datetime(end_date) > self._cache_end_date)):
                short_mom_features = self.short_momentum.compute(self.market, end_date=end_date)
                self._features_cache[cache_key] = short_mom_features
                if end_date:
                    self._cache_end_date = pd.to_datetime(end_date)
            features["SHORT_MOMENTUM"] = self._features_cache[cache_key]
        
        # Compute residual trend features if requested
        if "RESIDUAL_TREND" in feature_types:
            cache_key = "RESIDUAL_TREND"
            if (cache_key not in self._features_cache or 
                self._cache_end_date is None or 
                (end_date and pd.to_datetime(end_date) > self._cache_end_date)):
                residual_trend_features = self.residual_trend.compute(self.market, end_date=end_date)
                self._features_cache[cache_key] = residual_trend_features
                if end_date:
                    self._cache_end_date = pd.to_datetime(end_date)
            features["RESIDUAL_TREND"] = self._features_cache[cache_key]
        
        # Compute persistence features if requested
        if "PERSISTENCE" in feature_types:
            cache_key = "PERSISTENCE"
            if (cache_key not in self._features_cache or 
                self._cache_end_date is None or 
                (end_date and pd.to_datetime(end_date) > self._cache_end_date)):
                persistence_features = self.persistence.compute(self.market, end_date=end_date)
                self._features_cache[cache_key] = persistence_features
                if end_date:
                    self._cache_end_date = pd.to_datetime(end_date)
            features["PERSISTENCE"] = self._features_cache[cache_key]
        
        return features
    
    def clear_cache(self):
        """Clear all cached features."""
        self._features_cache = {}
        self._cache_end_date = None
        logger.debug("[FeatureService] Cleared feature cache")

