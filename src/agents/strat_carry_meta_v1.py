"""
Carry Meta-Sleeve v1 (Layer 1 — Engine Signals)

Meta-sleeve that combines carry signals across all four asset classes:
1. Equity Carry: Implied dividend yield (ES, NQ, RTY)
2. FX Carry: Interest rate differentials (6E, 6B, 6J)
3. Rates Carry: Rolldown / curve slope (ZT, ZF, ZN, UB)
4. Commodity Carry: Backwardation / contango (CL, GC)

Economic Theme: Intertemporal price of capital and storage
Role in Portfolio: Orthogonal structural return source
Always-On: Yes
Unconditional: Yes
State-Aware: No

Phase-0 Contract (as specified):
- Sign-only signals (no z-scoring, no normalization)
- Equal-weight per asset
- Daily rebalance
- No gating
- No cross-sectional ranking
- No volatility normalization

Architectural Compliance:
- Carry is a canonical unconditional economic return source
- Equal footing with Trend, VRP, Curve RV
- Expresses continuous pricing pressure embedded in forward curves and funding markets
- Fully admissible as an Engine (Meta-Sleeve) per SYSTEM_CONSTRUCTION.md
"""

import logging
from typing import Optional, Union, Dict, Sequence
from datetime import datetime
import pandas as pd
import numpy as np

from .feature_equity_carry import EquityCarryFeatures
from .feature_carry_fx_commod import FxCommodCarryFeatures
from .feature_rates_carry import RatesCarryFeatures

logger = logging.getLogger(__name__)


class CarryMetaV1:
    """
    Carry Meta-Sleeve v1: Unified carry signals across asset classes.
    
    Combines carry from:
    - Equities: Implied dividend yield (ES, NQ, RTY)
    - FX: Interest rate differentials (6E, 6B, 6J)
    - Rates: Rolldown / curve slope (ZT, ZF, ZN, UB)
    - Commodities: Backwardation / contango (CL, GC)
    
    Phase-0 Implementation:
    - Sign-only: sign(raw_carry)
    - Equal-weight across all assets
    - No z-scoring (Phase-1)
    - No vol normalization (Phase-1)
    - No cross-sectional ranking (Phase-1)
    """
    
    # Database symbol mapping
    SYMBOL_MAP = {
        # Equities
        "ES": "ES_FRONT_CALENDAR_2D",
        "NQ": "NQ_FRONT_CALENDAR_2D",
        "RTY": "RTY_FRONT_CALENDAR_2D",
        # FX
        "6E": "6E_FRONT_CALENDAR",
        "6B": "6B_FRONT_CALENDAR",
        "6J": "6J_FRONT_CALENDAR",
        # Rates
        "ZT": "ZT_FRONT_VOLUME",
        "ZF": "ZF_FRONT_VOLUME",
        "ZN": "ZN_FRONT_VOLUME",
        "UB": "UB_FRONT_VOLUME",
        # Commodities
        "CL": "CL_FRONT_VOLUME",
        "GC": "GC_FRONT_VOLUME"
    }
    
    def __init__(
        self,
        enabled_asset_classes: Optional[Sequence[str]] = None,
        phase: int = 0,
        equity_symbols: Optional[Sequence[str]] = None,
        fx_symbols: Optional[Sequence[str]] = None,
        rates_symbols: Optional[Sequence[str]] = None,
        commodity_symbols: Optional[Sequence[str]] = None,
        window: int = 252,
        clip: float = 3.0
    ):
        """
        Initialize Carry Meta-Sleeve v1.
        
        Args:
            enabled_asset_classes: List of enabled asset classes
                                  (default: ["equity", "fx", "rates", "commodity"])
            phase: Sleeve lifecycle phase (0, 1, or 2)
                   - 0: Sign-only, equal-weight
                   - 1: Add z-scoring, vol normalization, cross-sectional ranking
                   - 2: Add overlays, policy integration
            equity_symbols: Equity futures to include (default: ["ES", "NQ", "RTY"])
            fx_symbols: FX futures to include (default: ["6E", "6B", "6J"])
            rates_symbols: Rates futures to include (default: ["ZT", "ZF", "ZN", "UB"])
            commodity_symbols: Commodity futures to include (default: ["CL", "GC"])
            window: Rolling window for z-score (Phase-1+, default: 252)
            clip: Z-score clipping bounds (Phase-1+, default: 3.0)
        """
        self.enabled_asset_classes = enabled_asset_classes or ["equity", "fx", "rates", "commodity"]
        self.phase = phase
        self.window = window
        self.clip = clip
        
        # Asset symbols per class
        self.equity_symbols = list(equity_symbols or ["ES", "NQ", "RTY"])
        self.fx_symbols = list(fx_symbols or ["6E", "6B", "6J"])
        self.rates_symbols = list(rates_symbols or ["ZT", "ZF", "ZN", "UB"])
        self.commodity_symbols = list(commodity_symbols or ["CL", "GC"])
        
        # Initialize feature calculators
        self.feature_calcs = {}
        
        if "equity" in self.enabled_asset_classes:
            self.feature_calcs["equity"] = EquityCarryFeatures(
                futures_symbols=self.equity_symbols,
                window=self.window,
                clip=self.clip
            )
        
        if "fx" in self.enabled_asset_classes or "commodity" in self.enabled_asset_classes:
            # FxCommodCarryFeatures handles both FX and commodities
            roots = []
            if "fx" in self.enabled_asset_classes:
                roots.extend(self.fx_symbols)
            if "commodity" in self.enabled_asset_classes:
                roots.extend(self.commodity_symbols)
            
            self.feature_calcs["fx_commod"] = FxCommodCarryFeatures(
                roots=roots,
                window=self.window,
                clip=self.clip
            )
        
        if "rates" in self.enabled_asset_classes:
            self.feature_calcs["rates"] = RatesCarryFeatures(
                symbols=self.rates_symbols,
                window=self.window,
                clip=self.clip
            )
        
        # Cache for computed features
        self._features_cache = None
        self._cache_end_date = None
        
        logger.info(
            f"[CarryMetaV1] Initialized Phase-{self.phase} with "
            f"asset_classes={self.enabled_asset_classes}, "
            f"equity={self.equity_symbols}, fx={self.fx_symbols}, "
            f"rates={self.rates_symbols}, commodity={self.commodity_symbols}"
        )
    
    def signals(
        self,
        market,
        date: Union[str, datetime],
        features: Optional[pd.DataFrame] = None
    ) -> pd.Series:
        """
        Generate carry signals for all enabled asset classes at given date.
        
        Phase-0 Contract:
        - Returns sign(raw_carry) for each asset
        - Equal-weight (no normalization)
        - Values in {-1, 0, 1}
        
        Phase-1 Contract:
        - Returns z-scored carry signals
        - Cross-sectional ranking
        - Clipped to ±3.0
        
        Args:
            market: MarketData instance
            date: Current rebalance date
            features: Optional pre-computed features DataFrame
            
        Returns:
            Series with signals for all enabled assets
        """
        date_dt = pd.to_datetime(date)
        
        # Get or compute features
        if features is None:
            # Check cache
            if (self._features_cache is None or 
                self._cache_end_date is None or 
                date_dt > self._cache_end_date):
                # Compute features up to current date
                self._features_cache = self._compute_all_features(market, end_date=date_dt)
                self._cache_end_date = date_dt
            features = self._features_cache
        
        if features.empty:
            logger.warning(f"[CarryMetaV1] No features available for date {date_dt}")
            # Return zero signals for all assets
            return self._zero_signals()
        
        # Find the closest available date (forward-fill from previous available date)
        if date_dt not in features.index:
            available_dates = features.index[features.index <= date_dt]
            if len(available_dates) == 0:
                logger.warning(
                    f"[CarryMetaV1] No features available for date {date_dt} (no prior data)"
                )
                return self._zero_signals()
            use_date = available_dates[-1]
            logger.debug(
                f"[CarryMetaV1] Using features from {use_date} for date {date_dt}"
            )
        else:
            use_date = date_dt
        
        # Get feature row for this date
        f = features.loc[use_date]
        
        # Generate signals based on phase
        if self.phase == 0:
            return self._signals_phase0(f)
        elif self.phase == 1:
            # Phase-1 needs full features DataFrame for rolling z-score
            return self._signals_phase1(features, market, date_dt)
        elif self.phase == 1.1:
            # Phase-1.1: Asset-class risk parity
            return self._signals_phase1_1(features, market, date_dt)
        else:
            # Phase 2+ not yet implemented
            logger.warning(f"[CarryMetaV1] Phase {self.phase} not yet implemented, using Phase-1")
            return self._signals_phase1(features, market, date_dt)
    
    def _compute_all_features(
        self,
        market,
        end_date: Optional[Union[str, datetime]] = None
    ) -> pd.DataFrame:
        """
        Compute features from all enabled asset classes.
        
        Args:
            market: MarketData instance
            end_date: End date for feature computation
            
        Returns:
            Combined DataFrame with all features
        """
        all_features = []
        
        for asset_class, calc in self.feature_calcs.items():
            try:
                features = calc.compute(market, end_date=end_date)
                if not features.empty:
                    all_features.append(features)
            except Exception as e:
                logger.warning(
                    f"[CarryMetaV1] Error computing {asset_class} features: {e}"
                )
        
        if not all_features:
            logger.warning("[CarryMetaV1] No features computed from any asset class")
            return pd.DataFrame()
        
        # Concatenate all features
        combined = pd.concat(all_features, axis=1)
        combined = combined.sort_index()
        
        # Canonical NA handling: drop rows with ANY NaN (PROCEDURES.md requirement)
        rows_before = len(combined)
        combined = combined.dropna(how="any")
        rows_dropped = rows_before - len(combined)
        if rows_dropped > 0:
            logger.debug(
                f"[CarryMetaV1] Dropped {rows_dropped} rows with NaN after feature concatenation "
                f"(before: {rows_before}, after: {len(combined)})"
            )
        
        return combined
    
    def _signals_phase0(self, features_row: pd.Series) -> pd.Series:
        """
        Phase-0: Sign-only signals, equal-weight.
        
        For each asset:
        - Equity: sign(equity_carry_raw_{symbol})
        - FX: sign(carry_ts_z_{root}) — use time-series feature from FxCommodCarry
        - Rates: sign(rates_carry_raw_{symbol})
        - Commodity: sign(carry_ts_z_{root}) — use time-series feature from FxCommodCarry
        
        Args:
            features_row: Features for a single date
            
        Returns:
            Series with sign-only signals in {-1, 0, 1}
        """
        signals = {}
        
        # Equity carry
        if "equity" in self.enabled_asset_classes:
            for sym in self.equity_symbols:
                db_sym = self.SYMBOL_MAP[sym]
                feature_name = f"equity_carry_raw_{sym}"
                
                if feature_name in features_row.index:
                    raw_val = features_row[feature_name]
                    if pd.notna(raw_val):
                        signals[db_sym] = np.sign(raw_val)
                    else:
                        signals[db_sym] = 0.0
                else:
                    signals[db_sym] = 0.0
        
        # FX carry
        if "fx" in self.enabled_asset_classes:
            for sym in self.fx_symbols:
                db_sym = self.SYMBOL_MAP[sym]
                # FX carry uses time-series feature from FxCommodCarryFeatures
                feature_name = f"carry_ts_z_{sym}"
                
                if feature_name in features_row.index:
                    raw_val = features_row[feature_name]
                    if pd.notna(raw_val):
                        # For Phase-0, we want sign of the RAW carry, not the z-scored value
                        # But FxCommodCarryFeatures only returns z-scored values
                        # So we use sign of the z-scored value as a proxy
                        signals[db_sym] = np.sign(raw_val)
                    else:
                        signals[db_sym] = 0.0
                else:
                    signals[db_sym] = 0.0
        
        # Rates carry
        if "rates" in self.enabled_asset_classes:
            for sym in self.rates_symbols:
                db_sym = self.SYMBOL_MAP[sym]
                feature_name = f"rates_carry_raw_{sym}"
                
                if feature_name in features_row.index:
                    raw_val = features_row[feature_name]
                    if pd.notna(raw_val):
                        signals[db_sym] = np.sign(raw_val)
                    else:
                        signals[db_sym] = 0.0
                else:
                    signals[db_sym] = 0.0
        
        # Commodity carry
        if "commodity" in self.enabled_asset_classes:
            for sym in self.commodity_symbols:
                db_sym = self.SYMBOL_MAP[sym]
                # Commodity carry uses time-series feature from FxCommodCarryFeatures
                feature_name = f"carry_ts_z_{sym}"
                
                if feature_name in features_row.index:
                    raw_val = features_row[feature_name]
                    if pd.notna(raw_val):
                        # For Phase-0, sign of z-scored value
                        signals[db_sym] = np.sign(raw_val)
                    else:
                        signals[db_sym] = 0.0
                else:
                    signals[db_sym] = 0.0
        
        result = pd.Series(signals)
        
        logger.debug(
            f"[CarryMetaV1 Phase-0] Generated signals: "
            f"mean={result.mean():.3f}, "
            f"non-zero={(result != 0).sum()}/{len(result)}"
        )
        
        return result
    
    def _signals_phase1(
        self,
        features: pd.DataFrame,
        market,
        date: pd.Timestamp
    ) -> pd.Series:
        """
        Phase-1: Z-scored, vol-normalized signals with clipping.
        
        Step B: Rolling z-score per instrument (252d window)
        Step C: Clip at ±3.0
        Step D: Vol-normalize within sleeve (equal risk per asset)
        Step E: Optional cross-sectional ranking within asset class
        
        Args:
            features: Full features DataFrame (with history for rolling z-score)
            market: MarketData instance (for vol normalization)
            date: Current rebalance date
            
        Returns:
            Series with z-scored, clipped, vol-normalized signals
        """
        # Find the closest available date (forward-fill)
        if date not in features.index:
            available_dates = features.index[features.index <= date]
            if len(available_dates) == 0:
                return self._zero_signals()
            use_date = available_dates[-1]
        else:
            use_date = date
        
        # Get features up to current date (for rolling calculations)
        features_history = features.loc[features.index <= use_date]
        
        if features_history.empty:
            return self._zero_signals()
        
        signals = {}
        
        # Step B & C: Compute rolling z-score and clip for each asset
        carry_z_dict = {}
        
        # Equity carry
        if "equity" in self.enabled_asset_classes:
            for sym in self.equity_symbols:
                db_sym = self.SYMBOL_MAP[sym]
                feature_name = f"equity_carry_raw_{sym}"
                
                if feature_name in features_history.columns:
                    carry_raw = features_history[feature_name].dropna()
                    if len(carry_raw) >= self.window // 2:  # Minimum periods
                        # Rolling z-score (252d window)
                        mu = carry_raw.rolling(window=self.window, min_periods=self.window // 2).mean()
                        sigma = carry_raw.rolling(window=self.window, min_periods=self.window // 2).std()
                        carry_z = (carry_raw - mu) / sigma.replace(0.0, np.nan)
                        # Step C: Clip at ±3.0
                        carry_z_clipped = carry_z.clip(-self.clip, self.clip)
                        # Get value at use_date
                        if use_date in carry_z_clipped.index:
                            carry_z_dict[db_sym] = carry_z_clipped.loc[use_date]
                        else:
                            carry_z_dict[db_sym] = 0.0
                    else:
                        carry_z_dict[db_sym] = 0.0
                else:
                    carry_z_dict[db_sym] = 0.0
        
        # FX carry (FxCommodCarryFeatures provides carry_ts_z which is already z-scored)
        # For Phase-1, we re-z-score on top of the raw roll yield if available
        # Otherwise use the pre-computed z-score
        if "fx" in self.enabled_asset_classes:
            for sym in self.fx_symbols:
                db_sym = self.SYMBOL_MAP[sym]
                # FxCommodCarryFeatures provides carry_ts_z (already z-scored)
                # For Phase-1, we'll use this directly (it's already standardized)
                feature_name_z = f"carry_ts_z_{sym}"
                
                if feature_name_z in features_history.columns:
                    carry_z_series = features_history[feature_name_z].dropna()
                    if use_date in carry_z_series.index:
                        # Already z-scored and clipped, use directly
                        carry_z_dict[db_sym] = carry_z_series.loc[use_date]
                    else:
                        carry_z_dict[db_sym] = 0.0
                else:
                    carry_z_dict[db_sym] = 0.0
        
        # Rates carry
        if "rates" in self.enabled_asset_classes:
            for sym in self.rates_symbols:
                db_sym = self.SYMBOL_MAP[sym]
                feature_name = f"rates_carry_raw_{sym}"
                
                if feature_name in features_history.columns:
                    carry_raw = features_history[feature_name].dropna()
                    if len(carry_raw) >= self.window // 2:
                        mu = carry_raw.rolling(window=self.window, min_periods=self.window // 2).mean()
                        sigma = carry_raw.rolling(window=self.window, min_periods=self.window // 2).std()
                        carry_z = (carry_raw - mu) / sigma.replace(0.0, np.nan)
                        carry_z_clipped = carry_z.clip(-self.clip, self.clip)
                        if use_date in carry_z_clipped.index:
                            carry_z_dict[db_sym] = carry_z_clipped.loc[use_date]
                        else:
                            carry_z_dict[db_sym] = 0.0
                    else:
                        carry_z_dict[db_sym] = 0.0
                else:
                    carry_z_dict[db_sym] = 0.0
        
        # Commodity carry (similar to FX - uses carry_ts_z from FxCommodCarryFeatures)
        if "commodity" in self.enabled_asset_classes:
            for sym in self.commodity_symbols:
                db_sym = self.SYMBOL_MAP[sym]
                feature_name_z = f"carry_ts_z_{sym}"
                
                if feature_name_z in features_history.columns:
                    carry_z_series = features_history[feature_name_z].dropna()
                    if use_date in carry_z_series.index:
                        # Already z-scored and clipped, use directly
                        carry_z_dict[db_sym] = carry_z_series.loc[use_date]
                    else:
                        carry_z_dict[db_sym] = 0.0
                else:
                    carry_z_dict[db_sym] = 0.0
        
        # Step E: Optional cross-sectional ranking within asset classes
        # For Phase-1, we'll skip this initially (can add later)
        # If enabled, rank within: equity vs equity, rates vs rates, etc.
        
        # Step D: Vol-normalize within sleeve (equal risk per asset)
        # Get asset returns for vol calculation
        try:
            # Get returns up to current date (252d lookback for vol)
            returns_start = (use_date - pd.Timedelta(days=365)).strftime('%Y-%m-%d')
            returns_df = market.get_returns(
                symbols=list(carry_z_dict.keys()),
                start=returns_start,
                end=use_date.strftime('%Y-%m-%d'),
                method="log"
            )
            
            # Compute rolling volatility (252d annualized)
            vol_dict = {}
            vol_history_dict = {}  # Store full vol history for floor calculation
            
            for sym in carry_z_dict.keys():
                if sym in returns_df.columns:
                    ret_series = returns_df[sym].dropna()
                    if len(ret_series) >= 63:  # Minimum for vol estimate
                        vol = ret_series.rolling(window=252, min_periods=63).std() * np.sqrt(252)
                        vol_history_dict[sym] = vol  # Store for floor calculation
                        
                        if use_date in vol.index and pd.notna(vol.loc[use_date]):
                            vol_dict[sym] = vol.loc[use_date]
                        else:
                            # Use most recent available vol
                            vol_dict[sym] = vol.dropna().iloc[-1] if len(vol.dropna()) > 0 else 0.15
                    else:
                        vol_dict[sym] = 0.15  # Default to 15% if insufficient data
                        vol_history_dict[sym] = pd.Series([0.15], index=[use_date])
                else:
                    vol_dict[sym] = 0.15  # Default to 15% if symbol not found
                    vol_history_dict[sym] = pd.Series([0.15], index=[use_date])
            
            # Apply vol floor: prevent low-vol assets (e.g., rates) from dominating
            # Floor = 5th percentile of asset's vol history, or 4% minimum
            vol_floor_global = 0.04  # 4% annualized minimum
            for sym in vol_dict.keys():
                vol_hist = vol_history_dict.get(sym, pd.Series([0.15]))
                vol_hist_clean = vol_hist.dropna()
                if len(vol_hist_clean) > 0:
                    vol_floor_asset = max(vol_hist_clean.quantile(0.05), vol_floor_global)
                    vol_dict[sym] = max(vol_dict[sym], vol_floor_asset)
                else:
                    vol_dict[sym] = max(vol_dict[sym], vol_floor_global)
            
            # Vol-normalize: target unit risk per asset
            # signal_normalized = signal_z / asset_vol
            # This makes each asset contribute equal risk to the sleeve
            # Units: z-score (std devs of carry) / vol (annualized) = carry strength per unit realized vol
            for sym in carry_z_dict.keys():
                vol = vol_dict.get(sym, 0.15)
                if vol > 1e-6:  # Avoid division by zero
                    # Normalize: z-score / vol gives risk-adjusted signal
                    # For unit risk, we want signal * vol = constant
                    # So signal = z / vol (this makes high vol assets have smaller signals)
                    signals[sym] = carry_z_dict[sym] / vol
                else:
                    signals[sym] = carry_z_dict[sym]
            
            # Scale all signals to target unit gross exposure
            # This prevents the sleeve from having excessive leverage
            total_abs = sum(abs(s) for s in signals.values())
            if total_abs > 1e-6:
                # Scale to unit gross (sum of absolute values = 1.0)
                scale_factor = 1.0 / total_abs
                signals = {k: v * scale_factor for k, v in signals.items()}
        
        except Exception as e:
            logger.warning(
                f"[CarryMetaV1 Phase-1] Vol normalization failed: {e}. "
                f"Using z-scored signals without vol normalization."
            )
            # Fallback: use z-scored signals without vol normalization
            signals = carry_z_dict.copy()
        
        result = pd.Series(signals)
        
        # Replace NaN with 0
        result = result.fillna(0.0)
        
        logger.debug(
            f"[CarryMetaV1 Phase-1] Generated signals: "
            f"mean={result.mean():.3f}, std={result.std():.3f}, "
            f"non-zero={(result != 0).sum()}/{len(result)}"
        )
        
        return result
    
    def _signals_phase1_1(
        self,
        features: pd.DataFrame,
        market,
        date: pd.Timestamp
    ) -> pd.Series:
        """
        Phase-1.1: Asset-class risk parity (25% per class) with mild vol scaling within class.
        
        Changes from Phase-1:
        - Fixed class weights: 25% each (equity, fx, rates, commodity)
        - Within each class: allocate equal gross (or mild vol scaling)
        - Prevents rates from dominating due to low vol
        
        Args:
            features: Full features DataFrame (with history for rolling z-score)
            market: MarketData instance (for vol normalization within class)
            date: Current rebalance date
            
        Returns:
            Series with asset-class risk parity signals
        """
        # Find the closest available date (forward-fill)
        if date not in features.index:
            available_dates = features.index[features.index <= date]
            if len(available_dates) == 0:
                return self._zero_signals()
            use_date = available_dates[-1]
        else:
            use_date = date
        
        # Get features up to current date (for rolling calculations)
        features_history = features.loc[features.index <= use_date]
        
        if features_history.empty:
            return self._zero_signals()
        
        signals = {}
        
        # Step B & C: Compute rolling z-score and clip for each asset (same as Phase-1)
        carry_z_dict = {}
        
        # Equity carry
        if "equity" in self.enabled_asset_classes:
            for sym in self.equity_symbols:
                db_sym = self.SYMBOL_MAP[sym]
                feature_name = f"equity_carry_raw_{sym}"
                
                if feature_name in features_history.columns:
                    carry_raw = features_history[feature_name].dropna()
                    if len(carry_raw) >= self.window // 2:
                        mu = carry_raw.rolling(window=self.window, min_periods=self.window // 2).mean()
                        sigma = carry_raw.rolling(window=self.window, min_periods=self.window // 2).std()
                        carry_z = (carry_raw - mu) / sigma.replace(0.0, np.nan)
                        carry_z_clipped = carry_z.clip(-self.clip, self.clip)
                        if use_date in carry_z_clipped.index:
                            carry_z_dict[db_sym] = carry_z_clipped.loc[use_date]
                        else:
                            carry_z_dict[db_sym] = 0.0
                    else:
                        carry_z_dict[db_sym] = 0.0
                else:
                    carry_z_dict[db_sym] = 0.0
        
        # FX carry (use pre-computed z-score)
        if "fx" in self.enabled_asset_classes:
            for sym in self.fx_symbols:
                db_sym = self.SYMBOL_MAP[sym]
                feature_name_z = f"carry_ts_z_{sym}"
                
                if feature_name_z in features_history.columns:
                    carry_z_series = features_history[feature_name_z].dropna()
                    if use_date in carry_z_series.index:
                        carry_z_dict[db_sym] = carry_z_series.loc[use_date]
                    else:
                        carry_z_dict[db_sym] = 0.0
                else:
                    carry_z_dict[db_sym] = 0.0
        
        # Rates carry
        if "rates" in self.enabled_asset_classes:
            for sym in self.rates_symbols:
                db_sym = self.SYMBOL_MAP[sym]
                feature_name = f"rates_carry_raw_{sym}"
                
                if feature_name in features_history.columns:
                    carry_raw = features_history[feature_name].dropna()
                    if len(carry_raw) >= self.window // 2:
                        mu = carry_raw.rolling(window=self.window, min_periods=self.window // 2).mean()
                        sigma = carry_raw.rolling(window=self.window, min_periods=self.window // 2).std()
                        carry_z = (carry_raw - mu) / sigma.replace(0.0, np.nan)
                        carry_z_clipped = carry_z.clip(-self.clip, self.clip)
                        if use_date in carry_z_clipped.index:
                            carry_z_dict[db_sym] = carry_z_clipped.loc[use_date]
                        else:
                            carry_z_dict[db_sym] = 0.0
                    else:
                        carry_z_dict[db_sym] = 0.0
                else:
                    carry_z_dict[db_sym] = 0.0
        
        # Commodity carry (use pre-computed z-score)
        if "commodity" in self.enabled_asset_classes:
            for sym in self.commodity_symbols:
                db_sym = self.SYMBOL_MAP[sym]
                feature_name_z = f"carry_ts_z_{sym}"
                
                if feature_name_z in features_history.columns:
                    carry_z_series = features_history[feature_name_z].dropna()
                    if use_date in carry_z_series.index:
                        carry_z_dict[db_sym] = carry_z_series.loc[use_date]
                    else:
                        carry_z_dict[db_sym] = 0.0
                else:
                    carry_z_dict[db_sym] = 0.0
        
        # Phase-1.1: Asset-class risk parity
        # Step 1: Group assets by class
        asset_class_groups = {
            "equity": [self.SYMBOL_MAP[s] for s in self.equity_symbols if self.SYMBOL_MAP[s] in carry_z_dict],
            "fx": [self.SYMBOL_MAP[s] for s in self.fx_symbols if self.SYMBOL_MAP[s] in carry_z_dict],
            "rates": [self.SYMBOL_MAP[s] for s in self.rates_symbols if self.SYMBOL_MAP[s] in carry_z_dict],
            "commodity": [self.SYMBOL_MAP[s] for s in self.commodity_symbols if self.SYMBOL_MAP[s] in carry_z_dict]
        }
        
        # Step 2: Fixed class weights (25% each)
        class_weights = {
            "equity": 0.25,
            "fx": 0.25,
            "rates": 0.25,
            "commodity": 0.25
        }
        
        # Step 3: Get vol for mild scaling within class (optional)
        vol_dict = {}
        try:
            returns_start = (use_date - pd.Timedelta(days=365)).strftime('%Y-%m-%d')
            returns_df = market.get_returns(
                symbols=list(carry_z_dict.keys()),
                start=returns_start,
                end=use_date.strftime('%Y-%m-%d'),
                method="log"
            )
            
            for sym in carry_z_dict.keys():
                if sym in returns_df.columns:
                    ret_series = returns_df[sym].dropna()
                    if len(ret_series) >= 63:
                        vol = ret_series.rolling(window=252, min_periods=63).std() * np.sqrt(252)
                        if use_date in vol.index and pd.notna(vol.loc[use_date]):
                            vol_dict[sym] = vol.loc[use_date]
                        else:
                            vol_dict[sym] = vol.dropna().iloc[-1] if len(vol.dropna()) > 0 else 0.15
                    else:
                        vol_dict[sym] = 0.15
                else:
                    vol_dict[sym] = 0.15
        except:
            pass  # If vol calculation fails, skip mild scaling
        
        # Step 4: Within each class, allocate equal gross (with optional mild vol scaling)
        for class_name, class_symbols in asset_class_groups.items():
            if len(class_symbols) == 0:
                continue
            
            class_gross = class_weights.get(class_name, 0.25)
            
            # Get z_clip values for this class
            class_z_clip = {sym: carry_z_dict[sym] for sym in class_symbols}
            
            # Optional: Mild vol scaling within class (prevents ZT from crushing ES)
            class_signals = {}
            for sym in class_symbols:
                vol = vol_dict.get(sym, 0.15)
                # Mild scaling: use sqrt(vol) instead of vol to reduce impact
                vol_scale = np.sqrt(vol / 0.15)  # Normalize to 15% baseline
                class_signals[sym] = class_z_clip[sym] / max(vol_scale, 0.5)  # Floor at 0.5x
            
            # Normalize within class to sum(abs) = class_gross
            class_total_abs = sum(abs(s) for s in class_signals.values())
            if class_total_abs > 1e-6:
                class_scale = class_gross / class_total_abs
                for sym in class_symbols:
                    signals[sym] = class_signals[sym] * class_scale
            else:
                # If all signals are zero, assign equal weight
                for sym in class_symbols:
                    signals[sym] = class_gross / len(class_symbols) if class_z_clip[sym] >= 0 else -class_gross / len(class_symbols)
        
        result = pd.Series(signals)
        result = result.fillna(0.0)
        
        logger.debug(
            f"[CarryMetaV1 Phase-1.1] Generated signals: "
            f"mean={result.mean():.3f}, std={result.std():.3f}, "
            f"non-zero={(result != 0).sum()}/{len(result)}"
        )
        
        return result
    
    def _zero_signals(self) -> pd.Series:
        """Generate zero signals for all enabled assets."""
        signals = {}
        
        if "equity" in self.enabled_asset_classes:
            for sym in self.equity_symbols:
                signals[self.SYMBOL_MAP[sym]] = 0.0
        
        if "fx" in self.enabled_asset_classes:
            for sym in self.fx_symbols:
                signals[self.SYMBOL_MAP[sym]] = 0.0
        
        if "rates" in self.enabled_asset_classes:
            for sym in self.rates_symbols:
                signals[self.SYMBOL_MAP[sym]] = 0.0
        
        if "commodity" in self.enabled_asset_classes:
            for sym in self.commodity_symbols:
                signals[self.SYMBOL_MAP[sym]] = 0.0
        
        return pd.Series(signals)
    
    def warmup_periods(self) -> int:
        """
        Return number of trading days required for warmup.
        
        For Phase-0, warmup is determined by the longest feature lookback.
        Default: 252 days (1 year) for rolling z-score windows.
        """
        return self.window
    
    def clear_cache(self):
        """Clear cached features."""
        self._features_cache = None
        self._cache_end_date = None
