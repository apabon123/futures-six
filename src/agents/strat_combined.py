"""
CombinedStrategy: Wrapper that combines multiple strategy sleeves.

Combines signals from multiple strategies (TSMOM, SR3 carry/curve, etc.)
with configurable weights before passing to overlays.
"""

import logging
from typing import Dict, Optional, Union, List, Tuple, Set, FrozenSet
from datetime import datetime
import pandas as pd

logger = logging.getLogger(__name__)

# Sleeve-scoped instrument eligibility: VRP sleeves may ONLY trade these symbols.
# This is enforced at signal combination time to prevent unintentional leakage.
VRP_ALLOWED_SYMBOLS: FrozenSet[str] = frozenset({'VX1', 'VX2', 'VX3'})

# Sleeve names that belong to the VRP engine
VRP_SLEEVE_NAMES: FrozenSet[str] = frozenset({
    'vrp_core_meta', 'vrp_convergence_meta', 'vrp_alt_meta'
})


class CombinedStrategy:
    """
    Combines multiple strategy sleeves into a single signal.
    
    Each strategy generates signals independently, then signals are
    combined with configurable weights before being passed to overlays.
    """
    
    def __init__(
        self,
        strategies: Dict,
        weights: Dict[str, float],
        features: Optional[Dict] = None,
        feature_service: Optional[object] = None,
        strict_universe: bool = False
    ):
        """
        Initialize CombinedStrategy.
        
        Args:
            strategies: Dictionary of strategy instances keyed by name
            weights: Dictionary of weights for each strategy (must sum to 1.0)
            features: Optional dictionary of pre-computed features
            feature_service: Optional FeatureService instance for on-demand feature computation
            strict_universe: If True, raise error when any sleeve emits symbols not in master universe
        """
        self.strategies = strategies
        self.weights = weights
        self.features = features or {}
        self.feature_service = feature_service
        self.strict_universe = strict_universe
        
        # Validate weights sum to 1.0
        total_weight = sum(weights.values())
        if abs(total_weight - 1.0) > 1e-6:
            logger.warning(
                f"[CombinedStrategy] Weights sum to {total_weight:.6f}, normalizing to 1.0"
            )
            self.weights = {k: v / total_weight for k, v in weights.items()}
        
        # Cache for last signals
        self._last_signals = None
        self._last_date = None
    
    def signals(
        self,
        market,
        date: Union[str, datetime],
        return_components: bool = False
    ) -> Union[pd.Series, Tuple[pd.Series, Dict[str, pd.Series]]]:
        """
        Generate combined signals from all enabled strategies.
        
        Args:
            market: MarketData instance
            date: Current rebalance date
            return_components: If True, return (blended, components_dict) instead of just blended
            
        Returns:
            If return_components=False: Combined signals as pd.Series
            If return_components=True: Tuple of (blended_signals, components_dict)
                where components_dict maps sleeve names to their weighted signal Series
        """
        date_dt = pd.to_datetime(date)
        
        # Update features if feature_service is available
        if self.feature_service:
            # Get features up to current date (point-in-time)
            updated_features = self.feature_service.get_features(end_date=date_dt)
            # Merge with existing features
            for key, value in updated_features.items():
                self.features[key] = value
        
        # Collect signals from all strategies
        all_signals = {}
        components = {}  # Track per-sleeve weighted signals if return_components=True
        
        for strategy_name, strategy in self.strategies.items():
            weight = self.weights.get(strategy_name, 0.0)
            
            if weight == 0.0:
                continue  # Skip disabled strategies
            
            try:
                # Generate signals (some strategies may need features)
                if hasattr(strategy, 'signals'):
                    # Check if strategy needs features
                    import inspect
                    sig = inspect.signature(strategy.signals)
                    if 'features' in sig.parameters:
                        # Pass features if available (SR3 uses SR3_CURVE, rates uses RATES_CURVE, tsmom uses LONG_MOMENTUM)
                        if strategy_name == "sr3_carry_curve":
                            features_dict = self.features.get("SR3_CURVE")
                            strategy_sigs = strategy.signals(market, date_dt, features=features_dict)
                        elif strategy_name == "rates_curve":
                            features_dict = self.features.get("RATES_CURVE")
                            strategy_sigs = strategy.signals(features_dict, date_dt)
                        elif strategy_name == "fx_commod_carry":
                            features_dict = self.features.get("CARRY_FX_COMMOD")
                            strategy_sigs = strategy.signals(market, date_dt, features=features_dict)
                        elif strategy_name == "tsmom":
                            features_dict = self.features.get("LONG_MOMENTUM")
                            strategy_sigs = strategy.signals(market, date_dt, features=features_dict)
                        elif strategy_name == "tsmom_med" or strategy_name == "medium_momentum":
                            features_dict = self.features.get("MEDIUM_MOMENTUM")
                            strategy_sigs = strategy.signals(market, date_dt, features=features_dict)
                        elif strategy_name == "tsmom_med_canonical" or strategy_name == "canonical_medium_momentum":
                            features_dict = self.features.get("CANONICAL_MEDIUM_MOMENTUM")
                            strategy_sigs = strategy.signals(market, date_dt, features=features_dict)
                        elif strategy_name == "tsmom_short" or strategy_name == "short_momentum":
                            features_dict = self.features.get("SHORT_MOMENTUM")
                            strategy_sigs = strategy.signals(market, date_dt, features=features_dict)
                        elif strategy_name == "tsmom_long" or strategy_name == "long_momentum":
                            features_dict = self.features.get("LONG_MOMENTUM")
                            strategy_sigs = strategy.signals(market, date_dt, features=features_dict)
                        elif strategy_name == "tsmom_multihorizon":
                            # Multi-horizon needs all three momentum feature types + residual trend (4th atomic sleeve)
                            # + canonical medium-term (for medium_variant: "canonical")
                            features_dict = {
                                "LONG_MOMENTUM": self.features.get("LONG_MOMENTUM", pd.DataFrame()),
                                "MEDIUM_MOMENTUM": self.features.get("MEDIUM_MOMENTUM", pd.DataFrame()),
                                "CANONICAL_MEDIUM_MOMENTUM": self.features.get("CANONICAL_MEDIUM_MOMENTUM", pd.DataFrame()),
                                "SHORT_MOMENTUM": self.features.get("SHORT_MOMENTUM", pd.DataFrame()),
                                "RESIDUAL_TREND": self.features.get("RESIDUAL_TREND", pd.DataFrame())
                            }
                            strategy_sigs = strategy.signals(market, date_dt, features=features_dict)
                        elif strategy_name == "csmom_meta" or strategy_name == "cross_sectional":
                            # CSMOM doesn't need features, just market and date
                            strategy_sigs = strategy.signals(market, date_dt)
                        elif strategy_name == "residual_trend":
                            # Residual trend uses RESIDUAL_TREND features
                            features_dict = self.features.get("RESIDUAL_TREND")
                            strategy_sigs = strategy.signals(market, date_dt, features=features_dict)
                        elif strategy_name == "persistence":
                            # Persistence uses PERSISTENCE features
                            features_dict = self.features.get("PERSISTENCE")
                            strategy_sigs = strategy.signals(market, date_dt, features=features_dict)
                        else:
                            # Generic: try to pass features
                            features_dict = self.features.get("SR3_CURVE")
                            strategy_sigs = strategy.signals(market, date_dt, features=features_dict)
                    elif 'rates_features' in sig.parameters:
                        # Rates curve strategy signature: signals(rates_features, date)
                        features_dict = self.features.get("RATES_CURVE")
                        strategy_sigs = strategy.signals(features_dict, date_dt)
                    else:
                        strategy_sigs = strategy.signals(market, date_dt)
                    
                    # Validate VRP sleeve scope: VRP sleeves may only emit VRP_ALLOWED_SYMBOLS
                    if strategy_name in VRP_SLEEVE_NAMES:
                        emitted = set(strategy_sigs.keys())
                        invalid = emitted - VRP_ALLOWED_SYMBOLS
                        if invalid:
                            logger.error(
                                f"[CombinedStrategy] VRP sleeve '{strategy_name}' emitted "
                                f"symbols outside VRP scope: {invalid}. Dropping them."
                            )
                            strategy_sigs = {k: v for k, v in strategy_sigs.items()
                                             if k in VRP_ALLOWED_SYMBOLS}
                    else:
                        # Non-VRP sleeves must NOT trade VX instruments.
                        # Strip any VX signals to prevent TSMOM/Carry/etc from
                        # accidentally generating positions in VX.
                        vx_emitted = set(strategy_sigs.keys()) & VRP_ALLOWED_SYMBOLS
                        if vx_emitted:
                            logger.debug(
                                f"[CombinedStrategy] Non-VRP sleeve '{strategy_name}' "
                                f"emitted VX symbols {vx_emitted}; stripping them "
                                f"(only VRP sleeves may trade VX)."
                            )
                            strategy_sigs = {k: v for k, v in strategy_sigs.items()
                                             if k not in VRP_ALLOWED_SYMBOLS}
                    
                    # Apply weight and add to combined signals
                    weighted_sigs = {}
                    for symbol, signal in strategy_sigs.items():
                        weighted_signal = weight * signal
                        if symbol not in all_signals:
                            all_signals[symbol] = 0.0
                        all_signals[symbol] += weighted_signal
                        weighted_sigs[symbol] = weighted_signal
                    
                    # Store component if requested
                    if return_components:
                        weighted_series = pd.Series(weighted_sigs)
                        # Ensure all universe symbols are present (fill missing with 0)
                        for symbol in market.universe:
                            if symbol not in weighted_series.index:
                                weighted_series[symbol] = 0.0
                        weighted_series = weighted_series.reindex(market.universe, fill_value=0.0)
                        components[strategy_name] = weighted_series
                        
            except Exception as e:
                logger.warning(
                    f"[CombinedStrategy] Error getting signals from {strategy_name}: {e}"
                )
                continue
        
        # Convert to Series
        if not all_signals:
            # Return zero signals for all symbols in universe
            return pd.Series(0.0, index=market.universe)
        
        combined = pd.Series(all_signals)
        
        # GUARDRAIL: Check for symbols emitted by sleeves that are NOT in master universe.
        # This prevents silent drops that cause zero PnL for entire sleeves.
        emitted_symbols = set(combined.index)
        universe_set = set(market.universe)
        dropped = emitted_symbols - universe_set
        
        if dropped:
            msg = (
                f"[CombinedStrategy] UNIVERSE MISMATCH: Sleeves emitted symbols not in master universe. "
                f"Dropped symbols: {sorted(dropped)}. "
                f"Master universe ({len(universe_set)} symbols): {sorted(universe_set)}. "
                f"This means those sleeves contribute ZERO PnL. "
                f"Fix: add these symbols to configs/data.yaml universe or vx_universe."
            )
            if self.strict_universe:
                raise RuntimeError(msg)
            else:
                logger.warning(msg)
        
        # Reindex to master universe: keep all universe symbols, fill missing with 0.
        # Symbols NOT in the universe are dropped (with warning above).
        for symbol in market.universe:
            if symbol not in combined.index:
                combined[symbol] = 0.0
        
        combined = combined.reindex(market.universe, fill_value=0.0)
        
        # Cache for potential reuse
        self._last_signals = combined
        self._last_date = date_dt
        
        logger.debug(
            f"[CombinedStrategy] Combined signals at {date_dt}: "
            f"mean={combined.mean():.3f}, std={combined.std():.3f}, "
            f"sum={combined.sum():.3f}"
        )
        
        if return_components:
            return combined, components
        return combined
    
    def describe(self) -> dict:
        """Return strategy description."""
        return {
            "type": "CombinedStrategy",
            "strategies": list(self.strategies.keys()),
            "weights": self.weights
        }

