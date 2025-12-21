"""
RegimeClassifierV1: Regime Classification from Allocator State

Consumes allocator state features and classifies portfolio regimes:
- NORMAL: Typical market conditions
- ELEVATED: Increased volatility or correlation
- STRESS: Significant drawdown or volatility spike
- CRISIS: Extreme conditions requiring defensive positioning

Uses deterministic, rule-based classification with hysteresis to avoid thrash.
"""

import logging
from typing import Optional, Dict
import pandas as pd
import numpy as np

from .regime_rules_v1 import (
    get_default_thresholds,
    validate_thresholds,
    REGIMES,
    REGIME_LEVELS
)

logger = logging.getLogger(__name__)


class RegimeClassifierV1:
    """
    Regime classifier for portfolio state.
    
    Uses four stress condition signals:
    1. S_vol_fast: Volatility acceleration (short-term >> long-term)
    2. S_corr_spike: Correlation shock (sudden correlation increase)
    3. S_dd_deep: Deep drawdown
    4. S_dd_worsening: Drawdown deteriorating rapidly
    
    Combines these into a risk score and maps to regimes with hysteresis.
    """
    
    VERSION = "v1.0"
    
    def __init__(self, thresholds: Optional[Dict] = None):
        """
        Initialize RegimeClassifierV1.
        
        Args:
            thresholds: Optional dict of threshold parameters.
                       If None, uses default thresholds from regime_rules_v1.
        """
        self.thresholds = thresholds or get_default_thresholds()
        validate_thresholds(self.thresholds)
        
        logger.info(f"[RegimeClassifierV1] Initialized (version {self.VERSION})")
        logger.info(f"[RegimeClassifierV1] Thresholds: {self.thresholds}")
    
    def classify(
        self,
        state_df: pd.DataFrame
    ) -> pd.Series:
        """
        Classify regime from allocator state features.
        
        Args:
            state_df: Allocator state DataFrame with canonical features
        
        Returns:
            Series of regime labels indexed by date
            Values: 'NORMAL', 'ELEVATED', 'STRESS', 'CRISIS'
        """
        if state_df.empty:
            logger.warning("[RegimeClassifierV1] Empty state DataFrame, returning empty regime series")
            return pd.Series(dtype=str)
        
        logger.info(f"[RegimeClassifierV1] Classifying regime for {len(state_df)} dates")
        
        # Initialize regime series
        regime = pd.Series(index=state_df.index, dtype=str)
        
        # Track days in current regime for anti-thrash
        days_in_regime = 0
        current_regime = 'NORMAL'
        
        for i, date in enumerate(state_df.index):
            row = state_df.loc[date]
            
            # Compute stress condition signals (ENTER thresholds)
            s_vol_fast_enter = row['vol_accel'] >= self.thresholds['vol_accel_enter']
            s_corr_spike_enter = row['corr_shock'] >= self.thresholds['corr_shock_enter']
            s_dd_deep_enter = row['dd_level'] <= self.thresholds['dd_enter']
            s_dd_worsening_enter = row['dd_slope_10d'] <= self.thresholds['dd_slope_enter']
            
            # Compute risk score (ENTER)
            risk_score_enter = sum([
                s_vol_fast_enter,
                s_corr_spike_enter,
                s_dd_deep_enter,
                s_dd_worsening_enter
            ])
            
            # Compute stress condition signals (EXIT thresholds)
            s_vol_fast_exit = row['vol_accel'] >= self.thresholds['vol_accel_exit']
            s_corr_spike_exit = row['corr_shock'] >= self.thresholds['corr_shock_exit']
            s_dd_deep_exit = row['dd_level'] <= self.thresholds['dd_exit']
            s_dd_worsening_exit = row['dd_slope_10d'] <= self.thresholds['dd_slope_exit']
            
            # Compute risk score (EXIT)
            risk_score_exit = sum([
                s_vol_fast_exit,
                s_corr_spike_exit,
                s_dd_deep_exit,
                s_dd_worsening_exit
            ])
            
            # Determine target regime based on ENTER logic
            target_regime = self._compute_enter_regime(
                row,
                risk_score_enter,
                s_vol_fast_enter,
                s_corr_spike_enter,
                s_dd_worsening_enter
            )
            
            # Check if we should stay in current regime (hysteresis + anti-thrash)
            if i == 0:
                # First date: use target regime
                regime.iloc[i] = target_regime
                current_regime = target_regime
                days_in_regime = 1
            else:
                # Check if we can downgrade (use EXIT thresholds)
                can_downgrade = self._can_downgrade(
                    current_regime,
                    target_regime,
                    days_in_regime,
                    risk_score_exit
                )
                
                if can_downgrade:
                    # Downgrade to target regime
                    regime.iloc[i] = target_regime
                    if target_regime != current_regime:
                        logger.debug(
                            f"[RegimeClassifierV1] {date}: Downgrade {current_regime} -> {target_regime} "
                            f"(days_in_regime={days_in_regime})"
                        )
                        current_regime = target_regime
                        days_in_regime = 1
                    else:
                        days_in_regime += 1
                elif REGIME_LEVELS[target_regime] > REGIME_LEVELS[current_regime]:
                    # Upgrade immediately (no hysteresis on upgrades)
                    regime.iloc[i] = target_regime
                    logger.debug(
                        f"[RegimeClassifierV1] {date}: Upgrade {current_regime} -> {target_regime}"
                    )
                    current_regime = target_regime
                    days_in_regime = 1
                else:
                    # Stay in current regime
                    regime.iloc[i] = current_regime
                    days_in_regime += 1
        
        # Log regime statistics
        regime_counts = regime.value_counts()
        logger.info(f"[RegimeClassifierV1] Regime distribution: {regime_counts.to_dict()}")
        
        return regime
    
    def _compute_enter_regime(
        self,
        row: pd.Series,
        risk_score: int,
        s_vol_fast: bool,
        s_corr_spike: bool,
        s_dd_worsening: bool
    ) -> str:
        """
        Compute target regime based on ENTER thresholds.
        
        Args:
            row: State features for current date
            risk_score: Sum of stress condition signals
            s_vol_fast: Volatility acceleration signal
            s_corr_spike: Correlation shock signal
            s_dd_worsening: Drawdown worsening signal
        
        Returns:
            Target regime: 'NORMAL', 'ELEVATED', 'STRESS', or 'CRISIS'
        """
        # CRISIS conditions (most severe)
        if (row['dd_level'] <= self.thresholds['dd_crisis_enter'] or
            risk_score >= 3 or
            (s_vol_fast and s_corr_spike and s_dd_worsening)):
            return 'CRISIS'
        
        # STRESS conditions
        if (risk_score >= 2 or
            (s_vol_fast and s_corr_spike) or
            row['dd_level'] <= self.thresholds['dd_stress_enter']):
            return 'STRESS'
        
        # ELEVATED conditions
        if risk_score >= 1:
            return 'ELEVATED'
        
        # NORMAL (default)
        return 'NORMAL'
    
    def _can_downgrade(
        self,
        current_regime: str,
        target_regime: str,
        days_in_regime: int,
        risk_score_exit: int
    ) -> bool:
        """
        Check if regime can be downgraded (with hysteresis and anti-thrash).
        
        Args:
            current_regime: Current regime
            target_regime: Target regime based on ENTER thresholds
            days_in_regime: Number of consecutive days in current regime
            risk_score_exit: Risk score computed with EXIT thresholds
        
        Returns:
            True if downgrade is allowed
        """
        # Can't downgrade if target is higher severity
        if REGIME_LEVELS[target_regime] >= REGIME_LEVELS[current_regime]:
            return False
        
        # Anti-thrash: must be in regime for MIN_DAYS before downgrade
        if days_in_regime < self.thresholds['min_days_in_regime']:
            return False
        
        # Additional check: EXIT risk score must be low enough
        # For CRISIS->STRESS, require risk_score_exit < 3
        # For STRESS->ELEVATED, require risk_score_exit < 2
        # For ELEVATED->NORMAL, require risk_score_exit < 1
        if current_regime == 'CRISIS' and risk_score_exit >= 3:
            return False
        if current_regime == 'STRESS' and risk_score_exit >= 2:
            return False
        if current_regime == 'ELEVATED' and risk_score_exit >= 1:
            return False
        
        return True
    
    def describe(self) -> dict:
        """
        Return description of RegimeClassifierV1.
        
        Returns:
            Dict with version and description
        """
        return {
            'agent': 'RegimeClassifierV1',
            'version': self.VERSION,
            'role': 'Classify portfolio regime from allocator state',
            'regimes': REGIMES,
            'thresholds': self.thresholds,
            'input': 'allocator_state_v1',
            'output': 'regime_series'
        }

