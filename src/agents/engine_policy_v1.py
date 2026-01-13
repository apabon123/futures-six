"""
Engine Policy v1: Binary Gate for Engine Validity Filtering

This module implements Layer 2 (Engine Policy) in the canonical execution stack:
1. Engine Signals (alpha)
2. **Engine Policy** (gates / throttles) <- THIS MODULE
3. Portfolio Construction (static weights)
4. Discretionary Overlay (bounded tilts)
5. Risk Targeting (vol → leverage)
6. Allocator (risk brake)
7. Margin & Execution Constraints

Key Principle: Engine Policy is a validity filter, not an optimizer.

v1 Implementation:
- Binary gate (multiplier ∈ {0, 1})
- Inputs: context features like gamma / vol-of-vol
- NOT allowed: portfolio drawdown, correlation, sizing (allocator territory)

Artifacts:
- engine_policy_state_v1.csv (daily)
- engine_policy_applied_v1.csv (rebalance dates)
- engine_policy_v1_meta.json (once per run)
"""

import logging
import json
import hashlib
from typing import Dict, Optional, Union, Any
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


# ==============================================================================
# Constants
# ==============================================================================

VERSION = "v1.0"

VALID_MODES = ["off", "compute", "precomputed"]

# Default rules (engine-local)
DEFAULT_RULES = {
    "gamma_vol_stress_gate_v1": {
        "description": "Binary gate based on gamma/vol stress proxy",
        "feature": "gamma_stress_proxy",
        "threshold": 1,
        "invert": False,  # OFF when feature >= threshold
    }
}


# ==============================================================================
# Helper Functions
# ==============================================================================

def compute_determinism_hash(config: dict, state_df: Optional[pd.DataFrame] = None) -> str:
    """
    Compute deterministic hash for reproducibility verification.
    
    Args:
        config: Engine policy configuration
        state_df: Optional state DataFrame for content hash
    
    Returns:
        Hexadecimal hash string
    """
    hasher = hashlib.sha256()
    
    # Hash config
    config_str = json.dumps(config, sort_keys=True, default=str)
    hasher.update(config_str.encode('utf-8'))
    
    # Hash state content if provided
    if state_df is not None and not state_df.empty:
        hasher.update(state_df.to_csv().encode('utf-8'))
    
    return hasher.hexdigest()[:16]


def compute_csv_hash(csv_path: Union[str, Path]) -> str:
    """
    Compute hash of a CSV file for verification.
    
    Args:
        csv_path: Path to CSV file
    
    Returns:
        Hexadecimal hash string
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    hasher = hashlib.sha256()
    with open(csv_path, 'rb') as f:
        hasher.update(f.read())
    
    return hasher.hexdigest()[:16]


# ==============================================================================
# Engine Policy v1
# ==============================================================================

class EnginePolicyV1:
    """
    Engine Policy v1: Binary gate for engine validity filtering.
    
    This class implements a binary gate (ON/OFF) for engines based on
    context features like gamma stress proxy. It sits between Engine Signals
    and Portfolio Construction in the canonical stack.
    
    Key Design Principles:
    - Binary gate: multiplier is always 0 or 1
    - Context-based: uses external features, not portfolio metrics
    - Engine-local: each engine can have its own rule
    - Lagged: applies t-1 policy decision to t rebalance (lag_rebalances=1)
    - Deterministic: fully reproducible given same config and data
    """
    
    VERSION = VERSION
    
    def __init__(
        self,
        config: Dict[str, Any],
        artifact_writer: Optional[Any] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize EnginePolicyV1.
        
        Args:
            config: Engine policy configuration dict with keys:
                - enabled: bool
                - mode: "off" | "compute" | "precomputed"
                - precomputed_run_id: Optional[str]
                - lag_rebalances: int (default: 1)
                - engines: Dict[str, engine_config]
            artifact_writer: Optional ArtifactWriter for saving outputs
            logger: Optional custom logger
        """
        self.config = config
        self.artifact_writer = artifact_writer
        self._logger = logger or logging.getLogger(__name__)
        
        # Parse config
        self.enabled = config.get('enabled', False)
        self.mode = config.get('mode', 'off')
        self.precomputed_run_id = config.get('precomputed_run_id', None)
        self.lag_rebalances = config.get('lag_rebalances', 1)
        self.engines_config = config.get('engines', {})
        
        # Validate mode
        if self.mode not in VALID_MODES:
            raise ValueError(f"Invalid mode '{self.mode}'. Must be one of: {VALID_MODES}")
        
        # Validate precomputed mode requirements
        if self.mode == 'precomputed' and not self.precomputed_run_id:
            raise ValueError("mode='precomputed' requires precomputed_run_id to be set")
        
        # Cache for precomputed multipliers
        self._precomputed_multipliers: Optional[pd.DataFrame] = None
        self._meta_written = False
        
        self._logger.info(
            f"[EnginePolicyV1] Initialized: enabled={self.enabled}, mode={self.mode}, "
            f"lag_rebalances={self.lag_rebalances}, engines={list(self.engines_config.keys())}"
        )
    
    def compute_state(
        self,
        market,
        start: Union[str, datetime],
        end: Union[str, datetime]
    ) -> pd.DataFrame:
        """
        Compute daily policy state for all enabled engines.
        
        This method computes the policy state (ON/OFF) for each engine
        on each trading day based on the configured rules and features.
        
        Args:
            market: MarketData instance with features dict
            start: Start date
            end: End date
        
        Returns:
            DataFrame with columns:
                - date
                - engine
                - stress_value
                - stress_percentile (optional)
                - policy_state ("ON" | "OFF")
                - policy_multiplier (0 | 1)
        """
        start_dt = pd.to_datetime(start)
        end_dt = pd.to_datetime(end)
        
        self._logger.info(
            f"[EnginePolicyV1] Computing state from {start_dt.date()} to {end_dt.date()}"
        )
        
        # Get trading days
        trading_days = market.trading_days()
        trading_days = trading_days[(trading_days >= start_dt) & (trading_days <= end_dt)]
        
        if len(trading_days) == 0:
            self._logger.warning("[EnginePolicyV1] No trading days in range")
            return pd.DataFrame()
        
        # Collect state for each engine
        state_records = []
        
        for engine_name, engine_cfg in self.engines_config.items():
            if not engine_cfg.get('enabled', False):
                continue
            
            rule_name = engine_cfg.get('rule', 'gamma_vol_stress_gate_v1')
            feature_name = engine_cfg.get('feature', 'gamma_stress_proxy')
            threshold = engine_cfg.get('threshold', 1)
            invert = engine_cfg.get('invert', False)
            
            self._logger.debug(
                f"[EnginePolicyV1] Processing engine '{engine_name}': "
                f"rule={rule_name}, feature={feature_name}, threshold={threshold}, invert={invert}"
            )
            
            # Get feature from market
            feature_series = self._get_feature(market, feature_name)
            
            if feature_series is None or feature_series.empty:
                self._logger.warning(
                    f"[EnginePolicyV1] Feature '{feature_name}' not found or empty, "
                    f"defaulting to policy_multiplier=1 (ON)"
                )
                # Default: policy ON when feature unavailable
                for date in trading_days:
                    state_records.append({
                        'date': date,
                        'engine': engine_name,
                        'stress_value': np.nan,
                        'stress_percentile': np.nan,
                        'policy_state': 'ON',
                        'policy_multiplier': 1
                    })
                continue
            
            # Compute state for each trading day
            for date in trading_days:
                # Get feature value at date
                if date in feature_series.index:
                    stress_value = feature_series.loc[date]
                else:
                    # Forward fill to get most recent value
                    prior_values = feature_series[feature_series.index <= date]
                    if len(prior_values) > 0:
                        stress_value = prior_values.iloc[-1]
                    else:
                        stress_value = np.nan
                
                # Compute stress percentile (optional, for diagnostics)
                if not np.isnan(stress_value):
                    prior_values = feature_series[feature_series.index <= date]
                    stress_percentile = (prior_values < stress_value).mean() * 100
                else:
                    stress_percentile = np.nan
                
                # Apply rule: OFF when feature >= threshold (unless inverted)
                if np.isnan(stress_value):
                    policy_state = 'ON'
                    policy_multiplier = 1
                else:
                    stress_triggered = stress_value >= threshold
                    if invert:
                        stress_triggered = not stress_triggered
                    
                    policy_state = 'OFF' if stress_triggered else 'ON'
                    policy_multiplier = 0 if stress_triggered else 1
                
                state_records.append({
                    'date': date,
                    'engine': engine_name,
                    'stress_value': stress_value,
                    'stress_percentile': stress_percentile,
                    'policy_state': policy_state,
                    'policy_multiplier': policy_multiplier
                })
        
        if not state_records:
            self._logger.warning("[EnginePolicyV1] No enabled engines found")
            return pd.DataFrame()
        
        state_df = pd.DataFrame(state_records)
        state_df['date'] = pd.to_datetime(state_df['date'])
        state_df = state_df.sort_values(['date', 'engine']).reset_index(drop=True)
        
        # Log summary
        n_off = (state_df['policy_state'] == 'OFF').sum()
        n_total = len(state_df)
        pct_off = n_off / n_total * 100 if n_total > 0 else 0
        
        self._logger.info(
            f"[EnginePolicyV1] Computed state: {n_total} records, "
            f"{n_off} OFF ({pct_off:.1f}%)"
        )
        
        # Write artifact
        if self.artifact_writer is not None:
            self.artifact_writer.write_csv(
                "engine_policy_state_v1.csv",
                state_df,
                mode="overwrite"
            )
            self._logger.info("[EnginePolicyV1] Saved engine_policy_state_v1.csv")
        
        return state_df
    
    def compute_applied_multipliers(
        self,
        state_df: pd.DataFrame,
        rebalance_dates: pd.DatetimeIndex
    ) -> pd.DataFrame:
        """
        Compute multipliers to apply at each rebalance date with lag.
        
        This method maps daily policy state to rebalance dates, applying
        the configured lag (default: 1 rebalance).
        
        Args:
            state_df: Daily policy state from compute_state()
            rebalance_dates: DatetimeIndex of rebalance dates
        
        Returns:
            DataFrame with columns:
                - rebalance_date
                - engine
                - policy_multiplier_used
                - source_run_id (null for compute mode)
        """
        if state_df.empty:
            self._logger.warning("[EnginePolicyV1] Empty state_df, returning empty multipliers")
            return pd.DataFrame()
        
        self._logger.info(
            f"[EnginePolicyV1] Computing applied multipliers for "
            f"{len(rebalance_dates)} rebalance dates with lag={self.lag_rebalances}"
        )
        
        # Get unique engines
        engines = state_df['engine'].unique()
        
        # Build multiplier records
        multiplier_records = []
        
        for i, rebal_date in enumerate(rebalance_dates):
            # Determine which state to use (with lag)
            # lag_rebalances=1 means use state from previous rebalance
            state_idx = i - self.lag_rebalances
            
            if state_idx < 0:
                # Not enough history, use default (ON)
                for engine in engines:
                    multiplier_records.append({
                        'rebalance_date': rebal_date,
                        'engine': engine,
                        'policy_multiplier_used': 1,
                        'source_run_id': None
                    })
                continue
            
            # Get the state date corresponding to lagged rebalance
            state_date = rebalance_dates[state_idx]
            
            # Get state for each engine at that date
            for engine in engines:
                engine_state = state_df[
                    (state_df['engine'] == engine) &
                    (state_df['date'] <= state_date)
                ]
                
                if len(engine_state) > 0:
                    # Get most recent state up to (and including) state_date
                    most_recent = engine_state.iloc[-1]
                    multiplier = most_recent['policy_multiplier']
                else:
                    # No state available, default to ON
                    multiplier = 1
                
                multiplier_records.append({
                    'rebalance_date': rebal_date,
                    'engine': engine,
                    'policy_multiplier_used': multiplier,
                    'source_run_id': None
                })
        
        applied_df = pd.DataFrame(multiplier_records)
        applied_df['rebalance_date'] = pd.to_datetime(applied_df['rebalance_date'])
        applied_df = applied_df.sort_values(['rebalance_date', 'engine']).reset_index(drop=True)
        
        # Log summary
        n_gated = (applied_df['policy_multiplier_used'] == 0).sum()
        n_total = len(applied_df)
        pct_gated = n_gated / n_total * 100 if n_total > 0 else 0
        
        self._logger.info(
            f"[EnginePolicyV1] Computed applied multipliers: {n_total} records, "
            f"{n_gated} gated OFF ({pct_gated:.1f}%)"
        )
        
        # Write artifact
        if self.artifact_writer is not None:
            self.artifact_writer.write_csv(
                "engine_policy_applied_v1.csv",
                applied_df,
                mode="overwrite"
            )
            self._logger.info("[EnginePolicyV1] Saved engine_policy_applied_v1.csv")
        
        return applied_df
    
    def load_precomputed_multipliers(self, run_dir: Union[str, Path]) -> pd.DataFrame:
        """
        Load precomputed multipliers from a prior run.
        
        Args:
            run_dir: Path to run directory (e.g., reports/runs/{run_id})
        
        Returns:
            DataFrame with applied multipliers
        """
        run_dir = Path(run_dir)
        multipliers_file = run_dir / "engine_policy_applied_v1.csv"
        
        if not multipliers_file.exists():
            raise FileNotFoundError(
                f"Precomputed multipliers file not found: {multipliers_file}\n"
                f"Make sure the baseline run has engine_policy_applied_v1.csv generated."
            )
        
        self._logger.info(f"[EnginePolicyV1] Loading precomputed multipliers from: {multipliers_file}")
        
        # Read CSV
        df = pd.read_csv(multipliers_file, parse_dates=['rebalance_date'])
        
        # Update source_run_id
        df['source_run_id'] = self.precomputed_run_id
        
        self._precomputed_multipliers = df
        
        self._logger.info(
            f"[EnginePolicyV1] Loaded {len(df)} precomputed multipliers from "
            f"{df['rebalance_date'].min().date()} to {df['rebalance_date'].max().date()}"
        )
        
        return df
    
    def get_multiplier_at_rebalance(
        self,
        date: Union[str, datetime],
        engine: str,
        applied_df: Optional[pd.DataFrame] = None
    ) -> int:
        """
        Get the policy multiplier for a specific engine at a rebalance date.
        
        Args:
            date: Rebalance date
            engine: Engine name (e.g., "trend")
            applied_df: Optional DataFrame of applied multipliers
        
        Returns:
            Policy multiplier (0 or 1)
        """
        date_dt = pd.to_datetime(date)
        
        # Use precomputed if available
        if applied_df is None and self._precomputed_multipliers is not None:
            applied_df = self._precomputed_multipliers
        
        if applied_df is None or applied_df.empty:
            # Default: policy ON
            return 1
        
        # Find matching record
        mask = (applied_df['rebalance_date'] == date_dt) & (applied_df['engine'] == engine)
        matching = applied_df[mask]
        
        if len(matching) > 0:
            return int(matching.iloc[0]['policy_multiplier_used'])
        
        # Not found, try forward fill from most recent
        prior = applied_df[
            (applied_df['engine'] == engine) &
            (applied_df['rebalance_date'] <= date_dt)
        ]
        
        if len(prior) > 0:
            return int(prior.iloc[-1]['policy_multiplier_used'])
        
        # Default: policy ON
        return 1
    
    def apply(
        self,
        engine_weights_dict: Dict[str, pd.Series],
        multipliers_at_rebalance: Dict[str, int]
    ) -> Dict[str, pd.Series]:
        """
        Apply policy multipliers to engine weights.
        
        This method multiplies each engine's weights by its policy multiplier.
        For v1, multiplier is binary (0 or 1), so this either zeros out
        the engine's contribution or leaves it unchanged.
        
        Args:
            engine_weights_dict: Dict mapping engine names to weight Series
            multipliers_at_rebalance: Dict mapping engine names to multipliers
        
        Returns:
            Modified engine_weights_dict with policy applied
        """
        result = {}
        
        for engine_name, weights in engine_weights_dict.items():
            multiplier = multipliers_at_rebalance.get(engine_name, 1)
            
            if multiplier == 0:
                # Gate OFF: zero out weights
                self._logger.info(
                    f"[EnginePolicyV1] Engine '{engine_name}' GATED OFF (multiplier=0)"
                )
                result[engine_name] = weights * 0
            else:
                # Gate ON: pass through unchanged
                result[engine_name] = weights.copy()
        
        return result
    
    def apply_to_signals(
        self,
        signals: pd.Series,
        date: Union[str, datetime],
        applied_df: Optional[pd.DataFrame] = None,
        engine_name: str = "trend"
    ) -> pd.Series:
        """
        Apply policy to signals at a specific date (convenience method).
        
        For combined strategies where signals are already aggregated,
        this method applies the policy multiplier directly to the signal Series.
        
        Args:
            signals: Signal Series from strategy
            date: Rebalance date
            applied_df: Optional DataFrame of applied multipliers
            engine_name: Engine name for multiplier lookup (default: "trend")
        
        Returns:
            Scaled signals
        """
        multiplier = self.get_multiplier_at_rebalance(date, engine_name, applied_df)
        
        if multiplier == 0:
            self._logger.info(
                f"[EnginePolicyV1] Applying multiplier=0 to {engine_name} at {date}"
            )
            return signals * 0
        
        return signals.copy()
    
    def write_meta(
        self,
        run_id: str,
        state_df: Optional[pd.DataFrame] = None,
        applied_df: Optional[pd.DataFrame] = None
    ) -> None:
        """
        Write metadata JSON artifact.
        
        Args:
            run_id: Current run ID
            state_df: Optional state DataFrame for hash
            applied_df: Optional applied DataFrame for stats
        """
        if self._meta_written:
            return
        
        # Compute determinism hash
        det_hash = compute_determinism_hash(self.config, state_df)
        
        # Build engine list with rules
        engines_list = []
        for engine_name, engine_cfg in self.engines_config.items():
            if engine_cfg.get('enabled', False):
                engines_list.append({
                    'name': engine_name,
                    'rule': engine_cfg.get('rule', 'gamma_vol_stress_gate_v1'),
                    'feature': engine_cfg.get('feature', 'gamma_stress_proxy'),
                    'threshold': engine_cfg.get('threshold', 1),
                    'invert': engine_cfg.get('invert', False)
                })
        
        # Compute stats from applied_df
        stats = {}
        if applied_df is not None and not applied_df.empty:
            stats = {
                'n_rebalances': len(applied_df['rebalance_date'].unique()),
                'n_gated': int((applied_df['policy_multiplier_used'] == 0).sum()),
                'pct_gated': float((applied_df['policy_multiplier_used'] == 0).mean() * 100),
                'engines_affected': applied_df[
                    applied_df['policy_multiplier_used'] == 0
                ]['engine'].unique().tolist()
            }
        
        meta = {
            'version': self.VERSION,
            'run_id': run_id,
            'enabled': self.enabled,
            'mode': self.mode,
            'precomputed_run_id': self.precomputed_run_id,
            'lag_rebalances': self.lag_rebalances,
            'engines': engines_list,
            'determinism_hash': det_hash,
            'config_snapshot': self.config,
            'stats': stats,
            'generated_at': datetime.now().isoformat()
        }
        
        if self.artifact_writer is not None:
            self.artifact_writer.write_json(
                "engine_policy_v1_meta.json",
                meta,
                mode="once"
            )
            self._logger.info("[EnginePolicyV1] Saved engine_policy_v1_meta.json")
        
        self._meta_written = True
    
    def _get_feature(
        self,
        market,
        feature_name: str
    ) -> Optional[pd.Series]:
        """
        Get feature series from market data.
        
        This method abstracts feature retrieval, allowing flexibility
        in how features are stored (market.features dict, separate DataFrame, etc.)
        
        Args:
            market: MarketData instance
            feature_name: Name of feature to retrieve
        
        Returns:
            Feature Series indexed by date, or None if not found
        """
        # Try market.features dict
        if hasattr(market, 'features') and isinstance(market.features, dict):
            if feature_name in market.features:
                feature = market.features[feature_name]
                if isinstance(feature, pd.Series):
                    return feature
                elif isinstance(feature, pd.DataFrame):
                    # If DataFrame, try to find column with feature_name
                    if feature_name in feature.columns:
                        return feature[feature_name]
                    # Or return first column if single column
                    if len(feature.columns) == 1:
                        return feature.iloc[:, 0]
        
        # Try direct attribute
        if hasattr(market, feature_name):
            feature = getattr(market, feature_name)
            if isinstance(feature, pd.Series):
                return feature
        
        # Try market.get_feature() method
        if hasattr(market, 'get_feature'):
            try:
                return market.get_feature(feature_name)
            except Exception:
                pass
        
        return None
    
    def describe(self) -> dict:
        """Return description of EnginePolicyV1."""
        return {
            'agent': 'EnginePolicyV1',
            'version': self.VERSION,
            'role': 'Binary gate for engine validity filtering',
            'enabled': self.enabled,
            'mode': self.mode,
            'lag_rebalances': self.lag_rebalances,
            'engines': list(self.engines_config.keys()),
            'placement': 'Layer 2 in canonical stack (between Engine Signals and Portfolio Construction)'
        }


# ==============================================================================
# Factory Function
# ==============================================================================

def create_engine_policy_v1(
    config: Dict[str, Any],
    artifact_writer: Optional[Any] = None,
    logger: Optional[logging.Logger] = None
) -> Optional[EnginePolicyV1]:
    """
    Factory function to create EnginePolicyV1 instance.
    
    Args:
        config: Engine policy configuration
        artifact_writer: Optional ArtifactWriter
        logger: Optional custom logger
    
    Returns:
        EnginePolicyV1 instance if enabled, None otherwise
    """
    if config is None:
        return None
    
    if not config.get('enabled', False):
        return None
    
    return EnginePolicyV1(config, artifact_writer, logger)

