"""
ExecSim: Backtest Orchestrator & Metrics Agent

Runs the complete backtest loop:
1. Build rebalance schedule where risk data is available
2. On each rebalance date:
   - Get strategy signals
   - Apply vol-managed overlay scaling
   - Allocate to final weights
   - Apply returns (next-day convention: close-to-close)
3. Apply slippage/commissions on rebalance days only
4. Compute equity curve and performance metrics

No data writes. No look-ahead. Deterministic outputs given MarketData snapshot.
"""

import logging
from typing import Dict, Optional, Union, Tuple
from datetime import datetime
from pathlib import Path
import os
import json
import shutil

import yaml
import pandas as pd
import numpy as np

from src.config.trading_calendar import is_us_trading_day, previous_us_trading_day

logger = logging.getLogger(__name__)



class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super(NumpyEncoder, self).default(obj)


class ExecSim:
    """
    Backtest orchestrator and metrics calculator.
    
    Coordinates all agents (strategy, overlay, allocator) and simulates
    portfolio execution over a rebalance schedule, producing equity curves,
    weights panels, and performance metrics.
    """
    
    def __init__(
        self,
        rebalance: str = "W-FRI",
        slippage_bps: float = 0.5,
        commission_per_contract: float = 0.0,
        cash_rate: float = 0.0,
        position_notional_scale: float = 1.0,
        config_path: str = "configs/strategies.yaml",
        filter_roll_jumps: bool = True,
        roll_jump_threshold_bp: float = 100.0
    ):
        """
        Initialize ExecSim orchestrator.
        
        Args:
            rebalance: Rebalance frequency ("W-FRI", "M", etc.)
            slippage_bps: Slippage in basis points (applied to turnover on rebalance days)
            commission_per_contract: Commission per contract (placeholder, not used until contract sizing)
            cash_rate: Risk-free rate for cash (default: 0.0)
            position_notional_scale: Scale factor for positions (default: 1.0 = unit portfolio)
            config_path: Path to configuration YAML file
            filter_roll_jumps: Whether to filter roll jumps from returns used for P&L (default: True)
            roll_jump_threshold_bp: Threshold in basis points for detecting roll jumps (default: 100.0)
        """
        # Use explicit params (don't load from config for ExecSim tests)
        self.rebalance = rebalance
        self.slippage_bps = slippage_bps
        self.commission_per_contract = commission_per_contract
        self.cash_rate = cash_rate
        self.position_notional_scale = position_notional_scale
        self.filter_roll_jumps = filter_roll_jumps
        self.roll_jump_threshold_bp = roll_jump_threshold_bp
        
        # Validate parameters
        if self.slippage_bps < 0:
            raise ValueError(f"slippage_bps must be >= 0, got {self.slippage_bps}")
        
        if self.commission_per_contract < 0:
            raise ValueError(f"commission_per_contract must be >= 0, got {self.commission_per_contract}")
        
        logger.info(
            f"[ExecSim] Initialized: rebalance={self.rebalance}, "
            f"slippage_bps={self.slippage_bps}, commission={self.commission_per_contract}, "
            f"cash_rate={self.cash_rate}, scale={self.position_notional_scale}, "
            f"filter_roll_jumps={self.filter_roll_jumps}, roll_jump_threshold_bp={self.roll_jump_threshold_bp}"
        )
    
    @staticmethod
    def _to_ts(x):
        """Normalize date-like objects to pandas.Timestamp."""
        return x if isinstance(x, pd.Timestamp) else pd.Timestamp(x)
    
    @staticmethod
    def _get_sleeve_to_engine_mapping() -> Dict[str, str]:
        """
        Get explicit mapping from sleeve names to policy engine names.
        
        This is part of the permission layer contract and must be explicit and auditable.
        
        Returns:
            Dict mapping sleeve_name -> policy_engine_name ("trend" or "vrp")
        """
        return {
            # Trend-related sleeves
            'tsmom_multihorizon': 'trend',
            'tsmom': 'trend',
            'tsmom_long': 'trend',
            'tsmom_med': 'trend',
            'tsmom_short': 'trend',
            'tsmom_med_canonical': 'trend',
            'canonical_medium_momentum': 'trend',
            'medium_momentum': 'trend',
            'short_momentum': 'trend',
            'long_momentum': 'trend',
            'residual_trend': 'trend',
            # VRP-related sleeves
            'vrp_core_meta': 'vrp',
            'vrp_convergence_meta': 'vrp',
            'vrp_alt_meta': 'vrp',
            'vrp_core': 'vrp',
            'vrp_convergence': 'vrp',
            'vrp_alt': 'vrp',
        }
    
    @staticmethod
    def _get_policy_exempt_sleeves() -> set:
        """
        Get set of sleeve names that are explicitly exempt from policy.
        
        These sleeves are allowed to exist without policy mapping.
        
        Returns:
            Set of exempt sleeve names
        """
        return {
            'csmom_meta',
            'csmom',
            'cross_sectional',
            'rates_curve',
            'sr3_carry_curve',
            'fx_commod_carry',
            'vx_calendar_carry',
            'sr3_curve_rv_meta',
            'persistence',
        }
    
    @staticmethod
    def _map_sleeve_to_policy_engine(sleeve_name: str, mapping: Optional[Dict[str, str]] = None) -> Optional[str]:
        """
        Map sleeve name to policy engine name using explicit mapping.
        
        Args:
            sleeve_name: Strategy/sleeve name (e.g., "tsmom_multihorizon", "vrp_core_meta")
            mapping: Optional explicit mapping dict (default: uses _get_sleeve_to_engine_mapping())
            
        Returns:
            Policy engine name ("trend", "vrp") or None if not mapped
        """
        if mapping is None:
            mapping = ExecSim._get_sleeve_to_engine_mapping()
        return mapping.get(sleeve_name)
    
    def _load_config(self, config_path: str) -> Optional[dict]:
        """Load configuration from YAML file."""
        path = Path(config_path)
        if not path.exists():
            logger.warning(f"[ExecSim] Config file not found: {config_path}, using defaults")
            return None
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            logger.warning(f"[ExecSim] Failed to load config: {e}, using defaults")
            return None
    
    def _build_rebalance_dates(
        self,
        market,
        risk_vol,
        start: Union[str, datetime],
        end: Union[str, datetime]
    ) -> pd.DatetimeIndex:
        """
        Build union of rebalancing dates where risk data is available for all tradable symbols.
        
        Args:
            market: MarketData instance
            risk_vol: RiskVol instance
            start: Start date
            end: End date
            
        Returns:
            DatetimeIndex of rebalance dates
        """
        start = pd.to_datetime(start)
        end = pd.to_datetime(end)
        
        # Get all trading days in range
        all_dates = market.trading_days()
        
        if len(all_dates) == 0:
            logger.warning(f"[ExecSim] No trading days available")
            return pd.DatetimeIndex([])
        
        # Filter to date range
        date_range = all_dates[(all_dates >= start) & (all_dates <= end)]
        
        if len(date_range) == 0:
            logger.warning(f"[ExecSim] No trading days in range {start} to {end}")
            return pd.DatetimeIndex([])
        
        # Generate rebalance schedule
        if self.rebalance == "W-FRI":
            schedule = pd.date_range(start=start, end=end, freq='W-FRI')
            # Use US exchange calendar: rebalance on Friday only if it's a trading day,
            # else on previous US trading day (avoids Good Friday etc. causing "No signal")
            actual_schedule = []
            for friday in schedule:
                d = friday.date() if hasattr(friday, 'date') else friday
                if is_us_trading_day(d):
                    actual_schedule.append(pd.Timestamp(d))
                else:
                    prev = previous_us_trading_day(d)
                    actual_schedule.append(pd.Timestamp(prev))
            schedule = pd.DatetimeIndex(actual_schedule)
        elif self.rebalance == "M":
            # Month-end - use 'M' for backward compatibility, but snap to nearest business day
            schedule = pd.date_range(start=start, end=end, freq='M')
            # For month-end, find nearest trading day to each month-end date
            rebalance_candidates = []
            for me_date in schedule:
                # Find closest trading day to month-end
                if me_date in date_range:
                    rebalance_candidates.append(me_date)
                else:
                    # Find nearest trading day before or at month-end
                    prev_days = date_range[date_range <= me_date]
                    if len(prev_days) > 0:
                        rebalance_candidates.append(prev_days[-1])
            schedule = pd.DatetimeIndex(rebalance_candidates)
        elif self.rebalance == "D":
            schedule = pd.date_range(start=start, end=end, freq='D')
        else:
            raise ValueError(f"Unknown rebalance frequency: {self.rebalance}")
        
        # Only keep dates that exist in trading calendar (for non-M frequencies)
        if self.rebalance != "M":
            rebalance_dates = schedule.intersection(date_range)
        else:
            rebalance_dates = schedule  # Already filtered above
        
        # Filter to dates where risk data is available
        valid_dates = []
        for date in rebalance_dates:
            try:
                # Check if we can get a mask (valid symbols with data)
                mask = risk_vol.mask(market, date)
                if len(mask) > 0:
                    valid_dates.append(date)
            except Exception as e:
                logger.debug(f"[ExecSim] Skipping {date}: {e}")
                continue
        
        rebalance_dates = pd.DatetimeIndex(valid_dates)
        
        logger.info(f"[ExecSim] Built {len(rebalance_dates)} rebalance dates from {start} to {end}")
        
        return rebalance_dates
    
    def _compute_portfolio_return(
        self,
        weights: pd.Series,
        returns: pd.Series,
        is_rebalance: bool
    ) -> float:
        """
        Compute portfolio return for one period.
        
        Uses close-to-close convention: weights held from t to t+1, applied to returns[t+1].
        On rebalance days, apply slippage to turnover.
        
        Args:
            weights: Portfolio weights at start of period
            returns: Asset returns for the period
            is_rebalance: Whether this is a rebalance day (apply slippage)
            
        Returns:
            Portfolio return (including transaction costs if rebalance)
        """
        # Align weights and returns
        common = weights.index.intersection(returns.index)
        
        if len(common) == 0:
            return 0.0
        
        w = weights.loc[common].fillna(0)
        r = returns.loc[common].fillna(0)
        
        # Gross portfolio return
        port_ret = (w * r).sum()
        
        # No slippage/commissions for now (placeholder)
        # In future: apply slippage_bps to turnover on rebalance days
        
        return port_ret
    
    def _compute_turnover(
        self,
        weights_prev: pd.Series,
        weights_new: pd.Series
    ) -> float:
        """
        Compute turnover between two weight vectors.
        
        Args:
            weights_prev: Previous weights
            weights_new: New weights
            
        Returns:
            Turnover (sum of absolute changes)
        """
        all_symbols = weights_prev.index.union(weights_new.index)
        w_prev = weights_prev.reindex(all_symbols, fill_value=0)
        w_new = weights_new.reindex(all_symbols, fill_value=0)
        
        turnover = (w_new - w_prev).abs().sum()
        return turnover
    
    @staticmethod
    def _materialize_policy_artifact(
        *,
        source_run_dir: Path,
        target_run_dir: Path,
        filename: str = "engine_policy_applied_v1.csv",
    ) -> bool:
        """
        Copy policy artifact from source run into the current run dir.
        
        This ensures precomputed runs are self-contained and auditable.
        The artifact is copied as-is (preserves exact bytes for determinism).
        
        Args:
            source_run_dir: Path to source run directory
            target_run_dir: Path to target run directory
            filename: Artifact filename (default: "engine_policy_applied_v1.csv")
            
        Returns:
            True if copied, False if source missing
            
        Raises:
            FileNotFoundError: If source artifact is missing (hard requirement for precomputed mode)
        """
        src = source_run_dir / filename
        dst = target_run_dir / filename
        
        if not src.exists():
            raise FileNotFoundError(
                f"Precomputed policy artifact not found: {src}\n"
                f"Precomputed mode requires engine_policy_applied_v1.csv in source run: {source_run_dir.name}\n"
                f"This is a hard requirement for production-safe baselines."
            )
        
        # Make the precomputed run self-contained by copying the canonical artifact
        shutil.copyfile(src, dst)
        logger.info(f"[ExecSim] Materialized policy artifact: {dst.name} from source run: {source_run_dir.name}")
        return True
    
    def run(
        self,
        market,
        start: Union[str, datetime],
        end: Union[str, datetime],
        components: Dict,
        run_id: Optional[str] = None,
        out_dir: str = "reports/runs",
        strategy_profile: Optional[str] = None,
        config_path: Optional[str] = None
    ) -> Dict:
        """
        Run backtest over date range.
        
        Args:
            market: MarketData instance
            start: Start date
            end: End date
            components: Dict with required keys:
                    - 'strategy': Strategy agent (e.g., TSMOM)
                    - 'overlay': VolManagedOverlay agent
                    - 'risk_vol': RiskVol agent
                    - 'allocator': Allocator agent
                Optional keys:
                    - 'macro_overlay': Regime-based overlay applied before vol overlay
            run_id: Optional run identifier for saving artifacts. If None, generates timestamp-based ID.
            out_dir: Base directory for saving run artifacts (default: "reports/runs")
                
        Returns:
            Dict with keys:
                - 'equity_curve': pd.Series of cumulative returns
                - 'weights_panel': pd.DataFrame [date x symbol]
                - 'signals_panel': pd.DataFrame [date x symbol]
                - 'report': dict of performance metrics
                - 'run_id': run_id used (or generated)
        """
        logger.info(f"[ExecSim] Starting backtest from {start} to {end}")
        
        # Extract components
        strategy = components['strategy']
        overlay = components['overlay']
        macro_overlay = components.get('macro_overlay')
        risk_vol = components['risk_vol']
        risk_targeting = components.get('risk_targeting')  # Layer 5: Risk Targeting
        allocator = components['allocator']
        
        # Generate run_id early (needed for error artifacts and logging)
        if run_id is None:
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            logger.info(f"[ExecSim] Generated run_id: {run_id}")
        
        # Create run directory early (needed for error artifacts)
        run_dir = Path(out_dir) / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        
        # CRITICAL: Log run_id and run_dir for debugging (addresses root cause #2)
        logger.info(f"[RUN] run_id={run_id} run_dir={run_dir.resolve()}")
        
        # Initialize artifact writer if Risk Targeting, Allocator v1, or Engine Policy v1 is enabled
        artifact_writer = None
        if (risk_targeting is not None or 
            components.get('allocator_v1_config', {}).get('enabled', False) or
            components.get('engine_policy_v1_config', {}).get('enabled', False)):
            from src.layers import create_artifact_writer
            # Use run_dir already created above
            artifact_writer = create_artifact_writer(run_dir)
            
            # Pass artifact writer to Risk Targeting layer
            if risk_targeting is not None:
                risk_targeting.artifact_writer = artifact_writer
                # Write params.json once at start
                risk_targeting._write_params()
        
        # Stage 5.5: Check allocator_v1 config and mode
        allocator_v1_config = components.get('allocator_v1_config', {})
        allocator_v1_enabled = allocator_v1_config.get('enabled', False)
        allocator_v1_mode = allocator_v1_config.get('mode', 'off')
        allocator_v1_profile = allocator_v1_config.get('profile', None)
        allocator_v1_precomputed_run_id = allocator_v1_config.get('precomputed_run_id', None)
        
        # Log allocator config at startup (critical for debugging)
        logger.info(
            f"[ExecSim] Allocator v1 config: "
            f"enabled={allocator_v1_enabled} "
            f"mode={allocator_v1_mode} "
            f"profile={allocator_v1_profile} "
            f"precomputed_run_id={allocator_v1_precomputed_run_id}"
        )
        
        # Precomputed mode: load scalars from prior run
        precomputed_scalars = None
        if allocator_v1_enabled and allocator_v1_mode == 'precomputed':
            try:
                from src.allocator.scalar_loader import load_precomputed_applied_scalars
                
                precomputed_run_id = allocator_v1_config.get('precomputed_run_id')
                if not precomputed_run_id:
                    raise ValueError(
                        "allocator_v1.mode='precomputed' requires precomputed_run_id to be set"
                    )
                
                scalar_filename = allocator_v1_config.get(
                    'precomputed_scalar_filename',
                    'allocator_risk_v1_applied.csv'
                )
                
                # Load scalars from baseline run
                baseline_run_dir = Path(out_dir) / precomputed_run_id
                precomputed_scalars = load_precomputed_applied_scalars(
                    baseline_run_dir,
                    scalar_filename
                )
                
                logger.info(
                    f"[ExecSim] ✅ Allocator v1 ENABLED (mode='precomputed'). "
                    f"Loaded {len(precomputed_scalars)} precomputed scalars from run: {precomputed_run_id}"
                )
                logger.info(
                    f"[ExecSim] Precomputed scalar range: "
                    f"{precomputed_scalars.index[0].strftime('%Y-%m-%d')} to "
                    f"{precomputed_scalars.index[-1].strftime('%Y-%m-%d')}"
                )
            except Exception as e:
                logger.error(f"[ExecSim] Failed to load precomputed scalars: {e}")
                logger.error("[ExecSim] Disabling allocator and continuing with baseline run")
                allocator_v1_enabled = False
                allocator_v1_mode = 'off'
        elif allocator_v1_enabled and allocator_v1_mode == 'compute':
            logger.info(
                "[ExecSim] ⚠️  Allocator v1 ENABLED (mode='compute'). "
                "Risk scalars will be computed on-the-fly (may be empty early due to warmup). "
                "Recommendation: Use mode='precomputed' for production."
            )
        elif allocator_v1_enabled:
            logger.warning(
                f"[ExecSim] ⚠️  Allocator v1 enabled=true but mode='{allocator_v1_mode}'. "
                f"Treating as 'off' (artifacts only, no scaling)."
            )
            allocator_v1_mode = 'off'
        else:
            logger.info(
                "[ExecSim] Allocator v1 is DISABLED (mode='off'). "
                "State/regime/risk artifacts will be computed and saved for analysis only."
            )
        
        # Phase 2: Check engine_policy_v1 config and materialize artifact in precomputed mode
        engine_policy_v1_config = components.get('engine_policy_v1_config', {})
        engine_policy_v1_enabled = engine_policy_v1_config.get('enabled', False)
        engine_policy_v1_mode = engine_policy_v1_config.get('mode', 'off')
        engine_policy_v1_precomputed_run_id = engine_policy_v1_config.get('precomputed_run_id', None)
        
        # Materialize policy artifact in precomputed mode (hard requirement for production-safe baselines)
        engine_policy_map = None
        if engine_policy_v1_enabled and engine_policy_v1_mode == 'precomputed':
            if not engine_policy_v1_precomputed_run_id:
                raise ValueError(
                    "engine_policy_v1.mode='precomputed' requires precomputed_run_id to be set"
                )
            
            source_run_dir = Path(out_dir) / engine_policy_v1_precomputed_run_id
            try:
                self._materialize_policy_artifact(
                    source_run_dir=source_run_dir,
                    target_run_dir=run_dir,
                    filename="engine_policy_applied_v1.csv"
                )
                logger.info(
                    f"[ExecSim] ✅ Engine Policy v1 ENABLED (mode='precomputed'). "
                    f"Materialized policy artifact from run: {engine_policy_v1_precomputed_run_id}"
                )
                
                # Load policy multipliers from materialized artifact
                policy_csv_path = run_dir / "engine_policy_applied_v1.csv"
                if policy_csv_path.exists():
                    policy_df = pd.read_csv(policy_csv_path, parse_dates=['rebalance_date'])
                    policy_df = policy_df.set_index('rebalance_date')
                    engine_policy_map = policy_df.to_dict(orient='index')
                    logger.info(
                        f"[ExecSim] Loaded {len(engine_policy_map)} precomputed policy multipliers from artifact"
                    )
            except FileNotFoundError as e:
                logger.error(f"[ExecSim] Failed to materialize policy artifact: {e}")
                raise  # Hard requirement: fail loudly for production-safe baselines
        elif engine_policy_v1_enabled:
            logger.info(
                f"[ExecSim] Engine Policy v1 ENABLED (mode='{engine_policy_v1_mode}'). "
                f"Policy artifacts will be computed/generated during run."
            )
        
        # Build rebalance schedule
        rebalance_dates = self._build_rebalance_dates(market, risk_vol, start, end)
        rebalance_dates = pd.to_datetime(rebalance_dates)
        
        if len(rebalance_dates) == 0:
            logger.error("[ExecSim] No valid rebalance dates, cannot run backtest")
            return {
                'equity_curve': pd.Series(dtype=float),
                'weights_panel': pd.DataFrame(),
                'signals_panel': pd.DataFrame(),
                'report': {}
            }
        
        # Phase 2: Compute engine policy in compute mode (before rebalance loop)
        # Note: engine_policy_map may already be loaded in precomputed mode above
        if engine_policy_v1_enabled and engine_policy_v1_mode == 'compute' and engine_policy_map is None:
            from src.agents.engine_policy_v1 import EnginePolicyV1
            
            logger.info("[ExecSim] ENGINE POLICY ENABLED: entering compute/apply path")
            
            # Ensure artifact_writer is created
            if artifact_writer is None:
                from src.layers import create_artifact_writer
                artifact_writer = create_artifact_writer(run_dir)
            
            # Instantiate EnginePolicyV1
            policy = EnginePolicyV1(
                config=engine_policy_v1_config,
                artifact_writer=artifact_writer
            )
            
            # Compute state (daily policy state for all enabled engines)
            state_df = policy.compute_state(market, start, end)
            
            # Compute applied multipliers (maps state to rebalance dates, writes artifact)
            if not state_df.empty:
                applied_multipliers_df = policy.compute_applied_multipliers(state_df, rebalance_dates)
                
                # Build lookup map for applying multipliers during rebalance loop
                # Convert pivot format to lookup map: date -> {trend_multiplier, vrp_multiplier}
                if not applied_multipliers_df.empty:
                    # Read the canonical pivot format artifact
                    canonical_df = policy._to_canonical_pivot_format(applied_multipliers_df)
                    canonical_df['rebalance_date'] = pd.to_datetime(canonical_df['rebalance_date'])
                    canonical_df = canonical_df.set_index('rebalance_date')
                    engine_policy_map = canonical_df.to_dict(orient='index')
                    
                    logger.info(
                        f"[ExecSim] Engine Policy v1: computed {len(applied_multipliers_df)} multiplier records, "
                        f"built lookup map with {len(engine_policy_map)} dates"
                    )
                else:
                    logger.warning("[ExecSim] Engine Policy v1: computed empty multipliers")
            else:
                logger.warning("[ExecSim] Engine Policy v1: computed empty state, skipping multipliers")
        
        # Phase 2.5: Validate sleeve→engine mapping and log audit (if policy enabled)
        if engine_policy_v1_enabled and engine_policy_map is not None:
            # Strategy is already extracted above, validate mapping
            if hasattr(strategy, 'strategies') and hasattr(strategy, 'weights'):
                # This is CombinedStrategy, validate all sleeves are mapped or exempt
                sleeve_mapping = ExecSim._get_sleeve_to_engine_mapping()
                exempt_sleeves = ExecSim._get_policy_exempt_sleeves()
                
                # Get enabled sleeves
                enabled_sleeves = {name for name, weight in strategy.weights.items() if weight > 0}
                
                # Validate mapping
                unmapped_sleeves = []
                mapped_sleeves = {}
                exempt_sleeves_found = []
                
                for sleeve_name in enabled_sleeves:
                    if sleeve_name in exempt_sleeves:
                        exempt_sleeves_found.append(sleeve_name)
                    elif sleeve_name in sleeve_mapping:
                        engine_name = sleeve_mapping[sleeve_name]
                        mapped_sleeves[sleeve_name] = engine_name
                    else:
                        unmapped_sleeves.append(sleeve_name)
                
                # Hard-fail if unmapped sleeves exist (Phase-3A strictness)
                if unmapped_sleeves:
                    raise ValueError(
                        f"[ExecSim] Policy enabled but unmapped sleeves found: {unmapped_sleeves}\n"
                        f"Sleeves must be either mapped to a policy engine or explicitly exempt.\n"
                        f"Add mapping in ExecSim._get_sleeve_to_engine_mapping() or exempt in ExecSim._get_policy_exempt_sleeves()."
                    )
                
                # Log mapping audit
                logger.info("[ExecSim] ========================================")
                logger.info("[ExecSim] ENGINE POLICY MAPPING AUDIT")
                logger.info("[ExecSim] ========================================")
                logger.info(f"[ExecSim] Enabled sleeves: {len(enabled_sleeves)}")
                logger.info(f"[ExecSim] Mapped to policy engines:")
                for sleeve_name, engine_name in sorted(mapped_sleeves.items()):
                    multiplier_col = f"{engine_name}_multiplier"
                    logger.info(f"[ExecSim]   {sleeve_name:30s} → {engine_name:10s} → {multiplier_col}")
                if exempt_sleeves_found:
                    logger.info(f"[ExecSim] Explicitly exempt (no policy):")
                    for sleeve_name in sorted(exempt_sleeves_found):
                        logger.info(f"[ExecSim]   {sleeve_name:30s} → (exempt)")
                logger.info("[ExecSim] ========================================")
                
                # Count rebalances with policy binding (multiplier < 1.0)
                if engine_policy_map:
                    binding_count = 0
                    for date, multipliers in engine_policy_map.items():
                        trend_mult = multipliers.get('trend_multiplier', 1.0)
                        vrp_mult = multipliers.get('vrp_multiplier', 1.0)
                        if trend_mult < 0.999 or vrp_mult < 0.999:
                            binding_count += 1
                    logger.info(f"[ExecSim] Policy binding summary: {binding_count}/{len(engine_policy_map)} rebalances with multiplier < 1.0")
        
        # Initialize tracking
        weights_history = []
        signals_history = []
        returns_history = []
        dates_history = []
        turnover_history = []
        macro_scaler_history = [] if macro_overlay is not None else None
        
        # Stage 4A: Tracking for optional allocator state features
        sleeve_signals_history = {}  # Track per-sleeve signals before blending
        
        # Stage 5: Tracking for risk scalar application
        risk_scalar_applied_history = []  # Track applied scalars at each rebalance
        risk_scalar_computed_history = []  # Track computed scalars (for diagnostics)
        weights_raw_history = []  # Track pre-scaling weights (for diagnostics)
        rebalance_dates_history = []  # Track dates when rebalances occurred
        
        # Risk Targeting governance tracking (Layer 5)
        rt_leverage_history = []  # Track leverage multipliers
        rt_current_vol_history = []  # Track computed current vol
        rt_weights_pre_history = []  # Track weights before RT
        rt_weights_post_history = []  # Track weights after RT
        rt_returns_available_history = []  # Track if returns were available
        rt_cov_available_history = []  # Track if cov was computable
        
        # Allocator v1 governance tracking (Layer 6)
        alloc_v1_state_computed_history = []  # Track if state was computed successfully
        alloc_v1_regime_history = []  # Track regime classifications
        alloc_v1_scalar_computed_history = []  # Track computed scalars
        alloc_v1_inputs_available_history = []  # Track if required inputs were available
        
        # Engine Policy: Track pre-policy and post-policy components for layered weights trace
        components_pre_policy_history = []  # Track components before policy application
        components_post_policy_history = []  # Track components after policy application
        
        prev_weights = None
        prev_risk_scalar = None  # Track previous scalar for lag
        
        # Get continuous returns for the full period (we'll slice by date as needed)
        # Use continuous returns for P&L calculation (no roll jumps in back-adjusted series)
        returns_cont = market.returns_cont
        
        if returns_cont.empty:
            logger.error("[ExecSim] No continuous returns data available")
            return {
                'equity_curve': pd.Series(dtype=float),
                'weights_panel': pd.DataFrame(),
                'signals_panel': pd.DataFrame(),
                'report': {}
            }
        
        # Filter by date range
        returns_df = returns_cont.copy()
        if start:
            start_dt = pd.to_datetime(start)
            returns_df = returns_df[returns_df.index >= start_dt]
        if end:
            end_dt = pd.to_datetime(end)
            returns_df = returns_df[returns_df.index <= end_dt]
        
        if returns_df.empty:
            logger.error("[ExecSim] No returns data available after filtering")
            return {
                'equity_curve': pd.Series(dtype=float),
                'weights_panel': pd.DataFrame(),
                'signals_panel': pd.DataFrame(),
                'report': {}
            }
        
        # Note: We no longer filter roll jumps because continuous returns are back-adjusted
        # and don't have roll jumps. The backward-panama adjustment removes price gaps
        # at roll points, so P&L is computed correctly from continuous returns.
        
        # Loop over rebalance dates
        for i, date in enumerate(rebalance_dates):
            date = self._to_ts(date)
            logger.debug(f"[ExecSim] Rebalance {i+1}/{len(rebalance_dates)}: {date}")
            
            try:
                # Step 1: Get strategy signals (with policy application if enabled)
                signals = None
                components_pre_policy = None
                components_post_policy = None
                
                # If policy enabled and CombinedStrategy, use components for policy application
                if (engine_policy_v1_enabled and engine_policy_map is not None and 
                    hasattr(strategy, 'strategies') and hasattr(strategy, 'weights')):
                    # Request components from CombinedStrategy
                    signals, components_pre_policy = strategy.signals(market, date, return_components=True)
                    
                    # Test 1: Blend equivalence check (first 3 rebalances only)
                    # Verify that sum(components) == blended to ensure we're not changing portfolio construction
                    if i < 3:
                        blended_from_components = pd.Series(0.0, index=market.universe)
                        for component_sigs in components_pre_policy.values():
                            blended_from_components = blended_from_components + component_sigs
                        # Ensure same index order
                        blended_from_components = blended_from_components.reindex(market.universe, fill_value=0.0)
                        signals_reindexed = signals.reindex(market.universe, fill_value=0.0)
                        
                        # Check equality (allow small numerical differences)
                        max_diff = (signals_reindexed - blended_from_components).abs().max()
                        if max_diff > 1e-6:
                            raise RuntimeError(
                                f"[ExecSim] Blend equivalence check FAILED at rebalance {i+1} (date {date}):\n"
                                f"max difference: {max_diff:.2e}\n"
                                f"signals from strategy.signals(): {signals_reindexed.head().to_dict()}\n"
                                f"signals from sum(components): {blended_from_components.head().to_dict()}\n"
                                f"This indicates components are not pre-weighted signals that can be summed.\n"
                                f"Policy application logic must use CombinedStrategy's internal blend method, not sum()."
                            )
                        else:
                            logger.debug(f"[ExecSim] Blend equivalence check PASSED at rebalance {i+1}: max_diff={max_diff:.2e}")
                    
                    # Apply policy multipliers to components
                    multipliers = engine_policy_map.get(date, {})
                    trend_mult = multipliers.get('trend_multiplier', 1.0)
                    vrp_mult = multipliers.get('vrp_multiplier', 1.0)
                    
                    components_post_policy = {}
                    sleeve_mapping = ExecSim._get_sleeve_to_engine_mapping()
                    for sleeve_name, component_sigs in components_pre_policy.items():
                        policy_engine = ExecSim._map_sleeve_to_policy_engine(sleeve_name, mapping=sleeve_mapping)
                        if policy_engine == 'trend':
                            components_post_policy[sleeve_name] = component_sigs * trend_mult
                        elif policy_engine == 'vrp':
                            components_post_policy[sleeve_name] = component_sigs * vrp_mult
                        else:
                            # No policy mapping, keep as-is
                            components_post_policy[sleeve_name] = component_sigs.copy()
                    
                    # Re-blend components (sum them, as CombinedStrategy does internally)
                    signals = pd.Series(0.0, index=market.universe)
                    for component_sigs in components_post_policy.values():
                        signals = signals + component_sigs
                    
                    # Track components for artifacts
                    components_pre_policy_history.append({
                        'date': date,
                        'components': {k: v.copy() for k, v in components_pre_policy.items()}
                    })
                    components_post_policy_history.append({
                        'date': date,
                        'components': {k: v.copy() for k, v in components_post_policy.items()}
                    })
                else:
                    # Policy not enabled or not CombinedStrategy, use normal path
                    signals = strategy.signals(market, date)
                
                # Stage 4A: Capture per-sleeve signals if CombinedStrategy
                if hasattr(strategy, 'strategies') and hasattr(strategy, 'weights'):
                    # This is a CombinedStrategy, capture individual sleeve signals
                    for sleeve_name, sleeve_strategy in strategy.strategies.items():
                        sleeve_weight = strategy.weights.get(sleeve_name, 0.0)
                        if sleeve_weight > 0:
                            try:
                                # Get individual sleeve signals (before blending)
                                import inspect
                                sig = inspect.signature(sleeve_strategy.signals)
                                if 'features' in sig.parameters:
                                    # Replicate the feature routing logic from CombinedStrategy
                                    features_dict = None
                                    if sleeve_name in ["tsmom", "tsmom_long", "long_momentum"]:
                                        features_dict = strategy.features.get("LONG_MOMENTUM")
                                    elif sleeve_name in ["tsmom_med", "medium_momentum"]:
                                        features_dict = strategy.features.get("MEDIUM_MOMENTUM")
                                    elif sleeve_name in ["tsmom_med_canonical", "canonical_medium_momentum"]:
                                        features_dict = strategy.features.get("CANONICAL_MEDIUM_MOMENTUM")
                                    elif sleeve_name in ["tsmom_short", "short_momentum"]:
                                        features_dict = strategy.features.get("SHORT_MOMENTUM")
                                    elif sleeve_name == "tsmom_multihorizon":
                                        features_dict = {
                                            "LONG_MOMENTUM": strategy.features.get("LONG_MOMENTUM", pd.DataFrame()),
                                            "MEDIUM_MOMENTUM": strategy.features.get("MEDIUM_MOMENTUM", pd.DataFrame()),
                                            "CANONICAL_MEDIUM_MOMENTUM": strategy.features.get("CANONICAL_MEDIUM_MOMENTUM", pd.DataFrame()),
                                            "SHORT_MOMENTUM": strategy.features.get("SHORT_MOMENTUM", pd.DataFrame()),
                                            "RESIDUAL_TREND": strategy.features.get("RESIDUAL_TREND", pd.DataFrame())
                                        }
                                    elif sleeve_name == "sr3_carry_curve":
                                        features_dict = strategy.features.get("SR3_CURVE")
                                    elif sleeve_name == "rates_curve":
                                        features_dict = strategy.features.get("RATES_CURVE")
                                    elif sleeve_name == "fx_commod_carry":
                                        features_dict = strategy.features.get("CARRY_FX_COMMOD")
                                    elif sleeve_name == "residual_trend":
                                        features_dict = strategy.features.get("RESIDUAL_TREND")
                                    elif sleeve_name == "persistence":
                                        features_dict = strategy.features.get("PERSISTENCE")
                                    
                                    if 'rates_features' in sig.parameters:
                                        sleeve_sigs = sleeve_strategy.signals(features_dict, date)
                                    else:
                                        sleeve_sigs = sleeve_strategy.signals(market, date, features=features_dict)
                                else:
                                    sleeve_sigs = sleeve_strategy.signals(market, date)
                                
                                # Store weighted sleeve signals
                                if sleeve_name not in sleeve_signals_history:
                                    sleeve_signals_history[sleeve_name] = []
                                sleeve_signals_history[sleeve_name].append((date, sleeve_sigs * sleeve_weight))
                            except Exception as e:
                                logger.debug(f"[ExecSim] Could not capture {sleeve_name} signals: {e}")
                                continue
                
                macro_k = 1.0
                if macro_overlay is not None:
                    macro_k = macro_overlay.scaler(market, date)
                    macro_scaler_history.append(macro_k)
                
                # Step 2: Apply vol-managed overlay on raw signals
                scaled_signals = overlay.scale(signals, market, date)
                
                # Step 2b: Apply macro scaler after vol targeting to preserve risk reduction
                if macro_overlay is not None:
                    scaled_signals = scaled_signals * macro_k
                
                # Step 3: Get covariance matrix and validity mask for allocator
                # Pass signals to ensure risk model uses same universe
                try:
                    cov = risk_vol.covariance(market, date, signals=scaled_signals)
                    mask = risk_vol.mask(market, date, signals=scaled_signals)
                except (ValueError, KeyError) as e:
                    # Skip dates where risk data is insufficient (warmup period)
                    if "Insufficient history" in str(e) or "need" in str(e).lower():
                        logger.debug(f"[ExecSim] Skipping {date}: {e}")
                        continue
                    else:
                        # Re-raise other errors
                        raise
                
                # Step 3b: Zero out signals for assets failing validity mask
                # This prevents allocator from wasting budget on NaNs that got imputed
                valid_symbols = mask.intersection(scaled_signals.index)
                if len(valid_symbols) < len(scaled_signals):
                    invalid_symbols = scaled_signals.index.difference(valid_symbols)
                    logger.debug(
                        f"[ExecSim] Zeroing {len(invalid_symbols)} invalid signals: {list(invalid_symbols)[:5]}"
                    )
                    scaled_signals = scaled_signals.copy()
                    scaled_signals.loc[invalid_symbols] = 0.0
                
                # Step 4: Allocate to final weights (Portfolio Construction - Layer 3)
                weights_raw = allocator.solve(scaled_signals, cov, weights_prev=prev_weights)
                
                # Step 5: Risk Targeting Layer (Layer 5: vol → leverage)
                weights_pre_rt = weights_raw.copy()  # Save pre-RT weights for artifacts
                rt_leverage = 1.0
                rt_current_vol = None
                rt_returns_available = False
                rt_cov_available = False
                
                if risk_targeting is not None:
                    # Get historical returns for volatility estimation
                    # Use returns up to (but not including) current date
                    if date in returns_df.index:
                        date_idx = returns_df.index.get_loc(date)
                        if date_idx > 0:
                            # Get returns up to previous day
                            returns_for_vol = returns_df.iloc[:date_idx]
                        else:
                            returns_for_vol = pd.DataFrame()
                    else:
                        # Find last date <= current date
                        valid_dates = returns_df.index[returns_df.index < date]
                        if len(valid_dates) > 0:
                            returns_for_vol = returns_df.loc[:valid_dates[-1]]
                        else:
                            returns_for_vol = pd.DataFrame()
                    
                    # Track returns availability for governance
                    rt_returns_available = not returns_for_vol.empty and returns_for_vol.notna().any().any()
                    
                    # Convert log returns to simple returns for Risk Targeting
                    # Risk Targeting expects simple returns (it will compute cov internally)
                    if not returns_for_vol.empty:
                        returns_simple = np.exp(returns_for_vol) - 1.0
                    else:
                        returns_simple = pd.DataFrame()
                    
                    # Try to compute current vol to check if cov is available
                    try:
                        rt_current_vol = risk_targeting.compute_portfolio_vol(
                            weights=weights_pre_rt,
                            returns=returns_simple,
                            date=date
                        )
                        rt_cov_available = rt_current_vol is not None and np.isfinite(rt_current_vol) and rt_current_vol > 0
                        if rt_cov_available:
                            rt_leverage = risk_targeting.compute_leverage(rt_current_vol)
                    except Exception as e:
                        logger.debug(f"[ExecSim] RT vol computation failed at {date}: {e}")
                        rt_cov_available = False
                    
                    # Scale weights using Risk Targeting
                    weights_raw = risk_targeting.scale_weights(
                        weights=weights_pre_rt,
                        returns=returns_simple,
                        date=date
                    )
                    
                    # Extract actual leverage from RT (recompute to get exact value)
                    # This ensures we capture the actual leverage used, even if initial computation failed
                    try:
                        if rt_current_vol is not None and np.isfinite(rt_current_vol):
                            rt_leverage = risk_targeting.compute_leverage(rt_current_vol)
                        else:
                            # Fallback: compute leverage from actual weight scaling ratio
                            # This captures the actual leverage applied even if vol computation failed
                            pre_rt_gross = weights_pre_rt.abs().sum()
                            post_rt_gross = weights_raw.abs().sum()
                            if pre_rt_gross > 1e-6:  # Avoid division by zero
                                rt_leverage = post_rt_gross / pre_rt_gross
                            else:
                                rt_leverage = 1.0
                    except Exception:
                        # Final fallback: compute from weight ratio
                        try:
                            pre_rt_gross = weights_pre_rt.abs().sum()
                            post_rt_gross = weights_raw.abs().sum()
                            if pre_rt_gross > 1e-6:
                                rt_leverage = post_rt_gross / pre_rt_gross
                            else:
                                rt_leverage = 1.0
                        except Exception:
                            rt_leverage = 1.0
                    
                    logger.debug(
                        f"[ExecSim] {date}: Risk Targeting applied. "
                        f"Pre-RT leverage: {weights_pre_rt.abs().sum():.2f}x, "
                        f"Post-RT leverage: {weights_raw.abs().sum():.2f}x, "
                        f"RT multiplier: {rt_leverage:.3f}x"
                    )
                else:
                    # No Risk Targeting: weights_pre_rt = weights_raw
                    pass
                
                # Track RT governance data - ensure we only append finite values
                # rt_leverage should always be finite (defaults to 1.0, computed from weights if needed)
                if rt_leverage is not None and np.isfinite(rt_leverage):
                    rt_leverage_history.append(rt_leverage)
                else:
                    rt_leverage_history.append(1.0)  # Fallback to 1.0 if somehow None/NaN
                
                # rt_current_vol can be None if computation failed, but we still track it
                rt_current_vol_history.append(rt_current_vol)
                rt_weights_pre_history.append(weights_pre_rt.copy())
                rt_weights_post_history.append(weights_raw.copy())
                rt_returns_available_history.append(rt_returns_available)
                rt_cov_available_history.append(rt_cov_available)
                
                # Stage 5.5: Apply risk scalar (mode-dependent) - Allocator Layer 6
                current_risk_scalar = 1.0  # Default: no scaling
                risk_scalar_applied = 1.0
                
                if allocator_v1_enabled and allocator_v1_mode == 'precomputed' and precomputed_scalars is not None:
                    # Precomputed mode: lookup scalar from loaded series
                    try:
                        # Reindex to current date with forward fill
                        scalar_at_date = precomputed_scalars.reindex([date], method='ffill')
                        
                        if scalar_at_date.isna().iloc[0]:
                            # Date before scalar series starts, use default
                            apply_missing_as = allocator_v1_config.get('apply_missing_scalar_as', 1.0)
                            risk_scalar_applied = apply_missing_as
                            logger.debug(
                                f"[ExecSim] {date}: No precomputed scalar available (pre-start), "
                                f"using default {apply_missing_as:.3f}"
                            )
                        else:
                            risk_scalar_applied = float(scalar_at_date.iloc[0])
                        
                        # In precomputed mode, computed=applied (no lag within Pass 2)
                        current_risk_scalar = risk_scalar_applied
                        
                        # Populate history for meta.json governance stats
                        alloc_v1_state_computed_history.append(True)
                        alloc_v1_regime_history.append('NORMAL') # Precomputed assumes NORMAL/Baseline implicitly
                        alloc_v1_scalar_computed_history.append(current_risk_scalar)
                        alloc_v1_inputs_available_history.append(False) # Inputs not checked in precomputed mode
                        
                        # Emit allocator artifacts for precomputed mode
                        if artifact_writer is not None:
                            profile_name = allocator_v1_config.get('profile', 'H')
                            
                            # For precomputed mode, we don't have regime, so use NORMAL
                            # (regime was determined in baseline run)
                            regime_df = pd.DataFrame({
                                'date': [date.strftime('%Y-%m-%d')],
                                'regime': ['NORMAL'],  # Precomputed: regime from baseline
                                'profile': [profile_name],
                            })
                            artifact_writer.write_csv("allocator/regime_series.csv", regime_df, mode="append")
                            
                            multiplier_df = pd.DataFrame({
                                'date': [date.strftime('%Y-%m-%d')],
                                'multiplier': [risk_scalar_applied],
                                'profile': [profile_name],
                            })
                            artifact_writer.write_csv("allocator/multiplier_series.csv", multiplier_df, mode="append")
                    except Exception as e:
                        logger.warning(f"[ExecSim] Failed to lookup precomputed scalar at {date}: {e}")
                        risk_scalar_applied = 1.0
                        current_risk_scalar = 1.0
                
                elif allocator_v1_enabled and allocator_v1_mode == 'compute':
                    # Compute mode: on-the-fly state/regime/risk computation
                    if len(returns_history) > 0 and len(dates_history) > 0:
                        try:
                            from src.allocator.state_v1 import AllocatorStateV1
                            from src.allocator.regime_v1 import RegimeClassifierV1
                            from src.allocator.risk_v1 import RiskTransformerV1
                            
                            # Build portfolio returns Series from history
                            portfolio_returns_so_far = pd.Series(returns_history, index=dates_history)
                            equity_so_far = (1 + portfolio_returns_so_far).cumprod()
                            
                            # Get asset returns through previous rebalance
                            asset_returns_up_to_date = returns_df.loc[:dates_history[-1]]
                            asset_returns_simple = np.exp(asset_returns_up_to_date) - 1.0
                            
                            # Compute allocator state
                            state_computed = False
                            inputs_available = (
                                not portfolio_returns_so_far.empty and
                                not equity_so_far.empty and
                                not asset_returns_simple.empty
                            )
                            
                            state_computer = AllocatorStateV1()
                            state_df = state_computer.compute(
                                portfolio_returns=portfolio_returns_so_far,
                                equity_curve=equity_so_far,
                                asset_returns=asset_returns_simple,
                                trend_unit_returns=None,
                                sleeve_returns=None
                            )
                            
                            state_computed = not state_df.empty and state_df.notna().any().any()
                            
                            if state_computed:
                                # Compute regime and risk
                                classifier = RegimeClassifierV1()
                                regime = classifier.classify(state_df)
                                
                                if not regime.empty:
                                    # Create transformer with profile from config
                                    profile_name = allocator_v1_config.get('profile', 'H')
                                    from src.allocator.risk_v1 import create_risk_transformer_from_profile
                                    transformer = create_risk_transformer_from_profile(profile_name)
                                    risk_scalars = transformer.transform(state_df, regime)
                                    
                                    if not risk_scalars.empty:
                                        current_risk_scalar = float(risk_scalars['risk_scalar'].iloc[-1])
                                        
                                        # Track governance data
                                        alloc_v1_state_computed_history.append(True)
                                        alloc_v1_regime_history.append(regime.iloc[-1] if not regime.empty else None)
                                        alloc_v1_scalar_computed_history.append(current_risk_scalar)
                                        alloc_v1_inputs_available_history.append(inputs_available)
                                        
                                        # Emit allocator artifacts
                                        if artifact_writer is not None:
                                            # Get profile name from config
                                            profile_name = allocator_v1_config.get('profile', 'H')
                                            
                                            # Write regime series
                                            regime_df = pd.DataFrame({
                                                'date': [date.strftime('%Y-%m-%d')],
                                                'regime': [regime.iloc[-1]],
                                                'profile': [profile_name],
                                            })
                                            artifact_writer.write_csv("allocator/regime_series.csv", regime_df, mode="append")
                                            
                                            # Write multiplier series
                                            multiplier_df = pd.DataFrame({
                                                'date': [date.strftime('%Y-%m-%d')],
                                                'multiplier': [current_risk_scalar],
                                                'profile': [profile_name],
                                            })
                                            artifact_writer.write_csv("allocator/multiplier_series.csv", multiplier_df, mode="append")
                        except Exception as e:
                            logger.warning(f"[ExecSim] Failed to compute risk scalar at {date}: {e}")
                            current_risk_scalar = 1.0
                            # Track failed computation
                            alloc_v1_state_computed_history.append(False)
                            alloc_v1_regime_history.append(None)
                            alloc_v1_scalar_computed_history.append(1.0)
                            alloc_v1_inputs_available_history.append(False)
                    
                    # Apply with 1-rebalance lag
                    risk_scalar_applied = prev_risk_scalar if prev_risk_scalar is not None else 1.0
                    prev_risk_scalar = current_risk_scalar
                
                # Store computed and applied scalars
                risk_scalar_computed_history.append(current_risk_scalar)
                risk_scalar_applied_history.append(risk_scalar_applied)
                weights_raw_history.append(weights_raw.copy())
                rebalance_dates_history.append(date)
                
                # Apply scalar to weights
                if allocator_v1_enabled and allocator_v1_mode in ['precomputed', 'compute'] and risk_scalar_applied < 1.0:
                    logger.info(
                        f"[ExecSim] {date}: Applying risk_scalar={risk_scalar_applied:.3f} "
                        f"(mode={allocator_v1_mode}, gross before: {weights_raw.abs().sum():.2f}x)"
                    )
                    weights = weights_raw * risk_scalar_applied
                    logger.info(f"[ExecSim] {date}: After scaling: {weights.abs().sum():.2f}x")
                else:
                    weights = weights_raw.copy()
                
                # Record signals and weights (weights are now scaled if enabled)
                signals_history.append(scaled_signals)
                weights_history.append(weights)
                
                # Compute turnover (if not first date)
                if prev_weights is not None:
                    turnover = self._compute_turnover(prev_weights, weights)
                    turnover_history.append(turnover)
                else:
                    turnover_history.append(weights.abs().sum())  # Initial turnover = gross leverage
                
                # Step 4: Compute returns for holding period (date to next rebalance or end)
                # Find next date for return calculation
                if i < len(rebalance_dates) - 1:
                    next_date = self._to_ts(rebalance_dates[i + 1])
                else:
                    # Last rebalance: hold until end
                    next_date = self._to_ts(returns_df.index[-1]) if date < returns_df.index[-1] else date
                
                # Get returns from current date to next date
                # Convention: close-to-close, so returns[date+1] is the first return we earn
                if date in returns_df.index:
                    date_idx = returns_df.index.get_loc(date)
                    
                    # If not the last date, compute returns for holding period
                    if date_idx < len(returns_df) - 1:
                        # Returns from date+1 to next_date (inclusive)
                        if next_date in returns_df.index:
                            next_idx = returns_df.index.get_loc(next_date)
                        else:
                            # Find last date <= next_date
                            valid_dates = returns_df.index[returns_df.index <= next_date]
                            if len(valid_dates) > 0:
                                next_idx = returns_df.index.get_loc(valid_dates[-1])
                            else:
                                next_idx = date_idx
                        
                        # Compute period returns (weights are fixed over [t, next_t))
                        period_returns = returns_df.iloc[date_idx + 1:next_idx + 1]
                        
                        # For log returns: sum is correct (additive)
                        # For simple returns: need to compound with prod
                        # Since we use log returns by default, sum is correct
                        # But handle both cases for robustness
                        if len(period_returns) > 0:
                            # Check if returns are log or simple by examining if they can be negative
                            # Log returns can be negative, simple returns typically > -1
                            # For safety, assume log returns (default) and use sum
                            holding_returns = period_returns.sum()
                        else:
                            holding_returns = pd.Series(0.0, index=returns_df.columns)
                        
                        # Compute portfolio return
                        port_ret = self._compute_portfolio_return(
                            weights,
                            holding_returns,
                            is_rebalance=True
                        )
                        
                        # Apply slippage to turnover on rebalance day
                        slippage_cost = (turnover_history[-1] * self.slippage_bps / 10000)
                        port_ret_net = port_ret - slippage_cost
                        
                        returns_history.append(port_ret_net)
                        dates_history.append(date)
                        
                        # Diagnostics: what-moved report
                        if prev_weights is not None:
                            weight_changes = (weights - prev_weights).abs().sort_values(ascending=False)
                            top_movers = weight_changes.head(5)
                            logger.info(
                                f"[ExecSim] {date}: port_ret={port_ret:.4f}, "
                                f"slippage={slippage_cost:.4f}, net_ret={port_ret_net:.4f}, "
                                f"turnover={turnover_history[-1]:.3f}, "
                                f"k={macro_k:.3f}, top_movers={dict(top_movers.head(3))}"
                            )
                        else:
                            logger.info(
                                f"[ExecSim] {date}: port_ret={port_ret:.4f}, "
                                f"slippage={slippage_cost:.4f}, net_ret={port_ret_net:.4f}, "
                                f"turnover={turnover_history[-1]:.3f}, k={macro_k:.3f}"
                            )
                
                # Update previous weights
                prev_weights = weights.copy()
                
            except Exception as e:
                logger.error(f"[ExecSim] Error on {date}: {e}")
                import traceback
                traceback.print_exc()
                raise
        
        # Handle Curve RV returns if enabled
        curve_rv_meta = components.get('curve_rv_meta')
        curve_rv_weight = components.get('curve_rv_weight', 0.0)
        curve_rv_returns = None
        
        if curve_rv_meta is not None and curve_rv_weight > 0:
            logger.info(f"[ExecSim] Computing Curve RV returns (weight={curve_rv_weight:.1%})...")
            try:
                curve_rv_returns = curve_rv_meta.compute_returns(market, start, end)
                logger.info(f"[ExecSim] Curve RV returns: n={len(curve_rv_returns)}, "
                           f"mean={curve_rv_returns.mean():.6f}, std={curve_rv_returns.std():.6f}")
            except Exception as e:
                logger.error(f"[ExecSim] Failed to compute Curve RV returns: {e}")
                curve_rv_returns = None

        vx_calendar_carry_returns = None
        vx_calendar_carry_meta = components.get('vx_calendar_carry_meta')
        if vx_calendar_carry_meta is not None:
            logger.info("[ExecSim] Computing VX calendar carry returns...")
            try:
                vx_calendar_carry_returns = vx_calendar_carry_meta.compute_returns(market, start, end)
                logger.info(f"[ExecSim] VX calendar carry returns: n={len(vx_calendar_carry_returns)}, "
                           f"mean={vx_calendar_carry_returns.mean():.6f}, std={vx_calendar_carry_returns.std():.6f}")
            except Exception as e:
                logger.error(f"[ExecSim] Failed to compute VX calendar carry returns: {e}")
                vx_calendar_carry_returns = None

        sr3_calendar_spread_carry_returns = None
        sr3_calendar_spread_carry_meta = components.get('sr3_calendar_spread_carry_meta')
        if sr3_calendar_spread_carry_meta is not None:
            logger.info("[ExecSim] Computing SR3 calendar spread carry returns...")
            try:
                sr3_calendar_spread_carry_returns = sr3_calendar_spread_carry_meta.compute_returns(market, start, end)
                logger.info(f"[ExecSim] SR3 calendar spread carry returns: n={len(sr3_calendar_spread_carry_returns)}, "
                           f"mean={sr3_calendar_spread_carry_returns.mean():.6f}, std={sr3_calendar_spread_carry_returns.std():.6f}")
            except Exception as e:
                logger.error(f"[ExecSim] Failed to compute SR3 calendar spread carry returns: {e}")
                sr3_calendar_spread_carry_returns = None
        
        # Build results
        logger.info(f"[ExecSim] Completed {len(dates_history)} holding periods")
        
        # Equity curve: cumulative sum of log returns (rebalance-frequency, for backward compatibility)
        if len(returns_history) > 0:
            equity_curve = pd.Series(returns_history, index=dates_history).cumsum()
            equity_curve = np.exp(equity_curve)  # Convert log to arithmetic
        else:
            equity_curve = pd.Series(dtype=float)
        
        # Weights panel
        if len(weights_history) > 0:
            weights_panel = pd.DataFrame(weights_history, index=rebalance_dates[:len(weights_history)])
        else:
            weights_panel = pd.DataFrame()
        
        # Signals panel
        if len(signals_history) > 0:
            signals_panel = pd.DataFrame(signals_history, index=rebalance_dates[:len(signals_history)])
        else:
            signals_panel = pd.DataFrame()
        
        # Macro scaler series
        if macro_scaler_history is not None and len(macro_scaler_history) > 0:
            macro_scaler_series = pd.Series(macro_scaler_history, index=rebalance_dates[:len(macro_scaler_history)])
        else:
            macro_scaler_series = pd.Series(dtype=float)
        
        # Compute daily equity curve for accurate metrics
        # This ensures metrics use the same equity curve as the CSV output
        equity_curve_for_metrics = equity_curve  # Default to rebalance-frequency
        if not weights_panel.empty and not returns_df.empty:
            # Forward-fill weights to daily dates
            weights_daily = weights_panel.reindex(returns_df.index).ffill().fillna(0.0)
            
            # Align columns
            common_symbols = weights_daily.columns.intersection(returns_df.columns)
            if len(common_symbols) > 0:
                weights_aligned = weights_daily[common_symbols]
                returns_aligned = returns_df[common_symbols]
                
                # Compute daily portfolio returns (log returns)
                portfolio_returns_log = (weights_aligned * returns_aligned).sum(axis=1)
                portfolio_returns_daily = np.exp(portfolio_returns_log) - 1.0
                
                # Add Curve RV / VX / SR3 calendar spread returns if enabled (blend by weight)
                vx_weight = components.get('vx_calendar_carry_weight', 0.0)
                sr3_spread_weight = components.get('sr3_calendar_spread_carry_weight', 0.0)
                blend_weight = curve_rv_weight + vx_weight + sr3_spread_weight
                if blend_weight > 0:
                    base_weight = 1.0 - blend_weight
                    blended = base_weight * portfolio_returns_daily
                    if curve_rv_returns is not None and curve_rv_weight > 0:
                        curve_rv_aligned = curve_rv_returns.reindex(portfolio_returns_daily.index).fillna(0.0)
                        blended = blended + curve_rv_weight * curve_rv_aligned
                        logger.info(f"[ExecSim] Added Curve RV returns to portfolio (weight={curve_rv_weight:.1%})")
                    if vx_calendar_carry_returns is not None and vx_weight > 0:
                        vx_aligned = vx_calendar_carry_returns.reindex(portfolio_returns_daily.index).fillna(0.0)
                        blended = blended + vx_weight * vx_aligned
                        logger.info(f"[ExecSim] Added VX calendar carry returns to portfolio (weight={vx_weight:.1%})")
                    if sr3_calendar_spread_carry_returns is not None and sr3_spread_weight > 0:
                        sr3_aligned = sr3_calendar_spread_carry_returns.reindex(portfolio_returns_daily.index).fillna(0.0)
                        blended = blended + sr3_spread_weight * sr3_aligned
                        logger.info(f"[ExecSim] Added SR3 calendar spread carry returns to portfolio (weight={sr3_spread_weight:.1%})")
                    portfolio_returns_daily = blended
                
                # Compute daily equity curve
                equity_daily = (1 + portfolio_returns_daily).cumprod()
                equity_daily.iloc[0] = 1.0
                
                # Filter to start from first rebalance date (actual trading start)
                first_rebalance_date = weights_panel.index[0]
                equity_daily_filtered = equity_daily[equity_daily.index >= first_rebalance_date].copy()
                if len(equity_daily_filtered) > 0:
                    # Recompute from first rebalance date to ensure clean start
                    portfolio_returns_from_start = portfolio_returns_daily[portfolio_returns_daily.index >= first_rebalance_date]
                    equity_daily_filtered = (1 + portfolio_returns_from_start).cumprod()
                    equity_daily_filtered.iloc[0] = 1.0
                    equity_curve_for_metrics = equity_daily_filtered
        
        # Compute metrics using daily equity curve (filtered from first rebalance date)
        report = self._compute_metrics(
            equity_curve_for_metrics,
            returns_history,
            weights_panel,
            turnover_history
        )
        
        logger.info(
            f"[ExecSim] Backtest complete: "
            f"CAGR={report.get('cagr', 0):.2%}, "
            f"Sharpe={report.get('sharpe', 0):.2f}, "
            f"MaxDD={report.get('max_drawdown', 0):.2%}"
        )
        
        # VX tradability diagnostic: log non-zero weight counts per VX symbol
        vx_symbols = ["VX1", "VX2", "VX3"]
        vx_counts = {}
        for sym in vx_symbols:
            if sym in weights_panel.columns:
                non_zero = (weights_panel[sym].abs() > 1e-9).sum()
                total = len(weights_panel)
                pct = 100.0 * non_zero / total if total > 0 else 0
                vx_counts[sym] = non_zero
                logger.info(
                    f"[ExecSim] VX tradability: {sym} non-zero weights: "
                    f"{non_zero}/{total} rebalances ({pct:.1f}%)"
                )
        if vx_counts:
            n1, n2, n3 = vx_counts.get('VX1', 0), vx_counts.get('VX2', 0), vx_counts.get('VX3', 0)
            logger.info(f"[ExecSim] VX leg summary: VX1 non-zero={n1}, VX2 non-zero={n2}, VX3 non-zero={n3}")
        
        # Save artifacts (run_id already generated above if needed)
        self._save_run_artifacts(
            run_id=run_id,
            out_dir=out_dir,
            equity_curve=equity_curve_for_metrics,  # Use daily equity curve for CSV
            weights_panel=weights_panel,
            returns_df=returns_df,
            start=start,
            end=end,
            components=components,
            market=market,
            sleeve_signals_history=sleeve_signals_history,  # Stage 4A
            curve_rv_returns=curve_rv_returns,
            vx_calendar_carry_returns=vx_calendar_carry_returns,
            sr3_calendar_spread_carry_returns=sr3_calendar_spread_carry_returns,
            risk_scalar_applied_history=risk_scalar_applied_history,  # Stage 5
            risk_scalar_computed_history=risk_scalar_computed_history,  # Stage 5
            weights_raw_history=weights_raw_history,  # Stage 5
            allocator_v1_enabled=allocator_v1_enabled,  # Stage 5
            allocator_v1_mode=allocator_v1_mode,  # Stage 5.5
            precomputed_scalars=precomputed_scalars,  # Stage 5.5
            strategy_profile=strategy_profile,
            config_path=config_path,
            # Risk Targeting governance tracking
            rt_leverage_history=rt_leverage_history,
            rt_current_vol_history=rt_current_vol_history,
            rt_returns_available_history=rt_returns_available_history,
            rt_cov_available_history=rt_cov_available_history,
            # Allocator v1 governance tracking
            alloc_v1_state_computed_history=alloc_v1_state_computed_history,
            alloc_v1_regime_history=alloc_v1_regime_history,
            alloc_v1_scalar_computed_history=alloc_v1_scalar_computed_history,
            alloc_v1_inputs_available_history=alloc_v1_inputs_available_history,
            # Dual metrics computation inputs
            rebalance_dates=rebalance_dates,
            returns_history=returns_history,
            turnover_history=turnover_history
        )
        
        # Stage 5: Print allocator summary
        if allocator_v1_enabled and len(risk_scalar_applied_history) > 0:
            scalars = pd.Series(risk_scalar_applied_history)
            n_scaled = (scalars < 1.0).sum()
            pct_scaled = n_scaled / len(scalars) * 100
            
            logger.info("\n" + "=" * 80)
            logger.info("Allocator v1 Summary")
            logger.info("=" * 80)
            logger.info(f"Risk scalars applied: {n_scaled}/{len(scalars)} rebalances ({pct_scaled:.1f}%)")
            logger.info(f"Average scalar: {scalars.mean():.3f}")
            logger.info(f"Min scalar: {scalars.min():.3f}")
            logger.info(f"Max scalar: {scalars.max():.3f}")
            
            # Top 5 largest de-risk events
            if n_scaled > 0:
                scaled_dates = [rebalance_dates_history[i] for i, s in enumerate(risk_scalar_applied_history) if s < 1.0]
                scaled_values = [s for s in risk_scalar_applied_history if s < 1.0]
                scaled_df = pd.DataFrame({'date': scaled_dates, 'scalar': scaled_values})
                scaled_df = scaled_df.sort_values('scalar').head(5)
                
                logger.info(f"\nTop 5 largest de-risk events:")
                for idx, row in scaled_df.iterrows():
                    reduction_pct = (1 - row['scalar']) * 100
                    logger.info(f"  {row['date'].strftime('%Y-%m-%d')}: scalar={row['scalar']:.3f} (−{reduction_pct:.1f}%)")
            logger.info("=" * 80)
        
        return {
            'equity_curve': equity_curve_for_metrics,  # Return daily equity curve for consistency
            'weights_panel': weights_panel,
            'signals_panel': signals_panel,
            'report': report,
            'macro_scaler': macro_scaler_series,
            'run_id': run_id
        }
    
    def _compute_metrics(
        self,
        equity_curve: pd.Series,
        returns_list: list,
        weights_panel: pd.DataFrame,
        turnover_history: list
    ) -> Dict:
        """
        Compute performance metrics.
        
        Returns dict with:
        - cagr: Compound annual growth rate
        - vol: Annualized volatility
        - sharpe: Sharpe ratio
        - max_drawdown: Maximum drawdown
        - hit_rate: Percentage of positive returns
        - avg_turnover: Average turnover per rebalance
        - avg_gross: Average gross leverage
        - avg_net: Average net exposure
        """
        if equity_curve.empty or len(returns_list) == 0:
            return {
                'cagr': 0.0,
                'vol': 0.0,
                'sharpe': 0.0,
                'max_drawdown': 0.0,
                'hit_rate': 0.0,
                'avg_turnover': 0.0,
                'avg_gross': 0.0,
                'avg_net': 0.0,
                'n_periods': 0
            }
        
        # CRITICAL: Compute returns from equity_curve (daily), not returns_list (rebalance-frequency)
        # This ensures Sharpe, vol, and drawdown use the same series as CAGR
        returns_daily = equity_curve.pct_change().dropna()
        
        # CAGR - use actual time period, not number of periods
        total_ret = equity_curve.iloc[-1] / equity_curve.iloc[0] - 1.0
        # Calculate actual years from first to last date
        n_days = (equity_curve.index[-1] - equity_curve.index[0]).days
        n_years = n_days / 365.25
        if n_years > 0:
            cagr = (1 + total_ret) ** (1 / n_years) - 1
        else:
            cagr = 0.0
        
        # Volatility (annualized) - use daily returns
        vol = returns_daily.std() * np.sqrt(252)  # Daily returns -> annualized
        
        # Sharpe ratio (rf = 0 for now)
        mean_ret = returns_daily.mean()
        sharpe = (mean_ret * 252) / vol if vol > 0 else 0.0
        
        # Maximum drawdown - use equity curve directly
        running_max = equity_curve.expanding().max()
        drawdown = (equity_curve - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Hit rate - use daily returns
        hit_rate = (returns_daily > 0).mean()
        
        # Average turnover
        avg_turnover = np.mean(turnover_history) if len(turnover_history) > 0 else 0.0
        
        # Average gross and net leverage
        if not weights_panel.empty:
            avg_gross = weights_panel.abs().sum(axis=1).mean()
            avg_net = weights_panel.sum(axis=1).abs().mean()
        else:
            avg_gross = 0.0
            avg_net = 0.0
        
        return {
            'cagr': cagr,
            'vol': vol,
            'sharpe': sharpe,
            'max_drawdown': max_drawdown,
            'hit_rate': hit_rate,
            'avg_turnover': avg_turnover,
            'avg_gross': avg_gross,
            'avg_net': avg_net,
            'n_periods': len(returns_list)
        }
    
    def to_parquet(self, results: Dict, outdir: str):
        """
        Save results to parquet files (optional).
        
        Args:
            results: Output from run()
            outdir: Output directory path
        """
        outpath = Path(outdir)
        outpath.mkdir(parents=True, exist_ok=True)
        
        # Save equity curve
        if not results['equity_curve'].empty:
            results['equity_curve'].to_frame('equity').to_parquet(
                outpath / 'equity_curve.parquet'
            )
            logger.info(f"[ExecSim] Saved equity curve to {outpath / 'equity_curve.parquet'}")
        
        # Save weights panel
        if not results['weights_panel'].empty:
            results['weights_panel'].to_parquet(outpath / 'weights_panel.parquet')
            logger.info(f"[ExecSim] Saved weights panel to {outpath / 'weights_panel.parquet'}")
        
        # Save signals panel
        if not results['signals_panel'].empty:
            results['signals_panel'].to_parquet(outpath / 'signals_panel.parquet')
            logger.info(f"[ExecSim] Saved signals panel to {outpath / 'signals_panel.parquet'}")
        
        # Save metrics report as JSON
        import json
        with open(outpath / 'report.json', 'w') as f:
            json.dump(results['report'], f, indent=2)
        logger.info(f"[ExecSim] Saved report to {outpath / 'report.json'}")
    
    def _save_run_artifacts(
        self,
        run_id: str,
        out_dir: str,
        equity_curve: pd.Series,
        weights_panel: pd.DataFrame,
        returns_df: pd.DataFrame,
        start: Union[str, datetime],
        end: Union[str, datetime],
        components: Dict,
        market,
        sleeve_signals_history: Optional[Dict] = None,
        risk_scalar_applied_history: Optional[list] = None,
        risk_scalar_computed_history: Optional[list] = None,
        weights_raw_history: Optional[list] = None,
        allocator_v1_enabled: bool = False,
        allocator_v1_mode: str = 'off',
        precomputed_scalars: Optional[pd.Series] = None,
        strategy_profile: Optional[str] = None,
        config_path: Optional[str] = None,
        # Risk Targeting governance tracking
        rt_leverage_history: Optional[list] = None,
        rt_current_vol_history: Optional[list] = None,
        rt_returns_available_history: Optional[list] = None,
        rt_cov_available_history: Optional[list] = None,
        # Allocator v1 governance tracking
        alloc_v1_state_computed_history: Optional[list] = None,
        alloc_v1_regime_history: Optional[list] = None,
        alloc_v1_scalar_computed_history: Optional[list] = None,
        alloc_v1_inputs_available_history: Optional[list] = None,
        # Dual metrics computation inputs
        rebalance_dates: Optional[pd.DatetimeIndex] = None,
        returns_history: Optional[list] = None,
        turnover_history: Optional[list] = None,
        curve_rv_returns: Optional[pd.Series] = None,
        vx_calendar_carry_returns: Optional[pd.Series] = None,
        sr3_calendar_spread_carry_returns: Optional[pd.Series] = None
    ):
        """
        Save run artifacts to disk for diagnostics.
        
        Args:
            run_id: Run identifier
            out_dir: Base directory for saving
            equity_curve: Equity curve Series
            weights_panel: Weights DataFrame
            returns_df: Daily asset returns DataFrame
            start: Start date
            end: End date
            components: Components dict (for metadata)
            market: MarketData instance (for universe)
        """
        run_dir = Path(out_dir) / run_id
        os.makedirs(run_dir, exist_ok=True)
        
        logger.info(f"[ExecSim] Saving artifacts to {run_dir}")
        
        # Compute daily portfolio returns and equity curve
        portfolio_returns_daily = None
        equity_daily = None
        
        if not weights_panel.empty and not returns_df.empty:
            # Forward-fill weights to daily dates
            weights_daily = weights_panel.reindex(returns_df.index).ffill().fillna(0.0)
            
            # Align columns (ensure weights and returns have same symbols)
            common_symbols = weights_daily.columns.intersection(returns_df.columns)
            if len(common_symbols) > 0:
                weights_aligned = weights_daily[common_symbols]
                returns_aligned = returns_df[common_symbols]
                
                # Compute daily portfolio returns: sum(weight * return) for each day
                # Note: returns_df contains log returns, so portfolio_returns is log returns
                # Convert to simple returns for diagnostics: r_simple = exp(r_log) - 1
                portfolio_returns_log = (weights_aligned * returns_aligned).sum(axis=1)
                portfolio_returns_daily = np.exp(portfolio_returns_log) - 1.0
                
                # Add Curve RV returns if enabled (for artifact saving)
                curve_rv_meta = components.get('curve_rv_meta')
                curve_rv_weight = components.get('curve_rv_weight', 0.0)
                if curve_rv_meta is not None and curve_rv_weight > 0:
                    try:
                        curve_rv_returns = curve_rv_meta.compute_returns(market, start, end)
                        # Align Curve RV returns to portfolio returns dates
                        common_dates = portfolio_returns_daily.index.intersection(curve_rv_returns.index)
                        if len(common_dates) > 0:
                            # Scale Curve RV returns by weight and add to portfolio
                            base_weight = 1.0 - curve_rv_weight
                            curve_rv_aligned = curve_rv_returns.loc[common_dates]
                            portfolio_returns_daily.loc[common_dates] = (
                                base_weight * portfolio_returns_daily.loc[common_dates] +
                                curve_rv_weight * curve_rv_aligned
                            )
                    except Exception as e:
                        logger.warning(f"[ExecSim] Failed to add Curve RV returns in artifact saving: {e}")
                
                # Compute equity curve: cumulative product (simple returns)
                # Start at 1.0
                equity_daily = (1 + portfolio_returns_daily).cumprod()
                equity_daily.iloc[0] = 1.0  # Ensure starting value is 1.0
        
        # 1. Portfolio returns (daily)
        if portfolio_returns_daily is not None:
            portfolio_returns_daily.name = 'ret'
            portfolio_returns_daily.to_csv(
                run_dir / 'portfolio_returns.csv',
                header=True
            )
        else:
            # Create empty file if no data
            pd.DataFrame(columns=['date', 'ret']).to_csv(
                run_dir / 'portfolio_returns.csv',
                index=False
            )
        
        # 2. Equity curve (daily)
        # CRITICAL: Only include dates from first rebalance date onwards
        # This ensures consistency between CSV and metrics calculation
        # Before first rebalance, weights are 0 so equity stays at 1.0
        if equity_daily is not None:
            # Get first rebalance date (actual start of trading)
            if not weights_panel.empty:
                first_rebalance_date = weights_panel.index[0]
                # Filter equity_daily to start from first rebalance date
                equity_daily_filtered = equity_daily[equity_daily.index >= first_rebalance_date].copy()
                # Normalize to start at 1.0 at first rebalance date (should already be 1.0, but ensure it)
                if len(equity_daily_filtered) > 0:
                    equity_daily_filtered.iloc[0] = 1.0
                    # Recompute from first rebalance date to ensure consistency
                    portfolio_returns_from_start = portfolio_returns_daily[portfolio_returns_daily.index >= first_rebalance_date]
                    if len(portfolio_returns_from_start) > 0:
                        equity_daily_filtered = (1 + portfolio_returns_from_start).cumprod()
                        equity_daily_filtered.iloc[0] = 1.0
            else:
                equity_daily_filtered = equity_daily
            
            equity_daily_filtered.name = 'equity'
            equity_daily_filtered.to_csv(
                run_dir / 'equity_curve.csv',
                header=True
            )
        elif not equity_curve.empty:
            # Fallback: use rebalance-frequency equity curve
            equity_curve.name = 'equity'
            equity_curve.to_csv(
                run_dir / 'equity_curve.csv',
                header=True
            )
        else:
            pd.DataFrame(columns=['date', 'equity']).to_csv(
                run_dir / 'equity_curve.csv',
                index=False
            )
        
        # 3. Asset returns (daily, all symbols in universe)
        # Convert from log returns to simple returns for consistency
        if not returns_df.empty:
            # Convert log returns to simple returns: r_simple = exp(r_log) - 1
            asset_returns_simple = np.exp(returns_df) - 1.0
            asset_returns_simple.to_csv(run_dir / 'asset_returns.csv')
        else:
            pd.DataFrame().to_csv(run_dir / 'asset_returns.csv')
        
        # 4. Weights (rebalance dates only)
        if not weights_panel.empty:
            weights_panel.to_csv(run_dir / 'weights.csv')
        else:
            pd.DataFrame().to_csv(run_dir / 'weights.csv')
        
        # 4A. Stage 5.5: Raw weights and scalar artifacts (if allocator enabled)
        if allocator_v1_enabled and weights_raw_history and len(weights_raw_history) > 0:
            # Save raw weights (before risk scalar applied)
            weights_raw_panel = pd.DataFrame(weights_raw_history, index=weights_panel.index)
            weights_raw_panel.to_csv(run_dir / 'weights_raw.csv')
            logger.info(f"[ExecSim] Saved weights_raw.csv (pre-scaling weights)")
            
            # Rename weights.csv to weights_scaled.csv for clarity
            if (run_dir / 'weights.csv').exists():
                import shutil
                shutil.copy(run_dir / 'weights.csv', run_dir / 'weights_scaled.csv')
                logger.info(f"[ExecSim] Saved weights_scaled.csv (post-scaling weights)")
            
            # Save computed and applied scalars at each rebalance
            if risk_scalar_computed_history and len(risk_scalar_computed_history) > 0:
                scalars_df = pd.DataFrame({
                    'risk_scalar_computed': risk_scalar_computed_history,
                    'risk_scalar_applied': risk_scalar_applied_history
                }, index=weights_panel.index)
                scalars_df.to_csv(run_dir / 'allocator_scalars_at_rebalances.csv', index_label='rebalance_date')
                logger.info(f"[ExecSim] Saved allocator_scalars_at_rebalances.csv")
            
            # Stage 5.5: Precomputed mode artifacts
            if allocator_v1_mode == 'precomputed' and precomputed_scalars is not None:
                # Save source scalars (full series from baseline run)
                precomputed_scalars.to_csv(run_dir / 'allocator_risk_v1_applied_source.csv', header=True)
                logger.info(f"[ExecSim] Saved allocator_risk_v1_applied_source.csv (loaded from baseline)")
                
                # Save used scalars (at rebalance dates only, after reindex/ffill)
                if risk_scalar_applied_history:
                    used_scalars_df = pd.DataFrame({
                        'risk_scalar_used': risk_scalar_applied_history
                    }, index=weights_panel.index)
                    used_scalars_df.to_csv(run_dir / 'allocator_risk_v1_applied_used.csv', index_label='rebalance_date')
                    logger.info(f"[ExecSim] Saved allocator_risk_v1_applied_used.csv (actually applied)")
                
                # Save metadata for precomputed mode
                allocator_v1_config = components.get('allocator_v1_config', {})
                precomputed_meta = {
                    'mode': 'precomputed',
                    'source_run_id': allocator_v1_config.get('precomputed_run_id'),
                    'source_filename': allocator_v1_config.get('precomputed_scalar_filename'),
                    'n_rebalances': len(weights_panel),
                    'n_scaled': sum(1 for s in risk_scalar_applied_history if s < 1.0),
                    'pct_scaled': sum(1 for s in risk_scalar_applied_history if s < 1.0) / len(risk_scalar_applied_history) * 100 if risk_scalar_applied_history else 0,
                    'mean_scalar': float(pd.Series(risk_scalar_applied_history).mean()) if risk_scalar_applied_history else 1.0,
                    'min_scalar': float(pd.Series(risk_scalar_applied_history).min()) if risk_scalar_applied_history else 1.0,
                    'max_scalar': float(pd.Series(risk_scalar_applied_history).max()) if risk_scalar_applied_history else 1.0,
                    'generated_at': datetime.now().isoformat()
                }
                
                with open(run_dir / 'allocator_precomputed_meta.json', 'w') as f:
                    json.dump(precomputed_meta, f, indent=2)
                
                logger.info(f"[ExecSim] Saved allocator_precomputed_meta.json")
        
        # 4B. Stage 4A: Trend Unit Returns (optional, for allocator state)
        trend_unit_returns_df = None
        if not weights_panel.empty and not returns_df.empty:
            try:
                logger.info("[ExecSim] Computing trend unit returns...")
                
                # Forward-fill weights to daily
                weights_daily = weights_panel.reindex(returns_df.index).ffill().fillna(0.0)
                
                # Align symbols
                common_symbols = weights_daily.columns.intersection(returns_df.columns)
                if len(common_symbols) > 0:
                    # Compute trend unit returns: weight[t-1] * return[t]
                    # Shift weights by 1 day (lagged weights)
                    weights_lagged = weights_daily[common_symbols].shift(1).fillna(0.0)
                    returns_aligned = returns_df[common_symbols]
                    
                    # Per-asset unit returns (still in log space)
                    # Convert to simple returns for interpretability
                    returns_simple = np.exp(returns_aligned) - 1.0
                    trend_unit_returns_df = weights_lagged * returns_simple
                    
                    # Save to CSV
                    trend_unit_returns_df.to_csv(run_dir / 'trend_unit_returns.csv')
                    logger.info(f"[ExecSim] Saved trend_unit_returns.csv: {len(trend_unit_returns_df)} rows, {len(common_symbols)} assets")
            except Exception as e:
                logger.warning(f"[ExecSim] Failed to compute trend_unit_returns: {e}")
        
        # 4B. Stage 4A: Sleeve Returns (optional, for allocator state and attribution)
        sleeve_returns_df = None
        if (sleeve_signals_history and 'strategy' in components and hasattr(components['strategy'], 'strategies')
                or curve_rv_returns is not None or vx_calendar_carry_returns is not None
                or sr3_calendar_spread_carry_returns is not None):
            try:
                logger.info("[ExecSim] Computing sleeve returns...")
                sleeve_returns_data = {}
                strategy = components.get('strategy')
                if sleeve_signals_history and strategy is not None and hasattr(strategy, 'strategies'):
                    # For each sleeve, compute its return contribution
                    for sleeve_name, sleeve_strategy in strategy.strategies.items():
                        sleeve_weight = strategy.weights.get(sleeve_name, 0.0)
                        if sleeve_weight > 0:
                            # Get the sleeve's signals (if captured)
                            if sleeve_name in sleeve_signals_history and len(sleeve_signals_history[sleeve_name]) > 0:
                                # Reconstruct sleeve signals as a DataFrame
                                sleeve_dates = [d for d, _ in sleeve_signals_history[sleeve_name]]
                                sleeve_sigs_list = [s for _, s in sleeve_signals_history[sleeve_name]]
                                
                                if len(sleeve_sigs_list) > 0:
                                    sleeve_sigs_df = pd.DataFrame(sleeve_sigs_list, index=sleeve_dates)
                                    
                                    # Forward-fill to daily frequency
                                    sleeve_sigs_daily = sleeve_sigs_df.reindex(returns_df.index).ffill().fillna(0.0)
                                    
                                    # Compute sleeve return as: sum(sleeve_signal[t-1] * asset_return[t])
                                    sleeve_sigs_lagged = sleeve_sigs_daily.shift(1).fillna(0.0)
                                    
                                    common_assets = sleeve_sigs_lagged.columns.intersection(returns_df.columns)
                                    if len(common_assets) > 0:
                                        sigs_aligned = sleeve_sigs_lagged[common_assets]
                                        rets_aligned = returns_df[common_assets]
                                        rets_simple = np.exp(rets_aligned) - 1.0
                                        sleeve_returns_data[sleeve_name] = (sigs_aligned * rets_simple).sum(axis=1)

                # Include precomputed sleeve returns for curve_rv and vx_calendar_carry (not from signals)
                if curve_rv_returns is not None and len(curve_rv_returns) > 0:
                    idx = returns_df.index if not returns_df.empty else curve_rv_returns.index
                    aligned = curve_rv_returns.reindex(idx).fillna(0.0)
                    sleeve_returns_data["sr3_curve_rv_meta"] = aligned
                if vx_calendar_carry_returns is not None and len(vx_calendar_carry_returns) > 0:
                    idx = returns_df.index if not returns_df.empty else vx_calendar_carry_returns.index
                    aligned = vx_calendar_carry_returns.reindex(idx).fillna(0.0)
                    sleeve_returns_data["vx_calendar_carry"] = aligned
                if sr3_calendar_spread_carry_returns is not None and len(sr3_calendar_spread_carry_returns) > 0:
                    idx = returns_df.index if not returns_df.empty else sr3_calendar_spread_carry_returns.index
                    aligned = sr3_calendar_spread_carry_returns.reindex(idx).fillna(0.0)
                    sleeve_returns_data["sr3_calendar_spread_carry"] = aligned

                if sleeve_returns_data:
                    sleeve_returns_df = pd.DataFrame(sleeve_returns_data)
                    sleeve_returns_df.to_csv(run_dir / 'sleeve_returns.csv')
                    logger.info(f"[ExecSim] Saved sleeve_returns.csv: {len(sleeve_returns_df)} rows, {len(sleeve_returns_data)} sleeves")
            except Exception as e:
                logger.warning(f"[ExecSim] Failed to compute sleeve_returns: {e}")
                import traceback
                traceback.print_exc()
        
        # 5. Allocator State v1 (optional, compute if we have the necessary data)
        allocator_state_success = False
        try:
            from src.allocator.state_v1 import AllocatorStateV1, LOOKBACKS
            from src.allocator.state_validate import validate_allocator_state_v1, validate_inputs_aligned
            
            if (portfolio_returns_daily is not None and 
                equity_daily_filtered is not None and 
                not returns_df.empty):
                
                logger.info("[ExecSim] Computing allocator state v1...")
                
                # Validate input alignment
                validate_inputs_aligned(
                    portfolio_returns=portfolio_returns_daily,
                    equity_curve=equity_daily_filtered,
                    asset_returns=asset_returns_simple
                )
                
                # Initialize allocator state computer
                allocator_state = AllocatorStateV1()
                
                # Compute state features
                # Stage 4A: Pass trend_unit_returns and sleeve_returns if available
                state_df = allocator_state.compute(
                    portfolio_returns=portfolio_returns_daily,
                    equity_curve=equity_daily_filtered,
                    asset_returns=asset_returns_simple,
                    trend_unit_returns=trend_unit_returns_df,
                    sleeve_returns=sleeve_returns_df
                )
                
                if not state_df.empty:
                    # Extract metadata from state.attrs
                    features_present = state_df.attrs.get('features_present', list(state_df.columns))
                    features_missing = state_df.attrs.get('features_missing', [])
                    rows_before = state_df.attrs.get('rows_before_dropna', len(state_df))
                    rows_after = state_df.attrs.get('rows_after_dropna', len(state_df))
                    rows_dropped = state_df.attrs.get('rows_dropped', 0)
                    required_features = state_df.attrs.get('required_features', [])
                    optional_features = state_df.attrs.get('optional_features', [])
                    
                    # Compute effective start shift
                    effective_start_shift_days = 0
                    effective_start = state_df.index[0].strftime('%Y-%m-%d')
                    effective_end = state_df.index[-1].strftime('%Y-%m-%d')
                    
                    requested_start_dt = pd.Timestamp(str(start))
                    effective_start_dt = pd.Timestamp(effective_start)
                    effective_start_shift_days = (effective_start_dt - requested_start_dt).days
                    
                    # Create metadata with canonical structure
                    state_meta = {
                        'allocator_state_version': AllocatorStateV1.VERSION,
                        'lookbacks': LOOKBACKS,
                        'required_features': required_features,
                        'optional_features': optional_features,
                        'features_present': features_present,
                        'features_missing': features_missing,
                        'rows_requested': rows_before,
                        'rows_valid': rows_after,
                        'rows_dropped': rows_dropped,
                        'effective_start_date': effective_start,
                        'effective_end_date': effective_end,
                        'requested_start_date': str(start),
                        'requested_end_date': str(end),
                        'effective_start_shift_days': effective_start_shift_days,
                        'generated_at': datetime.now().isoformat()
                    }
                    
                    # Validate state before saving
                    validate_allocator_state_v1(state_df, state_meta)
                    
                    # Save allocator state CSV
                    state_df.to_csv(run_dir / 'allocator_state_v1.csv')
                    
                    # Save metadata
                    with open(run_dir / 'allocator_state_v1_meta.json', 'w') as f:
                        json.dump(state_meta, f, indent=2)
                    
                    logger.info(
                        f"[ExecSim] ✓ Saved allocator_state_v1.csv: {len(state_df)} rows, "
                        f"{len(features_present)} features "
                        f"({len(required_features)} required, {len(features_present) - len(required_features)} optional)"
                    )
                    allocator_state_success = True
                    
                    # Stage 4B-C: Compute regime and risk scalars from allocator state
                    # These are ALWAYS computed and saved (independent of allocator_v1.enabled)
                    try:
                        from src.allocator.regime_v1 import RegimeClassifierV1
                        from src.allocator.risk_v1 import RiskTransformerV1
                        from src.allocator.regime_rules_v1 import get_default_thresholds
                        
                        logger.info("[ExecSim] Computing allocator regime v1...")
                        classifier = RegimeClassifierV1()
                        regime = classifier.classify(state_df)
                        
                        if not regime.empty:
                            # Save regime series
                            regime_df = regime.to_frame('regime')
                            regime_df.to_csv(run_dir / 'allocator_regime_v1.csv')
                            
                            # Compute regime statistics
                            regime_counts = regime.value_counts().sort_index()
                            regime_pct = (regime_counts / len(regime) * 100).round(1)
                            
                            # Compute transition counts
                            transitions = {}
                            for i in range(len(regime) - 1):
                                from_regime = regime.iloc[i]
                                to_regime = regime.iloc[i + 1]
                                key = f"{from_regime}->{to_regime}"
                                transitions[key] = transitions.get(key, 0) + 1
                            
                            # Compute max consecutive days per regime
                            max_consecutive = {}
                            current_regime = None
                            current_count = 0
                            
                            for r in regime:
                                if r == current_regime:
                                    current_count += 1
                                else:
                                    if current_regime is not None:
                                        max_consecutive[current_regime] = max(
                                            max_consecutive.get(current_regime, 0),
                                            current_count
                                        )
                                    current_regime = r
                                    current_count = 1
                            
                            if current_regime is not None:
                                max_consecutive[current_regime] = max(
                                    max_consecutive.get(current_regime, 0),
                                    current_count
                                )
                            
                            # Save regime metadata
                            regime_meta = {
                                'version': RegimeClassifierV1.VERSION,
                                'thresholds': get_default_thresholds(),
                                'regime_day_counts': regime_counts.to_dict(),
                                'regime_percentages': regime_pct.to_dict(),
                                'transition_counts': transitions,
                                'max_consecutive_days': max_consecutive,
                                'effective_start_date': regime.index[0].strftime('%Y-%m-%d'),
                                'effective_end_date': regime.index[-1].strftime('%Y-%m-%d'),
                                'generated_at': datetime.now().isoformat()
                            }
                            
                            with open(run_dir / 'allocator_regime_v1_meta.json', 'w') as f:
                                json.dump(regime_meta, f, indent=2)
                            
                            logger.info(
                                f"[ExecSim] ✓ Saved allocator_regime_v1.csv: {len(regime)} rows, "
                                f"distribution: {regime_counts.to_dict()}"
                            )
                            
                            # Stage 4C: Compute risk scalars from regime
                            logger.info("[ExecSim] Computing allocator risk v1...")
                            transformer = RiskTransformerV1()
                            risk_scalars = transformer.transform(state_df, regime)
                            
                            if not risk_scalars.empty:
                                # Save risk scalars
                                risk_scalars.to_csv(run_dir / 'allocator_risk_v1.csv')
                                
                                # Compute statistics
                                risk_scalar = risk_scalars['risk_scalar']
                                risk_stats = {
                                    'mean': float(risk_scalar.mean()),
                                    'std': float(risk_scalar.std()),
                                    'min': float(risk_scalar.min()),
                                    'max': float(risk_scalar.max()),
                                    'median': float(risk_scalar.median()),
                                    'q25': float(risk_scalar.quantile(0.25)),
                                    'q75': float(risk_scalar.quantile(0.75))
                                }
                                
                                # Compute risk scalar by regime
                                from src.allocator.risk_v1 import DEFAULT_REGIME_SCALARS, RISK_MIN, RISK_MAX
                                aligned_regime = regime.reindex(risk_scalar.index)
                                risk_by_regime = {}
                                for regime_name in DEFAULT_REGIME_SCALARS.keys():
                                    regime_mask = aligned_regime == regime_name
                                    if regime_mask.any():
                                        risk_by_regime[regime_name] = {
                                            'mean': float(risk_scalar[regime_mask].mean()),
                                            'min': float(risk_scalar[regime_mask].min()),
                                            'max': float(risk_scalar[regime_mask].max()),
                                            'count': int(regime_mask.sum())
                                        }
                                
                                # Save risk metadata
                                risk_meta = {
                                    'version': RiskTransformerV1.VERSION,
                                    'regime_scalar_mapping': DEFAULT_REGIME_SCALARS,
                                    'smoothing_alpha': transformer.smoothing_alpha,
                                    'smoothing_half_life': transformer.smoothing_half_life,
                                    'risk_bounds': [RISK_MIN, RISK_MAX],
                                    'risk_scalar_stats': risk_stats,
                                    'risk_scalar_by_regime': risk_by_regime,
                                    'effective_start_date': risk_scalar.index[0].strftime('%Y-%m-%d'),
                                    'effective_end_date': risk_scalar.index[-1].strftime('%Y-%m-%d'),
                                    'generated_at': datetime.now().isoformat()
                                }
                                
                                with open(run_dir / 'allocator_risk_v1_meta.json', 'w') as f:
                                    json.dump(risk_meta, f, indent=2)
                                
                                logger.info(
                                    f"[ExecSim] ✓ Saved allocator_risk_v1.csv: {len(risk_scalar)} rows, "
                                    f"mean={risk_stats['mean']:.3f}, range=[{risk_stats['min']:.3f}, {risk_stats['max']:.3f}]"
                                )
                                
                                # Stage 5: Create applied risk_scalar series (lagged, at rebalance dates)
                                # Reindex risk_scalar to rebalance dates (forward fill to get most recent value)
                                # Then shift by 1 to create 1-bar lag (use yesterday's scalar today)
                                risk_scalar_at_rebalances = risk_scalar.reindex(weights_panel.index, method='ffill')
                                # Fill any remaining NaNs with 1.0 (no scaling for early rebalances before state available)
                                risk_scalar_at_rebalances = risk_scalar_at_rebalances.fillna(1.0)
                                risk_scalar_applied = risk_scalar_at_rebalances.shift(1, fill_value=1.0)
                                
                                # Create DataFrame
                                risk_scalar_applied_df = risk_scalar_applied.to_frame('risk_scalar_applied')
                                risk_scalar_applied_df.to_csv(run_dir / 'allocator_risk_v1_applied.csv')
                                
                                # Compute statistics for applied scalars
                                applied_stats = {
                                    'mean': float(risk_scalar_applied.mean()),
                                    'std': float(risk_scalar_applied.std()),
                                    'min': float(risk_scalar_applied.min()),
                                    'max': float(risk_scalar_applied.max()),
                                    'median': float(risk_scalar_applied.median()),
                                    'n_rebalances': len(risk_scalar_applied),
                                    'n_scaled': int((risk_scalar_applied < 1.0).sum()),
                                    'pct_scaled': float((risk_scalar_applied < 1.0).sum() / len(risk_scalar_applied) * 100)
                                }
                                
                                # Save metadata for applied scalars
                                applied_meta = {
                                    'version': RiskTransformerV1.VERSION,
                                    'description': 'Risk scalars that would be applied to weights (1-rebalance lag)',
                                    'lag_convention': '1-rebalance lag (apply risk_scalar[t-1] to weights[t])',
                                    'construction': 'Reindex daily risk_scalar to rebalance dates (forward fill), then shift by 1',
                                    'applied_stats': applied_stats,
                                    'first_rebalance': weights_panel.index[0].strftime('%Y-%m-%d'),
                                    'last_rebalance': weights_panel.index[-1].strftime('%Y-%m-%d'),
                                    'generated_at': datetime.now().isoformat()
                                }
                                
                                with open(run_dir / 'allocator_risk_v1_applied_meta.json', 'w') as f:
                                    json.dump(applied_meta, f, indent=2)
                                
                                logger.info(
                                    f"[ExecSim] ✓ Saved allocator_risk_v1_applied.csv: {len(risk_scalar_applied)} rebalances, "
                                    f"mean={applied_stats['mean']:.3f}, "
                                    f"scaled {applied_stats['n_scaled']}/{applied_stats['n_rebalances']} times "
                                    f"({applied_stats['pct_scaled']:.1f}%)"
                                )
                            else:
                                logger.warning("[ExecSim] ⚠️  Risk transformation returned empty DataFrame")
                        else:
                            logger.warning("[ExecSim] ⚠️  Regime classification returned empty series")
                    except Exception as e:
                        logger.error(f"[ExecSim] ❌ Failed to compute regime/risk: {e}")
                        import traceback
                        logger.error(traceback.format_exc())
                else:
                    logger.warning("[ExecSim] ⚠️  Allocator state computation returned empty DataFrame")
        except Exception as e:
            # Fail soft but loud: write error JSON and log red flag
            import traceback
            error_traceback = traceback.format_exc()
            
            error_info = {
                'error': str(e),
                'traceback': error_traceback,
                'inputs_present': {
                    'portfolio_returns': portfolio_returns_daily is not None,
                    'equity_curve': equity_daily_filtered is not None,
                    'asset_returns': not returns_df.empty
                },
                'generated_at': datetime.now().isoformat()
            }
            
            # Write error JSON
            with open(run_dir / 'allocator_state_v1_error.json', 'w') as f:
                json.dump(error_info, f, indent=2)
            
            logger.error(f"[ExecSim] ❌ Failed to compute allocator state v1: {e}")
            logger.error(f"[ExecSim] Error details written to allocator_state_v1_error.json")
        
        # 6. Meta JSON
        strategy_name = "unknown"
        if 'strategy' in components:
            strategy = components['strategy']
            if hasattr(strategy, '__class__'):
                strategy_name = strategy.__class__.__name__
        
        universe = []
        if hasattr(market, 'universe'):
            universe = list(market.universe) if market.universe else []
        
        # Extract source run IDs for precomputed mode (for auditability)
        allocator_v1_config = components.get('allocator_v1_config', {})
        engine_policy_v1_config = components.get('engine_policy_v1_config', {})
        
        allocator_source_run_id = None
        engine_policy_source_run_id = None
        
        if allocator_v1_enabled and allocator_v1_mode == 'precomputed':
            allocator_source_run_id = allocator_v1_config.get('precomputed_run_id')
        
        if components.get('engine_policy_v1_config', {}).get('enabled', False):
            engine_policy_v1_mode = components.get('engine_policy_v1_config', {}).get('mode', 'off')
            if engine_policy_v1_mode == 'precomputed':
                engine_policy_source_run_id = engine_policy_v1_config.get('precomputed_run_id')
        
        # Compute effective start date (first rebalance date)
        effective_start_date = None
        if not weights_panel.empty:
            effective_start_date = str(weights_panel.index[0])
            
        # Phase 3A: Determine Evaluation Start Date (Governance)
        # Approach: evaluation_start_date = max(policy_start, rt_start, alloc_start)
        # Each stage's effective start is the first date where it's effective (non-inert)
        evaluation_start_date = None
        policy_effective_start = None
        rt_effective_start = None
        alloc_effective_start = None
        
        try:
            n_points = len(rt_leverage_history) # Should match rebalance_dates length
            
            # 1. Policy Effective Start
            # Policy is effective when data is present and not-all-NaN
            # For simplicity, we assume policy is effective from the start if enabled
            # (in practice, policy features become available very early in the backtest)
            # If you have explicit policy multiplier history, check for first non-1.0 multiplier
            policy_effective_start = rebalance_dates[0] if len(rebalance_dates) > 0 else None
            
            # 2. Risk Targeting Effective Start
            # RT is effective when covariance is available (sufficient lookback)
            rt_enabled = components.get('risk_targeting') is not None
            if rt_enabled and len(rt_cov_available_history) > 0:
                for i in range(n_points):
                    if i < len(rt_cov_available_history) and rt_cov_available_history[i]:
                        rt_effective_start = rebalance_dates[i]
                        logger.info(f"[ExecSim] RT Effective Start: {rt_effective_start.strftime('%Y-%m-%d')} (index {i})")
                        break
            
            # 3. Allocator Effective Start
            # Allocator is effective when scalars are computed/available
            if allocator_v1_enabled and allocator_v1_mode != 'off':
                for i in range(n_points):
                    has_scalars = i < len(alloc_v1_scalar_computed_history) and alloc_v1_scalar_computed_history[i] is not None
                    
                    # Mode-specific validity
                    is_effective = False
                    if allocator_v1_mode == 'compute':
                        # In compute mode, must have inputs AND scalars
                        has_inputs = i < len(alloc_v1_inputs_available_history) and alloc_v1_inputs_available_history[i]
                        is_effective = has_inputs and has_scalars
                    elif allocator_v1_mode == 'precomputed':
                        # In precomputed mode, MUST have scalars (from source) AND valid source ID
                        is_effective = has_scalars and allocator_source_run_id is not None
                    
                    if is_effective:
                        alloc_effective_start = rebalance_dates[i]
                        logger.info(f"[ExecSim] Allocator Effective Start: {alloc_effective_start.strftime('%Y-%m-%d')} (index {i}, mode={allocator_v1_mode})")
                        break
            
            # Compute evaluation_start_date as max of all enabled stages
            stage_starts = []
            if policy_effective_start is not None:
                stage_starts.append(policy_effective_start)
            if rt_effective_start is not None:
                stage_starts.append(rt_effective_start)
            if alloc_effective_start is not None:
                stage_starts.append(alloc_effective_start)
            
            if stage_starts:
                evaluation_start_date = max(stage_starts)
                logger.info(
                    f"[ExecSim] Evaluation Start Date: {evaluation_start_date.strftime('%Y-%m-%d')} "
                    f"(max of policy={policy_effective_start.strftime('%Y-%m-%d') if policy_effective_start else 'N/A'}, "
                    f"rt={rt_effective_start.strftime('%Y-%m-%d') if rt_effective_start else 'N/A'}, "
                    f"alloc={alloc_effective_start.strftime('%Y-%m-%d') if alloc_effective_start else 'N/A'})"
                )
            else:
                logger.warning("[ExecSim] Could not determine evaluation start date (no enabled governance stages?)")

        except Exception as e:
             logger.error(f"[ExecSim] Error determining evaluation start date: {e}")
        
        # Compute Dual Metrics
        # 1. Full Run Metrics (Risk Context)
        # NOTE: equity_curve starts from effective_start_date (first rebalance), not data_start_date
        # This is because weights are 0 / undefined before first rebalance
        # So "full run" here means "full backtest run from first weights application"
        metrics_full = self._compute_metrics(
             equity_curve,
             returns_history, # rebalance freq, less accurate for sharpe but passed for n_periods
             weights_panel,
             turnover_history
        )
        # Note: _compute_metrics uses daily equity_curve internally for CAGR/Sharpe/DD
        
        # 2. Evaluation Window Metrics (Authoritative Performance)
        metrics_eval = {}
        if evaluation_start_date is not None:
            eval_start_ts = pd.Timestamp(evaluation_start_date)
            
            # Filter daily equity curve to evaluation window
            # equity_curve is already daily (passed from run method)
            if not equity_curve.empty:
                 equity_eval = equity_curve[equity_curve.index >= eval_start_ts].copy()
                 if not equity_eval.empty:
                     # Normalize to 1.0 at start of evaluation
                     # Determine scalar to rebase: first value should become 1.0
                     # Rebase: series / series[0]
                     equity_eval = equity_eval / equity_eval.iloc[0]
                     
                     # Filter weights and turnover (rebalance frequency)
                     # Find index in rebalance_dates corresponding to evaluation_start_date
                     # rebalance_dates is pd.DatetimeIndex
                     eval_start_idx = -1
                     if evaluation_start_date in rebalance_dates:
                         eval_start_idx = rebalance_dates.get_loc(evaluation_start_date)
                     
                     if eval_start_idx >= 0:
                         # Slice history lists from this index
                         # returns_history is per-holding-period return
                         returns_hist_eval = returns_history[eval_start_idx:] if eval_start_idx < len(returns_history) else []
                         turnover_hist_eval = turnover_history[eval_start_idx:] if eval_start_idx < len(turnover_history) else []
                         
                         # Weights panel slicing
                         weights_eval = weights_panel.loc[evaluation_start_date:]
                         
                         metrics_eval = self._compute_metrics(
                             equity_eval,
                             returns_hist_eval,
                             weights_eval,
                             turnover_hist_eval
                         )
        
        if not metrics_eval:
             # Fallback if evaluation window invalid or empty
             metrics_eval = {k: 0.0 for k in metrics_full.keys()}
             metrics_eval['n_periods'] = 0
        
        # Compute config hash for reproducibility
        config_hash = None
        if config_path:
            try:
                import hashlib
                with open(config_path, 'rb') as f:
                    config_bytes = f.read()
                    config_hash = hashlib.sha256(config_bytes).hexdigest()[:16]
            except Exception as e:
                logger.warning(f"[ExecSim] Could not compute config hash: {e}")
        
        # Load full config for meta.json (needed for diagnostics governance)
        full_config = {}
        if config_path:
            try:
                import yaml
                with open(config_path, 'r', encoding='utf-8') as f:
                    full_config = yaml.safe_load(f) or {}
            except Exception as e:
                logger.warning(f"[ExecSim] Could not load config for meta.json: {e}")
        
        # Check if dates match canonical window
        from src.utils.canonical_window import load_canonical_window
        canonical_start, canonical_end = load_canonical_window()
        is_canonical_window = (str(start) == canonical_start and str(end) == canonical_end)
        
        # Document policy features in meta (for Phase 3A governance)
        policy_features_meta = {}
        if hasattr(market, 'policy_features') and isinstance(market.policy_features, dict):
            # Document which policy features are present (data-driven, not file-based)
            for feature_name in ['gamma_stress_proxy', 'vx_backwardation', 'vrp_stress_proxy']:
                if feature_name in market.policy_features:
                    feature_series = market.policy_features[feature_name]
                    # Check if feature has data (not all NaN)
                    has_data = not feature_series.empty and feature_series.notna().any()
                    policy_features_meta[feature_name] = {
                        'present': True,
                        'has_data': bool(has_data),
                        'n_dates': len(feature_series) if not feature_series.empty else 0,
                        'n_valid': int(feature_series.notna().sum()) if not feature_series.empty else 0
                    }
                else:
                    policy_features_meta[feature_name] = {
                        'present': False,
                        'has_data': False,
                        'n_dates': 0,
                        'n_valid': 0
                    }
        
        # Document Risk Targeting governance (Layer 5)
        risk_targeting_meta = {}
        risk_targeting = components.get('risk_targeting')
        if risk_targeting is not None:
            # Check inputs availability from history
            rt_returns_available = bool(any(rt_returns_available_history) if rt_returns_available_history and len(rt_returns_available_history) > 0 else False)
            rt_cov_available = bool(any(rt_cov_available_history) if rt_cov_available_history and len(rt_cov_available_history) > 0 else False)
            rt_has_data = bool(rt_returns_available or rt_cov_available)
            
            # Check if RT had teeth (leverage != 1.0 at least once)
            rt_has_teeth = False
            if rt_leverage_history and len(rt_leverage_history) > 0:
                leverage_array = np.array([x for x in rt_leverage_history if x is not None and np.isfinite(x)])
                if len(leverage_array) > 0:
                    rt_has_teeth = bool(np.any(np.abs(leverage_array - 1.0) > 1e-12))
            
            # Compute stats - ensure we have finite values
            rt_multiplier_stats = {}
            if rt_leverage_history and len(rt_leverage_history) > 0:
                leverage_array = np.array([x for x in rt_leverage_history if x is not None and np.isfinite(x)])
                if len(leverage_array) > 0:
                    rt_multiplier_stats = {
                        'p5': float(np.percentile(leverage_array, 5)),
                        'p50': float(np.percentile(leverage_array, 50)),
                        'p95': float(np.percentile(leverage_array, 95)),
                        'at_cap': float(np.sum(leverage_array >= risk_targeting.leverage_cap - 1e-12) / len(leverage_array) * 100),
                        'at_floor': float(np.sum(leverage_array <= risk_targeting.leverage_floor + 1e-12) / len(leverage_array) * 100)
                    }
                else:
                    # If all values filtered out, stats should be empty dict (not NaN)
                    rt_multiplier_stats = {}
            
            rt_vol_stats = {}
            if rt_current_vol_history and len(rt_current_vol_history) > 0:
                vol_array = np.array([x for x in rt_current_vol_history if x is not None and np.isfinite(x)])
                if len(vol_array) > 0:
                    rt_vol_stats = {
                        'p50': float(np.percentile(vol_array, 50)),
                        'p95': float(np.percentile(vol_array, 95))
                    }
                else:
                    # If all values filtered out, stats should be empty dict (not NaN)
                    rt_vol_stats = {}
            
            risk_targeting_meta = {
                'enabled': True,
                'inputs_present': {
                    'asset_returns_df': {
                        'present': True,  # Always present if RT is enabled (from market)
                        'has_data': rt_returns_available
                    },
                    'cov_matrix': {
                        'present': False,  # Not pre-computed, computed from returns
                        'has_data': rt_cov_available
                    },
                    'weights_pre_rt': {
                        'present': True,  # Always present if RT is enabled
                        'has_data': bool(len(rt_leverage_history) > 0 if rt_leverage_history else False)  # If we computed leverage, weights were present
                    }
                },
                'inputs_missing': not rt_has_data,
                'effective': rt_has_data,
                'has_teeth': rt_has_teeth,
                'multiplier_stats': rt_multiplier_stats,
                'vol_stats': rt_vol_stats,
                'n_rebalances': len(rt_leverage_history) if rt_leverage_history else 0
            }
        else:
            risk_targeting_meta = {
                'enabled': False,
                'inputs_present': {},
                'inputs_missing': False,
                'effective': False,
                'has_teeth': False
            }
        
        # Document Allocator v1 governance (Layer 6)
        allocator_v1_meta = {}
        if allocator_v1_enabled:
            # Check inputs availability
            alloc_v1_inputs_available = any(alloc_v1_inputs_available_history) if alloc_v1_inputs_available_history and len(alloc_v1_inputs_available_history) > 0 else False
            alloc_v1_state_computed = any(alloc_v1_state_computed_history) if alloc_v1_state_computed_history and len(alloc_v1_state_computed_history) > 0 else False
            
            # Check if allocator had teeth (scalar < 1.0 at least once)
            alloc_v1_has_teeth = False
            if alloc_v1_scalar_computed_history:
                scalar_array = np.array([x for x in alloc_v1_scalar_computed_history if x is not None and np.isfinite(x)])
                if len(scalar_array) > 0:
                    alloc_v1_has_teeth = np.any(scalar_array < 1.0 - 1e-6)
            
            # Compute regime distribution
            regime_dist = {}
            if alloc_v1_regime_history and len(alloc_v1_regime_history) > 0:
                regimes = [r for r in alloc_v1_regime_history if r is not None]
                if regimes:
                    from collections import Counter
                    regime_counts = Counter(regimes)
                    total = len(regimes)
                    regime_dist = {k: float(v / total * 100) for k, v in regime_counts.items()}
            
            # Compute scalar stats
            scalar_stats = {}
            if alloc_v1_scalar_computed_history and len(alloc_v1_scalar_computed_history) > 0:
                scalar_array = np.array([x for x in alloc_v1_scalar_computed_history if x is not None and np.isfinite(x)])
                if len(scalar_array) > 0:
                    scalar_stats = {
                        'p5': float(np.percentile(scalar_array, 5)),
                        'p50': float(np.percentile(scalar_array, 50)),
                        'p95': float(np.percentile(scalar_array, 95)),
                        'at_min': float(np.sum(scalar_array <= 0.25 + 1e-6) / len(scalar_array) * 100)
                    }
            
            allocator_v1_meta = {
                'enabled': True,
                'mode': allocator_v1_mode,
                'profile': allocator_v1_config.get('profile', 'H'),
                'inputs_present': {
                    'portfolio_returns': {
                        'present': True,  # Always present if allocator enabled
                        'has_data': alloc_v1_inputs_available
                    },
                    'equity_curve': {
                        'present': True,
                        'has_data': alloc_v1_inputs_available
                    },
                    'asset_returns': {
                        'present': True,
                        'has_data': alloc_v1_inputs_available
                    }
                },
                'inputs_missing': not alloc_v1_inputs_available,
                'state_computed': alloc_v1_state_computed,
                'effective': alloc_v1_state_computed and (alloc_v1_inputs_available or allocator_v1_mode == 'precomputed'),
                'has_teeth': alloc_v1_has_teeth,
                'regime_distribution': regime_dist,
                'scalar_stats': scalar_stats,
                'n_rebalances': len(alloc_v1_scalar_computed_history) if alloc_v1_scalar_computed_history else 0
            }
        else:
            allocator_v1_meta = {
                'enabled': False,
                'inputs_present': {},
                'inputs_missing': False,
                'effective': False,
                'has_teeth': False
            }
        
        
        # Document per-stage effective starts (Phase 3A auditability)
        per_stage_effective_starts = {
            'policy_effective_start': str(policy_effective_start.date()) if policy_effective_start is not None else None,
            'rt_effective_start': str(rt_effective_start.date()) if rt_effective_start is not None else None,
            'alloc_effective_start': str(alloc_effective_start.date()) if alloc_effective_start is not None else None
        }
        
        # Validate precomputed allocator source ID matches expected compute baseline
        allocator_source_valid = True
        if allocator_v1_enabled and allocator_v1_mode == 'precomputed' and allocator_source_run_id is not None:
            # Check if source ID matches the compute baseline run ID parameter
            allocator_v1_config = components.get('allocator_v1_config', {})
            expected_source = allocator_v1_config.get('precomputed_run_id')
            if expected_source and allocator_source_run_id != expected_source:
                logger.warning(
                    f"[ExecSim] Allocator source ID mismatch: "
                    f"meta has {allocator_source_run_id}, config expects {expected_source}"
                )
                allocator_source_valid = False
        
        meta = {
            'run_id': run_id,
            'start_date': str(start),  # Requested start date
            'end_date': str(end),
            'effective_start_date': effective_start_date,  # First rebalance date (after warmup)
            'evaluation_start_date': str(evaluation_start_date.date()) if evaluation_start_date is not None else None,
            # Per-stage effective starts for auditability
            'per_stage_effective_starts': per_stage_effective_starts,
            'strategy_profile': strategy_profile,  # Strategy profile name (if used)
            'strategy_config_name': strategy_name,
            'universe': universe,
            'rebalance': self.rebalance,
            'slippage_bps': self.slippage_bps,
            'n_rebalances': len(weights_panel) if not weights_panel.empty else 0,
            'n_trading_days': len(returns_df) if not returns_df.empty else 0,
            'canonical_window': is_canonical_window,
            'config_hash': config_hash,  # Hash of config file for reproducibility
            'config': full_config,  # Full config for diagnostics governance (Phase 3A)
            # Precomputed mode source links (for auditability)
            'allocator_source_run_id': allocator_source_run_id,
            'allocator_source_valid': allocator_source_valid,
            'engine_policy_source_run_id': engine_policy_source_run_id,
            # Phase 3A: Policy features documentation (data-driven, not file-based)
            'policy_features': policy_features_meta if policy_features_meta else None,
            # Risk Targeting governance (Layer 5)
            'risk_targeting': risk_targeting_meta,
            # Allocator v1 governance (Layer 6)
            # Allocator v1 governance (Layer 6)
            'allocator_v1': allocator_v1_meta,
            # Dual Metrics Reporting (Phase 3A)
            'metrics_full': metrics_full,
            'metrics_eval': metrics_eval
        }
        
        with open(run_dir / 'meta.json', 'w') as f:
            json.dump(meta, f, indent=2, cls=NumpyEncoder)
        
        # Sleeve weights artifact (source of truth for attribution when using return-based path)
        strategies = full_config.get('strategies', {})
        sleeve_weights = {
            name: float(cfg.get('weight', 0))
            for name, cfg in strategies.items()
            if isinstance(cfg, dict) and cfg.get('enabled') and (cfg.get('weight') or 0) > 0
        }
        if sleeve_weights:
            (run_dir / 'analysis').mkdir(parents=True, exist_ok=True)
            with open(run_dir / 'analysis' / 'sleeve_weights.json', 'w') as f:
                json.dump({'sleeve_weights': sleeve_weights, 'source': 'ExecSim'}, f, indent=2)
            logger.info(f"[ExecSim] Saved analysis/sleeve_weights.json ({len(sleeve_weights)} sleeves)")
        
        # Leverage telemetry: gross exposure from weights, RT scaling, target/realized vol
        (run_dir / 'analysis').mkdir(parents=True, exist_ok=True)
        gross_series = weights_panel.abs().sum(axis=1) if not weights_panel.empty else pd.Series(dtype=float)
        leverage_summary = {
            'run_id': run_id,
            'gross_exposure_avg': float(gross_series.mean()) if len(gross_series) > 0 else None,
            'gross_exposure_median': float(gross_series.median()) if len(gross_series) > 0 else None,
            'gross_exposure_max': float(gross_series.max()) if len(gross_series) > 0 else None,
            'gross_exposure_p95': float(gross_series.quantile(0.95)) if len(gross_series) > 0 else None,
            'scaling_factor_avg': float(np.mean(rt_leverage_history)) if rt_leverage_history and len(rt_leverage_history) > 0 else None,
            'target_vol': float(risk_targeting.target_vol) if risk_targeting is not None else None,
            'realized_vol': float(metrics_eval.get('vol')) if metrics_eval and metrics_eval.get('vol') is not None else (float(metrics_full.get('vol')) if metrics_full and metrics_full.get('vol') is not None else None),
            'leverage_cap': float(risk_targeting.leverage_cap) if risk_targeting is not None and getattr(risk_targeting, 'leverage_cap', None) is not None else None,
        }
        with open(run_dir / 'analysis' / 'leverage_summary.json', 'w') as f:
            json.dump(leverage_summary, f, indent=2)
        logger.info(f"[ExecSim] Saved analysis/leverage_summary.json")
        
        logger.info(f"[ExecSim] Saved artifacts to {run_dir}")
        
        # Attribution: compute portfolio-consistent sleeve-level return attribution
        try:
            from src.attribution.core import compute_attribution
            from src.attribution.artifacts import generate_attribution_artifacts
            
            if (sleeve_signals_history and portfolio_returns_daily is not None
                    and not weights_panel.empty and not returns_df.empty):
                logger.info("[ExecSim] Computing portfolio-consistent attribution...")
                
                # Convert log returns to simple returns for attribution input
                asset_returns_simple_for_attr = np.exp(returns_df) - 1.0
                
                # Default metasleeve mapping: each sleeve is its own metasleeve
                # Group VRP sleeves under a "vrp" metasleeve
                metasleeve_map = {}
                vrp_sleeves = {'vrp_core_meta', 'vrp_convergence_meta', 'vrp_alt_meta'}
                for sname in sleeve_signals_history.keys():
                    if sname in vrp_sleeves:
                        metasleeve_map[sname] = 'vrp_combined'
                    else:
                        metasleeve_map[sname] = sname
                
                attr_result = compute_attribution(
                    weights_panel=weights_panel,
                    asset_returns_simple=asset_returns_simple_for_attr,
                    portfolio_returns_simple=portfolio_returns_daily,
                    sleeve_signals_history=sleeve_signals_history,
                    universe=universe,
                    metasleeve_mapping=metasleeve_map,
                )
                
                attr_dir = generate_attribution_artifacts(
                    attribution_result=attr_result,
                    portfolio_returns_simple=portfolio_returns_daily,
                    run_dir=run_dir,
                    run_id=run_id,
                    metasleeve_mapping=metasleeve_map,
                )
                logger.info(f"[ExecSim] Attribution artifacts saved to {attr_dir}")
                
                # Also save sleeve weight decomposition for post-hoc recomputation
                decomp = attr_result.get("sleeve_weight_decomposition")
                if decomp is not None and not decomp.empty:
                    decomp.to_csv(run_dir / "sleeve_weight_decomposition.csv")
                    logger.info("[ExecSim] Saved sleeve_weight_decomposition.csv")
            else:
                logger.info("[ExecSim] Skipping attribution (missing sleeve signals or portfolio data)")
        except Exception as e:
            logger.warning(f"[ExecSim] Attribution computation failed (non-fatal): {e}")
            import traceback
            traceback.print_exc()
        
        # Red flag if allocator state failed
        if not allocator_state_success:
            logger.warning(
                f"[ExecSim] 🚩 RED FLAG: Allocator state v1 computation failed or returned empty. "
                f"See allocator_state_v1_error.json for details."
            )
    
    def describe(self) -> dict:
        """
        Return configuration and description of the ExecSim agent.
        
        Returns:
            dict with configuration parameters
        """
        return {
            'agent': 'ExecSim',
            'role': 'Backtest orchestrator and metrics calculator',
            'rebalance': self.rebalance,
            'slippage_bps': self.slippage_bps,
            'commission_per_contract': self.commission_per_contract,
            'cash_rate': self.cash_rate,
            'position_notional_scale': self.position_notional_scale,
            'outputs': ['run(market, start, end, components)', 'to_parquet(results, outdir)']
        }

