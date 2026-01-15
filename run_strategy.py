"""
Main entry point for running the futures-six TSMOM strategy.

This script orchestrates the full pipeline:
1. MarketData broker - read OHLCV data
2. TSMOM strategy - generate momentum signals
3. MacroRegimeFilter - regime scaler applied to signals
4. RiskVol - calculate volatility and covariance
5. VolManaged overlay - scale signals by volatility target
6. Allocator - convert signals to portfolio weights
7. ExecSim - simulate execution and generate backtest results
"""

import sys
import argparse
from pathlib import Path
import pandas as pd
import logging
import yaml

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
CONFIG_PATH = Path("configs/strategies.yaml")

# Import canonical dates
from src.config.backtest_window import CANONICAL_START, CANONICAL_END

# Import agents
from src.agents import MarketData
from src.agents.strat_momentum import TSMOM
from src.agents.strat_sr3_carry_curve import Sr3CarryCurveStrategy
from src.agents.strat_rates_curve import RatesCurveStrategy
from src.agents.strat_carry_fx_commod import CarryFxCommodStrategy
from src.agents.strat_momentum_medium import MediumTermMomentumStrategy, CanonicalMediumTermMomentumStrategy
from src.agents.strat_momentum_short import ShortTermMomentumStrategy
from src.agents.strat_tsmom_multihorizon import TSMOMMultiHorizonStrategy
from src.agents.strat_cross_sectional import CSMOMMeta
from src.agents.strat_residual_trend import ResidualTrendStrategy
from src.agents.strat_momentum_persistence import MomentumPersistence
from src.agents.strat_vrp_core import VRPCoreMeta
from src.agents.strat_vrp_convergence import VRPConvergenceMeta
from src.agents.strat_vrp_alt import VRPAltMeta
from src.agents.strat_combined import CombinedStrategy
from src.agents.feature_service import FeatureService
from src.agents.overlay_volmanaged import VolManagedOverlay
from src.agents.overlay_macro_regime import MacroRegimeFilter
from src.agents.risk_vol import RiskVol
from src.agents.allocator import Allocator
from src.agents.exec_sim import ExecSim
from src.layers import RiskTargetingLayer, create_risk_targeting_layer, ArtifactWriter, create_artifact_writer


def load_config(path: Path = CONFIG_PATH) -> dict:
    """Load strategy configuration from YAML."""
    if not path.exists():
        logger.warning("Config file not found at %s; using defaults.", path)
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def main():
    """Run the complete strategy backtest."""
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run futures-six TSMOM strategy backtest")
    parser.add_argument(
        "--strategy_profile",
        type=str,
        default=None,
        help="Strategy profile name from configs/strategies.yaml (e.g., 'momentum_only_v1'). If not specified, uses default strategies config."
    )
    parser.add_argument(
        "--start",
        type=str,
        default=CANONICAL_START,
        help=f"Start date for backtest (YYYY-MM-DD). Default: {CANONICAL_START}"
    )
    parser.add_argument(
        "--end",
        type=str,
        default=CANONICAL_END,
        help=f"End date for backtest (YYYY-MM-DD). Default: {CANONICAL_END}"
    )
    parser.add_argument(
        "--run_id",
        type=str,
        default=None,
        help="Run identifier for saving artifacts. If not specified, generates timestamp-based ID."
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default=None,
        help="Path to strategies.yaml config file. If not specified, uses configs/strategies.yaml"
    )
    
    args = parser.parse_args()
    
    start_date = args.start
    end_date = args.end
    run_id = args.run_id
    strategy_profile = args.strategy_profile
    config_path = Path(args.config_path) if args.config_path else CONFIG_PATH
    
    logger.info("=" * 80)
    logger.info("FUTURES-SIX: TSMOM Strategy Backtest")
    logger.info("=" * 80)
    
    logger.info(f"\nBacktest Period: {start_date} to {end_date}")
    if strategy_profile:
        logger.info(f"Strategy Profile: {strategy_profile}")
    if run_id:
        logger.info(f"Run ID: {run_id}")
    if args.config_path:
        logger.info(f"Config Path: {config_path}")
    
    try:
        config = load_config(config_path)
        
        # Log config source and key settings (for A/B backtest verification)
        logger.info(f"\n[Config] Loaded from: {config_path}")
        allocator_v1_cfg = config.get('allocator_v1', {})
        risk_targeting_cfg = config.get('risk_targeting', {})
        logger.info(f"[Config] allocator_v1.enabled={allocator_v1_cfg.get('enabled')}, "
                   f"mode={allocator_v1_cfg.get('mode')}, profile={allocator_v1_cfg.get('profile')}")
        logger.info(f"[Config] risk_targeting.enabled={risk_targeting_cfg.get('enabled')}, "
                   f"target_vol={risk_targeting_cfg.get('target_vol')}, "
                   f"leverage_cap={risk_targeting_cfg.get('leverage_cap')}")
        
        # Load strategy profile if specified
        if strategy_profile:
            strategy_profiles = config.get("strategy_profiles", {})
            if strategy_profile not in strategy_profiles:
                logger.error(f"Strategy profile '{strategy_profile}' not found in config")
                logger.info(f"Available profiles: {list(strategy_profiles.keys())}")
                return None
            
            profile_config = strategy_profiles[strategy_profile]
            # Override strategies config with profile
            strategies_cfg = profile_config.get("strategies", config.get("strategies", {}))
            # Override macro_regime config with profile
            profile_macro = profile_config.get("macro_regime", {})
            if profile_macro:
                macro_cfg = {**config.get("macro_regime", {}), **profile_macro}
            else:
                macro_cfg = config.get("macro_regime", {})
        else:
            strategies_cfg = config.get("strategies", {})
            macro_cfg = config.get("macro_regime", {})
        
        features_cfg = config.get("features", {}) if config else {}
        
        # 1. Initialize MarketData
        logger.info("\n[1/9] Initializing MarketData broker...")
        market = MarketData()
        logger.info(f"  Universe: {market.universe}")
        logger.info(f"  Table: {market.table_name}")
        
        # 1b. Build Policy Features (upstream of EnginePolicyV1)
        logger.info("\n[2/9] Building Policy Features...")
        from src.agents.policy_feature_builder import PolicyFeatureBuilder
        
        # Check if engine policy is enabled (if so, we need policy features)
        engine_policy_cfg = config.get("engine_policy_v1", {}) if config else {}
        engine_policy_enabled = engine_policy_cfg.get("enabled", False)
        
        if engine_policy_enabled:
            policy_builder = PolicyFeatureBuilder(market)
            try:
                policy_features = policy_builder.build(
                    start_date=start_date,
                    end_date=end_date,
                    attach_to_market=True
                )
                logger.info(f"  Policy features built: {list(policy_features.keys())}")
            except Exception as e:
                logger.error(f"  Error building policy features: {e}")
                logger.warning("  Continuing without policy features (EnginePolicyV1 may fail)")
        else:
            logger.info("  Engine Policy disabled, skipping policy features")
        
        # 1c. Initialize FeatureService
        logger.info("\n[3/9] Initializing FeatureService...")
        feature_service = FeatureService(market, config=features_cfg)
        logger.info("  FeatureService initialized")
        
        # 2. Initialize individual strategies
        logger.info("\n[4/9] Initializing strategy sleeves...")
        strategy_instances = {}
        strategy_weights = {}
        
        # TSMOM strategy
        tsmom_cfg = strategies_cfg.get("tsmom", {})
        if tsmom_cfg.get("enabled", True):
            logger.info("  Initializing TSMOM...")
            tsmom_params = tsmom_cfg.get("params", {})
            tsmom_weights = tsmom_params.get("weights", {})
            tsmom_strategy = TSMOM(
                symbols=None,  # Will use market.universe
                weights=tsmom_weights,
                signal_cap=tsmom_params.get("signal_cap", 3.0),
                rebalance=tsmom_params.get("rebalance", "W-FRI")
            )
            tsmom_strategy.fit_in_sample(market, start=start_date, end=end_date)
            strategy_instances["tsmom"] = tsmom_strategy
            strategy_weights["tsmom"] = tsmom_cfg.get("weight", 0.6)
            logger.info(f"    Config: weights={tsmom_weights}, "
                       f"signal_cap={tsmom_params.get('signal_cap')}, "
                       f"rebalance={tsmom_params.get('rebalance')}")
            logger.info(f"    Weight: {strategy_weights['tsmom']}")
        
        # SR3 Carry/Curve strategy
        sr3_cfg = strategies_cfg.get("sr3_carry_curve", {})
        if sr3_cfg.get("enabled", False):
            logger.info("  Initializing SR3 Carry/Curve...")
            sr3_params = sr3_cfg.get("params", {})
            sr3_strategy = Sr3CarryCurveStrategy(
                root="SR3",
                w_carry=sr3_params.get("w_carry", 0.30),
                w_curve=sr3_params.get("w_curve", 0.25),
                w_pack_slope=sr3_params.get("w_pack_slope", 0.20),
                w_front_lvl=sr3_params.get("w_front_lvl", 0.10),
                w_curv_belly=sr3_params.get("w_curv_belly", 0.15),
                cap=sr3_params.get("cap", 3.0),
                window=sr3_params.get("window", 252)
            )
            strategy_instances["sr3_carry_curve"] = sr3_strategy
            strategy_weights["sr3_carry_curve"] = sr3_cfg.get("weight", 0.15)
            logger.info(f"    Config: w_carry={sr3_params.get('w_carry')}, "
                       f"w_curve={sr3_params.get('w_curve')}, "
                       f"w_pack_slope={sr3_params.get('w_pack_slope')}, "
                       f"w_front_lvl={sr3_params.get('w_front_lvl')}, "
                       f"w_curv_belly={sr3_params.get('w_curv_belly')}")
            logger.info(f"    Weight: {strategy_weights['sr3_carry_curve']}")
        else:
            logger.info("  SR3 Carry/Curve disabled in config")
        
        # Rates Curve strategy
        rates_cfg = strategies_cfg.get("rates_curve", {})
        if rates_cfg.get("enabled", False):
            logger.info("  Initializing Rates Curve...")
            rates_params = rates_cfg.get("params", {})
            # Map rates symbols from market universe
            # Default to standard names, but allow override from config
            sym_2y = rates_params.get("sym_2y", "ZT_FRONT_VOLUME")
            sym_5y = rates_params.get("sym_5y", "ZF_FRONT_VOLUME")
            sym_10y = rates_params.get("sym_10y", "ZN_FRONT_VOLUME")
            sym_30y = rates_params.get("sym_30y", "UB_FRONT_VOLUME")
            
            # Validate symbols exist in universe
            universe_set = set(market.universe)
            if sym_2y not in universe_set:
                logger.warning(f"  Rates Curve: {sym_2y} not in universe, using default")
                sym_2y = "ZT_FRONT_VOLUME"
            if sym_5y not in universe_set:
                logger.warning(f"  Rates Curve: {sym_5y} not in universe, using default")
                sym_5y = "ZF_FRONT_VOLUME"
            if sym_10y not in universe_set:
                logger.warning(f"  Rates Curve: {sym_10y} not in universe, using default")
                sym_10y = "ZN_FRONT_VOLUME"
            if sym_30y not in universe_set:
                logger.warning(f"  Rates Curve: {sym_30y} not in universe, using default")
                sym_30y = "UB_FRONT_VOLUME"
            
            rates_strategy = RatesCurveStrategy(
                z_cap=rates_params.get("z_cap", 3.0),
                w_slope_2s10s=rates_params.get("w_slope_2s10s", 0.35),
                w_slope_5s30s=rates_params.get("w_slope_5s30s", 0.35),
                w_curv_2s5s10s=rates_params.get("w_curv_2s5s10s", 0.15),
                w_curv_5s10s30s=rates_params.get("w_curv_5s10s30s", 0.15),
                sym_2y=sym_2y,
                sym_5y=sym_5y,
                sym_10y=sym_10y,
                sym_30y=sym_30y
            )
            strategy_instances["rates_curve"] = rates_strategy
            strategy_weights["rates_curve"] = rates_cfg.get("weight", 0.15)
            logger.info(f"    Config: w_slope_2s10s={rates_params.get('w_slope_2s10s')}, "
                       f"w_slope_5s30s={rates_params.get('w_slope_5s30s')}, "
                       f"w_curv_2s5s10s={rates_params.get('w_curv_2s5s10s')}, "
                       f"w_curv_5s10s30s={rates_params.get('w_curv_5s10s30s')}")
            logger.info(f"    Weight: {strategy_weights['rates_curve']}")
        else:
            logger.info("  Rates Curve disabled in config")
        
        # FX/Commodity Carry strategy
        fx_commod_cfg = strategies_cfg.get("fx_commod_carry", {})
        if fx_commod_cfg.get("enabled", False):
            logger.info("  Initializing FX/Commodity Carry...")
            fx_commod_params = fx_commod_cfg.get("params", {})
            fx_commod_strategy = CarryFxCommodStrategy(
                roots=fx_commod_params.get("roots", ["CL", "GC", "6E", "6B", "6J"]),
                w_ts=fx_commod_params.get("w_ts", 0.6),
                w_xs=fx_commod_params.get("w_xs", 0.25),
                w_mom=fx_commod_params.get("w_mom", 0.15),
                clip=fx_commod_params.get("clip", 3.0),
                window=fx_commod_params.get("window", 252)
            )
            strategy_instances["fx_commod_carry"] = fx_commod_strategy
            strategy_weights["fx_commod_carry"] = fx_commod_cfg.get("weight", 0.10)
            logger.info(f"    Config: roots={fx_commod_params.get('roots')}, "
                       f"w_ts={fx_commod_params.get('w_ts')}, "
                       f"w_xs={fx_commod_params.get('w_xs')}, "
                       f"w_mom={fx_commod_params.get('w_mom')}, "
                       f"clip={fx_commod_params.get('clip')}")
            logger.info(f"    Weight: {strategy_weights['fx_commod_carry']}")
        else:
            logger.info("  FX/Commodity Carry disabled in config")
        
        # Medium-Term Momentum strategy (Legacy)
        tsmom_med_cfg = strategies_cfg.get("tsmom_med", {})
        if tsmom_med_cfg.get("enabled", False):
            logger.info("  Initializing Medium-Term Momentum (Legacy)...")
            tsmom_med_params = tsmom_med_cfg.get("params", {})
            tsmom_med_weights = tsmom_med_params.get("weights", {})
            tsmom_med_strategy = MediumTermMomentumStrategy(
                symbols=None,  # Will use market.universe
                weights=tsmom_med_weights,
                signal_cap=tsmom_med_params.get("signal_cap", 3.0),
                rebalance=tsmom_med_params.get("rebalance", "W-FRI")
            )
            strategy_instances["tsmom_med"] = tsmom_med_strategy
            strategy_weights["tsmom_med"] = tsmom_med_cfg.get("weight", 0.0)
            logger.info(f"    Config: weights={tsmom_med_weights}, "
                       f"signal_cap={tsmom_med_params.get('signal_cap')}, "
                       f"rebalance={tsmom_med_params.get('rebalance')}")
            logger.info(f"    Weight: {strategy_weights['tsmom_med']}")
        else:
            logger.info("  Medium-Term Momentum (Legacy) disabled in config")
        
        # Canonical Medium-Term Momentum strategy
        tsmom_med_canonical_cfg = strategies_cfg.get("tsmom_med_canonical", {})
        if tsmom_med_canonical_cfg.get("enabled", False):
            logger.info("  Initializing Canonical Medium-Term Momentum...")
            tsmom_med_canonical_params = tsmom_med_canonical_cfg.get("params", {})
            tsmom_med_canonical_strategy = CanonicalMediumTermMomentumStrategy(
                symbols=None,  # Will use market.universe
                signal_cap=tsmom_med_canonical_params.get("signal_cap", 3.0),
                rebalance=tsmom_med_canonical_params.get("rebalance", "W-FRI")
            )
            strategy_instances["tsmom_med_canonical"] = tsmom_med_canonical_strategy
            strategy_weights["tsmom_med_canonical"] = tsmom_med_canonical_cfg.get("weight", 0.0)
            logger.info(f"    Config: signal_cap={tsmom_med_canonical_params.get('signal_cap')}, "
                       f"rebalance={tsmom_med_canonical_params.get('rebalance')}")
            logger.info(f"    Weight: {strategy_weights['tsmom_med_canonical']}")
        else:
            logger.info("  Canonical Medium-Term Momentum disabled in config")
        
        # Short-Term Momentum strategy
        tsmom_short_cfg = strategies_cfg.get("tsmom_short", {})
        if tsmom_short_cfg.get("enabled", False):
            logger.info("  Initializing Short-Term Momentum...")
            tsmom_short_params = tsmom_short_cfg.get("params", {})
            tsmom_short_weights = tsmom_short_params.get("weights", {})
            tsmom_short_strategy = ShortTermMomentumStrategy(
                symbols=None,  # Will use market.universe
                weights=tsmom_short_weights,
                signal_cap=tsmom_short_params.get("signal_cap", 3.0),
                rebalance=tsmom_short_params.get("rebalance", "W-FRI")
            )
            strategy_instances["tsmom_short"] = tsmom_short_strategy
            strategy_weights["tsmom_short"] = tsmom_short_cfg.get("weight", 0.0)
            logger.info(f"    Config: weights={tsmom_short_weights}, "
                       f"signal_cap={tsmom_short_params.get('signal_cap')}, "
                       f"rebalance={tsmom_short_params.get('rebalance')}")
            logger.info(f"    Weight: {strategy_weights['tsmom_short']}")
        else:
            logger.info("  Short-Term Momentum disabled in config")
        
        # Multi-Horizon TSMOM strategy (v2)
        tsmom_mh_cfg = strategies_cfg.get("tsmom_multihorizon", {})
        if tsmom_mh_cfg.get("enabled", False):
            logger.info("  Initializing Multi-Horizon TSMOM (v2)...")
            tsmom_mh_params = tsmom_mh_cfg.get("params", {})
            tsmom_mh_strategy = TSMOMMultiHorizonStrategy(
                symbols=None,  # Will use market.universe
                horizon_weights=tsmom_mh_params.get("horizon_weights", {}),
                feature_weights=tsmom_mh_params.get("feature_weights", {}),
                signal_cap=tsmom_mh_params.get("signal_cap", 3.0),
                rebalance=tsmom_mh_params.get("rebalance", "W-FRI"),
                medium_variant=tsmom_mh_params.get("medium_variant", "legacy")
            )
            tsmom_mh_strategy.fit_in_sample(market, start=start_date, end=end_date)
            strategy_instances["tsmom_multihorizon"] = tsmom_mh_strategy
            strategy_weights["tsmom_multihorizon"] = tsmom_mh_cfg.get("weight", 0.6)
            logger.info(f"    Config: horizon_weights={tsmom_mh_params.get('horizon_weights')}, "
                       f"feature_weights={tsmom_mh_params.get('feature_weights')}, "
                       f"signal_cap={tsmom_mh_params.get('signal_cap')}, "
                       f"rebalance={tsmom_mh_params.get('rebalance')}")
            logger.info(f"    Weight: {strategy_weights['tsmom_multihorizon']}")
        else:
            logger.info("  Multi-Horizon TSMOM disabled in config")
        
        # CSMOM Meta-Sleeve
        csmom_cfg = strategies_cfg.get("csmom_meta", {})
        if csmom_cfg.get("enabled", False):
            logger.info("  Initializing CSMOM Meta-Sleeve...")
            csmom_params = csmom_cfg.get("params", {})
            csmom_strategy = CSMOMMeta(
                symbols=None,  # Will use market.universe
                lookbacks=csmom_params.get("lookbacks", [63, 126, 252]),
                weights=csmom_params.get("horizon_weights", [0.4, 0.35, 0.25]),
                vol_lookback=csmom_params.get("vol_lookback", 63),
                rebalance_freq=csmom_params.get("rebalance", "D"),
                neutralize_cross_section=csmom_params.get("neutralize_cross_section", True),
                clip_score=csmom_params.get("clip", 3.0)
            )
            csmom_strategy.fit_in_sample(market, start=start_date, end=end_date)
            strategy_instances["csmom_meta"] = csmom_strategy
            strategy_weights["csmom_meta"] = csmom_cfg.get("weight", 0.25)
            logger.info(f"    Config: lookbacks={csmom_params.get('lookbacks')}, "
                       f"horizon_weights={csmom_params.get('horizon_weights')}, "
                       f"vol_lookback={csmom_params.get('vol_lookback')}, "
                       f"rebalance={csmom_params.get('rebalance')}")
            logger.info(f"    Weight: {strategy_weights['csmom_meta']}")
        else:
            logger.info("  CSMOM Meta-Sleeve disabled in config")
        
        # VRP-Core Meta-Sleeve
        vrp_core_cfg = strategies_cfg.get("vrp_core_meta", {})
        if vrp_core_cfg.get("enabled", False):
            logger.info("  Initializing VRP-Core Meta-Sleeve...")
            vrp_core_params = vrp_core_cfg.get("params", {})
            # Determine DB path from config
            import yaml
            config_path = Path("configs/data.yaml")
            db_path = None
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    data_config = yaml.safe_load(f)
                db_path = data_config.get('db', {}).get('path')
            
            vrp_core_strategy = VRPCoreMeta(
                rv_lookback=vrp_core_params.get("rv_lookback", 21),
                zscore_window=vrp_core_params.get("zscore_window", 252),
                clip=vrp_core_params.get("clip", 3.0),
                signal_mode=vrp_core_params.get("signal_mode", "zscore"),
                db_path=db_path
            )
            strategy_instances["vrp_core_meta"] = vrp_core_strategy
            strategy_weights["vrp_core_meta"] = vrp_core_cfg.get("weight", 0.10)
            logger.info(f"    Config: rv_lookback={vrp_core_params.get('rv_lookback')}, "
                       f"zscore_window={vrp_core_params.get('zscore_window')}, "
                       f"clip={vrp_core_params.get('clip')}, "
                       f"signal_mode={vrp_core_params.get('signal_mode')}")
            logger.info(f"    Weight: {strategy_weights['vrp_core_meta']}")
        else:
            logger.info("  VRP-Core Meta-Sleeve disabled in config")
        
        # VRP-Convergence Meta-Sleeve
        vrp_conv_cfg = strategies_cfg.get("vrp_convergence_meta", {})
        if vrp_conv_cfg.get("enabled", False):
            logger.info("  Initializing VRP-Convergence Meta-Sleeve...")
            vrp_conv_params = vrp_conv_cfg.get("params", {})
            # Determine DB path from config
            import yaml
            config_path = Path("configs/data.yaml")
            db_path = None
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    data_config = yaml.safe_load(f)
                db_path = data_config.get('db', {}).get('path')
            
            vrp_conv_strategy = VRPConvergenceMeta(
                zscore_window=vrp_conv_params.get("zscore_window", 252),
                clip=vrp_conv_params.get("clip", 3.0),
                signal_mode=vrp_conv_params.get("signal_mode", "zscore"),
                target_vol=vrp_conv_params.get("target_vol", 0.10),
                vol_lookback=vrp_conv_params.get("vol_lookback", 63),
                vol_floor=vrp_conv_params.get("vol_floor", 0.05),
                db_path=db_path
            )
            strategy_instances["vrp_convergence_meta"] = vrp_conv_strategy
            strategy_weights["vrp_convergence_meta"] = vrp_conv_cfg.get("weight", 0.05)
            logger.info(f"    Config: zscore_window={vrp_conv_params.get('zscore_window')}, "
                       f"clip={vrp_conv_params.get('clip')}, "
                       f"signal_mode={vrp_conv_params.get('signal_mode')}, "
                       f"target_vol={vrp_conv_params.get('target_vol')}")
            logger.info(f"    Weight: {strategy_weights['vrp_convergence_meta']}")
        else:
            logger.info("  VRP-Convergence Meta-Sleeve disabled in config")
        
        # VRP-Alt Meta-Sleeve
        vrp_alt_cfg = strategies_cfg.get("vrp_alt_meta", {})
        if vrp_alt_cfg.get("enabled", False):
            logger.info("  Initializing VRP-Alt Meta-Sleeve...")
            vrp_alt_params = vrp_alt_cfg.get("params", {})
            # Determine DB path from config
            import yaml
            config_path = Path("configs/data.yaml")
            db_path = None
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    data_config = yaml.safe_load(f)
                db_path = data_config.get('db', {}).get('path')
            
            vrp_alt_strategy = VRPAltMeta(
                zscore_window=vrp_alt_params.get("zscore_window", 252),
                clip=vrp_alt_params.get("clip", 3.0),
                signal_mode=vrp_alt_params.get("signal_mode", "zscore"),
                target_vol=vrp_alt_params.get("target_vol", 0.10),
                vol_lookback=vrp_alt_params.get("vol_lookback", 63),
                vol_floor=vrp_alt_params.get("vol_floor", 0.05),
                db_path=db_path
            )
            strategy_instances["vrp_alt_meta"] = vrp_alt_strategy
            strategy_weights["vrp_alt_meta"] = vrp_alt_cfg.get("weight", 0.05)
            logger.info(f"    Config: zscore_window={vrp_alt_params.get('zscore_window')}, "
                       f"clip={vrp_alt_params.get('clip')}, "
                       f"signal_mode={vrp_alt_params.get('signal_mode')}, "
                       f"target_vol={vrp_alt_params.get('target_vol')}")
            logger.info(f"    Weight: {strategy_weights['vrp_alt_meta']}")
        else:
            logger.info("  VRP-Alt Meta-Sleeve disabled in config")
        
        # SR3 Curve RV Meta-Sleeve
        curve_rv_cfg = strategies_cfg.get("sr3_curve_rv_meta", {})
        curve_rv_meta = None
        curve_rv_weight = 0.0
        if curve_rv_cfg.get("enabled", False):
            logger.info("  Initializing SR3 Curve RV Meta-Sleeve...")
            from src.agents.strat_curve_rv_meta import SR3CurveRVMeta
            curve_rv_params = curve_rv_cfg.get("params", {})
            curve_rv_meta = SR3CurveRVMeta(
                enabled_atomics=curve_rv_params.get("enabled_atomics", ["rank_fly"]),
                atomic_weights=curve_rv_params.get("atomic_weights"),
                zscore_window=curve_rv_params.get("zscore_window", 252),
                clip=curve_rv_params.get("clip", 3.0),
                target_vol=curve_rv_params.get("target_vol", 0.10),
                vol_lookback=curve_rv_params.get("vol_lookback", 63),
                min_vol_floor=curve_rv_params.get("min_vol_floor", 0.01),
                max_leverage=curve_rv_params.get("max_leverage", 10.0),
                lag=curve_rv_params.get("lag", 1)
            )
            curve_rv_weight = curve_rv_cfg.get("weight", 0.05)
            logger.info(f"    Config: enabled_atomics={curve_rv_params.get('enabled_atomics')}, "
                       f"zscore_window={curve_rv_params.get('zscore_window')}, "
                       f"clip={curve_rv_params.get('clip')}, "
                       f"target_vol={curve_rv_params.get('target_vol')}")
            logger.info(f"    Weight: {curve_rv_weight}")
        else:
            logger.info("  SR3 Curve RV Meta-Sleeve disabled in config")
        
        # Residual Trend strategy
        residual_trend_cfg = strategies_cfg.get("residual_trend", {})
        if residual_trend_cfg.get("enabled", False):
            logger.info("  Initializing Residual Trend...")
            residual_trend_params = residual_trend_cfg.get("params", {})
            residual_trend_strategy = ResidualTrendStrategy(
                symbols=None,  # Will use market.universe
                long_lookback=residual_trend_params.get("long_lookback", 252),
                short_lookback=residual_trend_params.get("short_lookback", 21),
                signal_cap=residual_trend_params.get("signal_cap", 3.0),
                rebalance=residual_trend_params.get("rebalance", "W-FRI")
            )
            residual_trend_strategy.fit_in_sample(market, start=start_date, end=end_date)
            strategy_instances["residual_trend"] = residual_trend_strategy
            strategy_weights["residual_trend"] = residual_trend_cfg.get("weight", 0.20)
            logger.info(f"    Config: long_lookback={residual_trend_params.get('long_lookback')}, "
                       f"short_lookback={residual_trend_params.get('short_lookback')}, "
                       f"signal_cap={residual_trend_params.get('signal_cap')}, "
                       f"rebalance={residual_trend_params.get('rebalance')}")
            logger.info(f"    Weight: {strategy_weights['residual_trend']}")
        else:
            logger.info("  Residual Trend disabled in config")
        
        # Persistence strategy
        persistence_cfg = strategies_cfg.get("persistence", {})
        if persistence_cfg.get("enabled", False):
            logger.info("  Initializing Persistence...")
            persistence_params = persistence_cfg.get("params", {})
            persistence_strategy = MomentumPersistence(
                symbols=None,  # Will use market.universe
                weights=persistence_params.get("weights", {
                    "slope_accel": 0.80,
                    "breakout_accel": 0.10,
                    "return_accel": 0.10
                }),
                signal_cap=persistence_params.get("signal_cap", 3.0),
                rebalance=persistence_params.get("rebalance", "W-FRI")
            )
            strategy_instances["persistence"] = persistence_strategy
            strategy_weights["persistence"] = persistence_cfg.get("weight", 0.20)
            logger.info(f"    Config: weights={persistence_params.get('weights')}, "
                       f"signal_cap={persistence_params.get('signal_cap')}, "
                       f"rebalance={persistence_params.get('rebalance')}")
            logger.info(f"    Weight: {strategy_weights['persistence']}")
        else:
            logger.info("  Persistence disabled in config")
        
        # Pre-compute features if any feature-based strategy is enabled
        features_dict = {}
        if ("sr3_carry_curve" in strategy_instances or
            "rates_curve" in strategy_instances or
            "fx_commod_carry" in strategy_instances or
            "tsmom" in strategy_instances or
            "tsmom_med" in strategy_instances or
            "tsmom_med_canonical" in strategy_instances or
            "tsmom_short" in strategy_instances or
            "tsmom_multihorizon" in strategy_instances or
            "residual_trend" in strategy_instances or
            "persistence" in strategy_instances):
            logger.info("  Pre-computing features...")
            # For multi-horizon, we need all three momentum feature types
            feature_types = []
            if "tsmom" in strategy_instances or "tsmom_multihorizon" in strategy_instances:
                feature_types.append("LONG_MOMENTUM")
            if "tsmom_med" in strategy_instances or "tsmom_multihorizon" in strategy_instances:
                feature_types.append("MEDIUM_MOMENTUM")
            if "tsmom_med_canonical" in strategy_instances:
                feature_types.append("CANONICAL_MEDIUM_MOMENTUM")
            if "tsmom_short" in strategy_instances or "tsmom_multihorizon" in strategy_instances:
                feature_types.append("SHORT_MOMENTUM")
            # Residual trend is now part of TSMOMMultiHorizon as 4th atomic sleeve
            if "tsmom_multihorizon" in strategy_instances:
                feature_types.append("RESIDUAL_TREND")
                # Also need canonical medium-term if medium_variant is canonical
                tsmom_mh_params = strategies_cfg.get("tsmom_multihorizon", {}).get("params", {})
                if tsmom_mh_params.get("medium_variant", "legacy") == "canonical":
                    feature_types.append("CANONICAL_MEDIUM_MOMENTUM")
            if "sr3_carry_curve" in strategy_instances:
                feature_types.append("SR3_CURVE")
            if "rates_curve" in strategy_instances:
                feature_types.append("RATES_CURVE")
            if "fx_commod_carry" in strategy_instances:
                feature_types.append("CARRY_FX_COMMOD")
            if "residual_trend" in strategy_instances:
                feature_types.append("RESIDUAL_TREND")
            if "persistence" in strategy_instances:
                feature_types.append("PERSISTENCE")
            
            features_dict = feature_service.get_features(end_date=end_date, feature_types=feature_types)
            logger.info(f"    Computed features: {list(features_dict.keys())}")
        
        # 3. Initialize CombinedStrategy
        logger.info("\n[5/9] Initializing CombinedStrategy...")
        strategy = CombinedStrategy(
            strategies=strategy_instances,
            weights=strategy_weights,
            features=features_dict,
            feature_service=feature_service
        )
        logger.info(f"  Config: {strategy.describe()}")
        
        # 4. Initialize MacroRegimeFilter (if enabled)
        macro_enabled = macro_cfg.get("enabled", True)
        if macro_enabled:
            logger.info("\n[6/9] Initializing MacroRegimeFilter...")
            fred_series = macro_cfg.get("fred_series")
            if fred_series and isinstance(fred_series, list):
                fred_series = tuple(fred_series)
            macro_overlay = MacroRegimeFilter(
                rebalance=macro_cfg.get("rebalance", "W-FRI"),
                vol_thresholds=macro_cfg.get("vol_thresholds"),
                k_bounds=macro_cfg.get("k_bounds"),
                smoothing=macro_cfg.get("smoothing", 0.2),
                vol_lookback=macro_cfg.get("vol_lookback", 21),
                breadth_lookback=macro_cfg.get("breadth_lookback", 200),
                proxy_symbols=tuple(macro_cfg.get("proxy_symbols", ("ES", "NQ"))),
                fred_series=fred_series,
                fred_lookback=macro_cfg.get("fred_lookback", 252),
                fred_weight=macro_cfg.get("fred_weight", 0.3)
            )
            logger.info(f"  Config: {macro_overlay.describe()}")
        else:
            logger.info("\n[6/9] MacroRegimeFilter disabled (enabled: false)")
            macro_overlay = None
        
        # 5. Initialize RiskVol (needed by VolManaged)
        logger.info("\n[7/9] Initializing RiskVol agent...")
        risk = RiskVol()
        logger.info(f"  Config: {risk.describe()}")
        
        # 6. Initialize VolManaged Overlay
        logger.info("\n[8/9] Initializing VolManaged overlay...")
        vol_overlay = VolManagedOverlay(risk_vol=risk)
        logger.info(f"  Config: {vol_overlay.describe()}")
        
        # 7. Initialize Risk Targeting Layer (Layer 5)
        logger.info("\n[9/10] Initializing Risk Targeting Layer...")
        risk_targeting_cfg = config.get('risk_targeting', {})
        risk_targeting_enabled = risk_targeting_cfg.get('enabled', False)
        risk_targeting_layer = None
        
        if risk_targeting_enabled:
            # Use profile if specified, otherwise use config values
            profile = risk_targeting_cfg.get('profile', 'default')
            if profile:
                risk_targeting_layer = create_risk_targeting_layer(profile=profile)
            else:
                # Create from config values
                risk_targeting_layer = RiskTargetingLayer(
                    target_vol=risk_targeting_cfg.get('target_vol', 0.20),
                    leverage_cap=risk_targeting_cfg.get('leverage_cap', 7.0),
                    leverage_floor=risk_targeting_cfg.get('leverage_floor', 1.0),
                    vol_lookback=risk_targeting_cfg.get('vol_lookback', 63),
                    update_frequency=risk_targeting_cfg.get('update_frequency', 'static'),
                    config_path=None  # Don't load from config when explicitly set
                )
            logger.info(f"  Config: {risk_targeting_layer.describe()}")
        else:
            logger.info("  Risk Targeting disabled (enabled: false)")
        
        # 8. Initialize Allocator
        logger.info("\n[10/10] Initializing Allocator...")
        allocator = Allocator()
        logger.info(f"  Config: {allocator.describe()}")
        
        # 9. Initialize and Run ExecSim
        logger.info("\nRunning backtest with ExecSim...")
        exec_sim = ExecSim()
        logger.info(f"  Config: {exec_sim.describe()}")
        
        # Run backtest
        logger.info(f"\nExecuting backtest from {start_date} to {end_date}...")
        
        # Package components
        # Stage 4D: Include allocator_v1 config for integration point
        allocator_v1_config = config.get('allocator_v1', {})
        # Phase 2: Include engine_policy_v1 config for integration point
        engine_policy_v1_config = config.get('engine_policy_v1', {})
        
        components = {
            'strategy': strategy,
            'macro_overlay': macro_overlay,
            'overlay': vol_overlay,
            'risk_vol': risk,
            'risk_targeting': risk_targeting_layer,  # Layer 5: Risk Targeting
            'allocator': allocator,
            'curve_rv_meta': curve_rv_meta,  # Add Curve RV meta-sleeve if enabled
            'curve_rv_weight': curve_rv_weight,  # Add Curve RV weight
            'allocator_v1_config': allocator_v1_config,  # Stage 4D: Allocator v1 config
            'engine_policy_v1_config': engine_policy_v1_config  # Phase 2: Engine Policy v1 config
        }
        
        results = exec_sim.run(
            market=market,
            start=start_date,
            end=end_date,
            components=components,
            run_id=run_id,
            strategy_profile=strategy_profile,  # Pass strategy profile for meta.json
            config_path=str(config_path)  # Pass config path for meta.json
        )
        
        # Extract results
        equity = results['equity_curve']  # This is now the daily equity curve filtered from first rebalance
        report = results['report']
        weights_panel = results['weights_panel']
        signals_panel = results['signals_panel']
        
        # Display Results
        logger.info("\n" + "=" * 80)
        logger.info("BACKTEST RESULTS")
        logger.info("=" * 80)
        
        logger.info(f"\nEquity Curve:")
        # Show actual first trading date (first rebalance date), not requested start date
        actual_start_date = equity.index[0]
        effective_start_date = actual_start_date  # Effective start = first rebalance date (after warmup)
        warmup_days = (pd.Timestamp(effective_start_date) - pd.Timestamp(start_date)).days if start_date else 0
        logger.info(f"  Requested Start:   {start_date}")
        logger.info(f"  Effective Start:   {effective_start_date} (first rebalance date, after warmup)")
        if warmup_days > 0:
            logger.info(f"  Warmup Period:     {warmup_days} days")
        logger.info(f"  End Date:           {equity.index[-1]}")
        logger.info(f"  Starting Value:     ${equity.iloc[0]:,.2f}")
        logger.info(f"  Ending Value:       ${equity.iloc[-1]:,.2f}")
        logger.info(f"  Total Return:       {(equity.iloc[-1] / equity.iloc[0] - 1) * 100:.2f}%")
        
        logger.info(f"\nPerformance Metrics:")
        for metric, value in report.items():
            if isinstance(value, (int, float)):
                if 'vol' in metric.lower() or 'drawdown' in metric.lower() or 'sharpe' in metric.lower() or 'turnover' in metric.lower() or 'rate' in metric.lower():
                    logger.info(f"  {metric:20}: {value:8.4f}")
                elif 'cagr' in metric.lower() or 'return' in metric.lower():
                    logger.info(f"  {metric:20}: {value:8.2%}")
                elif 'gross' in metric.lower() or 'net' in metric.lower():
                    logger.info(f"  {metric:20}: {value:8.2f}x")
                else:
                    logger.info(f"  {metric:20}: {value}")
            else:
                logger.info(f"  {metric:20}: {value}")
        
        logger.info(f"\nPortfolio Statistics:")
        logger.info(f"  Number of rebalances: {len(weights_panel)}")
        logger.info(f"  Assets in universe:   {len(market.universe)}")
        
        # Show final weights
        if not weights_panel.empty:
            final_weights = weights_panel.iloc[-1]
            logger.info(f"\nFinal Portfolio Weights (as of {weights_panel.index[-1]}):")
            for symbol, weight in final_weights.items():
                if abs(weight) > 0.001:  # Only show non-zero weights
                    logger.info(f"  {symbol:25}: {weight:7.2%}")
        
        # Close market data connection
        market.close()
        
        logger.info("\n" + "=" * 80)
        logger.info("Backtest completed successfully!")
        logger.info("=" * 80)
        
        return results
    
    except Exception as e:
        logger.error(f"\nError during backtest: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    results = main()
    sys.exit(0 if results is not None else 1)

