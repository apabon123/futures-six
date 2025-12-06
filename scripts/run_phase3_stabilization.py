"""
Phase 3 Stabilization: Run all diagnostic validations in sequence.

This script orchestrates the complete Phase 3 validation pipeline:
1. Feature Validation
2. Sleeve Validation
3. Overlay Validation
4. Combined Strategy Validation
5. Minimal Charts

Run from project root:
    python scripts/run_phase3_stabilization.py
"""

import sys
from pathlib import Path
import logging
import yaml
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import diagnostics
from src.diagnostics.feature_validation import run_feature_validation
from src.diagnostics.sleeve_validation import run_sleeve_validation
from src.diagnostics.overlay_validation import run_overlay_validation
from src.diagnostics.combined_validation import run_combined_validation
from src.diagnostics.minimal_charts import run_minimal_charts
from src.diagnostics.continuous_price_validation import validate_continuous_prices
from src.diagnostics.universe_consistency import universe_consistency_report

# Import agents
from src.agents import MarketData
from src.agents.strat_momentum import TSMOM
from src.agents.strat_sr3_carry_curve import Sr3CarryCurveStrategy
from src.agents.strat_rates_curve import RatesCurveStrategy
from src.agents.strat_carry_fx_commod import CarryFxCommodStrategy
from src.agents.strat_momentum_medium import MediumTermMomentumStrategy
from src.agents.strat_momentum_short import ShortTermMomentumStrategy
from src.agents.strat_combined import CombinedStrategy
from src.agents.feature_service import FeatureService
from src.agents.overlay_volmanaged import VolManagedOverlay
from src.agents.overlay_macro_regime import MacroRegimeFilter
from src.agents.risk_vol import RiskVol
from src.agents.allocator import Allocator
from src.agents.exec_sim import ExecSim


def load_config(path: Path = Path("configs/strategies.yaml")) -> dict:
    """Load strategy configuration from YAML."""
    if not path.exists():
        logger.warning("Config file not found at %s; using defaults.", path)
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


class Env:
    """Environment object to hold all components."""
    pass


def load_environment(start_date: str = "2021-01-01", end_date: str = "2025-11-05") -> Env:
    """
    Load all components into environment object.
    
    Args:
        start_date: Start date for backtest
        end_date: End date for backtest
    
    Returns:
        Env object with all components
    """
    logger.info("Loading environment components...")
    
    config = load_config()
    macro_cfg = config.get("macro_regime", {}) if config else {}
    strategies_cfg = config.get("strategies", {}) if config else {}
    features_cfg = config.get("features", {}) if config else {}
    
    env = Env()
    
    # 1. MarketData
    logger.info("  Initializing MarketData...")
    env.market = MarketData()
    
    # 2. FeatureService
    logger.info("  Initializing FeatureService...")
    env.feature_service = FeatureService(env.market, config=features_cfg)
    
    # 3. Get features (for validation)
    logger.info("  Computing features...")
    env.features = env.feature_service.get_features(end_date=end_date)
    
    # 4. Get prices (for validation)
    logger.info("  Loading prices...")
    env.prices = env.market.get_price_panel(
        symbols=env.market.universe,
        fields=("close",),
        end=end_date,
        tidy=False
    )
    
    # 5. Initialize strategy sleeves
    logger.info("  Initializing strategy sleeves...")
    strategy_instances = {}
    strategy_weights = {}
    
    # TSMOM
    tsmom_cfg = strategies_cfg.get("tsmom", {})
    if tsmom_cfg.get("enabled", True):
        tsmom_params = tsmom_cfg.get("params", {})
        tsmom_weights = tsmom_params.get("weights", {})
        tsmom_strategy = TSMOM(
            symbols=None,
            weights=tsmom_weights,
            signal_cap=tsmom_params.get("signal_cap", 3.0),
            rebalance=tsmom_params.get("rebalance", "W-FRI")
        )
        tsmom_strategy.fit_in_sample(env.market, start=start_date, end=end_date)
        strategy_instances["tsmom"] = tsmom_strategy
        strategy_weights["tsmom"] = tsmom_cfg.get("weight", 0.6)
    
    # SR3 Carry/Curve
    sr3_cfg = strategies_cfg.get("sr3_carry_curve", {})
    if sr3_cfg.get("enabled", False):
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
    
    # Rates Curve
    rates_cfg = strategies_cfg.get("rates_curve", {})
    if rates_cfg.get("enabled", False):
        rates_params = rates_cfg.get("params", {})
        rates_strategy = RatesCurveStrategy(
            z_cap=rates_params.get("z_cap", 3.0),
            w_slope_2s10s=rates_params.get("w_slope_2s10s", 0.35),
            w_slope_5s30s=rates_params.get("w_slope_5s30s", 0.35),
            w_curv_2s5s10s=rates_params.get("w_curv_2s5s10s", 0.15),
            w_curv_5s10s30s=rates_params.get("w_curv_5s10s30s", 0.15)
        )
        strategy_instances["rates_curve"] = rates_strategy
        strategy_weights["rates_curve"] = rates_cfg.get("weight", 0.10)
    
    # FX/Commodity Carry
    fx_commod_cfg = strategies_cfg.get("fx_commod_carry", {})
    if fx_commod_cfg.get("enabled", False):
        fx_commod_params = fx_commod_cfg.get("params", {})
        fx_commod_strategy = CarryFxCommodStrategy(
            roots=fx_commod_params.get("roots", ["CL", "GC", "6E", "6B", "6J"]),
            window=fx_commod_params.get("window", 252),
            clip=fx_commod_params.get("clip", 3.0)
        )
        strategy_instances["fx_commod_carry"] = fx_commod_strategy
        strategy_weights["fx_commod_carry"] = fx_commod_cfg.get("weight", 0.10)
    
    # Medium-term Momentum
    med_mom_cfg = strategies_cfg.get("medium_momentum", {})
    if med_mom_cfg.get("enabled", False):
        med_mom_params = med_mom_cfg.get("params", {})
        med_mom_strategy = MediumTermMomentumStrategy(
            symbols=None,
            weights=med_mom_params.get("weights", {}),
            signal_cap=med_mom_params.get("signal_cap", 3.0),
            rebalance=med_mom_params.get("rebalance", "W-FRI")
        )
        med_mom_strategy.fit_in_sample(env.market, start=start_date, end=end_date)
        strategy_instances["medium_momentum"] = med_mom_strategy
        strategy_weights["medium_momentum"] = med_mom_cfg.get("weight", 0.05)
    
    # Short-term Momentum
    short_mom_cfg = strategies_cfg.get("short_momentum", {})
    if short_mom_cfg.get("enabled", False):
        short_mom_params = short_mom_cfg.get("params", {})
        short_mom_strategy = ShortTermMomentumStrategy(
            symbols=None,
            weights=short_mom_params.get("weights", {}),
            signal_cap=short_mom_params.get("signal_cap", 3.0),
            rebalance=short_mom_params.get("rebalance", "W-FRI")
        )
        short_mom_strategy.fit_in_sample(env.market, start=start_date, end=end_date)
        strategy_instances["short_momentum"] = short_mom_strategy
        strategy_weights["short_momentum"] = short_mom_cfg.get("weight", 0.05)
    
    env.sleeves = list(strategy_instances.values())
    
    # 6. CombinedStrategy
    logger.info("  Initializing CombinedStrategy...")
    env.combined_strategy = CombinedStrategy(
        strategies=strategy_instances,
        weights=strategy_weights,
        features=env.features,
        feature_service=env.feature_service
    )
    
    # 7. MacroRegimeFilter
    logger.info("  Initializing MacroRegimeFilter...")
    fred_series = macro_cfg.get("fred_series")
    if fred_series and isinstance(fred_series, list):
        fred_series = tuple(fred_series)
    env.macro_overlay = MacroRegimeFilter(
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
    
    # 8. RiskVol
    logger.info("  Initializing RiskVol...")
    env.risk_vol = RiskVol()
    
    # 9. VolManagedOverlay
    logger.info("  Initializing VolManagedOverlay...")
    env.vol_overlay = VolManagedOverlay(risk_vol=env.risk_vol)
    
    # 10. Allocator
    logger.info("  Initializing Allocator...")
    env.allocator = Allocator()
    
    # 11. ExecSim
    logger.info("  Initializing ExecSim...")
    env.exec_sim = ExecSim()
    
    # 12. Dates
    env.start = start_date
    env.end = end_date
    
    # 13. Z-score feature columns (for validation)
    # These are columns that should be z-scored (have _z suffix typically)
    env.zscore_features = []
    for feat_type, feat_df in env.features.items():
        if not feat_df.empty:
            zscore_cols = [col for col in feat_df.columns if col.endswith('_z')]
            env.zscore_features.extend(zscore_cols)
    
    logger.info("Environment loaded successfully")
    return env


def main():
    """Run all Phase 3 validations in sequence."""
    
    print("\n" + "=" * 80)
    print("PHASE 3: STABILIZATION VALIDATION")
    print("=" * 80)
    
    # Configuration
    start_date = "2021-01-01"
    end_date = "2025-11-05"
    
    print(f"\nValidation Period: {start_date} to {end_date}")
    
    try:
        # Load environment
        print("\n[SETUP] Loading environment components...")
        env = load_environment(start_date=start_date, end_date=end_date)
        
        # Prepare output directory
        out_dir = Path("reports/phase3")
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # 0. Continuous Price Validation (new - validate back-adjustment logic)
        print("\n" + "=" * 80)
        print("PHASE 3.0: VALIDATE CONTINUOUS PRICES")
        print("=" * 80)
        validate_continuous_prices(env.market, verbose=True)
        
        # 1. Feature Validation
        print("\n" + "=" * 80)
        print("PHASE 3.1: VALIDATE FEATURES")
        print("=" * 80)
        run_feature_validation(
            market=env.market,
            feature_service=env.feature_service,
            prices=env.prices,
            zscore_cols=env.zscore_features,
            end_date=end_date
        )
        
        # 2. Sleeve Validation
        print("\n" + "=" * 80)
        print("PHASE 3.2: VALIDATE SLEEVES")
        print("=" * 80)
        run_sleeve_validation(
            all_sleeves=env.sleeves,
            market=env.market,
            exec_sim=env.exec_sim,
            start=start_date,
            end=end_date,
            features=env.features,
            risk_vol=env.risk_vol,
            overlay=env.vol_overlay,
            allocator=env.allocator
        )
        
        # 3. Overlay Validation
        # For overlay validation, we need a sample combined signal
        # Get it from a sample date
        print("\n" + "=" * 80)
        print("PHASE 3.3: VALIDATE OVERLAYS")
        print("=" * 80)
        
        # Get a sample date for overlay validation
        sample_date = env.prices.index[-1] if not env.prices.empty else end_date
        sample_signal = env.combined_strategy.signals(env.market, sample_date)
        
        # Get covariance for allocator
        cov = env.risk_vol.covariance(env.market, sample_date)
        
        run_overlay_validation(
            macro_overlay=env.macro_overlay,
            vol_overlay=env.vol_overlay,
            allocator=env.allocator,
            combined_signal=sample_signal,
            market=env.market,
            date=sample_date,
            cov=cov
        )
        
        # 4. Combined Strategy Validation
        print("\n" + "=" * 80)
        print("PHASE 3.4: VALIDATE COMBINED STRATEGY")
        print("=" * 80)
        combined_results = run_combined_validation(env)
        
        # 4.5. Universe Consistency Report
        print("\n" + "=" * 80)
        print("PHASE 3.4.5: UNIVERSE CONSISTENCY REPORT")
        print("=" * 80)
        universe_info = universe_consistency_report(env, combined_results)
        
        # 5. Minimal Charts
        print("\n" + "=" * 80)
        print("PHASE 3.5: GENERATE MINIMAL CHARTS")
        print("=" * 80)
        
        if not combined_results.get("weights", pd.DataFrame()).empty and \
           not combined_results.get("returns", pd.Series()).empty:
            run_minimal_charts(
                weights=combined_results["weights"],
                returns=combined_results["returns"],
                out_dir=str(out_dir)
            )
        else:
            logger.warning("Skipping charts: missing weights or returns data")
        
        print("\n" + "=" * 80)
        print("PHASE 3 FINISHED SUCCESSFULLY")
        print("=" * 80)
        print(f"\nResults saved to: {out_dir}")
        print("\nNext steps:")
        print("  1. Review console output for warnings")
        print("  2. Check charts in reports/phase3/")
        print("  3. Address any validation failures")
        
    except Exception as e:
        logger.error(f"Phase 3 validation failed: {e}", exc_info=True)
        sys.exit(1)
    finally:
        # Cleanup
        if hasattr(env, 'market'):
            env.market.close()


if __name__ == "__main__":
    main()

