"""
Universe Consistency Diagnostic: Track which assets exist at each stage.

This module helps debug where assets disappear in the pipeline by tracking
the universe of symbols at each stage:
- Raw prices
- Continuous prices  
- Features
- Sleeve signals (individual)
- Combined signals
- Allocator weights
"""

import pandas as pd
import logging
from typing import Dict, Set, Optional, List, Union
from datetime import datetime

logger = logging.getLogger(__name__)


def print_universe_stage(name: str, objs: List[Union[pd.DataFrame, pd.Series, Dict]]) -> Set[str]:
    """
    Extract and print universe of symbols from a list of objects.
    
    Args:
        name: Name of this stage
        objs: List of DataFrames, Series, or dicts containing symbol data
    
    Returns:
        Set of symbol names found
    """
    print(f"\n=== Universe at stage: {name} ===")
    cols = set()
    
    for obj in objs:
        if obj is None:
            continue
        
        if isinstance(obj, pd.Series):
            cols |= set(obj.index)
        elif isinstance(obj, pd.DataFrame):
            cols |= set(obj.columns)
        elif isinstance(obj, dict):
            # For dicts (like features), extract columns from all DataFrames/Series
            for key, value in obj.items():
                if isinstance(value, pd.Series):
                    cols |= set(value.index)
                elif isinstance(value, pd.DataFrame):
                    cols |= set(value.columns)
        else:
            logger.warning(f"[UniverseConsistency] Unknown object type: {type(obj)}")
    
    print(sorted(cols))
    return cols


def universe_consistency_report(env, combined_results: Dict) -> Dict:
    """
    Track which assets exist at each stage of the pipeline.
    
    Stages tracked:
    1. Raw prices (from MarketData)
    2. Continuous prices (from MarketData)
    3. Contract IDs (from MarketData)
    4. Features (from FeatureService)
    5. Individual sleeve signals (for each sleeve)
    6. Combined signals (from CombinedStrategy)
    7. Allocator weights (final weights through time)
    
    Args:
        env: Environment object with:
            - market (MarketData instance)
            - features (dict of feature DataFrames)
            - sleeves (list of sleeve instances)
            - combined_strategy (CombinedStrategy instance)
        combined_results: Dict returned from run_combined_validation with:
            - weights (pd.DataFrame): weights through time
    
    Returns:
        Dict with universe sets for each stage
    """
    print("\n" + "=" * 80)
    print("UNIVERSE CONSISTENCY REPORT")
    print("=" * 80)
    
    market = env.market
    
    # Get prices_raw, prices_cont, contract_ids
    try:
        prices_raw = market.prices_raw
        prices_cont = market.prices_cont
        contract_ids = market.contract_ids
    except Exception as e:
        logger.warning(f"[UniverseConsistency] Error getting prices: {e}")
        prices_raw = pd.DataFrame()
        prices_cont = pd.DataFrame()
        contract_ids = pd.DataFrame()
    
    # 1) Raw & continuous prices
    raw_univ = print_universe_stage("raw prices", [prices_raw])
    cont_univ = print_universe_stage("continuous prices", [prices_cont])
    cid_univ = print_universe_stage("contract_ids", [contract_ids])
    
    # 2) Features universe
    features = env.features if hasattr(env, 'features') else {}
    feat_univ = print_universe_stage("features", [features])
    
    # 3) Sleeve signals (for each sleeve individually)
    sleeve_univs = {}
    sleeves = env.sleeves if hasattr(env, 'sleeves') else []
    combined_strategy = env.combined_strategy if hasattr(env, 'combined_strategy') else None
    
    # Get a sample date for testing signals
    if not prices_cont.empty:
        sample_date = prices_cont.index[-1]
    else:
        sample_date = pd.Timestamp.now()
    
    for sleeve in sleeves:
        try:
            # Get sleeve name
            sleeve_name = getattr(sleeve, 'name', None)
            if not sleeve_name:
                sleeve_name = sleeve.__class__.__name__
            
            # Try to get signals from this sleeve
            # Different sleeves have different signal methods
            strategy_sigs = None
            
            # Check if sleeve is in combined_strategy's strategies dict
            if combined_strategy and hasattr(combined_strategy, 'strategies'):
                for name, strategy in combined_strategy.strategies.items():
                    if strategy is sleeve:
                        # Get features for this strategy type
                        features_dict = None
                        if name == "sr3_carry_curve":
                            features_dict = features.get("SR3_CURVE")
                        elif name == "rates_curve":
                            features_dict = features.get("RATES_CURVE")
                        elif name == "fx_commod_carry":
                            features_dict = features.get("CARRY_FX_COMMOD")
                        elif name == "tsmom":
                            features_dict = features.get("LONG_MOMENTUM")
                        elif name in ["tsmom_med", "medium_momentum"]:
                            features_dict = features.get("MEDIUM_MOMENTUM")
                        elif name in ["tsmom_short", "short_momentum"]:
                            features_dict = features.get("SHORT_MOMENTUM")
                        
                        # Try to call signals method
                        if hasattr(sleeve, 'signals'):
                            import inspect
                            sig = inspect.signature(sleeve.signals)
                            if 'features' in sig.parameters:
                                strategy_sigs = sleeve.signals(market, sample_date, features=features_dict)
                            else:
                                strategy_sigs = sleeve.signals(market, sample_date)
                        break
            
            if strategy_sigs is None:
                # Fallback: try direct call
                if hasattr(sleeve, 'signals'):
                    try:
                        strategy_sigs = sleeve.signals(market, sample_date)
                    except Exception:
                        pass
            
            if strategy_sigs is not None and isinstance(strategy_sigs, pd.Series):
                u = print_universe_stage(f"sleeve '{sleeve_name}' signals", [strategy_sigs])
                sleeve_univs[sleeve_name] = u
            else:
                print(f"\n=== Universe at stage: sleeve '{sleeve_name}' signals ===")
                print("[WARN] Could not extract signals from this sleeve")
                sleeve_univs[sleeve_name] = set()
                
        except Exception as e:
            logger.warning(f"[UniverseConsistency] Error getting signals from sleeve {sleeve_name}: {e}")
            sleeve_univs[sleeve_name] = set()
    
    # 4) Combined signals
    comb_univ = set()
    if combined_strategy:
        try:
            combined_signals = combined_strategy.signals(market, sample_date)
            if isinstance(combined_signals, pd.Series):
                comb_univ = print_universe_stage("combined signals", [combined_signals])
            else:
                print("\n=== Universe at stage: combined signals ===")
                print("[WARN] Combined signals is not a Series")
        except Exception as e:
            logger.warning(f"[UniverseConsistency] Error getting combined signals: {e}")
            print("\n=== Universe at stage: combined signals ===")
            print(f"[WARN] Error: {e}")
    else:
        print("\n=== Universe at stage: combined signals ===")
        print("[WARN] No combined_strategy available")
    
    # 5) Allocated weights (through time)
    weights = combined_results.get("weights", pd.DataFrame())
    w_univ = print_universe_stage("allocator weights", [weights])
    
    # 6) Missing / dropped assets summary
    print("\n" + "=" * 80)
    print("UNIVERSE DIFF SUMMARY")
    print("=" * 80)
    
    def diff(label_from: str, from_univ: Set[str], label_to: str, to_univ: Set[str]):
        """Print differences between two universes."""
        missing = sorted(from_univ - to_univ)
        if missing:
            print(f"\n{label_from} but NOT in {label_to}:")
            for sym in missing:
                print(f"  - {sym}")
        else:
            print(f"\n[OK] No assets lost going from {label_from} to {label_to}.")
    
    diff("raw", raw_univ, "continuous", cont_univ)
    diff("continuous", cont_univ, "features", feat_univ)
    
    # Compare features to combined signals (via all sleeves)
    all_sleeve_univ = set()
    for sleeve_name, sleeve_univ in sleeve_univs.items():
        all_sleeve_univ |= sleeve_univ
    
    if all_sleeve_univ:
        diff("features", feat_univ, "sleeve signals (union)", all_sleeve_univ)
        diff("sleeve signals (union)", all_sleeve_univ, "combined signals", comb_univ)
    else:
        diff("features", feat_univ, "combined signals", comb_univ)
    
    diff("combined signals", comb_univ, "allocator weights", w_univ)
    
    # Per-sleeve breakdown
    if sleeve_univs:
        print("\n" + "=" * 80)
        print("PER-SLEEVE UNIVERSE BREAKDOWN")
        print("=" * 80)
        for sleeve_name, sleeve_univ in sleeve_univs.items():
            print(f"\nSleeve '{sleeve_name}': {len(sleeve_univ)} symbols")
            print(f"  Symbols: {sorted(sleeve_univ)}")
            missing_from_features = sorted(feat_univ - sleeve_univ)
            if missing_from_features:
                print(f"  Missing from features: {missing_from_features}")
    
    return {
        "raw": raw_univ,
        "cont": cont_univ,
        "contract_ids": cid_univ,
        "features": feat_univ,
        "sleeves": sleeve_univs,
        "combined": comb_univ,
        "weights": w_univ,
    }

