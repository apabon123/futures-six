"""
Combined Validation: Minimal validation of full pipeline.

Validates that the complete strategy pipeline runs end-to-end without
numerical explosions, NaNs, or infinities.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
from datetime import datetime


def validate_combined_strategy(
    combined_strategy,
    macro_overlay: Optional[object],
    vol_overlay: object,
    allocator: object,
    exec_sim: object,
    market: object,
    start: str,
    end: str,
    risk_vol: object,
    features: Optional[Dict] = None
) -> Dict:
    """
    Minimal combined validation:
    - Run full backtest via ExecSim
    - Check for NaNs/infinities
    - Check basic stats
    - Check exposure sanity
    - Check rolling Sharpe
    
    Args:
        combined_strategy: CombinedStrategy instance
        macro_overlay: MacroRegimeFilter instance (optional)
        vol_overlay: VolManagedOverlay instance
        allocator: Allocator instance
        exec_sim: ExecSim instance
        market: MarketData instance
        start: Start date for backtest
        end: End date for backtest
        risk_vol: RiskVol instance
        features: Optional dict of pre-computed features
    
    Returns:
        Dict with validation results and data
    """
    print("=" * 70)
    print("Combined Strategy Validation")
    print("=" * 70)
    
    # Package components for ExecSim
    components = {
        'strategy': combined_strategy,
        'overlay': vol_overlay,
        'risk_vol': risk_vol,
        'allocator': allocator
    }
    
    if macro_overlay is not None:
        components['macro_overlay'] = macro_overlay
    
    # Run backtest
    print("\n[1/5] Running full backtest...")
    try:
        results = exec_sim.run(
            market=market,
            start=start,
            end=end,
            components=components
        )
    except Exception as e:
        print(f"[ERROR] Backtest failed: {e}")
        return {
            "signals": pd.DataFrame(),
            "weights": pd.DataFrame(),
            "returns": pd.Series(dtype=float),
            "sharpe": np.nan,
            "error": str(e)
        }
    
    # Extract results
    equity_curve = results.get('equity_curve', pd.Series(dtype=float))
    weights_panel = results.get('weights_panel', pd.DataFrame())
    signals_panel = results.get('signals_panel', pd.DataFrame())
    
    # 2) Check for NaNs in signals
    print("\n[2/5] Checking signals for NaNs...")
    if not signals_panel.empty:
        if signals_panel.isna().any().any():
            print("[WARN] Raw combined signals contain NaNs")
        else:
            print("[OK] No NaNs in signals")
    else:
        print("[WARN] Signals panel is empty")
    
    # 3) Check for NaNs in weights
    print("\n[3/5] Checking weights for NaNs...")
    if not weights_panel.empty:
        if weights_panel.isna().any().any():
            print("[WARN] Allocator produced NaNs in combined validation")
        else:
            print("[OK] No NaNs in weights")
    else:
        print("[WARN] Weights panel is empty")
    
    # 4) Check returns for NaNs/infinities
    print("\n[4/5] Checking returns for NaNs/infinities...")
    if not equity_curve.empty:
        # Convert equity curve to returns
        returns = equity_curve.pct_change().dropna()
        
        if returns.isna().any():
            print("[WARN] Returns contain NaNs")
        
        if np.isinf(returns).any():
            print("[WARN] Returns contain infinities")
        
        if len(returns) > 0:
            # Basic stats
            mean = returns.mean()
            vol = returns.std()
            sharpe = mean / vol * np.sqrt(252) if vol != 0 else 0
            
            print(f"[Combined] Sharpe={sharpe:.2f}, mean={mean:.5f}, vol={vol:.5f}")
            
            # Check for extreme values
            if abs(sharpe) > 10:
                print(f"[WARN] Extreme Sharpe ratio: {sharpe:.2f}")
            
            if vol > 1.0:
                print(f"[WARN] Extreme volatility: {vol:.5f}")
        else:
            sharpe = np.nan
            print("[WARN] No valid returns")
    else:
        returns = pd.Series(dtype=float)
        sharpe = np.nan
        print("[WARN] Equity curve is empty")
    
    # 5) Exposure sanity check
    print("\n[5/5] Checking exposure sanity...")
    if not weights_panel.empty:
        gross = weights_panel.abs().sum(axis=1)
        net = weights_panel.sum(axis=1)
        
        gross_max = gross.max()
        net_max = net.abs().max()
        
        print(f"[Combined] Gross max={gross_max:.2f}, Net max={net_max:.2f}")
        
        # Check against allocator constraints
        if hasattr(allocator, 'gross_cap'):
            if gross_max > allocator.gross_cap + 1e-6:
                print(f"[WARN] Gross exposure exceeding cap: {gross_max:.2f} > {allocator.gross_cap}")
        
        if hasattr(allocator, 'net_cap'):
            if net_max > allocator.net_cap + 1e-6:
                print(f"[WARN] Net exposure exceeding cap: {net_max:.2f} > {allocator.net_cap}")
        
        # Check for negative gross (shouldn't happen)
        gross_min = gross.min()
        if gross_min < 0:
            print(f"[WARN] Negative gross exposure detected: {gross_min:.2f}")
    else:
        print("[WARN] Cannot check exposure (weights panel is empty)")
    
    # 6) Rolling Sharpe check
    print("\n[6/6] Checking rolling Sharpe...")
    if not returns.empty and len(returns) >= 252:
        rolling_sharpe = returns.rolling(252).mean() / returns.rolling(252).std() * np.sqrt(252)
        rolling_sharpe = rolling_sharpe.dropna()
        
        if not rolling_sharpe.empty:
            if rolling_sharpe.isna().any():
                print("[WARN] Rolling Sharpe contains NaNs")
            
            if np.isinf(rolling_sharpe).any():
                print("[WARN] Rolling Sharpe contains infinities")
            
            sharpe_min = rolling_sharpe.min()
            sharpe_max = rolling_sharpe.max()
            
            print(f"[Combined] Rolling Sharpe: min={sharpe_min:.2f}, max={sharpe_max:.2f}")
            
            if sharpe_min < -10 or sharpe_max > 10:
                print(f"[WARN] Extreme rolling Sharpe values detected")
        else:
            print("[WARN] No valid rolling Sharpe values")
    else:
        print(f"[WARN] Insufficient data for rolling Sharpe (need >= 252 days, got {len(returns)})")
    
    print("\n" + "=" * 70)
    
    return {
        "signals": signals_panel,
        "weights": weights_panel,
        "returns": returns if not returns.empty else equity_curve.pct_change().dropna(),
        "equity_curve": equity_curve,
        "sharpe": sharpe,
        "report": results.get('report', {})
    }


def run_combined_validation(env):
    """
    Run combined validation using environment object.
    
    Args:
        env: Environment object with attributes:
            - combined_strategy (or strategy)
            - macro_overlay (optional)
            - vol_overlay (or overlay)
            - allocator
            - exec_sim
            - market
            - start
            - end
            - risk_vol
            - features (optional)
    
    Returns:
        Dict with validation results
    """
    return validate_combined_strategy(
        combined_strategy=getattr(env, 'combined_strategy', getattr(env, 'strategy', None)),
        macro_overlay=getattr(env, 'macro_overlay', getattr(env, 'macro', None)),
        vol_overlay=getattr(env, 'vol_overlay', getattr(env, 'overlay', None)),
        allocator=getattr(env, 'allocator', getattr(env, 'alloc', None)),
        exec_sim=getattr(env, 'exec_sim', None),
        market=getattr(env, 'market', None),
        start=getattr(env, 'start', None),
        end=getattr(env, 'end', None),
        risk_vol=getattr(env, 'risk_vol', getattr(env, 'risk', None)),
        features=getattr(env, 'features', None)
    )

