"""
Sleeve Validation: Validate each sleeve independently.

Backtests each sleeve alone to check:
- Sharpe ratio
- Turnover
- Correlation to other sleeves
- Basic distribution of signals
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime


def validate_sleeve(sleeve, market, features: Optional[Dict] = None, date: Optional[datetime] = None):
    """
    Compute sleeve-only signals and basic stats.
    
    Args:
        sleeve: Strategy sleeve instance (e.g., TSMOM, SR3CarryCurve)
        market: MarketData instance
        features: Optional dict of pre-computed features
        date: Optional date for signal computation (default: last available)
    
    Returns:
        Signal DataFrame [date x symbol]
    """
    # Get sleeve name
    sleeve_name = getattr(sleeve, 'name', sleeve.__class__.__name__)
    
    # For validation, we need to get signals across multiple dates
    # If date is provided, use it; otherwise get last available date
    if date is None:
        # Get last available date from market
        prices = market.get_price_panel(
            symbols=market.universe[:1],  # Just one symbol to get dates
            fields=("close",),
            tidy=False
        )
        if prices.empty:
            print(f"[WARN] No price data for {sleeve_name}")
            return pd.DataFrame()
        date = prices.index[-1]
    
    # Get signals for this date
    # Sleeves have signals(market, date, features=...) method
    try:
        if features is not None:
            # Determine which features this sleeve needs
            # This is a simplified approach - in practice, you'd know which features each sleeve needs
            sig = sleeve.signals(market, date, features=features)
        else:
            sig = sleeve.signals(market, date)
        
        if isinstance(sig, pd.Series):
            sig = sig.to_frame().T
            sig.index = [date]
        
        # Basic stats
        mean = sig.mean().mean()
        std = sig.std().mean()
        print(f"[{sleeve_name}] signal mean={mean:.3f}, std={std:.3f}")
        
        # Look for dead signals
        if sig.std().sum() == 0:
            print(f"[WARN] Sleeve {sleeve_name} signal is constant.")
        
        return sig
    
    except Exception as e:
        print(f"[WARN] Error getting signals from {sleeve_name}: {e}")
        return pd.DataFrame()


def backtest_sleeve_only(
    sleeve,
    market,
    exec_sim,
    start: str,
    end: str,
    features: Optional[Dict] = None,
    risk_vol=None,
    overlay=None,
    allocator=None
):
    """
    Backtest a single sleeve independently.
    
    Args:
        sleeve: Strategy sleeve instance
        market: MarketData instance
        exec_sim: ExecSim instance
        start: Start date
        end: End date
        features: Optional dict of pre-computed features
        risk_vol: RiskVol instance (required for ExecSim)
        overlay: VolManagedOverlay instance (required for ExecSim)
        allocator: Allocator instance (required for ExecSim)
    
    Returns:
        Tuple of (signals DataFrame, returns Series)
    """
    sleeve_name = getattr(sleeve, 'name', sleeve.__class__.__name__)
    
    # Create a minimal CombinedStrategy with just this sleeve
    from src.agents.strat_combined import CombinedStrategy
    
    # Wrap sleeve in CombinedStrategy
    combined_strategy = CombinedStrategy(
        strategies={sleeve_name: sleeve},
        weights={sleeve_name: 1.0},
        features=features
    )
    
    # Run backtest
    components = {
        'strategy': combined_strategy,
        'overlay': overlay,
        'risk_vol': risk_vol,
        'allocator': allocator
    }
    
    results = exec_sim.run(market, start, end, components)
    
    returns = results.get('equity_curve', pd.Series(dtype=float))
    
    if len(returns) > 0:
        # Calculate Sharpe (annualized)
        returns_daily = returns.pct_change().dropna()
        if len(returns_daily) > 0 and returns_daily.std() > 0:
            sharpe = returns_daily.mean() / returns_daily.std() * np.sqrt(252)
            print(f"[{sleeve_name}] Sharpe={sharpe:.2f}")
        else:
            print(f"[{sleeve_name}] Sharpe=N/A (insufficient data)")
    else:
        print(f"[{sleeve_name}] Sharpe=N/A (no returns)")
    
    signals_panel = results.get('signals_panel', pd.DataFrame())
    
    return signals_panel, returns


def sleeve_correlation_table(sleeve_signals: Dict[str, pd.DataFrame]):
    """
    Compute correlation matrix of sleeve signals.
    
    Args:
        sleeve_signals: Dict mapping sleeve name to signal DataFrame [date x symbol]
    
    Returns:
        Correlation matrix DataFrame
    """
    # Convert each sleeve's signals to a single time series (mean across symbols)
    signal_series = {}
    
    for name, sig_df in sleeve_signals.items():
        if not sig_df.empty:
            # Take mean across symbols for each date
            signal_series[name] = sig_df.mean(axis=1)
    
    if not signal_series:
        print("[WARN] No sleeve signals available for correlation")
        return pd.DataFrame()
    
    # Combine into DataFrame
    df = pd.DataFrame(signal_series)
    
    # Align by index (take intersection of all dates)
    df = df.dropna()
    
    if df.empty:
        print("[WARN] No overlapping dates for correlation")
        return pd.DataFrame()
    
    # Compute correlation
    corr = df.corr()
    
    print("\nSleeve Signal Correlation Matrix:")
    print(corr)
    
    return corr


def run_sleeve_validation(
    all_sleeves: List,
    market,
    exec_sim,
    start: str,
    end: str,
    features: Optional[Dict] = None,
    risk_vol=None,
    overlay=None,
    allocator=None
):
    """
    Run validation for all sleeves.
    
    Args:
        all_sleeves: List of sleeve instances
        market: MarketData instance
        exec_sim: ExecSim instance
        start: Start date for backtest
        end: End date for backtest
        features: Optional dict of pre-computed features
        risk_vol: RiskVol instance (required)
        overlay: VolManagedOverlay instance (required)
        allocator: Allocator instance (required)
    """
    print("=" * 70)
    print("Sleeve Validation")
    print("=" * 70)
    
    sleeve_sigs = {}
    
    for sleeve in all_sleeves:
        sleeve_name = getattr(sleeve, 'name', sleeve.__class__.__name__)
        print(f"\n[{sleeve_name}] Running validation...")
        
        try:
            sig, returns = backtest_sleeve_only(
                sleeve, market, exec_sim, start, end,
                features=features,
                risk_vol=risk_vol,
                overlay=overlay,
                allocator=allocator
            )
            sleeve_sigs[sleeve_name] = sig
        except Exception as e:
            print(f"[ERROR] Failed to backtest {sleeve_name}: {e}")
            continue
    
    # Compute correlation table
    if len(sleeve_sigs) > 1:
        print("\n" + "=" * 70)
        sleeve_correlation_table(sleeve_sigs)
    
    print("\n" + "=" * 70)

