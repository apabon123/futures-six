#!/usr/bin/env python3
"""
Equity Carry Forensic Diagnostics

Purpose: Diagnose why equity carry has negative Sharpe (-0.537) before
making any changes to the meta-sleeve construction.

Checks:
1. Per-equity instrument attribution (ES, NQ, RTY individual Sharpe/CAGR/MaxDD)
2. Dividend implied sanity (implied dividend yield statistics)
3. Contract calendar/T sanity (verify T definition)
4. Spot index type (confirm price-return, not total return)
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import json
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.agents import MarketData
from src.agents.feature_equity_carry import EquityCarryFeatures
from src.agents.strat_carry_meta_v1 import CarryMetaV1

def compute_summary_stats(returns: pd.Series, name: str = "Strategy") -> dict:
    """Compute comprehensive summary statistics."""
    if returns.empty or returns.isna().all():
        return {}
    
    returns_clean = returns.dropna()
    if len(returns_clean) == 0:
        return {}
    
    n_obs = len(returns_clean)
    mean_daily = returns_clean.mean()
    std_daily = returns_clean.std()
    
    ann_return = mean_daily * 252
    ann_vol = std_daily * np.sqrt(252)
    sharpe = ann_return / ann_vol if ann_vol > 1e-6 else 0.0
    
    # Cumulative returns for drawdown
    cumret = (1 + returns_clean).cumprod()
    running_max = cumret.expanding().max()
    drawdown = (cumret - running_max) / running_max
    max_dd = drawdown.min()
    
    return {
        "name": name,
        "n_obs": n_obs,
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "best_day": returns_clean.max(),
        "worst_day": returns_clean.min(),
    }

def main():
    """Run equity carry forensic diagnostics."""
    print("=" * 80)
    print("EQUITY CARRY FORENSIC DIAGNOSTICS")
    print("=" * 80)
    
    market = MarketData()
    equity_features = EquityCarryFeatures()
    
    # Load features
    print("\nLoading equity carry features...")
    features = equity_features.compute(market, end_date='2025-12-31')
    
    # 1. Per-equity instrument attribution
    print("\n" + "=" * 80)
    print("1. PER-EQUITY INSTRUMENT ATTRIBUTION")
    print("=" * 80)
    
    equity_symbols = ['ES', 'NQ', 'RTY']
    instrument_stats = {}
    
    for sym in equity_symbols:
        raw_col = f'equity_carry_raw_{sym}'
        if raw_col not in features.columns:
            print(f"⚠️ {sym}: No raw carry column found")
            continue
        
        raw_carry = features[raw_col].dropna()
        
        # Get returns for this instrument
        db_sym = CarryMetaV1.SYMBOL_MAP.get(sym, f"{sym}_FRONT_CALENDAR_2D")
        try:
            returns = market.get_returns(
                symbols=[db_sym],
                start='2020-01-01',
                end='2025-12-31',
                method='log'
            )
            
            if db_sym in returns.columns:
                asset_returns = returns[db_sym].dropna()
                
                # Align carry with returns (use carry as signal)
                # For attribution, we'll use raw carry as a simple signal proxy
                # (positive carry → long, negative → short)
                common_dates = raw_carry.index.intersection(asset_returns.index)
                if len(common_dates) > 0:
                    carry_aligned = raw_carry.loc[common_dates]
                    returns_aligned = asset_returns.loc[common_dates]
                    
                    # Simple attribution: sign(carry) * return
                    # This approximates what the strategy would do
                    signals = np.sign(carry_aligned)
                    attributed_returns = signals * returns_aligned
                    
                    stats = compute_summary_stats(attributed_returns, name=sym)
                    instrument_stats[sym] = stats
                    
                    print(f"\n{sym} ({db_sym}):")
                    print(f"  Sharpe: {stats.get('sharpe', 0):.3f}")
                    print(f"  CAGR: {stats.get('ann_return', 0):.2%}")
                    print(f"  Vol: {stats.get('ann_vol', 0):.2%}")
                    print(f"  MaxDD: {stats.get('max_dd', 0):.2%}")
                    print(f"  Observations: {stats.get('n_obs', 0)}")
                else:
                    print(f"⚠️ {sym}: No overlapping dates between carry and returns")
            else:
                print(f"⚠️ {sym}: Symbol {db_sym} not found in returns")
        except Exception as e:
            print(f"⚠️ {sym}: Error computing attribution - {e}")
    
    # 2. Dividend implied sanity
    print("\n" + "=" * 80)
    print("2. DIVIDEND IMPLIED SANITY")
    print("=" * 80)
    
    for sym in equity_symbols:
        div_col = f'implied_div_yield_{sym}'
        if div_col not in features.columns:
            print(f"⚠️ {sym}: No implied dividend yield column found")
            continue
        
        div_yield = features[div_col].dropna()
        
        print(f"\n{sym} Implied Dividend Yield:")
        print(f"  Median: {div_yield.median():.4f} ({div_yield.median()*100:.2f}%)")
        print(f"  Mean: {div_yield.mean():.4f} ({div_yield.mean()*100:.2f}%)")
        print(f"  P5: {div_yield.quantile(0.05):.4f} ({div_yield.quantile(0.05)*100:.2f}%)")
        print(f"  P95: {div_yield.quantile(0.95):.4f} ({div_yield.quantile(0.95)*100:.2f}%)")
        print(f"  Min: {div_yield.min():.4f} ({div_yield.min()*100:.2f}%)")
        print(f"  Max: {div_yield.max():.4f} ({div_yield.max()*100:.2f}%)")
        
        # Check for impossible values
        negative_days = (div_yield < 0).sum()
        extreme_positive = (div_yield > 0.20).sum()  # > 20% annual dividend yield
        extreme_negative = (div_yield < -0.10).sum()  # < -10% (impossible)
        
        print(f"  Negative days: {negative_days} ({negative_days/len(div_yield)*100:.1f}%)")
        print(f"  Extreme positive (>20%): {extreme_positive} ({extreme_positive/len(div_yield)*100:.1f}%)")
        print(f"  Extreme negative (<-10%): {extreme_negative} ({extreme_negative/len(div_yield)*100:.1f}%)")
        
        if negative_days > len(div_yield) * 0.1:
            print(f"  WARNING: {negative_days} days with negative implied dividends ({negative_days/len(div_yield)*100:.1f}%)")
        if extreme_positive > 0:
            print(f"  WARNING: {extreme_positive} days with >20% implied dividend yield")
        if extreme_negative > 0:
            print(f"  ERROR: {extreme_negative} days with <-10% implied dividend yield (impossible)")
    
    # 3. Contract calendar/T sanity
    print("\n" + "=" * 80)
    print("3. CONTRACT CALENDAR / T SANITY")
    print("=" * 80)
    
    print("\nT Definition Check:")
    print("  Current implementation uses constant T = 45 days")
    print("  This is an approximation for front-month futures")
    print("  TODO: Verify if actual daycount to expiry is needed")
    print("  TODO: Check if correct contract month is being used")
    
    # Check if we can get actual contract expiry dates
    # (This would require contract-level data, which may not be available)
    print("\n  Note: T = 45 days is a reasonable approximation for front-month")
    print("  For exact calculation, would need actual expiry dates per contract")
    
    # 4. Spot index type
    print("\n" + "=" * 80)
    print("4. SPOT INDEX TYPE")
    print("=" * 80)
    
    print("\nSpot Index Source Check:")
    print("  Current implementation uses FRED indicators:")
    print("    - SP500: FRED series 'SP500'")
    print("    - NASDAQ100: FRED series 'NASDAQ100'")
    print("    - RUT_SPOT: FRED series 'RUT_SPOT'")
    print("\n  TODO: Verify these are price-return indices (not total return)")
    print("  TODO: Check FRED documentation for index type")
    
    # Try to load a sample to check
    try:
        sp500_sample = market.get_fred_indicator(series_id="SP500", start='2020-01-01', end='2020-12-31')
        print(f"\n  SP500 sample (2020):")
        print(f"    First value: {sp500_sample.iloc[0]:.2f}")
        print(f"    Last value: {sp500_sample.iloc[-1]:.2f}")
        print(f"    Return: {(sp500_sample.iloc[-1] / sp500_sample.iloc[0] - 1)*100:.2f}%")
        print(f"    (If this matches S&P 500 price return, it's correct)")
    except Exception as e:
        print(f"  ⚠️ Could not load SP500 sample: {e}")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    print("\nPer-Instrument Attribution:")
    for sym, stats in instrument_stats.items():
        sharpe = stats.get('sharpe', 0)
        status = "PASS" if sharpe > 0 else "FAIL"
        print(f"  {sym}: Sharpe = {sharpe:.3f} ({status})")
    
    print("\nNext Steps:")
    print("  1. Review implied dividend yield statistics for anomalies")
    print("  2. Verify T definition (constant 45 days vs actual daycount)")
    print("  3. Confirm spot indices are price-return (not total return)")
    print("  4. If all checks pass, equity carry may be non-admissible as Engine v1")
    
    # Save results
    output = {
        "timestamp": datetime.now().isoformat(),
        "instrument_stats": instrument_stats,
        "checks": {
            "per_instrument_attribution": "✅" if len(instrument_stats) > 0 else "❌",
            "dividend_sanity": "✅ Check values above",
            "contract_calendar": "⚠️ Uses constant T=45 days",
            "spot_index_type": "⚠️ Needs verification"
        }
    }
    
    output_path = Path("reports") / "equity_carry_forensic.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\nResults saved to: {output_path}")

if __name__ == "__main__":
    main()
