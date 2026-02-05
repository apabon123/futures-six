#!/usr/bin/env python3
"""
Debug Carry Phase-1 Equity Carry Issue

Purpose: Investigate why equity carry has negative Sharpe (-0.537) despite
positive raw carry values (mean ES = 0.72, mean NQ = 11.51).

Checks:
1. Raw carry values vs signals
2. Z-score computation
3. Vol normalization
4. Sign logic (positive carry → long?)
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.agents import MarketData
from src.agents.strat_carry_meta_v1 import CarryMetaV1
from src.agents.feature_equity_carry import EquityCarryFeatures

# Load data
market = MarketData()
carry = CarryMetaV1(phase=1, window=252, clip=3.0)
features = carry._compute_all_features(market, end_date='2025-12-31')

# Check equity carry
print("=" * 80)
print("EQUITY CARRY DIAGNOSTIC")
print("=" * 80)

# Sample dates
sample_dates = pd.date_range('2020-06-01', '2025-06-01', freq='MS')[:10]

print("\nSample Equity Carry Values:")
print("-" * 80)
for date in sample_dates:
    if date in features.index:
        raw_es = features.loc[date, 'equity_carry_raw_ES'] if 'equity_carry_raw_ES' in features.columns else np.nan
        raw_nq = features.loc[date, 'equity_carry_raw_NQ'] if 'equity_carry_raw_NQ' in features.columns else np.nan
        z_es = features.loc[date, 'equity_carry_z_ES'] if 'equity_carry_z_ES' in features.columns else np.nan
        
        # Get Phase-1 signal
        sig = carry.signals(market, date, features=features)
        sig_es = sig.get('ES_FRONT_CALENDAR_2D', 0)
        
        print(f"{date.strftime('%Y-%m-%d')}: "
              f"raw_ES={raw_es:.4f}, z_ES={z_es:.4f}, signal_ES={sig_es:.4f}")

# Check correlation: raw carry vs signal
print("\n" + "=" * 80)
print("CORRELATION ANALYSIS")
print("=" * 80)

if 'equity_carry_raw_ES' in features.columns:
    raw_es = features['equity_carry_raw_ES'].dropna()
    
    # Get signals for all dates
    signals_series = []
    for date in raw_es.index:
        try:
            sig = carry.signals(market, date, features=features)
            sig_es = sig.get('ES_FRONT_CALENDAR_2D', 0)
            signals_series.append(sig_es)
        except:
            signals_series.append(0.0)
    
    signals_es = pd.Series(signals_series, index=raw_es.index)
    
    # Align
    common = raw_es.index.intersection(signals_es.index)
    if len(common) > 0:
        raw_aligned = raw_es.loc[common]
        sig_aligned = signals_es.loc[common]
        
        corr = raw_aligned.corr(sig_aligned)
        print(f"\nCorrelation (raw_ES vs signal_ES): {corr:.4f}")
        print(f"Expected: Positive (positive carry -> positive signal)")
        
        if corr < 0:
            print("⚠️ WARNING: Negative correlation suggests sign error!")
        
        # Check sign alignment
        same_sign = ((raw_aligned > 0) & (sig_aligned > 0)).sum() + ((raw_aligned < 0) & (sig_aligned < 0)).sum()
        total = len(common)
        print(f"Same sign: {same_sign}/{total} ({same_sign/total*100:.1f}%)")

print("\n" + "=" * 80)
print("VOL NORMALIZATION CHECK")
print("=" * 80)

# Check vol values used
try:
    returns = market.get_returns(
        symbols=['ES_FRONT_CALENDAR_2D', 'ZT_FRONT_VOLUME'],
        start='2020-01-01',
        end='2025-12-31',
        method='log'
    )
    
    vol_es = returns['ES_FRONT_CALENDAR_2D'].rolling(252, min_periods=63).std() * np.sqrt(252)
    vol_zt = returns['ZT_FRONT_VOLUME'].rolling(252, min_periods=63).std() * np.sqrt(252)
    
    print(f"\nES Vol: min={vol_es.min():.4f}, median={vol_es.median():.4f}, max={vol_es.max():.4f}")
    print(f"ZT Vol: min={vol_zt.min():.4f}, median={vol_zt.median():.4f}, max={vol_zt.max():.4f}")
    print(f"\nVol ratio (ES/ZT): median={vol_es.median()/vol_zt.median():.2f}x")
    print("(If ES vol >> ZT vol, then ES signals get smaller after vol normalization)")
    
except Exception as e:
    print(f"Could not compute vol: {e}")

print("\n" + "=" * 80)
