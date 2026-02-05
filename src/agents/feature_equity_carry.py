"""
Equity Carry Features: Basis-Only Equity Carry (No SOFR Dependency).

Computes equity carry signals from spot indices and futures prices using basis-only formula:

    carry_eq(t) = (1/T) * ln(S_t / F_t) = d_t - r_t

Where:
    - S_t = spot index price
    - F_t = futures price
    - T = time to maturity (in years)
    - d_t = implied dividend yield
    - r_t = funding rate (SOFR)

This is mathematically equivalent to d - r but computed directly from basis without SOFR,
making it more robust and avoiding SOFR unit mismatch issues.

Positive carry (S > F): Backwardation → Long futures
Negative carry (S < F): Contango → Short futures

Implied dividend yield is also computed for diagnostic purposes (with sanity checks):
    d_t = r_t - (1/T) * ln(F_t / S_t)

Sanity bounds: -5% to +10% (values outside range set to NaN, diagnostic only)

Features computed (per equity index):
- equity_carry_raw_{symbol}: Raw carry signal (basis-only, ln(S/F)/T)
- equity_carry_z_{symbol}: Z-scored carry signal
- implied_div_yield_{symbol}: Implied dividend yield (diagnostic, with sanity bounds)
"""

import logging
from typing import Optional, Union, List, Dict
from datetime import datetime
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def _zscore_rolling(
    s: pd.Series,
    window: int = 252,
    clip: float = 3.0,
    min_periods: Optional[int] = None
) -> pd.Series:
    """
    Compute rolling z-score with clipping.
    
    Args:
        s: Input series
        window: Rolling window size in days
        clip: Z-score clipping bounds
        min_periods: Minimum periods for rolling calculation (default: window // 2)
        
    Returns:
        Z-scored and clipped series
    """
    if min_periods is None:
        min_periods = max(window // 2, 126)  # At least 126 days (half year)
    
    mu = s.rolling(window=window, min_periods=min_periods).mean()
    sigma = s.rolling(window=window, min_periods=min_periods).std()
    z = (s - mu) / sigma
    return z.clip(-clip, clip)


class EquityCarryFeatures:
    """
    Builds equity carry features from spot indices, futures, and SOFR.
    
    Computes implied dividend yield and equity carry signal for each equity index.
    
    Required data:
    - Spot indices: SP500, NASDAQ100, RUT_SPOT (price-return indices, NOT total return)
    - Futures: ES, NQ, RTY (front contract continuous series)
    - Funding: SOFR (annualized rate)
    
    Features computed (3 per equity):
    - equity_carry_raw_{symbol}: Raw carry signal (r - d)
    - equity_carry_z_{symbol}: Z-scored carry signal (time-series standardized)
    - implied_div_yield_{symbol}: Implied dividend yield (informational)
    """
    
    # Mapping: futures symbol -> spot index symbol
    EQUITY_MAP = {
        "ES": "SP500",           # S&P 500 E-mini futures -> SPX price return
        "NQ": "NASDAQ100",       # Nasdaq-100 E-mini futures -> NDX price return
        "RTY": "RUT_SPOT"        # Russell 2000 futures -> RUT price return
    }
    
    def __init__(
        self,
        futures_symbols: Optional[List[str]] = None,
        window: int = 252,
        clip: float = 3.0,
        contract_multiplier: Optional[Dict[str, float]] = None
    ):
        """
        Initialize equity carry features calculator.
        
        Args:
            futures_symbols: List of futures symbols (default: ["ES", "NQ", "RTY"])
            window: Rolling window for z-score standardization (default: 252 days)
            clip: Z-score clipping bounds (default: 3.0)
            contract_multiplier: Optional dict of contract multipliers for DV01 calc
        """
        self.futures_symbols = futures_symbols or ["ES", "NQ", "RTY"]
        self.window = window
        self.clip = clip
        self.contract_multiplier = contract_multiplier or {}
        
        # Validate that all futures have spot mappings
        for fut in self.futures_symbols:
            if fut not in self.EQUITY_MAP:
                raise ValueError(
                    f"No spot index mapping for futures symbol: {fut}. "
                    f"Available: {list(self.EQUITY_MAP.keys())}"
                )
    
    def compute(
        self,
        market,
        end_date: Optional[Union[str, datetime]] = None
    ) -> pd.DataFrame:
        """
        Compute equity carry features up to end_date.
        
        Args:
            market: MarketData instance
            end_date: End date for feature computation (default: last available)
            
        Returns:
            DataFrame indexed by date with columns:
            - equity_carry_raw_{symbol}: Raw carry signal (r - d)
            - equity_carry_z_{symbol}: Z-scored carry signal
            - implied_div_yield_{symbol}: Implied dividend yield (informational)
        """
        features_dict = {}
        all_dates = None
        
        # Get SOFR (funding rate) from FRED observations table
        try:
            sofr_data = market.get_fred_indicator(
                series_id="SOFR",
                start=None,
                end=end_date
            )
            
            if sofr_data.empty:
                logger.warning(
                    "[EquityCarry] SOFR data not found in f_fred_observations. "
                    "Using placeholder constant rate of 4.5% (0.045)."
                )
                sofr_data = pd.Series(0.045, index=pd.date_range(
                    start="2010-01-01", end=end_date or pd.Timestamp.now(), freq='D'
                ))
            else:
                # SOFR normalization: check if it's in percent or decimal
                # Heuristic: if median > 1.0, it's in percent (e.g., 5.3), divide by 100
                if sofr_data.median() > 1.0:
                    sofr_data = sofr_data / 100.0
                    logger.debug(f"[EquityCarry] SOFR converted from percent to decimal")
                logger.info(f"[EquityCarry] Loaded SOFR data: {len(sofr_data)} observations")
        
        except Exception as e:
            logger.warning(
                f"[EquityCarry] Error loading SOFR: {e}. "
                f"Using placeholder constant rate."
            )
            sofr_data = pd.Series(0.045, index=pd.date_range(
                start="2010-01-01", end=end_date or pd.Timestamp.now(), freq='D'
            ))
        
        # Iterate through each futures symbol
        for fut_sym in self.futures_symbols:
            spot_sym = self.EQUITY_MAP[fut_sym]
            
            try:
                # 1. Load futures prices (front contract)
                # Futures symbols in database: ES_FRONT_CALENDAR_2D, NQ_FRONT_CALENDAR_2D, RTY_FRONT_CALENDAR_2D
                db_fut_sym = f"{fut_sym}_FRONT_CALENDAR_2D"
                
                fut_prices = market.get_price_panel(
                    symbols=[db_fut_sym],
                    fields=("close",),
                    start=None,
                    end=end_date,
                    tidy=False
                )
                
                if fut_prices.empty:
                    logger.warning(
                        f"[EquityCarry] No futures data found for {db_fut_sym}. "
                        f"Skipping {fut_sym}."
                    )
                    continue
                
                # get_price_panel with one field returns DataFrame with symbol as column
                F = fut_prices[db_fut_sym]
                
                # 2. Load spot index prices from FRED observations table
                try:
                    spot_series = market.get_fred_indicator(
                        series_id=spot_sym,
                        start=None,
                        end=end_date
                    )
                    
                    if spot_series.empty:
                        logger.warning(
                            f"[EquityCarry] No spot index data found for {spot_sym} in f_fred_observations. "
                            f"Skipping {fut_sym}."
                        )
                        continue
                    
                    S = spot_series
                    logger.debug(f"[EquityCarry] Loaded {spot_sym}: {len(S)} observations")
                
                except Exception as e:
                    logger.warning(
                        f"[EquityCarry] Error loading spot index {spot_sym}: {e}. "
                        f"Skipping {fut_sym}."
                    )
                    continue
                
                # 3. Align dates and handle missing data
                # Combine all series on common dates
                df = pd.DataFrame({
                    'F': F,
                    'S': S,
                    'r': sofr_data
                })
                
                # Forward-fill and back-fill to handle minor gaps
                df = df.ffill().bfill()
                
                # Canonical NA handling: drop rows with ANY NaN (PROCEDURES.md requirement)
                rows_before = len(df)
                df = df.dropna(how="any")
                rows_dropped = rows_before - len(df)
                if rows_dropped > 0:
                    logger.debug(
                        f"[EquityCarry] Dropped {rows_dropped} rows with NaN "
                        f"(before: {rows_before}, after: {len(df)})"
                    )
                
                if len(df) < self.window:
                    logger.warning(
                        f"[EquityCarry] Insufficient data for {fut_sym}: "
                        f"{len(df)} days, need {self.window}. Continuing anyway."
                    )
                
                # 4. Compute time to maturity (T)
                # For front continuous contracts with 2-day calendar roll, approximate T
                # Typical time to expiry for equity index futures: ~45-60 days on average
                # Guard against tiny T: use min_days = 7 days minimum (prevents explosion)
                T_days = 45  # Average days to expiry for front equity futures
                min_days = 7  # Minimum days to expiry (prevents explosion when T is tiny)
                T_days = max(T_days, min_days)
                T_years = T_days / 365.25
                
                # Log T statistics for monitoring
                logger.debug(
                    f"[EquityCarry] {fut_sym}: Using T = {T_days} days "
                    f"({T_years:.4f} years)"
                )
                
                # 5. Compute equity carry (basis-only, no SOFR required)
                # Formula: carry_eq(t) = (1/T) * ln(S/F) = d - r
                # This is the net cost of carry: positive = backwardation (long), negative = contango (short)
                log_ratio = np.log(df['S'] / df['F'])  # ln(S/F), not ln(F/S)
                carry_raw = log_ratio / T_years  # This is (d - r), the equity carry
                
                # Handle any inf or extreme values
                carry_raw = carry_raw.replace([np.inf, -np.inf], np.nan)
                
                # 6. Compute implied dividend yield (diagnostic only, with sanity checks)
                # d_t = r_t - (1/T) * ln(F_t / S_t) = r_t + (1/T) * ln(S_t / F_t)
                # Since ln(S/F) = -ln(F/S), we have: d = r + carry_raw
                # But we need SOFR for this, so compute from basis:
                log_ratio_fs = np.log(df['F'] / df['S'])  # ln(F/S) for dividend calc
                basis_net = log_ratio_fs / T_years  # This is (r - d)
                d_implied = df['r'] - basis_net  # This is d
                
                # Handle any inf or extreme values
                carry_raw = carry_raw.replace([np.inf, -np.inf], np.nan)
                
                # Sanity checks for implied dividend (diagnostic only, don't let it break carry)
                # Dividend yields for broad indices should be reasonable: -5% to +10%
                d_implied = d_implied.replace([np.inf, -np.inf], np.nan)
                d_valid_mask = (d_implied >= -0.05) & (d_implied <= 0.10)
                d_implied_sane = d_implied.copy()
                d_implied_sane[~d_valid_mask] = np.nan
                
                # Log extreme values for debugging
                extreme_count = (~d_valid_mask).sum()
                if extreme_count > 0:
                    logger.warning(
                        f"[EquityCarryFeatures] {fut_sym}: {extreme_count} days with "
                        f"implied dividend outside [-5%, +10%] range. Setting to NaN."
                    )
                
                # 7. Standardize carry with rolling z-score
                carry_z = _zscore_rolling(
                    carry_raw,
                    window=self.window,
                    clip=self.clip,
                    min_periods=max(self.window // 2, 126)
                )
                
                # Store features
                features_dict[f"equity_carry_raw_{fut_sym}"] = carry_raw
                features_dict[f"equity_carry_z_{fut_sym}"] = carry_z
                features_dict[f"implied_div_yield_{fut_sym}"] = d_implied_sane  # Sanitized version
                
                # Track dates for alignment
                if all_dates is None:
                    all_dates = carry_raw.index
                else:
                    all_dates = all_dates.union(carry_raw.index)
                
                logger.debug(
                    f"[EquityCarry] Computed features for {fut_sym}: "
                    f"{len(carry_raw.dropna())} non-null values"
                )
            
            except Exception as e:
                logger.warning(
                    f"[EquityCarry] Error computing features for {fut_sym}: {e}"
                )
                continue
        
        if not features_dict:
            logger.warning("[EquityCarry] No features could be computed")
            return pd.DataFrame()
        
        # Combine into DataFrame
        features = pd.DataFrame(features_dict, index=all_dates)
        features = features.sort_index()
        
        logger.info(
            f"[EquityCarry] Computed features for {len(features)} dates "
            f"(non-null: {features.notna().sum().to_dict()})"
        )
        
        return features
