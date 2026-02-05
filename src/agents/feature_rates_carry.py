"""
Rates Carry Features: Rolldown / Curve Slope for Treasury Futures.

Computes rates carry signals from term structure slope (rolldown) for
Treasury futures (ZT, ZF, ZN, UB).

Carry definition for rates:
    carry_rates(t) = (P_near(t) - P_far(t)) / (T_far - T_near)

Where:
    - P_near = price of near contract (front)
    - P_far = price of far contract (second)
    - T_near = maturity of near contract
    - T_far = maturity of far contract

Interpretation:
    - Positive carry => upward sloping curve => long duration carry
    - Negative carry => inverted curve => short duration carry

For Phase-0, we use a simplified approach:
    - Near contract = front (rank 0)
    - Far contract = second (rank 1)
    - Approximate maturity spacing based on contract type

Features computed (per rates future):
- rates_carry_raw_{symbol}: Raw rolldown signal
- rates_carry_z_{symbol}: Z-scored rolldown signal (time-series standardized)
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


class RatesCarryFeatures:
    """
    Builds rates carry features from Treasury futures term structure.
    
    Computes rolldown/curve slope as a carry signal for each rates future.
    
    Required data:
    - Rates futures: ZT, ZF, ZN, UB (front and second contract continuous series)
    
    Features computed (2 per rates future):
    - rates_carry_raw_{symbol}: Raw rolldown signal (curve slope)
    - rates_carry_z_{symbol}: Z-scored rolldown signal
    """
    
    # Maturity spacing approximations (in years) for each Treasury future
    # These are rough estimates of the spacing between front and second contracts
    MATURITY_SPACING = {
        "ZT": 0.25,    # 2-Year: ~3 month spacing
        "ZF": 0.25,    # 5-Year: ~3 month spacing
        "ZN": 0.25,    # 10-Year: ~3 month spacing
        "UB": 0.25     # Ultra: ~3 month spacing
    }
    
    def __init__(
        self,
        symbols: Optional[List[str]] = None,
        window: int = 252,
        clip: float = 3.0
    ):
        """
        Initialize rates carry features calculator.
        
        Args:
            symbols: List of rates symbols (default: ["ZT", "ZF", "ZN", "UB"])
            window: Rolling window for z-score standardization (default: 252 days)
            clip: Z-score clipping bounds (default: 3.0)
        """
        self.symbols = symbols or ["ZT", "ZF", "ZN", "UB"]
        self.window = window
        self.clip = clip
    
    def compute(
        self,
        market,
        end_date: Optional[Union[str, datetime]] = None
    ) -> pd.DataFrame:
        """
        Compute rates carry features up to end_date.
        
        Args:
            market: MarketData instance
            end_date: End date for feature computation (default: last available)
            
        Returns:
            DataFrame indexed by date with columns:
            - rates_carry_raw_{symbol}: Raw rolldown signal
            - rates_carry_z_{symbol}: Z-scored rolldown signal
        """
        features_dict = {}
        all_dates = None
        
        for symbol in self.symbols:
            try:
                # Load rank 0 (front) and rank 1 (second) contracts
                # Database symbols: ZT_FRONT_VOLUME, ZT_RANK_1_VOLUME, etc.
                # **NOTE**: Rank 1 continuous series need to exist in database
                close = market.get_contracts_by_root(
                    root=symbol,
                    ranks=[0, 1],
                    fields=("close",),
                    start=None,
                    end=end_date
                )
                
                if close.empty:
                    logger.warning(
                        f"[RatesCarry] No data found for {symbol}. "
                        f"This requires rank 0 AND rank 1 continuous series to exist in the database. "
                        f"**ACTION REQUIRED**: Load rank 1 continuous series for {symbol}."
                    )
                    continue
                
                # Ensure we have both ranks
                if 0 not in close.columns or 1 not in close.columns:
                    logger.warning(
                        f"[RatesCarry] Missing required ranks for {symbol}. "
                        f"Available: {list(close.columns)}. "
                        f"**ACTION REQUIRED**: Load rank 1 continuous series."
                    )
                    continue
                
                # Extract front and second contract prices
                P_near = close[0]
                P_far = close[1]
                
                # Forward-fill and back-fill to handle minor gaps
                P_near = P_near.ffill().bfill()
                P_far = P_far.ffill().bfill()
                
                # Check for sufficient data
                complete_mask = (P_near.notna() & P_far.notna())
                complete_dates = complete_mask.sum()
                
                if complete_dates < self.window:
                    logger.warning(
                        f"[RatesCarry] Insufficient complete data for {symbol}: "
                        f"{complete_dates} days with both ranks, need {self.window}. "
                        f"Continuing anyway."
                    )
                
                # Get maturity spacing for this symbol
                T_spacing = self.MATURITY_SPACING.get(symbol, 0.25)
                
                # Compute rolldown (curve slope)
                # carry_rates = (P_near - P_far) / T_spacing
                # Positive => upward sloping curve (near more expensive than far)
                # Negative => inverted curve (near cheaper than far)
                carry_raw = (P_near - P_far) / T_spacing
                
                # Handle any inf or extreme values
                carry_raw = carry_raw.replace([np.inf, -np.inf], np.nan)
                
                # Standardize with rolling z-score
                carry_z = _zscore_rolling(
                    carry_raw,
                    window=self.window,
                    clip=self.clip,
                    min_periods=max(self.window // 2, 126)
                )
                
                # Store features
                features_dict[f"rates_carry_raw_{symbol}"] = carry_raw
                features_dict[f"rates_carry_z_{symbol}"] = carry_z
                
                # Track dates for alignment
                if all_dates is None:
                    all_dates = carry_raw.index
                else:
                    all_dates = all_dates.union(carry_raw.index)
                
                logger.debug(
                    f"[RatesCarry] Computed features for {symbol}: "
                    f"{len(carry_raw.dropna())} non-null values"
                )
            
            except Exception as e:
                logger.warning(
                    f"[RatesCarry] Error computing features for {symbol}: {e}"
                )
                continue
        
        if not features_dict:
            logger.warning("[RatesCarry] No features could be computed")
            return pd.DataFrame()
        
        # Combine into DataFrame
        features = pd.DataFrame(features_dict, index=all_dates)
        features = features.sort_index()
        
        logger.info(
            f"[RatesCarry] Computed features for {len(features)} dates "
            f"(non-null: {features.notna().sum().to_dict()})"
        )
        
        return features
