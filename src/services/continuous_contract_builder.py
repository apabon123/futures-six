"""
ContinuousContractBuilder: Build back-adjusted continuous price series.

This module provides the backward-panama back-adjustment algorithm to create
smooth continuous price series from raw contract data. Used for all "what did
the market do?" calculations (returns, features, vol, covariance, P&L).

Raw prices (from DB) are used for sizing/notional calculations only.
"""

import logging
from pathlib import Path
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class ContinuousContractBuilder:
    """
    Builds back-adjusted continuous price series from raw contract data.
    
    Uses backward-panama style adjustment:
    - At each roll point (contract_id changes), compute gap between old and new contract
    - Accumulate that gap into an adjustment term
    - Subtract cumulative adjustment from all subsequent prices
    - Result: continuous series has no jumps, but level differences between
      contracts are preserved as history
    """
    
    def build_back_adjusted(self, df: pd.DataFrame) -> pd.Series:
        """
        Build back-adjusted continuous price series from raw contract data.
        
        Args:
            df: DataFrame with columns ["close", "contract_id"], index sorted by date.
                - index: date (sorted ascending)
                - close: raw settlement/close price
                - contract_id: contract identifier (e.g., "ESZ2024", "ESH2025")
                
        Returns:
            pd.Series of back-adjusted 'close' prices with same index as input.
            First price is unadjusted (no roll has occurred yet).
        """
        # Validate input
        if df.empty:
            logger.warning("[ContinuousContractBuilder] Empty DataFrame provided")
            return pd.Series(dtype=float)
        
        required_cols = {"close", "contract_id"}
        missing_cols = required_cols - set(df.columns)
        if missing_cols:
            raise ValueError(
                f"Missing required columns: {missing_cols}. "
                f"Required: {required_cols}, got: {set(df.columns)}"
            )
        
        # Sort by date (ensure ascending)
        df = df.sort_index()
        
        # Initialize continuous series with raw prices
        cont = df["close"].copy()
        
        # Track cumulative adjustment
        adj = 0.0
        
        # Iterate forward; when contract_id changes, update adjustment
        for i in range(1, len(df)):
            cid_now = df["contract_id"].iloc[i]
            cid_prev = df["contract_id"].iloc[i - 1]
            
            if cid_now != cid_prev:
                # Roll day: compute gap between new and old contract
                raw_new = df["close"].iloc[i]
                raw_old = df["close"].iloc[i - 1]
                gap = raw_new - raw_old
                
                # Accumulate adjustment
                adj += gap
                
                logger.debug(
                    f"[ContinuousContractBuilder] Roll detected at {df.index[i]}: "
                    f"{cid_prev} -> {cid_now}, gap={gap:.2f}, cumulative_adj={adj:.2f}"
                )
            
            # Apply cumulative adjustment to current price
            cont.iloc[i] = df["close"].iloc[i] - adj
        
        # First point: just use raw (no adjustment yet)
        cont.iloc[0] = df["close"].iloc[0]
        
        return cont

