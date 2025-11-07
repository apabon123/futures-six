"""
RollJumpFilter: Overlay that makes momentum signals robust to roll gaps.

Filters non-back-adjusted roll jumps from returns used in signal construction.
Does NOT modify raw prices used for P&L calculations - filter is applied only
to the signal construction pipeline.

Two modes:
- "drop": Remove flagged dates (set to 0)
- "winsorize": Clip flagged returns to ±threshold_bp

Pure function. No data writes. Deterministic outputs.
"""

import logging
from typing import Optional
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class RollJumpFilter:
    """
    Filter to make momentum signals robust to non-back-adjusted roll gaps.
    
    Takes returns DataFrame used for signal construction and a jumps DataFrame
    from MarketData.flag_roll_jumps(), then applies filtering to reduce the
    impact of roll jumps on momentum calculations.
    
    Two filtering modes:
    - "drop": Set flagged returns to 0 (removes roll days from signal)
    - "winsorize": Clip flagged returns to ±threshold_bp (caps extreme moves)
    
    This filter is applied ONLY to returns used for signal building, not to
    prices used for P&L calculations.
    """
    
    def __init__(
        self,
        threshold_bp: float = 100.0,
        mode: str = "winsorize"
    ):
        """
        Initialize RollJumpFilter.
        
        Args:
            threshold_bp: Threshold in basis points for filtering (default: 100 = 1%)
                         In winsorize mode, returns are clipped to ±threshold_bp
            mode: Filtering mode - "drop" or "winsorize" (default: "winsorize")
        
        Raises:
            ValueError: If parameters are invalid
        """
        # Validate parameters
        if threshold_bp <= 0:
            raise ValueError(f"threshold_bp must be > 0, got {threshold_bp}")
        
        if mode not in ("drop", "winsorize"):
            raise ValueError(f"mode must be 'drop' or 'winsorize', got '{mode}'")
        
        self.threshold_bp = threshold_bp
        self.mode = mode
        
        logger.info(
            f"[RollJumpFilter] Initialized: threshold_bp={self.threshold_bp:.1f}, "
            f"mode={self.mode}"
        )
    
    def apply(
        self,
        returns_df: pd.DataFrame,
        jumps_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Apply roll jump filter to returns DataFrame.
        
        Args:
            returns_df: Returns DataFrame (wide format: index=dates, columns=symbols)
                       Simple returns for signal building
            jumps_df: Jumps DataFrame from MarketData.flag_roll_jumps()
                     Tidy format with columns [date, symbol, return, flagged]
        
        Returns:
            Filtered returns DataFrame with same shape/index/columns as input
        """
        # Handle empty inputs
        if returns_df.empty:
            logger.debug("[RollJumpFilter] Empty returns_df, returning as-is")
            return returns_df.copy()
        
        if jumps_df.empty:
            logger.debug("[RollJumpFilter] No jumps to filter, returning as-is")
            return returns_df.copy()
        
        # Make a copy to avoid modifying original
        filtered = returns_df.copy()
        
        # Convert threshold to decimal for calculations
        threshold_decimal = self.threshold_bp / 10000
        
        # Track how many jumps we filter
        n_filtered = 0
        
        # Process each flagged jump
        for _, row in jumps_df.iterrows():
            date = pd.to_datetime(row['date'])
            symbol = row['symbol']
            
            # Check if this date/symbol exists in returns_df
            if date not in filtered.index or symbol not in filtered.columns:
                continue
            
            original_return = filtered.loc[date, symbol]
            
            # Skip if NaN
            if pd.isna(original_return):
                continue
            
            # Apply filtering based on mode
            if self.mode == "drop":
                # Set to 0 (removes this day from signal calculation)
                filtered.loc[date, symbol] = 0.0
                n_filtered += 1
                
                logger.debug(
                    f"[RollJumpFilter] DROP: {date.date()} {symbol} "
                    f"return={original_return:.4f} -> 0.0"
                )
            
            elif self.mode == "winsorize":
                # Clip to ±threshold
                if original_return > threshold_decimal:
                    filtered.loc[date, symbol] = threshold_decimal
                    n_filtered += 1
                    logger.debug(
                        f"[RollJumpFilter] WINSORIZE: {date.date()} {symbol} "
                        f"return={original_return:.4f} -> {threshold_decimal:.4f}"
                    )
                elif original_return < -threshold_decimal:
                    filtered.loc[date, symbol] = -threshold_decimal
                    n_filtered += 1
                    logger.debug(
                        f"[RollJumpFilter] WINSORIZE: {date.date()} {symbol} "
                        f"return={original_return:.4f} -> {-threshold_decimal:.4f}"
                    )
                # Otherwise leave unchanged
        
        if n_filtered > 0:
            logger.info(
                f"[RollJumpFilter] Filtered {n_filtered} roll jumps "
                f"(mode={self.mode}, threshold={self.threshold_bp:.1f}bp)"
            )
        else:
            logger.debug("[RollJumpFilter] No jumps exceeded threshold")
        
        return filtered
    
    def describe(self) -> dict:
        """
        Return configuration and description of the RollJumpFilter.
        
        Returns:
            dict with configuration parameters
        """
        return {
            'agent': 'RollJumpFilter',
            'role': 'Filter roll jumps from signal construction returns',
            'threshold_bp': self.threshold_bp,
            'mode': self.mode,
            'modes_available': {
                'drop': 'Set flagged returns to 0',
                'winsorize': 'Clip flagged returns to ±threshold_bp'
            },
            'outputs': ['apply(returns_df, jumps_df)'],
            'note': 'Applied only to signal construction, not P&L prices'
        }

