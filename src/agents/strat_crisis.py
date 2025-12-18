"""
Crisis Meta-Sleeve: Always-on crisis hedges for tail risk mitigation.

Phase-0 Implementation:
- Crisis Sleeve A: Long VX2 (always-on)
- Crisis Sleeve B: Long VX2 - VX1 spread (always-on, dollar-neutral)
- Crisis Sleeve C: Long Duration (UB or ZN, always-on)

Design Principles:
- Always-on exposure (no signals, no timing)
- Small fixed weight (5% of portfolio capital)
- No macro filters
- No vol targeting
- No allocator logic
- No interaction with other sleeves

These sleeves are evaluated on tail behavior, not average returns.
"""

import logging
from typing import Optional, Union, Sequence
from datetime import datetime
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class CrisisVX2Long:
    """
    Crisis Sleeve A: Long VX2 (Always-On).
    
    Instrument: VX2 continuous future
    Position: Always long VX2
    Constant notional exposure
    
    Purpose: Pure convexity benchmark, establish upper bound on crisis protection.
    """
    
    def __init__(self, symbol: str = "VX2"):
        """
        Initialize Crisis VX2 Long sleeve.
        
        Args:
            symbol: VX2 symbol (default: "VX2")
        """
        self.symbol = symbol
        logger.info(f"[CrisisVX2Long] Initialized with symbol={symbol}")
    
    def signals(
        self,
        market,
        date: Union[str, datetime],
        universe: Optional[Sequence[str]] = None
    ) -> pd.Series:
        """
        Get always-on long VX2 signal.
        
        Args:
            market: MarketData instance
            date: Date to get signals for
            universe: Ignored (crisis sleeve trades VX2, not universe assets)
            
        Returns:
            Series with single entry: VX2 signal = 1.0 (always long)
        """
        # Always return long signal = 1.0
        return pd.Series({self.symbol: 1.0})


class CrisisVXSpread:
    """
    Crisis Sleeve B: Long VX2 - VX1 Spread (Always-On).
    
    Instruments:
        - VX2 (long)
        - VX1 (short)
    Construction: Dollar-neutral spread, equal notional long/short
    No dynamic scaling
    
    Purpose: Reduced carry bleed, convex response to term-structure inversion.
    """
    
    def __init__(self, vx2_symbol: str = "VX2", vx1_symbol: str = "VX1"):
        """
        Initialize Crisis VX Spread sleeve.
        
        Args:
            vx2_symbol: VX2 symbol (default: "VX2")
            vx1_symbol: VX1 symbol (default: "VX1")
        """
        self.vx2_symbol = vx2_symbol
        self.vx1_symbol = vx1_symbol
        logger.info(
            f"[CrisisVXSpread] Initialized with vx2={vx2_symbol}, vx1={vx1_symbol}"
        )
    
    def signals(
        self,
        market,
        date: Union[str, datetime],
        universe: Optional[Sequence[str]] = None
    ) -> pd.Series:
        """
        Get always-on VX2-VX1 spread signal (long VX2, short VX1).
        
        Args:
            market: MarketData instance
            date: Date to get signals for
            universe: Ignored (crisis sleeve trades VX spreads, not universe assets)
            
        Returns:
            Series with two entries:
                - VX2: 1.0 (long)
                - VX1: -1.0 (short)
        """
        # Dollar-neutral spread: long VX2, short VX1
        return pd.Series({
            self.vx2_symbol: 1.0,   # Long VX2
            self.vx1_symbol: -1.0   # Short VX1
        })


class CrisisDurationLong:
    """
    Crisis Sleeve C: Long Duration (Always-On).
    
    Instrument: UB (preferred) or ZN (alternative)
    Position: Always long
    Constant exposure
    
    Purpose: Flight-to-safety hedge, non-volatility-based crisis protection.
    """
    
    def __init__(self, symbol: str = "UB"):
        """
        Initialize Crisis Duration Long sleeve.
        
        Args:
            symbol: Duration instrument symbol (default: "UB", alternative: "ZN")
        """
        self.symbol = symbol
        logger.info(f"[CrisisDurationLong] Initialized with symbol={symbol}")
    
    def signals(
        self,
        market,
        date: Union[str, datetime],
        universe: Optional[Sequence[str]] = None
    ) -> pd.Series:
        """
        Get always-on long duration signal.
        
        Args:
            market: MarketData instance
            date: Date to get signals for
            universe: Ignored (crisis sleeve trades duration, not universe assets)
            
        Returns:
            Series with single entry: Duration signal = 1.0 (always long)
        """
        # Always return long signal = 1.0
        return pd.Series({self.symbol: 1.0})

