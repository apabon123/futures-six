"""
RiskVol: Covariance & Risk Targeting Agent

Provides rolling returns, volatilities, and a shrunk covariance matrix 
for the current date. Ensures no look-ahead, no data writes, and 
deterministic outputs given a MarketData snapshot.
"""

import logging
from pathlib import Path
from typing import Optional, Union
from datetime import datetime

import yaml
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class RiskVol:
    """
    Risk & Volatility agent for covariance matrix and volatility calculations.
    
    Provides:
    - Rolling annualized volatilities per asset
    - Shrunk covariance matrix (Ledoit-Wolf or diagonal blend)
    - Mask of tradable symbols at a given date
    
    All calculations are point-in-time (no look-ahead) and deterministic
    given the MarketData snapshot.
    """
    
    def __init__(
        self,
        cov_lookback: Optional[int] = None,
        vol_lookback: Optional[int] = None,
        shrinkage: Optional[str] = None,
        nan_policy: Optional[str] = None,
        config_path: str = "configs/strategies.yaml"
    ):
        """
        Initialize RiskVol agent.
        
        Args:
            cov_lookback: Rolling window for covariance (default: from config or 252)
            vol_lookback: Rolling window for volatility (default: from config or 63)
            shrinkage: Shrinkage method ("lw" for Ledoit-Wolf, "none" for sample)
            nan_policy: How to handle NaN ("drop-row" or "mask-asset")
            config_path: Path to configuration YAML file
        """
        # Load config defaults
        config = self._load_config(config_path)
        config_defaults = {}
        if config and 'risk_vol' in config:
            config_defaults = config['risk_vol']
        
        # Set parameters: explicit args > config > hardcoded defaults
        self.cov_lookback = (
            cov_lookback if cov_lookback is not None
            else config_defaults.get('cov_lookback', 252)
        )
        self.vol_lookback = (
            vol_lookback if vol_lookback is not None
            else config_defaults.get('vol_lookback', 63)
        )
        self.shrinkage = (
            shrinkage if shrinkage is not None
            else config_defaults.get('shrinkage', 'lw')
        )
        self.nan_policy = (
            nan_policy if nan_policy is not None
            else config_defaults.get('nan_policy', 'mask-asset')
        )
        
        # Validate parameters
        if self.cov_lookback < 2:
            raise ValueError(f"cov_lookback must be >= 2, got {self.cov_lookback}")
        
        if self.vol_lookback < 2:
            raise ValueError(f"vol_lookback must be >= 2, got {self.vol_lookback}")
        
        if self.shrinkage not in ("lw", "none"):
            raise ValueError(f"shrinkage must be 'lw' or 'none', got {self.shrinkage}")
        
        if self.nan_policy not in ("drop-row", "mask-asset"):
            raise ValueError(f"nan_policy must be 'drop-row' or 'mask-asset', got {self.nan_policy}")
        
        logger.info(
            f"[RiskVol] Initialized: cov_lookback={self.cov_lookback}, "
            f"vol_lookback={self.vol_lookback}, shrinkage={self.shrinkage}, "
            f"nan_policy={self.nan_policy}"
        )
    
    def _load_config(self, config_path: str) -> Optional[dict]:
        """Load configuration from YAML file."""
        path = Path(config_path)
        if not path.exists():
            logger.warning(f"[RiskVol] Config file not found: {config_path}, using defaults")
            return None
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            logger.warning(f"[RiskVol] Failed to load config: {e}, using defaults")
            return None
    
    def _get_returns_window(
        self,
        market,
        date: Union[str, datetime],
        lookback: int,
        signals: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """
        Get returns window up to and including date.
        
        Args:
            market: MarketData instance
            date: End date for window
            lookback: Number of days to look back
            signals: Optional signals Series to use for filtering universe
            
        Returns:
            DataFrame of log returns [dates x symbols]
        """
        date = pd.to_datetime(date)
        
        # Get all returns up to date (using asof if available)
        # Use continuous returns for covariance/vol calculations
        returns = market.returns_cont
        
        if returns.empty:
            raise ValueError(f"No returns data available up to {date}")
        
        # If signals provided, filter to signal universe first
        if signals is not None and not signals.empty:
            sig_cols = list(signals.index)
            logger.debug(f"[RISK] signals.columns: {sorted(sig_cols)}")
            # Only use columns that exist in returns
            available_cols = [c for c in sig_cols if c in returns.columns]
            if len(available_cols) < len(sig_cols):
                missing = set(sig_cols) - set(available_cols)
                logger.warning(f"[RISK] Some signal symbols missing from returns_cont: {sorted(missing)}")
            returns = returns[available_cols]
        
        # Ensure date is in index
        if date not in returns.index:
            # Find the last available date <= target date
            valid_dates = returns.index[returns.index <= date]
            if len(valid_dates) == 0:
                raise ValueError(f"No data available on or before {date}")
            date = valid_dates[-1]
        
        # Get position of date in index
        date_pos = returns.index.get_loc(date)
        
        # Calculate start position
        start_pos = max(0, date_pos - lookback + 1)
        
        # Extract window
        window = returns.iloc[start_pos:date_pos + 1]
        
        logger.debug(f"[RISK] returns_window initial columns ({date}): {sorted(window.columns)}")
        logger.debug(f"[RISK] NaN counts per column:")
        nan_counts = window.isna().sum().sort_values(ascending=False)
        for col, count in nan_counts.items():
            if count > 0:
                logger.debug(f"[RISK]   {col}: {count}/{len(window)} NaNs ({100*count/len(window):.1f}%)")
        
        if len(window) < lookback:
            raise ValueError(
                f"Insufficient history: need {lookback} days, got {len(window)} "
                f"(date={date})"
            )
        
        return window
    
    def vols(
        self,
        market,
        date: Union[str, datetime],
        signals: Optional[pd.Series] = None
    ) -> pd.Series:
        """
        Calculate annualized volatility for each symbol at a given date.
        
        Args:
            market: MarketData instance
            date: Date to calculate volatility for
            signals: Optional signals Series to use for filtering universe
            
        Returns:
            pd.Series of annualized volatilities indexed by symbol
        """
        # Get returns window
        window = self._get_returns_window(market, date, self.vol_lookback, signals)
        
        # Calculate standard deviation per symbol
        if self.nan_policy == "drop-row":
            # Drop rows where any symbol is NaN
            clean_window = window.dropna(axis=0, how='any')
            if len(clean_window) < 2:
                raise ValueError(
                    f"After dropping NaN rows, insufficient data: "
                    f"{len(clean_window)} rows remaining"
                )
            vol = clean_window.std(ddof=1)
        else:  # mask-asset
            # Calculate per-column std, ignoring NaN
            vol = window.std(ddof=1)
        
        # Annualize with √252
        vol_ann = vol * np.sqrt(252)
        
        # Remove any symbols with NaN volatility
        dropped = vol_ann.isna().sum()
        vol_ann = vol_ann.dropna()
        
        if dropped > 0:
            logger.debug(f"[RiskVol] Dropped {dropped} symbols with NaN volatility")
        
        if len(vol_ann) == 0:
            raise ValueError(f"No valid volatilities calculated for date {date}")
        
        logger.debug(f"[RiskVol] vols({date}): {len(vol_ann)} symbols")
        
        return vol_ann
    
    def covariance(
        self,
        market,
        date: Union[str, datetime],
        signals: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """
        Calculate shrunk covariance matrix at a given date.
        
        Args:
            market: MarketData instance
            date: Date to calculate covariance for
            signals: Optional signals Series to use for filtering universe
            
        Returns:
            pd.DataFrame covariance matrix [symbols x symbols], annualized
        """
        # Get returns window
        window = self._get_returns_window(market, date, self.cov_lookback, signals)
        
        # Handle NaN policy
        if self.nan_policy == "drop-row":
            # Drop rows where any symbol is NaN
            clean_window = window.dropna(axis=0, how='any')
            if len(clean_window) < 2:
                raise ValueError(
                    f"After dropping NaN rows, insufficient data: "
                    f"{len(clean_window)} rows remaining"
                )
        else:  # mask-asset
            # Drop columns that are all NaN
            clean_window = window.dropna(axis=1, how='all')
            logger.debug(f"[RISK] returns_window after dropna(axis=1, how='all'): {sorted(clean_window.columns)}")
            
            if clean_window.shape[1] == 0:
                raise ValueError(f"No valid symbols with data in window ending {date}")
            
            # Allow some missing data; drop dates that have any NaNs
            # Only drop a column if it has too few observations
            min_obs = min(self.cov_lookback // 2, 60)  # Require at least 50% or 60 obs, whichever is smaller
            valid_cols = [c for c in clean_window.columns if clean_window[c].count() >= min_obs]
            
            if len(valid_cols) < len(clean_window.columns):
                dropped_cols = set(clean_window.columns) - set(valid_cols)
                logger.debug(f"[RISK] Dropped {len(dropped_cols)} columns with < {min_obs} observations: {sorted(dropped_cols)}")
                clean_window = clean_window[valid_cols]
            
            # For mask-asset, we need to drop rows with ANY NaN for covariance calculation
            # (covariance requires pairwise complete observations)
            clean_window = clean_window.dropna(axis=0, how='any')
            logger.debug(f"[RISK] returns_window after dropna(axis=0): {sorted(clean_window.columns)}, {len(clean_window)} rows")
            
            if len(clean_window) < 2:
                raise ValueError(
                    f"Insufficient complete observations for covariance: "
                    f"{len(clean_window)} rows"
                )
        
        symbols = clean_window.columns
        
        # Calculate covariance with shrinkage
        if self.shrinkage == "lw":
            try:
                from sklearn.covariance import LedoitWolf
                
                lw = LedoitWolf()
                cov_matrix = lw.fit(clean_window.values).covariance_
                cov = pd.DataFrame(cov_matrix, index=symbols, columns=symbols)
                
                logger.debug(f"[RiskVol] Applied Ledoit-Wolf shrinkage for date {date}")
                
            except ImportError:
                logger.warning(
                    "[RiskVol] sklearn not available for Ledoit-Wolf, "
                    "using diagonal blend fallback"
                )
                # Diagonal blend fallback
                sample_cov = clean_window.cov(ddof=1)
                diag = np.diag(np.diag(sample_cov.values))
                # Simple 50/50 blend as fallback
                cov_matrix = 0.5 * sample_cov.values + 0.5 * diag
                cov = pd.DataFrame(cov_matrix, index=symbols, columns=symbols)
                
        else:  # shrinkage == "none"
            cov = clean_window.cov(ddof=1)
        
        # Annualize with 252
        cov_ann = cov * 252
        
        # Add min vol floor (50 bps annualized) to avoid exploding leverage in ultra-calm regimes
        min_vol_floor = 0.005  # 50 bps = 0.5% annualized
        for sym in symbols:
            vol_sq = cov_ann.loc[sym, sym]
            if vol_sq < min_vol_floor ** 2:
                cov_ann.loc[sym, sym] = min_vol_floor ** 2
        
        # Ensure symmetry (numerical precision)
        cov_ann = (cov_ann + cov_ann.T) / 2
        
        logger.debug(
            f"[RiskVol] covariance({date}): {cov_ann.shape[0]}x{cov_ann.shape[1]} matrix"
        )
        
        return cov_ann
    
    def mask(
        self,
        market,
        date: Union[str, datetime],
        signals: Optional[pd.Series] = None
    ) -> pd.Index:
        """
        Get tradable symbols at a given date.
        
        Returns symbols that have sufficient history and valid data
        for both volatility and covariance calculations.
        
        Args:
            market: MarketData instance
            date: Date to check tradability
            signals: Optional signals Series to use for filtering universe
            
        Returns:
            pd.Index of tradable symbol names
        """
        try:
            # Try to calculate volatilities - this will tell us which symbols are valid
            vol = self.vols(market, date, signals)
            tradable = vol.index
            
            logger.debug(f"[RiskVol] mask({date}): {len(tradable)} tradable symbols")
            
            return tradable
            
        except Exception as e:
            logger.warning(f"[RiskVol] Could not determine mask for {date}: {e}")
            return pd.Index([])
    
    def describe(self) -> dict:
        """
        Return configuration and description of the RiskVol agent.
        
        Returns:
            dict with configuration parameters
        """
        return {
            'agent': 'RiskVol',
            'role': 'Provide rolling returns, volatilities, and shrunk covariance matrix',
            'cov_lookback': self.cov_lookback,
            'vol_lookback': self.vol_lookback,
            'shrinkage': self.shrinkage,
            'nan_policy': self.nan_policy,
            'outputs': ['vols(market, date)', 'covariance(market, date)', 'mask(market, date)']
        }

