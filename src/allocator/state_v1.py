"""
AllocatorStateV1: Canonical Allocator State Feature Service

Computes 10 state features for regime classification and risk management:
- Portfolio volatility (20d, 60d) and acceleration
- Drawdown level and slope
- Cross-asset correlation (20d, 60d) and shock
- Trend breadth (20d)
- Sleeve concentration (60d)

No thresholds, no regime mapping - pure feature computation.
Designed for artifact output so later stages can consume without refactors.
"""

import logging
from typing import Optional, List

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# Canonical 10-feature contract
REQUIRED_FEATURES: List[str] = [
    'port_rvol_20d',      # 20-day realized volatility
    'port_rvol_60d',      # 60-day realized volatility
    'vol_accel',          # Volatility acceleration (20d/60d)
    'dd_level',           # Drawdown level
    'dd_slope_10d',       # 10-day drawdown slope
    'corr_20d',           # 20-day average pairwise correlation
    'corr_60d',           # 60-day average pairwise correlation
    'corr_shock'          # Correlation shock (20d - 60d)
]

OPTIONAL_FEATURES: List[str] = [
    'trend_breadth_20d',       # Trend breadth (requires trend_unit_returns)
    'sleeve_concentration_60d' # Sleeve concentration (requires sleeve_returns)
]

ALL_FEATURES: List[str] = REQUIRED_FEATURES + OPTIONAL_FEATURES

# Canonical lookback windows
LOOKBACKS = {
    'rvol_fast': 20,
    'rvol_slow': 60,
    'dd_slope': 10,
    'corr_fast': 20,
    'corr_slow': 60,
    'trend_breadth': 20,
    'sleeve_concentration': 60
}


class AllocatorStateV1:
    """
    Canonical allocator state feature service.
    
    Computes 10 state features from portfolio and asset data:
    1. port_rvol_20d: 20-day realized volatility (annualized)
    2. port_rvol_60d: 60-day realized volatility (annualized)
    3. vol_accel: Volatility acceleration (20d / 60d)
    4. dd_level: Drawdown level (current equity / running max - 1)
    5. dd_slope_10d: 10-day drawdown slope
    6. corr_20d: Average pairwise correlation (20d)
    7. corr_60d: Average pairwise correlation (60d)
    8. corr_shock: Correlation shock (20d - 60d)
    9. trend_breadth_20d: Fraction of assets with positive trend (20d)
    10. sleeve_concentration_60d: Herfindahl index of sleeve contributions (60d)
    
    Output is a DataFrame indexed by date with these 10 columns.
    Rows with any NaN are dropped (canonical rule).
    """
    
    VERSION = "v1.0"
    
    def __init__(self):
        """Initialize AllocatorStateV1."""
        logger.info(f"[AllocatorStateV1] Initialized (version {self.VERSION})")
    
    def compute(
        self,
        portfolio_returns: pd.Series,
        equity_curve: pd.Series,
        asset_returns: pd.DataFrame,
        trend_unit_returns: Optional[pd.DataFrame] = None,
        sleeve_returns: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Compute allocator state features.
        
        Args:
            portfolio_returns: Daily portfolio returns indexed by date
            equity_curve: Daily equity curve indexed by date
            asset_returns: Daily asset returns (columns=assets, index=date)
            trend_unit_returns: Optional daily trend unit returns (columns=assets, index=date)
            sleeve_returns: Optional daily sleeve returns (columns=sleeves, index=date)
        
        Returns:
            DataFrame with 10 feature columns indexed by date.
            Rows with any NaN are dropped.
        """
        logger.info("[AllocatorStateV1] Computing state features...")
        
        # Align all inputs on common dates
        common_dates = portfolio_returns.index.intersection(equity_curve.index)
        common_dates = common_dates.intersection(asset_returns.index)
        
        if len(common_dates) == 0:
            logger.warning("[AllocatorStateV1] No common dates found, returning empty DataFrame")
            return pd.DataFrame()
        
        portfolio_returns = portfolio_returns.loc[common_dates]
        equity_curve = equity_curve.loc[common_dates]
        asset_returns = asset_returns.loc[common_dates]
        
        # Initialize state DataFrame
        state = pd.DataFrame(index=common_dates)
        
        # 1-3: Volatility features
        logger.debug("[AllocatorStateV1] Computing volatility features...")
        state['port_rvol_20d'] = self._compute_realized_vol(portfolio_returns, window=20)
        state['port_rvol_60d'] = self._compute_realized_vol(portfolio_returns, window=60)
        state['vol_accel'] = state['port_rvol_20d'] / state['port_rvol_60d']
        
        # 4-5: Drawdown features
        logger.debug("[AllocatorStateV1] Computing drawdown features...")
        state['dd_level'] = self._compute_drawdown_level(equity_curve)
        state['dd_slope_10d'] = self._compute_drawdown_slope(state['dd_level'], window=10)
        
        # 6-8: Correlation features
        logger.debug("[AllocatorStateV1] Computing correlation features...")
        state['corr_20d'] = self._compute_rolling_correlation(asset_returns, window=20)
        state['corr_60d'] = self._compute_rolling_correlation(asset_returns, window=60)
        state['corr_shock'] = state['corr_20d'] - state['corr_60d']
        
        # 9: Trend breadth (optional) - explicit handling
        logger.debug("[AllocatorStateV1] Computing trend breadth...")
        has_trend = False
        if trend_unit_returns is not None and not trend_unit_returns.empty:
            # Align trend_unit_returns with common_dates
            trend_aligned = trend_unit_returns.reindex(common_dates)
            state['trend_breadth_20d'] = self._compute_trend_breadth(trend_aligned, window=LOOKBACKS['trend_breadth'])
            has_trend = True
            logger.info("[AllocatorStateV1] trend_breadth_20d computed (optional feature present)")
        else:
            logger.info("[AllocatorStateV1] trend_unit_returns not provided, trend_breadth_20d excluded")
        
        # 10: Sleeve concentration (optional) - explicit handling
        logger.debug("[AllocatorStateV1] Computing sleeve concentration...")
        has_sleeve = False
        if sleeve_returns is not None and not sleeve_returns.empty:
            # Align sleeve_returns with common_dates
            sleeve_aligned = sleeve_returns.reindex(common_dates)
            state['sleeve_concentration_60d'] = self._compute_sleeve_concentration(sleeve_aligned, window=LOOKBACKS['sleeve_concentration'])
            has_sleeve = True
            logger.info("[AllocatorStateV1] sleeve_concentration_60d computed (optional feature present)")
        else:
            logger.info("[AllocatorStateV1] sleeve_returns not provided, sleeve_concentration_60d excluded")
        
        # Drop rows with any NaN in REQUIRED_FEATURES only (canonical rule)
        # Optional features are excluded from output if not computed
        rows_before = len(state)
        
        # Determine which columns to keep
        cols_present = REQUIRED_FEATURES.copy()
        if has_trend:
            cols_present.append('trend_breadth_20d')
        if has_sleeve:
            cols_present.append('sleeve_concentration_60d')
        
        # Select only present columns and drop NAs on REQUIRED_FEATURES
        state = state[cols_present].copy()
        state = state.dropna(subset=REQUIRED_FEATURES, how='any')
        rows_after = len(state)
        
        # Track feature coverage
        features_present = [f for f in ALL_FEATURES if f in state.columns]
        features_missing = [f for f in OPTIONAL_FEATURES if f not in state.columns]
        
        logger.info(
            f"[AllocatorStateV1] Computed {len(features_present)} features "
            f"({len(REQUIRED_FEATURES)} required, {len(features_present) - len(REQUIRED_FEATURES)} optional). "
            f"Rows: {rows_before} -> {rows_after} (dropped {rows_before - rows_after} with NaN)"
        )
        
        if features_missing:
            logger.info(f"[AllocatorStateV1] Optional features missing: {', '.join(features_missing)}")
        
        if len(state) > 0:
            logger.info(f"[AllocatorStateV1] Effective date range: {state.index[0]} to {state.index[-1]}")
        
        # Store metadata as attribute for retrieval by callers
        state.attrs['features_present'] = features_present
        state.attrs['features_missing'] = features_missing
        state.attrs['required_features'] = REQUIRED_FEATURES
        state.attrs['optional_features'] = OPTIONAL_FEATURES
        state.attrs['rows_before_dropna'] = rows_before
        state.attrs['rows_after_dropna'] = rows_after
        state.attrs['rows_dropped'] = rows_before - rows_after
        
        return state
    
    def _compute_realized_vol(self, returns: pd.Series, window: int) -> pd.Series:
        """
        Compute rolling realized volatility (annualized).
        
        Args:
            returns: Daily returns
            window: Rolling window size (days)
        
        Returns:
            Annualized realized volatility
        """
        return returns.rolling(window).std() * np.sqrt(252)
    
    def _compute_drawdown_level(self, equity_curve: pd.Series) -> pd.Series:
        """
        Compute drawdown level.
        
        Args:
            equity_curve: Equity curve
        
        Returns:
            Drawdown level (current / running_max - 1)
        """
        running_max = equity_curve.cummax()
        return equity_curve / running_max - 1.0
    
    def _compute_drawdown_slope(self, dd_level: pd.Series, window: int) -> pd.Series:
        """
        Compute drawdown slope.
        
        Args:
            dd_level: Drawdown level series
            window: Lookback window (days)
        
        Returns:
            Drawdown slope (current - lagged)
        """
        return dd_level - dd_level.shift(window)
    
    def _compute_rolling_correlation(self, asset_returns: pd.DataFrame, window: int) -> pd.Series:
        """
        Compute rolling average pairwise correlation.
        
        Args:
            asset_returns: Asset returns DataFrame (columns=assets)
            window: Rolling window size (days)
        
        Returns:
            Series of average pairwise correlations
        """
        # Use rolling window to compute correlation at each date
        corr_series = pd.Series(index=asset_returns.index, dtype=float)
        
        for i in range(window - 1, len(asset_returns)):
            window_data = asset_returns.iloc[i - window + 1:i + 1]
            
            # Skip if insufficient data
            if len(window_data) < window:
                corr_series.iloc[i] = np.nan
                continue
            
            # Compute average pairwise correlation
            avg_corr = self._avg_pairwise_corr(window_data)
            corr_series.iloc[i] = avg_corr
        
        return corr_series
    
    def _avg_pairwise_corr(self, returns_window: pd.DataFrame) -> float:
        """
        Compute average pairwise correlation for a window of returns.
        
        Args:
            returns_window: DataFrame of returns (rows=dates, columns=assets)
        
        Returns:
            Average pairwise correlation (off-diagonal mean)
        """
        # Drop columns with all NaN or insufficient variance
        valid_cols = []
        for col in returns_window.columns:
            col_data = returns_window[col].dropna()
            if len(col_data) >= 2 and col_data.std() > 1e-10:
                valid_cols.append(col)
        
        if len(valid_cols) < 2:
            # Need at least 2 assets for correlation
            return np.nan
        
        returns_valid = returns_window[valid_cols]
        
        # Compute correlation matrix
        corr_matrix = returns_valid.corr()
        
        # Extract off-diagonal elements
        n = len(corr_matrix)
        if n < 2:
            return np.nan
        
        # Get upper triangle (excluding diagonal)
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        off_diag = corr_matrix.where(mask).stack()
        
        # Return mean of off-diagonal correlations
        if len(off_diag) == 0:
            return np.nan
        
        return off_diag.mean()
    
    def _compute_trend_breadth(self, trend_unit_returns: pd.DataFrame, window: int) -> pd.Series:
        """
        Compute trend breadth: fraction of assets with positive trend over window.
        
        For each date: sum each asset's trend unit returns over window → sign → fraction positive.
        
        Args:
            trend_unit_returns: Trend unit returns (columns=assets, index=date)
            window: Rolling window size (days)
        
        Returns:
            Series of trend breadth (fraction of assets with positive trend)
        """
        # Sum trend unit returns over rolling window
        trend_sum = trend_unit_returns.rolling(window).sum()
        
        # Fraction of assets with positive sum
        positive_fraction = (trend_sum > 0).mean(axis=1)
        
        return positive_fraction
    
    def _compute_sleeve_concentration(self, sleeve_returns: pd.DataFrame, window: int) -> pd.Series:
        """
        Compute sleeve concentration using Herfindahl index.
        
        Concentration = sum of squared shares of absolute contributions over window.
        
        Args:
            sleeve_returns: Sleeve returns (columns=sleeves, index=date)
            window: Rolling window size (days)
        
        Returns:
            Series of Herfindahl concentration index
        """
        # Compute rolling sum of sleeve returns (PnL contributions)
        sleeve_pnl = sleeve_returns.rolling(window).sum()
        
        # Compute absolute contributions
        abs_contrib = sleeve_pnl.abs()
        
        # Compute shares (normalize by total absolute contribution)
        total_abs = abs_contrib.sum(axis=1)
        
        # Avoid division by zero
        shares = abs_contrib.div(total_abs, axis=0)
        
        # Herfindahl index: sum of squared shares
        hhi = (shares ** 2).sum(axis=1)
        
        return hhi
    
    def describe(self) -> dict:
        """
        Return description of AllocatorStateV1.
        
        Returns:
            Dict with version and feature list
        """
        return {
            'agent': 'AllocatorStateV1',
            'version': self.VERSION,
            'role': 'Compute canonical allocator state features',
            'required_features': REQUIRED_FEATURES,
            'optional_features': OPTIONAL_FEATURES,
            'all_features': ALL_FEATURES,
            'lookbacks': LOOKBACKS
        }

