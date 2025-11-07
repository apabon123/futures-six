"""
ExecSim: Backtest Orchestrator & Metrics Agent

Runs the complete backtest loop:
1. Build rebalance schedule where risk data is available
2. On each rebalance date:
   - Get strategy signals
   - Apply vol-managed overlay scaling
   - Allocate to final weights
   - Apply returns (next-day convention: close-to-close)
3. Apply slippage/commissions on rebalance days only
4. Compute equity curve and performance metrics

No data writes. No look-ahead. Deterministic outputs given MarketData snapshot.
"""

import logging
from typing import Dict, Optional, Union, Tuple
from datetime import datetime
from pathlib import Path

import yaml
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class ExecSim:
    """
    Backtest orchestrator and metrics calculator.
    
    Coordinates all agents (strategy, overlay, allocator) and simulates
    portfolio execution over a rebalance schedule, producing equity curves,
    weights panels, and performance metrics.
    """
    
    def __init__(
        self,
        rebalance: str = "W-FRI",
        slippage_bps: float = 0.5,
        commission_per_contract: float = 0.0,
        cash_rate: float = 0.0,
        position_notional_scale: float = 1.0,
        config_path: str = "configs/strategies.yaml"
    ):
        """
        Initialize ExecSim orchestrator.
        
        Args:
            rebalance: Rebalance frequency ("W-FRI", "M", etc.)
            slippage_bps: Slippage in basis points (applied to turnover on rebalance days)
            commission_per_contract: Commission per contract (placeholder, not used until contract sizing)
            cash_rate: Risk-free rate for cash (default: 0.0)
            position_notional_scale: Scale factor for positions (default: 1.0 = unit portfolio)
            config_path: Path to configuration YAML file
        """
        # Use explicit params (don't load from config for ExecSim tests)
        self.rebalance = rebalance
        self.slippage_bps = slippage_bps
        self.commission_per_contract = commission_per_contract
        self.cash_rate = cash_rate
        self.position_notional_scale = position_notional_scale
        
        # Validate parameters
        if self.slippage_bps < 0:
            raise ValueError(f"slippage_bps must be >= 0, got {self.slippage_bps}")
        
        if self.commission_per_contract < 0:
            raise ValueError(f"commission_per_contract must be >= 0, got {self.commission_per_contract}")
        
        logger.info(
            f"[ExecSim] Initialized: rebalance={self.rebalance}, "
            f"slippage_bps={self.slippage_bps}, commission={self.commission_per_contract}, "
            f"cash_rate={self.cash_rate}, scale={self.position_notional_scale}"
        )
    
    def _load_config(self, config_path: str) -> Optional[dict]:
        """Load configuration from YAML file."""
        path = Path(config_path)
        if not path.exists():
            logger.warning(f"[ExecSim] Config file not found: {config_path}, using defaults")
            return None
        
        try:
            with open(path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            logger.warning(f"[ExecSim] Failed to load config: {e}, using defaults")
            return None
    
    def _build_rebalance_dates(
        self,
        market,
        risk_vol,
        start: Union[str, datetime],
        end: Union[str, datetime]
    ) -> pd.DatetimeIndex:
        """
        Build union of rebalancing dates where risk data is available for all tradable symbols.
        
        Args:
            market: MarketData instance
            risk_vol: RiskVol instance
            start: Start date
            end: End date
            
        Returns:
            DatetimeIndex of rebalance dates
        """
        start = pd.to_datetime(start)
        end = pd.to_datetime(end)
        
        # Get all trading days in range
        all_dates = market.trading_days()
        
        if len(all_dates) == 0:
            logger.warning(f"[ExecSim] No trading days available")
            return pd.DatetimeIndex([])
        
        # Filter to date range
        date_range = all_dates[(all_dates >= start) & (all_dates <= end)]
        
        if len(date_range) == 0:
            logger.warning(f"[ExecSim] No trading days in range {start} to {end}")
            return pd.DatetimeIndex([])
        
        # Generate rebalance schedule
        if self.rebalance == "W-FRI":
            schedule = pd.date_range(start=start, end=end, freq='W-FRI')
        elif self.rebalance == "M":
            # Month-end - use 'M' for backward compatibility, but snap to nearest business day
            schedule = pd.date_range(start=start, end=end, freq='M')
            # For month-end, find nearest trading day to each month-end date
            rebalance_candidates = []
            for me_date in schedule:
                # Find closest trading day to month-end
                if me_date in date_range:
                    rebalance_candidates.append(me_date)
                else:
                    # Find nearest trading day before or at month-end
                    prev_days = date_range[date_range <= me_date]
                    if len(prev_days) > 0:
                        rebalance_candidates.append(prev_days[-1])
            schedule = pd.DatetimeIndex(rebalance_candidates)
        elif self.rebalance == "D":
            schedule = pd.date_range(start=start, end=end, freq='D')
        else:
            raise ValueError(f"Unknown rebalance frequency: {self.rebalance}")
        
        # Only keep dates that exist in trading calendar (for non-M frequencies)
        if self.rebalance != "M":
            rebalance_dates = schedule.intersection(date_range)
        else:
            rebalance_dates = schedule  # Already filtered above
        
        # Filter to dates where risk data is available
        valid_dates = []
        for date in rebalance_dates:
            try:
                # Check if we can get a mask (valid symbols with data)
                mask = risk_vol.mask(market, date)
                if len(mask) > 0:
                    valid_dates.append(date)
            except Exception as e:
                logger.debug(f"[ExecSim] Skipping {date}: {e}")
                continue
        
        rebalance_dates = pd.DatetimeIndex(valid_dates)
        
        logger.info(f"[ExecSim] Built {len(rebalance_dates)} rebalance dates from {start} to {end}")
        
        return rebalance_dates
    
    def _compute_portfolio_return(
        self,
        weights: pd.Series,
        returns: pd.Series,
        is_rebalance: bool
    ) -> float:
        """
        Compute portfolio return for one period.
        
        Uses close-to-close convention: weights held from t to t+1, applied to returns[t+1].
        On rebalance days, apply slippage to turnover.
        
        Args:
            weights: Portfolio weights at start of period
            returns: Asset returns for the period
            is_rebalance: Whether this is a rebalance day (apply slippage)
            
        Returns:
            Portfolio return (including transaction costs if rebalance)
        """
        # Align weights and returns
        common = weights.index.intersection(returns.index)
        
        if len(common) == 0:
            return 0.0
        
        w = weights.loc[common].fillna(0)
        r = returns.loc[common].fillna(0)
        
        # Gross portfolio return
        port_ret = (w * r).sum()
        
        # No slippage/commissions for now (placeholder)
        # In future: apply slippage_bps to turnover on rebalance days
        
        return port_ret
    
    def _compute_turnover(
        self,
        weights_prev: pd.Series,
        weights_new: pd.Series
    ) -> float:
        """
        Compute turnover between two weight vectors.
        
        Args:
            weights_prev: Previous weights
            weights_new: New weights
            
        Returns:
            Turnover (sum of absolute changes)
        """
        all_symbols = weights_prev.index.union(weights_new.index)
        w_prev = weights_prev.reindex(all_symbols, fill_value=0)
        w_new = weights_new.reindex(all_symbols, fill_value=0)
        
        turnover = (w_new - w_prev).abs().sum()
        return turnover
    
    def run(
        self,
        market,
        start: Union[str, datetime],
        end: Union[str, datetime],
        components: Dict
    ) -> Dict:
        """
        Run backtest over date range.
        
        Args:
            market: MarketData instance
            start: Start date
            end: End date
        components: Dict with required keys:
                - 'strategy': Strategy agent (e.g., TSMOM)
                - 'overlay': VolManagedOverlay agent
                - 'risk_vol': RiskVol agent
                - 'allocator': Allocator agent
            Optional keys:
                - 'macro_overlay': Regime-based overlay applied before vol overlay
                
        Returns:
            Dict with keys:
                - 'equity_curve': pd.Series of cumulative returns
                - 'weights_panel': pd.DataFrame [date x symbol]
                - 'signals_panel': pd.DataFrame [date x symbol]
                - 'report': dict of performance metrics
        """
        logger.info(f"[ExecSim] Starting backtest from {start} to {end}")
        
        # Extract components
        strategy = components['strategy']
        overlay = components['overlay']
        macro_overlay = components.get('macro_overlay')
        risk_vol = components['risk_vol']
        allocator = components['allocator']
        
        # Build rebalance schedule
        rebalance_dates = self._build_rebalance_dates(market, risk_vol, start, end)
        
        if len(rebalance_dates) == 0:
            logger.error("[ExecSim] No valid rebalance dates, cannot run backtest")
            return {
                'equity_curve': pd.Series(dtype=float),
                'weights_panel': pd.DataFrame(),
                'signals_panel': pd.DataFrame(),
                'report': {}
            }
        
        # Initialize tracking
        weights_history = []
        signals_history = []
        returns_history = []
        dates_history = []
        turnover_history = []
        macro_scaler_history = [] if macro_overlay is not None else None
        
        prev_weights = None
        
        # Get returns for the full period (we'll slice by date as needed)
        returns_df = market.get_returns(method="log", start=start, end=end)
        
        if returns_df.empty:
            logger.error("[ExecSim] No returns data available")
            return {
                'equity_curve': pd.Series(dtype=float),
                'weights_panel': pd.DataFrame(),
                'signals_panel': pd.DataFrame(),
                'report': {}
            }
        
        # Loop over rebalance dates
        for i, date in enumerate(rebalance_dates):
            logger.debug(f"[ExecSim] Rebalance {i+1}/{len(rebalance_dates)}: {date}")
            
            try:
                # Step 1: Get strategy signals
                signals = strategy.signals(market, date)
                
                macro_k = 1.0
                if macro_overlay is not None:
                    macro_k = macro_overlay.scaler(market, date)
                    macro_scaler_history.append(macro_k)
                
                # Step 2: Apply vol-managed overlay on raw signals
                scaled_signals = overlay.scale(signals, market, date)
                
                # Step 2b: Apply macro scaler after vol targeting to preserve risk reduction
                if macro_overlay is not None:
                    scaled_signals = scaled_signals * macro_k
                
                # Step 3: Get covariance matrix for allocator
                cov = risk_vol.covariance(market, date)
                
                # Step 4: Allocate to final weights
                weights = allocator.solve(scaled_signals, cov, weights_prev=prev_weights)
                
                # Record signals and weights
                signals_history.append(scaled_signals)
                weights_history.append(weights)
                
                # Compute turnover (if not first date)
                if prev_weights is not None:
                    turnover = self._compute_turnover(prev_weights, weights)
                    turnover_history.append(turnover)
                else:
                    turnover_history.append(weights.abs().sum())  # Initial turnover = gross leverage
                
                # Step 4: Compute returns for holding period (date to next rebalance or end)
                # Find next date for return calculation
                if i < len(rebalance_dates) - 1:
                    next_date = rebalance_dates[i + 1]
                else:
                    # Last rebalance: hold until end
                    next_date = returns_df.index[-1] if date < returns_df.index[-1] else date
                
                # Get returns from current date to next date
                # Convention: close-to-close, so returns[date+1] is the first return we earn
                if date in returns_df.index:
                    date_idx = returns_df.index.get_loc(date)
                    
                    # If not the last date, compute returns for holding period
                    if date_idx < len(returns_df) - 1:
                        # Returns from date+1 to next_date (inclusive)
                        if next_date in returns_df.index:
                            next_idx = returns_df.index.get_loc(next_date)
                        else:
                            # Find last date <= next_date
                            valid_dates = returns_df.index[returns_df.index <= next_date]
                            if len(valid_dates) > 0:
                                next_idx = returns_df.index.get_loc(valid_dates[-1])
                            else:
                                next_idx = date_idx
                        
                        # Sum returns over holding period (log returns are additive)
                        holding_returns = returns_df.iloc[date_idx + 1:next_idx + 1].sum()
                        
                        # Compute portfolio return
                        port_ret = self._compute_portfolio_return(
                            weights,
                            holding_returns,
                            is_rebalance=True
                        )
                        
                        # Apply slippage to turnover on rebalance day
                        slippage_cost = (turnover_history[-1] * self.slippage_bps / 10000)
                        port_ret_net = port_ret - slippage_cost
                        
                        returns_history.append(port_ret_net)
                        dates_history.append(date)
                        
                        logger.debug(
                            f"[ExecSim] {date}: port_ret={port_ret:.4f}, "
                            f"slippage={slippage_cost:.4f}, net_ret={port_ret_net:.4f}, "
                            f"turnover={turnover_history[-1]:.3f}"
                        )
                
                # Update previous weights
                prev_weights = weights.copy()
                
            except Exception as e:
                logger.error(f"[ExecSim] Error on {date}: {e}")
                continue
        
        # Build results
        logger.info(f"[ExecSim] Completed {len(dates_history)} holding periods")
        
        # Equity curve: cumulative sum of log returns
        if len(returns_history) > 0:
            equity_curve = pd.Series(returns_history, index=dates_history).cumsum()
            equity_curve = np.exp(equity_curve)  # Convert log to arithmetic
        else:
            equity_curve = pd.Series(dtype=float)
        
        # Weights panel
        if len(weights_history) > 0:
            weights_panel = pd.DataFrame(weights_history, index=rebalance_dates[:len(weights_history)])
        else:
            weights_panel = pd.DataFrame()
        
        # Signals panel
        if len(signals_history) > 0:
            signals_panel = pd.DataFrame(signals_history, index=rebalance_dates[:len(signals_history)])
        else:
            signals_panel = pd.DataFrame()
        
        # Macro scaler series
        if macro_scaler_history is not None and len(macro_scaler_history) > 0:
            macro_scaler_series = pd.Series(macro_scaler_history, index=rebalance_dates[:len(macro_scaler_history)])
        else:
            macro_scaler_series = pd.Series(dtype=float)
        
        # Compute metrics
        report = self._compute_metrics(
            equity_curve,
            returns_history,
            weights_panel,
            turnover_history
        )
        
        logger.info(
            f"[ExecSim] Backtest complete: "
            f"CAGR={report.get('cagr', 0):.2%}, "
            f"Sharpe={report.get('sharpe', 0):.2f}, "
            f"MaxDD={report.get('max_drawdown', 0):.2%}"
        )
        
        return {
            'equity_curve': equity_curve,
            'weights_panel': weights_panel,
            'signals_panel': signals_panel,
            'report': report,
            'macro_scaler': macro_scaler_series
        }
    
    def _compute_metrics(
        self,
        equity_curve: pd.Series,
        returns_list: list,
        weights_panel: pd.DataFrame,
        turnover_history: list
    ) -> Dict:
        """
        Compute performance metrics.
        
        Returns dict with:
        - cagr: Compound annual growth rate
        - vol: Annualized volatility
        - sharpe: Sharpe ratio
        - max_drawdown: Maximum drawdown
        - hit_rate: Percentage of positive returns
        - avg_turnover: Average turnover per rebalance
        - avg_gross: Average gross leverage
        - avg_net: Average net exposure
        """
        if equity_curve.empty or len(returns_list) == 0:
            return {
                'cagr': 0.0,
                'vol': 0.0,
                'sharpe': 0.0,
                'max_drawdown': 0.0,
                'hit_rate': 0.0,
                'avg_turnover': 0.0,
                'avg_gross': 0.0,
                'avg_net': 0.0,
                'n_periods': 0
            }
        
        # Convert returns list to series for calculations
        returns = pd.Series(returns_list)
        
        # CAGR
        total_ret = equity_curve.iloc[-1] - 1.0  # Minus initial value of 1.0
        n_years = len(equity_curve) / 52  # Approximate (weekly rebalancing)
        if n_years > 0:
            cagr = (1 + total_ret) ** (1 / n_years) - 1
        else:
            cagr = 0.0
        
        # Volatility (annualized)
        vol = returns.std() * np.sqrt(52)  # Approximate weekly
        
        # Sharpe ratio
        sharpe = (returns.mean() * 52) / vol if vol > 0 else 0.0
        
        # Maximum drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Hit rate
        hit_rate = (returns > 0).mean()
        
        # Average turnover
        avg_turnover = np.mean(turnover_history) if len(turnover_history) > 0 else 0.0
        
        # Average gross and net leverage
        if not weights_panel.empty:
            avg_gross = weights_panel.abs().sum(axis=1).mean()
            avg_net = weights_panel.sum(axis=1).abs().mean()
        else:
            avg_gross = 0.0
            avg_net = 0.0
        
        return {
            'cagr': cagr,
            'vol': vol,
            'sharpe': sharpe,
            'max_drawdown': max_drawdown,
            'hit_rate': hit_rate,
            'avg_turnover': avg_turnover,
            'avg_gross': avg_gross,
            'avg_net': avg_net,
            'n_periods': len(returns_list)
        }
    
    def to_parquet(self, results: Dict, outdir: str):
        """
        Save results to parquet files (optional).
        
        Args:
            results: Output from run()
            outdir: Output directory path
        """
        outpath = Path(outdir)
        outpath.mkdir(parents=True, exist_ok=True)
        
        # Save equity curve
        if not results['equity_curve'].empty:
            results['equity_curve'].to_frame('equity').to_parquet(
                outpath / 'equity_curve.parquet'
            )
            logger.info(f"[ExecSim] Saved equity curve to {outpath / 'equity_curve.parquet'}")
        
        # Save weights panel
        if not results['weights_panel'].empty:
            results['weights_panel'].to_parquet(outpath / 'weights_panel.parquet')
            logger.info(f"[ExecSim] Saved weights panel to {outpath / 'weights_panel.parquet'}")
        
        # Save signals panel
        if not results['signals_panel'].empty:
            results['signals_panel'].to_parquet(outpath / 'signals_panel.parquet')
            logger.info(f"[ExecSim] Saved signals panel to {outpath / 'signals_panel.parquet'}")
        
        # Save metrics report as JSON
        import json
        with open(outpath / 'report.json', 'w') as f:
            json.dump(results['report'], f, indent=2)
        logger.info(f"[ExecSim] Saved report to {outpath / 'report.json'}")
    
    def describe(self) -> dict:
        """
        Return configuration and description of the ExecSim agent.
        
        Returns:
            dict with configuration parameters
        """
        return {
            'agent': 'ExecSim',
            'role': 'Backtest orchestrator and metrics calculator',
            'rebalance': self.rebalance,
            'slippage_bps': self.slippage_bps,
            'commission_per_contract': self.commission_per_contract,
            'cash_rate': self.cash_rate,
            'position_notional_scale': self.position_notional_scale,
            'outputs': ['run(market, start, end, components)', 'to_parquet(results, outdir)']
        }

