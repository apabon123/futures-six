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
import os
import json

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
        config_path: str = "configs/strategies.yaml",
        filter_roll_jumps: bool = True,
        roll_jump_threshold_bp: float = 100.0
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
            filter_roll_jumps: Whether to filter roll jumps from returns used for P&L (default: True)
            roll_jump_threshold_bp: Threshold in basis points for detecting roll jumps (default: 100.0)
        """
        # Use explicit params (don't load from config for ExecSim tests)
        self.rebalance = rebalance
        self.slippage_bps = slippage_bps
        self.commission_per_contract = commission_per_contract
        self.cash_rate = cash_rate
        self.position_notional_scale = position_notional_scale
        self.filter_roll_jumps = filter_roll_jumps
        self.roll_jump_threshold_bp = roll_jump_threshold_bp
        
        # Validate parameters
        if self.slippage_bps < 0:
            raise ValueError(f"slippage_bps must be >= 0, got {self.slippage_bps}")
        
        if self.commission_per_contract < 0:
            raise ValueError(f"commission_per_contract must be >= 0, got {self.commission_per_contract}")
        
        logger.info(
            f"[ExecSim] Initialized: rebalance={self.rebalance}, "
            f"slippage_bps={self.slippage_bps}, commission={self.commission_per_contract}, "
            f"cash_rate={self.cash_rate}, scale={self.position_notional_scale}, "
            f"filter_roll_jumps={self.filter_roll_jumps}, roll_jump_threshold_bp={self.roll_jump_threshold_bp}"
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
            # Handle holidays: if Friday is not a trading day, use previous business day
            actual_schedule = []
            date_range_list = list(date_range)  # Convert to list for easier indexing
            for friday in schedule:
                if friday in date_range_list:
                    actual_schedule.append(friday)
                else:
                    # Find previous business day
                    prev_days = [d for d in date_range_list if d <= friday]
                    if len(prev_days) > 0:
                        actual_schedule.append(prev_days[-1])
            schedule = pd.DatetimeIndex(actual_schedule)
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
        components: Dict,
        run_id: Optional[str] = None,
        out_dir: str = "reports/runs"
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
            run_id: Optional run identifier for saving artifacts. If None, generates timestamp-based ID.
            out_dir: Base directory for saving run artifacts (default: "reports/runs")
                
        Returns:
            Dict with keys:
                - 'equity_curve': pd.Series of cumulative returns
                - 'weights_panel': pd.DataFrame [date x symbol]
                - 'signals_panel': pd.DataFrame [date x symbol]
                - 'report': dict of performance metrics
                - 'run_id': run_id used (or generated)
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
        
        # Get continuous returns for the full period (we'll slice by date as needed)
        # Use continuous returns for P&L calculation (no roll jumps in back-adjusted series)
        returns_cont = market.returns_cont
        
        if returns_cont.empty:
            logger.error("[ExecSim] No continuous returns data available")
            return {
                'equity_curve': pd.Series(dtype=float),
                'weights_panel': pd.DataFrame(),
                'signals_panel': pd.DataFrame(),
                'report': {}
            }
        
        # Filter by date range
        returns_df = returns_cont.copy()
        if start:
            start_dt = pd.to_datetime(start)
            returns_df = returns_df[returns_df.index >= start_dt]
        if end:
            end_dt = pd.to_datetime(end)
            returns_df = returns_df[returns_df.index <= end_dt]
        
        if returns_df.empty:
            logger.error("[ExecSim] No returns data available after filtering")
            return {
                'equity_curve': pd.Series(dtype=float),
                'weights_panel': pd.DataFrame(),
                'signals_panel': pd.DataFrame(),
                'report': {}
            }
        
        # Note: We no longer filter roll jumps because continuous returns are back-adjusted
        # and don't have roll jumps. The backward-panama adjustment removes price gaps
        # at roll points, so P&L is computed correctly from continuous returns.
        
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
                
                # Step 3: Get covariance matrix and validity mask for allocator
                # Pass signals to ensure risk model uses same universe
                cov = risk_vol.covariance(market, date, signals=scaled_signals)
                mask = risk_vol.mask(market, date, signals=scaled_signals)
                
                # Step 3b: Zero out signals for assets failing validity mask
                # This prevents allocator from wasting budget on NaNs that got imputed
                valid_symbols = mask.intersection(scaled_signals.index)
                if len(valid_symbols) < len(scaled_signals):
                    invalid_symbols = scaled_signals.index.difference(valid_symbols)
                    logger.debug(
                        f"[ExecSim] Zeroing {len(invalid_symbols)} invalid signals: {list(invalid_symbols)[:5]}"
                    )
                    scaled_signals = scaled_signals.copy()
                    scaled_signals.loc[invalid_symbols] = 0.0
                
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
                        
                        # Compute period returns (weights are fixed over [t, next_t))
                        period_returns = returns_df.iloc[date_idx + 1:next_idx + 1]
                        
                        # For log returns: sum is correct (additive)
                        # For simple returns: need to compound with prod
                        # Since we use log returns by default, sum is correct
                        # But handle both cases for robustness
                        if len(period_returns) > 0:
                            # Check if returns are log or simple by examining if they can be negative
                            # Log returns can be negative, simple returns typically > -1
                            # For safety, assume log returns (default) and use sum
                            holding_returns = period_returns.sum()
                        else:
                            holding_returns = pd.Series(0.0, index=returns_df.columns)
                        
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
                        
                        # Diagnostics: what-moved report
                        if prev_weights is not None:
                            weight_changes = (weights - prev_weights).abs().sort_values(ascending=False)
                            top_movers = weight_changes.head(5)
                            logger.info(
                                f"[ExecSim] {date}: port_ret={port_ret:.4f}, "
                                f"slippage={slippage_cost:.4f}, net_ret={port_ret_net:.4f}, "
                                f"turnover={turnover_history[-1]:.3f}, "
                                f"k={macro_k:.3f}, top_movers={dict(top_movers.head(3))}"
                            )
                        else:
                            logger.info(
                                f"[ExecSim] {date}: port_ret={port_ret:.4f}, "
                                f"slippage={slippage_cost:.4f}, net_ret={port_ret_net:.4f}, "
                                f"turnover={turnover_history[-1]:.3f}, k={macro_k:.3f}"
                            )
                
                # Update previous weights
                prev_weights = weights.copy()
                
            except Exception as e:
                logger.error(f"[ExecSim] Error on {date}: {e}")
                continue
        
        # Build results
        logger.info(f"[ExecSim] Completed {len(dates_history)} holding periods")
        
        # Equity curve: cumulative sum of log returns (rebalance-frequency, for backward compatibility)
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
        
        # Compute daily equity curve for accurate metrics
        # This ensures metrics use the same equity curve as the CSV output
        equity_curve_for_metrics = equity_curve  # Default to rebalance-frequency
        if not weights_panel.empty and not returns_df.empty:
            # Forward-fill weights to daily dates
            weights_daily = weights_panel.reindex(returns_df.index).ffill().fillna(0.0)
            
            # Align columns
            common_symbols = weights_daily.columns.intersection(returns_df.columns)
            if len(common_symbols) > 0:
                weights_aligned = weights_daily[common_symbols]
                returns_aligned = returns_df[common_symbols]
                
                # Compute daily portfolio returns (log returns)
                portfolio_returns_log = (weights_aligned * returns_aligned).sum(axis=1)
                portfolio_returns_daily = np.exp(portfolio_returns_log) - 1.0
                
                # Compute daily equity curve
                equity_daily = (1 + portfolio_returns_daily).cumprod()
                equity_daily.iloc[0] = 1.0
                
                # Filter to start from first rebalance date (actual trading start)
                first_rebalance_date = weights_panel.index[0]
                equity_daily_filtered = equity_daily[equity_daily.index >= first_rebalance_date].copy()
                if len(equity_daily_filtered) > 0:
                    # Recompute from first rebalance date to ensure clean start
                    portfolio_returns_from_start = portfolio_returns_daily[portfolio_returns_daily.index >= first_rebalance_date]
                    equity_daily_filtered = (1 + portfolio_returns_from_start).cumprod()
                    equity_daily_filtered.iloc[0] = 1.0
                    equity_curve_for_metrics = equity_daily_filtered
        
        # Compute metrics using daily equity curve (filtered from first rebalance date)
        report = self._compute_metrics(
            equity_curve_for_metrics,
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
        
        # Save artifacts (always save, generate run_id if not provided)
        if run_id is None:
            # Generate default run_id from timestamp
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            logger.info(f"[ExecSim] Generated run_id: {run_id}")
        
        self._save_run_artifacts(
            run_id=run_id,
            out_dir=out_dir,
            equity_curve=equity_curve_for_metrics,  # Use daily equity curve for CSV
            weights_panel=weights_panel,
            returns_df=returns_df,
            start=start,
            end=end,
            components=components,
            market=market
        )
        
        return {
            'equity_curve': equity_curve_for_metrics,  # Return daily equity curve for consistency
            'weights_panel': weights_panel,
            'signals_panel': signals_panel,
            'report': report,
            'macro_scaler': macro_scaler_series,
            'run_id': run_id
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
        
        # CRITICAL: Compute returns from equity_curve (daily), not returns_list (rebalance-frequency)
        # This ensures Sharpe, vol, and drawdown use the same series as CAGR
        returns_daily = equity_curve.pct_change().dropna()
        
        # CAGR - use actual time period, not number of periods
        total_ret = equity_curve.iloc[-1] / equity_curve.iloc[0] - 1.0
        # Calculate actual years from first to last date
        n_days = (equity_curve.index[-1] - equity_curve.index[0]).days
        n_years = n_days / 365.25
        if n_years > 0:
            cagr = (1 + total_ret) ** (1 / n_years) - 1
        else:
            cagr = 0.0
        
        # Volatility (annualized) - use daily returns
        vol = returns_daily.std() * np.sqrt(252)  # Daily returns -> annualized
        
        # Sharpe ratio (rf = 0 for now)
        mean_ret = returns_daily.mean()
        sharpe = (mean_ret * 252) / vol if vol > 0 else 0.0
        
        # Maximum drawdown - use equity curve directly
        running_max = equity_curve.expanding().max()
        drawdown = (equity_curve - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Hit rate - use daily returns
        hit_rate = (returns_daily > 0).mean()
        
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
    
    def _save_run_artifacts(
        self,
        run_id: str,
        out_dir: str,
        equity_curve: pd.Series,
        weights_panel: pd.DataFrame,
        returns_df: pd.DataFrame,
        start: Union[str, datetime],
        end: Union[str, datetime],
        components: Dict,
        market
    ):
        """
        Save run artifacts to disk for diagnostics.
        
        Args:
            run_id: Run identifier
            out_dir: Base directory for saving
            equity_curve: Equity curve Series
            weights_panel: Weights DataFrame
            returns_df: Daily asset returns DataFrame
            start: Start date
            end: End date
            components: Components dict (for metadata)
            market: MarketData instance (for universe)
        """
        run_dir = Path(out_dir) / run_id
        os.makedirs(run_dir, exist_ok=True)
        
        logger.info(f"[ExecSim] Saving artifacts to {run_dir}")
        
        # Compute daily portfolio returns and equity curve
        portfolio_returns_daily = None
        equity_daily = None
        
        if not weights_panel.empty and not returns_df.empty:
            # Forward-fill weights to daily dates
            weights_daily = weights_panel.reindex(returns_df.index).ffill().fillna(0.0)
            
            # Align columns (ensure weights and returns have same symbols)
            common_symbols = weights_daily.columns.intersection(returns_df.columns)
            if len(common_symbols) > 0:
                weights_aligned = weights_daily[common_symbols]
                returns_aligned = returns_df[common_symbols]
                
                # Compute daily portfolio returns: sum(weight * return) for each day
                # Note: returns_df contains log returns, so portfolio_returns is log returns
                # Convert to simple returns for diagnostics: r_simple = exp(r_log) - 1
                portfolio_returns_log = (weights_aligned * returns_aligned).sum(axis=1)
                portfolio_returns_daily = np.exp(portfolio_returns_log) - 1.0
                
                # Compute equity curve: cumulative product (simple returns)
                # Start at 1.0
                equity_daily = (1 + portfolio_returns_daily).cumprod()
                equity_daily.iloc[0] = 1.0  # Ensure starting value is 1.0
        
        # 1. Portfolio returns (daily)
        if portfolio_returns_daily is not None:
            portfolio_returns_daily.name = 'ret'
            portfolio_returns_daily.to_csv(
                run_dir / 'portfolio_returns.csv',
                header=True
            )
        else:
            # Create empty file if no data
            pd.DataFrame(columns=['date', 'ret']).to_csv(
                run_dir / 'portfolio_returns.csv',
                index=False
            )
        
        # 2. Equity curve (daily)
        # CRITICAL: Only include dates from first rebalance date onwards
        # This ensures consistency between CSV and metrics calculation
        # Before first rebalance, weights are 0 so equity stays at 1.0
        if equity_daily is not None:
            # Get first rebalance date (actual start of trading)
            if not weights_panel.empty:
                first_rebalance_date = weights_panel.index[0]
                # Filter equity_daily to start from first rebalance date
                equity_daily_filtered = equity_daily[equity_daily.index >= first_rebalance_date].copy()
                # Normalize to start at 1.0 at first rebalance date (should already be 1.0, but ensure it)
                if len(equity_daily_filtered) > 0:
                    equity_daily_filtered.iloc[0] = 1.0
                    # Recompute from first rebalance date to ensure consistency
                    portfolio_returns_from_start = portfolio_returns_daily[portfolio_returns_daily.index >= first_rebalance_date]
                    if len(portfolio_returns_from_start) > 0:
                        equity_daily_filtered = (1 + portfolio_returns_from_start).cumprod()
                        equity_daily_filtered.iloc[0] = 1.0
            else:
                equity_daily_filtered = equity_daily
            
            equity_daily_filtered.name = 'equity'
            equity_daily_filtered.to_csv(
                run_dir / 'equity_curve.csv',
                header=True
            )
        elif not equity_curve.empty:
            # Fallback: use rebalance-frequency equity curve
            equity_curve.name = 'equity'
            equity_curve.to_csv(
                run_dir / 'equity_curve.csv',
                header=True
            )
        else:
            pd.DataFrame(columns=['date', 'equity']).to_csv(
                run_dir / 'equity_curve.csv',
                index=False
            )
        
        # 3. Asset returns (daily, all symbols in universe)
        # Convert from log returns to simple returns for consistency
        if not returns_df.empty:
            # Convert log returns to simple returns: r_simple = exp(r_log) - 1
            asset_returns_simple = np.exp(returns_df) - 1.0
            asset_returns_simple.to_csv(run_dir / 'asset_returns.csv')
        else:
            pd.DataFrame().to_csv(run_dir / 'asset_returns.csv')
        
        # 4. Weights (rebalance dates only)
        if not weights_panel.empty:
            weights_panel.to_csv(run_dir / 'weights.csv')
        else:
            pd.DataFrame().to_csv(run_dir / 'weights.csv')
        
        # 5. Meta JSON
        strategy_name = "unknown"
        if 'strategy' in components:
            strategy = components['strategy']
            if hasattr(strategy, '__class__'):
                strategy_name = strategy.__class__.__name__
        
        universe = []
        if hasattr(market, 'universe'):
            universe = list(market.universe) if market.universe else []
        
        meta = {
            'run_id': run_id,
            'start_date': str(start),
            'end_date': str(end),
            'strategy_config_name': strategy_name,
            'universe': universe,
            'rebalance': self.rebalance,
            'slippage_bps': self.slippage_bps,
            'n_rebalances': len(weights_panel) if not weights_panel.empty else 0,
            'n_trading_days': len(returns_df) if not returns_df.empty else 0
        }
        
        with open(run_dir / 'meta.json', 'w') as f:
            json.dump(meta, f, indent=2)
        
        logger.info(f"[ExecSim] Saved artifacts to {run_dir}")
    
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

