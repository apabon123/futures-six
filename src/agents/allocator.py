"""
Allocator: Portfolio Weight Allocation Agent

Takes vol-managed signals and produces final weights that satisfy:
- Gross leverage cap
- Net leverage cap
- Per-asset bounds
- Turnover cap/penalty

Implements two methods:
1. "signal-beta": Use signals directly with constraints
2. "erc": Equal Risk Contribution (variance-parity based)

No look-ahead. Deterministic outputs.
"""

import logging
from typing import Optional
from pathlib import Path

import yaml
import pandas as pd
import numpy as np
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


class Allocator:
    """
    Portfolio weight allocator agent.
    
    Converts vol-managed signals into final portfolio weights while enforcing:
    - Gross leverage constraints
    - Net leverage constraints
    - Per-asset position bounds
    - Turnover constraints
    
    Implements two allocation methods:
    - "signal-beta": Map signals to weights with constraints
    - "erc": Equal Risk Contribution (risk-parity approach)
    """
    
    def __init__(
        self,
        method: str = "signal-beta",
        gross_cap: float = 7.0,
        net_cap: float = 2.0,
        w_bounds_per_asset = None,
        turnover_cap: float = 0.5,
        lambda_turnover: float = 0.001,
        config_path: str = "configs/strategies.yaml"
    ):
        """
        Initialize Allocator agent.
        
        Args:
            method: Allocation method ("signal-beta" or "erc")
            gross_cap: Maximum sum of absolute weights
            net_cap: Maximum absolute value of net exposure
            w_bounds_per_asset: [min, max] position size per asset (default: [-1.5, 1.5])
            turnover_cap: Maximum turnover (default: 0.5)
            lambda_turnover: Penalty coefficient for turnover (default: 0.001)
            config_path: Path to configuration YAML file
        """
        # Load config defaults - but only use them as fallbacks
        config = self._load_config(config_path)
        config_alloc = config.get('allocator', {}) if config else {}
        
        # Use explicit params if provided, otherwise fall back to config, then defaults
        self.method = method
        self.gross_cap = gross_cap
        self.net_cap = net_cap
        if w_bounds_per_asset is None:
            w_bounds_config = config_alloc.get('w_bounds_per_asset', [-1.5, 1.5])
            self.w_bounds_per_asset = w_bounds_config if isinstance(w_bounds_config, list) else list(w_bounds_config)
        else:
            self.w_bounds_per_asset = w_bounds_per_asset if isinstance(w_bounds_per_asset, list) else list(w_bounds_per_asset)
        self.turnover_cap = turnover_cap
        self.lambda_turnover = lambda_turnover
        
        # Ensure w_bounds_per_asset is a list
        if isinstance(self.w_bounds_per_asset, tuple):
            self.w_bounds_per_asset = list(self.w_bounds_per_asset)
        
        # Validate parameters
        if self.method not in ("signal-beta", "erc", "meanvar"):
            raise ValueError(f"method must be 'signal-beta', 'erc', or 'meanvar', got {self.method}")
        
        if self.gross_cap <= 0:
            raise ValueError(f"gross_cap must be positive, got {self.gross_cap}")
        
        if self.net_cap <= 0:
            raise ValueError(f"net_cap must be positive, got {self.net_cap}")
        
        if self.net_cap > self.gross_cap:
            raise ValueError(f"net_cap ({self.net_cap}) cannot exceed gross_cap ({self.gross_cap})")
        
        if not isinstance(self.w_bounds_per_asset, list) or len(self.w_bounds_per_asset) != 2:
            raise ValueError(f"w_bounds_per_asset must have 2 elements, got {self.w_bounds_per_asset}")
        
        if self.w_bounds_per_asset[0] > self.w_bounds_per_asset[1]:
            raise ValueError(f"w_bounds_per_asset min > max: {self.w_bounds_per_asset}")
        
        if self.turnover_cap is not None and not (0 <= self.turnover_cap <= 1):
            raise ValueError(f"turnover_cap must be in [0, 1], got {self.turnover_cap}")
        
        if self.lambda_turnover < 0:
            raise ValueError(f"lambda_turnover must be non-negative, got {self.lambda_turnover}")
        
        logger.info(
            f"[Allocator] Initialized: method={self.method}, "
            f"gross_cap={self.gross_cap}, net_cap={self.net_cap}, "
            f"w_bounds={self.w_bounds_per_asset}, turnover_cap={self.turnover_cap}, "
            f"lambda_turnover={self.lambda_turnover}"
        )
    
    def _load_config(self, config_path: str) -> Optional[dict]:
        """Load configuration from YAML file."""
        path = Path(config_path)
        if not path.exists():
            logger.warning(f"[Allocator] Config file not found: {config_path}, using defaults")
            return None
        
        try:
            with open(path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            logger.warning(f"[Allocator] Failed to load config: {e}, using defaults")
            return None
    
    def _apply_position_bounds(self, weights: pd.Series) -> pd.Series:
        """
        Clip weights to per-asset position bounds.
        
        Args:
            weights: Raw weights
            
        Returns:
            Bounded weights
        """
        return weights.clip(
            lower=self.w_bounds_per_asset[0],
            upper=self.w_bounds_per_asset[1]
        )
    
    def _apply_gross_cap(self, weights: pd.Series) -> pd.Series:
        """
        Scale down weights if gross leverage exceeds cap.
        
        Args:
            weights: Portfolio weights
            
        Returns:
            Scaled weights
        """
        gross_leverage = weights.abs().sum()
        
        if gross_leverage > self.gross_cap:
            # Scale down proportionally
            scale = self.gross_cap / gross_leverage
            weights = weights * scale
            
            logger.debug(
                f"[Allocator] Gross cap applied: "
                f"gross={gross_leverage:.2f} > cap={self.gross_cap}, "
                f"scale={scale:.3f}"
            )
        
        return weights
    
    def _apply_net_cap(self, weights: pd.Series) -> pd.Series:
        """
        Scale down weights if net exposure exceeds cap.
        
        Args:
            weights: Portfolio weights
            
        Returns:
            Scaled weights
        """
        net_exposure = abs(weights.sum())
        
        if net_exposure > self.net_cap:
            # Scale down proportionally
            scale = self.net_cap / net_exposure
            weights = weights * scale
            
            logger.debug(
                f"[Allocator] Net cap applied: "
                f"net={net_exposure:.2f} > cap={self.net_cap}, "
                f"scale={scale:.3f}"
            )
        
        return weights
    
    def _apply_turnover_constraint(
        self,
        target_weights: pd.Series,
        prev_weights: Optional[pd.Series]
    ) -> pd.Series:
        """
        Apply turnover constraint.
        
        Args:
            target_weights: Desired weights
            prev_weights: Previous weights (if any)
            
        Returns:
            Adjusted weights respecting turnover constraint
        """
        if prev_weights is None or prev_weights.empty or self.turnover_cap is None:
            # No turnover constraint if no previous weights or no cap
            return target_weights
        
        # Align indices
        all_symbols = target_weights.index.union(prev_weights.index)
        target_aligned = target_weights.reindex(all_symbols, fill_value=0)
        prev_aligned = prev_weights.reindex(all_symbols, fill_value=0)
        
        # Calculate desired turnover
        desired_turnover = (target_aligned - prev_aligned).abs().sum()
        
        if desired_turnover <= self.turnover_cap:
            # Within cap, no adjustment needed
            return target_weights
        
        # Scale down the trade to respect turnover cap
        # w_new = w_prev + scale * (w_target - w_prev)
        # where scale ensures turnover = turnover_cap
        scale = self.turnover_cap / desired_turnover
        
        adjusted = prev_aligned + scale * (target_aligned - prev_aligned)
        
        # Return only the symbols in target
        result = adjusted.loc[target_weights.index]
        
        logger.debug(
            f"[Allocator] Turnover cap applied: "
            f"desired={desired_turnover:.3f} > cap={self.turnover_cap}, "
            f"scale={scale:.3f}"
        )
        
        return result
    
    def _solve_signal_beta(
        self,
        signals: pd.Series,
        cov: pd.DataFrame,
        prev_weights: Optional[pd.Series] = None
    ) -> pd.Series:
        """
        Signal-beta allocation: Use signals as baseline weights with constraints.
        
        Args:
            signals: Strategy signals
            cov: Covariance matrix (not used in signal-beta, kept for API consistency)
            prev_weights: Previous weights (for turnover control)
            
        Returns:
            Portfolio weights
        """
        # Remove NaN signals
        weights = signals.fillna(0)
        
        # Apply constraints - order matters!
        # 1. Gross and net caps first (to scale overall portfolio)
        weights = self._apply_gross_cap(weights)
        weights = self._apply_net_cap(weights)
        
        # 2. Then position bounds (to clip individual positions)
        weights = self._apply_position_bounds(weights)
        
        # 3. Finally turnover (to moderate trading)
        weights = self._apply_turnover_constraint(weights, prev_weights)
        
        return weights
    
    def _solve_erc(
        self,
        signals: pd.Series,
        cov: pd.DataFrame,
        prev_weights: Optional[pd.Series] = None
    ) -> pd.Series:
        """
        Equal Risk Contribution allocation.
        
        Uses optimization to find weights where each asset contributes equally to portfolio risk,
        while respecting signal direction and constraints.
        
        Args:
            signals: Strategy signals (determines direction)
            cov: Covariance matrix
            prev_weights: Previous weights (for turnover control)
            
        Returns:
            Portfolio weights
        """
        # Align signals and cov
        common = signals.index.intersection(cov.index)
        
        if len(common) == 0:
            logger.warning("[Allocator] No common symbols between signals and covariance")
            return signals * 0
        
        signals_aligned = signals.loc[common]
        cov_aligned = cov.loc[common, common]
        
        # Remove zero signals
        nonzero_mask = signals_aligned.abs() > 1e-10
        if nonzero_mask.sum() == 0:
            # All signals are zero
            return signals_aligned * 0
        
        active_assets = signals_aligned[nonzero_mask].index
        signals_active = signals_aligned.loc[active_assets]
        cov_active = cov_aligned.loc[active_assets, active_assets]
        
        n = len(active_assets)
        
        # Objective: minimize sum of squared differences in risk contributions
        def objective(w):
            w = np.array(w)
            portfolio_var = w @ cov_active.values @ w
            
            if portfolio_var < 1e-12:
                return 1e6  # Penalize degenerate solutions
            
            # Risk contributions: w_i * (Σw)_i
            marginal_contrib = cov_active.values @ w
            risk_contrib = w * marginal_contrib
            
            # Target: equal risk contributions
            target_rc = portfolio_var / n
            
            # Sum of squared deviations from target
            deviation = np.sum((risk_contrib - target_rc) ** 2)
            
            return deviation
        
        # Constraints and bounds
        bounds = [(self.w_bounds_per_asset[0], self.w_bounds_per_asset[1]) for _ in range(n)]
        
        # Respect signal direction
        for i, asset in enumerate(active_assets):
            if signals_active[asset] > 0:
                bounds[i] = (max(bounds[i][0], 0), bounds[i][1])  # Force positive
            else:
                bounds[i] = (bounds[i][0], min(bounds[i][1], 0))  # Force negative
        
        # Constraints
        constraints = []
        
        # Gross cap constraint
        constraints.append({
            'type': 'ineq',
            'fun': lambda w: self.gross_cap - np.abs(w).sum()
        })
        
        # Net cap constraint
        constraints.append({
            'type': 'ineq',
            'fun': lambda w: self.net_cap - np.abs(w.sum())
        })
        
        # Initial guess: equal-weight respecting signal direction
        w0 = np.ones(n) * (1.0 / n)
        w0 = w0 * np.sign(signals_active.values)
        
        # Optimize
        try:
            result = minimize(
                objective,
                w0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 500, 'ftol': 1e-9}
            )
            
            if result.success:
                weights_active = pd.Series(result.x, index=active_assets)
            else:
                logger.warning(f"[Allocator] ERC optimization did not converge: {result.message}")
                # Fallback to signal-beta
                weights_active = signals_active.copy()
                weights_active = self._apply_position_bounds(weights_active)
                weights_active = self._apply_gross_cap(weights_active)
                weights_active = self._apply_net_cap(weights_active)
        
        except Exception as e:
            logger.error(f"[Allocator] ERC optimization failed: {e}")
            # Fallback to signal-beta
            weights_active = signals_active.copy()
            weights_active = self._apply_position_bounds(weights_active)
            weights_active = self._apply_gross_cap(weights_active)
            weights_active = self._apply_net_cap(weights_active)
        
        # Map back to full signal index
        weights = pd.Series(0.0, index=signals.index)
        weights.loc[active_assets] = weights_active
        
        # Apply turnover constraint
        weights = self._apply_turnover_constraint(weights, prev_weights)
        
        return weights
    
    def _solve_meanvar(
        self,
        signals: pd.Series,
        cov: pd.DataFrame,
        prev_weights: Optional[pd.Series] = None
    ) -> pd.Series:
        """
        Mean-variance optimization.
        
        Maximizes: signals @ w - risk_aversion * w @ cov @ w
        subject to constraints.
        
        Args:
            signals: Expected returns/signals
            cov: Covariance matrix
            prev_weights: Previous weights (for turnover control)
            
        Returns:
            Portfolio weights
        """
        # Align signals and cov
        common = signals.index.intersection(cov.index)
        
        if len(common) == 0:
            logger.warning("[Allocator] No common symbols between signals and covariance")
            return signals * 0
        
        signals_aligned = signals.loc[common]
        cov_aligned = cov.loc[common, common]
        
        # Remove NaN signals
        signals_aligned = signals_aligned.fillna(0)
        
        # Remove zero signals if desired (for now keep them)
        n = len(common)
        
        # Objective: maximize utility = return - risk_aversion * variance
        # We'll use a simple risk aversion parameter
        risk_aversion = 0.5
        
        def objective(w):
            w = np.array(w)
            returns = signals_aligned.values @ w
            risk = w @ cov_aligned.values @ w
            utility = returns - risk_aversion * risk
            return -utility  # Minimize negative utility
        
        # Constraints and bounds
        bounds = [(self.w_bounds_per_asset[0], self.w_bounds_per_asset[1]) for _ in range(n)]
        
        # Constraints
        constraints = []
        
        # Gross cap constraint
        constraints.append({
            'type': 'ineq',
            'fun': lambda w: self.gross_cap - np.abs(w).sum()
        })
        
        # Net cap constraint  
        constraints.append({
            'type': 'ineq',
            'fun': lambda w: self.net_cap - np.abs(w.sum())
        })
        
        # Initial guess: proportional to signals
        signal_sum = signals_aligned.abs().sum()
        if signal_sum > 1e-10:
            w0 = signals_aligned.values / signal_sum
        else:
            w0 = np.ones(n) / n
        
        # Optimize
        try:
            result = minimize(
                objective,
                w0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 500, 'ftol': 1e-9}
            )
            
            if result.success:
                weights = pd.Series(result.x, index=common)
            else:
                logger.warning(f"[Allocator] Mean-var optimization did not converge: {result.message}")
                # Fallback to signal-beta
                weights = self._solve_signal_beta(signals_aligned, cov_aligned, None)
        
        except Exception as e:
            logger.error(f"[Allocator] Mean-var optimization failed: {e}")
            # Fallback to signal-beta
            weights = self._solve_signal_beta(signals_aligned, cov_aligned, None)
        
        # Apply turnover constraint
        weights = self._apply_turnover_constraint(weights, prev_weights)
        
        return weights
    
    def solve(
        self,
        signals: pd.Series,
        cov: pd.DataFrame,
        weights_prev: Optional[pd.Series] = None
    ) -> pd.Series:
        """
        Compute optimal portfolio weights from signals.
        
        Args:
            signals: Vol-managed signals from overlay (pd.Series indexed by symbol)
            cov: Covariance matrix (pd.DataFrame, symbols x symbols)
            weights_prev: Previous portfolio weights (optional, for turnover control)
            
        Returns:
            Portfolio weights (pd.Series with same index as overlapping signals/cov)
        """
        # Handle empty signals
        if signals.empty or signals.abs().sum() == 0:
            logger.debug("[Allocator] Empty or zero signals, returning zeros")
            return signals * 0
        
        logger.debug(f"[ALLOC] signals universe: {sorted(signals.index)}")
        logger.debug(f"[ALLOC] risk returns universe (cov): {sorted(cov.index)}")
        
        # Check for overlap between signals and cov
        common = signals.index.intersection(cov.index)
        logger.debug(f"[ALLOC] common (used in allocation): {sorted(common)}")
        
        if len(common) == 0:
            logger.warning("[Allocator] No common symbols between signals and covariance")
            return pd.Series(dtype=float)  # Return empty series
        
        # Filter to common assets
        signals_common = signals.loc[common]
        cov_common = cov.loc[common, common]
        
        # Route to appropriate method
        if self.method == "signal-beta":
            weights = self._solve_signal_beta(signals_common, cov_common, weights_prev)
        elif self.method == "erc":
            weights = self._solve_erc(signals_common, cov_common, weights_prev)
        elif self.method == "meanvar":
            weights = self._solve_meanvar(signals_common, cov_common, weights_prev)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        # Log diagnostics
        gross_lev = weights.abs().sum()
        net_exp = weights.sum()
        
        if weights_prev is not None and not weights_prev.empty:
            prev_aligned = weights_prev.reindex(weights.index, fill_value=0)
            turnover = (weights - prev_aligned).abs().sum()
        else:
            turnover = gross_lev
        
        logger.debug(
            f"[Allocator] Final weights: "
            f"gross={gross_lev:.2f}, net={net_exp:.2f}, "
            f"turnover={turnover:.3f}, n_positions={(weights.abs() > 1e-6).sum()}"
        )
        
        return weights
    
    def describe(self) -> dict:
        """
        Return configuration and description of the Allocator agent.
        
        Returns:
            dict with configuration parameters
        """
        return {
            'agent': 'Allocator',
            'role': 'Convert vol-managed signals to final portfolio weights',
            'method': self.method,
            'gross_cap': self.gross_cap,
            'net_cap': self.net_cap,
            'w_bounds_per_asset': self.w_bounds_per_asset,
            'turnover_cap': self.turnover_cap,
            'lambda_turnover': self.lambda_turnover,
            'outputs': ['solve(signals, cov, weights_prev)']
        }
