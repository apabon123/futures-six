"""
PortfolioSleeveAllocator: Combines multiple sleeves into total portfolio weights.

Aggregates sleeve positions using risk budgets, enforces gross/net caps,
per-asset bounds, and optional turnover constraints.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class AllocationConstraints:
    """Portfolio constraints for sleeve allocation."""
    
    bounds_per_asset: Tuple[float, float] = (-1.5, 1.5)
    gross_cap: float = 7.0
    net_cap: float = 2.0
    turnover_cap: Optional[float] = None


class PortfolioSleeveAllocator:
    """
    Combines multiple strategy sleeves into total portfolio weights.
    
    Operates on scaled signals that have already passed through MacroRegime
    and VolManaged overlays. Uses explicit risk budgets to combine sleeves
    and enforces portfolio-level constraints.
    """
    
    def __init__(self, constraints: Optional[AllocationConstraints] = None):
        """
        Initialize the allocator.
        
        Args:
            constraints: Portfolio constraints. Uses defaults if None.
        """
        self.constraints = constraints or AllocationConstraints()
    
    def combine(
        self,
        sleeve_positions: Dict[str, pd.Series],
        risk_budgets: Dict[str, float],
        prev_weights: Optional[pd.Series] = None
    ) -> Dict[str, pd.Series]:
        """
        Combine sleeve positions into total portfolio weights.
        
        Args:
            sleeve_positions: Dict mapping sleeve name to position Series.
                             Positions are in risk units, post-overlay.
            risk_budgets: Dict mapping sleeve name to risk budget weight.
                         Should sum to 1.0.
            prev_weights: Previous total weights for turnover constraint.
        
        Returns:
            Dict with keys:
                - 'total_weights': Combined portfolio weights
                - 'per_sleeve_contrib': DataFrame of per-sleeve contributions
                - 'feasibility': Dict with constraint violation info
        """
        # Validate inputs
        self._validate_inputs(sleeve_positions, risk_budgets)
        
        # Get all assets across sleeves (include prev_weights assets if provided)
        all_assets = self._get_all_assets(sleeve_positions, prev_weights)
        
        # Align all sleeve positions to same index
        aligned_positions = self._align_positions(sleeve_positions, all_assets)
        
        # Aggregate with risk budgets in risk space
        total_weights = self._aggregate_sleeves(aligned_positions, risk_budgets)
        
        # Store per-sleeve contributions before constraint enforcement
        per_sleeve_contrib = self._calculate_contributions(
            aligned_positions, risk_budgets
        )
        
        # Enforce constraints
        total_weights, feasibility_info = self._enforce_constraints(
            total_weights, prev_weights
        )
        
        return {
            'total_weights': total_weights,
            'per_sleeve_contrib': per_sleeve_contrib,
            'feasibility': feasibility_info
        }
    
    def _validate_inputs(
        self,
        sleeve_positions: Dict[str, pd.Series],
        risk_budgets: Dict[str, float]
    ) -> None:
        """Validate input parameters."""
        if not sleeve_positions:
            raise ValueError("sleeve_positions cannot be empty")
        
        if not risk_budgets:
            raise ValueError("risk_budgets cannot be empty")
        
        # Check that all sleeves have risk budgets
        missing_budgets = set(sleeve_positions.keys()) - set(risk_budgets.keys())
        if missing_budgets:
            raise ValueError(f"Missing risk budgets for sleeves: {missing_budgets}")
        
        # Check that risk budgets sum to approximately 1.0
        total_budget = sum(risk_budgets.values())
        if not np.isclose(total_budget, 1.0, atol=1e-6):
            raise ValueError(
                f"Risk budgets must sum to 1.0, got {total_budget:.6f}"
            )
        
        # Check for negative budgets
        if any(b < 0 for b in risk_budgets.values()):
            raise ValueError("Risk budgets must be non-negative")
    
    def _get_all_assets(
        self, sleeve_positions: Dict[str, pd.Series], prev_weights: Optional[pd.Series] = None
    ) -> pd.Index:
        """Get union of all assets across sleeves and previous weights."""
        all_assets = set()
        for positions in sleeve_positions.values():
            all_assets.update(positions.index)
        # Include assets from previous weights for turnover calculation
        if prev_weights is not None:
            all_assets.update(prev_weights.index)
        return pd.Index(sorted(all_assets))
    
    def _align_positions(
        self,
        sleeve_positions: Dict[str, pd.Series],
        all_assets: pd.Index
    ) -> Dict[str, pd.Series]:
        """Align all sleeve positions to same asset index (fill missing with 0)."""
        aligned = {}
        for sleeve_name, positions in sleeve_positions.items():
            aligned[sleeve_name] = positions.reindex(all_assets, fill_value=0.0)
        return aligned
    
    def _aggregate_sleeves(
        self,
        aligned_positions: Dict[str, pd.Series],
        risk_budgets: Dict[str, float]
    ) -> pd.Series:
        """
        Aggregate sleeve positions using risk budgets.
        
        total = Σ (w_sleeve * pos_sleeve)
        """
        # Get all assets
        assets = next(iter(aligned_positions.values())).index
        total = pd.Series(0.0, index=assets)
        
        for sleeve_name, positions in aligned_positions.items():
            budget = risk_budgets.get(sleeve_name, 0.0)
            total += budget * positions
        
        return total
    
    def _calculate_contributions(
        self,
        aligned_positions: Dict[str, pd.Series],
        risk_budgets: Dict[str, float]
    ) -> pd.DataFrame:
        """Calculate per-sleeve contributions to each asset."""
        contributions = {}
        for sleeve_name, positions in aligned_positions.items():
            budget = risk_budgets.get(sleeve_name, 0.0)
            contributions[sleeve_name] = budget * positions
        
        return pd.DataFrame(contributions)
    
    def _enforce_constraints(
        self,
        weights: pd.Series,
        prev_weights: Optional[pd.Series] = None
    ) -> Tuple[pd.Series, Dict]:
        """
        Enforce portfolio constraints on weights.
        
        Uses L2 projection to nearest feasible solution if infeasible.
        """
        feasibility_info = {
            'initial_violations': {},
            'final_violations': {},
            'projection_applied': False
        }
        
        # Check initial violations
        feasibility_info['initial_violations'] = self._check_violations(
            weights, prev_weights
        )
        
        # If no violations, return as-is
        if not any(feasibility_info['initial_violations'].values()):
            return weights, feasibility_info
        
        # Apply L2 projection to enforce constraints
        constrained_weights = self._project_to_feasible(weights, prev_weights)
        feasibility_info['projection_applied'] = True
        
        # Check final violations
        feasibility_info['final_violations'] = self._check_violations(
            constrained_weights, prev_weights
        )
        
        return constrained_weights, feasibility_info
    
    def _check_violations(
        self,
        weights: pd.Series,
        prev_weights: Optional[pd.Series] = None
    ) -> Dict[str, bool]:
        """Check which constraints are violated."""
        violations = {}
        
        # Per-asset bounds
        lower, upper = self.constraints.bounds_per_asset
        violations['bounds'] = (weights < lower).any() or (weights > upper).any()
        
        # Gross cap
        gross_exposure = weights.abs().sum()
        violations['gross_cap'] = gross_exposure > self.constraints.gross_cap
        
        # Net cap
        net_exposure = abs(weights.sum())
        violations['net_cap'] = net_exposure > self.constraints.net_cap
        
        # Turnover cap
        if self.constraints.turnover_cap is not None and prev_weights is not None:
            prev_aligned = prev_weights.reindex(weights.index, fill_value=0.0)
            turnover = (weights - prev_aligned).abs().sum()
            violations['turnover_cap'] = turnover > self.constraints.turnover_cap
        else:
            violations['turnover_cap'] = False
        
        return violations
    
    def _project_to_feasible(
        self,
        weights: pd.Series,
        prev_weights: Optional[pd.Series] = None
    ) -> pd.Series:
        """
        Project weights to nearest feasible solution using L2 norm.
        
        Implements a simple iterative projection algorithm that cycles
        through constraints until convergence.
        """
        w = weights.copy()
        lower, upper = self.constraints.bounds_per_asset
        
        # Prepare previous weights if needed
        if prev_weights is not None:
            prev_aligned = prev_weights.reindex(w.index, fill_value=0.0)
        else:
            prev_aligned = None
        
        max_iterations = 100
        tolerance = 1e-6
        
        for iteration in range(max_iterations):
            w_old = w.copy()
            
            # 1. Enforce per-asset bounds
            w = w.clip(lower=lower, upper=upper)
            
            # 2. Enforce turnover cap (if applicable)
            if (self.constraints.turnover_cap is not None and 
                prev_aligned is not None):
                w = self._project_turnover(w, prev_aligned)
            
            # 3. Enforce gross cap
            w = self._project_gross(w)
            
            # 4. Enforce net cap
            w = self._project_net(w)
            
            # Check convergence
            if np.allclose(w, w_old, atol=tolerance):
                break
        
        return w
    
    def _project_turnover(
        self,
        weights: pd.Series,
        prev_weights: pd.Series
    ) -> pd.Series:
        """Project to satisfy turnover constraint."""
        if self.constraints.turnover_cap is None:
            return weights
        
        turnover = (weights - prev_weights).abs().sum()
        
        if turnover <= self.constraints.turnover_cap:
            return weights
        
        # Scale the change to respect turnover cap
        change = weights - prev_weights
        scale = self.constraints.turnover_cap / turnover
        return prev_weights + scale * change
    
    def _project_gross(self, weights: pd.Series) -> pd.Series:
        """Project to satisfy gross exposure cap."""
        gross = weights.abs().sum()
        
        if gross <= self.constraints.gross_cap:
            return weights
        
        # Scale all positions proportionally
        scale = self.constraints.gross_cap / gross
        return weights * scale
    
    def _project_net(self, weights: pd.Series) -> pd.Series:
        """Project to satisfy net exposure cap."""
        net = weights.sum()
        
        if abs(net) <= self.constraints.net_cap:
            return weights
        
        # Shift all positions equally to respect net cap
        excess = net - np.sign(net) * self.constraints.net_cap
        shift = excess / len(weights)
        return weights - shift

