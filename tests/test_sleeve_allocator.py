"""
Tests for PortfolioSleeveAllocator.
"""

import pytest
import pandas as pd
import numpy as np
from src.agents.sleeve_allocator import (
    PortfolioSleeveAllocator,
    AllocationConstraints
)


class TestSleeveAllocatorBasics:
    """Test basic sleeve allocation functionality."""
    
    def test_simple_two_sleeve_combination(self):
        """Test combining two sleeves with risk budgets."""
        allocator = PortfolioSleeveAllocator()
        
        # Create simple sleeve positions
        sleeve_positions = {
            'tsmom': pd.Series({'ES': 1.0, 'GC': 0.5}),
            'xsec': pd.Series({'ES': -0.5, 'CL': 1.0})
        }
        
        risk_budgets = {'tsmom': 0.6, 'xsec': 0.4}
        
        result = allocator.combine(sleeve_positions, risk_budgets)
        total = result['total_weights']
        
        # Check aggregation: 0.6*1.0 + 0.4*(-0.5) = 0.4
        assert np.isclose(total['ES'], 0.4)
        # Check: 0.6*0.5 + 0.4*0.0 = 0.3
        assert np.isclose(total['GC'], 0.3)
        # Check: 0.6*0.0 + 0.4*1.0 = 0.4
        assert np.isclose(total['CL'], 0.4)
    
    def test_per_sleeve_contributions(self):
        """Test that per-sleeve contributions are correctly calculated."""
        allocator = PortfolioSleeveAllocator()
        
        sleeve_positions = {
            'tsmom': pd.Series({'ES': 1.0, 'GC': 0.5}),
            'xsec': pd.Series({'ES': -0.5, 'CL': 1.0})
        }
        
        risk_budgets = {'tsmom': 0.6, 'xsec': 0.4}
        
        result = allocator.combine(sleeve_positions, risk_budgets)
        contrib = result['per_sleeve_contrib']
        
        # Check TSMOM contributions
        assert np.isclose(contrib.loc['ES', 'tsmom'], 0.6)
        assert np.isclose(contrib.loc['GC', 'tsmom'], 0.3)
        assert np.isclose(contrib.loc['CL', 'tsmom'], 0.0)
        
        # Check XSec contributions
        assert np.isclose(contrib.loc['ES', 'xsec'], -0.2)
        assert np.isclose(contrib.loc['GC', 'xsec'], 0.0)
        assert np.isclose(contrib.loc['CL', 'xsec'], 0.4)
        
        # Sum should equal total weights
        total = result['total_weights']
        contrib_sum = contrib.sum(axis=1)
        pd.testing.assert_series_equal(total, contrib_sum, check_names=False)
    
    def test_three_sleeve_combination(self):
        """Test combining three sleeves."""
        allocator = PortfolioSleeveAllocator()
        
        sleeve_positions = {
            'tsmom': pd.Series({'ES': 1.0}),
            'xsec': pd.Series({'ES': -0.5}),
            'carry': pd.Series({'ES': 0.3})
        }
        
        risk_budgets = {'tsmom': 0.5, 'xsec': 0.3, 'carry': 0.2}
        
        result = allocator.combine(sleeve_positions, risk_budgets)
        total = result['total_weights']
        
        # 0.5*1.0 + 0.3*(-0.5) + 0.2*0.3 = 0.5 - 0.15 + 0.06 = 0.41
        assert np.isclose(total['ES'], 0.41)
    
    def test_deterministic_output(self):
        """Test that output is deterministic (same input -> same output)."""
        allocator = PortfolioSleeveAllocator()
        
        sleeve_positions = {
            'tsmom': pd.Series({'ES': 1.0, 'GC': 0.5, 'CL': -0.3}),
            'xsec': pd.Series({'ES': -0.5, 'CL': 1.0, 'SI': 0.2})
        }
        
        risk_budgets = {'tsmom': 0.6, 'xsec': 0.4}
        
        # Run multiple times
        results = [
            allocator.combine(sleeve_positions, risk_budgets)
            for _ in range(5)
        ]
        
        # All results should be identical
        for i in range(1, len(results)):
            pd.testing.assert_series_equal(
                results[0]['total_weights'],
                results[i]['total_weights']
            )
    
    def test_linear_scaling_by_risk_budgets(self):
        """Test that scaling by risk budgets is linear."""
        allocator = PortfolioSleeveAllocator()
        
        sleeve_positions = {
            'tsmom': pd.Series({'ES': 1.0, 'GC': 0.5}),
            'xsec': pd.Series({'ES': -0.5, 'CL': 1.0})
        }
        
        # Test with different budget splits
        budget_sets = [
            {'tsmom': 0.6, 'xsec': 0.4},
            {'tsmom': 0.8, 'xsec': 0.2},
            {'tsmom': 0.5, 'xsec': 0.5}
        ]
        
        for budgets in budget_sets:
            result = allocator.combine(sleeve_positions, budgets)
            total = result['total_weights']
            
            # Manually calculate expected
            expected_es = (budgets['tsmom'] * 1.0 + 
                          budgets['xsec'] * (-0.5))
            expected_gc = budgets['tsmom'] * 0.5
            expected_cl = budgets['xsec'] * 1.0
            
            assert np.isclose(total['ES'], expected_es)
            assert np.isclose(total['GC'], expected_gc)
            assert np.isclose(total['CL'], expected_cl)


class TestConstraintEnforcement:
    """Test constraint enforcement."""
    
    def test_per_asset_bounds_enforcement(self):
        """Test that per-asset bounds are enforced."""
        constraints = AllocationConstraints(
            bounds_per_asset=(-0.5, 0.5),
            gross_cap=10.0,
            net_cap=10.0
        )
        allocator = PortfolioSleeveAllocator(constraints)
        
        # Create positions that violate bounds
        sleeve_positions = {
            'tsmom': pd.Series({'ES': 2.0, 'GC': -1.5})
        }
        
        risk_budgets = {'tsmom': 1.0}
        
        result = allocator.combine(sleeve_positions, risk_budgets)
        total = result['total_weights']
        
        # Should be clipped to bounds
        assert total['ES'] <= 0.5
        assert total['GC'] >= -0.5
        
        # Check violation was detected
        assert result['feasibility']['initial_violations']['bounds']
        assert result['feasibility']['projection_applied']
    
    def test_gross_cap_enforcement(self):
        """Test that gross exposure cap is enforced."""
        constraints = AllocationConstraints(
            bounds_per_asset=(-5.0, 5.0),
            gross_cap=2.0,
            net_cap=10.0
        )
        allocator = PortfolioSleeveAllocator(constraints)
        
        # Create positions that violate gross cap
        sleeve_positions = {
            'tsmom': pd.Series({'ES': 1.5, 'GC': 1.0, 'CL': -1.5})
        }
        
        risk_budgets = {'tsmom': 1.0}
        
        result = allocator.combine(sleeve_positions, risk_budgets)
        total = result['total_weights']
        
        # Gross should be at or below cap
        gross = total.abs().sum()
        assert gross <= constraints.gross_cap + 1e-6
        
        # Check violation was detected
        assert result['feasibility']['initial_violations']['gross_cap']
    
    def test_net_cap_enforcement(self):
        """Test that net exposure cap is enforced."""
        constraints = AllocationConstraints(
            bounds_per_asset=(-5.0, 5.0),
            gross_cap=10.0,
            net_cap=1.0
        )
        allocator = PortfolioSleeveAllocator(constraints)
        
        # Create positions that violate net cap
        sleeve_positions = {
            'tsmom': pd.Series({'ES': 2.0, 'GC': 1.5, 'CL': 1.0})
        }
        
        risk_budgets = {'tsmom': 1.0}
        
        result = allocator.combine(sleeve_positions, risk_budgets)
        total = result['total_weights']
        
        # Net should be at or below cap
        net = abs(total.sum())
        assert net <= constraints.net_cap + 1e-6
        
        # Check violation was detected
        assert result['feasibility']['initial_violations']['net_cap']
    
    def test_turnover_cap_enforcement(self):
        """Test that turnover cap is enforced when prev_weights provided."""
        constraints = AllocationConstraints(
            bounds_per_asset=(-5.0, 5.0),
            gross_cap=10.0,
            net_cap=10.0,
            turnover_cap=0.5
        )
        allocator = PortfolioSleeveAllocator(constraints)
        
        # Create positions with high turnover
        sleeve_positions = {
            'tsmom': pd.Series({'ES': 2.0, 'GC': 1.0})
        }
        
        risk_budgets = {'tsmom': 1.0}
        prev_weights = pd.Series({'ES': 0.0, 'GC': 0.0})
        
        result = allocator.combine(
            sleeve_positions, risk_budgets, prev_weights
        )
        total = result['total_weights']
        
        # Turnover should be at or below cap
        turnover = (total - prev_weights).abs().sum()
        assert turnover <= constraints.turnover_cap + 1e-6
        
        # Check violation was detected
        assert result['feasibility']['initial_violations']['turnover_cap']
    
    def test_no_violations_when_feasible(self):
        """Test that no projection is applied when already feasible."""
        constraints = AllocationConstraints(
            bounds_per_asset=(-2.0, 2.0),
            gross_cap=10.0,
            net_cap=5.0
        )
        allocator = PortfolioSleeveAllocator(constraints)
        
        # Create feasible positions
        sleeve_positions = {
            'tsmom': pd.Series({'ES': 0.5, 'GC': -0.3})
        }
        
        risk_budgets = {'tsmom': 1.0}
        
        result = allocator.combine(sleeve_positions, risk_budgets)
        
        # No projection should be applied
        assert not result['feasibility']['projection_applied']
        assert not any(result['feasibility']['initial_violations'].values())
    
    def test_multiple_constraints_simultaneously(self):
        """Test handling multiple violated constraints at once."""
        constraints = AllocationConstraints(
            bounds_per_asset=(-0.5, 0.5),
            gross_cap=1.5,
            net_cap=0.8
        )
        allocator = PortfolioSleeveAllocator(constraints)
        
        # Create positions that violate multiple constraints
        sleeve_positions = {
            'tsmom': pd.Series({'ES': 2.0, 'GC': 1.5, 'CL': 1.0})
        }
        
        risk_budgets = {'tsmom': 1.0}
        
        result = allocator.combine(sleeve_positions, risk_budgets)
        total = result['total_weights']
        
        # All constraints should be satisfied
        assert (total >= constraints.bounds_per_asset[0] - 1e-6).all()
        assert (total <= constraints.bounds_per_asset[1] + 1e-6).all()
        assert total.abs().sum() <= constraints.gross_cap + 1e-6
        assert abs(total.sum()) <= constraints.net_cap + 1e-6


class TestInputValidation:
    """Test input validation."""
    
    def test_empty_sleeve_positions_raises_error(self):
        """Test that empty sleeve_positions raises error."""
        allocator = PortfolioSleeveAllocator()
        
        with pytest.raises(ValueError, match="sleeve_positions cannot be empty"):
            allocator.combine({}, {'tsmom': 1.0})
    
    def test_empty_risk_budgets_raises_error(self):
        """Test that empty risk_budgets raises error."""
        allocator = PortfolioSleeveAllocator()
        
        sleeve_positions = {'tsmom': pd.Series({'ES': 1.0})}
        
        with pytest.raises(ValueError, match="risk_budgets cannot be empty"):
            allocator.combine(sleeve_positions, {})
    
    def test_missing_risk_budgets_raises_error(self):
        """Test that missing risk budgets for sleeves raises error."""
        allocator = PortfolioSleeveAllocator()
        
        sleeve_positions = {
            'tsmom': pd.Series({'ES': 1.0}),
            'xsec': pd.Series({'ES': 0.5})
        }
        
        risk_budgets = {'tsmom': 1.0}  # Missing xsec
        
        with pytest.raises(ValueError, match="Missing risk budgets"):
            allocator.combine(sleeve_positions, risk_budgets)
    
    def test_risk_budgets_not_summing_to_one_raises_error(self):
        """Test that risk budgets not summing to 1.0 raises error."""
        allocator = PortfolioSleeveAllocator()
        
        sleeve_positions = {
            'tsmom': pd.Series({'ES': 1.0}),
            'xsec': pd.Series({'ES': 0.5})
        }
        
        risk_budgets = {'tsmom': 0.6, 'xsec': 0.5}  # Sums to 1.1
        
        with pytest.raises(ValueError, match="must sum to 1.0"):
            allocator.combine(sleeve_positions, risk_budgets)
    
    def test_negative_risk_budgets_raises_error(self):
        """Test that negative risk budgets raise error."""
        allocator = PortfolioSleeveAllocator()
        
        sleeve_positions = {
            'tsmom': pd.Series({'ES': 1.0}),
            'xsec': pd.Series({'ES': 0.5})
        }
        
        risk_budgets = {'tsmom': 1.2, 'xsec': -0.2}
        
        with pytest.raises(ValueError, match="must be non-negative"):
            allocator.combine(sleeve_positions, risk_budgets)


class TestEdgeCases:
    """Test edge cases."""
    
    def test_single_asset(self):
        """Test with single asset across sleeves."""
        allocator = PortfolioSleeveAllocator()
        
        sleeve_positions = {
            'tsmom': pd.Series({'ES': 1.0}),
            'xsec': pd.Series({'ES': -0.5})
        }
        
        risk_budgets = {'tsmom': 0.6, 'xsec': 0.4}
        
        result = allocator.combine(sleeve_positions, risk_budgets)
        total = result['total_weights']
        
        assert len(total) == 1
        assert np.isclose(total['ES'], 0.4)
    
    def test_no_overlapping_assets(self):
        """Test sleeves with completely different assets."""
        allocator = PortfolioSleeveAllocator()
        
        sleeve_positions = {
            'tsmom': pd.Series({'ES': 1.0, 'GC': 0.5}),
            'xsec': pd.Series({'CL': 1.0, 'SI': -0.3})
        }
        
        risk_budgets = {'tsmom': 0.6, 'xsec': 0.4}
        
        result = allocator.combine(sleeve_positions, risk_budgets)
        total = result['total_weights']
        
        # Should have all 4 assets
        assert len(total) == 4
        assert np.isclose(total['ES'], 0.6)
        assert np.isclose(total['GC'], 0.3)
        assert np.isclose(total['CL'], 0.4)
        assert np.isclose(total['SI'], -0.12)
    
    def test_zero_positions_in_sleeve(self):
        """Test sleeve with all zero positions."""
        allocator = PortfolioSleeveAllocator()
        
        sleeve_positions = {
            'tsmom': pd.Series({'ES': 1.0, 'GC': 0.5}),
            'xsec': pd.Series({'ES': 0.0, 'GC': 0.0})
        }
        
        risk_budgets = {'tsmom': 0.6, 'xsec': 0.4}
        
        result = allocator.combine(sleeve_positions, risk_budgets)
        total = result['total_weights']
        
        # Only TSMOM should contribute
        assert np.isclose(total['ES'], 0.6)
        assert np.isclose(total['GC'], 0.3)
    
    def test_equal_risk_budgets(self):
        """Test with equal risk budgets across sleeves."""
        allocator = PortfolioSleeveAllocator()
        
        sleeve_positions = {
            'tsmom': pd.Series({'ES': 1.0}),
            'xsec': pd.Series({'ES': 1.0}),
            'carry': pd.Series({'ES': 1.0})
        }
        
        risk_budgets = {
            'tsmom': 1.0/3.0,
            'xsec': 1.0/3.0,
            'carry': 1.0/3.0
        }
        
        result = allocator.combine(sleeve_positions, risk_budgets)
        total = result['total_weights']
        
        # Should be average
        assert np.isclose(total['ES'], 1.0)
    
    def test_previous_weights_with_new_assets(self):
        """Test with prev_weights that have different assets."""
        constraints = AllocationConstraints(turnover_cap=1.0)
        allocator = PortfolioSleeveAllocator(constraints)
        
        sleeve_positions = {
            'tsmom': pd.Series({'ES': 1.0, 'GC': 0.5})
        }
        
        risk_budgets = {'tsmom': 1.0}
        
        # Previous weights have different assets
        prev_weights = pd.Series({'CL': 0.3, 'ES': 0.2})
        
        result = allocator.combine(
            sleeve_positions, risk_budgets, prev_weights
        )
        
        # Should handle gracefully
        assert 'total_weights' in result
        assert len(result['total_weights']) == 3  # ES, GC, CL


class TestRealWorldScenarios:
    """Test realistic portfolio scenarios."""
    
    def test_standard_60_40_tsmom_xsec(self):
        """Test standard 60/40 TSMOM/XSec allocation."""
        constraints = AllocationConstraints(
            bounds_per_asset=(-1.5, 1.5),
            gross_cap=7.0,
            net_cap=2.0
        )
        allocator = PortfolioSleeveAllocator(constraints)
        
        # Create realistic positions
        sleeve_positions = {
            'tsmom': pd.Series({
                'ES': 0.8, 'GC': -0.4, 'CL': 0.6,
                'SI': 0.3, 'NG': -0.5
            }),
            'xsec': pd.Series({
                'ES': -0.3, 'GC': 0.7, 'CL': -0.2,
                'SI': 0.5, 'HG': 0.4
            })
        }
        
        risk_budgets = {'tsmom': 0.6, 'xsec': 0.4}
        
        result = allocator.combine(sleeve_positions, risk_budgets)
        total = result['total_weights']
        contrib = result['per_sleeve_contrib']
        
        # Check basic properties
        assert len(total) == 6  # All unique assets
        
        # Verify contributions sum to total
        contrib_sum = contrib.sum(axis=1)
        for asset in total.index:
            assert np.isclose(total[asset], contrib_sum[asset])
        
        # Verify constraints satisfied
        assert (total >= constraints.bounds_per_asset[0] - 1e-6).all()
        assert (total <= constraints.bounds_per_asset[1] + 1e-6).all()
        assert total.abs().sum() <= constraints.gross_cap + 1e-6
        assert abs(total.sum()) <= constraints.net_cap + 1e-6
    
    def test_three_sleeve_portfolio(self):
        """Test portfolio with three sleeves."""
        constraints = AllocationConstraints(
            bounds_per_asset=(-1.0, 1.0),
            gross_cap=5.0,
            net_cap=1.5
        )
        allocator = PortfolioSleeveAllocator(constraints)
        
        sleeve_positions = {
            'tsmom': pd.Series({'ES': 0.5, 'GC': -0.3, 'CL': 0.4}),
            'xsec': pd.Series({'ES': -0.2, 'GC': 0.4, 'SI': 0.3}),
            'carry': pd.Series({'GC': 0.6, 'CL': -0.2, 'SI': 0.1})
        }
        
        risk_budgets = {'tsmom': 0.5, 'xsec': 0.3, 'carry': 0.2}
        
        result = allocator.combine(sleeve_positions, risk_budgets)
        total = result['total_weights']
        contrib = result['per_sleeve_contrib']
        
        # Verify all three sleeves contribute
        assert 'tsmom' in contrib.columns
        assert 'xsec' in contrib.columns
        assert 'carry' in contrib.columns
        
        # Check ES specifically (has positions in tsmom and xsec only)
        es_contrib = contrib.loc['ES']
        expected_es = 0.5 * 0.5 + 0.3 * (-0.2)  # 0.25 - 0.06 = 0.19
        assert np.isclose(es_contrib['tsmom'] + es_contrib['xsec'], expected_es)
        assert np.isclose(es_contrib['carry'], 0.0)
    
    def test_portfolio_rebalance_with_turnover_limit(self):
        """Test rebalancing with turnover constraint."""
        constraints = AllocationConstraints(
            bounds_per_asset=(-1.5, 1.5),
            gross_cap=7.0,
            net_cap=2.0,
            turnover_cap=2.0
        )
        allocator = PortfolioSleeveAllocator(constraints)
        
        # Current positions
        prev_weights = pd.Series({
            'ES': 0.5, 'GC': -0.2, 'CL': 0.3, 'SI': 0.1
        })
        
        # New target positions
        sleeve_positions = {
            'tsmom': pd.Series({
                'ES': 1.0, 'GC': 0.5, 'CL': -0.8, 'SI': 0.6
            })
        }
        
        risk_budgets = {'tsmom': 1.0}
        
        result = allocator.combine(
            sleeve_positions, risk_budgets, prev_weights
        )
        total = result['total_weights']
        
        # Calculate actual turnover
        turnover = (total - prev_weights).abs().sum()
        
        # Should respect turnover cap
        assert turnover <= constraints.turnover_cap + 1e-6

