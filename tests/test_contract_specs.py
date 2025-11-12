"""
Tests for ContractSpecs service.

Validates alias resolution, spec field presence, USD move calculations,
and USD returns conversions.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from src.agents.contract_specs import ContractSpecs


@pytest.fixture
def specs():
    """Create ContractSpecs instance for testing."""
    config_path = Path("configs/contracts.yaml")
    return ContractSpecs(config_path=str(config_path))


class TestAliasResolution:
    """Test alias resolution to canonical roots."""
    
    def test_alias_resolves_to_correct_root(self, specs):
        """Database symbols resolve to correct root symbols."""
        assert specs.canonical_root("ES_FRONT_CALENDAR_2D") == "ES"
        assert specs.canonical_root("NQ_FRONT_CALENDAR_2D") == "NQ"
        assert specs.canonical_root("ZN_FRONT_VOLUME") == "ZN"
        assert specs.canonical_root("ZF_FRONT_VOLUME") == "ZF"
        assert specs.canonical_root("ZT_FRONT_VOLUME") == "ZT"
        assert specs.canonical_root("UB_FRONT_VOLUME") == "UB"
        assert specs.canonical_root("CL_FRONT_VOLUME") == "CL"
        assert specs.canonical_root("GC_FRONT_VOLUME") == "GC"
        assert specs.canonical_root("6E_FRONT_CALENDAR_2D") == "6E"
        assert specs.canonical_root("6B_FRONT_CALENDAR_2D") == "6B"
        assert specs.canonical_root("6J_FRONT_CALENDAR_2D") == "6J"
        assert specs.canonical_root("RTY_FRONT_CALENDAR_2D") == "RTY"
    
    def test_root_symbol_returns_itself(self, specs):
        """Root symbols resolve to themselves."""
        for root in ["ES", "NQ", "RTY", "ZN", "ZF", "ZT", "UB", "CL", "GC", "6E", "6B", "6J"]:
            assert specs.canonical_root(root) == root
    
    def test_unknown_symbol_raises_error(self, specs):
        """Unknown symbols raise ValueError."""
        with pytest.raises(ValueError, match="Unknown symbol or alias"):
            specs.canonical_root("INVALID_SYMBOL")


class TestSpecFieldsPresent:
    """Test that all required spec fields are present and valid."""
    
    def test_all_required_fields_present(self, specs):
        """All contracts have required fields."""
        required_fields = ['root', 'multiplier', 'point_value', 'tick_size', 'currency', 'fx_base']
        
        for symbol in ['ES', 'NQ', 'RTY', 'ZN', 'ZF', 'ZT', 'UB', 'CL', 'GC', '6E', '6B', '6J']:
            spec = specs.spec(symbol)
            for field in required_fields:
                assert field in spec, f"{symbol} missing field: {field}"
    
    def test_numeric_fields_are_numeric(self, specs):
        """Numeric fields contain numeric values."""
        numeric_fields = ['multiplier', 'point_value', 'tick_size']
        
        for symbol in ['ES', 'NQ', 'RTY', 'ZN', 'ZF', 'ZT', 'UB', 'CL', 'GC', '6E', '6B', '6J']:
            spec = specs.spec(symbol)
            for field in numeric_fields:
                value = spec[field]
                assert isinstance(value, (int, float)), \
                    f"{symbol}.{field} is not numeric: {type(value)}"
                assert value > 0, f"{symbol}.{field} must be positive: {value}"
    
    def test_meta_field_exists(self, specs):
        """Meta field exists in returned spec (even if empty)."""
        for symbol in ['ES', 'NQ', 'RTY', 'ZN', 'ZF', 'ZT', 'UB', 'CL', 'GC', '6E', '6B', '6J']:
            spec = specs.spec(symbol)
            assert 'meta' in spec
            assert isinstance(spec['meta'], dict)
    
    def test_spec_via_alias(self, specs):
        """Spec can be retrieved using database alias."""
        spec_via_alias = specs.spec("ES_FRONT_CALENDAR_2D")
        spec_via_root = specs.spec("ES")
        
        assert spec_via_alias == spec_via_root
        assert spec_via_alias['root'] == 'ES'


class TestUsdMoveSign:
    """Test USD move calculation sign and magnitude."""
    
    def test_positive_price_change_positive_usd_move(self, specs):
        """Positive price change produces positive USD move."""
        for symbol in ['ES', 'NQ', 'RTY', 'ZN', 'ZF', 'ZT', 'UB', 'CL', 'GC', '6E', '6B', '6J']:
            usd_move = specs.usd_move(symbol, 1.0)
            assert usd_move > 0, f"{symbol} positive move should be positive: {usd_move}"
    
    def test_negative_price_change_negative_usd_move(self, specs):
        """Negative price change produces negative USD move."""
        for symbol in ['ES', 'NQ', 'RTY', 'ZN', 'ZF', 'ZT', 'UB', 'CL', 'GC', '6E', '6B', '6J']:
            usd_move = specs.usd_move(symbol, -1.0)
            assert usd_move < 0, f"{symbol} negative move should be negative: {usd_move}"
    
    def test_zero_price_change_zero_usd_move(self, specs):
        """Zero price change produces zero USD move."""
        for symbol in ['ES', 'NQ', 'RTY', 'ZN', 'ZF', 'ZT', 'UB', 'CL', 'GC', '6E', '6B', '6J']:
            usd_move = specs.usd_move(symbol, 0.0)
            assert usd_move == 0.0, f"{symbol} zero move should be zero: {usd_move}"
    
    def test_es_one_point_move(self, specs):
        """ES 1-point move equals $50."""
        usd_move = specs.usd_move("ES", 1.0)
        assert usd_move == 50.0
    
    def test_nq_one_point_move(self, specs):
        """NQ 1-point move equals $20."""
        usd_move = specs.usd_move("NQ", 1.0)
        assert usd_move == 20.0
    
    def test_cl_one_point_move(self, specs):
        """CL 1-point move equals $1000."""
        usd_move = specs.usd_move("CL", 1.0)
        assert usd_move == 1000.0
    
    def test_usd_move_with_fx_rates(self, specs):
        """USD move calculation with optional fx_rates parameter."""
        # For USD-based contracts, fx_rates shouldn't affect result
        usd_move_no_fx = specs.usd_move("ES", 1.0)
        usd_move_with_fx = specs.usd_move("ES", 1.0, fx_rates={'EURUSD': 1.05})
        assert usd_move_no_fx == usd_move_with_fx
    
    def test_usd_move_via_alias(self, specs):
        """USD move calculation works with database aliases."""
        move_via_alias = specs.usd_move("ES_FRONT_CALENDAR_2D", 1.0)
        move_via_root = specs.usd_move("ES", 1.0)
        assert move_via_alias == move_via_root


class TestToUsdReturnsShape:
    """Test USD returns conversion maintains shape and alignment."""
    
    def test_output_aligned_to_input_index(self, specs):
        """Output series has same index as input series."""
        dates = pd.date_range('2024-01-01', periods=10, freq='D')
        prices = pd.Series(range(4000, 4010), index=dates)
        
        returns = specs.to_usd_returns(prices, "ES")
        
        assert isinstance(returns, pd.Series)
        assert len(returns) == len(prices)
        assert returns.index.equals(prices.index)
    
    def test_output_same_length_as_input(self, specs):
        """Output series has same length as input series."""
        prices = pd.Series([4000, 4001, 4002, 4003, 4004])
        returns = specs.to_usd_returns(prices, "ES")
        
        assert len(returns) == len(prices)
    
    def test_first_return_is_nan(self, specs):
        """First return is NaN (no previous price for comparison)."""
        prices = pd.Series([4000, 4001, 4002])
        returns = specs.to_usd_returns(prices, "ES")
        
        assert pd.isna(returns.iloc[0])
    
    def test_no_mutation_of_input(self, specs):
        """Input series is not mutated."""
        prices = pd.Series([4000, 4001, 4002, 4003])
        prices_copy = prices.copy()
        
        specs.to_usd_returns(prices, "ES")
        
        pd.testing.assert_series_equal(prices, prices_copy)
    
    def test_returns_calculation_es(self, specs):
        """Returns are calculated correctly for ES."""
        # ES: point_value = 50
        # Price moves from 4000 to 4001 (+1 point = +$50)
        # Notional at 4000 = 4000 * 50 = 200,000
        # Return = 50 / 200,000 = 0.00025
        prices = pd.Series([4000.0, 4001.0])
        returns = specs.to_usd_returns(prices, "ES")
        
        assert pd.isna(returns.iloc[0])
        assert returns.iloc[1] == pytest.approx(0.00025, rel=1e-6)
    
    def test_returns_with_custom_index(self, specs):
        """Returns work with custom datetime index."""
        dates = pd.date_range('2024-06-01', periods=5, freq='H')
        prices = pd.Series([4000, 4001, 4000, 4002, 4001], index=dates)
        
        returns = specs.to_usd_returns(prices, "ES")
        
        assert returns.index.equals(dates)
        assert len(returns) == 5
    
    def test_returns_via_alias(self, specs):
        """Returns calculation works with database aliases."""
        prices = pd.Series([4000.0, 4001.0])
        
        returns_via_alias = specs.to_usd_returns(prices, "ES_FRONT_CALENDAR_2D")
        returns_via_root = specs.to_usd_returns(prices, "ES")
        
        pd.testing.assert_series_equal(returns_via_alias, returns_via_root)
    
    def test_returns_with_optional_fx_series(self, specs):
        """Returns calculation accepts optional fx_series parameter."""
        prices = pd.Series([4000.0, 4001.0])
        fx_series = pd.Series([1.0, 1.0])
        
        # Should not raise error
        returns = specs.to_usd_returns(prices, "ES", fx_series=fx_series)
        assert len(returns) == len(prices)
    
    def test_invalid_price_series_type_raises_error(self, specs):
        """Non-Series price input raises TypeError."""
        with pytest.raises(TypeError, match="price_series must be a pandas Series"):
            specs.to_usd_returns([4000, 4001, 4002], "ES")
    
    def test_invalid_fx_series_type_raises_error(self, specs):
        """Non-Series fx_series raises TypeError."""
        prices = pd.Series([4000.0, 4001.0])
        
        with pytest.raises(TypeError, match="fx_series must be a pandas Series"):
            specs.to_usd_returns(prices, "ES", fx_series=[1.0, 1.0])


class TestDeterministicBehavior:
    """Test that all operations are deterministic."""
    
    def test_repeated_spec_calls_identical(self, specs):
        """Multiple calls to spec() return identical results."""
        spec1 = specs.spec("ES")
        spec2 = specs.spec("ES")
        spec3 = specs.spec("ES")
        
        assert spec1 == spec2 == spec3
    
    def test_repeated_usd_move_calls_identical(self, specs):
        """Multiple calls to usd_move() return identical results."""
        move1 = specs.usd_move("ES", 1.5)
        move2 = specs.usd_move("ES", 1.5)
        move3 = specs.usd_move("ES", 1.5)
        
        assert move1 == move2 == move3
    
    def test_repeated_returns_calls_identical(self, specs):
        """Multiple calls to to_usd_returns() return identical results."""
        prices = pd.Series([4000, 4001, 4002, 4003])
        
        returns1 = specs.to_usd_returns(prices, "ES")
        returns2 = specs.to_usd_returns(prices, "ES")
        returns3 = specs.to_usd_returns(prices, "ES")
        
        pd.testing.assert_series_equal(returns1, returns2)
        pd.testing.assert_series_equal(returns2, returns3)

