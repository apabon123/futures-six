"""
Contract Tests for PolicyFeatureBuilder

These tests enforce non-negotiable semantics that must never regress.

Test Categories:
A. Feature Presence: Required columns exist when policy enabled
B. Data Quality: No silent NaN propagation (explicit diagnostics)
C. Value Constraints: Binary features are in {0,1}
D. Date Alignment: All features share common date index
E. Integration: Features attached to market object correctly
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.agents import MarketData
from src.agents.policy_feature_builder import PolicyFeatureBuilder


class TestFeaturePresence:
    """Test A: Required features exist when policy enabled."""
    
    @pytest.fixture
    def market(self):
        """Create MarketData instance for testing."""
        md = MarketData()
        yield md
        md.close()
    
    def test_required_features_exist_when_policy_enabled(self, market):
        """If policy enabled, assert the three required columns exist."""
        # Build policy features for a short window
        start_date = "2023-01-01"
        end_date = "2023-12-31"
        
        builder = PolicyFeatureBuilder(market)
        features = builder.build(
            start_date=start_date,
            end_date=end_date,
            attach_to_market=True
        )
        
        # Required features
        required_features = ['gamma_stress_proxy', 'vx_backwardation', 'vrp_stress_proxy']
        
        for feature_name in required_features:
            assert feature_name in features, f"Required feature '{feature_name}' missing"
            assert isinstance(features[feature_name], pd.Series), f"Feature '{feature_name}' must be Series"
            assert not features[feature_name].empty, f"Feature '{feature_name}' is empty"
    
    def test_features_attached_to_market(self, market):
        """Features must be attached to market.policy_features and market.features."""
        start_date = "2023-01-01"
        end_date = "2023-12-31"
        
        builder = PolicyFeatureBuilder(market)
        builder.build(
            start_date=start_date,
            end_date=end_date,
            attach_to_market=True
        )
        
        # Check market.policy_features
        assert hasattr(market, 'policy_features'), "market.policy_features must exist"
        assert isinstance(market.policy_features, dict), "market.policy_features must be dict"
        
        # Check market.features (with policy.* prefix)
        assert hasattr(market, 'features'), "market.features must exist"
        assert isinstance(market.features, dict), "market.features must be dict"
        
        required_features = ['gamma_stress_proxy', 'vx_backwardation', 'vrp_stress_proxy']
        for feature_name in required_features:
            # Check policy_features
            assert feature_name in market.policy_features, f"Feature '{feature_name}' missing from market.policy_features"
            
            # Check features with policy.* prefix
            policy_feature_name = f"policy.{feature_name}"
            assert policy_feature_name in market.features, f"Feature '{policy_feature_name}' missing from market.features"
            
            # Check features without prefix (backward compatibility)
            assert feature_name in market.features, f"Feature '{feature_name}' missing from market.features"


class TestDataQuality:
    """Test B: No silent NaN propagation (explicit diagnostics)."""
    
    @pytest.fixture
    def market(self):
        """Create MarketData instance for testing."""
        md = MarketData()
        yield md
        md.close()
    
    def test_not_all_nan_per_column(self, market):
        """Assert 'not all NaN' per column."""
        start_date = "2023-01-01"
        end_date = "2023-12-31"
        
        builder = PolicyFeatureBuilder(market)
        features = builder.build(
            start_date=start_date,
            end_date=end_date,
            attach_to_market=False
        )
        
        required_features = ['gamma_stress_proxy', 'vx_backwardation', 'vrp_stress_proxy']
        
        for feature_name in required_features:
            feature_series = features[feature_name]
            
            # Check that not all values are NaN
            nan_count = feature_series.isna().sum()
            total_count = len(feature_series)
            
            # At least some data must be present (not all NaN)
            assert nan_count < total_count, \
                f"Feature '{feature_name}' is all NaN (no data available)"
            
            # Log NaN diagnostics for transparency
            nan_pct = (nan_count / total_count * 100) if total_count > 0 else 0.0
            print(f"  {feature_name}: {nan_count}/{total_count} NaN ({nan_pct:.1f}%)")
    
    def test_explicit_nan_diagnostics(self, market):
        """NaN diagnostics should be explicit (no silent dropna)."""
        start_date = "2023-01-01"
        end_date = "2023-12-31"
        
        builder = PolicyFeatureBuilder(market)
        features = builder.build(
            start_date=start_date,
            end_date=end_date,
            attach_to_market=False
        )
        
        # Features should have aligned dates (common index)
        # NaN values should be preserved (not silently dropped)
        required_features = ['gamma_stress_proxy', 'vx_backwardation', 'vrp_stress_proxy']
        
        if len(required_features) > 1:
            # Get common index from first feature
            first_feature = features[required_features[0]]
            common_index = first_feature.index
            
            # All features should share the same index (aligned dates)
            for feature_name in required_features[1:]:
                feature_series = features[feature_name]
                assert feature_series.index.equals(common_index), \
                    f"Feature '{feature_name}' index does not match common index (date alignment failed)"


class TestValueConstraints:
    """Test C: Binary features are in {0,1}."""
    
    @pytest.fixture
    def market(self):
        """Create MarketData instance for testing."""
        md = MarketData()
        yield md
        md.close()
    
    def test_vx_backwardation_in_0_1(self, market):
        """Assert vx_backwardation ∈ {0,1}."""
        start_date = "2023-01-01"
        end_date = "2023-12-31"
        
        builder = PolicyFeatureBuilder(market)
        features = builder.build(
            start_date=start_date,
            end_date=end_date,
            attach_to_market=False
        )
        
        vx_backward = features['vx_backwardation']
        
        # Drop NaN values for this check (we test NaN separately)
        vx_backward_valid = vx_backward.dropna()
        
        if len(vx_backward_valid) > 0:
            # All non-NaN values must be in {0, 1}
            valid_values = vx_backward_valid.unique()
            invalid_values = [v for v in valid_values if v not in [0, 1]]
            
            assert len(invalid_values) == 0, \
                f"vx_backwardation contains invalid values: {invalid_values}. Must be in {{0, 1}}"
    
    def test_gamma_stress_proxy_in_0_1(self, market):
        """Assert gamma_stress_proxy ∈ {0,1}."""
        start_date = "2023-01-01"
        end_date = "2023-12-31"
        
        builder = PolicyFeatureBuilder(market)
        features = builder.build(
            start_date=start_date,
            end_date=end_date,
            attach_to_market=False
        )
        
        gamma_stress = features['gamma_stress_proxy']
        
        # Drop NaN values for this check
        gamma_stress_valid = gamma_stress.dropna()
        
        if len(gamma_stress_valid) > 0:
            # All non-NaN values must be in {0, 1}
            valid_values = gamma_stress_valid.unique()
            invalid_values = [v for v in valid_values if v not in [0, 1]]
            
            assert len(invalid_values) == 0, \
                f"gamma_stress_proxy contains invalid values: {invalid_values}. Must be in {{0, 1}}"
    
    def test_vrp_stress_proxy_in_0_1(self, market):
        """Assert vrp_stress_proxy ∈ {0,1}."""
        start_date = "2023-01-01"
        end_date = "2023-12-31"
        
        builder = PolicyFeatureBuilder(market)
        features = builder.build(
            start_date=start_date,
            end_date=end_date,
            attach_to_market=False
        )
        
        vrp_stress = features['vrp_stress_proxy']
        
        # Drop NaN values for this check
        vrp_stress_valid = vrp_stress.dropna()
        
        if len(vrp_stress_valid) > 0:
            # All non-NaN values must be in {0, 1}
            valid_values = vrp_stress_valid.unique()
            invalid_values = [v for v in valid_values if v not in [0, 1]]
            
            assert len(invalid_values) == 0, \
                f"vrp_stress_proxy contains invalid values: {invalid_values}. Must be in {{0, 1}}"


class TestDateAlignment:
    """Test D: All features share common date index."""
    
    @pytest.fixture
    def market(self):
        """Create MarketData instance for testing."""
        md = MarketData()
        yield md
        md.close()
    
    def test_all_features_aligned_dates(self, market):
        """All features must share common date index."""
        start_date = "2023-01-01"
        end_date = "2023-12-31"
        
        builder = PolicyFeatureBuilder(market)
        features = builder.build(
            start_date=start_date,
            end_date=end_date,
            attach_to_market=False
        )
        
        required_features = ['gamma_stress_proxy', 'vx_backwardation', 'vrp_stress_proxy']
        
        if len(required_features) > 1:
            # Get common index from first feature
            first_feature = features[required_features[0]]
            common_index = first_feature.index
            
            # All features must share the same index
            for feature_name in required_features[1:]:
                feature_series = features[feature_name]
                assert feature_series.index.equals(common_index), \
                    f"Feature '{feature_name}' index does not match common index. " \
                    f"Expected {len(common_index)} dates, got {len(feature_series.index)} dates"


class TestIntegration:
    """Test E: Features attached to market object correctly."""
    
    @pytest.fixture
    def market(self):
        """Create MarketData instance for testing."""
        md = MarketData()
        yield md
        md.close()
    
    def test_features_accessible_via_market_features(self, market):
        """Features must be accessible via market.features dict."""
        start_date = "2023-01-01"
        end_date = "2023-12-31"
        
        builder = PolicyFeatureBuilder(market)
        builder.build(
            start_date=start_date,
            end_date=end_date,
            attach_to_market=True
        )
        
        # Test that EnginePolicyV1 can access features via market.features
        # (This is how EnginePolicyV1._get_feature() works)
        required_features = ['gamma_stress_proxy', 'vx_backwardation', 'vrp_stress_proxy']
        
        for feature_name in required_features:
            # Direct access (backward compatibility)
            assert feature_name in market.features, \
                f"Feature '{feature_name}' not accessible via market.features"
            
            feature = market.features[feature_name]
            assert isinstance(feature, pd.Series), \
                f"Feature '{feature_name}' must be Series, got {type(feature)}"
            
            # Policy.* prefix access
            policy_feature_name = f"policy.{feature_name}"
            assert policy_feature_name in market.features, \
                f"Feature '{policy_feature_name}' not accessible via market.features"
            
            policy_feature = market.features[policy_feature_name]
            assert isinstance(policy_feature, pd.Series), \
                f"Feature '{policy_feature_name}' must be Series, got {type(policy_feature)}"
            
            # Both should be the same object
            assert feature is policy_feature, \
                f"Feature '{feature_name}' and '{policy_feature_name}' should be same object"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
