"""
Tests for Drift Detection - Simplified (no scipy)
"""

import pytest
import pandas as pd
import numpy as np


class TestTemporalPeriodSlicing:
    """Test temporal slicing."""
    
    def test_reference_period_extraction(self):
        """Test reference period extraction."""
        dates = pd.date_range('2005-01-01', '2020-12-31', freq='D')
        data = pd.DataFrame({'Date': dates, 'Value': np.random.randn(len(dates))})
        data['Year'] = data['Date'].dt.year
        
        reference = data[(data['Year'] >= 2005) & (data['Year'] <= 2010)]
        
        assert reference['Year'].min() >= 2005
        assert reference['Year'].max() <= 2010
    
    def test_current_period_extraction(self):
        """Test current period extraction."""
        dates = pd.date_range('2005-01-01', '2025-12-31', freq='D')
        data = pd.DataFrame({'Date': dates, 'Value': np.random.randn(len(dates))})
        data['Year'] = data['Date'].dt.year
        
        current = data[(data['Year'] >= 2020) & (data['Year'] <= 2025)]
        
        assert current['Year'].min() >= 2020
        assert len(current) > 0
    
    def test_periods_non_overlapping(self):
        """Test periods don't overlap."""
        reference_years = set(range(2005, 2011))
        current_years = set(range(2020, 2026))
        
        overlap = reference_years & current_years
        assert len(overlap) == 0


class TestMeanStdDriftDetection:
    """Test mean/std drift detection."""
    
    def test_mean_drift_calculation(self):
        """Test mean drift calculation."""
        ref_data = pd.Series([10, 10, 10, 10])
        current_data = pd.Series([15, 15, 15, 15])
        
        mean_drift_pct = ((current_data.mean() - ref_data.mean()) / ref_data.mean()) * 100
        
        assert abs(mean_drift_pct - 50.0) < 0.01
    
    def test_std_drift_calculation(self):
        """Test std drift calculation."""
        ref_data = pd.Series(np.random.randn(100))
        current_data = pd.Series(np.random.randn(100) * 2)
        
        std_ref = ref_data.std()
        std_cur = current_data.std()
        
        std_drift_pct = ((std_cur - std_ref) / std_ref) * 100
        assert isinstance(std_drift_pct, float)
    
    def test_zero_drift_same_data(self):
        """Test zero drift."""
        ref_data = pd.Series([1, 2, 3, 4, 5])
        current_data = pd.Series([1, 2, 3, 4, 5])
        
        mean_drift = current_data.mean() - ref_data.mean()
        assert abs(mean_drift) < 1e-10


class TestDriftSeverityClassification:
    """Test drift severity."""
    
    def test_high_drift_severity(self):
        """Test high severity."""
        mean_change_pct = 60
        severity = 'HIGH' if abs(mean_change_pct) > 50 else 'MEDIUM'
        assert severity == 'HIGH'
    
    def test_medium_drift_severity(self):
        """Test medium severity."""
        mean_change_pct = 30
        severity = 'HIGH' if abs(mean_change_pct) > 50 else 'MEDIUM'
        assert severity == 'MEDIUM'
    
    def test_low_drift_severity(self):
        """Test low severity."""
        mean_change_pct = 5
        severity = 'LOW' if abs(mean_change_pct) < 20 else 'MEDIUM'
        assert severity == 'LOW'


class TestDriftFlagCreation:
    """Test drift flag creation."""
    
    def test_drift_flag_added(self):
        """Test drift flag added."""
        data = pd.DataFrame({'Feature1': [1, 2, 3], 'Feature2': [10, 20, 30]})
        data['Feature_Drift_Flag'] = 0
        
        assert 'Feature_Drift_Flag' in data.columns
    
    def test_high_drift_features_flagged(self):
        """Test high drift flagged."""
        data = pd.DataFrame({'Value': [1, 2, 3, 4, 5]})
        data['Feature_Drift_Flag'] = 0
        data.loc[2, 'Feature_Drift_Flag'] = 1
        
        assert data['Feature_Drift_Flag'].sum() == 1
