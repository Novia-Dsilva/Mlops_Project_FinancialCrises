"""
Tests for Validation Checkpoints

Coverage target: >75%
"""

import pytest
import pandas as pd
import numpy as np


class TestSchemaValidation:
    """Test schema validation."""
    
    def test_required_columns_present(self, sample_fred_data):
        """Test required columns exist."""
        required = ['Date', 'GDP', 'CPI', 'Unemployment_Rate']
        
        for col in required:
            assert col in sample_fred_data.columns
    
    def test_date_column_is_datetime(self, sample_fred_data):
        """Test Date column is datetime type."""
        assert pd.api.types.is_datetime64_any_dtype(sample_fred_data['Date'])
    
    def test_numeric_columns_are_numeric(self, sample_fred_data):
        """Test numeric columns have correct types."""
        numeric_cols = ['GDP', 'CPI', 'Unemployment_Rate']
        
        for col in numeric_cols:
            assert pd.api.types.is_numeric_dtype(sample_fred_data[col])


class TestDataQualityChecks:
    """Test data quality validation."""
    
    def test_no_inf_values(self, sample_fred_data):
        """Test no inf values in data."""
        numeric_cols = sample_fred_data.select_dtypes(include=[np.number]).columns
        
        assert not np.isinf(sample_fred_data[numeric_cols]).any().any()
    
    def test_missing_values_threshold(self, sample_fred_data):
        """Test missing values below threshold."""
        missing_pct = (sample_fred_data.isna().sum().sum() / sample_fred_data.size) * 100
        
        # After cleaning, should be < 5%
        assert missing_pct < 5
    
    def test_no_duplicate_rows(self, sample_fred_data):
        """Test no duplicate rows."""
        duplicates = sample_fred_data.duplicated(subset=['Date'])
        
        assert not duplicates.any()


class TestValueRanges:
    """Test value range validation."""
    
    def test_vix_in_range(self, sample_market_data):
        """Test VIX is in valid range."""
        assert (sample_market_data['VIX'] >= 5).all()
        assert (sample_market_data['VIX'] <= 100).all()
    
    def test_unemployment_in_range(self, sample_fred_data):
        """Test unemployment rate in valid range."""
        assert (sample_fred_data['Unemployment_Rate'] >= 0).all()
        assert (sample_fred_data['Unemployment_Rate'] <= 30).all()
    
    def test_stock_price_positive(self, sample_company_prices):
        """Test stock prices are positive."""
        assert (sample_company_prices['Stock_Price'] > 0).all()