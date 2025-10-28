"""
Tests for Data Collection (Step 0)

Coverage target: >80%
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, Mock
from pathlib import Path


# ============================================================================
# TESTS FOR DATA COLLECTION
# ============================================================================

class TestFREDDataCollection:
    """Test FRED data fetching."""
    
    def test_fred_data_structure(self, sample_fred_data):
        """Test FRED data has correct structure."""
        assert 'Date' in sample_fred_data.columns
        assert 'GDP' in sample_fred_data.columns
        assert 'CPI' in sample_fred_data.columns
        assert len(sample_fred_data) > 0
    
    def test_fred_data_types(self, sample_fred_data):
        """Test FRED data types are correct."""
        assert pd.api.types.is_datetime64_any_dtype(sample_fred_data['Date'])
        assert pd.api.types.is_numeric_dtype(sample_fred_data['GDP'])
        assert pd.api.types.is_numeric_dtype(sample_fred_data['CPI'])
    
    def test_fred_no_nulls_in_date(self, sample_fred_data):
        """Test Date column has no nulls."""
        assert sample_fred_data['Date'].notna().all()
    
    def test_fred_date_range(self, sample_fred_data):
        """Test date range is reasonable."""
        date_range = (sample_fred_data['Date'].max() - sample_fred_data['Date'].min()).days
        assert date_range > 300  # At least ~1 year


class TestMarketDataCollection:
    """Test market data fetching."""
    
    def test_market_data_structure(self, sample_market_data):
        """Test market data structure."""
        assert 'Date' in sample_market_data.columns
        assert 'VIX' in sample_market_data.columns
        assert 'SP500' in sample_market_data.columns
    
    def test_vix_positive(self, sample_market_data):
        """Test VIX values are positive."""
        assert (sample_market_data['VIX'] > 0).all()
    
    def test_sp500_reasonable_range(self, sample_market_data):
        """Test S&P 500 in reasonable range."""
        assert sample_market_data['SP500'].min() > 1000
        assert sample_market_data['SP500'].max() < 10000


class TestCompanyDataCollection:
    """Test company price data fetching."""
    
    def test_company_data_structure(self, sample_company_prices):
        """Test company data structure."""
        assert 'Date' in sample_company_prices.columns
        assert 'Company' in sample_company_prices.columns
        assert 'Stock_Price' in sample_company_prices.columns
        assert 'Sector' in sample_company_prices.columns
    
    def test_multiple_companies(self, sample_company_prices):
        """Test data contains multiple companies."""
        assert sample_company_prices['Company'].nunique() >= 2
    
    def test_stock_price_positive(self, sample_company_prices):
        """Test stock prices are positive."""
        assert (sample_company_prices['Stock_Price'] > 0).all()
    
    def test_volume_positive(self, sample_company_prices):
        """Test trading volume is positive."""
        assert (sample_company_prices['Volume'] > 0).all()
    
    def test_no_duplicate_company_dates(self, sample_company_prices):
        """Test no duplicate (Company, Date) pairs."""
        duplicates = sample_company_prices.duplicated(subset=['Company', 'Date'])
        assert not duplicates.any()


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestDataCollectionIntegration:
    """Integration tests for complete data collection."""
    
