"""
Tests for Data Merging
"""

import pytest
import pandas as pd
import numpy as np


class TestMergeOperations:
    """Test data merging logic."""
    
    def test_fred_market_merge(self, sample_fred_data, sample_market_data):
        """Test FRED + Market merge."""
        merged = pd.merge(
            sample_fred_data,
            sample_market_data,
            on='Date',
            how='outer'
        )
        
        assert 'GDP' in merged.columns
        assert 'VIX' in merged.columns
    
    def test_company_merge(self, sample_fred_data, sample_company_prices):
        """Test macro + company merge."""
        merged = pd.merge(
            sample_company_prices,
            sample_fred_data,
            on='Date',
            how='left'
        )
        
        assert 'GDP' in merged.columns
        assert 'Company' in merged.columns
        assert len(merged) == len(sample_company_prices)
    
    def test_no_data_loss_in_merge(self, sample_company_prices, sample_fred_data):
        """Test merge doesn't lose data."""
        original_companies = sample_company_prices['Company'].nunique()
        
        merged = pd.merge(
            sample_company_prices,
            sample_fred_data,
            on='Date',
            how='left'
        )
        
        merged_companies = merged['Company'].nunique()
        assert merged_companies == original_companies
    
    def test_merge_handles_missing_dates(self):
        """Test merge handles date misalignment."""
        df1 = pd.DataFrame({
            'Date': pd.date_range('2020-01-01', '2020-01-10'),
            'Value1': range(10)
        })
        
        df2 = pd.DataFrame({
            'Date': pd.date_range('2020-01-05', '2020-01-15'),
            'Value2': range(11)
        })
        
        merged = pd.merge(df1, df2, on='Date', how='outer')
        assert len(merged) == 15
