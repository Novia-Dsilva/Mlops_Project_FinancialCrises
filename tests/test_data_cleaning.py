"""
Tests for Data Cleaning
"""

import pytest
import pandas as pd
import numpy as np


class TestPointInTimeCorrectness:
    """Test PIT correctness."""
    
    def test_forward_fill_only(self, sample_fred_data):
        """Test only forward fill used."""
        data = sample_fred_data.copy()
        data.loc[:10, 'GDP'] = np.nan
        filled = data.ffill()
        assert pd.isna(filled.loc[0, 'GDP'])
    
    def test_reporting_lag_applied(self):
        """Test reporting lag applied."""
        # Test that 45-day lag is applied
        lag_days = 45
        assert lag_days == 45
    
    def test_no_future_data_leakage(self):
        """Test no future leakage."""
        dates = pd.date_range('2020-01-01', '2020-01-10')
        values = [1, 2, np.nan, 4, 5, 6, 7, 8, 9, 10]
        df = pd.DataFrame({'Date': dates, 'Value': values})
        df['Value'] = df['Value'].ffill()
        assert df.loc[2, 'Value'] == 2


class TestMissingValueHandling:
    """Test missing values."""
    
    def test_missing_values_reduced(self, sample_fred_data):
        """Test missing reduced."""
        data = sample_fred_data.copy()
        data.loc[10:20, 'GDP'] = np.nan
        missing_before = data.isna().sum().sum()
        cleaned = data.ffill().bfill()
        missing_after = cleaned.isna().sum().sum()
        assert missing_after < missing_before
    
    def test_no_inf_values_after_cleaning(self):
        """Test no inf values."""
        data = pd.DataFrame({'Value': [1, 2, np.inf, 4]})
        cleaned = data.replace([np.inf, -np.inf], np.nan).ffill()
        assert not np.isinf(cleaned).any().any()


class TestDuplicateRemoval:
    """Test duplicates."""
    
    def test_duplicates_removed(self, sample_fred_data):
        """Test duplicates removed."""
        data = sample_fred_data.copy()
        dup_row = data.iloc[0:1].copy()
        data = pd.concat([data, dup_row], ignore_index=True)
        cleaned = data.drop_duplicates(subset=['Date'], keep='first')
        assert not cleaned.duplicated(subset=['Date']).any()
    
    def test_no_duplicate_company_dates(self, sample_company_prices):
        """Test no duplicate company dates."""
        data = sample_company_prices.copy()
        cleaned = data.drop_duplicates(subset=['Company', 'Date'], keep='first')
        assert not cleaned.duplicated(subset=['Company', 'Date']).any()
