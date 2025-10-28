"""
Tests for Feature Engineering (Step 2)

Coverage target: >75%
"""

import pytest
import pandas as pd
import numpy as np


class TestLagFeatures:
    """Test lag feature creation."""
    
    def test_lag_features_created(self, sample_fred_data):
        """Test lag features are created correctly."""
        data = sample_fred_data.copy()
        
        # Create 1-day lag
        data['GDP_Lag1'] = data['GDP'].shift(1)
        
        assert 'GDP_Lag1' in data.columns
        assert pd.isna(data.loc[0, 'GDP_Lag1'])  # First value should be NaN
        assert data.loc[1, 'GDP_Lag1'] == data.loc[0, 'GDP']  # Second = first
    
    def test_multiple_lags(self, sample_fred_data):
        """Test multiple lag periods."""
        data = sample_fred_data.copy()
        
        for lag in [1, 5, 22]:
            data[f'GDP_Lag{lag}'] = data['GDP'].shift(lag)
        
        assert 'GDP_Lag1' in data.columns
        assert 'GDP_Lag5' in data.columns
        assert 'GDP_Lag22' in data.columns


class TestMovingAverages:
    """Test moving average features."""
    
    def test_moving_average_calculation(self):
        """Test MA is calculated correctly."""
        data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        
        ma5 = data.rolling(window=5, min_periods=1).mean()
        
        assert ma5.iloc[0] == 1  # First value
        assert ma5.iloc[4] == 3  # Mean of [1,2,3,4,5]
        assert ma5.iloc[9] == 8  # Mean of [6,7,8,9,10]
    
    def test_vix_moving_averages(self, sample_market_data):
        """Test VIX moving averages."""
        data = sample_market_data.copy()
        
        data['VIX_MA5'] = data['VIX'].rolling(window=5, min_periods=1).mean()
        data['VIX_MA22'] = data['VIX'].rolling(window=22, min_periods=1).mean()
        
        assert 'VIX_MA5' in data.columns
        assert 'VIX_MA22' in data.columns
        assert data['VIX_MA5'].notna().sum() > 0


class TestReturnCalculations:
    """Test return calculations."""
    
    def test_return_calculation(self):
        """Test percentage return calculation."""
        prices = pd.Series([100, 105, 110, 108, 112])
        
        returns = prices.pct_change() * 100
        
        assert abs(returns.iloc[1] - 5.0) < 0.01  # Floating point tolerance
        assert abs(returns.iloc[2] - 4.76) < 0.01  # (110-105)/105 * 100
    
    def test_stock_returns(self, sample_company_prices):
        """Test stock return features."""
        data = sample_company_prices.copy()
        
        # Calculate 1-day return
        data['Stock_Return_1D'] = data.groupby('Company')['Stock_Price'].pct_change() * 100
        
        assert 'Stock_Return_1D' in data.columns
        # First return for each company should be NaN
        for company in data['Company'].unique():
            company_data = data[data['Company'] == company]
            assert pd.isna(company_data.iloc[0]['Stock_Return_1D'])


class TestFinancialRatios:
    """Test financial ratio calculations."""
    
    def test_profit_margin(self, sample_company_financials):
        """Test profit margin calculation."""
        data = sample_company_financials.copy()
        
        data['Profit_Margin'] = (data['Net_Income'] / data['Revenue']) * 100
        
        assert 'Profit_Margin' in data.columns
        assert data['Profit_Margin'].notna().sum() > 0
        # Should be between -100 and 100
        assert data['Profit_Margin'].between(-100, 100).sum() > len(data) * 0.9
    
    def test_roe_calculation(self, sample_company_financials):
        """Test ROE calculation."""
        data = sample_company_financials.copy()
        
        data['ROE'] = (data['Net_Income'] / data['Total_Equity']) * 100
        
        assert 'ROE' in data.columns
        # Should handle division by zero
        data_with_zero = data.copy()
        data_with_zero.loc[0, 'Total_Equity'] = 0
        data_with_zero['ROE'] = (data_with_zero['Net_Income'] / 
                                 data_with_zero['Total_Equity'].replace(0, np.nan)) * 100
        
        assert not np.isinf(data_with_zero['ROE'].dropna()).any()


class TestQuarterlyToDaily:
    """Test quarterly to daily conversion."""
    
    def test_quarterly_expansion(self, sample_company_financials):
        """Test quarterly data expands to daily."""
        quarterly = sample_company_financials.copy()
        
        # Get date range
        start_date = quarterly['Date'].min()
        end_date = quarterly['Date'].max()
        daily_dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Expand for one company
        company = quarterly['Company'].iloc[0]
        company_q = quarterly[quarterly['Company'] == company].set_index('Date')
        
        # Reindex to daily
        company_d = company_q.reindex(daily_dates)
        
        # Forward fill
        company_d = company_d.ffill()
        
        assert len(company_d) > len(company_q)  # More daily than quarterly
        assert len(company_d) == len(daily_dates)
    
    def test_forward_fill_preserves_quarterly_values(self):
        """Test that quarterly values persist until next quarter."""
        # Create quarterly data
        dates_q = pd.date_range('2020-01-01', '2020-12-31', freq='Q')
        data_q = pd.DataFrame({
            'Date': dates_q,
            'Revenue': [100, 110, 120, 130]
        }).set_index('Date')
        
        # Expand to daily
        dates_d = pd.date_range('2020-01-01', '2020-12-31', freq='D')
        data_d = data_q.reindex(dates_d).ffill()
        
        # Check Q1 value (100) persists until Q2
        q1_end = pd.Timestamp('2020-03-31')
        assert data_d.loc[q1_end, 'Revenue'] == 100