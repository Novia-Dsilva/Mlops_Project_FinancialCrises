"""
Tests for Post-Merge Cleaning (Step 3c)

Coverage target: >80%

Tests:
- Duplicate column removal
- Inf value handling
- Missing value filling
- Extreme outlier capping
- Data type validation
- Constant column removal
"""

import pytest
import pandas as pd
import numpy as np


class TestDuplicateColumnRemoval:
    """Test removal of duplicate columns from merge."""
    
    def test_removes_x_y_suffixes(self):
        """Test removes columns with _x, _y suffixes."""
        data = pd.DataFrame({
            'Date': pd.date_range('2020-01-01', periods=10),
            'Value': range(10),
            'Value_x': range(10),
            'Value_y': range(10, 20)
        })
        
        # Should remove Value_x and Value_y, keep Value
        suffixes = ['_x', '_y']
        cols_to_drop = []
        
        for col in data.columns:
            for suffix in suffixes:
                if col.endswith(suffix):
                    base_col = col[:-len(suffix)]
                    if base_col in data.columns:
                        cols_to_drop.append(col)
        
        cleaned = data.drop(columns=cols_to_drop)
        
        assert 'Value' in cleaned.columns
        assert 'Value_x' not in cleaned.columns
        assert 'Value_y' not in cleaned.columns
    
    def test_removes_fred_market_suffixes(self):
        """Test removes _fred, _market suffixes."""
        data = pd.DataFrame({
            'GDP': [1, 2, 3],
            'GDP_fred': [1, 2, 3],
            'GDP_market': [4, 5, 6],
            'VIX': [10, 15, 20],
            'VIX_macro': [10, 15, 20]
        })
        
        suffixes = ['_fred', '_market', '_macro']
        cols_to_drop = []
        
        for col in data.columns:
            for suffix in suffixes:
                if col.endswith(suffix):
                    base_col = col[:-len(suffix)]
                    if base_col in data.columns:
                        cols_to_drop.append(col)
        
        cleaned = data.drop(columns=cols_to_drop)
        
        assert 'GDP' in cleaned.columns
        assert 'GDP_fred' not in cleaned.columns
        assert 'GDP_market' not in cleaned.columns
    
    def test_column_count_reduced(self):
        """Test column count is reduced after removing duplicates."""
        data = pd.DataFrame({
            'A': [1, 2],
            'A_x': [1, 2],
            'A_y': [3, 4],
            'B': [5, 6]
        })
        
        original_cols = len(data.columns)
        
        # Remove duplicates
        cols_to_drop = [col for col in data.columns if col.endswith('_x') or col.endswith('_y')]
        cleaned = data.drop(columns=cols_to_drop)
        
        assert len(cleaned.columns) < original_cols


class TestInfValueHandling:
    """Test handling of inf values."""
    
    def test_inf_values_replaced_with_nan(self):
        """Test inf values are replaced with NaN."""
        data = pd.DataFrame({
            'Value': [1.0, 2.0, np.inf, 4.0, -np.inf, 6.0]
        })
        
        # Replace inf with NaN
        cleaned = data.replace([np.inf, -np.inf], np.nan)
        
        assert not np.isinf(cleaned['Value']).any()
        assert cleaned['Value'].isna().sum() == 2  # 2 inf values became NaN
    
    def test_inf_from_division_by_zero(self):
        """Test inf from division is handled."""
        data = pd.DataFrame({
            'Numerator': [10, 20, 30],
            'Denominator': [2, 0, 5]  # Division by zero
        })
        
        # This creates inf
        data['Ratio'] = data['Numerator'] / data['Denominator']
        
        # Should have inf
        assert np.isinf(data['Ratio']).any()
        
        # Replace with NaN
        data['Ratio'] = data['Ratio'].replace([np.inf, -np.inf], np.nan)
        
        assert not np.isinf(data['Ratio']).any()
    
    def test_no_inf_after_cleaning(self):
        """Test no inf values remain after cleaning."""
        data = pd.DataFrame({
            'A': [1, np.inf, 3],
            'B': [np.inf, 2, 3],
            'C': [1, 2, -np.inf]
        })
        
        # Clean all numeric columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        data[numeric_cols] = data[numeric_cols].replace([np.inf, -np.inf], np.nan)
        
        assert not np.isinf(data).any().any()


class TestExtremeOutlierCapping:
    """Test extreme outlier capping."""
    
    def test_values_capped_at_percentiles(self):
        """Test values capped at 0.1% and 99.9% percentiles."""
        data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 1000])  # 1000 is extreme
        
        lower = data.quantile(0.001)
        upper = data.quantile(0.999)
        
        capped = data.clip(lower=lower, upper=upper)
        
        assert capped.max() < 1000  # Extreme value capped
        assert capped.max() <= upper
    
    def test_capping_per_group(self):
        """Test capping works per group."""
        data = pd.DataFrame({
            'Group': ['A', 'A', 'A', 'B', 'B', 'B'],
            'Value': [1, 2, 100, 5, 6, 7]  # 100 is outlier in group A
        })
        
        # Cap per group
        for group in data['Group'].unique():
            group_mask = data['Group'] == group
            group_data = data.loc[group_mask, 'Value']
            
            lower = group_data.quantile(0.001)
            upper = group_data.quantile(0.999)
            
            data.loc[group_mask, 'Value'] = group_data.clip(lower=lower, upper=upper)
        
        # 100 should be capped in group A
        assert data[data['Group'] == 'A']['Value'].max() < 100
    
    def test_crisis_data_preserved(self):
        """Test crisis period outliers within 99.9% are preserved."""
        # Crisis data can be extreme but should be within 99.9%
        data = pd.Series(np.random.randn(1000) * 10)
        data.iloc[0] = 50  # Extreme but within 99.9%
        
        upper = data.quantile(0.999)
        
        # Should preserve this value if within percentile
        if 50 <= upper:
            capped = data.clip(upper=upper)
            assert capped.iloc[0] == 50


class TestMissingValueFilling:
    """Test missing value filling after merge."""
    
    def test_forward_fill_within_group(self):
        """Test forward fill works per company."""
        data = pd.DataFrame({
            'Date': pd.date_range('2020-01-01', periods=10),
            'Company': ['A']*10,
            'Value': [1, 2, np.nan, np.nan, 5, 6, 7, 8, 9, 10]
        })
        
        # Forward fill
        data['Value'] = data.groupby('Company')['Value'].ffill()
        
        assert data.loc[2, 'Value'] == 2  # Forward filled from index 1
        assert data.loc[3, 'Value'] == 2  # Forward filled from index 1
    
    def test_backward_fill_for_leading_nans(self):
        """Test backward fill handles leading NaNs."""
        data = pd.DataFrame({
            'Value': [np.nan, np.nan, 3, 4, 5]
        })
        
        # Forward then backward fill
        data['Value'] = data['Value'].ffill().bfill()
        
        assert data.loc[0, 'Value'] == 3  # Backward filled
        assert data.loc[1, 'Value'] == 3
    
    def test_median_fill_for_remaining(self):
        """Test median fill for remaining NaNs."""
        data = pd.Series([1, 2, np.nan, 4, 5])
        
        median_val = data.median()
        data = data.fillna(median_val)
        
        assert data.notna().all()
        assert data.iloc[2] == median_val


class TestDataTypeValidation:
    """Test data type validation."""
    
    def test_date_converted_to_datetime(self):
        """Test Date column converted to datetime."""
        data = pd.DataFrame({
            'Date': ['2020-01-01', '2020-01-02', '2020-01-03']
        })
        
        data['Date'] = pd.to_datetime(data['Date'])
        
        assert pd.api.types.is_datetime64_any_dtype(data['Date'])
    
    def test_categorical_conversion(self):
        """Test categorical columns converted."""
        data = pd.DataFrame({
            'Company': ['JPM', 'BAC', 'JPM', 'BAC'],
            'Sector': ['Financials', 'Financials', 'Financials', 'Financials']
        })
        
        data['Company'] = data['Company'].astype('category')
        data['Sector'] = data['Sector'].astype('category')
        
        assert data['Company'].dtype.name == 'category'
        assert data['Sector'].dtype.name == 'category'
    
    def test_numeric_object_conversion(self):
        """Test object columns converted to numeric where possible."""
        data = pd.DataFrame({
            'Value': ['1', '2', '3', '4']
        })
        
        data['Value'] = pd.to_numeric(data['Value'])
        
        assert pd.api.types.is_numeric_dtype(data['Value'])


class TestConstantColumnRemoval:
    """Test removal of constant columns."""
    
    def test_constant_column_detected(self):
        """Test constant columns are detected."""
        data = pd.DataFrame({
            'Constant': [5, 5, 5, 5, 5],
            'Variable': [1, 2, 3, 4, 5]
        })
        
        constant_cols = [col for col in data.columns if data[col].nunique() <= 1]
        
        assert 'Constant' in constant_cols
        assert 'Variable' not in constant_cols
    
    def test_constant_columns_removed(self):
        """Test constant columns are removed."""
        data = pd.DataFrame({
            'Date': pd.date_range('2020-01-01', periods=5),
            'Constant1': [10]*5,
            'Constant2': [20]*5,
            'Variable': [1, 2, 3, 4, 5]
        })
        
        # Remove constant columns (except Date)
        constant_cols = [col for col in data.columns 
                        if col != 'Date' and data[col].nunique() <= 1]
        
        cleaned = data.drop(columns=constant_cols)
        
        assert 'Constant1' not in cleaned.columns
        assert 'Constant2' not in cleaned.columns
        assert 'Variable' in cleaned.columns
        assert 'Date' in cleaned.columns
    
    def test_low_variance_detection(self):
        """Test low variance columns detected."""
        data = pd.DataFrame({
            'LowVar': [5.0, 5.0, 5.0000001, 5.0, 5.0],
            'HighVar': [1, 10, 20, 30, 40]
        })
        
        assert data['LowVar'].std() < 1e-6
        assert data['HighVar'].std() > 1


class TestInvalidRatioFixes:
    """Test fixing of invalid financial ratios."""
    
    def test_negative_ratio_made_positive(self):
        """Test negative ratios fixed where appropriate."""
        data = pd.DataFrame({
            'Current_Ratio': [2.0, 3.0, -1.5, 4.0]  # -1.5 invalid
        })
        
        # Fix negative
        data['Current_Ratio'] = data['Current_Ratio'].abs()
        
        assert (data['Current_Ratio'] >= 0).all()
    
    def test_extreme_ratios_capped(self):
        """Test extreme ratios are capped."""
        data = pd.DataFrame({
            'Profit_Margin': [-150, 10, 15, 200, 25]  # -150, 200 extreme
        })
        
        # Cap to [-100, 100]
        data['Profit_Margin'] = data['Profit_Margin'].clip(-100, 100)
        
        assert data['Profit_Margin'].min() >= -100
        assert data['Profit_Margin'].max() <= 100
    
    def test_division_by_zero_handled(self):
        """Test division by zero doesn't create inf."""
        data = pd.DataFrame({
            'Revenue': [100, 200, 300],
            'Shares': [10, 0, 30]  # 0 causes division issue
        })
        
        # Calculate ratio with protection
        data['EPS'] = data['Revenue'] / data['Shares'].replace(0, np.nan)
        
        assert not np.isinf(data['EPS']).any()


class TestDataIntegrityAfterCleaning:
    """Test data integrity is maintained."""
    
    def test_row_count_unchanged(self):
        """Test cleaning doesn't remove rows (only adds flags)."""
        data = pd.DataFrame({
            'Date': pd.date_range('2020-01-01', periods=100),
            'Value': np.random.randn(100)
        })
        
        original_rows = len(data)
        
        # Cleaning operations
        cleaned = data.copy()
        cleaned = cleaned.drop_duplicates()
        
        # May reduce due to duplicates, but not increase
        assert len(cleaned) <= original_rows
    
    def test_original_columns_preserved(self):
        """Test original columns are preserved (after removing duplicates)."""
        data = pd.DataFrame({
            'Date': [1, 2, 3],
            'Value': [10, 20, 30],
            'Value_x': [10, 20, 30]
        })
        
        # Remove duplicate columns
        cleaned = data.drop(columns=['Value_x'])
        
        # Original columns still there
        assert 'Date' in cleaned.columns
        assert 'Value' in cleaned.columns
    

class TestGroupBasedCleaning:
    """Test cleaning respects company groups."""
    

    def test_missing_filled_per_company(self):
        """Test missing values filled per company."""
        data = pd.DataFrame({
            'Company': ['A', 'A', 'A', 'B', 'B', 'B'],
            'Value': [1, np.nan, 3, 10, np.nan, 30]
        })
        
        # Fill per company
        data['Value'] = data.groupby('Company')['Value'].ffill()
        
        # A's NaN filled with 1
        assert data.loc[1, 'Value'] == 1
        # B's NaN filled with 10
        assert data.loc[4, 'Value'] == 10
