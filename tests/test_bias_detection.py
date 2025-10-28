"""
Tests for Bias Detection (Step 4)

Coverage target: >75%
"""

import pytest
import pandas as pd
import numpy as np


class TestDataSlicing:
    """Test data slicing functionality."""
    
    def test_slice_by_company(self, sample_company_prices):
        """Test slicing by company."""
        companies = sample_company_prices['Company'].unique()
        
        slices = {}
        for company in companies:
            slices[company] = sample_company_prices[sample_company_prices['Company'] == company]
        
        assert len(slices) == len(companies)
        
        # Each slice should have only one company
        for company, slice_df in slices.items():
            assert slice_df['Company'].nunique() == 1
    
    def test_slice_by_sector(self, sample_company_prices):
        """Test slicing by sector."""
        sectors = sample_company_prices['Sector'].unique()
        
        slices = {}
        for sector in sectors:
            slices[sector] = sample_company_prices[sample_company_prices['Sector'] == sector]
        
        assert len(slices) == len(sectors)
    
    def test_temporal_slicing(self, sample_merged_data):
        """Test temporal slicing."""
        data = sample_merged_data.copy()
        data['Year'] = data['Date'].dt.year
        
        # Create time slices
        pre_2020 = data[data['Year'] < 2020]
        post_2020 = data[data['Year'] >= 2020]
        
        assert len(pre_2020) + len(post_2020) <= len(data)


class TestRepresentationBias:
    """Test representation bias detection."""
    
    def test_detect_underrepresentation(self):
        """Test detection of underrepresented groups."""
        # Create imbalanced data
        data = pd.DataFrame({
            'Company': ['A']*100 + ['B']*100 + ['C']*10,  # C is underrepresented
            'Value': np.random.randn(210)
        })
        
        company_counts = data['Company'].value_counts()
        expected = len(data) / 3  # 70 per company
        
        # C should be flagged
        c_count = company_counts['C']
        deviation = (c_count - expected) / expected
        
        assert deviation < -0.3  # More than 30% below expected
    
    def test_balanced_representation(self, sample_company_prices):
        """Test balanced data is not flagged."""
        company_counts = sample_company_prices['Company'].value_counts()
        
        # Check if reasonably balanced
        max_count = company_counts.max()
        min_count = company_counts.min()
        ratio = max_count / min_count
        
        # Should be reasonably balanced (depends on test data)
        assert ratio < 10  # Not more than 10x imbalance


class TestDistributionComparison:
    """Test distribution comparison across slices."""
    
    def test_ks_test_detects_different_distributions(self):
        """Test KS test detects distribution differences."""
        from scipy.stats import ks_2samp
        
        # Two different distributions
        dist1 = np.random.normal(0, 1, 1000)
        dist2 = np.random.normal(5, 1, 1000)  # Different mean
        
        ks_stat, p_value = ks_2samp(dist1, dist2)
        
        assert p_value < 0.05  # Significantly different
    
    def test_ks_test_same_distribution(self):
        """Test KS test doesn't flag similar distributions."""
        from scipy.stats import ks_2samp
        
        # Same distribution
        dist1 = np.random.normal(0, 1, 1000)
        dist2 = np.random.normal(0, 1, 1000)
        
        ks_stat, p_value = ks_2samp(dist1, dist2)
        
        assert p_value > 0.05  # Not significantly different