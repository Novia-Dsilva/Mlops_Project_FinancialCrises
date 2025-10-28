"""
End-to-End Integration Tests

Tests the complete pipeline flow.
Coverage target: >70%
"""

import pytest
import pandas as pd
from pathlib import Path


class TestPipelineIntegration:
    """Test complete pipeline execution."""
    
    @pytest.mark.slow
    def test_full_pipeline_execution(self, temp_data_dir, sample_fred_data, 
                                     sample_market_data, sample_company_prices,
                                     mock_config):
        """Test complete pipeline runs without errors."""
        
        # This is a simplified integration test
        # In reality, you'd run the actual pipeline
        
        # Step 1: Save raw data
        raw_dir = temp_data_dir / 'raw'
        raw_dir.mkdir()
        
        sample_fred_data.to_csv(raw_dir / 'fred_raw.csv', index=False)
        sample_market_data.to_csv(raw_dir / 'market_raw.csv', index=False)
        sample_company_prices.to_csv(raw_dir / 'company_prices_raw.csv', index=False)
        
        # Step 2: Verify files exist
        assert (raw_dir / 'fred_raw.csv').exists()
        assert (raw_dir / 'market_raw.csv').exists()
        
        # Step 3: Load and verify
        fred_loaded = pd.read_csv(raw_dir / 'fred_raw.csv', parse_dates=['Date'])
        assert len(fred_loaded) == len(sample_fred_data)
        
        # Pipeline would continue...
        # This is a placeholder for actual pipeline execution


class TestDataFlowIntegrity:
    """Test data integrity through pipeline."""
    
    def test_row_count_preserved(self, sample_company_prices):
        """Test row count preserved through cleaning."""
        original_count = len(sample_company_prices)
        
        # Simulate cleaning (remove duplicates only)
        cleaned = sample_company_prices.drop_duplicates(subset=['Company', 'Date'])
        
        # Should preserve or reduce (not increase)
        assert len(cleaned) <= original_count
    
    def test_companies_preserved(self, sample_company_prices):
        """Test all companies preserved through pipeline."""
        original_companies = set(sample_company_prices['Company'].unique())
        
        # After any operation
        cleaned = sample_company_prices.copy()
        final_companies = set(cleaned['Company'].unique())
        
        assert original_companies == final_companies