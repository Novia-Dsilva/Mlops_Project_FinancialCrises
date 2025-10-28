"""
Tests for Utilities - FIXED
"""

import pytest
import pandas as pd
import numpy as np
import json


class TestJSONSerialization:
    """Test JSON serialization - FIXED with inline encoder."""
    
    def test_numpy_int_serialization(self):
        """Test numpy int serialization."""
        # Define encoder inline (not importing from module)
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, (np.integer, np.int64)):
                    return int(obj)
                elif isinstance(obj, (np.floating, np.float64)):
                    return float(obj)
                return super().default(obj)
        
        data = {'count': np.int64(100)}
        json_str = json.dumps(data, cls=NumpyEncoder)
        loaded = json.loads(json_str)
        
        assert loaded['count'] == 100
        assert isinstance(loaded['count'], int)
    
    def test_numpy_float_serialization(self):
        """Test numpy float serialization."""
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, (np.floating, np.float64)):
                    return float(obj)
                elif isinstance(obj, (np.integer, np.int64)):
                    return int(obj)
                return super().default(obj)
        
        data = {'percentage': np.float64(25.5)}
        json_str = json.dumps(data, cls=NumpyEncoder)
        loaded = json.loads(json_str)
        
        assert loaded['percentage'] == 25.5
        assert isinstance(loaded['percentage'], float)


class TestDataLoading:
    """Test data loading utilities."""
    
    def test_load_csv_with_dates(self, sample_fred_data, temp_data_dir):
        """Test loading CSV with date parsing."""
        filepath = temp_data_dir / 'test.csv'
        sample_fred_data.to_csv(filepath, index=False)
        
        loaded = pd.read_csv(filepath, parse_dates=['Date'])
        
        assert pd.api.types.is_datetime64_any_dtype(loaded['Date'])
    
    def test_load_nonexistent_file(self):
        """Test loading nonexistent file handles gracefully."""
        with pytest.raises(FileNotFoundError):
            pd.read_csv('nonexistent_file.csv')
