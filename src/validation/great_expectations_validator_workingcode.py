"""
Great Expectations Validator - FIXED for GE 0.18.x API
Validates FRED, Market, and Company data using correct ExpectationConfiguration
"""

import pandas as pd
import numpy as np
import great_expectations as gx
from great_expectations.core import ExpectationConfiguration
from typing import Dict, Tuple
import os
import json
from datetime import datetime


class GreatExpectationsValidator:
    """Validates data using Great Expectations 0.18.x API."""
    
    def __init__(self, context_root_dir: str = "great_expectations"):
        """Initialize Great Expectations context."""
        self.context_root_dir = context_root_dir
        
        if os.path.exists(context_root_dir):
            self.context = gx.get_context(context_root_dir=context_root_dir)
            print(f"✓ Loaded existing GE context from {context_root_dir}")
        else:
            self.context = gx.get_context()
            print(f"✓ Created new GE context")
        
        self._setup_datasource()
        self._created_assets = set()  # Track created assets
    
    def _setup_datasource(self):
        """Setup pandas datasource for runtime validation."""
        try:
            self.datasource = self.context.get_datasource("pandas_runtime")
            print("✓ Using existing pandas_runtime datasource")
        except:
            self.datasource = self.context.sources.add_pandas("pandas_runtime")
            print("✓ Created pandas_runtime datasource")
    
    def create_fred_expectations(self) -> str:
        """Create expectation suite for FRED macroeconomic data."""
        suite_name = "fred_macro_suite"
        
        # Always recreate the suite to ensure expectations are fresh
        try:
            self.context.delete_expectation_suite(suite_name)
            print(f"  ✓ Deleted existing suite: {suite_name}")
        except:
            pass
        
        suite = self.context.add_expectation_suite(suite_name)
        print(f"✓ Creating new suite: {suite_name}")
        
        # Define expectations using ExpectationConfiguration (GE 0.18.x API)
        expectations = [
            # Column existence checks
            ExpectationConfiguration(
                expectation_type="expect_column_to_exist",
                kwargs={"column": "GDP_Growth"}
            ),
            ExpectationConfiguration(
                expectation_type="expect_column_to_exist",
                kwargs={"column": "CPI_Inflation"}
            ),
            ExpectationConfiguration(
                expectation_type="expect_column_to_exist",
                kwargs={"column": "Unemployment_Rate"}
            ),
            ExpectationConfiguration(
                expectation_type="expect_column_to_exist",
                kwargs={"column": "Federal_Funds_Rate"}
            ),
            
            # Value range checks
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_be_between",
                kwargs={
                    "column": "GDP_Growth",
                    "min_value": -20,
                    "max_value": 20,
                    "mostly": 0.95
                }
            ),
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_be_between",
                kwargs={
                    "column": "CPI_Inflation",
                    "min_value": -5,
                    "max_value": 20,
                    "mostly": 0.95
                }
            ),
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_be_between",
                kwargs={
                    "column": "Unemployment_Rate",
                    "min_value": 0,
                    "max_value": 30,
                    "mostly": 0.95
                }
            ),
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_be_between",
                kwargs={
                    "column": "Federal_Funds_Rate",
                    "min_value": 0,
                    "max_value": 25,
                    "mostly": 0.95
                }
            ),
            
            # Completeness checks
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_not_be_null",
                kwargs={
                    "column": "GDP_Growth",
                    "mostly": 0.5  # Allow 50% missing (different frequencies)
                }
            ),
            
            # Row count check
            ExpectationConfiguration(
                expectation_type="expect_table_row_count_to_be_between",
                kwargs={
                    "min_value": 100,
                    "max_value": None
                }
            )
        ]
        
        # Add expectations to suite using add_expectation
        for exp_config in expectations:
            suite.add_expectation(expectation_configuration=exp_config)
        
        self.context.add_or_update_expectation_suite(expectation_suite=suite)
        print(f"✓ Created {len(expectations)} expectations for FRED data")
        
        return suite_name
    
    def create_market_expectations(self) -> str:
        """Create expectation suite for market data (VIX, S&P 500)."""
        suite_name = "market_data_suite"
        
        # Always recreate the suite to ensure expectations are fresh
        try:
            self.context.delete_expectation_suite(suite_name)
            print(f"  ✓ Deleted existing suite: {suite_name}")
        except:
            pass
        
        suite = self.context.add_expectation_suite(suite_name)
        print(f"✓ Creating new suite: {suite_name}")
        
        expectations = [
            # VIX checks
            ExpectationConfiguration(
                expectation_type="expect_column_to_exist",
                kwargs={"column": "VIX"}
            ),
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_be_between",
                kwargs={
                    "column": "VIX",
                    "min_value": 0,
                    "max_value": 100,
                    "mostly": 0.99
                }
            ),
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_not_be_null",
                kwargs={
                    "column": "VIX",
                    "mostly": 0.9
                }
            ),
            
            # SP500 checks
            ExpectationConfiguration(
                expectation_type="expect_column_to_exist",
                kwargs={"column": "SP500"}
            ),
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_be_between",
                kwargs={
                    "column": "SP500",
                    "min_value": 0,
                    "max_value": 10000,
                    "mostly": 0.99
                }
            ),
            
            # SP500_Return checks
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_be_between",
                kwargs={
                    "column": "SP500_Return",
                    "min_value": -20,
                    "max_value": 20,
                    "mostly": 0.95
                }
            ),
            
            # Row count
            ExpectationConfiguration(
                expectation_type="expect_table_row_count_to_be_between",
                kwargs={
                    "min_value": 1000,
                    "max_value": None
                }
            )
        ]
        
        # Add expectations to suite using add_expectation
        for exp_config in expectations:
            suite.add_expectation(expectation_configuration=exp_config)
        
        self.context.add_or_update_expectation_suite(expectation_suite=suite)
        print(f"✓ Created {len(expectations)} expectations for market data")
        
        return suite_name
    
    def create_company_expectations(self) -> str:
        """Create expectation suite for company financial data."""
        suite_name = "company_data_suite"
        
        # Always recreate the suite to ensure expectations are fresh
        try:
            self.context.delete_expectation_suite(suite_name)
            print(f"  ✓ Deleted existing suite: {suite_name}")
        except:
            pass
        
        suite = self.context.add_expectation_suite(suite_name)
        print(f"✓ Creating new suite: {suite_name}")
        
        expectations = [
            # Required columns
            ExpectationConfiguration(
                expectation_type="expect_column_to_exist",
                kwargs={"column": "Company"}
            ),
            ExpectationConfiguration(
                expectation_type="expect_column_to_exist",
                kwargs={"column": "Sector"}
            ),
            ExpectationConfiguration(
                expectation_type="expect_column_to_exist",
                kwargs={"column": "Stock_Price"}
            ),
            
            # Stock_Price checks
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_be_between",
                kwargs={
                    "column": "Stock_Price",
                    "min_value": 0,
                    "max_value": 10000,
                    "mostly": 0.99
                }
            ),
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_not_be_null",
                kwargs={
                    "column": "Stock_Price",
                    "mostly": 0.95
                }
            ),
            
            # Company not null
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_not_be_null",
                kwargs={"column": "Company"}
            ),
            
            # Sector validation
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_be_in_set",
                kwargs={
                    "column": "Sector",
                    "value_set": [
                        "Financials", "Technology", "Communication Services",
                        "Consumer Discretionary", "Consumer Staples", "Energy",
                        "Healthcare", "Industrials", "Materials"
                    ]
                }
            ),
            
            # Stock_Return range
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_be_between",
                kwargs={
                    "column": "Stock_Return",
                    "min_value": -50,
                    "max_value": 50,
                    "mostly": 0.95
                }
            ),
            
            # Row count
            ExpectationConfiguration(
                expectation_type="expect_table_row_count_to_be_between",
                kwargs={
                    "min_value": 1000,
                    "max_value": None
                }
            )
        ]
        
        # Add expectations to suite using add_expectation
        for exp_config in expectations:
            suite.add_expectation(expectation_configuration=exp_config)
        
        self.context.add_or_update_expectation_suite(expectation_suite=suite)
        print(f"✓ Created {len(expectations)} expectations for company data")
        
        return suite_name
    
    def validate_dataframe(
        self,
        df: pd.DataFrame,
        suite_name: str,
        data_asset_name: str
    ) -> Tuple[bool, Dict]:
        """Validate a DataFrame against an expectation suite."""
        print(f"\n{'='*70}")
        print(f"🔍 VALIDATING {data_asset_name.upper()}")
        print(f"{'='*70}")
        
        try:
            # Create unique asset name with timestamp to avoid conflicts
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            unique_asset_name = f"{data_asset_name}_{timestamp}"
            
            # Create new data asset with unique name
            data_asset = self.datasource.add_dataframe_asset(name=unique_asset_name)
            print(f"  ✓ Created data asset: {unique_asset_name}")
            
            # Create batch request
            batch_request = data_asset.build_batch_request(dataframe=df)
            
            # Get validator
            validator = self.context.get_validator(
                batch_request=batch_request,
                expectation_suite_name=suite_name
            )
            
            # Run validation
            results = validator.validate()
            
            # Parse results with safe access
            is_valid = results.success
            stats = results.statistics if hasattr(results, 'statistics') else {}
            
            # Safely extract statistics with defaults
            total_expectations = stats.get('evaluated_expectations', 0)
            successful = stats.get('successful_expectations', 0)
            failed = stats.get('unsuccessful_expectations', 0)
            success_percent = stats.get('success_percent', 0.0)
            
            # Handle None values
            if success_percent is None:
                success_percent = 0.0
            
            print(f"\nValidation Results:")
            print(f"  Total expectations: {total_expectations}")
            print(f"  Successful: {successful}")
            print(f"  Failed: {failed}")
            print(f"  Success rate: {success_percent:.1f}%")
            
            # Collect failed expectations
            failed_expectations = []
            if hasattr(results, 'results'):
                for result in results.results:
                    if not result.success:
                        exp_type = result.expectation_config.expectation_type
                        kwargs = result.expectation_config.kwargs
                        failed_expectations.append({
                            'expectation': exp_type,
                            'column': kwargs.get('column', 'N/A'),
                            'details': str(result.result)
                        })
            
            # Log failures
            if failed_expectations:
                print(f"\n❌ {len(failed_expectations)} FAILED EXPECTATIONS:")
                for i, failure in enumerate(failed_expectations[:5], 1):
                    print(f"  {i}. {failure['expectation']}")
                    print(f"     Column: {failure['column']}")
            
            # Create report
            report = {
                'is_valid': is_valid,
                'data_asset': data_asset_name,
                'suite_name': suite_name,
                'total_expectations': total_expectations,
                'successful': successful,
                'failed': failed,
                'success_rate': success_percent,
                'failed_expectations': failed_expectations,
                'timestamp': datetime.now().isoformat()
            }
            
            if is_valid:
                print(f"\n✅ VALIDATION PASSED")
            else:
                print(f"\n❌ VALIDATION FAILED")
            
            print(f"{'='*70}\n")
            
            # Save report
            self._save_report(report, data_asset_name)
            
            return is_valid, report
        
        except Exception as e:
            print(f"❌ Validation error: {str(e)}")
            import traceback
            traceback.print_exc()
            
            return False, {
                'is_valid': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat(),
                'success_rate': 0.0,
                'total_expectations': 0,
                'successful': 0,
                'failed': 0
            }
    
    def _save_report(self, report: Dict, data_asset_name: str):
        """Save validation report to disk."""
        os.makedirs('data/validation_reports', exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"data/validation_reports/{data_asset_name}_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"💾 Validation report saved: {filename}")


# ============================================================================
# CONVENIENCE FUNCTIONS FOR ALL THREE DATA SOURCES
# ============================================================================

def validate_fred_with_ge(df: pd.DataFrame) -> Tuple[bool, Dict]:
    """Validate FRED data using Great Expectations."""
    validator = GreatExpectationsValidator()
    suite_name = validator.create_fred_expectations()
    is_valid, report = validator.validate_dataframe(
        df, 
        suite_name=suite_name,
        data_asset_name="fred_macro_data"
    )
    return is_valid, report


def validate_market_with_ge(df: pd.DataFrame) -> Tuple[bool, Dict]:
    """Validate market data using Great Expectations."""
    validator = GreatExpectationsValidator()
    suite_name = validator.create_market_expectations()
    is_valid, report = validator.validate_dataframe(
        df,
        suite_name=suite_name,
        data_asset_name="market_data"
    )
    return is_valid, report


def validate_company_with_ge(df: pd.DataFrame) -> Tuple[bool, Dict]:
    """Validate company data using Great Expectations."""
    validator = GreatExpectationsValidator()
    suite_name = validator.create_company_expectations()
    is_valid, report = validator.validate_dataframe(
        df,
        suite_name=suite_name,
        data_asset_name="company_data"
    )
    return is_valid, report


# ============================================================================
# TEST THE VALIDATOR
# ============================================================================

if __name__ == "__main__":
    """Test the validator with sample data."""
    print("\n" + "="*70)
    print("GREAT EXPECTATIONS VALIDATOR - TEST MODE")
    print("="*70)
    
    # Create sample FRED data
    dates = pd.date_range('2020-01-01', '2024-12-31', freq='D')
    fred_data = pd.DataFrame({
        'GDP_Growth': np.random.randn(len(dates)) * 2 + 2,
        'CPI_Inflation': np.random.randn(len(dates)) * 1 + 3,
        'Unemployment_Rate': np.random.randn(len(dates)) * 1 + 5,
        'Federal_Funds_Rate': np.random.randn(len(dates)) * 0.5 + 2,
        'Yield_Curve_Spread': np.random.randn(len(dates)) * 0.5 + 0.5
    }, index=dates)
    
    # Validate
    is_valid, report = validate_fred_with_ge(fred_data)
    
    print(f"\n{'='*70}")
    print(f"TEST RESULT: {'✅ PASSED' if is_valid else '❌ FAILED'}")
    print(f"Success Rate: {report.get('success_rate', 0):.1f}%")
    print(f"Total Expectations: {report.get('total_expectations', 0)}")
    print(f"Failed: {report.get('failed', 0)}")
    if 'error' in report:
        print(f"Error: {report['error']}")
    print(f"{'='*70}")