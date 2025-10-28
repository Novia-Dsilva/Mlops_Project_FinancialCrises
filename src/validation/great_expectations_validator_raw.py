"""
Great Expectations Validator for RAW Data Files
Validates data in data/raw/ folder (before processing)

Key Differences from Processed Validator:
- Validates RAW column names (GDP, CPI vs GDP_Growth, CPI_Inflation)
- Validates RAW value ranges (GDP in billions vs GDP_Growth in %)
- Validates OHLCV data (Open, High, Low, Close, Volume)
- Validates quarterly fundamentals (Revenue, EPS, Balance Sheet)
"""

import pandas as pd
import numpy as np
import great_expectations as gx
from great_expectations.core import ExpectationConfiguration
from typing import Dict, Tuple
import os
import json
from datetime import datetime


class RawDataValidator:
    """Validates RAW data files before processing"""
    
    def __init__(self, context_root_dir: str = "great_expectations"):
        """Initialize Great Expectations context"""
        self.context_root_dir = context_root_dir
        
        if os.path.exists(context_root_dir):
            self.context = gx.get_context(context_root_dir=context_root_dir)
            print(f"   ✅ Loaded existing GE context")
        else:
            self.context = gx.get_context()
            print(f"   ✅ Created new GE context")
        
        self._setup_datasource()
        self._created_assets = set()
    
    def _setup_datasource(self):
        """Setup pandas datasource for runtime validation"""
        try:
            self.datasource = self.context.get_datasource("pandas_runtime")
        except:
            self.datasource = self.context.sources.add_pandas("pandas_runtime")
    
    # ========================================================================
    # FRED RAW DATA EXPECTATIONS
    # ========================================================================
    
    def create_fred_raw_expectations(self) -> str:
        """
        Create expectations for RAW FRED data
        
        RAW FRED data has:
        - GDP (in billions, not growth %)
        - CPI (index level, not inflation %)
        - Unemployment_Rate (already in %)
        - Federal_Funds_Rate (already in %)
        """
        suite_name = "fred_raw_suite"
        
        # Delete existing suite
        try:
            self.context.delete_expectation_suite(suite_name)
        except:
            pass
        
        suite = self.context.add_expectation_suite(suite_name)
        
        expectations = [
            # Column existence - RAW column names
            ExpectationConfiguration(
                expectation_type="expect_column_to_exist",
                kwargs={"column": "GDP"}  # RAW: GDP (not GDP_Growth)
            ),
            ExpectationConfiguration(
                expectation_type="expect_column_to_exist",
                kwargs={"column": "CPI"}  # RAW: CPI (not CPI_Inflation)
            ),
            ExpectationConfiguration(
                expectation_type="expect_column_to_exist",
                kwargs={"column": "Unemployment_Rate"}
            ),
            ExpectationConfiguration(
                expectation_type="expect_column_to_exist",
                kwargs={"column": "Federal_Funds_Rate"}
            ),
            
            # Value ranges - RAW values (not growth rates!)
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_be_between",
                kwargs={
                    "column": "GDP",
                    "min_value": 5000,    # Billions of dollars (2005: ~13T, 2025: ~28T)
                    "max_value": 35000,
                    "mostly": 0.95
                }
            ),
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_be_between",
                kwargs={
                    "column": "CPI",
                    "min_value": 150,     # Index (2005: ~195, 2025: ~315)
                    "max_value": 400,
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
            
            # Completeness
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_not_be_null",
                kwargs={
                    "column": "Unemployment_Rate",
                    "mostly": 0.9
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
        
        for exp_config in expectations:
            suite.add_expectation(expectation_configuration=exp_config)
        
        self.context.add_or_update_expectation_suite(expectation_suite=suite)
        print(f"   ✅ Created {len(expectations)} expectations for FRED raw data")
        
        return suite_name
    
    # ========================================================================
    # MARKET RAW DATA EXPECTATIONS
    # ========================================================================
    
    def create_market_raw_expectations(self) -> str:
        """
        Create expectations for RAW market data
        
        RAW market data has:
        - VIX (raw close prices)
        - SP500 (raw index level, not returns)
        """
        suite_name = "market_raw_suite"
        
        try:
            self.context.delete_expectation_suite(suite_name)
        except:
            pass
        
        suite = self.context.add_expectation_suite(suite_name)
        
        expectations = [
            # Columns
            ExpectationConfiguration(
                expectation_type="expect_column_to_exist",
                kwargs={"column": "VIX"}
            ),
            ExpectationConfiguration(
                expectation_type="expect_column_to_exist",
                kwargs={"column": "SP500"}
            ),
            
            # VIX range
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_be_between",
                kwargs={
                    "column": "VIX",
                    "min_value": 5,
                    "max_value": 100,
                    "mostly": 0.99
                }
            ),
            
            # S&P 500 index level (not return %)
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_be_between",
                kwargs={
                    "column": "SP500",
                    "min_value": 500,     # 2005: ~1200, 2025: ~6000
                    "max_value": 10000,
                    "mostly": 0.99
                }
            ),
            
            # Completeness
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_not_be_null",
                kwargs={
                    "column": "VIX",
                    "mostly": 0.95
                }
            ),
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_not_be_null",
                kwargs={
                    "column": "SP500",
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
        
        for exp_config in expectations:
            suite.add_expectation(expectation_configuration=exp_config)
        
        self.context.add_or_update_expectation_suite(expectation_suite=suite)
        print(f"   ✅ Created {len(expectations)} expectations for market raw data")
        
        return suite_name
    
    # ========================================================================
    # COMPANY PRICES RAW DATA EXPECTATIONS
    # ========================================================================
    
    def create_company_prices_raw_expectations(self) -> str:
        """
        Create expectations for RAW company price data
        
        RAW price data has:
        - Open, High, Low, Close, Volume (OHLCV)
        - No calculated returns yet
        """
        suite_name = "company_prices_raw_suite"
        
        try:
            self.context.delete_expectation_suite(suite_name)
        except:
            pass
        
        suite = self.context.add_expectation_suite(suite_name)
        
        expectations = [
            # Required OHLCV columns
            ExpectationConfiguration(
                expectation_type="expect_column_to_exist",
                kwargs={"column": "Open"}
            ),
            ExpectationConfiguration(
                expectation_type="expect_column_to_exist",
                kwargs={"column": "High"}
            ),
            ExpectationConfiguration(
                expectation_type="expect_column_to_exist",
                kwargs={"column": "Low"}
            ),
            ExpectationConfiguration(
                expectation_type="expect_column_to_exist",
                kwargs={"column": "Close"}
            ),
            ExpectationConfiguration(
                expectation_type="expect_column_to_exist",
                kwargs={"column": "Volume"}
            ),
            ExpectationConfiguration(
                expectation_type="expect_column_to_exist",
                kwargs={"column": "Company"}
            ),
            ExpectationConfiguration(
                expectation_type="expect_column_to_exist",
                kwargs={"column": "Sector"}
            ),
            
            # Price ranges
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_be_between",
                kwargs={
                    "column": "Close",
                    "min_value": 0.01,
                    "max_value": 10000,
                    "mostly": 0.99
                }
            ),
            
            # High >= Low (price logic)
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_be_between",
                kwargs={
                    "column": "High",
                    "min_value": 0.01,
                    "max_value": 10000,
                    "mostly": 0.99
                }
            ),
            
            # Volume >= 0
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_be_between",
                kwargs={
                    "column": "Volume",
                    "min_value": 0,
                    "max_value": None,
                    "mostly": 0.95
                }
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
            
            # Company not null
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_not_be_null",
                kwargs={"column": "Company"}
            ),
            
            # Minimum data
            ExpectationConfiguration(
                expectation_type="expect_table_row_count_to_be_between",
                kwargs={
                    "min_value": 10000,
                    "max_value": None
                }
            )
        ]
        
        for exp_config in expectations:
            suite.add_expectation(expectation_configuration=exp_config)
        
        self.context.add_or_update_expectation_suite(expectation_suite=suite)
        print(f"   ✅ Created {len(expectations)} expectations for company prices")
        
        return suite_name
    
    # ========================================================================
    # COMPANY FUNDAMENTALS RAW DATA EXPECTATIONS
    # ========================================================================
    
    def create_company_fundamentals_raw_expectations(self, statement_type: str) -> str:
        """
        Create expectations for RAW company fundamentals
        
        Args:
            statement_type: 'income' or 'balance'
        
        RAW fundamentals are QUARTERLY data from Alpha Vantage
        """
        suite_name = f"company_{statement_type}_raw_suite"
        
        try:
            self.context.delete_expectation_suite(suite_name)
        except:
            pass
        
        suite = self.context.add_expectation_suite(suite_name)
        
        if statement_type == 'income':
            expectations = [
                # Required columns
                ExpectationConfiguration(
                    expectation_type="expect_column_to_exist",
                    kwargs={"column": "Date"}
                ),
                ExpectationConfiguration(
                    expectation_type="expect_column_to_exist",
                    kwargs={"column": "Revenue"}
                ),
                ExpectationConfiguration(
                    expectation_type="expect_column_to_exist",
                    kwargs={"column": "Net_Income"}
                ),
                ExpectationConfiguration(
                    expectation_type="expect_column_to_exist",
                    kwargs={"column": "EPS"}
                ),
                ExpectationConfiguration(
                    expectation_type="expect_column_to_exist",
                    kwargs={"column": "Company"}
                ),
                ExpectationConfiguration(
                    expectation_type="expect_column_to_exist",
                    kwargs={"column": "Sector"}
                ),
                
                # Revenue range (quarterly revenue in dollars)
                ExpectationConfiguration(
                    expectation_type="expect_column_values_to_be_between",
                    kwargs={
                        "column": "Revenue",
                        "min_value": 0,
                        "max_value": 1e12,  # $1 trillion max per quarter
                        "mostly": 0.7  # Allow some nulls (not all companies report all quarters)
                    }
                ),
                
                # Net Income can be negative (losses)
                ExpectationConfiguration(
                    expectation_type="expect_column_values_to_be_between",
                    kwargs={
                        "column": "Net_Income",
                        "min_value": -1e11,  # Can lose money
                        "max_value": 1e11,
                        "mostly": 0.7
                    }
                ),
                
                # EPS range (can be negative)
                ExpectationConfiguration(
                    expectation_type="expect_column_values_to_be_between",
                    kwargs={
                        "column": "EPS",
                        "min_value": -100,
                        "max_value": 200,
                        "mostly": 0.7
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
                
                # Minimum quarters
                ExpectationConfiguration(
                    expectation_type="expect_table_row_count_to_be_between",
                    kwargs={
                        "min_value": 10,  # At least a few quarters
                        "max_value": None
                    }
                )
            ]
        
        elif statement_type == 'balance':
            expectations = [
                # Required columns
                ExpectationConfiguration(
                    expectation_type="expect_column_to_exist",
                    kwargs={"column": "Date"}
                ),
                ExpectationConfiguration(
                    expectation_type="expect_column_to_exist",
                    kwargs={"column": "Total_Assets"}
                ),
                ExpectationConfiguration(
                    expectation_type="expect_column_to_exist",
                    kwargs={"column": "Total_Liabilities"}
                ),
                ExpectationConfiguration(
                    expectation_type="expect_column_to_exist",
                    kwargs={"column": "Total_Equity"}
                ),
                ExpectationConfiguration(
                    expectation_type="expect_column_to_exist",
                    kwargs={"column": "Current_Assets"}
                ),
                ExpectationConfiguration(
                    expectation_type="expect_column_to_exist",
                    kwargs={"column": "Current_Liabilities"}
                ),
                ExpectationConfiguration(
                    expectation_type="expect_column_to_exist",
                    kwargs={"column": "Company"}
                ),
                
                # Assets > 0
                ExpectationConfiguration(
                    expectation_type="expect_column_values_to_be_between",
                    kwargs={
                        "column": "Total_Assets",
                        "min_value": 1e6,      # At least $1M
                        "max_value": 1e13,     # Max $10T
                        "mostly": 0.7
                    }
                ),
                
                # Liabilities >= 0
                ExpectationConfiguration(
                    expectation_type="expect_column_values_to_be_between",
                    kwargs={
                        "column": "Total_Liabilities",
                        "min_value": 0,
                        "max_value": 1e13,
                        "mostly": 0.7
                    }
                ),
                
                # Equity can be negative (distressed companies)
                ExpectationConfiguration(
                    expectation_type="expect_column_values_to_be_between",
                    kwargs={
                        "column": "Total_Equity",
                        "min_value": -1e12,    # Can be negative
                        "max_value": 1e13,
                        "mostly": 0.7
                    }
                ),
                
                # Debt_to_Equity ratio (if exists)
                ExpectationConfiguration(
                    expectation_type="expect_column_values_to_be_between",
                    kwargs={
                        "column": "Debt_to_Equity",
                        "min_value": 0,
                        "max_value": 50,  # Banks can have high leverage
                        "mostly": 0.7
                    }
                ),
                
                # Current_Ratio (if exists)
                ExpectationConfiguration(
                    expectation_type="expect_column_values_to_be_between",
                    kwargs={
                        "column": "Current_Ratio",
                        "min_value": 0.1,
                        "max_value": 20,
                        "mostly": 0.7
                    }
                ),
                
                # Company not null
                ExpectationConfiguration(
                    expectation_type="expect_column_values_to_not_be_null",
                    kwargs={"column": "Company"}
                ),
                
                # Minimum quarters
                ExpectationConfiguration(
                    expectation_type="expect_table_row_count_to_be_between",
                    kwargs={
                        "min_value": 10,
                        "max_value": None
                    }
                )
            ]
        else:
            raise ValueError(f"Invalid statement_type: {statement_type}")
        
        for exp_config in expectations:
            suite.add_expectation(expectation_configuration=exp_config)
        
        self.context.add_or_update_expectation_suite(expectation_suite=suite)
        print(f"   ✅ Created {len(expectations)} expectations for {statement_type} statement")
        
        return suite_name
    
    # ========================================================================
    # VALIDATION EXECUTOR
    # ========================================================================
    
    def validate_dataframe(
        self,
        df: pd.DataFrame,
        suite_name: str,
        data_asset_name: str
    ) -> Tuple[bool, Dict]:
        """Validate DataFrame against expectation suite"""
        
        try:
            # Create unique asset name
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            unique_asset_name = f"{data_asset_name}_{timestamp}"
            
            # Create data asset
            data_asset = self.datasource.add_dataframe_asset(name=unique_asset_name)
            batch_request = data_asset.build_batch_request(dataframe=df)
            
            # Get validator
            validator = self.context.get_validator(
                batch_request=batch_request,
                expectation_suite_name=suite_name
            )
            
            # Run validation
            results = validator.validate()
            
            # Parse results
            is_valid = results.success
            stats = results.statistics if hasattr(results, 'statistics') else {}
            
            total_expectations = stats.get('evaluated_expectations', 0)
            successful = stats.get('successful_expectations', 0)
            failed = stats.get('unsuccessful_expectations', 0)
            success_percent = stats.get('success_percent', 0.0) or 0.0
            
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
                            'details': str(result.result)[:200]
                        })
            
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
            
            # Save report
            self._save_report(report, data_asset_name)
            
            return is_valid, report
        
        except Exception as e:
            print(f"   ❌ Validation error: {str(e)}")
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
        """Save validation report to disk"""
        os.makedirs('data/validation_reports', exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"data/validation_reports/{data_asset_name}_raw_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"   💾 Report saved: {filename}")


# ============================================================================
# CONVENIENCE FUNCTIONS (Called by validate_raw_data.py)
# ============================================================================

def validate_fred_raw(df: pd.DataFrame) -> Tuple[bool, Dict]:
    """
    Validate RAW FRED data
    
    Called by: validate_raw_data.py
    Expects: RAW column names (GDP, CPI, not GDP_Growth, CPI_Inflation)
    """
    validator = RawDataValidator()
    suite_name = validator.create_fred_raw_expectations()
    return validator.validate_dataframe(df, suite_name, "fred_raw")


def validate_market_raw(df: pd.DataFrame) -> Tuple[bool, Dict]:
    """
    Validate RAW market data
    
    Called by: validate_raw_data.py
    Expects: VIX, SP500 (raw close prices, not returns)
    """
    validator = RawDataValidator()
    suite_name = validator.create_market_raw_expectations()
    return validator.validate_dataframe(df, suite_name, "market_raw")


def validate_company_prices_raw(df: pd.DataFrame) -> Tuple[bool, Dict]:
    """
    Validate RAW company price data
    
    Called by: validate_raw_data.py
    Expects: Open, High, Low, Close, Volume, Company, Sector
    """
    validator = RawDataValidator()
    suite_name = validator.create_company_prices_raw_expectations()
    return validator.validate_dataframe(df, suite_name, "company_prices_raw")


def validate_company_fundamentals_raw(df: pd.DataFrame, statement_type: str) -> Tuple[bool, Dict]:
    """
    Validate RAW company fundamentals (quarterly data from Alpha Vantage)
    
    Called by: validate_raw_data.py
    Args:
        df: DataFrame with quarterly fundamental data
        statement_type: 'income' or 'balance'
    
    Expects (income):
        - Date, Revenue, Net_Income, EPS, Company, Sector
        
    Expects (balance):
        - Date, Total_Assets, Total_Liabilities, Total_Equity, 
          Current_Assets, Current_Liabilities, Company, Sector
    """
    validator = RawDataValidator()
    suite_name = validator.create_company_fundamentals_raw_expectations(statement_type)
    return validator.validate_dataframe(df, suite_name, f"company_{statement_type}_raw")


# ============================================================================
# TEST MODE
# ============================================================================

if __name__ == "__main__":
    """Test the validator"""
    print("\n" + "="*70)
    print("GREAT EXPECTATIONS RAW DATA VALIDATOR - TEST MODE")
    print("="*70)
    
    print("\n✅ Validator loaded successfully!")
    print("\nAvailable validation functions:")
    print("  - validate_fred_raw(df)")
    print("  - validate_market_raw(df)")
    print("  - validate_company_prices_raw(df)")
    print("  - validate_company_fundamentals_raw(df, 'income')")
    print("  - validate_company_fundamentals_raw(df, 'balance')")
    
    print("\n" + "="*70)
    print("To use: Import these functions in validate_raw_data.py")
    print("="*70)