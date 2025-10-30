"""
CHECKPOINT 3: Validate Merged Data
Runs after Step 3 (merging), before cleaning merged data

Focus:
- Merge quality (no data loss)
- Alignment (dates match)
- Completeness (all companies have macro context)
"""

import pandas as pd
import sys
from pathlib import Path
from robust_validator import RobustValidator
from ge_validator_base import GEValidatorBase, ValidationSeverity
from great_expectations.core import ExpectationConfiguration
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


class MergedDataValidator:
    """Checkpoint 3: Validate merged data."""
    
    def __init__(self):
        self.features_dir = Path("data/features")
        self.ge_validator = GEValidatorBase()
        self.all_reports = {}
    
    def validate_macro_features(self) -> bool:
        """Validate macro_features.csv (FRED + Market merged)."""
        logger.info("\n[1/2] Validating macro_features.csv...")
        
        filepath = self.features_dir / "macro_features.csv"
        if not filepath.exists():
            logger.error(f"❌ File not found: {filepath}")
            return False
        
        df = pd.read_csv(filepath, parse_dates=['Date'])
        
        # GE expectations
        expectations = [
            # Has columns from both sources
            ExpectationConfiguration(
                expectation_type="expect_column_to_exist",
                kwargs={"column": "GDP"}  # From FRED
            ),
            ExpectationConfiguration(
                expectation_type="expect_column_to_exist",
                kwargs={"column": "VIX"}  # From Market
            ),
            
            # Row count reasonable
            ExpectationConfiguration(
                expectation_type="expect_table_row_count_to_be_between",
                kwargs={
                    "min_value": 3000,
                    "max_value": 10000
                }
            ),
            
            # Column count increased (FRED + Market features)
            ExpectationConfiguration(
                expectation_type="expect_table_column_count_to_be_between",
                kwargs={
                    "min_value": 50,  # Should have many features
                    "max_value": 150
                }
            )
        ]
        
        suite_name = self.ge_validator.create_expectation_suite("macro_features_suite", expectations)
        ge_passed, ge_report = self.ge_validator.validate_dataframe(
            df, suite_name, "macro_features", ValidationSeverity.CRITICAL
        )
        
        # RobustValidator
        robust_validator = RobustValidator(
            dataset_name="macro_features",
            enable_auto_fix=True,
            enable_temporal_checks=True,
            enable_business_rules=True
        )
        
        df_validated, robust_report = robust_validator.validate(df)
        
        # Save if fixes applied
        if robust_report.remediation_log:
            df_validated.to_csv(filepath, index=False)
        
        critical_count = robust_report.count_by_severity()['CRITICAL']
        robust_passed = (critical_count == 0)
        
        passed = ge_passed and robust_passed
        
        self.all_reports['macro_features'] = {
            'ge_report': ge_report,
            'robust_report': robust_report.to_dict(),
            'passed': passed
        }
        
        if passed:
            logger.info("  ✅ macro_features.csv validation PASSED")
        else:
            logger.error("  ❌ macro_features.csv validation FAILED")
        
        return passed
    
    def validate_merged_features(self) -> bool:
        """Validate merged_features.csv (Macro + Market + Company)."""
        logger.info("\n[2/2] Validating merged_features.csv...")
        
        filepath = self.features_dir / "merged_features.csv"
        if not filepath.exists():
            logger.error(f"❌ File not found: {filepath}")
            return False
        
        df = pd.read_csv(filepath, parse_dates=['Date'])
        
        # GE expectations
        expectations = [
            # Has columns from all sources
            ExpectationConfiguration(
                expectation_type="expect_column_to_exist",
                kwargs={"column": "GDP"}  # Macro
            ),
            ExpectationConfiguration(
                expectation_type="expect_column_to_exist",
                kwargs={"column": "VIX"}  # Market
            ),
            ExpectationConfiguration(
                expectation_type="expect_column_to_exist",
                kwargs={"column": "Stock_Price"}  # Company
            ),
            ExpectationConfiguration(
                expectation_type="expect_column_to_exist",
                kwargs={"column": "Company"}
            ),
            
            # Company not null
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_not_be_null",
                kwargs={"column": "Company"}
            ),
            
            # Reasonable company count
            ExpectationConfiguration(
                expectation_type="expect_column_unique_value_count_to_be_between",
                kwargs={
                    "column": "Company",
                    "min_value": 2,
                    "max_value": 50
                }
            ),
            
            # No duplicate (Date, Company) pairs
            ExpectationConfiguration(
                expectation_type="expect_compound_columns_to_be_unique",
                kwargs={"column_list": ["Date", "Company"]}
            ),
            
            # Row count
            ExpectationConfiguration(
                expectation_type="expect_table_row_count_to_be_between",
                kwargs={
                    "min_value": 5000,
                    "max_value": 200000
                }
            )
        ]
        
        suite_name = self.ge_validator.create_expectation_suite("merged_features_suite", expectations)
        ge_passed, ge_report = self.ge_validator.validate_dataframe(
            df, suite_name, "merged_features", ValidationSeverity.CRITICAL
        )
        
        # RobustValidator
        robust_validator = RobustValidator(
            dataset_name="merged_features",
            enable_auto_fix=True,
            enable_temporal_checks=True,
            enable_business_rules=True
        )
        
        df_validated, robust_report = robust_validator.validate(df)
        
        if robust_report.remediation_log:
            df_validated.to_csv(filepath, index=False)
        
        critical_count = robust_report.count_by_severity()['CRITICAL']
        robust_passed = (critical_count == 0)
        
        passed = ge_passed and robust_passed
        
        self.all_reports['merged_features'] = {
            'ge_report': ge_report,
            'robust_report': robust_report.to_dict(),
            'passed': passed
        }
        
        if passed:
            logger.info("  ✅ merged_features.csv validation PASSED")
        else:
            logger.error("  ❌ merged_features.csv validation FAILED")
        
        return passed
    
    def run_all_validations(self) -> bool:
        """Run all merged data validations."""
        logger.info("\n" + "="*80)
        logger.info("CHECKPOINT 3: MERGED DATA VALIDATION")
        logger.info("="*80)
        
        results = {
            'macro': self.validate_macro_features(),
            'merged': self.validate_merged_features()
        }
        
        all_passed = all(results.values())
        
        # Summary
        logger.info("\n" + "="*80)
        logger.info("CHECKPOINT 3 SUMMARY")
        logger.info("="*80)
        
        for name, passed in results.items():
            status = "✅ PASSED" if passed else "❌ FAILED"
            logger.info(f"{name:20s}: {status}")
        
        if all_passed:
            logger.info("\n✅ CHECKPOINT 3 PASSED - Proceeding to Step 3c (Clean Merged)")
            return True
        else:
            logger.error("\n❌ CHECKPOINT 3 FAILED - Pipeline stopped")
            return False


def main():
    validator = MergedDataValidator()
    success = validator.run_all_validations()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()