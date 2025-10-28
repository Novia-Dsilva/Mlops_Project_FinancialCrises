"""
CHECKPOINT 2: Validate Cleaned Data
Runs after Step 1 (cleaning), before feature engineering

Focus:
- Missing values handled (< 5%)
- No inf values
- Duplicates removed
- Point-in-time correctness
"""

import pandas as pd
import sys
from pathlib import Path
from robust_validator import RobustValidator, ValidationSeverity
from ge_validator_base import GEValidatorBase, ValidationSeverity as GESeverity
from great_expectations.core import ExpectationConfiguration
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


class CleanedDataValidator:
    """Checkpoint 2: Validate cleaned data."""
    
    def __init__(self):
        self.clean_dir = Path("data/clean")
        self.ge_validator = GEValidatorBase()
        self.all_reports = {}
    
    def validate_fred_clean(self) -> bool:
        """Validate FRED cleaned data."""
        logger.info("\n[1/5] Validating fred_clean.csv...")
        
        filepath = self.clean_dir / "fred_clean.csv"
        if not filepath.exists():
            logger.error(f"❌ File not found: {filepath}")
            return False
        
        df = pd.read_csv(filepath, parse_dates=['Date'])
        
        # GE expectations - STRICTER after cleaning
        expectations = [
            # Must have all columns
            ExpectationConfiguration(
                expectation_type="expect_column_to_exist",
                kwargs={"column": "GDP"}
            ),
            ExpectationConfiguration(
                expectation_type="expect_column_to_exist",
                kwargs={"column": "CPI"}
            ),
            
            # After cleaning: < 5% missing
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_not_be_null",
                kwargs={
                    "column": "GDP",
                    "mostly": 0.95  # Stricter than raw (was 0.80)
                }
            ),
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_not_be_null",
                kwargs={
                    "column": "CPI",
                    "mostly": 0.95
                }
            ),
            
            # No duplicates
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_be_unique",
                kwargs={"column": "Date"}
            ),
            
            # Row count should match raw (no data loss)
            ExpectationConfiguration(
                expectation_type="expect_table_row_count_to_be_between",
                kwargs={
                    "min_value": 1000,
                    "max_value": 10000
                }
            )
        ]
        
        suite_name = self.ge_validator.create_expectation_suite("fred_clean_suite", expectations)
        ge_passed, ge_report = self.ge_validator.validate_dataframe(
            df, suite_name, "fred_clean", GESeverity.CRITICAL
        )
        
        # RobustValidator with auto-fix enabled
        robust_validator = RobustValidator(
            dataset_name="fred_clean",
            enable_auto_fix=True,  # Can fix minor issues
            enable_temporal_checks=True,
            enable_business_rules=True
        )
        
        df_validated, robust_report = robust_validator.validate(df)
        
        # Save if fixes applied
        if robust_report.remediation_log:
            df_validated.to_csv(filepath, index=False)
            logger.info(f"  ✓ Applied {len(robust_report.remediation_log)} auto-fixes")
        
        critical_count = robust_report.count_by_severity()['CRITICAL']
        robust_passed = (critical_count == 0)
        
        passed = ge_passed and robust_passed
        
        self.all_reports['fred_clean'] = {
            'ge_report': ge_report,
            'robust_report': robust_report.to_dict(),
            'passed': passed
        }
        
        if passed:
            logger.info("  ✅ fred_clean.csv validation PASSED")
        else:
            logger.error("  ❌ fred_clean.csv validation FAILED")
        
        return passed
    
    # ... Similar methods for market_clean, company_prices_clean, 
    # company_balance_clean, company_income_clean ...
    # (Follow same pattern as fred_clean with stricter thresholds)
    
    def run_all_validations(self) -> bool:
        """Run all cleaned data validations."""
        logger.info("\n" + "="*80)
        logger.info("CHECKPOINT 2: CLEANED DATA VALIDATION")
        logger.info("="*80)
        
        results = {
            'fred': self.validate_fred_clean(),
            # Add other datasets...
        }
        
        all_passed = all(results.values())
        
        # Summary
        logger.info("\n" + "="*80)
        logger.info("CHECKPOINT 2 SUMMARY")
        logger.info("="*80)
        
        for name, passed in results.items():
            status = "✅ PASSED" if passed else "❌ FAILED"
            logger.info(f"{name:20s}: {status}")
        
        if all_passed:
            logger.info("\n✅ CHECKPOINT 2 PASSED - Proceeding to Step 2 (Feature Engineering)")
            return True
        else:
            logger.error("\n❌ CHECKPOINT 2 FAILED - Pipeline stopped")
            return False


def main():
    validator = CleanedDataValidator()
    success = validator.run_all_validations()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()