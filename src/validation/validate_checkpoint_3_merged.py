"""
CHECKPOINT 4: Validate Clean Merged Data
Runs after Step 3c (post-merge cleaning), before Step 3b (interaction features)

Focus:
- No inf values (CRITICAL)
- Minimal missing values (< 2%)
- No duplicate columns with suffixes
- Valid data types
- Proper value ranges
- Company-level data integrity

Validates:
- macro_features_clean.csv
- merged_features_clean.csv

Usage:
    python src/validation/validate_checkpoint_4_clean_merged.py

Exit codes:
    0: All validations passed
    1: Validation failed
"""

import pandas as pd
import sys
from pathlib import Path
from robust_validator import RobustValidator, ValidationSeverity
from ge_validator_base import GEValidatorBase, ValidationSeverity as GESeverity
from great_expectations.core import ExpectationConfiguration
import logging
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


class CleanMergedDataValidator:
    """Checkpoint 4: Validate cleaned merged data."""
    
    def __init__(self):
        self.features_dir = Path("data/features")
        self.ge_validator = GEValidatorBase()
        self.all_reports = {}
    
    def validate_macro_features_clean(self) -> bool:
        """
        Validate macro_features_clean.csv.
        
        After post-merge cleaning, this should be pristine:
        - 0 inf values
        - < 1% missing
        - No duplicate columns
        - Proper data types
        """
        logger.info("\n[1/2] Validating macro_features_clean.csv...")
        
        filepath = self.features_dir / "macro_features_clean.csv"
        if not filepath.exists():
            logger.error(f"❌ File not found: {filepath}")
            logger.error("   Run Step 4 first: python step4_post_merge_cleaning.py")
            return False
        
        df = pd.read_csv(filepath, parse_dates=['Date'])
        
        # CRITICAL FIX: Ensure Date is datetime
        if 'Date' in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(df['Date']):
                df['Date'] = pd.to_datetime(df['Date'], format='mixed', errors='coerce')

        logger.info(f"   Loaded: {df.shape}")
        logger.info(f"   Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")
        
        # === GREAT EXPECTATIONS VALIDATION ===
        logger.info("\n   Running Great Expectations checks...")
        
        expectations = [
            # === CRITICAL: Column existence ===
            ExpectationConfiguration(
                expectation_type="expect_column_to_exist",
                kwargs={"column": "Date"}
            ),
            ExpectationConfiguration(
                expectation_type="expect_column_to_exist",
                kwargs={"column": "GDP"}  # From FRED
            ),
            ExpectationConfiguration(
                expectation_type="expect_column_to_exist",
                kwargs={"column": "CPI"}  # From FRED
            ),
            ExpectationConfiguration(
                expectation_type="expect_column_to_exist",
                kwargs={"column": "VIX"}  # From Market
            ),
            
            # === CRITICAL: No duplicate columns with suffixes ===
            # We check this in RobustValidator
            
            # === CRITICAL: Completeness after cleaning ===
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_not_be_null",
                kwargs={
                    "column": "GDP",
                    "mostly": 0.99  # Very strict: < 1% missing
                }
            ),
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_not_be_null",
                kwargs={
                    "column": "CPI",
                    "mostly": 0.99
                }
            ),
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_not_be_null",
                kwargs={
                    "column": "VIX",
                    "mostly": 0.99
                }
            ),
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_not_be_null",
                kwargs={
                    "column": "Unemployment_Rate",
                    "mostly": 0.99
                }
            ),
            
            # === ERROR: Value ranges ===
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_be_between",
                kwargs={
                    "column": "GDP",
                    "min_value": 5000,
                    "max_value": 35000,
                    "mostly": 0.99
                }
            ),
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_be_between",
                kwargs={
                    "column": "CPI",
                    "min_value": 150,
                    "max_value": 400,
                    "mostly": 0.99
                }
            ),
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_be_between",
                kwargs={
                    "column": "VIX",
                    "min_value": 5,
                    "max_value": 100,
                    "mostly": 0.99
                }
            ),
            
            # === CRITICAL: No duplicates ===
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_be_unique",
                kwargs={"column": "Date"}
            ),
            
            # === ERROR: Row count ===
            ExpectationConfiguration(
                expectation_type="expect_table_row_count_to_be_between",
                kwargs={
                    "min_value": 3000,
                    "max_value": 10000
                }
            ),
            
            # === ERROR: Column count (should be reasonable after cleanup) ===
            ExpectationConfiguration(
                expectation_type="expect_table_column_count_to_be_between",
                kwargs={
                    "min_value": 40,   # Minimum useful features
                    "max_value": 120   # Maximum before cleanup
                }
            )
        ]
        
        suite_name = self.ge_validator.create_expectation_suite(
            "macro_features_clean_suite", 
            expectations
        )
        
        ge_passed, ge_report = self.ge_validator.validate_dataframe(
            df, 
            suite_name, 
            "macro_features_clean",
            severity_threshold=GESeverity.CRITICAL
        )
        
        # === ROBUST VALIDATOR ===
        logger.info("\n   Running RobustValidator checks...")
        
        robust_validator = RobustValidator(
            dataset_name="macro_features_clean",
            enable_auto_fix=False,  # Should already be clean
            enable_temporal_checks=True,
            enable_business_rules=True
        )
        
        df_validated, robust_report = robust_validator.validate(df)
        
        # === ADDITIONAL POST-CLEANING CHECKS ===
        logger.info("\n   Running post-cleaning specific checks...")
        
        post_clean_issues = []
        
        # 1. Check for inf values (CRITICAL - should be 0)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        inf_count = np.isinf(df[numeric_cols]).sum().sum()
        
        if inf_count > 0:
            post_clean_issues.append(f"CRITICAL: Found {inf_count} inf values (should be 0)")
            logger.error(f"   ❌ Found {inf_count} inf values")
        else:
            logger.info(f"   ✓ No inf values (0)")
        
        # 2. Check for duplicate columns with suffixes
        suffix_cols = [col for col in df.columns if any(
            col.endswith(suffix) for suffix in ['_x', '_y', '_fred', '_market', '_macro', '_dup']
        )]
        
        if suffix_cols:
            post_clean_issues.append(f"CRITICAL: Found {len(suffix_cols)} duplicate columns: {suffix_cols}")
            logger.error(f"   ❌ Found {len(suffix_cols)} duplicate columns with suffixes")
        else:
            logger.info(f"   ✓ No duplicate columns (0)")
        
        # 3. Check missing percentage
        missing_pct = (df.isna().sum().sum() / df.size) * 100
        if missing_pct > 1.0:
            post_clean_issues.append(f"ERROR: {missing_pct:.2f}% missing (expected < 1%)")
            logger.warning(f"   ⚠️  Missing: {missing_pct:.2f}% (expected < 1%)")
        else:
            logger.info(f"   ✓ Missing: {missing_pct:.2f}% (< 1%)")
        
        # 4. Check for constant columns
        constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
        if constant_cols:
            post_clean_issues.append(f"WARNING: Found {len(constant_cols)} constant columns")
            logger.warning(f"   ⚠️  Found {len(constant_cols)} constant columns (should be removed)")
        else:
            logger.info(f"   ✓ No constant columns (0)")
        
        # === FINAL DECISION ===
        critical_count = robust_report.count_by_severity()['CRITICAL']
        critical_from_post_clean = len([i for i in post_clean_issues if 'CRITICAL' in i])
        
        robust_passed = (critical_count == 0 and critical_from_post_clean == 0)
        passed = ge_passed and robust_passed
        
        self.all_reports['macro_features_clean'] = {
            'ge_report': ge_report,
            'robust_report': robust_report.to_dict(),
            'post_clean_issues': post_clean_issues,
            'passed': passed
        }
        
        if passed:
            logger.info("\n  ✅ macro_features_clean.csv validation PASSED")
        else:
            logger.error("\n  ❌ macro_features_clean.csv validation FAILED")
            if not ge_passed:
                logger.error(f"     GE failures: {ge_report['critical_failures']} critical")
            if not robust_passed:
                logger.error(f"     Robust failures: {critical_count} critical")
            if post_clean_issues:
                logger.error(f"     Post-clean issues: {len(post_clean_issues)}")
                for issue in post_clean_issues:
                    logger.error(f"       - {issue}")
        
        return passed
    
    def validate_merged_features_clean(self) -> bool:
        """
        Validate merged_features_clean.csv.
        
        This is the most important validation - ensures company-level data
        is clean and ready for modeling.
        """
        logger.info("\n[2/2] Validating merged_features_clean.csv...")
        
        filepath = self.features_dir / "merged_features_clean.csv"
        if not filepath.exists():
            logger.error(f"❌ File not found: {filepath}")
            logger.error("   Run Step 3c first: python step3c_post_merge_cleaning.py")
            return False
        
        df = pd.read_csv(filepath, parse_dates=['Date'])
        # CRITICAL FIX: Ensure Date is datetime
        if 'Date' in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(df['Date']):
                df['Date'] = pd.to_datetime(df['Date'], format='mixed', errors='coerce')
        logger.info(f"   Loaded: {df.shape}")
        logger.info(f"   Companies: {df['Company'].nunique()}")
        logger.info(f"   Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")
        
        # === GREAT EXPECTATIONS VALIDATION ===
        logger.info("\n   Running Great Expectations checks...")
        
        expectations = [
            # === CRITICAL: Column existence from all sources ===
            ExpectationConfiguration(
                expectation_type="expect_column_to_exist",
                kwargs={"column": "Date"}
            ),
            ExpectationConfiguration(
                expectation_type="expect_column_to_exist",
                kwargs={"column": "Company"}
            ),
            ExpectationConfiguration(
                expectation_type="expect_column_to_exist",
                kwargs={"column": "GDP"}  # Macro
            ),
            ExpectationConfiguration(
                expectation_type="expect_column_to_exist",
                kwargs={"column": "VIX"}  # Market
            ),
            
            # Check for Stock_Price or Close
            # Note: GE will check if at least one exists via RobustValidator
            
            # === CRITICAL: Company data integrity ===
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_not_be_null",
                kwargs={"column": "Company"}
            ),
            ExpectationConfiguration(
                expectation_type="expect_column_unique_value_count_to_be_between",
                kwargs={
                    "column": "Company",
                    "min_value": 2,
                    "max_value": 50
                }
            ),
            
            # === CRITICAL: No duplicate (Date, Company) pairs ===
            ExpectationConfiguration(
                expectation_type="expect_compound_columns_to_be_unique",
                kwargs={"column_list": ["Date", "Company"]}
            ),
            
            # === CRITICAL: Key columns completeness (very strict after cleaning) ===
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_not_be_null",
                kwargs={
                    "column": "GDP",
                    "mostly": 0.99
                }
            ),
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_not_be_null",
                kwargs={
                    "column": "VIX",
                    "mostly": 0.99
                }
            ),
            
            # === ERROR: Value ranges ===
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_be_between",
                kwargs={
                    "column": "GDP",
                    "min_value": 5000,
                    "max_value": 35000,
                    "mostly": 0.99
                }
            ),
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_be_between",
                kwargs={
                    "column": "VIX",
                    "min_value": 5,
                    "max_value": 100,
                    "mostly": 0.99
                }
            ),
            
            # === ERROR: Financial ratios in reasonable ranges ===
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_be_between",
                kwargs={
                    "column": "Profit_Margin",
                    "min_value": -100,  # Can have losses
                    "max_value": 100,
                    "mostly": 0.95
                }
            ),
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_be_between",
                kwargs={
                    "column": "ROE",
                    "min_value": -200,
                    "max_value": 200,
                    "mostly": 0.95
                }
            ),
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_be_between",
                kwargs={
                    "column": "Debt_to_Equity",
                    "min_value": 0,
                    "max_value": 100,
                    "mostly": 0.95
                }
            ),
            
            # === ERROR: Row count ===
            ExpectationConfiguration(
                expectation_type="expect_table_row_count_to_be_between",
                kwargs={
                    "min_value": 5000,
                    "max_value": 300000
                }
            ),
            
            # === ERROR: Column count ===
            ExpectationConfiguration(
                expectation_type="expect_table_column_count_to_be_between",
                kwargs={
                    "min_value": 40,   # At least this many features
                    "max_value": 200   # Not too many (duplicates removed)
                }
            )
        ]
        
        suite_name = self.ge_validator.create_expectation_suite(
            "merged_features_clean_suite", 
            expectations
        )
        
        ge_passed, ge_report = self.ge_validator.validate_dataframe(
            df, 
            suite_name, 
            "merged_features_clean",
            severity_threshold=GESeverity.CRITICAL
        )
        
        # === ROBUST VALIDATOR ===
        logger.info("\n   Running RobustValidator checks...")
        
        robust_validator = RobustValidator(
            dataset_name="merged_features_clean",
            enable_auto_fix=False,  # Should NOT need fixes at this stage
            enable_temporal_checks=True,
            enable_business_rules=True
        )
        
        df_validated, robust_report = robust_validator.validate(df)
        
        # === POST-CLEANING SPECIFIC CHECKS ===
        logger.info("\n   Running post-cleaning specific checks...")
        
        post_clean_issues = []
        
        # 1. CRITICAL: Zero inf values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        inf_count = np.isinf(df[numeric_cols]).sum().sum()
        
        if inf_count > 0:
            post_clean_issues.append({
                'severity': 'CRITICAL',
                'check': 'No Inf Values',
                'message': f"Found {inf_count} inf values (should be 0 after cleaning)"
            })
            logger.error(f"   ❌ CRITICAL: {inf_count} inf values found")
            
            # Show which columns
            inf_cols = []
            for col in numeric_cols:
                col_inf = np.isinf(df[col]).sum()
                if col_inf > 0:
                    inf_cols.append((col, col_inf))
            
            logger.error(f"   Columns with inf:")
            for col, count in sorted(inf_cols, key=lambda x: x[1], reverse=True)[:5]:
                logger.error(f"     - {col}: {count}")
        else:
            logger.info(f"   ✓ No inf values (0)")
        
        # 2. CRITICAL: No duplicate columns with suffixes
        suffix_cols = [col for col in df.columns if any(
            col.endswith(suffix) for suffix in ['_x', '_y', '_fred', '_market', '_macro', '_dup', '_fin']
        )]
        
        if suffix_cols:
            post_clean_issues.append({
                'severity': 'CRITICAL',
                'check': 'No Duplicate Columns',
                'message': f"Found {len(suffix_cols)} columns with merge suffixes"
            })
            logger.error(f"   ❌ CRITICAL: {len(suffix_cols)} duplicate columns")
            logger.error(f"   Columns: {suffix_cols[:5]}...")
        else:
            logger.info(f"   ✓ No duplicate columns (0)")
        
        # 3. ERROR: Missing values < 2%
        missing_pct = (df.isna().sum().sum() / df.size) * 100
        
        if missing_pct > 2.0:
            post_clean_issues.append({
                'severity': 'ERROR',
                'check': 'Missing Values',
                'message': f"{missing_pct:.2f}% missing (expected < 2%)"
            })
            logger.error(f"   ❌ ERROR: {missing_pct:.2f}% missing")
        elif missing_pct > 1.0:
            post_clean_issues.append({
                'severity': 'WARNING',
                'check': 'Missing Values',
                'message': f"{missing_pct:.2f}% missing (ideally < 1%)"
            })
            logger.warning(f"   ⚠️  WARNING: {missing_pct:.2f}% missing")
        else:
            logger.info(f"   ✓ Missing: {missing_pct:.2f}% (< 1%)")
        
        # 4. WARNING: Constant columns
        constant_cols = [col for col in df.columns if col not in ['Date', 'Company'] and df[col].nunique() <= 1]
        
        if constant_cols:
            post_clean_issues.append({
                'severity': 'WARNING',
                'check': 'No Constant Columns',
                'message': f"Found {len(constant_cols)} constant columns"
            })
            logger.warning(f"   ⚠️  WARNING: {len(constant_cols)} constant columns")
            logger.warning(f"   Columns: {constant_cols[:5]}...")
        else:
            logger.info(f"   ✓ No constant columns (0)")
        
        # 5. Check for each company
        logger.info(f"\n   Checking data availability per company...")
        
        companies_with_issues = []
        for company in df['Company'].unique():
            company_data = df[df['Company'] == company]
            company_missing_pct = (company_data.isna().sum().sum() / company_data.size) * 100
            
            if company_missing_pct > 5.0:
                companies_with_issues.append((company, company_missing_pct))
        
        if companies_with_issues:
            logger.warning(f"   ⚠️  {len(companies_with_issues)} companies have > 5% missing:")
            for company, pct in companies_with_issues[:5]:
                logger.warning(f"      - {company}: {pct:.1f}%")
        else:
            logger.info(f"   ✓ All companies have < 5% missing")
        
        # === FINAL DECISION ===
        critical_count = robust_report.count_by_severity()['CRITICAL']
        critical_from_post_clean = len([i for i in post_clean_issues if i['severity'] == 'CRITICAL'])
        
        robust_passed = (critical_count == 0 and critical_from_post_clean == 0)
        passed = ge_passed and robust_passed
        
        self.all_reports['merged_features_clean'] = {
            'ge_report': ge_report,
            'robust_report': robust_report.to_dict(),
            'post_clean_issues': post_clean_issues,
            'passed': passed
        }
        
        if passed:
            logger.info("\n  ✅ merged_features_clean.csv validation PASSED")
        else:
            logger.error("\n  ❌ merged_features_clean.csv validation FAILED")
            if not ge_passed:
                logger.error(f"     GE failures: {ge_report.get('critical_failures', 0)} critical")
            if not robust_passed:
                logger.error(f"     Robust failures: {critical_count} critical")
                logger.error(f"     Post-clean issues: {critical_from_post_clean} critical")
        
        return passed
    
    def run_all_validations(self) -> bool:
        """Run all cleaned merged data validations."""
        logger.info("\n" + "="*80)
        logger.info("CHECKPOINT 4: CLEAN MERGED DATA VALIDATION")
        logger.info("="*80)
        logger.info("Running AFTER: Step 3c (post-merge cleaning)")
        logger.info("Running BEFORE: Step 3b (interaction features)")
        logger.info("\nExpectations:")
        logger.info("  - 0 inf values (CRITICAL)")
        logger.info("  - < 1% missing (ERROR)")
        logger.info("  - No duplicate columns (CRITICAL)")
        logger.info("  - Valid data types (CRITICAL)")
        logger.info("  - Proper value ranges (ERROR)")
        logger.info("="*80)
        
        results = {
            'macro_clean': self.validate_macro_features_clean(),
            'merged_clean': self.validate_merged_features_clean()
        }
        
        all_passed = all(results.values())
        
        # === DETAILED SUMMARY ===
        logger.info("\n" + "="*80)
        logger.info("CHECKPOINT 4 SUMMARY")
        logger.info("="*80)
        
        for dataset_name, passed in results.items():
            status = "✅ PASSED" if passed else "❌ FAILED"
            
            # Get detailed stats
            report_key = dataset_name.replace('_clean', '_features_clean')
            if report_key in self.all_reports:
                report = self.all_reports[report_key]
                ge_rate = report['ge_report'].get('success_rate', 0)
                robust_counts = report['robust_report'].get('issue_counts', {})
                post_issues = len(report.get('post_clean_issues', []))
                
                logger.info(f"\n{dataset_name:20s}: {status}")
                logger.info(f"  GE Success Rate:   {ge_rate:.1f}%")
                logger.info(f"  Robust CRITICAL:   {robust_counts.get('CRITICAL', 0)}")
                logger.info(f"  Robust ERROR:      {robust_counts.get('ERROR', 0)}")
                logger.info(f"  Robust WARNING:    {robust_counts.get('WARNING', 0)}")
                logger.info(f"  Post-clean issues: {post_issues}")
            else:
                logger.info(f"{dataset_name:20s}: {status}")
        
        logger.info("\n" + "="*80)
        
        if all_passed:
            logger.info("✅ CHECKPOINT 4 PASSED")
            logger.info("="*80)
            logger.info("\n✓ All cleaned merged datasets validated")
            logger.info("✓ Data quality is production-ready")
            logger.info("✓ Ready for interaction feature engineering")
            logger.info("\nNext step:")
            logger.info("  python step3b_interaction_features.py")
            return True
        else:
            logger.error("❌ CHECKPOINT 4 FAILED")
            logger.error("="*80)
            logger.error("\n✗ Data quality issues found in cleaned merged data")
            logger.error("✗ Review issues above and re-run Step 3c")
            logger.error("\nTo debug:")
            logger.error("  1. Check validation reports in data/validation_reports/")
            logger.error("  2. Review post-cleaning code in step3c_post_merge_cleaning.py")
            logger.error("  3. Fix issues and re-run Step 3c")
            
            # Save detailed failure report
            self._save_failure_report()
            
            return False
    
    def _save_failure_report(self):
        """Save detailed failure report for debugging."""
        report_path = Path("data/validation_reports/checkpoint_4_failures.json")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        import json
        
        failure_report = {
            'checkpoint': 'checkpoint_4_clean_merged',
            'timestamp': pd.Timestamp.now().isoformat(),
            'datasets': {}
        }
        
        for dataset_name, report in self.all_reports.items():
            failure_report['datasets'][dataset_name] = {
                'passed': report['passed'],
                'ge_success_rate': report['ge_report'].get('success_rate', 0),
                'ge_critical_failures': report['ge_report'].get('critical_failures', 0),
                'robust_critical': report['robust_report'].get('issue_counts', {}).get('CRITICAL', 0),
                'robust_error': report['robust_report'].get('issue_counts', {}).get('ERROR', 0),
                'post_clean_issues': report.get('post_clean_issues', [])
            }
        
        with open(report_path, 'w') as f:
            json.dump(failure_report, f, indent=2)
        
        logger.info(f"\n📄 Detailed failure report saved: {report_path}")


def main():
    """Execute Checkpoint 4 validation."""
    
    validator = CleanMergedDataValidator()
    
    try:
        success = validator.run_all_validations()
        
        if success:
            logger.info("\n✅ Validation complete - Pipeline can continue")
            sys.exit(0)
        else:
            logger.error("\n❌ Validation failed - Pipeline stopped")
            logger.error("\nFix data quality issues and re-run Step 3c")
            sys.exit(1)
    
    except FileNotFoundError as e:
        logger.error(f"\n❌ Error: {e}")
        logger.error("Run Step 3c first: python step3c_post_merge_cleaning.py")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()