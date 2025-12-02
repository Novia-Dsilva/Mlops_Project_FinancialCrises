"""
CHECKPOINT 2-3: Validate Cleaned Quarterly Data (Combined)
============================================================

Runs after Step 1 (combined cleaning + merging), before feature engineering

Validates: quarterly_data_complete.csv (ONE ROW PER QUARTER PER COMPANY)

Focus:
- Data completeness (< 5% missing for critical fields)
- No inf values (CRITICAL)
- No duplicates
- Point-in-time correctness
- Company-level data integrity
- Proper merging (allows minor cleanup of duplicate columns)
- Valid data ranges

Exit codes:
- 0: All validations passed
- 1: Critical failures detected
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


class QuarterlyDataValidator:
    """
    Checkpoint 2-3: Validate quarterly_data_complete.csv
    
    This validates the output of the combined cleaning + merging pipeline.
    ONE ROW PER QUARTER PER COMPANY with all features merged.
    """
    
    def __init__(self):
        self.processed_dir = Path("data/processed")
        self.ge_validator = GEValidatorBase()
        self.validation_report = {}
    
    def validate_quarterly_data_complete(self) -> bool:
        """
        Validate quarterly_data_complete.csv
        
        This is the main output of Step 1 (cleaning + merging).
        Should contain:
        - Company fundamentals (Balance + Income)
        - Stock prices & returns
        - FRED macro data (aggregated to quarterly)
        - Market data (VIX, SP500 aggregated to quarterly)
        """
        logger.info("\n" + "="*80)
        logger.info("VALIDATING: quarterly_data_complete.csv")
        logger.info("="*80)
        
        filepath = self.processed_dir / "quarterly_data_complete.csv"
        if not filepath.exists():
            logger.error(f"❌ File not found: {filepath}")
            logger.error("   Run Step 1 first: python step1_quarterly_cleaning.py")
            return False
        
        # Load data
        df = pd.read_csv(filepath, parse_dates=['Date'])
        
        # Ensure Date is datetime
        if 'Date' in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(df['Date']):
                df['Date'] = pd.to_datetime(df['Date'], format='mixed', errors='coerce')
        
        logger.info(f"\n📊 Dataset Overview:")
        logger.info(f"   Shape: {df.shape}")
        logger.info(f"   Companies: {df['Company'].nunique()}")
        logger.info(f"   Quarters: {len(df):,}")
        logger.info(f"   Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")
        
        # =================================================================
        # AUTO-FIX: Remove duplicate/constant columns before validation
        # =================================================================
        logger.info("\n" + "="*80)
        logger.info("[AUTO-FIX] Cleaning up duplicate/constant columns...")
        logger.info("="*80)
        
        original_cols = df.shape[1]
        
        # Remove duplicate columns with merge suffixes
        duplicate_cols = [col for col in df.columns if any(
            suffix in col for suffix in 
            ['_fred', '_market', '_macro', '_bal', '_inc', '_px', '_mkt', '_x', '_y']
        )]
        
        if duplicate_cols:
            logger.info(f"   Found {len(duplicate_cols)} duplicate columns with merge suffixes")
            df = df.drop(columns=duplicate_cols)
            logger.info(f"   ✓ Removed {len(duplicate_cols)} columns")
        
        # Remove constant columns (excluding metadata)
        metadata_cols = ['Date', 'Quarter', 'Quarter_Num', 'Year', 'Company', 'Company_Name', 'Sector', 'Original_Quarter_End']
        constant_cols = []
        
        for col in df.columns:
            if col not in metadata_cols:
                if df[col].nunique() <= 1:
                    constant_cols.append(col)
        
        if constant_cols:
            logger.info(f"   Found {len(constant_cols)} constant columns:")
            for col in constant_cols[:5]:
                logger.info(f"      - {col}")
            df = df.drop(columns=constant_cols)
            logger.info(f"   ✓ Removed {len(constant_cols)} columns")
        
        # Handle duplicate Quarter_End_Date columns
        quarter_end_cols = [col for col in df.columns if 'Quarter_End_Date' in col]
        if len(quarter_end_cols) > 1:
            cols_to_remove = [col for col in quarter_end_cols if col != 'Quarter_End_Date']
            if cols_to_remove:
                df = df.drop(columns=cols_to_remove)
                logger.info(f"   ✓ Kept 'Quarter_End_Date', removed {len(cols_to_remove)} duplicates")
        
        cleaned_cols = df.shape[1]
        if original_cols != cleaned_cols:
            logger.info(f"\n   📊 Columns: {original_cols} → {cleaned_cols} ({original_cols - cleaned_cols} removed)")
            
            # Save cleaned version
            logger.info(f"   💾 Saving cleaned version...")
            df.to_csv(filepath, index=False)
            logger.info(f"   ✓ Saved: {filepath}")
        else:
            logger.info(f"   ✓ No cleanup needed")
        
        # =================================================================
        # SECTION 1: GREAT EXPECTATIONS - Schema & Ranges
        # =================================================================
        logger.info("\n" + "="*80)
        logger.info("[1/4] GREAT EXPECTATIONS VALIDATION")
        logger.info("="*80)
        
        expectations = [
            # === CRITICAL: Core columns existence ===
            ExpectationConfiguration(
                expectation_type="expect_column_to_exist",
                kwargs={"column": "Date"}
            ),
            ExpectationConfiguration(
                expectation_type="expect_column_to_exist",
                kwargs={"column": "Quarter"}
            ),
            ExpectationConfiguration(
                expectation_type="expect_column_to_exist",
                kwargs={"column": "Company"}
            ),
            
            # === CRITICAL: Fundamentals columns ===
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
                kwargs={"column": "Total_Assets"}
            ),
            ExpectationConfiguration(
                expectation_type="expect_column_to_exist",
                kwargs={"column": "Total_Liabilities"}
            ),
            
            # === CRITICAL: Stock metrics ===
            ExpectationConfiguration(
                expectation_type="expect_column_to_exist",
                kwargs={"column": "q_return"}
            ),
            ExpectationConfiguration(
                expectation_type="expect_column_to_exist",
                kwargs={"column": "q_price"}
            ),
            
            # === CRITICAL: Macro data ===
            ExpectationConfiguration(
                expectation_type="expect_column_to_exist",
                kwargs={"column": "GDP_last"}
            ),
            ExpectationConfiguration(
                expectation_type="expect_column_to_exist",
                kwargs={"column": "Unemployment_Rate_last"}
            ),
            
            # === CRITICAL: Market data ===
            ExpectationConfiguration(
                expectation_type="expect_column_to_exist",
                kwargs={"column": "vix_q_mean"}
            ),
            ExpectationConfiguration(
                expectation_type="expect_column_to_exist",
                kwargs={"column": "sp500_q_return"}
            ),
            
            # === CRITICAL: No duplicate (Date, Company) pairs ===
            ExpectationConfiguration(
                expectation_type="expect_compound_columns_to_be_unique",
                kwargs={"column_list": ["Date", "Company"]}
            ),
            
            # === CRITICAL: Company column completeness ===
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_not_be_null",
                kwargs={"column": "Company"}
            ),
            
            # === ERROR: Key columns completeness (< 5% missing) ===
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_not_be_null",
                kwargs={
                    "column": "Revenue",
                    "mostly": 0.95
                }
            ),
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_not_be_null",
                kwargs={
                    "column": "Total_Assets",
                    "mostly": 0.95
                }
            ),
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_not_be_null",
                kwargs={
                    "column": "q_return",
                    "mostly": 0.90  # Allow some missing in early periods
                }
            ),
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_not_be_null",
                kwargs={
                    "column": "GDP_last",
                    "mostly": 0.95
                }
            ),
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_not_be_null",
                kwargs={
                    "column": "vix_q_mean",
                    "mostly": 0.95
                }
            ),
            
            # === ERROR: Value ranges for key metrics ===
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_be_between",
                kwargs={
                    "column": "Revenue",
                    "min_value": 0,
                    "max_value": 1e12,
                    "mostly": 0.95
                }
            ),
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_be_between",
                kwargs={
                    "column": "Total_Assets",
                    "min_value": 1e6,
                    "max_value": 1e13,
                    "mostly": 0.90
                }
            ),
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_be_between",
                kwargs={
                    "column": "q_return",
                    "min_value": -100,
                    "max_value": 500,
                    "mostly": 0.99
                }
            ),
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_be_between",
                kwargs={
                    "column": "GDP_last",
                    "min_value": 5000,
                    "max_value": 35000,
                    "mostly": 0.95
                }
            ),
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_be_between",
                kwargs={
                    "column": "vix_q_mean",
                    "min_value": 5,
                    "max_value": 100,
                    "mostly": 0.99
                }
            ),
            
            # === ERROR: Company count ===
            ExpectationConfiguration(
                expectation_type="expect_column_unique_value_count_to_be_between",
                kwargs={
                    "column": "Company",
                    "min_value": 50,
                    "max_value": 150
                }
            ),
            
            # === ERROR: Row count ===
            ExpectationConfiguration(
                expectation_type="expect_table_row_count_to_be_between",
                kwargs={
                    "min_value": 5000,
                    "max_value": 100000
                }
            ),
            
            # === ERROR: Column count ===
            ExpectationConfiguration(
                expectation_type="expect_table_column_count_to_be_between",
                kwargs={
                    "min_value": 50,
                    "max_value": 200
                }
            )
        ]
        
        suite_name = self.ge_validator.create_expectation_suite(
            "quarterly_data_complete_suite",
            expectations
        )
        
        ge_passed, ge_report = self.ge_validator.validate_dataframe(
            df,
            suite_name,
            "quarterly_data_complete",
            severity_threshold=GESeverity.CRITICAL
        )
        
        logger.info(f"\n   GE Results:")
        logger.info(f"   Success Rate: {ge_report.get('success_rate', 0):.1f}%")
        logger.info(f"   Critical Failures: {ge_report.get('critical_failures', 0)}")
        logger.info(f"   Total Failures: {ge_report.get('total_failures', 0)}")
        
        # =================================================================
        # SECTION 2: ROBUST VALIDATOR - Business Logic & Anomalies
        # =================================================================
        logger.info("\n" + "="*80)
        logger.info("[2/4] ROBUST VALIDATOR CHECKS")
        logger.info("="*80)
        
        robust_validator = RobustValidator(
            dataset_name="quarterly_data_complete",
            enable_auto_fix=False,
            enable_temporal_checks=True,
            enable_business_rules=True
        )
        
        df_validated, robust_report = robust_validator.validate(df)
        
        robust_counts = robust_report.count_by_severity()
        logger.info(f"\n   Robust Results:")
        logger.info(f"   CRITICAL: {robust_counts['CRITICAL']}")
        logger.info(f"   ERROR:    {robust_counts['ERROR']}")
        logger.info(f"   WARNING:  {robust_counts['WARNING']}")
        logger.info(f"   INFO:     {robust_counts['INFO']}")
        
        # =================================================================
        # SECTION 3: POST-CLEANING SPECIFIC CHECKS
        # =================================================================
        logger.info("\n" + "="*80)
        logger.info("[3/4] POST-CLEANING SPECIFIC CHECKS")
        logger.info("="*80)
        
        post_clean_issues = []
        
        # Check 1: Inf values (CRITICAL - should be 0)
        logger.info("\n   [3.1] Checking for inf values...")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        inf_count = np.isinf(df[numeric_cols]).sum().sum()
        
        if inf_count > 0:
            post_clean_issues.append({
                'severity': 'CRITICAL',
                'check': 'No Inf Values',
                'message': f"Found {inf_count} inf values"
            })
            logger.error(f"   ❌ CRITICAL: {inf_count} inf values found")
            
            # Show which columns
            inf_cols = []
            for col in numeric_cols:
                col_inf = np.isinf(df[col]).sum()
                if col_inf > 0:
                    inf_cols.append((col, col_inf))
            
            for col, count in sorted(inf_cols, key=lambda x: x[1], reverse=True)[:5]:
                logger.error(f"      - {col}: {count}")
        else:
            logger.info(f"   ✓ No inf values (0)")
        
        # Check 2: Duplicate columns (INFO only - we auto-fixed)
        logger.info("\n   [3.2] Checking for duplicate columns...")
        suffix_cols = [col for col in df.columns if any(
            suffix in col for suffix in 
            ['_fred', '_market', '_macro', '_bal', '_inc', '_px', '_mkt', '_x', '_y']
        )]
        
        if suffix_cols:
            logger.info(f"   ℹ️  Note: {len(suffix_cols)} merge suffix columns (auto-fixed)")
        else:
            logger.info(f"   ✓ No duplicate columns (0)")
        
        # Check 3: Missing data percentage (WARNING if > 5%)
        logger.info("\n   [3.3] Checking missing data...")
        missing_pct = (df.isna().sum().sum() / df.size) * 100
        
        if missing_pct > 5.0:
            post_clean_issues.append({
                'severity': 'WARNING',
                'check': 'Missing Data',
                'message': f"{missing_pct:.2f}% missing (expected < 5%)"
            })
            logger.warning(f"   ⚠️  WARNING: {missing_pct:.2f}% missing")
        elif missing_pct > 2.0:
            logger.info(f"   ℹ️  Missing: {missing_pct:.2f}% (acceptable)")
        else:
            logger.info(f"   ✓ Missing: {missing_pct:.2f}% (< 2%)")
        
        # Show top missing columns
        missing_by_col = (df.isna().sum() / len(df) * 100).sort_values(ascending=False).head(10)
        if (missing_by_col > 5).any():
            logger.info(f"\n   Top columns with missing data:")
            for col, pct in missing_by_col.items():
                if pct > 5:
                    logger.info(f"   ⚠️ {col:30s}: {pct:5.1f}%")
        
        # Check 4: Date monotonicity per company (ERROR)
        logger.info("\n   [3.4] Checking date ordering per company...")
        date_issues = []
        
        for company in df['Company'].unique():
            company_dates = df[df['Company'] == company]['Date'].sort_values()
            if not company_dates.is_monotonic_increasing:
                date_issues.append(company)
        
        if date_issues:
            post_clean_issues.append({
                'severity': 'ERROR',
                'check': 'Date Monotonicity',
                'message': f"{len(date_issues)} companies have non-monotonic dates"
            })
            logger.error(f"   ❌ ERROR: {len(date_issues)} companies have date issues")
            logger.error(f"   Companies: {date_issues[:5]}")
        else:
            logger.info(f"   ✓ All companies have monotonic dates")
        
        # =================================================================
        # SECTION 4: COMPANY-LEVEL DATA INTEGRITY
        # =================================================================
        logger.info("\n" + "="*80)
        logger.info("[4/4] COMPANY-LEVEL DATA INTEGRITY")
        logger.info("="*80)
        
        logger.info(f"\n   [4.1] Data availability per company...")
        
        company_stats = []
        for company in df['Company'].unique():
            company_data = df[df['Company'] == company]
            quarters = len(company_data)
            missing_pct = (company_data.isna().sum().sum() / company_data.size) * 100
            
            company_stats.append({
                'company': company,
                'quarters': quarters,
                'missing_pct': missing_pct
            })
        
        company_stats_df = pd.DataFrame(company_stats)
        
        # Companies with issues
        companies_low_data = company_stats_df[company_stats_df['quarters'] < 20]
        companies_high_missing = company_stats_df[company_stats_df['missing_pct'] > 10]
        
        if len(companies_low_data) > 0:
            logger.info(f"\n   ℹ️  {len(companies_low_data)} companies have < 20 quarters (acceptable)")
        
        if len(companies_high_missing) > 0:
            logger.warning(f"\n   ⚠️  {len(companies_high_missing)} companies have > 10% missing:")
            for _, row in companies_high_missing.head(5).iterrows():
                logger.warning(f"      - {row['company']:6s}: {row['missing_pct']:.1f}%")
        
        if len(companies_low_data) == 0 and len(companies_high_missing) == 0:
            logger.info(f"   ✓ All companies have adequate data")
        
        # Summary stats
        logger.info(f"\n   Summary:")
        logger.info(f"   Average quarters per company: {company_stats_df['quarters'].mean():.1f}")
        logger.info(f"   Average missing per company:  {company_stats_df['missing_pct'].mean():.1f}%")
        logger.info(f"   Companies with full data:     {len(company_stats_df[company_stats_df['missing_pct'] < 5])}")
        
        # =================================================================
        # FINAL DECISION
        # =================================================================
        logger.info("\n" + "="*80)
        logger.info("VALIDATION SUMMARY")
        logger.info("="*80)
        
        # Count critical issues only
        critical_from_post_clean = len([i for i in post_clean_issues if i['severity'] == 'CRITICAL'])
        
        # Decision logic - only CRITICAL issues fail validation
        ge_critical_pass = ge_report.get('critical_failures', 0) == 0
        robust_critical_pass = robust_counts['CRITICAL'] == 0
        post_clean_critical_pass = critical_from_post_clean == 0
        
        passed = ge_critical_pass and robust_critical_pass and post_clean_critical_pass
        
        # Store report
        self.validation_report = {
            'dataset': 'quarterly_data_complete.csv',
            'passed': passed,
            'ge_report': ge_report,
            'robust_report': robust_report.to_dict(),
            'post_clean_issues': post_clean_issues,
            'company_stats': company_stats_df.to_dict('records')
        }
        
        # Print summary
        logger.info(f"\n   Great Expectations:")
        logger.info(f"   - Success Rate:      {ge_report.get('success_rate', 0):.1f}%")
        logger.info(f"   - Critical Failures: {ge_report.get('critical_failures', 0)}")
        
        logger.info(f"\n   Robust Validator:")
        logger.info(f"   - CRITICAL: {robust_counts['CRITICAL']}")
        logger.info(f"   - ERROR:    {robust_counts['ERROR']}")
        logger.info(f"   - WARNING:  {robust_counts['WARNING']}")
        
        logger.info(f"\n   Post-Clean Checks:")
        logger.info(f"   - CRITICAL: {critical_from_post_clean}")
        
        logger.info("\n" + "="*80)
        
        if passed:
            logger.info("✅ CHECKPOINT 2-3 PASSED")
            logger.info("="*80)
            logger.info("\n✓ quarterly_data_complete.csv is production-ready")
            logger.info("✓ Data quality meets all critical requirements")
            logger.info("✓ Ready for feature engineering")
            logger.info("\nNext step:")
            logger.info("  python step2_feature_engineering.py")
            return True
        else:
            logger.error("❌ CHECKPOINT 2-3 FAILED")
            logger.error("="*80)
            logger.error("\n✗ Critical data quality issues found")
            
            if not ge_critical_pass:
                logger.error(f"✗ GE: {ge_report.get('critical_failures', 0)} critical failures")
            if not robust_critical_pass:
                logger.error(f"✗ Robust: {robust_counts['CRITICAL']} critical issues")
            if not post_clean_critical_pass:
                logger.error(f"✗ Post-clean: {critical_from_post_clean} critical issues")
            
            logger.error("\nTo debug:")
            logger.error("  1. Check validation reports in data/validation_reports/")
            logger.error("  2. Review critical issues above")
            logger.error("  3. Re-run Step 1: python step1_quarterly_cleaning.py")
            
            # Save failure report
            self._save_failure_report()
            
            return False
    
    def _save_failure_report(self):
        """Save detailed failure report for debugging."""
        report_path = Path("data/validation_reports/checkpoint_2_3_failures.json")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        import json
        
        failure_report = {
            'checkpoint': 'checkpoint_2_3_quarterly_data',
            'timestamp': pd.Timestamp.now().isoformat(),
            'validation_report': self.validation_report
        }
        
        with open(report_path, 'w') as f:
            json.dump(failure_report, f, indent=2, default=str)
        
        logger.info(f"\n📄 Detailed failure report saved: {report_path}")


def main():
    """Execute Checkpoint 2-3 validation."""
    
    logger.info("\n" + "="*80)
    logger.info("CHECKPOINT 2-3: VALIDATE CLEANED QUARTERLY DATA")
    logger.info("="*80)
    logger.info("\nRuns AFTER:  Step 1 (quarterly cleaning + merging)")
    logger.info("Runs BEFORE: Step 2 (feature engineering)")
    logger.info("\nValidates: quarterly_data_complete.csv")
    logger.info("Format: ONE ROW PER QUARTER PER COMPANY")
    logger.info("\nChecks:")
    logger.info("  ✓ Schema & column existence")
    logger.info("  ✓ Data completeness (< 5% missing)")
    logger.info("  ✓ No inf values")
    logger.info("  ✓ No duplicates")
    logger.info("  ✓ Valid ranges")
    logger.info("  ✓ Company-level integrity")
    logger.info("  ✓ Auto-fixes minor issues (duplicate columns)")
    logger.info("="*80)
    
    validator = QuarterlyDataValidator()
    
    try:
        success = validator.validate_quarterly_data_complete()
        
        if success:
            logger.info("\n✅ Validation complete - Pipeline can continue")
            sys.exit(0)
        else:
            logger.error("\n❌ Validation failed - Pipeline stopped")
            logger.error("\nFix data quality issues and re-run Step 1")
            sys.exit(1)
    
    except FileNotFoundError as e:
        logger.error(f"\n❌ Error: {e}")
        logger.error("Run Step 1 first: python step1_quarterly_cleaning.py")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()