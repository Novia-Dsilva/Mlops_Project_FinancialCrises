"""
Great Expectations Validator Base Class
Unified validator for all pipeline stages with multi-level severity
"""

import pandas as pd
import numpy as np
import great_expectations as gx
from great_expectations.core import ExpectationConfiguration
from typing import Dict, Tuple, List
import os
import json
from datetime import datetime
from enum import Enum


class ValidationSeverity(Enum):
    """Severity levels matching RobustValidator"""
    CRITICAL = "CRITICAL"
    ERROR = "ERROR"
    WARNING = "WARNING"
    INFO = "INFO"


class GEValidatorBase:
    """
    Base class for Great Expectations validation across all pipeline stages.
    
    Integrates with RobustValidator for multi-level severity checking.
    """
    
    def __init__(self, context_root_dir: str = "great_expectations"):
        """Initialize GE context."""
        self.context_root_dir = context_root_dir
        
        if os.path.exists(context_root_dir):
            self.context = gx.get_context(context_root_dir=context_root_dir)
            print(f"✓ Loaded existing GE context")
        else:
            self.context = gx.get_context()
            print(f"✓ Created new GE context")
        
        self._setup_datasource()
        self.validation_reports = []
    
    def _setup_datasource(self):
        """Setup pandas datasource."""
        try:
            self.datasource = self.context.get_datasource("pandas_runtime")
        except:
            self.datasource = self.context.sources.add_pandas("pandas_runtime")
    
    def create_expectation_suite(
        self,
        suite_name: str,
        expectations: List[ExpectationConfiguration]
    ) -> str:
        """
        Generic method to create expectation suite.
        
        Args:
            suite_name: Name of the suite
            expectations: List of ExpectationConfiguration objects
        """
        # Delete existing suite
        try:
            self.context.delete_expectation_suite(suite_name)
        except:
            pass
        
        suite = self.context.add_expectation_suite(suite_name)
        
        # Add expectations
        for exp_config in expectations:
            suite.add_expectation(expectation_configuration=exp_config)
        
        self.context.add_or_update_expectation_suite(expectation_suite=suite)
        
        return suite_name
    
    def validate_dataframe(
        self,
        df: pd.DataFrame,
        suite_name: str,
        data_asset_name: str,
        severity_threshold: ValidationSeverity = ValidationSeverity.ERROR
    ) -> Tuple[bool, Dict]:
        """
        Validate DataFrame with severity-based decision.
        
        Args:
            df: DataFrame to validate
            suite_name: Expectation suite name
            data_asset_name: Asset name for reporting
            severity_threshold: Minimum severity to fail validation
        
        Returns:
            (is_valid, report)
        """
        print(f"\n{'='*70}")
        print(f"🔍 VALIDATING {data_asset_name.upper()}")
        print(f"{'='*70}")
        
        try:
            # Create unique asset
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            unique_asset_name = f"{data_asset_name}_{timestamp}"
            
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
            
            # Collect failed expectations with severity
            failed_expectations = []
            critical_failures = 0
            error_failures = 0
            warning_failures = 0
            
            if hasattr(results, 'results'):
                for result in results.results:
                    if not result.success:
                        exp_type = result.expectation_config.expectation_type
                        kwargs = result.expectation_config.kwargs
                        column = kwargs.get('column', 'N/A')
                        
                        # Determine severity based on expectation type
                        severity = self._determine_severity(exp_type, column)
                        
                        failed_expectations.append({
                            'expectation': exp_type,
                            'column': column,
                            'severity': severity.value,
                            'details': str(result.result)[:200]
                        })
                        
                        # Count by severity
                        if severity == ValidationSeverity.CRITICAL:
                            critical_failures += 1
                        elif severity == ValidationSeverity.ERROR:
                            error_failures += 1
                        elif severity == ValidationSeverity.WARNING:
                            warning_failures += 1
            
            # Print results
            print(f"\nValidation Results:")
            print(f"  Total expectations: {total_expectations}")
            print(f"  Successful: {successful}")
            print(f"  Failed: {failed}")
            print(f"  Success rate: {success_percent:.1f}%")
            print(f"\nFailures by Severity:")
            print(f"  CRITICAL: {critical_failures}")
            print(f"  ERROR: {error_failures}")
            print(f"  WARNING: {warning_failures}")
            
            # Determine if validation passes based on severity threshold
            validation_passed = is_valid
            if not is_valid:
                if severity_threshold == ValidationSeverity.CRITICAL:
                    validation_passed = (critical_failures == 0)
                elif severity_threshold == ValidationSeverity.ERROR:
                    validation_passed = (critical_failures == 0 and error_failures == 0)
                elif severity_threshold == ValidationSeverity.WARNING:
                    validation_passed = (critical_failures == 0 and error_failures == 0 and warning_failures == 0)
            
            # Create report
            report = {
                'is_valid': validation_passed,
                'ge_success': is_valid,
                'data_asset': data_asset_name,
                'suite_name': suite_name,
                'total_expectations': total_expectations,
                'successful': successful,
                'failed': failed,
                'success_rate': success_percent,
                'failed_expectations': failed_expectations,
                'critical_failures': critical_failures,
                'error_failures': error_failures,
                'warning_failures': warning_failures,
                'severity_threshold': severity_threshold.value,
                'timestamp': datetime.now().isoformat()
            }
            
            # Print final status
            if validation_passed:
                print(f"\n✅ VALIDATION PASSED (threshold: {severity_threshold.value})")
            else:
                print(f"\n❌ VALIDATION FAILED (threshold: {severity_threshold.value})")
            
            # Show critical failures
            if critical_failures > 0:
                print(f"\n🚨 CRITICAL FAILURES:")
                for failure in [f for f in failed_expectations if f['severity'] == 'CRITICAL'][:5]:
                    print(f"  • {failure['expectation']} on {failure['column']}")
            
            print(f"{'='*70}\n")
            
            # Save report
            self._save_report(report, data_asset_name)
            self.validation_reports.append(report)
            
            return validation_passed, report
        
        except Exception as e:
            print(f"❌ Validation error: {str(e)}")
            import traceback
            traceback.print_exc()
            
            return False, {
                'is_valid': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat(),
                'success_rate': 0.0
            }
    
    def _determine_severity(self, expectation_type: str, column: str) -> ValidationSeverity:
        """
        Determine severity based on expectation type and column.
        
        Rules:
        - Column existence: CRITICAL
        - NOT NULL on key columns: CRITICAL
        - Type checks: CRITICAL
        - Range checks: ERROR
        - Value set checks: ERROR
        - Statistical checks: WARNING
        """
        # CRITICAL: Must pass
        if 'exist' in expectation_type:
            return ValidationSeverity.CRITICAL
        
        if 'not_be_null' in expectation_type:
            key_columns = ['Date', 'Company', 'GDP', 'CPI', 'VIX', 'Stock_Price', 'Revenue']
            if column in key_columns:
                return ValidationSeverity.CRITICAL
            return ValidationSeverity.ERROR
        
        if 'type' in expectation_type.lower():
            return ValidationSeverity.CRITICAL
        
        # ERROR: Should pass
        if 'between' in expectation_type:
            return ValidationSeverity.ERROR
        
        if 'in_set' in expectation_type:
            return ValidationSeverity.ERROR
        
        # WARNING: Nice to pass
        if any(keyword in expectation_type for keyword in ['mean', 'median', 'std', 'quantile']):
            return ValidationSeverity.WARNING
        
        # Default to ERROR
        return ValidationSeverity.ERROR
    
    def _save_report(self, report: Dict, data_asset_name: str):
        """Save validation report."""
        os.makedirs('data/validation_reports', exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"data/validation_reports/ge_{data_asset_name}_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"💾 Report saved: {filename}")
    
    def get_all_reports(self) -> List[Dict]:
        """Get all validation reports from this session."""
        return self.validation_reports