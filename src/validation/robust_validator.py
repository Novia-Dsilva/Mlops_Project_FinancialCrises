"""
Robust Validator - Production-Grade Data Validation Engine

Multi-level validation with:
- CRITICAL: Must pass (blocks pipeline)
- ERROR: Should pass (proceed with caution)
- WARNING: Nice to pass (log and continue)
- INFO: Informational only

Features:
- Automated remediation
- Temporal anomaly detection
- Business rule validation
- Historical metrics tracking
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import warnings
from scipy import stats

warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# VALIDATION SEVERITY LEVELS
# ============================================================================

class ValidationSeverity(Enum):
    """Severity levels for validation failures."""
    CRITICAL = "CRITICAL"  # Must fix - blocks pipeline
    ERROR = "ERROR"        # Should fix - proceed with caution
    WARNING = "WARNING"    # Nice to fix - log and continue
    INFO = "INFO"          # Informational - no action needed


# ============================================================================
# VALIDATION RESULT CLASSES
# ============================================================================

@dataclass
class ValidationIssue:
    """Single validation issue."""
    severity: ValidationSeverity
    check_name: str
    message: str
    dataset: str
    column: Optional[str] = None
    row_indices: Optional[List[int]] = None
    value: Optional[Any] = None
    expected: Optional[Any] = None
    timestamp: datetime = field(default_factory=datetime.now)
    auto_fixable: bool = False
    fix_applied: bool = False
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'severity': self.severity.value,
            'check_name': self.check_name,
            'message': self.message,
            'dataset': self.dataset,
            'column': self.column,
            'affected_rows': len(self.row_indices) if self.row_indices else 0,
            'value': str(self.value) if self.value is not None else None,
            'expected': str(self.expected) if self.expected is not None else None,
            'timestamp': self.timestamp.isoformat(),
            'auto_fixable': self.auto_fixable,
            'fix_applied': self.fix_applied
        }


@dataclass
class ValidationReport:
    """Comprehensive validation report."""
    dataset_name: str
    validation_timestamp: datetime
    passed: bool
    issues: List[ValidationIssue] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    data_snapshot: Dict[str, Any] = field(default_factory=dict)
    remediation_log: List[str] = field(default_factory=list)
    
    def add_issue(self, issue: ValidationIssue):
        """Add validation issue."""
        self.issues.append(issue)
        
        # Update passed status
        if issue.severity == ValidationSeverity.CRITICAL:
            self.passed = False
    
    def count_by_severity(self) -> Dict[str, int]:
        """Count issues by severity."""
        counts = {
            'CRITICAL': 0,
            'ERROR': 0,
            'WARNING': 0,
            'INFO': 0
        }
        
        for issue in self.issues:
            counts[issue.severity.value] += 1
        
        return counts
    
    def get_critical_issues(self) -> List[ValidationIssue]:
        """Get critical issues that must be fixed."""
        return [i for i in self.issues if i.severity == ValidationSeverity.CRITICAL]
    
    def get_auto_fixable_issues(self) -> List[ValidationIssue]:
        """Get issues that can be auto-fixed."""
        return [i for i in self.issues if i.auto_fixable and not i.fix_applied]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'dataset_name': self.dataset_name,
            'validation_timestamp': self.validation_timestamp.isoformat(),
            'passed': self.passed,
            'issue_counts': self.count_by_severity(),
            'total_issues': len(self.issues),
            'issues': [issue.to_dict() for issue in self.issues],
            'metrics': self.metrics,
            'data_snapshot': self.data_snapshot,
            'remediation_log': self.remediation_log
        }
    
    def save(self, output_dir: Path):
        """Save report to file."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        filename = f"robust_validation_{self.dataset_name}_{self.validation_timestamp.strftime('%Y%m%d_%H%M%S')}.json"
        filepath = output_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        
        return filepath


# ============================================================================
# ROBUST VALIDATOR ENGINE
# ============================================================================

class RobustValidator:
    """
    Production-grade validator with multi-level checks and auto-remediation.
    """
    
    def __init__(self, 
                 dataset_name: str,
                 enable_auto_fix: bool = True,
                 enable_temporal_checks: bool = True,
                 enable_business_rules: bool = True):
        
        self.dataset_name = dataset_name
        self.enable_auto_fix = enable_auto_fix
        self.enable_temporal_checks = enable_temporal_checks
        self.enable_business_rules = enable_business_rules
        
        self.report = ValidationReport(
            dataset_name=dataset_name,
            validation_timestamp=datetime.now(),
            passed=True
        )
        
        self.logger = logging.getLogger(f"RobustValidator.{dataset_name}")
        
        # Historical metrics for comparison
        self.historical_metrics_file = Path(f"data/validation_history/{dataset_name}_metrics.json")
        self.historical_metrics = self._load_historical_metrics()
    
    # ========== CORE VALIDATION METHODS ==========
    
    def validate(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, ValidationReport]:
        """
        Run complete validation suite.
        
        Returns:
            (cleaned_df, validation_report)
        """
        self.logger.info("="*80)
        self.logger.info(f"ROBUST VALIDATION: {self.dataset_name}")
        self.logger.info("="*80)
        
        original_df = df.copy()
        
        # Store data snapshot
        self._capture_data_snapshot(df)
        
        # LEVEL 1: CRITICAL CHECKS (Must pass)
        self.logger.info("\n[LEVEL 1] CRITICAL checks...")
        df = self._run_critical_checks(df)
        
        # LEVEL 2: ERROR CHECKS (Should pass)
        self.logger.info("\n[LEVEL 2] ERROR checks...")
        df = self._run_error_checks(df)
        
        # LEVEL 3: WARNING CHECKS (Nice to pass)
        self.logger.info("\n[LEVEL 3] WARNING checks...")
        df = self._run_warning_checks(df)
        
        # LEVEL 4: INFO CHECKS (Informational)
        self.logger.info("\n[LEVEL 4] INFO checks...")
        df = self._run_info_checks(df)
        
        # TEMPORAL VALIDATION
        if self.enable_temporal_checks and 'Date' in df.columns:
            self.logger.info("\n[TEMPORAL] Time-series checks...")
            df = self._run_temporal_checks(df)
        
        # BUSINESS LOGIC VALIDATION
        if self.enable_business_rules:
            self.logger.info("\n[BUSINESS] Business logic checks...")
            df = self._run_business_checks(df)
        
        # AUTO-REMEDIATION
        if self.enable_auto_fix:
            self.logger.info("\n[AUTO-FIX] Automatic remediation...")
            df = self._auto_remediate(df)
        
        # FINAL SUMMARY
        self._print_summary()
        
        # Save historical metrics
        self._save_current_metrics()
        
        return df, self.report
    
    # ========== LEVEL 1: CRITICAL CHECKS ==========
    
    def _run_critical_checks(self, df: pd.DataFrame) -> pd.DataFrame:
        """CRITICAL checks - pipeline MUST stop if these fail."""
        
        # 1. DataFrame not empty
        if len(df) == 0:
            self.report.add_issue(ValidationIssue(
                severity=ValidationSeverity.CRITICAL,
                check_name="DataFrame Not Empty",
                message="DataFrame is completely empty",
                dataset=self.dataset_name,
                auto_fixable=False
            ))
            return df
        
        # 2. Required columns exist
        required_cols = self._get_required_columns()
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            self.report.add_issue(ValidationIssue(
                severity=ValidationSeverity.CRITICAL,
                check_name="Required Columns Exist",
                message=f"Missing required columns: {missing_cols}",
                dataset=self.dataset_name,
                expected=required_cols,
                value=list(df.columns),
                auto_fixable=False
            ))
        
        # 3. Date column is datetime
        if 'Date' in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(df['Date']):
                self.report.add_issue(ValidationIssue(
                    severity=ValidationSeverity.CRITICAL,
                    check_name="Date Column Type",
                    message="Date column is not datetime type",
                    dataset=self.dataset_name,
                    column='Date',
                    auto_fixable=True
                ))
        
        # 4. No completely null columns
        null_cols = df.columns[df.isnull().all()].tolist()
        if null_cols:
            self.report.add_issue(ValidationIssue(
                severity=ValidationSeverity.CRITICAL,
                check_name="No Null Columns",
                message=f"Completely null columns found: {null_cols}",
                dataset=self.dataset_name,
                value=null_cols,
                auto_fixable=True
            ))
        
        # 5. No inf values in numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            inf_count = np.isinf(df[col]).sum()
            if inf_count > 0:
                inf_indices = df[np.isinf(df[col])].index.tolist()
                self.report.add_issue(ValidationIssue(
                    severity=ValidationSeverity.CRITICAL,
                    check_name="No Inf Values",
                    message=f"Column {col} contains {inf_count} inf values",
                    dataset=self.dataset_name,
                    column=col,
                    row_indices=inf_indices,
                    auto_fixable=True
                ))
        
        return df
    
    # ========== LEVEL 2: ERROR CHECKS ==========
    
    def _run_error_checks(self, df: pd.DataFrame) -> pd.DataFrame:
        """ERROR checks - serious issues that should be fixed."""
        
        # 1. Duplicate rows
        if 'Date' in df.columns and 'Company' in df.columns:
            duplicates = df.duplicated(subset=['Date', 'Company'], keep=False)
            dup_count = duplicates.sum()
            
            if dup_count > 0:
                dup_indices = df[duplicates].index.tolist()
                self.report.add_issue(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    check_name="No Duplicate Rows",
                    message=f"Found {dup_count} duplicate (Date, Company) pairs",
                    dataset=self.dataset_name,
                    row_indices=dup_indices,
                    auto_fixable=True
                ))
        elif 'Date' in df.columns:
            duplicates = df.duplicated(subset=['Date'], keep=False)
            dup_count = duplicates.sum()
            
            if dup_count > 0:
                self.report.add_issue(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    check_name="No Duplicate Dates",
                    message=f"Found {dup_count} duplicate dates",
                    dataset=self.dataset_name,
                    auto_fixable=True
                ))
        
        # 2. Missing values in key columns
        key_columns = self._get_key_columns()
        for col in key_columns:
            if col in df.columns:
                missing_count = df[col].isnull().sum()
                missing_pct = (missing_count / len(df)) * 100
                
                if missing_pct > 5:
                    missing_indices = df[df[col].isnull()].index.tolist()
                    self.report.add_issue(ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        check_name="Key Column Completeness",
                        message=f"Key column {col} has {missing_pct:.1f}% missing values",
                        dataset=self.dataset_name,
                        column=col,
                        row_indices=missing_indices,
                        expected="< 5% missing",
                        value=f"{missing_pct:.1f}%",
                        auto_fixable=True
                    ))
        
        # 3. Value range violations
        range_checks = self._get_range_checks()
        for col, (min_val, max_val) in range_checks.items():
            if col in df.columns:
                out_of_range = (df[col] < min_val) | (df[col] > max_val)
                out_count = out_of_range.sum()
                
                if out_count > 0:
                    out_indices = df[out_of_range].index.tolist()
                    self.report.add_issue(ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        check_name="Value Range Check",
                        message=f"Column {col} has {out_count} values outside [{min_val}, {max_val}]",
                        dataset=self.dataset_name,
                        column=col,
                        row_indices=out_indices,
                        expected=f"[{min_val}, {max_val}]",
                        value=f"min={df[col].min():.2f}, max={df[col].max():.2f}",
                        auto_fixable=True
                    ))
        
        return df
    
    # ========== LEVEL 3: WARNING CHECKS ==========
    
    def _run_warning_checks(self, df: pd.DataFrame) -> pd.DataFrame:
        """WARNING checks - issues to be aware of but not blocking."""
        
        # 1. High missing percentage
        for col in df.columns:
            missing_pct = (df[col].isnull().sum() / len(df)) * 100
            
            if missing_pct > 20 and col not in self._get_key_columns():
                self.report.add_issue(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    check_name="Missing Value Check",
                    message=f"Column {col} has {missing_pct:.1f}% missing values",
                    dataset=self.dataset_name,
                    column=col,
                    value=f"{missing_pct:.1f}%",
                    auto_fixable=False
                ))
        
        # 2. Outliers
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].notna().sum() < 10:
                continue
                
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR
            
            outliers = ((df[col] < lower_bound) | (df[col] > upper_bound))
            outlier_count = outliers.sum()
            outlier_pct = (outlier_count / len(df)) * 100
            
            if outlier_pct > 1:
                self.report.add_issue(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    check_name="Outlier Detection",
                    message=f"Column {col} has {outlier_pct:.1f}% outliers",
                    dataset=self.dataset_name,
                    column=col,
                    value=f"{outlier_pct:.1f}% ({outlier_count} values)",
                    auto_fixable=False
                ))
        
        return df
    
    # ========== LEVEL 4: INFO CHECKS ==========
    
    def _run_info_checks(self, df: pd.DataFrame) -> pd.DataFrame:
        """INFO checks - informational, no action needed."""
        
        # Dataset size
        memory_mb = df.memory_usage(deep=True).sum() / 1024**2
        
        self.report.add_issue(ValidationIssue(
            severity=ValidationSeverity.INFO,
            check_name="Dataset Size",
            message=f"Dataset size: {len(df):,} rows × {len(df.columns)} columns ({memory_mb:.2f} MB)",
            dataset=self.dataset_name,
            auto_fixable=False
        ))
        
        # Date range
        if 'Date' in df.columns:
            date_range_days = (df['Date'].max() - df['Date'].min()).days
            
            self.report.add_issue(ValidationIssue(
                severity=ValidationSeverity.INFO,
                check_name="Date Range",
                message=f"Date range: {df['Date'].min()} to {df['Date'].max()} ({date_range_days} days)",
                dataset=self.dataset_name,
                auto_fixable=False
            ))
        
        return df
    
    # ========== TEMPORAL VALIDATION ==========
    
    def _run_temporal_checks(self, df: pd.DataFrame) -> pd.DataFrame:
        """Time-series specific validation."""
        
        if 'Company' in df.columns:
            # Check per company
            for company in df['Company'].unique():
                company_df = df[df['Company'] == company].copy()
                df = self._check_temporal_anomalies(company_df, df, group=company)
        else:
            # Check entire dataset
            df = self._check_temporal_anomalies(df, df)
        
        return df
    
    def _check_temporal_anomalies(self, subset_df: pd.DataFrame, 
                                   full_df: pd.DataFrame, 
                                   group: Optional[str] = None) -> pd.DataFrame:
        """Detect time-series anomalies."""
        
        subset_df = subset_df.sort_values('Date')
        
        # Date gaps
        if len(subset_df) > 1:
            date_diffs = subset_df['Date'].diff()
            max_gap = date_diffs.max().days
            
            if max_gap > 30:
                self.report.add_issue(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    check_name="Date Gap Check",
                    message=f"{'Company ' + group + ': ' if group else ''}Maximum date gap is {max_gap} days",
                    dataset=self.dataset_name,
                    column='Date',
                    value=f"{max_gap} days",
                    expected="< 30 days",
                    auto_fixable=False
                ))
        
        return full_df
    
    # ========== BUSINESS LOGIC VALIDATION ==========
    
    def _run_business_checks(self, df: pd.DataFrame) -> pd.DataFrame:
        """Domain-specific business rules validation."""
        
        # Financial ratios make sense
        if 'Debt_to_Equity' in df.columns:
            extreme_leverage = df['Debt_to_Equity'] > 100
            if extreme_leverage.any():
                self.report.add_issue(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    check_name="Business Rule: Leverage",
                    message=f"Debt-to-Equity > 100 for {extreme_leverage.sum()} observations",
                    dataset=self.dataset_name,
                    column='Debt_to_Equity',
                    auto_fixable=False
                ))
        
        return df
    
    # ========== AUTO-REMEDIATION ==========
    
    def _auto_remediate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Automatically fix issues that are safe to fix."""
        
        fixable_issues = self.report.get_auto_fixable_issues()
        
        if not fixable_issues:
            self.logger.info("No auto-fixable issues found")
            return df
        
        self.logger.info(f"Found {len(fixable_issues)} auto-fixable issues")
        
        for issue in fixable_issues:
            try:
                if issue.check_name == "Date Column Type":
                    df['Date'] = pd.to_datetime(df['Date'])
                    issue.fix_applied = True
                    self.report.remediation_log.append("Converted Date column to datetime")
                
                elif issue.check_name == "No Null Columns" and issue.value:
                    df = df.drop(columns=issue.value)
                    issue.fix_applied = True
                    self.report.remediation_log.append(f"Dropped null columns: {issue.value}")
                
                elif issue.check_name == "No Inf Values" and issue.column:
                    df[issue.column] = df[issue.column].replace([np.inf, -np.inf], np.nan)
                    issue.fix_applied = True
                    self.report.remediation_log.append(f"Replaced inf in {issue.column}")
                
                elif issue.check_name == "No Duplicate Rows":
                    if 'Company' in df.columns:
                        df = df.drop_duplicates(subset=['Date', 'Company'], keep='first')
                    else:
                        df = df.drop_duplicates(subset=['Date'], keep='first')
                    issue.fix_applied = True
                    self.report.remediation_log.append("Removed duplicates")
                
                self.logger.info(f"  ✓ Fixed: {issue.check_name}")
                
            except Exception as e:
                self.logger.error(f"  ✗ Failed to fix {issue.check_name}: {e}")
        
        return df
    
    # ========== HELPER METHODS ==========
    
    def _get_required_columns(self) -> List[str]:
        """Get required columns based on dataset type."""
        if 'macro' in self.dataset_name.lower():
            return ['Date', 'GDP', 'CPI']
        elif 'market' in self.dataset_name.lower():
            return ['Date', 'VIX']
        elif 'company' in self.dataset_name.lower() or 'merged' in self.dataset_name.lower():
            return ['Date', 'Company']
        else:
            return ['Date']
    
    def _get_key_columns(self) -> List[str]:
        """Get key columns that should have minimal missing values."""
        if 'macro' in self.dataset_name.lower():
            return ['Date', 'GDP', 'CPI', 'Unemployment_Rate']
        elif 'market' in self.dataset_name.lower():
            return ['Date', 'VIX', 'SP500']
        elif 'price' in self.dataset_name.lower():
            return ['Date', 'Company', 'Close', 'Stock_Price']
        elif 'balance' in self.dataset_name.lower():
            return ['Date', 'Company', 'Total_Assets']
        elif 'income' in self.dataset_name.lower():
            return ['Date', 'Company', 'Revenue']
        else:
            return ['Date']
    
    def _get_range_checks(self) -> Dict[str, Tuple[float, float]]:
        """Get valid ranges for numeric columns."""
        return {
            'GDP': (5000, 35000),
            'CPI': (150, 400),
            'Unemployment_Rate': (0, 30),
            'Federal_Funds_Rate': (-5, 25),
            'VIX': (5, 100),
            'SP500': (500, 10000),
            'Stock_Price': (0.01, 1000),
            'Close': (0.01, 1000),
        }
    
    def _capture_data_snapshot(self, df: pd.DataFrame):
        """Capture key metrics for historical tracking."""
        self.report.data_snapshot = {
            'n_rows': len(df),
            'n_cols': len(df.columns),
            'memory_mb': df.memory_usage(deep=True).sum() / 1024**2,
            'missing_pct': (df.isnull().sum().sum() / df.size) * 100,
        }
        
        if 'Date' in df.columns:
            self.report.data_snapshot['date_range_days'] = (df['Date'].max() - df['Date'].min()).days
    
    def _load_historical_metrics(self) -> Dict:
        """Load historical metrics for comparison."""
        if self.historical_metrics_file.exists():
            try:
                with open(self.historical_metrics_file, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def _save_current_metrics(self):
        """Save current metrics to history."""
        self.historical_metrics_file.parent.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().isoformat()
        if 'history' not in self.historical_metrics:
            self.historical_metrics['history'] = []
        
        self.historical_metrics['history'].append({
            'timestamp': timestamp,
            'snapshot': self.report.data_snapshot,
            'issue_counts': self.report.count_by_severity()
        })
        
        # Keep last 100 runs
        if len(self.historical_metrics['history']) > 100:
            self.historical_metrics['history'] = self.historical_metrics['history'][-100:]
        
        with open(self.historical_metrics_file, 'w') as f:
            json.dump(self.historical_metrics, f, indent=2)
    
    def _print_summary(self):
        """Print validation summary."""
        self.logger.info("\n" + "="*80)
        self.logger.info("VALIDATION SUMMARY")
        self.logger.info("="*80)
        
        counts = self.report.count_by_severity()
        
        self.logger.info(f"\nStatus: {'✅ PASSED' if self.report.passed else '❌ FAILED'}")
        self.logger.info(f"Total Issues: {len(self.report.issues)}")
        self.logger.info(f"  CRITICAL: {counts['CRITICAL']}")
        self.logger.info(f"  ERROR:    {counts['ERROR']}")
        self.logger.info(f"  WARNING:  {counts['WARNING']}")
        self.logger.info(f"  INFO:     {counts['INFO']}")
        
        if self.report.remediation_log:
            self.logger.info(f"\nAuto-Remediation Applied:")
            for log in self.report.remediation_log:
                self.logger.info(f"  ✓ {log}")
        
        critical_issues = self.report.get_critical_issues()
        if critical_issues:
            self.logger.error(f"\n🚨 CRITICAL ISSUES:")
            for issue in critical_issues:
                self.logger.error(f"  • {issue.message}")