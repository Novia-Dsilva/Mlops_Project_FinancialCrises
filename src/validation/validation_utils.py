"""
Validation Utilities

Shared helper functions for all validators.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json


def load_dataset(filepath: Path, parse_dates: List[str] = None) -> pd.DataFrame:
    """
    Safely load dataset with error handling.
    
    Args:
        filepath: Path to CSV file
        parse_dates: List of columns to parse as dates
        
    Returns:
        DataFrame or None if error
    """
    try:
        if parse_dates:
            df = pd.read_csv(filepath, parse_dates=parse_dates)
        else:
            df = pd.read_csv(filepath)
        return df
    except FileNotFoundError:
        print(f"❌ File not found: {filepath}")
        return None
    except Exception as e:
        print(f"❌ Error loading {filepath}: {e}")
        return None


def check_file_exists(filepath: Path) -> bool:
    """Check if file exists and print status."""
    if filepath.exists():
        size_mb = filepath.stat().st_size / (1024 * 1024)
        print(f"✓ Found: {filepath.name} ({size_mb:.2f} MB)")
        return True
    else:
        print(f"✗ Missing: {filepath.name}")
        return False


def compute_basic_stats(df: pd.DataFrame) -> Dict:
    """
    Compute basic statistics for a dataset.
    
    Returns:
        Dictionary with stats
    """
    stats = {
        'n_rows': len(df),
        'n_cols': len(df.columns),
        'memory_mb': df.memory_usage(deep=True).sum() / 1024**2,
        'missing_count': df.isnull().sum().sum(),
        'missing_pct': (df.isnull().sum().sum() / df.size) * 100
    }
    
    # Date range if Date column exists
    if 'Date' in df.columns:
        stats['date_min'] = str(df['Date'].min())
        stats['date_max'] = str(df['Date'].max())
        stats['date_range_days'] = (df['Date'].max() - df['Date'].min()).days
    
    # Company count if applicable
    if 'Company' in df.columns:
        stats['n_companies'] = df['Company'].nunique()
    
    return stats


def save_validation_summary(
    results: Dict,
    output_dir: Path,
    checkpoint_name: str
):
    """
    Save validation summary to JSON.
    
    Args:
        results: Dictionary of validation results
        output_dir: Where to save
        checkpoint_name: Name of checkpoint (e.g., "checkpoint_1_raw")
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    summary = {
        'checkpoint': checkpoint_name,
        'timestamp': datetime.now().isoformat(),
        'overall_passed': all(r.get('passed', False) for r in results.values()),
        'datasets': {}
    }
    
    for dataset_name, result in results.items():
        summary['datasets'][dataset_name] = {
            'passed': result.get('passed', False),
            'ge_success_rate': result.get('ge_report', {}).get('success_rate', 0),
            'robust_critical': result.get('robust_report', {}).get('issue_counts', {}).get('CRITICAL', 0),
            'robust_error': result.get('robust_report', {}).get('issue_counts', {}).get('ERROR', 0),
            'robust_warning': result.get('robust_report', {}).get('issue_counts', {}).get('WARNING', 0)
        }
    
    filename = f"{checkpoint_name}_summary_{timestamp}.json"
    filepath = output_dir / filename
    
    with open(filepath, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n💾 Summary saved: {filepath}")


def print_validation_table(results: Dict):
    """
    Print a nice table of validation results.
    
    Args:
        results: Dictionary of validation results
    """
    print("\n" + "="*80)
    print(f"{'Dataset':<25} {'Status':<12} {'GE Rate':<12} {'Critical':<10} {'Errors':<10}")
    print("-"*80)
    
    for dataset_name, result in results.items():
        passed = result.get('passed', False)
        status = "✅ PASSED" if passed else "❌ FAILED"
        
        ge_rate = result.get('ge_report', {}).get('success_rate', 0)
        critical = result.get('robust_report', {}).get('issue_counts', {}).get('CRITICAL', 0)
        errors = result.get('robust_report', {}).get('issue_counts', {}).get('ERROR', 0)
        
        print(f"{dataset_name:<25} {status:<12} {ge_rate:>6.1f}% {critical:>10} {errors:>10}")
    
    print("="*80)


def detect_data_drift(
    current_stats: Dict,
    historical_stats: Dict,
    threshold: float = 0.2
) -> List[str]:
    """
    Detect data drift by comparing current stats to historical.
    
    Args:
        current_stats: Current dataset statistics
        historical_stats: Historical statistics
        threshold: Percentage threshold for drift (default 20%)
        
    Returns:
        List of drift warnings
    """
    warnings = []
    
    # Check row count drift
    if 'n_rows' in current_stats and 'n_rows' in historical_stats:
        hist_rows = historical_stats['n_rows']
        curr_rows = current_stats['n_rows']
        
        if hist_rows > 0:
            drift = abs(curr_rows - hist_rows) / hist_rows
            if drift > threshold:
                warnings.append(
                    f"Row count drift: {curr_rows:,} vs historical {hist_rows:,} ({drift*100:.1f}% change)"
                )
    
    # Check missing value drift
    if 'missing_pct' in current_stats and 'missing_pct' in historical_stats:
        hist_missing = historical_stats['missing_pct']
        curr_missing = current_stats['missing_pct']
        
        if curr_missing > hist_missing * (1 + threshold):
            warnings.append(
                f"Missing value drift: {curr_missing:.1f}% vs historical {hist_missing:.1f}%"
            )
    
    return warnings


def compare_column_sets(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    name1: str = "Dataset 1",
    name2: str = "Dataset 2"
) -> Dict:
    """
    Compare columns between two datasets.
    
    Returns:
        Dictionary with added, removed, common columns
    """
    cols1 = set(df1.columns)
    cols2 = set(df2.columns)
    
    return {
        'added': list(cols2 - cols1),
        'removed': list(cols1 - cols2),
        'common': list(cols1 & cols2),
        'n_added': len(cols2 - cols1),
        'n_removed': len(cols1 - cols2),
        'n_common': len(cols1 & cols2)
    }


def validate_merge_quality(
    df_left: pd.DataFrame,
    df_right: pd.DataFrame,
    df_merged: pd.DataFrame,
    merge_key: str
) -> Dict:
    """
    Validate merge quality - check for data loss.
    
    Args:
        df_left: Left DataFrame
        df_right: Right DataFrame
        df_merged: Merged DataFrame
        merge_key: Column(s) used for merge
        
    Returns:
        Dictionary with merge statistics
    """
    if isinstance(merge_key, str):
        merge_key = [merge_key]
    
    # Check unique keys
    left_keys = df_left[merge_key].drop_duplicates()
    right_keys = df_right[merge_key].drop_duplicates()
    merged_keys = df_merged[merge_key].drop_duplicates()
    
    stats = {
        'left_unique_keys': len(left_keys),
        'right_unique_keys': len(right_keys),
        'merged_unique_keys': len(merged_keys),
        'expected_keys': len(left_keys),
        'keys_preserved': len(merged_keys) == len(left_keys),
        'data_loss': len(left_keys) - len(merged_keys) if len(merged_keys) < len(left_keys) else 0
    }
    
    return stats


# ============================================================================
# DATASET-SPECIFIC VALIDATORS
# ============================================================================

def validate_fred_columns(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """Validate FRED dataset has required columns."""
    required = ['Date', 'GDP', 'CPI', 'Unemployment_Rate', 'Federal_Funds_Rate']
    missing = [col for col in required if col not in df.columns]
    return (len(missing) == 0), missing


def validate_market_columns(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """Validate Market dataset has required columns."""
    required = ['Date', 'VIX', 'SP500']
    missing = [col for col in required if col not in df.columns]
    return (len(missing) == 0), missing


def validate_company_columns(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """Validate Company dataset has required columns."""
    required = ['Date', 'Company', 'Sector']
    missing = [col for col in required if col not in df.columns]
    return (len(missing) == 0), missing


def check_date_alignment(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    tolerance_days: int = 7
) -> Dict:
    """
    Check if two datasets have overlapping date ranges.
    
    Args:
        df1, df2: DataFrames with Date columns
        tolerance_days: Allowed gap in days
        
    Returns:
        Dictionary with alignment info
    """
    if 'Date' not in df1.columns or 'Date' not in df2.columns:
        return {'aligned': False, 'reason': 'Missing Date column'}
    
    df1_min = df1['Date'].min()
    df1_max = df1['Date'].max()
    df2_min = df2['Date'].min()
    df2_max = df2['Date'].max()
    
    # Check overlap
    overlap_start = max(df1_min, df2_min)
    overlap_end = min(df1_max, df2_max)
    
    has_overlap = overlap_start <= overlap_end
    overlap_days = (overlap_end - overlap_start).days if has_overlap else 0
    
    return {
        'aligned': has_overlap,
        'df1_range': (str(df1_min), str(df1_max)),
        'df2_range': (str(df2_min), str(df2_max)),
        'overlap_range': (str(overlap_start), str(overlap_end)) if has_overlap else None,
        'overlap_days': overlap_days
    }