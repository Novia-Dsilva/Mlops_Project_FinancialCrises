"""
STEP 4: POST-MERGE DATA CLEANING

Runs AFTER Step 3 (merging), BEFORE Step 5 (bias detection)

Purpose:
- Clean merged datasets to address merge-specific issues
- Handle missing values from merge operations
- Remove duplicate columns with suffixes (_x, _y, _fred, _market)
- Fix inf values from ratio calculations
- Cap extreme outliers
- Validate data types
- Remove constant/low-variance columns

Merge-Specific Issues Addressed:
1. Duplicate columns from merge suffixes
2. Missing values from date misalignment
3. Inf values from division operations
4. Type inconsistencies across merged sources
5. Extreme outliers from calculated features

Input:
    - data/features/macro_features.csv
    - data/features/merged_features.csv

Output:
    - data/features/macro_features_clean.csv
    - data/features/merged_features_clean.csv

Usage:
    python step4_post_merge_cleaning.py

Next Step:
    python src/data/step5_bias_detection_with_explicit_slicing.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Tuple
import time
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


class PostMergeDataCleaner:
    """Clean merged datasets to ensure quality before modeling."""

    def __init__(self, features_dir: str = "data/features"):
        self.features_dir = Path(features_dir)
        self.reports_dir = Path("data/reports")
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    # ========== STATISTICS FUNCTIONS ==========

    def compute_statistics(self, df: pd.DataFrame, name: str) -> Dict:
        """Compute comprehensive statistics."""
        stats = {
            'dataset_name': name,
            'n_rows': len(df),
            'n_cols': len(df.columns),
            'memory_mb': df.memory_usage(deep=True).sum() / 1024**2,
        }

        # Date range
        if 'Date' in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(df['Date']):
                df['Date'] = pd.to_datetime(df['Date'], format='mixed', errors='coerce')

            
            stats['date_min'] = str(df['Date'].min())
            stats['date_max'] = str(df['Date'].max())
            stats['date_range_days'] = (df['Date'].max() - df['Date'].min()).days

        # Missing values
        missing = df.isna().sum()
        stats['total_missing'] = missing.sum()
        stats['missing_pct'] = round((missing.sum() / df.size) * 100, 2)
        stats['cols_with_missing'] = (missing > 0).sum()

        # Numeric statistics
        numeric_df = df.select_dtypes(include=[np.number])
        if not numeric_df.empty:
            stats['n_numeric_cols'] = len(numeric_df.columns)
            
            # Check for inf values
            inf_count = np.isinf(numeric_df).sum().sum()
            stats['inf_values'] = inf_count
            
            # Check for negative values in columns that should be positive
            non_negative_cols = ['VIX', 'Volume', 'CPI', 'Total_Assets', 'Revenue']
            negative_count = 0
            for col in non_negative_cols:
                if col in numeric_df.columns:
                    negative_count += (numeric_df[col] < 0).sum()
            stats['invalid_negatives'] = negative_count

        # Duplicates
        if 'Date' in df.columns and 'Company' in df.columns:
            stats['duplicates'] = df.duplicated(subset=['Date', 'Company']).sum()
        elif 'Date' in df.columns:
            stats['duplicates'] = df.duplicated(subset=['Date']).sum()
        else:
            stats['duplicates'] = df.duplicated().sum()

        return stats

    def print_statistics_comparison(self, before_stats: Dict, after_stats: Dict):
        """Print before/after comparison."""
        logger.info(f"\n{'='*80}")
        logger.info(f"STATISTICS: {before_stats['dataset_name']}")
        logger.info(f"{'='*80}")

        comparisons = [
            ('Rows', 'n_rows'),
            ('Columns', 'n_cols'),
            ('Memory (MB)', 'memory_mb'),
            ('Date Range (days)', 'date_range_days'),
            ('Total Missing', 'total_missing'),
            ('Missing %', 'missing_pct'),
            ('Cols with Missing', 'cols_with_missing'),
            ('Inf Values', 'inf_values'),
            ('Invalid Negatives', 'invalid_negatives'),
            ('Duplicates', 'duplicates'),
        ]

        print(f"\n{'Metric':<30} {'BEFORE':>15} {'AFTER':>15} {'Change':>15}")
        print("-" * 75)

        for label, key in comparisons:
            before_val = before_stats.get(key, 'N/A')
            after_val = after_stats.get(key, 'N/A')

            if isinstance(before_val, (int, float)) and isinstance(after_val, (int, float)):
                change = after_val - before_val
                if isinstance(before_val, float):
                    print(f"{label:<30} {before_val:>15.2f} {after_val:>15.2f} {change:>15.2f}")
                else:
                    print(f"{label:<30} {before_val:>15,} {after_val:>15,} {change:>15,}")
            else:
                print(f"{label:<30} {str(before_val):>15} {str(after_val):>15} {'':>15}")

    # ========== CLEANING FUNCTIONS ==========

    def remove_duplicate_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove duplicate columns created during merge.
        
        Common patterns:
        - col, col_x, col_y (from merge)
        - col, col_fred, col_market (from merge with suffixes)
        - col, col_macro, col_company (from multiple merges)
        """
        logger.info("\n1. Removing duplicate columns...")
        
        df = df.copy()
        original_cols = len(df.columns)
        
        # Find columns with common suffixes
        suffixes = ['_x', '_y', '_fred', '_market', '_macro', '_company', '_dup', '_fin']
        
        cols_to_drop = []
        cols_to_rename = {}
        
        for col in df.columns:
            # Check if this is a suffixed duplicate
            for suffix in suffixes:
                if col.endswith(suffix):
                    base_col = col[:-len(suffix)]
                    
                    # If base column exists, drop the suffixed version
                    if base_col in df.columns:
                        cols_to_drop.append(col)
                        logger.info(f"   Found duplicate: '{col}' (keeping '{base_col}')")
                    else:
                        # If base doesn't exist, rename to base
                        cols_to_rename[col] = base_col
                        logger.info(f"   Renaming: '{col}' → '{base_col}'")
        
        # Drop duplicates
        if cols_to_drop:
            df.drop(columns=cols_to_drop, inplace=True)
            logger.info(f"   ✓ Removed {len(cols_to_drop)} duplicate columns")
        
        # Rename orphaned suffixed columns
        if cols_to_rename:
            df.rename(columns=cols_to_rename, inplace=True)
            logger.info(f"   ✓ Renamed {len(cols_to_rename)} orphaned suffixed columns")
        
        if not cols_to_drop and not cols_to_rename:
            logger.info(f"   ✓ No duplicate columns found")
        
        final_cols = len(df.columns)
        logger.info(f"   Columns: {original_cols} → {final_cols} (removed {original_cols - final_cols})")
        
        return df

    def handle_inf_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Replace inf/-inf values with NaN.
        
        Inf values often come from:
        - Division by zero in ratio calculations
        - Log of zero/negative numbers
        - Percentage calculations with zero denominators
        """
        logger.info("\n2. Handling inf values...")
        
        df = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        # Count inf values before
        inf_before = np.isinf(df[numeric_cols]).sum().sum()
        
        if inf_before > 0:
            logger.info(f"   Found {inf_before:,} inf values across {len(numeric_cols)} numeric columns")
            
            # Identify which columns have inf
            inf_cols = []
            for col in numeric_cols:
                inf_count = np.isinf(df[col]).sum()
                if inf_count > 0:
                    inf_cols.append((col, inf_count))
            
            # Show top offenders
            inf_cols_sorted = sorted(inf_cols, key=lambda x: x[1], reverse=True)
            logger.info(f"   Top columns with inf values:")
            for col, count in inf_cols_sorted[:10]:
                logger.info(f"     - {col}: {count:,} inf values")
            
            # Replace inf with NaN
            df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
            
            logger.info(f"   ✓ Replaced {inf_before:,} inf values with NaN")
        else:
            logger.info(f"   ✓ No inf values found")
        
        return df

    def cap_extreme_outliers(self, df: pd.DataFrame, 
                            group_col: str = None,
                            percentile_low: float = 0.001,
                            percentile_high: float = 0.999) -> pd.DataFrame:
        """
        Cap extreme outliers at percentile thresholds.
        
        This is more conservative than removing outliers - we keep the data
        but prevent extreme values from dominating models.
        
        Args:
            df: DataFrame
            group_col: If provided, cap within groups (e.g., per Company)
            percentile_low: Lower percentile threshold (default: 0.1%)
            percentile_high: Upper percentile threshold (default: 99.9%)
        """
        logger.info(f"\n3. Capping extreme outliers (outside {percentile_low:.1%}-{percentile_high:.1%})...")
        
        df = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        # Exclude ID-like columns
        exclude_cols = ['Date', 'Year', 'Month', 'Day', 'Quarter']
        numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        capped_count = 0
        capped_details = []
        
        if group_col and group_col in df.columns:
            # Cap within groups
            logger.info(f"   Capping per {group_col}...")
            
            for col in numeric_cols:
                col_capped = 0
                for group_name in df[group_col].unique():
                    group_mask = df[group_col] == group_name
                    group_data = df.loc[group_mask, col]
                    
                    if group_data.notna().sum() > 10:  # Need enough data
                        lower = group_data.quantile(percentile_low)
                        upper = group_data.quantile(percentile_high)
                        
                        # Count values being capped
                        n_capped = ((group_data < lower) | (group_data > upper)).sum()
                        col_capped += n_capped
                        
                        # Cap values
                        df.loc[group_mask, col] = group_data.clip(lower=lower, upper=upper)
                
                if col_capped > 0:
                    capped_count += col_capped
                    capped_details.append((col, col_capped))
        else:
            # Cap entire dataset
            logger.info(f"   Capping globally across all data...")
            
            for col in numeric_cols:
                if df[col].notna().sum() > 10:
                    lower = df[col].quantile(percentile_low)
                    upper = df[col].quantile(percentile_high)
                    
                    # Count values being capped
                    n_capped = ((df[col] < lower) | (df[col] > upper)).sum()
                    
                    if n_capped > 0:
                        capped_count += n_capped
                        capped_details.append((col, n_capped))
                    
                    # Cap values
                    df[col] = df[col].clip(lower=lower, upper=upper)
        
        if capped_count > 0:
            logger.info(f"   ✓ Capped {capped_count:,} extreme values across {len(numeric_cols)} columns")
            
            # Show top capped columns
            capped_sorted = sorted(capped_details, key=lambda x: x[1], reverse=True)
            logger.info(f"   Top capped columns:")
            for col, count in capped_sorted[:10]:
                logger.info(f"     - {col}: {count:,} values capped")
        else:
            logger.info(f"   ✓ No extreme outliers found")
        
        return df

    def handle_missing_values_post_merge(self, df: pd.DataFrame, 
                                         group_col: str = None) -> pd.DataFrame:
        """
        Handle missing values created by merge operations.
        
        Strategy:
        1. For time series columns: Forward fill then backward fill
        2. For cross-sectional columns: Fill with group median
        3. For sparse columns (>50% missing): Flag for review
        
        Args:
            df: DataFrame
            group_col: Column to group by (e.g., 'Company')
        """
        logger.info("\n4. Handling missing values from merge...")
        
        df = df.copy()
        original_missing = df.isna().sum().sum()
        original_missing_pct = (original_missing / df.size) * 100
        
        logger.info(f"   Total missing values: {original_missing:,} ({original_missing_pct:.2f}%)")
        
        # === IDENTIFY HIGH-MISSING COLUMNS ===
        missing_pct = (df.isna().sum() / len(df)) * 100
        high_missing = missing_pct[missing_pct > 50].sort_values(ascending=False)
        
        if len(high_missing) > 0:
            logger.warning(f"\n   ⚠️  Columns with >50% missing (may not be useful):")
            for col, pct in high_missing.items():
                logger.warning(f"      - {col}: {pct:.1f}%")
            logger.info(f"   Note: Keeping these columns but they may be dropped later")
        
        # === FILL MISSING VALUES ===
        if group_col and group_col in df.columns:
            logger.info(f"\n   Filling missing values per {group_col}...")
            
            companies_processed = 0
            for company in df[group_col].unique():
                company_mask = df[group_col] == company
                company_data = df.loc[company_mask].copy()
                
                # Forward fill (time series)
                company_data = company_data.ffill()
                
                # Backward fill (for leading NaNs)
                company_data = company_data.bfill()
                
                # For any remaining NaNs, use column median
                for col in company_data.columns:
                    if company_data[col].isna().any():
                        if pd.api.types.is_numeric_dtype(company_data[col]):
                            median_val = company_data[col].median()
                            if not np.isnan(median_val):
                                company_data[col].fillna(median_val, inplace=True)
                
                df.loc[company_mask] = company_data
                companies_processed += 1
            
            logger.info(f"   ✓ Processed {companies_processed} companies")
        else:
            logger.info(f"\n   Filling missing values globally...")
            
            # Forward fill
            df = df.ffill()
            
            # Backward fill
            df = df.bfill()
            
            # Fill remaining with median
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if df[col].isna().any():
                    median_val = df[col].median()
                    if not np.isnan(median_val):
                        df[col].fillna(median_val, inplace=True)
        
        final_missing = df.isna().sum().sum()
        final_missing_pct = (final_missing / df.size) * 100
        filled = original_missing - final_missing
        
        logger.info(f"\n   ✓ Filled {filled:,} missing values")
        logger.info(f"   Remaining: {final_missing:,} ({final_missing_pct:.2f}%)")
        
        return df

    def validate_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure proper data types for all columns.
        
        Common issues from merge:
        - Numeric columns stored as object
        - Date columns as string
        - Category columns as object
        """
        logger.info("\n5. Validating data types...")
        
        df = df.copy()
        conversions = []
        
        # === DATE COLUMNS ===
        if 'Date' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['Date']):
            #df['Date'] = pd.to_datetime(df['Date'])
            df['Date'] = pd.to_datetime(df['Date'], format='mixed', errors='coerce')
            conversions.append("Date → datetime64")
        
        # === CATEGORICAL COLUMNS ===
        categorical_cols = ['Company', 'Sector', 'Company_Name', 'VIX_Regime']
        for col in categorical_cols:
            if col in df.columns:
                if df[col].dtype == 'object':
                    df[col] = df[col].astype('category')
                    conversions.append(f"{col} → category")
                elif df[col].dtype == 'float64':
                    # VIX_Regime might be stored as float after merge
                    if col == 'VIX_Regime':
                        df[col] = df[col].astype('category')
                        conversions.append(f"{col} → category (from float)")
        
        # === NUMERIC COLUMNS (that may be stored as object) ===
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    # Try to convert to numeric
                    converted = pd.to_numeric(df[col], errors='coerce')
                    # Only convert if most values are numeric
                    if converted.notna().sum() / len(df) > 0.5:
                        df[col] = converted
                        conversions.append(f"{col} → numeric (from object)")
                except (ValueError, TypeError):
                    pass  # Keep as object if conversion fails
        
        if conversions:
            logger.info(f"   ✓ Converted {len(conversions)} columns:")
            for conv in conversions[:10]:  # Show first 10
                logger.info(f"      - {conv}")
            if len(conversions) > 10:
                logger.info(f"      ... and {len(conversions) - 10} more")
        else:
            logger.info(f"   ✓ All data types correct")
        
        return df

    # def remove_constant_columns(self, df: pd.DataFrame) -> pd.DataFrame:
    #     """
    #     Remove columns with constant values (no variance).
        
    #     These provide no information for modeling.
    #     """
    #     logger.info("\n6. Removing constant/low-variance columns...")
        
    #     df = df.copy()
    #     original_cols = len(df.columns)
        
    #     # === IDENTIFY CONSTANT COLUMNS ===
    #     constant_cols = []
    #     low_variance_cols = []
        
    #     for col in df.columns:
    #         if col not in ['Date', 'Company', 'Sector']:  # Keep these
    #             unique_count = df[col].nunique()
                
    #             if unique_count <= 1:
    #                 constant_cols.append(col)
    #             elif unique_count == 2 and pd.api.types.is_numeric_dtype(df[col]):
    #                 # Check if it's essentially constant (e.g., 0 and 0.0000001)
    #                 if df[col].std() < 1e-10:
    #                     low_variance_cols.append(col)
        
    #     # Drop constant columns
    #     if constant_cols:
    #         df.drop(columns=constant_cols, inplace=True)
    #         logger.info(f"   ✓ Removed {len(constant_cols)} constant columns:")
    #         for col in constant_cols[:5]:
    #             logger.info(f"      - {col}")
    #         if len(constant_cols) > 5:
    #             logger.info(f"      ... and {len(constant_cols) - 5} more")
    #     else:
    #         logger.info(f"   ✓ No constant columns found")
        
    #     # Drop low variance columns
    #     if low_variance_cols:
    #         df.drop(columns=low_variance_cols, inplace=True)
    #         logger.info(f"   ✓ Removed {len(low_variance_cols)} low-variance columns:")
    #         for col in low_variance_cols[:5]:
    #             logger.info(f"      - {col}")
        
    #     final_cols = len(df.columns)
    #     if original_cols > final_cols:
    #         logger.info(f"   Columns: {original_cols} → {final_cols} (removed {original_cols - final_cols})")
        
    #     return df

    def remove_constant_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove columns with constant values (no variance).
        
        These provide no information for modeling.
        """
        logger.info("\n6. Removing constant/low-variance columns...")
        
        df = df.copy()
        original_cols = len(df.columns)
        
        # === IDENTIFY CONSTANT COLUMNS ===
        constant_cols = []
        low_variance_cols = []
        
        for col in df.columns:
            if col not in ['Date', 'Company', 'Sector']:  # Keep these
                unique_count = df[col].nunique()
                
                if unique_count <= 1:
                    constant_cols.append(col)
                elif unique_count == 2 and pd.api.types.is_numeric_dtype(df[col]):
                    # Check if it's essentially constant (e.g., 0 and 0.0000001)
                    if df[col].std() < 1e-10:
                        low_variance_cols.append(col)
        
        # Drop constant columns
        if constant_cols:
            df.drop(columns=constant_cols, inplace=True)
            logger.info(f"   ✓ Removed {len(constant_cols)} constant columns:")
            for col in constant_cols[:5]:
                logger.info(f"      - {col}")
            if len(constant_cols) > 5:
                logger.info(f"      ... and {len(constant_cols) - 5} more")
        else:
            logger.info(f"   ✓ No constant columns found")
        
        # Drop low variance columns
        if low_variance_cols:
            df.drop(columns=low_variance_cols, inplace=True)
            logger.info(f"   ✓ Removed {len(low_variance_cols)} low-variance columns:")
            for col in low_variance_cols[:5]:
                logger.info(f"      - {col}")
        
        final_cols = len(df.columns)
        if original_cols > final_cols:
            logger.info(f"   Columns: {original_cols} → {final_cols} (removed {original_cols - final_cols})")
        
        return df

    def fix_invalid_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fix invalid ratios that may have been created during merge.
        
        Common issues:
        - Negative ratios that should be positive
        - Extreme ratio values from division by near-zero
        """
        logger.info("\n7. Fixing invalid ratios...")
        
        df = df.copy()
        fixes = []
        
        # === IDENTIFY RATIO COLUMNS ===
        ratio_cols = [col for col in df.columns if any(
            keyword in col.lower() for keyword in 
            ['ratio', 'margin', 'pct', 'percent', '_to_', 'vs_ma', 'roe', 'roa']
        )]
        
        logger.info(f"   Found {len(ratio_cols)} ratio/percentage columns")
        
        for col in ratio_cols:
            if col in df.columns:
                # === FIX 1: Cap extreme ratios ===
                extreme_high = (df[col] > 1000).sum()
                extreme_low = (df[col] < -1000).sum()
                
                if extreme_high > 0 or extreme_low > 0:
                    df[col] = df[col].clip(-1000, 1000)
                    fixes.append(f"{col}: capped {extreme_high + extreme_low} extreme values to [-1000, 1000]")
                
                # === FIX 2: Handle negative values in ratios that should be positive ===
                # Some ratios can be negative (e.g., Profit_Margin during losses)
                # But others should always be positive
                always_positive = ['current_ratio', 'debt_to_assets', 'volume_vs']
                
                if any(keyword in col.lower() for keyword in always_positive):
                    neg_count = (df[col] < 0).sum()
                    if neg_count > 0:
                        df[col] = df[col].abs()
                        fixes.append(f"{col}: made {neg_count} negative values positive")
        
        if fixes:
            logger.info(f"   ✓ Fixed {len(fixes)} ratio issues:")
            for fix in fixes[:10]:
                logger.info(f"      - {fix}")
            if len(fixes) > 10:
                logger.info(f"      ... and {len(fixes) - 10} more")
        else:
            logger.info(f"   ✓ No invalid ratios found")
        
        return df

    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate rows."""
        logger.info("\n8. Removing duplicate rows...")
        
        df = df.copy()
        original_rows = len(df)
        
        if 'Date' in df.columns and 'Company' in df.columns:
            # Remove duplicates based on Date + Company
            duplicates_before = df.duplicated(subset=['Date', 'Company']).sum()
            
            if duplicates_before > 0:
                df = df.drop_duplicates(subset=['Date', 'Company'], keep='first')
                duplicates_removed = original_rows - len(df)
                logger.info(f"   ✓ Removed {duplicates_removed:,} duplicate (Date, Company) pairs")
            else:
                logger.info(f"   ✓ No duplicate (Date, Company) pairs found")
        elif 'Date' in df.columns:
            # Remove duplicates based on Date only
            duplicates_before = df.duplicated(subset=['Date']).sum()
            
            if duplicates_before > 0:
                df = df.drop_duplicates(subset=['Date'], keep='first')
                duplicates_removed = original_rows - len(df)
                logger.info(f"   ✓ Removed {duplicates_removed:,} duplicate dates")
            else:
                logger.info(f"   ✓ No duplicate dates found")
        else:
            # Remove completely duplicate rows
            duplicates_before = df.duplicated().sum()
            
            if duplicates_before > 0:
                df = df.drop_duplicates(keep='first')
                duplicates_removed = original_rows - len(df)
                logger.info(f"   ✓ Removed {duplicates_removed:,} duplicate rows")
            else:
                logger.info(f"   ✓ No duplicate rows found")
        
        return df

    # ========== MAIN CLEANING PIPELINES ==========

    def clean_macro_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict, Dict]:
        """
        Clean macro_features.csv after merging.
        
        This dataset contains FRED + Market data merged on Date.
        """
        logger.info("\n" + "="*80)
        logger.info("CLEANING MACRO_FEATURES.CSV (FRED + Market Merged)")
        logger.info("="*80)
        
        # Before statistics
        before_stats = self.compute_statistics(df, 'macro_features')
        
        logger.info(f"\nBEFORE CLEANING:")
        logger.info(f"  Shape: {df.shape}")
        logger.info(f"  Missing: {before_stats['total_missing']:,} ({before_stats['missing_pct']:.2f}%)")
        logger.info(f"  Inf values: {before_stats.get('inf_values', 0):,}")
        logger.info(f"  Invalid negatives: {before_stats.get('invalid_negatives', 0):,}")
        
        # Apply cleaning steps IN ORDER
        df = self.remove_duplicate_columns(df)
        df = self.handle_inf_values(df)
        df = self.cap_extreme_outliers(df)
        df = self.handle_missing_values_post_merge(df)
        df = self.validate_data_types(df)
        df = self.remove_constant_columns(df)
        df = self.fix_invalid_ratios(df)
        df = self.remove_duplicates(df)
        
        # After statistics
        after_stats = self.compute_statistics(df, 'macro_features')
        
        logger.info(f"\nAFTER CLEANING:")
        logger.info(f"  Shape: {df.shape}")
        logger.info(f"  Missing: {after_stats['total_missing']:,} ({after_stats['missing_pct']:.2f}%)")
        logger.info(f"  Inf values: {after_stats.get('inf_values', 0):,}")
        logger.info(f"  Invalid negatives: {after_stats.get('invalid_negatives', 0):,}")
        
        return df, before_stats, after_stats

    def clean_merged_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict, Dict]:
        """
        Clean merged_features.csv after merging.
        
        This dataset contains FRED + Market + Company data merged on Date + Company.
        """
        logger.info("\n" + "="*80)
        logger.info("CLEANING MERGED_FEATURES.CSV (Macro + Market + Company)")
        logger.info("="*80)
        
        # Before statistics
        before_stats = self.compute_statistics(df, 'merged_features')
        
        logger.info(f"\nBEFORE CLEANING:")
        logger.info(f"  Shape: {df.shape}")
        logger.info(f"  Companies: {df['Company'].nunique() if 'Company' in df.columns else 'N/A'}")
        logger.info(f"  Missing: {before_stats['total_missing']:,} ({before_stats['missing_pct']:.2f}%)")
        logger.info(f"  Inf values: {before_stats.get('inf_values', 0):,}")
        logger.info(f"  Invalid negatives: {before_stats.get('invalid_negatives', 0):,}")
        
        # Apply cleaning steps (with company grouping)
        df = self.remove_duplicate_columns(df)
        df = self.handle_inf_values(df)
        df = self.cap_extreme_outliers(df, group_col='Company')  # Cap per company
        df = self.handle_missing_values_post_merge(df, group_col='Company')  # Fill per company
        df = self.validate_data_types(df)
        df = self.remove_constant_columns(df)
        df = self.fix_invalid_ratios(df)
        df = self.remove_duplicates(df)
        
        # After statistics
        after_stats = self.compute_statistics(df, 'merged_features')
        
        logger.info(f"\nAFTER CLEANING:")
        logger.info(f"  Shape: {df.shape}")
        logger.info(f"  Companies: {df['Company'].nunique() if 'Company' in df.columns else 'N/A'}")
        logger.info(f"  Missing: {after_stats['total_missing']:,} ({after_stats['missing_pct']:.2f}%)")
        logger.info(f"  Inf values: {after_stats.get('inf_values', 0):,}")
        logger.info(f"  Invalid negatives: {after_stats.get('invalid_negatives', 0):,}")
        
        return df, before_stats, after_stats

    def save_cleaning_report(self, all_stats: Dict):
        """Save detailed cleaning report."""
        report_data = []
        
        for dataset_name, stats_pair in all_stats.items():
            before = stats_pair['before']
            after = stats_pair['after']
            
            report_data.append({
                'Dataset': dataset_name,
                'Rows_Before': before['n_rows'],
                'Rows_After': after['n_rows'],
                'Cols_Before': before['n_cols'],
                'Cols_After': after['n_cols'],
                'Missing_Before': before['total_missing'],
                'Missing_After': after['total_missing'],
                'Missing_Pct_Before': before['missing_pct'],
                'Missing_Pct_After': after['missing_pct'],
                'Inf_Before': before.get('inf_values', 0),
                'Inf_After': after.get('inf_values', 0),
                'Invalid_Neg_Before': before.get('invalid_negatives', 0),
                'Invalid_Neg_After': after.get('invalid_negatives', 0),
                'Duplicates_Before': before['duplicates'],
                'Duplicates_After': after['duplicates']
            })
        
        report_df = pd.DataFrame(report_data)
        report_path = self.reports_dir / 'step3c_post_merge_cleaning_report.csv'
        report_df.to_csv(report_path, index=False)
        logger.info(f"\n✓ Cleaning report saved to: {report_path}")
        
        return report_df

    # ========== MAIN PIPELINE ==========

    def run_post_merge_cleaning(self):
        """Execute complete post-merge cleaning pipeline."""
        logger.info("\n" + "="*80)
        logger.info("STEP 4: POST-MERGE DATA CLEANING")
        logger.info("="*80)
        logger.info("\nCleaning merged datasets from Step 3...")
        logger.info("\nWhy needed:")
        logger.info("  - Merge creates duplicate columns (col_x, col_y)")
        logger.info("  - Date misalignment causes missing values")
        logger.info("  - Ratio calculations create inf values")
        logger.info("  - Type inconsistencies across sources")
        logger.info("="*80)

        overall_start = time.time()
        
        all_stats = {}

        # === CLEAN MACRO_FEATURES ===
        macro_path = self.features_dir / 'macro_features.csv'
        if macro_path.exists():
            logger.info(f"\n{'='*80}")
            logger.info("[1/2] LOADING macro_features.csv")
            logger.info(f"{'='*80}")
            
            df_macro = pd.read_csv(macro_path, parse_dates=['Date'])
            logger.info(f"Loaded: {df_macro.shape}")
            
            df_macro_clean, before_macro, after_macro = self.clean_macro_features(df_macro)
            
            # Save
            output_path = self.features_dir / 'macro_features_clean.csv'
            df_macro_clean.to_csv(output_path, index=False)
            logger.info(f"\n✓ Saved: {output_path}")
            logger.info(f"  Size: {output_path.stat().st_size / (1024*1024):.2f} MB")
            
            all_stats['macro_features'] = {'before': before_macro, 'after': after_macro}
        else:
            logger.warning(f"\n⚠️  macro_features.csv not found at {macro_path}")
            logger.warning("     Skipping macro_features cleaning")

        # === CLEAN MERGED_FEATURES ===
        merged_path = self.features_dir / 'merged_features.csv'
        if merged_path.exists():
            logger.info(f"\n{'='*80}")
            logger.info("[2/2] LOADING merged_features.csv")
            logger.info(f"{'='*80}")
            
            df_merged = pd.read_csv(merged_path, parse_dates=['Date'])
            logger.info(f"Loaded: {df_merged.shape}")
            
            df_merged_clean, before_merged, after_merged = self.clean_merged_features(df_merged)
            
            # Save
            output_path = self.features_dir / 'merged_features_clean.csv'
            df_merged_clean.to_csv(output_path, index=False)
            logger.info(f"\n✓ Saved: {output_path}")
            logger.info(f"  Size: {output_path.stat().st_size / (1024*1024):.2f} MB")
            
            all_stats['merged_features'] = {'before': before_merged, 'after': after_merged}
        else:
            logger.warning(f"\n⚠️  merged_features.csv not found at {merged_path}")
            logger.warning("     Skipping merged_features cleaning")

        if not all_stats:
            logger.error("\n❌ No datasets were cleaned!")
            logger.error("Make sure Step 3 (merging) was run successfully")
            return None

        # === PRINT BEFORE/AFTER COMPARISONS ===
        logger.info("\n\n" + "="*80)
        logger.info("BEFORE vs AFTER COMPARISON")
        logger.info("="*80)
        
        for name, stats in all_stats.items():
            self.print_statistics_comparison(stats['before'], stats['after'])

        # === SAVE COMPREHENSIVE REPORT ===
        report = self.save_cleaning_report(all_stats)

        # === FINAL SUMMARY ===
        elapsed = time.time() - overall_start

        logger.info("\n\n" + "="*80)
        logger.info("CLEANING SUMMARY")
        logger.info("="*80)
        print("\n" + report.to_string(index=False))

        logger.info("\n" + "="*80)
        logger.info("POST-MERGE CLEANING COMPLETE")
        logger.info("="*80)
        
        logger.info(f"\n📁 Cleaned files saved:")
        if 'macro_features' in all_stats:
            logger.info(f"  ✓ data/features/macro_features_clean.csv")
        if 'merged_features' in all_stats:
            logger.info(f"  ✓ data/features/merged_features_clean.csv")

        logger.info(f"\n📊 Reports:")
        logger.info(f"  ✓ data/reports/step3c_post_merge_cleaning_report.csv")

        logger.info(f"\n⏱️  Total Time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
        logger.info("="*80)

        logger.info("\n✅ Step 3c Complete!")
        logger.info("\n➡️  Next Steps:")
        logger.info("   1. (Optional) Validate: python src/validation/validate_checkpoint_4_clean_merged.py")

        return all_stats


def main():
    """Execute post-merge cleaning."""
    
    cleaner = PostMergeDataCleaner(features_dir="data/features")
    
    try:
        stats = cleaner.run_post_merge_cleaning()
        
        if stats:
            logger.info("\n" + "="*80)
            logger.info("✅ STEP 4 SUCCESSFULLY COMPLETED")
            logger.info("="*80)
            return stats
        else:
            logger.error("\n❌ Cleaning failed - no datasets processed")
            return None
        
    except FileNotFoundError as e:
        logger.error(f"\n❌ ERROR: {e}")
        logger.error("\nMake sure you've run Step 3 (merging) first!")
        logger.error("  Run: python step3_data_merging.py")
        return None
    except Exception as e:
        logger.error(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    cleaning_stats = main()
    