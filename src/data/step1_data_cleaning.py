"""
STEP 1: DATA CLEANING WITH POINT-IN-TIME CORRECTNESS

This script cleans all raw data files while preserving point-in-time correctness.

Key Features:
1. Forward-fill ONLY (no backward fill = no look-ahead bias)
2. Apply reporting lags to quarterly financials (45 days)
3. Detect outliers but DON'T remove (crises are real!)
4. Per-company handling (no cross-contamination)
5. Comprehensive before/after statistics

Input:  data/raw/*.csv (5 files)
Output: data/clean/*.csv (5 files)

Usage:
    python step1_data_cleaning.py

Next Step:
    python src/validation/validate_checkpoint_2_clean.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, Tuple, List
import time
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


class PointInTimeDataCleaner:
    """Clean data while preserving point-in-time correctness."""

    # Reporting lags (days after quarter-end when data becomes available)
    REPORTING_LAGS = {
        'earnings': 45,      # Earnings reported ~45 days after quarter end
        'balance_sheet': 45, # Balance sheet same as earnings
        'macro': 30          # Macro data (GDP, CPI) ~30 days lag
    }

    def __init__(self, raw_dir: str = "data/raw", clean_dir: str = "data/clean"):
        self.raw_dir = Path(raw_dir)
        self.clean_dir = Path(clean_dir)
        self.clean_dir.mkdir(parents=True, exist_ok=True)

        # Create reports directory
        self.report_dir = Path("data/reports")
        self.report_dir.mkdir(parents=True, exist_ok=True)

    # ========== STATISTICS FUNCTIONS ==========

    def compute_statistics(self, df: pd.DataFrame, name: str) -> Dict:
        """Compute comprehensive statistics for a dataset."""
        stats = {
            'dataset_name': name,
            'n_rows': len(df),
            'n_cols': len(df.columns),
            'memory_mb': df.memory_usage(deep=True).sum() / 1024**2,
        }

        # Date range
        if 'Date' in df.columns:
            # Ensure Date is datetime
            if not pd.api.types.is_datetime64_any_dtype(df['Date']):
                df['Date'] = pd.to_datetime(df['Date'])

            stats['date_min'] = str(df['Date'].min())
            stats['date_max'] = str(df['Date'].max())
            stats['date_range_days'] = (df['Date'].max() - df['Date'].min()).days

        # Missing values
        missing = df.isna().sum()
        stats['total_missing'] = missing.sum()
        stats['missing_pct'] = round((missing.sum() / df.size) * 100, 2)
        stats['cols_with_missing'] = (missing > 0).sum()

        # Duplicates
        if 'Date' in df.columns and 'Company' in df.columns:
            stats['duplicates'] = df.duplicated(subset=['Date', 'Company']).sum()
        elif 'Date' in df.columns:
            stats['duplicates'] = df.duplicated(subset=['Date']).sum()
        else:
            stats['duplicates'] = df.duplicated().sum()

        # Numeric statistics
        numeric_df = df.select_dtypes(include=[np.number])
        if not numeric_df.empty:
            stats['n_numeric_cols'] = len(numeric_df.columns)
            stats['mean_value'] = numeric_df.mean().mean()
            stats['std_value'] = numeric_df.std().mean()

        # Categorical
        categorical_df = df.select_dtypes(exclude=[np.number])
        stats['n_categorical_cols'] = len(categorical_df.columns)

        return stats

    def print_statistics_comparison(self, before_stats: Dict, after_stats: Dict):
        """Print before/after comparison in clean format."""
        logger.info(f"\n{'='*80}")
        logger.info(f"STATISTICS: {before_stats['dataset_name']}")
        logger.info(f"{'='*80}")

        comparisons = [
            ('Rows', 'n_rows'),
            ('Columns', 'n_cols'),
            ('Memory (MB)', 'memory_mb'),
            ('Date Range (days)', 'date_range_days'),
            ('Total Missing Values', 'total_missing'),
            ('Missing %', 'missing_pct'),
            ('Columns with Missing', 'cols_with_missing'),
            ('Duplicate Rows', 'duplicates'),
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

    # ========== POINT-IN-TIME FUNCTIONS ==========

    def apply_reporting_lag(self, df: pd.DataFrame, lag_days: int,
                           group_col: str = None) -> pd.DataFrame:
        """
        Apply reporting lag to quarterly data for point-in-time correctness.

        Example: Q1 2020 earnings (3/31) are reported 45 days later (5/15)
        So on any day before 5/15, we should use Q4 2019 data, not Q1 2020.

        Args:
            df: DataFrame with quarterly data
            lag_days: Number of days after quarter-end when data is available
            group_col: If provided, shift within groups (e.g., per Company)
        """
        logger.info(f"\n⏰ Applying {lag_days}-day reporting lag for point-in-time correctness...")

        df = df.copy()

        # Shift dates forward by reporting lag
        df['Date'] = df['Date'] + pd.Timedelta(days=lag_days)

        # Log the transformation
        example_date = pd.Timestamp('2020-03-31')
        example_available = example_date + pd.Timedelta(days=lag_days)
        logger.info(f"  Example: Q1 2020 (3/31) → Available on {example_available.date()}")

        return df

    def handle_nulls_no_lookahead(self, df: pd.DataFrame, date_col: str = 'Date',
                                  group_col: str = None) -> pd.DataFrame:
        """
        Handle nulls using ONLY forward fill (no backward fill = no look-ahead).

        For leading NaNs (at start), use median of first 10 valid values.
        """
        df = df.copy()
        df_original = df.copy()

        # Ensure date is datetime
        if date_col in df.columns and not pd.api.types.is_datetime64_any_dtype(df[date_col]):
            df[date_col] = pd.to_datetime(df[date_col])

        if group_col:
            # Fill within groups
            numeric_cols = df.select_dtypes(include=[np.number]).columns

            for col in numeric_cols:
                # Forward fill within group
                df[col] = df.groupby(group_col)[col].ffill()

                # For remaining leading NaNs, use group median
                for group_name in df[group_col].unique():
                    group_mask = df[group_col] == group_name
                    group_data = df.loc[group_mask, col]

                    if group_data.isna().any():
                        valid_data = group_data.dropna()
                        if len(valid_data) > 0:
                            fill_value = valid_data.head(min(10, len(valid_data))).median()
                            df.loc[group_mask, col] = df.loc[group_mask, col].fillna(fill_value)
        else:
            # Fill entire dataset
            df = df.set_index(date_col) if date_col in df.columns else df

            # Forward fill
            df = df.ffill()

            # For remaining leading NaNs, use median of first valid values
            for col in df.columns:
                if df[col].isna().any():
                    valid_data = df[col].dropna()
                    if len(valid_data) > 0:
                        fill_value = valid_data.head(min(10, len(valid_data))).median()
                        df[col] = df[col].fillna(fill_value)

            if date_col in df.index.names or df.index.name == date_col:
                df = df.reset_index()

        # Log what was filled
        filled_count = df_original.isna().sum().sum() - df.isna().sum().sum()
        if filled_count > 0:
            logger.info(f"  ✓ Filled {filled_count} null values (forward fill + median for leading NaNs)")

        return df

    # ========== CLEAN INDIVIDUAL DATASETS ==========

    def clean_fred(self) -> Tuple[pd.DataFrame, Dict, Dict]:
        """Clean FRED data with point-in-time correctness."""
        logger.info("\n" + "="*80)
        logger.info("CLEANING FRED DATA")
        logger.info("="*80)

        # Load
        filepath = self.raw_dir / 'fred_raw.csv'
        df = pd.read_csv(filepath)
        before_stats = self.compute_statistics(df, 'FRED')

        logger.info(f"\nBEFORE CLEANING:")
        logger.info(f"  Shape: {df.shape}")
        logger.info(f"  Missing values: {df.isna().sum().sum()} ({before_stats['missing_pct']}%)")
        logger.info(f"  Duplicates: {before_stats['duplicates']}")

        # Standardize column names
        if 'DATE' in df.columns:
            df.rename(columns={'DATE': 'Date'}, inplace=True)
        
        df['Date'] = pd.to_datetime(df['Date'])
        df.sort_values('Date', inplace=True)

        # Handle nulls (forward fill only)
        df = self.handle_nulls_no_lookahead(df, date_col='Date')

        # Remove duplicates
        df = df.drop_duplicates(subset=['Date'], keep='last')

        # After statistics
        after_stats = self.compute_statistics(df, 'FRED')

        logger.info(f"\nAFTER CLEANING:")
        logger.info(f"  Shape: {df.shape}")
        logger.info(f"  Missing values: {df.isna().sum().sum()} ({after_stats['missing_pct']}%)")
        logger.info(f"  Duplicates: {after_stats['duplicates']}")

        # Save
        output_path = self.clean_dir / 'fred_clean.csv'
        df.to_csv(output_path, index=False)
        logger.info(f"\n✓ Saved to: {output_path}")

        return df, before_stats, after_stats

    def clean_market(self) -> Tuple[pd.DataFrame, Dict, Dict]:
        """Clean market data (real-time, no lag needed)."""
        logger.info("\n" + "="*80)
        logger.info("CLEANING MARKET DATA")
        logger.info("="*80)

        # Load
        filepath = self.raw_dir / 'market_raw.csv'
        df = pd.read_csv(filepath)
        before_stats = self.compute_statistics(df, 'Market')

        logger.info(f"\nBEFORE CLEANING:")
        logger.info(f"  Shape: {df.shape}")
        logger.info(f"  Missing: {before_stats['total_missing']} ({before_stats['missing_pct']}%)")

        # Parse date
        df['Date'] = pd.to_datetime(df['Date'])
        df.sort_values('Date', inplace=True)

        # Rename columns for consistency
        if 'SP500' in df.columns:
            df.rename(columns={'SP500': 'SP500_Close'}, inplace=True)

        # Handle nulls (no reporting lag - market data is real-time)
        logger.info("\n  Market data is real-time (no reporting lag needed)")
        df = self.handle_nulls_no_lookahead(df, date_col='Date')

        # Remove duplicates
        df = df.drop_duplicates(subset=['Date'], keep='last')

        after_stats = self.compute_statistics(df, 'Market')

        logger.info(f"\nAFTER CLEANING:")
        logger.info(f"  Shape: {df.shape}")
        logger.info(f"  Missing: {after_stats['total_missing']} ({after_stats['missing_pct']}%)")

        output_path = self.clean_dir / 'market_clean.csv'
        df.to_csv(output_path, index=False)
        logger.info(f"\n✓ Saved to: {output_path}")

        return df, before_stats, after_stats

    def clean_company_prices(self) -> Tuple[pd.DataFrame, Dict, Dict]:
        """Clean company stock prices (real-time, no lag)."""
        logger.info("\n" + "="*80)
        logger.info("CLEANING COMPANY PRICES")
        logger.info("="*80)

        # Load
        filepath = self.raw_dir / 'company_prices_raw.csv'
        df = pd.read_csv(filepath)
        before_stats = self.compute_statistics(df, 'Company Prices')

        logger.info(f"\nBEFORE CLEANING:")
        logger.info(f"  Shape: {df.shape}")
        logger.info(f"  Companies: {df['Company'].nunique() if 'Company' in df.columns else 'N/A'}")
        logger.info(f"  Missing: {before_stats['total_missing']} ({before_stats['missing_pct']}%)")

        # Parse date
        df['Date'] = pd.to_datetime(df['Date'])

        # Keep needed columns, use Adj_Close (accounts for splits/dividends)
        keep_cols = ['Date', 'Close', 'Volume', 'Company', 'Company_Name', 'Sector']
        
        # Check which columns exist
        available_cols = [col for col in keep_cols if col in df.columns]
        
        # Handle Adj Close if it exists
        if 'Adj Close' in df.columns:
            df['Stock_Price'] = df['Adj Close']
        elif 'Adj_Close' in df.columns:
            df['Stock_Price'] = df['Adj_Close']
        elif 'Close' in df.columns:
            df['Stock_Price'] = df['Close']
        
        # Add Stock_Price to keep list
        if 'Stock_Price' not in available_cols:
            available_cols.append('Stock_Price')
        
        # Keep only available columns
        df = df[available_cols].copy()

        # Sort
        df.sort_values(['Company', 'Date'], inplace=True)

        # Handle nulls per company (no reporting lag - prices are real-time)
        logger.info("\n  Stock prices are real-time (no reporting lag needed)")
        df = self.handle_nulls_no_lookahead(df, date_col='Date', group_col='Company')

        # Remove duplicates
        df = df.drop_duplicates(subset=['Date', 'Company'], keep='last')

        after_stats = self.compute_statistics(df, 'Company Prices')

        logger.info(f"\nAFTER CLEANING:")
        logger.info(f"  Shape: {df.shape}")
        logger.info(f"  Missing: {after_stats['total_missing']} ({after_stats['missing_pct']}%)")

        # Per-company summary
        logger.info(f"\n  Per-company summary:")
        for company in df['Company'].unique()[:5]:  # Show first 5
            company_df = df[df['Company'] == company]
            logger.info(f"    {company}: {len(company_df):,} days, " +
                       f"{company_df['Date'].min()} to {company_df['Date'].max()}")

        output_path = self.clean_dir / 'company_prices_clean.csv'
        df.to_csv(output_path, index=False)
        logger.info(f"\n✓ Saved to: {output_path}")

        return df, before_stats, after_stats

    def clean_balance_sheet(self) -> Tuple[pd.DataFrame, Dict, Dict]:
        """Clean balance sheet with 45-day reporting lag."""
        logger.info("\n" + "="*80)
        logger.info("CLEANING BALANCE SHEET")
        logger.info("="*80)

        # Load
        filepath = self.raw_dir / 'company_balance_raw.csv'
        df = pd.read_csv(filepath)
        before_stats = self.compute_statistics(df, 'Balance Sheet')

        logger.info(f"\nBEFORE CLEANING:")
        logger.info(f"  Shape: {df.shape}")
        logger.info(f"  Companies: {df['Company'].nunique() if 'Company' in df.columns else 'N/A'}")
        logger.info(f"  Missing: {before_stats['total_missing']} ({before_stats['missing_pct']}%)")

        # Parse date
        df['Date'] = pd.to_datetime(df['Date'])
        df.sort_values(['Company', 'Date'], inplace=True)

        # CRITICAL: Apply 45-day reporting lag
        logger.info(f"\n⏰ Applying {self.REPORTING_LAGS['balance_sheet']}-day reporting lag...")
        logger.info("  Why: Balance sheets for Q1 (3/31) are filed ~45 days later (5/15)")
        logger.info("  Effect: Q1 data becomes 'available' on 5/15, not 3/31")

        df = self.apply_reporting_lag(df, lag_days=self.REPORTING_LAGS['balance_sheet'])

        logger.info(f"\n  Example transformation:")
        logger.info(f"    Q1 2020 (3/31) → Available {pd.Timestamp('2020-03-31') + pd.Timedelta(days=45)}")
        logger.info(f"    Q2 2020 (6/30) → Available {pd.Timestamp('2020-06-30') + pd.Timedelta(days=45)}")

        # Handle missing Long_Term_Debt
        logger.info("\n  Handling missing Long_Term_Debt...")
        before_ltd = df['Long_Term_Debt'].isna().sum() if 'Long_Term_Debt' in df.columns else 0
        if 'Long_Term_Debt' in df.columns:
            df['Long_Term_Debt'] = df.groupby('Company')['Long_Term_Debt'].ffill()
        after_ltd = df['Long_Term_Debt'].isna().sum() if 'Long_Term_Debt' in df.columns else 0
        logger.info(f"    Long_Term_Debt: {before_ltd} → {after_ltd} missing")

        # Calculate Total_Debt
        if 'Long_Term_Debt' in df.columns and 'Short_Term_Debt' in df.columns:
            df['Total_Debt'] = df['Long_Term_Debt'].fillna(0) + df['Short_Term_Debt'].fillna(0)

        # Handle other nulls per company (forward fill only)
        df = self.handle_nulls_no_lookahead(df, date_col='Date', group_col='Company')

        # Remove duplicates
        df = df.drop_duplicates(subset=['Date', 'Company'], keep='last')

        after_stats = self.compute_statistics(df, 'Balance Sheet')

        logger.info(f"\nAFTER CLEANING:")
        logger.info(f"  Shape: {df.shape}")
        logger.info(f"  Missing: {after_stats['total_missing']} ({after_stats['missing_pct']}%)")

        output_path = self.clean_dir / 'company_balance_clean.csv'
        df.to_csv(output_path, index=False)
        logger.info(f"\n✓ Saved to: {output_path}")

        return df, before_stats, after_stats

    def clean_income_statement(self) -> Tuple[pd.DataFrame, Dict, Dict]:
        """Clean income statement with 45-day reporting lag."""
        logger.info("\n" + "="*80)
        logger.info("CLEANING INCOME STATEMENT")
        logger.info("="*80)

        # Load
        filepath = self.raw_dir / 'company_income_raw.csv'
        df = pd.read_csv(filepath)
        before_stats = self.compute_statistics(df, 'Income Statement')

        logger.info(f"\nBEFORE CLEANING:")
        logger.info(f"  Shape: {df.shape}")
        logger.info(f"  Missing: {before_stats['total_missing']} ({before_stats['missing_pct']}%)")

        # Parse date
        df['Date'] = pd.to_datetime(df['Date'])
        df.sort_values(['Company', 'Date'], inplace=True)

        # Apply 45-day reporting lag
        logger.info(f"\n⏰ Applying {self.REPORTING_LAGS['earnings']}-day reporting lag...")
        df = self.apply_reporting_lag(df, lag_days=self.REPORTING_LAGS['earnings'])

        # Handle nulls per company (forward fill only)
        df = self.handle_nulls_no_lookahead(df, date_col='Date', group_col='Company')

        # Remove duplicates
        df = df.drop_duplicates(subset=['Date', 'Company'], keep='last')

        after_stats = self.compute_statistics(df, 'Income Statement')

        logger.info(f"\nAFTER CLEANING:")
        logger.info(f"  Shape: {df.shape}")
        logger.info(f"  Missing: {after_stats['total_missing']} ({after_stats['missing_pct']}%)")

        output_path = self.clean_dir / 'company_income_clean.csv'
        df.to_csv(output_path, index=False)
        logger.info(f"\n✓ Saved to: {output_path}")

        return df, before_stats, after_stats

    def save_statistics_report(self, all_stats: Dict):
        """Save detailed cleaning report."""
        report_data = []

        for dataset_name, stats_pair in all_stats.items():
            before = stats_pair['before']
            after = stats_pair['after']

            report_data.append({
                'Dataset': dataset_name,
                'Rows_Before': before['n_rows'],
                'Rows_After': after['n_rows'],
                'Missing_Before': before['total_missing'],
                'Missing_After': after['total_missing'],
                'Missing_Pct_Before': before['missing_pct'],
                'Missing_Pct_After': after['missing_pct'],
                'Duplicates_Before': before['duplicates'],
                'Duplicates_After': after['duplicates']
            })

        report_df = pd.DataFrame(report_data)
        report_path = self.report_dir / 'step1_cleaning_report.csv'
        report_df.to_csv(report_path, index=False)
        logger.info(f"\n✓ Cleaning report saved to: {report_path}")

        return report_df

    # ========== OUTLIER DETECTION (for reporting, not removal) ==========

    def detect_and_report_outliers(self, df: pd.DataFrame, name: str,
                                   columns_to_check: List[str],
                                   group_col: str = None) -> pd.DataFrame:
        """
        Detect outliers and create detailed report.
        Uses IQR method (robust to extreme values).
        DOES NOT REMOVE - just reports!
        """
        logger.info(f"\n📊 Detecting outliers in {name}...")

        outlier_records = []

        if group_col and group_col in df.columns:
            # Check outliers per group
            for group_name in df[group_col].unique():
                group_df = df[df[group_col] == group_name]

                for col in columns_to_check:
                    if col not in group_df.columns:
                        continue

                    outliers = self.detect_outliers(group_df, col, method='iqr', threshold=3.0)
                    n_outliers = outliers.sum()

                    if n_outliers > 0:
                        outlier_vals = group_df.loc[outliers, col]

                        outlier_records.append({
                            'Dataset': name,
                            'Group': group_name,
                            'Column': col,
                            'N_Outliers': n_outliers,
                            'Pct': round(n_outliers / len(group_df) * 100, 2),
                            'Min_Outlier': round(outlier_vals.min(), 2),
                            'Max_Outlier': round(outlier_vals.max(), 2),
                            'Outlier_Years': group_df.loc[outliers, 'Date'].dt.year.unique().tolist()
                        })
        else:
            # Check outliers for entire dataset
            for col in columns_to_check:
                if col not in df.columns:
                    continue

                outliers = self.detect_outliers(df, col, method='iqr', threshold=3.0)
                n_outliers = outliers.sum()

                if n_outliers > 0:
                    outlier_vals = df.loc[outliers, col]

                    outlier_records.append({
                        'Dataset': name,
                        'Column': col,
                        'N_Outliers': n_outliers,
                        'Pct': round(n_outliers / len(df) * 100, 2),
                        'Min_Outlier': round(outlier_vals.min(), 2),
                        'Max_Outlier': round(outlier_vals.max(), 2),
                        'Outlier_Years': df.loc[outliers, 'Date'].dt.year.unique().tolist()
                    })

        if outlier_records:
            outlier_df = pd.DataFrame(outlier_records)
            logger.info(f"\n  ⚠️  Outliers detected (NOT removed):")
            print(outlier_df.to_string(index=False))

            logger.info(f"\n  📌 Crisis years in data: 2008-2009 (Financial Crisis), 2020 (COVID)")
            logger.info(f"     → Outliers during these years are EXPECTED and VALID")

            # Save report
            report_path = self.report_dir / f'{name.lower().replace(" ", "_")}_outliers.csv'
            outlier_df.to_csv(report_path, index=False)
            logger.info(f"\n  ✓ Outlier report saved: {report_path}")

            return outlier_df
        else:
            logger.info("  ✓ No outliers detected")
            return pd.DataFrame()

    def detect_outliers(self, df: pd.DataFrame, column: str,
                       method: str = 'iqr', threshold: float = 3.0) -> pd.Series:
        """Detect outliers using IQR method."""
        if column not in df.columns or df[column].isna().all():
            return pd.Series([False] * len(df), index=df.index)

        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR

        outliers = (df[column] < lower_bound) | (df[column] > upper_bound)

        return outliers

    # ========== MASTER CLEANING PIPELINE ==========

    def clean_all(self) -> Dict[str, Tuple[pd.DataFrame, Dict, Dict]]:
        """Run complete point-in-time cleaning pipeline with full statistics."""
        logger.info("\n" + "="*80)
        logger.info("STEP 1: DATA CLEANING PIPELINE")
        logger.info("="*80)
        logger.info("\nKey Principles:")
        logger.info("  1. Forward fill ONLY (no backward fill = no look-ahead bias)")
        logger.info("  2. Apply reporting lags to quarterly financials (45 days)")
        logger.info("  3. Detect outliers but DON'T remove (crises are real!)")
        logger.info("  4. Per-company handling (no cross-contamination)")
        logger.info("="*80)

        overall_start = time.time()
        
        all_results = {}
        all_stats = {}

        # Clean each dataset
        logger.info("\n[1/5] Cleaning FRED data...")
        df_fred, before_fred, after_fred = self.clean_fred()
        all_results['fred'] = df_fred
        all_stats['fred'] = {'before': before_fred, 'after': after_fred}

        logger.info("\n[2/5] Cleaning Market data...")
        df_market, before_market, after_market = self.clean_market()
        all_results['market'] = df_market
        all_stats['market'] = {'before': before_market, 'after': after_market}

        logger.info("\n[3/5] Cleaning Company Prices...")
        df_prices, before_prices, after_prices = self.clean_company_prices()
        all_results['prices'] = df_prices
        all_stats['prices'] = {'before': before_prices, 'after': after_prices}

        logger.info("\n[4/5] Cleaning Balance Sheet...")
        df_balance, before_balance, after_balance = self.clean_balance_sheet()
        all_results['balance'] = df_balance
        all_stats['balance'] = {'before': before_balance, 'after': after_balance}

        logger.info("\n[5/5] Cleaning Income Statement...")
        df_income, before_income, after_income = self.clean_income_statement()
        all_results['income'] = df_income
        all_stats['income'] = {'before': before_income, 'after': after_income}

        # ========== PRINT BEFORE/AFTER COMPARISONS ==========

        logger.info("\n\n" + "="*80)
        logger.info("BEFORE vs AFTER COMPARISON - ALL DATASETS")
        logger.info("="*80)

        for name, stats in all_stats.items():
            self.print_statistics_comparison(stats['before'], stats['after'])

        # ========== DETECT OUTLIERS ==========

        logger.info("\n\n" + "="*80)
        logger.info("OUTLIER DETECTION REPORT (NOT REMOVED)")
        logger.info("="*80)

        self.detect_and_report_outliers(
            df_fred, 'FRED',
            columns_to_check=['GDP', 'CPI', 'Unemployment_Rate', 'Federal_Funds_Rate']
        )

        self.detect_and_report_outliers(
            df_market, 'Market',
            columns_to_check=['VIX', 'SP500_Close']
        )

        self.detect_and_report_outliers(
            df_prices, 'Company Prices',
            columns_to_check=['Stock_Price', 'Volume'],
            group_col='Company'
        )

        # ========== SAVE COMPREHENSIVE REPORT ==========

        summary_report = self.save_statistics_report(all_stats)

        # ========== FINAL SUMMARY ==========
        
        elapsed = time.time() - overall_start

        logger.info("\n\n" + "="*80)
        logger.info("STEP 1 COMPLETE - SUMMARY")
        logger.info("="*80)
        
        logger.info(f"\n📊 DATA CLEANED:")
        logger.info(f"  1. FRED:            {df_fred.shape[0]:,} rows × {df_fred.shape[1]} cols")
        logger.info(f"  2. Market:          {df_market.shape[0]:,} rows × {df_market.shape[1]} cols")
        logger.info(f"  3. Company Prices:  {df_prices.shape[0]:,} rows ({df_prices['Company'].nunique()} companies)")
        logger.info(f"  4. Balance Sheets:  {df_balance.shape[0]:,} rows ({df_balance['Company'].nunique()} companies)")
        logger.info(f"  5. Income Stmts:    {df_income.shape[0]:,} rows ({df_income['Company'].nunique()} companies)")

        logger.info(f"\n📁 OUTPUT FILES (data/clean/):")
        logger.info(f"  - fred_clean.csv")
        logger.info(f"  - market_clean.csv")
        logger.info(f"  - company_prices_clean.csv")
        logger.info(f"  - company_balance_clean.csv")
        logger.info(f"  - company_income_clean.csv")

        logger.info(f"\n📊 REPORTS (data/reports/):")
        logger.info(f"  - step1_cleaning_report.csv")
        logger.info(f"  - Outlier reports for each dataset")

        logger.info(f"\n⏱️  Total Time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
        logger.info("="*80)
        
        logger.info("\n✅ Step 1 Complete!")
        logger.info("\n➡️  Next Steps:")
        logger.info("   1. Run validation: python src/validation/validate_checkpoint_2_clean.py")
        logger.info("   2. If validation passes: python step2_feature_engineering.py")

        return all_results, all_stats


def main():
    """Execute Step 1: Data Cleaning."""

    cleaner = PointInTimeDataCleaner(raw_dir="data/raw", clean_dir="data/clean")
    
    try:
        cleaned_data, statistics = cleaner.clean_all()
        
        logger.info("\n" + "="*80)
        logger.info("✅ STEP 1 SUCCESSFULLY COMPLETED")
        logger.info("="*80)
        
        return cleaned_data, statistics
        
    except FileNotFoundError as e:
        logger.error(f"\n❌ ERROR: {e}")
        logger.error("Make sure raw data files exist in data/raw/")
        logger.error("Run Step 0 first: python step0_data_collection.py")
        return None, None
    except Exception as e:
        logger.error(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return None, None


if __name__ == "__main__":
    cleaned, stats = main()