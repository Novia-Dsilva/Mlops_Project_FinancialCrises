"""
STEP 3: DATA MERGING

Combine feature-engineered datasets into two final merged datasets:

Pipeline 1 (VAE - Scenario Generation):
    macro_features.csv = FRED + Market
    
    - Daily macro/market data only
    - Used to train VAE for generating stress scenarios
    - ~5,500 rows × ~65 columns

Pipeline 2 (XGBoost/LSTM - Prediction):
    merged_features.csv = FRED + Market + Company
    - Daily company-date observations with full macro context
    - Used to train predictive models
    - ~187,000 rows × ~135 columns (25 companies × ~7,500 days)

Merge Strategy:
- Pipeline 1: Simple merge on Date (outer join)
- Pipeline 2: Merge macro+market first, then merge with company data on Date
- Handle missing values appropriately
- Validate merge quality

Input:  data/features/fred_features.csv, market_features.csv, company_features.csv
Output: data/features/macro_features.csv, merged_features.csv

Usage:
    python step3_data_merging.py

Next Step:
    python src/validation/validate_checkpoint_3_merged.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, Tuple
import time
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


class DataMerger:
    """Merge feature-engineered datasets into final datasets for modeling."""

    def __init__(self, features_dir: str = "data/features"):
        self.features_dir = Path(features_dir)
        self.reports_dir = Path("data/reports")
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    # ========== LOAD FEATURE-ENGINEERED DATA ==========

    # def load_feature_datasets(self) -> Dict[str, pd.DataFrame]:
    #     """Load all feature-engineered datasets from Step 2."""
    #     logger.info("="*80)
    #     logger.info("LOADING FEATURE-ENGINEERED DATASETS")
    #     logger.info("="*80)

    #     data = {}

    #     # Load FRED features
    #     fred_path = self.features_dir / 'fred_features.csv'
    #     if fred_path.exists():
    #         data['fred'] = pd.read_csv(fred_path, parse_dates=['Date'])
    #         logger.info(f"\n✓ Loaded fred_features: {data['fred'].shape}")
    #         logger.info(f"  Date range: {data['fred']['Date'].min().date()} to {data['fred']['Date'].max().date()}")
    #         logger.info(f"  Columns: {list(data['fred'].columns[:10])}...")
    #     else:
    #         logger.error(f"\n❌ fred_features.csv not found!")
    #         raise FileNotFoundError(f"{fred_path} does not exist. Run Step 2 first.")

    #     # Load Market features
    #     market_path = self.features_dir / 'market_features.csv'
    #     if market_path.exists():
    #         data['market'] = pd.read_csv(market_path, parse_dates=['Date'])
    #         logger.info(f"\n✓ Loaded market_features: {data['market'].shape}")
    #         logger.info(f"  Date range: {data['market']['Date'].min().date()} to {data['market']['Date'].max().date()}")
    #         logger.info(f"  Columns: {list(data['market'].columns[:10])}...")
    #     else:
    #         logger.error(f"\n❌ market_features.csv not found!")
    #         raise FileNotFoundError(f"{market_path} does not exist. Run Step 2 first.")

    #     # Load Company features
    #     company_path = self.features_dir / 'company_features.csv'
    #     if company_path.exists():
    #         data['company'] = pd.read_csv(company_path, parse_dates=['Date'])
    #         logger.info(f"\n✓ Loaded company_features: {data['company'].shape}")
    #         logger.info(f"  Date range: {data['company']['Date'].min().date()} to {data['company']['Date'].max().date()}")
    #         logger.info(f"  Companies: {data['company']['Company'].nunique()}")
    #         logger.info(f"    {sorted(data['company']['Company'].unique())}")
    #     else:
    #         logger.error(f"\n❌ company_features.csv not found!")
    #         raise FileNotFoundError(f"{company_path} does not exist. Run Step 2 first.")

    #     return data
    def load_feature_datasets(self) -> Dict[str, pd.DataFrame]:
        """Load all feature-engineered datasets from Step 2."""
        logger.info("="*80)
        logger.info("LOADING FEATURE-ENGINEERED DATASETS")
        logger.info("="*80)

        data = {}

        # --- FRED FEATURES ---
        fred_path = self.features_dir / 'fred_features.csv'
        if fred_path.exists():
            data['fred'] = pd.read_csv(fred_path, parse_dates=['Date'])
            # ✅ Defensive datetime enforcement
            data['fred']['Date'] = pd.to_datetime(data['fred']['Date'], errors='coerce')
            logger.info(f"\n✓ Loaded fred_features: {data['fred'].shape}")
            logger.info(f"  Date range: {data['fred']['Date'].min().date()} to {data['fred']['Date'].max().date()}")
            logger.info(f"  Columns: {list(data['fred'].columns[:10])}...")
        else:
            logger.error(f"\n❌ fred_features.csv not found!")
            raise FileNotFoundError(f"{fred_path} does not exist. Run Step 2 first.")

        # --- MARKET FEATURES ---
        market_path = self.features_dir / 'market_features.csv'
        if market_path.exists():
            data['market'] = pd.read_csv(market_path, parse_dates=['Date'])
            # ✅ Defensive datetime enforcement
            data['market']['Date'] = pd.to_datetime(data['market']['Date'], errors='coerce')
            logger.info(f"\n✓ Loaded market_features: {data['market'].shape}")
            logger.info(f"  Date range: {data['market']['Date'].min().date()} to {data['market']['Date'].max().date()}")
            logger.info(f"  Columns: {list(data['market'].columns[:10])}...")
        else:
            logger.error(f"\n❌ market_features.csv not found!")
            raise FileNotFoundError(f"{market_path} does not exist. Run Step 2 first.")

        # --- COMPANY FEATURES ---
        company_path = self.features_dir / 'company_features.csv'
        if company_path.exists():
            data['company'] = pd.read_csv(company_path, parse_dates=['Date'])
            # ✅ Defensive datetime enforcement
            data['company']['Date'] = pd.to_datetime(data['company']['Date'], errors='coerce')
            logger.info(f"\n✓ Loaded company_features: {data['company'].shape}")
            logger.info(f"  Date range: {data['company']['Date'].min().date()} to {data['company']['Date'].max().date()}")
            logger.info(f"  Companies: {data['company']['Company'].nunique()}")
            logger.info(f"    {sorted(data['company']['Company'].unique())}")
        else:
            logger.error(f"\n❌ company_features.csv not found!")
            raise FileNotFoundError(f"{company_path} does not exist. Run Step 2 first.")

        return data

    # ========== MERGE PIPELINE 1: MACRO + MARKET ==========

    def merge_pipeline1(self, fred_df: pd.DataFrame, market_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge FRED and Market data for Pipeline 1 (VAE).

        Strategy: Outer join on Date to keep all dates from both datasets.
        
        Args:
            fred_df: FRED features DataFrame
            market_df: Market features DataFrame
            
        Returns:
            Merged DataFrame for VAE training
        """
        logger.info("\n" + "="*80)
        logger.info("PIPELINE 1: MERGING FRED + MARKET (FOR VAE)")
        logger.info("="*80)

        logger.info(f"\nInput datasets:")
        logger.info(f"  FRED:   {fred_df.shape} ({fred_df['Date'].min().date()} to {fred_df['Date'].max().date()})")
        logger.info(f"  Market: {market_df.shape} ({market_df['Date'].min().date()} to {market_df['Date'].max().date()})")

        # Check for overlapping date ranges
        fred_dates = set(fred_df['Date'])
        market_dates = set(market_df['Date'])
        overlap = len(fred_dates & market_dates)
        
        logger.info(f"\n  Date overlap: {overlap:,} common dates")
        logger.info(f"  FRED-only dates: {len(fred_dates - market_dates):,}")
        logger.info(f"  Market-only dates: {len(market_dates - fred_dates):,}")

        # Merge on Date (outer join to keep all dates)
        logger.info(f"\nMerging on: Date (outer join)")
        
        merged = pd.merge(
            fred_df,
            market_df,
            on='Date',
            how='outer',
            suffixes=('_fred', '_market')
        )

        # Sort by date
        merged.sort_values('Date', inplace=True)
        merged.reset_index(drop=True, inplace=True)

        logger.info(f"\n✓ Merged shape: {merged.shape}")
        logger.info(f"  Date range: {merged['Date'].min().date()} to {merged['Date'].max().date()}")
        logger.info(f"  Total days: {len(merged):,}")

        # === CHECK FOR DUPLICATE COLUMNS ===
        logger.info(f"\nChecking for duplicate columns...")
        
        duplicate_cols = [col for col in merged.columns if col.endswith('_fred') or col.endswith('_market')]
        
        if duplicate_cols:
            logger.warning(f"  ⚠️  Found {len(duplicate_cols)} duplicate columns with suffixes:")
            for col in duplicate_cols[:5]:
                logger.warning(f"     - {col}")
            logger.info(f"  Note: These will be cleaned in Step 3c")
        else:
            logger.info(f"  ✓ No duplicate columns")

        # === CHECK FOR MISSING VALUES ===
        logger.info(f"\nChecking missing values after merge...")
        
        missing_pct = (merged.isna().sum() / len(merged)) * 100
        high_missing = missing_pct[missing_pct > 5].sort_values(ascending=False)

        if len(high_missing) > 0:
            logger.warning(f"\n  ⚠️  Columns with >5% missing values:")
            for col, pct in high_missing.head(10).items():
                logger.warning(f"     {col}: {pct:.1f}%")

            logger.info(f"\n  Filling missing values with forward fill...")
            merged = merged.ffill().bfill()

            # Check again
            missing_after = (merged.isna().sum() / len(merged)) * 100
            total_missing = missing_after.sum()
            logger.info(f"  ✓ Total missing after fill: {total_missing:.2f}%")
        else:
            logger.info(f"  ✓ No significant missing values")

        # === VERIFY KEY COLUMNS ===
        logger.info(f"\nVerifying key columns...")
        
        key_macro_cols = ['GDP', 'CPI', 'Unemployment_Rate', 'Federal_Funds_Rate']
        key_market_cols = ['VIX', 'SP500_Return_1D']
        
        # Handle SP500_Close vs SP500
        if 'SP500_Close' in merged.columns:
            key_market_cols.append('SP500_Close')
        elif 'SP500' in merged.columns:
            key_market_cols.append('SP500')

        missing_key_cols = []
        for col in key_macro_cols + key_market_cols:
            if col not in merged.columns:
                missing_key_cols.append(col)

        if missing_key_cols:
            logger.warning(f"  ⚠️  Key columns not found: {missing_key_cols}")
        else:
            logger.info(f"  ✓ All key columns present")

        return merged

    # ========== MERGE PIPELINE 2: MACRO + MARKET + COMPANY ==========

    def merge_pipeline2(self, fred_df: pd.DataFrame, market_df: pd.DataFrame,
                       company_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge FRED, Market, and Company data for Pipeline 2 (XGBoost/LSTM).

        Strategy:
        1. Merge FRED + Market on Date (same as Pipeline 1)
        2. Merge result with Company on Date
        3. This creates Company-Date observations with full macro context
        
        Args:
            fred_df: FRED features
            market_df: Market features
            company_df: Company features
            
        Returns:
            Merged DataFrame for prediction models
        """
        logger.info("\n" + "="*80)
        logger.info("PIPELINE 2: MERGING FRED + MARKET + COMPANY (FOR XGBOOST/LSTM)")
        logger.info("="*80)

        # === STEP 1: Merge FRED + Market ===
        logger.info(f"\nStep 1: Merging FRED + Market...")
        
        macro_market = pd.merge(
            fred_df,
            market_df,
            on='Date',
            how='outer',
            suffixes=('_fred', '_market')
        )
        
        macro_market.sort_values('Date', inplace=True)
        logger.info(f"  ✓ Macro+Market shape: {macro_market.shape}")

        # === STEP 2: Merge with Company data ===
        logger.info(f"\nStep 2: Merging (Macro+Market) with Company data...")
        logger.info(f"  Company data shape: {company_df.shape}")
        logger.info(f"  Companies: {company_df['Company'].nunique()}")
        logger.info(f"  Date range: {company_df['Date'].min().date()} to {company_df['Date'].max().date()}")

        # Check date alignment
        company_dates = set(company_df['Date'])
        macro_dates = set(macro_market['Date'])
        overlap = len(company_dates & macro_dates)
        
        logger.info(f"\n  Date alignment check:")
        logger.info(f"    Common dates: {overlap:,}")
        logger.info(f"    Company-only dates: {len(company_dates - macro_dates):,}")
        logger.info(f"    Macro-only dates: {len(macro_dates - company_dates):,}")

        # Merge on Date (left join - keep all company-date observations)
        logger.info(f"\nMerging on: Date (left join from Company)")
        
        merged = pd.merge(
            company_df,
            macro_market,
            on='Date',
            how='left',
            suffixes=('', '_macro')
        )

        # Sort by Company and Date
        merged.sort_values(['Company', 'Date'], inplace=True)
        merged.reset_index(drop=True, inplace=True)

        logger.info(f"\n✓ Final merged shape: {merged.shape}")
        logger.info(f"  Companies: {merged['Company'].nunique()}")
        logger.info(f"  Date range: {merged['Date'].min().date()} to {merged['Date'].max().date()}")
        logger.info(f"  Rows per company: ~{len(merged) / merged['Company'].nunique():.0f}")

        # === MERGE QUALITY CHECK ===
        logger.info(f"\n" + "="*80)
        logger.info(f"MERGE QUALITY CHECK")
        logger.info(f"="*80)

        # Check for missing values by source
        company_cols = [col for col in company_df.columns if col not in ['Date', 'Company']]
        fred_cols = [col for col in fred_df.columns if col not in ['Date']]
        market_cols = [col for col in market_df.columns if col not in ['Date']]

        missing_pct = (merged.isna().sum() / len(merged)) * 100

        logger.info(f"\nMissing values by source:")

        # Company features
        company_missing = missing_pct[[col for col in company_cols if col in merged.columns]].mean() if company_cols else 0
        logger.info(f"  Company features: {company_missing:.1f}% avg missing")

        # Macro features
        macro_missing = missing_pct[[col for col in fred_cols if col in merged.columns]].mean() if fred_cols else 0
        logger.info(f"  Macro features:   {macro_missing:.1f}% avg missing")

        # Market features
        market_missing = missing_pct[[col for col in market_cols if col in merged.columns]].mean() if market_cols else 0
        logger.info(f"  Market features:  {market_missing:.1f}% avg missing")

        # Overall
        total_missing = missing_pct.mean()
        logger.info(f"  Overall:          {total_missing:.1f}% avg missing")

        # === HANDLE MISSING VALUES FROM MERGE ===
        if total_missing > 1:
            logger.info(f"\n  Filling missing values from merge misalignment...")

            # For each company separately (to avoid cross-contamination)
            filled_dfs = []
            for company in merged['Company'].unique():
                company_data = merged[merged['Company'] == company].copy()

                # Forward fill within company
                company_data = company_data.ffill()

                # Backward fill any remaining (at start of series)
                company_data = company_data.bfill()

                filled_dfs.append(company_data)

            merged = pd.concat(filled_dfs, ignore_index=True)
            merged.sort_values(['Company', 'Date'], inplace=True)

            # Check after filling
            missing_after = (merged.isna().sum() / len(merged)) * 100
            total_missing_after = missing_after.mean()
            logger.info(f"  ✓ Overall missing after fill: {total_missing_after:.2f}%")
        else:
            logger.info(f"  ✓ Minimal missing values, no filling needed")

        # === VERIFY DATA INTEGRITY FOR SAMPLE COMPANY ===
        logger.info(f"\n" + "="*80)
        logger.info(f"DATA INTEGRITY CHECK (Sample Company)")
        logger.info(f"="*80)

        sample_company = merged['Company'].iloc[0]
        sample_data = merged[merged['Company'] == sample_company]

        logger.info(f"\nCompany: {sample_company}")
        logger.info(f"  Total rows: {len(sample_data):,}")
        logger.info(f"  Date range: {sample_data['Date'].min().date()} to {sample_data['Date'].max().date()}")

        # Check key columns have data
        key_checks = {
            'Stock_Price': 'Company data',
            'Revenue': 'Company financials',
            'GDP': 'Macro data',
            'VIX': 'Market data'
        }

        logger.info(f"\n  Key columns availability:")
        for col, source in key_checks.items():
            # Handle alternative column names
            check_col = col
            if col == 'Stock_Price' and col not in sample_data.columns:
                if 'Close' in sample_data.columns:
                    check_col = 'Close'
            
            if check_col in sample_data.columns:
                avail_count = sample_data[check_col].notna().sum()
                avail_pct = (avail_count / len(sample_data)) * 100
                logger.info(f"    {col:20s} ({source:20s}): {avail_count:6,} rows ({avail_pct:5.1f}%)")
            else:
                logger.warning(f"    {col:20s} ({source:20s}): ❌ NOT FOUND")

        # Show sample rows
        logger.info(f"\n  Sample rows (first 3):")
        display_cols = ['Date', 'Company', 'Stock_Price', 'Revenue', 'GDP', 'VIX']
        
        # Handle alternative column names
        if 'Stock_Price' not in sample_data.columns and 'Close' in sample_data.columns:
            display_cols[display_cols.index('Stock_Price')] = 'Close'
        
        available_display = [col for col in display_cols if col in sample_data.columns]
        print(sample_data[available_display].head(3).to_string(index=False))

        return merged

    # ========== CHECK DUPLICATE COLUMNS ==========

    def check_duplicate_columns(self, df: pd.DataFrame, pipeline_name: str):
        """Check for duplicate columns with suffixes."""
        logger.info(f"\nChecking for duplicate columns in {pipeline_name}...")
        
        # Look for common suffixes
        suffixes = ['_x', '_y', '_fred', '_market', '_macro', '_company']
        duplicate_cols = [col for col in df.columns if any(col.endswith(suffix) for suffix in suffixes)]
        
        if duplicate_cols:
            logger.warning(f"  ⚠️  Found {len(duplicate_cols)} columns with suffixes:")
            
            # Group by base name
            dup_groups = {}
            for col in duplicate_cols:
                for suffix in suffixes:
                    if col.endswith(suffix):
                        base = col[:-len(suffix)]
                        if base not in dup_groups:
                            dup_groups[base] = []
                        dup_groups[base].append(col)
            
            for base, cols in list(dup_groups.items())[:5]:  # Show first 5
                logger.warning(f"     {base}: {cols}")
            
            if len(dup_groups) > 5:
                logger.warning(f"     ... and {len(dup_groups) - 5} more")
            
            logger.info(f"  Note: These will be cleaned in Step 3c (Post-Merge Cleaning)")
        else:
            logger.info(f"  ✓ No duplicate columns with suffixes")

    # ========== SAVE MERGED DATASETS ==========

    def save_merged_datasets(self, pipeline1_df: pd.DataFrame, pipeline2_df: pd.DataFrame):
        """Save merged datasets to CSV format."""
        logger.info("\n" + "="*80)
        logger.info("SAVING MERGED DATASETS")
        logger.info("="*80)

        # === Save Pipeline 1 ===
        pipeline1_path = self.features_dir / 'macro_features.csv'
        pipeline1_df.to_csv(pipeline1_path, index=False)
        
        logger.info(f"\n✓ Saved Pipeline 1 (VAE):")
        logger.info(f"  Path:  {pipeline1_path}")
        logger.info(f"  Shape: {pipeline1_df.shape}")
        logger.info(f"  Size:  {pipeline1_path.stat().st_size / 1024 / 1024:.2f} MB")

        # === Save Pipeline 2 ===
        pipeline2_path = self.features_dir / 'merged_features.csv'
        pipeline2_df.to_csv(pipeline2_path, index=False)
        
        logger.info(f"\n✓ Saved Pipeline 2 (XGBoost/LSTM):")
        logger.info(f"  Path:  {pipeline2_path}")
        logger.info(f"  Shape: {pipeline2_df.shape}")
        logger.info(f"  Size:  {pipeline2_path.stat().st_size / 1024 / 1024:.2f} MB")

        # === Save column lists for reference ===
        logger.info(f"\nSaving column lists for reference...")
        
        with open(self.features_dir / 'pipeline1_columns.txt', 'w') as f:
            f.write("PIPELINE 1 (VAE) - COLUMN LIST\n")
            f.write("="*80 + "\n\n")
            for col in sorted(pipeline1_df.columns):
                f.write(f"{col}\n")

        with open(self.features_dir / 'pipeline2_columns.txt', 'w') as f:
            f.write("PIPELINE 2 (XGBOOST/LSTM) - COLUMN LIST\n")
            f.write("="*80 + "\n\n")
            for col in sorted(pipeline2_df.columns):
                f.write(f"{col}\n")

        logger.info(f"  ✓ Column lists saved:")
        logger.info(f"     - {self.features_dir / 'pipeline1_columns.txt'}")
        logger.info(f"     - {self.features_dir / 'pipeline2_columns.txt'}")

    # ========== GENERATE MERGE REPORT ==========

    def generate_merge_report(self, fred_df: pd.DataFrame, market_df: pd.DataFrame,
                            company_df: pd.DataFrame, pipeline1_df: pd.DataFrame,
                            pipeline2_df: pd.DataFrame):
        """Generate detailed merge quality report."""
        logger.info(f"\nGenerating merge quality report...")
        
        report_data = []

        # Pipeline 1 stats
        report_data.append({
            'Pipeline': 'Pipeline 1 (VAE)',
            'Dataset': 'macro_features.csv',
            'Input_Datasets': 'FRED + Market',
            'Rows': len(pipeline1_df),
            'Columns': len(pipeline1_df.columns),
            'Missing_Pct': (pipeline1_df.isna().sum().sum() / pipeline1_df.size) * 100,
            'Date_Range_Days': (pipeline1_df['Date'].max() - pipeline1_df['Date'].min()).days
        })

        # Pipeline 2 stats
        report_data.append({
            'Pipeline': 'Pipeline 2 (Prediction)',
            'Dataset': 'merged_features.csv',
            'Input_Datasets': 'FRED + Market + Company',
            'Rows': len(pipeline2_df),
            'Columns': len(pipeline2_df.columns),
            'Missing_Pct': (pipeline2_df.isna().sum().sum() / pipeline2_df.size) * 100,
            'Date_Range_Days': (pipeline2_df['Date'].max() - pipeline2_df['Date'].min()).days
        })

        report_df = pd.DataFrame(report_data)
        report_path = self.reports_dir / 'step3_merge_report.csv'
        report_df.to_csv(report_path, index=False)
        
        logger.info(f"  ✓ Report saved: {report_path}")
        
        return report_df

    # ========== MAIN PIPELINE ==========

    def run_merging_pipeline(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Execute complete data merging pipeline."""
        logger.info("\n" + "="*80)
        logger.info("STEP 3: DATA MERGING PIPELINE")
        logger.info("="*80)
        logger.info("\nCreating two merged datasets:")
        logger.info("  1. macro_features.csv (FRED + Market) for VAE")
        logger.info("  2. merged_features.csv (FRED + Market + Company) for XGBoost/LSTM")
        logger.info("="*80)

        overall_start = time.time()

        # Load feature datasets
        data = self.load_feature_datasets()

        # === MERGE PIPELINE 1: FRED + Market ===
        pipeline1_merged = self.merge_pipeline1(data['fred'], data['market'])
        
        # Check for duplicates
        self.check_duplicate_columns(pipeline1_merged, "Pipeline 1")

        # === MERGE PIPELINE 2: FRED + Market + Company ===
        pipeline2_merged = self.merge_pipeline2(data['fred'], data['market'], data['company'])
        
        # Check for duplicates
        self.check_duplicate_columns(pipeline2_merged, "Pipeline 2")

        # === SAVE MERGED DATASETS ===
        self.save_merged_datasets(pipeline1_merged, pipeline2_merged)

        # === GENERATE REPORT ===
        report = self.generate_merge_report(
            data['fred'], data['market'], data['company'],
            pipeline1_merged, pipeline2_merged
        )

        # === FINAL SUMMARY ===
        elapsed = time.time() - overall_start

        logger.info("\n" + "="*80)
        logger.info("MERGING COMPLETE - SUMMARY")
        logger.info("="*80)

        print("\n" + report.to_string(index=False))

        logger.info(f"\n📊 PIPELINE 1 (VAE - Scenario Generation):")
        logger.info(f"  Dataset:    macro_features.csv")
        logger.info(f"  Purpose:    Train VAE to generate stress scenarios")
        logger.info(f"  Shape:      {pipeline1_merged.shape[0]:,} rows × {pipeline1_merged.shape[1]} columns")
        logger.info(f"  Frequency:  Daily")
        logger.info(f"  Date range: {pipeline1_merged['Date'].min().date()} to {pipeline1_merged['Date'].max().date()}")
        logger.info(f"  Features:   Macro + Market indicators")

        logger.info(f"\n📊 PIPELINE 2 (XGBoost/LSTM - Prediction):")
        logger.info(f"  Dataset:    merged_features.csv")
        logger.info(f"  Purpose:    Train models to predict company outcomes")
        logger.info(f"  Shape:      {pipeline2_merged.shape[0]:,} rows × {pipeline2_merged.shape[1]} columns")
        logger.info(f"  Frequency:  Daily")
        logger.info(f"  Companies:  {pipeline2_merged['Company'].nunique()}")
        logger.info(f"  Date range: {pipeline2_merged['Date'].min().date()} to {pipeline2_merged['Date'].max().date()}")
        logger.info(f"  Features:   Macro + Market + Company indicators")

        logger.info(f"\n⏱️  Total Time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
        logger.info("="*80)

        logger.info("\n✅ Step 3 Complete!")
        logger.info("\n➡️  Next Steps:")
        logger.info("   1. Validate merged data: python src/validation/validate_checkpoint_3_merged.py")
        logger.info("   2. Clean merged data: python step3c_post_merge_cleaning.py")
        logger.info("   3. Then interaction features: python step3b_interaction_features.py")

        return pipeline1_merged, pipeline2_merged


def main():
    """Execute Step 3: Data Merging."""

    merger = DataMerger(features_dir="data/features")

    try:
        pipeline1, pipeline2 = merger.run_merging_pipeline()

        # === SHOW SAMPLE DATA ===
        logger.info("\n" + "="*80)
        logger.info("SAMPLE DATA PREVIEW")
        logger.info("="*80)

        logger.info("\n1. PIPELINE 1 (macro_features.csv) - First 5 rows, key columns:")
        p1_cols = ['Date', 'GDP', 'CPI', 'Unemployment_Rate', 'VIX', 'SP500_Return_1D']
        available_p1_cols = [col for col in p1_cols if col in pipeline1.columns]
        print(pipeline1[available_p1_cols].head().to_string())

        logger.info("\n2. PIPELINE 2 (merged_features.csv) - First 5 rows, key columns:")
        p2_cols = ['Date', 'Company', 'Stock_Price', 'Revenue', 'GDP', 'VIX', 'Stock_Return_1D']
        
        # Handle alternative column names
        if 'Stock_Price' not in pipeline2.columns and 'Close' in pipeline2.columns:
            p2_cols[p2_cols.index('Stock_Price')] = 'Close'
        
        available_p2_cols = [col for col in p2_cols if col in pipeline2.columns]
        print(pipeline2[available_p2_cols].head().to_string())

        logger.info("\n" + "="*80)
        logger.info("COLUMN COUNT BY DATASET")
        logger.info("="*80)
        logger.info(f"Pipeline 1 (macro_features):  {len(pipeline1.columns)} columns")
        logger.info(f"Pipeline 2 (merged_features): {len(pipeline2.columns)} columns")

        # Show feature categories
        logger.info("\n" + "="*80)
        logger.info("FEATURE CATEGORIES IN PIPELINE 2")
        logger.info("="*80)
        
        # Count features by category
        macro_features = [col for col in pipeline2.columns if any(
            keyword in col for keyword in ['GDP', 'CPI', 'Unemployment', 'Federal', 'Inflation']
        )]
        
        market_features = [col for col in pipeline2.columns if any(
            keyword in col for keyword in ['VIX', 'SP500']
        )]
        
        stock_features = [col for col in pipeline2.columns if any(
            keyword in col for keyword in ['Stock', 'Close', 'Volume', 'RSI', 'MACD']
        )]
        
        financial_features = [col for col in pipeline2.columns if any(
            keyword in col for keyword in ['Revenue', 'Income', 'Assets', 'Debt', 'Equity', 'ROE', 'ROA', 'Margin']
        )]
        
        logger.info(f"Macro features:     {len(macro_features)}")
        logger.info(f"Market features:    {len(market_features)}")
        logger.info(f"Stock features:     {len(stock_features)}")
        logger.info(f"Financial features: {len(financial_features)}")

        logger.info("\n" + "="*80)
        logger.info("✅ STEP 3 SUCCESSFULLY COMPLETED")
        logger.info("="*80)

        return pipeline1, pipeline2

    except FileNotFoundError as e:
        logger.error(f"\n❌ ERROR: {e}")
        logger.error("\nMake sure you've run Step 2 first!")
        logger.error("  Run: python step2_feature_engineering.py")
        return None, None
    except Exception as e:
        logger.error(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return None, None


if __name__ == "__main__":
    merged_data = main()