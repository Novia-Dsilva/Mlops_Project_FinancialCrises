"""
COMPLETE QUARTERLY DATA PIPELINE - WITH FRED & MARKET DATA
===========================================================

This script creates ONE ROW PER QUARTER PER COMPANY with:
1. Quarterly fundamentals (Balance Sheet + Income Statement)  
2. Quarterly stock price aggregates
3. Quarterly macro aggregates (FRED: GDP, CPI, Unemployment, etc.)
4. Quarterly market aggregates (VIX, SP500)

Output: ONE ROW PER QUARTER PER COMPANY with all features

Author: Financial ML Pipeline
Date: 2025
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import timedelta
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class QuarterlyDataPipeline:
    """Clean and merge financial data at quarterly frequency."""
    
    REPORTING_LAG_DAYS = 45  # Quarterly financials reported 45 days after quarter end
    
    def __init__(self, raw_dir='data/raw', output_dir='data/processed'):
        self.raw_dir = Path(raw_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    # ========== HELPER FUNCTIONS ==========
    
    def get_quarter_end(self, date):
        """Get the quarter end date for any given date."""
        quarter = (date.month - 1) // 3 + 1
        if quarter == 1:
            return pd.Timestamp(year=date.year, month=3, day=31)
        elif quarter == 2:
            return pd.Timestamp(year=date.year, month=6, day=30)
        elif quarter == 3:
            return pd.Timestamp(year=date.year, month=9, day=30)
        else:
            return pd.Timestamp(year=date.year, month=12, day=31)
    
    def add_quarter_info(self, df):
        """Add quarter information columns."""
        df['Year'] = df['Date'].dt.year
        df['Quarter_Num'] = df['Date'].dt.quarter
        df['Quarter'] = df['Year'].astype(str) + 'Q' + df['Quarter_Num'].astype(str)
        df['Quarter_End_Date'] = df['Date'].apply(self.get_quarter_end)
        return df
    
    # ========== CLEANING FUNCTIONS ==========
    
    def clean_quarterly_fundamentals(self, df, data_type='balance'):
        """Clean quarterly fundamental data."""
        logger.info(f"\nCleaning {data_type.title()} Sheet...")
        df = df.copy()
        
        # Parse dates
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values(['Company', 'Date']).reset_index(drop=True)
        
        logger.info(f"  Raw shape: {df.shape}")
        logger.info(f"  Companies: {df['Company'].nunique()}")
        logger.info(f"  Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")
        
        # Store original quarter end dates BEFORE applying lag
        df = self.add_quarter_info(df)
        df['Original_Quarter_End'] = df['Date'].copy()
        
        # Apply 45-day reporting lag
        logger.info(f"  Applying {self.REPORTING_LAG_DAYS}-day reporting lag...")
        df['Date'] = df['Date'] + pd.Timedelta(days=self.REPORTING_LAG_DAYS)
        
        # Handle specific columns
        if data_type == 'balance':
            if 'Long_Term_Debt' in df.columns:
                for company in df['Company'].unique():
                    mask = df['Company'] == company
                    median_debt = df.loc[mask, 'Long_Term_Debt'].median()
                    if not pd.isna(median_debt):
                        df.loc[mask, 'Long_Term_Debt'] = df.loc[mask, 'Long_Term_Debt'].fillna(median_debt)
            
            if 'Total_Debt' not in df.columns:
                if 'Long_Term_Debt' in df.columns and 'Short_Term_Debt' in df.columns:
                    df['Total_Debt'] = df['Long_Term_Debt'].fillna(0) + df['Short_Term_Debt'].fillna(0)
        
        elif data_type == 'income':
            if 'EPS' in df.columns:
                null_count = df['EPS'].isna().sum()
                if null_count > 0:
                    logger.info(f"  Filling {null_count:,} EPS nulls with 0")
                    df['EPS'] = df['EPS'].fillna(0.0)
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['Date', 'Company'], keep='last')
        
        logger.info(f"  Final shape: {df.shape}")
        logger.info(f"  ✓ One row per quarter per company")
        
        return df
    
    def clean_quarterly_prices(self, df):
        """Clean quarterly stock prices and calculate returns."""
        logger.info("\nCleaning Quarterly Stock Prices...")
        df = df.copy()
        
        # Parse dates
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values(['Company', 'Date']).reset_index(drop=True)
        
        logger.info(f"  Raw shape: {df.shape}")
        logger.info(f"  Companies: {df['Company'].nunique()}")
        logger.info(f"  Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")
        
        # Use Adj_Close
        if 'Adj_Close' in df.columns:
            df['Stock_Price'] = df['Adj_Close']
        elif 'Close' in df.columns:
            df['Stock_Price'] = df['Close']
        else:
            logger.error("  ✗ No price column found")
            return None
        
        # Add quarter info
        df = self.add_quarter_info(df)
        df['Original_Quarter_End'] = df['Quarter_End_Date'].copy()
        
        # Keep essential columns
        df['q_price'] = df['Stock_Price']
        if 'Volume' in df.columns:
            df['q_volume'] = df['Volume']
        if 'High' in df.columns:
            df['q_high'] = df['High']
        if 'Low' in df.columns:
            df['q_low'] = df['Low']
        if 'Open' in df.columns:
            df['q_open'] = df['Open']
        
        # Calculate quarter-over-quarter return
        logger.info(f"  Calculating quarter-over-quarter returns...")
        df = df.sort_values(['Company', 'Date'])
        df['prev_q_price'] = df.groupby('Company')['Stock_Price'].shift(1)
        
        df['q_return'] = np.where(
            df['prev_q_price'] > 0,
            (df['Stock_Price'] - df['prev_q_price']) / df['prev_q_price'] * 100,
            np.nan
        )
        
        # Calculate price range
        if 'q_high' in df.columns and 'q_low' in df.columns:
            df['q_price_range_pct'] = np.where(
                df['q_price'] > 0,
                (df['q_high'] - df['q_low']) / df['q_price'] * 100,
                np.nan
            )
        
        # Stats
        valid = (~df['q_return'].isna() & (df['q_return'] != 0)).sum()
        logger.info(f"  Valid returns: {valid} ({valid/len(df)*100:.1f}%)")
        logger.info(f"  Mean return: {df['q_return'].mean():.2f}%")
        
        df = df.drop(columns=['prev_q_price'], errors='ignore')
        
        return df
    
    def aggregate_fred_to_quarterly(self, df):
        """Aggregate daily FRED data to quarterly."""
        logger.info("\nAggregating FRED Data to Quarterly...")
        df = df.copy()
        
        # Parse dates
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)
        
        logger.info(f"  Raw FRED data: {df.shape}")
        logger.info(f"  Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")
        
        # Add quarter info
        df = self.add_quarter_info(df)
        
        # Identify quarterly vs daily columns
        # GDP, Unemployment, CPI are quarterly
        # Others are daily
        
        quarterly_cols = ['GDP', 'CPI', 'Unemployment_Rate']
        daily_cols = [col for col in df.columns if col not in quarterly_cols + ['Date', 'Year', 'Quarter_Num', 'Quarter', 'Quarter_End_Date']]
        
        logger.info(f"  Aggregating {len(daily_cols)} daily metrics to quarterly...")
        
        # Aggregate to quarterly
        agg_dict = {}
        
        # For quarterly metrics, take last value
        for col in quarterly_cols:
            if col in df.columns:
                agg_dict[col] = 'last'
        
        # For daily metrics, calculate mean, max, std
        for col in daily_cols:
            if col in df.columns:
                agg_dict[col] = ['mean', 'max', 'std']
        
        quarterly_fred = df.groupby(['Quarter', 'Year', 'Quarter_Num']).agg(agg_dict).reset_index()
        
        # Flatten multi-level columns
        new_cols = []
        for col in quarterly_fred.columns:
            if isinstance(col, tuple):
                if col[1] == '':
                    new_cols.append(col[0])
                else:
                    new_cols.append(f"{col[0]}_{col[1]}")
            else:
                new_cols.append(col)
        
        quarterly_fred.columns = new_cols
        
        # Get quarter end date
        quarterly_fred['Quarter_End_Date'] = quarterly_fred.apply(
            lambda row: self.get_quarter_end(pd.Timestamp(year=int(row['Year']), month=int(row['Quarter_Num'])*3, day=1)),
            axis=1
        )
        
        logger.info(f"  ✓ FRED aggregated: {quarterly_fred.shape}")
        logger.info(f"  Quarters: {len(quarterly_fred)}")
        
        return quarterly_fred
    
    def aggregate_market_to_quarterly(self, df):
        """Aggregate daily market data (VIX, SP500) to quarterly."""
        logger.info("\nAggregating Market Data to Quarterly...")
        df = df.copy()
        
        # Parse dates
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)
        
        logger.info(f"  Raw market data: {df.shape}")
        logger.info(f"  Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")
        
        # Add quarter info
        df = self.add_quarter_info(df)
        
        # Aggregate to quarterly
        quarterly_market = df.groupby(['Quarter', 'Year', 'Quarter_Num']).agg({
            'VIX': ['mean', 'max', 'std'],
            'SP500': ['first', 'last', 'mean', 'min', 'max']
        }).reset_index()
        
        # Flatten columns
        quarterly_market.columns = [
            '_'.join(col).strip() if col[1] else col[0] 
            for col in quarterly_market.columns.values
        ]
        
        # Rename for clarity
        quarterly_market = quarterly_market.rename(columns={
            'VIX_mean': 'vix_q_mean',
            'VIX_max': 'vix_q_max',
            'VIX_std': 'vix_q_std',
            'SP500_first': 'sp500_q_start',
            'SP500_last': 'sp500_q_end',
            'SP500_mean': 'sp500_q_mean',
            'SP500_min': 'sp500_q_min',
            'SP500_max': 'sp500_q_max'
        })
        
        # Calculate SP500 quarterly return
        quarterly_market['sp500_q_return'] = (
            (quarterly_market['sp500_q_end'] - quarterly_market['sp500_q_start']) 
            / quarterly_market['sp500_q_start'] * 100
        )
        
        # Get quarter end date
        quarterly_market['Quarter_End_Date'] = quarterly_market.apply(
            lambda row: self.get_quarter_end(pd.Timestamp(year=int(row['Year']), month=int(row['Quarter_Num'])*3, day=1)),
            axis=1
        )
        
        logger.info(f"  ✓ Market data aggregated: {quarterly_market.shape}")
        logger.info(f"  Quarters: {len(quarterly_market)}")
        logger.info(f"  SP500 return mean: {quarterly_market['sp500_q_return'].mean():.2f}%")
        
        return quarterly_market
    
    # ========== MERGING FUNCTION ==========
    
    def merge_all_quarterly_data(self, balance_df, income_df, prices_df, fred_df, market_df):
        """Merge all quarterly datasets."""
        logger.info("\n" + "="*80)
        logger.info("MERGING ALL QUARTERLY DATA")
        logger.info("="*80)
        
        # Step 1: Merge Balance + Income
        logger.info("\n[1/5] Merging Balance Sheet + Income Statement...")
        
        merge_cols = ['Original_Quarter_End', 'Company', 'Quarter', 'Year', 'Quarter_Num']
        optional_cols = ['Company_Name', 'Sector']
        for col in optional_cols:
            if col in balance_df.columns and col in income_df.columns:
                merge_cols.append(col)
        
        fundamentals = pd.merge(
            balance_df,
            income_df,
            on=merge_cols,
            how='outer',
            suffixes=('', '_inc')
        )
        
        # Remove duplicates
        dup_cols = [col for col in fundamentals.columns if col.endswith('_inc') and col.replace('_inc', '') in fundamentals.columns]
        if dup_cols:
            fundamentals = fundamentals.drop(columns=dup_cols)
        
        logger.info(f"  Fundamentals: {len(fundamentals)} quarters, {fundamentals['Company'].nunique()} companies")
        
        # Step 2: Merge Fundamentals + Prices
        logger.info("\n[2/5] Merging Fundamentals + Stock Prices...")
        
        company_data = pd.merge(
            fundamentals,
            prices_df,
            on=['Company', 'Original_Quarter_End'],
            how='inner',
            suffixes=('', '_px')
        )
        
        # Clean duplicate columns
        dup_cols = [col for col in company_data.columns if col.endswith('_px') and col.replace('_px', '') in company_data.columns]
        if dup_cols:
            company_data = company_data.drop(columns=dup_cols)
        
        logger.info(f"  Company data: {len(company_data)} quarters")
        
        # Step 3: Merge FRED data
        logger.info("\n[3/5] Merging FRED Macro Data...")
        
        company_data = pd.merge(
            company_data,
            fred_df,
            on=['Quarter', 'Year', 'Quarter_Num'],
            how='left',
            suffixes=('', '_fred')
        )
        
        logger.info(f"  After FRED merge: {len(company_data)} quarters")
        
        # Step 4: Merge Market data
        logger.info("\n[4/5] Merging Market Data (VIX, SP500)...")
        
        final_df = pd.merge(
            company_data,
            market_df,
            on=['Quarter', 'Year', 'Quarter_Num'],
            how='left',
            suffixes=('', '_mkt')
        )
        
        # Clean duplicate columns
        dup_cols = [col for col in final_df.columns if col.endswith('_mkt') and col.replace('_mkt', '') in final_df.columns]
        if dup_cols:
            final_df = final_df.drop(columns=dup_cols)
        
        logger.info(f"  Final dataset: {len(final_df)} quarters")
        logger.info(f"  Companies: {final_df['Company'].nunique()}")
        
        # Step 5: Summary
        logger.info("\n[5/5] Company data summary...")
        
        for company in sorted(final_df['Company'].unique())[:5]:
            company_df = final_df[final_df['Company'] == company]
            logger.info(f"    {company}: {len(company_df)} qtrs  ({company_df['Quarter'].min()} → {company_df['Quarter'].max()})")
        
        if final_df['Company'].nunique() > 5:
            logger.info(f"    ... and {final_df['Company'].nunique() - 5} more companies")
        
        return final_df
    
    # ========== VALIDATION ==========
    
    def validate_quarterly_data(self, df):
        """Validate the final quarterly dataset."""
        logger.info("\n" + "="*80)
        logger.info("VALIDATION REPORT")
        logger.info("="*80)
        
        logger.info(f"\n1. Dataset Overview:")
        logger.info(f"   ✓ ONE ROW PER QUARTER PER COMPANY")
        logger.info(f"   Shape: {df.shape}")
        logger.info(f"   Quarters: {len(df):,}")
        logger.info(f"   Companies: {df['Company'].nunique()}")
        logger.info(f"   Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")
        
        logger.info(f"\n2. Data Completeness:")
        key_cols = ['Revenue', 'Net_Income', 'Total_Assets', 'q_return', 'q_price', 'vix_q_mean', 'sp500_q_return']
        
        for col in key_cols:
            if col in df.columns:
                missing = df[col].isna().sum()
                pct = missing / len(df) * 100
                status = '✓' if pct < 5 else '⚠️'
                logger.info(f"   {status} {col:20s}: {missing:>6,} missing ({pct:>5.1f}%)")
        
        logger.info(f"\n3. Sample Data:")
        sample_cols = ['Date', 'Quarter', 'Company', 'Revenue', 'q_return', 'sp500_q_return', 'vix_q_mean']
        sample_cols = [col for col in sample_cols if col in df.columns]
        print("\n" + df[sample_cols].head(5).to_string(index=False))
        
        return df
    
    # ========== MAIN PIPELINE ==========
    
    def run_pipeline(self):
        """Execute complete pipeline."""
        logger.info("\n" + "="*80)
        logger.info("COMPLETE QUARTERLY DATA PIPELINE")
        logger.info("="*80)
        
        # Load all raw files
        logger.info("\nLoading raw files...")
        try:
            balance_raw = pd.read_csv(self.raw_dir / 'company_balance_raw.csv')
            income_raw = pd.read_csv(self.raw_dir / 'company_income_raw.csv')
            prices_raw = pd.read_csv(self.raw_dir / 'company_prices_raw.csv')
            fred_raw = pd.read_csv(self.raw_dir / 'fred_raw.csv')
            market_raw = pd.read_csv(self.raw_dir / 'market_raw.csv')
            
            logger.info(f"✓ Balance Sheet: {balance_raw.shape}")
            logger.info(f"✓ Income Statement: {income_raw.shape}")
            logger.info(f"✓ Stock Prices: {prices_raw.shape}")
            logger.info(f"✓ FRED Data: {fred_raw.shape}")
            logger.info(f"✓ Market Data: {market_raw.shape}")
        except FileNotFoundError as e:
            logger.error(f"✗ Error: {e}")
            return None
        
        # Step 1: Clean fundamentals
        logger.info("\n" + "="*80)
        logger.info("STEP 1: CLEANING FUNDAMENTALS")
        logger.info("="*80)
        
        balance_clean = self.clean_quarterly_fundamentals(balance_raw, 'balance')
        income_clean = self.clean_quarterly_fundamentals(income_raw, 'income')
        
        # Step 2: Clean prices
        logger.info("\n" + "="*80)
        logger.info("STEP 2: CLEANING STOCK PRICES")
        logger.info("="*80)
        
        prices_clean = self.clean_quarterly_prices(prices_raw)
        if prices_clean is None:
            return None
        
        # Step 3: Aggregate FRED
        logger.info("\n" + "="*80)
        logger.info("STEP 3: AGGREGATING FRED DATA")
        logger.info("="*80)
        
        fred_quarterly = self.aggregate_fred_to_quarterly(fred_raw)
        
        # Step 4: Aggregate Market
        logger.info("\n" + "="*80)
        logger.info("STEP 4: AGGREGATING MARKET DATA")
        logger.info("="*80)
        
        market_quarterly = self.aggregate_market_to_quarterly(market_raw)
        
        # Step 5: Merge all
        logger.info("\n" + "="*80)
        logger.info("STEP 5: MERGING ALL DATA")
        logger.info("="*80)
        
        merged_df = self.merge_all_quarterly_data(
            balance_clean, income_clean, prices_clean,
            fred_quarterly, market_quarterly
        )
        
        if merged_df is None:
            return None
        
        # Step 6: Validate
        logger.info("\n" + "="*80)
        logger.info("STEP 6: VALIDATION")
        logger.info("="*80)
        
        validated_df = self.validate_quarterly_data(merged_df)
        
        # Sort and save
        validated_df = validated_df.sort_values(['Company', 'Date']).reset_index(drop=True)
        
        output_path = self.output_dir / 'quarterly_data_complete.csv'
        validated_df.to_csv(output_path, index=False)
        logger.info(f"\n✓ Saved to: {output_path}")
        
        # Summary
        logger.info("\n" + "="*80)
        logger.info("PIPELINE COMPLETE ✅")
        logger.info("="*80)
        logger.info(f"\n📊 Final Dataset:")
        logger.info(f"   Shape: {validated_df.shape}")
        logger.info(f"   Quarters: {len(validated_df):,}")
        logger.info(f"   Companies: {validated_df['Company'].nunique()}")
        logger.info(f"   Features: {validated_df.shape[1]}")
        logger.info(f"\n   Includes:")
        logger.info(f"   ✓ Company fundamentals (Balance + Income)")
        logger.info(f"   ✓ Stock prices & returns")
        logger.info(f"   ✓ Macro indicators (GDP, CPI, etc.)")
        logger.info(f"   ✓ Market metrics (VIX, SP500)")
        
        return validated_df


# ========== USAGE ==========

if __name__ == "__main__":
    pipeline = QuarterlyDataPipeline(
        raw_dir='data/raw',
        output_dir='data/processed'
    )
    
    quarterly_data = pipeline.run_pipeline()
    
    if quarterly_data is not None:
        print("\n✅ SUCCESS! Complete quarterly dataset ready.")
        print(f"\nOutput: data/processed/quarterly_data_complete.csv")
        print(f"\nFormat: ONE ROW PER QUARTER PER COMPANY")
        print(f"\nIncludes:")
        print(f"  • Fundamentals (Revenue, Assets, Debt, etc.)")
        print(f"  • Stock metrics (q_return, q_price, q_volume)")
        print(f"  • Macro data (GDP, CPI, Unemployment)")
        print(f"  • Market data (VIX, SP500)")