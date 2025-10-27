"""
Financial Stress Test - Data Cleaning and Merging Pipeline
✨ Cleans individual datasets and merges them into ML-ready formats
✨ Handles missing values, alignment, and feature engineering
✨ Outputs multiple merged datasets for different use cases
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class FinancialDataMerger:
    """
    Cleans and merges FRED, Market, and Company data into ML-ready formats.
    """
    
    def __init__(self, data_dir='data/processed'):
        self.data_dir = data_dir
        self.fred_df = None
        self.market_df = None
        self.company_df = None
        
    def load_data(self):
        """Load all three datasets."""
        print("\n" + "="*70)
        print("📂 LOADING DATA")
        print("="*70)
        
        # Load FRED data
        fred_path = os.path.join(self.data_dir, 'fred_data_daily_oct25.csv')
        self.fred_df = pd.read_csv(fred_path, index_col=0, parse_dates=True)
        print(f"✅ FRED data loaded: {self.fred_df.shape}")
        print(f"   Date range: {self.fred_df.index.min()} to {self.fred_df.index.max()}")
        
        # Load Market data
        market_path = os.path.join(self.data_dir, 'market_data_daily_oct25.csv')
        self.market_df = pd.read_csv(market_path, index_col=0, parse_dates=True)
        print(f"✅ Market data loaded: {self.market_df.shape}")
        print(f"   Date range: {self.market_df.index.min()} to {self.market_df.index.max()}")
        
        # Load Company data
        company_path = os.path.join(self.data_dir, 'company_data_daily_oct25.csv')
        self.company_df = pd.read_csv(company_path, parse_dates=['Date'])
        print(f"✅ Company data loaded: {self.company_df.shape}")
        print(f"   Companies: {self.company_df['Company'].nunique()}")
        print(f"   Date range: {self.company_df['Date'].min()} to {self.company_df['Date'].max()}")
        
    def clean_fred_data(self):
        """Clean FRED macroeconomic data."""
        print("\n" + "="*70)
        print("🧹 CLEANING FRED DATA")
        print("="*70)
        
        df = self.fred_df.copy()
        
        # Check missing values before
        print(f"\n📊 Missing values before cleaning:")
        missing_before = df.isna().sum()
        for col in missing_before[missing_before > 0].index:
            print(f"   {col}: {missing_before[col]} ({missing_before[col]/len(df)*100:.1f}%)")
        
        # Forward fill for time series (use last known value)
        df_filled = df.fillna(method='ffill')
        
        # Backward fill for any remaining NaN at the start
        df_filled = df_filled.fillna(method='bfill')
        
        # For remaining NaN (if any), use column median
        for col in df_filled.columns:
            if df_filled[col].isna().any():
                median_val = df_filled[col].median()
                df_filled[col].fillna(median_val, inplace=True)
                print(f"   ⚠️  Filled {col} remaining NaN with median: {median_val:.2f}")
        
        # Check missing values after
        missing_after = df_filled.isna().sum().sum()
        print(f"\n✅ Missing values after cleaning: {missing_after}")
        
        self.fred_df = df_filled
        return df_filled
        
    def clean_market_data(self):
        """Clean market data (VIX, S&P 500)."""
        print("\n" + "="*70)
        print("🧹 CLEANING MARKET DATA")
        print("="*70)
        
        df = self.market_df.copy()
        
        # Check missing values
        print(f"\n📊 Missing values before cleaning:")
        missing_before = df.isna().sum()
        for col in missing_before[missing_before > 0].index:
            print(f"   {col}: {missing_before[col]} ({missing_before[col]/len(df)*100:.1f}%)")
        
        # Forward fill (market closed on weekends)
        df_filled = df.fillna(method='ffill')
        
        # Backward fill for start
        df_filled = df_filled.fillna(method='bfill')
        
        # Check for outliers in VIX (should be 0-100)
        if 'VIX' in df_filled.columns:
            vix_outliers = ((df_filled['VIX'] < 0) | (df_filled['VIX'] > 100)).sum()
            if vix_outliers > 0:
                print(f"   ⚠️  Found {vix_outliers} VIX outliers")
                df_filled.loc[df_filled['VIX'] < 0, 'VIX'] = np.nan
                df_filled.loc[df_filled['VIX'] > 100, 'VIX'] = np.nan
                df_filled['VIX'].fillna(method='ffill', inplace=True)
        
        missing_after = df_filled.isna().sum().sum()
        print(f"\n✅ Missing values after cleaning: {missing_after}")
        
        self.market_df = df_filled
        return df_filled
        
    def clean_company_data(self):
        """Clean company financial data."""
        print("\n" + "="*70)
        print("🧹 CLEANING COMPANY DATA")
        print("="*70)
        
        df = self.company_df.copy()
        
        # Check missing values by company
        print(f"\n📊 Missing values by company:")
        for company in df['Company'].unique():
            company_data = df[df['Company'] == company]
            missing = company_data.isna().sum().sum()
            if missing > 0:
                print(f"   {company}: {missing} missing values")
        
        # Forward fill within each company group
        df_filled = df.groupby('Company').apply(
            lambda group: group.fillna(method='ffill').fillna(method='bfill')
        ).reset_index(drop=True)
        
        # Remove rows where Stock_Price is still missing (no valid data)
        initial_rows = len(df_filled)
        df_filled = df_filled.dropna(subset=['Stock_Price'])
        removed_rows = initial_rows - len(df_filled)
        if removed_rows > 0:
            print(f"   ⚠️  Removed {removed_rows} rows with missing Stock_Price")
        
        # Check for negative stock prices (data errors)
        negative_prices = (df_filled['Stock_Price'] < 0).sum()
        if negative_prices > 0:
            print(f"   ⚠️  Found {negative_prices} negative stock prices - removing")
            df_filled = df_filled[df_filled['Stock_Price'] >= 0]
        
        missing_after = df_filled.isna().sum().sum()
        print(f"\n✅ Missing values after cleaning: {missing_after}")
        print(f"✅ Final shape: {df_filled.shape}")
        
        self.company_df = df_filled
        return df_filled
        
    def create_macro_market_merged(self, frequency='daily'):
        """
        Merge FRED and Market data at specified frequency.
        
        Args:
            frequency: 'daily', 'weekly', or 'monthly'
        """
        print("\n" + "="*70)
        print(f"🔗 MERGING FRED + MARKET DATA ({frequency.upper()})")
        print("="*70)
        
        # Resample if needed
        if frequency == 'monthly':
            fred = self.fred_df.resample('M').last()
            market = self.market_df.resample('M').last()
        elif frequency == 'weekly':
            fred = self.fred_df.resample('W').last()
            market = self.market_df.resample('W').last()
        else:  # daily
            fred = self.fred_df
            market = self.market_df
        
        # Merge on index (date)
        merged = fred.join(market, how='outer')
        
        # Forward fill FRED data (economic indicators don't update daily)
        merged[fred.columns] = merged[fred.columns].fillna(method='ffill')
        
        # Forward fill market data (weekends)
        merged[market.columns] = merged[market.columns].fillna(method='ffill')
        
        print(f"✅ Merged shape: {merged.shape}")
        print(f"   Date range: {merged.index.min()} to {merged.index.max()}")
        print(f"   Missing values: {merged.isna().sum().sum()}")
        
        return merged
        
    def create_company_with_context(self, frequency='monthly'):
        """
        Create company-level dataset with macroeconomic and market context.
        Each company gets matched with macro/market data for that date.
        
        Args:
            frequency: 'daily', 'weekly', or 'monthly'
        """
        print("\n" + "="*70)
        print(f"🔗 MERGING COMPANIES WITH MACRO CONTEXT ({frequency.upper()})")
        print("="*70)
        
        # Get macro-market merged data
        macro_market = self.create_macro_market_merged(frequency=frequency)
        
        # Resample company data
        company = self.company_df.copy()
        company['Date'] = pd.to_datetime(company['Date'])
        
        if frequency == 'monthly':
            # Group by company and month
            company['YearMonth'] = company['Date'].dt.to_period('M')
            company_resampled = company.groupby(['Company', 'YearMonth']).agg({
                'Stock_Price': 'last',
                'Stock_Return': 'sum',  # Sum returns over month
                'Stock_Volume': 'mean',
                'Company_Name': 'first',
                'Sector': 'first'
            }).reset_index()
            company_resampled['Date'] = company_resampled['YearMonth'].dt.to_timestamp()
            company_resampled = company_resampled.drop('YearMonth', axis=1)
        elif frequency == 'weekly':
            company['Week'] = company['Date'].dt.to_period('W')
            company_resampled = company.groupby(['Company', 'Week']).agg({
                'Stock_Price': 'last',
                'Stock_Return': 'sum',
                'Stock_Volume': 'mean',
                'Company_Name': 'first',
                'Sector': 'first'
            }).reset_index()
            company_resampled['Date'] = company_resampled['Week'].dt.to_timestamp()
            company_resampled = company_resampled.drop('Week', axis=1)
        else:  # daily
            company_resampled = company
        
        # Merge with macro-market data
        company_resampled['Date'] = pd.to_datetime(company_resampled['Date'])
        company_resampled = company_resampled.set_index('Date')
        
        merged = company_resampled.join(macro_market, how='left')
        
        # Forward fill macro data
        macro_cols = list(self.fred_df.columns) + list(self.market_df.columns)
        merged[macro_cols] = merged.groupby('Company')[macro_cols].fillna(method='ffill')
        
        merged = merged.reset_index()
        
        print(f"✅ Merged shape: {merged.shape}")
        print(f"   Companies: {merged['Company'].nunique()}")
        print(f"   Date range: {merged['Date'].min()} to {merged['Date'].max()}")
        print(f"   Missing values: {merged.isna().sum().sum()}")
        
        return merged
        
    def create_single_wide_table(self):
        """
        Create a SINGLE WIDE TABLE with ALL features:
        - FRED macroeconomic indicators
        - Market data (VIX, S&P 500)
        - Company data (25 companies)
        
        Daily frequency with forward-fill for missing values.
        """
        print("\n" + "="*70)
        print("🔗 CREATING SINGLE WIDE TABLE (ALL FEATURES)")
        print("="*70)
        
        # Step 1: Get macro-market merged (daily)
        macro_market = self.create_macro_market_merged('daily')
        print(f"\n✅ Step 1: Macro + Market data ready")
        print(f"   Shape: {macro_market.shape}")
        
        # Step 2: Prepare company data
        company = self.company_df.copy()
        company['Date'] = pd.to_datetime(company['Date'])
        company = company.set_index('Date')
        
        # Step 3: Pivot company data to wide format
        # Each company gets its own columns
        print(f"\n🔄 Step 2: Pivoting company data...")
        
        company_features = ['Stock_Price', 'Stock_Return', 'Stock_Volume']
        
        # Create wide format for each feature
        wide_dfs = []
        for feature in company_features:
            pivot = company.pivot_table(
                index='Date',
                columns='Company',
                values=feature,
                aggfunc='last'
            )
            # Rename columns to Company_Feature format (e.g., JPM_Stock_Price)
            pivot.columns = [f"{col}_{feature}" for col in pivot.columns]
            wide_dfs.append(pivot)
        
        # Combine all company features
        company_wide = pd.concat(wide_dfs, axis=1)
        print(f"✅ Company data pivoted")
        print(f"   Shape: {company_wide.shape}")
        print(f"   Features per company: {len(company_features)}")
        print(f"   Total company features: {company_wide.shape[1]}")
        
        # Step 4: Merge everything
        print(f"\n🔗 Step 3: Merging all data...")
        final_merged = macro_market.join(company_wide, how='outer')
        
        # Step 5: Forward fill ALL missing values
        print(f"\n📊 Missing values before forward fill: {final_merged.isna().sum().sum()}")
        final_merged = final_merged.fillna(method='ffill')
        
        # Backward fill for any remaining NaN at start
        final_merged = final_merged.fillna(method='bfill')
        
        print(f"✅ Missing values after forward fill: {final_merged.isna().sum().sum()}")
        
        # Step 6: Add metadata columns
        final_merged['Year'] = final_merged.index.year
        final_merged['Month'] = final_merged.index.month
        final_merged['Quarter'] = final_merged.index.quarter
        final_merged['DayOfWeek'] = final_merged.index.dayofweek
        final_merged['IsMonthEnd'] = final_merged.index.is_month_end
        
        print(f"\n✅ SINGLE WIDE TABLE CREATED")
        print(f"   Shape: {final_merged.shape}")
        print(f"   Date range: {final_merged.index.min()} to {final_merged.index.max()}")
        print(f"   Total features: {final_merged.shape[1]}")
        print(f"   - FRED indicators: {len(self.fred_df.columns)}")
        print(f"   - Market indicators: {len(self.market_df.columns)}")
        print(f"   - Company features: {company_wide.shape[1]} ({self.company_df['Company'].nunique()} companies × {len(company_features)} features)")
        print(f"   - Time features: 5")
        
        return final_merged
        
    def save_merged_datasets(self):
        """Save the single wide table."""
        print("\n" + "="*70)
        print("💾 SAVING SINGLE WIDE TABLE")
        print("="*70)
        
        output_dir = os.path.join(self.data_dir, 'merged')
        os.makedirs(output_dir, exist_ok=True)
        
        # Create the single wide table
        wide_table = self.create_single_wide_table()
        
        # Save as CSV
        path = os.path.join(output_dir, 'financial_data_complete_daily.csv')
        wide_table.to_csv(path)
        print(f"\n✅ Saved: {path}")
        print(f"   Size: {os.path.getsize(path) / (1024**2):.1f} MB")
        
        # Also save a feature list for reference
        feature_list = pd.DataFrame({
            'Feature': wide_table.columns,
            'Type': ['FRED' if col in self.fred_df.columns 
                    else 'Market' if col in self.market_df.columns
                    else 'Time' if col in ['Year', 'Month', 'Quarter', 'DayOfWeek', 'IsMonthEnd']
                    else 'Company' 
                    for col in wide_table.columns]
        })
        
        feature_path = os.path.join(output_dir, 'feature_list.csv')
        feature_list.to_csv(feature_path, index=False)
        print(f"✅ Feature list saved: {feature_path}")
        
        # Save basic statistics
        stats_path = os.path.join(output_dir, 'data_statistics.csv')
        wide_table.describe().to_csv(stats_path)
        print(f"✅ Statistics saved: {stats_path}")
        
        print(f"\n🎉 Single wide table saved to: {output_dir}")
        
        return wide_table
        

def main():
    """
    Main pipeline: Clean and merge all data sources into SINGLE WIDE TABLE.
    """
    print("\n" + "="*70)
    print("🔄 DATA CLEANING AND MERGING PIPELINE")
    print("   OUTPUT: Single Wide Table (Daily)")
    print("="*70)
    
    # Initialize merger
    merger = FinancialDataMerger(data_dir='data/processed')
    
    # Load data
    merger.load_data()
    
    # Clean each dataset
    merger.clean_fred_data()
    merger.clean_market_data()
    merger.clean_company_data()
    
    # Create and save single wide table
    wide_table = merger.save_merged_datasets()
    
    # Summary statistics
    print("\n" + "="*70)
    print("📊 FINAL DATASET SUMMARY")
    print("="*70)
    
    print(f"\n📋 Single Wide Table:")
    print(f"  Shape: {wide_table.shape[0]:,} rows × {wide_table.shape[1]} columns")
    print(f"  Date range: {wide_table.index.min()} to {wide_table.index.max()}")
    print(f"  Total days: {(wide_table.index.max() - wide_table.index.min()).days:,} days")
    print(f"  Missing values: {wide_table.isna().sum().sum()}")
    
    # Calculate memory usage safely
    memory_usage = wide_table.memory_usage(deep=True)
    if isinstance(memory_usage, int):
        total_memory = memory_usage / 1024**2
    else:
        total_memory = memory_usage.sum() / 1024**2
    print(f"  Memory usage: {total_memory:.1f} MB")
    
    # Feature breakdown
    print(f"\n📊 Feature Breakdown:")
    fred_cols = [col for col in wide_table.columns if col in merger.fred_df.columns]
    market_cols = [col for col in wide_table.columns if col in merger.market_df.columns]
    company_cols = [col for col in wide_table.columns if any(ticker in col for ticker in merger.company_df['Company'].unique())]
    time_cols = ['Year', 'Month', 'Quarter', 'DayOfWeek', 'IsMonthEnd']
    
    print(f"  FRED indicators: {len(fred_cols)}")
    print(f"  Market indicators: {len(market_cols)}")
    print(f"  Company features: {len(company_cols)}")
    print(f"  Time features: {len(time_cols)}")
    print(f"  TOTAL: {len(wide_table.columns)}")
    
    # Sample of features
    print(f"\n🔍 Sample Features:")
    if fred_cols:
        print(f"  FRED: {', '.join(fred_cols[:3])}...")
    if market_cols:
        print(f"  Market: {', '.join(market_cols[:2])}...")
    if company_cols:
        print(f"  Companies: {', '.join([col for col in company_cols[:3]])}...")
    
    print("\n" + "="*70)
    print("✅ PIPELINE COMPLETE!")
    print("="*70)
    print("\n📁 Output files in: data/processed/merged/")
    print("   📄 financial_data_complete_daily.csv - MAIN DATASET")
    print("   📄 feature_list.csv - List of all features")
    print("   📄 data_statistics.csv - Summary statistics")
    print("\n💡 This single wide table is ready for:")
    print("   • Feature engineering")
    print("   • Train/validation/test splits")
    print("   • ML model training")
    print("   • Scenario generation")
    

if __name__ == "__main__":
    main()