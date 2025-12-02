"""
src/preprocessing/create_targets.py

Creates 5 target variables for next quarter prediction:
1. target_revenue
2. target_eps
3. target_debt_equity
4. target_profit_margin
5. target_stock_return

Run this BEFORE temporal split!
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def create_targets(input_file: str, output_file: str):
    """
    Create all 5 target variables (next quarter predictions)
    """
    
    print("="*80)
    print("🎯 CREATING TARGET VARIABLES (NEXT QUARTER)")
    print("="*80)
    
    # ========================================
    # 1. LOAD DATA
    # ========================================
    
    print("\n1️⃣ Loading quarterly data...")
    df = pd.read_csv(input_file)
    
    print(f"   Loaded: {len(df):,} rows")
    print(f"   Companies: {df['Company'].nunique()}")
    print(f"   Columns: {len(df.columns)}")
    
    # ========================================
    # 2. SORT DATA (CRITICAL!)
    # ========================================
    
    print("\n2️⃣ Sorting data by Company and Date...")
    
    # Ensure Date is datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Sort by Company, then Date (CRITICAL for shift to work correctly!)
    df = df.sort_values(['Company', 'Date']).reset_index(drop=True)
    
    print(f"   ✅ Sorted by Company and Date")
    print(f"   Date range: {df['Date'].min()} to {df['Date'].max()}")
    
    # ========================================
    # 3. TARGET 1: REVENUE
    # ========================================
    
    print("\n3️⃣ Creating Target 1: Revenue (next quarter)...")
    
    if 'Revenue' in df.columns:
        # Shift Revenue by -1 within each company
        df['target_revenue'] = df.groupby('Company')['Revenue'].shift(-1)
        
        # Count valid targets
        valid_count = df['target_revenue'].notna().sum()
        null_count = df['target_revenue'].isna().sum()
        
        print(f"   ✅ target_revenue created")
        print(f"      Valid: {valid_count:,} ({valid_count/len(df)*100:.1f}%)")
        print(f"      Null: {null_count:,} (last quarter per company)")
        
        # Show example
        example = df[df['target_revenue'].notna()][['Company', 'Date', 'Revenue', 'target_revenue']].head(3)
        print(f"\n   Example:")
        print(example.to_string(index=False))
    else:
        print(f"   ❌ Revenue column not found - skipping")
    
    # ========================================
    # 4. TARGET 2: EPS
    # ========================================
    
    print("\n4️⃣ Creating Target 2: EPS (next quarter)...")
    
    # Check if EPS column exists and has data
    has_eps_col = 'EPS' in df.columns
    has_eps_data = has_eps_col and df['EPS'].notna().sum() > 0
    
    if has_eps_data:
        print(f"   Found existing EPS column with data")
        df['target_eps'] = df.groupby('Company')['EPS'].shift(-1)
        
        valid_count = df['target_eps'].notna().sum()
        print(f"   ✅ target_eps created from existing EPS")
        print(f"      Valid: {valid_count:,}")
        
    else:
        print(f"   EPS column missing or empty - calculating proxy...")
        
        # Calculate EPS proxy: Net_Income / Total_Equity
        if 'Net_Income' in df.columns and 'Total_Equity' in df.columns:
            # Calculate EPS proxy
            df['EPS_calculated'] = df['Net_Income'] / df['Total_Equity']
            
            # Handle infinities and extreme values
            df['EPS_calculated'] = df['EPS_calculated'].replace([np.inf, -np.inf], np.nan)
            
            # Winsorize at 1% and 99%
            lower = df['EPS_calculated'].quantile(0.01)
            upper = df['EPS_calculated'].quantile(0.99)
            df['EPS_calculated'] = df['EPS_calculated'].clip(lower=lower, upper=upper)
            
            print(f"      Calculated EPS = Net_Income / Total_Equity")
            print(f"      Winsorized at [{lower:.4f}, {upper:.4f}]")
            
            # Shift to create target
            df['target_eps'] = df.groupby('Company')['EPS_calculated'].shift(-1)
            
            valid_count = df['target_eps'].notna().sum()
            print(f"   ✅ target_eps created (calculated)")
            print(f"      Valid: {valid_count:,}")
            
        else:
            print(f"   ❌ Cannot calculate EPS - missing Net_Income or Total_Equity")
            df['target_eps'] = np.nan
    
    # ========================================
    # 5. TARGET 3: DEBT/EQUITY
    # ========================================
    
    print("\n5️⃣ Creating Target 3: Debt/Equity (next quarter)...")
    
    if 'Debt_to_Equity' in df.columns:
        # Shift Debt_to_Equity by -1 within each company
        df['target_debt_equity'] = df.groupby('Company')['Debt_to_Equity'].shift(-1)
        
        # Handle extreme values (winsorize)
        lower = df['target_debt_equity'].quantile(0.01)
        upper = df['target_debt_equity'].quantile(0.99)
        
        print(f"   Original range: [{df['target_debt_equity'].min():.2f}, {df['target_debt_equity'].max():.2f}]")
        print(f"   Winsorizing at: [{lower:.2f}, {upper:.2f}]")
        
        df['target_debt_equity'] = df['target_debt_equity'].clip(lower=lower, upper=upper)
        
        valid_count = df['target_debt_equity'].notna().sum()
        print(f"   ✅ target_debt_equity created")
        print(f"      Valid: {valid_count:,}")
        print(f"      Capped range: [{df['target_debt_equity'].min():.2f}, {df['target_debt_equity'].max():.2f}]")
        
    else:
        print(f"   ❌ Debt_to_Equity column not found - skipping")
    
    # ========================================
    # 6. TARGET 4: PROFIT MARGIN
    # ========================================
    
    print("\n6️⃣ Creating Target 4: Profit Margin (next quarter)...")
    
    # Try existing net_margin_q first
    if 'net_margin_q' in df.columns and df['net_margin_q'].notna().sum() > 0:
        print(f"   Using existing net_margin_q column")
        df['target_profit_margin'] = df.groupby('Company')['net_margin_q'].shift(-1)
        
        valid_count = df['target_profit_margin'].notna().sum()
        print(f"   ✅ target_profit_margin created from net_margin_q")
        print(f"      Valid: {valid_count:,}")
        
    # Otherwise calculate from Net_Income and Revenue
    elif 'Net_Income' in df.columns and 'Revenue' in df.columns:
        print(f"   Calculating Profit Margin = (Net_Income / Revenue) * 100")
        
        # Calculate profit margin
        df['profit_margin_calculated'] = (df['Net_Income'] / df['Revenue']) * 100
        
        # Handle infinities
        df['profit_margin_calculated'] = df['profit_margin_calculated'].replace([np.inf, -np.inf], np.nan)
        
        # Winsorize
        lower = df['profit_margin_calculated'].quantile(0.01)
        upper = df['profit_margin_calculated'].quantile(0.99)
        df['profit_margin_calculated'] = df['profit_margin_calculated'].clip(lower=lower, upper=upper)
        
        print(f"      Winsorized at [{lower:.2f}%, {upper:.2f}%]")
        
        # Shift to create target
        df['target_profit_margin'] = df.groupby('Company')['profit_margin_calculated'].shift(-1)
        
        valid_count = df['target_profit_margin'].notna().sum()
        print(f"   ✅ target_profit_margin created (calculated)")
        print(f"      Valid: {valid_count:,}")
        
    else:
        print(f"   ❌ Cannot calculate Profit Margin - missing columns")
        df['target_profit_margin'] = np.nan
    
    # ========================================
    # 7. TARGET 5: STOCK RETURN (3-MONTH)
    # ========================================
    
    print("\n7️⃣ Creating Target 5: Stock Return (next quarter)...")
    
    # Try using existing stock_q_return first
    if 'stock_q_return' in df.columns and df['stock_q_return'].notna().sum() > 0:
        print(f"   Using existing stock_q_return column")
        
        # Shift to get next quarter's return
        df['target_stock_return'] = df.groupby('Company')['stock_q_return'].shift(-1)
        
        # Winsorize to handle outliers
        lower = df['target_stock_return'].quantile(0.01)
        upper = df['target_stock_return'].quantile(0.99)
        
        print(f"   Original range: [{df['target_stock_return'].min():.2%}, {df['target_stock_return'].max():.2%}]")
        print(f"   Winsorizing at: [{lower:.2%}, {upper:.2%}]")
        
        df['target_stock_return'] = df['target_stock_return'].clip(lower=lower, upper=upper)
        
        valid_count = df['target_stock_return'].notna().sum()
        print(f"   ✅ target_stock_return created")
        print(f"      Valid: {valid_count:,}")
        print(f"      Capped range: [{df['target_stock_return'].min():.2%}, {df['target_stock_return'].max():.2%}]")
        
    # Otherwise calculate from Stock_Price
    elif 'Stock_Price' in df.columns:
        print(f"   Calculating quarterly return from Stock_Price")
        
        # Calculate quarter-over-quarter return
        df['return_calculated'] = df.groupby('Company')['Stock_Price'].pct_change(1)
        
        # Shift to get next quarter's return
        df['target_stock_return'] = df.groupby('Company')['return_calculated'].shift(-1)
        
        # Winsorize
        lower = df['target_stock_return'].quantile(0.01)
        upper = df['target_stock_return'].quantile(0.99)
        
        print(f"   Original range: [{df['target_stock_return'].min():.2%}, {df['target_stock_return'].max():.2%}]")
        print(f"   Winsorizing at: [{lower:.2%}, {upper:.2%}]")
        
        df['target_stock_return'] = df['target_stock_return'].clip(lower=lower, upper=upper)
        
        valid_count = df['target_stock_return'].notna().sum()
        print(f"   ✅ target_stock_return created (calculated)")
        print(f"      Valid: {valid_count:,}")
        print(f"      Capped range: [{df['target_stock_return'].min():.2%}, {df['target_stock_return'].max():.2%}]")
        
    else:
        print(f"   ❌ Cannot calculate stock return - missing stock price data")
        df['target_stock_return'] = np.nan
    
    # ========================================
    # 8. SUMMARY
    # ========================================
    
    print("\n" + "="*80)
    print("📊 TARGET CREATION SUMMARY")
    print("="*80)
    
    target_cols = [
        'target_revenue',
        'target_eps', 
        'target_debt_equity',
        'target_profit_margin',
        'target_stock_return'
    ]
    
    print(f"\n{'Target':<25} {'Valid':<10} {'Null':<10} {'% Valid':<10}")
    print("-"*55)
    
    for col in target_cols:
        if col in df.columns:
            valid = df[col].notna().sum()
            null = df[col].isna().sum()
            pct = valid / len(df) * 100
            print(f"{col:<25} {valid:<10,} {null:<10,} {pct:<10.1f}%")
        else:
            print(f"{col:<25} {'NOT CREATED':<10}")
    
    print("-"*55)
    print(f"{'Total rows':<25} {len(df):<10,}")
    
    # Show statistics for each target
    print("\n" + "="*80)
    print("📈 TARGET STATISTICS")
    print("="*80)
    
    for col in target_cols:
        if col in df.columns and df[col].notna().sum() > 0:
            print(f"\n{col}:")
            print(f"   Mean: {df[col].mean():.4f}")
            print(f"   Std: {df[col].std():.4f}")
            print(f"   Min: {df[col].min():.4f}")
            print(f"   25%: {df[col].quantile(0.25):.4f}")
            print(f"   Median: {df[col].median():.4f}")
            print(f"   75%: {df[col].quantile(0.75):.4f}")
            print(f"   Max: {df[col].max():.4f}")
    
    # ========================================
    # 9. SAVE
    # ========================================
    
    print("\n" + "="*80)
    print("💾 SAVING DATA WITH TARGETS")
    print("="*80)
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    
    file_size = Path(output_file).stat().st_size / (1024*1024)
    
    print(f"\n✅ Saved to: {output_file}")
    print(f"   Rows: {len(df):,}")
    print(f"   Columns: {len(df.columns)} (added 5 target columns)")
    print(f"   File size: {file_size:.1f} MB")
    
    # ========================================
    # 10. VALIDATION
    # ========================================
    
    print("\n" + "="*80)
    print("🔍 DATA QUALITY CHECKS")
    print("="*80)
    
    # Check for companies with no targets (shouldn't happen)
    companies_no_targets = []
    for company in df['Company'].unique():
        company_df = df[df['Company'] == company]
        if company_df['target_revenue'].notna().sum() == 0:
            companies_no_targets.append(company)
    
    if companies_no_targets:
        print(f"\n⚠️ Warning: {len(companies_no_targets)} companies have no targets:")
        print(f"   {companies_no_targets}")
    else:
        print(f"\n✅ All companies have at least some target values")
    
    # Check last quarter nulls (expected)
    last_quarters = df.groupby('Company').tail(1)
    last_quarter_nulls = last_quarters['target_revenue'].isna().sum()
    
    print(f"\n✅ Last quarter nulls (expected): {last_quarter_nulls}/{len(last_quarters)}")
    
    # Show example of predictions we can make
    print("\n" + "="*80)
    print("📋 EXAMPLE: WHAT WE CAN PREDICT")
    print("="*80)
    
    # Get a random company with data
    sample_company = df[df['target_revenue'].notna()]['Company'].iloc[0]
    sample = df[
        (df['Company'] == sample_company) & 
        (df['target_revenue'].notna())
    ][['Company', 'Date', 'Revenue', 'target_revenue', 'target_eps', 
       'target_debt_equity', 'target_profit_margin', 'target_stock_return']].head(5)
    
    print(f"\nSample predictions for {sample_company}:")
    print(sample.to_string(index=False))
    
    print("\n" + "="*80)
    print("✅ TARGET CREATION COMPLETE!")
    print("="*80)
    print(f"\n🎯 Next steps:")
    print(f"   1. Run temporal split: python src/preprocessing/create_temporal_splits.py")
    print(f"   2. Train models on each target")
    print(f"   3. Evaluate and compare")
    
    return df


if __name__ == "__main__":
    
    # File paths
    input_file = 'data/processed/features_engineered.csv'
    output_file = 'data/features/quarterly_data_with_targets.csv'
    
    # Create targets
    df_with_targets = create_targets(input_file, output_file)
    
    print("\n🚀 Ready for model training!")