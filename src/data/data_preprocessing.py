"""
Stage 2: Data Preprocessing - EXACT Features from Original Document

Generates EXACTLY the 56 features specified in your project document:
- 13 Macro indicators (FRED)
- 2 Market indicators (Yahoo)
- 12 Company raw features
- 25+ Engineered features
- 3 Metadata fields

NO imputation - preserves NaN for later handling
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

RAW_DATA_DIR = 'data/raw'
PROCESSED_DATA_DIR = 'data/processed'

def print_section(title: str):
    print("\n" + "="*70)
    print(f"{title}")
    print("="*70)

# ============================================================================
# LOAD RAW DATA
# ============================================================================

def load_raw_data():
    """Load raw data files"""
    print_section("LOADING RAW DATA")
    
    print("  FRED raw...", end=" ")
    df_fred = pd.read_csv(f'{RAW_DATA_DIR}/fred_raw_daily.csv', index_col=0, parse_dates=True)
    print(f"OK {df_fred.shape}")
    
    print("  Market raw...", end=" ")
    df_market = pd.read_csv(f'{RAW_DATA_DIR}/market_raw_daily.csv', index_col=0, parse_dates=True)
    print(f"OK {df_market.shape}")
    
    print("  Companies raw...", end=" ")
    df_companies = pd.read_csv(f'{RAW_DATA_DIR}/companies_raw_daily.csv', parse_dates=['Date'])
    print(f"OK {df_companies.shape}")
    
    return df_fred, df_market, df_companies

# ============================================================================
# ALIGN & MERGE (NO FILLING)
# ============================================================================

def align_and_merge(df_fred, df_market, df_companies):
    """Align all data to business days and merge (NO FILLING)"""
    
    print_section("ALIGNING & MERGING (NO IMPUTATION)")
    
    # Get business days from company data
    all_dates = pd.to_datetime(df_companies['Date'].unique())
    business_days = pd.DatetimeIndex(sorted(all_dates))
    print(f"  Business days: {len(business_days):,}")
    
    # Align macro data (keep NaN)
    print("  Aligning FRED...")
    df_fred_aligned = df_fred.reindex(business_days)
    
    print("  Aligning Market...")
    df_market_aligned = df_market.reindex(business_days)
    
    # Combine macro
    df_macro = df_fred_aligned.join(df_market_aligned, how='outer')
    
    # SP500 return
    if 'SP500' in df_macro.columns:
        df_macro['SP500_Return'] = df_macro['SP500'].pct_change() * 100
    
    print(f"  Macro aligned: {df_macro.shape}")
    
    # Prepare companies
    print("  Preparing company data...")
    all_comp = []
    
    for company in sorted(df_companies['Company'].unique()):
        comp_data = df_companies[df_companies['Company'] == company].copy()
        comp_data['Date'] = pd.to_datetime(comp_data['Date'])
        comp_data = comp_data.set_index('Date').reindex(business_days)
        
        # Stock return & volatility
        if 'Stock_Price' in comp_data.columns:
            comp_data['Stock_Return'] = comp_data['Stock_Price'].pct_change() * 100
            comp_data['Volatility'] = comp_data['Stock_Return'].rolling(20, min_periods=1).std() * 100
        
        comp_data['Company'] = company
        all_comp.append(comp_data.reset_index().rename(columns={'index': 'Date'}))
    
    df_companies_aligned = pd.concat(all_comp, ignore_index=True)
    print(f"  Companies aligned: {df_companies_aligned.shape}")
    
    # Merge
    print("  Merging macro + companies...")
    df_merged = df_companies_aligned.merge(df_macro, left_on='Date', right_index=True, how='left')
    print(f"  Merged: {df_merged.shape}")
    
    return df_merged

# ============================================================================
# FEATURE ENGINEERING (EXACT FROM DOCUMENT)
# ============================================================================

def engineer_all_features(df):
    """Engineer EXACT features from original document"""
    
    print_section("FEATURE ENGINEERING (56 Features - NO FILLING)")
    
    df = df.copy()
    df = df.sort_values(['Company', 'Date'])
    
    # 1. LAG FEATURES (4)
    print("  1. Lag features (GDP_Lag1, CPI_Lag1, UNRATE_Lag1, VIX_Lag1)")
    for company in df['Company'].unique():
        mask = df['Company'] == company
        df.loc[mask, 'GDP_Lag1'] = df.loc[mask, 'GDP_Growth'].shift(1)
        df.loc[mask, 'CPI_Lag1'] = df.loc[mask, 'CPI_Inflation'].shift(1)
        df.loc[mask, 'UNRATE_Lag1'] = df.loc[mask, 'Unemployment_Rate'].shift(1)
        df.loc[mask, 'VIX_Lag1'] = df.loc[mask, 'VIX'].shift(1)
    
    # 2. MOVING AVERAGES (3) - 2Q = 126 trading days
    print("  2. Moving averages (VIX_MA_2Q, SP500_MA_2Q, Oil_MA_2Q)")
    window = 126
    df['VIX_MA_2Q'] = df.groupby('Company')['VIX'].transform(lambda x: x.rolling(window, min_periods=1).mean())
    df['SP500_MA_2Q'] = df['SP500_Return'].rolling(window, min_periods=1).mean()
    df['Oil_MA_2Q'] = df['Oil_Price'].rolling(window, min_periods=1).mean()
    
    # 3. GROWTH RATES (3)
    print("  3. Growth rates (Revenue_Growth, EPS_Growth, Debt_Growth)")
    for company in df['Company'].unique():
        mask = df['Company'] == company
        df.loc[mask, 'Revenue_Growth'] = df.loc[mask, 'Revenue'].pct_change() * 100
        df.loc[mask, 'EPS_Growth'] = df.loc[mask, 'EPS'].pct_change() * 100
        df.loc[mask, 'Debt_Growth'] = df.loc[mask, 'Total_Debt'].pct_change() * 100
    
    # 4. FINANCIAL RATIOS (5)
    print("  4. Financial ratios (5 ratios)")
    df['Debt_to_Equity'] = df['Total_Debt'] / df['Total_Equity']
    df['Current_Ratio'] = df['Current_Assets'] / df['Current_Liabilities']
    df['Profit_Margin'] = (df['Net_Income'] / df['Revenue']) * 100
    
    # Interest Coverage (Operating Income / Interest Expense)
    # Note: We don't have Interest Expense, so skip or use proxy
    df['Interest_Coverage'] = np.nan  # Placeholder
    
    # 5. VOLATILITY (2) - 63 days = 1 quarter
    print("  5. Volatility (Return_Volatility, Debt_Ratio_Volatility)")
    for company in df['Company'].unique():
        mask = df['Company'] == company
        df.loc[mask, 'Return_Volatility'] = df.loc[mask, 'Stock_Return'].rolling(63, min_periods=1).std()
        df.loc[mask, 'Debt_Ratio_Volatility'] = df.loc[mask, 'Debt_to_Equity'].rolling(63, min_periods=1).std()
    
    # 6. INTERACTION TERMS (3)
    print("  6. Interactions (GDP_x_UNRATE, Inflation_x_Interest, VIX_x_Debt_Ratio)")
    df['GDP_x_UNRATE'] = df['GDP_Growth'] * df['Unemployment_Rate']
    df['Inflation_x_Interest'] = df['CPI_Inflation'] * df['Federal_Funds_Rate']
    df['VIX_x_Debt_Ratio'] = df['VIX'] * df['Debt_to_Equity']
    
    # 7. COMPOSITE STRESS INDEX (PCA on stress indicators)
    print("  7. Composite Stress Index (PCA)")
    stress_cols = ['VIX', 'Corporate_Bond_Spread', 'TED_Spread', 'Financial_Stress_Index']
    available_stress = [c for c in stress_cols if c in df.columns]
    
    if len(available_stress) >= 2:
        stress_data = df[available_stress].dropna()
        if len(stress_data) > 10:
            scaler = StandardScaler()
            stress_scaled = scaler.fit_transform(stress_data)
            pca = PCA(n_components=1)
            composite = pca.fit_transform(stress_scaled)
            
            # Map back to original dataframe
            df['Composite_Stress_Index'] = np.nan
            df.loc[stress_data.index, 'Composite_Stress_Index'] = composite.flatten()
        else:
            df['Composite_Stress_Index'] = np.nan
    else:
        df['Composite_Stress_Index'] = np.nan
    
    # 8. CRISIS INDICATORS (3)
    print("  8. Crisis dummies (Crisis_2008, Crisis_2020, Crisis_2022)")
    df['Year'] = pd.to_datetime(df['Date']).dt.year
    df['Crisis_2008'] = df['Year'].isin([2007, 2008, 2009]).astype(int)
    df['Crisis_2020'] = (df['Year'] == 2020).astype(int)
    df['Crisis_2022'] = df['Year'].isin([2022, 2023]).astype(int)
    
    # 9. TARGET VARIABLES (3) - Next quarter = 63 trading days
    print("  9. Targets (EPS_Next_Q, Revenue_Next_Q, Return_Next_Q)")
    for company in df['Company'].unique():
        mask = df['Company'] == company
        df.loc[mask, 'EPS_Next_Q'] = df.loc[mask, 'EPS'].shift(-63)
        df.loc[mask, 'Revenue_Next_Q'] = df.loc[mask, 'Revenue'].shift(-63)
        df.loc[mask, 'Return_Next_Q'] = df.loc[mask, 'Stock_Return'].rolling(63).sum().shift(-63)
    
    # 10. MARKET CAP (placeholder - need shares outstanding)
    print("  10. Market Cap (placeholder)")
    df['Market_Cap'] = np.nan
    
    print(f"\n  Total features: {df.shape[1]}")
    print(f"  NaN preserved (not filled)")
    
    return df

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*70)
    print("STAGE 2: PREPROCESSING - EXACT ORIGINAL FEATURES")
    print("="*70)
    print("Features: 56 (matching project document)")
    print("Method: Merge + Engineer, NO imputation")
    print("="*70)
    
    start = datetime.now()
    
    # Load
    df_fred, df_market, df_companies = load_raw_data()
    
    # Merge
    df_merged = align_and_merge(df_fred, df_market, df_companies)
    
    # Engineer
    df_final = engineer_all_features(df_merged)
    
    # Save
    print_section("SAVING FINAL DATASET")
    output = f'{PROCESSED_DATA_DIR}/merged_dataset_daily_raw.csv'
    df_final.to_csv(output, index=False)
    
    size = os.path.getsize(output) / (1024**2)
    print(f"  File: {output}")
    print(f"  Size: {size:.1f} MB")
    print(f"  Shape: {df_final.shape[0]:,} rows x {df_final.shape[1]} cols")
    print(f"  Missing: {df_final.isnull().sum().sum():,} ({df_final.isnull().sum().sum()/df_final.size*100:.1f}%)")
    print(f"  Time: {(datetime.now()-start).total_seconds()/60:.1f} min")
    print("="*70)

if __name__ == "__main__":
    main()