"""
src/preprocessing/drop_leakage_features.py

Drops leakage features from the engineered dataset.
Keeps only safe, non-leaking features for predicting:
1. Revenue (next quarter)
2. EPS (next quarter)
3. Debt-to-Equity (next quarter)
4. Profit Margin (next quarter)
5. Stock Return (next quarter)

Input:  data/processed/features_engineered.csv
Output: data/processed/features_engineered.csv (overwritten)
Backup: data/processed/features_engineered_backup.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


def drop_leakage_features(input_file: str, create_backup: bool = True):
    """
    Drop features that cause data leakage
    
    Args:
        input_file: Path to features_engineered.csv
        create_backup: If True, creates backup before overwriting
    """
    
    print("="*80)
    print("🧹 DROPPING LEAKAGE FEATURES")
    print("="*80)
    
    # ========================================
    # 1. LOAD DATA
    # ========================================
    
    print("\n1️⃣ Loading engineered features...")
    
    input_path = Path(input_file)
    
    if not input_path.exists():
        raise FileNotFoundError(f"File not found: {input_file}")
    
    df = pd.read_csv(input_path)
    
    print(f"   ✅ Loaded: {len(df):,} rows × {len(df.columns)} columns")
    print(f"   Date range: {df['Date'].min()} to {df['Date'].max()}")
    
    original_columns = df.columns.tolist()
    original_shape = df.shape
    
    # ========================================
    # 2. DEFINE FEATURES TO DROP
    # ========================================
    
    print("\n2️⃣ Defining features to drop...")
    
    FEATURES_TO_DROP = [
        # ========================================
        # A. CURRENT QUARTER VALUES OF TARGETS
        # ========================================
        'Revenue',                    # Predicting next quarter
        'EPS',                        # Target
        'Debt_to_Equity',             # Target
        'net_margin',                 # Target (profit margin)
        'gross_margin',               # Derived from target
        'operating_margin',           # Derived from target
        'roa',                        # Derived from targets
        'roe',                        # Derived from targets
        'q_return',                   # Target (stock return)
        'next_q_return',              # THIS IS THE TARGET!
        
        # ========================================
        # B. GROWTH FEATURES FROM CURRENT QUARTER
        # ========================================
        'Revenue_growth_1q',          # Uses current Revenue
        'Revenue_growth_4q',
        'Net_Income_growth_1q',       # Uses current Net_Income
        'Net_Income_growth_4q',
        'Gross_Profit_growth_1q',     # Uses current Gross_Profit
        'Gross_Profit_growth_4q',
        'Operating_Income_growth_1q',
        'Operating_Income_growth_4q',
        'eps_growth_1q',              # Uses current EPS
        'eps_growth_4q',
        
        # ========================================
        # C. CURRENT QUARTER FUNDAMENTALS
        # ========================================
        'Net_Income',                 # Use lagged instead
        'Gross_Profit',
        'Operating_Income',
        'EBITDA',
        'Total_Debt',
        'Total_Assets',
        'Total_Liabilities',
        'Total_Equity',
        'Current_Assets',
        'Current_Liabilities',
        'Long_Term_Debt',
        'Short_Term_Debt',
        'Cash',
        'Current_Ratio',
        
        # ========================================
        # D. CURRENT QUARTER STOCK PRICES
        # ========================================
        'Stock_Price',
        'q_price',
        'q_volume',
        'q_high',
        'q_low',
        'q_open',
        'q_price_range_pct',
        'Open',                       # Daily data
        'High',
        'Low',
        'Close',
        'Adj_Close',
        'Volume',
        
        # ========================================
        # E. RATIOS USING TARGET VARIABLES
        # ========================================
        'pe_ratio',                   # Uses EPS (target)
        'debt_to_assets',             # Current quarter leverage
        'debt_to_ebitda',             # Uses current EBITDA
        'cash_ratio',                 # Current quarter liquidity
        
        # ========================================
        # F. ENGINEERED FEATURES WITH LEAKAGE
        # ========================================
        'revenue_acceleration',       # Uses current Revenue growth
        'net_margin_trend',           # Uses current net_margin
        'return_momentum',            # Uses current q_return
        'revenue_declining',          # Uses current Revenue
        'high_leverage',              # Uses current Debt_to_Equity
        'liquidity_risk',             # Uses current ratios
        'composite_stress_score',     # Composite of targets
        'leverage_x_vix',             # Uses current leverage
        'margin_x_market',            # Uses current margin
        'revenue_decline_x_vix',      # Uses current Revenue
        'excess_return',              # Uses current q_return
        'return_vs_sector',           # Uses current q_return
        'revenue_growth_vs_sector',   # Uses current Revenue growth
        'debt_x_rates',               # Uses current debt (if not lagged)
        
        # ========================================
        # G. Z-SCORES OF TARGETS
        # ========================================
        'net_margin_zscore',          # Standardized target
        'roa_zscore',
        'roe_zscore',
        'debt_to_assets_zscore',
        
        # ========================================
        # H. LOG TRANSFORMS OF CURRENT VALUES
        # ========================================
        'log_Revenue',                # Current quarter
        'log_Total_Assets',
        'log_Total_Debt',
        'log_Cash',
        
        # ========================================
        # I. CLASSIFICATION LABELS
        # ========================================
        'crisis_flag',
        
        # ========================================
        # J. REDUNDANT DATE COLUMNS
        # ========================================
        'Quarter_End_Date',
        'Original_Quarter_End',
        'Quarter_End_Date_fred',
        
        # ========================================
        # K. REDUNDANT IDENTIFIERS
        # ========================================
        'Company_Name',
    ]
    
    print(f"   Features to drop: {len(FEATURES_TO_DROP)}")
    
    # ========================================
    # 3. CHECK WHICH FEATURES EXIST
    # ========================================
    
    print("\n3️⃣ Checking which features exist in dataset...")
    
    existing_to_drop = [col for col in FEATURES_TO_DROP if col in df.columns]
    missing_to_drop = [col for col in FEATURES_TO_DROP if col not in df.columns]
    
    print(f"   Existing (will drop): {len(existing_to_drop)}")
    print(f"   Missing (already absent): {len(missing_to_drop)}")
    
    if existing_to_drop:
        print(f"\n   📋 Dropping {len(existing_to_drop)} features:")
        for i, col in enumerate(existing_to_drop[:20], 1):
            print(f"      {i}. {col}")
        if len(existing_to_drop) > 20:
            print(f"      ... and {len(existing_to_drop) - 20} more")
    
    if missing_to_drop:
        print(f"\n   ℹ️  {len(missing_to_drop)} features already absent:")
        for col in missing_to_drop[:10]:
            print(f"      - {col}")
        if len(missing_to_drop) > 10:
            print(f"      ... and {len(missing_to_drop) - 10} more")
    
    # ========================================
    # 4. CREATE BACKUP
    # ========================================
    
    if create_backup:
        print(f"\n4️⃣ Creating backup...")
        
        backup_file = input_path.parent / f"{input_path.stem}_backup{input_path.suffix}"
        df.to_csv(backup_file, index=False)
        
        backup_size = backup_file.stat().st_size / (1024*1024)
        
        print(f"   ✅ Backup created: {backup_file}")
        print(f"      Size: {backup_size:.1f} MB")
    else:
        print(f"\n4️⃣ Skipping backup (create_backup=False)")
    
    # ========================================
    # 5. DROP FEATURES
    # ========================================
    
    print(f"\n5️⃣ Dropping leakage features...")
    
    df_clean = df.drop(columns=existing_to_drop)
    
    print(f"   ✅ Dropped {len(existing_to_drop)} features")
    print(f"   Before: {original_shape[0]:,} rows × {original_shape[1]} columns")
    print(f"   After:  {df_clean.shape[0]:,} rows × {df_clean.shape[1]} columns")
    print(f"   Columns removed: {original_shape[1] - df_clean.shape[1]}")
    
    # ========================================
    # 6. VALIDATE REMAINING FEATURES
    # ========================================
    
    print(f"\n6️⃣ Validating remaining features...")
    
    remaining_cols = df_clean.columns.tolist()
    
    # Categorize remaining features
    identifier_cols = [c for c in remaining_cols if c in ['Date', 'Year', 'Quarter', 'Quarter_Num', 'Company', 'Sector']]
    macro_cols = [c for c in remaining_cols if any(x in c for x in ['GDP', 'CPI', 'Unemployment', 'Federal', 'Yield', 'Consumer', 'Oil', 'Trade', 'Corporate', 'TED', 'Treasury', 'Financial_Stress', 'High_Yield', 'vix', 'sp500'])]
    lag_cols = [c for c in remaining_cols if '_lag_' in c]
    rolling_cols = [c for c in remaining_cols if 'rolling' in c]
    growth_cols = [c for c in remaining_cols if 'growth' in c]
    other_cols = [c for c in remaining_cols if c not in identifier_cols + macro_cols + lag_cols + rolling_cols + growth_cols]
    
    print(f"\n   📊 Remaining feature breakdown:")
    print(f"      Identifiers:     {len(identifier_cols)}")
    print(f"      Macro features:  {len(macro_cols)}")
    print(f"      Lagged features: {len(lag_cols)}")
    print(f"      Rolling features: {len(rolling_cols)}")
    print(f"      Growth features: {len(growth_cols)}")
    print(f"      Other features:  {len(other_cols)}")
    print(f"      TOTAL:           {len(remaining_cols)}")
    
    # Check for suspicious remaining features
    suspicious = []
    
    # Check for any current-quarter target remnants
    target_keywords = ['Revenue', 'EPS', 'net_margin', 'Debt_to_Equity', 'q_return']
    for col in remaining_cols:
        # Allow lagged versions
        if any(keyword in col for keyword in target_keywords):
            if not any(x in col for x in ['_lag_', 'rolling', 'growth', 'sector_avg']):
                suspicious.append(col)
    
    if suspicious:
        print(f"\n   ⚠️  WARNING: Potentially leaky features remain:")
        for col in suspicious:
            print(f"      - {col}")
    else:
        print(f"\n   ✅ No suspicious features detected")
    
    # ========================================
    # 7. SAVE CLEANED DATA
    # ========================================
    
    print(f"\n7️⃣ Saving cleaned dataset...")
    
    df_clean.to_csv(input_path, index=False)
    
    output_size = input_path.stat().st_size / (1024*1024)
    
    print(f"   ✅ Saved to: {input_path}")
    print(f"      Size: {output_size:.1f} MB")
    print(f"      Size reduction: {(original_shape[1] - df_clean.shape[1]) / original_shape[1] * 100:.1f}%")
    
    # ========================================
    # 8. SAVE DROPPED FEATURES LOG
    # ========================================
    
    print(f"\n8️⃣ Saving dropped features log...")
    
    log_file = input_path.parent / 'dropped_features_log.txt'
    
    with open(log_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("DROPPED FEATURES LOG\n")
        f.write("="*80 + "\n")
        f.write(f"Date: {datetime.now().isoformat()}\n")
        f.write(f"Original columns: {original_shape[1]}\n")
        f.write(f"Remaining columns: {df_clean.shape[1]}\n")
        f.write(f"Dropped: {len(existing_to_drop)}\n")
        f.write("\n" + "="*80 + "\n")
        f.write("FEATURES DROPPED:\n")
        f.write("="*80 + "\n\n")
        
        for i, col in enumerate(existing_to_drop, 1):
            f.write(f"{i}. {col}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("FEATURES NOT FOUND (already absent):\n")
        f.write("="*80 + "\n\n")
        
        for col in missing_to_drop:
            f.write(f"- {col}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("REMAINING FEATURES:\n")
        f.write("="*80 + "\n\n")
        
        for i, col in enumerate(remaining_cols, 1):
            f.write(f"{i}. {col}\n")
    
    print(f"   ✅ Log saved: {log_file}")
    
    # ========================================
    # 9. SUMMARY
    # ========================================
    
    print(f"\n{'='*80}")
    print(f"📊 FEATURE CLEANING SUMMARY")
    print(f"{'='*80}")
    
    print(f"\nOriginal Dataset:")
    print(f"   Rows:    {original_shape[0]:,}")
    print(f"   Columns: {original_shape[1]}")
    
    print(f"\nCleaned Dataset:")
    print(f"   Rows:    {df_clean.shape[0]:,}")
    print(f"   Columns: {df_clean.shape[1]}")
    
    print(f"\nChanges:")
    print(f"   Columns dropped: {len(existing_to_drop)}")
    print(f"   Columns kept:    {df_clean.shape[1]}")
    print(f"   Reduction:       {(original_shape[1] - df_clean.shape[1]) / original_shape[1] * 100:.1f}%")
    
    print(f"\nFiles:")
    print(f"   Original (backup): {input_path.parent / f'{input_path.stem}_backup{input_path.suffix}'}")
    print(f"   Cleaned:           {input_path}")
    print(f"   Log:               {log_file}")
    
    print(f"\n{'='*80}")
    print(f"✅ FEATURE CLEANING COMPLETE!")
    print(f"{'='*80}")
    
    print(f"\n🎯 Next steps:")
    print(f"   1. Review dropped_features_log.txt")
    print(f"   2. Verify remaining features look correct")
    print(f"   3. Proceed with target creation:")
    print(f"      python src/preprocessing/create_targets.py")
    
    return df_clean, existing_to_drop


if __name__ == "__main__":
    """
    Main execution: Drop leakage features
    """
    
    # Input file
    input_file = 'data/processed/features_engineered.csv'
    
    # Create backup before overwriting
    create_backup = True
    
    try:
        print(f"\n{'='*80}")
        print(f"🚀 STARTING FEATURE CLEANING")
        print(f"{'='*80}")
        print(f"\nInput file: {input_file}")
        
        # Drop leakage features
        df_clean, dropped = drop_leakage_features(
            input_file=input_file,
            create_backup=create_backup
        )
        
        print(f"\n{'='*80}")
        print(f"✅ SUCCESS! Feature cleaning completed!")
        print(f"{'='*80}")
        
        print(f"\n📋 Summary:")
        print(f"   Dropped {len(dropped)} leakage features")
        print(f"   Remaining: {len(df_clean.columns)} clean features")
        print(f"   Dataset saved to: {input_file}")
        
    except FileNotFoundError as e:
        print(f"\n❌ FILE NOT FOUND ERROR:")
        print(f"   {e}")
        print(f"\n💡 Make sure the file exists:")
        print(f"   data/processed/features_engineered.csv")
        
    except Exception as e:
        print(f"\n❌ ERROR:")
        print(f"   {e}")
        import traceback
        traceback.print_exc()