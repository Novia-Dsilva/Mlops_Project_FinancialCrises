# """
# src/preprocessing/drop_leakage_features.py

# Drops leakage features from the engineered dataset.
# Keeps only safe, non-leaking features for predicting:
# 1. Revenue (next quarter)
# 2. EPS (next quarter)
# 3. Debt-to-Equity (next quarter)
# 4. Profit Margin (next quarter)
# 5. Stock Return (next quarter)

# Input:  data/processed/features_engineered.csv
# Output: data/processed/features_engineered.csv (overwritten)
# Backup: data/processed/features_engineered_backup.csv
# """

# import pandas as pd
# import numpy as np
# from pathlib import Path
# from datetime import datetime
# import warnings
# warnings.filterwarnings('ignore')


# def drop_leakage_features(input_file: str, create_backup: bool = True):
#     """
#     Drop features that cause data leakage
    
#     Args:
#         input_file: Path to features_engineered.csv
#         create_backup: If True, creates backup before overwriting
#     """
    
#     print("="*80)
#     print("🧹 DROPPING LEAKAGE FEATURES")
#     print("="*80)
    
#     # ========================================
#     # 1. LOAD DATA
#     # ========================================
    
#     print("\n1️⃣ Loading engineered features...")
    
#     input_path = Path(input_file)
    
#     if not input_path.exists():
#         raise FileNotFoundError(f"File not found: {input_file}")
    
#     df = pd.read_csv(input_path)
    
#     print(f"   ✅ Loaded: {len(df):,} rows × {len(df.columns)} columns")
#     print(f"   Date range: {df['Date'].min()} to {df['Date'].max()}")
    
#     original_columns = df.columns.tolist()
#     original_shape = df.shape
    
#     # ========================================
#     # 2. DEFINE FEATURES TO DROP
#     # ========================================
    
#     print("\n2️⃣ Defining features to drop...")
    
#     FEATURES_TO_DROP = [
#         # ========================================
#         # A. CURRENT QUARTER VALUES OF TARGETS
#         # ========================================
#         'Revenue',                    # Predicting next quarter
#         'EPS',                        # Target
#         'Debt_to_Equity',             # Target
#         'net_margin',                 # Target (profit margin)
#         'gross_margin',               # Derived from target
#         'operating_margin',           # Derived from target
#         'roa',                        # Derived from targets
#         'roe',                        # Derived from targets
#         'q_return',                   # Target (stock return)
#         'next_q_return',              # THIS IS THE TARGET!
        
#         # ========================================
#         # B. GROWTH FEATURES FROM CURRENT QUARTER
#         # ========================================
#         'Revenue_growth_1q',          # Uses current Revenue
#         'Revenue_growth_4q',
#         'Net_Income_growth_1q',       # Uses current Net_Income
#         'Net_Income_growth_4q',
#         'Gross_Profit_growth_1q',     # Uses current Gross_Profit
#         'Gross_Profit_growth_4q',
#         'Operating_Income_growth_1q',
#         'Operating_Income_growth_4q',
#         'eps_growth_1q',              # Uses current EPS
#         'eps_growth_4q',
        
#         # ========================================
#         # C. CURRENT QUARTER FUNDAMENTALS
#         # ========================================
#         'Net_Income',                 # Use lagged instead
#         'Gross_Profit',
#         'Operating_Income',
#         'EBITDA',
#         'Total_Debt',
#         'Total_Assets',
#         'Total_Liabilities',
#         'Total_Equity',
#         'Current_Assets',
#         'Current_Liabilities',
#         'Long_Term_Debt',
#         'Short_Term_Debt',
#         'Cash',
#         'Current_Ratio',
        
#         # ========================================
#         # D. CURRENT QUARTER STOCK PRICES
#         # ========================================
#         'Stock_Price',
#         'q_price',
#         'q_volume',
#         'q_high',
#         'q_low',
#         'q_open',
#         'q_price_range_pct',
#         'Open',                       # Daily data
#         'High',
#         'Low',
#         'Close',
#         'Adj_Close',
#         'Volume',
        
#         # ========================================
#         # E. RATIOS USING TARGET VARIABLES
#         # ========================================
#         'pe_ratio',                   # Uses EPS (target)
#         'debt_to_assets',             # Current quarter leverage
#         'debt_to_ebitda',             # Uses current EBITDA
#         'cash_ratio',                 # Current quarter liquidity
        
#         # ========================================
#         # F. ENGINEERED FEATURES WITH LEAKAGE
#         # ========================================
#         'revenue_acceleration',       # Uses current Revenue growth
#         'net_margin_trend',           # Uses current net_margin
#         'return_momentum',            # Uses current q_return
#         'revenue_declining',          # Uses current Revenue
#         'high_leverage',              # Uses current Debt_to_Equity
#         'liquidity_risk',             # Uses current ratios
#         'composite_stress_score',     # Composite of targets
#         'leverage_x_vix',             # Uses current leverage
#         'margin_x_market',            # Uses current margin
#         'revenue_decline_x_vix',      # Uses current Revenue
#         'excess_return',              # Uses current q_return
#         'return_vs_sector',           # Uses current q_return
#         'revenue_growth_vs_sector',   # Uses current Revenue growth
#         'debt_x_rates',               # Uses current debt (if not lagged)
        
#         # ========================================
#         # G. Z-SCORES OF TARGETS
#         # ========================================
#         'net_margin_zscore',          # Standardized target
#         'roa_zscore',
#         'roe_zscore',
#         'debt_to_assets_zscore',
        
#         # ========================================
#         # H. LOG TRANSFORMS OF CURRENT VALUES
#         # ========================================
#         'log_Revenue',                # Current quarter
#         'log_Total_Assets',
#         'log_Total_Debt',
#         'log_Cash',
        
#         # ========================================
#         # I. CLASSIFICATION LABELS
#         # ========================================
#         'crisis_flag',
        
#         # ========================================
#         # J. REDUNDANT DATE COLUMNS
#         # ========================================
#         'Quarter_End_Date',
#         'Original_Quarter_End',
#         'Quarter_End_Date_fred',
        
#         # ========================================
#         # K. REDUNDANT IDENTIFIERS
#         # ========================================
#         'Company_Name',
#     ]
    
#     print(f"   Features to drop: {len(FEATURES_TO_DROP)}")
    
#     # ========================================
#     # 3. CHECK WHICH FEATURES EXIST
#     # ========================================
    
#     print("\n3️⃣ Checking which features exist in dataset...")
    
#     existing_to_drop = [col for col in FEATURES_TO_DROP if col in df.columns]
#     missing_to_drop = [col for col in FEATURES_TO_DROP if col not in df.columns]
    
#     print(f"   Existing (will drop): {len(existing_to_drop)}")
#     print(f"   Missing (already absent): {len(missing_to_drop)}")
    
#     if existing_to_drop:
#         print(f"\n   📋 Dropping {len(existing_to_drop)} features:")
#         for i, col in enumerate(existing_to_drop[:20], 1):
#             print(f"      {i}. {col}")
#         if len(existing_to_drop) > 20:
#             print(f"      ... and {len(existing_to_drop) - 20} more")
    
#     if missing_to_drop:
#         print(f"\n   ℹ️  {len(missing_to_drop)} features already absent:")
#         for col in missing_to_drop[:10]:
#             print(f"      - {col}")
#         if len(missing_to_drop) > 10:
#             print(f"      ... and {len(missing_to_drop) - 10} more")
    
#     # ========================================
#     # 4. CREATE BACKUP
#     # ========================================
    
#     if create_backup:
#         print(f"\n4️⃣ Creating backup...")
        
#         backup_file = input_path.parent / f"{input_path.stem}_backup{input_path.suffix}"
#         df.to_csv(backup_file, index=False)
        
#         backup_size = backup_file.stat().st_size / (1024*1024)
        
#         print(f"   ✅ Backup created: {backup_file}")
#         print(f"      Size: {backup_size:.1f} MB")
#     else:
#         print(f"\n4️⃣ Skipping backup (create_backup=False)")
    
#     # ========================================
#     # 5. DROP FEATURES
#     # ========================================
    
#     print(f"\n5️⃣ Dropping leakage features...")
    
#     df_clean = df.drop(columns=existing_to_drop)
    
#     print(f"   ✅ Dropped {len(existing_to_drop)} features")
#     print(f"   Before: {original_shape[0]:,} rows × {original_shape[1]} columns")
#     print(f"   After:  {df_clean.shape[0]:,} rows × {df_clean.shape[1]} columns")
#     print(f"   Columns removed: {original_shape[1] - df_clean.shape[1]}")
    
#     # ========================================
#     # 6. VALIDATE REMAINING FEATURES
#     # ========================================
    
#     print(f"\n6️⃣ Validating remaining features...")
    
#     remaining_cols = df_clean.columns.tolist()
    
#     # Categorize remaining features
#     identifier_cols = [c for c in remaining_cols if c in ['Date', 'Year', 'Quarter', 'Quarter_Num', 'Company', 'Sector']]
#     macro_cols = [c for c in remaining_cols if any(x in c for x in ['GDP', 'CPI', 'Unemployment', 'Federal', 'Yield', 'Consumer', 'Oil', 'Trade', 'Corporate', 'TED', 'Treasury', 'Financial_Stress', 'High_Yield', 'vix', 'sp500'])]
#     lag_cols = [c for c in remaining_cols if '_lag_' in c]
#     rolling_cols = [c for c in remaining_cols if 'rolling' in c]
#     growth_cols = [c for c in remaining_cols if 'growth' in c]
#     other_cols = [c for c in remaining_cols if c not in identifier_cols + macro_cols + lag_cols + rolling_cols + growth_cols]
    
#     print(f"\n   📊 Remaining feature breakdown:")
#     print(f"      Identifiers:     {len(identifier_cols)}")
#     print(f"      Macro features:  {len(macro_cols)}")
#     print(f"      Lagged features: {len(lag_cols)}")
#     print(f"      Rolling features: {len(rolling_cols)}")
#     print(f"      Growth features: {len(growth_cols)}")
#     print(f"      Other features:  {len(other_cols)}")
#     print(f"      TOTAL:           {len(remaining_cols)}")
    
#     # Check for suspicious remaining features
#     suspicious = []
    
#     # Check for any current-quarter target remnants
#     target_keywords = ['Revenue', 'EPS', 'net_margin', 'Debt_to_Equity', 'q_return']
#     for col in remaining_cols:
#         # Allow lagged versions
#         if any(keyword in col for keyword in target_keywords):
#             if not any(x in col for x in ['_lag_', 'rolling', 'growth', 'sector_avg']):
#                 suspicious.append(col)
    
#     if suspicious:
#         print(f"\n   ⚠️  WARNING: Potentially leaky features remain:")
#         for col in suspicious:
#             print(f"      - {col}")
#     else:
#         print(f"\n   ✅ No suspicious features detected")
    
#     # ========================================
#     # 7. SAVE CLEANED DATA
#     # ========================================
    
#     print(f"\n7️⃣ Saving cleaned dataset...")
    
#     df_clean.to_csv(input_path, index=False)
    
#     output_size = input_path.stat().st_size / (1024*1024)
    
#     print(f"   ✅ Saved to: {input_path}")
#     print(f"      Size: {output_size:.1f} MB")
#     print(f"      Size reduction: {(original_shape[1] - df_clean.shape[1]) / original_shape[1] * 100:.1f}%")
    
#     # ========================================
#     # 8. SAVE DROPPED FEATURES LOG
#     # ========================================
    
#     print(f"\n8️⃣ Saving dropped features log...")
    
#     log_file = input_path.parent / 'dropped_features_log.txt'
    
#     with open(log_file, 'w') as f:
#         f.write("="*80 + "\n")
#         f.write("DROPPED FEATURES LOG\n")
#         f.write("="*80 + "\n")
#         f.write(f"Date: {datetime.now().isoformat()}\n")
#         f.write(f"Original columns: {original_shape[1]}\n")
#         f.write(f"Remaining columns: {df_clean.shape[1]}\n")
#         f.write(f"Dropped: {len(existing_to_drop)}\n")
#         f.write("\n" + "="*80 + "\n")
#         f.write("FEATURES DROPPED:\n")
#         f.write("="*80 + "\n\n")
        
#         for i, col in enumerate(existing_to_drop, 1):
#             f.write(f"{i}. {col}\n")
        
#         f.write("\n" + "="*80 + "\n")
#         f.write("FEATURES NOT FOUND (already absent):\n")
#         f.write("="*80 + "\n\n")
        
#         for col in missing_to_drop:
#             f.write(f"- {col}\n")
        
#         f.write("\n" + "="*80 + "\n")
#         f.write("REMAINING FEATURES:\n")
#         f.write("="*80 + "\n\n")
        
#         for i, col in enumerate(remaining_cols, 1):
#             f.write(f"{i}. {col}\n")
    
#     print(f"   ✅ Log saved: {log_file}")
    
#     # ========================================
#     # 9. SUMMARY
#     # ========================================
    
#     print(f"\n{'='*80}")
#     print(f"📊 FEATURE CLEANING SUMMARY")
#     print(f"{'='*80}")
    
#     print(f"\nOriginal Dataset:")
#     print(f"   Rows:    {original_shape[0]:,}")
#     print(f"   Columns: {original_shape[1]}")
    
#     print(f"\nCleaned Dataset:")
#     print(f"   Rows:    {df_clean.shape[0]:,}")
#     print(f"   Columns: {df_clean.shape[1]}")
    
#     print(f"\nChanges:")
#     print(f"   Columns dropped: {len(existing_to_drop)}")
#     print(f"   Columns kept:    {df_clean.shape[1]}")
#     print(f"   Reduction:       {(original_shape[1] - df_clean.shape[1]) / original_shape[1] * 100:.1f}%")
    
#     print(f"\nFiles:")
#     print(f"   Original (backup): {input_path.parent / f'{input_path.stem}_backup{input_path.suffix}'}")
#     print(f"   Cleaned:           {input_path}")
#     print(f"   Log:               {log_file}")
    
#     print(f"\n{'='*80}")
#     print(f"✅ FEATURE CLEANING COMPLETE!")
#     print(f"{'='*80}")
    
#     print(f"\n🎯 Next steps:")
#     print(f"   1. Review dropped_features_log.txt")
#     print(f"   2. Verify remaining features look correct")
#     print(f"   3. Proceed with target creation:")
#     print(f"      python src/preprocessing/create_targets.py")
    
#     return df_clean, existing_to_drop


# if __name__ == "__main__":
#     """
#     Main execution: Drop leakage features
#     """
    
#     # Input file
<<<<<<< HEAD
#     input_file = r'data\processed\features_engineered.csv'
=======
#     input_file = 'data/processed/features_engineered.csv'
>>>>>>> 8d1ecdb96c99c112cf86a9fd5fc949c4961a8f42
    
#     # Create backup before overwriting
#     create_backup = True
    
#     try:
#         print(f"\n{'='*80}")
#         print(f"🚀 STARTING FEATURE CLEANING")
#         print(f"{'='*80}")
#         print(f"\nInput file: {input_file}")
        
#         # Drop leakage features
#         df_clean, dropped = drop_leakage_features(
#             input_file=input_file,
#             create_backup=create_backup
#         )
        
#         print(f"\n{'='*80}")
#         print(f"✅ SUCCESS! Feature cleaning completed!")
#         print(f"{'='*80}")
        
#         print(f"\n📋 Summary:")
#         print(f"   Dropped {len(dropped)} leakage features")
#         print(f"   Remaining: {len(df_clean.columns)} clean features")
#         print(f"   Dataset saved to: {input_file}")
        
#     except FileNotFoundError as e:
#         print(f"\n❌ FILE NOT FOUND ERROR:")
#         print(f"   {e}")
#         print(f"\n💡 Make sure the file exists:")
#         print(f"   data/processed/features_engineered.csv")
        
#     except Exception as e:
#         print(f"\n❌ ERROR:")
#         print(f"   {e}")
#         import traceback
#         traceback.print_exc()

<<<<<<< HEAD
"""
src/preprocessing/drop_leakage_features.py

Drops leakage features from the engineered dataset.
Now updated for percentage-change targets:

KEEP THESE TARGETS:
    - target_revenue_pct
    - target_eps_pct
    - target_debt_equity_pct
    - target_profit_margin_pct
    - target_stock_return_pct

Drop:
    - ALL raw target columns (if they exist)
    - ALL current-quarter fundamentals
    - ALL growth features that use current values
    - ALL engineered indicators using current values
=======



"""
src/preprocessing/drop_leakage_features.py

Drops leakage features AFTER targets are created.
Keeps target columns + safe features only.
>>>>>>> 8d1ecdb96c99c112cf86a9fd5fc949c4961a8f42

Input:  data/features/quarterly_data_with_targets.csv (has targets + leakage)
Output: data/features/quarterly_data_with_targets_clean.csv (targets + safe features only)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


<<<<<<< HEAD
def drop_leakage_features(input_file: str, create_backup: bool = True):

=======
def drop_leakage_features(input_file: str, output_file: str):
    """
    Drop features that cause data leakage (but KEEP target columns!)
    
    Args:
        input_file: Path to quarterly_data_with_targets.csv
        output_file: Path to save cleaned file
    """
    
>>>>>>> 8d1ecdb96c99c112cf86a9fd5fc949c4961a8f42
    print("="*80)
    print("🧹 DROPPING LEAKAGE FEATURES")
    print("="*80)

    # ============================
    # 1. LOAD DATA
<<<<<<< HEAD
    # ============================

    print("\n1️⃣ Loading engineered features...")

    input_path = Path(input_file)

=======
    # ========================================
    
    print("\n1️⃣ Loading data with targets...")
    
    input_path = Path(input_file)
    output_path = Path(output_file)
    
>>>>>>> 8d1ecdb96c99c112cf86a9fd5fc949c4961a8f42
    if not input_path.exists():
        raise FileNotFoundError(f"File not found: {input_file}")

    df = pd.read_csv(input_path)

    print(f"   ✅ Loaded {len(df):,} rows × {len(df.columns)} columns")
    print(f"   Date range: {df['Date'].min()} → {df['Date'].max()}")

    original_cols = df.columns.tolist()
    original_shape = df.shape
<<<<<<< HEAD

    # ============================
    # 2. FEATURES TO DROP
    # ============================
    # ❗ Updated to ensure new % targets are kept.

    print("\n2️⃣ Defining leakage features...")

    RAW_TARGETS = [
        "target_revenue",
        "target_eps",
        "target_debt_equity",
        "target_profit_margin",
        "target_stock_return",
        "next_q_return",
        "q_return"
    ]

    NEW_PCT_TARGETS = [
        "target_revenue_pct",
        "target_eps_pct",
        "target_debt_equity_pct",
        "target_profit_margin_pct",
        "target_stock_return_pct",
    ]

    CURRENT_QUARTER_FEATURES = [
        'Revenue', 'EPS', 'Debt_to_Equity', 'net_margin', 'gross_margin',
        'operating_margin', 'roa', 'roe',

        # Price data
        'Stock_Price', 'q_price', 'q_volume', 'q_high', 'q_low', 'q_open',
        'q_price_range_pct', 'Open', 'High', 'Low', 'Close', 'Adj_Close', 'Volume',

        # Other fundamentals
        'Net_Income', 'Gross_Profit', 'Operating_Income', 'EBITDA',
        'Total_Debt', 'Total_Assets', 'Total_Liabilities', 'Total_Equity',
        'Current_Assets', 'Current_Liabilities', 'Long_Term_Debt',
        'Short_Term_Debt', 'Cash', 'Current_Ratio',
    ]

    GROWTH_FEATURES = [
        'Revenue_growth_1q', 'Revenue_growth_4q',
        'Net_Income_growth_1q', 'Net_Income_growth_4q',
        'Gross_Profit_growth_1q', 'Gross_Profit_growth_4q',
        'Operating_Income_growth_1q', 'Operating_Income_growth_4q',
        'eps_growth_1q', 'eps_growth_4q'
    ]

    ENGINEERED_LEAKAGE = [
        'revenue_acceleration', 'net_margin_trend', 'return_momentum',
        'revenue_declining', 'high_leverage', 'liquidity_risk',
        'composite_stress_score', 'leverage_x_vix', 'margin_x_market',
        'revenue_decline_x_vix', 'excess_return', 'return_vs_sector',
        'revenue_growth_vs_sector', 'debt_x_rates',
        'net_margin_zscore', 'roa_zscore', 'roe_zscore',
        'debt_to_assets_zscore'
    ]

    LOG_TRANSFORMED = [
        'log_Revenue', 'log_Total_Assets', 'log_Total_Debt', 'log_Cash'
    ]

    REDUNDANT = [
        'crisis_flag', 'Quarter_End_Date', 'Original_Quarter_End',
        'Quarter_End_Date_fred', 'Company_Name'
    ]

    FEATURES_TO_DROP = (
        RAW_TARGETS +
        CURRENT_QUARTER_FEATURES +
        GROWTH_FEATURES +
        ENGINEERED_LEAKAGE +
        LOG_TRANSFORMED +
        REDUNDANT
    )

    print(f"   Total possible leakage features: {len(FEATURES_TO_DROP)}")

    # ============================
    # 3. FILTER EXISTING FEATURES
    # ============================

    print("\n3️⃣ Checking existence in dataset...")

    existing_to_drop = [c for c in FEATURES_TO_DROP if c in df.columns]
    missing = [c for c in FEATURES_TO_DROP if c not in df.columns]

    print(f"   Will drop: {len(existing_to_drop)}")
    print(f"   Missing (safe): {len(missing)}")

    # ============================
    # 4. BACKUP
    # ============================

    if create_backup:
        print("\n4️⃣ Creating backup...")
        backup_file = input_path.parent / (input_path.stem + "_backup.csv")
        df.to_csv(backup_file, index=False)
        print(f"   Backup saved → {backup_file}")

    # ============================
    # 5. DROP FEATURES
    # ============================

    print("\n5️⃣ Dropping features...")

    df_clean = df.drop(columns=existing_to_drop)
    print(f"   Dropped {len(existing_to_drop)} columns")

    # ============================
    # 6. VALIDATE TARGETS
    # ============================

    print("\n6️⃣ Validating retained % targets...")

    missing_pct = [c for c in NEW_PCT_TARGETS if c not in df_clean.columns]

    if missing_pct:
        print("   ❌ ERROR: Missing new % targets:")
        for c in missing_pct:
            print(f"      - {c}")
        raise ValueError("Percent-change targets missing!")

    print("   ✅ All % targets present!")

    # ============================
    # 7. SAVE RESULT
    # ============================

    print("\n7️⃣ Saving cleaned dataset...")

    df_clean.to_csv(input_path, index=False)

    print(f"   Saved cleaned file → {input_path}")
    print(f"   Final columns: {len(df_clean.columns)}")

    # ============================
    # 8. SUMMARY
    # ============================

    print("\n" + "="*80)
    print("📊 CLEANING SUMMARY")
    print("="*80)
    print(f"Original: {original_shape[1]} columns")
    print(f"Cleaned:  {df_clean.shape[1]} columns")
    print(f"Dropped:  {len(existing_to_drop)}")
    print("="*80)

=======
    
    # Check for target columns
    target_cols = [col for col in df.columns if col.startswith('target_')]
    print(f"   Target columns found: {len(target_cols)}")
    for col in target_cols:
        print(f"      ✅ {col}")
    
    if not target_cols:
        print(f"   ⚠️  WARNING: No target columns found!")
        print(f"      Expected: target_revenue, target_eps, etc.")
        print(f"      Run create_targets.py first!")
    
    # ========================================
    # 2. DEFINE FEATURES TO DROP
    # ========================================
    
    print("\n2️⃣ Defining leakage features to drop...")
    
    FEATURES_TO_DROP = [
        # ========================================
        # A. CURRENT QUARTER VALUES OF TARGETS
        # ========================================
        'Revenue',                    # Predicting next quarter (keep target_revenue!)
        'EPS',                        # Target (keep target_eps!)
        'Debt_to_Equity',             # Target (keep target_debt_equity!)
        'net_margin',                 # Target (keep target_profit_margin!)
        'net_margin_q',               # Same as net_margin
        'gross_margin',               # Derived from target
        'operating_margin',           # Derived from target
        'roa',                        # Derived from targets
        'roe',                        # Derived from targets
        'q_return',                   # Target (keep target_stock_return!)
        'stock_q_return',             # Same as q_return
        'next_q_return',              # Future return (leakage!)
        
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
        'debt_x_rates',               # Uses current debt
        
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
        
        # ========================================
        # L. CALCULATED INTERMEDIATE COLUMNS
        # ========================================
        'EPS_calculated',             # Intermediate calculation
        'profit_margin_calculated',   # Intermediate calculation
        'return_calculated',          # Intermediate calculation
    ]
    
    print(f"   Defined {len(FEATURES_TO_DROP)} features to drop")
    
    # ========================================
    # 3. CHECK WHICH FEATURES EXIST
    # ========================================
    
    print("\n3️⃣ Checking which features exist...")
    
    # CRITICAL: Don't drop target columns!
    existing_to_drop = [
        col for col in FEATURES_TO_DROP 
        if col in df.columns and not col.startswith('target_')
    ]
    
    missing_to_drop = [col for col in FEATURES_TO_DROP if col not in df.columns]
    
    print(f"   Will drop: {len(existing_to_drop)}")
    print(f"   Already absent: {len(missing_to_drop)}")
    
    if existing_to_drop:
        print(f"\n   📋 Dropping {len(existing_to_drop)} leakage features:")
        for i, col in enumerate(existing_to_drop[:15], 1):
            print(f"      {i}. {col}")
        if len(existing_to_drop) > 15:
            print(f"      ... and {len(existing_to_drop) - 15} more")
    
    # ========================================
    # 4. DROP FEATURES (KEEP TARGETS!)
    # ========================================
    
    print(f"\n4️⃣ Dropping leakage features (keeping targets)...")
    
    df_clean = df.drop(columns=existing_to_drop)
    
    print(f"   ✅ Dropped {len(existing_to_drop)} features")
    print(f"   Before: {original_shape[1]} columns")
    print(f"   After:  {df_clean.shape[1]} columns")
    print(f"   Reduction: {len(existing_to_drop)} columns")
    
    # Verify target columns are still there
    remaining_targets = [col for col in df_clean.columns if col.startswith('target_')]
    print(f"\n   ✅ Target columns preserved: {len(remaining_targets)}")
    for col in remaining_targets:
        valid_count = df_clean[col].notna().sum()
        print(f"      ✅ {col}: {valid_count:,} valid values")
    
    # ========================================
    # 5. VALIDATE REMAINING FEATURES
    # ========================================
    
    print(f"\n5️⃣ Validating cleaned dataset...")
    
    remaining_cols = df_clean.columns.tolist()
    
    # Categorize
    identifier_cols = [c for c in remaining_cols if c in ['Date', 'Year', 'Quarter', 'Quarter_Num', 'Company', 'Sector']]
    target_features = [c for c in remaining_cols if c.startswith('target_')]
    macro_cols = [c for c in remaining_cols if any(x in c for x in ['GDP', 'CPI', 'Unemployment', 'Federal', 'Yield', 'Consumer', 'Oil', 'vix', 'sp500'])]
    lag_cols = [c for c in remaining_cols if '_lag_' in c]
    rolling_cols = [c for c in remaining_cols if 'rolling' in c]
    
    print(f"\n   📊 Cleaned dataset composition:")
    print(f"      Identifiers:      {len(identifier_cols)}")
    print(f"      Target variables: {len(target_features)} ← KEPT!")
    print(f"      Macro features:   {len(macro_cols)}")
    print(f"      Lagged features:  {len(lag_cols)}")
    print(f"      Rolling features: {len(rolling_cols)}")
    print(f"      TOTAL:            {len(remaining_cols)}")
    
    # Check for any remaining leakage
    suspicious = []
    leakage_keywords = ['Revenue', 'EPS', 'net_margin', 'Debt_to_Equity', 'q_return', 'Stock_Price']
    
    for col in remaining_cols:
        # Skip target columns (they're supposed to be there!)
        if col.startswith('target_'):
            continue
        # Check for current-quarter values
        if any(keyword in col for keyword in leakage_keywords):
            # Allow lagged/rolling/sector versions
            if not any(x in col for x in ['_lag_', 'rolling', 'sector_avg', 'sp500']):
                suspicious.append(col)
    
    if suspicious:
        print(f"\n   ⚠️  WARNING: Potentially leaky features remain:")
        for col in suspicious:
            print(f"      - {col}")
    else:
        print(f"\n   ✅ No leakage detected in remaining features")
    
    # ========================================
    # 6. SAVE TO NEW FILE
    # ========================================
    
    print(f"\n6️⃣ Saving cleaned dataset to NEW file...")
    
    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save to NEW file (not overwriting!)
    df_clean.to_csv(output_path, index=False)
    
    output_size = output_path.stat().st_size / (1024*1024)
    
    print(f"   ✅ Saved to: {output_path}")
    print(f"      Size: {output_size:.1f} MB")
    print(f"      Rows: {df_clean.shape[0]:,}")
    print(f"      Columns: {df_clean.shape[1]}")
    
    # ========================================
    # 7. SAVE DROPPED FEATURES LOG
    # ========================================
    
    print(f"\n7️⃣ Saving log...")
    
    log_file = output_path.parent / 'dropped_features_log.txt'
    
    with open(log_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("DROPPED FEATURES LOG\n")
        f.write("="*80 + "\n")
        f.write(f"Date: {datetime.now().isoformat()}\n")
        f.write(f"Input file: {input_file}\n")
        f.write(f"Output file: {output_file}\n")
        f.write(f"Original columns: {original_shape[1]}\n")
        f.write(f"Remaining columns: {df_clean.shape[1]}\n")
        f.write(f"Dropped: {len(existing_to_drop)}\n")
        f.write("\n" + "="*80 + "\n")
        f.write("FEATURES DROPPED:\n")
        f.write("="*80 + "\n\n")
        
        for i, col in enumerate(existing_to_drop, 1):
            f.write(f"{i}. {col}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("TARGET COLUMNS PRESERVED:\n")
        f.write("="*80 + "\n\n")
        
        for col in remaining_targets:
            f.write(f"✅ {col}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("FEATURES NOT FOUND (already absent):\n")
        f.write("="*80 + "\n\n")
        
        for col in missing_to_drop:
            f.write(f"- {col}\n")
    
    print(f"   ✅ Log saved: {log_file}")
    
    # ========================================
    # 8. SUMMARY
    # ========================================
    
    print(f"\n{'='*80}")
    print(f"📊 FEATURE CLEANING SUMMARY")
    print(f"{'='*80}")
    
    print(f"\n✅ INPUT FILE (preserved):")
    print(f"   {input_file}")
    print(f"   Rows: {original_shape[0]:,}")
    print(f"   Columns: {original_shape[1]}")
    
    print(f"\n✅ OUTPUT FILE (cleaned):")
    print(f"   {output_file}")
    print(f"   Rows: {df_clean.shape[0]:,}")
    print(f"   Columns: {df_clean.shape[1]}")
    
    print(f"\n📉 Changes:")
    print(f"   Leakage features dropped: {len(existing_to_drop)}")
    print(f"   Target columns preserved: {len(remaining_targets)}")
    print(f"   Reduction: {len(existing_to_drop)/original_shape[1]*100:.1f}%")
    
    print(f"\n📁 Files:")
    print(f"   Original: {input_file} (unchanged)")
    print(f"   Cleaned:  {output_file} (new)")
    print(f"   Log:      {log_file}")
    
    print(f"\n{'='*80}")
    print(f"✅ FEATURE CLEANING COMPLETE!")
    print(f"{'='*80}")
    
    print(f"\n🎯 Next step:")
    print(f"   Create temporal splits:")
    print(f"   python src/preprocessing/create_temporal_splits.py")
    
>>>>>>> 8d1ecdb96c99c112cf86a9fd5fc949c4961a8f42
    return df_clean, existing_to_drop


if __name__ == "__main__":
<<<<<<< HEAD

    input_file = r"data/processed/features_engineered.csv"

    print("\n🚀 Starting feature cleaning...")
    df_clean, dropped = drop_leakage_features(input_file=input_file)
    print("\n✅ Done!")
=======
    """
    Main execution: Drop leakage features after targets are created
    """
    
    # Input: File WITH targets (from create_targets.py)
    input_file = 'data/features/quarterly_data_with_targets.csv'
    
    # Output: NEW cleaned file (targets + safe features only)
    output_file = 'data/features/quarterly_data_with_targets_clean.csv'
    
    try:
        print(f"\n{'='*80}")
        print(f"🚀 STARTING FEATURE CLEANING")
        print(f"{'='*80}")
        print(f"\n📥 Input:  {input_file}")
        print(f"📤 Output: {output_file}")
        print(f"   (Original file will be preserved!)")
        
        # Drop leakage features
        df_clean, dropped = drop_leakage_features(
            input_file=input_file,
            output_file=output_file
        )
        
        print(f"\n{'='*80}")
        print(f"✅ SUCCESS! Feature cleaning completed!")
        print(f"{'='*80}")
        
        print(f"\n📋 What happened:")
        print(f"   ✅ Read: {input_file}")
        print(f"   ✅ Dropped {len(dropped)} leakage features")
        print(f"   ✅ Kept target columns")
        print(f"   ✅ Saved: {output_file}")
        
        print(f"\n📁 You now have:")
        print(f"   Original (with leakage): {input_file}")
        print(f"   Cleaned (safe):          {output_file}")
        
    except FileNotFoundError as e:
        print(f"\n❌ FILE NOT FOUND ERROR:")
        print(f"   {e}")
        print(f"\n💡 Run create_targets.py first to create:")
        print(f"   data/features/quarterly_data_with_targets.csv")
        
    except Exception as e:
        print(f"\n❌ ERROR:")
        print(f"   {e}")
        import traceback
        traceback.print_exc()
>>>>>>> 8d1ecdb96c99c112cf86a9fd5fc949c4961a8f42
