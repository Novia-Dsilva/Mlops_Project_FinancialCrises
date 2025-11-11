# File: scripts/01_prepare_data.py

"""
Prepare data with temporal shifting for next quarter prediction
Creates features from current quarter → targets from next quarter
"""

import json
import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime

# ============================================================
# CONFIGURATION
# ============================================================

DATA_FILE = r"C:\Users\Sushmitha Sudharsan\Desktop\Mlops_Project_FinancialCrises\data\features\merged_features_clean_with_anomaly_flags_with_drift_flags.csv"
OUTPUT_DIR = "data/processed"

# Columns to predict (will be shifted to "next quarter")
TARGET_COLS = [
    'Revenue',
    'Debt_to_Equity',
    'Profit_Margin',
    'Stock_Price'
]

DATE_COL = 'Date'
COMPANY_COL = 'Company'

# Train/test split date
TRAIN_END_DATE = '2022-12-31'

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# LOAD DATA
# ============================================================

print("="*70)
print("📂 STEP 1: LOADING DATA")
print("="*70)

df = pd.read_csv(DATA_FILE)
print(f"✅ Loaded {len(df)} rows, {len(df.columns)} columns")

# Parse date
df[DATE_COL] = pd.to_datetime(df[DATE_COL])
print(f"✅ Date range: {df[DATE_COL].min()} to {df[DATE_COL].max()}")

# Check for company column
if COMPANY_COL in df.columns:
    print(f"✅ Companies: {df[COMPANY_COL].nunique()}")
    companies = df[COMPANY_COL].unique()
    print(f"   {companies[:5]}..." if len(
        companies) > 5 else f"   {companies}")
else:
    print("⚠️  No Company column found - treating as single entity")
    df[COMPANY_COL] = 'ALL'

# ============================================================
# CREATE TEMPORAL SHIFT (NEXT QUARTER TARGETS)
# ============================================================

print("\n" + "="*70)
print("🔄 STEP 2: CREATING NEXT QUARTER TARGETS")
print("="*70)

# Sort by company and date (critical!)
df = df.sort_values([COMPANY_COL, DATE_COL]).reset_index(drop=True)
print(f"✅ Data sorted by {COMPANY_COL} and {DATE_COL}")

# Create next quarter targets
print("\n📊 Creating temporal shift for targets...")
for target in TARGET_COLS:
    if target not in df.columns:
        print(f"❌ ERROR: Target column '{target}' not found!")
        exit(1)

    # Shift -1 within each company (get next quarter's value)
    df[f'{target}_Next_Q'] = df.groupby(COMPANY_COL)[target].shift(-1)

    # Count how many valid next quarter values we have
    valid_count = df[f'{target}_Next_Q'].notna().sum()
    print(f"  ✅ {target:<20} → {target}_Next_Q ({valid_count} valid values)")

# Drop rows where we don't have next quarter data
# (last quarter for each company won't have next quarter values)
before_drop = len(df)
df = df.dropna(subset=[f'{col}_Next_Q' for col in TARGET_COLS])
after_drop = len(df)

print(
    f"\n✅ Dropped {before_drop - after_drop} rows (last quarter per company)")
print(f"✅ Remaining: {after_drop} rows with valid next quarter targets")

# Show example
print("\n📋 Example: Current → Next Quarter")
print("-" * 70)
sample = df.iloc[0]
print(f"Company: {sample[COMPANY_COL]}")
print(f"Date: {sample[DATE_COL]}")
print(f"\nCurrent Quarter → Next Quarter:")
for target in TARGET_COLS:
    print(
        f"  {target:<20}: {sample[target]:>10.2f} → {sample[f'{target}_Next_Q']:>10.2f}")

# ============================================================
# DEFINE FEATURES (EXCLUDE NEXT QUARTER TARGETS!)
# ============================================================

print("\n" + "="*70)
print("🔍 STEP 3: DEFINING FEATURES")
print("="*70)

# Columns to exclude from features
exclude_cols = [
    DATE_COL,
    COMPANY_COL
] + TARGET_COLS  # Exclude current quarter targets (those are what we're predicting!)

# Also exclude the next quarter target columns (they're our y, not X)
next_q_targets = [f'{col}_Next_Q' for col in TARGET_COLS]

# Exclude anomaly/drift flags (optional - you can keep if useful for prediction)
exclude_patterns = ['_anomaly', '_drift', 'Unnamed']

# Get all numeric columns
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# Filter features
FEATURE_COLS = [
    col for col in numeric_cols
    if col not in exclude_cols
    and col not in next_q_targets
    and not any(pattern in col for pattern in exclude_patterns)
]

print(f"✅ Selected {len(FEATURE_COLS)} feature columns")
print(f"\n📋 Features (first 20):")
for i, col in enumerate(FEATURE_COLS[:20], 1):
    print(f"  {i:3d}. {col}")
if len(FEATURE_COLS) > 20:
    print(f"  ... and {len(FEATURE_COLS) - 20} more")

print(f"\n🎯 Target columns (Next Quarter):")
for i, col in enumerate(next_q_targets, 1):
    print(f"  {i}. {col}")

# ============================================================
# HANDLE MISSING VALUES
# ============================================================

print("\n" + "="*70)
print("🧹 STEP 4: HANDLING MISSING VALUES")
print("="*70)

print(f"Missing values in features: {df[FEATURE_COLS].isnull().sum().sum()}")
print(f"Missing values in targets: {df[next_q_targets].isnull().sum().sum()}")

# Drop any remaining rows with missing values
before_clean = len(df)
df_clean = df.dropna(subset=FEATURE_COLS + next_q_targets)
after_clean = len(df_clean)

print(f"✅ Dropped {before_clean - after_clean} rows with missing values")
print(f"✅ Final dataset: {after_clean} rows")

if after_clean < 100:
    print("⚠️  WARNING: Very few samples! Check your data.")

# ============================================================
# TEMPORAL TRAIN/TEST SPLIT
# ============================================================

print("\n" + "="*70)
print("📅 STEP 5: TEMPORAL TRAIN/TEST SPLIT")
print("="*70)

print(f"Split date: {TRAIN_END_DATE}")
print(f"  Train: All data ≤ {TRAIN_END_DATE}")
print(f"  Test:  All data > {TRAIN_END_DATE}")

train_mask = df_clean[DATE_COL] <= TRAIN_END_DATE
test_mask = df_clean[DATE_COL] > TRAIN_END_DATE

train_df = df_clean[train_mask].copy()
test_df = df_clean[test_mask].copy()

print(
    f"\n✅ Train: {len(train_df)} samples ({len(train_df)/len(df_clean)*100:.1f}%)")
print(
    f"   Date range: {train_df[DATE_COL].min()} to {train_df[DATE_COL].max()}")

print(
    f"✅ Test:  {len(test_df)} samples ({len(test_df)/len(df_clean)*100:.1f}%)")
print(f"   Date range: {test_df[DATE_COL].min()} to {test_df[DATE_COL].max()}")

if len(test_df) == 0:
    print("❌ ERROR: No test data! Adjust TRAIN_END_DATE to an earlier date.")
    exit(1)

# Verify no leakage
latest_train = train_df[DATE_COL].max()
earliest_test = test_df[DATE_COL].min()
print(f"\n🔒 Leakage check:")
print(f"   Latest train date: {latest_train}")
print(f"   Earliest test date: {earliest_test}")
if latest_train >= earliest_test:
    print("❌ ERROR: Temporal leakage detected!")
    exit(1)
else:
    print("✅ No temporal leakage - train and test are properly separated")

# ============================================================
# EXTRACT X AND y
# ============================================================

print("\n" + "="*70)
print("📊 STEP 6: EXTRACTING FEATURES AND TARGETS")
print("="*70)

X_train = train_df[FEATURE_COLS].values
X_test = test_df[FEATURE_COLS].values

y_train = train_df[next_q_targets].values
y_test = test_df[next_q_targets].values

print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_test shape:  {X_test.shape}")
print(f"y_test shape:  {y_test.shape}")

# ============================================================
# SAVE PROCESSED DATA
# ============================================================

print("\n" + "="*70)
print("💾 STEP 7: SAVING PROCESSED DATA")
print("="*70)

# Save arrays
np.save(f'{OUTPUT_DIR}/X_train.npy', X_train)
np.save(f'{OUTPUT_DIR}/X_test.npy', X_test)
np.save(f'{OUTPUT_DIR}/y_train.npy', y_train)
np.save(f'{OUTPUT_DIR}/y_test.npy', y_test)

print(f"✅ Saved numpy arrays to {OUTPUT_DIR}/")

# Save metadata
metadata = {
    'feature_cols': FEATURE_COLS,
    'target_cols': next_q_targets,
    'original_target_cols': TARGET_COLS,
    'n_features': len(FEATURE_COLS),
    'n_targets': len(TARGET_COLS),
    'train_samples': len(X_train),
    'test_samples': len(X_test),
    'train_date_range': f"{train_df[DATE_COL].min()} to {train_df[DATE_COL].max()}",
    'test_date_range': f"{test_df[DATE_COL].min()} to {test_df[DATE_COL].max()}",
    'split_date': TRAIN_END_DATE,
    'created_at': datetime.now().isoformat()
}

with open(f'{OUTPUT_DIR}/metadata.pkl', 'wb') as f:
    pickle.dump(metadata, f)

print(f"✅ Saved metadata to {OUTPUT_DIR}/metadata.pkl")

# Also save as JSON for readability
with open(f'{OUTPUT_DIR}/metadata.json', 'w') as f:
    # Convert to JSON-serializable format
    json_metadata = {k: v if not isinstance(v, (list, np.ndarray)) else
                     v[:10] if isinstance(v, list) and len(v) > 10 else v
                     for k, v in metadata.items() if k != 'feature_cols'}
    json_metadata['n_features'] = metadata['n_features']
    json_metadata['feature_cols_sample'] = FEATURE_COLS[:10]
    json.dump(json_metadata, f, indent=2)

print(f"✅ Saved metadata.json (human readable)")

# Save full dataframes for reference
train_df.to_csv(f'{OUTPUT_DIR}/train_data.csv', index=False)
test_df.to_csv(f'{OUTPUT_DIR}/test_data.csv', index=False)

print(f"✅ Saved train_data.csv and test_data.csv")

# ============================================================
# SUMMARY
# ============================================================

print("\n" + "="*70)
print("✅ DATA PREPARATION COMPLETE!")
print("="*70)

print(f"\n📊 Summary:")
print(f"  Original data: {len(df)} rows")
print(f"  After temporal shift: {len(df_clean)} rows")
print(f"  Train samples: {len(X_train)}")
print(f"  Test samples: {len(X_test)}")
print(f"  Features: {len(FEATURE_COLS)}")
print(f"  Targets: {len(TARGET_COLS)}")

print(f"\n📁 Output files in '{OUTPUT_DIR}/':")
print(f"  - X_train.npy, X_test.npy")
print(f"  - y_train.npy, y_test.npy")
print(f"  - metadata.pkl, metadata.json")
print(f"  - train_data.csv, test_data.csv")

print(f"\n➡️  Next step: Run scripts/02_train_xgboost.py")
print("="*70)
