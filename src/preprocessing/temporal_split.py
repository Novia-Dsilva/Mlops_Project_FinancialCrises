"""
src/preprocessing/create_temporal_splits.py

Creates temporal splits using EXPLICIT DATES (better than percentages!)
"""

import pandas as pd
from pathlib import Path
from datetime import datetime


def create_date_based_splits(
    input_file: str,
    output_dir: str,
    # train_end_date: str = '2019-12-31',
    # val_end_date: str = '2022-12-31',
    train_end_date: str = '2015-12-31',
    val_end_date: str = '2019-12-31',
    date_column: str = 'Date'
):
    """
    Create temporal splits using explicit dates.
    
    Args:
        input_file: Path to quarterly data with targets
        output_dir: Directory to save splits
        train_end_date: Last date for training (inclusive)
        val_end_date: Last date for validation (inclusive)
        date_column: Name of date column
    
    Returns:
        train_df, val_df, test_df
    """
    
    print("="*80)
    print("🔄 CREATING DATE-BASED TEMPORAL SPLITS")
    print("="*80)
    
    # ========================================
    # 1. LOAD DATA
    # ========================================
    
    print("\n1️⃣ Loading quarterly data with targets...")
    
    if not Path(input_file).exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    df = pd.read_csv(input_file)
    
    print(f"   ✅ Loaded: {len(df):,} rows")
    print(f"   Columns: {len(df.columns)}")
    
    # Convert date column
    df[date_column] = pd.to_datetime(df[date_column])
    
    print(f"   Date range: {df[date_column].min()} to {df[date_column].max()}")
    
    # Check for target columns
    target_cols = [col for col in df.columns if col.startswith('target_')]
    print(f"   Target columns: {len(target_cols)}")
    for col in target_cols:
        valid_count = df[col].notna().sum()
        print(f"      - {col}: {valid_count:,} valid values")
    
    if not target_cols:
        raise ValueError("No target columns found! Run create_targets.py first!")
    
    # ========================================
    # 2. SORT BY DATE (CRITICAL!)
    # ========================================
    
    print("\n2️⃣ Sorting data by Date...")
    
    df = df.sort_values([date_column, 'Company']).reset_index(drop=True)
    
    print(f"   ✅ Sorted by Date and Company")
    
    # ========================================
    # 3. PERFORM DATE-BASED SPLIT
    # ========================================
    
    print("\n3️⃣ Performing date-based split...")
    
    # Convert split dates
    train_end = pd.to_datetime(train_end_date)
    val_end = pd.to_datetime(val_end_date)
    
    # Split by dates
    train_mask = df[date_column] <= train_end
    val_mask = (df[date_column] > train_end) & (df[date_column] <= val_end)
    test_mask = df[date_column] > val_end
    
    train_df = df[train_mask].copy()
    val_df = df[val_mask].copy()
    test_df = df[test_mask].copy()
    
    # ========================================
    # 4. PRINT SPLIT INFORMATION
    # ========================================
    
    print("\n" + "="*80)
    print("📊 DATE-BASED SPLIT SUMMARY")
    print("="*80)
    
    # Train set
    train_start = train_df[date_column].min()
    train_end_actual = train_df[date_column].max()
    train_pct = (len(train_df) / len(df)) * 100
    
    print(f"\n🔵 TRAIN SET: {len(train_df):,} rows ({train_pct:.1f}%)")
    print(f"   Date range: {train_start.date()} to {train_end_actual.date()}")
    print(f"   Duration: {(train_end_actual - train_start).days} days")
    print(f"   Companies: {train_df['Company'].nunique()}")
    
    # Show target stats for train
    print(f"   Target statistics:")
    for target_col in target_cols[:3]:  # Show first 3
        if target_col in train_df.columns:
            valid = train_df[target_col].notna().sum()
            mean = train_df[target_col].mean()
            print(f"      {target_col}: {valid:,} valid, mean={mean:.2f}")
    
    # Validation set
    val_start = val_df[date_column].min()
    val_end_actual = val_df[date_column].max()
    val_pct = (len(val_df) / len(df)) * 100
    
    print(f"\n🟢 VALIDATION SET: {len(val_df):,} rows ({val_pct:.1f}%)")
    print(f"   Date range: {val_start.date()} to {val_end_actual.date()}")
    print(f"   Duration: {(val_end_actual - val_start).days} days")
    print(f"   Companies: {val_df['Company'].nunique()}")
    print(f"   📌 Contains: COVID-19 crisis period ⚠️")
    
    # Test set
    test_start = test_df[date_column].min()
    test_end_actual = test_df[date_column].max()
    test_pct = (len(test_df) / len(df)) * 100
    
    print(f"\n🔴 TEST SET: {len(test_df):,} rows ({test_pct:.1f}%)")
    print(f"   Date range: {test_start.date()} to {test_end_actual.date()}")
    print(f"   Duration: {(test_end_actual - test_start).days} days")
    print(f"   Companies: {test_df['Company'].nunique()}")
    print(f"   📌 Most recent period (current conditions)")
    
    print("\n" + "="*80)
    
    # ========================================
    # 5. VALIDATE NO OVERLAP
    # ========================================
    
    print("\n4️⃣ Validating temporal integrity...")
    
    # Check no overlap
    if train_end_actual >= val_start:
        print(f"   ❌ ERROR: Train overlaps with Val!")
        print(f"      Train ends: {train_end_actual.date()}")
        print(f"      Val starts: {val_start.date()}")
        raise ValueError("Temporal overlap detected!")
    
    if val_end_actual >= test_start:
        print(f"   ❌ ERROR: Val overlaps with Test!")
        print(f"      Val ends: {val_end_actual.date()}")
        print(f"      Test starts: {test_start.date()}")
        raise ValueError("Temporal overlap detected!")
    
    print(f"   ✅ No temporal overlap")
    print(f"      Train → Val gap: {(val_start - train_end_actual).days} days")
    print(f"      Val → Test gap: {(test_start - val_end_actual).days} days")
    
    # ========================================
    # 6. CHECK COMPANY COVERAGE
    # ========================================
    
    print("\n5️⃣ Checking company coverage...")
    
    train_companies = set(train_df['Company'].unique())
    val_companies = set(val_df['Company'].unique())
    test_companies = set(test_df['Company'].unique())
    
    all_companies = train_companies | val_companies | test_companies
    
    print(f"   Total unique companies: {len(all_companies)}")
    print(f"   In train: {len(train_companies)}")
    print(f"   In val: {len(val_companies)}")
    print(f"   In test: {len(test_companies)}")
    
    # Companies only in certain sets
    only_train = train_companies - val_companies - test_companies
    only_val = val_companies - train_companies - test_companies
    only_test = test_companies - train_companies - val_companies
    
    if only_train:
        print(f"   ⚠️  {len(only_train)} companies ONLY in train: {list(only_train)[:3]}...")
    if only_val:
        print(f"   ⚠️  {len(only_val)} companies ONLY in val: {list(only_val)[:3]}...")
    if only_test:
        print(f"   ⚠️  {len(only_test)} companies ONLY in test: {list(only_test)[:3]}...")
    
    # Companies in all sets (ideal)
    in_all_sets = train_companies & val_companies & test_companies
    print(f"   ✅ {len(in_all_sets)} companies in ALL sets (ideal!)")
    
    # ========================================
    # 7. CREATE OUTPUT DIRECTORY
    # ========================================
    
    print("\n6️⃣ Creating output directory...")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"   ✅ Output directory: {output_path}")
    
    # ========================================
    # 8. SAVE SPLITS
    # ========================================
    
    print("\n7️⃣ Saving splits to disk...")
    
    train_file = output_path / 'train_data.csv'
    val_file = output_path / 'val_data.csv'
    test_file = output_path / 'test_data.csv'
    
    train_df.to_csv(train_file, index=False)
    val_df.to_csv(val_file, index=False)
    test_df.to_csv(test_file, index=False)
    
    print(f"   ✅ Saved train: {train_file}")
    print(f"      Size: {Path(train_file).stat().st_size / (1024*1024):.1f} MB")
    
    print(f"   ✅ Saved val: {val_file}")
    print(f"      Size: {Path(val_file).stat().st_size / (1024*1024):.1f} MB")
    
    print(f"   ✅ Saved test: {test_file}")
    print(f"      Size: {Path(test_file).stat().st_size / (1024*1024):.1f} MB")
    
    # ========================================
    # 9. SAVE SPLIT METADATA
    # ========================================
    
    import json
    
    metadata = {
        'split_method': 'date-based',
        'train': {
            'start_date': str(train_start.date()),
            'end_date': str(train_end_actual.date()),
            'rows': int(len(train_df)),
            'percentage': float(train_pct)
        },
        'val': {
            'start_date': str(val_start.date()),
            'end_date': str(val_end_actual.date()),
            'rows': int(len(val_df)),
            'percentage': float(val_pct),
            'note': 'Contains COVID-19 crisis period'
        },
        'test': {
            'start_date': str(test_start.date()),
            'end_date': str(test_end_actual.date()),
            'rows': int(len(test_df)),
            'percentage': float(test_pct),
            'note': 'Most recent period'
        },
        'companies': {
            'total': len(all_companies),
            'in_all_sets': len(in_all_sets)
        }
    }
    
    metadata_file = output_path / 'split_metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"   💾 Saved metadata: {metadata_file}")
    
    # ========================================
    # 10. FINAL SUMMARY
    # ========================================
    
    print("\n" + "="*80)
    print("✅ DATE-BASED SPLITS CREATED SUCCESSFULLY!")
    print("="*80)
    
    print(f"\n📋 Summary:")
    print(f"   Total data: {len(df):,} rows")
    print(f"   Train: {len(train_df):,} rows ({train_pct:.1f}%) | {train_start.date()} to {train_end_actual.date()}")
    print(f"   Val:   {len(val_df):,} rows ({val_pct:.1f}%) | {val_start.date()} to {val_end_actual.date()}")
    print(f"   Test:  {len(test_df):,} rows ({test_pct:.1f}%) | {test_start.date()} to {test_end_actual.date()}")
    
    print(f"\n🎯 Next steps:")
    print(f"   1. Handle outliers: python src/preprocessing/handle_outliers_after_split.py")
    print(f"   2. Train models: python src/models/train_xgboost.py")
    
    return train_df, val_df, test_df


if __name__ == "__main__":
    
    # ========================================
    # CONFIGURATION
    # ========================================
    
    # Input file (quarterly data with targets)
    input_file = 'data/features/quarterly_data_with_targets.csv'
    
    # Output directory
    output_dir = 'data/splits'
    
    # ========================================
    # DATE-BASED SPLIT BOUNDARIES
    # ========================================
    
    # Option 1: Your original proposal (with fix)
    # train_end_date = '2019-12-31'  # Train: 1990-2019
    # val_end_date = '2022-12-31'    # Val: 2020-2022 (COVID period)
    # Updated split boundaries
    train_end_date = '2015-12-31'  # Train: 1990-2015
    val_end_date = '2019-12-31'    # Val: 2016-2019
    # Test: 2023-present (most recent)
    
    # Option 2: More recent test data
    # train_end_date = '2019-12-31'  # Train: 1990-2019
    # val_end_date = '2023-06-30'    # Val: 2020-mid 2023
    # Test: mid 2023-present
    
    # ========================================
    # CREATE SPLITS
    # ========================================
    
    try:
        train_df, val_df, test_df = create_date_based_splits(
            input_file=input_file,
            output_dir=output_dir,
            train_end_date=train_end_date,
            val_end_date=val_end_date
        )
        
        print("\n🚀 Splits ready!")
        
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: {e}")
        print(f"\n💡 Make sure you ran create_targets.py first!")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()