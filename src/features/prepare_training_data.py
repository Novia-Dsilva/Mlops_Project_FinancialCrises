"""
Prepare Training Data
Splits merged dataset into train/test sets for model training
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import yaml


def load_params():
    """Load parameters from params.yaml"""
    with open('params.yaml', 'r') as f:
        return yaml.safe_load(f)


def load_merged_data():
    """Load the merged feature dataset"""

    print("\n" + "="*70)
    print("PREPARING TRAINING DATA")
    print("="*70)

    # Try to find the merged data file in multiple locations
    possible_paths = [
        # Your actual path
        'data/processed/labeled_data.csv',  # If labels were created
        'data/features/merged_features_clean_with_anomaly_flags_with_drift_flags.csv',
        'data/processed/merged_features_clean_with_anomaly_flags_with_drift_flags.csv',
        'merged_features_clean_with_anomaly_flags_with_drift_flags.csv'
    ]

    data_path = None
    for path in possible_paths:
        if Path(path).exists():
            data_path = Path(path)
            break

    if data_path is None:
        raise FileNotFoundError(
            f"Could not find merged data file. Tried:\n" +
            "\n".join(f"  - {p}" for p in possible_paths)
        )

    print(f"\n📥 Loading merged dataset...")
    print(f"  Path: {data_path}")

    df = pd.read_csv(data_path)

    print(f"  ✓ Data loaded")
    print(f"  Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")

    # Check for Date column
    if 'Date' in df.columns:
        print(f"  Date range: {df['Date'].min()} to {df['Date'].max()}")

    return df, data_path


def identify_target_column(df):
    """Identify the target column (what we're predicting)"""

    print("\n🎯 Identifying target column...")

    # Common target column names
    possible_targets = [
        'Financial_Distress',
        'At_Risk',
        'Company_At_Risk',
        'Distressed',
        'Target',
        'Label',
        'is_distressed'
    ]

    target_col = None
    for col in possible_targets:
        if col in df.columns:
            target_col = col
            break

    if target_col is None:
        print("\n⚠ Could not auto-detect target column.")
        print("Available columns:")
        for i, col in enumerate(df.columns, 1):
            print(f"  {i:2d}. {col}")

        # You'll need to specify manually
        raise ValueError(
            "Please specify target column name.\n"
            "Common options: Financial_Distress, At_Risk, Company_At_Risk"
        )

    print(f"  ✓ Target column: '{target_col}'")
    print(f"  Distribution: {df[target_col].value_counts().to_dict()}")

    return target_col


def identify_feature_columns(df, target_col):
    """Identify feature columns (exclude metadata and target)"""

    print("\n📊 Identifying feature columns...")

    # Columns to exclude
    exclude_cols = [
        target_col,           # Target
        'Date',               # Temporal identifier
        'Company',            # Company identifier
        'Company_Name',       # Company name
        'Sector',             # Can include as categorical if you want
        'Ticker',             # Ticker symbol
        'Quarter',            # Temporal
        'Year',               # Temporal
    ]

    # Get feature columns
    feature_cols = [col for col in df.columns if col not in exclude_cols]

    print(f"  ✓ Features identified: {len(feature_cols)} columns")
    print(f"  Excluded: {[col for col in exclude_cols if col in df.columns]}")

    # Show sample features
    print(f"\n  Sample features:")
    for i, col in enumerate(feature_cols[:10], 1):
        print(f"    {i:2d}. {col}")
    if len(feature_cols) > 10:
        print(f"    ... and {len(feature_cols) - 10} more")

    return feature_cols


def create_temporal_split(df, split_date='2020-01-01'):
    """
    Create temporal train/test split

    Train: Before split_date
    Test: After split_date
    """

    print(f"\n📅 Creating temporal split...")
    print(f"  Split date: {split_date}")

    # Convert Date to datetime
    df['Date'] = pd.to_datetime(df['Date'])

    # Split
    train_df = df[df['Date'] < split_date].copy()
    test_df = df[df['Date'] >= split_date].copy()

    print(
        f"  Train: {len(train_df):,} rows ({df['Date'].min()} to {split_date})")
    print(
        f"  Test:  {len(test_df):,} rows ({split_date} to {df['Date'].max()})")
    print(f"  Split: {len(train_df)/(len(train_df)+len(test_df))*100:.1f}% train, {len(test_df)/(len(train_df)+len(test_df))*100:.1f}% test")

    return train_df, test_df


def create_random_split(df, test_size=0.15, random_state=42):
    """
    Create random train/test split

    Alternative to temporal split
    """

    print(f"\n🎲 Creating random split...")
    print(f"  Test size: {test_size*100:.0f}%")

    from sklearn.model_selection import train_test_split

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[target_col] if target_col in df.columns else None
    )

    print(
        f"  Train: {len(train_df):,} rows ({len(train_df)/(len(df))*100:.1f}%)")
    print(
        f"  Test:  {len(test_df):,} rows ({len(test_df)/(len(df))*100:.1f}%)")

    return train_df, test_df


def prepare_training_data():
    """Main function to prepare training data"""

    # Load parameters
    params = load_params()
    train_params = params['train']

    # Load merged data
    df, data_path = load_merged_data()

    # Identify target and features
    target_col = identify_target_column(df)
    feature_cols = identify_feature_columns(df, target_col)

    # Create train/test split
    if train_params['split_strategy'] == 'temporal':
        split_date = train_params.get('temporal_split_date', '2020-01-01')
        train_df, test_df = create_temporal_split(df, split_date)
    else:
        test_size = train_params['test_split']
        random_state = train_params['random_state']
        train_df, test_df = create_random_split(df, test_size, random_state)

    # Extract X and y
    print("\n📦 Extracting features and targets...")

    X_train = train_df[feature_cols].copy()
    y_train = train_df[target_col].copy()

    X_test = test_df[feature_cols].copy()
    y_test = test_df[target_col].copy()

    print(f"  X_train: {X_train.shape}")
    print(f"  y_train: {y_train.shape}")
    print(f"  X_test:  {X_test.shape}")
    print(f"  y_test:  {y_test.shape}")

    # CRITICAL: Handle non-numeric data
    print("\n🔧 Cleaning non-numeric data...")

    # Identify non-numeric columns
    non_numeric_cols = []
    for col in X_train.columns:
        if X_train[col].dtype == 'object' or X_train[col].dtype == 'string':
            non_numeric_cols.append(col)

    if non_numeric_cols:
        print(f"  Found {len(non_numeric_cols)} non-numeric columns:")
        for col in non_numeric_cols:
            sample_vals = X_train[col].unique()[:5]
            print(f"    - {col}: {sample_vals}")

        # Option 1: One-hot encode categorical columns
        print(f"\n  Encoding categorical columns...")
        X_train = pd.get_dummies(
            X_train, columns=non_numeric_cols, drop_first=True)
        X_test = pd.get_dummies(
            X_test, columns=non_numeric_cols, drop_first=True)

        # Align columns (test set might have different categories)
        X_train, X_test = X_train.align(
            X_test, join='left', axis=1, fill_value=0)

        print(f"  ✓ Encoded: {X_train.shape[1]} features after encoding")
    else:
        print(f"  ✓ All columns are numeric")

    # Convert any remaining object columns to numeric (force)
    for col in X_train.columns:
        if X_train[col].dtype == 'object':
            print(f"  Converting {col} to numeric...")
            X_train[col] = pd.to_numeric(X_train[col], errors='coerce')
            X_test[col] = pd.to_numeric(X_test[col], errors='coerce')

    # Check for missing values
    print("\n🔍 Checking data quality...")
    print(f"  Missing values in X_train: {X_train.isna().sum().sum():,}")
    print(f"  Missing values in X_test: {X_test.isna().sum().sum():,}")

    # Handle missing values if any
    if X_train.isna().sum().sum() > 0:
        print("\n⚠ Handling missing values with forward fill...")
        X_train = X_train.fillna(method='ffill').fillna(
            method='bfill').fillna(0)
        X_test = X_test.fillna(method='ffill').fillna(method='bfill').fillna(0)
        print("  ✓ Missing values handled")

    # Save to CSV
    print("\n💾 Saving training data splits...")

    output_dir = Path('data/features')
    output_dir.mkdir(parents=True, exist_ok=True)

    X_train.to_csv(output_dir / 'X_train.csv', index=False)
    X_test.to_csv(output_dir / 'X_test.csv', index=False)
    y_train.to_csv(output_dir / 'y_train.csv', index=False, header=['target'])
    y_test.to_csv(output_dir / 'y_test.csv', index=False, header=['target'])

    print(f"  ✓ X_train.csv ({X_train.shape})")
    print(f"  ✓ X_test.csv ({X_test.shape})")
    print(f"  ✓ y_train.csv ({y_train.shape})")
    print(f"  ✓ y_test.csv ({y_test.shape})")

    # Save feature names
    feature_info = {
        'feature_columns': feature_cols,
        'n_features': len(feature_cols),
        'target_column': target_col,
        'created_date': datetime.now().isoformat(),
        'source_file': str(data_path),
        'split_strategy': train_params['split_strategy']
    }

    with open(output_dir / 'feature_info.json', 'w') as f:
        json.dump(feature_info, f, indent=2)

    print(f"\n  ✓ Feature info saved to feature_info.json")

    # Summary statistics
    print("\n" + "="*70)
    print("📊 DATA SPLIT SUMMARY")
    print("="*70)
    print(f"Total samples: {len(df):,}")
    print(
        f"Training samples: {len(X_train):,} ({len(X_train)/len(df)*100:.1f}%)")
    print(f"Test samples: {len(X_test):,} ({len(X_test)/len(df)*100:.1f}%)")
    print(f"Total features: {len(feature_cols)}")
    print(f"Target column: {target_col}")
    print(f"\nClass balance (train):")
    print(
        f"  Class 0 (Healthy): {(y_train==0).sum():,} ({(y_train==0).sum()/len(y_train)*100:.1f}%)")
    print(
        f"  Class 1 (Distressed): {(y_train==1).sum():,} ({(y_train==1).sum()/len(y_train)*100:.1f}%)")
    print(f"\nClass balance (test):")
    print(
        f"  Class 0 (Healthy): {(y_test==0).sum():,} ({(y_test==0).sum()/len(y_test)*100:.1f}%)")
    print(
        f"  Class 1 (Distressed): {(y_test==1).sum():,} ({(y_test==1).sum()/len(y_test)*100:.1f}%)")
    print("="*70)

    print("\n✅ Training data preparation complete!")
    print("\n🎯 Next step: Run training")
    print("  python src/models/step4_train_model.py")

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    import sys
    import json

    try:
        X_train, X_test, y_train, y_test = prepare_training_data()
        sys.exit(0)

    except Exception as e:
        print(f"\n❌ Data preparation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
