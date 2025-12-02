"""
src/preprocessing/handle_outliers_after_split.py

Handles outliers AFTER temporal split to prevent data leakage.
Calculates thresholds from TRAINING data only!

Author: Financial Crisis MLOps Team
Date: 2025
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


def calculate_winsorization_thresholds(
    train_df: pd.DataFrame,
    target_col: str,
    lower_percentile: float = 1.0,
    upper_percentile: float = 99.0
) -> Tuple[float, float]:
    """
    Calculate winsorization thresholds from TRAINING data only.
    
    CRITICAL: Only uses train_df to prevent data leakage!
    
    Args:
        train_df: Training dataframe
        target_col: Target column name
        lower_percentile: Lower cap percentile (e.g., 1.0 = 1st percentile)
        upper_percentile: Upper cap percentile (e.g., 99.0 = 99th percentile)
    
    Returns:
        (lower_cap, upper_cap) - threshold values
    """
    # Remove NaN values for percentile calculation
    valid_data = train_df[target_col].dropna()
    
    if len(valid_data) == 0:
        raise ValueError(f"No valid data in {target_col} for threshold calculation")
    
    # Calculate from TRAINING data only (no leakage!)
    lower_cap = valid_data.quantile(lower_percentile / 100.0)
    upper_cap = valid_data.quantile(upper_percentile / 100.0)
    
    return float(lower_cap), float(upper_cap)


def apply_winsorization(
    df: pd.DataFrame,
    target_col: str,
    lower_cap: float,
    upper_cap: float,
    set_name: str = ""
) -> Tuple[pd.DataFrame, Dict]:
    """
    Apply winsorization using pre-calculated thresholds.
    
    Args:
        df: Dataframe to clean
        target_col: Target column name
        lower_cap: Lower cap value (from training data)
        upper_cap: Upper cap value (from training data)
        set_name: Name for logging (Train/Val/Test)
    
    Returns:
        Cleaned dataframe and statistics dictionary
    """
    df_clean = df.copy()
    
    # Get valid data only
    valid_mask = df[target_col].notna()
    valid_data = df[target_col][valid_mask]
    
    if len(valid_data) == 0:
        print(f"   {set_name:5s}: No valid data to cap")
        return df_clean, {}
    
    # Count values to cap
    lower_count = (valid_data < lower_cap).sum()
    upper_count = (valid_data > upper_cap).sum()
    total_capped = lower_count + upper_count
    
    # Original stats
    original_min = valid_data.min()
    original_max = valid_data.max()
    original_std = valid_data.std()
    original_mean = valid_data.mean()
    
    # Apply caps
    df_clean[target_col] = df_clean[target_col].clip(lower=lower_cap, upper=upper_cap)
    
    # New stats
    new_valid_data = df_clean[target_col][valid_mask]
    new_min = new_valid_data.min()
    new_max = new_valid_data.max()
    new_std = new_valid_data.std()
    new_mean = new_valid_data.mean()
    
    # Statistics
    pct_capped = (total_capped / len(valid_data)) * 100
    
    stats = {
        'set_name': set_name,
        'total_values': int(len(valid_data)),
        'original_range': [float(original_min), float(original_max)],
        'new_range': [float(new_min), float(new_max)],
        'lower_count': int(lower_count),
        'upper_count': int(upper_count),
        'total_capped': int(total_capped),
        'pct_capped': float(pct_capped),
        'original_mean': float(original_mean),
        'new_mean': float(new_mean),
        'original_std': float(original_std),
        'new_std': float(new_std),
        'std_reduction_pct': float(((original_std - new_std) / original_std) * 100) if original_std > 0 else 0.0
    }
    
    # Log
    if set_name:
        print(f"   {set_name:5s}: [{original_min:15,.2f}, {original_max:15,.2f}] → "
              f"[{new_min:15,.2f}, {new_max:15,.2f}] | "
              f"Capped: {total_capped:5,} ({pct_capped:5.2f}%)")
    
    return df_clean, stats


def handle_outliers_after_split(
    splits_dir: str,
    output_dir: Optional[str] = None,
    target_configs: Optional[Dict[str, Dict]] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
    """
    Handle outliers AFTER temporal split (no data leakage).
    
    Calculates thresholds from training data ONLY,
    then applies to train/val/test.
    
    Args:
        splits_dir: Directory containing train/val/test CSV files
        output_dir: Where to save cleaned files (default: same as splits_dir)
        target_configs: Optional custom config for each target
    
    Returns:
        train_df, val_df, test_df, all_thresholds
    """
    
    print("="*80)
    print("🔧 OUTLIER HANDLING (AFTER SPLIT - NO LEAKAGE)")
    print("="*80)
    
    # Default output to same directory
    if output_dir is None:
        output_dir = splits_dir
    
    # ========================================
    # 1. LOAD SPLITS
    # ========================================
    
    print("\n1️⃣ Loading train/val/test splits...")
    
    splits_path = Path(splits_dir)
    
    train_file = splits_path / 'train_data.csv'
    val_file = splits_path / 'val_data.csv'
    test_file = splits_path / 'test_data.csv'
    
    # Check files exist
    if not train_file.exists():
        raise FileNotFoundError(f"Train file not found: {train_file}")
    if not val_file.exists():
        raise FileNotFoundError(f"Val file not found: {val_file}")
    if not test_file.exists():
        raise FileNotFoundError(f"Test file not found: {test_file}")
    
    # Load data
    train_df = pd.read_csv(train_file)
    val_df = pd.read_csv(val_file)
    test_df = pd.read_csv(test_file)
    
    print(f"   ✅ Train: {len(train_df):,} rows, {len(train_df.columns)} columns")
    print(f"   ✅ Val:   {len(val_df):,} rows, {len(val_df.columns)} columns")
    print(f"   ✅ Test:  {len(test_df):,} rows, {len(test_df.columns)} columns")
    
    # Find target columns
    target_cols = [col for col in train_df.columns if col.startswith('target_')]
    
    if not target_cols:
        raise ValueError("No target columns found! Did you run create_targets.py?")
    
    print(f"\n   Target columns found: {len(target_cols)}")
    for col in target_cols:
        train_valid = train_df[col].notna().sum()
        val_valid = val_df[col].notna().sum()
        test_valid = test_df[col].notna().sum()
        print(f"      - {col}: Train={train_valid:,}, Val={val_valid:,}, Test={test_valid:,}")
    
    # ========================================
    # 2. DEFAULT CONFIGURATIONS
    # ========================================
    
    if target_configs is None:
        target_configs = {
            'target_revenue': {
                'lower_percentile': 1.0,
                'upper_percentile': 99.0,
                'description': 'Revenue (Next Quarter)'
            },
            'target_eps': {
                'lower_percentile': 1.0,
                'upper_percentile': 99.0,
                'description': 'EPS (Next Quarter)'
            },
            'target_debt_equity': {
                'lower_percentile': 1.0,
                'upper_percentile': 99.0,
                'description': 'Debt/Equity (Next Quarter)'
            },
            'target_profit_margin': {
                'lower_percentile': 1.0,
                'upper_percentile': 99.0,
                'description': 'Profit Margin (Next Quarter)'
            },
            'target_stock_return': {
                'lower_percentile': 1.0,
                'upper_percentile': 99.0,
                'description': 'Stock Return (Next Quarter)'
            }
        }
    
    # ========================================
    # 3. HANDLE OUTLIERS FOR EACH TARGET
    # ========================================
    
    print("\n2️⃣ Handling outliers (using TRAIN statistics only)...")
    print("="*80)
    
    all_thresholds = {}
    all_stats = {}
    
    for target_col, config in target_configs.items():
        
        if target_col not in train_df.columns:
            print(f"\n⚠️  Skipping {target_col}: Not found in data")
            continue
        
        # Check if column has any valid data
        train_valid_count = train_df[target_col].notna().sum()
        if train_valid_count == 0:
            print(f"\n⚠️  Skipping {target_col}: No valid values in train set")
            continue
        
        desc = config['description']
        print(f"\n📊 {desc}")
        print(f"   Column: {target_col}")
        print("-" * 80)
        
        try:
            # ========================================
            # Calculate thresholds from TRAIN only!
            # ========================================
            
            lower_cap, upper_cap = calculate_winsorization_thresholds(
                train_df,
                target_col,
                config['lower_percentile'],
                config['upper_percentile']
            )
            
            all_thresholds[target_col] = {
                'lower_cap': float(lower_cap),
                'upper_cap': float(upper_cap),
                'lower_percentile': float(config['lower_percentile']),
                'upper_percentile': float(config['upper_percentile']),
                'description': desc
            }
            
            print(f"\n   🎯 Thresholds (calculated from TRAIN set only):")
            print(f"      {config['lower_percentile']:.1f}th percentile: {lower_cap:15,.2f}")
            print(f"      {config['upper_percentile']:.1f}th percentile: {upper_cap:15,.2f}")
            
            # ========================================
            # Apply to ALL sets using TRAIN thresholds
            # ========================================
            
            print(f"\n   🔧 Applying winsorization:")
            
            train_df, train_stats = apply_winsorization(
                train_df, target_col, lower_cap, upper_cap, "Train"
            )
            
            val_df, val_stats = apply_winsorization(
                val_df, target_col, lower_cap, upper_cap, "Val"
            )
            
            test_df, test_stats = apply_winsorization(
                test_df, target_col, lower_cap, upper_cap, "Test"
            )
            
            # Store stats
            all_stats[target_col] = {
                'train': train_stats,
                'val': val_stats,
                'test': test_stats,
                'thresholds': all_thresholds[target_col]
            }
            
            print(f"   ✅ {target_col} cleaned successfully (NO data leakage!)")
            
        except Exception as e:
            print(f"   ❌ ERROR processing {target_col}: {e}")
            continue
    
    # ========================================
    # 4. SAVE CLEANED SPLITS
    # ========================================
    
    print("\n" + "="*80)
    print("3️⃣ Saving cleaned splits...")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save to output directory
    train_out = output_path / 'train_data.csv'
    val_out = output_path / 'val_data.csv'
    test_out = output_path / 'test_data.csv'
    
    train_df.to_csv(train_out, index=False)
    val_df.to_csv(val_out, index=False)
    test_df.to_csv(test_out, index=False)
    
    train_size = train_out.stat().st_size / (1024*1024)
    val_size = val_out.stat().st_size / (1024*1024)
    test_size = test_out.stat().st_size / (1024*1024)
    
    print(f"   ✅ Saved train: {train_out}")
    print(f"      Size: {train_size:.1f} MB")
    print(f"   ✅ Saved val:   {val_out}")
    print(f"      Size: {val_size:.1f} MB")
    print(f"   ✅ Saved test:  {test_out}")
    print(f"      Size: {test_size:.1f} MB")
    
    # ========================================
    # 5. SAVE THRESHOLDS (for production use)
    # ========================================
    
    thresholds_file = output_path / 'outlier_thresholds.json'
    
    with open(thresholds_file, 'w') as f:
        json.dump(all_thresholds, f, indent=2)
    
    print(f"\n   💾 Saved thresholds: {thresholds_file}")
    print(f"      (Use these in production for new data)")
    
    # Save detailed stats
    stats_file = output_path / 'outlier_stats.json'
    
    with open(stats_file, 'w') as f:
        json.dump(all_stats, f, indent=2, default=str)
    
    print(f"   💾 Saved detailed stats: {stats_file}")
    
    # ========================================
    # 6. SUMMARY
    # ========================================
    
    print("\n" + "="*80)
    print("📊 OUTLIER HANDLING SUMMARY")
    print("="*80)
    
    if all_thresholds:
        print(f"\nTargets cleaned: {len(all_thresholds)}")
        
        for target_col, threshold_info in all_thresholds.items():
            print(f"\n   {threshold_info['description']}:")
            print(f"      Column: {target_col}")
            print(f"      Caps: [{threshold_info['lower_cap']:,.2f}, {threshold_info['upper_cap']:,.2f}]")
            
            if target_col in all_stats:
                train_pct = all_stats[target_col]['train'].get('pct_capped', 0)
                val_pct = all_stats[target_col]['val'].get('pct_capped', 0)
                test_pct = all_stats[target_col]['test'].get('pct_capped', 0)
                
                print(f"      Capped: Train {train_pct:.2f}%, Val {val_pct:.2f}%, Test {test_pct:.2f}%")
    else:
        print("\n   ⚠️  No targets were cleaned")
    
    print("\n" + "="*80)
    print("✅ OUTLIER HANDLING COMPLETE (NO DATA LEAKAGE!)")
    print("="*80)
    
    print(f"\n🎯 Cleaned data ready for model training:")
    print(f"   Train: {train_out}")
    print(f"   Val:   {val_out}")
    print(f"   Test:  {test_out}")
    
    print(f"\n📝 Thresholds saved for production use:")
    print(f"   {thresholds_file}")
    
    print(f"\n🚀 Next step: Train models!")
    print(f"   python src/models/train_xgboost.py --target revenue")
    
    return train_df, val_df, test_df, all_thresholds


if __name__ == "__main__":
    """
    Main execution: Handle outliers after temporal split
    """
    
    # ========================================
    # CONFIGURATION
    # ========================================
    
    # Directory containing train/val/test splits
    splits_dir = 'data/splits'
    
    # Output directory (same as input by default)
    output_dir = 'data/splits'
    
    # Optional: Custom configurations for specific targets
    # Leave as None to use defaults (1st and 99th percentiles for all)
    custom_configs = None
    
    # Example of custom configs (uncomment to use):
    # custom_configs = {
    #     'target_revenue': {
    #         'lower_percentile': 0.5,  # More aggressive
    #         'upper_percentile': 99.5,
    #         'description': 'Revenue (Next Quarter)'
    #     },
    #     'target_stock_return': {
    #         'lower_percentile': 2.0,  # Less aggressive
    #         'upper_percentile': 98.0,
    #         'description': 'Stock Return (Next Quarter)'
    #     }
    # }
    
    # ========================================
    # RUN OUTLIER HANDLING
    # ========================================
    
    try:
        print("\n" + "="*80)
        print("🚀 STARTING OUTLIER HANDLING PROCESS")
        print("="*80)
        print(f"\nInput directory:  {splits_dir}")
        print(f"Output directory: {output_dir}")
        
        # Handle outliers
        train_df, val_df, test_df, thresholds = handle_outliers_after_split(
            splits_dir=splits_dir,
            output_dir=output_dir,
            target_configs=custom_configs
        )
        
        print("\n" + "="*80)
        print("✅ SUCCESS! Outlier handling completed successfully!")
        print("="*80)
        
    except FileNotFoundError as e:
        print(f"\n❌ FILE NOT FOUND ERROR:")
        print(f"   {e}")
        print(f"\n💡 Make sure you ran these scripts first:")
        print(f"   1. python src/preprocessing/create_targets.py")
        print(f"   2. python src/preprocessing/create_temporal_splits.py")
        
    except ValueError as e:
        print(f"\n❌ VALUE ERROR:")
        print(f"   {e}")
        print(f"\n💡 Check that your data has target columns (target_*)")
        
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR:")
        print(f"   {e}")
        print(f"\n📋 Full traceback:")
        import traceback
        traceback.print_exc()