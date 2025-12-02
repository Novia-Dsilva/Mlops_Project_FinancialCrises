"""
Utility functions for data splitting
"""

import pandas as pd
import numpy as np
from typing import Tuple, List

def get_feature_target_split(
    df: pd.DataFrame,
    target_col: str = 'target_revenue',
    exclude_cols: List[str] = None,
    encode_categoricals: bool = True
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Split dataframe into features (X) and target (y).
    """
    if exclude_cols is None:
        exclude_cols = ['Date', 'Year', 'Quarter']
    
    print(f"\n{'='*80}")
    print(f"🎯 FEATURE-TARGET SPLIT FOR: {target_col}")
    print(f"{'='*80}")
    
    # Base exclusions
    base_exclusions = exclude_cols.copy()
    
    # Exclude current value being predicted
    print(f"\n🔒 LEAKAGE PREVENTION:")
    
    target_to_current = {
        'target_revenue': ['Revenue'],
        'target_eps': ['EPS', 'EPS_calculated'],
        'target_debt_equity': ['Debt_to_Equity'],
        'target_profit_margin': ['net_margin_q', 'Profit_Margin'],
        'target_stock_return': ['stock_q_return', 'Stock_Price', 'Close']
    }
    
    if target_col in target_to_current:
        current_cols = target_to_current[target_col]
        existing = [c for c in current_cols if c in df.columns]
        if existing:
            print(f"   ❌ Excluding current value: {existing}")
            base_exclusions.extend(existing)
    
    # Exclude rolling features
    rolling_patterns = {
        'target_revenue': ['rolling4q_avg_log_revenue'],
        'target_profit_margin': ['rolling4q_avg_net_margin'],
        'target_debt_equity': ['rolling4q_avg_debt_to_assets'],
        'target_stock_return': ['rolling4q_avg_stock_q_return']
    }
    
    if target_col in rolling_patterns:
        rolling_cols = [c for c in df.columns if any(p in c for p in rolling_patterns[target_col])]
        if rolling_cols:
            print(f"   ❌ Excluding {len(rolling_cols)} rolling features")
            base_exclusions.extend(rolling_cols)
    
    # Keep lagged features
    lagged_features = [c for c in df.columns if '_t_' in c or 'Δ' in c]
    if lagged_features:
        print(f"   ✅ KEEPING {len(lagged_features)} lagged/delta features")
    
    # Exclude other targets
    other_targets = [c for c in df.columns if c.startswith('target_') and c != target_col]
    if other_targets:
        print(f"   ❌ Excluding {len(other_targets)} other targets")
        base_exclusions.extend(other_targets)
    
    # Get features
    feature_cols = [c for c in df.columns if c != target_col and c not in base_exclusions]
    
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    
    print(f"\n📊 Initial feature set: {len(feature_cols)} columns")
    
    # Handle categoricals
    categorical_cols = X.select_dtypes(include=['object', 'string']).columns.tolist()
    
    if categorical_cols and encode_categoricals:
        print(f"\n🔄 ENCODING {len(categorical_cols)} categoricals")
        for col in categorical_cols:
            print(f"   - {col}: {X[col].nunique()} unique")
        
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True, dtype=int)
        print(f"   ✅ After encoding: {X.shape[1]} features")
    
    # Check non-numeric
    non_numeric = X.select_dtypes(include=['object', 'string']).columns.tolist()
    if non_numeric:
        raise ValueError(f"Non-numeric columns: {non_numeric}")
    
    # Replace inf
    X = X.replace([np.inf, -np.inf], np.nan)
    
    print(f"\n{'='*80}")
    print(f"Features (X): {X.shape[1]} columns × {X.shape[0]} rows")
    print(f"Target (y):   {y.notna().sum():,} valid values")
    print(f"{'='*80}\n")
    
    return X, y

def drop_nan_targets(
    X: pd.DataFrame,
    y: pd.Series,
    split_name: str = "Split"
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Drop rows where the target is NaN (these rows are unlabeled for this task).

    Used so that:
      - Models only train on rows with a known next-quarter target.
      - Metrics (RMSE/R²/MAE) are computed only where y_true exists.

    We STILL keep these rows in the raw CSVs for inference, but
    they are excluded from training/evaluation.
    """
    mask = y.notna()
    dropped = (~mask).sum()
    if dropped > 0:
        print(f"   {split_name}: Dropped {dropped} rows with NaN targets")
    return X.loc[mask], y.loc[mask]


def drop_nan_targets(
    X: pd.DataFrame,
    y: pd.Series,
    split_name: str = "Split"
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Drop rows where the target is NaN (these rows are unlabeled for this task).

    Used so that:
      - Models only train on rows with a known next-quarter target.
      - Metrics (RMSE/R²/MAE) are computed only where y_true exists.

    We STILL keep these rows in the raw CSVs for inference, but
    they are excluded from training/evaluation.
    """
    mask = y.notna()
    dropped = (~mask).sum()
    if dropped > 0:
        print(f"   {split_name}: Dropped {dropped} rows with NaN targets")
    return X.loc[mask], y.loc[mask]
