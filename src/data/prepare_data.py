import pandas as pd
import numpy as np
from typing import Tuple, List


def temporal_train_test_split(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
    date_column: str = "Date"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data chronologically into train, validation, test sets.
    """
    assert abs(train_ratio + val_ratio + test_ratio -
               1.0) < 1e-3, "Ratios must sum to 1"

    df = df.sort_values(date_column).reset_index(drop=True)
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    print(
        f"Train: {len(train_df)} rows, Val: {len(val_df)} rows, Test: {len(test_df)} rows")
    return train_df, val_df, test_df


def get_feature_target_split(
    df: pd.DataFrame,
    target_col: str,
    exclude_cols: List[str] = None,
    encode_categoricals: bool = True
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Return X, y after excluding leakage features and encoding categoricals.
    """
    if exclude_cols is None:
        exclude_cols = ["Date", "Company"]

    # Exclude trivial features
    price_leakage = [c for c in ["Close", "Stock_Price",
                                 "Stock_MA20", "Stock_MA50", "Stock_MA200"] if c in df.columns]
    exclude_cols += price_leakage

    feature_cols = [
        c for c in df.columns if c not in exclude_cols and c != target_col]
    X = df[feature_cols].copy()
    y = df[target_col].copy()

    # Encode categoricals
    cat_cols = X.select_dtypes(include=["object", "string"]).columns.tolist()
    if encode_categoricals and cat_cols:
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True, dtype=int)

    return X, y


def build_company_sequences(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    seq_len: int = 4
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build company-aware input sequences (sliding window per company).
    """
    X_seqs = []
    y_seqs = []

    for company, group in df.groupby("Company"):
        group = group.sort_values("Date")
        X_arr = group[feature_cols].values
        y_arr = group[target_col].values

        for i in range(len(group) - seq_len):
            X_seqs.append(X_arr[i:i+seq_len])
            y_seqs.append(y_arr[i+seq_len])

    X_seqs = np.array(X_seqs)
    y_seqs = np.array(y_seqs)
    print(f"Built sequences: {X_seqs.shape}, Targets: {y_seqs.shape}")
    return X_seqs, y_seqs
