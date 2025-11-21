"""
Load Prepared Data with Proper Scaling
Integrates with your existing data pipeline
"""
from features.feature_scaler import prepare_scaled_datasets, get_raw_datasets_for_xgboost
import pandas as pd
import numpy as np
import logging
from pathlib import Path
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Loads data and prepares both scaled (for LSTM) and raw (for XGBoost) versions
    """

    def __init__(self, data_path, feature_cols, target_cols):
        """
        Args:
            data_path: Path to merged_dataset.csv
            feature_cols: List of top 20 feature names
            target_cols: List of 3 target names (no EPS)
        """
        self.data_path = data_path
        self.feature_cols = feature_cols
        self.target_cols = target_cols

        self.df = None
        self.train_df = None
        self.val_df = None
        self.test_df = None

    def load_and_split(self,
                       train_end='2018-12-31',
                       val_end='2022-12-31'):
        """
        Load data and create temporal splits

        Args:
            train_end: End date for training set
            val_end: End date for validation set

        Returns:
            Self for method chaining
        """
        logger.info("=" * 70)
        logger.info("LOADING & TEMPORAL SPLITTING")
        logger.info("=" * 70)

        # Load data
        logger.info(f"Loading data from {self.data_path}")
        self.df = pd.read_csv(self.data_path)
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        self.df = self.df.sort_values('Date').reset_index(drop=True)

        logger.info(f"Total rows: {len(self.df):,}")
        logger.info(
            f"Date range: {self.df['Date'].min()} to {self.df['Date'].max()}")
        logger.info(f"Companies: {self.df['Company'].nunique()}")

        # Temporal split
        train_end = pd.to_datetime(train_end)
        val_end = pd.to_datetime(val_end)

        self.train_df = self.df[self.df['Date'] <= train_end].copy()
        self.val_df = self.df[(self.df['Date'] > train_end) &
                              (self.df['Date'] <= val_end)].copy()
        self.test_df = self.df[self.df['Date'] > val_end].copy()

        logger.info(
            f"\nTrain: {len(self.train_df):,} rows ({len(self.train_df)/len(self.df)*100:.1f}%)")
        logger.info(
            f"       {self.train_df['Date'].min()} to {self.train_df['Date'].max()}")
        logger.info(
            f"Val:   {len(self.val_df):,} rows ({len(self.val_df)/len(self.df)*100:.1f}%)")
        logger.info(
            f"       {self.val_df['Date'].min()} to {self.val_df['Date'].max()}")
        logger.info(
            f"Test:  {len(self.test_df):,} rows ({len(self.test_df)/len(self.df)*100:.1f}%)")
        logger.info(
            f"       {self.test_df['Date'].min()} to {self.test_df['Date'].max()}")

        return self

    def prepare_arrays(self):
        """
        Extract feature and target arrays, dropping NaN targets

        Returns:
            Tuple of (X_train, y_train, X_val, y_val, X_test, y_test)
        """
        logger.info("\n" + "=" * 70)
        logger.info("EXTRACTING FEATURES & TARGETS")
        logger.info("=" * 70)

        logger.info(
            f"Feature columns ({len(self.feature_cols)}): {self.feature_cols[:5]}...")
        logger.info(
            f"Target columns ({len(self.target_cols)}): {self.target_cols}")

        # Drop rows with NaN in targets
        train_clean = self.train_df.dropna(subset=self.target_cols)
        val_clean = self.val_df.dropna(subset=self.target_cols)
        test_clean = self.test_df.dropna(subset=self.target_cols)

        dropped_train = len(self.train_df) - len(train_clean)
        dropped_val = len(self.val_df) - len(val_clean)
        dropped_test = len(self.test_df) - len(test_clean)

        logger.info(f"\nDropped rows with NaN targets:")
        logger.info(
            f"  Train: {dropped_train} ({dropped_train/len(self.train_df)*100:.1f}%)")
        logger.info(
            f"  Val:   {dropped_val} ({dropped_val/len(self.val_df)*100:.1f}%)")
        logger.info(
            f"  Test:  {dropped_test} ({dropped_test/len(self.test_df)*100:.1f}%)")

        # Extract arrays
        X_train = train_clean[self.feature_cols].values
        y_train = train_clean[self.target_cols].values
        X_val = val_clean[self.feature_cols].values
        y_val = val_clean[self.target_cols].values
        X_test = test_clean[self.feature_cols].values
        y_test = test_clean[self.target_cols].values

        logger.info(f"\nFinal shapes:")
        logger.info(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
        logger.info(f"  X_val:   {X_val.shape}, y_val:   {y_val.shape}")
        logger.info(f"  X_test:  {X_test.shape}, y_test:  {y_test.shape}")

        return X_train, y_train, X_val, y_val, X_test, y_test

    def get_data_for_lstm(self, scaler_path='models/scalers/standard_scaler.pkl'):
        """
        Get SCALED data for LSTM/Neural Network models

        Returns:
            Dict with scaled data and targets
        """
        logger.info("\n🔧 Preparing SCALED data for LSTM...")

        X_train, y_train, X_val, y_val, X_test, y_test = self.prepare_arrays()

        # Scale features (fit on train only!)
        X_train_scaled, X_val_scaled, X_test_scaled, scaler = prepare_scaled_datasets(
            X_train, X_val, X_test,
            feature_names=self.feature_cols,
            scaler_save_path=scaler_path
        )

        return {
            'X_train': X_train_scaled,
            'y_train': y_train,
            'X_val': X_val_scaled,
            'y_val': y_val,
            'X_test': X_test_scaled,
            'y_test': y_test,
            'scaler': scaler,
            'feature_cols': self.feature_cols,
            'target_cols': self.target_cols
        }

    def get_data_for_xgboost(self):
        """
        Get RAW (unscaled) data for XGBoost/Tree models

        Returns:
            Dict with raw data and targets
        """
        logger.info("\n🌳 Preparing RAW data for XGBoost...")

        X_train, y_train, X_val, y_val, X_test, y_test = self.prepare_arrays()

        # XGBoost doesn't need scaling
        X_train_raw, X_val_raw, X_test_raw = get_raw_datasets_for_xgboost(
            X_train, X_val, X_test
        )

        return {
            'X_train': X_train_raw,
            'y_train': y_train,
            'X_val': X_val_raw,
            'y_val': y_val,
            'X_test': X_test_raw,
            'y_test': y_test,
            'feature_cols': self.feature_cols,
            'target_cols': self.target_cols
        }


# Convenience function
def load_data_for_model(model_type, data_path, feature_cols, target_cols):
    """
    Load data appropriate for specific model type

    Args:
        model_type: 'lstm', 'xgboost', 'linear', 'rf'
        data_path: Path to merged_dataset.csv
        feature_cols: List of feature names
        target_cols: List of target names

    Returns:
        Data dictionary ready for training
    """
    loader = DataLoader(data_path, feature_cols, target_cols)
    loader.load_and_split()

    if model_type in ['lstm', 'gru', 'linear', 'rf']:
        return loader.get_data_for_lstm()
    elif model_type == 'xgboost':
        return loader.get_data_for_xgboost()
    else:
        raise ValueError(f"Unknown model type: {model_type}")
