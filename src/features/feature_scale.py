"""
Feature Scaling Module
Handles proper scaling with no data leakage (fit on train only)
"""
import numpy as np
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)


class FeatureScaler:
    """
    Manages feature scaling for different model types.

    Key Rules:
    - LSTM/Neural Networks: MUST scale (StandardScaler)
    - XGBoost/Tree models: NO scaling (raw features)
    - Linear models: SHOULD scale
    - Scaler fitted ONLY on training data
    """

    def __init__(self, scaler_type='standard'):
        """
        Args:
            scaler_type: 'standard', 'minmax', or 'robust'
        """
        if scaler_type == 'standard':
            self.scaler = StandardScaler()
        else:
            from sklearn.preprocessing import MinMaxScaler, RobustScaler
            self.scaler = MinMaxScaler() if scaler_type == 'minmax' else RobustScaler()

        self.scaler_type = scaler_type
        self.is_fitted = False
        self.feature_names = None

    def fit(self, X_train, feature_names=None):
        """
        Fit scaler on training data ONLY

        Args:
            X_train: Training features (n_samples, n_features)
            feature_names: List of feature names for logging

        Returns:
            Self for method chaining
        """
        logger.info(f"Fitting {self.scaler_type} scaler on training data...")
        logger.info(f"  Training samples: {X_train.shape[0]}")
        logger.info(f"  Features: {X_train.shape[1]}")

        self.scaler.fit(X_train)
        self.is_fitted = True
        self.feature_names = feature_names

        if hasattr(self.scaler, 'mean_'):
            logger.info(f"  Feature means (first 5): {self.scaler.mean_[:5]}")
            logger.info(f"  Feature stds (first 5): {self.scaler.scale_[:5]}")

        return self

    def transform(self, X):
        """
        Transform features using fitted scaler

        Args:
            X: Features to transform

        Returns:
            Scaled features (same shape as input)
        """
        if not self.is_fitted:
            raise ValueError("Scaler not fitted! Call fit() first.")

        X_scaled = self.scaler.transform(X)

        logger.debug(f"Transformed {X.shape[0]} samples")
        logger.debug(f"  Input range: [{X.min():.2f}, {X.max():.2f}]")
        logger.debug(
            f"  Output range: [{X_scaled.min():.2f}, {X_scaled.max():.2f}]")

        return X_scaled

    def fit_transform(self, X_train, feature_names=None):
        """Convenience: fit and transform in one call"""
        return self.fit(X_train, feature_names).transform(X_train)

    def inverse_transform(self, X_scaled):
        """Convert scaled features back to original scale"""
        if not self.is_fitted:
            raise ValueError("Scaler not fitted!")
        return self.scaler.inverse_transform(X_scaled)

    def save(self, path):
        """Save fitted scaler for inference"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        save_dict = {
            'scaler': self.scaler,
            'scaler_type': self.scaler_type,
            'feature_names': self.feature_names,
            'is_fitted': self.is_fitted
        }

        joblib.dump(save_dict, path)
        logger.info(f"Saved scaler to {path}")

    @classmethod
    def load(cls, path):
        """Load fitted scaler from disk"""
        save_dict = joblib.load(path)

        instance = cls(scaler_type=save_dict['scaler_type'])
        instance.scaler = save_dict['scaler']
        instance.is_fitted = save_dict['is_fitted']
        instance.feature_names = save_dict.get('feature_names')

        logger.info(f"Loaded scaler from {path}")
        return instance


def prepare_scaled_datasets(X_train, X_val, X_test,
                            feature_names=None,
                            scaler_save_path='models/scalers/standard_scaler.pkl'):
    """
    Prepare scaled datasets for LSTM/Linear models

    Args:
        X_train, X_val, X_test: Raw feature arrays
        feature_names: Optional list of feature names
        scaler_save_path: Where to save fitted scaler

    Returns:
        Tuple of (X_train_scaled, X_val_scaled, X_test_scaled, scaler)
    """
    logger.info("=" * 70)
    logger.info("PREPARING SCALED DATASETS")
    logger.info("=" * 70)

    # Initialize and fit scaler
    scaler = FeatureScaler(scaler_type='standard')
    scaler.fit(X_train, feature_names=feature_names)

    # Transform all datasets
    logger.info("Transforming datasets...")
    X_train_scaled = scaler.transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Save scaler
    scaler.save(scaler_save_path)

    # Verify no leakage
    logger.info("\nVerifying no data leakage:")
    train_mean = X_train.mean(axis=0)[:3]
    scaler_mean = scaler.scaler.mean_[:3]
    logger.info(f"  Train mean (first 3): {train_mean}")
    logger.info(f"  Scaler mean (first 3): {scaler_mean}")
    logger.info(f"  Match: {np.allclose(train_mean, scaler_mean)}")

    logger.info("=" * 70)

    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


def get_raw_datasets_for_xgboost(X_train, X_val, X_test):
    """
    Return raw (unscaled) datasets for XGBoost

    XGBoost doesn't need scaling, so just return copies
    """
    logger.info("Using RAW (unscaled) features for XGBoost")
    return X_train.copy(), X_val.copy(), X_test.copy()
