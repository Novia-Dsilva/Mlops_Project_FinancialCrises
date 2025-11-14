"""
XGBoost Training Script
Trains SEPARATE models per target (fixes negative R²)
Compares Full Features vs Top 20 Features
NO feature scaling (tree-based models don't need it)
"""

import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import mlflow.xgboost
import mlflow
import logging
from datetime import datetime
import argparse
import yaml
import pickle
import numpy as np
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================


def load_config():
    """Load params.yaml"""
    config_path = project_root / 'config' / 'params.yaml'
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_feature_selection_results():
    """Load feature selection results"""
    results_path = project_root / 'data' / \
        'processed' / 'feature_selection_results.pkl'
    with open(results_path, 'rb') as f:
        return pickle.load(f)

# ============================================================================
# DATA LOADER (NO SCALING FOR XGBOOST!)
# ============================================================================


class XGBoostDataLoader:
    """Load RAW (unscaled) data for XGBoost"""

    def __init__(self, feature_set, config):
        self.feature_set = feature_set
        self.config = config
        self.data_dir = project_root / 'data' / 'processed'

    def load_data(self, feature_selection_results):
        """Load data with specified feature set - NO SCALING"""
        logger.info("=" * 70)
        logger.info(f"LOADING DATA - XGBoost {self.feature_set.upper()}")
        logger.info("=" * 70)

        # Load full datasets
        X_train_full = np.load(self.data_dir / 'X_train.npy')
        X_test_full = np.load(self.data_dir / 'X_test.npy')
        y_train = np.load(self.data_dir / 'y_train.npy')
        y_test = np.load(self.data_dir / 'y_test.npy')

        # Select features
        if self.feature_set == 'full':
            X_train = X_train_full
            X_test = X_test_full
            feature_names = feature_selection_results['all_features']
            logger.info(f"   Using ALL features: {X_train.shape[1]}")
        elif self.feature_set == 'top20':
            indices = feature_selection_results['top_feature_indices']
            X_train = X_train_full[:, indices]
            X_test = X_test_full[:, indices]
            feature_names = feature_selection_results['top_features']
            logger.info(f"   Using TOP 20 features: {X_train.shape[1]}")

        logger.info(f"   Train samples: {X_train.shape[0]:,}")
        logger.info(f"   Test samples: {X_test.shape[0]:,}")
        logger.info(f"   NO SCALING (tree-based model)")

        # Split train into train/val
        split_idx = int(0.8 * len(X_train))
        X_train_split = X_train[:split_idx]
        X_val = X_train[split_idx:]
        y_train_split = y_train[:split_idx]
        y_val = y_train[split_idx:]

        return {
            'X_train': X_train_split,
            'y_train': y_train_split,
            'X_val': X_val,
            'y_val': y_val,
            'X_test': X_test,
            'y_test': y_test,
            'feature_names': feature_names,
            'target_names': feature_selection_results['target_cols']
        }

# ============================================================================
# XGBOOST TRAINER (SEPARATE MODELS PER TARGET)
# ============================================================================


class XGBoostTrainer:
    """Train separate XGBoost model for each target"""

    def __init__(self, data, config, feature_set):
        self.data = data
        self.config = config
        self.feature_set = feature_set
        self.model_config = config['xgboost'][f'{feature_set}_features']['model']
        self.models = {}  # One model per target

    def train_all_targets(self):
        """Train separate model for each target with MLflow tracking"""
        # Get experiment config
        exp_key = f"xgboost_{self.feature_set}"
        exp_config = self.config['mlflow']['experiments'][exp_key]

        mlflow.set_experiment(exp_config['name'])

        logger.info("\n" + "=" * 70)
        logger.info(f"TRAINING: {exp_config['name']}")
        logger.info("=" * 70)
        logger.info(f"Strategy: SEPARATE models per target (fixes negative R²)")

        # Train one model per target
        for target_idx, target_name in enumerate(self.data['target_names']):
            self._train_single_target(target_idx, target_name, exp_config)

        logger.info(f"\n✅ Trained {len(self.models)} XGBoost models")

    def _train_single_target(self, target_idx, target_name, exp_config):
        """Train XGBoost for a single target"""
        logger.info(f"\n{'─'*70}")
        logger.info(
            f"Target {target_idx + 1}/{len(self.data['target_names'])}: {target_name}")
        logger.info(f"{'─'*70}")

        # Extract single target
        y_train = self.data['y_train'][:, target_idx]
        y_val = self.data['y_val'][:, target_idx]
        y_test = self.data['y_test'][:, target_idx]

        # Generate run name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        run_name = f"XGBoost_{self.feature_set}_{target_name}_{timestamp}"

        with mlflow.start_run(run_name=run_name):
            # Log tags
            for key, value in exp_config['tags'].items():
                mlflow.set_tag(key, value)
            for key, value in self.config['mlflow']['common_tags'].items():
                mlflow.set_tag(key, value)
            mlflow.set_tag('target', target_name)

            # Log parameters
            mlflow.log_param("n_features", self.data['X_train'].shape[1])
            mlflow.log_param("target", target_name)
            mlflow.log_param("train_samples", len(y_train))
            mlflow.log_param("val_samples", len(y_val))
            mlflow.log_param("test_samples", len(y_test))

            for key, value in self.model_config.items():
                mlflow.log_param(key, value)

            # Initialize model
            model = XGBRegressor(**self.model_config)

            # Train
            logger.info("Training...")
            model.fit(
                self.data['X_train'], y_train,
                eval_set=[(self.data['X_val'], y_val)],
                verbose=False
            )

            logger.info(f"Best iteration: {model.best_iteration}")

            # Evaluate
            self._evaluate_and_log(model, y_train, y_val, y_test, target_name)

            # Save model
            mlflow.xgboost.log_model(model, "model")

            model_dir = project_root / 'models' / 'trained'
            model_dir.mkdir(parents=True, exist_ok=True)
            model_path = model_dir / \
                f'xgboost_{self.feature_set}_{target_name}.pkl'
            joblib.dump(model, model_path)

            self.models[target_name] = model
            logger.info(f"✅ Saved: {model_path.name}")

    def _evaluate_and_log(self, model, y_train, y_val, y_test, target_name):
        """Evaluate model and log metrics"""
        logger.info("\nEvaluation:")

        for split_name, X, y_true in [
            ('train', self.data['X_train'], y_train),
            ('val', self.data['X_val'], y_val),
            ('test', self.data['X_test'], y_test)
        ]:
            y_pred = model.predict(X)

            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            r2 = r2_score(y_true, y_pred)

            logger.info(
                f"  {split_name.upper()}: MAE={mae:.4f}, RMSE={rmse:.4f}, R²={r2:.4f}")

            mlflow.log_metric(f"{split_name}_mae", mae)
            mlflow.log_metric(f"{split_name}_rmse", rmse)
            mlflow.log_metric(f"{split_name}_r2", r2)
            mlflow.log_metric(f"{split_name}_{target_name}_r2", r2)

    def get_feature_importance(self, target_name, top_n=10):
        """Get feature importance for a target"""
        import pandas as pd

        model = self.models[target_name]
        importance = model.feature_importances_

        df_importance = pd.DataFrame({
            'feature': self.data['feature_names'][:len(importance)],
            'importance': importance
        }).sort_values('importance', ascending=False)

        logger.info(f"\nTop {top_n} features for {target_name}:")
        logger.info(df_importance.head(top_n).to_string(index=False))

        return df_importance

# ============================================================================
# MAIN
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description='Train XGBoost models')
    parser.add_argument(
        '--feature-set',
        type=str,
        choices=['full', 'top20', 'both'],
        default='both',
        help='Feature set: full, top20, or both (default: both)'
    )
    args = parser.parse_args()

    # Load configurations
    config = load_config()
    feature_selection = load_feature_selection_results()

    # Determine which feature sets to train
    if args.feature_set == 'both':
        feature_sets = ['full', 'top20']
    else:
        feature_sets = [args.feature_set]

    # Train on each feature set
    for feature_set in feature_sets:
        logger.info("\n" + "=" * 70)
        logger.info(f"TRAINING XGBOOST ON {feature_set.upper()} FEATURES")
        logger.info("=" * 70)

        # Load data
        data_loader = XGBoostDataLoader(feature_set, config)
        data = data_loader.load_data(feature_selection)

        # Train models
        trainer = XGBoostTrainer(data, config, feature_set)
        trainer.train_all_targets()

        # Show feature importance
        logger.info("\n" + "=" * 70)
        logger.info("FEATURE IMPORTANCE")
        logger.info("=" * 70)
        for target_name in data['target_names']:
            trainer.get_feature_importance(target_name, top_n=10)

        logger.info(f"\n✅ Completed: XGBoost {feature_set}")

    logger.info("\n" + "=" * 70)
    logger.info("✅ ALL XGBOOST TRAINING COMPLETE!")
    logger.info("=" * 70)
    logger.info("\nView results: mlflow ui --port 5000")
    logger.info("Compare experiments to see full vs top20 performance")


if __name__ == "__main__":
    main()