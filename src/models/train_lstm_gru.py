"""
LSTM + GRU Hybrid Training Script
Combines LSTM and GRU layers for improved temporal modeling
Compares Full Features vs Top 20 Features
"""

import joblib
from tensorflow.keras import layers
from tensorflow import keras
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import mlflow.keras
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
# DATA LOADER
# ============================================================================


class LSTMGRUDataLoader:
    """Load and prepare data with proper scaling"""

    def __init__(self, feature_set, config):
        self.feature_set = feature_set
        self.config = config
        self.data_dir = project_root / 'data' / 'processed'
        self.scaler = StandardScaler()

    def load_data(self, feature_selection_results):
        """Load data and apply scaling"""
        logger.info("=" * 70)
        logger.info(f"LOADING DATA - LSTM+GRU {self.feature_set.upper()}")
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

        # Reshape for LSTM/GRU: (samples, timesteps, features)
        # Using timesteps=1 for now (can be extended later)
        X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
        X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

        # Split train into train/val
        split_idx = int(0.8 * len(X_train))
        X_train_split = X_train[:split_idx]
        X_val = X_train[split_idx:]
        y_train_split = y_train[:split_idx]
        y_val = y_train[split_idx:]

        logger.info(f"\n   Reshaped for recurrent layers:")
        logger.info(
            f"   Shape: {X_train_split.shape} (samples, timesteps, features)")

        # CRITICAL: Fit scaler on training data ONLY
        # For LSTM/GRU with (samples, timesteps, features), we scale features dimension
        logger.info(f"\n🔧 Fitting StandardScaler on TRAIN ONLY...")

        # Reshape to 2D for scaling
        X_train_2d = X_train_split.reshape(-1, X_train_split.shape[-1])
        self.scaler.fit(X_train_2d)

        # Scale and reshape back
        X_train_scaled = self.scaler.transform(
            X_train_split.reshape(-1, X_train_split.shape[-1])
        ).reshape(X_train_split.shape)

        X_val_scaled = self.scaler.transform(
            X_val.reshape(-1, X_val.shape[-1])
        ).reshape(X_val.shape)

        X_test_scaled = self.scaler.transform(
            X_test.reshape(-1, X_test.shape[-1])
        ).reshape(X_test.shape)

        logger.info(
            f"   Scaled range: [{X_train_scaled.min():.2f}, {X_train_scaled.max():.2f}]")

        # Save scaler
        scaler_dir = project_root / 'models' / 'scalers'
        scaler_dir.mkdir(parents=True, exist_ok=True)
        scaler_path = scaler_dir / f'scaler_lstm_gru_{self.feature_set}.pkl'
        joblib.dump(self.scaler, scaler_path)
        logger.info(f"   Saved: {scaler_path.name}")

        return {
            'X_train': X_train_scaled,
            'y_train': y_train_split,
            'X_val': X_val_scaled,
            'y_val': y_val,
            'X_test': X_test_scaled,
            'y_test': y_test,
            'feature_names': feature_names,
            'target_names': feature_selection_results['target_cols']
        }

# ============================================================================
# LSTM + GRU TRAINER
# ============================================================================


class LSTMGRUTrainer:
    """Train LSTM+GRU hybrid model"""

    def __init__(self, data, config, feature_set):
        self.data = data
        self.config = config
        self.feature_set = feature_set
        self.model_config = config['lstm_gru'][f'{feature_set}_features']
        self.model = None

    def build_model(self):
        """Build LSTM + GRU hybrid architecture"""
        timesteps = self.data['X_train'].shape[1]
        n_features = self.data['X_train'].shape[2]
        n_targets = self.data['y_train'].shape[1]

        arch = self.model_config['architecture']
        lstm_units = arch['lstm_units']
        gru_units = arch['gru_units']
        dropout = arch['dropout_rate']
        rec_dropout = arch['recurrent_dropout']

        logger.info("\n" + "=" * 70)
        logger.info("BUILDING LSTM + GRU HYBRID MODEL")
        logger.info("=" * 70)
        logger.info(
            f"   Input: ({timesteps} timesteps, {n_features} features)")
        logger.info(f"   LSTM units: {lstm_units}")
        logger.info(f"   GRU units: {gru_units}")
        logger.info(f"   Output: {n_targets} targets")

        self.model = keras.Sequential([
            layers.Input(shape=(timesteps, n_features)),

            # LSTM layer
            layers.LSTM(
                lstm_units,
                return_sequences=True,
                dropout=dropout,
                recurrent_dropout=rec_dropout
            ),

            # GRU layer
            layers.GRU(
                gru_units,
                dropout=dropout,
                recurrent_dropout=rec_dropout
            ),

            # Output layer
            layers.Dense(n_targets)
        ])

        train_config = self.model_config['training']
        self.model.compile(
            optimizer=keras.optimizers.Adam(
                learning_rate=train_config['learning_rate']),
            loss='mse',
            metrics=['mae', 'mse']
        )

        self.model.summary(print_fn=lambda x: logger.info(x))

    def train(self):
        """Train model with MLflow tracking"""
        # Get experiment config
        exp_key = f"lstm_gru_{self.feature_set}"
        exp_config = self.config['mlflow']['experiments'][exp_key]

        mlflow.set_experiment(exp_config['name'])

        # Generate run name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        run_name = f"LSTM_GRU_{self.feature_set}_{timestamp}"

        with mlflow.start_run(run_name=run_name):
            logger.info("\n" + "=" * 70)
            logger.info(f"TRAINING: {exp_config['name']}")
            logger.info("=" * 70)

            # Log tags
            for key, value in exp_config['tags'].items():
                mlflow.set_tag(key, value)
            for key, value in self.config['mlflow']['common_tags'].items():
                mlflow.set_tag(key, value)

            # Log parameters
            mlflow.log_param("n_features", self.data['X_train'].shape[2])
            mlflow.log_param("n_targets", self.data['y_train'].shape[1])
            mlflow.log_param("timesteps", self.data['X_train'].shape[1])
            mlflow.log_param("train_samples", self.data['X_train'].shape[0])
            mlflow.log_param("val_samples", self.data['X_val'].shape[0])
            mlflow.log_param("test_samples", self.data['X_test'].shape[0])

            for key, value in self.model_config['architecture'].items():
                mlflow.log_param(f"arch_{key}", value)
            for key, value in self.model_config['training'].items():
                mlflow.log_param(key, value)

            # Callbacks
            train_config = self.model_config['training']
            callbacks = [
                keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=train_config['early_stopping_patience'],
                    restore_best_weights=True,
                    verbose=1
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=5,
                    min_lr=1e-6,
                    verbose=1
                )
            ]

            # Train
            logger.info("\nStarting training...")
            history = self.model.fit(
                self.data['X_train'], self.data['y_train'],
                validation_data=(self.data['X_val'], self.data['y_val']),
                epochs=train_config['epochs'],
                batch_size=train_config['batch_size'],
                callbacks=callbacks,
                verbose=1
            )

            # Evaluate
            self._evaluate_and_log()

            # Save model
            model_dir = project_root / 'models' / 'trained'
            model_dir.mkdir(parents=True, exist_ok=True)
            model_path = model_dir / f'lstm_gru_{self.feature_set}.h5'
            self.model.save(model_path)

            mlflow.keras.log_model(self.model, "model")
            logger.info(f"\n✅ Saved: {model_path.name}")

    def _evaluate_and_log(self):
        """Evaluate and log metrics"""
        logger.info("\n" + "=" * 70)
        logger.info("EVALUATION RESULTS")
        logger.info("=" * 70)

        for split_name, X, y in [
            ('train', self.data['X_train'], self.data['y_train']),
            ('val', self.data['X_val'], self.data['y_val']),
            ('test', self.data['X_test'], self.data['y_test'])
        ]:
            y_pred = self.model.predict(X, verbose=0)

            # Overall metrics
            mae = mean_absolute_error(y, y_pred)
            rmse = np.sqrt(mean_squared_error(y, y_pred))
            r2 = r2_score(y, y_pred)

            logger.info(f"\n{split_name.upper()}:")
            logger.info(f"   MAE:  {mae:.4f}")
            logger.info(f"   RMSE: {rmse:.4f}")
            logger.info(f"   R²:   {r2:.4f}")

            mlflow.log_metric(f"{split_name}_mae", mae)
            mlflow.log_metric(f"{split_name}_rmse", rmse)
            mlflow.log_metric(f"{split_name}_r2", r2)

            # Per-target metrics
            logger.info(f"\n   Per-Target R²:")
            for idx, target_name in enumerate(self.data['target_names']):
                target_r2 = r2_score(y[:, idx], y_pred[:, idx])
                target_mae = mean_absolute_error(y[:, idx], y_pred[:, idx])

                logger.info(
                    f"     {target_name}: R²={target_r2:.4f}, MAE={target_mae:.4f}")

                mlflow.log_metric(f"{split_name}_{target_name}_r2", target_r2)
                mlflow.log_metric(
                    f"{split_name}_{target_name}_mae", target_mae)

# ============================================================================
# MAIN
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description='Train LSTM+GRU hybrid model')
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
        logger.info(f"TRAINING LSTM+GRU ON {feature_set.upper()} FEATURES")
        logger.info("=" * 70)

        # Load data
        data_loader = LSTMGRUDataLoader(feature_set, config)
        data = data_loader.load_data(feature_selection)

        # Train model
        trainer = LSTMGRUTrainer(data, config, feature_set)
        trainer.build_model()
        trainer.train()

        logger.info(f"\n✅ Completed: LSTM+GRU {feature_set}")

    logger.info("\n" + "=" * 70)
    logger.info("✅ ALL LSTM+GRU TRAINING COMPLETE!")
    logger.info("=" * 70)
    logger.info("\nView results: mlflow ui --port 5000")
    logger.info("Compare experiments to see full vs top20 performance")


if __name__ == "__main__":
    main()
