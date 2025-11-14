"""
TRUE LSTM with Sequences - Daily Data
Uses last N days to predict next day/quarter targets
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
import pandas as pd
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
    if results_path.exists():
        with open(results_path, 'rb') as f:
            return pickle.load(f)
    return None

# ============================================================================
# SEQUENCE DATA LOADER
# ============================================================================


class LSTMSequenceDataLoader:
    """Create sequences from daily data for LSTM"""

    def __init__(self, feature_set, lookback_days, config):
        self.feature_set = feature_set
        self.lookback_days = lookback_days  # e.g., 20, 60, 90 days
        self.config = config
        self.scaler = StandardScaler()

    def load_data(self):
        """Load daily data from CSV"""
        logger.info("=" * 70)
        logger.info(f"LOADING DAILY DATA - LSTM Sequences")
        logger.info(f"Lookback window: {self.lookback_days} days")
        logger.info("=" * 70)

        # Load CSV
        data_path = project_root / 'data' / 'features' / \
            'merged_features_clean_with_anomaly_flags_with_drift_flags.csv'
        df = pd.read_csv(data_path)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values(['Company', 'Date']).reset_index(drop=True)

        logger.info(f"Loaded: {len(df):,} rows, {len(df.columns)} columns")
        logger.info(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
        logger.info(f"Companies: {df['Company'].nunique()}")

        return df

    def prepare_features_targets(self, df, feature_selection_results):
        """Identify feature and target columns"""
        logger.info("\n" + "=" * 70)
        logger.info("PREPARING FEATURES & TARGETS")
        logger.info("=" * 70)

        # Exclude columns
        exclude_cols = [
            'Date', 'Company', 'Company_Name', 'Sector', 'Year',
            'Quarter', 'Ticker', 'Close', 'Volume',
            # Exclude extreme flags and outlier flags to reduce noise
            'Debt_to_Equity_Outlier_Flag', 'Stock_Return_1D_Outlier_Flag',
            'Stock_Return_22D_Outlier_Flag', 'Stock_Volatility_22D_Outlier_Flag',
            'Profit_Margin_Outlier_Flag', 'ROA_Outlier_Flag', 'ROE_Outlier_Flag',
            'Revenue_Growth_YoY_Outlier_Flag', 'VIX_Outlier_Flag',
            'SP500_Return_1D_Outlier_Flag', 'Profit_Margin_Extreme_Flag',
            'ROE_Extreme_Flag', 'Debt_to_Equity_Extreme_Flag',
            'Stock_Price_Jump_Flag', 'Close_Jump_Flag', 'Revenue_Jump_Flag',
            'Feature_Drift_Flag'
        ]

        # Define targets - what we want to predict
        target_cols = [
            'Stock_Return_1D',  # Next day return
            'Debt_to_Equity',   # Current leverage
            'Revenue'           # Current revenue
        ]

        logger.info(f"Target columns: {target_cols}")

        # Feature columns
        all_cols = set(df.columns)
        exclude_set = set(exclude_cols + target_cols)

        if self.feature_set == 'top20' and feature_selection_results:
            # Use pre-selected top 20 features
            feature_cols = feature_selection_results['top_features']
            logger.info(f"Using TOP 20 SELECTED features")
        elif self.feature_set == 'top20':
            # If top20 requested but no selection file, use most important manual selection
            logger.warning(
                "Feature selection file not found, using manual top 20")
            feature_cols = [
                'Stock_Price', 'VIX', 'SP500_Close', 'Stock_Volatility_22D',
                'Revenue', 'Net_Income', 'Total_Assets', 'Debt_to_Equity',
                'ROE', 'Profit_Margin', 'GDP', 'CPI', 'Unemployment_Rate',
                'Federal_Funds_Rate', 'Oil_Price', 'Treasury_10Y_Yield',
                'SP500_Return_1D', 'Stock_MA20', 'VIX_MA22', 'Revenue_Growth_YoY'
            ]
        else:
            # Use ALL features
            feature_cols = sorted(list(all_cols - exclude_set))
            logger.info(f"Using ALL features")

        # Ensure all columns exist
        feature_cols = [f for f in feature_cols if f in df.columns]
        target_cols = [t for t in target_cols if t in df.columns]

        logger.info(
            f"Final: {len(feature_cols)} features, {len(target_cols)} targets")
        logger.info(f"Features: {feature_cols[:10]}..." if len(
            feature_cols) > 10 else f"Features: {feature_cols}")

        return feature_cols, target_cols

    def create_sequences(self, df, feature_cols, target_cols):
        """Create sequences: [last N days] → predict [next day]"""
        logger.info("\n" + "=" * 70)
        logger.info(
            f"CREATING SEQUENCES (Lookback: {self.lookback_days} days)")
        logger.info("=" * 70)

        # Convert to numeric and handle NaN
        for col in feature_cols + target_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        X_sequences = []
        y_sequences = []
        dates = []
        companies = []

        # Create sequences per company
        for company in df['Company'].unique():
            company_df = df[df['Company'] == company].sort_values(
                'Date').reset_index(drop=True)

            # Need at least lookback_days + 1 for one sequence
            if len(company_df) < self.lookback_days + 1:
                logger.warning(f"Skipping {company} - insufficient data")
                continue

            # Create rolling windows
            for i in range(len(company_df) - self.lookback_days):
                # Input: last N days of features
                X_window = company_df.iloc[i:i +
                                           self.lookback_days][feature_cols].values

                # Target: next day's values
                y_target = company_df.iloc[i +
                                           self.lookback_days][target_cols].values

                # Skip if NaN (now safe to check)
                if np.isnan(X_window).any() or np.isnan(y_target).any():
                    continue

                X_sequences.append(X_window)
                y_sequences.append(y_target)
                dates.append(company_df.iloc[i+self.lookback_days]['Date'])
                companies.append(company)

        # (samples, lookback_days, features)
        X_sequences = np.array(X_sequences)
        y_sequences = np.array(y_sequences)  # (samples, targets)

        logger.info(f"Created {len(X_sequences):,} sequences")
        logger.info(
            f"X shape: {X_sequences.shape} (samples, timesteps, features)")
        logger.info(f"y shape: {y_sequences.shape} (samples, targets)")

        return X_sequences, y_sequences, dates, companies

    def temporal_split(self, X, y, dates, companies):
        """Split by date (2005-2018 train, 2019-2022 val, 2023+ test)"""
        logger.info("\n" + "=" * 70)
        logger.info("TEMPORAL SPLIT")
        logger.info("=" * 70)

        dates = pd.to_datetime(dates)

        train_mask = dates <= '2018-12-31'
        val_mask = (dates > '2018-12-31') & (dates <= '2022-12-31')
        test_mask = dates > '2022-12-31'

        X_train = X[train_mask]
        y_train = y[train_mask]
        X_val = X[val_mask]
        y_val = y[val_mask]
        X_test = X[test_mask]
        y_test = y[test_mask]

        logger.info(f"Train: {len(X_train):,} sequences (2005-2018)")
        logger.info(f"Val:   {len(X_val):,} sequences (2019-2022)")
        logger.info(f"Test:  {len(X_test):,} sequences (2023-2025)")

        return X_train, y_train, X_val, y_val, X_test, y_test

    def scale_sequences(self, X_train, X_val, X_test):
        """Scale features - FIT ON TRAIN ONLY"""
        logger.info("\n" + "=" * 70)
        logger.info("SCALING SEQUENCES")
        logger.info("=" * 70)

        # Reshape to 2D for scaling
        n_samples_train, n_timesteps, n_features = X_train.shape
        X_train_2d = X_train.reshape(-1, n_features)

        # Fit scaler on train
        logger.info("Fitting StandardScaler on TRAIN sequences only...")
        self.scaler.fit(X_train_2d)

        # Transform all sets
        X_train_scaled = self.scaler.transform(
            X_train_2d).reshape(X_train.shape)
        X_val_scaled = self.scaler.transform(
            X_val.reshape(-1, n_features)).reshape(X_val.shape)
        X_test_scaled = self.scaler.transform(
            X_test.reshape(-1, n_features)).reshape(X_test.shape)

        logger.info(
            f"Scaled range: [{X_train_scaled.min():.2f}, {X_train_scaled.max():.2f}]")

        # Save scaler
        scaler_dir = project_root / 'models' / 'scalers'
        scaler_dir.mkdir(parents=True, exist_ok=True)
        scaler_path = scaler_dir / \
            f'scaler_lstm_seq_{self.lookback_days}d_{self.feature_set}.pkl'
        joblib.dump(self.scaler, scaler_path)
        logger.info(f"Saved: {scaler_path.name}")

        return X_train_scaled, X_val_scaled, X_test_scaled

# ============================================================================
# TRUE LSTM TRAINER
# ============================================================================


class TrueLSTMTrainer:
    """Train LSTM with recurrent layers on sequences"""

    def __init__(self, data, config, feature_set, lookback_days):
        self.data = data
        self.config = config
        self.feature_set = feature_set
        self.lookback_days = lookback_days
        self.model = None

    def build_model(self, lstm_units=[64, 32], dropout=0.3):
        """Build true LSTM with recurrent layers"""
        n_timesteps = self.data['X_train'].shape[1]
        n_features = self.data['X_train'].shape[2]
        n_targets = self.data['y_train'].shape[1]

        logger.info("\n" + "=" * 70)
        logger.info("BUILDING TRUE LSTM MODEL")
        logger.info("=" * 70)
        logger.info(f"Input: ({n_timesteps} timesteps, {n_features} features)")
        logger.info(f"LSTM units: {lstm_units}")
        logger.info(f"Output: {n_targets} targets")

        self.model = keras.Sequential([
            layers.Input(shape=(n_timesteps, n_features)),

            # First LSTM layer (returns sequences for stacking)
            layers.LSTM(
                lstm_units[0],
                return_sequences=True,
                dropout=dropout,
                recurrent_dropout=0.2
            ),

            # Second LSTM layer (returns single output)
            layers.LSTM(
                lstm_units[1],
                dropout=dropout,
                recurrent_dropout=0.2
            ),

            # Dense layer for output
            layers.Dense(n_targets)
        ])

        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae', 'mse']
        )

        self.model.summary(print_fn=lambda x: logger.info(x))

    def train(self):
        """Train with MLflow tracking"""
        exp_name = f"LSTM_Sequences_{self.lookback_days}d_{self.feature_set}"
        mlflow.set_experiment(exp_name)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        run_name = f"LSTM_Seq{self.lookback_days}d_{self.feature_set}_{timestamp}"

        with mlflow.start_run(run_name=run_name):
            logger.info("\n" + "=" * 70)
            logger.info(f"TRAINING: {exp_name}")
            logger.info("=" * 70)

            # Log parameters
            mlflow.log_param("model_type", "LSTM_Sequences")
            mlflow.log_param("lookback_days", self.lookback_days)
            mlflow.log_param("feature_set", self.feature_set)
            mlflow.log_param("n_features", self.data['X_train'].shape[2])
            mlflow.log_param("n_targets", self.data['y_train'].shape[1])
            mlflow.log_param("train_sequences", self.data['X_train'].shape[0])
            mlflow.log_param("val_sequences", self.data['X_val'].shape[0])
            mlflow.log_param("test_sequences", self.data['X_test'].shape[0])

            # Callbacks
            callbacks = [
                keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=10,
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
                epochs=50,
                batch_size=64,  # Larger batch for daily data
                callbacks=callbacks,
                verbose=1
            )

            # Evaluate
            self._evaluate_and_log()

            # Save model
            model_dir = project_root / 'models' / 'trained'
            model_dir.mkdir(parents=True, exist_ok=True)
            model_path = model_dir / \
                f'lstm_seq{self.lookback_days}d_{self.feature_set}.h5'
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

# ============================================================================
# MAIN
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description='Train TRUE LSTM with sequences')
    parser.add_argument(
        '--lookback-days',
        type=int,
        default=20,
        help='Number of days to look back (default: 20)'
    )
    parser.add_argument(
        '--feature-set',
        type=str,
        choices=['full', 'top20'],
        default='full',
        help='Feature set: full or top20'
    )
    args = parser.parse_args()

    # Load config
    config = load_config()
    feature_selection = load_feature_selection_results()

    logger.info("\n" + "=" * 70)
    logger.info(f"TRUE LSTM with {args.lookback_days}-DAY SEQUENCES")
    logger.info("=" * 70)

    # Load data
    data_loader = LSTMSequenceDataLoader(
        args.feature_set, args.lookback_days, config)
    df = data_loader.load_data()

    # Prepare features and targets
    feature_cols, target_cols = data_loader.prepare_features_targets(
        df, feature_selection)

    # Create sequences
    X, y, dates, companies = data_loader.create_sequences(
        df, feature_cols, target_cols)

    # Temporal split
    X_train, y_train, X_val, y_val, X_test, y_test = data_loader.temporal_split(
        X, y, dates, companies
    )

    # Scale sequences
    X_train_scaled, X_val_scaled, X_test_scaled = data_loader.scale_sequences(
        X_train, X_val, X_test
    )

    # Package data
    data = {
        'X_train': X_train_scaled,
        'y_train': y_train,
        'X_val': X_val_scaled,
        'y_val': y_val,
        'X_test': X_test_scaled,
        'y_test': y_test
    }

    # Train model
    trainer = TrueLSTMTrainer(
        data, config, args.feature_set, args.lookback_days)
    trainer.build_model(lstm_units=[64, 32], dropout=0.3)
    trainer.train()

    logger.info("\n" + "=" * 70)
    logger.info("✅ TRUE LSTM TRAINING COMPLETE!")
    logger.info("=" * 70)
    logger.info("\nView results: mlflow ui --port 5000")


if __name__ == "__main__":
    main()
