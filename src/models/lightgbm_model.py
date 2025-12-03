"""
src/models/lightgbm_model.py
Train LightGBM models - Often faster and better than XGBoost!

Expected: R² = 0.47-0.52 (competitive with or better than XGBoost)

Usage:
    python src/models/lightgbm_model.py --target profit_margin
    python src/models/lightgbm_model.py --target all
"""
import sys
import json
from datetime import datetime
from pathlib import Path
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Try different LightGBM import methods
try:
    import lightgbm as lgb
    print("✅ LightGBM imported successfully")
except ImportError:
    print("❌ LightGBM not installed!")
    print("   Run: pip install lightgbm")
    sys.exit(1)

warnings.filterwarnings("ignore")

# Setup paths
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from utils.split_utils import get_feature_target_split, drop_nan_targets

print("✅ Imports successful\n")


class LightGBMTrainer:
    """
    LightGBM trainer for financial forecasting
    """

    def __init__(self, target_name: str, params: dict = None):
        self.target_name = target_name
        self.target_col = f"target_{target_name}"

        # Default optimized hyperparameters
        if params is None:
            params = {
                "objective": "regression",
                "metric": "rmse",
                "boosting_type": "gbdt",
                "num_leaves": 31,
                "max_depth": 8,
                "learning_rate": 0.05,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "min_child_samples": 20,
                "reg_alpha": 0.1,
                "reg_lambda": 1.0,
                "random_state": 42,
                "verbose": -1,
                # num_boost_round will be passed separately to lgb.train()
            }

        self.params = params
        self.num_boost_round = 500  # Number of trees
        self.model = None
        self.feature_names = None
        self.best_iteration = None
        self.train_metrics = None
        self.val_metrics = None
        self.test_metrics = None

    def load_data(self, splits_dir: str):
        """Load pre-split data"""
        print(f"\n{'=' * 80}")
        print(f"📂 LOADING DATA FOR: {self.target_name.upper()}")
        print(f"{'=' * 80}")

        splits_path = Path(splits_dir)
        
        train_df = pd.read_csv(splits_path / "train_data.csv")
        val_df = pd.read_csv(splits_path / "val_data.csv")
        test_df = pd.read_csv(splits_path / "test_data.csv")

        print(f"   ✅ Train: {len(train_df):,} rows")
        print(f"   ✅ Val:   {len(val_df):,} rows")
        print(f"   ✅ Test:  {len(test_df):,} rows")

        return train_df, val_df, test_df

    def prepare_features(self, train_df, val_df, test_df):
        """Prepare features"""
        print(f"\n{'=' * 80}")
        print(f"🔧 PREPARING FEATURES")
        print(f"{'=' * 80}")

        X_train, y_train = get_feature_target_split(train_df, self.target_col, encode_categoricals=True)
        X_val, y_val = get_feature_target_split(val_df, self.target_col, encode_categoricals=True)
        X_test, y_test = get_feature_target_split(test_df, self.target_col, encode_categoricals=True)

        # Align columns
        for col in X_train.columns:
            if col not in X_val.columns:
                X_val[col] = 0
            if col not in X_test.columns:
                X_test[col] = 0

        X_val = X_val[X_train.columns]
        X_test = X_test[X_train.columns]

        # Impute missing
        if X_train.isna().sum().sum() > 0:
            imputer = SimpleImputer(strategy="median")
            X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
            X_val = pd.DataFrame(imputer.transform(X_val), columns=X_val.columns, index=X_val.index)
            X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns, index=X_test.index)

        # Drop NaN targets
        X_train, y_train = drop_nan_targets(X_train, y_train, "Train")
        X_val, y_val = drop_nan_targets(X_val, y_val, "Val")
        X_test, y_test = drop_nan_targets(X_test, y_test, "Test")

        self.feature_names = X_train.columns.tolist()

        return X_train, y_train, X_val, y_val, X_test, y_test

    def train(self, X_train, y_train, X_val, y_val):
        """Train LightGBM"""
        print(f"\n{'=' * 80}")
        print(f"🚀 TRAINING LIGHTGBM: {self.target_name.upper()}")
        print(f"{'=' * 80}")

        # Create train and validation datasets in LightGBM format
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        # Train using lgb.train (native API - more reliable)
        print(f"   Training with early stopping (patience=50)...")
        
        callbacks = [
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=0)  # Silent
        ]
        
        self.model = lgb.train(
            self.params,
            train_data,
            valid_sets=[train_data, val_data],
            valid_names=['train', 'val'],
            num_boost_round=self.num_boost_round,
            callbacks=callbacks
        )

        self.best_iteration = self.model.best_iteration
        print(f"   ✅ Best iteration: {self.best_iteration}")

        # Evaluate
        train_pred = self.model.predict(X_train, num_iteration=self.best_iteration)
        val_pred = self.model.predict(X_val, num_iteration=self.best_iteration)

        self.train_metrics = {
            "rmse": float(np.sqrt(mean_squared_error(y_train, train_pred))),
            "mae": float(mean_absolute_error(y_train, train_pred)),
            "r2": float(r2_score(y_train, train_pred)),
        }

        self.val_metrics = {
            "rmse": float(np.sqrt(mean_squared_error(y_val, val_pred))),
            "mae": float(mean_absolute_error(y_val, val_pred)),
            "r2": float(r2_score(y_val, val_pred)),
        }

        print(f"\n   📊 Train R²: {self.train_metrics['r2']:.4f}")
        print(f"   📊 Val R²:   {self.val_metrics['r2']:.4f}")

    def evaluate_test(self, X_test, y_test):
        """Evaluate on test set"""
        print(f"\n{'=' * 80}")
        print(f"📈 TEST SET EVALUATION")
        print(f"{'=' * 80}")

        test_pred = self.model.predict(X_test, num_iteration=self.best_iteration)
        
        self.test_metrics = {
            "rmse": float(np.sqrt(mean_squared_error(y_test, test_pred))),
            "mae": float(mean_absolute_error(y_test, test_pred)),
            "r2": float(r2_score(y_test, test_pred)),
        }

        print(f"   📊 Test R²:   {self.test_metrics['r2']:.4f}")
        print(f"   📊 Test RMSE: {self.test_metrics['rmse']:.2f}")

        return test_pred

    def save_model(self, output_dir: str):
        """Save model and results"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        model_file = output_path / f"lightgbm_{self.target_name}.pkl"
        joblib.dump({
            "model": self.model,
            "params": self.params,
            "feature_names": self.feature_names,
            "best_iteration": self.best_iteration,
            "train_metrics": self.train_metrics,
            "val_metrics": self.val_metrics,
            "test_metrics": self.test_metrics,
        }, model_file)

        # Save metrics in standard format (for model_selection.py)
        metrics_file = output_path / f"lightgbm_{self.target_name}_metrics.json"
        with open(metrics_file, "w") as f:
            json.dump({
                "target": self.target_name,
                "target_column": self.target_col,
                "train": self.train_metrics,
                "val": self.val_metrics,
                "test": self.test_metrics,
                "hyperparameters": self.params,
                "best_iteration": self.best_iteration,
                "n_features": len(self.feature_names),
                "timestamp": datetime.now().isoformat(),
            }, f, indent=2)

        print(f"   ✅ Model: {model_file}")
        print(f"   ✅ Metrics: {metrics_file}")

        return model_file, metrics_file


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Train LightGBM models")
    parser.add_argument('--target', type=str, default='all',
                       choices=['revenue', 'eps', 'debt_equity', 'profit_margin', 'stock_return', 'all'])
    parser.add_argument('--splits-dir', type=str, default='data/splits')
    parser.add_argument('--output-dir', type=str, default='models/lightgbm')
    
    args = parser.parse_args()
    
    targets = ['revenue', 'eps', 'debt_equity', 'profit_margin', 'stock_return'] if args.target == 'all' else [args.target]
    
    print(f"\n{'=' * 80}")
    print(f"🚀 LIGHTGBM TRAINING")
    print(f"{'=' * 80}\n")
    
    results = {}
    
    for target in targets:
        print(f"\n{'=' * 80}")
        print(f"TARGET: {target.upper()}")
        print(f"{'=' * 80}")
        
        try:
            trainer = LightGBMTrainer(target_name=target)
            train_df, val_df, test_df = trainer.load_data(args.splits_dir)
            X_train, y_train, X_val, y_val, X_test, y_test = trainer.prepare_features(train_df, val_df, test_df)
            trainer.train(X_train, y_train, X_val, y_val)
            trainer.evaluate_test(X_test, y_test)
            trainer.save_model(args.output_dir)
            
            results[target] = trainer.test_metrics
            
        except Exception as e:
            print(f"❌ Error: {e}")
            continue
    
    # Summary
    if results:
        print(f"\n\n{'=' * 80}")
        print(f"📊 LIGHTGBM RESULTS")
        print(f"{'=' * 80}\n")
        
        print(f"{'Target':<20} {'Test R²':>10} {'Test RMSE':>15}")
        print("-" * 50)
        for target, metrics in results.items():
            print(f"{target:<20} {metrics['r2']:>10.4f} {metrics['rmse']:>15,.2f}")
        
        if len(results) > 1:
            avg_r2 = np.mean([m['r2'] for m in results.values()])
            print("-" * 50)
            print(f"{'AVERAGE':<20} {avg_r2:>10.4f}")
        
        print(f"\n🎯 Next: Compare with XGBoost")
        print(f"   python src/models/model_selection.py --target profit_margin")


if __name__ == "__main__":
    main()