"""
src/models/xgboost_model.py
Train XGBoost models on pre-cleaned, split data WITH MLFLOW TRACKING
Uses data from data/splits/ (already split and outlier-handled!)
Usage:
    python src/models/xgboost_model.py                    # Trains ALL 5 targets
    python src/models/xgboost_model.py --target revenue   # Train single target
"""
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path
import warnings
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
warnings.filterwarnings("ignore")

# ========================================
# Setup paths so we can import from src/
# ========================================
project_root = Path(__file__).resolve().parent.parent.parent
print(f"✅ Project root: {project_root}")

# Add src/ to sys.path (so 'utils' is importable)
sys.path.insert(0, str(project_root / "src"))

# Now import split utilities and MLflow tracker
from utils.split_utils import get_feature_target_split, drop_nan_targets  # noqa: E402
from utils.mlflow_tracker import MLflowTracker, track_xgboost_run  # noqa: E402

print("✅ Imports successful (split_utils, mlflow_tracker)\n")


# ============================================
# XGBoost Model Trainer
# ============================================
class XGBoostTrainer:
    """
    XGBoost trainer for financial forecasting
    """

    def __init__(self, target_name: str, params: dict = None):
        """
        Initialize XGBoost trainer
        Args:
            target_name: Name of target (revenue, eps, etc.)
            params: Optional custom hyperparameters
        """
        self.target_name = target_name
        self.target_col = f"target_{target_name}"

        # Default optimized hyperparameters
        if params is None:
            params = {
                "n_estimators": 500,
                "max_depth": 8,
                "learning_rate": 0.05,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "min_child_weight": 3,
                "gamma": 0.1,
                "reg_alpha": 0.1,
                "reg_lambda": 1.0,
                "random_state": 42,
                "n_jobs": -1,
                "tree_method": "hist",
                "verbosity": 0,
            }

        self.params = params
        self.model = xgb.XGBRegressor(**params)
        self.feature_names = None
        self.best_iteration = None
        self.train_metrics = None
        self.val_metrics = None
        self.test_metrics = None

    def load_data(self, splits_dir: str):
        """
        Load pre-split and cleaned data
        Args:
            splits_dir: Directory containing train/val/test CSVs
        """
        print(f"\n{'=' * 80}")
        print(f"📂 LOADING DATA FOR: {self.target_name.upper()}")
        print(f"{'=' * 80}")

        splits_path = Path(splits_dir)

        # Load splits
        train_file = splits_path / "train_data.csv"
        val_file = splits_path / "val_data.csv"
        test_file = splits_path / "test_data.csv"

        print(f"\n1️⃣ Loading pre-split data...")
        train_df = pd.read_csv(train_file)
        val_df = pd.read_csv(val_file)
        test_df = pd.read_csv(test_file)

        print(f"   ✅ Train: {len(train_df):,} rows")
        print(f"   ✅ Val:   {len(val_df):,} rows")
        print(f"   ✅ Test:  {len(test_df):,} rows")

        # Check target exists
        if self.target_col not in train_df.columns:
            raise ValueError(f"Target column '{self.target_col}' not found!")

        # Check valid samples
        train_valid = train_df[self.target_col].notna().sum()
        val_valid = val_df[self.target_col].notna().sum()
        test_valid = test_df[self.target_col].notna().sum()

        print(f"\n2️⃣ Target: {self.target_col}")
        print(f"   Train valid: {train_valid:,} ({train_valid/len(train_df)*100:.1f}%)")
        print(f"   Val valid:   {val_valid:,} ({val_valid/len(val_df)*100:.1f}%)")
        print(f"   Test valid:  {test_valid:,} ({test_valid/len(test_df)*100:.1f}%)")

        if train_valid < 100:
            raise ValueError(f"Insufficient training data: only {train_valid} valid samples")

        # Show target statistics
        print(f"\n3️⃣ Target statistics (after outlier handling):")
        train_target = train_df[self.target_col].dropna()
        print(f"   Mean:   {train_target.mean():,.2f}")
        print(f"   Std:    {train_target.std():,.2f}")
        print(f"   Min:    {train_target.min():,.2f}")
        print(f"   25%:    {train_target.quantile(0.25):,.2f}")
        print(f"   Median: {train_target.median():,.2f}")
        print(f"   75%:    {train_target.quantile(0.75):,.2f}")
        print(f"   Max:    {train_target.max():,.2f}")

        return train_df, val_df, test_df

    def prepare_features(self, train_df, val_df, test_df):
        """
        Prepare features using split_utils
        """
        print(f"\n{'=' * 80}")
        print(f"🔧 PREPARING FEATURES")
        print(f"{'=' * 80}")

        # Use split_utils to get features and handle encoding
        X_train, y_train = get_feature_target_split(
            train_df,
            target_col=self.target_col,
            encode_categoricals=True,
        )

        X_val, y_val = get_feature_target_split(
            val_df,
            target_col=self.target_col,
            encode_categoricals=True,
        )

        X_test, y_test = get_feature_target_split(
            test_df,
            target_col=self.target_col,
            encode_categoricals=True,
        )

        # Align columns (val/test might have different categories)
        print(f"\n4️⃣ Aligning features across splits...")
        train_cols = set(X_train.columns)

        # Add missing columns to val/test and fill with 0
        for col in train_cols:
            if col not in X_val.columns:
                X_val[col] = 0
            if col not in X_test.columns:
                X_test[col] = 0

        # Reorder columns to match train
        X_val = X_val[X_train.columns]
        X_test = X_test[X_train.columns]

        print(f"   ✅ All splits have {len(X_train.columns)} features")

        # Handle missing feature values
        print(f"\n5️⃣ Handling missing feature values...")
        nan_train = X_train.isna().sum().sum()
        nan_val = X_val.isna().sum().sum()
        nan_test = X_test.isna().sum().sum()

        if nan_train > 0 or nan_val > 0 or nan_test > 0:
            print(f"   Missing: Train={nan_train:,}, Val={nan_val:,}, Test={nan_test:,}")
            print(f"   Imputing with median...")

            imputer = SimpleImputer(strategy="median")
            X_train = pd.DataFrame(
                imputer.fit_transform(X_train),
                columns=X_train.columns,
                index=X_train.index,
            )
            X_val = pd.DataFrame(
                imputer.transform(X_val),
                columns=X_val.columns,
                index=X_val.index,
            )
            X_test = pd.DataFrame(
                imputer.transform(X_test),
                columns=X_test.columns,
                index=X_test.index,
            )
            print(f"   ✅ Imputation complete")
        else:
            print(f"   ✅ No missing values in features")

        # Drop rows with NaN targets (unlabeled rows)
        print(f"\n6️⃣ Removing rows with NaN targets from each split...")
        X_train, y_train = drop_nan_targets(X_train, y_train, "Train")
        X_val, y_val = drop_nan_targets(X_val, y_val, "Val")
        X_test, y_test = drop_nan_targets(X_test, y_test, "Test")

        print(f"\n{'=' * 80}")
        print(f"✅ FEATURES READY")
        print(f"{'=' * 80}")
        print(f"   Features: {X_train.shape[1]}")
        print(f"   Train samples: {len(X_train):,}")
        print(f"   Val samples:   {len(X_val):,}")
        print(f"   Test samples:  {len(X_test):,}")

        self.feature_names = X_train.columns.tolist()

        return X_train, y_train, X_val, y_val, X_test, y_test

    def train(self, X_train, y_train, X_val, y_val):
        """
        Train XGBoost with early stopping
        """
        print(f"\n{'=' * 80}")
        print(f"🚀 TRAINING XGBOOST: {self.target_name.upper()}")
        print(f"{'=' * 80}")

        # Enable early stopping
        self.model.set_params(early_stopping_rounds=50)

        # Train
        print(f"\n   Training with early stopping (patience=50)...")
        self.model.fit(
            X_train,
            y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=False,
        )

        # Get best iteration
        try:
            self.best_iteration = self.model.best_iteration
            print(f"   ✅ Best iteration: {self.best_iteration}")
        except Exception:
            self.best_iteration = self.params["n_estimators"]
            print(f"   ✅ Used all {self.best_iteration} iterations")

        # Evaluate
        print(f"\n6️⃣ Evaluating performance...")

        # Train set
        train_pred = self.model.predict(X_train)
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        train_mae = mean_absolute_error(y_train, train_pred)
        train_r2 = r2_score(y_train, train_pred)

        self.train_metrics = {
            "rmse": train_rmse,
            "mae": train_mae,
            "r2": train_r2,
        }

        print(f"\n   📊 Train Set:")
        print(f"      RMSE: {train_rmse:,.4f}")
        print(f"      MAE:  {train_mae:,.4f}")
        print(f"      R²:   {train_r2:.4f}")

        # Validation set
        val_pred = self.model.predict(X_val)
        val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
        val_mae = mean_absolute_error(y_val, val_pred)
        val_r2 = r2_score(y_val, val_pred)

        self.val_metrics = {
            "rmse": val_rmse,
            "mae": val_mae,
            "r2": val_r2,
        }

        print(f"\n   📊 Validation Set:")
        print(f"      RMSE: {val_rmse:,.4f}")
        print(f"      MAE:  {val_mae:,.4f}")
        print(f"      R²:   {val_r2:.4f}")

        # Check overfitting
        overfit_pct = ((val_rmse / train_rmse) - 1) * 100
        if overfit_pct > 30:
            print(f"\n   ⚠️  Overfitting: Val RMSE is {overfit_pct:.1f}% higher")
        elif overfit_pct < 0:
            print(f"\n   ⚠️  Suspicious: Val RMSE is better than train")
        else:
            print(f"\n   ✅ Good fit: Val RMSE is {overfit_pct:.1f}% higher")

    def evaluate_test(self, X_test, y_test):
        """
        Evaluate on test set
        """
        print(f"\n{'=' * 80}")
        print(f"📈 TEST SET EVALUATION")
        print(f"{'=' * 80}")

        test_pred = self.model.predict(X_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        test_mae = mean_absolute_error(y_test, test_pred)
        test_r2 = r2_score(y_test, test_pred)

        self.test_metrics = {
            "rmse": test_rmse,
            "mae": test_mae,
            "r2": test_r2,
        }

        print(f"\n   📊 Test Set Performance:")
        print(f"      RMSE: {test_rmse:,.4f}")
        print(f"      MAE:  {test_mae:,.4f}")
        print(f"      R²:   {test_r2:.4f}")

        # Performance assessment
        if test_r2 > 0.70:
            status = "EXCELLENT ✅"
        elif test_r2 > 0.50:
            status = "GOOD ✅"
        elif test_r2 > 0.30:
            status = "ACCEPTABLE ⚠️"
        else:
            status = "NEEDS IMPROVEMENT ❌"

        print(f"\n   Status: {status}")

        return test_pred

    def get_feature_importance(self, top_n=20):
        """
        Get and display feature importance
        """
        print(f"\n{'=' * 80}")
        print(f"🔍 FEATURE IMPORTANCE (Top {top_n})")
        print(f"{'=' * 80}")

        importance_dict = self.model.get_booster().get_score(importance_type="gain")

        if not importance_dict:
            print("   ⚠️  No importance available")
            return pd.DataFrame()

        importance_df = pd.DataFrame(
            {
                "feature": list(importance_dict.keys()),
                "importance": list(importance_dict.values()),
            }
        ).sort_values("importance", ascending=False)

        print()
        for _, row in importance_df.head(top_n).iterrows():
            print(f"   {row['feature'][:50]:50s} {row['importance']:>10,.2f}")

        return importance_df

    def save_model(self, output_dir: str):
        """
        Save model and results
        """
        print(f"\n{'=' * 80}")
        print(f"💾 SAVING MODEL AND RESULTS")
        print(f"{'=' * 80}")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save model
        model_file = output_path / f"xgboost_{self.target_name}.pkl"
        joblib.dump(
            {
                "model": self.model,
                "params": self.params,
                "feature_names": self.feature_names,
                "best_iteration": self.best_iteration,
                "train_metrics": self.train_metrics,
                "val_metrics": self.val_metrics,
                "test_metrics": self.test_metrics,
            },
            model_file,
        )
        print(f"\n   ✅ Model: {model_file}")

        # Save metrics
        metrics_file = output_path / f"xgboost_{self.target_name}_metrics.json"
        with open(metrics_file, "w") as f:
            json.dump(
                {
                    "target": self.target_name,
                    "target_column": self.target_col,
                    "train": self.train_metrics,
                    "val": self.val_metrics,
                    "test": self.test_metrics,
                    "hyperparameters": self.params,
                    "best_iteration": self.best_iteration,
                    "n_features": len(self.feature_names),
                    "timestamp": datetime.now().isoformat(),
                },
                f,
                indent=2,
            )
        print(f"   ✅ Metrics: {metrics_file}")

        # Save feature importance
        importance_df = self.get_feature_importance(top_n=50)
        if not importance_df.empty:
            importance_file = output_path / f"xgboost_{self.target_name}_importance.csv"
            importance_df.to_csv(importance_file, index=False)
            print(f"   ✅ Importance: {importance_file}")

        return model_file, metrics_file


# ============================================
# Main Training Function
# ============================================
def train_xgboost_model(
    target_name: str, 
    splits_dir: str, 
    output_dir: str,
    mlflow_tracker: MLflowTracker = None
):
    """
    Train XGBoost for a single target
    Args:
        target_name: Target to predict (revenue, eps, debt_equity, profit_margin, stock_return)
        splits_dir: Directory with train/val/test splits
        output_dir: Where to save model
        mlflow_tracker: Optional MLflow tracker for logging
    """
    print(f"\n{'=' * 80}")
    print(f"🎯 XGBOOST TRAINING: {target_name.upper()}")
    print(f"{'=' * 80}")
    print(f"   Input: {splits_dir}")
    print(f"   Output: {output_dir}")

    # Initialize trainer
    trainer = XGBoostTrainer(target_name=target_name)

    # Load data
    train_df, val_df, test_df = trainer.load_data(splits_dir)

    # Prepare features
    X_train, y_train, X_val, y_val, X_test, y_test = trainer.prepare_features(
        train_df, val_df, test_df
    )

    # Train
    trainer.train(X_train, y_train, X_val, y_val)

    # Evaluate on test
    _ = trainer.evaluate_test(X_test, y_test)

    # Save
    model_file, _ = trainer.save_model(output_dir)

    # MLflow tracking
    if mlflow_tracker is not None:
        print(f"\n{'=' * 80}")
        print(f"📊 LOGGING TO MLFLOW")
        print(f"{'=' * 80}")
        track_xgboost_run(mlflow_tracker, target_name, trainer, X_test, y_test)

    print(f"\n{'=' * 80}")
    print(f"✅ TRAINING COMPLETE: {target_name.upper()}")
    print(f"{'=' * 80}")
    print(f"   Test R²: {trainer.test_metrics['r2']:.4f}")
    print(f"   Model: {model_file}")

    return trainer


# ============================================
# CLI Interface
# ============================================
def main():
    parser = argparse.ArgumentParser(
        description="Train XGBoost models for financial forecasting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python xgboost_model.py                    # Train ALL 5 targets (default)
  python xgboost_model.py --target revenue   # Train only revenue
  python xgboost_model.py --target all       # Train all targets
  python xgboost_model.py --no-mlflow        # Train without MLflow tracking
        """,
    )
    parser.add_argument(
        "--target",
        type=str,
        default="all",
        choices=["revenue", "eps", "debt_equity", "profit_margin", "stock_return", "all"],
        help="Target to predict (default: all)",
    )
    parser.add_argument(
        "--splits-dir",
        type=str,
        default="data/splits",
        help="Directory containing train/val/test splits",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/xgboost",
        help="Directory to save models",
    )
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Disable MLflow tracking",
    )
    parser.add_argument(
        "--mlflow-uri",
        type=str,
        default="mlruns",
        help="MLflow tracking URI (default: mlruns)",
    )

    args = parser.parse_args()

    # Define all targets
    all_targets = ["revenue", "eps", "debt_equity", "profit_margin", "stock_return"]

    # Determine which targets to train
    if args.target == "all":
        targets = all_targets
        print(f"\n{'=' * 80}")
        print(f"🎯 TRAINING XGBOOST FOR ALL {len(targets)} TARGETS")
        print(f"{'=' * 80}")
        print(f"Targets: {', '.join(targets)}")
        print(f"{'=' * 80}\n")
    else:
        targets = [args.target]
        print(f"\n🎯 Training single target: {args.target}\n")

    # Initialize MLflow tracker (if not disabled)
    mlflow_tracker = None
    if not args.no_mlflow:
        mlflow_tracker = MLflowTracker(
            experiment_name="financial_forecasting_xgboost",
            tracking_uri=args.mlflow_uri
        )
        print()

    # Train each target
    results = {}
    failed = []

    for i, target in enumerate(targets, 1):
        print(f"\n{'=' * 80}")
        print(f"📊 MODEL {i}/{len(targets)}: {target.upper()}")
        print(f"{'=' * 80}")

        try:
            trainer = train_xgboost_model(
                target_name=target,
                splits_dir=args.splits_dir,
                output_dir=args.output_dir,
                mlflow_tracker=mlflow_tracker,
            )
            results[target] = trainer.test_metrics

            print(f"\n✅ {target.upper()} completed successfully!")
            print(f"   Test R²: {trainer.test_metrics['r2']:.4f}")

        except Exception as e:
            print(f"\n❌ ERROR training {target}:")
            print(f"   {str(e)}")
            failed.append(target)
            import traceback
            print(f"\n   Full traceback:")
            traceback.print_exc()
            continue

    # Final Summary
    if len(results) > 0:
        print(f"\n\n{'=' * 80}")
        print(f"📊 FINAL TRAINING SUMMARY - ALL TARGETS")
        print(f"{'=' * 80}\n")

        print(f"{'Target':<20} {'Test R²':>10} {'Test RMSE':>15} {'Test MAE':>15}")
        print(f"-" * 65)

        for target, metrics in results.items():
            print(
                f"{target:<20} {metrics['r2']:>10.4f} "
                f"{metrics['rmse']:>15,.2f} {metrics['mae']:>15,.2f}"
            )

        if len(results) > 1:
            avg_r2 = np.mean([m["r2"] for m in results.values()])
            print(f"-" * 65)
            print(f"{'AVERAGE':<20} {avg_r2:>10.4f}")

        print(f"\n{'=' * 80}")
        print(
            f"✅ TRAINING COMPLETE: {len(results)}/{len(targets)} models trained successfully!"
        )

        if failed:
            print(f"❌ Failed: {len(failed)} models - {', '.join(failed)}")

        print(f"{'=' * 80}")
        print(f"\n📁 Models saved to: {args.output_dir}/")
        for target in results.keys():
            print(f"   - xgboost_{target}.pkl")
            print(f"   - xgboost_{target}_metrics.json")
            print(f"   - xgboost_{target}_importance.csv")
        
        if not args.no_mlflow:
            print(f"\n📊 MLflow UI:")
            print(f"   Run: mlflow ui --backend-store-uri {args.mlflow_uri}")
            print(f"   Then open: http://localhost:5000")

    else:
        print(f"\n{'=' * 80}")
        print(f"❌ NO MODELS TRAINED SUCCESSFULLY")
        print(f"{'=' * 80}")

        if failed:
            print(f"\nFailed targets: {', '.join(failed)}")

        print(f"\nPlease check:")
        print(f"   1. Data splits exist in: {args.splits_dir}/")
        print(f"   2. Target columns exist in data")
        print(f"   3. Sufficient valid samples for training")
        print(f"   4. Error messages above for details")


if __name__ == "__main__":
    print(f"\n{'=' * 80}")
    print(f"🚀 XGBOOST MODEL TRAINING WITH MLFLOW")
    print(f"{'=' * 80}")
    print("This script will train XGBoost models for financial forecasting")
    print("Using pre-split and cleaned data from data/splits/")
    print(f"{'=' * 80}\n")

    main()