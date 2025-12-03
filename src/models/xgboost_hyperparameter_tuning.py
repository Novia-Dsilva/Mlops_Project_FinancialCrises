"""
src/models/xgboost_hyperparameter_tuning.py
Hyperparameter tuning for XGBoost models using Optuna with MLflow tracking

Usage:
    # Tune single target (recommended for problem targets like profit_margin)
    python src/models/xgboost_hyperparameter_tuning.py --target profit_margin --trials 50
    
    # Tune all targets
    python src/models/xgboost_hyperparameter_tuning.py --target all --trials 30
    
    # Quick test with fewer trials
    python src/models/xgboost_hyperparameter_tuning.py --target eps --trials 10
"""
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
import warnings
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
import optuna
from optuna.integration import XGBoostPruningCallback

warnings.filterwarnings("ignore")

# Setup paths
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from utils.split_utils import get_feature_target_split, drop_nan_targets
from utils.mlflow_tracker import MLflowTracker

print("✅ Imports successful (split_utils, mlflow_tracker, optuna)\n")


# ============================================
# Hyperparameter Search Space
# ============================================
def get_search_space(trial: optuna.Trial) -> dict:
    """
    Define hyperparameter search space for XGBoost
    
    Args:
        trial: Optuna trial object
    
    Returns:
        Dictionary of hyperparameters to try
    """
    params = {
        # Tree structure
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        
        # Learning
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=50),
        
        # Sampling
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.6, 1.0),
        
        # Regularization
        "gamma": trial.suggest_float("gamma", 0.0, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 2.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 2.0),
        
        # Fixed params
        "random_state": 42,
        "n_jobs": -1,
        "tree_method": "hist",
        "verbosity": 0,
    }
    
    return params


# ============================================
# Optuna Objective Function
# ============================================
class XGBoostObjective:
    """
    Objective function for Optuna hyperparameter optimization
    """
    
    def __init__(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        target_name: str,
    ):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.target_name = target_name
        self.best_score = float('-inf')
        
    def __call__(self, trial: optuna.Trial) -> float:
        """
        Objective function to minimize (negative R²)
        """
        # Get hyperparameters for this trial
        params = get_search_space(trial)
        
        # Create model
        model = xgb.XGBRegressor(**params)
        
        # Train with early stopping (no pruning callback to avoid compatibility issues)
        model.set_params(early_stopping_rounds=50)
        
        try:
            # Try with Optuna pruning callback (for XGBoost 1.x)
            pruning_callback = XGBoostPruningCallback(trial, "validation_0-rmse")
            model.fit(
                self.X_train,
                self.y_train,
                eval_set=[(self.X_val, self.y_val)],
                verbose=False,
                callbacks=[pruning_callback],
            )
        except (TypeError, AttributeError):
            # Fallback: train without pruning callback (for XGBoost 2.0+)
            model.fit(
                self.X_train,
                self.y_train,
                eval_set=[(self.X_val, self.y_val)],
                verbose=False,
            )
        
        # Evaluate
        val_pred = model.predict(self.X_val)
        val_r2 = r2_score(self.y_val, val_pred)
        val_rmse = np.sqrt(mean_squared_error(self.y_val, val_pred))
        
        # Track best score
        if val_r2 > self.best_score:
            self.best_score = val_r2
        
        # Log to trial
        trial.set_user_attr("val_r2", val_r2)
        trial.set_user_attr("val_rmse", val_rmse)
        trial.set_user_attr("n_iterations", model.best_iteration if hasattr(model, 'best_iteration') else params['n_estimators'])
        
        # Return negative R² (Optuna minimizes by default)
        return -val_r2


# ============================================
# Tuner Class
# ============================================
class XGBoostTuner:
    """
    Hyperparameter tuner for XGBoost
    """
    
    def __init__(self, target_name: str):
        self.target_name = target_name
        self.target_col = f"target_{target_name}"
        self.study = None
        self.best_params = None
        self.best_model = None
        self.feature_names = None
        
    def load_and_prepare_data(self, splits_dir: str):
        """
        Load and prepare data for tuning
        """
        print(f"\n{'=' * 80}")
        print(f"📂 LOADING DATA: {self.target_name.upper()}")
        print(f"{'=' * 80}")
        
        splits_path = Path(splits_dir)
        
        # Load splits
        train_df = pd.read_csv(splits_path / "train_data.csv")
        val_df = pd.read_csv(splits_path / "val_data.csv")
        test_df = pd.read_csv(splits_path / "test_data.csv")
        
        print(f"   Train: {len(train_df):,} rows")
        print(f"   Val:   {len(val_df):,} rows")
        print(f"   Test:  {len(test_df):,} rows")
        
        # Prepare features
        print(f"\n🔧 Preparing features...")
        X_train, y_train = get_feature_target_split(train_df, self.target_col, encode_categoricals=True)
        X_val, y_val = get_feature_target_split(val_df, self.target_col, encode_categoricals=True)
        X_test, y_test = get_feature_target_split(test_df, self.target_col, encode_categoricals=True)
        
        # Align columns
        train_cols = set(X_train.columns)
        for col in train_cols:
            if col not in X_val.columns:
                X_val[col] = 0
            if col not in X_test.columns:
                X_test[col] = 0
        
        X_val = X_val[X_train.columns]
        X_test = X_test[X_train.columns]
        
        # Impute missing values
        if X_train.isna().sum().sum() > 0:
            print(f"   Imputing missing values...")
            imputer = SimpleImputer(strategy="median")
            X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
            X_val = pd.DataFrame(imputer.transform(X_val), columns=X_val.columns, index=X_val.index)
            X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns, index=X_test.index)
        
        # Drop NaN targets
        X_train, y_train = drop_nan_targets(X_train, y_train, "Train")
        X_val, y_val = drop_nan_targets(X_val, y_val, "Val")
        X_test, y_test = drop_nan_targets(X_test, y_test, "Test")
        
        self.feature_names = X_train.columns.tolist()
        
        print(f"   ✅ Features: {len(self.feature_names)}")
        print(f"   ✅ Train samples: {len(X_train):,}")
        print(f"   ✅ Val samples: {len(X_val):,}")
        print(f"   ✅ Test samples: {len(X_test):,}")
        
        return X_train, y_train, X_val, y_val, X_test, y_test
    
    def tune(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        n_trials: int = 50,
        timeout: int = None,
    ):
        """
        Run hyperparameter optimization
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            n_trials: Number of optimization trials
            timeout: Timeout in seconds (optional)
        """
        print(f"\n{'=' * 80}")
        print(f"🔍 HYPERPARAMETER TUNING: {self.target_name.upper()}")
        print(f"{'=' * 80}")
        print(f"   Strategy: Bayesian Optimization (Optuna)")
        print(f"   Trials: {n_trials}")
        print(f"   Metric: R² (validation set)")
        print(f"{'=' * 80}\n")
        
        # Create study
        study_name = f"xgboost_{self.target_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.study = optuna.create_study(
            study_name=study_name,
            direction="maximize",  # Maximize R²
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=10),
        )
        
        # Create objective
        objective = XGBoostObjective(X_train, y_train, X_val, y_val, self.target_name)
        
        # Optimize
        print("Starting optimization...\n")
        self.study.optimize(
            lambda trial: -objective(trial),  # Minimize negative R²
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=True,
            callbacks=[self._log_callback],
        )
        
        # Get best parameters
        self.best_params = self.study.best_params
        
        print(f"\n{'=' * 80}")
        print(f"✅ OPTIMIZATION COMPLETE")
        print(f"{'=' * 80}")
        print(f"   Best R²: {-self.study.best_value:.4f}")
        print(f"   Best trial: #{self.study.best_trial.number}")
        print(f"   Total trials: {len(self.study.trials)}")
        print(f"   Pruned trials: {len([t for t in self.study.trials if t.state == optuna.trial.TrialState.PRUNED])}")
        
        return self.best_params
    
    def _log_callback(self, study: optuna.Study, trial: optuna.trial.FrozenTrial):
        """Callback to log progress during optimization"""
        if trial.number % 5 == 0:
            best_r2 = -study.best_value
            current_r2 = trial.user_attrs.get("val_r2", 0)
            print(f"   Trial {trial.number:3d}: R² = {current_r2:.4f} | Best so far: {best_r2:.4f}")
    
    def train_best_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ):
        """
        Train final model with best parameters
        """
        print(f"\n{'=' * 80}")
        print(f"🚀 TRAINING FINAL MODEL WITH BEST PARAMETERS")
        print(f"{'=' * 80}\n")
        
        # Show best parameters
        print("Best Hyperparameters:")
        for param, value in sorted(self.best_params.items()):
            print(f"   {param:20s}: {value}")
        
        # Train model
        self.best_model = xgb.XGBRegressor(**self.best_params, random_state=42, n_jobs=-1, tree_method="hist", verbosity=0)
        self.best_model.set_params(early_stopping_rounds=50)
        
        self.best_model.fit(
            X_train,
            y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=False,
        )
        
        # Evaluate on all splits
        print(f"\n{'=' * 80}")
        print(f"📊 MODEL PERFORMANCE")
        print(f"{'=' * 80}\n")
        
        results = {}
        for name, X, y in [("Train", X_train, y_train), ("Val", X_val, y_val), ("Test", X_test, y_test)]:
            pred = self.best_model.predict(X)
            r2 = r2_score(y, pred)
            rmse = np.sqrt(mean_squared_error(y, pred))
            
            results[name.lower()] = {"r2": r2, "rmse": rmse}
            
            print(f"{name:5s} - R²: {r2:.4f}, RMSE: {rmse:,.2f}")
        
        return results
    
    def save_results(self, output_dir: str, results: dict):
        """
        Save tuning results and best model
        """
        print(f"\n{'=' * 80}")
        print(f"💾 SAVING RESULTS")
        print(f"{'=' * 80}")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save best model
        model_file = output_path / f"xgboost_{self.target_name}_tuned.pkl"
        joblib.dump(
            {
                "model": self.best_model,
                "params": self.best_params,
                "feature_names": self.feature_names,
                "results": results,
                "study_name": self.study.study_name,
                "n_trials": len(self.study.trials),
                "timestamp": timestamp,
            },
            model_file,
        )
        print(f"   ✅ Model: {model_file}")
        
        # Save tuning results
        tuning_file = output_path / f"xgboost_{self.target_name}_tuning_results.json"
        
        # Get trial history
        trials_data = []
        for trial in self.study.trials:
            trials_data.append({
                "number": trial.number,
                "params": trial.params,
                "val_r2": trial.user_attrs.get("val_r2", None),
                "val_rmse": trial.user_attrs.get("val_rmse", None),
                "state": str(trial.state),
            })
        
        tuning_results = {
            "target": self.target_name,
            "best_params": self.best_params,
            "best_val_r2": -self.study.best_value,
            "results": results,
            "n_trials": len(self.study.trials),
            "n_pruned": len([t for t in self.study.trials if t.state == optuna.trial.TrialState.PRUNED]),
            "trials": trials_data,
            "timestamp": timestamp,
        }
        
        with open(tuning_file, "w") as f:
            json.dump(tuning_results, f, indent=2)
        print(f"   ✅ Tuning results: {tuning_file}")
        
        # Save optimization history plot (if optuna has visualization)
        try:
            import plotly.graph_objects as go
            
            # Create optimization history
            trial_nums = [t.number for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            r2_values = [-t.value for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=trial_nums,
                y=r2_values,
                mode='markers+lines',
                name='Trial R²',
                marker=dict(size=8, color=r2_values, colorscale='Viridis', showscale=True),
            ))
            
            # Add best value line
            best_r2 = -self.study.best_value
            fig.add_hline(y=best_r2, line_dash="dash", line_color="red", 
                         annotation_text=f"Best: {best_r2:.4f}")
            
            fig.update_layout(
                title=f"Hyperparameter Optimization: {self.target_name}",
                xaxis_title="Trial Number",
                yaxis_title="Validation R²",
                hovermode='closest',
            )
            
            plot_file = output_path / f"xgboost_{self.target_name}_optimization.html"
            fig.write_html(plot_file)
            print(f"   ✅ Optimization plot: {plot_file}")
            
        except ImportError:
            print(f"   ⚠️  Plotly not available for visualization")
        
        return model_file, tuning_file
    
    def compare_with_baseline(self, baseline_metrics: dict, tuned_metrics: dict):
        """
        Compare tuned model with baseline
        """
        print(f"\n{'=' * 80}")
        print(f"📊 COMPARISON: BASELINE vs TUNED")
        print(f"{'=' * 80}\n")
        
        print(f"{'Split':<10} {'Metric':<8} {'Baseline':>12} {'Tuned':>12} {'Δ':>12} {'Improvement':>12}")
        print("-" * 75)
        
        for split in ["train", "val", "test"]:
            if split in baseline_metrics and split in tuned_metrics:
                base_r2 = baseline_metrics[split]["r2"]
                tuned_r2 = tuned_metrics[split]["r2"]
                delta_r2 = tuned_r2 - base_r2
                improvement = (delta_r2 / abs(base_r2)) * 100 if base_r2 != 0 else 0
                
                print(f"{split.capitalize():<10} {'R²':<8} {base_r2:>12.4f} {tuned_r2:>12.4f} {delta_r2:>+12.4f} {improvement:>11.2f}%")
                
                base_rmse = baseline_metrics[split]["rmse"]
                tuned_rmse = tuned_metrics[split]["rmse"]
                delta_rmse = tuned_rmse - base_rmse
                improvement = (delta_rmse / base_rmse) * 100 if base_rmse != 0 else 0
                
                print(f"{'':10} {'RMSE':<8} {base_rmse:>12,.2f} {tuned_rmse:>12,.2f} {delta_rmse:>+12,.2f} {improvement:>11.2f}%")
                print()


# ============================================
# Main Tuning Function
# ============================================
def tune_target(
    target_name: str,
    splits_dir: str,
    output_dir: str,
    n_trials: int = 50,
    baseline_file: str = None,
):
    """
    Tune hyperparameters for a single target
    """
    print(f"\n{'=' * 80}")
    print(f"🎯 TUNING TARGET: {target_name.upper()}")
    print(f"{'=' * 80}")
    
    # Initialize tuner
    tuner = XGBoostTuner(target_name=target_name)
    
    # Load data
    X_train, y_train, X_val, y_val, X_test, y_test = tuner.load_and_prepare_data(splits_dir)
    
    # Run tuning
    best_params = tuner.tune(X_train, y_train, X_val, y_val, n_trials=n_trials)
    
    # Train best model
    results = tuner.train_best_model(X_train, y_train, X_val, y_val, X_test, y_test)
    
    # Save results
    model_file, tuning_file = tuner.save_results(output_dir, results)
    
    # Compare with baseline if provided
    if baseline_file:
        baseline_path = Path(baseline_file)
        if baseline_path.exists():
            with open(baseline_path, "r") as f:
                baseline_data = json.load(f)
            
            baseline_metrics = {
                "train": baseline_data.get("train", {}),
                "val": baseline_data.get("val", {}),
                "test": baseline_data.get("test", {}),
            }
            
            tuner.compare_with_baseline(baseline_metrics, results)
        else:
            print(f"\n⚠️  Baseline file not found: {baseline_file}")
    
    print(f"\n{'=' * 80}")
    print(f"✅ TUNING COMPLETE: {target_name.upper()}")
    print(f"{'=' * 80}")
    print(f"   Test R²: {results['test']['r2']:.4f}")
    print(f"   Best params saved to: {tuning_file}")
    print(f"   Best model saved to: {model_file}")
    
    return tuner, results


# ============================================
# CLI Interface
# ============================================
def main():
    parser = argparse.ArgumentParser(
        description="Hyperparameter tuning for XGBoost models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Tune profit_margin (worst performer) with 50 trials
  python xgboost_hyperparameter_tuning.py --target profit_margin --trials 50
  
  # Quick test with 10 trials
  python xgboost_hyperparameter_tuning.py --target eps --trials 10
  
  # Tune all targets with 30 trials each
  python xgboost_hyperparameter_tuning.py --target all --trials 30
  
  # Compare with baseline model
  python xgboost_hyperparameter_tuning.py --target profit_margin --trials 50 \\
      --baseline models/xgboost/xgboost_profit_margin_metrics.json
        """,
    )
    
    parser.add_argument(
        "--target",
        type=str,
        required=True,
        choices=["revenue", "eps", "debt_equity", "profit_margin", "stock_return", "all"],
        help="Target to tune",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=50,
        help="Number of optimization trials (default: 50)",
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
        default="models/xgboost_tuned",
        help="Directory to save tuned models",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        help="Path to baseline metrics JSON for comparison (optional)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        help="Timeout in seconds for optimization (optional)",
    )
    
    args = parser.parse_args()
    
    # Determine targets
    all_targets = ["revenue", "eps", "debt_equity", "profit_margin", "stock_return"]
    
    if args.target == "all":
        targets = all_targets
        print(f"\n{'=' * 80}")
        print(f"🔍 TUNING ALL {len(targets)} TARGETS")
        print(f"{'=' * 80}")
        print(f"Targets: {', '.join(targets)}")
        print(f"Trials per target: {args.trials}")
        print(f"{'=' * 80}\n")
    else:
        targets = [args.target]
    
    # Tune each target
    results_summary = {}
    
    for i, target in enumerate(targets, 1):
        print(f"\n\n{'#' * 80}")
        print(f"TARGET {i}/{len(targets)}: {target.upper()}")
        print(f"{'#' * 80}\n")
        
        try:
            # Get baseline file path if provided
            baseline_file = None
            if args.baseline:
                baseline_file = args.baseline
            else:
                # Try to find baseline in default location
                default_baseline = Path(f"models/xgboost/xgboost_{target}_metrics.json")
                if default_baseline.exists():
                    baseline_file = str(default_baseline)
                    print(f"📊 Found baseline metrics: {baseline_file}\n")
            
            # Run tuning
            tuner, results = tune_target(
                target_name=target,
                splits_dir=args.splits_dir,
                output_dir=args.output_dir,
                n_trials=args.trials,
                baseline_file=baseline_file,
            )
            
            results_summary[target] = results
            
        except Exception as e:
            print(f"\n❌ ERROR tuning {target}:")
            print(f"   {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    # Final summary
    if results_summary:
        print(f"\n\n{'=' * 80}")
        print(f"📊 FINAL TUNING SUMMARY")
        print(f"{'=' * 80}\n")
        
        print(f"{'Target':<20} {'Test R²':>12} {'Test RMSE':>15}")
        print("-" * 50)
        
        for target, results in results_summary.items():
            test_r2 = results["test"]["r2"]
            test_rmse = results["test"]["rmse"]
            print(f"{target:<20} {test_r2:>12.4f} {test_rmse:>15,.2f}")
        
        if len(results_summary) > 1:
            avg_r2 = np.mean([r["test"]["r2"] for r in results_summary.values()])
            print("-" * 50)
            print(f"{'AVERAGE':<20} {avg_r2:>12.4f}")
        
        print(f"\n{'=' * 80}")
        print(f"✅ ALL TUNING COMPLETE!")
        print(f"{'=' * 80}")
        print(f"\n📁 Tuned models saved to: {args.output_dir}/")
        print(f"\n💡 Next steps:")
        print(f"   1. Review tuning results in {args.output_dir}/")
        print(f"   2. Compare baseline vs tuned performance")
        print(f"   3. Use best parameters for production models")
        print(f"   4. Consider ensemble methods if single model performance plateaus")


if __name__ == "__main__":
    print(f"\n{'=' * 80}")
    print(f"🔍 XGBOOST HYPERPARAMETER TUNING")
    print(f"{'=' * 80}")
    print("Using Optuna for Bayesian Optimization")
    print(f"{'=' * 80}\n")
    
    main()