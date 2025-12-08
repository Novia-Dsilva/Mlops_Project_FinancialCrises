

# """
# src/models/predict_with_best_models.py

# Load and use best models for predictions

# This script provides easy-to-use functions for loading best models
# and making predictions on new data.

# Usage:
#     # Make predictions for all targets
#     python src/models/predict_with_best_models.py --input data/new_data.csv
    
#     # Predict specific target
#     python src/models/predict_with_best_models.py --input data/new_data.csv --target revenue
    
#     # From Python
#     from predict_with_best_models import BestModelPredictor
#     predictor = BestModelPredictor()
#     predictions = predictor.predict_all(X_new)
# """

# import sys
# import argparse
# from pathlib import Path
# from typing import Dict, Union
# import warnings
# import joblib
# import numpy as np
# import pandas as pd
# from sklearn.impute import SimpleImputer

# warnings.filterwarnings("ignore")

# # Setup paths
# project_root = Path(__file__).resolve().parent.parent.parent.parent  
# sys.path.insert(0, str(project_root))

# from src.utils.split_utils import get_feature_target_split


# class BestModelPredictor:
#     """
#     Load and use best models for predictions
#     """
    
#     def __init__(self, models_dir: str = "models/best_models"):
#         """
#         Initialize predictor
        
#         Args:
#             models_dir: Directory containing best model files
#         """
#         self.models_dir = Path(models_dir)
#         self.models = {}
#         self.feature_names = {}
#         self.model_info = {}
        
#         if not self.models_dir.exists():
#             raise FileNotFoundError(f"Models directory not found: {models_dir}")
    
#     def load_model(self, target: str) -> bool:
#         """
#         Load best model for a specific target
        
#         Args:
#             target: Target variable name
        
#         Returns:
#             True if loaded successfully
#         """
#         model_file = self.models_dir / f"{target}_best.pkl"
        
#         if not model_file.exists():
#             print(f"Model not found: {model_file}")
#             return False
        
#         try:
#             model_data = joblib.load(model_file)
            
#             self.models[target] = model_data["model"]
#             self.feature_names[target] = model_data["feature_names"]
            
#             self.model_info[target] = {
#                 "model_type": model_data.get("model_type", "unknown"),
#                 "test_r2": model_data.get("test_metrics", {}).get("r2", None),
#                 "test_rmse": model_data.get("test_metrics", {}).get("rmse", None),
#                 "selection_reasoning": model_data.get("selection_reasoning", ""),
#                 "timestamp": model_data.get("timestamp", "unknown")
#             }
            
#             print(f"Loaded {target}: {model_data.get('model_type', 'unknown')} "
#                   f"(R2={model_data.get('test_metrics', {}).get('r2', 0):.4f})")
            
#             return True
            
#         except Exception as e:
#             print(f"Error loading {target}: {str(e)}")
#             return False
    
#     def load_all_models(self) -> Dict[str, bool]:
#         """
#         Load all available best models
        
#         Returns:
#             Dictionary mapping target names to load success status
#         """
#         print(f"\n{'='*80}")
#         print(f"LOADING BEST MODELS")
#         print(f"{'='*80}\n")
        
#         all_targets = ["revenue", "eps", "debt_equity", "profit_margin", "stock_return"]
        
#         results = {}
#         for target in all_targets:
#             results[target] = self.load_model(target)
        
#         loaded_count = sum(results.values())
#         print(f"\nLoaded {loaded_count}/{len(all_targets)} models")
        
#         return results
    
#     def prepare_features(self, X: pd.DataFrame, target: str) -> pd.DataFrame:
#         """
#         Prepare features to match model's expected format
        
#         Args:
#             X: Input features DataFrame
#             target: Target variable name
        
#         Returns:
#             Prepared features DataFrame
#         """
#         if target not in self.feature_names:
#             raise ValueError(f"Model not loaded for target: {target}")
        
#         expected_features = self.feature_names[target]
        
#         # Add missing columns (fill with 0)
#         for col in expected_features:
#             if col not in X.columns:
#                 X[col] = 0
        
#         # Remove extra columns and reorder
#         X_prepared = X[expected_features].copy()
        
#         # Handle missing values
#         if X_prepared.isna().any().any():
#             imputer = SimpleImputer(strategy="median")
#             X_prepared = pd.DataFrame(
#                 imputer.fit_transform(X_prepared),
#                 columns=X_prepared.columns,
#                 index=X_prepared.index
#             )
        
#         return X_prepared
    
#     def predict(self, X: pd.DataFrame, target: str) -> np.ndarray:
#         """
#         Make predictions for a specific target
        
#         Args:
#             X: Input features DataFrame
#             target: Target variable name
        
#         Returns:
#             Predictions array
#         """
#         if target not in self.models:
#             raise ValueError(f"Model not loaded for target: {target}. Call load_model() first.")
        
#         # Prepare features
#         X_prepared = self.prepare_features(X, target)
        
#         # Make predictions
#         model = self.models[target]
        
#         # Handle different model types
#         try:
#             # Try standard predict
#             predictions = model.predict(X_prepared)
#         except AttributeError:
#             # For LightGBM native API
#             try:
#                 predictions = model.predict(X_prepared, num_iteration=model.best_iteration)
#             except Exception as e:
#                 raise RuntimeError(f"Prediction failed: {str(e)}")
        
#         return predictions
    
#     def predict_all(self, X: pd.DataFrame, return_dict: bool = True) -> Union[pd.DataFrame, Dict[str, np.ndarray]]:
#         """
#         Make predictions for all loaded targets
        
#         Args:
#             X: Input features DataFrame
#             return_dict: If True, return dict; if False, return DataFrame
        
#         Returns:
#             Predictions for all targets
#         """
#         predictions = {}
        
#         for target in self.models.keys():
#             try:
#                 predictions[target] = self.predict(X, target)
#             except Exception as e:
#                 print(f"Failed to predict {target}: {str(e)}")
#                 predictions[target] = np.full(len(X), np.nan)
        
#         if return_dict:
#             return predictions
#         else:
#             # Convert to DataFrame
#             pred_df = pd.DataFrame(predictions, index=X.index)
#             return pred_df
    
#     def print_model_info(self):
#         """Print information about loaded models"""
        
#         print(f"\n{'='*80}")
#         print(f"LOADED MODELS INFORMATION")
#         print(f"{'='*80}\n")
        
#         if not self.models:
#             print("No models loaded.")
#             return
        
#         print(f"{'Target':<20} {'Model Type':<20} {'Test R2':>10} {'Test RMSE':>15}")
#         print(f"{'─'*70}")
        
#         for target, info in self.model_info.items():
#             model_type = info.get("model_type", "unknown")
#             test_r2 = info.get("test_r2", 0)
#             test_rmse = info.get("test_rmse", 0)
            
#             r2_str = f"{test_r2:.4f}" if test_r2 is not None else "N/A"
#             rmse_str = f"{test_rmse:,.2f}" if test_rmse is not None else "N/A"
            
#             print(f"{target:<20} {model_type:<20} {r2_str:>10} {rmse_str:>15}")
        
#         print(f"\n{'='*80}\n")
    
#     def evaluate_predictions(self, X: pd.DataFrame, y_true: Dict[str, pd.Series]) -> pd.DataFrame:
#         """
#         Evaluate predictions against ground truth
        
#         Args:
#             X: Input features
#             y_true: Dictionary mapping target names to true values
        
#         Returns:
#             DataFrame with evaluation metrics
#         """
#         from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
#         results = []
        
#         for target in self.models.keys():
#             if target not in y_true:
#                 continue
            
#             try:
#                 # Make predictions
#                 y_pred = self.predict(X, target)
#                 y_actual = y_true[target]
                
#                 # Remove NaN values
#                 mask = ~(pd.isna(y_actual) | pd.isna(y_pred))
#                 y_actual_clean = y_actual[mask]
#                 y_pred_clean = y_pred[mask]
                
#                 if len(y_actual_clean) == 0:
#                     continue
                
#                 # Calculate metrics
#                 rmse = np.sqrt(mean_squared_error(y_actual_clean, y_pred_clean))
#                 mae = mean_absolute_error(y_actual_clean, y_pred_clean)
#                 r2 = r2_score(y_actual_clean, y_pred_clean)
                
#                 results.append({
#                     "target": target,
#                     "model_type": self.model_info[target]["model_type"],
#                     "n_samples": len(y_actual_clean),
#                     "rmse": rmse,
#                     "mae": mae,
#                     "r2": r2
#                 })
                
#             except Exception as e:
#                 print(f"Failed to evaluate {target}: {str(e)}")
#                 continue
        
#         return pd.DataFrame(results)


# # ============================================
# # CLI Interface
# # ============================================

# def main():
#     parser = argparse.ArgumentParser(
#         description="Make predictions using best models",
#         formatter_class=argparse.RawDescriptionHelpFormatter,
#         epilog="""
# Examples:
#   # Predict all targets
#   python predict_with_best_models.py --input data/test_data.csv --output predictions.csv
  
#   # Predict specific target
#   python predict_with_best_models.py --input data/test_data.csv --target revenue
  
#   # With ground truth for evaluation
#   python predict_with_best_models.py --input data/test_data.csv --evaluate
#         """
#     )
    
#     parser.add_argument(
#         "--input",
#         type=str,
#         default="data/splits/test_data.csv",
#         help="Input CSV file with features (default: data/splits/test_data.csv)"
#     )
#     parser.add_argument(
#         "--output",
#         type=str,
#         default=None,
#         help="Output CSV file for predictions (optional)"
#     )
#     parser.add_argument(
#         "--target",
#         type=str,
#         nargs="+",
#         default=["all"],
#         help="Target(s) to predict (default: all)"
#     )
#     parser.add_argument(
#         "--models-dir",
#         type=str,
#         default="models/best_models",
#         help="Directory containing best models"
#     )
#     parser.add_argument(
#         "--evaluate",
#         action="store_true",
#         help="Evaluate predictions if ground truth is available in input file"
#     )
    
#     args = parser.parse_args()
    
#     # Load input data
#     print(f"\n{'='*80}")
#     print(f"LOADING INPUT DATA")
#     print(f"{'='*80}\n")
    
#     try:
#         input_df = pd.read_csv(args.input)
#         print(f"Loaded {len(input_df):,} rows from {args.input}")
#     except Exception as e:
#         print(f"Error loading input file: {str(e)}")
#         return
    
#     # Initialize predictor
#     predictor = BestModelPredictor(models_dir=args.models_dir)
    
#     # Determine targets
#     all_targets = ["revenue", "eps", "debt_equity", "profit_margin", "stock_return"]
    
#     if "all" in args.target:
#         targets_to_predict = all_targets
#     else:
#         targets_to_predict = [t for t in args.target if t in all_targets]
    
#     # Load models
#     for target in targets_to_predict:
#         predictor.load_model(target)
    
#     if not predictor.models:
#         print(f"\nNo models loaded. Cannot make predictions.")
#         return
    
#     # Show model info
#     predictor.print_model_info()
    
#     # Make predictions
#     print(f"\n{'='*80}")
#     print(f"MAKING PREDICTIONS")
#     print(f"{'='*80}\n")
    
#     try:
#         predictions_dict = predictor.predict_all(input_df, return_dict=False)
        
#         print(f"Predictions generated for {len(predictions_dict.columns)} targets")
#         print(f"\nPrediction summary:")
#         print(predictions_dict.describe())
        
#         # Save predictions
#         if args.output:
#             output_df = input_df.copy()
            
#             # Add predictions
#             for col in predictions_dict.columns:
#                 output_df[f"pred_{col}"] = predictions_dict[col]
            
#             output_df.to_csv(args.output, index=False)
#             print(f"\nPredictions saved to: {args.output}")
        
#         # Evaluate if requested
#         if args.evaluate:
#             print(f"\n{'='*80}")
#             print(f"EVALUATING PREDICTIONS")
#             print(f"{'='*80}\n")
            
#             # Check if ground truth is available
#             y_true = {}
#             for target in predictor.models.keys():
#                 target_col = f"target_{target}"
#                 if target_col in input_df.columns:
#                     y_true[target] = input_df[target_col]
            
#             if y_true:
#                 eval_df = predictor.evaluate_predictions(input_df, y_true)
                
#                 print(f"\n{'Target':<20} {'Model':<20} {'N':>8} {'RMSE':>12} {'MAE':>12} {'R2':>10}")
#                 print(f"{'─'*90}")
                
#                 for _, row in eval_df.iterrows():
#                     print(f"{row['target']:<20} {row['model_type']:<20} {row['n_samples']:>8} "
#                           f"{row['rmse']:>12,.2f} {row['mae']:>12,.2f} {row['r2']:>10.4f}")
                
#                 if len(eval_df) > 1:
#                     avg_r2 = eval_df["r2"].mean()
#                     print(f"{'─'*90}")
#                     print(f"{'AVERAGE':<20} {'':20} {'':8} {'':12} {'':12} {avg_r2:>10.4f}")
#             else:
#                 print(f"No ground truth found in input file")
#                 print(f"   Expected columns: target_revenue, target_eps, etc.")
        
#         print(f"\n{'='*80}")
#         print(f"PREDICTION COMPLETE")
#         print(f"{'='*80}")
        
#     except Exception as e:
#         print(f"\nError during prediction: {str(e)}")
#         import traceback
#         traceback.print_exc()


# if __name__ == "__main__":
#     main()




#--------------------------------------------


"""
src/models/unified_model_training_and_selection.py

UNIFIED TRAINING PIPELINE: Train ALL models, automatically select BEST for each target

This script:
1. Trains 4 model types for each target (XGBoost, XGBoost-tuned, LightGBM, LightGBM-tuned)
2. Evaluates all models on test set
3. Selects best model based on Test R² (with overfitting consideration)
4. Saves best model as {target}_best.pkl
5. Generates comprehensive comparison report

Usage:
    # Train all targets with all models
    python src/models/unified_model_training_and_selection.py
    
    # Train specific target
    python src/models/unified_model_training_and_selection.py --target profit_margin
    
    # Quick mode (skip tuning for faster results)
    python src/models/unified_model_training_and_selection.py --quick
    
    # Custom number of tuning trials
    python src/models/unified_model_training_and_selection.py --trials 20
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
import lightgbm as lgb
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import optuna
from typing import Dict, Tuple, List

warnings.filterwarnings("ignore")

# Setup paths
project_root = Path(__file__).resolve().parent.parent.parent.parent  
sys.path.insert(0, str(project_root))

from src.utils.split_utils import get_feature_target_split, drop_nan_targets

print("✅ Imports successful\n")


# ============================================
# Model Trainers (Base Classes)
# ============================================

class BaseModelTrainer:
    """Base class for all model trainers"""
    
    def __init__(self, target_name: str):
        self.target_name = target_name
        self.target_col = f"target_{target_name}"
        self.model = None
        self.feature_names = None
        self.train_metrics = None
        self.val_metrics = None
        self.test_metrics = None
    
    def load_and_prepare_data(self, splits_dir: str):
        """Load and prepare data (common for all models)"""
        splits_path = Path(splits_dir)
        
        train_df = pd.read_csv(splits_path / "train_data.csv")
        val_df = pd.read_csv(splits_path / "val_data.csv")
        test_df = pd.read_csv(splits_path / "test_data.csv")
        
        # Prepare features
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
    
    def evaluate_all_splits(self, X_train, y_train, X_val, y_val, X_test, y_test):
        """Evaluate model on all splits"""
        results = {}
        
        for name, X, y in [("train", X_train, y_train), ("val", X_val, y_val), ("test", X_test, y_test)]:
            pred = self.predict(X)
            
            results[name] = {
                "rmse": float(np.sqrt(mean_squared_error(y, pred))),
                "mae": float(mean_absolute_error(y, pred)),
                "r2": float(r2_score(y, pred))
            }
        
        self.train_metrics = results["train"]
        self.val_metrics = results["val"]
        self.test_metrics = results["test"]
        
        return results
    
    def predict(self, X):
        """Prediction method (to be implemented by subclasses)"""
        raise NotImplementedError
    
    def get_model_info(self):
        """Get model information for reporting"""
        return {
            "model_type": self.__class__.__name__,
            "train": self.train_metrics,
            "val": self.val_metrics,
            "test": self.test_metrics,
            "n_features": len(self.feature_names) if self.feature_names else 0
        }


class XGBoostTrainer(BaseModelTrainer):
    """XGBoost baseline trainer"""
    
    def __init__(self, target_name: str):
        super().__init__(target_name)
        self.model_type = "xgboost"
    
    def train(self, X_train, y_train, X_val, y_val):
        """Train XGBoost baseline"""
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
        
        self.model = xgb.XGBRegressor(**params)
        self.model.set_params(early_stopping_rounds=50)
        
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        return self
    
    def predict(self, X):
        return self.model.predict(X)


class XGBoostTunedTrainer(BaseModelTrainer):
    """XGBoost with Optuna tuning"""
    
    def __init__(self, target_name: str):
        super().__init__(target_name)
        self.model_type = "xgboost_tuned"
        self.best_params = None
    
    def train(self, X_train, y_train, X_val, y_val, n_trials=30):
        """Train XGBoost with hyperparameter tuning"""
        
        def objective(trial):
            params = {
                "max_depth": trial.suggest_int("max_depth", 3, 12),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.3, log=True),
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=30),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.6, 1.0),
                "gamma": trial.suggest_float("gamma", 0.0, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 2.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 2.0),
                "random_state": 42,
                "n_jobs": -1,
                "tree_method": "hist",
                "verbosity": 0,
            }
            
            model = xgb.XGBRegressor(**params)
            model.set_params(early_stopping_rounds=30)
            
            try:
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
                val_pred = model.predict(X_val)
                val_r2 = r2_score(y_val, val_pred)
                return -val_r2
            except Exception:
                raise optuna.TrialPruned()
        
        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=10)
        )
        
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        
        self.best_params = study.best_params
        self.model = xgb.XGBRegressor(**self.best_params, random_state=42, n_jobs=-1, tree_method="hist", verbosity=0)
        self.model.set_params(early_stopping_rounds=30)
        self.model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        
        return self
    
    def predict(self, X):
        return self.model.predict(X)


class LightGBMTrainer(BaseModelTrainer):
    """LightGBM baseline trainer"""
    
    def __init__(self, target_name: str):
        super().__init__(target_name)
        self.model_type = "lightgbm"
    
    def train(self, X_train, y_train, X_val, y_val):
        """Train LightGBM baseline"""
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
        }
        
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        callbacks = [
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=0)
        ]
        
        self.model = lgb.train(
            params,
            train_data,
            valid_sets=[train_data, val_data],
            valid_names=['train', 'val'],
            num_boost_round=500,
            callbacks=callbacks
        )
        
        return self
    
    def predict(self, X):
        return self.model.predict(X, num_iteration=self.model.best_iteration)


class LightGBMTunedTrainer(BaseModelTrainer):
    """LightGBM with Optuna tuning"""
    
    def __init__(self, target_name: str):
        super().__init__(target_name)
        self.model_type = "lightgbm_tuned"
        self.best_params = None
    
    def train(self, X_train, y_train, X_val, y_val, n_trials=30):
        """Train LightGBM with hyperparameter tuning"""
        
        def objective(trial):
            params = {
                'num_leaves': trial.suggest_int('num_leaves', 20, 100),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=50),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'subsample_freq': trial.suggest_int('subsample_freq', 1, 5),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 2.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 2.0),
                'random_state': 42,
                'n_jobs': -1,
                'verbose': -1,
            }
            
            model = lgb.LGBMRegressor(**params)
            
            try:
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric='rmse')
                val_pred = model.predict(X_val)
                val_r2 = r2_score(y_val, val_pred)
                return -val_r2
            except Exception:
                raise optuna.TrialPruned()
        
        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=10)
        )
        
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        
        self.best_params = study.best_params
        self.model = lgb.LGBMRegressor(**self.best_params, random_state=42, n_jobs=-1, verbose=-1)
        self.model.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric='rmse')
        
        return self
    
    def predict(self, X):
        return self.model.predict(X)


# ============================================
# Unified Training Pipeline
# ============================================

class UnifiedModelPipeline:
    """
    Unified pipeline that trains all models and selects the best
    """
    
    def __init__(self, splits_dir: str = "data/splits", output_dir: str = "models/best_models"):
        self.splits_dir = splits_dir
        self.output_dir = output_dir
        self.results = {}
        self.best_models = {}
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    def train_all_models_for_target(self, target: str, n_trials: int = 30, skip_tuning: bool = False):
        """
        Train all 4 model types for a single target
        
        Args:
            target: Target variable name
            n_trials: Number of Optuna trials for tuned models
            skip_tuning: If True, only train baseline models (faster)
        
        Returns:
            Dictionary with results for all models
        """
        print(f"\n{'='*80}")
        print(f"🎯 TRAINING ALL MODELS: {target.upper()}")
        print(f"{'='*80}\n")
        
        target_results = {}
        
        # Initialize all trainers
        trainers = [
            ("XGBoost", XGBoostTrainer(target)),
            ("LightGBM", LightGBMTrainer(target)),
        ]
        
        if not skip_tuning:
            trainers.extend([
                ("XGBoost-Tuned", XGBoostTunedTrainer(target)),
                ("LightGBM-Tuned", LightGBMTunedTrainer(target)),
            ])
        
        # Load data once
        print(f"📂 Loading data...")
        trainer = trainers[0][1]
        X_train, y_train, X_val, y_val, X_test, y_test = trainer.load_and_prepare_data(self.splits_dir)
        print(f"   ✅ Train: {len(X_train):,}, Val: {len(X_val):,}, Test: {len(X_test):,}")
        
        # Train each model
        for model_name, trainer in trainers:
            print(f"\n{'─'*80}")
            print(f"🚀 Training: {model_name}")
            print(f"{'─'*80}")
            
            try:
                # Load data for this trainer
                X_train, y_train, X_val, y_val, X_test, y_test = trainer.load_and_prepare_data(self.splits_dir)
                
                # Train
                start_time = pd.Timestamp.now()
                if "Tuned" in model_name:
                    trainer.train(X_train, y_train, X_val, y_val, n_trials=n_trials)
                else:
                    trainer.train(X_train, y_train, X_val, y_val)
                
                training_time = (pd.Timestamp.now() - start_time).total_seconds()
                
                # Evaluate
                results = trainer.evaluate_all_splits(X_train, y_train, X_val, y_val, X_test, y_test)
                
                # Store results
                target_results[model_name] = {
                    "trainer": trainer,
                    "results": results,
                    "training_time": training_time,
                    "model_type": trainer.model_type
                }
                
                # Print summary
                print(f"   ✅ Train R²: {results['train']['r2']:.4f}")
                print(f"   ✅ Val R²:   {results['val']['r2']:.4f}")
                print(f"   ✅ Test R²:  {results['test']['r2']:.4f}")
                print(f"   ⏱️  Time:    {training_time:.1f}s")
                
            except Exception as e:
                print(f"   ❌ ERROR: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
        
        return target_results
    
    def select_best_model(self, target: str, target_results: Dict):
        """
        Select best model based on multiple criteria
        
        Selection Criteria (in order):
        1. Test R² (primary)
        2. Overfitting gap < 30% (quality check)
        3. RMSE (tiebreaker)
        
        Args:
            target: Target variable name
            target_results: Results from all models
        
        Returns:
            Best model name and selection reasoning
        """
        print(f"\n{'='*80}")
        print(f"🏆 MODEL SELECTION: {target.upper()}")
        print(f"{'='*80}\n")
        
        # Create comparison table
        comparison = []
        for model_name, data in target_results.items():
            results = data["results"]
            
            train_r2 = results["train"]["r2"]
            test_r2 = results["test"]["r2"]
            overfit_gap = train_r2 - test_r2
            overfit_pct = (overfit_gap / train_r2 * 100) if train_r2 > 0 else 0
            
            comparison.append({
                "model": model_name,
                "test_r2": test_r2,
                "train_r2": train_r2,
                "overfit_gap": overfit_gap,
                "overfit_pct": overfit_pct,
                "test_rmse": results["test"]["rmse"],
                "trainer": data["trainer"]
            })
        
        # Sort by test R² (descending)
        comparison_sorted = sorted(comparison, key=lambda x: x["test_r2"], reverse=True)
        
        # Print comparison table
        print(f"{'Model':<20} {'Test R²':>10} {'Train R²':>10} {'Overfit %':>12} {'Test RMSE':>12}")
        print(f"{'─'*70}")
        
        for item in comparison_sorted:
            overfit_symbol = "✅" if item["overfit_pct"] < 30 else "⚠️"
            print(f"{item['model']:<20} {item['test_r2']:>10.4f} {item['train_r2']:>10.4f} "
                  f"{overfit_symbol} {item['overfit_pct']:>10.1f}% {item['test_rmse']:>12,.2f}")
        
        # Selection logic
        best = comparison_sorted[0]
        
        # Check for excessive overfitting
        if best["overfit_pct"] > 30:
            print(f"\n⚠️  WARNING: Best model has {best['overfit_pct']:.1f}% overfitting!")
            
            # Look for alternative with less overfitting
            alternatives = [c for c in comparison_sorted[1:] if c["overfit_pct"] < 30]
            if alternatives:
                alternative = alternatives[0]
                r2_sacrifice = best["test_r2"] - alternative["test_r2"]
                r2_sacrifice_pct = (r2_sacrifice / best["test_r2"]) * 100
                
                if r2_sacrifice_pct < 3.0:  # Accept up to 3% R² sacrifice for better generalization
                    print(f"   → Switching to {alternative['model']} (better generalization)")
                    print(f"   → R² sacrifice: {r2_sacrifice:.4f} ({r2_sacrifice_pct:.1f}%)")
                    best = alternative
        
        print(f"\n{'─'*80}")
        print(f"🏆 SELECTED: {best['model']}")
        print(f"{'─'*80}")
        print(f"   Test R²: {best['test_r2']:.4f}")
        print(f"   Overfit: {best['overfit_pct']:.1f}%")
        
        # Selection reasoning
        reasoning = f"Selected {best['model']} with Test R²={best['test_r2']:.4f}. "
        if best["overfit_pct"] < 20:
            reasoning += "Excellent generalization."
        elif best["overfit_pct"] < 30:
            reasoning += "Good generalization."
        else:
            reasoning += f"Warning: {best['overfit_pct']:.1f}% overfitting detected."
        
        return best["model"], best["trainer"], reasoning, comparison_sorted
    
    def save_best_model(self, target: str, model_name: str, trainer: BaseModelTrainer, reasoning: str):
        """
        Save best model as {target}_best.pkl
        
        Args:
            target: Target variable name
            model_name: Name of selected model
            trainer: Trained model object
            reasoning: Selection reasoning
        """
        output_file = Path(self.output_dir) / f"{target}_best.pkl"
        
        model_data = {
            "target": target,
            "model_type": model_name,
            "model": trainer.model,
            "feature_names": trainer.feature_names,
            "train_metrics": trainer.train_metrics,
            "val_metrics": trainer.val_metrics,
            "test_metrics": trainer.test_metrics,
            "selection_reasoning": reasoning,
            "timestamp": datetime.now().isoformat()
        }
        
        # Add model-specific parameters
        if hasattr(trainer, 'best_params'):
            model_data["hyperparameters"] = trainer.best_params
        
        joblib.dump(model_data, output_file)
        
        print(f"\n💾 Model saved: {output_file}")
        
        return output_file
    
    def generate_comparison_report(self, all_results: Dict, output_dir: str = None):
        """
        Generate comprehensive comparison report for all targets
        
        Args:
            all_results: Results from all targets
            output_dir: Output directory (default: self.output_dir)
        """
        if output_dir is None:
            output_dir = self.output_dir
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*80}")
        print(f"📊 GENERATING COMPARISON REPORT")
        print(f"{'='*80}\n")
        
        # Prepare report data
        report = {
            "timestamp": datetime.now().isoformat(),
            "total_targets": len(all_results),
            "targets": {}
        }
        
        summary_table = []
        
        for target, data in all_results.items():
            best_model_name = data["best_model"]
            best_trainer = data["best_trainer"]
            comparison = data["comparison"]
            
            # Target-specific report
            report["targets"][target] = {
                "selected_model": best_model_name,
                "selection_reasoning": data["reasoning"],
                "test_r2": best_trainer.test_metrics["r2"],
                "test_rmse": best_trainer.test_metrics["rmse"],
                "test_mae": best_trainer.test_metrics["mae"],
                "all_models": [
                    {
                        "model": c["model"],
                        "test_r2": c["test_r2"],
                        "overfit_pct": c["overfit_pct"]
                    }
                    for c in comparison
                ]
            }
            
            summary_table.append({
                "target": target,
                "model": best_model_name,
                "test_r2": best_trainer.test_metrics["r2"],
                "test_rmse": best_trainer.test_metrics["rmse"]
            })
        
        # Save JSON report
        report_file = output_path / "model_comparison_report.json"
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)
        
        print(f"   ✅ JSON report: {report_file}")
        
        # Create summary table
        print(f"\n{'='*80}")
        print(f"📊 FINAL MODEL SUMMARY")
        print(f"{'='*80}\n")
        
        print(f"{'Target':<20} {'Selected Model':<20} {'Test R²':>10} {'Test RMSE':>15}")
        print(f"{'─'*70}")
        
        for row in summary_table:
            print(f"{row['target']:<20} {row['model']:<20} {row['test_r2']:>10.4f} {row['test_rmse']:>15,.2f}")
        
        # Average performance
        avg_r2 = np.mean([r["test_r2"] for r in summary_table])
        print(f"{'─'*70}")
        print(f"{'AVERAGE':<20} {'':20} {avg_r2:>10.4f}")
        
        # Save markdown summary
        md_content = self._create_markdown_summary(report, summary_table)
        md_file = output_path / "MODEL_SELECTION_SUMMARY.md"
        with open(md_file, "w") as f:
            f.write(md_content)
        
        print(f"\n   ✅ Markdown summary: {md_file}")
        
        return report_file
    
    def _create_markdown_summary(self, report: Dict, summary_table: List[Dict]) -> str:
        """Create markdown summary"""
        
        md = "# Unified Model Training & Selection Report\n\n"
        md += f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        md += "## Overview\n\n"
        md += "This report summarizes the unified model training pipeline where all 4 model types "
        md += "(XGBoost, XGBoost-Tuned, LightGBM, LightGBM-Tuned) were trained for each target, "
        md += "and the best model was automatically selected based on Test R² and generalization quality.\n\n"
        
        md += "## Selection Criteria\n\n"
        md += "1. **Primary:** Test R² (higher is better)\n"
        md += "2. **Quality Check:** Overfitting gap < 30% (train-test R² difference)\n"
        md += "3. **Tiebreaker:** Test RMSE (lower is better)\n\n"
        
        md += "## Selected Models Summary\n\n"
        md += "| Target | Selected Model | Test R² | Test RMSE | Notes |\n"
        md += "|--------|---------------|---------|-----------|-------|\n"
        
        for row in summary_table:
            target_data = report["targets"][row["target"]]
            reasoning = target_data["selection_reasoning"]
            
            # Shortened reasoning for table
            notes = "✅ Best" if "Excellent" in reasoning else "⚠️ Check" if "Warning" in reasoning else "✓ Good"
            
            md += f"| {row['target']} | {row['model']} | {row['test_r2']:.4f} | "
            md += f"{row['test_rmse']:,.2f} | {notes} |\n"
        
        avg_r2 = np.mean([r["test_r2"] for r in summary_table])
        md += f"| **AVERAGE** | - | **{avg_r2:.4f}** | - | - |\n\n"
        
        md += "## Detailed Selection Reasoning\n\n"
        
        for target, data in report["targets"].items():
            md += f"### {target.upper()}\n\n"
            md += f"**Selected Model:** {data['selected_model']}\n\n"
            md += f"**Reasoning:** {data['selection_reasoning']}\n\n"
            
            md += "**All Models Comparison:**\n\n"
            md += "| Model | Test R² | Overfitting % |\n"
            md += "|-------|---------|---------------|\n"
            
            for model_data in data["all_models"]:
                md += f"| {model_data['model']} | {model_data['test_r2']:.4f} | {model_data['overfit_pct']:.1f}% |\n"
            
            md += "\n---\n\n"
        
        md += "## Model Files\n\n"
        md += "Best models saved to: `models/best_models/`\n\n"
        
        for target in report["targets"].keys():
            md += f"- `{target}_best.pkl`\n"
        
        md += "\n## Usage\n\n"
        md += "```python\n"
        md += "import joblib\n\n"
        md += "# Load best model for a target\n"
        md += "model_data = joblib.load('models/best_models/revenue_best.pkl')\n"
        md += "model = model_data['model']\n"
        md += "feature_names = model_data['feature_names']\n\n"
        md += "# Make predictions\n"
        md += "predictions = model.predict(X_new)\n"
        md += "```\n"
        
        return md
    
    def run_pipeline(self, targets: List[str], n_trials: int = 30, skip_tuning: bool = False):
        """
        Run complete pipeline for all targets
        
        Args:
            targets: List of target variable names
            n_trials: Number of Optuna trials for tuned models
            skip_tuning: If True, only train baseline models
        
        Returns:
            Dictionary with all results
        """
        print(f"\n{'='*80}")
        print(f"🚀 UNIFIED MODEL TRAINING PIPELINE")
        print(f"{'='*80}")
        print(f"Targets: {', '.join(targets)}")
        print(f"Models: XGBoost, LightGBM" + ("" if skip_tuning else ", XGBoost-Tuned, LightGBM-Tuned"))
        print(f"Tuning trials: {n_trials if not skip_tuning else 'N/A (skipped)'}")
        print(f"{'='*80}\n")
        
        all_results = {}
        
        for i, target in enumerate(targets, 1):
            print(f"\n{'#'*80}")
            print(f"TARGET {i}/{len(targets)}: {target.upper()}")
            print(f"{'#'*80}")
            
            try:
                # Train all models for this target
                target_results = self.train_all_models_for_target(target, n_trials, skip_tuning)
                
                # Select best model
                best_model_name, best_trainer, reasoning, comparison = self.select_best_model(target, target_results)
                
                # Save best model
                model_file = self.save_best_model(target, best_model_name, best_trainer, reasoning)
                
                # Store results
                all_results[target] = {
                    "best_model": best_model_name,
                    "best_trainer": best_trainer,
                    "reasoning": reasoning,
                    "comparison": comparison,
                    "model_file": str(model_file),
                    "all_models": target_results
                }
                
                print(f"\n✅ {target.upper()} COMPLETE")
                
            except Exception as e:
                print(f"\n❌ ERROR processing {target}:")
                print(f"   {str(e)}")
                import traceback
                traceback.print_exc()
                continue
        
        # Generate comparison report
        if all_results:
            self.generate_comparison_report(all_results)
        
        return all_results


# ============================================
# CLI Interface
# ============================================

def main():
    parser = argparse.ArgumentParser(
        description="Unified model training and selection pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train all targets with all models (default)
  python unified_model_training_and_selection.py
  
  # Train specific target
  python unified_model_training_and_selection.py --target profit_margin
  
  # Quick mode (skip tuned models for faster results)
  python unified_model_training_and_selection.py --quick
  
  # Custom number of tuning trials
  python unified_model_training_and_selection.py --trials 20
  
  # Multiple specific targets
  python unified_model_training_and_selection.py --target revenue eps
        """
    )
    
    parser.add_argument(
        "--target",
        type=str,
        nargs="+",
        default=["all"],
        help="Target(s) to train (default: all). Options: revenue, eps, debt_equity, profit_margin, stock_return, all"
    )
    parser.add_argument(
        "--splits-dir",
        type=str,
        default="data/splits",
        help="Directory containing train/val/test splits"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/best_models",
        help="Directory to save best models"
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=30,
        help="Number of Optuna trials for tuned models (default: 30)"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: skip tuned models (faster, only baseline XGBoost and LightGBM)"
    )
    
    args = parser.parse_args()
    
    # Determine targets
    all_targets = ["revenue", "eps", "debt_equity", "profit_margin", "stock_return"]
    
    if "all" in args.target:
        targets = all_targets
    else:
        targets = [t for t in args.target if t in all_targets]
        if not targets:
            print(f"❌ Invalid target(s): {args.target}")
            print(f"   Valid options: {', '.join(all_targets)}, all")
            return
    
    # Initialize pipeline
    pipeline = UnifiedModelPipeline(
        splits_dir=args.splits_dir,
        output_dir=args.output_dir
    )
    
    # Run pipeline
    results = pipeline.run_pipeline(
        targets=targets,
        n_trials=args.trials,
        skip_tuning=args.quick
    )
    
    # Final summary
    if results:
        print(f"\n{'='*80}")
        print(f"✅ PIPELINE COMPLETE")
        print(f"{'='*80}")
        print(f"\n📁 Best models saved to: {args.output_dir}/")
        print(f"\nFiles created:")
        for target in results.keys():
            print(f"   - {target}_best.pkl")
        print(f"   - model_comparison_report.json")
        print(f"   - MODEL_SELECTION_SUMMARY.md")
        
        print(f"\n🎯 Next steps:")
        print(f"   1. Review model_comparison_report.json")
        print(f"   2. Load and test best models")
        print(f"   3. Deploy to production")
        
        print(f"\n💡 To use a best model:")
        print(f"   ```python")
        print(f"   import joblib")
        print(f"   model_data = joblib.load('{args.output_dir}/revenue_best.pkl')")
        print(f"   model = model_data['model']")
        print(f"   predictions = model.predict(X_new)")
        print(f"   ```")
    else:
        print(f"\n❌ No models trained successfully")


if __name__ == "__main__":
    main()

