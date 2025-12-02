#!/usr/bin/env python3
"""
============================================================================
Model 3: Anomaly Detection - Complete Training Pipeline
============================================================================
Author: Parth Saraykar
Requirements: Follows Model Development Guidelines
  - Data loading from pipeline
  - Model training & selection
  - Hyperparameter tuning
  - MLflow experiment tracking
from typing import Dict, List, Tuple, Optional, Any
from typing import Dict, List, Tuple, Optional, Any
  - Model validation
  - Bias detection (sector slicing)
  - Sensitivity analysis (SHAP)
  - Model registry push
============================================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path
import yaml
import logging
from datetime import datetime
import warnings
import sys
import pickle
import json
from typing import Dict, List, Tuple, Optional, Any

# ML libraries
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, average_precision_score
)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# MLflow
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

# SHAP for sensitivity analysis
import shap

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Configuration loader"""
    
    def __init__(self, config_path: str = "configs/model_config.yaml"):
        with open(config_path, 'r') as f:
            self.cfg = yaml.safe_load(f)
    
    def __getitem__(self, key):
        return self.cfg[key]


# ============================================================================
# LOGGER
# ============================================================================

def setup_logger():
    """Setup comprehensive logger"""
    Path("logs").mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'logs/model_training_{timestamp}.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


# ============================================================================
# STEP 1: LOAD DATA FROM PIPELINE
# ============================================================================

class DataLoader:
    """Load data from Snorkel labeling pipeline"""
    
    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
    
    def load_labeled_data(self) -> pd.DataFrame:
        """Load labeled data from Snorkel pipeline output"""
        self.logger.info("="*80)
        self.logger.info("STEP 1: LOADING DATA FROM PIPELINE")
        self.logger.info("="*80)
        
        data_path = self.config['data']['labeled_data_path']
        self.logger.info(f"Loading from: {data_path}")
        
        df = pd.read_csv(data_path)
        df['Date'] = pd.to_datetime(df[self.config['data']['date_column']])
        df['Year'] = df['Date'].dt.year
        df['Quarter'] = df['Date'].dt.quarter
        
        self.logger.info(f"✓ Loaded {len(df):,} labeled samples")
        self.logger.info(f"  Date range: {df['Date'].min()} to {df['Date'].max()}")
        self.logger.info(f"  Companies: {df[self.config['data']['company_column']].nunique()}")
        self.logger.info(f"  Sectors: {df[self.config['data']['sector_column']].nunique()}")
        
        # Label distribution
        at_risk = (df[self.config['data']['target_column']] == 1).sum()
        self.logger.info(f"  AT_RISK: {at_risk:,} ({at_risk/len(df):.2%})")
        self.logger.info(f"  NOT_AT_RISK: {len(df)-at_risk:,} ({(len(df)-at_risk)/len(df):.2%})")
        
        return df
    
    def temporal_split(self, df: pd.DataFrame):
        """Temporal train/val/test split"""
        self.logger.info("\nPerforming temporal split...")
        
        train_mask = df['Year'] <= self.config['split']['train_end_year']
        val_mask = (df['Year'] >= self.config['split']['val_start_year']) & \
                   (df['Year'] <= self.config['split']['val_end_year'])
        test_mask = df['Year'] >= self.config['split']['test_start_year']
        
        train_df = df[train_mask].copy()
        val_df = df[val_mask].copy()
        test_df = df[test_mask].copy()
        
        self.logger.info(f"✓ Train: {len(train_df):,} samples (≤{self.config['split']['train_end_year']})")
        self.logger.info(f"✓ Val:   {len(val_df):,} samples ({self.config['split']['val_start_year']}-{self.config['split']['val_end_year']})")
        self.logger.info(f"✓ Test:  {len(test_df):,} samples (≥{self.config['split']['test_start_year']})")
        
        return train_df, val_df, test_df
    
    def prepare_features(self, train_df, val_df, test_df):
        """Prepare features and target"""
        self.logger.info("\nPreparing features...")
        
        # Get feature columns
        feature_cols = []
        for category in self.config['data']['feature_columns'].values():
            feature_cols.extend(category)
        
        self.logger.info(f"Using {len(feature_cols)} features")
        
        # Extract features and target
        X_train = train_df[feature_cols].copy()
        X_val = val_df[feature_cols].copy()
        X_test = test_df[feature_cols].copy()
        
        y_train = train_df[self.config['data']['target_column']].values
        y_val = val_df[self.config['data']['target_column']].values
        y_test = test_df[self.config['data']['target_column']].values
        
        # Handle missing values
        X_train = X_train.fillna(X_train.median())
        X_val = X_val.fillna(X_train.median())
        X_test = X_test.fillna(X_train.median())
        
        # Handle inf
        X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(X_train.median())
        X_val = X_val.replace([np.inf, -np.inf], np.nan).fillna(X_train.median())
        X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(X_train.median())
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        self.logger.info(f"✓ Features scaled using StandardScaler")
        self.logger.info(f"✓ Train at-risk rate: {y_train.sum()/len(y_train):.2%}")
        self.logger.info(f"✓ Val at-risk rate: {y_val.sum()/len(y_val):.2%}")
        
        return (X_train_scaled, y_train), (X_val_scaled, y_val), (X_test_scaled, y_test), scaler, feature_cols


# ============================================================================
# STEP 2: HYPERPARAMETER TUNING
# ============================================================================

class HyperparameterTuner:
    """Hyperparameter tuning for anomaly detection models"""
    
    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
    
    def tune_isolation_forest(self, X_train, y_train, X_val, y_val):
        """Grid search for Isolation Forest"""
        self.logger.info("\nHyperparameter tuning: Isolation Forest")
        
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_samples': [128, 256, 'auto'],
            'contamination': [0.02, 0.035, 0.05],
            'max_features': [0.8, 1.0]
        }
        
        best_score = -np.inf
        best_params = None
        
        for n_est in param_grid['n_estimators']:
            for max_samp in param_grid['max_samples']:
                for cont in param_grid['contamination']:
                    for max_feat in param_grid['max_features']:
                        
                        model = IsolationForest(
                            n_estimators=n_est,
                            max_samples=max_samp,
                            contamination=cont,
                            max_features=max_feat,
                            random_state=42
                        )
                        
                        model.fit(X_train)
                        scores = model.score_samples(X_val)
                        
                        # Evaluate ROC-AUC
                        try:
                            roc_auc = roc_auc_score(y_val, -scores)
                            
                            if roc_auc > best_score:
                                best_score = roc_auc
                                best_params = {
                                    'n_estimators': n_est,
                                    'max_samples': max_samp,
                                    'contamination': cont,
                                    'max_features': max_feat
                                }
                        except:
                            continue
        
        self.logger.info(f"✓ Best params: {best_params}")
        self.logger.info(f"✓ Best ROC-AUC: {best_score:.4f}")
        
        return best_params


# ============================================================================
# STEP 3: MODEL TRAINING WITH MLFLOW
# ============================================================================

class ModelTrainer:
    """Train anomaly detection models with MLflow tracking"""
    
    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
    
    def train_isolation_forest(self, X_train, y_train, X_val, y_val, params=None):
        """Train Isolation Forest with MLflow"""
        self.logger.info("\n" + "="*80)
        self.logger.info("TRAINING: ISOLATION FOREST")
        self.logger.info("="*80)
        
        if params is None:
            params = self.config['models']['isolation_forest']['params']
        
        with mlflow.start_run(run_name="isolation_forest") as run:
            # Log parameters
            mlflow.log_params(params)
            mlflow.set_tags(self.config['mlflow']['tags'])
            
            # Train
            model = IsolationForest(**params)
            model.fit(X_train)
            
            self.logger.info("✓ Model trained")
            
            # Predict
            y_pred_val = model.predict(X_val)
            scores_val = model.score_samples(X_val)
            y_pred_binary = (y_pred_val == -1).astype(int)
            
            # Evaluate
            metrics = self._evaluate(y_val, y_pred_binary, scores_val)
            mlflow.log_metrics(metrics)
            
            # Log model with signature
            signature = infer_signature(X_train, y_pred_binary[:len(X_train)])
            mlflow.sklearn.log_model(model, "model", signature=signature)
            
            # Log model info
            run_id = run.info.run_id
            
            self.logger.info(f"✓ ROC-AUC: {metrics['roc_auc']:.4f}")
            self.logger.info(f"✓ Precision@10%: {metrics['precision_at_10pct']:.4f}")
            self.logger.info(f"✓ MLflow Run ID: {run_id}")
            
            return model, metrics, run_id
    
    def train_lof(self, X_train, y_train, X_val, y_val):
        """Train LOF with MLflow"""
        self.logger.info("\n" + "="*80)
        self.logger.info("TRAINING: LOCAL OUTLIER FACTOR")
        self.logger.info("="*80)
        
        params = self.config['models']['local_outlier_factor']['params']
        
        with mlflow.start_run(run_name="local_outlier_factor") as run:
            mlflow.log_params(params)
            mlflow.set_tags(self.config['mlflow']['tags'])
            
            model = LocalOutlierFactor(**params)
            model.fit(X_train)
            
            y_pred_val = model.predict(X_val)
            scores_val = model.score_samples(X_val)
            y_pred_binary = (y_pred_val == -1).astype(int)
            
            metrics = self._evaluate(y_val, y_pred_binary, scores_val)
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(model, "model")
            
            run_id = run.info.run_id
            
            self.logger.info(f"✓ ROC-AUC: {metrics['roc_auc']:.4f}")
            self.logger.info(f"✓ Precision@10%: {metrics['precision_at_10pct']:.4f}")
            
            return model, metrics, run_id
    
    def train_one_class_svm(self, X_train, y_train, X_val, y_val):
        """Train One-Class SVM with MLflow"""
        self.logger.info("\n" + "="*80)
        self.logger.info("TRAINING: ONE-CLASS SVM")
        self.logger.info("="*80)
        
        params = self.config['models']['one_class_svm']['params']
        
        with mlflow.start_run(run_name="one_class_svm") as run:
            mlflow.log_params(params)
            mlflow.set_tags(self.config['mlflow']['tags'])
            
            model = OneClassSVM(**params)
            model.fit(X_train)
            
            y_pred_val = model.predict(X_val)
            scores_val = model.score_samples(X_val)
            y_pred_binary = (y_pred_val == -1).astype(int)
            
            metrics = self._evaluate(y_val, y_pred_binary, scores_val)
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(model, "model")
            
            run_id = run.info.run_id
            
            self.logger.info(f"✓ ROC-AUC: {metrics['roc_auc']:.4f}")
            self.logger.info(f"✓ Precision@10%: {metrics['precision_at_10pct']:.4f}")
            
            return model, metrics, run_id
    
    def train_dbscan(self, X_train, y_train, X_val, y_val):
        """Train DBSCAN with MLflow"""
        self.logger.info("\n" + "="*80)
        self.logger.info("TRAINING: DBSCAN")
        self.logger.info("="*80)
        
        params = self.config['models']['dbscan']['params']
        
        with mlflow.start_run(run_name="dbscan") as run:
            mlflow.log_params(params)
            mlflow.set_tags(self.config['mlflow']['tags'])
            
            model = DBSCAN(**params)
            clusters = model.fit_predict(X_val)
            
            # Noise points (cluster -1) are anomalies
            y_pred_binary = (clusters == -1).astype(int)
            scores_val = -np.ones(len(X_val))
            scores_val[clusters != -1] = 1
            
            metrics = self._evaluate(y_val, y_pred_binary, scores_val)
            mlflow.log_metrics(metrics)
            
            # Log cluster info
            n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
            n_noise = list(clusters).count(-1)
            mlflow.log_param("n_clusters", n_clusters)
            mlflow.log_param("n_noise_points", n_noise)
            
            run_id = run.info.run_id
            
            self.logger.info(f"✓ Clusters: {n_clusters}, Noise: {n_noise}")
            self.logger.info(f"✓ ROC-AUC: {metrics['roc_auc']:.4f}")
            
            return model, metrics, run_id
    
    def _evaluate(self, y_true, y_pred, scores):
        """Evaluate model performance"""
        
        metrics = {}
        
        # ROC-AUC
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, -scores)
        except:
            metrics['roc_auc'] = 0.0
        
        # Standard metrics
        metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
        metrics['f1_score'] = f1_score(y_true, y_pred, zero_division=0)
        
        # Average Precision
        try:
            metrics['average_precision'] = average_precision_score(y_true, -scores)
        except:
            metrics['average_precision'] = 0.0
        
        # Precision@K and Recall@K
        k = int(len(y_true) * 0.10)
        top_k_idx = np.argsort(scores)[:k]
        top_k_preds = np.zeros(len(y_true))
        top_k_preds[top_k_idx] = 1
        
        metrics['precision_at_10pct'] = precision_score(y_true, top_k_preds, zero_division=0)
        metrics['recall_at_10pct'] = recall_score(y_true, top_k_preds, zero_division=0)
        
        # Confusion matrix components
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics['true_positives'] = int(tp)
        metrics['true_negatives'] = int(tn)
        metrics['false_positives'] = int(fp)
        metrics['false_negatives'] = int(fn)
        
        return metrics


# ============================================================================
# STEP 4: MODEL VALIDATION
# ============================================================================

class ModelValidator:
    """Comprehensive model validation"""
    
    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
    
    def validate_against_targets(self, results: Dict) -> Dict:
        """Validate models against performance targets"""
        self.logger.info("\n" + "="*80)
        self.logger.info("STEP 4: MODEL VALIDATION AGAINST TARGETS")
        self.logger.info("="*80)
        
        targets = self.config['evaluation']['targets']
        validation_results = {}
        
        for model_name, result in results.items():
            metrics = result['metrics']
            
            meets_roc_auc = metrics['roc_auc'] >= targets['roc_auc_min']
            meets_precision = metrics['precision_at_10pct'] >= targets['precision_at_k_min']
            
            passes = meets_roc_auc and meets_precision
            
            validation_results[model_name] = {
                'passes': passes,
                'roc_auc_check': meets_roc_auc,
                'precision_check': meets_precision
            }
            
            status = "✓ PASS" if passes else "✗ FAIL"
            self.logger.info(f"\n{model_name}: {status}")
            self.logger.info(f"  ROC-AUC: {metrics['roc_auc']:.4f} (target: ≥{targets['roc_auc_min']}) {'✓' if meets_roc_auc else '✗'}")
            self.logger.info(f"  Precision@10%: {metrics['precision_at_10pct']:.4f} (target: ≥{targets['precision_at_k_min']}) {'✓' if meets_precision else '✗'}")
        
        return validation_results


# ============================================================================
# STEP 5: BIAS DETECTION (SECTOR SLICING)
# ============================================================================

class BiasDetector:
    """Detect bias using sector slicing"""
    
    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
    
    def detect_sector_bias(self, model, X_val, y_val, val_df, model_name):
        """Detect bias across sectors"""
        self.logger.info("\n" + "="*80)
        self.logger.info(f"BIAS DETECTION: {model_name} (Sector Slicing)")
        self.logger.info("="*80)
        
        sector_col = self.config['data']['sector_column']
        sectors = val_df[sector_col].values
        
        # Predict
        if hasattr(model, 'predict'):
            y_pred = model.predict(X_val)
            scores = model.score_samples(X_val) if hasattr(model, 'score_samples') else None
        else:
            # DBSCAN
            clusters = model.fit_predict(X_val)
            y_pred = (clusters == -1).astype(int)
            scores = None
        
        y_pred_binary = (y_pred == -1).astype(int) if not isinstance(y_pred[0], (int, np.integer)) else y_pred
        
        # Analyze by sector
        unique_sectors = np.unique(sectors)
        sector_metrics = []
        
        for sector in unique_sectors:
            sector_mask = sectors == sector
            
            if sector_mask.sum() == 0:
                continue
            
            y_true_sector = y_val[sector_mask]
            y_pred_sector = y_pred_binary[sector_mask]
            
            # Metrics
            # Handle cases where sector has only one class
            try:
                if len(np.unique(y_true_sector)) < 2 or len(np.unique(y_pred_sector)) < 2:
                    precision = 0.0
                    recall = 0.0
                    f1 = 0.0
                else:
                    precision = precision_score(y_true_sector, y_pred_sector, zero_division=0)
                    recall = recall_score(y_true_sector, y_pred_sector, zero_division=0)
                    f1 = f1_score(y_true_sector, y_pred_sector, zero_division=0)
            except:
                precision = 0.0
                recall = 0.0
                f1 = 0.0
            
            
            sector_metrics.append({
                'Sector': sector,
                'Samples': sector_mask.sum(),
                'True_At_Risk': y_true_sector.sum(),
                'Predicted_At_Risk': y_pred_sector.sum(),
                'Precision': precision,
                'Recall': recall,
                'F1_Score': f1
            })
        
        sector_df = pd.DataFrame(sector_metrics)
        
        # Calculate disparity
        f1_std = sector_df['F1_Score'].std()
        precision_std = sector_df['Precision'].std()
        
        self.logger.info(f"\nSector Performance Summary:")
        self.logger.info(f"  F1-Score Std Dev: {f1_std:.4f}")
        self.logger.info(f"  Precision Std Dev: {precision_std:.4f}")
        
        # Flag high disparity
        bias_threshold = 0.15
        if f1_std > bias_threshold or precision_std > bias_threshold:
            self.logger.warning(f"⚠️  HIGH DISPARITY DETECTED across sectors!")
            self.logger.warning(f"   Consider bias mitigation strategies")
        else:
            self.logger.info(f"✓ Low disparity - model is fair across sectors")
        
        # Log to MLflow
        mlflow.log_metric("sector_f1_std", f1_std)
        mlflow.log_metric("sector_precision_std", precision_std)
        mlflow.log_param("bias_check_passed", f1_std <= bias_threshold and precision_std <= bias_threshold)
        
        # Save sector analysis
        sector_path = Path(self.config['output']['results_dir']) / f"{model_name}_sector_analysis.csv"
        sector_df.to_csv(sector_path, index=False)
        mlflow.log_artifact(str(sector_path))
        
        self.logger.info(f"✓ Sector analysis saved: {sector_path}")
        
        return sector_df, f1_std <= bias_threshold


# ============================================================================
# STEP 6: SENSITIVITY ANALYSIS (SHAP)
# ============================================================================

class SensitivityAnalyzer:
    """Perform sensitivity analysis using SHAP"""
    
    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
    
    def analyze_feature_importance(self, model, X_val, feature_names, model_name):
        """SHAP-based feature importance"""
        self.logger.info("\n" + "="*80)
        self.logger.info(f"SENSITIVITY ANALYSIS: {model_name} (SHAP)")
        self.logger.info("="*80)
        
        try:
            # Use TreeExplainer for tree-based models
            if 'Forest' in model_name:
                # Use a wrapper for anomaly score
                def model_predict(X):
                    return -model.score_samples(X)
                
                # Sample for speed (SHAP is slow)
                X_sample = X_val[:min(100, len(X_val))]
                
                explainer = shap.Explainer(model_predict, X_sample)
                shap_values = explainer(X_sample)
                
                # Plot
                fig, ax = plt.subplots(figsize=(10, 8))
                shap.summary_plot(shap_values, X_sample, feature_names=feature_names, 
                                 show=False, max_display=15)
                plt.tight_layout()
                
                shap_path = Path(self.config['output']['plots_dir']) / f"{model_name}_shap_summary.png"
                plt.savefig(shap_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                self.logger.info(f"✓ SHAP analysis saved: {shap_path}")
                mlflow.log_artifact(str(shap_path))
                
                # Get feature importance
                importance = np.abs(shap_values.values).mean(axis=0)
                feature_importance = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': importance
                }).sort_values('Importance', ascending=False)
                
                self.logger.info(f"\nTop 5 Most Important Features:")
                for idx, row in feature_importance.head(5).iterrows():
                    self.logger.info(f"  {row['Feature']:30s}: {row['Importance']:.4f}")
                
                # Save importance
                importance_path = Path(self.config['output']['results_dir']) / f"{model_name}_feature_importance.csv"
                feature_importance.to_csv(importance_path, index=False)
                mlflow.log_artifact(str(importance_path))
                
                return feature_importance
                
        except Exception as e:
            self.logger.warning(f"Could not perform SHAP analysis: {str(e)}")
            return None


# ============================================================================
# STEP 7: MODEL SELECTION
# ============================================================================

class ModelSelector:
    """Select best model based on validation and bias"""
    
    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
    
    def select_best_model(self, results: Dict, validation_results: Dict):
        """Select best model considering performance and bias"""
        self.logger.info("\n" + "="*80)
        self.logger.info("STEP 7: MODEL SELECTION")
        self.logger.info("="*80)
        
        # Filter to models that pass validation
        passing_models = {
            name: result for name, result in results.items()
            if validation_results[name]['passes']
        }
        
        if not passing_models:
            self.logger.warning("⚠️  No models passed validation targets!")
            self.logger.info("Selecting best performing model anyway...")
            passing_models = results
        
        # Select best by ROC-AUC
        best_model_name = max(
            passing_models.keys(),
            key=lambda x: passing_models[x]['metrics']['roc_auc']
        )
        
        best_result = passing_models[best_model_name]
        
        self.logger.info(f"\n✓ BEST MODEL: {best_model_name}")
        self.logger.info(f"  ROC-AUC: {best_result['metrics']['roc_auc']:.4f}")
        self.logger.info(f"  Precision@10%: {best_result['metrics']['precision_at_10pct']:.4f}")
        self.logger.info(f"  F1-Score: {best_result['metrics']['f1_score']:.4f}")
        self.logger.info(f"  Bias Check: {'✓ Passed' if best_result.get('bias_passed', True) else '✗ Failed'}")
        
        return best_model_name, best_result


# ============================================================================
# STEP 8: MODEL REGISTRY PUSH
# ============================================================================

class ModelRegistryManager:
    """Push model to MLflow Model Registry"""
    
    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
    
    def register_model(self, model_name: str, run_id: str):
        """Register model in MLflow Model Registry"""
        self.logger.info("\n" + "="*80)
        self.logger.info("STEP 8: PUSHING TO MODEL REGISTRY")
        self.logger.info("="*80)
        
        try:
            model_uri = f"runs:/{run_id}/model"
            registered_model_name = f"financial_stress_{model_name.lower()}"
            
            # Register model
            model_version = mlflow.register_model(
                model_uri=model_uri,
                name=registered_model_name
            )
            
            self.logger.info(f"✓ Model registered: {registered_model_name}")
            self.logger.info(f"✓ Version: {model_version.version}")
            self.logger.info(f"✓ Run ID: {run_id}")
            
            # Transition to staging
            client = mlflow.tracking.MlflowClient()
            client.transition_model_version_stage(
                name=registered_model_name,
                version=model_version.version,
                stage="Staging"
            )
            
            self.logger.info(f"✓ Model promoted to Staging")
            
            return model_version
            
        except Exception as e:
            self.logger.error(f"Error registering model: {str(e)}")
            return None
    
    def save_model_locally(self, model, model_name: str, scaler, feature_cols):
        """Save model artifacts locally"""
        self.logger.info(f"\nSaving {model_name} artifacts locally...")
        
        model_dir = Path(self.config['output']['models_dir']) / model_name
        model_dir.mkdir(exist_ok=True, parents=True)
        
        # Save model
        model_path = model_dir / "model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        # Save scaler
        scaler_path = model_dir / "scaler.pkl"
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        
        # Save feature names
        features_path = model_dir / "features.json"
        with open(features_path, 'w') as f:
            json.dump({'features': feature_cols}, f)
        
        self.logger.info(f"✓ Saved model artifacts to: {model_dir}")


# ============================================================================
# VISUALIZATIONS
# ============================================================================

class Visualizer:
    """Create comprehensive visualizations"""
    
    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
        Path(self.config['output']['plots_dir']).mkdir(exist_ok=True, parents=True)
    
    def plot_model_comparison(self, results: Dict):
        """Compare all models"""
        self.logger.info("\nCreating model comparison plot...")
        
        model_names = list(results.keys())
        roc_aucs = [results[m]['metrics']['roc_auc'] for m in model_names]
        precisions_at_k = [results[m]['metrics']['precision_at_10pct'] for m in model_names]
        f1_scores = [results[m]['metrics']['f1_score'] for m in model_names]
        recalls = [results[m]['metrics']['recall'] for m in model_names]
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # ROC-AUC
        axes[0, 0].bar(model_names, roc_aucs, color='steelblue', edgecolor='black')
        axes[0, 0].axhline(y=0.85, color='red', linestyle='--', linewidth=2, label='Target: 0.85')
        axes[0, 0].set_ylabel('ROC-AUC', fontsize=12)
        axes[0, 0].set_title('ROC-AUC by Model', fontsize=14, fontweight='bold')
        axes[0, 0].set_ylim([0, 1])
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3, axis='y')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Precision@10%
        axes[0, 1].bar(model_names, precisions_at_k, color='coral', edgecolor='black')
        axes[0, 1].axhline(y=0.80, color='red', linestyle='--', linewidth=2, label='Target: 0.80')
        axes[0, 1].set_ylabel('Precision@10%', fontsize=12)
        axes[0, 1].set_title('Precision@10% by Model', fontsize=14, fontweight='bold')
        axes[0, 1].set_ylim([0, 1])
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # F1-Score
        axes[1, 0].bar(model_names, f1_scores, color='lightgreen', edgecolor='black')
        axes[1, 0].set_ylabel('F1-Score', fontsize=12)
        axes[1, 0].set_title('F1-Score by Model', fontsize=14, fontweight='bold')
        axes[1, 0].set_ylim([0, 1])
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Recall
        axes[1, 1].bar(model_names, recalls, color='gold', edgecolor='black')
        axes[1, 1].set_ylabel('Recall', fontsize=12)
        axes[1, 1].set_title('Recall by Model', fontsize=14, fontweight='bold')
        axes[1, 1].set_ylim([0, 1])
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        output_path = Path(self.config['output']['plots_dir']) / "model_comparison.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"✓ Saved: {output_path}")
        mlflow.log_artifact(str(output_path))
    
    def plot_confusion_matrices(self, results: Dict, y_val):
        """Plot confusion matrices"""
        self.logger.info("Creating confusion matrices...")
        
        n_models = len(results)
        fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 4))
        
        if n_models == 1:
            axes = [axes]
        
        for idx, (model_name, result) in enumerate(results.items()):
            metrics = result['metrics']
            cm = [[metrics['true_negatives'], metrics['false_positives']],
                  [metrics['false_negatives'], metrics['true_positives']]]
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       ax=axes[idx], cbar=False,
                       xticklabels=['NOT_AT_RISK', 'AT_RISK'],
                       yticklabels=['NOT_AT_RISK', 'AT_RISK'])
            axes[idx].set_title(f'{model_name}', fontsize=12, fontweight='bold')
            axes[idx].set_ylabel('True', fontsize=10)
            axes[idx].set_xlabel('Predicted', fontsize=10)
        
        plt.tight_layout()
        
        output_path = Path(self.config['output']['plots_dir']) / "confusion_matrices.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"✓ Saved: {output_path}")
        mlflow.log_artifact(str(output_path))
    
    def plot_sector_bias(self, sector_df, model_name):
        """Plot sector bias analysis"""
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        
        # Precision by sector
        axes[0].barh(sector_df['Sector'], sector_df['Precision'], color='steelblue', edgecolor='black')
        axes[0].set_xlabel('Precision', fontsize=11)
        axes[0].set_title(f'{model_name}: Precision by Sector', fontsize=12, fontweight='bold')
        axes[0].grid(True, alpha=0.3, axis='x')
        
        # Recall by sector
        axes[1].barh(sector_df['Sector'], sector_df['Recall'], color='coral', edgecolor='black')
        axes[1].set_xlabel('Recall', fontsize=11)
        axes[1].set_title(f'{model_name}: Recall by Sector', fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3, axis='x')
        
        # F1 by sector
        axes[2].barh(sector_df['Sector'], sector_df['F1_Score'], color='lightgreen', edgecolor='black')
        axes[2].set_xlabel('F1-Score', fontsize=11)
        axes[2].set_title(f'{model_name}: F1-Score by Sector', fontsize=12, fontweight='bold')
        axes[2].grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        output_path = Path(self.config['output']['plots_dir']) / f"{model_name}_sector_bias.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"✓ Saved sector bias plot: {output_path}")
        mlflow.log_artifact(str(output_path))


# ============================================================================
# REPORT GENERATION
# ============================================================================

def generate_comprehensive_report(results, validation_results, best_model_info, config, logger):
    """Generate final comprehensive report"""
    logger.info("\n" + "="*80)
    logger.info("GENERATING COMPREHENSIVE REPORT")
    logger.info("="*80)
    
    Path(config['output']['reports_dir']).mkdir(exist_ok=True, parents=True)
    
    report = f"""
{'='*80}
MODEL 3: ANOMALY DETECTION - COMPREHENSIVE TRAINING REPORT
{'='*80}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Author: Parth Saraykar
MLOps Innovation Expo - Financial Stress Test Generator

1. MODELS TRAINED
{'='*80}
Total Models: {len(results)}
"""
    
    for model_name, result in results.items():
        metrics = result['metrics']
        report += f"\n{model_name}:\n"
        report += f"  ROC-AUC:           {metrics['roc_auc']:.4f}\n"
        report += f"  Precision@10%:     {metrics['precision_at_10pct']:.4f}\n"
        report += f"  Recall@10%:        {metrics['recall_at_10pct']:.4f}\n"
        report += f"  F1-Score:          {metrics['f1_score']:.4f}\n"
        report += f"  Precision:         {metrics['precision']:.4f}\n"
        report += f"  Recall:            {metrics['recall']:.4f}\n"
        report += f"  MLflow Run ID:     {result['run_id']}\n"
    
    report += f"""
2. VALIDATION RESULTS
{'='*80}
Performance Targets:
  - ROC-AUC ≥ 0.85
  - Precision@10% ≥ 0.80

Model Validation Status:
"""
    
    for model_name, val_result in validation_results.items():
        status = "✓ PASS" if val_result['passes'] else "✗ FAIL"
        report += f"  {model_name:20s}: {status}\n"
    
    report += f"""
3. BEST MODEL SELECTED
{'='*80}
Model: {best_model_info['name']}
ROC-AUC: {best_model_info['metrics']['roc_auc']:.4f}
Precision@10%: {best_model_info['metrics']['precision_at_10pct']:.4f}
F1-Score: {best_model_info['metrics']['f1_score']:.4f}
Bias Check: {' ✓ Passed' if best_model_info.get('bias_passed', True) else '✗ Failed'}
MLflow Run ID: {best_model_info['run_id']}

4. BIAS DETECTION SUMMARY
{'='*80}
Slicing Dimension: Sector
Bias Metric: F1-Score Standard Deviation across sectors
Result: {'✓ No significant bias detected' if best_model_info.get('bias_passed', True) else '⚠️ Bias detected - mitigation recommended'}

5. SENSITIVITY ANALYSIS
{'='*80}
Method: SHAP (SHapley Additive exPlanations)
See outputs/models/plots/{best_model_info['name']}_shap_summary.png

6. MODEL REGISTRY
{'='*80}
Status: {'✓ Registered in MLflow Model Registry' if best_model_info.get('registered', False) else 'Local artifacts saved only'}
Local Path: models/anomaly_detection/{best_model_info['name']}/

7. OUTPUTS GENERATED
{'='*80}
✓ Model comparison plots
✓ Confusion matrices
✓ Sector bias analysis
✓ SHAP feature importance
✓ Model artifacts (model.pkl, scaler.pkl, features.json)
✓ MLflow experiment logs

8. NEXT STEPS
{'='*80}
1. Review MLflow UI: mlflow ui --port 5000
2. Examine sector bias analysis: outputs/models/results/*_sector_analysis.csv
3. Review SHAP plots: outputs/models/plots/*_shap_summary.png
4. Integrate with teammates' models (VAE/GAN, XGBoost/LSTM)
5. Deploy to production or proceed to ensemble model

{'='*80}
END OF REPORT
{'='*80}
"""
    
    report_path = Path(config['output']['reports_dir']) / "comprehensive_training_report.txt"
    with open(report_path, 'w') as f:
        f.write(report)
    
    logger.info(f"✓ Report saved: {report_path}")
    print("\n" + report)
    
    return report


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """Complete training pipeline with all requirements"""
    
    # Setup
    logger = setup_logger()
    config = Config()
    
    logger.info("="*80)
    logger.info("MODEL 3: ANOMALY DETECTION - COMPREHENSIVE TRAINING PIPELINE")
    logger.info("="*80)
    logger.info("Requirements Implemented:")
    logger.info("  ✓ Data loading from pipeline")
    logger.info("  ✓ Model training & selection")
    logger.info("  ✓ Hyperparameter tuning")
    logger.info("  ✓ MLflow experiment tracking")
    logger.info("  ✓ Model validation")
    logger.info("  ✓ Bias detection (sector slicing)")
    logger.info("  ✓ Sensitivity analysis (SHAP)")
    logger.info("  ✓ Model registry push")
    logger.info("="*80)
    
    # Create output directories
    for d in [config['output']['models_dir'], config['output']['results_dir'],
              config['output']['plots_dir'], config['output']['reports_dir']]:
        Path(d).mkdir(exist_ok=True, parents=True)
    
    # Setup MLflow
    mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
    mlflow.set_experiment(config['mlflow']['experiment_name'])
    
    # Initialize components
    data_loader = DataLoader(config, logger)
    tuner = HyperparameterTuner(config, logger)
    trainer = ModelTrainer(config, logger)
    validator = ModelValidator(config, logger)
    bias_detector = BiasDetector(config, logger)
    sensitivity_analyzer = SensitivityAnalyzer(config, logger)
    model_selector = ModelSelector(config, logger)
    registry_manager = ModelRegistryManager(config, logger)
    visualizer = Visualizer(config, logger)
    
    # STEP 1: Load data
    df = data_loader.load_labeled_data()
    train_df, val_df, test_df = data_loader.temporal_split(df)
    (X_train, y_train), (X_val, y_val), (X_test, y_test), scaler, feature_cols = \
        data_loader.prepare_features(train_df, val_df, test_df)
    
    # STEP 2: Hyperparameter tuning (optional - Isolation Forest only)
    logger.info("\n" + "="*80)
    logger.info("STEP 2: HYPERPARAMETER TUNING")
    logger.info("="*80)
    best_if_params = tuner.tune_isolation_forest(X_train, y_train, X_val, y_val)
    
    # STEP 3: Train all models
    logger.info("\n" + "="*80)
    logger.info("STEP 3: TRAINING MODELS WITH MLFLOW")
    logger.info("="*80)
    
    results = {}
    
    # Isolation Forest (with tuned params)
    if config['models']['isolation_forest']['enabled']:
        model, metrics, run_id = trainer.train_isolation_forest(
            X_train, y_train, X_val, y_val, params=best_if_params
        )
        results['Isolation_Forest'] = {
            'model': model, 'metrics': metrics, 'run_id': run_id
        }
    
    # LOF
    if config['models']['local_outlier_factor']['enabled']:
        model, metrics, run_id = trainer.train_lof(X_train, y_train, X_val, y_val)
        results['LOF'] = {
            'model': model, 'metrics': metrics, 'run_id': run_id
        }
    
    # One-Class SVM
    if config['models']['one_class_svm']['enabled']:
        model, metrics, run_id = trainer.train_one_class_svm(X_train, y_train, X_val, y_val)
        results['One_Class_SVM'] = {
            'model': model, 'metrics': metrics, 'run_id': run_id
        }
    
    # DBSCAN
    if config['models']['dbscan']['enabled']:
        model, metrics, run_id = trainer.train_dbscan(X_train, y_train, X_val, y_val)
        results['DBSCAN'] = {
            'model': model, 'metrics': metrics, 'run_id': run_id
        }
    
    # STEP 4: Validation
    validation_results = validator.validate_against_targets(results)
    
    # STEP 5: Bias detection
    logger.info("\n" + "="*80)
    logger.info("STEP 5: BIAS DETECTION (SECTOR SLICING)")
    logger.info("="*80)
    
    for model_name, result in results.items():
        sector_df, bias_passed = bias_detector.detect_sector_bias(
            result['model'], X_val, y_val, val_df, model_name
        )
        result['bias_passed'] = bias_passed
        result['sector_analysis'] = sector_df
        
        # Visualize bias
        visualizer.plot_sector_bias(sector_df, model_name)
    
    # STEP 6: Sensitivity analysis
    logger.info("\n" + "="*80)
    logger.info("STEP 6: SENSITIVITY ANALYSIS (SHAP)")
    logger.info("="*80)
    
    for model_name, result in results.items():
        if 'Forest' in model_name:  # SHAP works best with tree models
            importance = sensitivity_analyzer.analyze_feature_importance(
                result['model'], X_val, feature_cols, model_name
            )
            result['feature_importance'] = importance
    
    # STEP 7: Select best model
    best_model_name, best_result = model_selector.select_best_model(results, validation_results)
    
    # STEP 8: Register best model
    registered_version = registry_manager.register_model(best_model_name, best_result['run_id'])
    best_result['registered'] = registered_version is not None
    
    # Save all models locally
    for model_name, result in results.items():
        registry_manager.save_model_locally(
            result['model'], model_name, scaler, feature_cols
        )
    
    # STEP 9: Visualizations
    logger.info("\n" + "="*80)
    logger.info("STEP 9: GENERATING VISUALIZATIONS")
    logger.info("="*80)
    
    visualizer.plot_model_comparison(results)
    visualizer.plot_confusion_matrices(results, y_val)
    
    # STEP 10: Final report
    best_model_info = {
        'name': best_model_name,
        'metrics': best_result['metrics'],
        'run_id': best_result['run_id'],
        'bias_passed': best_result['bias_passed'],
        'registered': best_result.get('registered', False)
    }
    
    report = generate_comprehensive_report(
        results, validation_results, best_model_info, config, logger
    )
    
    logger.info("\n" + "="*80)
    logger.info("✓ TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
    logger.info("="*80)
    logger.info("\nView results:")
    logger.info("  1. MLflow UI:      mlflow ui --port 5000")
    logger.info("  2. Report:         cat outputs/models/reports/comprehensive_training_report.txt")
    logger.info("  3. Visualizations: outputs/models/plots/")
    logger.info("  4. Model artifacts: models/anomaly_detection/")


if __name__ == "__main__":
    main()