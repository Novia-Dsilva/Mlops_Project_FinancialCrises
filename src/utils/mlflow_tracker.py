"""
src/utils/mlflow_tracker.py
MLflow tracking utilities for XGBoost, Linear, and LSTM models
"""
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.pytorch
from pathlib import Path
from typing import Dict, Any, Optional
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class MLflowTracker:
    """
    Unified MLflow tracker for all model types
    """
    
    def __init__(self, experiment_name: str, tracking_uri: str = "mlruns"):
        """
        Initialize MLflow tracker
        
        Args:
            experiment_name: Name of the MLflow experiment
            tracking_uri: Path to MLflow tracking directory
        """
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        
        # Set tracking URI
        mlflow.set_tracking_uri(tracking_uri)  # ← FIXED
        
        # Set or create experiment
        mlflow.set_experiment(experiment_name)  # ← FIXED
        
        print(f"✅ MLflow initialized")
        print(f"   Experiment: {experiment_name}")
        print(f"   Tracking URI: {tracking_uri}")
    
    def start_run(self, run_name: str, tags: Optional[Dict[str, str]] = None):
        """
        Start an MLflow run
        
        Args:
            run_name: Name for this run
            tags: Optional tags to add to the run
        """
        mlflow.start_run(run_name=run_name)  # ← FIXED
        
        # Set tags
        if tags:
            mlflow.set_tags(tags)  # ← FIXED
        
        print(f"\n🔄 Started MLflow run: {run_name}")
    
    def log_params(self, params: Dict[str, Any]):
        """
        Log parameters
        
        Args:
            params: Dictionary of parameters to log
        """
        for key, value in params.items():
            mlflow.log_param(key, value)  # ← FIXED
        
        print(f"   ✅ Logged {len(params)} parameters")
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        Log metrics
        
        Args:
            metrics: Dictionary of metrics to log
            step: Optional step number (for iterative logging)
        """
        for key, value in metrics.items():
            mlflow.log_metric(key, value, step=step)  # ← FIXED
        
        print(f"   ✅ Logged {len(metrics)} metrics")
    
    def log_model(self, model, model_type: str, artifact_path: str = "model"):
        """
        Log model to MLflow
        
        Args:
            model: Trained model object
            model_type: Type of model ('xgboost', 'sklearn', 'pytorch')
            artifact_path: Path within run to store model
        """
        if model_type == "xgboost":
            mlflow.xgboost.log_model(model, artifact_path)  # ← FIXED
        elif model_type == "sklearn":
            mlflow.sklearn.log_model(model, artifact_path)  # ← FIXED
        elif model_type == "pytorch":
            mlflow.pytorch.log_model(model, artifact_path)  # ← FIXED
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        print(f"   ✅ Logged {model_type} model")
    
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """
        Log a local file as an artifact
        
        Args:
            local_path: Path to local file
            artifact_path: Optional subdirectory within artifacts
        """
        mlflow.log_artifact(local_path, artifact_path)  # ← FIXED
        print(f"   ✅ Logged artifact: {local_path}")
    
    def log_dict(self, dictionary: Dict, filename: str):
        """
        Log a dictionary as JSON artifact
        
        Args:
            dictionary: Dictionary to save
            filename: Name of JSON file
        """
        mlflow.log_dict(dictionary, filename)  # ← FIXED
        print(f"   ✅ Logged dictionary: {filename}")
    
    def log_figure(self, fig, filename: str):
        """
        Log a matplotlib figure
        
        Args:
            fig: Matplotlib figure object
            filename: Name to save figure as
        """
        mlflow.log_figure(fig, filename)  # ← FIXED
        print(f"   ✅ Logged figure: {filename}")
    
    def end_run(self):
        """End the current MLflow run"""
        mlflow.end_run()  # ← FIXED
        print("   ✅ MLflow run ended\n")
    
    def log_training_metrics_over_time(self, history: list, metric_names: list):
        """
        Log training metrics over epochs/iterations
        
        Args:
            history: List of dictionaries with metrics at each step
            metric_names: List of metric names to log
        """
        for i, step_metrics in enumerate(history):
            for metric_name in metric_names:
                if metric_name in step_metrics:
                    mlflow.log_metric(metric_name, step_metrics[metric_name], step=i)  # ← FIXED
        
        print(f"   ✅ Logged {len(history)} steps of training history")
    
    def create_feature_importance_plot(
        self, 
        importance_df: pd.DataFrame, 
        top_n: int = 20,
        title: str = "Feature Importance"
    ):
        """
        Create and log feature importance plot
        
        Args:
            importance_df: DataFrame with 'feature' and 'importance' columns
            top_n: Number of top features to plot
            title: Plot title
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Get top N features
        top_features = importance_df.head(top_n)
        
        # Create horizontal bar plot
        ax.barh(range(len(top_features)), top_features['importance'])
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features['feature'])
        ax.invert_yaxis()
        ax.set_xlabel('Importance')
        ax.set_title(title)
        
        plt.tight_layout()
        
        # Log figure
        self.log_figure(fig, "feature_importance.png")
        plt.close(fig)
    
    def create_predictions_plot(
        self,
        y_true,
        y_pred,
        title: str = "Predictions vs Actual",
        dataset_name: str = "test"
    ):
        """
        Create and log predictions vs actual plot
        
        Args:
            y_true: True values
            y_pred: Predicted values
            title: Plot title
            dataset_name: Name of dataset (for filename)
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Scatter plot
        ax1.scatter(y_true, y_pred, alpha=0.5)
        ax1.plot([y_true.min(), y_true.max()], 
                [y_true.min(), y_true.max()], 
                'r--', lw=2)
        ax1.set_xlabel('Actual')
        ax1.set_ylabel('Predicted')
        ax1.set_title(f'{title} - Scatter')
        ax1.grid(True, alpha=0.3)
        
        # Residuals plot
        residuals = y_true - y_pred
        ax2.scatter(y_pred, residuals, alpha=0.5)
        ax2.axhline(y=0, color='r', linestyle='--', lw=2)
        ax2.set_xlabel('Predicted')
        ax2.set_ylabel('Residuals')
        ax2.set_title(f'{title} - Residuals')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Log figure
        self.log_figure(fig, f"predictions_{dataset_name}.png")
        plt.close(fig)
    
    def create_training_history_plot(
        self,
        history: list,
        metrics: list = ['train_loss', 'val_loss'],
        title: str = "Training History"
    ):
        """
        Create and log training history plot (for LSTM)
        
        Args:
            history: List of dictionaries with metrics at each epoch
            metrics: List of metrics to plot
            title: Plot title
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Extract metrics
        epochs = [h['epoch'] for h in history]
        
        for metric in metrics:
            values = [h[metric] for h in history if metric in h]
            if values:
                ax.plot(epochs[:len(values)], values, label=metric, marker='o')
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Log figure
        self.log_figure(fig, "training_history.png")
        plt.close(fig)


# ============================================
# Model-specific tracking functions
# ============================================

def track_xgboost_run(
    tracker: MLflowTracker,
    target_name: str,
    trainer,
    X_test,
    y_test
):
    """
    Track XGBoost model training
    
    Args:
        tracker: MLflowTracker instance
        target_name: Name of target variable
        trainer: XGBoostTrainer instance
        X_test: Test features
        y_test: Test targets
    """
    run_name = f"xgboost_{target_name}"
    
    tracker.start_run(
        run_name=run_name,
        tags={
            "model_type": "xgboost",
            "target": target_name,
            "framework": "xgboost"
        }
    )
    
    try:
        # Log parameters
        tracker.log_params({
            "model_type": "xgboost",
            "target": target_name,
            "n_features": len(trainer.feature_names),
            **trainer.params
        })
        
        # Log metrics
        all_metrics = {
            "train_rmse": trainer.train_metrics['rmse'],
            "train_mae": trainer.train_metrics['mae'],
            "train_r2": trainer.train_metrics['r2'],
            "val_rmse": trainer.val_metrics['rmse'],
            "val_mae": trainer.val_metrics['mae'],
            "val_r2": trainer.val_metrics['r2'],
            "test_rmse": trainer.test_metrics['rmse'],
            "test_mae": trainer.test_metrics['mae'],
            "test_r2": trainer.test_metrics['r2'],
            "best_iteration": trainer.best_iteration,
        }
        tracker.log_metrics(all_metrics)
        
        # Log model
        tracker.log_model(trainer.model, "xgboost")
        
        # Log feature importance
        importance_df = trainer.get_feature_importance(top_n=50)
        if not importance_df.empty:
            tracker.create_feature_importance_plot(
                importance_df,
                top_n=20,
                title=f"XGBoost Feature Importance - {target_name}"
            )
        
        # Log predictions plot
        test_pred = trainer.model.predict(X_test)
        tracker.create_predictions_plot(
            y_test.values,
            test_pred,
            title=f"XGBoost Predictions - {target_name}",
            dataset_name="test"
        )
        
    finally:
        tracker.end_run()


def track_linear_run(
    tracker: MLflowTracker,
    target_name: str,
    trainer,
    X_test,
    y_test
):
    """
    Track Linear model training
    
    Args:
        tracker: MLflowTracker instance
        target_name: Name of target variable
        trainer: LinearModelTrainer instance
        X_test: Test features
        y_test: Test targets
    """
    run_name = f"linear_{target_name}"
    
    tracker.start_run(
        run_name=run_name,
        tags={
            "model_type": "linear",
            "target": target_name,
            "framework": "sklearn"
        }
    )
    
    try:
        # Log parameters
        tracker.log_params({
            "model_type": "ridge_regression",
            "target": target_name,
            "n_features": len(trainer.feature_names),
            "alpha": trainer.alpha,
            "use_cv": trainer.use_cv,
        })
        
        # Log metrics
        all_metrics = {
            "train_rmse": trainer.train_metrics['rmse'],
            "train_mae": trainer.train_metrics['mae'],
            "train_r2": trainer.train_metrics['r2'],
            "val_rmse": trainer.val_metrics['rmse'],
            "val_mae": trainer.val_metrics['mae'],
            "val_r2": trainer.val_metrics['r2'],
            "test_rmse": trainer.test_metrics['rmse'],
            "test_mae": trainer.test_metrics['mae'],
            "test_r2": trainer.test_metrics['r2'],
        }
        tracker.log_metrics(all_metrics)
        
        # Log model (scaler + model together)
        tracker.log_model(trainer.model, "sklearn")
        
        # Log feature importance (coefficients)
        importance_df = trainer.get_feature_importance(top_n=50)
        if not importance_df.empty:
            tracker.create_feature_importance_plot(
                importance_df,
                top_n=20,
                title=f"Linear Feature Importance - {target_name}"
            )
        
        # Log predictions plot
        test_pred = trainer.predict(X_test)
        tracker.create_predictions_plot(
            y_test.values,
            test_pred,
            title=f"Linear Predictions - {target_name}",
            dataset_name="test"
        )
        
    finally:
        tracker.end_run()


def track_lstm_run(
    tracker: MLflowTracker,
    target_name: str,
    trainer,
    X_test,
    y_test
):
    """
    Track LSTM model training
    
    Args:
        tracker: MLflowTracker instance
        target_name: Name of target variable
        trainer: LSTMTrainer instance
        X_test: Test features
        y_test: Test targets
    """
    run_name = f"lstm_{target_name}"
    
    tracker.start_run(
        run_name=run_name,
        tags={
            "model_type": "lstm",
            "target": target_name,
            "framework": "pytorch"
        }
    )
    
    try:
        # Log parameters
        tracker.log_params({
            "model_type": "lstm",
            "target": target_name,
            "n_features": len(trainer.feature_names),
            "sequence_length": trainer.sequence_length,
            "hidden_size": trainer.hidden_size,
            "num_layers": trainer.num_layers,
            "learning_rate": trainer.learning_rate,
            "batch_size": trainer.batch_size,
            "epochs": trainer.epochs,
            "patience": trainer.patience,
        })
        
        # Log metrics
        all_metrics = {
            "val_rmse": trainer.val_metrics['rmse'],
            "val_mae": trainer.val_metrics['mae'],
            "val_r2": trainer.val_metrics['r2'],
            "test_rmse": trainer.test_metrics['rmse'],
            "test_mae": trainer.test_metrics['mae'],
            "test_r2": trainer.test_metrics['r2'],
            "best_epoch": trainer.best_epoch,
            "total_epochs_trained": len(trainer.training_history),
        }
        tracker.log_metrics(all_metrics)
        
        # Log training history over epochs
        if trainer.training_history:
            tracker.log_training_metrics_over_time(
                trainer.training_history,
                ['train_loss', 'val_loss', 'learning_rate']
            )
            
            # Create training history plot
            tracker.create_training_history_plot(
                trainer.training_history,
                metrics=['train_loss', 'val_loss'],
                title=f"LSTM Training History - {target_name}"
            )
        
        # Log model
        tracker.log_model(trainer.model, "pytorch")
        
        # Log predictions plot
        test_pred = trainer.evaluate_test(X_test, y_test)
        y_test_trimmed = y_test.values[trainer.sequence_length:]
        
        tracker.create_predictions_plot(
            y_test_trimmed,
            test_pred,
            title=f"LSTM Predictions - {target_name}",
            dataset_name="test"
        )
        
    finally:
        tracker.end_run()