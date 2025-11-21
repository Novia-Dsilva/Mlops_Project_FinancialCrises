"""
MLflow Configuration for Financial Stress Test Models
"""
import mlflow
import os
from pathlib import Path


class MLflowConfig:
    """MLflow configuration and helper functions"""
    
    def __init__(self, tracking_uri=None, experiment_name="Financial_Stress_Test_Scenarios"):
        """
        Initialize MLflow configuration
        
        Args:
            tracking_uri: Remote MLflow server URI (e.g., 'http://mlflow-server:5000')
            experiment_name: Name of the MLflow experiment
        """
        self.tracking_uri = tracking_uri or os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000')
        self.experiment_name = experiment_name
        
        # Set tracking URI
        mlflow.set_tracking_uri(self.tracking_uri)
        print(f"✓ MLflow Tracking URI: {self.tracking_uri}")
        
        # Set or create experiment
        try:
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(self.experiment_name)
                print(f"✓ Created new experiment: {self.experiment_name} (ID: {experiment_id})")
            else:
                experiment_id = experiment.experiment_id
                print(f"✓ Using existing experiment: {self.experiment_name} (ID: {experiment_id})")
            
            mlflow.set_experiment(self.experiment_name)
        except Exception as e:
            print(f"⚠ Error setting experiment: {e}")
            print(f"⚠ Falling back to default experiment")
    
    @staticmethod
    def log_params(params_dict):
        """Log parameters to MLflow"""
        try:
            for key, value in params_dict.items():
                if isinstance(value, (list, tuple)):
                    mlflow.log_param(key, str(value))
                else:
                    mlflow.log_param(key, value)
        except Exception as e:
            print(f"⚠ Error logging params: {e}")
    
    @staticmethod
    def log_metrics(metrics_dict, step=None):
        """Log metrics to MLflow"""
        try:
            for key, value in metrics_dict.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(key, value, step=step)
        except Exception as e:
            print(f"⚠ Error logging metrics: {e}")
    
    @staticmethod
    def log_artifacts(artifact_paths):
        """
        Log artifacts to MLflow
        
        Args:
            artifact_paths: List of file paths or single file path
        """
        try:
            if isinstance(artifact_paths, str):
                artifact_paths = [artifact_paths]
            
            for path in artifact_paths:
                if os.path.exists(path):
                    if os.path.isdir(path):
                        mlflow.log_artifacts(path)
                    else:
                        mlflow.log_artifact(path)
                else:
                    print(f"⚠ Artifact not found: {path}")
        except Exception as e:
            print(f"⚠ Error logging artifacts: {e}")
    
    @staticmethod
    def set_tags(tags_dict):
        """Set tags for the run"""
        try:
            for key, value in tags_dict.items():
                mlflow.set_tag(key, str(value))
        except Exception as e:
            print(f"⚠ Error setting tags: {e}")


def compare_models(experiment_name="Financial_Stress_Test_Scenarios"):
    """
    Compare all models in the experiment and identify the best one
    
    Returns:
        DataFrame with model comparison
    """
    import pandas as pd
    
    try:
        mlflow.set_experiment(experiment_name)
        
        # Get all runs from experiment
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            print(f"⚠ Experiment '{experiment_name}' not found")
            return None
        
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["metrics.ks_pass_rate DESC"]
        )
        
        if len(runs) == 0:
            print("⚠ No runs found in experiment")
            return None
        
        # Select relevant columns
        comparison_cols = [
            'run_id',
            'tags.mlflow.runName',
            'params.model_type',
            'metrics.ks_pass_rate',
            'metrics.correlation_mae',
            'metrics.wasserstein_distance',
            'metrics.training_time_seconds',
            'start_time'
        ]
        
        available_cols = [col for col in comparison_cols if col in runs.columns]
        comparison_df = runs[available_cols].copy()
        
        # Rename columns for readability
        comparison_df.columns = [col.split('.')[-1] for col in comparison_df.columns]
        
        # Sort by KS pass rate (primary) and correlation MAE (secondary)
        if 'ks_pass_rate' in comparison_df.columns and 'correlation_mae' in comparison_df.columns:
            comparison_df = comparison_df.sort_values(
                by=['ks_pass_rate', 'correlation_mae'],
                ascending=[False, True]
            )
        
        print("\n" + "="*80)
        print("MODEL COMPARISON REPORT")
        print("="*80)
        print(f"\nTotal Runs: {len(comparison_df)}")
        print(f"\nTop 5 Models (by KS Pass Rate):")
        print("-"*80)
        print(comparison_df.head(5).to_string(index=False))
        print("="*80)
        
        return comparison_df
        
    except Exception as e:
        print(f"⚠ Error comparing models: {e}")
        return None