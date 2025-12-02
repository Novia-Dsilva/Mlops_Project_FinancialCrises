"""
Evaluation metrics for regression models
"""

import numpy as np
from sklearn.metrics import (
    mean_squared_error, 
    mean_absolute_error, 
    r2_score,
    mean_absolute_percentage_error
)
from typing import Dict
import pandas as pd

def calculate_regression_metrics(y_true, y_pred, model_name: str = "Model") -> Dict:
    """
    Calculate comprehensive regression metrics.
    Optimized for percentage returns (handles near-zero values).
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        model_name: Name of model for display
    
    Returns:
        Dictionary of metrics
    """
    # Handle numpy arrays or pandas Series
    if isinstance(y_true, pd.Series):
        y_true = y_true.values
    if isinstance(y_pred, pd.Series):
        y_pred = y_pred.values
    
    # Calculate standard metrics
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # ============================================
    # FIX: Better metrics for returns prediction
    # ============================================
    
    # 1. Mean Absolute Error in percentage points
    # (e.g., 0.15 = 15 percentage points error)
    mae_pct_points = mae * 100
    
    # 2. Directional Accuracy
    # What % of time did we predict the right direction?
    direction_correct = np.sign(y_true) == np.sign(y_pred)
    directional_accuracy = direction_correct.mean() * 100
    
    # 3. MAPE only for non-zero values (avoid division by zero)
    # Only calculate for values where |actual| > 0.01 (1%)
    non_zero_mask = np.abs(y_true) > 0.01
    if non_zero_mask.sum() > 0:
        mape = np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / 
                              y_true[non_zero_mask])) * 100
    else:
        mape = np.nan  # Can't calculate MAPE
    
    # 4. Additional metrics
    max_error = np.max(np.abs(y_true - y_pred))
    median_ae = np.median(np.abs(y_true - y_pred))
    
    # 5. Error percentiles (helpful for understanding distribution)
    errors = np.abs(y_true - y_pred)
    error_95th = np.percentile(errors, 95)
    
    metrics = {
        'model_name': model_name,
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'mape': mape,  # Can be NaN if too many near-zero values
        'mae_pct_points': mae_pct_points,  # ⭐ Use this instead of MAPE for returns
        'directional_accuracy': directional_accuracy,  # ⭐ New metric!
        'max_error': max_error,
        'median_absolute_error': median_ae,
        'error_95th_percentile': error_95th
    }
    
    return metrics


def print_metrics(metrics: Dict, dataset_name: str = "Dataset"):
    """
    Pretty print metrics (updated for returns prediction).
    """
    print(f"\n{'='*80}")
    print(f"📊 {metrics['model_name']} Performance on {dataset_name}")
    print(f"{'='*80}")
    print(f"RMSE (Root Mean Squared Error):     {metrics['rmse']:>12.4f}")
    print(f"MAE  (Mean Absolute Error):         {metrics['mae']:>12.4f}")
    print(f"MAE  (Percentage Points):           {metrics['mae_pct_points']:>11.2f} pp")
    print(f"R²   (R-squared):                   {metrics['r2']:>12.4f}")
    
    # Only show MAPE if it's valid
    if not np.isnan(metrics['mape']):
        print(f"MAPE (Mean Absolute % Error):       {metrics['mape']:>11.2f}%")
    
    print(f"Directional Accuracy:               {metrics['directional_accuracy']:>11.2f}%")
    print(f"Max Error:                          {metrics['max_error']:>12.4f}")
    print(f"Median Absolute Error:              {metrics['median_absolute_error']:>12.4f}")
    print(f"95th Percentile Error:              {metrics['error_95th_percentile']:>12.4f}")
    print(f"{'='*80}\n")

def compare_models(metrics_list: list) -> pd.DataFrame:
    """
    Compare multiple models' metrics.
    
    Args:
        metrics_list: List of metric dictionaries
    
    Returns:
        DataFrame with comparison
    """
    comparison = pd.DataFrame(metrics_list)
    
    # Round for display
    comparison['rmse'] = comparison['rmse'].round(4)
    comparison['mae'] = comparison['mae'].round(4)
    comparison['r2'] = comparison['r2'].round(4)
    comparison['mape'] = comparison['mape'].round(2)
    
    # Rank models (lower RMSE = better)
    comparison = comparison.sort_values('rmse')
    comparison['rank'] = range(1, len(comparison) + 1)
    
    return comparison[['rank', 'model_name', 'rmse', 'mae', 'r2', 'mape']]