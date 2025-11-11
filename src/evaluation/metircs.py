# File: src/evaluation/metrics.py

"""
Evaluation metrics for model performance
"""

import numpy as np
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error
)
from typing import Dict, List


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                      target_names: List[str]) -> Dict[str, float]:
    """
    Calculate comprehensive metrics for all targets

    Args:
        y_true: True values (n_samples, n_targets)
        y_pred: Predicted values (n_samples, n_targets)
        target_names: List of target names

    Returns:
        Dictionary of metrics
    """
    metrics = {}

    # Per-target metrics
    for i, target in enumerate(target_names):
        rmse = np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i]))
        mae = mean_absolute_error(y_true[:, i], y_pred[:, i])
        r2 = r2_score(y_true[:, i], y_pred[:, i])

        # MAPE (handle zero values)
        mask = y_true[:, i] != 0
        if mask.sum() > 0:
            mape = np.mean(
                np.abs((y_true[mask, i] - y_pred[mask, i]) / y_true[mask, i])) * 100
        else:
            mape = np.nan

        metrics[f'{target}_rmse'] = rmse
        metrics[f'{target}_mae'] = mae
        metrics[f'{target}_r2'] = r2
        metrics[f'{target}_mape'] = mape

    # Overall metrics
    metrics['avg_rmse'] = np.mean([metrics[f'{t}_rmse'] for t in target_names])
    metrics['avg_mae'] = np.mean([metrics[f'{t}_mae'] for t in target_names])
    metrics['avg_r2'] = np.mean([metrics[f'{t}_r2'] for t in target_names])

    return metrics


def print_metrics(metrics: Dict[str, float], target_names: List[str]):
    """
    Print metrics in formatted table
    """
    print(f"\n{'Metric':<25} {'RMSE':>12} {'MAE':>12} {'R²':>10} {'MAPE %':>10}")
    print("-" * 75)

    for target in target_names:
        rmse = metrics.get(f'{target}_rmse', 0)
        mae = metrics.get(f'{target}_mae', 0)
        r2 = metrics.get(f'{target}_r2', 0)
        mape = metrics.get(f'{target}_mape', np.nan)

        print(f"{target:<25} {rmse:>12.2f} {mae:>12.2f} {r2:>10.3f} {mape:>10.2f}")

    print("-" * 75)
    print(
        f"{'AVERAGE':<25} {metrics['avg_rmse']:>12.2f} {metrics['avg_mae']:>12.2f} {metrics['avg_r2']:>10.3f}")


def compare_models(model_metrics: Dict[str, Dict[str, float]],
                   metric: str = 'avg_r2') -> str:
    """
    Compare models and return best

    Args:
        model_metrics: Dict of {model_name: {metric_name: value}}
        metric: Metric to use for comparison (default: avg_r2)

    Returns:
        Name of best model
    """
    if metric.endswith('_r2'):
        # Higher is better for R²
        best_model = max(model_metrics.items(),
                         key=lambda x: x[1].get(metric, -np.inf))
    else:
        # Lower is better for RMSE, MAE, MAPE
        best_model = min(model_metrics.items(),
                         key=lambda x: x[1].get(metric, np.inf))

    return best_model[0]
