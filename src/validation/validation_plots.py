"""
src/evaluation/create_validation_plots.py

OPTIONAL: Creates comprehensive validation visualizations for regression models
- Actual vs Predicted scatter plots
- Residual plots (error analysis)
- Error distribution histograms
- Feature importance comparison

NOTE: These are NICE-TO-HAVE, not critical for PDF requirements!
PRIORITY: Do bias_detection.py first (25% of grade!)

Usage:
    python src/evaluation/create_validation_plots.py --model lightgbm --target profit_margin
    python src/evaluation/create_validation_plots.py --model lightgbm --target all
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
import argparse
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from utils.split_utils import get_feature_target_split, drop_nan_targets

sns.set_style("whitegrid")


def load_model_and_data(model_type: str, target_name: str):
    """Load model and test data"""
    
    target_col = f'target_{target_name}'
    
    # Load model
    model_file = Path(f'models/{model_type}/{model_type}_{target_name}.pkl')
    if not model_file.exists():
        raise FileNotFoundError(f"Model not found: {model_file}")
    
    model_data = joblib.load(model_file)
    model = model_data['model']
    
    # Load test data
    test_df = pd.read_csv('data/splits/test_data.csv')
    
    # Prepare features
    X_test, y_test = get_feature_target_split(test_df, target_col, encode_categoricals=True)
    
    # Get feature names from model
    feature_names = model_data.get('feature_names', [])
    
    # Align columns
    for col in feature_names:
        if col not in X_test.columns:
            X_test[col] = 0
    X_test = X_test[feature_names]
    
    # Drop NaN targets
    X_test, y_test = drop_nan_targets(X_test, y_test, "Test")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    return model, X_test, y_test, y_pred, feature_names


def create_actual_vs_predicted_plot(y_test, y_pred, target_name, model_type, output_dir):
    """
    Scatter plot: Actual vs Predicted values
    Perfect predictions fall on diagonal line
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Scatter plot
    ax.scatter(y_test, y_pred, alpha=0.5, s=30, edgecolors='black', linewidth=0.5)
    
    # Perfect prediction line (diagonal)
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    # Calculate metrics
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    
    # Add metrics text box
    metrics_text = f'R² = {r2:.4f}\nRMSE = {rmse:.2f}\nMAE = {mae:.2f}'
    ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes,
           fontsize=12, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax.set_xlabel(f'Actual {target_name}', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'Predicted {target_name}', fontsize=12, fontweight='bold')
    ax.set_title(f'Actual vs Predicted: {target_name} ({model_type})', 
                fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_file = Path(output_dir) / f"{target_name}_{model_type}_actual_vs_predicted.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"   ✅ Actual vs Predicted: {plot_file}")
    plt.close()


def create_residual_plot(y_test, y_pred, target_name, model_type, output_dir):
    """
    Residual plot: Shows prediction errors
    Good model has residuals randomly scattered around zero
    """
    residuals = y_test - y_pred
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Residuals vs Predicted
    ax1.scatter(y_pred, residuals, alpha=0.5, s=30, edgecolors='black', linewidth=0.5)
    ax1.axhline(y=0, color='r', linestyle='--', linewidth=2, label='Zero Error')
    ax1.set_xlabel(f'Predicted {target_name}', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Residuals (Actual - Predicted)', fontsize=12, fontweight='bold')
    ax1.set_title(f'Residual Plot: {target_name}', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add standard deviation bands
    std_resid = residuals.std()
    ax1.axhline(y=std_resid, color='orange', linestyle=':', linewidth=1, alpha=0.7, label='+1 std')
    ax1.axhline(y=-std_resid, color='orange', linestyle=':', linewidth=1, alpha=0.7, label='-1 std')
    
    # Plot 2: Residual Distribution
    ax2.hist(residuals, bins=50, edgecolor='black', alpha=0.7)
    ax2.axvline(x=0, color='r', linestyle='--', linewidth=2, label='Zero Error')
    ax2.set_xlabel('Residuals', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax2.set_title(f'Residual Distribution: {target_name}', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add normality text
    mean_resid = residuals.mean()
    std_resid = residuals.std()
    stats_text = f'Mean: {mean_resid:.2f}\nStd: {std_resid:.2f}'
    ax2.text(0.95, 0.95, stats_text, transform=ax2.transAxes,
            fontsize=10, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plot_file = Path(output_dir) / f"{target_name}_{model_type}_residuals.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"   ✅ Residual Plot: {plot_file}")
    plt.close()


def create_error_distribution_plot(y_test, y_pred, target_name, model_type, output_dir):
    """
    Multiple error metrics visualization
    """
    residuals = y_test - y_pred
    abs_errors = np.abs(residuals)
    pct_errors = (abs_errors / np.abs(y_test)) * 100
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Absolute Errors
    axes[0, 0].hist(abs_errors, bins=50, edgecolor='black', alpha=0.7, color='skyblue')
    axes[0, 0].axvline(x=abs_errors.median(), color='r', linestyle='--', 
                       linewidth=2, label=f'Median: {abs_errors.median():.2f}')
    axes[0, 0].set_xlabel('Absolute Error', fontsize=11, fontweight='bold')
    axes[0, 0].set_ylabel('Frequency', fontsize=11, fontweight='bold')
    axes[0, 0].set_title('Absolute Error Distribution', fontsize=12, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Percentage Errors (capped at 100% for visualization)
    pct_errors_capped = np.clip(pct_errors, 0, 100)
    axes[0, 1].hist(pct_errors_capped, bins=50, edgecolor='black', alpha=0.7, color='lightcoral')
    axes[0, 1].axvline(x=pct_errors.median(), color='r', linestyle='--', 
                       linewidth=2, label=f'Median: {pct_errors.median():.2f}%')
    axes[0, 1].set_xlabel('Percentage Error (%)', fontsize=11, fontweight='bold')
    axes[0, 1].set_ylabel('Frequency', fontsize=11, fontweight='bold')
    axes[0, 1].set_title('Percentage Error Distribution', fontsize=12, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Error by Prediction Range
    # Bin predictions into quartiles
    pred_quartiles = pd.qcut(y_pred, q=4, labels=['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)'])
    error_by_quartile = pd.DataFrame({
        'Quartile': pred_quartiles,
        'Abs_Error': abs_errors
    })
    
    error_by_quartile.boxplot(column='Abs_Error', by='Quartile', ax=axes[1, 0])
    axes[1, 0].set_xlabel('Prediction Range', fontsize=11, fontweight='bold')
    axes[1, 0].set_ylabel('Absolute Error', fontsize=11, fontweight='bold')
    axes[1, 0].set_title('Error by Prediction Range', fontsize=12, fontweight='bold')
    plt.sca(axes[1, 0])
    plt.xticks(rotation=0)
    
    # Plot 4: Cumulative Error
    sorted_errors = np.sort(abs_errors)
    cumulative = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors) * 100
    
    axes[1, 1].plot(sorted_errors, cumulative, linewidth=2)
    axes[1, 1].axhline(y=50, color='r', linestyle='--', linewidth=1, label='50% of predictions')
    axes[1, 1].axhline(y=90, color='orange', linestyle='--', linewidth=1, label='90% of predictions')
    axes[1, 1].set_xlabel('Absolute Error', fontsize=11, fontweight='bold')
    axes[1, 1].set_ylabel('Cumulative %', fontsize=11, fontweight='bold')
    axes[1, 1].set_title('Cumulative Error Distribution', fontsize=12, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Add 90th percentile annotation
    p90_error = np.percentile(abs_errors, 90)
    axes[1, 1].text(0.6, 0.5, f'90% of predictions\nhave error < {p90_error:.2f}',
                   transform=axes[1, 1].transAxes, fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.suptitle(f'Error Analysis: {target_name} ({model_type})', 
                fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    plot_file = Path(output_dir) / f"{target_name}_{model_type}_error_analysis.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"   ✅ Error Analysis: {plot_file}")
    plt.close()


def create_feature_importance_plot(model, feature_names, target_name, model_type, output_dir):
    """
    Feature importance visualization
    """
    # Get feature importance (works for tree-based models)
    try:
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importance = np.abs(model.coef_)
        else:
            print(f"   ⚠️  Feature importance not available for this model type")
            return
        
        # Create dataframe
        feat_imp_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance
        }).sort_values('Importance', ascending=False).head(20)
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(feat_imp_df)))
        bars = ax.barh(feat_imp_df['Feature'], feat_imp_df['Importance'], 
                      color=colors, alpha=0.8, edgecolor='black')
        
        ax.set_xlabel('Importance', fontsize=12, fontweight='bold')
        ax.set_ylabel('Feature', fontsize=12, fontweight='bold')
        ax.set_title(f'Top 20 Features: {target_name} ({model_type})', 
                    fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        # Invert y-axis so highest is on top
        ax.invert_yaxis()
        
        plt.tight_layout()
        plot_file = Path(output_dir) / f"{target_name}_{model_type}_feature_importance.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"   ✅ Feature Importance: {plot_file}")
        plt.close()
        
    except Exception as e:
        print(f"   ⚠️  Could not create feature importance plot: {e}")


def create_prediction_intervals_plot(y_test, y_pred, target_name, model_type, output_dir):
    """
    Show prediction confidence/error bars
    """
    residuals = y_test - y_pred
    abs_errors = np.abs(residuals)
    
    # Sort by actual values for better visualization
    sorted_indices = np.argsort(y_test)
    y_test_sorted = y_test.iloc[sorted_indices].values
    y_pred_sorted = y_pred[sorted_indices]
    abs_errors_sorted = abs_errors.iloc[sorted_indices].values
    
    # Sample if too many points
    n_samples = min(200, len(y_test_sorted))
    step = len(y_test_sorted) // n_samples
    
    y_test_sample = y_test_sorted[::step]
    y_pred_sample = y_pred_sorted[::step]
    errors_sample = abs_errors_sorted[::step]
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x = np.arange(len(y_test_sample))
    
    # Plot actual values
    ax.plot(x, y_test_sample, 'o-', label='Actual', linewidth=2, markersize=6, alpha=0.7)
    
    # Plot predictions with error bars
    ax.errorbar(x, y_pred_sample, yerr=errors_sample, fmt='s-', 
               label='Predicted ± Error', linewidth=2, markersize=5, 
               alpha=0.6, capsize=3)
    
    ax.set_xlabel('Sample Index (sorted by actual value)', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'{target_name} Value', fontsize=12, fontweight='bold')
    ax.set_title(f'Predictions with Error Bars: {target_name} ({model_type})', 
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_file = Path(output_dir) / f"{target_name}_{model_type}_prediction_intervals.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"   ✅ Prediction Intervals: {plot_file}")
    plt.close()


def create_comprehensive_validation_report(y_test, y_pred, target_name, model_type, output_dir):
    """
    Create detailed validation metrics report
    """
    residuals = y_test - y_pred
    abs_errors = np.abs(residuals)
    pct_errors = (abs_errors / np.abs(y_test)) * 100
    
    # Calculate comprehensive metrics
    metrics = {
        'target': target_name,
        'model': model_type,
        'performance': {
            'r2': float(r2_score(y_test, y_pred)),
            'rmse': float(np.sqrt(mean_squared_error(y_test, y_pred))),
            'mae': float(mean_absolute_error(y_test, y_pred)),
            'mape': float(pct_errors.mean()),
        },
        'error_analysis': {
            'mean_error': float(residuals.mean()),
            'median_error': float(residuals.median()),
            'std_error': float(residuals.std()),
            'mean_abs_error': float(abs_errors.mean()),
            'median_abs_error': float(abs_errors.median()),
            '90th_percentile_error': float(np.percentile(abs_errors, 90)),
            '95th_percentile_error': float(np.percentile(abs_errors, 95)),
        },
        'prediction_quality': {
            'within_10pct': float((pct_errors < 10).sum() / len(pct_errors) * 100),
            'within_20pct': float((pct_errors < 20).sum() / len(pct_errors) * 100),
            'within_30pct': float((pct_errors < 30).sum() / len(pct_errors) * 100),
        },
        'sample_stats': {
            'n_predictions': int(len(y_test)),
            'actual_mean': float(y_test.mean()),
            'actual_std': float(y_test.std()),
            'predicted_mean': float(y_pred.mean()),
            'predicted_std': float(y_pred.std()),
        }
    }
    
    report_file = Path(output_dir) / f"{target_name}_{model_type}_validation_report.json"
    with open(report_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"   ✅ Validation Report: {report_file}")
    
    return metrics


def create_all_validation_plots(model_type: str, target_name: str, 
                                output_dir: str = "reports/validation"):
    """
    Create all validation plots for a model-target combination
    """
    print(f"\n{'=' * 80}")
    print(f"📊 CREATING VALIDATION PLOTS: {target_name} ({model_type})")
    print(f"{'=' * 80}\n")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load model and data
        print(f"Loading model and data...")
        model, X_test, y_test, y_pred, feature_names = load_model_and_data(model_type, target_name)
        print(f"   ✅ Loaded {len(y_test)} test samples")
        
        # Create plots
        print(f"\nCreating visualizations...")
        create_actual_vs_predicted_plot(y_test, y_pred, target_name, model_type, output_path)
        create_residual_plot(y_test, y_pred, target_name, model_type, output_path)
        create_error_distribution_plot(y_test, y_pred, target_name, model_type, output_path)
        create_feature_importance_plot(model, feature_names, target_name, model_type, output_path)
        create_prediction_intervals_plot(y_test, y_pred, target_name, model_type, output_path)
        
        # Create validation report
        print(f"\nGenerating validation report...")
        metrics = create_comprehensive_validation_report(y_test, y_pred, target_name, 
                                                        model_type, output_path)
        
        # Print summary
        print(f"\n{'=' * 80}")
        print(f"📋 VALIDATION SUMMARY")
        print(f"{'=' * 80}")
        print(f"\nPerformance:")
        print(f"   R²:   {metrics['performance']['r2']:.4f}")
        print(f"   RMSE: {metrics['performance']['rmse']:.2f}")
        print(f"   MAE:  {metrics['performance']['mae']:.2f}")
        print(f"   MAPE: {metrics['performance']['mape']:.2f}%")
        
        print(f"\nPrediction Quality:")
        print(f"   Within 10% error: {metrics['prediction_quality']['within_10pct']:.1f}%")
        print(f"   Within 20% error: {metrics['prediction_quality']['within_20pct']:.1f}%")
        print(f"   Within 30% error: {metrics['prediction_quality']['within_30pct']:.1f}%")
        
        print(f"\n{'=' * 80}")
        print(f"✅ VALIDATION COMPLETE: {target_name}")
        print(f"{'=' * 80}")
        
    except Exception as e:
        print(f"\n❌ Error creating validation plots: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Create comprehensive validation plots for regression models"
    )
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Model type (xgboost, lightgbm, lightgbm_tuned, etc.)'
    )
    parser.add_argument(
        '--target',
        type=str,
        required=True,
        choices=['revenue', 'eps', 'debt_equity', 'profit_margin', 'stock_return', 'all'],
        help='Target variable to validate'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='reports/validation',
        help='Output directory for plots and reports'
    )
    
    args = parser.parse_args()
    
    print(f"\n{'=' * 80}")
    print(f"🎯 VALIDATION PLOT GENERATION")
    print(f"{'=' * 80}")
    print(f"Model: {args.model}")
    print(f"Target: {args.target}")
    print(f"Output: {args.output_dir}/")
    print(f"{'=' * 80}")
    
    if args.target == 'all':
        targets = ['revenue', 'eps', 'debt_equity', 'profit_margin', 'stock_return']
        
        print(f"\n📊 Creating validation plots for ALL targets...")
        
        for target in targets:
            try:
                create_all_validation_plots(args.model, target, args.output_dir)
            except Exception as e:
                print(f"\n⚠️  Skipping {target}: {e}")
                continue
        
        print(f"\n{'=' * 80}")
        print(f"✅ ALL VALIDATION PLOTS COMPLETE")
        print(f"{'=' * 80}")
        print(f"\nPlots saved to: {args.output_dir}/")
        
    else:
        create_all_validation_plots(args.model, args.target, args.output_dir)
    
    print(f"\n{'=' * 80}")
    print(f"🎯 NEXT STEPS")
    print(f"{'=' * 80}")
    print(f"\n⚠️  REMINDER: Validation plots are OPTIONAL (5-10% extra credit)")
    print(f"\n🔥 CRITICAL: Still need to complete (60% of grade!):")
    print(f"   1. Bias detection (slicing by sector) - 25% of grade")
    print(f"   2. Model registry push (GCP) - 15% of grade")
    print(f"   3. CI/CD pipeline setup - 20% of grade")
    print(f"\n{'=' * 80}")


if __name__ == "__main__":
    main()