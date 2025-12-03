"""
src/evaluation/test_all_models_for_bias.py

CORRECT APPROACH: Test ALL models for bias BEFORE final selection!

This script:
1. Tests ALL available models (xgboost, lightgbm, lightgbm_tuned) for bias
2. Creates comparison of R² AND bias for each target
3. Allows informed selection considering BOTH metrics

Then use final_model_selection_after_bias.py to make final choice!

Usage:
    python src/evaluation/test_all_models_for_bias.py --target debt_equity
    python src/evaluation/test_all_models_for_bias.py --target all
"""

import json
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import argparse

# Setup
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root / "src"))
from utils.split_utils import get_feature_target_split, drop_nan_targets

sns.set_style("whitegrid")


def find_available_models(target_name: str):
    """Find all available trained models for a target"""
    
    # Check ALL possible model types (baseline + tuned)
    model_types = [
        'xgboost', 
        'xgboost_tuned',
        'lightgbm', 
        'lightgbm_tuned',
        'lstm',
        'lstm_tuned'
    ]
    
    available = []
    
    for model_type in model_types:
        # Try different filename patterns
        if '_tuned' in model_type:
            base_model = model_type.replace('_tuned', '')
            patterns = [
                f'models/{model_type}/{base_model}_{target_name}.pkl',
                f'models/{model_type}/{base_model}_{target_name}_tuned.pkl',
                f'models/{model_type}/{model_type}_{target_name}.pkl'
            ]
        else:
            patterns = [f'models/{model_type}/{model_type}_{target_name}.pkl']
        
        for pattern in patterns:
            if Path(pattern).exists():
                available.append({
                    'model_type': model_type,
                    'path': pattern
                })
                break
    
    return available


def load_model(model_path: str):
    """Load model"""
    model_data = joblib.load(model_path)
    return model_data['model'], model_data.get('feature_names', [])


def create_crisis_flag(test_df):
    """Create crisis flag"""
    test_df['Date'] = pd.to_datetime(test_df['Date'])
    test_df['crisis_flag'] = 0
    
    # Financial Crisis
    financial_crisis = (test_df['Date'] >= '2007-01-01') & (test_df['Date'] <= '2009-12-31')
    test_df.loc[financial_crisis, 'crisis_flag'] = 1
    
    # COVID Crisis
    covid_crisis = (test_df['Date'] >= '2020-01-01') & (test_df['Date'] <= '2021-12-31')
    test_df.loc[covid_crisis, 'crisis_flag'] = 1
    
    # VIX stress
    if 'vix_q_mean' in test_df.columns:
        vix_stress = test_df['vix_q_mean'] > 30
        test_df.loc[vix_stress, 'crisis_flag'] = 1
    
    return test_df


def calculate_metrics(y_true, y_pred):
    """Calculate metrics"""
    if len(y_true) < 5:
        return {
            'n_samples': len(y_true), 'r2': np.nan, 'rmse': np.nan,
            'mae': np.nan, 'mean_residual': np.nan
        }
    
    return {
        'n_samples': len(y_true),
        'r2': float(r2_score(y_true, y_pred)),
        'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
        'mae': float(mean_absolute_error(y_true, y_pred)),
        'mean_residual': float(np.mean(y_true - y_pred))
    }


def test_model_for_bias(model_type: str, model_path: str, target_name: str, test_df):
    """Test one model for crisis bias"""
    
    target_col = f'target_{target_name}'
    
    try:
        # Load model
        model, feature_names = load_model(model_path)
        
        # Prepare features
        X_test, y_test = get_feature_target_split(test_df, target_col, encode_categoricals=True)
        
        # Align features
        for col in feature_names:
            if col not in X_test.columns:
                X_test[col] = 0
        
        extra_cols = set(X_test.columns) - set(feature_names)
        if extra_cols:
            X_test = X_test.drop(columns=list(extra_cols))
        
        X_test = X_test[feature_names]
        X_test, y_test = drop_nan_targets(X_test, y_test, "Test")
        
        # Predict
        y_pred = model.predict(X_test)
        
        # Get crisis flags
        crisis_flags = test_df.iloc[X_test.index]['crisis_flag'].values
        
        # Split by crisis
        crisis_mask = crisis_flags == 1
        non_crisis_mask = crisis_flags == 0
        
        # Calculate metrics
        overall = calculate_metrics(y_test, y_pred)
        crisis = calculate_metrics(y_test[crisis_mask], y_pred[crisis_mask])
        non_crisis = calculate_metrics(y_test[non_crisis_mask], y_pred[non_crisis_mask])
        
        # Detect bias
        if crisis['n_samples'] < 5 or np.isnan(crisis['r2']):
            severity = 'NOT_EVALUATED'
            has_bias = None
            rmse_ratio = None
            r2_deg = None
        else:
            rmse_ratio = crisis['rmse'] / non_crisis['rmse'] if non_crisis['rmse'] > 0 else 0
            r2_deg = non_crisis['r2'] - crisis['r2']
            
            if rmse_ratio > 1.5 or r2_deg > 0.20:
                severity = 'CRITICAL'
                has_bias = True
            elif rmse_ratio > 1.2 or r2_deg > 0.10:
                severity = 'MODERATE'
                has_bias = True
            else:
                severity = 'NONE'
                has_bias = False
        
        return {
            'model_type': model_type,
            'overall': overall,
            'crisis': crisis,
            'non_crisis': non_crisis,
            'bias': {
                'has_bias': has_bias,
                'severity': severity,
                'rmse_ratio': rmse_ratio,
                'r2_degradation': r2_deg
            }
        }
        
    except Exception as e:
        print(f"      ❌ Error testing {model_type}: {e}")
        return None


def test_all_models_for_target(target_name: str, test_df, output_dir: str):
    """Test ALL available models for one target"""
    
    print(f"\n{'#'*80}")
    print(f"🔍 TESTING ALL MODELS FOR: {target_name.upper()}")
    print(f"{'#'*80}\n")
    
    # Find available models
    available_models = find_available_models(target_name)
    
    if not available_models:
        print(f"   ❌ No models found for {target_name}")
        return None
    
    print(f"   Found {len(available_models)} models to test:")
    for m in available_models:
        print(f"      - {m['model_type']}")
    
    # Test each model
    print(f"\n   Testing each model for crisis bias...\n")
    
    results = []
    
    for model_info in available_models:
        model_type = model_info['model_type']
        model_path = model_info['path']
        
        print(f"   Testing: {model_type}...")
        result = test_model_for_bias(model_type, model_path, target_name, test_df)
        
        if result:
            results.append(result)
            
            # Print summary
            bias_severity = result['bias']['severity']
            overall_r2 = result['overall']['r2']
            crisis_r2 = result['crisis']['r2']
            
            symbol = "✅" if bias_severity == "NONE" else "⚠️" if bias_severity == "MODERATE" else "🚨"
            
            print(f"      {symbol} R²: {overall_r2:.4f}, Crisis: {crisis_r2:.4f}, Bias: {bias_severity}")
    
    # Create comparison table
    print(f"\n{'='*80}")
    print(f"📊 COMPARISON: ALL MODELS FOR {target_name.upper()}")
    print(f"{'='*80}\n")
    
    print(f"{'Model':<20} {'Overall R²':>12} {'Crisis R²':>12} {'RMSE Ratio':>12} {'Bias':>15}")
    print(f"{'─'*75}")
    
    for result in sorted(results, key=lambda x: x['overall']['r2'], reverse=True):
        model = result['model_type']
        overall_r2 = result['overall']['r2']
        crisis_r2 = result['crisis']['r2']
        rmse_ratio = result['bias'].get('rmse_ratio', 0)
        severity = result['bias']['severity']
        
        rmse_str = f"{rmse_ratio:.2f}x" if rmse_ratio else "N/A"
        
        print(f"{model:<20} {overall_r2:>12.4f} {crisis_r2:>12.4f} {rmse_str:>12} {severity:>15}")
    
    print(f"{'─'*75}\n")
    
    # Identify best choice
    print(f"{'='*80}")
    print(f"🎯 SELECTION RECOMMENDATION")
    print(f"{'='*80}\n")
    
    # Filter out critical/severe bias
    acceptable = [r for r in results if r['bias']['severity'] != 'CRITICAL']
    
    if not acceptable:
        print(f"   🚨 ALL MODELS HAVE CRITICAL BIAS!")
        print(f"   Recommendation: Use best R² with heavy monitoring")
        best = max(results, key=lambda x: x['overall']['r2'])
    else:
        # From acceptable models, pick best R²
        best = max(acceptable, key=lambda x: x['overall']['r2'])
        
        print(f"   🏆 RECOMMENDED: {best['model_type']}")
        print(f"      R²: {best['overall']['r2']:.4f}")
        print(f"      Bias: {best['bias']['severity']}")
        
        # Check if we're sacrificing R² for fairness
        best_r2_model = max(results, key=lambda x: x['overall']['r2'])
        
        if best['model_type'] != best_r2_model['model_type']:
            r2_sacrifice = best_r2_model['overall']['r2'] - best['overall']['r2']
            print(f"\n   Trade-off: Accepting {r2_sacrifice:.4f} lower R² for fairness")
            print(f"      (Rejected {best_r2_model['model_type']}: R²={best_r2_model['overall']['r2']:.4f}, Bias={best_r2_model['bias']['severity']})")
        else:
            print(f"\n   ✅ Best R² AND lowest bias - clear winner!")
    
    # Save comparison
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    comparison = {
        'target': target_name,
        'models_tested': len(results),
        'models': results,
        'recommendation': {
            'model': best['model_type'],
            'overall_r2': best['overall']['r2'],
            'crisis_r2': best['crisis']['r2'],
            'bias_severity': best['bias']['severity']
        }
    }
    
    comparison_file = output_path / f"{target_name}_all_models_bias_comparison.json"
    with open(comparison_file, 'w') as f:
        json.dump(comparison, f, indent=2)
    
    print(f"\n   ✅ Comparison saved: {comparison_file}")
    
    return results


def create_comparison_visualization(target_name: str, results: list, output_dir: str):
    """Create visualization comparing all models"""
    
    if not results:
        return
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Model Comparison with Bias Analysis - {target_name.upper()}',
                 fontsize=16, fontweight='bold')
    
    df = pd.DataFrame([{
        'Model': r['model_type'],
        'Overall R²': r['overall']['r2'],
        'Crisis R²': r['crisis']['r2'],
        'RMSE Ratio': r['bias'].get('rmse_ratio', 0),
        'Bias': r['bias']['severity']
    } for r in results]).sort_values('Overall R²', ascending=False)
    
    # Plot 1: R² comparison
    ax1 = axes[0, 0]
    x = np.arange(len(df))
    width = 0.35
    
    ax1.bar(x - width/2, df['Overall R²'], width, label='Overall', alpha=0.8)
    ax1.bar(x + width/2, df['Crisis R²'], width, label='Crisis', alpha=0.8)
    
    ax1.set_xlabel('Model', fontweight='bold')
    ax1.set_ylabel('R²', fontweight='bold')
    ax1.set_title('R² Comparison: Overall vs Crisis', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(df['Model'], rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: RMSE Ratio
    ax2 = axes[0, 1]
    
    colors = []
    for severity in df['Bias']:
        if severity == 'CRITICAL':
            colors.append('red')
        elif severity == 'MODERATE':
            colors.append('orange')
        elif severity == 'NONE':
            colors.append('green')
        else:
            colors.append('gray')
    
    bars = ax2.barh(df['Model'], df['RMSE Ratio'], color=colors, alpha=0.8, edgecolor='black')
    ax2.axvline(x=1.5, color='red', linestyle='--', linewidth=2, label='Critical Threshold')
    ax2.axvline(x=1.2, color='orange', linestyle='--', linewidth=2, label='Moderate Threshold')
    
    ax2.set_xlabel('RMSE Ratio (Crisis/Normal)', fontweight='bold')
    ax2.set_ylabel('Model', fontweight='bold')
    ax2.set_title('Error Increase During Crisis', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='x')
    
    # Plot 3: Bias severity
    ax3 = axes[1, 0]
    
    bias_counts = df['Bias'].value_counts()
    colors_pie = []
    for bias_type in bias_counts.index:
        if bias_type == 'NONE':
            colors_pie.append('green')
        elif bias_type == 'MODERATE':
            colors_pie.append('orange')
        elif bias_type == 'CRITICAL':
            colors_pie.append('red')
        else:
            colors_pie.append('gray')
    
    ax3.pie(bias_counts.values, labels=bias_counts.index, colors=colors_pie,
           autopct='%1.0f%%', startangle=90, textprops={'fontsize': 11, 'fontweight': 'bold'})
    ax3.set_title('Bias Severity Distribution', fontweight='bold')
    
    # Plot 4: Performance-Bias scatter
    ax4 = axes[1, 1]
    
    scatter_colors = []
    for severity in df['Bias']:
        if severity == 'CRITICAL':
            scatter_colors.append('red')
        elif severity == 'MODERATE':
            scatter_colors.append('orange')
        elif severity == 'NONE':
            scatter_colors.append('green')
        else:
            scatter_colors.append('gray')
    
    ax4.scatter(df['Overall R²'], df['RMSE Ratio'], s=300, c=scatter_colors, 
               alpha=0.6, edgecolors='black', linewidth=2)
    
    # Add labels
    for _, row in df.iterrows():
        ax4.annotate(row['Model'], 
                    (row['Overall R²'], row['RMSE Ratio']),
                    fontsize=9, ha='center', va='bottom')
    
    # Add threshold lines
    ax4.axhline(y=1.5, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Critical')
    ax4.axhline(y=1.2, color='orange', linestyle='--', linewidth=1, alpha=0.5, label='Moderate')
    
    ax4.set_xlabel('Overall R² (Higher is better)', fontweight='bold')
    ax4.set_ylabel('RMSE Ratio (Lower is better)', fontweight='bold')
    ax4.set_title('Performance vs Bias Trade-off', fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Highlight optimal zone
    ax4.axhspan(0, 1.2, alpha=0.1, color='green', label='Low Bias Zone')
    
    plt.tight_layout()
    
    plot_file = output_path / f"{target_name}_all_models_bias_comparison.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    
    return plot_file


def test_all_models_for_target(target_name: str, test_df, output_dir: str = "reports/all_models_bias"):
    """Complete pipeline for one target"""
    
    print(f"\n{'='*80}")
    print(f"🔍 FINDING AVAILABLE MODELS: {target_name.upper()}")
    print(f"{'='*80}\n")
    
    # Find models
    available_models = find_available_models(target_name)
    
    if not available_models:
        print(f"   ❌ No models found for {target_name}")
        return None
    
    print(f"   ✅ Found {len(available_models)} models:")
    for m in available_models:
        print(f"      - {m['model_type']:<20} ({m['path']})")
    
    # Test all models
    print(f"\n{'='*80}")
    print(f"🚨 TESTING CRISIS BIAS FOR ALL MODELS")
    print(f"{'='*80}\n")
    
    all_results = []
    
    for model_info in available_models:
        model_type = model_info['model_type']
        model_path = model_info['path']
        
        print(f"   Testing {model_type}...")
        result = test_model_for_bias(model_type, model_path, target_name, test_df)
        
        if result:
            all_results.append(result)
            
            # Print summary
            bias_severity = result['bias']['severity']
            overall_r2 = result['overall']['r2']
            crisis_r2 = result['crisis']['r2']
            rmse_ratio = result['bias'].get('rmse_ratio', 0)
            
            symbol = "✅" if bias_severity == "NONE" else "⚠️" if bias_severity == "MODERATE" else "🚨"
            rmse_str = f"{rmse_ratio:.2f}x" if rmse_ratio else "N/A"
            
            print(f"      {symbol} Overall R²: {overall_r2:.4f}, Crisis R²: {crisis_r2:.4f}, "
                  f"RMSE Ratio: {rmse_str}, Bias: {bias_severity}")
    
    if not all_results:
        print(f"   ❌ No results for {target_name}")
        return None
    
    # Create comparison table
    print(f"\n{'='*80}")
    print(f"📊 COMPARISON: ALL MODELS FOR {target_name.upper()}")
    print(f"{'='*80}\n")
    
    print(f"{'Model':<20} {'Overall R²':>12} {'Crisis R²':>12} {'RMSE Ratio':>12} {'Bias':>15}")
    print(f"{'─'*75}")
    
    for result in sorted(all_results, key=lambda x: x['overall']['r2'], reverse=True):
        model = result['model_type']
        overall_r2 = result['overall']['r2']
        crisis_r2 = result['crisis']['r2']
        rmse_ratio = result['bias'].get('rmse_ratio', 0)
        severity = result['bias']['severity']
        
        rmse_str = f"{rmse_ratio:.2f}x" if rmse_ratio and not np.isnan(rmse_ratio) else "N/A"
        
        print(f"{model:<20} {overall_r2:>12.4f} {crisis_r2:>12.4f} {rmse_str:>12} {severity:>15}")
    
    print(f"{'─'*75}\n")
    
    # REMOVED RECOMMENDATION SECTION - Just report facts!
    print(f"{'='*80}")
    print(f"✅ BIAS TESTING COMPLETE FOR {target_name.upper()}")
    print(f"{'='*80}")
    print(f"\n   Tested {len(all_results)} models")
    print(f"   Results saved for final selection phase")
    
    # Save comparison (NO recommendation, just data!)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    comparison = {
        'target': target_name,
        'models_tested': len(all_results),
        'models': all_results,
        'note': 'This is bias testing only. Final selection happens in separate script.'
    }
    
    comparison_file = output_path / f"{target_name}_all_models_bias_comparison.json"
    with open(comparison_file, 'w') as f:
        json.dump(comparison, f, indent=2)
    
    print(f"\n   ✅ Comparison saved: {comparison_file}")
    
    # Create visualization
    if len(all_results) > 1:
        plot_file = create_comparison_visualization(target_name, all_results, output_dir)
        print(f"   ✅ Visualization: {plot_file}")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Test ALL models for bias BEFORE final selection (CORRECT approach!)"
    )
    parser.add_argument(
        '--target',
        type=str,
        required=True,
        choices=['revenue', 'eps', 'debt_equity', 'profit_margin', 'stock_return', 'all'],
        help='Target variable'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='reports/all_models_bias',
        help='Output directory'
    )
    
    args = parser.parse_args()
    
    print(f"\n{'='*80}")
    print(f"🎯 COMPREHENSIVE BIAS TESTING - ALL MODELS")
    print(f"{'='*80}")
    print(f"CORRECT APPROACH: Test ALL models BEFORE final selection!")
    print(f"{'='*80}\n")
    
    # Load test data once (reuse for all models)
    print(f"{'='*80}")
    print(f"📥 LOADING TEST DATA")
    print(f"{'='*80}\n")
    
    test_df = pd.read_csv('data/splits/test_data.csv')
    test_df = create_crisis_flag(test_df)
    
    crisis_count = test_df['crisis_flag'].sum()
    print(f"   ✅ Total samples: {len(test_df):,}")
    print(f"   ✅ Crisis: {crisis_count} ({crisis_count/len(test_df)*100:.1f}%)")
    print(f"   ✅ Normal: {len(test_df)-crisis_count} ({(len(test_df)-crisis_count)/len(test_df)*100:.1f}%)")
    
    if args.target == 'all':
        targets = ['revenue', 'eps', 'debt_equity', 'profit_margin', 'stock_return']
        
        all_target_results = {}
        
        for target in targets:
            results = test_all_models_for_target(target, test_df, args.output_dir)
            if results:
                all_target_results[target] = results
        
        # Summary (NO recommendations - just report what was tested!)
        print(f"\n{'='*80}")
        print(f"📊 BIAS TESTING SUMMARY - ALL TARGETS")
        print(f"{'='*80}\n")
        
        print(f"{'Target':<20} {'Models Tested':>15} {'Models with NO Bias':>20}")
        print(f"{'─'*60}")
        
        for target, results in all_target_results.items():
            models_tested = len(results)
            no_bias = sum(1 for r in results if r['bias']['severity'] == 'NONE')
            
            print(f"{target:<20} {models_tested:>15} {no_bias:>20}")
        
        print(f"\n{'='*80}")
        print(f"✅ BIAS TESTING COMPLETE FOR ALL TARGETS")
        print(f"{'='*80}")
        print(f"\n   All bias reports saved to: {args.output_dir}/")
        
    else:
        test_all_models_for_target(args.target, test_df, args.output_dir)
    
    print(f"\n{'='*80}")
    print(f"📁 OUTPUTS (Bias Reports Only)")
    print(f"{'='*80}")
    print(f"\nSaved to: {args.output_dir}/")
    print(f"   - {{target}}_all_models_bias_comparison.json (all models tested)")
    print(f"   - {{target}}_all_models_bias_comparison.png (visualizations)")
    
    print(f"\n{'='*80}")
    print(f"🎯 NEXT STEP: FINAL MODEL SELECTION")
    print(f"{'='*80}")
    print(f"\nNow that ALL models have been tested for bias,")
    print(f"run the selection script to make final decision:")
    print(f"\n   python src/evaluation/final_model_selection_after_bias.py")
    print(f"\nThis will:")
    print(f"   ✅ Compare R² vs bias trade-offs")
    print(f"   ✅ Make final production model choice")
    print(f"   ✅ Document reasoning")
    
    print(f"\n{'='*80}")


if __name__ == "__main__":
    main()