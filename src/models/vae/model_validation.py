"""
Model Validation Script
Validates the selected model on generated scenarios
Simple quality checks on generated data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

# ============================================
# 1. LOAD SCENARIOS
# ============================================

def load_validation_data(model_name='Dense_VAE_Optimized'):
    """Load generated scenarios"""
    
    print("="*60)
    print("MODEL VALIDATION")
    print("="*60 + "\n")
    
    # Determine model directory
    if model_name == 'Dense_VAE_Optimized':
        model_dir = Path('outputs/output_Dense_VAE_optimized')
        scenarios_file = 'dense_vae_optimized_scenarios.csv'
    else:
        model_dir = Path('outputs/output_Ensemble_VAE')
        scenarios_file = 'ensemble_vae_scenarios.csv'
    
    print(f"Validating: {model_name}")
    print(f"Model Directory: {model_dir}\n")
    
    # Load generated scenarios
    scenarios = pd.read_csv(model_dir / scenarios_file)
    
    # Drop categorical columns that might cause issues
    categorical_cols = scenarios.select_dtypes(include=['object']).columns.tolist()
    
    # Keep only Scenario and Severity columns, drop others
    cols_to_keep = ['Scenario', 'Severity', 'Stress_Level', 'Stress_Score', 'Crisis_Type', 'Crisis_Score']
    categorical_to_drop = [col for col in categorical_cols if col not in cols_to_keep]
    
    if categorical_to_drop:
        print(f"⚠️  Dropping categorical columns: {categorical_to_drop}")
        scenarios = scenarios.drop(columns=categorical_to_drop)
    
    print(f"✓ Loaded {len(scenarios)} generated scenarios")
    print(f"✓ Features: {len(scenarios.columns) - len([c for c in cols_to_keep if c in scenarios.columns])} numeric\n")
    
    return scenarios


# ============================================
# 2. COMPUTE METRICS
# ============================================

def compute_validation_metrics(scenarios):
    """Compute simple validation metrics"""
    
    print("="*60)
    print("COMPUTING VALIDATION METRICS")
    print("="*60 + "\n")
    
    # Remove non-numeric columns
    scenario_cols_to_drop = ['Scenario', 'Severity', 'Stress_Level', 'Stress_Score', 'Crisis_Type', 'Crisis_Score']
    scenarios_numeric = scenarios.drop(columns=scenario_cols_to_drop, errors='ignore')
    
    # Keep ONLY numeric columns
    scenarios_numeric = scenarios_numeric.select_dtypes(include=[np.number])
    
    print(f"Validating {len(scenarios_numeric.columns)} features\n")
    
    metrics = {}
    
    # 1. Data Quality
    invalid_count = scenarios_numeric.isnull().sum().sum()
    inf_count = np.isinf(scenarios_numeric.values).sum()
    
    metrics['invalid_values'] = int(invalid_count)
    metrics['inf_values'] = int(inf_count)
    metrics['data_quality'] = 'PASS' if (invalid_count == 0 and inf_count == 0) else 'FAIL'
    
    print(f"1. Data Quality Check:")
    print(f"   Invalid (NaN) values: {invalid_count}")
    print(f"   Infinite values: {inf_count}")
    print(f"   Status: {metrics['data_quality']}")
    
    # 2. Value Ranges
    print(f"\n2. Value Range Check:")
    range_issues = []
    
    if 'VIX' in scenarios_numeric.columns:
        vix_min = scenarios_numeric['VIX'].min()
        vix_max = scenarios_numeric['VIX'].max()
        print(f"   VIX range: [{vix_min:.1f}, {vix_max:.1f}]")
        if vix_min < 0 or vix_max > 150:
            range_issues.append('VIX')
    
    if 'Unemployment_Rate' in scenarios_numeric.columns:
        unemp_min = scenarios_numeric['Unemployment_Rate'].min()
        unemp_max = scenarios_numeric['Unemployment_Rate'].max()
        print(f"   Unemployment range: [{unemp_min:.1f}%, {unemp_max:.1f}%]")
        if unemp_min < 0 or unemp_max > 30:
            range_issues.append('Unemployment_Rate')
    
    if 'GDP' in scenarios_numeric.columns:
        gdp_min = scenarios_numeric['GDP'].min()
        gdp_max = scenarios_numeric['GDP'].max()
        print(f"   GDP range: [{gdp_min:,.0f}, {gdp_max:,.0f}]")
        if gdp_min < 1000 or gdp_max > 50000:
            range_issues.append('GDP')
    
    metrics['range_check'] = 'PASS' if len(range_issues) == 0 else 'FAIL'
    metrics['problematic_features'] = range_issues
    print(f"   Status: {metrics['range_check']}")
    
    # 3. Scenario Diversity
    if 'Severity' in scenarios.columns:
        severity_dist = scenarios['Severity'].value_counts()
        metrics['scenario_diversity'] = {k: int(v) for k, v in severity_dist.items()}
        
        print(f"\n3. Scenario Diversity:")
        for severity, count in severity_dist.items():
            pct = 100 * count / len(scenarios)
            print(f"   {severity}: {count} ({pct:.1f}%)")
        
        baseline_pct = (scenarios['Severity'] == 'Baseline').sum() / len(scenarios) * 100
        severe_pct = (scenarios['Severity'] == 'Severe').sum() / len(scenarios) * 100
        
        diversity_ok = baseline_pct >= 5 and severe_pct <= 70
        metrics['diversity_check'] = 'PASS' if diversity_ok else 'REVIEW'
        print(f"   Diversity Status: {metrics['diversity_check']}")
    
    # 4. Statistical Properties
    print(f"\n4. Statistical Properties:")
    
    metrics['mean_values'] = {col: float(val) for col, val in scenarios_numeric.mean().items()}
    metrics['std_values'] = {col: float(val) for col, val in scenarios_numeric.std().items()}
    
    print(f"   Features analyzed: {len(scenarios_numeric.columns)}")
    print(f"   Mean values computed: ✓")
    print(f"   Std deviations computed: ✓")
    
    # 5. Correlation Matrix
    print(f"\n5. Correlation Matrix:")
    n_features = min(20, len(scenarios_numeric.columns))
    corr_matrix = scenarios_numeric.iloc[:, :n_features].corr()
    
    perfect_corr = ((corr_matrix.abs() > 0.99) & (corr_matrix.abs() < 1.0)).sum().sum()
    
    metrics['perfect_correlations'] = int(perfect_corr)
    metrics['correlation_check'] = 'PASS' if perfect_corr == 0 else 'REVIEW'
    
    print(f"   Correlation matrix computed: {n_features}×{n_features}")
    print(f"   Perfect correlations found: {perfect_corr}")
    print(f"   Status: {metrics['correlation_check']}")
    
    print("\n" + "="*60 + "\n")
    
    return metrics


# ============================================
# 3. CREATE PLOTS
# ============================================

def create_validation_plots(scenarios, output_dir='outputs/validation'):
    """Create validation visualizations"""
    
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Remove non-numeric columns
    scenario_cols_to_drop = ['Scenario', 'Severity', 'Stress_Level', 'Stress_Score', 'Crisis_Type', 'Crisis_Score']
    scenarios_numeric = scenarios.drop(columns=scenario_cols_to_drop, errors='ignore')
    scenarios_numeric = scenarios_numeric.select_dtypes(include=[np.number])
    
    # Plot 1: Feature Distributions
    key_features = ['GDP', 'VIX', 'Unemployment_Rate', 'SP500_Close', 'CPI', 'Oil_Price']
    key_features = [f for f in key_features if f in scenarios_numeric.columns][:6]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, feature in enumerate(key_features):
        axes[idx].hist(scenarios_numeric[feature], bins=30, alpha=0.7, color='steelblue', edgecolor='black')
        axes[idx].set_title(f'{feature} Distribution', fontweight='bold')
        axes[idx].set_xlabel('Value')
        axes[idx].set_ylabel('Frequency')
        axes[idx].grid(alpha=0.3)
        
        mean_val = scenarios_numeric[feature].mean()
        axes[idx].axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.1f}')
        axes[idx].legend()
    
    for idx in range(len(key_features), 6):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/feature_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_dir}/feature_distributions.png")
    
    # Plot 2: Scenario Diversity
    if 'Severity' in scenarios.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        severity_counts = scenarios['Severity'].value_counts()
        colors = {'Baseline': '#2ecc71', 'Adverse': '#f39c12', 'Severe': '#e67e22', 'Extreme': '#e74c3c'}
        bar_colors = [colors.get(sev, 'steelblue') for sev in severity_counts.index]
        
        bars = ax.bar(severity_counts.index, severity_counts.values, color=bar_colors, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Severity Level', fontweight='bold')
        ax.set_ylabel('Count', fontweight='bold')
        ax.set_title('Generated Scenario Distribution', fontweight='bold', fontsize=14)
        ax.grid(axis='y', alpha=0.3)
        
        for bar, count in zip(bars, severity_counts.values):
            height = bar.get_height()
            pct = 100 * count / len(scenarios)
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{count}\n({pct:.1f}%)', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/scenario_diversity.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {output_dir}/scenario_diversity.png")
    
    # Plot 3: Correlation Heatmap
    n_features = min(15, len(scenarios_numeric.columns))
    corr_matrix = scenarios_numeric.iloc[:, :n_features].corr()
    
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, ax=ax, vmin=-1, vmax=1, square=True)
    ax.set_title('Feature Correlation Matrix', fontweight='bold', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_dir}/correlation_matrix.png\n")


# ============================================
# 4. SAVE REPORT
# ============================================

def save_validation_report(model_name, metrics, output_dir='outputs/validation'):
    """Save validation report"""
    
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    report_path = f'{output_dir}/validation_report.txt'
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("MODEL VALIDATION REPORT\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"Model: {model_name}\n")
        f.write(f"Validation Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("VALIDATION CHECKS:\n")
        f.write("-" * 60 + "\n\n")
        
        f.write("1. Data Quality\n")
        f.write(f"   Invalid values: {metrics['invalid_values']}\n")
        f.write(f"   Infinite values: {metrics['inf_values']}\n")
        f.write(f"   Status: {metrics['data_quality']}\n\n")
        
        f.write("2. Value Ranges\n")
        f.write(f"   Status: {metrics['range_check']}\n")
        if metrics['problematic_features']:
            f.write(f"   Issues: {', '.join(metrics['problematic_features'])}\n")
        else:
            f.write(f"   All features within realistic ranges\n")
        f.write("\n")
        
        f.write("3. Scenario Diversity\n")
        for severity, count in metrics['scenario_diversity'].items():
            f.write(f"   {severity}: {count}\n")
        f.write(f"   Status: {metrics['diversity_check']}\n\n")
        
        f.write("4. Correlation Quality\n")
        f.write(f"   Perfect correlations: {metrics['perfect_correlations']}\n")
        f.write(f"   Status: {metrics['correlation_check']}\n\n")
        
        f.write("="*60 + "\n")
        f.write("OVERALL ASSESSMENT\n")
        f.write("="*60 + "\n\n")
        
        all_pass = (
            metrics['data_quality'] == 'PASS' and
            metrics['range_check'] == 'PASS' and
            metrics['diversity_check'] == 'PASS'
        )
        
        if all_pass:
            f.write("[PASS] Model validated successfully\n")
        else:
            f.write("[REVIEW] Some checks need attention\n")
    
    print(f"✓ Saved validation report: {report_path}")
    
    # Save JSON
    json_path = f'{output_dir}/validation_metrics.json'
    with open(json_path, 'w') as f:
        json.dump({
            'model': model_name,
            'metrics': metrics,
            'timestamp': pd.Timestamp.now().isoformat()
        }, f, indent=2)
    
    print(f"✓ Saved validation metrics: {json_path}\n")


# ============================================
# 5. MAIN
# ============================================

def main(model_name='Dense_VAE_Optimized'):
    """Main validation pipeline"""
    
    scenarios = load_validation_data(model_name)
    metrics = compute_validation_metrics(scenarios)
    create_validation_plots(scenarios)
    save_validation_report(model_name, metrics)
    
    print("="*60)
    print("✅ MODEL VALIDATION COMPLETE")
    print("="*60)
    print(f"\nValidated Model: {model_name}")
    print(f"Results saved in: outputs/validation/")
    print("\nFiles created:")
    print("  - validation_report.txt")
    print("  - feature_distributions.png")
    print("  - scenario_diversity.png")
    print("  - correlation_matrix.png")
    print("="*60 + "\n")


if __name__ == "__main__":
    import sys
    model_name = sys.argv[1] if len(sys.argv) > 1 else 'Dense_VAE_Optimized'
    main(model_name)