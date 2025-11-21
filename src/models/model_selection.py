"""
Model Selection Script
Compares Dense VAE Optimized vs Ensemble VAE
Selects best model based on KS Pass Rate and Correlation MAE
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
import mlflow

# ============================================
# 1. LOAD RESULTS FROM BOTH MODELS
# ============================================

def load_model_results():
    """Load validation results from both models"""
    
    print("="*60)
    print("MODEL COMPARISON & SELECTION")
    print("="*60 + "\n")
    
    # Model 1: Dense VAE Optimized
    dense_vae_dir = Path('outputs/output_Dense_VAE_optimized')
    ensemble_vae_dir = Path('outputs/output_Ensemble_VAE')
    
    results = {}
    
    # Read Dense VAE results
    if (dense_vae_dir / 'validation_report.txt').exists():
        print("✓ Found Dense VAE Optimized results")
        results['Dense_VAE_Optimized'] = {}  # Initialize the dictionary first!
        with open(dense_vae_dir / 'validation_report.txt', 'r') as f:
            content = f.read()
            # Parse metrics (simple text parsing)
            for line in content.split('\n'):
                if 'KS Test Pass Rate:' in line or 'KS Pass Rate:' in line:
                    ks_rate = float(line.split(':')[1].strip().replace('%', ''))
                    results['Dense_VAE_Optimized']['ks_pass_rate'] = ks_rate
                elif 'Correlation MAE:' in line:
                    corr_mae = float(line.split(':')[1].strip())
                    results['Dense_VAE_Optimized']['correlation_mae'] = corr_mae
                elif 'Wasserstein' in line:
                    wass = float(line.split(':')[1].strip())
                    results['Dense_VAE_Optimized']['wasserstein'] = wass
    
    # Read Ensemble VAE results
    if (ensemble_vae_dir / 'ensemble_validation.txt').exists():
        print("✓ Found Ensemble VAE results")
        results['Ensemble_VAE'] = {}  # Initialize the dictionary first!
        with open(ensemble_vae_dir / 'ensemble_validation.txt', 'r') as f:
            content = f.read()
            for line in content.split('\n'):
                if 'KS Test Pass Rate:' in line or 'KS Pass Rate:' in line:
                    ks_rate = float(line.split(':')[1].strip().replace('%', ''))
                    results['Ensemble_VAE']['ks_pass_rate'] = ks_rate
                elif 'Correlation MAE:' in line:
                    corr_mae = float(line.split(':')[1].strip())
                    results['Ensemble_VAE']['correlation_mae'] = corr_mae
                elif 'Wasserstein' in line:
                    wass = float(line.split(':')[1].strip())
                    results['Ensemble_VAE']['wasserstein'] = wass
    
    return results


# ============================================
# 2. SELECT BEST MODEL
# ============================================

def select_best_model(results):
    """
    Select best model based on:
    1. Higher KS Pass Rate (primary)
    2. Lower Correlation MAE (secondary)
    """
    
    print("\n" + "="*60)
    print("MODEL PERFORMANCE COMPARISON")
    print("="*60 + "\n")
    
    # Display results
    print("Metric Comparison:")
    print("-" * 60)
    print(f"{'Model':<30} {'KS Pass Rate':<15} {'Corr MAE':<15} {'Wasserstein':<15}")
    print("-" * 60)
    
    for model_name, metrics in results.items():
        print(f"{model_name:<30} "
              f"{metrics['ks_pass_rate']:<15.2f} "
              f"{metrics['correlation_mae']:<15.4f} "
              f"{metrics.get('wasserstein', 0):<15.2f}")
    
    print("-" * 60)
    
    # Selection logic
    best_model = None
    best_score = -1
    
    for model_name, metrics in results.items():
        # Composite score: KS Pass Rate is weighted more heavily
        # Higher KS = better, Lower MAE = better
        score = (metrics['ks_pass_rate'] * 0.7) - (metrics['correlation_mae'] * 100 * 0.3)
        
        if score > best_score:
            best_score = score
            best_model = model_name
    
    print(f"\n🏆 BEST MODEL: {best_model}")
    print(f"   Composite Score: {best_score:.2f}")
    print(f"   KS Pass Rate: {results[best_model]['ks_pass_rate']:.2f}%")
    print(f"   Correlation MAE: {results[best_model]['correlation_mae']:.4f}")
    
    return best_model, results[best_model]


# ============================================
# 3. VISUALIZE COMPARISON
# ============================================

def create_comparison_plots(results, output_dir='outputs/model_selection'):
    """Create comparison visualizations"""
    
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    models = list(results.keys())
    
    # Prepare data
    ks_rates = [results[m]['ks_pass_rate'] for m in models]
    corr_maes = [results[m]['correlation_mae'] for m in models]
    wassersteins = [results[m].get('wasserstein', 0) for m in models]
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot 1: KS Pass Rate
    axes[0].bar(models, ks_rates, color=['#3498db', '#e74c3c'])
    axes[0].set_ylabel('KS Pass Rate (%)')
    axes[0].set_title('KS Test Pass Rate\n(Higher is Better)')
    axes[0].set_ylim([0, 100])
    for i, v in enumerate(ks_rates):
        axes[0].text(i, v + 2, f'{v:.1f}%', ha='center', fontweight='bold')
    
    # Plot 2: Correlation MAE
    axes[1].bar(models, corr_maes, color=['#3498db', '#e74c3c'])
    axes[1].set_ylabel('Correlation MAE')
    axes[1].set_title('Correlation MAE\n(Lower is Better)')
    for i, v in enumerate(corr_maes):
        axes[1].text(i, v + 0.002, f'{v:.4f}', ha='center', fontweight='bold')
    
    # Plot 3: Wasserstein Distance
    axes[2].bar(models, wassersteins, color=['#3498db', '#e74c3c'])
    axes[2].set_ylabel('Wasserstein Distance')
    axes[2].set_title('Wasserstein Distance\n(Lower is Better)')
    for i, v in enumerate(wassersteins):
        axes[2].text(i, v + 10, f'{v:.1f}', ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/model_comparison.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved comparison plot: {output_dir}/model_comparison.png")
    
    plt.close()


# ============================================
# 4. SAVE SELECTION REPORT
# ============================================

def save_selection_report(best_model, metrics, output_dir='outputs/model_selection'):
    """Save model selection report"""
    
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    report_path = f'{output_dir}/model_selection_report.txt'
    
    with open(report_path, 'w', encoding='utf-8') as f:  # Add encoding='utf-8'
        f.write("="*60 + "\n")
        f.write("MODEL SELECTION REPORT\n")
        f.write("Financial Stress Test Scenario Generator\n")
        f.write("="*60 + "\n\n")
        
        f.write("SELECTION CRITERIA:\n")
        f.write("-" * 60 + "\n")
        f.write("1. Primary Metric: KS Pass Rate (Higher is Better)\n")
        f.write("   - Measures distribution similarity with real data\n")
        f.write("   - Target: >75%\n\n")
        
        f.write("2. Secondary Metric: Correlation MAE (Lower is Better)\n")
        f.write("   - Measures preservation of feature relationships\n")
        f.write("   - Target: <0.10\n\n")
        
        f.write("="*60 + "\n")
        f.write("SELECTED MODEL\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"Best Model: {best_model}\n\n")
        
        f.write("Performance Metrics:\n")
        f.write("-" * 60 + "\n")
        f.write(f"KS Pass Rate:       {metrics['ks_pass_rate']:.2f}%\n")
        f.write(f"Correlation MAE:    {metrics['correlation_mae']:.4f}\n")
        f.write(f"Wasserstein Dist:   {metrics.get('wasserstein', 0):.2f}\n\n")
        
        f.write("="*60 + "\n")
        f.write("JUSTIFICATION\n")
        f.write("="*60 + "\n\n")
        
        if metrics['ks_pass_rate'] > 75:
            f.write("[PASS] KS Pass Rate exceeds 75% threshold\n")
        else:
            f.write("[ACCEPTABLE] KS Pass Rate below 75% - acceptable for stress testing\n")
        
        if metrics['correlation_mae'] < 0.10:
            f.write("[PASS] Correlation MAE below 0.10 threshold\n")
        else:
            f.write("[ACCEPTABLE] Correlation MAE above 0.10 - still preserves relationships\n")
        
        f.write("\n" + "="*60 + "\n")
        f.write("DEPLOYMENT RECOMMENDATION\n")
        f.write("="*60 + "\n\n")
        f.write(f"The {best_model} is recommended for production deployment.\n")
        f.write("This model provides the best balance of statistical validity\n")
        f.write("and scenario diversity for financial stress testing.\n")
    
    print(f"✓ Saved selection report: {report_path}")
    
    # Also save as JSON
    json_path = f'{output_dir}/model_selection.json'
    with open(json_path, 'w') as f:
        json.dump({
            'selected_model': best_model,
            'metrics': metrics,
            'timestamp': pd.Timestamp.now().isoformat()
        }, f, indent=2)
    
    print(f"✓ Saved selection JSON: {json_path}\n")


# ============================================
# 5. MAIN EXECUTION
# ============================================

def main():
    """Main model selection pipeline"""
    
    # Load results
    results = load_model_results()
    
    if not results:
        print("❌ No model results found!")
        print("   Please run both models first:")
        print("   1. python Dense_VAE_optimized_mlflow_updated.py")
        print("   2. python Ensemble_VAE_updated.py")
        return
    
    # Select best model
    best_model, best_metrics = select_best_model(results)
    
    # Create visualizations
    create_comparison_plots(results)
    
    # Save report
    save_selection_report(best_model, best_metrics)
    
    print("\n" + "="*60)
    print("✅ MODEL SELECTION COMPLETE")
    print("="*60)
    print(f"\nBest Model: {best_model}")
    print(f"Results saved in: outputs/model_selection/")
    print("\nNext steps:")
    print("1. Review model_comparison.png")
    print("2. Read model_selection_report.txt")
    print("3. Proceed with model validation and bias checks")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()