"""
Bias Detection Script
Checks model performance across different data slices
Ensures fair scenario generation across economic conditions
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import ks_2samp
import json

# ============================================
# 1. DEFINE DATA SLICES
# ============================================

def create_data_slices(data):
    """
    Create meaningful slices of financial data
    Based on economic conditions, not demographics
    """
    
    print("="*60)
    print("BIAS DETECTION - DATA SLICING")
    print("="*60 + "\n")
    
    slices = {}
    
    # Slice 1: GDP Levels (Economic Size)
    if 'GDP' in data.columns:
        gdp_25 = data['GDP'].quantile(0.25)
        gdp_75 = data['GDP'].quantile(0.75)
        
        slices['GDP_Low'] = data[data['GDP'] < gdp_25]
        slices['GDP_Medium'] = data[(data['GDP'] >= gdp_25) & (data['GDP'] < gdp_75)]
        slices['GDP_High'] = data[data['GDP'] >= gdp_75]
    
    # Slice 2: VIX Levels (Market Stress)
    if 'VIX' in data.columns:
        vix_25 = data['VIX'].quantile(0.25)
        vix_75 = data['VIX'].quantile(0.75)
        
        slices['VIX_Low'] = data[data['VIX'] < vix_25]
        slices['VIX_Medium'] = data[(data['VIX'] >= vix_25) & (data['VIX'] < vix_75)]
        slices['VIX_High'] = data[data['VIX'] >= vix_75]
    
    # Slice 3: Unemployment Levels
    if 'Unemployment_Rate' in data.columns:
        unemp_25 = data['Unemployment_Rate'].quantile(0.25)
        unemp_75 = data['Unemployment_Rate'].quantile(0.75)
        
        slices['Unemployment_Low'] = data[data['Unemployment_Rate'] < unemp_25]
        slices['Unemployment_Medium'] = data[(data['Unemployment_Rate'] >= unemp_25) & 
                                              (data['Unemployment_Rate'] < unemp_75)]
        slices['Unemployment_High'] = data[data['Unemployment_Rate'] >= unemp_75]
    
    # Slice 4: Time Periods (if Date available)
    # This helps detect temporal bias
    
    print("Created Data Slices:")
    print("-" * 60)
    for slice_name, slice_data in slices.items():
        print(f"  {slice_name:25s}: {len(slice_data):6,} samples")
    
    print("\n" + "="*60 + "\n")
    
    return slices


# ============================================
# 2. EVALUATE SLICES
# ============================================

def evaluate_slice_performance(original_slices, generated_data, feature_list):
    """Evaluate model performance on each slice"""
    
    print("="*60)
    print("SLICE PERFORMANCE EVALUATION")
    print("="*60 + "\n")
    
    # Remove non-numeric columns from generated data
    gen_cols_to_drop = ['Scenario', 'Severity', 'Stress_Level', 'Stress_Score', 'Crisis_Type', 'Crisis_Score']
    generated_numeric = generated_data.drop(columns=gen_cols_to_drop, errors='ignore')
    
    slice_results = {}
    
    for slice_name, original_slice in original_slices.items():
        print(f"Evaluating: {slice_name}")
        
        # Get common features
        common_features = list(set(original_slice.columns) & set(generated_numeric.columns) & set(feature_list))
        
        if len(common_features) == 0:
            print(f"  âš  No common features found")
            continue
        
        # For generated data, we need to identify which scenarios match this slice
        # We'll use the slice's feature characteristics
        slice_criteria = {}
        
        if 'GDP' in slice_name:
            if 'GDP' in generated_numeric.columns:
                if 'Low' in slice_name:
                    gdp_threshold = original_slice['GDP'].max()
                    matching_gen = generated_numeric[generated_numeric['GDP'] < gdp_threshold]
                elif 'High' in slice_name:
                    gdp_threshold = original_slice['GDP'].min()
                    matching_gen = generated_numeric[generated_numeric['GDP'] >= gdp_threshold]
                else:  # Medium
                    gdp_low = original_slice['GDP'].min()
                    gdp_high = original_slice['GDP'].max()
                    matching_gen = generated_numeric[(generated_numeric['GDP'] >= gdp_low) & 
                                                     (generated_numeric['GDP'] < gdp_high)]
        
        elif 'VIX' in slice_name:
            if 'VIX' in generated_numeric.columns:
                if 'Low' in slice_name:
                    vix_threshold = original_slice['VIX'].max()
                    matching_gen = generated_numeric[generated_numeric['VIX'] < vix_threshold]
                elif 'High' in slice_name:
                    vix_threshold = original_slice['VIX'].min()
                    matching_gen = generated_numeric[generated_numeric['VIX'] >= vix_threshold]
                else:  # Medium
                    vix_low = original_slice['VIX'].min()
                    vix_high = original_slice['VIX'].max()
                    matching_gen = generated_numeric[(generated_numeric['VIX'] >= vix_low) & 
                                                     (generated_numeric['VIX'] < vix_high)]
        
        elif 'Unemployment' in slice_name:
            if 'Unemployment_Rate' in generated_numeric.columns:
                if 'Low' in slice_name:
                    unemp_threshold = original_slice['Unemployment_Rate'].max()
                    matching_gen = generated_numeric[generated_numeric['Unemployment_Rate'] < unemp_threshold]
                elif 'High' in slice_name:
                    unemp_threshold = original_slice['Unemployment_Rate'].min()
                    matching_gen = generated_numeric[generated_numeric['Unemployment_Rate'] >= unemp_threshold]
                else:  # Medium
                    unemp_low = original_slice['Unemployment_Rate'].min()
                    unemp_high = original_slice['Unemployment_Rate'].max()
                    matching_gen = generated_numeric[(generated_numeric['Unemployment_Rate'] >= unemp_low) & 
                                                     (generated_numeric['Unemployment_Rate'] < unemp_high)]
        else:
            matching_gen = generated_numeric
        
        if len(matching_gen) == 0:
            print(f"  âš  No matching generated scenarios found")
            continue
        
        # Compute KS test for this slice
        ks_passed = 0
        ks_total = 0
        
        for feature in common_features:
            try:
                _, pval = ks_2samp(original_slice[feature], matching_gen[feature])
                ks_total += 1
                if pval > 0.05:
                    ks_passed += 1
            except:
                continue
        
        ks_pass_rate = (ks_passed / ks_total * 100) if ks_total > 0 else 0
        
        slice_results[slice_name] = {
            'original_count': len(original_slice),
            'generated_count': len(matching_gen),
            'ks_pass_rate': ks_pass_rate,
            'ks_passed': ks_passed,
            'ks_total': ks_total
        }
        
        print(f"  Original: {len(original_slice):,} | Generated: {len(matching_gen):,}")
        print(f"  KS Pass Rate: {ks_pass_rate:.1f}% ({ks_passed}/{ks_total})\n")
    
    return slice_results


# ============================================
# 3. DETECT BIAS
# ============================================

def detect_bias(slice_results):
    """Detect performance disparities across slices"""
    
    print("="*60)
    print("BIAS DETECTION ANALYSIS")
    print("="*60 + "\n")
    
    ks_pass_rates = [r['ks_pass_rate'] for r in slice_results.values()]
    
    if len(ks_pass_rates) == 0:
        print("âš  No slice results to analyze")
        return None
    
    mean_rate = np.mean(ks_pass_rates)
    std_rate = np.std(ks_pass_rates)
    min_rate = np.min(ks_pass_rates)
    max_rate = np.max(ks_pass_rates)
    range_rate = max_rate - min_rate
    
    print(f"Performance Across Slices:")
    print("-" * 60)
    print(f"  Mean KS Pass Rate:  {mean_rate:.1f}%")
    print(f"  Std Deviation:      {std_rate:.1f}%")
    print(f"  Min Pass Rate:      {min_rate:.1f}%")
    print(f"  Max Pass Rate:      {max_rate:.1f}%")
    print(f"  Performance Range:  {range_rate:.1f}%\n")
    
    # Bias detection thresholds
    bias_detected = False
    bias_severity = "None"
    
    if range_rate > 20:
        bias_detected = True
        bias_severity = "High"
        print("âš  HIGH BIAS DETECTED")
        print("  Performance varies >20% across slices")
    elif range_rate > 10:
        bias_detected = True
        bias_severity = "Moderate"
        print("âš  MODERATE BIAS DETECTED")
        print("  Performance varies 10-20% across slices")
    else:
        print("âœ“ NO SIGNIFICANT BIAS DETECTED")
        print("  Performance is consistent across slices")
    
    print("\n" + "="*60 + "\n")
    
    return {
        'bias_detected': bias_detected,
        'bias_severity': bias_severity,
        'mean_rate': mean_rate,
        'std_rate': std_rate,
        'range_rate': range_rate,
        'min_rate': min_rate,
        'max_rate': max_rate
    }


# ============================================
# 4. VISUALIZE BIAS
# ============================================

def create_bias_plots(slice_results, output_dir='outputs/bias_detection'):
    """Create bias detection visualizations"""
    
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract data for plotting
    slice_names = list(slice_results.keys())
    ks_rates = [slice_results[s]['ks_pass_rate'] for s in slice_names]
    
    # Plot: Performance across slices
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = ['green' if rate > 75 else 'orange' if rate > 60 else 'red' for rate in ks_rates]
    ax.barh(slice_names, ks_rates, color=colors, alpha=0.7)
    ax.axvline(x=75, color='green', linestyle='--', label='Target (75%)', linewidth=2)
    ax.axvline(x=np.mean(ks_rates), color='blue', linestyle='-', label=f'Mean ({np.mean(ks_rates):.1f}%)', linewidth=2)
    
    ax.set_xlabel('KS Pass Rate (%)')
    ax.set_title('Model Performance Across Data Slices\n(Bias Detection)')
    ax.legend()
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/bias_detection_slices.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"âœ“ Saved: {output_dir}/bias_detection_slices.png\n")


# ============================================
# 5. SAVE BIAS REPORT
# ============================================

def save_bias_report(model_name, slice_results, bias_analysis, output_dir='outputs/bias_detection'):
    """Save comprehensive bias detection report"""
    
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    report_path = f'{output_dir}/bias_detection_report.txt'
    
    with open(report_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("BIAS DETECTION REPORT\n")
        f.write("Financial Stress Test Scenario Generator\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"Model: {model_name}\n")
        f.write(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("="*60 + "\n")
        f.write("SLICING METHODOLOGY\n")
        f.write("="*60 + "\n\n")
        
        f.write("Data sliced by economic conditions:\n")
        f.write("1. GDP Levels (Low/Medium/High)\n")
        f.write("2. VIX Levels (Low/Medium/High) - Market Stress\n")
        f.write("3. Unemployment Levels (Low/Medium/High)\n\n")
        
        f.write("="*60 + "\n")
        f.write("SLICE PERFORMANCE\n")
        f.write("="*60 + "\n\n")
        
        for slice_name, results in slice_results.items():
            f.write(f"{slice_name}:\n")
            f.write(f"  Original Samples: {results['original_count']:,}\n")
            f.write(f"  Generated Samples: {results['generated_count']:,}\n")
            f.write(f"  KS Pass Rate: {results['ks_pass_rate']:.1f}%\n\n")
        
        f.write("="*60 + "\n")
        f.write("BIAS ANALYSIS\n")
        f.write("="*60 + "\n\n")
        
        if bias_analysis:
            f.write(f"Bias Detected: {bias_analysis['bias_detected']}\n")
            f.write(f"Severity: {bias_analysis['bias_severity']}\n\n")
            
            f.write(f"Mean Performance: {bias_analysis['mean_rate']:.1f}%\n")
            f.write(f"Performance Range: {bias_analysis['range_rate']:.1f}%\n")
            f.write(f"Min-Max: {bias_analysis['min_rate']:.1f}% - {bias_analysis['max_rate']:.1f}%\n\n")
            
            if bias_analysis['bias_detected']:
                f.write("="*60 + "\n")
                f.write("MITIGATION RECOMMENDATIONS\n")
                f.write("="*60 + "\n\n")
                
                if bias_analysis['bias_severity'] == 'High':
                    f.write("1. Re-train model with balanced sampling across slices\n")
                    f.write("2. Apply slice-specific weight adjustments\n")
                    f.write("3. Consider ensemble approach with slice-specific models\n")
                else:
                    f.write("1. Monitor slice performance in production\n")
                    f.write("2. Consider minor adjustments to training data\n")
            else:
                f.write("âœ“ Model performs consistently across all data slices.\n")
                f.write("  No bias mitigation required.\n")
        
        f.write("\n" + "="*60 + "\n")
        f.write("CONCLUSION\n")
        f.write("="*60 + "\n\n")
        
        if bias_analysis and not bias_analysis['bias_detected']:
            f.write("The model demonstrates fair and consistent performance\n")
            f.write("across different economic conditions. No significant bias\n")
            f.write("was detected in scenario generation.\n")
        else:
            f.write("Performance disparities were detected across slices.\n")
            f.write("Implement recommended mitigation strategies before\n")
            f.write("production deployment.\n")
    
    print(f"âœ“ Saved bias report: {report_path}")
    
    # Save as JSON
    json_path = f'{output_dir}/bias_analysis.json'
    with open(json_path, 'w') as f:
        json.dump({
            'model': model_name,
            'slice_results': slice_results,
            'bias_analysis': bias_analysis,
            'timestamp': pd.Timestamp.now().isoformat()
        }, f, indent=2)
    
    print(f"âœ“ Saved bias analysis JSON: {json_path}\n")


# ============================================
# 6. MAIN EXECUTION
# ============================================

def main(model_name='Dense_VAE_Optimized'):
    """Main bias detection pipeline"""
    
    # Determine model directory
    if model_name == 'Dense_VAE_Optimized':
        model_dir = Path('outputs/output_Dense_VAE_optimized')
        scenarios_file = 'dense_vae_optimized_scenarios.csv'
    else:
        model_dir = Path('outputs/output_Ensemble_VAE')
        scenarios_file = 'ensemble_vae_scenarios.csv'
    
    # Load data
    print(f"Loading data for: {model_name}\n")
    scenarios = pd.read_csv(model_dir / scenarios_file)
    
    # Load original data - try multiple paths
    possible_paths = [
        'data/features/macro_features_clean.csv',
        '../../data/features/macro_features_clean.csv',
        '../../../data/features/macro_features_clean.csv',
        Path(__file__).parent.parent.parent / 'data/features/macro_features_clean.csv'
    ]
    
    original_data = None
    for data_path in possible_paths:
        try:
            original_data = pd.read_csv(data_path)
            print(f"âœ“ Found data at: {data_path}\n")
            break
        except FileNotFoundError:
            continue
    
    if original_data is None:
        raise FileNotFoundError(
            "Could not find macro_features_clean.csv\n" +
            "Please ensure the file exists in: data/features/macro_features_clean.csv"
        )
    
    if 'Date' in original_data.columns:
        original_data = original_data.drop('Date', axis=1)
    
    # ============================================
    # CRITICAL: Use hold-out test set only
    # ============================================
    
    print("âš ï¸  IMPORTANT: Using hold-out test set for bias detection!")
    print("Replicating training split...\n")
    
    from sklearn.model_selection import train_test_split
    
    # Use SAME split as training
    train_val, test_data = train_test_split(
        original_data,
        test_size=0.2,
        random_state=42
    )
    
    print(f"Full data: {len(original_data):,} rows")
    print(f"Test set: {len(test_data):,} rows (hold-out) âœ…\n")
    
    # Create slices from TEST DATA ONLY
    original_slices = create_data_slices(test_data)
    
    # Evaluate performance on slices
    feature_list = list(original_data.columns)
    slice_results = evaluate_slice_performance(original_slices, scenarios, feature_list)
    
    # Detect bias
    bias_analysis = detect_bias(slice_results)
    
    # Create visualizations
    create_bias_plots(slice_results)
    
    # Save report
    save_bias_report(model_name, slice_results, bias_analysis)
    
    print("="*60)
    print("âœ… BIAS DETECTION COMPLETE")
    print("="*60)
    print(f"\nAnalyzed Model: {model_name}")
    print(f"Results saved in: outputs/bias_detection/")
    print("\nNext steps:")
    print("1. Review bias_detection_report.txt")
    print("2. Examine bias_detection_slices.png")
    print("3. Implement mitigation if needed")
    print("="*60 + "\n")


if __name__ == "__main__":
    import sys
    
    # Get model name from command line or use default
    model_name = sys.argv[1] if len(sys.argv) > 1 else 'Dense_VAE_Optimized'
    
    main(model_name)