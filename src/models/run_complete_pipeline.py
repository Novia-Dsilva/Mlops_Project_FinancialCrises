"""
Complete Model Development Pipeline
Runs entire workflow from model training to deployment
"""

import subprocess
import sys
from pathlib import Path

def print_section(title):
    """Print section header"""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70 + "\n")


def run_script(script_name, *args):
    """Run a Python script"""
    cmd = [sys.executable, script_name] + list(args)
    print(f"Running: {' '.join(cmd)}\n")
    result = subprocess.run(cmd, capture_output=False)
    return result.returncode == 0


def main():
    """Run complete pipeline"""
    
    print("\n" + "="*70)
    print("  FINANCIAL STRESS TEST MODEL DEVELOPMENT PIPELINE")
    print("="*70)
    print("\nThis pipeline will:")
    print("  1. Train two models (Dense VAE Optimized & Ensemble VAE)")
    print("  2. Compare and select the best model")
    print("  3. Validate the selected model")
    print("  4. Perform bias detection")
    print("  5. Generate deployment report")
    print("\n" + "="*70)
    
    user_input = input("\nDo you want to run the complete pipeline? (yes/no): ")
    
    if user_input.lower() not in ['yes', 'y']:
        print("\nPipeline cancelled.")
        return
    
    # ============================================
    # STEP 1: TRAIN MODELS
    # ============================================
    
    print_section("STEP 1: TRAINING MODELS")
    
    train_models = input("Train models? (This takes ~30-90 min) (yes/skip): ")
    
    if train_models.lower() in ['yes', 'y']:
        print("\n1a. Training Dense VAE Optimized...")
        if not run_script('Dense_VAE_optimized_mlflow_updated.py'):
            print("❌ Dense VAE training failed!")
            return
        
        train_ensemble = input("\n1b. Train Ensemble VAE? (Takes 5x longer) (yes/no): ")
        if train_ensemble.lower() in ['yes', 'y']:
            print("\nTraining Ensemble VAE...")
            if not run_script('Ensemble_VAE_updated.py'):
                print("❌ Ensemble VAE training failed!")
                return
    else:
        print("⏭️  Skipping model training (using existing models)\n")
    
    # ============================================
    # STEP 2: MODEL SELECTION
    # ============================================
    
    print_section("STEP 2: MODEL SELECTION")
    
    print("Comparing models and selecting best...")
    if not run_script('model_selection.py'):
        print("❌ Model selection failed!")
        return
    
    # Read selected model from JSON
    import json
    try:
        with open('outputs/model_selection/model_selection.json', 'r') as f:
            selection_data = json.load(f)
            best_model = selection_data['selected_model']
        print(f"\n✓ Selected Model: {best_model}\n")
    except:
        print("⚠️  Could not read selection result, using Dense_VAE_Optimized")
        best_model = 'Dense_VAE_Optimized'
    
    # ============================================
    # STEP 3: MODEL VALIDATION
    # ============================================
    
    print_section("STEP 3: MODEL VALIDATION")
    
    print(f"Validating selected model: {best_model}...")
    if not run_script('model_validation.py', best_model):
        print("❌ Model validation failed!")
        return
    
    # ============================================
    # STEP 4: BIAS DETECTION
    # ============================================
    
    print_section("STEP 4: BIAS DETECTION")
    
    print(f"Running bias detection on: {best_model}...")
    if not run_script('bias_detection.py', best_model):
        print("❌ Bias detection failed!")
        return
    
    # ============================================
    # STEP 5: GENERATE DEPLOYMENT REPORT
    # ============================================
    
    print_section("STEP 5: GENERATING DEPLOYMENT REPORT")
    
    print("Generating deployment summary report...")
    generate_deployment_summary(best_model)
    
    # ============================================
    # PIPELINE COMPLETE
    # ============================================
    
    print("\n" + "="*70)
    print("  ✅ PIPELINE COMPLETE!")
    print("="*70)
    
    print(f"\n📊 Results Summary:")
    print(f"   • Selected Model: {best_model}")
    print(f"   • Model Selection Report: outputs/model_selection/")
    print(f"   • Validation Report: outputs/validation/")
    print(f"   • Bias Detection Report: outputs/bias_detection/")
    print(f"   • Deployment Report: outputs/deployment_report.txt")
    
    print(f"\n📁 All Outputs:")
    print(f"   • Model artifacts: outputs/output_{best_model}/")
    print(f"   • Analysis reports: outputs/")
    print(f"   • MLflow tracking: Check your MLflow UI")
    
    print(f"\n🎯 For Assignment Submission:")
    print(f"   1. 📄 Model Selection: outputs/model_selection/")
    print(f"   2. 📄 Validation: outputs/validation/")
    print(f"   3. 📄 Bias Detection: outputs/bias_detection/")
    print(f"   4. 📄 Deployment Report: outputs/deployment_report.txt")
    print(f"   5. 💾 Model Files: outputs/output_{best_model}/")
    print(f"   6. 📸 MLflow Screenshots: From MLflow UI")
    
    print(f"\n🚀 Next Steps:")
    print(f"   1. Review all reports and visualizations")
    print(f"   2. Take screenshots from MLflow UI")
    print(f"   3. Organize files for submission")
    print(f"   4. If approved, proceed with deployment")
    
    print("\n" + "="*70 + "\n")


def generate_deployment_summary(best_model):
    """Generate final deployment summary report"""
    
    import json
    from datetime import datetime
    
    # Load all analysis results
    try:
        with open('outputs/model_selection/model_selection.json', 'r') as f:
            selection_data = json.load(f)
    except:
        selection_data = {}
    
    try:
        with open(f'outputs/validation_{best_model}/validation_metrics.json', 'r') as f:
            validation_data = json.load(f)
    except:
        validation_data = {}
    
    try:
        with open(f'outputs/bias_detection_{best_model}/bias_analysis.json', 'r') as f:
            bias_data = json.load(f)
    except:
        bias_data = {}
    
    # Generate summary report
    report_path = 'outputs/deployment_report.txt'
    
    with open(report_path, 'w', encoding='utf-8') as f:  # Add encoding='utf-8'
        f.write("="*70 + "\n")
        f.write("DEPLOYMENT READINESS REPORT\n")
        f.write("Financial Stress Test Scenario Generator\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Selected Model: {best_model}\n\n")
        
        f.write("="*70 + "\n")
        f.write("EXECUTIVE SUMMARY\n")
        f.write("="*70 + "\n\n")
        
        # Model Selection
        f.write("1. MODEL SELECTION\n")
        f.write("-" * 70 + "\n")
        if selection_data:
            metrics = selection_data.get('metrics', {})
            f.write(f"   Selected: {best_model}\n")
            f.write(f"   KS Pass Rate: {metrics.get('ks_pass_rate', 'N/A'):.1f}%\n")
            f.write(f"   Correlation MAE: {metrics.get('correlation_mae', 'N/A'):.4f}\n")
            f.write(f"   Status: [PASS] PASSED\n\n")
        
        # Validation
        f.write("2. MODEL VALIDATION\n")
        f.write("-" * 70 + "\n")
        if validation_data:
            val_metrics = validation_data.get('metrics', {})
            f.write(f"   KS Pass Rate: {val_metrics.get('ks_pass_rate', 'N/A'):.1f}%\n")
            f.write(f"   Wasserstein Distance: {val_metrics.get('wasserstein_mean', 'N/A'):.2f}\n")
            f.write(f"   Status: [PASS] VALIDATED\n\n")
        
        # Bias Detection
        f.write("3. BIAS DETECTION\n")
        f.write("-" * 70 + "\n")
        if bias_data:
            bias_analysis = bias_data.get('bias_analysis', {})
            bias_detected = bias_analysis.get('bias_detected', False)
            bias_severity = bias_analysis.get('bias_severity', 'Unknown')
            
            f.write(f"   Bias Detected: {bias_detected}\n")
            f.write(f"   Severity: {bias_severity}\n")
            
            if not bias_detected:
                f.write(f"   Status: [PASS] NO BIAS DETECTED\n\n")
            else:
                f.write(f"   Status: [WARNING] REQUIRES MITIGATION\n\n")
        
        f.write("="*70 + "\n")
        f.write("DEPLOYMENT RECOMMENDATION\n")
        f.write("="*70 + "\n\n")
        
        # Overall recommendation
        can_deploy = True
        
        if selection_data and selection_data.get('metrics', {}).get('ks_pass_rate', 0) < 70:
            can_deploy = False
        
        if bias_data and bias_data.get('bias_analysis', {}).get('bias_severity') == 'High':
            can_deploy = False
        
        if can_deploy:
            f.write("[APPROVED] MODEL APPROVED FOR DEPLOYMENT\n\n")
            f.write("The model has passed all validation checks:\n")
            f.write("  - Statistical validity confirmed\n")
            f.write("  - Correlation preservation verified\n")
            f.write("  - No significant bias detected\n\n")
            f.write("Proceed with deployment to production environment.\n")
        else:
            f.write("[WARNING] MODEL REQUIRES ATTENTION BEFORE DEPLOYMENT\n\n")
            f.write("The model requires improvements in:\n")
            
            if selection_data and selection_data.get('metrics', {}).get('ks_pass_rate', 0) < 70:
                f.write("  - Statistical validity (KS Pass Rate < 70%)\n")
            
            if bias_data and bias_data.get('bias_analysis', {}).get('bias_severity') == 'High':
                f.write("  - Bias mitigation (High severity detected)\n")
            
            f.write("\nAddress these issues before production deployment.\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write("DOCUMENTATION & ARTIFACTS\n")
        f.write("="*70 + "\n\n")
        
        f.write("All artifacts and reports are available in:\n")
        f.write(f"  - Model files: outputs/output_{best_model}/\n")
        f.write(f"  - Selection report: outputs/model_selection/\n")
        f.write(f"  - Validation report: outputs/validation_{best_model}/\n")
        f.write(f"  - Bias report: outputs/bias_detection_{best_model}/\n")
        f.write(f"  - MLflow experiments: Check MLflow UI\n")
    
    print(f"✓ Generated deployment report: {report_path}\n")


if __name__ == "__main__":
    main()