
# """
# src/evaluation/detect_crisis_bias.py

# Crisis-Based Model Bias Detection - For Financial Stress Testing

# Detects if model:
# 1. Underperforms during crisis periods (higher RMSE in crises)
# 2. Systematically underestimates losses during crises (optimistic bias)
# 3. Works well in calm periods but fails in stress scenarios

# Primary Slice: Crisis vs Non-Crisis (2007-2009, 2020-2021, VIX>30)

# Usage:
#     python src/evaluation/detect_crisis_bias.py --model xgboost --target revenue
#     python src/evaluation/detect_crisis_bias.py --model xgboost --target all
#     python src/evaluation/detect_crisis_bias.py --model xgboost --target revenue --test-data data/splits/val_data.csv
# """

# import pandas as pd
# import numpy as np
# from pathlib import Path
# import json
# import argparse
# import warnings
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# from sklearn.impute import SimpleImputer
# import joblib
# import sys

# warnings.filterwarnings('ignore')

# # Setup paths
# project_root = Path(__file__).resolve().parent.parent.parent
# sys.path.insert(0, str(project_root / "src"))


# class CrisisBiasDetector:
#     """
#     Detects model bias during crisis vs non-crisis periods
#     """
    
#     def __init__(self, model_type: str, target_name: str):
#         self.model_type = model_type
#         self.target_name = target_name
#         self.target_col = f'target_{target_name}'
#         self.model = None
#         self.feature_names = None
#         self.test_df_raw = None
#         self.predictions = None
#         self.actuals = None
#         self.crisis_flags = None
#         self.sectors = None
#         self.metrics = {}
#         self.bias_report = {}
        
#     def load_model(self, model_path: str):
#         """Load trained model"""
#         print(f"\n{'='*80}")
#         print(f"📥 LOADING MODEL: {self.model_type.upper()}")
#         print(f"{'='*80}")
        
#         model_file = Path(model_path)
#         if not model_file.exists():
#             raise FileNotFoundError(f"Model not found: {model_path}")
        
#         model_data = joblib.load(model_file)
#         self.model = model_data['model']
#         self.feature_names = model_data['feature_names']
        
#         print(f"   ✅ Loaded {self.model_type.upper()} model")
#         print(f"   Features: {len(self.feature_names)}")
    
#     def load_and_prepare_test_data(self, test_data_path: str):
#         """Load test data and create crisis flag"""
#         print(f"\n{'='*80}")
#         print(f"📥 LOADING AND PREPARING TEST DATA")
#         print(f"{'='*80}")
        
#         # Load raw test data
#         self.test_df_raw = pd.read_csv(test_data_path)
#         print(f"   ✅ Loaded: {len(self.test_df_raw):,} rows")
        
#         # Filter valid targets
#         valid_mask = self.test_df_raw[self.target_col].notna()
#         test_df_valid = self.test_df_raw[valid_mask].copy()
#         print(f"   ✅ Valid targets: {len(test_df_valid):,} rows")
        
#         # Store sector info (for optional sector analysis)
#         if 'Sector' in test_df_valid.columns:
#             self.sectors = test_df_valid['Sector'].values
        
#         # Create crisis flag
#         test_df_valid = self._create_crisis_flag(test_df_valid)
        
#         # Store actuals and crisis flags
#         self.actuals = test_df_valid[self.target_col].values
#         self.crisis_flags = test_df_valid['crisis_flag'].values
        
#         # Prepare features
#         X_test_aligned = self._prepare_features(test_df_valid)
        
#         # Generate predictions
#         print(f"\n   🔮 Generating predictions...")
#         self.predictions = self.model.predict(X_test_aligned)
#         print(f"   ✅ Predictions generated: {len(self.predictions):,}")
        
#         # Show crisis distribution
#         crisis_count = self.crisis_flags.sum()
#         non_crisis_count = len(self.crisis_flags) - crisis_count
#         print(f"\n   📊 Test Set Distribution:")
#         print(f"      Crisis periods: {crisis_count} ({crisis_count/len(self.crisis_flags)*100:.1f}%)")
#         print(f"      Normal periods: {non_crisis_count} ({non_crisis_count/len(self.crisis_flags)*100:.1f}%)")
    
#     def _create_crisis_flag(self, df: pd.DataFrame) -> pd.DataFrame:
#         """
#         Create crisis_flag column
        
#         Crisis periods:
#         - 2007-2009: Financial Crisis
#         - 2020-2021: COVID Crisis
#         - VIX > 30: Market stress
#         """
#         print(f"\n   🚨 Creating crisis flag...")
        
#         df['Date'] = pd.to_datetime(df['Date'])
        
#         # Initialize crisis flag
#         df['crisis_flag'] = 0
        
#         # Method 1: Time-based crisis periods
#         financial_crisis = (df['Date'] >= '2007-01-01') & (df['Date'] <= '2009-12-31')
#         covid_crisis = (df['Date'] >= '2020-01-01') & (df['Date'] <= '2021-12-31')
        
#         df.loc[financial_crisis | covid_crisis, 'crisis_flag'] = 1
        
#         # Method 2: VIX-based stress (if available)
#         if 'vix_q_mean' in df.columns:
#             vix_stress = df['vix_q_mean'] > 30
#             df.loc[vix_stress, 'crisis_flag'] = 1
        
#         crisis_count = df['crisis_flag'].sum()
#         print(f"      ✅ Crisis periods marked: {crisis_count} rows ({crisis_count/len(df)*100:.1f}%)")
#         print(f"      Periods: 2007-2009 (Financial Crisis), 2020-2021 (COVID), VIX>30")
        
#         return df
    
#     def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
#         """
#         Prepare features EXACTLY as done during training
#         """
#         print(f"\n   🔧 Preparing features to match training pipeline...")
        
#         # Import training utilities
#         from utils.split_utils import get_feature_target_split
        
#         # Drop crisis_flag (we already extracted it)
#         df_for_features = df.drop(columns=['crisis_flag'], errors='ignore')
        
#         # Use SAME preprocessing as training
#         X_test, y_test = get_feature_target_split(
#             df_for_features,
#             target_col=self.target_col,
#             encode_categoricals=True
#         )
        
#         print(f"      ✅ Features prepared using training pipeline")
#         print(f"      Shape: {X_test.shape}")
        
#         # Align columns
#         missing_cols = set(self.feature_names) - set(X_test.columns)
#         extra_cols = set(X_test.columns) - set(self.feature_names)
        
#         if missing_cols:
#             print(f"      ⚠️  Adding {len(missing_cols)} missing columns with zeros")
#             for col in missing_cols:
#                 X_test[col] = 0
        
#         if extra_cols:
#             print(f"      ⚠️  Dropping {len(extra_cols)} extra columns")
#             X_test = X_test.drop(columns=list(extra_cols))
        
#         # Reorder to match model
#         X_test = X_test[self.feature_names]
        
#         # Simple NaN handling - fill with 0 (since we aligned columns already)
#         if X_test.isna().any().any():
#             print(f"      🔧 Filling remaining NaNs with 0...")
#             X_test = X_test.fillna(0)
        
#         print(f"      ✅ Final shape: {X_test.shape}")
        
#         return X_test
     
    
#     def calculate_metrics(self, y_true, y_pred, label=""):
#         """Calculate performance metrics"""
#         if len(y_true) < 5:
#             return {
#                 'rmse': np.nan, 'mae': np.nan, 'r2': np.nan,
#                 'mean_residual': np.nan, 'n_samples': len(y_true), 'label': label
#             }
        
#         rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
#         mae = float(mean_absolute_error(y_true, y_pred))
#         r2 = float(r2_score(y_true, y_pred))
        
#         # Mean residual (Actual - Predicted)
#         # Positive = underestimating (predicting lower than actual)
#         residual = y_true - y_pred
#         mean_residual = float(np.mean(residual))
        
#         return {
#             'rmse': rmse,
#             'mae': mae,
#             'r2': r2,
#             'mean_residual': mean_residual,
#             'n_samples': len(y_true),
#             'label': label
#         }
    
#     def analyze_crisis_performance(self):
#         """Analyze performance during crisis vs non-crisis"""
#         print(f"\n{'='*80}")
#         print(f"🚨 ANALYZING CRISIS vs NON-CRISIS PERFORMANCE")
#         print(f"{'='*80}")
        
#         # Split by crisis flag
#         crisis_mask = self.crisis_flags == 1
#         non_crisis_mask = self.crisis_flags == 0
        
#         # Calculate metrics
#         overall_metrics = self.calculate_metrics(self.actuals, self.predictions, "Overall")
#         crisis_metrics = self.calculate_metrics(
#             self.actuals[crisis_mask], 
#             self.predictions[crisis_mask], 
#             "Crisis"
#         )
#         non_crisis_metrics = self.calculate_metrics(
#             self.actuals[non_crisis_mask], 
#             self.predictions[non_crisis_mask], 
#             "Non-Crisis"
#         )
        
#         self.metrics['overall'] = overall_metrics
#         self.metrics['crisis'] = crisis_metrics
#         self.metrics['non_crisis'] = non_crisis_metrics
        
#         # Print results
#         print(f"\n   📊 OVERALL Performance:")
#         print(f"      Samples: {overall_metrics['n_samples']}")
#         print(f"      R²: {overall_metrics['r2']:.4f}")
#         print(f"      RMSE: {overall_metrics['rmse']:,.2f}")
#         print(f"      MAE: {overall_metrics['mae']:,.2f}")
#         print(f"      Mean Residual: {overall_metrics['mean_residual']:,.2f}")
        
#         print(f"\n   {'='*60}")
#         print(f"   🔴 CRISIS PERIODS (2007-2009, 2020-2021, VIX>30):")
#         print(f"      Samples: {crisis_metrics['n_samples']}")
#         if crisis_metrics['n_samples'] > 0:
#             print(f"      R²: {crisis_metrics['r2']:.4f}")
#             print(f"      RMSE: {crisis_metrics['rmse']:,.2f}")
#             print(f"      MAE: {crisis_metrics['mae']:,.2f}")
#             print(f"      Mean Residual: {crisis_metrics['mean_residual']:,.2f}")
#         else:
#             print(f"      R²: nan")
#             print(f"      RMSE: nan")
#             print(f"      MAE: nan")
#             print(f"      Mean Residual: nan")
        
#         print(f"\n   ✅ NON-CRISIS PERIODS (Normal market conditions):")
#         print(f"      Samples: {non_crisis_metrics['n_samples']}")
#         print(f"      R²: {non_crisis_metrics['r2']:.4f}")
#         print(f"      RMSE: {non_crisis_metrics['rmse']:,.2f}")
#         print(f"      MAE: {non_crisis_metrics['mae']:,.2f}")
#         print(f"      Mean Residual: {non_crisis_metrics['mean_residual']:,.2f}")
    
#     def detect_crisis_bias(self):
#         """Detect if model has crisis-specific bias"""
#         print(f"\n{'='*80}")
#         print(f"🔍 DETECTING CRISIS BIAS")
#         print(f"{'='*80}")
        
#         crisis = self.metrics['crisis']
#         non_crisis = self.metrics['non_crisis']
        
#         # ========================================
#         # EARLY EXIT: Not enough crisis samples
#         # ========================================
#         if crisis['n_samples'] < 5 or np.isnan(crisis['rmse']):
#             print("\n   ⚠️  NOT ENOUGH CRISIS SAMPLES TO ASSESS BIAS")
#             print(f"      Crisis samples: {crisis['n_samples']}")
#             print(f"      Non-crisis samples: {non_crisis['n_samples']}")
#             print("\n   ℹ️  Explanation:")
#             print("      The test set does not contain sufficient crisis periods")
#             print("      (2007-2009, 2020-2021, or VIX>30) to evaluate crisis bias.")
#             print("\n   📊 We can only report overall + non-crisis performance.")
#             print("      For crisis bias evaluation, consider:")
#             print("      - Using validation set (which may contain crisis periods)")
#             print("      - Temporal cross-validation across crisis/non-crisis folds")
#             print("      - Historical backtesting on 2008 or 2020 data")
            
#             self.bias_report = {
#                 'has_crisis_bias': None,  # Not evaluable
#                 'severity': 'NOT_EVALUATED',
#                 'bias_flags': [
#                     f"Insufficient crisis samples in test set ({crisis['n_samples']} samples)",
#                     "Cannot reliably assess crisis-specific bias"
#                 ],
#                 'rmse_ratio': None,
#                 'r2_degradation': None,
#                 'crisis_samples': crisis['n_samples'],
#                 'non_crisis_samples': non_crisis['n_samples']
#             }
#             return
        
#         # ========================================
#         # PROCEED WITH BIAS DETECTION
#         # ========================================
        
#         # Calculate ratios
#         rmse_ratio = crisis['rmse'] / non_crisis['rmse'] if non_crisis['rmse'] > 0 else 0
#         r2_diff = non_crisis['r2'] - crisis['r2']
        
#         # Bias flags
#         bias_flags = []
        
#         print(f"\n   🎯 Bias Detection Criteria:")
#         print(f"      1. RMSE Ratio (Crisis/Normal): {rmse_ratio:.2f}x")
#         print(f"      2. R² Degradation: {r2_diff:+.3f}")
#         print(f"      3. Crisis Mean Residual: {crisis['mean_residual']:,.2f}")
#         print(f"      4. Normal Mean Residual: {non_crisis['mean_residual']:,.2f}")
        
#         print(f"\n   {'='*60}")
        
#         # Check 1: Higher error in crisis
#         if rmse_ratio > 1.5:
#             bias_flags.append(f"🔴 CRITICAL: RMSE is {rmse_ratio:.2f}x higher during crises")
#             print(f"   ❌ CRISIS BIAS: Error rate {rmse_ratio:.2f}x higher in crisis periods!")
#             print(f"      → Model FAILS when it matters most!")
#         elif rmse_ratio > 1.2:
#             bias_flags.append(f"⚠️  MODERATE: RMSE is {rmse_ratio:.2f}x higher during crises")
#             print(f"   ⚠️  WARNING: Errors moderately higher in crises ({rmse_ratio:.2f}x)")
#             print(f"      → Model degrades during stress")
#         else:
#             print(f"   ✅ ERROR STABILITY: Similar error rates in crisis vs normal ({rmse_ratio:.2f}x)")
        
#         # Check 2: Performance degradation
#         if r2_diff > 0.20:
#             bias_flags.append(f"🔴 CRITICAL: R² drops by {r2_diff:.3f} during crises")
#             print(f"   ❌ PERFORMANCE DROP: Model much worse in crises (R² drops {r2_diff:.3f})")
#             print(f"      → Predictive power collapses under stress!")
#         elif r2_diff > 0.10:
#             bias_flags.append(f"⚠️  MODERATE: R² drops by {r2_diff:.3f} during crises")
#             print(f"   ⚠️  WARNING: Performance degrades in crises (R² drops {r2_diff:.3f})")
#         else:
#             print(f"   ✅ STABLE PERFORMANCE: R² similar across crisis/normal periods")
        
#         # Check 3: Optimistic bias in crisis (MOST CRITICAL!)
#         mean_target = np.mean(self.actuals)
#         if mean_target != 0:
#             crisis_bias_pct = (crisis['mean_residual'] / abs(mean_target)) * 100
#             non_crisis_bias_pct = (non_crisis['mean_residual'] / abs(mean_target)) * 100
            
#             if self.target_name in ['revenue', 'eps', 'profit_margin']:
#                 # For positive targets: positive residual = underestimating drops
#                 if crisis['mean_residual'] > 0 and abs(crisis_bias_pct) > 10:
#                     bias_flags.append(
#                         f"🔴 DANGEROUS: Underestimating {self.target_name} drops by {abs(crisis_bias_pct):.1f}% in crises"
#                     )
#                     print(f"\n   ❌ OPTIMISTIC BIAS IN CRISIS:")
#                     print(f"      Model underestimates crisis drops by {abs(crisis_bias_pct):.1f}%")
#                     print(f"      → TOO OPTIMISTIC during stress - DANGEROUS for risk management!")
#                     print(f"      → Bank/regulator would underestimate losses")
#                 elif crisis['mean_residual'] < 0 and abs(crisis_bias_pct) > 10:
#                     print(f"\n   ⚠️  PESSIMISTIC BIAS IN CRISIS:")
#                     print(f"      Model overestimates drops by {abs(crisis_bias_pct):.1f}%")
#                     print(f"      → Conservative (safer than optimistic bias)")
#                 else:
#                     print(f"\n   ✅ UNBIASED IN CRISIS: Predictions well-calibrated")
#                     print(f"      Crisis bias: {crisis_bias_pct:+.1f}%")
#                     print(f"      Normal bias: {non_crisis_bias_pct:+.1f}%")
            
#             elif self.target_name == 'stock_return':
#                 # For returns: negative residual = underestimating losses
#                 if crisis['mean_residual'] < 0 and abs(crisis_bias_pct) > 10:
#                     bias_flags.append(
#                         f"🔴 DANGEROUS: Underestimating stock losses by {abs(crisis_bias_pct):.1f}% in crises"
#                     )
#                     print(f"\n   ❌ OPTIMISTIC BIAS IN CRISIS:")
#                     print(f"      Model underestimates stock losses by {abs(crisis_bias_pct):.1f}%")
#                     print(f"      → TOO OPTIMISTIC about portfolio resilience!")
        
#         # Summary
#         print(f"\n   {'='*60}")
#         print(f"   VERDICT:")
#         if len(bias_flags) > 0:
#             severity = 'CRITICAL' if any('CRITICAL' in f for f in bias_flags) else 'MODERATE'
#             print(f"   ⚠️  CRISIS BIAS DETECTED - Severity: {severity}")
#             print(f"   Found {len(bias_flags)} issue(s):")
#             for flag in bias_flags:
#                 print(f"      • {flag}")
            
#             print(f"\n   🔧 RECOMMENDED ACTIONS:")
#             print(f"      1. Re-weight crisis periods in training (higher sample weights)")
#             print(f"      2. Add crisis-specific features (VIX interactions, stress indicators)")
#             print(f"      3. Apply post-hoc calibration for crisis regime")
#             print(f"      4. Consider ensemble with crisis-specialist sub-model")
            
#             self.bias_report['has_crisis_bias'] = True
#             self.bias_report['severity'] = severity
#         else:
#             print(f"   ✅ NO SIGNIFICANT CRISIS BIAS DETECTED")
#             print(f"      Model performs consistently across crisis/normal regimes")
#             self.bias_report['has_crisis_bias'] = False
#             self.bias_report['severity'] = 'NONE'
        
#         self.bias_report['bias_flags'] = bias_flags
#         self.bias_report['rmse_ratio'] = rmse_ratio
#         self.bias_report['r2_degradation'] = r2_diff
#         self.bias_report['crisis_samples'] = crisis['n_samples']
#         self.bias_report['non_crisis_samples'] = non_crisis['n_samples']
    
#     def create_visualization(self, output_dir: str):
#         """Create visualizations"""
#         print(f"\n{'='*80}")
#         print(f"📊 CREATING VISUALIZATIONS")
#         print(f"{'='*80}")
        
#         output_path = Path(output_dir)
#         output_path.mkdir(parents=True, exist_ok=True)
        
#         fig, axes = plt.subplots(2, 2, figsize=(16, 12))
#         fig.suptitle(f'Crisis Bias Analysis - {self.target_name.upper()} ({self.model_type.upper()})',
#                      fontsize=16, fontweight='bold')
        
#         # Plot 1: R² Comparison
#         ax1 = axes[0, 0]
#         periods = ['Overall', 'Non-Crisis', 'Crisis']
#         r2_values = [
#             self.metrics['overall']['r2'],
#             self.metrics['non_crisis']['r2'],
#             self.metrics['crisis']['r2'] if self.metrics['crisis']['n_samples'] > 0 else 0
#         ]
#         colors = ['gray', 'green', 'red']
#         bars1 = ax1.bar(periods, r2_values, color=colors)
#         ax1.set_ylabel('R² Score', fontweight='bold')
#         ax1.set_title('Model Performance: Crisis vs Non-Crisis', fontweight='bold')
#         ax1.grid(True, alpha=0.3, axis='y')
        
#         # Add value labels
#         for bar, val in zip(bars1, r2_values):
#             if not np.isnan(val):
#                 height = bar.get_height()
#                 ax1.text(bar.get_x() + bar.get_width()/2., height,
#                         f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
        
#         # Plot 2: RMSE Comparison
#         ax2 = axes[0, 1]
#         rmse_values = [
#             self.metrics['overall']['rmse'],
#             self.metrics['non_crisis']['rmse'],
#             self.metrics['crisis']['rmse'] if self.metrics['crisis']['n_samples'] > 0 else 0
#         ]
#         bars2 = ax2.bar(periods, rmse_values, color=colors)
#         ax2.set_ylabel('RMSE', fontweight='bold')
#         ax2.set_title('Error Magnitude: Crisis vs Non-Crisis', fontweight='bold')
#         ax2.grid(True, alpha=0.3, axis='y')
        
#         # Add value labels
#         for bar, val in zip(bars2, rmse_values):
#             if not np.isnan(val) and val > 0:
#                 height = bar.get_height()
#                 ax2.text(bar.get_x() + bar.get_width()/2., height,
#                         f'{val:,.0f}', ha='center', va='bottom', fontsize=9)
        
#         # Plot 3: Mean Residual (Bias Direction)
#         ax3 = axes[1, 0]
#         residuals = [
#             self.metrics['overall']['mean_residual'],
#             self.metrics['non_crisis']['mean_residual'],
#             self.metrics['crisis']['mean_residual'] if self.metrics['crisis']['n_samples'] > 0 else 0
#         ]
#         colors_bias = ['gray', 'green', 'red' if residuals[2] > 0 else 'orange']
#         bars3 = ax3.bar(periods, residuals, color=colors_bias)
#         ax3.axhline(y=0, color='blue', linestyle='--', linewidth=2, label='No Bias')
#         ax3.set_ylabel('Mean Residual (Actual - Predicted)', fontweight='bold')
#         ax3.set_title('Prediction Bias Direction', fontweight='bold')
#         ax3.legend()
#         ax3.grid(True, alpha=0.3, axis='y')
        
#         # Add annotation
#         ax3.text(0.02, 0.98, 'Positive = Underestimating drops\n(Optimistic bias - DANGEROUS for crises)\nNegative = Overestimating drops\n(Conservative - safer)',
#                 transform=ax3.transAxes, fontsize=8, verticalalignment='top',
#                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
#         # Plot 4: Sample Distribution
#         ax4 = axes[1, 1]
#         sample_counts = [
#             self.metrics['non_crisis']['n_samples'],
#             self.metrics['crisis']['n_samples']
#         ]
#         bars4 = ax4.bar(['Non-Crisis', 'Crisis'], sample_counts, color=['green', 'red'])
#         ax4.set_ylabel('Number of Samples', fontweight='bold')
#         ax4.set_title('Test Set Distribution', fontweight='bold')
#         ax4.grid(True, alpha=0.3, axis='y')
        
#         for bar, count in zip(bars4, sample_counts):
#             height = bar.get_height()
#             ax4.text(bar.get_x() + bar.get_width()/2., height,
#                     str(count), ha='center', va='bottom', fontsize=12, fontweight='bold')
        
#         plt.tight_layout()
        
#         viz_file = output_path / f'crisis_bias_{self.model_type}_{self.target_name}.png'
#         plt.savefig(viz_file, dpi=300, bbox_inches='tight')
#         print(f"   ✅ Visualization: {viz_file}")
#         plt.close()
    
#     def generate_reports(self, output_dir: str):
#         """Generate reports"""
#         print(f"\n{'='*80}")
#         print(f"📝 GENERATING REPORTS")
#         print(f"{'='*80}")
        
#         output_path = Path(output_dir)
#         output_path.mkdir(parents=True, exist_ok=True)
        
#         # JSON report
#         report_data = {
#             'model_type': self.model_type,
#             'target': self.target_name,
#             'metrics': self.metrics,
#             'bias_report': self.bias_report,
#             'timestamp': pd.Timestamp.now().isoformat()
#         }
        
#         json_file = output_path / f'crisis_bias_report_{self.model_type}_{self.target_name}.json'
#         with open(json_file, 'w') as f:
#             json.dump(report_data, f, indent=2)
#         print(f"   ✅ JSON: {json_file}")
        
#         # CSV report
#         csv_data = []
#         for period, metrics in self.metrics.items():
#             csv_data.append({
#                 'Period': period,
#                 'N': metrics['n_samples'],
#                 'R2': metrics['r2'],
#                 'RMSE': metrics['rmse'],
#                 'MAE': metrics['mae'],
#                 'Mean_Residual': metrics['mean_residual']
#             })
        
#         df_report = pd.DataFrame(csv_data)
#         csv_file = output_path / f'crisis_metrics_{self.model_type}_{self.target_name}.csv'
#         df_report.to_csv(csv_file, index=False)
#         print(f"   ✅ CSV: {csv_file}")


# def main():
#     parser = argparse.ArgumentParser(
#         description='Detect crisis-based model bias',
#         formatter_class=argparse.RawDescriptionHelpFormatter,
#         epilog="""
# Examples:
#   python detect_crisis_bias.py --model xgboost --target revenue
#   python detect_crisis_bias.py --model xgboost --target all
#   python detect_crisis_bias.py --model xgboost --target revenue --test-data data/splits/val_data.csv
#         """
#     )
#     parser.add_argument('--model', type=str, required=True, 
#                        choices=['xgboost', 'lstm', 'lightgbm'],
#                        help='Model type to analyze')
#     parser.add_argument('--target', type=str, required=True,
#                        choices=['revenue', 'eps', 'debt_equity', 'profit_margin', 'stock_return', 'all'],
#                        help='Target to analyze (or "all" for all targets)')
#     parser.add_argument('--model-dir', type=str, default='models',
#                        help='Directory containing trained models')
#     parser.add_argument('--test-data', type=str, default='data/splits/test_data.csv',
#                        help='Path to test data (use val_data.csv for validation set)')
#     parser.add_argument('--output', type=str, default='reports/crisis_bias',
#                        help='Output directory for reports')
    
#     args = parser.parse_args()
    
#     targets = ['revenue', 'eps', 'debt_equity', 'profit_margin', 'stock_return'] if args.target == 'all' else [args.target]
    
#     print(f"\n{'='*80}")
#     print(f"🚨 CRISIS BIAS DETECTION - {args.model.upper()}")
#     print(f"{'='*80}")
#     print(f"   Targets: {', '.join(targets)}")
#     print(f"   Data: {args.test_data}")
#     print(f"{'='*80}\n")
    
#     results_summary = []
    
#     for target in targets:
#         print(f"\n{'='*80}")
#         print(f"🚨 ANALYZING: {args.model.upper()} - {target.upper()}")
#         print(f"{'='*80}")
        
#         try:
#             detector = CrisisBiasDetector(args.model, target)
            
#             # Load model
#             model_file = Path(args.model_dir) / args.model / f'{args.model}_{target}.pkl'
#             detector.load_model(model_file)
            
#             # Prepare data and predict
#             detector.load_and_prepare_test_data(args.test_data)
            
#             # Analyze
#             detector.analyze_crisis_performance()
#             detector.detect_crisis_bias()
#             detector.create_visualization(args.output)
#             detector.generate_reports(args.output)
            
#             print(f"\n✅ {target} complete!")
            
#             # Store results
#             results_summary.append({
#                 'target': target,
#                 'overall_r2': detector.metrics['overall']['r2'],
#                 'crisis_samples': detector.metrics['crisis']['n_samples'],
#                 'has_bias': detector.bias_report.get('has_crisis_bias'),
#                 'severity': detector.bias_report.get('severity')
#             })
            
#         except Exception as e:
#             print(f"\n❌ ERROR: {e}")
#             import traceback
#             traceback.print_exc()
            
#             results_summary.append({
#                 'target': target,
#                 'overall_r2': None,
#                 'crisis_samples': None,
#                 'has_bias': None,
#                 'severity': 'ERROR'
#             })
    
#     # Final summary
#     if len(results_summary) > 1:
#         print(f"\n\n{'='*80}")
#         print(f"📊 CRISIS BIAS DETECTION SUMMARY")
#         print(f"{'='*80}\n")
        
#         print(f"{'Target':<20} {'Overall R²':>12} {'Crisis N':>10} {'Bias?':>10} {'Severity':>15}")
#         print(f"-" * 70)
        
#         for result in results_summary:
#             r2_str = f"{result['overall_r2']:.4f}" if result['overall_r2'] is not None else "N/A"
#             crisis_str = str(result['crisis_samples']) if result['crisis_samples'] is not None else "N/A"
#             bias_str = str(result['has_bias']) if result['has_bias'] is not None else "N/A"
            
#             print(f"{result['target']:<20} {r2_str:>12} {crisis_str:>10} {bias_str:>10} {result['severity']:>15}")
        
#         print(f"\n{'='*80}")
#         print(f"✅ Analysis complete for {len(results_summary)} target(s)")
#         print(f"   Reports saved to: {args.output}/")


# if __name__ == "__main__":
#     main()



"""
src/evaluation/detect_crisis_bias.py

Crisis-Based Model Bias Detection - Multi-Model Support

Supports: XGBoost, LSTM, Ridge, LightGBM

Usage:
    python src/evaluation/detect_crisis_bias.py --model xgboost --target revenue
    python src/evaluation/detect_crisis_bias.py --model lstm --target revenue
    python src/evaluation/detect_crisis_bias.py --model ridge --target all
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import argparse
import warnings
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import sys

warnings.filterwarnings('ignore')

# Setup paths
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root / "src"))


class CrisisBiasDetector:
    """
    Detects model bias during crisis vs non-crisis periods
    Supports: XGBoost, LSTM, Ridge, LightGBM
    """
    
    def __init__(self, model_type: str, target_name: str):
        self.model_type = model_type
        self.target_name = target_name
        self.target_col = f'target_{target_name}'
        self.model = None
        self.feature_names = None
        self.test_df_raw = None
        self.predictions = None
        self.actuals = None
        self.crisis_flags = None
        self.sectors = None
        self.metrics = {}
        self.bias_report = {}
        
        # LSTM-specific
        self.lstm_model = None
        self.scaler_X = None
        self.scaler_y = None
        self.sequence_length = None
        
    def load_model(self, model_path: str):
        """Load trained model (supports XGBoost, LSTM, Ridge, LightGBM)"""
        print(f"\n{'='*80}")
        print(f"📥 LOADING MODEL: {self.model_type.upper()}")
        print(f"{'='*80}")
        
        model_file = Path(model_path)
        if not model_file.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # ========================================
        # XGBOOST / RIDGE / LIGHTGBM (.pkl files)
        # ========================================
        if self.model_type in ['xgboost', 'ridge', 'lightgbm']:
            model_data = joblib.load(model_file)
            self.model = model_data['model']
            self.feature_names = model_data['feature_names']
            
            print(f"   ✅ Loaded {self.model_type.upper()} model")
            print(f"   Features: {len(self.feature_names)}")
        
        # ========================================
        # LSTM (.pth files)
        # ========================================
        elif self.model_type == 'lstm':
            import torch
            import torch.nn as nn
            
            # Load checkpoint
            checkpoint = torch.load(model_file, map_location='cpu')
            
            # Rebuild LSTM model architecture
            sys.path.insert(0, str(project_root / "src" / "models"))
            from train_lstm_model import LSTMRegressor
            
            self.feature_names = checkpoint['feature_names']
            input_size = len(self.feature_names)
            
            # Rebuild model
            self.lstm_model = LSTMRegressor(
                input_size=input_size,
                hidden_size=checkpoint['hidden_size'],
                num_layers=checkpoint['num_layers'],
                dropout=0.2
            )
            
            # Load weights
            self.lstm_model.load_state_dict(checkpoint['model_state_dict'])
            self.lstm_model.eval()
            
            # Store scalers and config
            self.scaler_X = checkpoint['scaler_X']
            self.scaler_y = checkpoint['scaler_y']
            self.sequence_length = checkpoint['sequence_length']
            
            print(f"   ✅ Loaded LSTM model")
            print(f"   Features: {len(self.feature_names)}")
            print(f"   Sequence Length: {self.sequence_length}")
            print(f"   Hidden Size: {checkpoint['hidden_size']}")
        
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def load_and_prepare_test_data(self, test_data_path: str):
        """Load test data and create crisis flag"""
        print(f"\n{'='*80}")
        print(f"📥 LOADING AND PREPARING TEST DATA")
        print(f"{'='*80}")
        
        # Load raw test data
        self.test_df_raw = pd.read_csv(test_data_path)
        print(f"   ✅ Loaded: {len(self.test_df_raw):,} rows")
        
        # Filter valid targets
        valid_mask = self.test_df_raw[self.target_col].notna()
        test_df_valid = self.test_df_raw[valid_mask].copy()
        print(f"   ✅ Valid targets: {len(test_df_valid):,} rows")
        
        # Store sector info
        if 'Sector' in test_df_valid.columns:
            self.sectors = test_df_valid['Sector'].values
        
        # Create crisis flag
        test_df_valid = self._create_crisis_flag(test_df_valid)
        
        # Store actuals and crisis flags
        self.actuals = test_df_valid[self.target_col].values
        self.crisis_flags = test_df_valid['crisis_flag'].values
        
        # Prepare features
        X_test_aligned = self._prepare_features(test_df_valid)
        
        # Generate predictions (model-specific)
        print(f"\n   🔮 Generating predictions...")
        self.predictions = self._generate_predictions(X_test_aligned)
        print(f"   ✅ Predictions generated: {len(self.predictions):,}")
        
        # Show crisis distribution
        crisis_count = self.crisis_flags.sum()
        non_crisis_count = len(self.crisis_flags) - crisis_count
        print(f"\n   📊 Test Set Distribution:")
        print(f"      Crisis periods: {crisis_count} ({crisis_count/len(self.crisis_flags)*100:.1f}%)")
        print(f"      Normal periods: {non_crisis_count} ({non_crisis_count/len(self.crisis_flags)*100:.1f}%)")
    
    def _create_crisis_flag(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create crisis_flag column"""
        print(f"\n   🚨 Creating crisis flag...")
        
        df['Date'] = pd.to_datetime(df['Date'])
        df['crisis_flag'] = 0
        
        # Time-based crisis periods
        financial_crisis = (df['Date'] >= '2007-01-01') & (df['Date'] <= '2009-12-31')
        covid_crisis = (df['Date'] >= '2020-01-01') & (df['Date'] <= '2021-12-31')
        df.loc[financial_crisis | covid_crisis, 'crisis_flag'] = 1
        
        # VIX-based stress
        if 'vix_q_mean' in df.columns:
            vix_stress = df['vix_q_mean'] > 30
            df.loc[vix_stress, 'crisis_flag'] = 1
        
        crisis_count = df['crisis_flag'].sum()
        print(f"      ✅ Crisis periods marked: {crisis_count} rows ({crisis_count/len(df)*100:.1f}%)")
        
        return df
    
    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features using training pipeline"""
        print(f"\n   🔧 Preparing features to match training pipeline...")
        
        from utils.split_utils import get_feature_target_split
        
        # Drop crisis_flag
        df_for_features = df.drop(columns=['crisis_flag'], errors='ignore')
        
        # Use training preprocessing
        X_test, y_test = get_feature_target_split(
            df_for_features,
            target_col=self.target_col,
            encode_categoricals=True
        )
        
        print(f"      ✅ Features prepared")
        print(f"      Shape: {X_test.shape}")
        
        # Align columns
        missing_cols = set(self.feature_names) - set(X_test.columns)
        extra_cols = set(X_test.columns) - set(self.feature_names)
        
        if missing_cols:
            for col in missing_cols:
                X_test[col] = 0
        
        if extra_cols:
            X_test = X_test.drop(columns=list(extra_cols))
        
        # Reorder
        X_test = X_test[self.feature_names]
        
        # Fill NaNs
        if X_test.isna().any().any():
            X_test = X_test.fillna(0)
        
        return X_test
    
    def _generate_predictions(self, X_test):
        """Generate predictions (model-specific)"""
        
        # ========================================
        # XGBOOST / RIDGE / LIGHTGBM
        # ========================================
        if self.model_type in ['xgboost', 'ridge', 'lightgbm']:
            return self.model.predict(X_test)
        
        # ========================================
        # LSTM (requires sequences)
        # ========================================
        elif self.model_type == 'lstm':
            import torch
            
            # Scale features
            X_scaled = self.scaler_X.transform(X_test)
            
            predictions = []
            with torch.no_grad():
                for i in range(self.sequence_length, len(X_scaled)):
                    # Create sequence
                    X_seq = X_scaled[i - self.sequence_length:i]
                    X_seq = torch.FloatTensor(X_seq).unsqueeze(0)
                    
                    # Predict
                    pred_scaled = self.lstm_model(X_seq).cpu().numpy()
                    
                    # Inverse scale
                    pred = self.scaler_y.inverse_transform(pred_scaled.reshape(-1, 1))[0, 0]
                    predictions.append(pred)
            
            # Trim actuals/flags to match predictions
            self.actuals = self.actuals[self.sequence_length:]
            self.crisis_flags = self.crisis_flags[self.sequence_length:]
            if self.sectors is not None:
                self.sectors = self.sectors[self.sequence_length:]
            
            return np.array(predictions)
    
    def calculate_metrics(self, y_true, y_pred, label=""):
        """Calculate performance metrics"""
        if len(y_true) < 5:
            return {
                'rmse': np.nan, 'mae': np.nan, 'r2': np.nan,
                'mean_residual': np.nan, 'n_samples': len(y_true), 'label': label
            }
        
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mae = float(mean_absolute_error(y_true, y_pred))
        r2 = float(r2_score(y_true, y_pred))
        residual = y_true - y_pred
        mean_residual = float(np.mean(residual))
        
        return {
            'rmse': rmse, 'mae': mae, 'r2': r2,
            'mean_residual': mean_residual,
            'n_samples': len(y_true), 'label': label
        }
    
    def analyze_crisis_performance(self):
        """Analyze performance during crisis vs non-crisis"""
        print(f"\n{'='*80}")
        print(f"🚨 ANALYZING CRISIS vs NON-CRISIS PERFORMANCE")
        print(f"{'='*80}")
        
        crisis_mask = self.crisis_flags == 1
        non_crisis_mask = self.crisis_flags == 0
        
        overall_metrics = self.calculate_metrics(self.actuals, self.predictions, "Overall")
        crisis_metrics = self.calculate_metrics(
            self.actuals[crisis_mask], 
            self.predictions[crisis_mask], 
            "Crisis"
        )
        non_crisis_metrics = self.calculate_metrics(
            self.actuals[non_crisis_mask], 
            self.predictions[non_crisis_mask], 
            "Non-Crisis"
        )
        
        self.metrics['overall'] = overall_metrics
        self.metrics['crisis'] = crisis_metrics
        self.metrics['non_crisis'] = non_crisis_metrics
        
        # Print results - FIXED
        print(f"\n   📊 OVERALL Performance:")
        print(f"      R²: {overall_metrics['r2']:.4f}")
        print(f"      RMSE: {overall_metrics['rmse']:,.2f}")
        print(f"      Samples: {overall_metrics['n_samples']}")
        
        print(f"\n   🔴 CRISIS PERIODS:")
        print(f"      Samples: {crisis_metrics['n_samples']}")
        if crisis_metrics['n_samples'] > 0:
            print(f"      R²: {crisis_metrics['r2']:.4f}")
            print(f"      RMSE: {crisis_metrics['rmse']:,.2f}")
        else:
            print(f"      R²: N/A")
            print(f"      RMSE: N/A")
        
        print(f"\n   ✅ NON-CRISIS PERIODS:")
        print(f"      R²: {non_crisis_metrics['r2']:.4f}")
        print(f"      RMSE: {non_crisis_metrics['rmse']:,.2f}")
        print(f"      Samples: {non_crisis_metrics['n_samples']}")
    
    def detect_crisis_bias(self):
        """Detect if model has crisis-specific bias"""
        print(f"\n{'='*80}")
        print(f"🔍 DETECTING CRISIS BIAS")
        print(f"{'='*80}")
        
        crisis = self.metrics['crisis']
        non_crisis = self.metrics['non_crisis']
        
        # Early exit if no crisis data
        if crisis['n_samples'] < 5 or np.isnan(crisis['rmse']):
            print("\n   ⚠️  NOT ENOUGH CRISIS SAMPLES")
            print(f"      Crisis: {crisis['n_samples']}, Non-crisis: {non_crisis['n_samples']}")
            
            self.bias_report = {
                'has_crisis_bias': None,
                'severity': 'NOT_EVALUATED',
                'bias_flags': [f"Insufficient crisis samples ({crisis['n_samples']})"],
                'rmse_ratio': None,
                'r2_degradation': None,
                'crisis_samples': crisis['n_samples'],
                'non_crisis_samples': non_crisis['n_samples']
            }
            return
        
        # Calculate metrics
        rmse_ratio = crisis['rmse'] / non_crisis['rmse'] if non_crisis['rmse'] > 0 else 0
        r2_diff = non_crisis['r2'] - crisis['r2']
        
        bias_flags = []
        
        print(f"\n   Metrics:")
        print(f"      RMSE Ratio (Crisis/Normal): {rmse_ratio:.2f}x")
        print(f"      R² Degradation: {r2_diff:+.3f}")
        
        # Check bias
        if rmse_ratio > 1.5:
            bias_flags.append(f"CRITICAL: RMSE {rmse_ratio:.2f}x higher in crisis")
            print(f"   ❌ CRISIS BIAS DETECTED: Error {rmse_ratio:.2f}x higher!")
        elif rmse_ratio > 1.2:
            bias_flags.append(f"MODERATE: RMSE {rmse_ratio:.2f}x higher in crisis")
            print(f"   ⚠️  Moderate degradation ({rmse_ratio:.2f}x)")
        else:
            print(f"   ✅ Stable performance")
        
        if r2_diff > 0.20:
            bias_flags.append(f"CRITICAL: R² drops {r2_diff:.3f} in crisis")
        
        # Summary
        severity = 'CRITICAL' if any('CRITICAL' in f for f in bias_flags) else 'MODERATE' if bias_flags else 'NONE'
        
        self.bias_report = {
            'has_crisis_bias': len(bias_flags) > 0,
            'severity': severity,
            'bias_flags': bias_flags,
            'rmse_ratio': rmse_ratio,
            'r2_degradation': r2_diff,
            'crisis_samples': crisis['n_samples'],
            'non_crisis_samples': non_crisis['n_samples']
        }
        
        print(f"\n   VERDICT: {severity}")
    
    def create_visualization(self, output_dir: str):
        """Create visualizations"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Crisis Bias - {self.target_name.upper()} ({self.model_type.upper()})',
                     fontsize=16, fontweight='bold')
        
        # R² comparison
        ax1 = axes[0, 0]
        periods = ['Overall', 'Non-Crisis', 'Crisis']
        r2_vals = [self.metrics['overall']['r2'], self.metrics['non_crisis']['r2'],
                   self.metrics['crisis']['r2'] if self.metrics['crisis']['n_samples'] > 0 else 0]
        ax1.bar(periods, r2_vals, color=['gray', 'green', 'red'])
        ax1.set_ylabel('R²', fontweight='bold')
        ax1.set_title('Performance Comparison')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # RMSE comparison
        ax2 = axes[0, 1]
        rmse_vals = [self.metrics['overall']['rmse'], self.metrics['non_crisis']['rmse'],
                     self.metrics['crisis']['rmse'] if self.metrics['crisis']['n_samples'] > 0 else 0]
        ax2.bar(periods, rmse_vals, color=['gray', 'green', 'red'])
        ax2.set_ylabel('RMSE', fontweight='bold')
        ax2.set_title('Error Magnitude')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Bias direction
        ax3 = axes[1, 0]
        resid_vals = [self.metrics['overall']['mean_residual'], 
                      self.metrics['non_crisis']['mean_residual'],
                      self.metrics['crisis']['mean_residual'] if self.metrics['crisis']['n_samples'] > 0 else 0]
        ax3.bar(periods, resid_vals, color=['gray', 'green', 'red'])
        ax3.axhline(y=0, color='blue', linestyle='--', linewidth=2)
        ax3.set_ylabel('Mean Residual', fontweight='bold')
        ax3.set_title('Bias Direction')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Sample distribution
        ax4 = axes[1, 1]
        counts = [self.metrics['non_crisis']['n_samples'], self.metrics['crisis']['n_samples']]
        ax4.bar(['Non-Crisis', 'Crisis'], counts, color=['green', 'red'])
        ax4.set_ylabel('Samples', fontweight='bold')
        ax4.set_title('Distribution')
        
        plt.tight_layout()
        
        viz_file = output_path / f'crisis_bias_{self.model_type}_{self.target_name}.png'
        plt.savefig(viz_file, dpi=300, bbox_inches='tight')
        print(f"\n   ✅ Saved: {viz_file}")
        plt.close()
    
    def generate_reports(self, output_dir: str):
        """Generate JSON and CSV reports"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # JSON
        report_data = {
            'model_type': self.model_type,
            'target': self.target_name,
            'metrics': self.metrics,
            'bias_report': self.bias_report,
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        json_file = output_path / f'crisis_bias_{self.model_type}_{self.target_name}.json'
        with open(json_file, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        # CSV
        csv_data = []
        for period, metrics in self.metrics.items():
            csv_data.append({
                'Period': period, 'N': metrics['n_samples'],
                'R2': metrics['r2'], 'RMSE': metrics['rmse'],
                'MAE': metrics['mae'], 'Mean_Residual': metrics['mean_residual']
            })
        
        csv_file = output_path / f'crisis_metrics_{self.model_type}_{self.target_name}.csv'
        pd.DataFrame(csv_data).to_csv(csv_file, index=False)
        
        print(f"   ✅ Reports saved")


def main():
    parser = argparse.ArgumentParser(description='Crisis bias detection - multi-model')
    parser.add_argument('--model', type=str, required=True, 
                       choices=['xgboost', 'lstm', 'lightgbm', 'ridge'])
    parser.add_argument('--target', type=str, required=True,
                       choices=['revenue', 'eps', 'debt_equity', 'profit_margin', 'stock_return', 'all'])
    parser.add_argument('--model-dir', type=str, default='models')
    parser.add_argument('--test-data', type=str, default='data/splits/test_data.csv')
    parser.add_argument('--output', type=str, default='reports/crisis_bias')
    
    args = parser.parse_args()
    
    targets = ['revenue', 'eps', 'debt_equity', 'profit_margin', 'stock_return'] if args.target == 'all' else [args.target]
    
    # Determine file extension
    file_ext = '.pth' if args.model == 'lstm' else '.pkl'
    
    for target in targets:
        print(f"\n{'='*80}")
        print(f"ANALYZING: {args.model.upper()} - {target.upper()}")
        print(f"{'='*80}")
        
        try:
            detector = CrisisBiasDetector(args.model, target)
            
            # Load model
            model_file = Path(args.model_dir) / args.model / f'{args.model}_{target}{file_ext}'
            detector.load_model(model_file)
            
            # Analyze
            detector.load_and_prepare_test_data(args.test_data)
            detector.analyze_crisis_performance()
            detector.detect_crisis_bias()
            detector.create_visualization(args.output)
            detector.generate_reports(args.output)
            
            print(f"\n✅ {target} complete!")
            
        except Exception as e:
            print(f"\n❌ ERROR: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()