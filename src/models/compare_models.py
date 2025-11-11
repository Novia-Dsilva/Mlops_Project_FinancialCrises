# File: src/models/compare_models_top20.py

"""
Compare models trained on top 20 features
"""

import mlflow
import pandas as pd
import numpy as np

print("="*70)
print("📊 MODEL COMPARISON - TOP 20 FEATURES")
print("="*70)

mlflow.set_experiment("modified_runs_and_testing")

# Get all runs from experiment
runs = mlflow.search_runs(
    experiment_names=["modified_runs_and_testing"],
    order_by=["start_time DESC"]
)

if len(runs) == 0:
    print("❌ No runs found!")
    exit(1)

# Extract results
results = []
for _, run in runs.iterrows():
    run_name = run.get('tags.mlflow.runName', 'Unknown')

    # Only include Top20 models
    if 'Top20' in run_name:
        results.append({
            'Model': run_name.split('_')[0],
            'Version': run_name.split('_')[1] if len(run_name.split('_')) > 1 else 'v1',
            'Run ID': run['run_id'][:8],
            'Avg RMSE': run.get('metrics.avg_rmse', np.nan),
            'Avg R²': run.get('metrics.avg_r2', np.nan),
            'Revenue R²': run.get('metrics.Revenue_r2', np.nan),
            'Debt/Equity R²': run.get('metrics.Debt_to_Equity_r2', np.nan),
            'Features': run.get('params.n_features', 'N/A'),
            'Status': run.get('tags.status', 'N/A')
        })

df_results = pd.DataFrame(results)

# Sort by Avg R² (descending)
df_results = df_results.sort_values('Avg R²', ascending=False)

print("\n")
print(df_results.to_string(index=False))
print("\n" + "="*70)

if len(df_results) > 0:
    best = df_results.iloc[0]
    print(f"\n🏆 BEST MODEL: {best['Model']} ({best['Version']})")
    print(f"   Avg RMSE: {best['Avg RMSE']:.2f}")
    print(f"   Avg R²: {best['Avg R²']:.3f}")
    print(f"   Run ID: {best['Run ID']}")
    print(f"   Features: {best['Features']}")

print("\n🔍 View detailed results:")
print("   mlflow ui")
print("   http://localhost:5000")
print("="*70)
