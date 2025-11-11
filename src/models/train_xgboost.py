# File: src/models/train_xgboost_top20.py

"""
Train XGBoost with top 20 features - Modified Runs and Testing
"""

import sys
from pathlib import Path
import numpy as np
import pickle
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow
import mlflow.sklearn
from datetime import datetime

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = project_root / "data" / "processed"
MODEL_DIR = project_root / "models" / "xgboost_top20"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# MLflow configuration
EXPERIMENT_NAME = "modified_runs_and_testing"
RUN_NAME = f"XGBoost_Top20_v1_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
MODEL_NAME = "financial_stress_xgboost"

# ============================================================
# LOAD DATA (TOP 20 FEATURES)
# ============================================================

print("="*70)
print("📂 LOADING TOP 20 FEATURES DATASET")
print("="*70)

X_train = np.load(DATA_DIR / 'X_train_top20.npy')
X_test = np.load(DATA_DIR / 'X_test_top20.npy')
y_train = np.load(DATA_DIR / 'y_train_top20.npy')
y_test = np.load(DATA_DIR / 'y_test_top20.npy')

with open(DATA_DIR / 'metadata_top20.pkl', 'rb') as f:
    metadata = pickle.load(f)

TARGET_COLS = metadata['original_target_cols']
FEATURE_COLS = metadata['feature_cols']
n_features = metadata['n_features']

print(f"✅ X_train: {X_train.shape}")
print(f"✅ y_train: {y_train.shape}")
print(f"✅ Features: {n_features}")
print(f"✅ Targets: {TARGET_COLS}")

print(f"\n📋 Top 20 Features:")
for i, feat in enumerate(FEATURE_COLS, 1):
    print(f"  {i:2d}. {feat}")

# ============================================================
# SCALE DATA
# ============================================================

print("\n⚖️  Scaling...")

scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train)

pickle.dump(scaler_X, open(MODEL_DIR / 'scaler_X.pkl', 'wb'))
pickle.dump(scaler_y, open(MODEL_DIR / 'scaler_y.pkl', 'wb'))

# ============================================================
# TRAIN XGBOOST WITH MLFLOW
# ============================================================

print("\n" + "="*70)
print("🚀 TRAINING XGBOOST - TOP 20 FEATURES")
print("="*70)

# Set experiment
mlflow.set_experiment(EXPERIMENT_NAME)

# Hyperparameters
xgb_params = {
    'n_estimators': 100,
    'max_depth': 6,
    'learning_rate': 0.1,
    'min_child_weight': 1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'verbosity': 0
}

with mlflow.start_run(run_name=RUN_NAME) as run:
    
    # Log run info
    run_id = run.info.run_id
    print(f"\n📝 MLflow Run ID: {run_id}")
    print(f"📝 Run Name: {RUN_NAME}")
    
    # Log parameters
    mlflow.log_params(xgb_params)
    mlflow.log_param("n_features", n_features)
    mlflow.log_param("n_targets", len(TARGET_COLS))
    mlflow.log_param("feature_selection", "top_20_xgboost_importance")
    mlflow.log_param("train_samples", len(X_train))
    mlflow.log_param("test_samples", len(X_test))
    
    # Train models
    xgb_models = []
    all_metrics = {}
    
    for i, target in enumerate(TARGET_COLS):
        print(f"\n📈 Training for {target}... ({i+1}/{len(TARGET_COLS)})")
        
        model = xgb.XGBRegressor(**xgb_params)
        model.fit(X_train_scaled, y_train_scaled[:, i])
        
        # Predict
        y_pred_scaled = model.predict(X_test_scaled)
        y_pred = scaler_y.inverse_transform(
            np.column_stack([y_pred_scaled if j == i else np.zeros(len(y_pred_scaled)) 
                            for j in range(len(TARGET_COLS))])
        )[:, i]
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test[:, i], y_pred))
        mae = mean_absolute_error(y_test[:, i], y_pred)
        r2 = r2_score(y_test[:, i], y_pred)
        mape = np.mean(np.abs((y_test[:, i] - y_pred) / (y_test[:, i] + 1e-10))) * 100
        
        print(f"  RMSE: {rmse:.2f} | MAE: {mae:.2f} | R²: {r2:.3f} | MAPE: {mape:.2f}%")
        
        all_metrics[f'{target}_rmse'] = rmse
        all_metrics[f'{target}_mae'] = mae
        all_metrics[f'{target}_r2'] = r2
        all_metrics[f'{target}_mape'] = mape
        
        xgb_models.append(model)
        
        # Save individual model
        model_path = MODEL_DIR / f'xgboost_{target}_top20.json'
        model.save_model(str(model_path))
        
        # Log model to MLflow
        mlflow.sklearn.log_model(
            model, 
            f"model_{target}",
            registered_model_name=f"{MODEL_NAME}_{target}"
        )
    
    # Calculate averages
    avg_rmse = np.mean([all_metrics[f'{t}_rmse'] for t in TARGET_COLS])
    avg_mae = np.mean([all_metrics[f'{t}_mae'] for t in TARGET_COLS])
    avg_r2 = np.mean([all_metrics[f'{t}_r2'] for t in TARGET_COLS])
    avg_mape = np.mean([all_metrics[f'{t}_mape'] for t in TARGET_COLS])
    
    all_metrics['avg_rmse'] = avg_rmse
    all_metrics['avg_mae'] = avg_mae
    all_metrics['avg_r2'] = avg_r2
    all_metrics['avg_mape'] = avg_mape
    
    print(f"\n{'='*70}")
    print(f"📊 OVERALL PERFORMANCE:")
    print(f"   Avg RMSE: {avg_rmse:.2f}")
    print(f"   Avg MAE:  {avg_mae:.2f}")
    print(f"   Avg R²:   {avg_r2:.3f}")
    print(f"   Avg MAPE: {avg_mape:.2f}%")
    print(f"{'='*70}")
    
    # Log all metrics
    mlflow.log_metrics(all_metrics)
    
    # Log feature list as artifact
    with open(MODEL_DIR / 'features_used.txt', 'w') as f:
        f.write("Top 20 Features Used:\n")
        for i, feat in enumerate(FEATURE_COLS, 1):
            f.write(f"{i}. {feat}\n")
    
    mlflow.log_artifact(str(MODEL_DIR / 'features_used.txt'))
    
    # Add tags
    mlflow.set_tags({
        "model_type": "XGBoost",
        "feature_selection": "top_20",
        "version": "v1",
        "status": "testing"
    })

print(f"\n✅ Models saved to {MODEL_DIR}/")
print(f"✅ MLflow Run ID: {run_id}")
print(f"\n🔍 View in MLflow UI:")
print(f"   mlflow ui")
print(f"   http://localhost:5000")
print(f"\n➡️  Next: python src/models/train_lstm_top20.py")