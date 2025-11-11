# File: src/models/train_lstm_gru_top20.py

"""
Train LSTM+GRU with top 20 features - Modified Runs and Testing
"""

import sys
from pathlib import Path
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow import keras
from tensorflow.keras import layers
import mlflow
import mlflow.keras
from datetime import datetime

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

DATA_DIR = project_root / "data" / "processed"
MODEL_DIR = project_root / "models" / "lstm_gru_top20"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

EXPERIMENT_NAME = "modified_runs_and_testing"
RUN_NAME = f"LSTM_GRU_Top20_v1_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
MODEL_NAME = "financial_stress_lstm_gru"

print("="*70)
print("📂 LOADING DATA")
print("="*70)

X_train = np.load(DATA_DIR / 'X_train_top20.npy')
X_test = np.load(DATA_DIR / 'X_test_top20.npy')
y_train = np.load(DATA_DIR / 'y_train_top20.npy')
y_test = np.load(DATA_DIR / 'y_test_top20.npy')

with open(DATA_DIR / 'metadata_top20.pkl', 'rb') as f:
    metadata = pickle.load(f)

TARGET_COLS = metadata['original_target_cols']
n_features = metadata['n_features']

print(f"✅ Loaded")

print("\n⚖️  Scaling...")

scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train)

X_train_lstm = X_train_scaled.reshape(
    X_train_scaled.shape[0], 1, X_train_scaled.shape[1])
X_test_lstm = X_test_scaled.reshape(
    X_test_scaled.shape[0], 1, X_test_scaled.shape[1])

pickle.dump(scaler_X, open(MODEL_DIR / 'scaler_X.pkl', 'wb'))
pickle.dump(scaler_y, open(MODEL_DIR / 'scaler_y.pkl', 'wb'))

print("\n" + "="*70)
print("🚀 TRAINING LSTM+GRU - TOP 20 FEATURES")
print("="*70)

mlflow.set_experiment(EXPERIMENT_NAME)

lstm_gru_params = {
    'lstm_units': 64,
    'gru_units': 32,
    'dropout': 0.2,
    'dense_units': 64,
    'learning_rate': 0.001,
    'epochs': 10,
    'batch_size': 32
}

with mlflow.start_run(run_name=RUN_NAME) as run:

    run_id = run.info.run_id
    print(f"\n📝 MLflow Run ID: {run_id}")

    model_hybrid = keras.Sequential([
        layers.LSTM(lstm_gru_params['lstm_units'], input_shape=(
            1, n_features), return_sequences=True),
        layers.Dropout(lstm_gru_params['dropout']),
        layers.GRU(lstm_gru_params['gru_units'], return_sequences=False),
        layers.Dropout(lstm_gru_params['dropout']),
        layers.Dense(lstm_gru_params['dense_units'], activation='relu'),
        layers.Dense(len(TARGET_COLS))
    ])

    model_hybrid.compile(
        optimizer=keras.optimizers.Adam(lstm_gru_params['learning_rate']),
        loss='mse',
        metrics=['mae']
    )

    model_hybrid.summary()

    mlflow.log_params(lstm_gru_params)
    mlflow.log_param("n_features", n_features)
    mlflow.log_param("feature_selection", "top_20")

    history = model_hybrid.fit(
        X_train_lstm, y_train_scaled,
        validation_split=0.2,
        epochs=lstm_gru_params['epochs'],
        batch_size=lstm_gru_params['batch_size'],
        verbose=1
    )

    # Log history
    for epoch in range(len(history.history['loss'])):
        mlflow.log_metric(
            "train_loss", history.history['loss'][epoch], step=epoch)
        mlflow.log_metric(
            "val_loss", history.history['val_loss'][epoch], step=epoch)

    y_pred_scaled = model_hybrid.predict(X_test_lstm, verbose=0)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)

    print("\n📊 Results:")
    print(f"{'Metric':<20} {'RMSE':>12} {'MAE':>12} {'R²':>10}")
    print("-" * 60)

    all_metrics = {}
    for i, target in enumerate(TARGET_COLS):
        rmse = np.sqrt(mean_squared_error(y_test[:, i], y_pred[:, i]))
        mae = mean_absolute_error(y_test[:, i], y_pred[:, i])
        r2 = r2_score(y_test[:, i], y_pred[:, i])

        all_metrics[f'{target}_rmse'] = rmse
        all_metrics[f'{target}_mae'] = mae
        all_metrics[f'{target}_r2'] = r2

        print(f"{target:<20} {rmse:>12.2f} {mae:>12.2f} {r2:>10.3f}")

    avg_rmse = np.mean([all_metrics[f'{t}_rmse'] for t in TARGET_COLS])
    avg_r2 = np.mean([all_metrics[f'{t}_r2'] for t in TARGET_COLS])
    all_metrics['avg_rmse'] = avg_rmse
    all_metrics['avg_r2'] = avg_r2

    print("-" * 60)
    print(f"{'AVERAGE':<20} {avg_rmse:>12.2f} {' ':>12} {avg_r2:>10.3f}")

    mlflow.log_metrics(all_metrics)

    model_path = MODEL_DIR / 'lstm_gru_model_top20.h5'
    model_hybrid.save(model_path)
    mlflow.keras.log_model(model_hybrid, "model",
                           registered_model_name=MODEL_NAME)

    mlflow.set_tags({
        "model_type": "LSTM_GRU",
        "feature_selection": "top_20",
        "version": "v1",
        "status": "testing"
    })

    print(f"\n✅ Saved to {MODEL_DIR}/")
    print(f"✅ MLflow Run ID: {run_id}")

print(f"\n➡️  Next: python src/models/compare_models_top20.py")
