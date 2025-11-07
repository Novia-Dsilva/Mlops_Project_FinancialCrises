"""
Step 4: Train Model
Trains LSTM model with MLflow tracking
"""

import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
from datetime import datetime
import yaml

import mlflow
import mlflow.keras
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

from model_builder import build_lstm_model, get_callbacks, load_params


def load_training_data():
    """Load preprocessed training data"""
    
    print("📥 Loading training data...")
    
    X_train = pd.read_csv('data/features/X_train.csv')
    X_test = pd.read_csv('data/features/X_test.csv')
    y_train = pd.read_csv('data/features/y_train.csv').values.ravel()
    y_test = pd.read_csv('data/features/y_test.csv').values.ravel()
    
    print(f"  Training samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")
    print(f"  Features: {X_train.shape[1]}")
    
    return X_train, X_test, y_train, y_test


def scale_features(X_train, X_test):
    """Scale features using StandardScaler"""
    
    print("🔧 Scaling features...")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save scaler
    Path('models').mkdir(exist_ok=True)
    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    print("  ✓ Features scaled and scaler saved")
    
    return X_train_scaled, X_test_scaled, scaler


def reshape_for_lstm(X, timesteps=12):
    """Reshape data for LSTM input (samples, timesteps, features)"""
    
    n_samples = X.shape[0]
    n_features = X.shape[1] // timesteps
    
    return X.reshape(n_samples, timesteps, n_features)


def calculate_class_weights(y_train):
    """Calculate class weights for imbalanced data"""
    
    classes = np.unique(y_train)
    weights = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weight_dict = dict(zip(classes, weights))
    
    print(f"  Class weights: {class_weight_dict}")
    
    return class_weight_dict


def train_model():
    """Main training function"""
    
    print("\n" + "="*70)
    print("STEP 4: MODEL TRAINING")
    print("="*70)
    
    # Load parameters
    params = load_params()
    train_params = params['train']
    
    # Load data
    X_train, X_test, y_train, y_test = load_training_data()
    
    # Scale features
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    
    # Reshape for LSTM
    X_train_lstm = reshape_for_lstm(X_train_scaled, timesteps=12)
    X_test_lstm = reshape_for_lstm(X_test_scaled, timesteps=12)
    
    print(f"\n📊 LSTM input shape: {X_train_lstm.shape}")
    
    # Calculate class weights
    class_weights = calculate_class_weights(y_train) if train_params['use_class_weights'] else None
    
    # MLflow setup
    mlflow.set_tracking_uri('mlruns')
    mlflow.set_experiment(params['registry']['experiment_name'])
    
    # Start MLflow run
    with mlflow.start_run(run_name=f"lstm_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        
        # Log parameters
        mlflow.log_params({
            'model_type': train_params['model_type'],
            'lstm_units': str(train_params['lstm_units']),
            'dropout_rate': train_params['dropout_rate'],
            'learning_rate': train_params['learning_rate'],
            'batch_size': train_params['batch_size'],
            'epochs': train_params['epochs'],
            'validation_split': train_params['validation_split']
        })
        
        # Build model
        print("\n🏗️  Building model...")
        input_shape = (X_train_lstm.shape[1], X_train_lstm.shape[2])
        model = build_lstm_model(input_shape, train_params)
        model.summary()
        
        # Get callbacks
        callbacks = get_callbacks(train_params)
        
        # Train model
        print("\n🚀 Training model...")
        history = model.fit(
            X_train_lstm, y_train,
            epochs=train_params['epochs'],
            batch_size=train_params['batch_size'],
            validation_split=train_params['validation_split'],
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate on test set
        print("\n📊 Evaluating on test set...")
        test_loss, test_acc, test_precision, test_recall = model.evaluate(X_test_lstm, y_test, verbose=0)
        
        print(f"  Test Loss: {test_loss:.4f}")
        print(f"  Test Accuracy: {test_acc:.4f}")
        print(f"  Test Precision: {test_precision:.4f}")
        print(f"  Test Recall: {test_recall:.4f}")
        
        # Log metrics
        mlflow.log_metrics({
            'test_loss': test_loss,
            'test_accuracy': test_acc,
            'test_precision': test_precision,
            'test_recall': test_recall,
            'final_train_loss': history.history['loss'][-1],
            'final_val_loss': history.history['val_loss'][-1]
        })
        
        # Save model
        print("\n💾 Saving model...")
        model_path = 'models/lstm_baseline'
        Path(model_path).mkdir(parents=True, exist_ok=True)
        model.save(f'{model_path}/model.h5')
        
        # Save metadata
        metadata = {
            'model_type': 'LSTM',
            'input_shape': input_shape,
            'training_date': datetime.now().isoformat(),
            'parameters': train_params,
            'performance': {
                'test_accuracy': float(test_acc),
                'test_loss': float(test_loss)
            }
        }
        
        with open(f'{model_path}/metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Log model to MLflow
        mlflow.keras.log_model(
            model,
            "model",
            registered_model_name=params['registry']['model_name']
        )
        
        # Log artifacts
        mlflow.log_artifact('models/metrics/training_history.csv')
        
        print("\n" + "="*70)
        print("✅ MODEL TRAINING COMPLETE")
        print("="*70)
        print(f"Model saved to: {model_path}")
        print(f"MLflow run ID: {mlflow.active_run().info.run_id}")
        
        return model, history


if __name__ == "__main__":
    train_model()