"""
Step 6: Hyperparameter Tuning
Uses Optuna to find optimal hyperparameters for LSTM model
"""

import pandas as pd
import numpy as np
import pickle
import yaml
import json
from pathlib import Path
from datetime import datetime

import optuna
from optuna.integration import TFKerasPruningCallback
import mlflow
import mlflow.keras
from tensorflow import keras
from sklearn.preprocessing import StandardScaler

from model_builder import build_lstm_model
from model_utils import reshape_for_lstm, calculate_class_weights


def load_params():
    """Load parameters from params.yaml"""
    with open('params.yaml', 'r') as f:
        return yaml.safe_load(f)


def load_data():
    """Load training data for hyperparameter tuning"""

    print("\n📥 Loading training data...")

    X_train = pd.read_csv('data/features/X_train.csv')
    X_test = pd.read_csv('data/features/X_test.csv')
    y_train = pd.read_csv('data/features/y_train.csv').values.ravel()
    y_test = pd.read_csv('data/features/y_test.csv').values.ravel()

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Reshape for LSTM
    timesteps = 12
    X_train_lstm = reshape_for_lstm(X_train_scaled, timesteps)
    X_test_lstm = reshape_for_lstm(X_test_scaled, timesteps)

    print(f"  ✓ Data loaded and preprocessed")
    print(f"  Train: {X_train_lstm.shape}")
    print(f"  Test: {X_test_lstm.shape}")

    return X_train_lstm, X_test_lstm, y_train, y_test


def create_model_for_trial(trial, input_shape):
    """
    Create model with hyperparameters suggested by Optuna trial

    Args:
        trial: Optuna trial object
        input_shape: Input shape for model

    Returns:
        keras.Model: Model with trial hyperparameters
    """

    # Suggest hyperparameters
    lstm_units_1 = trial.suggest_int('lstm_units_1', 64, 256, step=64)
    lstm_units_2 = trial.suggest_int('lstm_units_2', 32, 128, step=32)
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    dense_units = trial.suggest_categorical('dense_units', [16, 32, 64])

    # Build model with suggested parameters
    model = keras.Sequential([
        keras.layers.LSTM(
            lstm_units_1,
            return_sequences=True,
            input_shape=input_shape,
            name='lstm_1'
        ),
        keras.layers.Dropout(dropout_rate, name='dropout_1'),

        keras.layers.LSTM(
            lstm_units_2,
            return_sequences=False,
            name='lstm_2'
        ),
        keras.layers.Dropout(dropout_rate, name='dropout_2'),

        keras.layers.Dense(dense_units, activation='relu', name='dense_1'),
        keras.layers.Dropout(dropout_rate, name='dropout_3'),

        keras.layers.Dense(1, activation='sigmoid', name='output')
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.AUC(name='auc')]
    )

    # Store batch_size in trial for later use
    trial.set_user_attr('batch_size', batch_size)

    return model


def objective(trial, X_train, y_train, X_val, y_val, input_shape, class_weights):
    """
    Objective function for Optuna optimization

    Args:
        trial: Optuna trial
        X_train, y_train: Training data
        X_val, y_val: Validation data
        input_shape: Model input shape
        class_weights: Class weights for imbalanced data

    Returns:
        float: Validation AUC score (to maximize)
    """

    # Clear session to avoid memory issues
    keras.backend.clear_session()

    # Create model
    model = create_model_for_trial(trial, input_shape)

    # Get batch size from trial
    batch_size = trial.user_attrs['batch_size']

    # Callbacks for this trial
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=0
        ),
        TFKerasPruningCallback(trial, 'val_auc')  # Prune unpromising trials
    ]

    # Train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,  # Max epochs for tuning
        batch_size=batch_size,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=0  # Silent training
    )

    # Return best validation AUC
    best_auc = max(history.history['val_auc'])

    return best_auc


def tune_hyperparameters():
    """Main hyperparameter tuning function"""

    print("\n" + "="*70)
    print("STEP 6: HYPERPARAMETER TUNING WITH OPTUNA")
    print("="*70)

    # Load parameters
    params = load_params()
    tune_params = params['tune']

    # Check if tuning is enabled
    if not tune_params['enabled']:
        print("\n⚠ Hyperparameter tuning is DISABLED in params.yaml")
        print("  Set tune.enabled = true to enable")
        return None

    # Load data
    X_train, X_test, y_train, y_test = load_data()

    # Split train into train/val for tuning
    from sklearn.model_selection import train_test_split
    X_train_tune, X_val_tune, y_train_tune, y_val_tune = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )

    print(f"\n📊 Data split for tuning:")
    print(f"  Train: {X_train_tune.shape}")
    print(f"  Val: {X_val_tune.shape}")

    # Calculate class weights
    class_weights = calculate_class_weights(y_train_tune)

    # Get input shape
    input_shape = (X_train.shape[1], X_train.shape[2])

    # Create Optuna study
    print(f"\n🔬 Creating Optuna study...")
    print(f"  Optimization metric: {tune_params['optimization_metric']}")
    print(f"  Direction: {tune_params['direction']}")
    print(f"  Number of trials: {tune_params['n_trials']}")

    study = optuna.create_study(
        direction=tune_params['direction'],
        study_name='lstm-hyperparameter-tuning',
        pruner=optuna.pruners.MedianPruner(
            n_warmup_steps=5) if tune_params['pruning']['enabled'] else None
    )

    # Setup MLflow
    mlflow.set_experiment("hyperparameter-tuning")

    # Run optimization
    print(f"\n🚀 Starting optimization...")
    print(f"  This may take {tune_params['timeout']/3600:.1f} hours")
    print("="*70)

    def wrapped_objective(trial):
        """Wrapper to include MLflow logging"""

        with mlflow.start_run(nested=True, run_name=f"trial_{trial.number}"):
            # Log trial parameters
            mlflow.log_params(trial.params)

            # Run trial
            score = objective(
                trial,
                X_train_tune, y_train_tune,
                X_val_tune, y_val_tune,
                input_shape,
                class_weights
            )

            # Log result
            mlflow.log_metric('val_auc', score)

            return score

    study.optimize(
        wrapped_objective,
        n_trials=tune_params['n_trials'],
        timeout=tune_params.get('timeout'),
        show_progress_bar=True
    )

    print("\n" + "="*70)
    print("✅ HYPERPARAMETER TUNING COMPLETE")
    print("="*70)

    # Print results
    print(f"\n🏆 Best Trial Results:")
    print(f"  Trial number: {study.best_trial.number}")
    print(
        f"  Best {tune_params['optimization_metric']}: {study.best_value:.4f}")
    print(f"\n📋 Best Hyperparameters:")
    for param, value in study.best_params.items():
        print(f"  {param:20} : {value}")

    # Save best parameters
    best_params_path = Path('models/best_hyperparameters.json')
    with open(best_params_path, 'w') as f:
        json.dump({
            'best_params': study.best_params,
            'best_value': study.best_value,
            'best_trial': study.best_trial.number,
            'n_trials': len(study.trials),
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)

    print(f"\n💾 Best parameters saved to: {best_params_path}")

    # Update params.yaml suggestion
    print(f"\n📝 Suggested params.yaml update:")
    print(f"train:")
    print(
        f"  lstm_units: [{study.best_params['lstm_units_1']}, {study.best_params['lstm_units_2']}]")
    print(f"  dropout_rate: {study.best_params['dropout_rate']}")
    print(f"  learning_rate: {study.best_params['learning_rate']}")
    print(f"  batch_size: {study.best_params['batch_size']}")
    print(f"  dense_units: [{study.best_params.get('dense_units', 32)}]")

    # Optimization history
    print(f"\n📊 Optimization History:")
    print(f"  Total trials: {len(study.trials)}")
    print(
        f"  Completed: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")
    print(
        f"  Pruned: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")
    print(
        f"  Failed: {len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])}")

    # Plot optimization history
    try:
        import matplotlib.pyplot as plt

        # Optimization history plot
        fig = optuna.visualization.matplotlib.plot_optimization_history(study)
        plt.tight_layout()
        plt.savefig('models/metrics/optuna_optimization_history.png', dpi=300)
        plt.close()

        # Parameter importance plot
        fig = optuna.visualization.matplotlib.plot_param_importances(study)
        plt.tight_layout()
        plt.savefig('models/metrics/optuna_param_importance.png', dpi=300)
        plt.close()

        print(f"\n✓ Optimization plots saved to models/metrics/")

    except Exception as e:
        print(f"\n⚠ Could not generate plots: {e}")

    return study


def train_with_best_params(study):
    """
    Train final model with best hyperparameters found

    Args:
        study: Completed Optuna study
    """

    print("\n" + "="*70)
    print("TRAINING FINAL MODEL WITH BEST HYPERPARAMETERS")
    print("="*70)

    # Load full training data
    X_train, X_test, y_train, y_test = load_data()

    # Get best params
    best_params = study.best_params

    # Update params dict for model building
    params = load_params()
    train_params = params['train'].copy()
    train_params['lstm_units'] = [
        best_params['lstm_units_1'], best_params['lstm_units_2']]
    train_params['dropout_rate'] = best_params['dropout_rate']
    train_params['learning_rate'] = best_params['learning_rate']
    train_params['batch_size'] = best_params['batch_size']
    train_params['dense_units'] = [best_params.get('dense_units', 32)]

    # Build model
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_lstm_model(input_shape, train_params)

    # Calculate class weights
    class_weights = calculate_class_weights(y_train)

    # Train with best parameters
    print("\n🚀 Training with optimized hyperparameters...")

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            'models/lstm_optimized/model.h5',
            monitor='val_auc',
            save_best_only=True,
            verbose=1
        )
    ]

    history = model.fit(
        X_train, y_train,
        epochs=train_params['epochs'],
        batch_size=train_params['batch_size'],
        validation_split=0.2,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )

    # Evaluate
    test_results = model.evaluate(X_test, y_test, verbose=0)
    test_loss, test_acc, test_auc = test_results[0], test_results[1], test_results[2]

    print(f"\n📊 Optimized Model Performance:")
    print(f"  Test Accuracy: {test_acc:.4f}")
    print(f"  Test AUC: {test_auc:.4f}")

    # Save optimized model
    Path('models/lstm_optimized').mkdir(parents=True, exist_ok=True)
    model.save('models/lstm_optimized/model.h5')

    # Save metadata
    metadata = {
        'model_type': 'LSTM_Optimized',
        'optimization_method': 'Optuna',
        'n_trials': len(study.trials),
        'best_trial': study.best_trial.number,
        'best_hyperparameters': best_params,
        'training_date': datetime.now().isoformat(),
        'performance': {
            'test_accuracy': float(test_acc),
            'test_auc': float(test_auc),
            'test_loss': float(test_loss)
        }
    }

    with open('models/lstm_optimized/metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✅ Optimized model saved to models/lstm_optimized/")

    return model, history


def compare_with_baseline():
    """Compare optimized model with baseline"""

    print("\n" + "="*70)
    print("BASELINE VS OPTIMIZED COMPARISON")
    print("="*70)

    # Load metrics
    baseline_metrics = json.load(
        open('models/metrics/evaluation_metrics.json'))
    optimized_metrics = json.load(
        open('models/lstm_optimized/metadata.json'))['performance']

    print(f"\n{'Metric':<20} {'Baseline':>12} {'Optimized':>12} {'Improvement':>12}")
    print("-" * 70)

    for metric in ['test_accuracy', 'test_auc', 'test_loss']:
        baseline_val = baseline_metrics.get(
            metric, baseline_metrics.get(metric.replace('test_', ''), 0))
        optimized_val = optimized_metrics.get(metric, 0)

        improvement = optimized_val - baseline_val
        improvement_pct = (improvement / baseline_val *
                           100) if baseline_val != 0 else 0

        print(
            f"{metric:<20} {baseline_val:>12.4f} {optimized_val:>12.4f} {improvement_pct:>11.2f}%")

    print("-" * 70)


def main():
    """Main hyperparameter tuning pipeline"""

    print("\n" + "="*80)
    print("HYPERPARAMETER TUNING PIPELINE")
    print("="*80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load parameters
    params = load_params()
    tune_params = params['tune']

    # Check if enabled
    if not tune_params['enabled']:
        print("\n⚠ Tuning is DISABLED in params.yaml")
        print("  To enable:")
        print("  1. Edit params.yaml")
        print("  2. Set tune.enabled = true")
        print("  3. Run this script again")
        return

    # Load data
    X_train, X_test, y_train, y_test = load_data()

    # Split for tuning
    from sklearn.model_selection import train_test_split
    X_train_tune, X_val_tune, y_train_tune, y_val_tune = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )

    # Calculate class weights
    class_weights = calculate_class_weights(y_train_tune)

    # Get input shape
    input_shape = (X_train.shape[1], X_train.shape[2])

    # Create study
    study = optuna.create_study(
        direction=tune_params['direction'],
        study_name='lstm-hyperparameter-optimization',
        pruner=optuna.pruners.MedianPruner(
            n_warmup_steps=tune_params['pruning'].get('warmup_steps', 5)
        ) if tune_params['pruning']['enabled'] else None
    )

    # Optimize
    print(f"\n🔬 Running optimization...")
    print(f"  Trials: {tune_params['n_trials']}")
    print(f"  Timeout: {tune_params.get('timeout', 'None')}s")

    def wrapped_objective(trial):
        return objective(
            trial,
            X_train_tune, y_train_tune,
            X_val_tune, y_val_tune,
            input_shape,
            class_weights
        )

    study.optimize(
        wrapped_objective,
        n_trials=tune_params['n_trials'],
        timeout=tune_params.get('timeout'),
        show_progress_bar=True
    )

    # Print results
    print("\n" + "="*70)
    print("OPTIMIZATION RESULTS")
    print("="*70)
    print(f"\n🏆 Best Trial: #{study.best_trial.number}")
    print(
        f"   Best {tune_params['optimization_metric']}: {study.best_value:.4f}")
    print(f"\n📋 Best Hyperparameters:")
    for param, value in study.best_params.items():
        print(f"   {param:20} : {value}")

    # Save results
    results = {
        'best_params': study.best_params,
        'best_value': study.best_value,
        'best_trial': study.best_trial.number,
        'n_trials': len(study.trials),
        'completed_trials': len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
        'pruned_trials': len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
        'timestamp': datetime.now().isoformat()
    }

    Path('models').mkdir(exist_ok=True)
    with open('models/tuning_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n💾 Tuning results saved to models/tuning_results.json")

    # Train final model with best params
    print(f"\n{'='*70}")
    user_input = input("Train final model with best parameters? (y/n): ")

    if user_input.lower() == 'y':
        model, history = train_with_best_params(study)
        compare_with_baseline()

    print("\n" + "="*70)
    print("✅ HYPERPARAMETER TUNING COMPLETE")
    print("="*70)
    print(f"\n🎯 Next Steps:")
    print(f"  1. Review: models/tuning_results.json")
    print(f"  2. Update params.yaml with best hyperparameters")
    print(f"  3. Retrain: python src/models/step4_train_model.py")
    print(f"  4. Evaluate: python src/models/step5_evaluate_model.py")

    return study


if __name__ == "__main__":
    import sys

    try:
        study = tune_hyperparameters()
        sys.exit(0)

    except Exception as e:
        print(f"\n❌ Hyperparameter tuning failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
