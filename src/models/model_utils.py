"""
Model Utilities
Helper functions for model training, evaluation, and management
"""

import numpy as np
import pandas as pd
import pickle
import json
from pathlib import Path
from datetime import datetime
from tensorflow import keras


def load_model_with_metadata(model_path='models/lstm_baseline'):
    """
    Load model along with its metadata and preprocessing artifacts

    Args:
        model_path: Path to model directory

    Returns:
        tuple: (model, scaler, metadata, feature_names)
    """

    model_path = Path(model_path)

    # Load model
    model = keras.models.load_model(model_path / 'model.h5')

    # Load scaler
    scaler_path = Path('models/scaler.pkl')
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    # Load metadata
    with open(model_path / 'metadata.json', 'r') as f:
        metadata = json.load(f)

    # Load feature names
    with open(model_path / 'feature_names.json', 'r') as f:
        feature_names = json.load(f)

    return model, scaler, metadata, feature_names


def save_model_package(model, scaler, metadata, feature_names, output_dir='models/lstm_baseline'):
    """
    Save complete model package with all artifacts

    Args:
        model: Trained Keras model
        scaler: Fitted StandardScaler
        metadata: Dict with model information
        feature_names: List of feature names
        output_dir: Directory to save package
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save model
    model.save(output_dir / 'model.h5')

    # Save scaler
    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    # Save metadata
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    # Save feature names
    with open(output_dir / 'feature_names.json', 'w') as f:
        json.dump(feature_names, f, indent=2)

    print(f"✓ Model package saved to {output_dir}")


def reshape_for_lstm(X, timesteps):
    """
    Reshape flat feature matrix for LSTM input

    Args:
        X: numpy array or DataFrame of shape (samples, total_features)
        timesteps: number of time steps

    Returns:
        numpy array of shape (samples, timesteps, features_per_timestep)

    Example:
        X shape: (1000, 600)
        timesteps: 12
        Output shape: (1000, 12, 50)
    """

    if isinstance(X, pd.DataFrame):
        X = X.values

    n_samples = X.shape[0]
    n_features_per_timestep = X.shape[1] // timesteps

    if X.shape[1] % timesteps != 0:
        raise ValueError(
            f"Total features ({X.shape[1]}) must be divisible by timesteps ({timesteps}). "
            f"Got remainder: {X.shape[1] % timesteps}"
        )

    X_reshaped = X.reshape(n_samples, timesteps, n_features_per_timestep)

    return X_reshaped


def predict_with_preprocessing(model, scaler, X_raw, timesteps=12):
    """
    Make predictions with full preprocessing pipeline

    Args:
        model: Trained model
        scaler: Fitted scaler
        X_raw: Raw features (not scaled)
        timesteps: Sequence length

    Returns:
        numpy array: Predictions
    """

    # Scale features
    X_scaled = scaler.transform(X_raw)

    # Reshape for LSTM
    X_lstm = reshape_for_lstm(X_scaled, timesteps)

    # Predict
    predictions = model.predict(X_lstm, verbose=0)

    return predictions


def calculate_class_weights(y):
    """
    Calculate balanced class weights for imbalanced datasets

    Args:
        y: Target labels

    Returns:
        dict: Class weights {0: weight_0, 1: weight_1}
    """

    from sklearn.utils.class_weight import compute_class_weight

    classes = np.unique(y)
    weights = compute_class_weight('balanced', classes=classes, y=y)
    class_weight_dict = dict(zip(classes.astype(int), weights))

    return class_weight_dict


def get_model_size(model_path):
    """
    Get model file size in MB

    Args:
        model_path: Path to model.h5 file

    Returns:
        float: Size in MB
    """

    size_bytes = Path(model_path).stat().st_size
    size_mb = size_bytes / (1024 * 1024)

    return size_mb


def count_model_parameters(model):
    """
    Count trainable and non-trainable parameters

    Args:
        model: Keras model

    Returns:
        dict: Parameter counts
    """

    trainable = sum([np.prod(v.shape) for v in model.trainable_weights])
    non_trainable = sum([np.prod(v.shape)
                        for v in model.non_trainable_weights])

    return {
        'trainable': int(trainable),
        'non_trainable': int(non_trainable),
        'total': int(trainable + non_trainable)
    }


def create_model_summary_dict(model):
    """
    Create dictionary with model architecture summary

    Args:
        model: Keras model

    Returns:
        dict: Model summary information
    """

    summary = {
        'layers': [],
        'total_params': count_model_parameters(model)
    }

    for layer in model.layers:
        layer_info = {
            'name': layer.name,
            'type': layer.__class__.__name__,
            'output_shape': str(layer.output_shape),
            'params': layer.count_params()
        }
        summary['layers'].append(layer_info)

    return summary


def compare_model_performance(model1_metrics, model2_metrics):
    """
    Compare two models and determine which is better

    Args:
        model1_metrics: Dict of metrics for model 1
        model2_metrics: Dict of metrics for model 2

    Returns:
        dict: Comparison results
    """

    comparison = {
        'model1_better': [],
        'model2_better': [],
        'similar': []
    }

    for metric in model1_metrics.keys():
        if metric in model2_metrics:
            diff = model2_metrics[metric] - model1_metrics[metric]

            if abs(diff) < 0.01:  # Less than 1% difference
                comparison['similar'].append(metric)
            elif diff > 0:
                comparison['model2_better'].append((metric, diff))
            else:
                comparison['model1_better'].append((metric, abs(diff)))

    return comparison


def plot_feature_importance(feature_importance, feature_names, top_n=20, save_path='models/metrics/feature_importance.png'):
    """
    Plot feature importance bar chart

    Args:
        feature_importance: Array of importance scores
        feature_names: List of feature names
        top_n: Number of top features to show
        save_path: Path to save plot
    """

    import matplotlib.pyplot as plt

    # Create DataFrame
    df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    })

    # Sort and get top N
    df = df.sort_values('importance', ascending=False).head(top_n)

    # Plot
    plt.figure(figsize=(10, 8))
    plt.barh(df['feature'], df['importance'])
    plt.xlabel('Importance Score', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.title(f'Top {top_n} Feature Importance',
              fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Feature importance plot saved: {save_path}")


def create_prediction_explanation(model, scaler, X_sample, feature_names, method='basic'):
    """
    Generate explanation for a single prediction

    Args:
        model: Trained model
        scaler: Fitted scaler
        X_sample: Single sample to explain
        feature_names: List of feature names
        method: 'basic' or 'shap'

    Returns:
        dict: Explanation
    """

    # Make prediction
    X_scaled = scaler.transform(X_sample.reshape(1, -1))
    X_lstm = reshape_for_lstm(X_scaled, timesteps=12)
    prediction = model.predict(X_lstm, verbose=0)[0][0]

    explanation = {
        'prediction': float(prediction),
        'risk_level': 'HIGH' if prediction > 0.7 else 'MEDIUM' if prediction > 0.4 else 'LOW'
    }

    if method == 'basic':
        # Simple explanation: show top contributing features
        # (This is simplified - for advanced, use SHAP)
        explanation['method'] = 'basic'
        explanation['note'] = 'For detailed explanation, use SHAP analysis'

    return explanation


def validate_model_inputs(X, expected_shape):
    """
    Validate model inputs before prediction

    Args:
        X: Input data
        expected_shape: Expected shape tuple

    Raises:
        ValueError: If validation fails
    """

    if X.shape[1:] != expected_shape[1:]:
        raise ValueError(
            f"Input shape mismatch. Expected {expected_shape}, got {X.shape}"
        )

    # Check for NaN values
    if np.isnan(X).any():
        raise ValueError("Input contains NaN values")

    # Check for infinite values
    if np.isinf(X).any():
        raise ValueError("Input contains infinite values")

    print("✓ Input validation passed")


def generate_model_report(model, history, metrics, save_path='models/metrics/model_report.json'):
    """
    Generate comprehensive model report

    Args:
        model: Trained model
        history: Training history
        metrics: Evaluation metrics
        save_path: Path to save report

    Returns:
        dict: Complete model report
    """

    report = {
        'timestamp': datetime.now().isoformat(),
        'model_architecture': create_model_summary_dict(model),
        'training_history': {
            'epochs_trained': len(history.history['loss']),
            'final_train_loss': float(history.history['loss'][-1]),
            'final_val_loss': float(history.history['val_loss'][-1]),
            'best_val_loss': float(min(history.history['val_loss'])),
            'best_epoch': int(np.argmin(history.history['val_loss']))
        },
        'evaluation_metrics': metrics,
        'model_parameters': count_model_parameters(model)
    }

    # Save report
    with open(save_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"✓ Model report saved: {save_path}")

    return report


if __name__ == "__main__":
    # Test utilities
    print("Testing model utilities...")

    # Test reshape
    X_test = np.random.randn(100, 600)
    X_lstm = reshape_for_lstm(X_test, timesteps=12)
    print(f"✓ Reshape test: {X_test.shape} -> {X_lstm.shape}")

    # Test class weights
    y_test = np.array([0]*80 + [1]*20)
    weights = calculate_class_weights(y_test)
    print(f"✓ Class weights: {weights}")

    print("\n✅ Model utilities working correctly!")
