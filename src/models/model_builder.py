"""
Model Builder
Creates LSTM model architecture based on params.yaml configuration
"""

from tensorflow import keras
from tensorflow.keras import layers, regularizers
import yaml
from pathlib import Path


def load_params():
    """Load parameters from params.yaml"""
    params_file = Path('params.yaml')
    if not params_file.exists():
        raise FileNotFoundError("params.yaml not found")

    with open(params_file, 'r') as f:
        return yaml.safe_load(f)


def build_lstm_model(input_shape, params=None):
    """
    Build LSTM model for financial stress prediction

    Args:
        input_shape: tuple, (timesteps, features) e.g., (12, 50)
        params: dict, model parameters (if None, loads from params.yaml)

    Returns:
        keras.Model: Compiled LSTM model

    Example:
        model = build_lstm_model(input_shape=(12, 50))
        model.summary()
    """

    if params is None:
        all_params = load_params()
        params = all_params['train']

    print("\n🏗️  Building LSTM model...")
    print(f"  Input shape: {input_shape}")
    print(f"  Architecture: {params['lstm_units']} LSTM units")
    print(f"  Dropout: {params['dropout_rate']}")

    # Regularizer
    l2_reg = regularizers.l2(params.get('l2_regularization', 0.01))

    model = keras.Sequential([
        # First LSTM layer
        layers.LSTM(
            params['lstm_units'][0],
            return_sequences=True,
            input_shape=input_shape,
            recurrent_dropout=params.get('recurrent_dropout', 0.2),
            kernel_regularizer=l2_reg,
            name='lstm_layer_1'
        ),
        layers.Dropout(params['dropout_rate'], name='dropout_1'),

        # Second LSTM layer
        layers.LSTM(
            params['lstm_units'][1],
            return_sequences=False,
            recurrent_dropout=params.get('recurrent_dropout', 0.2),
            kernel_regularizer=l2_reg,
            name='lstm_layer_2'
        ),
        layers.Dropout(params['dropout_rate'], name='dropout_2'),

        # Dense layer
        layers.Dense(
            params['dense_units'][0],
            activation='relu',
            kernel_regularizer=l2_reg,
            name='dense_1'
        ),
        layers.Dropout(params['dropout_rate'], name='dropout_3'),

        # Output layer (binary classification)
        layers.Dense(1, activation='sigmoid', name='output')
    ])

    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=params['learning_rate']),
        loss=params.get('loss', 'binary_crossentropy'),
        metrics=[
            'accuracy',
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
            keras.metrics.AUC(name='auc')
        ]
    )

    print("  ✓ Model built and compiled")

    return model


def get_callbacks(params=None):
    """
    Create training callbacks based on params.yaml

    Returns:
        list: Keras callbacks
    """

    if params is None:
        all_params = load_params()
        params = all_params['train']

    callbacks = []

    # Early Stopping
    if params['early_stopping']['enabled']:
        callbacks.append(
            keras.callbacks.EarlyStopping(
                monitor=params['early_stopping']['monitor'],
                patience=params['early_stopping']['patience'],
                min_delta=params['early_stopping']['min_delta'],
                restore_best_weights=params['early_stopping']['restore_best_weights'],
                verbose=1
            )
        )

    # Reduce Learning Rate on Plateau
    if params['reduce_lr']['enabled']:
        callbacks.append(
            keras.callbacks.ReduceLROnPlateau(
                monitor=params['reduce_lr']['monitor'],
                factor=params['reduce_lr']['factor'],
                patience=params['reduce_lr']['patience'],
                min_lr=params['reduce_lr']['min_lr'],
                verbose=1
            )
        )

    # Model Checkpoint
    if params['model_checkpoint']['enabled']:
        Path('models/checkpoints').mkdir(parents=True, exist_ok=True)
        callbacks.append(
            keras.callbacks.ModelCheckpoint(
                filepath=params['model_checkpoint']['filepath'],
                monitor=params['model_checkpoint']['monitor'],
                save_best_only=params['model_checkpoint']['save_best_only'],
                verbose=1
            )
        )

    # CSV Logger
    Path('models/metrics').mkdir(parents=True, exist_ok=True)
    callbacks.append(
        keras.callbacks.CSVLogger('models/metrics/training_history.csv')
    )

    print(f"  ✓ {len(callbacks)} callbacks configured")

    return callbacks


if __name__ == "__main__":
    # Test model creation
    print("Testing model builder...")

    # Create test model
    model = build_lstm_model(input_shape=(12, 50))
    model.summary()

    print("\n✓ Model builder test successful!")
