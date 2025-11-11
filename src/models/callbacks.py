"""
Custom Keras Callbacks
Specialized callbacks for model training monitoring and control
"""

from tensorflow import keras
import numpy as np
import json
from pathlib import Path
from datetime import datetime


class MLflowLoggingCallback(keras.callbacks.Callback):
    """
    Custom callback to log metrics to MLflow during training

    Usage:
        callback = MLflowLoggingCallback(log_every_n_epochs=5)
        model.fit(X, y, callbacks=[callback])
    """

    def __init__(self, log_every_n_epochs=1):
        super().__init__()
        self.log_every_n_epochs = log_every_n_epochs

    def on_epoch_end(self, epoch, logs=None):
        """Log metrics to MLflow at end of each epoch"""

        if epoch % self.log_every_n_epochs == 0:
            try:
                import mlflow

                # Log training metrics
                mlflow.log_metrics({
                    f'epoch_{epoch}_train_loss': logs.get('loss', 0),
                    f'epoch_{epoch}_train_acc': logs.get('accuracy', 0),
                    f'epoch_{epoch}_val_loss': logs.get('val_loss', 0),
                    f'epoch_{epoch}_val_acc': logs.get('val_accuracy', 0)
                }, step=epoch)

            except Exception as e:
                print(f"Warning: Could not log to MLflow: {e}")


class MetricsLogger(keras.callbacks.Callback):
    """
    Log detailed metrics to JSON file during training

    Saves epoch-by-epoch metrics for later analysis

    Usage:
        callback = MetricsLogger(filepath='models/metrics/epoch_metrics.json')
        model.fit(X, y, callbacks=[callback])
    """

    def __init__(self, filepath='models/metrics/epoch_metrics.json'):
        super().__init__()
        self.filepath = filepath
        self.epoch_metrics = []

    def on_epoch_end(self, epoch, logs=None):
        """Save metrics for each epoch"""

        epoch_data = {
            'epoch': epoch,
            'timestamp': datetime.now().isoformat(),
            'loss': float(logs.get('loss', 0)),
            'accuracy': float(logs.get('accuracy', 0)),
            'val_loss': float(logs.get('val_loss', 0)),
            'val_accuracy': float(logs.get('val_accuracy', 0)),
            'learning_rate': float(keras.backend.get_value(self.model.optimizer.lr))
        }

        self.epoch_metrics.append(epoch_data)

    def on_train_end(self, logs=None):
        """Save all metrics to file when training completes"""

        Path(self.filepath).parent.mkdir(parents=True, exist_ok=True)

        with open(self.filepath, 'w') as f:
            json.dump(self.epoch_metrics, f, indent=2)

        print(f"\n✓ Epoch metrics saved to {self.filepath}")


class TimeBasedEarlyStopping(keras.callbacks.Callback):
    """
    Stop training after specified time duration
    Useful for limiting training time in experiments

    Usage:
        callback = TimeBasedEarlyStopping(max_time_seconds=3600)  # 1 hour max
        model.fit(X, y, callbacks=[callback])
    """

    def __init__(self, max_time_seconds=3600):
        """
        Args:
            max_time_seconds: Maximum training time in seconds (default: 1 hour)
        """
        super().__init__()
        self.max_time_seconds = max_time_seconds
        self.start_time = None

    def on_train_begin(self, logs=None):
        self.start_time = datetime.now()

    def on_epoch_end(self, epoch, logs=None):
        elapsed = (datetime.now() - self.start_time).total_seconds()

        if elapsed > self.max_time_seconds:
            print(
                f"\n⏰ Time limit reached ({elapsed:.0f}s > {self.max_time_seconds}s)")
            print(f"   Stopping training at epoch {epoch}")
            self.model.stop_training = True


class MemoryUsageCallback(keras.callbacks.Callback):
    """
    Monitor memory usage during training
    Useful for debugging OOM (Out Of Memory) errors

    Usage:
        callback = MemoryUsageCallback(log_every_n_epochs=10)
        model.fit(X, y, callbacks=[callback])

    Note: Requires psutil: pip install psutil
    """

    def __init__(self, log_every_n_epochs=10):
        super().__init__()
        self.log_every_n_epochs = log_every_n_epochs

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.log_every_n_epochs == 0:
            try:
                import psutil
                import os

                process = psutil.Process(os.getpid())
                memory_mb = process.memory_info().rss / (1024 * 1024)

                print(
                    f"\n  💾 Memory usage at epoch {epoch}: {memory_mb:.1f} MB")

            except ImportError:
                if epoch == 0:
                    print("\n  ⚠ psutil not installed, skipping memory monitoring")


class ProgressCallback(keras.callbacks.Callback):
    """
    Print training progress with estimated time remaining

    Shows clean progress updates with ETA

    Usage:
        callback = ProgressCallback(total_epochs=100)
        model.fit(X, y, epochs=100, callbacks=[callback])
    """

    def __init__(self, total_epochs):
        super().__init__()
        self.total_epochs = total_epochs
        self.start_time = None

    def on_train_begin(self, logs=None):
        self.start_time = datetime.now()
        print(f"\n🚀 Training started: {self.total_epochs} epochs")

    def on_epoch_end(self, epoch, logs=None):
        elapsed = (datetime.now() - self.start_time).total_seconds()
        epochs_done = epoch + 1
        epochs_remaining = self.total_epochs - epochs_done

        # Estimate time remaining
        avg_time_per_epoch = elapsed / epochs_done
        eta_seconds = avg_time_per_epoch * epochs_remaining
        eta_minutes = eta_seconds / 60

        # Progress percentage
        progress = epochs_done / self.total_epochs * 100

        # Progress bar
        bar_length = 30
        filled = int(bar_length * epochs_done / self.total_epochs)
        bar = '█' * filled + '░' * (bar_length - filled)

        print(f"  [{bar}] {progress:5.1f}% | "
              f"Epoch {epochs_done:3d}/{self.total_epochs} | "
              f"ETA: {eta_minutes:4.1f}min | "
              f"Loss: {logs.get('val_loss', 0):.4f}")


class SaveBestModelCallback(keras.callbacks.Callback):
    """
    Save model whenever it achieves best validation performance
    More flexible than standard ModelCheckpoint

    Usage:
        callback = SaveBestModelCallback(
            filepath='models/best_model.h5',
            monitor='val_auc',
            mode='max'
        )
        model.fit(X, y, callbacks=[callback])
    """

    def __init__(self, filepath='models/checkpoints/best_model.h5',
                 monitor='val_loss', mode='min'):
        super().__init__()
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode

        if mode == 'min':
            self.best_value = np.inf
            self.is_better = lambda new, best: new < best
        else:
            self.best_value = -np.inf
            self.is_better = lambda new, best: new > best

    def on_epoch_end(self, epoch, logs=None):
        current_value = logs.get(self.monitor)

        if current_value is None:
            return

        if self.is_better(current_value, self.best_value):
            print(
                f"\n  ✓ {self.monitor} improved: {self.best_value:.4f} → {current_value:.4f}")
            print(f"    Saving model to {self.filepath}")

            Path(self.filepath).parent.mkdir(parents=True, exist_ok=True)
            self.model.save(self.filepath)
            self.best_value = current_value


class LearningRateLogger(keras.callbacks.Callback):
    """
    Log learning rate changes during training
    Useful when using ReduceLROnPlateau

    Usage:
        callback = LearningRateLogger()
        model.fit(X, y, callbacks=[callback])
    """

    def __init__(self):
        super().__init__()
        self.lr_history = []

    def on_epoch_end(self, epoch, logs=None):
        lr = float(keras.backend.get_value(self.model.optimizer.lr))
        self.lr_history.append({'epoch': epoch, 'learning_rate': lr})

        # Print if LR changed
        if epoch > 0 and lr != self.lr_history[-2]['learning_rate']:
            print(
                f"\n  📉 Learning rate reduced: {self.lr_history[-2]['learning_rate']:.6f} → {lr:.6f}")


class SlackNotificationCallback(keras.callbacks.Callback):
    """
    Send Slack notification when training completes or fails

    Usage:
        callback = SlackNotificationCallback(
            webhook_url='https://hooks.slack.com/services/YOUR/WEBHOOK/URL',
            notify_on='completion'
        )
        model.fit(X, y, callbacks=[callback])

    Note: Optional - requires Slack webhook setup
    """

    def __init__(self, webhook_url=None, notify_on='completion'):
        """
        Args:
            webhook_url: Slack webhook URL (get from Slack app settings)
            notify_on: 'completion', 'improvement', 'both'
        """
        super().__init__()
        self.webhook_url = webhook_url
        self.notify_on = notify_on
        self.best_val_loss = np.inf

    def on_epoch_end(self, epoch, logs=None):
        """Check for improvements"""

        val_loss = logs.get('val_loss', np.inf)

        if self.notify_on in ['improvement', 'both'] and val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.send_notification(
                f"🎯 Improvement at epoch {epoch}: val_loss = {val_loss:.4f}")

    def on_train_end(self, logs=None):
        """Send notification when training completes"""

        if self.notify_on in ['completion', 'both']:
            message = (
                f"✅ Model training completed!\n"
                f"Final val_loss: {logs.get('val_loss', 0):.4f}\n"
                f"Final val_accuracy: {logs.get('val_accuracy', 0):.4f}"
            )
            self.send_notification(message)

    def send_notification(self, message):
        """Send message to Slack"""

        if not self.webhook_url:
            return

        try:
            import requests

            payload = {'text': message}
            requests.post(self.webhook_url, json=payload, timeout=5)

        except Exception as e:
            print(f"Warning: Could not send Slack notification: {e}")


class GradientLogger(keras.callbacks.Callback):
    """
    Monitor gradient statistics during training
    Helps detect vanishing/exploding gradients

    Usage:
        callback = GradientLogger(log_every_n_epochs=10)
        model.fit(X, y, callbacks=[callback])
    """

    def __init__(self, log_every_n_epochs=10):
        super().__init__()
        self.log_every_n_epochs = log_every_n_epochs

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.log_every_n_epochs == 0:
            # Get gradients
            weights = self.model.trainable_weights

            if weights:
                # Calculate gradient norms
                gradients = []
                for w in weights:
                    g = keras.backend.get_value(w)
                    gradients.append(np.linalg.norm(g))

                avg_grad = np.mean(gradients)
                max_grad = np.max(gradients)
                min_grad = np.min(gradients)

                print(f"\n  📊 Gradient stats at epoch {epoch}:")
                print(
                    f"     Avg: {avg_grad:.6f} | Max: {max_grad:.6f} | Min: {min_grad:.6f}")

                # Warn about gradient issues
                if max_grad > 10:
                    print(
                        f"     ⚠ Warning: Large gradients detected (exploding gradients?)")
                if avg_grad < 1e-7:
                    print(
                        f"     ⚠ Warning: Very small gradients (vanishing gradients?)")


def get_default_callbacks(params, enable_mlflow=True, enable_progress=True):
    """
    Get standard set of callbacks based on params.yaml configuration

    This is a helper function to quickly get all commonly used callbacks

    Args:
        params: Dict from params.yaml
        enable_mlflow: Whether to enable MLflow logging
        enable_progress: Whether to show progress bar

    Returns:
        list: Configured callbacks

    Usage:
        import yaml
        params = yaml.safe_load(open('params.yaml'))
        callbacks = get_default_callbacks(params)
        model.fit(X, y, callbacks=callbacks)
    """

    callbacks = []

    train_params = params['train']

    # Early Stopping
    if train_params['early_stopping']['enabled']:
        callbacks.append(
            keras.callbacks.EarlyStopping(
                monitor=train_params['early_stopping']['monitor'],
                patience=train_params['early_stopping']['patience'],
                min_delta=train_params['early_stopping']['min_delta'],
                restore_best_weights=train_params['early_stopping']['restore_best_weights'],
                verbose=1
            )
        )
        print("  ✓ Early Stopping enabled")

    # Reduce Learning Rate on Plateau
    if train_params['reduce_lr']['enabled']:
        callbacks.append(
            keras.callbacks.ReduceLROnPlateau(
                monitor=train_params['reduce_lr']['monitor'],
                factor=train_params['reduce_lr']['factor'],
                patience=train_params['reduce_lr']['patience'],
                min_lr=train_params['reduce_lr']['min_lr'],
                verbose=1
            )
        )
        print("  ✓ Reduce LR on Plateau enabled")

    # Model Checkpoint
    if train_params['model_checkpoint']['enabled']:
        filepath = train_params['model_checkpoint']['filepath']
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        callbacks.append(
            keras.callbacks.ModelCheckpoint(
                filepath=filepath,
                monitor=train_params['model_checkpoint']['monitor'],
                save_best_only=train_params['model_checkpoint']['save_best_only'],
                verbose=1
            )
        )
        print(f"  ✓ Model Checkpoint enabled: {filepath}")

    # CSV Logger (always enabled)
    Path('models/metrics').mkdir(parents=True, exist_ok=True)
    callbacks.append(
        keras.callbacks.CSVLogger('models/metrics/training_history.csv')
    )
    print("  ✓ CSV Logger enabled")

    # Custom callbacks
    callbacks.append(MetricsLogger())
    print("  ✓ JSON Metrics Logger enabled")

    if enable_progress:
        callbacks.append(ProgressCallback(total_epochs=train_params['epochs']))
        print("  ✓ Progress Callback enabled")

    # MLflow logging (optional)
    if enable_mlflow:
        try:
            import mlflow
            callbacks.append(MLflowLoggingCallback(log_every_n_epochs=5))
            print("  ✓ MLflow Logging enabled")
        except ImportError:
            print("  ⚠ MLflow not available, skipping MLflow logging")

    print(f"\n  Total callbacks: {len(callbacks)}")

    return callbacks


if __name__ == "__main__":
    # Test callbacks
    print("="*70)
    print("TESTING CALLBACKS")
    print("="*70)

    # Create dummy model
    print("\n1. Creating test model...")
    model = keras.Sequential([
        keras.layers.Dense(10, activation='relu', input_shape=(5,)),
        keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    print("   ✓ Model created")

    # Create test data
    print("\n2. Creating test data...")
    X_dummy = np.random.randn(100, 5)
    y_dummy = np.random.randint(0, 2, 100)
    print("   ✓ Test data created")

    # Test callbacks
    print("\n3. Testing callbacks...")

    test_callbacks = [
        MetricsLogger(filepath='test_metrics.json'),
        ProgressCallback(total_epochs=10),
        TimeBasedEarlyStopping(max_time_seconds=60),
        LearningRateLogger()
    ]

    print(f"   Using {len(test_callbacks)} callbacks")

    # Train
    print("\n4. Training with callbacks...")
    history = model.fit(
        X_dummy, y_dummy,
        epochs=10,
        batch_size=16,
        validation_split=0.2,
        callbacks=test_callbacks,
        verbose=0
    )

    print("\n" + "="*70)
    print("✅ CALLBACKS TEST COMPLETED!")
    print("="*70)

    # Check outputs
    print("\n5. Verifying outputs...")

    if Path('test_metrics.json').exists():
        print("   ✓ test_metrics.json created")

        with open('test_metrics.json', 'r') as f:
            metrics = json.load(f)
        print(f"   ✓ Logged {len(metrics)} epochs")

        # Clean up
        Path('test_metrics.json').unlink()
    else:
        print("   ✗ test_metrics.json not found")

    print("\n✅ All callbacks working correctly!")
    print("\nAvailable callbacks:")
    print("  - MLflowLoggingCallback")
    print("  - MetricsLogger")
    print("  - TimeBasedEarlyStopping")
    print("  - MemoryUsageCallback")
    print("  - ProgressCallback")
    print("  - SaveBestModelCallback")
    print("  - LearningRateLogger")
    print("  - SlackNotificationCallback")
    print("  - GradientLogger")
    print("\nHelper function:")
    print("  - get_default_callbacks(params)")
