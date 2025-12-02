
"""
src/models/train_lstm_model.py
Train LSTM models on pre-cleaned, split data with sequence windows.
Uses data from data/splits/ (already split and outlier-handled!)

Usage:
    python src/models/train_lstm_model.py                    # Trains ALL 5 targets
    python src/models/train_lstm_model.py --target revenue   # Train single target
"""
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings("ignore")

# ========================================
# Setup paths so we can import from src/
# ========================================
project_root = Path(__file__).resolve().parent.parent.parent
print(f"✅ Project root: {project_root}")

# Add src/ to sys.path (so 'utils' is importable)
sys.path.insert(0, str(project_root / "src"))

# Now import split utilities and MLflow tracker
from utils.split_utils import get_feature_target_split, drop_nan_targets  # noqa: E402
from utils.mlflow_tracker import MLflowTracker, track_lstm_run  # noqa: E402

print("✅ Imports successful (split_utils, mlflow_tracker)\n")


# ============================================
# Time Series Dataset for LSTM
# ============================================
class TimeSeriesDataset(Dataset):
    """
    Creates sequences for LSTM training using sliding windows
    """
    def __init__(self, X, y, sequence_length=8):
        """
        Args:
            X: Features array (samples, features)
            y: Target array (samples,)
            sequence_length: Number of time steps to look back
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.sequence_length = sequence_length
        
    def __len__(self):
        # We can create (total_samples - sequence_length) sequences
        return len(self.X) - self.sequence_length
    
    def __getitem__(self, idx):
        # Get sequence of length `sequence_length` ending at idx
        X_seq = self.X[idx:idx + self.sequence_length]
        # Predict the next value after the sequence
        y_target = self.y[idx + self.sequence_length]
        return X_seq, y_target


# ============================================
# LSTM Model Architecture
# ============================================
class LSTMRegressor(nn.Module):
    """
    LSTM Neural Network for Regression
    """
    def __init__(self, input_size, hidden_size=64, num_layers=1, dropout=0.2):
        super(LSTMRegressor, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0  # Only dropout if >1 layer
        )
        
        # Fully connected output layers
        self.fc1 = nn.Linear(hidden_size, 32)
        self.fc2 = nn.Linear(32, 1)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x shape: (batch, sequence_length, input_size)
        lstm_out, _ = self.lstm(x)
        
        # Take the last output of the sequence
        last_output = lstm_out[:, -1, :]
        
        # Pass through FC layers
        out = self.relu(self.fc1(last_output))
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out.squeeze()


# ============================================
# LSTM Model Trainer
# ============================================
class LSTMTrainer:
    """
    LSTM trainer for financial forecasting with sequence windows
    """

    def __init__(
        self,
        target_name: str,
        sequence_length: int = 8,
        hidden_size: int = 64,
        num_layers: int = 1,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        epochs: int = 50,
        patience: int = 5
    ):
        """
        Initialize LSTM trainer

        Args:
            target_name: Name of target (revenue, eps, etc.)
            sequence_length: Number of quarters to look back (default: 8 = 2 years)
            hidden_size: LSTM hidden dimension
            num_layers: Number of LSTM layers
            learning_rate: Learning rate for optimizer
            batch_size: Batch size for training
            epochs: Number of training epochs
            patience: Early stopping patience
        """
        self.target_name = target_name
        self.target_col = f"target_{target_name}"
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience

        self.model = None
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.feature_names = None
        self.best_epoch = None
        self.training_history = []
        self.train_metrics = None
        self.val_metrics = None
        self.test_metrics = None

    def load_data(self, splits_dir: str):
        """
        Load pre-split and cleaned data

        Args:
            splits_dir: Directory containing train/val/test CSVs
        """
        print(f"\n{'=' * 80}")
        print(f"📂 LOADING DATA FOR: {self.target_name.upper()}")
        print(f"{'=' * 80}")

        splits_path = Path(splits_dir)

        # Load splits
        train_file = splits_path / "train_data.csv"
        val_file = splits_path / "val_data.csv"
        test_file = splits_path / "test_data.csv"

        print(f"\n1️⃣ Loading pre-split data...")
        train_df = pd.read_csv(train_file)
        val_df = pd.read_csv(val_file)
        test_df = pd.read_csv(test_file)

        print(f"   ✅ Train: {len(train_df):,} rows")
        print(f"   ✅ Val:   {len(val_df):,} rows")
        print(f"   ✅ Test:  {len(test_df):,} rows")

        # Check target exists
        if self.target_col not in train_df.columns:
            raise ValueError(f"Target column '{self.target_col}' not found!")

        # Check valid samples
        train_valid = train_df[self.target_col].notna().sum()
        val_valid = val_df[self.target_col].notna().sum()
        test_valid = test_df[self.target_col].notna().sum()

        print(f"\n2️⃣ Target: {self.target_col}")
        print(f"   Train valid: {train_valid:,} ({train_valid/len(train_df)*100:.1f}%)")
        print(f"   Val valid:   {val_valid:,} ({val_valid/len(val_df)*100:.1f}%)")
        print(f"   Test valid:  {test_valid:,} ({test_valid/len(test_df)*100:.1f}%)")

        # For LSTM, we need enough data for sequences
        min_required = self.sequence_length + 100
        if train_valid < min_required:
            raise ValueError(
                f"Insufficient training data for sequences: "
                f"only {train_valid} valid samples, need at least {min_required}"
            )

        # Show target statistics
        print(f"\n3️⃣ Target statistics (after outlier handling):")
        train_target = train_df[self.target_col].dropna()
        print(f"   Mean:   {train_target.mean():,.2f}")
        print(f"   Std:    {train_target.std():,.2f}")
        print(f"   Min:    {train_target.min():,.2f}")
        print(f"   25%:    {train_target.quantile(0.25):,.2f}")
        print(f"   Median: {train_target.median():,.2f}")
        print(f"   75%:    {train_target.quantile(0.75):,.2f}")
        print(f"   Max:    {train_target.max():,.2f}")

        return train_df, val_df, test_df

    def prepare_features(self, train_df, val_df, test_df):
        """
        Prepare features using split_utils
        """
        print(f"\n{'=' * 80}")
        print(f"🔧 PREPARING FEATURES")
        print(f"{'=' * 80}")

        # Use split_utils to get features and handle encoding
        X_train, y_train = get_feature_target_split(
            train_df,
            target_col=self.target_col,
            encode_categoricals=True,
        )

        X_val, y_val = get_feature_target_split(
            val_df,
            target_col=self.target_col,
            encode_categoricals=True,
        )

        X_test, y_test = get_feature_target_split(
            test_df,
            target_col=self.target_col,
            encode_categoricals=True,
        )

        # Align columns (val/test might have different categories)
        print(f"\n4️⃣ Aligning features across splits...")
        train_cols = set(X_train.columns)

        # Add missing columns to val/test and fill with 0
        for col in train_cols:
            if col not in X_val.columns:
                X_val[col] = 0
            if col not in X_test.columns:
                X_test[col] = 0

        # Reorder columns to match train
        X_val = X_val[X_train.columns]
        X_test = X_test[X_train.columns]

        print(f"   ✅ All splits have {len(X_train.columns)} features")

        # Handle missing feature values
        print(f"\n5️⃣ Handling missing feature values...")
        nan_train = X_train.isna().sum().sum()
        nan_val = X_val.isna().sum().sum()
        nan_test = X_test.isna().sum().sum()

        if nan_train > 0 or nan_val > 0 or nan_test > 0:
            print(f"   Missing: Train={nan_train:,}, Val={nan_val:,}, Test={nan_test:,}")
            print(f"   Imputing with median...")

            imputer = SimpleImputer(strategy="median")
            X_train = pd.DataFrame(
                imputer.fit_transform(X_train),
                columns=X_train.columns,
                index=X_train.index,
            )
            X_val = pd.DataFrame(
                imputer.transform(X_val),
                columns=X_val.columns,
                index=X_val.index,
            )
            X_test = pd.DataFrame(
                imputer.transform(X_test),
                columns=X_test.columns,
                index=X_test.index,
            )
            print(f"   ✅ Imputation complete")
        else:
            print(f"   ✅ No missing values in features")

        # Drop rows with NaN targets (unlabeled rows)
        print(f"\n6️⃣ Removing rows with NaN targets from each split...")
        X_train, y_train = drop_nan_targets(X_train, y_train, "Train")
        X_val, y_val = drop_nan_targets(X_val, y_val, "Val")
        X_test, y_test = drop_nan_targets(X_test, y_test, "Test")

        print(f"\n{'=' * 80}")
        print(f"✅ FEATURES READY")
        print(f"{'=' * 80}")
        print(f"   Features: {X_train.shape[1]}")
        print(f"   Train samples: {len(X_train):,}")
        print(f"   Val samples:   {len(X_val):,}")
        print(f"   Test samples:  {len(X_test):,}")

        self.feature_names = X_train.columns.tolist()

        return X_train, y_train, X_val, y_val, X_test, y_test

    def train(self, X_train, y_train, X_val, y_val):
        """
        Train LSTM with sequence windows
        """
        print(f"\n{'=' * 80}")
        print(f"🚀 TRAINING LSTM: {self.target_name.upper()}")
        print(f"{'=' * 80}")
        print(f"   Device: {self.device}")
        print(f"   Sequence Length: {self.sequence_length} quarters")
        print(f"   Hidden Size: {self.hidden_size}")
        print(f"   Num Layers: {self.num_layers}")
        print(f"   Epochs: {self.epochs}")
        print(f"   Batch Size: {self.batch_size}")

        # Scale data
        print(f"\n⚙️  Scaling features and target...")
        X_train_scaled = self.scaler_X.fit_transform(X_train)
        X_val_scaled = self.scaler_X.transform(X_val)
        y_train_scaled = self.scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()
        y_val_scaled = self.scaler_y.transform(y_val.values.reshape(-1, 1)).ravel()

        # Create sequence datasets
        train_dataset = TimeSeriesDataset(X_train_scaled, y_train_scaled, self.sequence_length)
        val_dataset = TimeSeriesDataset(X_val_scaled, y_val_scaled, self.sequence_length)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        print(f"   ✅ Training sequences: {len(train_dataset):,} (from {len(X_train):,} samples)")
        print(f"   ✅ Validation sequences: {len(val_dataset):,} (from {len(X_val):,} samples)")

        # Initialize model
        input_size = X_train.shape[1]
        self.model = LSTMRegressor(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=0.2
        ).to(self.device)

        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3
        )

        # Training loop
        print(f"\n🔄 Training for {self.epochs} epochs...")
        print("=" * 80)

        best_val_loss = float('inf')
        patience_counter = 0
        checkpoint_path = Path('models/lstm') / f'lstm_{self.target_name}_checkpoint.pth'
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        for epoch in range(self.epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0

            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()

                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)

            # Validation phase
            self.model.eval()
            val_loss = 0.0

            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device)

                    outputs = self.model(X_batch)
                    loss = criterion(outputs, y_batch)
                    val_loss += loss.item()

            val_loss /= len(val_loader)

            # Learning rate scheduling
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]['lr']

            # Store history
            self.training_history.append({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'learning_rate': current_lr
            })

            # Print progress
            print(f"Epoch [{epoch+1:3d}/{self.epochs}] | "
                  f"Train Loss: {train_loss:.6f} | "
                  f"Val Loss: {val_loss:.6f} | "
                  f"LR: {current_lr:.6f}")

            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.best_epoch = epoch + 1
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), checkpoint_path)
            else:
                patience_counter += 1

            if patience_counter >= self.patience:
                print(f"\n⚠️  Early stopping at epoch {epoch+1}")
                print(f"   Best epoch: {self.best_epoch} (Val Loss: {best_val_loss:.6f})")
                break

        # Load best model
        self.model.load_state_dict(torch.load(checkpoint_path))

        print("=" * 80)
        print(f"✅ Training complete!")
        print(f"   Best epoch: {self.best_epoch}/{self.epochs}")
        print(f"   Best val loss: {best_val_loss:.6f}")

        # Calculate metrics on validation set
        print(f"\n6️⃣ Evaluating performance...")
        val_pred_scaled = self._predict_scaled(X_val)
        val_pred = self.scaler_y.inverse_transform(val_pred_scaled.reshape(-1, 1)).ravel()

        # Trim to match prediction length (we lose first sequence_length samples)
        y_val_trimmed = y_val.values[self.sequence_length:]
        val_pred_trimmed = val_pred[:len(y_val_trimmed)]

        val_rmse = float(np.sqrt(mean_squared_error(y_val_trimmed, val_pred_trimmed)))
        val_mae = float(mean_absolute_error(y_val_trimmed, val_pred_trimmed))
        val_r2 = float(r2_score(y_val_trimmed, val_pred_trimmed))

        self.val_metrics = {
            "rmse": val_rmse,
            "mae": val_mae,
            "r2": val_r2,
        }

        print(f"\n   📊 Validation Set:")
        print(f"      RMSE: {val_rmse:,.4f}")
        print(f"      MAE:  {val_mae:,.4f}")
        print(f"      R²:   {val_r2:.4f}")

    def _predict_scaled(self, X):
        """Internal: Predict on scaled data"""
        self.model.eval()
        X_scaled = self.scaler_X.transform(X)

        predictions = []
        with torch.no_grad():
            for i in range(self.sequence_length, len(X_scaled)):
                X_seq = X_scaled[i - self.sequence_length:i]
                X_seq = torch.FloatTensor(X_seq).unsqueeze(0).to(self.device)
                pred = self.model(X_seq).cpu().numpy()
                predictions.append(pred)

        return np.array(predictions)

    def evaluate_test(self, X_test, y_test):
        """
        Evaluate on test set
        """
        print(f"\n{'=' * 80}")
        print(f"📈 TEST SET EVALUATION")
        print(f"{'=' * 80}")

        test_pred_scaled = self._predict_scaled(X_test)
        test_pred = self.scaler_y.inverse_transform(test_pred_scaled.reshape(-1, 1)).ravel()

        # Trim to match prediction length
        y_test_trimmed = y_test.values[self.sequence_length:]
        test_pred_trimmed = test_pred[:len(y_test_trimmed)]

        test_rmse = float(np.sqrt(mean_squared_error(y_test_trimmed, test_pred_trimmed)))
        test_mae = float(mean_absolute_error(y_test_trimmed, test_pred_trimmed))
        test_r2 = float(r2_score(y_test_trimmed, test_pred_trimmed))

        self.test_metrics = {
            "rmse": test_rmse,
            "mae": test_mae,
            "r2": test_r2,
        }

        print(f"\n   📊 Test Set Performance:")
        print(f"      RMSE: {test_rmse:,.4f}")
        print(f"      MAE:  {test_mae:,.4f}")
        print(f"      R²:   {test_r2:.4f}")

        # Performance assessment
        if test_r2 > 0.70:
            status = "EXCELLENT ✅"
        elif test_r2 > 0.50:
            status = "GOOD ✅"
        elif test_r2 > 0.30:
            status = "ACCEPTABLE ⚠️"
        else:
            status = "NEEDS IMPROVEMENT ❌"

        print(f"\n   Status: {status}")

        return test_pred_trimmed

    def save_model(self, output_dir: str):
        """
        Save model and results
        """
        print(f"\n{'=' * 80}")
        print(f"💾 SAVING MODEL AND RESULTS")
        print(f"{'=' * 80}")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save model
        model_file = output_path / f"lstm_{self.target_name}.pth"
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'scaler_X': self.scaler_X,
            'scaler_y': self.scaler_y,
            'sequence_length': self.sequence_length,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'feature_names': self.feature_names,
            'best_epoch': self.best_epoch,
            'training_history': self.training_history,
            'train_metrics': self.train_metrics,
            'val_metrics': self.val_metrics,
            'test_metrics': self.test_metrics,
        }, model_file)
        print(f"\n   ✅ Model: {model_file}")

        # Save metrics
        metrics_file = output_path / f"lstm_{self.target_name}_metrics.json"
        with open(metrics_file, "w") as f:
            json.dump(
                {
                    "target": self.target_name,
                    "target_column": self.target_col,
                    "train": self.train_metrics,
                    "val": self.val_metrics,
                    "test": self.test_metrics,
                    "architecture": {
                        "sequence_length": self.sequence_length,
                        "hidden_size": self.hidden_size,
                        "num_layers": self.num_layers,
                        "learning_rate": self.learning_rate,
                        "batch_size": self.batch_size,
                    },
                    "best_epoch": self.best_epoch,
                    "total_epochs_trained": len(self.training_history),
                    "n_features": len(self.feature_names),
                    "timestamp": datetime.now().isoformat(),
                },
                f,
                indent=2,
            )
        print(f"   ✅ Metrics: {metrics_file}")

        # Save training history
        history_file = output_path / f"lstm_{self.target_name}_history.csv"
        history_df = pd.DataFrame(self.training_history)
        history_df.to_csv(history_file, index=False)
        print(f"   ✅ Training History: {history_file}")

        return model_file, metrics_file


# ============================================
# Main Training Function
# ============================================
def train_lstm_model(
    target_name: str,
    splits_dir: str,
    output_dir: str,
    sequence_length: int = 8,
    hidden_size: int = 64,
    num_layers: int = 1,
    epochs: int = 10,
    batch_size: int = 32
):
    """
    Train LSTM for a single target

    Args:
        target_name: Target to predict (revenue, eps, debt_equity, profit_margin, stock_return)
        splits_dir: Directory with train/val/test splits
        output_dir: Where to save model
        sequence_length: Number of quarters to look back
        hidden_size: LSTM hidden dimension
        num_layers: Number of LSTM layers
        epochs: Number of training epochs
        batch_size: Batch size
    """
    print(f"\n{'=' * 80}")
    print(f"🎯 LSTM TRAINING: {target_name.upper()}")
    print(f"{'=' * 80}")
    print(f"   Input: {splits_dir}")
    print(f"   Output: {output_dir}")

    # Initialize trainer
    trainer = LSTMTrainer(
        target_name=target_name,
        sequence_length=sequence_length,
        hidden_size=hidden_size,
        num_layers=num_layers,
        epochs=epochs,
        batch_size=batch_size
    )

    # Load data
    train_df, val_df, test_df = trainer.load_data(splits_dir)

    # Prepare features
    X_train, y_train, X_val, y_val, X_test, y_test = trainer.prepare_features(
        train_df, val_df, test_df
    )

    # Train
    trainer.train(X_train, y_train, X_val, y_val)

    # Evaluate on test
    _ = trainer.evaluate_test(X_test, y_test)

    # Save
    model_file, _ = trainer.save_model(output_dir)

    print(f"\n{'=' * 80}")
    print(f"✅ TRAINING COMPLETE: {target_name.upper()}")
    print(f"{'=' * 80}")
    print(f"   Test R²: {trainer.test_metrics['r2']:.4f}")
    print(f"   Model: {model_file}")

    return trainer


# ============================================
# CLI Interface
# ============================================
def main():
    parser = argparse.ArgumentParser(
        description="Train LSTM models for financial forecasting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train_lstm_model.py                    # Train ALL 5 targets (default)
  python train_lstm_model.py --target revenue   # Train only revenue
  python train_lstm_model.py --epochs 20 --layers 2  # Custom architecture
        """,
    )
    parser.add_argument(
        "--target",
        type=str,
        default="all",
        choices=["revenue", "eps", "debt_equity", "profit_margin", "stock_return", "all"],
        help="Target to predict (default: all)",
    )
    parser.add_argument(
        "--splits-dir",
        type=str,
        default="data/splits",
        help="Directory containing train/val/test splits",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/lstm",
        help="Directory to save models",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=8,
        help="Number of quarters to look back (default: 8 = 2 years)",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=64,
        help="LSTM hidden dimension (default: 64)",
    )
    parser.add_argument(
        "--layers",
        type=int,
        default=1,
        help="Number of LSTM layers (default: 1)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs (default: 50)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size (default: 32)",
    )

    args = parser.parse_args()

    # Define all targets
    all_targets = ["revenue", "eps", "debt_equity", "profit_margin", "stock_return"]

    # Determine which targets to train
    if args.target == "all":
        targets = all_targets
        print(f"\n{'=' * 80}")
        print(f"🎯 TRAINING LSTM FOR ALL {len(targets)} TARGETS")
        print(f"{'=' * 80}")
        print(f"Targets: {', '.join(targets)}")
        print(f"Sequence Length: {args.sequence_length} quarters")
        print(f"Hidden Size: {args.hidden_size}")
        print(f"Num Layers: {args.layers}")
        print(f"Epochs: {args.epochs}")
        print(f"{'=' * 80}\n")
    else:
        targets = [args.target]
        print(f"\n🎯 Training single target: {args.target}\n")

    # Train each target
    results = {}
    failed = []

    for i, target in enumerate(targets, 1):
        print(f"\n{'=' * 80}")
        print(f"📊 MODEL {i}/{len(targets)}: {target.upper()}")
        print(f"{'=' * 80}")

        try:
            trainer = train_lstm_model(
                target_name=target,
                splits_dir=args.splits_dir,
                output_dir=args.output_dir,
                sequence_length=args.sequence_length,
                hidden_size=args.hidden_size,
                num_layers=args.layers,
                epochs=args.epochs,
                batch_size=args.batch_size,
            )
            results[target] = trainer.test_metrics

            print(f"\n✅ {target.upper()} completed successfully!")
            print(f"   Test R²: {trainer.test_metrics['r2']:.4f}")

        except Exception as e:
            print(f"\n❌ ERROR training {target}:")
            print(f"   {str(e)}")
            failed.append(target)
            import traceback
            print(f"\n   Full traceback:")
            traceback.print_exc()
            continue

    # Final Summary
    if len(results) > 0:
        print(f"\n\n{'=' * 80}")
        print(f"📊 FINAL TRAINING SUMMARY - ALL TARGETS")
        print(f"{'=' * 80}\n")

        print(f"{'Target':<20} {'Test R²':>10} {'Test RMSE':>15} {'Test MAE':>15}")
        print(f"-" * 65)

        for target, metrics in results.items():
            print(
                f"{target:<20} {metrics['r2']:>10.4f} "
                f"{metrics['rmse']:>15,.2f} {metrics['mae']:>15,.2f}"
            )

        if len(results) > 1:
            avg_r2 = np.mean([m["r2"] for m in results.values()])
            print(f"-" * 65)
            print(f"{'AVERAGE':<20} {avg_r2:>10.4f}")

        print(f"\n{'=' * 80}")
        print(
            f"✅ TRAINING COMPLETE: {len(results)}/{len(targets)} models trained successfully!"
        )

        if failed:
            print(f"❌ Failed: {len(failed)} models - {', '.join(failed)}")

        print(f"{'=' * 80}")
        print(f"\n📁 Models saved to: {args.output_dir}/")
        for target in results.keys():
            print(f"   - lstm_{target}.pth")
            print(f"   - lstm_{target}_metrics.json")
            print(f"   - lstm_{target}_history.csv")

    else:
        print(f"\n{'=' * 80}")
        print(f"❌ NO MODELS TRAINED SUCCESSFULLY")
        print(f"{'=' * 80}")

        if failed:
            print(f"\nFailed targets: {', '.join(failed)}")

        print(f"\nPlease check:")
        print(f"   1. Data splits exist in: {args.splits_dir}/")
        print(f"   2. Target columns exist in data")
        print(f"   3. Sufficient valid samples for training")
        print(f"   4. Error messages above for details")


if __name__ == "__main__":
    print(f"\n{'=' * 80}")
    print(f"🚀 LSTM MODEL TRAINING")
    print(f"{'=' * 80}")
    print("This script will train LSTM models for financial forecasting")
    print("Using sequence windows on pre-split and cleaned data from data/splits/")
    print(f"{'=' * 80}\n")

    main()