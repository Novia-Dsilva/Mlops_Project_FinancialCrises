"""
src/models/lstm_hyperparameter_tuning.py
Hyperparameter tuning for LSTM models using Optuna

Expected improvement: May improve from R² 0.17 → 0.25-0.35
(Still won't beat XGBoost, but shows due diligence)

Usage:
    # Tune single target
    python src/models/lstm_hyperparameter_tuning.py --target profit_margin --trials 30
    
    # Quick test
    python src/models/lstm_hyperparameter_tuning.py --target profit_margin --trials 10
"""
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
import optuna

warnings.filterwarnings("ignore")

# Setup paths
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from utils.split_utils import get_feature_target_split, drop_nan_targets

print("✅ Imports successful (split_utils, optuna)\n")


# ============================================
# Copy Dataset and Model from train_lstm_model.py
# ============================================

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y, sequence_length=4):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.sequence_length = sequence_length
        
    def __len__(self):
        return len(self.X) - self.sequence_length
    
    def __getitem__(self, idx):
        X_seq = self.X[idx:idx + self.sequence_length]
        y_target = self.y[idx + self.sequence_length]
        return X_seq, y_target


class LSTMRegressor(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.2):
        super(LSTMRegressor, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc1 = nn.Linear(hidden_size, 32)
        self.fc2 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        out = self.relu(self.fc1(last_output))
        out = self.dropout(out)
        out = self.fc2(out)
        return out.squeeze()


# ============================================
# LSTM Hyperparameter Search Space
# ============================================

def get_lstm_search_space(trial: optuna.Trial) -> dict:
    """
    Define hyperparameter search space for LSTM
    
    Args:
        trial: Optuna trial object
    
    Returns:
        Dictionary of hyperparameters to try
    """
    params = {
        # Architecture
        'sequence_length': trial.suggest_int('sequence_length', 2, 8),  # 2 quarters to 2 years
        'hidden_size': trial.suggest_categorical('hidden_size', [32, 64, 128, 256]),
        'num_layers': trial.suggest_int('num_layers', 1, 3),
        
        # Training
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64]),
        'dropout': trial.suggest_float('dropout', 0.1, 0.5),
        
        # Fixed
        'epochs': 50,  # Will use early stopping anyway
        'patience': 10,
    }
    
    return params


# ============================================
# LSTM Objective Function
# ============================================

class LSTMObjective:
    """
    Objective function for Optuna hyperparameter optimization
    """
    
    def __init__(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        target_name: str,
        device: torch.device
    ):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.target_name = target_name
        self.device = device
        self.best_score = float('-inf')
        
    def __call__(self, trial: optuna.Trial) -> float:
        """
        Objective function to minimize (negative R²)
        """
        # Get hyperparameters
        params = get_lstm_search_space(trial)
        
        # Scale data
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        
        X_train_scaled = scaler_X.fit_transform(self.X_train)
        X_val_scaled = scaler_X.transform(self.X_val)
        y_train_scaled = scaler_y.fit_transform(self.y_train.values.reshape(-1, 1)).ravel()
        y_val_scaled = scaler_y.transform(self.y_val.values.reshape(-1, 1)).ravel()
        
        # Create datasets
        train_dataset = TimeSeriesDataset(X_train_scaled, y_train_scaled, params['sequence_length'])
        val_dataset = TimeSeriesDataset(X_val_scaled, y_val_scaled, params['sequence_length'])
        
        train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=False)
        
        # Create model
        input_size = self.X_train.shape[1]
        model = LSTMRegressor(
            input_size=input_size,
            hidden_size=params['hidden_size'],
            num_layers=params['num_layers'],
            dropout=params['dropout']
        ).to(self.device)
        
        # Train
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(params['epochs']):
            # Train
            model.train()
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            # Validate
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= params['patience']:
                break
            
            # Report to Optuna for pruning
            trial.report(val_loss, epoch)
            
            if trial.should_prune():
                raise optuna.TrialPruned()
        
        # Calculate R² on validation set
        model.eval()
        val_predictions = []
        with torch.no_grad():
            for i in range(params['sequence_length'], len(X_val_scaled)):
                X_seq = X_val_scaled[i - params['sequence_length']:i]
                X_seq = torch.FloatTensor(X_seq).unsqueeze(0).to(self.device)
                pred = model(X_seq).cpu().numpy()
                val_predictions.append(pred)
        
        val_predictions = np.array(val_predictions)
        val_predictions = scaler_y.inverse_transform(val_predictions.reshape(-1, 1)).ravel()
        
        y_val_trimmed = self.y_val.values[params['sequence_length']:]
        val_predictions = val_predictions[:len(y_val_trimmed)]
        
        val_r2 = r2_score(y_val_trimmed, val_predictions)
        
        # Track best
        if val_r2 > self.best_score:
            self.best_score = val_r2
        
        trial.set_user_attr('val_r2', val_r2)
        trial.set_user_attr('best_epoch', epoch + 1)
        
        # Return negative R² (Optuna minimizes)
        return -val_r2


# ============================================
# LSTM Tuner
# ============================================

class LSTMTuner:
    """
    Hyperparameter tuner for LSTM
    """
    
    def __init__(self, target_name: str):
        self.target_name = target_name
        self.target_col = f"target_{target_name}"
        self.study = None
        self.best_params = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def load_and_prepare_data(self, splits_dir: str):
        """
        Load and prepare data for tuning
        """
        print(f"\n{'=' * 80}")
        print(f"📂 LOADING DATA: {self.target_name.upper()}")
        print(f"{'=' * 80}")
        
        splits_path = Path(splits_dir)
        
        train_df = pd.read_csv(splits_path / "train_data.csv")
        val_df = pd.read_csv(splits_path / "val_data.csv")
        test_df = pd.read_csv(splits_path / "test_data.csv")
        
        print(f"   Train: {len(train_df):,} rows")
        print(f"   Val:   {len(val_df):,} rows")
        print(f"   Test:  {len(test_df):,} rows")
        
        # Prepare features
        print(f"\n🔧 Preparing features...")
        X_train, y_train = get_feature_target_split(train_df, self.target_col, encode_categoricals=True)
        X_val, y_val = get_feature_target_split(val_df, self.target_col, encode_categoricals=True)
        X_test, y_test = get_feature_target_split(test_df, self.target_col, encode_categoricals=True)
        
        # Align columns
        for col in X_train.columns:
            if col not in X_val.columns:
                X_val[col] = 0
            if col not in X_test.columns:
                X_test[col] = 0
        
        X_val = X_val[X_train.columns]
        X_test = X_test[X_train.columns]
        
        # Impute missing
        if X_train.isna().sum().sum() > 0:
            print(f"   Imputing missing values...")
            imputer = SimpleImputer(strategy="median")
            X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
            X_val = pd.DataFrame(imputer.transform(X_val), columns=X_val.columns)
            X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)
        
        # Drop NaN targets
        X_train, y_train = drop_nan_targets(X_train, y_train, "Train")
        X_val, y_val = drop_nan_targets(X_val, y_val, "Val")
        X_test, y_test = drop_nan_targets(X_test, y_test, "Test")
        
        print(f"   ✅ Train samples: {len(X_train):,}")
        print(f"   ✅ Val samples: {len(X_val):,}")
        print(f"   ✅ Test samples: {len(X_test):,}")
        
        return X_train, y_train, X_val, y_val, X_test, y_test
    
    def tune(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        n_trials: int = 30,
        timeout: int = None,
    ):
        """
        Run hyperparameter optimization
        """
        print(f"\n{'=' * 80}")
        print(f"🔍 LSTM HYPERPARAMETER TUNING: {self.target_name.upper()}")
        print(f"{'=' * 80}")
        print(f"   Strategy: Bayesian Optimization (Optuna)")
        print(f"   Trials: {n_trials}")
        print(f"   Metric: R² (validation set)")
        print(f"   Device: {self.device}")
        print(f"{'=' * 80}\n")
        
        # Create study
        study_name = f"lstm_{self.target_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.study = optuna.create_study(
            study_name=study_name,
            direction="maximize",  # Maximize R²
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=10),
        )
        
        # Create objective
        objective = LSTMObjective(X_train, y_train, X_val, y_val, self.target_name, self.device)
        
        # Optimize
        print("Starting optimization...\n")
        self.study.optimize(
            lambda trial: -objective(trial),  # Minimize negative R²
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=True,
            callbacks=[self._log_callback],
        )
        
        # Get best parameters
        self.best_params = self.study.best_params
        
        print(f"\n{'=' * 80}")
        print(f"✅ OPTIMIZATION COMPLETE")
        print(f"{'=' * 80}")
        print(f"   Best R²: {-self.study.best_value:.4f}")
        print(f"   Best trial: #{self.study.best_trial.number}")
        print(f"   Total trials: {len(self.study.trials)}")
        
        return self.best_params
    
    def _log_callback(self, study: optuna.Study, trial: optuna.trial.FrozenTrial):
        """Callback to log progress"""
        if trial.number % 5 == 0:
            best_r2 = -study.best_value
            current_r2 = trial.user_attrs.get("val_r2", 0)
            print(f"   Trial {trial.number:3d}: R² = {current_r2:.4f} | Best: {best_r2:.4f}")
    
    def train_best_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ):
        """
        Train final model with best parameters
        """
        print(f"\n{'=' * 80}")
        print(f"🚀 TRAINING FINAL LSTM WITH BEST PARAMETERS")
        print(f"{'=' * 80}\n")
        
        # Show best parameters
        print("Best Hyperparameters:")
        for param, value in sorted(self.best_params.items()):
            print(f"   {param:20s}: {value}")
        
        # Scale data
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        
        X_train_scaled = scaler_X.fit_transform(X_train)
        X_val_scaled = scaler_X.transform(X_val)
        X_test_scaled = scaler_X.transform(X_test)
        
        y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()
        y_val_scaled = scaler_y.transform(y_val.values.reshape(-1, 1)).ravel()
        y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1)).ravel()
        
        # Create datasets
        seq_len = self.best_params['sequence_length']
        train_dataset = TimeSeriesDataset(X_train_scaled, y_train_scaled, seq_len)
        val_dataset = TimeSeriesDataset(X_val_scaled, y_val_scaled, seq_len)
        
        train_loader = DataLoader(train_dataset, batch_size=self.best_params['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.best_params['batch_size'], shuffle=False)
        
        # Create model
        model = LSTMRegressor(
            input_size=X_train.shape[1],
            hidden_size=self.best_params['hidden_size'],
            num_layers=self.best_params['num_layers'],
            dropout=self.best_params['dropout']
        ).to(self.device)
        
        # Train
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.best_params['learning_rate'])
        
        print(f"\nTraining for {self.best_params['epochs']} epochs...")
        
        for epoch in range(self.best_params['epochs']):
            model.train()
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            if (epoch + 1) % 10 == 0:
                print(f"   Epoch {epoch+1}/{self.best_params['epochs']}")
        
        # Evaluate on all splits
        print(f"\n{'=' * 80}")
        print(f"📊 MODEL PERFORMANCE")
        print(f"{'=' * 80}\n")
        
        results = {}
        
        def predict_scaled(X_scaled_data):
            model.eval()
            predictions = []
            with torch.no_grad():
                for i in range(seq_len, len(X_scaled_data)):
                    X_seq = X_scaled_data[i - seq_len:i]
                    X_seq = torch.FloatTensor(X_seq).unsqueeze(0).to(self.device)
                    pred = model(X_seq).cpu().numpy()
                    predictions.append(pred)
            return np.array(predictions)
        
        for name, X_scaled, y_data in [
            ("train", X_train_scaled, y_train),
            ("val", X_val_scaled, y_val),
            ("test", X_test_scaled, y_test)
        ]:
            pred_scaled = predict_scaled(X_scaled)
            pred = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()
            
            y_trimmed = y_data.values[seq_len:]
            pred_trimmed = pred[:len(y_trimmed)]
            
            r2 = float(r2_score(y_trimmed, pred_trimmed))
            rmse = float(np.sqrt(mean_squared_error(y_trimmed, pred_trimmed)))
            
            results[name] = {"r2": r2, "rmse": rmse}
            print(f"{name.capitalize():5s} - R²: {r2:.4f}, RMSE: {rmse:,.2f}")
        
        # Save model
        self.model = model
        self.scaler_X = scaler_X
        self.scaler_y = scaler_y
        
        return results
    
    def save_results(self, output_dir: str, results: dict):
        """
        Save tuning results
        """
        print(f"\n{'=' * 80}")
        print(f"💾 SAVING RESULTS")
        print(f"{'=' * 80}")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_file = output_path / f"lstm_{self.target_name}_tuned.pth"
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'scaler_X': self.scaler_X,
            'scaler_y': self.scaler_y,
            'best_params': self.best_params,
            'results': results,
        }, model_file)
        print(f"   ✅ Model: {model_file}")
        
        # Save metrics (same format as baseline for comparison)
        metrics_file = output_path / f"lstm_{self.target_name}_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump({
                'target': self.target_name,
                'train': results['train'],
                'val': results['val'],
                'test': results['test'],
                'hyperparameters': self.best_params,
                'n_trials': len(self.study.trials),
                'timestamp': datetime.now().isoformat(),
            }, f, indent=2)
        print(f"   ✅ Metrics: {metrics_file}")
        
        # Save tuning history
        tuning_file = output_path / f"lstm_{self.target_name}_tuning_results.json"
        trials_data = [{
            'number': t.number,
            'params': t.params,
            'val_r2': t.user_attrs.get('val_r2'),
            'state': str(t.state)
        } for t in self.study.trials]
        
        with open(tuning_file, 'w') as f:
            json.dump({
                'target': self.target_name,
                'best_params': self.best_params,
                'best_val_r2': -self.study.best_value,
                'n_trials': len(self.study.trials),
                'trials': trials_data,
            }, f, indent=2)
        print(f"   ✅ Tuning results: {tuning_file}")


def main():
    parser = argparse.ArgumentParser(description="LSTM hyperparameter tuning")
    parser.add_argument('--target', type=str, required=True)
    parser.add_argument('--trials', type=int, default=30)
    parser.add_argument('--splits-dir', type=str, default='data/splits')
    parser.add_argument('--output-dir', type=str, default='models/lstm_tuned')
    parser.add_argument('--baseline', type=str, help='Baseline metrics JSON for comparison')
    
    args = parser.parse_args()
    
    print(f"\n{'=' * 80}")
    print(f"🔍 LSTM HYPERPARAMETER TUNING")
    print(f"{'=' * 80}")
    print(f"Target: {args.target}")
    print(f"Trials: {args.trials}")
    print(f"{'=' * 80}\n")
    
    # Initialize tuner
    tuner = LSTMTuner(target_name=args.target)
    
    # Load data
    X_train, y_train, X_val, y_val, X_test, y_test = tuner.load_and_prepare_data(args.splits_dir)
    
    # Run tuning
    best_params = tuner.tune(X_train, y_train, X_val, y_val, n_trials=args.trials)
    
    # Train best model
    results = tuner.train_best_model(X_train, y_train, X_val, y_val, X_test, y_test)
    
    # Save
    tuner.save_results(args.output_dir, results)
    
    # Compare with baseline if provided
    if args.baseline and Path(args.baseline).exists():
        with open(args.baseline, 'r') as f:
            baseline = json.load(f)
        
        print(f"\n{'=' * 80}")
        print(f"📊 COMPARISON: BASELINE vs TUNED")
        print(f"{'=' * 80}\n")
        
        print(f"{'Split':<10} {'Baseline R²':>12} {'Tuned R²':>12} {'Δ':>12}")
        print("-" * 50)
        for split in ['test']:
            base_r2 = baseline[split]['r2']
            tuned_r2 = results[split]['r2']
            delta = tuned_r2 - base_r2
            print(f"{split.capitalize():<10} {base_r2:>12.4f} {tuned_r2:>12.4f} {delta:>+12.4f}")
    
    print(f"\n{'=' * 80}")
    print(f"✅ LSTM TUNING COMPLETE")
    print(f"{'=' * 80}")
    print(f"   Test R²: {results['test']['r2']:.4f}")
    print(f"   Expected: Still below XGBoost (0.48), but improved from baseline")


if __name__ == "__main__":
    main()