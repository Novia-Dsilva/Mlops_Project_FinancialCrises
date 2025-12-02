# #!/usr/bin/env python3
# """
# src/models/train_tcn.py

# Train a Temporal Convolutional Network (TCN) on your quarterly data splits.

# - Robustly handles inf / -inf / NaN before scaling
# - Imputes NaNs with per-feature median from TRAIN only
# - Clips features to TRAIN 1st/99th percentiles (per-feature winsorization)
# - Log-transforms the target (log1p) and standardizes (train mean/std)
# - Saves outlier/clip/median info to model_dir/outlier_info_{target}.json
# - Saves target normalization info to model_dir/target_norm_{target}.json
# - Logs original-unit metrics (MAE/MSE/R2) to MLflow after training
# - Safely handles NaNs/infs when computing original-unit metrics (won't raise)
# """

# import argparse
# import json
# import os
# from pathlib import Path
# from typing import List, Tuple, Dict

# import joblib
# import mlflow
# import numpy as np
# import pandas as pd
# import tensorflow as tf
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# from sklearn.preprocessing import StandardScaler
# from tensorflow.keras import Input, Model
# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
# from tensorflow.keras.layers import (Activation, Add, BatchNormalization, Conv1D,
#                                      Dropout, GlobalAveragePooling1D, Dense)


# # ----------------------------
# # Utilities: data -> sequences
# # ----------------------------


# def load_split_csv(path: str) -> pd.DataFrame:
#     df = pd.read_csv(path, parse_dates=['Date'])
#     df = df.sort_values(['Company', 'Date']).reset_index(drop=True)
#     return df


# def build_sequences(df: pd.DataFrame, feature_cols: List[str], target_col: str, seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
#     """
#     Build X, y sequences from panel DataFrame grouped by Company.
#     For each company, for t >= seq_len, X[t] = features[t-seq_len:t], y[t] = target[t]
#     Returns:
#         X: np.ndarray shape (N_seq, seq_len, n_features) dtype float64
#         y: np.ndarray shape (N_seq,) dtype float32 (original units)
#     """
#     X_list, y_list = [], []
#     for comp, g in df.groupby('Company'):
#         g = g.sort_values('Date')
#         missing = [c for c in feature_cols if c not in g.columns]
#         if missing:
#             # Defensive: if group lacks feature columns, skip it
#             continue
#         features = g[feature_cols].values
#         targets = g[target_col].values
#         if len(features) <= seq_len:
#             continue
#         for end_idx in range(seq_len, len(features)):
#             start_idx = end_idx - seq_len
#             X_list.append(features[start_idx:end_idx])
#             y_list.append(targets[end_idx])
#     if not X_list:
#         return np.empty((0, seq_len, len(feature_cols))), np.empty((0,))
#     # use float64 for preprocessing stability
#     X = np.stack(X_list).astype(np.float64)
#     y = np.stack(y_list).astype(np.float32)
#     return X, y


# # ----------------------------
# # TCN building blocks
# # ----------------------------


# def residual_block(x, n_filters, kernel_size, dilation_rate, dropout_rate, name=None):
#     prev_x = x
#     conv1 = Conv1D(filters=n_filters, kernel_size=kernel_size, padding='causal',
#                    dilation_rate=dilation_rate)(x)
#     bn1 = BatchNormalization()(conv1)
#     act1 = Activation('relu')(bn1)
#     do1 = Dropout(dropout_rate)(act1)

#     conv2 = Conv1D(filters=n_filters, kernel_size=kernel_size, padding='causal',
#                    dilation_rate=dilation_rate)(do1)
#     bn2 = BatchNormalization()(conv2)
#     if prev_x.shape[-1] != n_filters:
#         prev_x = Conv1D(filters=n_filters, kernel_size=1,
#                         padding='same')(prev_x)
#     out = Add()([prev_x, bn2])
#     out = Activation('relu')(out)
#     return out


# def build_tcn_model(seq_len: int, n_features: int, n_filters: int = 64, kernel_size: int = 3,
#                     dilations: List[int] = [1, 2, 4, 8], dropout_rate: float = 0.1,
#                     dense_units: int = 64) -> tf.keras.Model:
#     inp = Input(shape=(seq_len, n_features), name='input_seq')
#     x = inp
#     for d in dilations:
#         x = residual_block(x, n_filters=n_filters, kernel_size=kernel_size,
#                            dilation_rate=d, dropout_rate=dropout_rate)
#     x = GlobalAveragePooling1D()(x)
#     x = Dense(dense_units, activation='relu')(x)
#     x = Dropout(dropout_rate)(x)
#     out = Dense(1, activation='linear', name='output')(x)
#     model = Model(inputs=inp, outputs=out)
#     return model


# # ----------------------------
# # Custom metrics
# # ----------------------------


# def r2_keras(y_true, y_pred):
#     SS_res = tf.reduce_sum(tf.square(y_true - y_pred))
#     SS_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
#     return 1 - SS_res / (SS_tot + tf.keras.backend.epsilon())


# # ----------------------------
# # Preprocessing helpers
# # ----------------------------
# def flatten_seq(X: np.ndarray) -> np.ndarray:
#     """Flatten shape (N, seq_len, n_feat) -> (N*seq_len, n_feat)"""
#     if X.size == 0:
#         return X.reshape(0, 0)
#     return X.reshape(-1, X.shape[-1])


# def reshape_back(X_flat: np.ndarray, orig_shape: tuple) -> np.ndarray:
#     if X_flat.size == 0:
#         return np.empty(orig_shape)
#     return X_flat.reshape(orig_shape)


# def compute_train_medians_and_clip(X_train_flat: np.ndarray, low_q: float = 1.0, high_q: float = 99.0) -> Dict[str, np.ndarray]:
#     """
#     Compute per-feature medians and clip bounds from TRAIN flattened matrix.
#     Returns dict with 'median', 'lower', 'upper' arrays.
#     """
#     # Ensure float64 for percentile stability
#     X = X_train_flat.astype(np.float64)
#     # Replace inf with nan to compute percentiles safely
#     X[np.isposinf(X)] = np.nan
#     X[np.isneginf(X)] = np.nan

#     medians = np.nanmedian(X, axis=0)
#     lower = np.nanpercentile(X, low_q, axis=0)
#     upper = np.nanpercentile(X, high_q, axis=0)

#     # if any lower==upper (constant column), expand small epsilon
#     eps = 1e-6
#     upper = np.where(upper == lower, lower + eps, upper)

#     return {'median': medians.tolist(), 'lower': lower.tolist(), 'upper': upper.tolist()}


# def apply_impute_and_clip(X_flat: np.ndarray, medians: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
#     """
#     Replace inf with nan, impute NaN with median, then clip between lower/upper.
#     Operates on flattened array shape (N, n_feat).
#     """
#     X = X_flat.astype(np.float64)
#     # replace inf with nan
#     X[np.isposinf(X)] = np.nan
#     X[np.isneginf(X)] = np.nan
#     # impute nan with medians
#     inds = np.where(np.isnan(X))
#     if inds[0].size > 0:
#         X[inds] = np.take(medians, inds[1])
#     # clip per-feature
#     X = np.maximum(X, lower)
#     X = np.minimum(X, upper)
#     return X


# # ----------------------------
# # Main training function
# # ----------------------------
# def train_tcn(args):
#     # PATHS
#     train_path = Path(args.data_dir) / 'train_data.csv'
#     val_path = Path(args.data_dir) / 'val_data.csv'
#     test_path = Path(args.data_dir) / 'test_data.csv'
#     model_dir = Path(args.model_dir)
#     model_dir.mkdir(parents=True, exist_ok=True)

#     print("Loading CSV splits...")
#     train_df = load_split_csv(str(train_path))
#     val_df = load_split_csv(str(val_path))
#     test_df = load_split_csv(str(test_path))

#     # Identify numeric feature columns (TRAIN only)
#     numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
#     numeric_feature_cols = [
#         c for c in numeric_cols if not c.startswith('target_')]
#     for col_to_exclude in ['Year', 'Quarter_Num', 'Quarter']:
#         if col_to_exclude in numeric_feature_cols:
#             numeric_feature_cols.remove(col_to_exclude)
#     numeric_feature_cols = sorted(numeric_feature_cols)
#     non_numeric_cols = [
#         c for c in train_df.columns if c not in numeric_cols and c not in ['Company', 'Date']]
#     if non_numeric_cols:
#         print("Non-numeric columns detected in TRAIN (excluded from features):",
#               non_numeric_cols)

#     feature_cols = numeric_feature_cols
#     print(
#         f"Using {len(feature_cols)} numeric features (example): {feature_cols[:6]} ...")

#     # -----------------------------
#     # Build sequences (raw targets)
#     # -----------------------------
#     print("Building train sequences...")
#     X_train, y_train_raw = build_sequences(
#         train_df, feature_cols, args.target, args.seq_len)
#     print("Building val sequences...")
#     X_val, y_val_raw = build_sequences(
#         val_df, feature_cols, args.target, args.seq_len)
#     print("Building test sequences...")
#     X_test, y_test_raw = build_sequences(
#         test_df, feature_cols, args.target, args.seq_len)

#     if X_train.shape[0] == 0:
#         raise ValueError(
#             "No training sequences were built. Consider lowering --seq_len or verifying data.")

#     print(
#         f"Sequences shapes -> Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

#     # -----------------------------
#     # LOG-TRANSFORM TARGET (train only stats)
#     # -----------------------------
#     # Use log1p to handle zeros and small values; then standardize using TRAIN stats
#     print("Applying log1p -> standardization to target (train stats only)...")
#     # protect against empty arrays
#     if y_train_raw.size == 0:
#         raise ValueError("No training targets found!")

#     y_train_log = np.log1p(y_train_raw.astype(np.float64))
#     y_val_log = np.log1p(y_val_raw.astype(np.float64)
#                          ) if y_val_raw.size else np.array([], dtype=np.float64)
#     y_test_log = np.log1p(y_test_raw.astype(
#         np.float64)) if y_test_raw.size else np.array([], dtype=np.float64)

#     y_mean = float(np.nanmean(y_train_log))
#     y_std = float(np.nanstd(y_train_log)) if float(
#         np.nanstd(y_train_log)) > 0 else 1.0

#     # Standardize log-targets (train mean/std)
#     y_train = ((y_train_log - y_mean) / y_std).astype(np.float32)
#     y_val = ((y_val_log - y_mean) / y_std).astype(
#         np.float32) if y_val_log.size else np.array([], dtype=np.float32)
#     y_test = ((y_test_log - y_mean) / y_std).astype(
#         np.float32) if y_test_log.size else np.array([], dtype=np.float32)

#     target_norm_info = {
#         "log_transform": True,
#         "y_mean": float(y_mean),
#         "y_std": float(y_std),
#         "notes": "To invert: y_log = y_scaled * y_std + y_mean; y_orig = expm1(y_log)"
#     }
#     print(f"Target log-mean={y_mean:.6f}, log-std={y_std:.6f}")

#     # Flatten for preprocessing (train only)
#     X_train_flat = flatten_seq(X_train)  # shape (N_train*seq_len, n_feat)
#     X_val_flat = flatten_seq(X_val)
#     X_test_flat = flatten_seq(X_test)

#     # Quick checks for inf/nan and extremes (before preprocessing)
#     print("Preprocessing checks (TRAIN):",
#           "any_inf=", np.isinf(X_train_flat).any(),
#           "any_nan=", np.isnan(X_train_flat).any(),
#           "max=", np.nanmax(X_train_flat),
#           "min=", np.nanmin(X_train_flat))

#     # Replace inf with nan so percentiles/medians ignore them
#     X_train_flat[np.isposinf(X_train_flat)] = np.nan
#     X_train_flat[np.isneginf(X_train_flat)] = np.nan
#     X_val_flat[np.isposinf(X_val_flat)] = np.nan
#     X_val_flat[np.isneginf(X_val_flat)] = np.nan
#     X_test_flat[np.isposinf(X_test_flat)] = np.nan
#     X_test_flat[np.isneginf(X_test_flat)] = np.nan

#     # Compute medians and clip thresholds from TRAIN (robust winsorization)
#     outlier_info = compute_train_medians_and_clip(
#         X_train_flat, low_q=1.0, high_q=99.0)
#     medians = np.array(outlier_info['median'], dtype=np.float64)
#     lower = np.array(outlier_info['lower'], dtype=np.float64)
#     upper = np.array(outlier_info['upper'], dtype=np.float64)

#     # Apply imputation (median) and clipping (train bounds) to train/val/test flattened arrays
#     X_train_flat_proc = apply_impute_and_clip(
#         X_train_flat, medians, lower, upper)
#     X_val_flat_proc = apply_impute_and_clip(
#         X_val_flat, medians, lower, upper) if X_val_flat.size else X_val_flat
#     X_test_flat_proc = apply_impute_and_clip(
#         X_test_flat, medians, lower, upper) if X_test_flat.size else X_test_flat

#     # Final check
#     print("After impute+clip (TRAIN): any_inf=", np.isinf(X_train_flat_proc).any(), "any_nan=", np.isnan(X_train_flat_proc).any(),
#           "max=", np.nanmax(X_train_flat_proc), "min=", np.nanmin(X_train_flat_proc))

#     # Fit scaler on processed TRAIN flattened array (float64 -> scaler)
#     scaler = StandardScaler()
#     scaler.fit(X_train_flat_proc)

#     # Transform flattened arrays and reshape back to sequences
#     X_train_flat_scaled = scaler.transform(X_train_flat_proc)
#     X_val_flat_scaled = scaler.transform(
#         X_val_flat_proc) if X_val_flat_proc.size else X_val_flat_proc
#     X_test_flat_scaled = scaler.transform(
#         X_test_flat_proc) if X_test_flat_proc.size else X_test_flat_proc

#     X_train = reshape_back(
#         X_train_flat_scaled.astype(np.float32), X_train.shape)
#     X_val = reshape_back(X_val_flat_scaled.astype(
#         np.float32), X_val.shape) if X_val_flat_scaled.size else X_val
#     X_test = reshape_back(X_test_flat_scaled.astype(
#         np.float32), X_test.shape) if X_test_flat_scaled.size else X_test

#     n_feat = X_train.shape[-1]

#     # Save outlier info for production use
#     outlier_file = model_dir / f'outlier_info_{args.target}.json'
#     with open(outlier_file, 'w') as f:
#         json.dump(outlier_info, f, indent=2)
#     print("Saved outlier info to:", outlier_file)

#     # Build model
#     tf.keras.backend.clear_session()
#     model = build_tcn_model(seq_len=args.seq_len, n_features=n_feat,
#                             n_filters=args.n_filters, kernel_size=args.kernel_size,
#                             dilations=args.dilations, dropout_rate=args.dropout, dense_units=args.dense_units)

#     optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)
#     model.compile(optimizer=optimizer, loss='mse', metrics=['mae', r2_keras])
#     model.summary()

#     # Callbacks
#     model_file = model_dir / f"tcn_{args.target}.h5"
#     ckpt = ModelCheckpoint(filepath=str(model_file),
#                            save_best_only=True, monitor='val_loss', verbose=1)
#     es = EarlyStopping(monitor='val_loss', patience=args.patience,
#                        verbose=1, restore_best_weights=True)
#     reduce_lr = ReduceLROnPlateau(
#         monitor='val_loss', factor=0.5, patience=5, verbose=1)

#     # MLflow logging
#     mlflow.set_experiment(args.mlflow_experiment)
#     with mlflow.start_run(run_name=f"tcn_{args.target}"):
#         mlflow.log_params({
#             'target': args.target,
#             'seq_len': args.seq_len,
#             'n_filters': args.n_filters,
#             'kernel_size': args.kernel_size,
#             'dilations': args.dilations,
#             'dropout': args.dropout,
#             'dense_units': args.dense_units,
#             'batch_size': args.batch_size,
#             'epochs': args.epochs,
#             'lr': args.lr,
#             'target_log_transform': True
#         })

#         # Fit (train on standardized log-target)
#         history = model.fit(
#             X_train, y_train,
#             validation_data=(X_val, y_val) if y_val.size else None,
#             epochs=args.epochs,
#             batch_size=args.batch_size,
#             callbacks=[ckpt, es, reduce_lr],
#             verbose=2
#         )

#         # Evaluate in scaled (training) space
#         val_metrics_scaled = model.evaluate(
#             X_val, y_val, verbose=0) if y_val.size else []
#         test_metrics_scaled = model.evaluate(
#             X_test, y_test, verbose=0) if y_test.size else []

#         metric_names = ['loss', 'mae', 'r2_keras']
#         if val_metrics_scaled:
#             mlflow.log_metrics({f"val_{n}": float(v)
#                                for n, v in zip(metric_names, val_metrics_scaled)})
#         if test_metrics_scaled:
#             mlflow.log_metrics({f"test_{n}": float(v)
#                                for n, v in zip(metric_names, test_metrics_scaled)})

#         # Save scaler & feature list
#         scaler_path = model_dir / f"scaler_{args.target}.pkl"
#         joblib.dump(scaler, scaler_path)
#         features_path = model_dir / f"features_{args.target}.json"
#         with open(features_path, 'w') as f:
#             json.dump(feature_cols, f)

#         # Save target normalization info
#         target_norm_path = model_dir / f"target_norm_{args.target}.json"
#         with open(target_norm_path, 'w') as f:
#             json.dump(target_norm_info, f, indent=2)

#         # Log artifacts
#         mlflow.log_artifact(str(model_file))
#         mlflow.log_artifact(str(scaler_path))
#         mlflow.log_artifact(str(features_path))
#         mlflow.log_artifact(str(outlier_file))
#         mlflow.log_artifact(str(target_norm_path))

#     # ----------------------------
#     # Post-train: compute original-unit predictions & metrics
#     # ----------------------------
#     # Load best model (checkpoint saved best by val_loss)
#     if model_file.exists():
#         best_model = tf.keras.models.load_model(
#             str(model_file), custom_objects={'r2_keras': r2_keras})

#         def invert_target(y_scaled: np.ndarray) -> np.ndarray:
#             """Inverse: y_scaled -> y_log -> y_orig"""
#             y_log = (y_scaled * y_std) + y_mean
#             return np.expm1(y_log)

#         # Helper to safely compute and log metrics in original units
#         def safe_metrics_and_log(y_true_log, y_pred_scaled, prefix: str):
#             """
#             y_true_log: array in log-space (log1p of original target)
#             y_pred_scaled: model predictions in standardized-log space
#             prefix: 'val' or 'test'
#             """
#             if y_pred_scaled is None or y_pred_scaled.size == 0 or y_true_log is None or y_true_log.size == 0:
#                 print(f"No data for {prefix} original-unit metrics.")
#                 return

#             # inverse-transform predictions and truths to original units
#             y_pred_scaled = np.asarray(y_pred_scaled).ravel()
#             y_true_log = np.asarray(y_true_log).ravel()

#             # predictions -> original
#             try:
#                 y_pred_orig = invert_target(y_pred_scaled)
#             except Exception as e:
#                 print(
#                     f"Error during inverse transform for {prefix} preds: {e}")
#                 y_pred_orig = np.full_like(
#                     y_pred_scaled, np.nan, dtype=np.float64)

#             # truths (already in log space): y_true_orig = expm1(y_true_log)
#             try:
#                 y_true_orig = np.expm1(y_true_log)
#             except Exception as e:
#                 print(
#                     f"Error during inverse transform for {prefix} truth: {e}")
#                 y_true_orig = np.full_like(
#                     y_true_log, np.nan, dtype=np.float64)

#             # replace infs with nan and create mask for valid pairs
#             y_pred_orig[~np.isfinite(y_pred_orig)] = np.nan
#             y_true_orig[~np.isfinite(y_true_orig)] = np.nan

#             print(f"{prefix.upper()} original-unit stats before filtering: truth_nan={np.isnan(y_true_orig).sum()}, pred_nan={np.isnan(y_pred_orig).sum()}, total={y_true_orig.size}")

#             valid = (~np.isnan(y_true_orig)) & (~np.isnan(y_pred_orig)
#                                                 ) & np.isfinite(y_true_orig) & np.isfinite(y_pred_orig)

#             if valid.sum() == 0:
#                 print(
#                     f"No valid {prefix} pairs to compute original-unit metrics (after filtering). Skipping.")
#                 return

#             y_t = y_true_orig[valid]
#             y_p = y_pred_orig[valid]

#             mse = float(mean_squared_error(y_t, y_p))
#             mae = float(mean_absolute_error(y_t, y_p))
#             r2 = float(r2_score(y_t, y_p))

#             print(
#                 f"{prefix.capitalize()} (orig units) -> count={valid.sum()}, MSE={mse:.4f}, MAE={mae:.4f}, R2={r2:.4f}")

#             mlflow.log_metrics({
#                 f'{prefix}_mse_orig': mse,
#                 f'{prefix}_mae_orig': mae,
#                 f'{prefix}_r2_orig': r2
#             })

#         # VAL
#         if y_val.size:
#             y_val_pred_scaled = best_model.predict(X_val, verbose=0).squeeze()
#             safe_metrics_and_log(y_val_log, y_val_pred_scaled, 'val')

#         # TEST
#         if y_test.size:
#             y_test_pred_scaled = best_model.predict(
#                 X_test, verbose=0).squeeze()
#             safe_metrics_and_log(y_test_log, y_test_pred_scaled, 'test')

#     # Final messages
#     print("Training complete. Best model saved to:", model_file)
#     print("Scaler saved to:", scaler_path)
#     print("Outlier info saved to:", outlier_file)
#     print("Target normalization saved to:", target_norm_path)


# # ----------------------------
# # CLI entrypoint
# # ----------------------------
# def parse_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--data_dir', type=str, default='data/splits',
#                         help='Directory with train/val/test CSVs')
#     parser.add_argument('--target', type=str, required=True,
#                         help='Target column, e.g., target_revenue')
#     parser.add_argument('--seq_len', type=int, default=8,
#                         help='Sequence length (quarters)')
#     parser.add_argument('--batch_size', type=int, default=64)
#     parser.add_argument('--epochs', type=int, default=100)
#     parser.add_argument('--patience', type=int, default=10)
#     parser.add_argument('--model_dir', type=str, default='models/tcn')
#     parser.add_argument('--mlflow_experiment', type=str,
#                         default='tcn_experiment')
#     parser.add_argument('--n_filters', type=int, default=64)
#     parser.add_argument('--kernel_size', type=int, default=3)
#     parser.add_argument('--dropout', type=float, default=0.1)
#     parser.add_argument('--dense_units', type=int, default=64)
#     parser.add_argument('--lr', type=float, default=1e-3)
#     parser.add_argument('--dilations', type=str, default='1,2,4,8',
#                         help='Comma-separated dilation rates')
#     args = parser.parse_args()
#     args.dilations = [int(x) for x in args.dilations.split(',') if x.strip()]
#     return args


# if __name__ == '__main__':
#     args = parse_args()
#     train_tcn(args)

#!/usr/bin/env python3
"""
src/models/train_tcn.py

Train a Temporal Convolutional Network (TCN) on your quarterly data splits.

Modifications for percent-change mode (QoQ / pct-change):
- Converts ALL numeric columns (features + target_*) to quarter-over-quarter percent-change
  grouped by Company before any preprocessing. This makes the model predict growth (% change)
  rather than raw levels.
- First-period pct_change per company is filled with 0.0 to preserve rows (CONFIG: KEEP_FIRST_AS_ZERO).
- Targets are NOT log-transformed. They are standardized (train mean/std) in pct-change units.
- Original-unit metrics (MAE/MSE/R2) are reported in pct-change units (e.g., 0.05 == 5%).

The rest of the original script's behavior (imputation with TRAIN medians, per-feature winsorization,
scaler fit on TRAIN, MLflow logging, checkpoints, callbacks, messages) is preserved.
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Tuple, Dict

import joblib
import mlflow
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import Input, Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import (Activation, Add, BatchNormalization, Conv1D,
                                     Dropout, GlobalAveragePooling1D, Dense)

# -----------------------
# CONFIG: pct-change behavior
# -----------------------
# If True, replace numeric columns with quarter-over-quarter percent-change grouped by Company.
APPLY_PERCENT_CHANGE = True
# If True, keep first pct_change NaN as 0.0 (preserve rows). If False, first rows will become NaN and
# those NaNs will be imputed later by medians (or you can change to drop).
KEEP_FIRST_AS_ZERO = True
# -----------------------

# ----------------------------
# Utilities: data -> sequences
# ----------------------------


def load_split_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=['Date'])
    # ensure sorted per company/time for pct_change grouping
    df = df.sort_values(['Company', 'Date']).reset_index(drop=True)
    return df


def pct_change_grouped(df: pd.DataFrame, by_col: str = "Company", numeric_only: bool = True) -> pd.DataFrame:
    """
    Compute period-over-period percent change per group (Company).
    Replaces numeric columns with pct_change. Non-numeric columns are left alone.
    If KEEP_FIRST_AS_ZERO is True, fills the first NaN (from pct_change) with 0.0.
    """
    out = df.copy()
    if numeric_only:
        numeric_cols = out.select_dtypes(include=[np.number]).columns.tolist()
    else:
        numeric_cols = out.columns.tolist()

    # compute pct_change per company for numeric columns
    out[numeric_cols] = out.groupby(by_col)[numeric_cols].pct_change()

    if KEEP_FIRST_AS_ZERO:
        out[numeric_cols] = out[numeric_cols].fillna(0.0)
    return out


def build_sequences(df: pd.DataFrame, feature_cols: List[str], target_col: str, seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build X, y sequences from panel DataFrame grouped by Company.
    For each company, for t >= seq_len, X[t] = features[t-seq_len:t], y[t] = target[t]
    Returns:
        X: np.ndarray shape (N_seq, seq_len, n_features) dtype float64
        y: np.ndarray shape (N_seq,) dtype float32 (pct-change units)
    """
    X_list, y_list = [], []
    for comp, g in df.groupby('Company'):
        g = g.sort_values('Date')
        missing = [c for c in feature_cols if c not in g.columns]
        if missing:
            # Defensive: if group lacks feature columns, skip it
            continue
        features = g[feature_cols].values
        targets = g[target_col].values
        if len(features) <= seq_len:
            continue
        for end_idx in range(seq_len, len(features)):
            start_idx = end_idx - seq_len
            X_list.append(features[start_idx:end_idx])
            y_list.append(targets[end_idx])
    if not X_list:
        return np.empty((0, seq_len, len(feature_cols))), np.empty((0,))
    # use float64 for preprocessing stability
    X = np.stack(X_list).astype(np.float64)
    y = np.stack(y_list).astype(np.float32)
    return X, y


# ----------------------------
# TCN building blocks
# ----------------------------


def residual_block(x, n_filters, kernel_size, dilation_rate, dropout_rate, name=None):
    prev_x = x
    conv1 = Conv1D(filters=n_filters, kernel_size=kernel_size, padding='causal',
                   dilation_rate=dilation_rate)(x)
    bn1 = BatchNormalization()(conv1)
    act1 = Activation('relu')(bn1)
    do1 = Dropout(dropout_rate)(act1)

    conv2 = Conv1D(filters=n_filters, kernel_size=kernel_size, padding='causal',
                   dilation_rate=dilation_rate)(do1)
    bn2 = BatchNormalization()(conv2)
    if prev_x.shape[-1] != n_filters:
        prev_x = Conv1D(filters=n_filters, kernel_size=1,
                        padding='same')(prev_x)
    out = Add()([prev_x, bn2])
    out = Activation('relu')(out)
    return out


def build_tcn_model(seq_len: int, n_features: int, n_filters: int = 64, kernel_size: int = 3,
                    dilations: List[int] = [1, 2, 4, 8], dropout_rate: float = 0.1,
                    dense_units: int = 64) -> tf.keras.Model:
    inp = Input(shape=(seq_len, n_features), name='input_seq')
    x = inp
    for d in dilations:
        x = residual_block(x, n_filters=n_filters, kernel_size=kernel_size,
                           dilation_rate=d, dropout_rate=dropout_rate)
    x = GlobalAveragePooling1D()(x)
    x = Dense(dense_units, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    out = Dense(1, activation='linear', name='output')(x)
    model = Model(inputs=inp, outputs=out)
    return model


# ----------------------------
# Custom metrics
# ----------------------------


def r2_keras(y_true, y_pred):
    SS_res = tf.reduce_sum(tf.square(y_true - y_pred))
    SS_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
    return 1 - SS_res / (SS_tot + tf.keras.backend.epsilon())


# ----------------------------
# Preprocessing helpers
# ----------------------------
def flatten_seq(X: np.ndarray) -> np.ndarray:
    """Flatten shape (N, seq_len, n_feat) -> (N*seq_len, n_feat)"""
    if X.size == 0:
        return X.reshape(0, 0)
    return X.reshape(-1, X.shape[-1])


def reshape_back(X_flat: np.ndarray, orig_shape: tuple) -> np.ndarray:
    if X_flat.size == 0:
        return np.empty(orig_shape)
    return X_flat.reshape(orig_shape)


def compute_train_medians_and_clip(X_train_flat: np.ndarray, low_q: float = 1.0, high_q: float = 99.0) -> Dict[str, np.ndarray]:
    """
    Compute per-feature medians and clip bounds from TRAIN flattened matrix.
    Returns dict with 'median', 'lower', 'upper' arrays.
    """
    # Ensure float64 for percentile stability
    X = X_train_flat.astype(np.float64)
    # Replace inf with nan to compute percentiles safely
    X[np.isposinf(X)] = np.nan
    X[np.isneginf(X)] = np.nan

    medians = np.nanmedian(X, axis=0)
    lower = np.nanpercentile(X, low_q, axis=0)
    upper = np.nanpercentile(X, high_q, axis=0)

    # if any lower==upper (constant column), expand small epsilon
    eps = 1e-6
    upper = np.where(upper == lower, lower + eps, upper)

    return {'median': medians.tolist(), 'lower': lower.tolist(), 'upper': upper.tolist()}


def apply_impute_and_clip(X_flat: np.ndarray, medians: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """
    Replace inf with nan, impute NaN with median, then clip between lower/upper.
    Operates on flattened array shape (N, n_feat).
    """
    X = X_flat.astype(np.float64)
    # replace inf with nan
    X[np.isposinf(X)] = np.nan
    X[np.isneginf(X)] = np.nan
    # impute nan with medians
    inds = np.where(np.isnan(X))
    if inds[0].size > 0:
        X[inds] = np.take(medians, inds[1])
    # clip per-feature
    X = np.maximum(X, lower)
    X = np.minimum(X, upper)
    return X


# ----------------------------
# Main training function
# ----------------------------
def train_tcn(args):
    # PATHS
    train_path = Path(args.data_dir) / 'train_data.csv'
    val_path = Path(args.data_dir) / 'val_data.csv'
    test_path = Path(args.data_dir) / 'test_data.csv'
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    print("Loading CSV splits...")
    train_df = load_split_csv(str(train_path))
    val_df = load_split_csv(str(val_path))
    test_df = load_split_csv(str(test_path))

    # -----------------------
    # Optionally convert to pct-change (QoQ) for all numeric cols
    # -----------------------
    if APPLY_PERCENT_CHANGE:
        print("Converting numeric features and targets to quarter-over-quarter percent-change (grouped by Company)...")
        # Ensure Date ordering per company
        train_df = train_df.sort_values(
            ['Company', 'Date']).reset_index(drop=True)
        val_df = val_df.sort_values(['Company', 'Date']).reset_index(drop=True)
        test_df = test_df.sort_values(
            ['Company', 'Date']).reset_index(drop=True)

        # apply percent change grouped by Company
        train_df = pct_change_grouped(
            train_df, by_col='Company', numeric_only=True)
        val_df = pct_change_grouped(
            val_df, by_col='Company', numeric_only=True)
        test_df = pct_change_grouped(
            test_df, by_col='Company', numeric_only=True)

        print("   ✅ Percent-change conversion applied.")
        # report a quick check
        print(
            f"   Post-conversion sizes -> Train: {len(train_df):,}, Val: {len(val_df):,}, Test: {len(test_df):,}")

    # Identify numeric feature columns (TRAIN only)
    numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_feature_cols = [
        c for c in numeric_cols if not c.startswith('target_')]
    for col_to_exclude in ['Year', 'Quarter_Num', 'Quarter']:
        if col_to_exclude in numeric_feature_cols:
            numeric_feature_cols.remove(col_to_exclude)
    numeric_feature_cols = sorted(numeric_feature_cols)
    non_numeric_cols = [
        c for c in train_df.columns if c not in numeric_cols and c not in ['Company', 'Date']]
    if non_numeric_cols:
        print("Non-numeric columns detected in TRAIN (excluded from features):",
              non_numeric_cols)

    feature_cols = numeric_feature_cols
    print(
        f"Using {len(feature_cols)} numeric features (example): {feature_cols[:6]} ...")

    # -----------------------------
    # Build sequences (pct-change targets)
    # -----------------------------
    print("Building train sequences...")
    X_train, y_train_raw = build_sequences(
        train_df, feature_cols, args.target, args.seq_len)
    print("Building val sequences...")
    X_val, y_val_raw = build_sequences(
        val_df, feature_cols, args.target, args.seq_len)
    print("Building test sequences...")
    X_test, y_test_raw = build_sequences(
        test_df, feature_cols, args.target, args.seq_len)

    if X_train.shape[0] == 0:
        raise ValueError(
            "No training sequences were built. Consider lowering --seq_len or verifying data.")

    print(
        f"Sequences shapes -> Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    # -----------------------------
    # TARGET: STANDARDIZE pct-change target (train only stats)
    # -----------------------------
    print("Standardizing pct-change target (train stats only)...")
    if y_train_raw.size == 0:
        raise ValueError("No training targets found!")

    # compute train mean/std in pct-change units (no log transform)
    y_mean = float(np.nanmean(y_train_raw.astype(np.float64)))
    y_std = float(np.nanstd(y_train_raw.astype(np.float64))) if float(
        np.nanstd(y_train_raw.astype(np.float64))) > 0 else 1.0

    # Standardize targets (train mean/std)
    y_train = ((y_train_raw.astype(np.float64) -
               y_mean) / y_std).astype(np.float32)
    y_val = ((y_val_raw.astype(np.float64) - y_mean) / y_std).astype(
        np.float32) if y_val_raw.size else np.array([], dtype=np.float32)
    y_test = ((y_test_raw.astype(np.float64) - y_mean) / y_std).astype(
        np.float32) if y_test_raw.size else np.array([], dtype=np.float32)

    target_norm_info = {
        "log_transform": False,
        "pct_change": True,
        "y_mean": float(y_mean),
        "y_std": float(y_std),
        "notes": "To invert: y_orig_pct = y_scaled * y_std + y_mean (e.g., 0.05 == 5% QoQ)"
    }
    print(f"Target mean={y_mean:.6f}, target std={y_std:.6f}")

    # Flatten for preprocessing (train only)
    X_train_flat = flatten_seq(X_train)  # shape (N_train*seq_len, n_feat)
    X_val_flat = flatten_seq(X_val)
    X_test_flat = flatten_seq(X_test)

    # Quick checks for inf/nan and extremes (before preprocessing)
    print("Preprocessing checks (TRAIN):",
          "any_inf=", np.isinf(X_train_flat).any(),
          "any_nan=", np.isnan(X_train_flat).any(),
          "max=", np.nanmax(X_train_flat),
          "min=", np.nanmin(X_train_flat))

    # Replace inf with nan so percentiles/medians ignore them
    X_train_flat[np.isposinf(X_train_flat)] = np.nan
    X_train_flat[np.isneginf(X_train_flat)] = np.nan
    X_val_flat[np.isposinf(X_val_flat)] = np.nan
    X_val_flat[np.isneginf(X_val_flat)] = np.nan
    X_test_flat[np.isposinf(X_test_flat)] = np.nan
    X_test_flat[np.isneginf(X_test_flat)] = np.nan

    # Compute medians and clip thresholds from TRAIN (robust winsorization)
    outlier_info = compute_train_medians_and_clip(
        X_train_flat, low_q=1.0, high_q=99.0)
    medians = np.array(outlier_info['median'], dtype=np.float64)
    lower = np.array(outlier_info['lower'], dtype=np.float64)
    upper = np.array(outlier_info['upper'], dtype=np.float64)

    # Apply imputation (median) and clipping (train bounds) to train/val/test flattened arrays
    X_train_flat_proc = apply_impute_and_clip(
        X_train_flat, medians, lower, upper)
    X_val_flat_proc = apply_impute_and_clip(
        X_val_flat, medians, lower, upper) if X_val_flat.size else X_val_flat
    X_test_flat_proc = apply_impute_and_clip(
        X_test_flat, medians, lower, upper) if X_test_flat.size else X_test_flat

    # Final check
    print("After impute+clip (TRAIN): any_inf=", np.isinf(X_train_flat_proc).any(), "any_nan=", np.isnan(X_train_flat_proc).any(),
          "max=", np.nanmax(X_train_flat_proc), "min=", np.nanmin(X_train_flat_proc))

    # Fit scaler on processed TRAIN flattened array (float64 -> scaler)
    scaler = StandardScaler()
    scaler.fit(X_train_flat_proc)

    # Transform flattened arrays and reshape back to sequences
    X_train_flat_scaled = scaler.transform(X_train_flat_proc)
    X_val_flat_scaled = scaler.transform(
        X_val_flat_proc) if X_val_flat_proc.size else X_val_flat_proc
    X_test_flat_scaled = scaler.transform(
        X_test_flat_proc) if X_test_flat_proc.size else X_test_flat_proc

    X_train = reshape_back(
        X_train_flat_scaled.astype(np.float32), X_train.shape)
    X_val = reshape_back(X_val_flat_scaled.astype(
        np.float32), X_val.shape) if X_val_flat_scaled.size else X_val
    X_test = reshape_back(X_test_flat_scaled.astype(
        np.float32), X_test.shape) if X_test_flat_scaled.size else X_test

    n_feat = X_train.shape[-1]

    # Save outlier info for production use
    outlier_file = model_dir / f'outlier_info_{args.target}.json'
    with open(outlier_file, 'w') as f:
        json.dump(outlier_info, f, indent=2)
    print("Saved outlier info to:", outlier_file)

    # Build model
    tf.keras.backend.clear_session()
    model = build_tcn_model(seq_len=args.seq_len, n_features=n_feat,
                            n_filters=args.n_filters, kernel_size=args.kernel_size,
                            dilations=args.dilations, dropout_rate=args.dropout, dense_units=args.dense_units)

    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae', r2_keras])
    model.summary()

    # Callbacks
    model_file = model_dir / f"tcn_{args.target}.h5"
    ckpt = ModelCheckpoint(filepath=str(model_file),
                           save_best_only=True, monitor='val_loss', verbose=1)
    es = EarlyStopping(monitor='val_loss', patience=args.patience,
                       verbose=1, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=5, verbose=1)

    # MLflow logging
    mlflow.set_experiment(args.mlflow_experiment)
    with mlflow.start_run(run_name=f"tcn_{args.target}"):
        mlflow.log_params({
            'target': args.target,
            'seq_len': args.seq_len,
            'n_filters': args.n_filters,
            'kernel_size': args.kernel_size,
            'dilations': args.dilations,
            'dropout': args.dropout,
            'dense_units': args.dense_units,
            'batch_size': args.batch_size,
            'epochs': args.epochs,
            'lr': args.lr,
            'target_pct_change': True
        })

        # Fit (train on standardized pct-change target)
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val) if y_val.size else None,
            epochs=args.epochs,
            batch_size=args.batch_size,
            callbacks=[ckpt, es, reduce_lr],
            verbose=2
        )

        # Evaluate in scaled (training) space
        val_metrics_scaled = model.evaluate(
            X_val, y_val, verbose=0) if y_val.size else []
        test_metrics_scaled = model.evaluate(
            X_test, y_test, verbose=0) if y_test.size else []

        metric_names = ['loss', 'mae', 'r2_keras']
        if val_metrics_scaled:
            mlflow.log_metrics({f"val_{n}": float(v)
                               for n, v in zip(metric_names, val_metrics_scaled)})
        if test_metrics_scaled:
            mlflow.log_metrics({f"test_{n}": float(v)
                               for n, v in zip(metric_names, test_metrics_scaled)})

        # Save scaler & feature list
        scaler_path = model_dir / f"scaler_{args.target}.pkl"
        joblib.dump(scaler, scaler_path)
        features_path = model_dir / f"features_{args.target}.json"
        with open(features_path, 'w') as f:
            json.dump(feature_cols, f)

        # Save target normalization info
        target_norm_path = model_dir / f"target_norm_{args.target}.json"
        with open(target_norm_path, 'w') as f:
            json.dump(target_norm_info, f, indent=2)

        # Log artifacts
        # NOTE: Model may not exist if training failed early; we still attempt to log
        try:
            mlflow.log_artifact(str(model_file))
        except Exception:
            pass
        mlflow.log_artifact(str(scaler_path))
        mlflow.log_artifact(str(features_path))
        mlflow.log_artifact(str(outlier_file))
        mlflow.log_artifact(str(target_norm_path))

    # ----------------------------
    # Post-train: compute pct-change predictions & metrics
    # ----------------------------
    # Load best model (checkpoint saved best by val_loss)
    if model_file.exists():
        best_model = tf.keras.models.load_model(
            str(model_file), custom_objects={'r2_keras': r2_keras})

        def invert_target(y_scaled: np.ndarray) -> np.ndarray:
            """Inverse: y_scaled -> y_orig_pct (pct-change units)"""
            return (y_scaled * y_std) + y_mean

        # Helper to safely compute and log metrics in pct-change units
        def safe_metrics_and_log(y_true_scaled_log, y_pred_scaled, prefix: str):
            """
            y_true_scaled_log: array in scaled pct-change space (we computed y_val_raw -> scaled)
            y_pred_scaled: model predictions in standardized space
            prefix: 'val' or 'test'
            """
            if y_pred_scaled is None or y_pred_scaled.size == 0 or y_true_scaled_log is None or y_true_scaled_log.size == 0:
                print(f"No data for {prefix} original-unit metrics.")
                return

            # predictions -> original pct-change units
            y_pred_scaled = np.asarray(y_pred_scaled).ravel()
            y_true_scaled_log = np.asarray(y_true_scaled_log).ravel()

            try:
                y_pred_orig = invert_target(y_pred_scaled)
            except Exception as e:
                print(
                    f"Error during inverse transform for {prefix} preds: {e}")
                y_pred_orig = np.full_like(
                    y_pred_scaled, np.nan, dtype=np.float64)

            try:
                y_true_orig = invert_target(y_true_scaled_log)
            except Exception as e:
                print(
                    f"Error during inverse transform for {prefix} truth: {e}")
                y_true_orig = np.full_like(
                    y_true_scaled_log, np.nan, dtype=np.float64)

            # replace infs with nan and create mask for valid pairs
            y_pred_orig[~np.isfinite(y_pred_orig)] = np.nan
            y_true_orig[~np.isfinite(y_true_orig)] = np.nan

            print(f"{prefix.upper()} original-unit stats before filtering: truth_nan={np.isnan(y_true_orig).sum()}, pred_nan={np.isnan(y_pred_orig).sum()}, total={y_true_orig.size}")

            valid = (~np.isnan(y_true_orig)) & (~np.isnan(y_pred_orig)
                                                ) & np.isfinite(y_true_orig) & np.isfinite(y_pred_orig)

            if valid.sum() == 0:
                print(
                    f"No valid {prefix} pairs to compute original-unit metrics (after filtering). Skipping.")
                return

            y_t = y_true_orig[valid]
            y_p = y_pred_orig[valid]

            mse = float(mean_squared_error(y_t, y_p))
            mae = float(mean_absolute_error(y_t, y_p))
            r2 = float(r2_score(y_t, y_p))

            print(f"{prefix.capitalize()} (pct-growth units) -> count={valid.sum()}, MSE={mse:.6f}, MAE={mae:.6f}, R2={r2:.4f}")

            mlflow.log_metrics({
                f'{prefix}_mse_pct': mse,
                f'{prefix}_mae_pct': mae,
                f'{prefix}_r2_pct': r2
            })

        # VAL
        if y_val.size:
            y_val_pred_scaled = best_model.predict(X_val, verbose=0).squeeze()
            safe_metrics_and_log(
                ((y_val * 1.0).astype(np.float64)), y_val_pred_scaled, 'val')

        # TEST
        if y_test.size:
            y_test_pred_scaled = best_model.predict(
                X_test, verbose=0).squeeze()
            safe_metrics_and_log(
                ((y_test * 1.0).astype(np.float64)), y_test_pred_scaled, 'test')

    # Final messages
    print("Training complete. Best model saved to:", model_file)
    print("Scaler saved to:", scaler_path)
    print("Outlier info saved to:", outlier_file)
    print("Target normalization saved to:", target_norm_path)


# ----------------------------
# CLI entrypoint
# ----------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/splits',
                        help='Directory with train/val/test CSVs')
    parser.add_argument('--target', type=str, required=True,
                        help='Target column, e.g., target_revenue (will be treated as pct-change)')
    parser.add_argument('--seq_len', type=int, default=8,
                        help='Sequence length (quarters)')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--model_dir', type=str, default='models/tcn')
    parser.add_argument('--mlflow_experiment', type=str,
                        default='tcn_experiment')
    parser.add_argument('--n_filters', type=int, default=64)
    parser.add_argument('--kernel_size', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--dense_units', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--dilations', type=str, default='1,2,4,8',
                        help='Comma-separated dilation rates')
    args = parser.parse_args()
    args.dilations = [int(x) for x in args.dilations.split(',') if x.strip()]
    return args


if __name__ == '__main__':
    args = parse_args()
    train_tcn(args)
