import json
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

# ------------------------
# CONFIG
# ------------------------
MODEL_PATH = "models/tcn/tcn_target_revenue.h5"
SCALER_PATH = "models/tcn/scaler_target_revenue.pkl"
FEATURES_PATH = "models/tcn/features_target_revenue.json"
OUTLIER_PATH = "models/tcn/outlier_info_target_revenue.json"
TARGET_NORM_PATH = "models/tcn/target_norm_target_revenue.json"
DATA_PATH = "data/splits/test_data.csv"
TARGET_COL = "target_revenue"
SEQ_LEN = 4  # number of past quarters to use

TARGET_YEAR = 2025
TARGET_QUARTER = 4
TARGET_DATE = pd.Timestamp(year=TARGET_YEAR, month=3*TARGET_QUARTER, day=31)

# ------------------------
# Load model + artifacts
# ------------------------
print("\nLoading model + artifacts...")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
scaler = joblib.load(SCALER_PATH)
with open(FEATURES_PATH, "r") as f:
    feature_cols = json.load(f)
print(f"Loaded {len(feature_cols)} features.")

# Load outlier info for clipping
with open(OUTLIER_PATH, "r") as f:
    outlier_info = json.load(f)
medians = np.array(outlier_info["median"], dtype=np.float64)
lower = np.array(outlier_info["lower"], dtype=np.float64)
upper = np.array(outlier_info["upper"], dtype=np.float64)

# ------------------------
# Load dataset
# ------------------------
df = pd.read_csv(DATA_PATH, parse_dates=["Date"])

# List all companies with enough history
companies_with_data = []
for comp, g in df.groupby("Company"):
    g_sorted = g.sort_values("Date")
    if g_sorted[g_sorted["Date"] < TARGET_DATE].shape[0] >= SEQ_LEN:
        companies_with_data.append(comp)

if not companies_with_data:
    raise ValueError(
        "No companies have enough historical quarters for prediction.")

print("\nCompanies with enough history:", companies_with_data)

# Choose company manually or automatically
COMPANY = input(
    "Enter company for prediction (or press Enter to pick first): ").strip()
if not COMPANY:
    COMPANY = companies_with_data[0]
if COMPANY not in companies_with_data:
    raise ValueError(f"Company {COMPANY} does not have enough history.")

print(f"\nSelected company for prediction: {COMPANY}")

df = df[df["Company"] == COMPANY].sort_values("Date").reset_index(drop=True)
last_seq = df[df["Date"] < TARGET_DATE].iloc[-SEQ_LEN:].copy()
print("\nUsing these last quarters:")
print(last_seq[["Date", TARGET_COL]])

if last_seq[TARGET_COL].isna().any():
    print("\n(⚠) WARNING: Some target values are missing (possible distress).")

# ------------------------
# Build input sequence
# ------------------------
X_seq = last_seq[feature_cols].values.astype(np.float32)

# Fill NaNs with medians
for i in range(len(feature_cols)):
    X_seq[:, i] = np.where(np.isnan(X_seq[:, i]), medians[i], X_seq[:, i])

# Clip to training ranges
for i in range(len(feature_cols)):
    X_seq[:, i] = np.clip(X_seq[:, i], lower[i], upper[i])

# Reshape for model
X_seq = X_seq.reshape(1, SEQ_LEN, len(feature_cols))
X_flat = X_seq.reshape(-1, len(feature_cols))
X_scaled = scaler.transform(X_flat).reshape(X_seq.shape)

# ------------------------
# Predict next quarter
# ------------------------
pred_scaled = model.predict(X_scaled, verbose=0).squeeze()

# Invert log-transform if available, with clipping
try:
    with open(TARGET_NORM_PATH, "r") as f:
        target_info = json.load(f)
    if target_info.get("log_transform", False):
        y_mean = target_info["y_mean"]
        y_std = target_info["y_std"]
        y_log_pred = pred_scaled * y_std + y_mean
        # Clip to avoid extreme outliers
        y_log_pred = np.clip(y_log_pred, y_mean - 5*y_std, y_mean + 5*y_std)
        pred_orig = np.expm1(y_log_pred)
    else:
        pred_orig = pred_scaled
except FileNotFoundError:
    print("\n(⚠) WARNING: No target normalization found. Returning scaled prediction.")
    pred_orig = pred_scaled

# ------------------------
# Output
# ------------------------
print("\n=======================================")
print(
    f" NEXT QUARTER REVENUE PREDICTION for {COMPANY} ({TARGET_YEAR} Q{TARGET_QUARTER})")
print("=======================================")
print(f"Predicted revenue: {pred_orig:,.0f}")
print("=======================================")

# Show actual value if available
true_row = df[df["Date"] == TARGET_DATE]
if not true_row.empty:
    true_val = true_row[TARGET_COL].values[0]
    print(f"Actual revenue: {true_val:,.0f}")
    pct_err = abs(pred_orig - true_val) / abs(true_val) * 100
    print(f"Prediction error: {pct_err:.2f}%")

# ------------------------
# Sanity checks
# ------------------------
last_revenues = last_seq[TARGET_COL].dropna().values
if len(last_revenues) > 0:
    min_expected = last_revenues.min() * 0.8
    max_expected = last_revenues.max() * 1.2
    if pred_orig < min_expected or pred_orig > max_expected:
        print("⚠ WARNING: Predicted revenue is unusually far from recent quarters!")
    else:
        print("✅ Prediction is within reasonable historical range.")

    last_quarter = last_revenues[-1]
    pct_change = (pred_orig - last_quarter) / last_quarter * 100
    print(f"Change vs last quarter: {pct_change:.2f}%")
    if abs(pct_change) > 50:
        print("⚠ Change is very large. Check model input/features.")

# Check for extreme feature values after scaling
for i, col in enumerate(feature_cols):
    col_min, col_max = X_scaled[:, :, i].min(), X_scaled[:, :, i].max()
    if col_min < -5 or col_max > 5:
        print(
            f"⚠ Feature {col} has extreme scaled value: {col_min:.2f}, {col_max:.2f}")

# Log-space prediction check
if target_info.get("log_transform", False):
    print(f"Prediction in log-space: {y_log_pred:.2f}")

# ------------------------
# Plot trend
# ------------------------
quarters = list(range(1, len(last_revenues)+1))
plt.plot(quarters, last_revenues, marker='o', label='History')
plt.scatter(len(last_revenues)+1, pred_orig, color='red', label='Predicted')
plt.ylabel("Revenue")
plt.xlabel("Quarter")
plt.title(f"{COMPANY} Revenue Prediction")
plt.legend()
plt.show()
