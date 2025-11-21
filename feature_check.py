import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -------------------------------
# Paths & Dashboard features
# -------------------------------
data_path = r"C:\Users\Sushmitha Sudharsan\Desktop\Mlops_Project_FinancialCrises\data\features\merged_features_clean_with_anomaly_flags_with_drift_flags.csv"

dashboard_features = ["Revenue", "Debt_to_Equity",
                      "Profit_Margin", "Stock_Price", "EPS"]

# -------------------------------
# Load dataset
# -------------------------------
df = pd.read_csv(data_path)
print("Dataset shape:", df.shape)
print("Columns:", df.columns.tolist())

# Separate features and targets
# all features except dashboard targets
X = df.drop(columns=dashboard_features)
y = df[dashboard_features]               # targets for dashboard

# Handle missing values (simple example: fill with median)
X = X.fillna(X.median())
y = y.fillna(y.median())

# -------------------------------
# Scale features for interpretability
# -------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------------------
# Train simple XGBoost for feature importance
# -------------------------------
model = XGBRegressor(n_estimators=100, max_depth=5,
                     learning_rate=0.1, random_state=42)

# Train a separate model for each dashboard target
for target in dashboard_features:
    print(f"\nTraining XGBoost for target: {target}")
    model.fit(X_scaled, y[target])

    # Predict and evaluate
    y_pred = model.predict(X_scaled)
    mae = mean_absolute_error(y[target], y_pred)
    rmse = mean_squared_error(y[target], y_pred, squared=False)
    r2 = r2_score(y[target], y_pred)

    print(f"{target} - MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")

    # Feature importance
    importances = model.feature_importances_
    feat_importance_df = pd.DataFrame({
        "feature": X.columns,
        "importance": importances
    }).sort_values(by="importance", ascending=False)

    print(f"Top 10 features for {target}:\n", feat_importance_df.head(25))
