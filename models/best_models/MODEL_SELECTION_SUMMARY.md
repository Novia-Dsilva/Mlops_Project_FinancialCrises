# Unified Model Training & Selection Report

**Generated:** 2025-12-08 15:47:07

## Overview

This report summarizes the unified model training pipeline where all 4 model types (XGBoost, XGBoost-Tuned, LightGBM, LightGBM-Tuned) were trained for each target, and the best model was automatically selected based on Test R² and generalization quality.

## Selection Criteria

1. **Primary:** Test R² (higher is better)
2. **Quality Check:** Overfitting gap < 30% (train-test R² difference)
3. **Tiebreaker:** Test RMSE (lower is better)

## Selected Models Summary

| Target | Selected Model | Test R² | Test RMSE | Notes |
|--------|---------------|---------|-----------|-------|
| revenue | LightGBM-Tuned | 0.9424 | 4,633,471,672.96 | ✅ Best |
| eps | LightGBM-Tuned | 0.7055 | 1.41 | ✓ Good |
| debt_equity | LightGBM-Tuned | 0.6675 | 3.41 | ✅ Best |
| profit_margin | LightGBM | 0.4965 | 10.29 | ⚠️ Check |
| stock_return | LightGBM-Tuned | 0.0575 | 0.14 | ⚠️ Check |
| **AVERAGE** | - | **0.5739** | - | - |

## Detailed Selection Reasoning

### REVENUE

**Selected Model:** LightGBM-Tuned

**Reasoning:** Selected LightGBM-Tuned with Test R²=0.9424. Excellent generalization.

**All Models Comparison:**

| Model | Test R² | Overfitting % |
|-------|---------|---------------|
| LightGBM-Tuned | 0.9424 | 5.7% |
| XGBoost-Tuned | 0.9375 | 6.2% |
| XGBoost | 0.9357 | 6.4% |
| LightGBM | 0.9233 | 7.0% |

---

### EPS

**Selected Model:** LightGBM-Tuned

**Reasoning:** Selected LightGBM-Tuned with Test R²=0.7055. Good generalization.

**All Models Comparison:**

| Model | Test R² | Overfitting % |
|-------|---------|---------------|
| LightGBM-Tuned | 0.7055 | 21.0% |
| XGBoost-Tuned | 0.6954 | 25.2% |
| LightGBM | 0.6906 | 25.8% |
| XGBoost | 0.6361 | 34.4% |

---

### DEBT_EQUITY

**Selected Model:** LightGBM-Tuned

**Reasoning:** Selected LightGBM-Tuned with Test R²=0.6675. Excellent generalization.

**All Models Comparison:**

| Model | Test R² | Overfitting % |
|-------|---------|---------------|
| LightGBM-Tuned | 0.6675 | 17.0% |
| XGBoost-Tuned | 0.6545 | 24.4% |
| LightGBM | 0.6406 | 29.9% |
| XGBoost | 0.5961 | 30.9% |

---

### PROFIT_MARGIN

**Selected Model:** LightGBM

**Reasoning:** Selected LightGBM with Test R²=0.4965. Warning: 38.6% overfitting detected.

**All Models Comparison:**

| Model | Test R² | Overfitting % |
|-------|---------|---------------|
| LightGBM | 0.4965 | 38.6% |
| LightGBM-Tuned | 0.4946 | 37.8% |
| XGBoost-Tuned | 0.4806 | 39.6% |
| XGBoost | 0.4586 | 51.5% |

---

### STOCK_RETURN

**Selected Model:** LightGBM-Tuned

**Reasoning:** Selected LightGBM-Tuned with Test R²=0.0575. Warning: 83.1% overfitting detected.

**All Models Comparison:**

| Model | Test R² | Overfitting % |
|-------|---------|---------------|
| LightGBM-Tuned | 0.0575 | 83.1% |
| XGBoost-Tuned | 0.0068 | 88.7% |
| LightGBM | 0.0053 | 84.2% |
| XGBoost | 0.0008 | 98.0% |

---

## Model Files

Best models saved to: `models/best_models/`

- `revenue_best.pkl`
- `eps_best.pkl`
- `debt_equity_best.pkl`
- `profit_margin_best.pkl`
- `stock_return_best.pkl`

## Usage

```python
import joblib

# Load best model for a target
model_data = joblib.load('models/best_models/revenue_best.pkl')
model = model_data['model']
feature_names = model_data['feature_names']

# Make predictions
predictions = model.predict(X_new)
```
