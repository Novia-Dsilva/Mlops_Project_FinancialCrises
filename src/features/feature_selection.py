# File: src/features/feature_selection.py

"""
Select top 20 most important features using XGBoost feature importance
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# CONFIGURATION
# ============================================================

project_root = Path(__file__).parent.parent.parent
DATA_DIR = project_root / "data" / "processed"
OUTPUT_DIR = project_root / "data" / "processed"
PLOTS_DIR = project_root / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

TOP_N_FEATURES = 20  # Select top 20 features

# ============================================================
# LOAD DATA
# ============================================================

print("="*70)
print("🔍 FEATURE SELECTION - TOP 20 FEATURES")
print("="*70)

X_train = np.load(DATA_DIR / 'X_train.npy')
y_train = np.load(DATA_DIR / 'y_train.npy')

with open(DATA_DIR / 'metadata.pkl', 'rb') as f:
    metadata = pickle.load(f)

FEATURE_COLS = metadata['feature_cols']
TARGET_COLS = metadata['original_target_cols']

print(f"\n📊 Original dataset:")
print(f"   Features: {len(FEATURE_COLS)}")
print(f"   Targets: {len(TARGET_COLS)}")
print(f"   Training samples: {len(X_train)}")

# ============================================================
# TRAIN XGBOOST FOR FEATURE IMPORTANCE
# ============================================================

print("\n" + "="*70)
print("🚀 TRAINING XGBOOST FOR FEATURE IMPORTANCE")
print("="*70)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)

# Train XGBoost on each target and aggregate importance
feature_importance_dict = {feat: 0.0 for feat in FEATURE_COLS}

for i, target in enumerate(TARGET_COLS):
    print(f"\n📈 Training for {target}... ({i+1}/{len(TARGET_COLS)})")

    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        verbosity=0
    )

    model.fit(X_scaled, y_train[:, i])

    # Get feature importance
    importance = model.feature_importances_

    # Accumulate importance across all targets
    for feat, imp in zip(FEATURE_COLS, importance):
        feature_importance_dict[feat] += imp

# Average importance across all targets
for feat in feature_importance_dict:
    feature_importance_dict[feat] /= len(TARGET_COLS)

print("\n✅ Feature importance calculated across all targets")

# ============================================================
# SELECT TOP N FEATURES
# ============================================================

print("\n" + "="*70)
print(f"📋 SELECTING TOP {TOP_N_FEATURES} FEATURES")
print("="*70)

# Sort by importance
feature_importance_sorted = sorted(
    feature_importance_dict.items(),
    key=lambda x: x[1],
    reverse=True
)

# Get top N
top_features = [feat[0] for feat in feature_importance_sorted[:TOP_N_FEATURES]]
top_importance = [feat[1]
                  for feat in feature_importance_sorted[:TOP_N_FEATURES]]

print(f"\n🏆 Top {TOP_N_FEATURES} Most Important Features:")
print(f"{'Rank':<6} {'Feature':<50} {'Importance':>12}")
print("-" * 70)

for rank, (feat, imp) in enumerate(feature_importance_sorted[:TOP_N_FEATURES], 1):
    print(f"{rank:<6} {feat:<50} {imp:>12.6f}")

# ============================================================
# VISUALIZE FEATURE IMPORTANCE
# ============================================================

print("\n📊 Creating visualization...")

plt.figure(figsize=(12, 8))
plt.barh(range(TOP_N_FEATURES), top_importance[::-1], color='steelblue')
plt.yticks(range(TOP_N_FEATURES), [top_features[i]
           for i in range(TOP_N_FEATURES-1, -1, -1)])
plt.xlabel('Average Feature Importance', fontsize=12)
plt.ylabel('Features', fontsize=12)
plt.title(f'Top {TOP_N_FEATURES} Most Important Features for Next Quarter Prediction',
          fontsize=14, fontweight='bold')
plt.tight_layout()

plot_path = PLOTS_DIR / 'feature_importance_top20.png'
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"✅ Plot saved: {plot_path}")
plt.close()

# ============================================================
# CREATE REDUCED DATASET
# ============================================================

print("\n" + "="*70)
print("💾 CREATING REDUCED DATASET")
print("="*70)

# Get indices of top features
top_feature_indices = [FEATURE_COLS.index(feat) for feat in top_features]

# Load full dataset
X_train_full = np.load(DATA_DIR / 'X_train.npy')
X_test_full = np.load(DATA_DIR / 'X_test.npy')
y_train_full = np.load(DATA_DIR / 'y_train.npy')
y_test_full = np.load(DATA_DIR / 'y_test.npy')

# Select only top features
X_train_reduced = X_train_full[:, top_feature_indices]
X_test_reduced = X_test_full[:, top_feature_indices]

print(f"✅ Reduced dataset shape:")
print(f"   X_train: {X_train_full.shape} → {X_train_reduced.shape}")
print(f"   X_test:  {X_test_full.shape} → {X_test_reduced.shape}")

# Save reduced dataset
np.save(OUTPUT_DIR / 'X_train_top20.npy', X_train_reduced)
np.save(OUTPUT_DIR / 'X_test_top20.npy', X_test_reduced)
np.save(OUTPUT_DIR / 'y_train_top20.npy', y_train_full)
np.save(OUTPUT_DIR / 'y_test_top20.npy', y_test_full)

# Save metadata
metadata_reduced = metadata.copy()
metadata_reduced['feature_cols'] = top_features
metadata_reduced['n_features'] = TOP_N_FEATURES
metadata_reduced['feature_selection_method'] = 'xgboost_importance'
metadata_reduced['original_n_features'] = len(FEATURE_COLS)

with open(OUTPUT_DIR / 'metadata_top20.pkl', 'wb') as f:
    pickle.dump(metadata_reduced, f)

print(f"\n✅ Saved reduced dataset to {OUTPUT_DIR}/")
print(f"   - X_train_top20.npy")
print(f"   - X_test_top20.npy")
print(f"   - y_train_top20.npy")
print(f"   - y_test_top20.npy")
print(f"   - metadata_top20.pkl")

# ============================================================
# SAVE FEATURE LIST FOR REFERENCE
# ============================================================

# Save as text file
with open(OUTPUT_DIR / 'top20_features.txt', 'w') as f:
    f.write(f"Top {TOP_N_FEATURES} Features Selected for Model Training\n")
    f.write("="*70 + "\n\n")
    for rank, (feat, imp) in enumerate(feature_importance_sorted[:TOP_N_FEATURES], 1):
        f.write(f"{rank:2d}. {feat:<50} {imp:.6f}\n")

print(f"✅ Feature list saved: top20_features.txt")

# ============================================================
# SUMMARY
# ============================================================

print("\n" + "="*70)
print("✅ FEATURE SELECTION COMPLETE!")
print("="*70)

print(f"\n📊 Summary:")
print(f"   Original features: {len(FEATURE_COLS)}")
print(f"   Selected features: {TOP_N_FEATURES}")
print(f"   Reduction: {(1 - TOP_N_FEATURES/len(FEATURE_COLS))*100:.1f}%")

print(f"\n📁 Output files:")
print(f"   - data/processed/X_train_top20.npy")
print(f"   - data/processed/X_test_top20.npy")
print(f"   - data/processed/metadata_top20.pkl")
print(f"   - plots/feature_importance_top20.png")
print(f"   - data/processed/top20_features.txt")

print(f"\n➡️  Next: Train models with reduced features")
print(f"   python src/models/train_xgboost_top20.py")
print("="*70)
