"""
Feature Selection - Updates Config Files
Identifies top 20 features and prepares configs for training comparison
"""

import numpy as np
import pandas as pd
import pickle
import yaml
from pathlib import Path
import xgboost as xgb
import matplotlib.pyplot as plt

# ============================================================
# PATHS
# ============================================================

project_root = Path(__file__).parent.parent.parent
DATA_DIR = project_root / "data" / "processed"
CONFIG_DIR = project_root / "config"
PLOTS_DIR = project_root / "plots"

PLOTS_DIR.mkdir(parents=True, exist_ok=True)
TOP_N_FEATURES = 20

# ============================================================
# LOAD DATA
# ============================================================

print("="*70)
print("🔍 FEATURE SELECTION - XGBoost Importance")
print("="*70)

X_train = np.load(DATA_DIR / 'X_train.npy')
y_train = np.load(DATA_DIR / 'y_train.npy')

with open(DATA_DIR / 'metadata.pkl', 'rb') as f:
    metadata = pickle.load(f)

FEATURE_COLS = metadata['feature_cols']
TARGET_COLS = metadata['original_target_cols']

print(f"\n📊 Dataset:")
print(f"   Features: {len(FEATURE_COLS)}")
print(f"   Targets: {TARGET_COLS}")
print(f"   Training samples: {len(X_train):,}")

# ============================================================
# COMPUTE FEATURE IMPORTANCE
# ============================================================

print("\n" + "="*70)
print("🚀 COMPUTING FEATURE IMPORTANCE")
print("="*70)

feature_importance_dict = {feat: 0.0 for feat in FEATURE_COLS}

for i, target in enumerate(TARGET_COLS):
    print(f"   Training XGBoost for {target}... ({i+1}/{len(TARGET_COLS)})")

    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        verbosity=0
    )

    model.fit(X_train, y_train[:, i])
    importance = model.feature_importances_

    for feat, imp in zip(FEATURE_COLS, importance):
        feature_importance_dict[feat] += imp

# Average across targets
for feat in feature_importance_dict:
    feature_importance_dict[feat] /= len(TARGET_COLS)

print("✅ Feature importance calculated")

# ============================================================
# SELECT TOP N
# ============================================================

print("\n" + "="*70)
print(f"📋 TOP {TOP_N_FEATURES} FEATURES")
print("="*70)

feature_importance_sorted = sorted(
    feature_importance_dict.items(),
    key=lambda x: x[1],
    reverse=True
)

top_features = [feat[0] for feat in feature_importance_sorted[:TOP_N_FEATURES]]
top_importance = [feat[1]
                  for feat in feature_importance_sorted[:TOP_N_FEATURES]]

print(f"\n{'Rank':<6} {'Feature':<50} {'Importance':>12}")
print("-" * 70)
for rank, (feat, imp) in enumerate(feature_importance_sorted[:TOP_N_FEATURES], 1):
    print(f"{rank:<6} {feat:<50} {imp:>12.6f}")

# ============================================================
# VISUALIZE
# ============================================================

print("\n📊 Creating visualization...")

fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# Top 20 bar chart
ax1 = axes[0]
ax1.barh(range(TOP_N_FEATURES), top_importance[::-1], color='steelblue')
ax1.set_yticks(range(TOP_N_FEATURES))
ax1.set_yticklabels([top_features[i] for i in range(TOP_N_FEATURES-1, -1, -1)])
ax1.set_xlabel('Average Importance', fontsize=12)
ax1.set_title(f'Top {TOP_N_FEATURES} Features', fontsize=14, fontweight='bold')
ax1.grid(axis='x', alpha=0.3)

# Importance decay curve
ax2 = axes[1]
all_importance = [feat[1] for feat in feature_importance_sorted]
ax2.plot(range(1, len(all_importance)+1),
         all_importance, marker='o', markersize=3)
ax2.axvline(x=TOP_N_FEATURES, color='red', linestyle='--', linewidth=2,
            label=f'Top {TOP_N_FEATURES} cutoff')
ax2.set_xlabel('Feature Rank', fontsize=12)
ax2.set_ylabel('Importance', fontsize=12)
ax2.set_title('Feature Importance Decay', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(alpha=0.3)

plt.tight_layout()
plot_path = PLOTS_DIR / 'feature_importance_analysis.png'
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"✅ Saved: {plot_path}")
plt.close()

# ============================================================
# SAVE RESULTS (NO DATA DUPLICATION!)
# ============================================================

print("\n" + "="*70)
print("💾 SAVING RESULTS")
print("="*70)

# Feature indices (for slicing arrays later)
top_feature_indices = [FEATURE_COLS.index(feat) for feat in top_features]

# Save feature selection results
feature_selection_results = {
    'top_features': top_features,
    'top_feature_indices': top_feature_indices,
    'top_importance': top_importance,
    'all_features': FEATURE_COLS,
    'all_importance': dict(feature_importance_sorted),
    'n_features_selected': TOP_N_FEATURES,
    'n_features_total': len(FEATURE_COLS),
    'target_cols': TARGET_COLS
}

with open(DATA_DIR / 'feature_selection_results.pkl', 'wb') as f:
    pickle.dump(feature_selection_results, f)
print(f"✅ Saved: feature_selection_results.pkl")

# Human-readable text file
with open(DATA_DIR / 'top20_features.txt', 'w') as f:
    f.write(f"Top {TOP_N_FEATURES} Features (XGBoost Importance)\n")
    f.write("="*70 + "\n\n")
    for rank, (feat, imp) in enumerate(feature_importance_sorted[:TOP_N_FEATURES], 1):
        f.write(f"{rank:2d}. {feat:<50} {imp:.6f}\n")
print(f"✅ Saved: top20_features.txt")

# ============================================================
# UPDATE CONFIG FILES
# ============================================================

print("\n" + "="*70)
print("⚙️  UPDATING CONFIG FILES")
print("="*70)

# Load existing feature_config.yaml
config_path = CONFIG_DIR / 'feature_config.yaml'
if config_path.exists():
    with open(config_path, 'r') as f:
        feature_config = yaml.safe_load(f)
else:
    feature_config = {}

# Update with actual values
feature_config['all_features'] = {
    'n_features': len(FEATURE_COLS),
    'source': 'data/processed/merged_dataset.csv'
}

feature_config['top_features'] = {
    'n_features': TOP_N_FEATURES,
    'selection_results': 'data/processed/feature_selection_results.pkl',
    'features': top_features  # Store actual feature names
}

feature_config['targets'] = {
    'names': TARGET_COLS,
    'n_targets': len(TARGET_COLS)
}

# Save updated config
with open(config_path, 'w') as f:
    yaml.dump(feature_config, f, default_flow_style=False, sort_keys=False)
print(f"✅ Updated: {config_path}")

# ============================================================
# SUMMARY
# ============================================================

print("\n" + "="*70)
print("✅ FEATURE SELECTION COMPLETE!")
print("="*70)

print(f"\n📊 Summary:")
print(f"   • Total features: {len(FEATURE_COLS)}")
print(f"   • Selected: {TOP_N_FEATURES}")
print(f"   • Reduction: {(1 - TOP_N_FEATURES/len(FEATURE_COLS))*100:.1f}%")

print(f"\n🎯 Training Strategy:")
print(f"   Models will be trained on BOTH:")
print(f"   1. Full features ({len(FEATURE_COLS)} features)")
print(f"   2. Top {TOP_N_FEATURES} features")

print(f"\n📁 Files created/updated:")
print(f"   • data/processed/feature_selection_results.pkl")
print(f"   • data/processed/top20_features.txt")
print(f"   • config/feature_config.yaml")
print(f"   • plots/feature_importance_analysis.png")

print(f"\n➡️  Next:")
print(f"   • Train LSTM: python src/models/train_lstm.py --feature-set full")
print(f"   • Train LSTM: python src/models/train_lstm.py --feature-set top20")
print(f"   • View results: mlflow ui --port 5000")

print("="*70)
