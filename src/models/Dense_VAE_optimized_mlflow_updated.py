"""
Dense VAE for Macroeconomic Scenario Generation
FINAL VERSION - Optimized for 1990-2023 dataset with multiple crises
WITH MLFLOW TRACKING

Key Features:
- ADASYN for crisis balancing
- Simple Dense VAE (no LSTM, no PCA complexity)
- Conditional severity sampling
- Crisis-focused generation (70% crisis scenarios)
- MLflow experiment tracking
- SEPARATE OUTPUT FOLDER (outputs/output_Dense_VAE_optimized)
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wasserstein_distance, ks_2samp
import os
import sys
import shutil
import warnings
import time
from datetime import datetime
warnings.filterwarnings('ignore')

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# MLflow imports
import mlflow
import mlflow.pytorch
from src.utils.mlflow_config import MLflowConfig


# ============================================
# 0. CLEANUP (MODEL-SPECIFIC)
# ============================================

def cleanup_outputs(output_dir='outputs/output_Dense_VAE_optimized'):
    """Delete old outputs for THIS MODEL ONLY"""
    if os.path.exists(output_dir):
        print(f"🗑️  Deleting old Dense VAE Optimized outputs...")
        try:
            shutil.rmtree(output_dir)
            print(f"✓ Cleaned up {output_dir}\n")
        except PermissionError:
            print(f"⚠  Files are open - please close Excel/CSV files")
            print(f"⚠  Skipping cleanup, will overwrite files instead\n")
        except Exception as e:
            print(f"⚠  Could not delete: {e}")
            print(f"⚠  Will overwrite files instead\n")
    
    # Create the directory
    os.makedirs(output_dir, exist_ok=True)
    print(f"✓ Created output directory: {output_dir}\n")


# ============================================
# 1. DATA LOADING WITH ADASYN
# ============================================

def load_data(csv_path, test_size=0.2, val_size=0.1, random_seed=42):
    """Load data with FIXED random seed for reproducibility"""
    
    print("="*60)
    print("LOADING DATA")
    print("="*60)
    print(f"File: {csv_path}")
    print(f"Random Seed: {random_seed} (for reproducibility)\n")
    
    # Set random seeds for reproducibility
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
    
    df = pd.read_csv(csv_path)
    
    # Store date for crisis analysis
    if 'Date' in df.columns:
        dates = pd.to_datetime(df['Date'])
        df_with_dates = df.copy()
        df = df.drop('Date', axis=1)
    else:
        dates = None
        df_with_dates = df.copy()
    
    # Handle categorical
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    if categorical_cols:
        print(f"Converting categorical columns: {categorical_cols}")
        for col in categorical_cols:
            if len(df[col].unique()) <= 10:
                df[col] = pd.Categorical(df[col]).codes
                print(f"  ✓ {col} encoded")
    
    # Numeric only
    df = df.select_dtypes(include=[np.number])
    df = df.fillna(method='ffill').fillna(method='bfill')
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    
    feature_names = df.columns.tolist()
    print(f"\n✓ Dataset: {len(df)} rows, {len(feature_names)} features")
    
    # ANALYZE CRISIS PERIODS IN YOUR DATA
    if dates is not None:
        analyze_crisis_periods(df_with_dates, dates, feature_names)
    
    # Split
    train_val, test = train_test_split(df, test_size=test_size, random_state=42)
    train, val = train_test_split(train_val, test_size=val_size/(1-test_size), random_state=42)
    
    print(f"✓ Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
    
    # Apply ADASYN
    train = apply_adasyn(train, feature_names)
    
    return train.values, val.values, test.values, feature_names


def analyze_crisis_periods(df, dates, feature_names):
    """Analyze and identify crisis periods in your dataset"""
    print("\n" + "="*60)
    print("CRISIS PERIOD ANALYSIS")
    print("="*60)
    
    df['Date'] = dates
    df['Year'] = dates.dt.year
    
    # Identify crisis periods based on VIX and Unemployment
    if 'VIX' in feature_names and 'Unemployment_Rate' in feature_names:
        
        vix_crisis = df['VIX'].quantile(0.90)
        unemp_crisis = df['Unemployment_Rate'].quantile(0.85)
        
        df['is_high_stress'] = (
            (df['VIX'] > vix_crisis) | 
            (df['Unemployment_Rate'] > unemp_crisis)
        )
        
        # Find crisis periods by year
        crisis_years = df[df['is_high_stress']].groupby('Year').size()
        crisis_years = crisis_years[crisis_years > 5]
        
        print(f"\nIdentified Crisis Periods in Your Data:")
        print("-" * 40)
        
        if len(crisis_years) > 0:
            for year, count in crisis_years.items():
                year_data = df[df['Year'] == year]
                avg_vix = year_data['VIX'].mean()
                avg_unemp = year_data['Unemployment_Rate'].mean()
                avg_gdp = year_data['GDP'].mean() if 'GDP' in feature_names else 0
                
                print(f"Year {year}: {count} stress months")
                print(f"  Avg VIX: {avg_vix:.1f}")
                print(f"  Avg Unemployment: {avg_unemp:.1f}%")
                if 'GDP' in feature_names:
                    print(f"  Avg GDP: {avg_gdp:,.0f}")
                
                # Identify specific crisis
                if year in [1990, 1991]:
                    print(f"  → Early 1990s Recession")
                elif year in [1997, 1998]:
                    print(f"  → Asian Financial Crisis")
                elif year in [2000, 2001, 2002]:
                    print(f"  → Dot-com Bubble Burst")
                elif year in [2008, 2009]:
                    print(f"  → Financial Crisis")
                elif year in [2020]:
                    print(f"  → COVID-19 Pandemic")
                print()
        else:
            print("⚠  No major crisis periods detected in dataset")
            print("  This may limit model's ability to generate crisis scenarios")
        
        # Overall statistics
        total_crisis_months = df['is_high_stress'].sum()
        crisis_pct = 100 * total_crisis_months / len(df)
        
        print("-" * 40)
        print(f"Total Crisis Months: {total_crisis_months} ({crisis_pct:.1f}%)")
        print(f"Normal Months: {len(df) - total_crisis_months} ({100-crisis_pct:.1f}%)")
        print(f"Unique Crisis Years: {len(crisis_years)}")
        
    print("="*60 + "\n")
    
    df = df.drop(['Date', 'Year', 'is_high_stress'], axis=1, errors='ignore')


def apply_adasyn(train_df, feature_names):
    """OPTIONAL: Apply ADASYN to balance crisis scenarios"""
    
    print("\n" + "="*60)
    print("ADASYN BALANCING")
    print("="*60)
    
    use_adasyn = False
    
    if not use_adasyn:
        print("⚠  ADASYN disabled (often gives better results)")
        print("✓ Using original data distribution")
        print("="*60 + "\n")
        return train_df
    
    # Define crisis: High VIX AND High Unemployment
    if 'VIX' in feature_names and 'Unemployment_Rate' in feature_names:
        
        vix_thresh = train_df['VIX'].quantile(0.80)
        unemp_thresh = train_df['Unemployment_Rate'].quantile(0.75)
        
        train_df['is_crisis'] = (
            (train_df['VIX'] > vix_thresh) &
            (train_df['Unemployment_Rate'] > unemp_thresh)
        ).astype(int)
        
        normal_count = (train_df['is_crisis'] == 0).sum()
        crisis_count = (train_df['is_crisis'] == 1).sum()
        
        print(f"Before ADASYN:")
        print(f"  Normal: {normal_count} ({100*normal_count/len(train_df):.1f}%)")
        print(f"  Crisis: {crisis_count} ({100*crisis_count/len(train_df):.1f}%)")
        
        try:
            from imblearn.over_sampling import ADASYN
            
            X = train_df.drop('is_crisis', axis=1).values
            y = train_df['is_crisis'].values
            
            adasyn = ADASYN(sampling_strategy=0.5, random_state=42, n_neighbors=5)
            X_resampled, y_resampled = adasyn.fit_resample(X, y)
            
            normal_after = (y_resampled == 0).sum()
            crisis_after = (y_resampled == 1).sum()
            
            print(f"\nAfter ADASYN:")
            print(f"  Normal: {normal_after} ({100*normal_after/len(y_resampled):.1f}%)")
            print(f"  Crisis: {crisis_after} ({100*crisis_after/len(y_resampled):.1f}%)")
            print(f"✓ Added {crisis_after - crisis_count} synthetic crisis scenarios")
            print("="*60 + "\n")
            
            return pd.DataFrame(X_resampled, columns=train_df.drop('is_crisis', axis=1).columns)
            
        except ImportError:
            print("\n⚠  imbalanced-learn not installed - skipping ADASYN")
            print("="*60 + "\n")
            return train_df.drop('is_crisis', axis=1)
    
    print("="*60 + "\n")
    return train_df


def normalize_data(train, val, test):
    """Normalize data"""
    
    print("Normalizing...")
    
    # Clip outliers
    for i in range(train.shape[1]):
        mean, std = train[:, i].mean(), train[:, i].std()
        lower, upper = mean - 5*std, mean + 5*std
        train[:, i] = np.clip(train[:, i], lower, upper)
        val[:, i] = np.clip(val[:, i], lower, upper)
        test[:, i] = np.clip(test[:, i], lower, upper)
    
    scaler = StandardScaler()
    train_s = scaler.fit_transform(train)
    val_s = scaler.transform(val)
    test_s = scaler.transform(test)
    
    print("✓ Normalized\n")
    
    return train_s, val_s, test_s, scaler


# ============================================
# 2. VAE MODEL
# ============================================

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim):
        super().__init__()
        layers = []
        curr = input_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(curr, h),
                nn.LayerNorm(h),             
                nn.SiLU(),                   
                nn.Dropout(0.1)              
            ])
            curr = h

        self.layers = nn.Sequential(*layers)
        self.fc_mu = nn.Linear(curr, latent_dim)
        self.fc_logvar = nn.Linear(curr, latent_dim)

    def forward(self, x):
        h = self.layers(x)
        return self.fc_mu(h), self.fc_logvar(h)


class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dims, output_dim):
        super().__init__()
        layers = []
        curr = latent_dim
        for h in reversed(hidden_dims):
            layers.extend([
                nn.Linear(curr, h),
                nn.LayerNorm(h),           
                nn.SiLU(),                 
                nn.Dropout(0.1)
            ])
            curr = h

        layers.append(nn.Linear(curr, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, z):
        return self.model(z)


class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim):
        super().__init__()
        self.encoder = Encoder(input_dim, hidden_dims, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dims, input_dim)
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar


def vae_loss(recon, x, mu, logvar, beta=1.0):
    recon_loss = nn.functional.mse_loss(recon, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl_loss, recon_loss, kl_loss


# ============================================
# 3. TRAINING (WITH MLFLOW)
# ============================================

def train_vae(train_loader, val_loader, input_dim, hidden_dims, latent_dim, 
              epochs, beta, device):
    """Train VAE with MLflow logging"""
    
    print("="*60)
    print("TRAINING")
    print("="*60)
    print(f"Device: {device}")
    print(f"Architecture: {hidden_dims} → {latent_dim}")
    print(f"Beta: {beta}\n")

    warmup_epochs = 30

    model = VAE(input_dim, hidden_dims, latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=15, factor=0.5)

    best_val = float('inf')
    patience_counter = 0
    epoch_times = []

    for epoch in range(epochs):
        epoch_start = time.time()
        
        # KL warmup
        curr_beta = min(beta, beta * epoch / warmup_epochs)

        # Train
        model.train()
        train_losses = []

        for batch in train_loader:
            if isinstance(batch, list):
                batch = batch[0]
            batch = batch.to(device)

            recon, mu, logvar = model(batch)
            loss, _, _ = vae_loss(recon, batch, mu, logvar, curr_beta)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_losses.append(loss.item() / len(batch))

        # Validate
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                if isinstance(batch, list):
                    batch = batch[0]
                batch = batch.to(device)

                recon, mu, logvar = model(batch)
                loss, _, _ = vae_loss(recon, batch, mu, logvar, beta)
                val_losses.append(loss.item() / len(batch))

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)

        scheduler.step(val_loss)
        
        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)

        # Log to MLflow every 20 epochs
        if (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] | Train: {train_loss:.4f} | Val: {val_loss:.4f}")
            mlflow.log_metrics({
                'train_loss': train_loss,
                'val_loss': val_loss,
                'learning_rate': optimizer.param_groups[0]['lr']
            }, step=epoch+1)

        # Early stopping
        if val_loss < best_val:
            best_val = val_loss
            patience_counter = 0
            best_state = model.state_dict()
        else:
            patience_counter += 1

        if patience_counter >= 30:
            print(f"\nEarly stopping at epoch {epoch+1}")
            mlflow.log_metric('epochs_completed', epoch+1)
            break

    model.load_state_dict(best_state)
    
    # Log final training metrics
    avg_epoch_time = np.mean(epoch_times)
    mlflow.log_metrics({
        'best_val_loss': best_val,
        'avg_epoch_time_seconds': avg_epoch_time
    })
    
    print(f"\n✓ Training complete! Best Val Loss: {best_val:.4f}\n")

    return model


# ============================================
# 4. GENERATION (WITH MLFLOW)
# ============================================

def generate_scenarios(model, scaler, features, n_baseline=10, n_adverse=20, 
                      n_severe=50, n_extreme=20, device='cpu'):
    """Generate crisis-focused scenarios"""
    
    print("="*60)
    print("GENERATING SCENARIOS (CRISIS-FOCUSED)")
    print("="*60)
    
    model.eval()
    latent_dim = model.decoder.model[0].in_features
    
    all_scenarios = []
    labels = []
    
    with torch.no_grad():
        # Baseline (σ=0.5)
        z = torch.randn(n_baseline, latent_dim, device=device) * 0.5
        s = model.decoder(z).cpu().numpy()
        all_scenarios.append(s)
        labels.extend(['Baseline'] * n_baseline)
        
        # Adverse (σ=1.5)
        z = torch.randn(n_adverse, latent_dim, device=device) * 1.5
        s = model.decoder(z).cpu().numpy()
        all_scenarios.append(s)
        labels.extend(['Adverse'] * n_adverse)
        
        # Severe (σ=2.5)
        z = torch.randn(n_severe, latent_dim, device=device) * 2.5
        s = model.decoder(z).cpu().numpy()
        all_scenarios.append(s)
        labels.extend(['Severe'] * n_severe)
        
        # Extreme (σ=3.5)
        z = torch.randn(n_extreme, latent_dim, device=device) * 3.5
        s = model.decoder(z).cpu().numpy()
        all_scenarios.append(s)
        labels.extend(['Extreme'] * n_extreme)
    
    # Combine
    all_scenarios = np.vstack(all_scenarios)
    
    # Denormalize
    scenarios_denorm = scaler.inverse_transform(all_scenarios)
    
    # Create DataFrame
    df = pd.DataFrame(scenarios_denorm, columns=features)
    df.insert(0, 'Scenario', [f'Scenario_{i+1}' for i in range(len(df))])
    df.insert(1, 'Severity', labels)
    
    print(f"\n✓ Generated {len(df)} scenarios:")
    print(f"  - {n_baseline} Baseline (normal) - {n_baseline}%")
    print(f"  - {n_adverse} Adverse (mild stress) - {n_adverse}%")
    print(f"  - {n_severe} Severe (crisis) - {n_severe}%")
    print(f"  - {n_extreme} Extreme (worst case) - {n_extreme}%")
    print(f"\n✓ Crisis scenarios: {n_adverse + n_severe + n_extreme} ({n_adverse + n_severe + n_extreme}%)")
    print("="*60 + "\n")
    
    # Log scenario distribution to MLflow
    mlflow.log_metrics({
        'n_baseline_scenarios': n_baseline,
        'n_adverse_scenarios': n_adverse,
        'n_severe_scenarios': n_severe,
        'n_extreme_scenarios': n_extreme,
        'total_scenarios': len(df),
        'crisis_scenario_percentage': (n_adverse + n_severe + n_extreme)
    })
    
    return df


def classify_stress_level(scenarios_df, feature_names):
    """Classify each scenario into stress levels"""
    
    print("="*60)
    print("CLASSIFYING STRESS LEVELS")
    print("="*60 + "\n")
    
    stress_levels = []
    stress_scores = []
    
    for idx, row in scenarios_df.iterrows():
        score = 0
        
        # Check GDP
        if 'GDP' in feature_names:
            gdp = row['GDP']
            if gdp < 12000:
                score += 4
            elif gdp < 14000:
                score += 3
            elif gdp < 16000:
                score += 2
            elif gdp < 18000:
                score += 1
        
        # Check VIX
        if 'VIX' in feature_names:
            vix = row['VIX']
            if vix > 45:
                score += 4
            elif vix > 30:
                score += 3
            elif vix > 20:
                score += 2
            elif vix > 15:
                score += 1
        
        # Check Unemployment
        if 'Unemployment_Rate' in feature_names:
            unemp = row['Unemployment_Rate']
            if unemp > 10:
                score += 4
            elif unemp > 8:
                score += 3
            elif unemp > 6:
                score += 2
            elif unemp > 5:
                score += 1
        
        # Check SP500
        if 'SP500_Close' in feature_names:
            sp500 = row['SP500_Close']
            if sp500 < 500:
                score += 4
            elif sp500 < 1000:
                score += 3
            elif sp500 < 1500:
                score += 2
            elif sp500 < 2000:
                score += 1
        
        # Check Oil
        if 'Oil_Price' in feature_names:
            oil = row['Oil_Price']
            if oil < 20 or oil > 120:
                score += 2
            elif oil < 30 or oil > 100:
                score += 1
        
        # Classify
        if score >= 12:
            stress_level = 'Extreme Crisis'
        elif score >= 8:
            stress_level = 'Severe Crisis'
        elif score >= 5:
            stress_level = 'Moderate Stress'
        elif score >= 2:
            stress_level = 'Mild Stress'
        else:
            stress_level = 'Normal'
        
        stress_levels.append(stress_level)
        stress_scores.append(score)
    
    scenarios_df.insert(2, 'Stress_Level', stress_levels)
    scenarios_df.insert(3, 'Stress_Score', stress_scores)
    
    # Print distribution
    print("Stress Level Distribution:")
    print("-" * 40)
    stress_dist = {}
    for level in ['Normal', 'Mild Stress', 'Moderate Stress', 'Severe Crisis', 'Extreme Crisis']:
        count = (scenarios_df['Stress_Level'] == level).sum()
        pct = 100 * count / len(scenarios_df)
        stress_dist[f'stress_level_{level.lower().replace(" ", "_")}'] = count
        print(f"  {level:20s}: {count:3d} ({pct:5.1f}%)")
    
    print(f"\nStress Score Range: {min(stress_scores)} - {max(stress_scores)}")
    print("="*60 + "\n")
    
    # Log stress distribution to MLflow
    mlflow.log_metrics(stress_dist)
    
    return scenarios_df


# ============================================
# 5. VALIDATION (WITH MLFLOW)
# ============================================

def validate(real_data, gen_data, features, save_dir='outputs/output_Dense_VAE_optimized'):
    """Validate scenarios and log to MLflow"""
    
    os.makedirs(save_dir, exist_ok=True)
    
    print("="*60)
    print("VALIDATION")
    print("="*60 + "\n")
    
    if isinstance(gen_data, pd.DataFrame):
        gen_data = gen_data.drop(['Scenario', 'Severity', 'Stress_Level', 'Stress_Score'], 
                                 axis=1, errors='ignore').values
    
    # KS tests
    ks_results = []
    for i in range(len(features)):
        stat, pval = ks_2samp(real_data[:, i], gen_data[:, i])
        ks_results.append({
            'feature': features[i],
            'statistic': stat,
            'p_value': pval,
            'passed': pval > 0.05
        })
    
    ks_df = pd.DataFrame(ks_results)
    pass_rate = 100 * ks_df['passed'].sum() / len(ks_df)
    
    # Wasserstein
    w_dists = [wasserstein_distance(real_data[:, i], gen_data[:, i]) 
               for i in range(min(20, real_data.shape[1]))]
    w_mean = np.mean(w_dists)
    
    # Correlation
    real_corr = np.corrcoef(real_data[:, :20].T)
    gen_corr = np.corrcoef(gen_data[:, :20].T)
    corr_mae = np.mean(np.abs(real_corr - gen_corr))
    
    print(f"1. KS Test Pass Rate: {pass_rate:.1f}% ({ks_df['passed'].sum()}/{len(ks_df)})")
    print(f"2. Correlation MAE: {corr_mae:.4f}")
    print(f"3. Wasserstein Distance: {w_mean:.2f}\n")
    
    # Log PRIMARY metrics to MLflow
    mlflow.log_metrics({
        'ks_pass_rate': pass_rate,
        'correlation_mae': corr_mae,
        'wasserstein_distance': w_mean,
        'n_features_passed_ks': int(ks_df['passed'].sum()),
        'n_features_total': len(ks_df)
    })
    
    # Save detailed report
    with open(f'{save_dir}/validation_report.txt', 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("DENSE VAE VALIDATION REPORT\n")
        f.write("="*60 + "\n\n")
        f.write(f"KS Test Pass Rate: {pass_rate:.1f}%\n")
        f.write(f"Correlation MAE: {corr_mae:.4f}\n")
        f.write(f"Wasserstein Distance: {w_mean:.2f}\n\n")
        
        f.write("Failed Features (p < 0.05):\n")
        failed = ks_df[~ks_df['passed']]
        for _, row in failed.iterrows():
            f.write(f"  - {row['feature']}: p={row['p_value']:.4f}\n")
    
    # Save KS results CSV
    ks_df.to_csv(f'{save_dir}/ks_test_results.csv', index=False)
    
    print(f"✓ Saved: {save_dir}/validation_report.txt")
    print(f"✓ Saved: {save_dir}/ks_test_results.csv\n")
    
    return {'pass_rate': pass_rate, 'wasserstein': w_mean, 'correlation': corr_mae}


# ============================================
# 6. VISUALIZATION
# ============================================

def create_plots(real_data, gen_data, features, save_dir='outputs/output_Dense_VAE_optimized'):
    """Create validation plots"""
    
    if isinstance(gen_data, pd.DataFrame):
        gen_data = gen_data.drop(['Scenario', 'Severity', 'Stress_Level', 'Stress_Score'], 
                                 axis=1, errors='ignore').values
    
    # Distribution plots
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    
    key_features = ['GDP', 'CPI', 'Unemployment_Rate', 'VIX', 'SP500_Close', 'Oil_Price']
    plot_idx = [features.index(f) for f in key_features if f in features][:6]
    
    for idx, feat_idx in enumerate(plot_idx):
        axes[idx].hist(real_data[:, feat_idx], bins=50, alpha=0.5, 
                      label='Real', density=True, color='blue')
        axes[idx].hist(gen_data[:, feat_idx], bins=50, alpha=0.5, 
                      label='Generated', density=True, color='orange')
        axes[idx].set_title(features[feat_idx])
        axes[idx].legend()
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Correlation heatmaps
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    real_corr = np.corrcoef(real_data[:, :20].T)
    gen_corr = np.corrcoef(gen_data[:, :20].T)
    
    sns.heatmap(real_corr, ax=axes[0], cmap='coolwarm', center=0, vmin=-1, vmax=1)
    axes[0].set_title('Real Correlation')
    
    sns.heatmap(gen_corr, ax=axes[1], cmap='coolwarm', center=0, vmin=-1, vmax=1)
    axes[1].set_title('Generated Correlation')
    
    sns.heatmap(real_corr - gen_corr, ax=axes[2], cmap='RdBu_r', center=0, vmin=-0.5, vmax=0.5)
    axes[2].set_title('Difference')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/correlations.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved plots to {save_dir}/\n")


# ============================================
# 7. MAIN PIPELINE (WITH MLFLOW)
# ============================================

def main(csv_path, hidden_dims=[256, 128, 64], latent_dim=32, 
         batch_size=64, epochs=300, beta=0.5,
         n_baseline=10, n_adverse=20, n_severe=50, n_extreme=20,
         mlflow_tracking_uri=None, experiment_name="Financial_Stress_Test_Scenarios",
         output_dir='outputs/output_Dense_VAE_optimized'):
    """Complete pipeline with MLflow tracking"""
    
    cleanup_outputs(output_dir)
    
    print("="*60)
    print("DENSE VAE OPTIMIZED - FINANCIAL STRESS TESTING")
    print("Dataset: Extended 1990-2025")
    print(f"Output Directory: {output_dir}")
    print("="*60 + "\n")
    
    # Initialize MLflow
    mlflow_config = MLflowConfig(
        tracking_uri=mlflow_tracking_uri,
        experiment_name=experiment_name
    )
    
    # Start MLflow run
    run_name = f"Dense_VAE_Optimized_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    with mlflow.start_run(run_name=run_name) as run:
        print(f"\n✓ MLflow Run Started: {run.info.run_id}")
        print(f"✓ Run Name: {run_name}\n")
        
        # Set tags
        mlflow_config.set_tags({
            'model_type': 'Dense_VAE_Optimized',
            'stage': 'development',
            'data_version': '1990_extended',
            'crisis_focus': 'True',
            'optimization': 'LayerNorm_SiLU',
            'output_folder': output_dir
        })
        
        # Log parameters
        params = {
            'hidden_dims': str(hidden_dims),
            'latent_dim': latent_dim,
            'batch_size': batch_size,
            'max_epochs': epochs,
            'beta': beta,
            'random_seed': 42,
            'n_baseline': n_baseline,
            'n_adverse': n_adverse,
            'n_severe': n_severe,
            'n_extreme': n_extreme,
            'test_size': 0.2,
            'val_size': 0.1,
            'warmup_epochs': 30,
            'activation': 'SiLU',
            'normalization': 'LayerNorm',
            'dropout': 0.1
        }
        mlflow_config.log_params(params)
        
        # Track total time
        start_time = time.time()
        
        # Load data
        train, val, test, features = load_data(csv_path)
        train_s, val_s, test_s, scaler = normalize_data(train, val, test)
        
        # Log data statistics
        mlflow.log_metrics({
            'n_train_samples': len(train),
            'n_val_samples': len(val),
            'n_test_samples': len(test),
            'n_features': len(features)
        })
        
        # Create loaders
        train_loader = DataLoader(
            TensorDataset(torch.FloatTensor(train_s)), 
            batch_size=batch_size, shuffle=True, drop_last=True
        )
        val_loader = DataLoader(
            TensorDataset(torch.FloatTensor(val_s)), 
            batch_size=batch_size, shuffle=False
        )
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        mlflow.log_param('device', str(device))
        
        # Train
        model = train_vae(train_loader, val_loader, train_s.shape[1], 
                         hidden_dims, latent_dim, epochs, beta, device)
        
        # Generate
        scenarios = generate_scenarios(model, scaler, features, 
                                       n_baseline, n_adverse, n_severe, n_extreme, device)
        
        # Validate BEFORE adding classification
        results = validate(test, scenarios, features, save_dir=output_dir)
        
        # Add stress classification AFTER validation
        scenarios = classify_stress_level(scenarios, features)
        
        # Visualize
        create_plots(test, scenarios, features, save_dir=output_dir)
        
        # Calculate training time
        training_time = time.time() - start_time
        mlflow.log_metric('training_time_seconds', training_time)
        
        # Save files
        print("="*60)
        print("SAVING")
        print("="*60 + "\n")
        
        scenarios.to_csv(f'{output_dir}/dense_vae_optimized_scenarios.csv', index=False)
        print(f"✓ Saved: {output_dir}/dense_vae_optimized_scenarios.csv")
        
        torch.save({
            'model': model.state_dict(),
            'scaler': scaler,
            'features': features,
            'config': {
                'hidden_dims': hidden_dims,
                'latent_dim': latent_dim,
                'beta': beta
            }
        }, f'{output_dir}/dense_vae_optimized_model.pth')
        print(f"✓ Saved: {output_dir}/dense_vae_optimized_model.pth\n")
        
        # Log artifacts to MLflow
        print("Logging artifacts to MLflow...")
        mlflow_config.log_artifacts([
            f'{output_dir}/dense_vae_optimized_scenarios.csv',
            f'{output_dir}/dense_vae_optimized_model.pth',
            f'{output_dir}/validation_report.txt',
            f'{output_dir}/ks_test_results.csv',
            f'{output_dir}/distributions.png',
            f'{output_dir}/correlations.png'
        ])
        
        # Log PyTorch model
        mlflow.pytorch.log_model(model, "model")
        
        print("✓ All artifacts logged to MLflow\n")
        
        # Final summary
        print("="*60)
        print("FINAL RESULTS")
        print("="*60)
        print(f"✓ KS Pass Rate: {results['pass_rate']:.1f}%")
        print(f"✓ Correlation MAE: {results['correlation']:.4f}")
        print(f"✓ Wasserstein: {results['wasserstein']:.1f}")
        print(f"✓ Total Scenarios: {len(scenarios)}")
        print(f"✓ Crisis Focus: {n_adverse+n_severe+n_extreme}% crisis scenarios")
        print(f"✓ Training Time: {training_time:.1f}s")
        print(f"✓ MLflow Run ID: {run.info.run_id}")
        print(f"✓ Output Directory: {output_dir}")
        print("="*60 + "\n")
        
        return model, scenarios, results, run.info.run_id


# ============================================
# 8. RUN
# ============================================

if __name__ == "__main__":
    
    import os
    from dotenv import load_dotenv
    
    # Load environment variables
    load_dotenv()
    
    # Get MLflow tracking URI from .env
    mlflow_uri = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000')
    experiment_name = os.getenv('MLFLOW_EXPERIMENT_NAME', 'Financial_Stress_Test_Scenarios')
    
    # YOUR DATA PATH
    data_path = r'data/features/macro_features_clean.csv'
    
    print(f"Looking for data: {data_path}\n")
    print(f"MLflow Tracking URI: {mlflow_uri}\n")
    
    # Run with MLflow tracking
    model, scenarios, results, run_id = main(
        csv_path=data_path,
        hidden_dims=[256, 128, 64],
        latent_dim=32,
        batch_size=64,
        epochs=300,
        beta=0.5,
        n_baseline=10,
        n_adverse=20,
        n_severe=50,
        n_extreme=20,
        mlflow_tracking_uri=mlflow_uri,
        experiment_name=experiment_name,
        output_dir='outputs/output_Dense_VAE_optimized'
    )
    
    print("="*60)
    print("🎉 SUCCESS!")
    print("="*60)
    print("\nFiles saved in: outputs/output_Dense_VAE_optimized/")
    print("  - dense_vae_optimized_scenarios.csv")
    print("  - dense_vae_optimized_model.pth")
    print("  - validation_report.txt")
    print("  - distributions.png")
    print("  - correlations.png")
    print("\nMLflow:")
    print(f"  - Run ID: {run_id}")
    print(f"  - View at: {mlflow_uri}")
    print("="*60)