"""
Ensemble VAE for Macroeconomic Scenario Generation
WITH MLFLOW TRACKING (Nested Runs)

Trains multiple Dense VAEs with different random seeds
Combines their outputs for better diversity and quality
SEPARATE OUTPUT FOLDER (outputs/output_Ensemble_VAE)
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
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
# CLEANUP (MODEL-SPECIFIC)
# ============================================

def cleanup_outputs(output_dir='outputs/output_Ensemble_VAE'):
    """Delete old outputs for THIS MODEL ONLY and recreate directory"""
    if os.path.exists(output_dir):
        print(f"🗑️  Deleting old Ensemble VAE outputs...")
        try:
            shutil.rmtree(output_dir)
            print(f"✓ Cleaned up {output_dir}\n")
        except:
            print(f"⚠  Could not delete outputs folder (files may be open)")
            print(f"⚠  Will overwrite files instead\n")
    
    # Recreate the directory
    os.makedirs(output_dir, exist_ok=True)
    print(f"✓ Created output directory: {output_dir}\n")


# [Rest of the code remains the same until main() function...]

# ============================================
# DATA LOADING
# ============================================

def load_data(csv_path, test_size=0.2, val_size=0.1, random_seed=42):
    """Load and split data"""
    
    print("="*60)
    print("LOADING DATA")
    print("="*60)
    print(f"File: {csv_path}\n")
    
    df = pd.read_csv(csv_path)
    
    if 'Date' in df.columns:
        df = df.drop('Date', axis=1)
    
    # Handle categorical
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    for col in categorical_cols:
        if len(df[col].unique()) <= 10:
            df[col] = pd.Categorical(df[col]).codes
    
    # Numeric only
    df = df.select_dtypes(include=[np.number])
    df = df.fillna(method='ffill').fillna(method='bfill')
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    
    features = df.columns.tolist()
    print(f"✓ Dataset: {len(df)} rows, {len(features)} features")
    
    # Split with fixed seed
    train_val, test = train_test_split(df, test_size=test_size, random_state=random_seed)
    train, val = train_test_split(train_val, test_size=val_size/(1-test_size), random_state=random_seed)
    
    print(f"✓ Train: {len(train)}, Val: {len(val)}, Test: {len(test)}\n")
    
    return train.values, val.values, test.values, features


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
# VAE ARCHITECTURE
# ============================================

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim):
        super().__init__()
        layers = []
        curr = input_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(curr, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(0.2)
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
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(0.2)
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


def vae_loss(recon, x, mu, logvar, beta=0.5):
    recon_loss = nn.functional.mse_loss(recon, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl_loss


# ============================================
# ENSEMBLE VAE WRAPPER (Custom MLflow Model)
# ============================================

class EnsembleVAEWrapper(mlflow.pyfunc.PythonModel):
    """Custom MLflow model wrapper for Ensemble VAE"""
    
    def load_context(self, context):
        """Load the ensemble models and scaler"""
        import torch
        import pickle
        
        ensemble_path = context.artifacts["ensemble_package"]
        ensemble_data = torch.load(ensemble_path, map_location='cpu')
        
        self.models = []
        self.config = ensemble_data['config']
        
        for model_state in ensemble_data['models']:
            model = VAE(
                input_dim=ensemble_data['input_dim'],
                hidden_dims=self.config['hidden_dims'],
                latent_dim=self.config['latent_dim']
            )
            model.load_state_dict(model_state)
            model.eval()
            self.models.append(model)
        
        self.scaler = ensemble_data['scaler']
        self.features = ensemble_data['features']
        self.n_models = len(self.models)
        
        print(f"✓ Loaded {self.n_models} ensemble models")
    
    def predict(self, context, model_input):
        """Generate scenarios using the ensemble"""
        import torch
        import pandas as pd
        
        n_scenarios = model_input.get('n_scenarios', 100)
        severity_dist = model_input.get('severity_dist', {
            'baseline': 10, 'adverse': 20, 'severe': 50, 'extreme': 20
        })
        sigma_values = model_input.get('sigma_values', {
            'baseline': 0.5, 'adverse': 1.5, 'severe': 2.5, 'extreme': 3.5
        })
        
        all_scenarios = []
        all_severities = []
        
        per_model_dist = {k: v // self.n_models for k, v in severity_dist.items()}
        
        device = torch.device('cpu')
        
        for model in self.models:
            model.eval()
            latent_dim = model.decoder.model[0].in_features
            
            with torch.no_grad():
                for severity, count in per_model_dist.items():
                    if count > 0:
                        sigma = sigma_values[severity]
                        z = torch.randn(count, latent_dim, device=device) * sigma
                        scenarios = model.decoder(z).cpu().numpy()
                        all_scenarios.append(scenarios)
                        all_severities.extend([severity.capitalize()] * count)
        
        all_scenarios = np.vstack(all_scenarios)
        scenarios_denorm = self.scaler.inverse_transform(all_scenarios)
        
        df = pd.DataFrame(scenarios_denorm, columns=self.features)
        df.insert(0, 'Scenario', [f'Scenario_{i+1}' for i in range(len(df))])
        df.insert(1, 'Severity', all_severities)
        
        return df


# ============================================
# TRAIN SINGLE VAE (WITH NESTED MLFLOW RUN)
# ============================================

def train_single_vae(train_loader, val_loader, input_dim, hidden_dims, 
                    latent_dim, epochs, beta, device, model_id, random_seed, output_dir):
    """Train one VAE with specific random seed and nested MLflow run"""
    
    print(f"\n{'='*60}")
    print(f"TRAINING VAE #{model_id} (seed={random_seed})")
    print(f"{'='*60}")
    
    # Set seed for this specific model
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
    
    # Start NESTED MLflow run for this model
    with mlflow.start_run(run_name=f"VAE_Model_{model_id}", nested=True) as nested_run:
        
        # Log model-specific parameters
        mlflow.log_params({
            'model_id': model_id,
            'random_seed': random_seed,
            'hidden_dims': str(hidden_dims),
            'latent_dim': latent_dim,
            'beta': beta
        })
        
        model_start_time = time.time()
        
        model = VAE(input_dim, hidden_dims, latent_dim).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=15, factor=0.5)
        
        best_val = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Train
            model.train()
            train_losses = []
            for batch in train_loader:
                if isinstance(batch, list):
                    batch = batch[0]
                batch = batch.to(device)
                recon, mu, logvar = model(batch)
                loss = vae_loss(recon, batch, mu, logvar, beta)
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
                    loss = vae_loss(recon, batch, mu, logvar, beta)
                    val_losses.append(loss.item() / len(batch))
            
            train_loss = np.mean(train_losses)
            val_loss = np.mean(val_losses)
            scheduler.step(val_loss)
            
            # Log to MLflow every 50 epochs
            if (epoch + 1) % 50 == 0:
                print(f"  Epoch [{epoch+1}/{epochs}] | Train: {train_loss:.4f} | Val: {val_loss:.4f}")
                mlflow.log_metrics({
                    f'model_{model_id}_train_loss': train_loss,
                    f'model_{model_id}_val_loss': val_loss
                }, step=epoch+1)
            
            # Early stopping
            if val_loss < best_val:
                best_val = val_loss
                patience_counter = 0
                best_state = model.state_dict()
            else:
                patience_counter += 1
            
            if patience_counter >= 30:
                print(f"  Early stopping at epoch {epoch+1}")
                mlflow.log_metric(f'model_{model_id}_epochs_completed', epoch+1)
                break
        
        model.load_state_dict(best_state)
        
        # Log final metrics for this model
        model_training_time = time.time() - model_start_time
        mlflow.log_metrics({
            f'model_{model_id}_best_val_loss': best_val,
            f'model_{model_id}_training_time_seconds': model_training_time
        })
        
        # Save model artifact
        model_path = f'{output_dir}/vae_model_{model_id}.pth'
        torch.save({
            'model': model.state_dict(),
            'seed': random_seed,
            'model_id': model_id,
            'best_val_loss': best_val
        }, model_path)
        
        mlflow.log_artifact(model_path)
        
        print(f"✓ VAE #{model_id} trained! Best Val Loss: {best_val:.4f}")
        print(f"✓ Logged to nested run: {nested_run.info.run_id}\n")
    
    return model


# ============================================
# ENSEMBLE GENERATION
# ============================================

def generate_ensemble_scenarios(models, scaler, features, scenarios_per_model=20, 
                                n_baseline=10, n_adverse=20, n_severe=50, n_extreme=20, device='cpu'):
    """Generate scenarios from ensemble of VAEs"""
    
    print("\n" + "="*60)
    print("GENERATING ENSEMBLE SCENARIOS")
    print("="*60)
    print(f"Number of models: {len(models)}")
    print(f"Scenarios per model: {scenarios_per_model}")
    print(f"Total scenarios: {len(models) * scenarios_per_model}\n")
    
    all_scenarios = []
    all_severities = []
    
    # Calculate scenarios per severity per model
    per_model_dist = {
        'baseline': n_baseline // len(models),
        'adverse': n_adverse // len(models),
        'severe': n_severe // len(models),
        'extreme': n_extreme // len(models)
    }
    
    for idx, model in enumerate(models):
        model.eval()
        latent_dim = model.decoder.model[0].in_features
        
        with torch.no_grad():
            # Baseline
            if per_model_dist['baseline'] > 0:
                z = torch.randn(per_model_dist['baseline'], latent_dim, device=device) * 0.5
                s = model.decoder(z).cpu().numpy()
                all_scenarios.append(s)
                all_severities.extend(['Baseline'] * per_model_dist['baseline'])
            
            # Adverse
            if per_model_dist['adverse'] > 0:
                z = torch.randn(per_model_dist['adverse'], latent_dim, device=device) * 1.5
                s = model.decoder(z).cpu().numpy()
                all_scenarios.append(s)
                all_severities.extend(['Adverse'] * per_model_dist['adverse'])
            
            # Severe
            if per_model_dist['severe'] > 0:
                z = torch.randn(per_model_dist['severe'], latent_dim, device=device) * 2.5
                s = model.decoder(z).cpu().numpy()
                all_scenarios.append(s)
                all_severities.extend(['Severe'] * per_model_dist['severe'])
            
            # Extreme
            if per_model_dist['extreme'] > 0:
                z = torch.randn(per_model_dist['extreme'], latent_dim, device=device) * 3.5
                s = model.decoder(z).cpu().numpy()
                all_scenarios.append(s)
                all_severities.extend(['Extreme'] * per_model_dist['extreme'])
        
        print(f"✓ Model {idx+1}/{len(models)} contributed {scenarios_per_model} scenarios")
    
    # Combine all scenarios
    all_scenarios = np.vstack(all_scenarios)
    
    # Denormalize
    scenarios_denorm = scaler.inverse_transform(all_scenarios)
    
    # Create DataFrame
    df = pd.DataFrame(scenarios_denorm, columns=features)
    df.insert(0, 'Scenario', [f'Scenario_{i+1}' for i in range(len(df))])
    df.insert(1, 'Severity', all_severities)
    
    print(f"\n✓ Total scenarios generated: {len(df)}")
    print(f"  - Baseline: {(df['Severity']=='Baseline').sum()}")
    print(f"  - Adverse: {(df['Severity']=='Adverse').sum()}")
    print(f"  - Severe: {(df['Severity']=='Severe').sum()}")
    print(f"  - Extreme: {(df['Severity']=='Extreme').sum()}\n")
    
    # Log scenario distribution to MLflow
    mlflow.log_metrics({
        'n_baseline_scenarios': int((df['Severity']=='Baseline').sum()),
        'n_adverse_scenarios': int((df['Severity']=='Adverse').sum()),
        'n_severe_scenarios': int((df['Severity']=='Severe').sum()),
        'n_extreme_scenarios': int((df['Severity']=='Extreme').sum()),
        'total_scenarios': len(df),
        'crisis_scenario_percentage': int((df['Severity'].isin(['Adverse', 'Severe', 'Extreme'])).sum())
    })
    
    return df


def classify_crisis_type(scenarios_df, feature_names, reference_data=None):
    """Classify each scenario by CRISIS TYPE using DATA-DRIVEN thresholds"""
    
    print("="*60)
    print("CLASSIFYING CRISIS TYPES (DATA-DRIVEN)")
    print("="*60 + "\n")
    
    # Use reference data if provided
    if reference_data is not None:
        ref_df = pd.DataFrame(reference_data, columns=feature_names)
        print("Using reference data (training set) for percentile thresholds")
    else:
        ref_df = scenarios_df
        print("Using generated scenarios for percentile thresholds")
    
    # Calculate thresholds
    thresholds = {}
    
    if 'GDP' in feature_names:
        thresholds['gdp_very_low'] = ref_df['GDP'].quantile(0.05)
        thresholds['gdp_low'] = ref_df['GDP'].quantile(0.25)
        thresholds['gdp_high'] = ref_df['GDP'].quantile(0.75)
        thresholds['gdp_very_high'] = ref_df['GDP'].quantile(0.95)
    
    if 'VIX' in feature_names:
        thresholds['vix_low'] = ref_df['VIX'].quantile(0.25)
        thresholds['vix_moderate'] = ref_df['VIX'].quantile(0.50)
        thresholds['vix_high'] = ref_df['VIX'].quantile(0.75)
        thresholds['vix_extreme'] = ref_df['VIX'].quantile(0.90)
    
    if 'Unemployment_Rate' in feature_names:
        thresholds['unemp_low'] = ref_df['Unemployment_Rate'].quantile(0.25)
        thresholds['unemp_moderate'] = ref_df['Unemployment_Rate'].quantile(0.50)
        thresholds['unemp_high'] = ref_df['Unemployment_Rate'].quantile(0.75)
        thresholds['unemp_extreme'] = ref_df['Unemployment_Rate'].quantile(0.90)
    
    if 'SP500_Close' in feature_names:
        thresholds['sp500_very_low'] = ref_df['SP500_Close'].quantile(0.05)
        thresholds['sp500_low'] = ref_df['SP500_Close'].quantile(0.25)
        thresholds['sp500_high'] = ref_df['SP500_Close'].quantile(0.75)
    
    if 'Oil_Price' in feature_names:
        thresholds['oil_very_low'] = ref_df['Oil_Price'].quantile(0.10)
        thresholds['oil_low'] = ref_df['Oil_Price'].quantile(0.25)
        thresholds['oil_high'] = ref_df['Oil_Price'].quantile(0.75)
        thresholds['oil_very_high'] = ref_df['Oil_Price'].quantile(0.90)
    
    print("Data-Driven Thresholds:")
    print("-" * 60)
    for key, value in list(thresholds.items())[:8]:
        print(f"  {key:25s}: {value:.2f}")
    if len(thresholds) > 8:
        print(f"  ... and {len(thresholds)-8} more thresholds")
    print()
    
    # Classify scenarios
    crisis_types = []
    crisis_scores = []
    
    for idx, row in scenarios_df.iterrows():
        gdp = row.get('GDP', thresholds.get('gdp_high', 18000))
        vix = row.get('VIX', thresholds.get('vix_moderate', 20))
        unemployment = row.get('Unemployment_Rate', thresholds.get('unemp_moderate', 5))
        
        # Simple classification logic
        if gdp > thresholds.get('gdp_high', 18000) and vix < thresholds.get('vix_low', 18):
            crisis_type = "Normal Economy"
            score = 1
        elif vix > thresholds.get('vix_extreme', 35):
            crisis_type = "Market Crash"
            score = 10
        elif unemployment > thresholds.get('unemp_extreme', 9):
            crisis_type = "Unemployment Crisis"
            score = 8
        else:
            crisis_type = "Moderate Stress"
            score = 4
        
        crisis_types.append(crisis_type)
        crisis_scores.append(score)
    
    scenarios_df.insert(2, 'Crisis_Type', crisis_types)
    scenarios_df.insert(3, 'Crisis_Score', crisis_scores)
    
    # Log to MLflow
    crisis_counts = scenarios_df['Crisis_Type'].value_counts()
    crisis_type_metrics = {}
    
    print("Crisis Type Distribution:")
    print("-" * 60)
    for crisis_type, count in crisis_counts.items():
        pct = 100 * count / len(scenarios_df)
        print(f"  {crisis_type:25s}: {count:3d} ({pct:5.1f}%)")
        crisis_type_key = crisis_type.lower().replace(' ', '_')
        crisis_type_metrics[f'crisis_type_{crisis_type_key}_count'] = int(count)
    
    mlflow.log_metrics({
        'unique_crisis_types': len(crisis_counts),
        'crisis_score_min': int(min(crisis_scores)),
        'crisis_score_max': int(max(crisis_scores)),
        'crisis_score_mean': float(np.mean(crisis_scores)),
        **crisis_type_metrics
    })
    
    print(f"\nTotal Unique Crisis Types: {len(crisis_counts)}")
    print("="*60 + "\n")
    
    return scenarios_df


# ============================================
# VALIDATION
# ============================================

def validate(real_data, gen_data, features, save_dir):
    """Validate scenarios and log to MLflow"""
    
    os.makedirs(save_dir, exist_ok=True)
    
    print("="*60)
    print("VALIDATION")
    print("="*60 + "\n")
    
    if isinstance(gen_data, pd.DataFrame):
        gen_data = gen_data.drop(['Scenario', 'Severity', 'Crisis_Type', 'Crisis_Score'], 
                                 axis=1, errors='ignore').values
    
    # KS tests
    ks_results = []
    for i in range(len(features)):
        stat, pval = ks_2samp(real_data[:, i], gen_data[:, i])
        ks_results.append({'feature': features[i], 'p_value': pval, 'passed': pval > 0.05})
    
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
    
    print(f"1. KS Pass Rate: {pass_rate:.1f}% ({ks_df['passed'].sum()}/{len(ks_df)})")
    print(f"2. Correlation MAE: {corr_mae:.4f}")
    print(f"3. Wasserstein: {w_mean:.2f}\n")
    
    # Log metrics to MLflow
    mlflow.log_metrics({
        'ks_pass_rate': pass_rate,
        'correlation_mae': corr_mae,
        'wasserstein_distance': w_mean,
        'n_features_passed_ks': int(ks_df['passed'].sum()),
        'n_features_total': len(ks_df)
    })
    
    # Save reports
    with open(f'{save_dir}/ensemble_validation.txt', 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("ENSEMBLE VAE VALIDATION REPORT\n")
        f.write("="*60 + "\n\n")
        f.write(f"KS Pass Rate: {pass_rate:.1f}%\n")
        f.write(f"Correlation MAE: {corr_mae:.4f}\n")
        f.write(f"Wasserstein: {w_mean:.2f}\n")
    
    ks_df.to_csv(f'{save_dir}/ensemble_ks_test_results.csv', index=False)
    
    print(f"✓ Saved: {save_dir}/ensemble_validation.txt")
    print(f"✓ Saved: {save_dir}/ensemble_ks_test_results.csv\n")
    
    return {'pass_rate': pass_rate, 'wasserstein': w_mean, 'correlation': corr_mae}


# ============================================
# MAIN ENSEMBLE PIPELINE
# ============================================

def main(csv_path, n_models=5, hidden_dims=[256, 128, 64], latent_dim=32,
         batch_size=128, epochs=300, beta=0.5, base_seed=42,
         n_baseline=10, n_adverse=20, n_severe=50, n_extreme=20,
         mlflow_tracking_uri=None, experiment_name="Financial_Stress_Test_Scenarios",
         output_dir='outputs/output_Ensemble_VAE'):
    """Train ensemble of VAEs with custom MLflow model logging"""
    
    cleanup_outputs(output_dir)
    
    print("="*60)
    print("ENSEMBLE VAE - FINANCIAL STRESS TESTING")
    print(f"Output Directory: {output_dir}")
    print("="*60)
    print(f"Training {n_models} independent VAEs")
    print(f"Expected improvement: 70.8% → 72-78% KS")
    print("="*60 + "\n")
    
    # Initialize MLflow
    mlflow_config = MLflowConfig(
        tracking_uri=mlflow_tracking_uri,
        experiment_name=experiment_name
    )
    
    # Start PARENT MLflow run
    run_name = f"Ensemble_VAE_{n_models}models_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    with mlflow.start_run(run_name=run_name) as parent_run:
        print(f"✓ MLflow Parent Run Started: {parent_run.info.run_id}")
        print(f"✓ Run Name: {run_name}\n")
        
        # Set tags
        mlflow_config.set_tags({
            'model_type': 'Ensemble_VAE',
            'n_models': n_models,
            'stage': 'development',
            'data_version': '1990_extended',
            'crisis_focus': 'True',
            'output_folder': output_dir
        })
        
        # Log parameters
        params = {
            'n_models': n_models,
            'hidden_dims': str(hidden_dims),
            'latent_dim': latent_dim,
            'batch_size': batch_size,
            'max_epochs': epochs,
            'beta': beta,
            'base_seed': base_seed,
            'n_baseline': n_baseline,
            'n_adverse': n_adverse,
            'n_severe': n_severe,
            'n_extreme': n_extreme
        }
        mlflow_config.log_params(params)
        
        start_time = time.time()
        
        # Load data
        train, val, test, features = load_data(csv_path, random_seed=base_seed)
        train_s, val_s, test_s, scaler = normalize_data(train, val, test)
        
        mlflow.log_metrics({
            'n_train_samples': len(train),
            'n_val_samples': len(val),
            'n_test_samples': len(test),
            'n_features': len(features)
        })
        
        # Create loaders
        train_loader = DataLoader(TensorDataset(torch.FloatTensor(train_s)), 
                                  batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(TensorDataset(torch.FloatTensor(val_s)), 
                                batch_size=batch_size, shuffle=False)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        input_dim = train_s.shape[1]
        
        # Train multiple VAEs with NESTED RUNS
        print("="*60)
        print(f"TRAINING {n_models} VAE MODELS")
        print("="*60)
        print(f"Expected total time: ~{n_models * 15} minutes\n")
        
        models = []
        seeds = [base_seed + i * 111 for i in range(n_models)]
        
        for i, seed in enumerate(seeds):
            model = train_single_vae(
                train_loader, val_loader, input_dim, hidden_dims, 
                latent_dim, epochs, beta, device, i+1, seed, output_dir
            )
            models.append(model)
        
        print("\n" + "="*60)
        print(f"✅ ALL {n_models} MODELS TRAINED!")
        print("="*60 + "\n")
        
        # Generate scenarios
        scenarios_per_model = (n_baseline + n_adverse + n_severe + n_extreme) // n_models
        scenarios = generate_ensemble_scenarios(
            models, scaler, features, scenarios_per_model,
            n_baseline, n_adverse, n_severe, n_extreme, device
        )
        
        # Classify crisis types
        scenarios = classify_crisis_type(scenarios, features, reference_data=train)
        
        # Validate
        results = validate(test, scenarios, features, save_dir=output_dir)
        
        training_time = time.time() - start_time
        mlflow.log_metric('training_time_seconds', training_time)
        
        # Save files
        print("="*60)
        print("SAVING")
        print("="*60 + "\n")
        
        scenarios.to_csv(f'{output_dir}/ensemble_vae_scenarios.csv', index=False)
        print(f"✓ Saved: {output_dir}/ensemble_vae_scenarios.csv")
        
        # Save scaler
        import pickle
        with open(f'{output_dir}/scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)
        print(f"✓ Saved: {output_dir}/scaler.pkl")
        
        # CREATE ENSEMBLE PACKAGE for MLflow custom model
        print("\nCreating ensemble package for MLflow...")
        ensemble_package = {
            'models': [model.state_dict() for model in models],
            'scaler': scaler,
            'features': features,
            'input_dim': input_dim,
            'config': {
                'n_models': n_models,
                'hidden_dims': hidden_dims,
                'latent_dim': latent_dim,
                'beta': beta,
                'seeds': seeds
            }
        }
        
        ensemble_package_path = f'{output_dir}/ensemble_vae_complete.pth'
        torch.save(ensemble_package, ensemble_package_path)
        print(f"✓ Saved ensemble package: {ensemble_package_path}")
        
        # LOG CUSTOM MLFLOW MODEL (Ensemble as pyfunc)
        print("\nLogging ensemble as MLflow custom model...")
        
        artifacts = {
            "ensemble_package": ensemble_package_path
        }
        
        conda_env = {
            'channels': ['defaults', 'conda-forge'],
            'dependencies': [
                'python=3.9',
                'pip',
                {
                    'pip': [
                        'mlflow',
                        'torch',
                        'numpy',
                        'pandas',
                        'scikit-learn'
                    ]
                }
            ],
            'name': 'ensemble_vae_env'
        }
        
        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=EnsembleVAEWrapper(),
            artifacts=artifacts,
            conda_env=conda_env
        )
        
        print("✅ Logged ensemble as MLflow pyfunc model!")
        print("   This model can be loaded and used for inference\n")
        
        # Log additional artifacts
        print("Logging additional artifacts...")
        mlflow_config.log_artifacts([
            f'{output_dir}/ensemble_vae_scenarios.csv',
            f'{output_dir}/ensemble_validation.txt',
            f'{output_dir}/ensemble_ks_test_results.csv'
        ])
        
        # Log individual model files
        for i in range(1, n_models + 1):
            model_path = f'{output_dir}/vae_model_{i}.pth'
            if os.path.exists(model_path):
                mlflow.log_artifact(model_path)
        
        print("✓ All artifacts logged to MLflow\n")
        
        # Results summary
        print("="*60)
        print("ENSEMBLE VAE RESULTS")
        print("="*60)
        print(f"KS Pass Rate: {results['pass_rate']:.1f}%")
        print(f"Correlation MAE: {results['correlation']:.4f}")
        print(f"Wasserstein: {results['wasserstein']:.1f}")
        print(f"\nTraining Time: {training_time:.1f}s ({training_time/60:.1f} min)")
        print(f"MLflow Run ID: {parent_run.info.run_id}")
        print(f"Output Directory: {output_dir}")
        print("="*60 + "\n")
        
        return models, scenarios, results, parent_run.info.run_id


# ============================================
# RUN
# ============================================

if __name__ == "__main__":
    
    from dotenv import load_dotenv
    
    load_dotenv()
    
    mlflow_uri = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000')
    experiment_name = os.getenv('MLFLOW_EXPERIMENT_NAME', 'Financial_Stress_Test_Scenarios')
    
    data_path = r'data/features/macro_features_clean.csv'
    
    print(f"Looking for data: {data_path}\n")
    print(f"MLflow Tracking URI: {mlflow_uri}\n")
    
    print("="*60)
    print("ENSEMBLE VAE APPROACH")
    print("="*60)
    print("Strategy: Train 5 independent VAEs, combine outputs")
    print("\nBenefits:")
    print("  ✅ More diverse scenarios")
    print("  ✅ Reduced overfitting")
    print("  ✅ Custom MLflow model for easy deployment")
    print("  ✅ Expected improvement: +2-7% KS")
    print("\nTrade-offs:")
    print("  ⚠  Takes 5x longer (75 minutes vs 15 minutes)")
    print("  ⚠  5 model files vs 1")
    print("="*60 + "\n")
    
    models, scenarios, results, run_id = main(
        csv_path=data_path,
        n_models=5,
        hidden_dims=[256, 128, 64],
        latent_dim=32,
        batch_size=128,
        epochs=300,
        beta=0.5,
        base_seed=42,
        n_baseline=10,
        n_adverse=20,
        n_severe=50,
        n_extreme=20,
        mlflow_tracking_uri=mlflow_uri,
        experiment_name=experiment_name,
        output_dir='outputs/output_Ensemble_VAE'
    )
    
    print("\n" + "="*60)
    print("🎉 ENSEMBLE VAE COMPLETE!")
    print("="*60)
    print(f"\nFinal Results:")
    print(f"  KS Pass Rate: {results['pass_rate']:.1f}%")
    print(f"  Correlation MAE: {results['correlation']:.4f}")
    print(f"  Wasserstein: {results['wasserstein']:.1f}")
    print(f"\nFiles saved in: outputs/output_Ensemble_VAE/")
    print(f"  - ensemble_vae_scenarios.csv")
    print(f"  - vae_model_1.pth to vae_model_5.pth")
    print(f"  - ensemble_validation.txt")
    print("\nMLflow:")
    print(f"  Run ID: {run_id}")
    print(f"  View at: {mlflow_uri}")
    print("\nTo load this model later:")
    print(f"  model = mlflow.pyfunc.load_model('runs:/{run_id}/model')")
    print("="*60)