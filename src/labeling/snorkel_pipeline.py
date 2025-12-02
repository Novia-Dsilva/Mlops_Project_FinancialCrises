#!/usr/bin/env python3
"""
============================================================================
Production-Grade Snorkel Weak Supervision Labeling Pipeline
============================================================================
Author: Parth Saraykar
Purpose: Generate weak supervision labels using EDA-derived thresholds
============================================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import yaml
import sys
import traceback
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns

from snorkel.labeling import PandasLFApplier, LFAnalysis
from snorkel.labeling.model import LabelModel

# Import labeling functions
try:
    from labeling_functions import get_enabled_lfs, ALL_LFS, ABSTAIN, NOT_AT_RISK, AT_RISK
except ImportError:
    print("Error: Could not import labeling_functions.py")
    print("Ensure labeling_functions.py is in the same directory or in Python path")
    sys.exit(1)

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class SnorkelConfig:
    """Configuration dataclass for Snorkel pipeline"""
    # Data paths
    input_path: str
    output_base_dir: str
    plots_dir: str
    data_dir: str
    reports_dir: str
    date_column: str
    company_column: str
    sector_column: str
    
    # Output filenames
    labeled_data_filename: str
    labeled_only_filename: str
    lf_summary_filename: str
    lf_analysis_filename: str
    
    # Label constants
    abstain: int
    not_at_risk: int
    at_risk: int
    cardinality: int
    
    # LabelModel parameters
    lm_epochs: int
    lm_lr: float
    lm_seed: int
    lm_log_freq: int
    
    # Quality thresholds
    min_coverage: float
    max_conflicts: float
    target_at_risk_rate_min: float
    target_at_risk_rate_max: float
    min_lf_coverage: float
    max_lf_conflicts: float
    
    # Validation periods
    validation_config: Dict
    
    # Logging
    log_level: str


def load_config(config_path: str = "configs/eda_config.yaml") -> SnorkelConfig:
    """Load Snorkel configuration from YAML"""
    try:
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        snorkel_config = config_dict['snorkel']
        logging_config = config_dict['logging']
        
        config = SnorkelConfig(
            # Data paths
            input_path=snorkel_config['data']['input_path'],
            output_base_dir=snorkel_config['output']['base_dir'],
            plots_dir=snorkel_config['output']['plots_dir'],
            data_dir=snorkel_config['output']['data_dir'],
            reports_dir=snorkel_config['output']['reports_dir'],
            date_column=snorkel_config['data']['date_column'],
            company_column=snorkel_config['data']['company_column'],
            sector_column=snorkel_config['data']['sector_column'],
            
            # Output filenames
            labeled_data_filename=snorkel_config['output']['labeled_data_filename'],
            labeled_only_filename=snorkel_config['output']['labeled_only_filename'],
            lf_summary_filename=snorkel_config['output']['lf_summary_filename'],
            lf_analysis_filename=snorkel_config['output']['lf_analysis_filename'],
            
            # Label constants
            abstain=snorkel_config['labels']['abstain'],
            not_at_risk=snorkel_config['labels']['not_at_risk'],
            at_risk=snorkel_config['labels']['at_risk'],
            cardinality=snorkel_config['labels']['cardinality'],
            
            # LabelModel parameters
            lm_epochs=snorkel_config['label_model']['epochs'],
            lm_lr=snorkel_config['label_model']['learning_rate'],
            lm_seed=snorkel_config['label_model']['seed'],
            lm_log_freq=snorkel_config['label_model']['log_freq'],
            
            # Quality thresholds
            min_coverage=snorkel_config['quality']['min_coverage'],
            max_conflicts=snorkel_config['quality']['max_conflicts'],
            target_at_risk_rate_min=snorkel_config['quality']['target_at_risk_rate_min'],
            target_at_risk_rate_max=snorkel_config['quality']['target_at_risk_rate_max'],
            min_lf_coverage=snorkel_config['quality']['min_lf_coverage'],
            max_lf_conflicts=snorkel_config['quality']['max_lf_conflicts'],
            
            # Validation
            validation_config=snorkel_config['validation'],
            
            # Logging
            log_level=logging_config['level']
        )
        
        return config
        
    except Exception as e:
        raise ValueError(f"Error loading configuration: {str(e)}")


# ============================================================================
# LOGGING SETUP
# ============================================================================

class SnorkelLogger:
    """Custom logger for Snorkel pipeline"""
    
    def __init__(self, log_dir: str = "logs", log_level: str = "INFO"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True, parents=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.log_dir / f"snorkel_{timestamp}.log"
        
        self.logger = logging.getLogger("Snorkel_Pipeline")
        self.logger.setLevel(getattr(logging, log_level.upper()))
        self.logger.handlers = []
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level.upper()))
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        self.logger.info(f"Logger initialized. Log file: {log_file}")
    
    def get_logger(self):
        return self.logger


def setup_directories(config: SnorkelConfig, logger: logging.Logger) -> None:
    """Create necessary output directories"""
    try:
        directories = [
            config.output_base_dir,
            config.plots_dir,
            config.data_dir,
            config.reports_dir
        ]
        
        for directory in directories:
            Path(directory).mkdir(exist_ok=True, parents=True)
            logger.debug(f"Created/verified directory: {directory}")
        
        logger.info("✓ All output directories created successfully")
        
    except Exception as e:
        logger.error(f"Error creating directories: {str(e)}")
        raise


# ============================================================================
# DATA LOADING
# ============================================================================

def load_data(config: SnorkelConfig, logger: logging.Logger) -> pd.DataFrame:
    """Load and prepare feature-engineered dataset"""
    logger.info(f"Loading data from: {config.input_path}")
    
    try:
        data_path = Path(config.input_path)
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {config.input_path}")
        
        df = pd.read_csv(data_path)
        logger.info(f"✓ Loaded {len(df):,} samples with {len(df.columns)} features")
        
        # Parse date and add temporal features
        df[config.date_column] = pd.to_datetime(df[config.date_column])
        df['Year'] = df[config.date_column].dt.year
        df['Quarter'] = df[config.date_column].dt.quarter
        
        logger.info(f"Date range: {df[config.date_column].min()} to {df[config.date_column].max()}")
        logger.info(f"Companies: {df[config.company_column].nunique()}")
        logger.info(f"Sectors: {df[config.sector_column].nunique()}")
        logger.info(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        return df
        
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        logger.debug(traceback.format_exc())
        raise


# ============================================================================
# APPLY LABELING FUNCTIONS
# ============================================================================

def apply_labeling_functions(
    df: pd.DataFrame,
    lfs: List,
    config: SnorkelConfig,
    logger: logging.Logger
) -> np.ndarray:
    """Apply all labeling functions to dataset"""
    logger.info(f"Applying {len(lfs)} labeling functions...")
    
    try:
        applier = PandasLFApplier(lfs=lfs)
        L_train = applier.apply(df=df)
        
        logger.info(f"✓ Label matrix shape: {L_train.shape}")
        logger.info(f"  Samples: {L_train.shape[0]:,}")
        logger.info(f"  Labeling functions: {L_train.shape[1]}")
        
        # Log sparsity
        abstain_rate = (L_train == config.abstain).sum() / L_train.size
        logger.info(f"  Abstain rate: {abstain_rate:.2%}")
        
        return L_train
        
    except Exception as e:
        logger.error(f"Error applying labeling functions: {str(e)}")
        logger.debug(traceback.format_exc())
        raise


# ============================================================================
# ANALYZE LABELING FUNCTION PERFORMANCE
# ============================================================================

def analyze_lf_performance(
    L_train: np.ndarray,
    lfs: List,
    config: SnorkelConfig,
    logger: logging.Logger
) -> pd.DataFrame:
    """Analyze coverage, conflicts, and overlaps"""
    logger.info("Analyzing labeling function performance...")
    
    try:
        lf_summary = LFAnalysis(L=L_train, lfs=lfs).lf_summary()
        
        logger.info("\n" + "="*80)
        logger.info("LABELING FUNCTION SUMMARY")
        logger.info("="*80)
        logger.info(f"\n{lf_summary.to_string()}\n")
        
        # Overall statistics
        coverage = (L_train != config.abstain).any(axis=1).sum() / len(L_train)
        conflicts = ((L_train != config.abstain).sum(axis=1) > 1).sum() / len(L_train)
        
        logger.info("="*80)
        logger.info("OVERALL STATISTICS")
        logger.info("="*80)
        logger.info(f"Coverage: {coverage:.2%} ({(L_train != config.abstain).any(axis=1).sum():,} samples)")
        logger.info(f"Conflicts: {conflicts:.2%} ({((L_train != config.abstain).sum(axis=1) > 1).sum():,} samples)")
        logger.info(f"Abstain rate: {(L_train == config.abstain).all(axis=1).sum() / len(L_train):.2%}")
        
        # Check against quality thresholds
        if coverage < config.min_coverage:
            logger.warning(f"⚠ Coverage {coverage:.2%} below target {config.min_coverage:.2%}")
        else:
            logger.info(f"✓ Coverage meets target ({config.min_coverage:.2%})")
        
        if conflicts > config.max_conflicts:
            logger.warning(f"⚠ Conflict rate {conflicts:.2%} above acceptable {config.max_conflicts:.2%}")
        else:
            logger.info(f"✓ Conflict rate acceptable (<{config.max_conflicts:.2%})")
        
        # Flag problematic LFs
        low_coverage_lfs = lf_summary[lf_summary['Coverage'] < config.min_lf_coverage]
        if len(low_coverage_lfs) > 0:
            logger.warning(f"⚠ {len(low_coverage_lfs)} LFs with coverage < {config.min_lf_coverage:.2%}:")
            for lf in low_coverage_lfs.index:
                logger.warning(f"    {lf}")
        
        high_conflict_lfs = lf_summary[lf_summary['Conflicts'] > config.max_lf_conflicts]
        if len(high_conflict_lfs) > 0:
            logger.warning(f"⚠ {len(high_conflict_lfs)} LFs with conflicts > {config.max_lf_conflicts:.2%}:")
            for lf in high_conflict_lfs.index:
                logger.warning(f"    {lf}")
        
        return lf_summary
        
    except Exception as e:
        logger.error(f"Error analyzing LF performance: {str(e)}")
        logger.debug(traceback.format_exc())
        raise


# ============================================================================
# TRAIN LABEL MODEL
# ============================================================================

def train_label_model(
    L_train: np.ndarray,
    config: SnorkelConfig,
    logger: logging.Logger
) -> LabelModel:
    """Train Snorkel LabelModel with error handling"""
    logger.info("Training Snorkel Label Model...")
    logger.info(f"Parameters: epochs={config.lm_epochs}, lr={config.lm_lr}, seed={config.lm_seed}")
    
    try:
        label_model = LabelModel(
            cardinality=config.cardinality,
            verbose=True
        )
        
        label_model.fit(
            L_train=L_train,
            n_epochs=config.lm_epochs,
            lr=config.lm_lr,
            seed=config.lm_seed,
            log_freq=config.lm_log_freq
        )
        
        logger.info("✓ Label model training completed")
        
        return label_model
        
    except Exception as e:
        logger.error(f"Error training label model: {str(e)}")
        logger.debug(traceback.format_exc())
        raise


# ============================================================================
# GENERATE LABELS
# ============================================================================

def generate_labels(
    label_model: LabelModel,
    L_train: np.ndarray,
    df: pd.DataFrame,
    config: SnorkelConfig,
    logger: logging.Logger
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate probabilistic and hard labels"""
    logger.info("Generating labels from trained model...")
    
    try:
        # Get probabilistic labels
        probs = label_model.predict_proba(L=L_train)
        
        # Get hard labels
        labels = label_model.predict(L=L_train, tie_break_policy="abstain")
        
        # Statistics
        n_labeled = (labels != config.abstain).sum()
        n_at_risk = (labels == config.at_risk).sum()
        n_not_at_risk = (labels == config.not_at_risk).sum()
        at_risk_rate = n_at_risk / n_labeled if n_labeled > 0 else 0
        
        logger.info("="*80)
        logger.info("LABEL GENERATION RESULTS")
        logger.info("="*80)
        logger.info(f"Total samples: {len(labels):,}")
        logger.info(f"Labeled samples: {n_labeled:,} ({n_labeled/len(labels):.2%})")
        logger.info(f"AT_RISK samples: {n_at_risk:,} ({at_risk_rate:.2%} of labeled)")
        logger.info(f"NOT_AT_RISK samples: {n_not_at_risk:,} ({n_not_at_risk/n_labeled:.2%} of labeled)")
        logger.info(f"Abstained samples: {(labels == config.abstain).sum():,}")
        
        # Check if at-risk rate is realistic
        if config.target_at_risk_rate_min <= at_risk_rate <= config.target_at_risk_rate_max:
            logger.info(f"✓ At-risk rate {at_risk_rate:.2%} is realistic")
        elif at_risk_rate < config.target_at_risk_rate_min:
            logger.warning(f"⚠ At-risk rate {at_risk_rate:.2%} is low (expected {config.target_at_risk_rate_min:.2%}-{config.target_at_risk_rate_max:.2%})")
        else:
            logger.warning(f"⚠ At-risk rate {at_risk_rate:.2%} is high (expected {config.target_at_risk_rate_min:.2%}-{config.target_at_risk_rate_max:.2%})")
        
        return probs, labels
        
    except Exception as e:
        logger.error(f"Error generating labels: {str(e)}")
        logger.debug(traceback.format_exc())
        raise


# ============================================================================
# VALIDATE LABELS
# ============================================================================

def validate_labels(
    df: pd.DataFrame,
    labels: np.ndarray,
    config: SnorkelConfig,
    logger: logging.Logger
) -> Dict:
    """Validate generated labels against historical crisis periods"""
    logger.info("Validating labels against historical crisis periods...")
    
    try:
        df_labeled = df.copy()
        df_labeled['snorkel_label'] = labels
        
        # Filter to labeled samples only
        df_labeled = df_labeled[df_labeled['snorkel_label'] != config.abstain]
        
        validation_results = {}
        
        # Validate crisis periods
        crisis_periods = config.validation_config['crisis_periods']
        
        for period_name, period_config in crisis_periods.items():
            start_year = period_config['start_year']
            end_year = period_config['end_year']
            
            period_df = df_labeled[
                (df_labeled['Year'] >= start_year) & 
                (df_labeled['Year'] <= end_year)
            ]
            
            if len(period_df) > 0:
                at_risk_rate = (period_df['snorkel_label'] == config.at_risk).sum() / len(period_df)
                expected_min = period_config['expected_at_risk_rate_min']
                expected_max = period_config['expected_at_risk_rate_max']
                
                validation_results[period_name] = {
                    'at_risk_rate': at_risk_rate,
                    'samples': len(period_df),
                    'passes_validation': expected_min <= at_risk_rate <= expected_max
                }
                
                logger.info(f"{period_name} ({start_year}-{end_year}): {at_risk_rate:.2%} labeled as AT_RISK")
                
                if expected_min <= at_risk_rate <= expected_max:
                    logger.info(f"  ✓ Within expected range ({expected_min:.2%}-{expected_max:.2%})")
                else:
                    logger.warning(f"  ⚠ Outside expected range ({expected_min:.2%}-{expected_max:.2%})")
        
        # Validate normal periods
        normal_periods = config.validation_config['normal_periods']
        
        for normal_config in normal_periods:
            years = normal_config['years']
            normal_df = df_labeled[df_labeled['Year'].isin(years)]
            
            if len(normal_df) > 0:
                at_risk_rate = (normal_df['snorkel_label'] == config.at_risk).sum() / len(normal_df)
                expected_max = normal_config['expected_at_risk_rate_max']
                
                validation_results[f"normal_{'-'.join(map(str, years))}"] = {
                    'at_risk_rate': at_risk_rate,
                    'samples': len(normal_df),
                    'passes_validation': at_risk_rate <= expected_max
                }
                
                logger.info(f"Normal periods {years}: {at_risk_rate:.2%} labeled as AT_RISK")
                
                if at_risk_rate <= expected_max:
                    logger.info(f"  ✓ Below expected maximum ({expected_max:.2%})")
                else:
                    logger.warning(f"  ⚠ Above expected maximum ({expected_max:.2%})")
        
        return validation_results
        
    except Exception as e:
        logger.error(f"Error validating labels: {str(e)}")
        logger.debug(traceback.format_exc())
        return {}


# ============================================================================
# SAVE RESULTS
# ============================================================================

def save_results(
    df: pd.DataFrame,
    labels: np.ndarray,
    probs: np.ndarray,
    lf_summary: pd.DataFrame,
    config: SnorkelConfig,
    logger: logging.Logger
) -> None:
    """Save labeled dataset and analysis results"""
    logger.info("Saving results...")
    
    try:
        # Add labels and probabilities to dataframe
        df_output = df.copy()
        df_output['snorkel_label'] = labels
        df_output['prob_not_at_risk'] = probs[:, 0]
        df_output['prob_at_risk'] = probs[:, 1]
        
        # Save complete labeled dataset
        output_path = Path(config.data_dir) / config.labeled_data_filename
        df_output.to_csv(output_path, index=False)
        logger.info(f"✓ Saved labeled data: {output_path}")
        
        # Save only labeled samples (excluding abstentions)
        df_labeled_only = df_output[df_output['snorkel_label'] != config.abstain]
        labeled_path = Path(config.data_dir) / config.labeled_only_filename
        df_labeled_only.to_csv(labeled_path, index=False)
        logger.info(f"✓ Saved labeled samples only ({len(df_labeled_only):,} samples): {labeled_path}")
        
        # Save LF summary
        lf_summary_path = Path(config.data_dir) / config.lf_summary_filename
        lf_summary.to_csv(lf_summary_path)
        logger.info(f"✓ Saved LF summary: {lf_summary_path}")
        
    except Exception as e:
        logger.error(f"Error saving results: {str(e)}")
        logger.debug(traceback.format_exc())
        raise


# ============================================================================
# VISUALIZATIONS
# ============================================================================

def create_visualizations(
    df: pd.DataFrame,
    labels: np.ndarray,
    lf_summary: pd.DataFrame,
    config: SnorkelConfig,
    logger: logging.Logger
) -> None:
    """Create comprehensive visualization plots"""
    logger.info("Creating visualizations...")
    
    try:
        df_viz = df.copy()
        df_viz['snorkel_label'] = labels
        df_viz_labeled = df_viz[df_viz['snorkel_label'] != config.abstain]
        
        # Plot 1: Labels over time
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        yearly_stats = df_viz_labeled.groupby('Year').agg({
            'snorkel_label': [
                ('total', 'count'),
                ('at_risk', lambda x: (x == config.at_risk).sum()),
                ('not_at_risk', lambda x: (x == config.not_at_risk).sum())
            ]
        })
        yearly_stats.columns = ['total', 'at_risk', 'not_at_risk']
        yearly_stats['at_risk_pct'] = yearly_stats['at_risk'] / yearly_stats['total'] * 100
        
        # Count plot
        axes[0].bar(yearly_stats.index, yearly_stats['at_risk'], 
                   color='darkred', alpha=0.7, label='AT_RISK', edgecolor='black')
        axes[0].axvline(x=2008, color='red', linestyle='--', linewidth=2, label='2008 Crisis')
        axes[0].axvline(x=2020, color='orange', linestyle='--', linewidth=2, label='2020 COVID')
        axes[0].set_xlabel('Year', fontsize=12)
        axes[0].set_ylabel('Number of AT_RISK Labels', fontsize=12)
        axes[0].set_title('Snorkel-Generated Labels Over Time (Count)', 
                         fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Rate plot
        axes[1].plot(yearly_stats.index, yearly_stats['at_risk_pct'], 
                    marker='o', linewidth=2, markersize=6, color='darkred')
        axes[1].axhline(y=yearly_stats['at_risk_pct'].mean(), 
                       color='blue', linestyle='--', linewidth=2, 
                       label=f'Average: {yearly_stats["at_risk_pct"].mean():.2f}%')
        axes[1].axvline(x=2008, color='red', linestyle='--', linewidth=2, alpha=0.5)
        axes[1].axvline(x=2020, color='orange', linestyle='--', linewidth=2, alpha=0.5)
        axes[1].fill_between(yearly_stats.index, 0, yearly_stats['at_risk_pct'], 
                            alpha=0.3, color='red')
        axes[1].set_xlabel('Year', fontsize=12)
        axes[1].set_ylabel('AT_RISK Rate (%)', fontsize=12)
        axes[1].set_title('AT_RISK Rate Over Time', fontsize=14, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(Path(config.plots_dir) / "labels_over_time.png", dpi=300)
        plt.close()
        logger.info("✓ Saved: labels_over_time.png")
        
        # Plot 2: LF performance
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        top_lfs = lf_summary.nlargest(15, 'Coverage')
        
        axes[0].barh(range(len(top_lfs)), top_lfs['Coverage'], color='steelblue', edgecolor='black')
        axes[0].set_yticks(range(len(top_lfs)))
        axes[0].set_yticklabels([lf.replace('lf_', '') for lf in top_lfs.index], fontsize=9)
        axes[0].set_xlabel('Coverage', fontsize=12)
        axes[0].set_title('Top 15 Labeling Functions by Coverage', fontsize=12, fontweight='bold')
        axes[0].grid(True, alpha=0.3, axis='x')
        
        axes[1].barh(range(len(top_lfs)), top_lfs['Conflicts'], color='coral', edgecolor='black')
        axes[1].set_yticks(range(len(top_lfs)))
        axes[1].set_yticklabels([lf.replace('lf_', '') for lf in top_lfs.index], fontsize=9)
        axes[1].set_xlabel('Conflicts', fontsize=12)
        axes[1].set_title('Top 15 Labeling Functions by Conflicts', fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig(Path(config.plots_dir) / "lf_performance.png", dpi=300)
        plt.close()
        logger.info("✓ Saved: lf_performance.png")
        
        # Plot 3: Label distribution by sector
        if config.sector_column in df_viz_labeled.columns:
            sector_stats = df_viz_labeled.groupby(config.sector_column).agg({
                'snorkel_label': [
                    ('total', 'count'),
                    ('at_risk', lambda x: (x == config.at_risk).sum())
                ]
            })
            sector_stats.columns = ['total', 'at_risk']
            sector_stats['at_risk_pct'] = sector_stats['at_risk'] / sector_stats['total'] * 100
            sector_stats = sector_stats.sort_values('at_risk_pct', ascending=False)
            
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.barh(range(len(sector_stats)), sector_stats['at_risk_pct'], color='coral', edgecolor='black')
            ax.set_yticks(range(len(sector_stats)))
            ax.set_yticklabels(sector_stats.index, fontsize=10)
            ax.set_xlabel('AT_RISK Rate (%)', fontsize=12)
            ax.set_title('AT_RISK Rate by Sector', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='x')
            
            plt.tight_layout()
            plt.savefig(Path(config.plots_dir) / "labels_by_sector.png", dpi=300)
            plt.close()
            logger.info("✓ Saved: labels_by_sector.png")
        
    except Exception as e:
        logger.error(f"Error creating visualizations: {str(e)}")
        logger.debug(traceback.format_exc())


# ============================================================================
# GENERATE REPORT
# ============================================================================

def generate_report(
    df: pd.DataFrame,
    labels: np.ndarray,
    lf_summary: pd.DataFrame,
    validation_results: Dict,
    config: SnorkelConfig,
    logger: logging.Logger
) -> str:
    """Generate comprehensive summary report"""
    logger.info("Generating summary report...")
    
    try:
        n_labeled = (labels != config.abstain).sum()
        n_at_risk = (labels == config.at_risk).sum()
        n_not_at_risk = (labels == config.not_at_risk).sum()
        at_risk_rate = n_at_risk / n_labeled if n_labeled > 0 else 0
        
        report = f"""
{'='*80}
SNORKEL WEAK SUPERVISION LABELING REPORT
Model 3: Anomaly Detection & Risk Scoring
{'='*80}
Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Author: Parth Saraykar

1. DATASET OVERVIEW
{'='*80}
Total Samples: {len(df):,}
Date Range: {df[config.date_column].min()} to {df[config.date_column].max()}
Companies: {df[config.company_column].nunique()}
Sectors: {df[config.sector_column].nunique()}

2. LABELING RESULTS
{'='*80}
Total Samples: {len(labels):,}
Labeled Samples: {n_labeled:,} ({n_labeled/len(labels):.2%})
AT_RISK Labels: {n_at_risk:,} ({at_risk_rate:.2%} of labeled)
NOT_AT_RISK Labels: {n_not_at_risk:,} ({n_not_at_risk/n_labeled:.2%} of labeled)
Abstained Samples: {(labels == config.abstain).sum():,}

3. LABELING FUNCTION PERFORMANCE
{'='*80}
Total LFs Applied: {len(lf_summary)}
Average Coverage: {lf_summary['Coverage'].mean():.2%}
Average Conflicts: {lf_summary['Conflicts'].mean():.2%}

Top 10 LFs by Coverage:
"""
        
        top_coverage = lf_summary.nlargest(10, 'Coverage')
        for idx, row in top_coverage.iterrows():
            report += f"  {idx:40s} | Coverage: {row['Coverage']:.2%}\n"
        
        report += f"\n4. VALIDATION RESULTS\n{'='*80}\n"
        
        for period_name, results in validation_results.items():
            status = "✓" if results['passes_validation'] else "⚠"
            report += f"{status} {period_name:30s} | AT_RISK: {results['at_risk_rate']:.2%} ({results['samples']:,} samples)\n"
        
        report += f"\n5. QUALITY METRICS\n{'='*80}\n"
        report += f"Coverage Target: {config.min_coverage:.2%} - {'✓ PASS' if n_labeled/len(labels) >= config.min_coverage else '✗ FAIL'}\n"
        report += f"At-Risk Rate Target: {config.target_at_risk_rate_min:.2%}-{config.target_at_risk_rate_max:.2%} - "
        if config.target_at_risk_rate_min <= at_risk_rate <= config.target_at_risk_rate_max:
            report += "✓ PASS\n"
        else:
            report += "✗ FAIL\n"
        
        report += f"\n6. OUTPUT FILES\n{'='*80}\n"
        report += f"✓ {config.labeled_data_filename}\n"
        report += f"✓ {config.labeled_only_filename}\n"
        report += f"✓ {config.lf_summary_filename}\n"
        report += f"✓ Visualization plots in {config.plots_dir}/\n"
        
        report += f"\n{'='*80}\nEND OF REPORT\n{'='*80}\n"
        
        # Save report
        report_path = Path(config.reports_dir) / "snorkel_labeling_report.txt"
        with open(report_path, 'w') as f:
            f.write(report)
        logger.info(f"✓ Saved report: {report_path}")
        
        return report
        
    except Exception as e:
        logger.error(f"Error generating report: {str(e)}")
        return f"Error generating report: {str(e)}"


# ============================================================================
# MAIN PIPELINE
# ============================================================================

class SnorkelPipeline:
    """Main Snorkel labeling pipeline orchestrator"""
    
    def __init__(self, config_path: str = "configs/eda_config.yaml"):
        """Initialize pipeline with configuration"""
        self.config = load_config(config_path)
        
        logger_instance = SnorkelLogger(log_level=self.config.log_level)
        self.logger = logger_instance.get_logger()
        
        self.logger.info("="*80)
        self.logger.info("SNORKEL WEAK SUPERVISION LABELING PIPELINE")
        self.logger.info("="*80)
        
        setup_directories(self.config, self.logger)
        
        self.results = {}
    
    def run(self) -> bool:
        """Execute complete Snorkel pipeline"""
        try:
            self.logger.info("Starting Snorkel pipeline execution...")
            
            # Step 1: Load data
            self.logger.info("\n" + "="*80)
            self.logger.info("STEP 1: DATA LOADING")
            self.logger.info("="*80)
            
            df = load_data(self.config, self.logger)
            self.results['dataframe'] = df
            
            # Step 2: Get enabled labeling functions
            self.logger.info("\n" + "="*80)
            self.logger.info("STEP 2: LOADING LABELING FUNCTIONS")
            self.logger.info("="*80)
            
            lfs = get_enabled_lfs()
            self.logger.info(f"Loaded {len(lfs)} enabled labeling functions")
            
            # Step 3: Apply labeling functions
            self.logger.info("\n" + "="*80)
            self.logger.info("STEP 3: APPLYING LABELING FUNCTIONS")
            self.logger.info("="*80)
            
            L_train = apply_labeling_functions(df, lfs, self.config, self.logger)
            self.results['L_train'] = L_train
            
            # Step 4: Analyze LF performance
            self.logger.info("\n" + "="*80)
            self.logger.info("STEP 4: ANALYZING LF PERFORMANCE")
            self.logger.info("="*80)
            
            lf_summary = analyze_lf_performance(L_train, lfs, self.config, self.logger)
            self.results['lf_summary'] = lf_summary
            
            # Step 5: Train label model
            self.logger.info("\n" + "="*80)
            self.logger.info("STEP 5: TRAINING LABEL MODEL")
            self.logger.info("="*80)
            
            label_model = train_label_model(L_train, self.config, self.logger)
            self.results['label_model'] = label_model
            
            # Step 6: Generate labels
            self.logger.info("\n" + "="*80)
            self.logger.info("STEP 6: GENERATING LABELS")
            self.logger.info("="*80)
            
            probs, labels = generate_labels(label_model, L_train, df, self.config, self.logger)
            self.results['probs'] = probs
            self.results['labels'] = labels
            
            # Step 7: Validate labels
            self.logger.info("\n" + "="*80)
            self.logger.info("STEP 7: VALIDATING LABELS")
            self.logger.info("="*80)
            
            validation_results = validate_labels(df, labels, self.config, self.logger)
            self.results['validation_results'] = validation_results
            
            # Step 8: Save results
            self.logger.info("\n" + "="*80)
            self.logger.info("STEP 8: SAVING RESULTS")
            self.logger.info("="*80)
            
            save_results(df, labels, probs, lf_summary, self.config, self.logger)
            
            # Step 9: Create visualizations
            self.logger.info("\n" + "="*80)
            self.logger.info("STEP 9: CREATING VISUALIZATIONS")
            self.logger.info("="*80)
            
            create_visualizations(df, labels, lf_summary, self.config, self.logger)
            
            # Step 10: Generate report
            self.logger.info("\n" + "="*80)
            self.logger.info("STEP 10: GENERATING REPORT")
            self.logger.info("="*80)
            
            report = generate_report(df, labels, lf_summary, validation_results, 
                                   self.config, self.logger)
            print("\n" + report)
            
            self.logger.info("\n" + "="*80)
            self.logger.info("✓ SNORKEL PIPELINE COMPLETED SUCCESSFULLY!")
            self.logger.info("="*80)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {str(e)}")
            self.logger.debug(traceback.format_exc())
            return False
    
    def get_results(self) -> Dict:
        """Get pipeline results"""
        return self.results


# ============================================================================
# ENTRY POINT
# ============================================================================

def main():
    """Main entry point"""
    try:
        pipeline = SnorkelPipeline(config_path="configs/eda_config.yaml")
        success = pipeline.run()
        
        if success:
            print("\n✓ Snorkel labeling completed successfully!")
            print("Check the following directories for outputs:")
            print("  - outputs/snorkel/data/     - Labeled datasets")
            print("  - outputs/snorkel/plots/    - Visualizations")
            print("  - outputs/snorkel/reports/  - Summary report")
            print("  - logs/                     - Execution logs")
            sys.exit(0)
        else:
            print("\n✗ Snorkel pipeline failed. Check logs for details.")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Fatal error: {str(e)}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()