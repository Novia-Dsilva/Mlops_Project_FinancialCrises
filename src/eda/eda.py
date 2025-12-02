#!/usr/bin/env python3
"""
============================================================================
Production-Grade EDA for Model 3: Anomaly Detection & Risk Scoring
============================================================================
Author: Parth Saraykar
Purpose: Comprehensive exploratory data analysis with robust error handling,
         logging, and validation
============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union
import yaml
import logging
import sys
import warnings
import traceback
from dataclasses import dataclass, asdict
import json

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION AND SETUP
# ============================================================================

@dataclass
class EDAConfig:
    """Configuration dataclass for EDA pipeline"""
    input_path: str
    output_base_dir: str
    plots_dir: str
    data_dir: str
    reports_dir: str
    date_column: str
    crisis_flag_column: str
    company_column: str
    sector_column: str
    risk_features: List[str]
    correlation_threshold: float
    outlier_zscore_threshold: float
    significance_level: float
    crisis_rate_min: float
    crisis_rate_max: float
    log_level: str
    dpi: int
    figure_format: str


class EDALogger:
    """Custom logger with console and file handlers"""
    
    def __init__(self, log_dir: str = "logs", log_level: str = "INFO"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True, parents=True)
        
        # Create timestamped log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.log_dir / f"eda_{timestamp}.log"
        
        # Configure logger
        self.logger = logging.getLogger("EDA_Pipeline")
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # Remove existing handlers
        self.logger.handlers = []
        
        # Create formatters
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
        file_handler.setLevel(logging.DEBUG)  # Always log everything to file
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        self.logger.info(f"Logger initialized. Log file: {log_file}")
    
    def get_logger(self):
        return self.logger


def load_config(config_path: str = "configs/eda_config.yaml") -> EDAConfig:
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        EDAConfig object
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config is invalid
    """
    config_file = Path(config_path)
    
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    try:
        with open(config_file, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Get EDA config section
        eda_config = config_dict['eda']
        
        # Flatten risk features
        risk_features = []
        for category in eda_config['features']['risk_features'].values():
            risk_features.extend(category)
        
        # Create EDAConfig object
        config = EDAConfig(
            input_path=eda_config['data']['input_path'],
            output_base_dir=eda_config['output']['base_dir'],
            plots_dir=eda_config['output']['plots_dir'],
            data_dir=eda_config['output']['data_dir'],
            reports_dir=eda_config['output']['reports_dir'],
            date_column=eda_config['data']['date_column'],
            crisis_flag_column=eda_config['data']['crisis_flag_column'],
            company_column=eda_config['data']['company_column'],
            sector_column=eda_config['data']['sector_column'],
            risk_features=risk_features,
            correlation_threshold=eda_config['analysis']['correlation_threshold'],
            outlier_zscore_threshold=eda_config['analysis']['outlier_zscore_threshold'],
            significance_level=eda_config['analysis']['significance_level'],
            crisis_rate_min=eda_config['analysis']['crisis_rate_threshold']['min'],
            crisis_rate_max=eda_config['analysis']['crisis_rate_threshold']['max'],
            log_level=config_dict['logging']['level'],
            dpi=eda_config['visualization']['dpi'],
            figure_format=eda_config['visualization']['figure_format']
        )
        
        return config
        
    except Exception as e:
        raise ValueError(f"Error loading configuration: {str(e)}")


def setup_directories(config: EDAConfig, logger: logging.Logger) -> None:
    """
    Create necessary output directories
    
    Args:
        config: EDAConfig object
        logger: Logger instance
    """
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
        
        logger.info("All output directories created successfully")
        
    except Exception as e:
        logger.error(f"Error creating directories: {str(e)}")
        raise


# ============================================================================
# DATA LOADING AND VALIDATION
# ============================================================================

class DataValidator:
    """Validate input data quality and structure"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.validation_results = {
            'passed': [],
            'warnings': [],
            'errors': []
        }
    
    def validate_dataframe(self, df: pd.DataFrame, config: EDAConfig) -> Tuple[bool, Dict]:
        """
        Comprehensive dataframe validation
        
        Args:
            df: Input dataframe
            config: EDAConfig object
            
        Returns:
            Tuple of (is_valid, validation_results)
        """
        self.logger.info("Starting dataframe validation...")
        
        try:
            # Check 1: DataFrame is not empty
            if df.empty:
                self.validation_results['errors'].append("DataFrame is empty")
                return False, self.validation_results
            self.validation_results['passed'].append("DataFrame is not empty")
            
            # Check 2: Required columns exist
            required_cols = [
                config.date_column,
                config.company_column,
                config.sector_column
            ]
            
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                self.validation_results['errors'].append(
                    f"Missing required columns: {missing_cols}"
                )
                return False, self.validation_results
            self.validation_results['passed'].append("All required columns present")
            
            # Check 3: Date column is valid
            try:
                pd.to_datetime(df[config.date_column])
                self.validation_results['passed'].append("Date column is valid")
            except Exception as e:
                self.validation_results['errors'].append(
                    f"Invalid date column: {str(e)}"
                )
                return False, self.validation_results
            
            # Check 4: Minimum sample size
            if len(df) < 100:
                self.validation_results['warnings'].append(
                    f"Small dataset: only {len(df)} samples"
                )
            else:
                self.validation_results['passed'].append(
                    f"Sufficient samples: {len(df)}"
                )
            
            # Check 5: Missing values analysis
            missing_pct = (df.isnull().sum() / len(df) * 100)
            critical_missing = missing_pct[missing_pct > 50]
            
            if len(critical_missing) > 0:
                self.validation_results['warnings'].append(
                    f"{len(critical_missing)} columns with >50% missing values"
                )
            
            # Check 6: Duplicate rows
            n_duplicates = df.duplicated().sum()
            if n_duplicates > 0:
                self.validation_results['warnings'].append(
                    f"Found {n_duplicates} duplicate rows"
                )
            else:
                self.validation_results['passed'].append("No duplicate rows")
            
            # Check 7: Crisis flag validation (if exists)
            if config.crisis_flag_column in df.columns:
                unique_values = df[config.crisis_flag_column].unique()
                if set(unique_values).issubset({0, 1, np.nan}):
                    self.validation_results['passed'].append(
                        "Crisis flag column is binary"
                    )
                else:
                    self.validation_results['warnings'].append(
                        f"Crisis flag has unexpected values: {unique_values}"
                    )
            
            # Check 8: Feature availability
            available_features = [f for f in config.risk_features if f in df.columns]
            missing_features = [f for f in config.risk_features if f not in df.columns]
            
            if missing_features:
                self.validation_results['warnings'].append(
                    f"{len(missing_features)} risk features not found in data"
                )
            
            self.validation_results['passed'].append(
                f"{len(available_features)}/{len(config.risk_features)} risk features available"
            )
            
            self.logger.info("Dataframe validation completed")
            return True, self.validation_results
            
        except Exception as e:
            self.logger.error(f"Validation error: {str(e)}")
            self.validation_results['errors'].append(f"Validation error: {str(e)}")
            return False, self.validation_results
    
    def log_validation_results(self) -> None:
        """Log validation results summary"""
        self.logger.info("=" * 80)
        self.logger.info("VALIDATION RESULTS SUMMARY")
        self.logger.info("=" * 80)
        
        self.logger.info(f"✓ Passed checks: {len(self.validation_results['passed'])}")
        for check in self.validation_results['passed']:
            self.logger.info(f"  ✓ {check}")
        
        if self.validation_results['warnings']:
            self.logger.warning(f"⚠ Warnings: {len(self.validation_results['warnings'])}")
            for warning in self.validation_results['warnings']:
                self.logger.warning(f"  ⚠ {warning}")
        
        if self.validation_results['errors']:
            self.logger.error(f"✗ Errors: {len(self.validation_results['errors'])}")
            for error in self.validation_results['errors']:
                self.logger.error(f"  ✗ {error}")


def load_data(config: EDAConfig, logger: logging.Logger) -> pd.DataFrame:
    """
    Load and perform initial preprocessing of data
    
    Args:
        config: EDAConfig object
        logger: Logger instance
        
    Returns:
        Loaded dataframe
        
    Raises:
        FileNotFoundError: If data file doesn't exist
        pd.errors.EmptyDataError: If file is empty
    """
    logger.info(f"Loading data from: {config.input_path}")
    
    try:
        # Check if file exists
        data_path = Path(config.input_path)
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {config.input_path}")
        
        # Load data
        df = pd.read_csv(data_path)
        logger.info(f"✓ Data loaded successfully: {df.shape[0]:,} rows × {df.shape[1]} columns")
        
        # Convert date column
        try:
            df[config.date_column] = pd.to_datetime(df[config.date_column])
            df['Year'] = df[config.date_column].dt.year
            df['Quarter'] = df[config.date_column].dt.quarter
            logger.info("✓ Date column parsed successfully")
        except Exception as e:
            logger.error(f"Error parsing date column: {str(e)}")
            raise
        
        # Log basic info
        logger.info(f"Date range: {df[config.date_column].min()} to {df[config.date_column].max()}")
        logger.info(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        return df
        
    except FileNotFoundError as e:
        logger.error(str(e))
        raise
    except pd.errors.EmptyDataError as e:
        logger.error("Data file is empty")
        raise
    except Exception as e:
        logger.error(f"Unexpected error loading data: {str(e)}")
        logger.debug(traceback.format_exc())
        raise


# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

class CrisisAnalyzer:
    """Analyze crisis flag distribution and patterns"""
    
    def __init__(self, df: pd.DataFrame, config: EDAConfig, logger: logging.Logger):
        self.df = df
        self.config = config
        self.logger = logger
        self.has_crisis_flag = config.crisis_flag_column in df.columns
    
    def analyze_crisis_distribution(self) -> Optional[Dict]:
        """
        Analyze crisis flag distribution
        
        Returns:
            Dictionary with crisis statistics or None if no crisis flag
        """
        if not self.has_crisis_flag:
            self.logger.warning(f"Crisis flag column '{self.config.crisis_flag_column}' not found")
            return None
        
        try:
            self.logger.info("Analyzing crisis distribution...")
            
            crisis_df = self.df[self.df[self.config.crisis_flag_column] == 1]
            normal_df = self.df[self.df[self.config.crisis_flag_column] == 0]
            
            crisis_rate = len(crisis_df) / len(self.df) * 100
            
            results = {
                'total_samples': len(self.df),
                'crisis_samples': len(crisis_df),
                'normal_samples': len(normal_df),
                'crisis_rate': crisis_rate,
                'is_realistic': self.config.crisis_rate_min <= crisis_rate <= self.config.crisis_rate_max
            }
            
            # Log results
            self.logger.info(f"Crisis samples: {results['crisis_samples']:,} ({crisis_rate:.2f}%)")
            self.logger.info(f"Normal samples: {results['normal_samples']:,} ({100-crisis_rate:.2f}%)")
            
            if results['is_realistic']:
                self.logger.info("✓ Crisis rate is within realistic range")
            elif crisis_rate < self.config.crisis_rate_min:
                self.logger.warning(f"⚠ Crisis rate is low (expected: {self.config.crisis_rate_min}-{self.config.crisis_rate_max}%)")
            else:
                self.logger.warning(f"⚠ Crisis rate is high (expected: {self.config.crisis_rate_min}-{self.config.crisis_rate_max}%)")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error analyzing crisis distribution: {str(e)}")
            self.logger.debug(traceback.format_exc())
            return None
    
    def analyze_temporal_distribution(self) -> Optional[pd.DataFrame]:
        """
        Analyze crisis distribution over time
        
        Returns:
            DataFrame with yearly crisis statistics
        """
        if not self.has_crisis_flag:
            return None
        
        try:
            self.logger.info("Analyzing temporal crisis distribution...")
            
            crisis_by_year = self.df.groupby('Year')[self.config.crisis_flag_column].agg([
                ('total', 'count'),
                ('crisis_count', 'sum'),
                ('crisis_rate', 'mean')
            ])
            crisis_by_year['crisis_pct'] = crisis_by_year['crisis_rate'] * 100
            
            # Identify peak crisis years
            top_years = crisis_by_year.nlargest(3, 'crisis_count')
            self.logger.info(f"Peak crisis years: {', '.join(map(str, top_years.index.tolist()))}")
            
            return crisis_by_year
            
        except Exception as e:
            self.logger.error(f"Error analyzing temporal distribution: {str(e)}")
            return None
    
    def compare_crisis_vs_normal(self, features: List[str]) -> Optional[pd.DataFrame]:
        """
        Statistical comparison between crisis and normal periods
        
        Args:
            features: List of features to compare
            
        Returns:
            DataFrame with comparison statistics
        """
        if not self.has_crisis_flag:
            return None
        
        try:
            self.logger.info(f"Comparing crisis vs normal for {len(features)} features...")
            
            crisis_df = self.df[self.df[self.config.crisis_flag_column] == 1]
            normal_df = self.df[self.df[self.config.crisis_flag_column] == 0]
            
            # Filter to available features
            available_features = [f for f in features if f in self.df.columns]
            self.logger.info(f"Analyzing {len(available_features)}/{len(features)} available features")
            
            comparison = []
            for feature in available_features:
                try:
                    crisis_data = crisis_df[feature].dropna()
                    normal_data = normal_df[feature].dropna()
                    
                    if len(crisis_data) == 0 or len(normal_data) == 0:
                        self.logger.debug(f"Skipping {feature}: insufficient data")
                        continue
                    
                    crisis_mean = crisis_data.mean()
                    normal_mean = normal_data.mean()
                    crisis_std = crisis_data.std()
                    normal_std = normal_data.std()
                    
                    # Perform t-test
                    t_stat, p_value = stats.ttest_ind(crisis_data, normal_data, equal_var=False)
                    
                    diff = crisis_mean - normal_mean
                    pct_diff = (diff / abs(normal_mean) * 100) if normal_mean != 0 else np.nan
                    
                    # Determine significance
                    if p_value < 0.001:
                        significance = '***'
                    elif p_value < 0.01:
                        significance = '**'
                    elif p_value < self.config.significance_level:
                        significance = '*'
                    else:
                        significance = ''
                    
                    comparison.append({
                        'Feature': feature,
                        'Crisis_Mean': crisis_mean,
                        'Normal_Mean': normal_mean,
                        'Crisis_Std': crisis_std,
                        'Normal_Std': normal_std,
                        'Difference': diff,
                        'Pct_Diff': pct_diff,
                        'T_Statistic': t_stat,
                        'P_Value': p_value,
                        'Significant': significance
                    })
                    
                except Exception as e:
                    self.logger.debug(f"Error comparing {feature}: {str(e)}")
                    continue
            
            if not comparison:
                self.logger.warning("No features could be compared")
                return None
            
            comparison_df = pd.DataFrame(comparison)
            comparison_df = comparison_df.sort_values('Pct_Diff', key=abs, ascending=False)
            
            # Log top differentiating features
            self.logger.info("Top 5 differentiating features:")
            for i, row in comparison_df.head(5).iterrows():
                self.logger.info(f"  {row['Feature']:30s} | Δ = {row['Pct_Diff']:+7.2f}% {row['Significant']}")
            
            return comparison_df
            
        except Exception as e:
            self.logger.error(f"Error in crisis vs normal comparison: {str(e)}")
            self.logger.debug(traceback.format_exc())
            return None


class CorrelationAnalyzer:
    """Analyze feature correlations"""
    
    def __init__(self, df: pd.DataFrame, config: EDAConfig, logger: logging.Logger):
        self.df = df
        self.config = config
        self.logger = logger
    
    def compute_correlation_matrix(self, features: List[str]) -> Optional[pd.DataFrame]:
        """
        Compute correlation matrix for features
        
        Args:
            features: List of features
            
        Returns:
            Correlation matrix or None if error
        """
        try:
            self.logger.info("Computing correlation matrix...")
            
            # Filter to available features
            available_features = [f for f in features if f in self.df.columns]
            
            if not available_features:
                self.logger.error("No features available for correlation analysis")
                return None
            
            corr_matrix = self.df[available_features].corr()
            self.logger.info(f"✓ Correlation matrix computed for {len(available_features)} features")
            
            return corr_matrix
            
        except Exception as e:
            self.logger.error(f"Error computing correlation matrix: {str(e)}")
            return None
    
    def find_high_correlations(self, corr_matrix: pd.DataFrame) -> List[Dict]:
        """
        Find highly correlated feature pairs
        
        Args:
            corr_matrix: Correlation matrix
            
        Returns:
            List of high correlation pairs
        """
        try:
            self.logger.info(f"Finding correlations above {self.config.correlation_threshold}...")
            
            high_corr = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i + 1, len(corr_matrix.columns)):
                    corr_value = corr_matrix.iloc[i, j]
                    if abs(corr_value) > self.config.correlation_threshold:
                        high_corr.append({
                            'Feature1': corr_matrix.columns[i],
                            'Feature2': corr_matrix.columns[j],
                            'Correlation': corr_value
                        })
            
            if high_corr:
                self.logger.warning(f"Found {len(high_corr)} highly correlated pairs")
                for pair in sorted(high_corr, key=lambda x: abs(x['Correlation']), reverse=True)[:5]:
                    self.logger.warning(f"  {pair['Feature1']} ↔ {pair['Feature2']}: {pair['Correlation']:.3f}")
            else:
                self.logger.info("✓ No highly correlated feature pairs found")
            
            return high_corr
            
        except Exception as e:
            self.logger.error(f"Error finding high correlations: {str(e)}")
            return []


class OutlierAnalyzer:
    """Detect and analyze outliers"""
    
    def __init__(self, df: pd.DataFrame, config: EDAConfig, logger: logging.Logger):
        self.df = df
        self.config = config
        self.logger = logger
    
    def detect_outliers_iqr(self, series: pd.Series) -> Tuple[int, float]:
        """
        Detect outliers using IQR method
        
        Args:
            series: Pandas Series
            
        Returns:
            Tuple of (outlier_count, outlier_percentage)
        """
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = series[(series < lower_bound) | (series > upper_bound)]
        return len(outliers), len(outliers) / len(series) * 100
    
    def detect_outliers_zscore(self, series: pd.Series) -> Tuple[int, float]:
        """
        Detect outliers using Z-score method
        
        Args:
            series: Pandas Series
            
        Returns:
            Tuple of (outlier_count, outlier_percentage)
        """
        z_scores = np.abs(stats.zscore(series.dropna()))
        outliers = series[z_scores > self.config.outlier_zscore_threshold]
        return len(outliers), len(outliers) / len(series) * 100
    
    def analyze_outliers(self, features: List[str]) -> Optional[pd.DataFrame]:
        """
        Analyze outliers across features
        
        Args:
            features: List of features to analyze
            
        Returns:
            DataFrame with outlier statistics
        """
        try:
            self.logger.info(f"Analyzing outliers for {len(features)} features...")
            
            available_features = [f for f in features if f in self.df.columns]
            
            outlier_summary = []
            for feature in available_features:
                try:
                    data = self.df[feature].dropna()
                    
                    if len(data) == 0:
                        continue
                    
                    n_iqr, pct_iqr = self.detect_outliers_iqr(data)
                    n_zscore, pct_zscore = self.detect_outliers_zscore(data)
                    
                    outlier_summary.append({
                        'Feature': feature,
                        'IQR_Count': n_iqr,
                        'IQR_Pct': pct_iqr,
                        'ZScore_Count': n_zscore,
                        'ZScore_Pct': pct_zscore
                    })
                    
                except Exception as e:
                    self.logger.debug(f"Error analyzing outliers for {feature}: {str(e)}")
                    continue
            
            if not outlier_summary:
                self.logger.warning("No outlier analysis results")
                return None
            
            outlier_df = pd.DataFrame(outlier_summary)
            
            # Log features with high outlier rates
            high_outliers = outlier_df[outlier_df['ZScore_Pct'] > 5]
            if len(high_outliers) > 0:
                self.logger.warning(f"{len(high_outliers)} features with >5% outliers")
            
            self.logger.info(f"✓ Outlier analysis completed for {len(outlier_df)} features")
            
            return outlier_df
            
        except Exception as e:
            self.logger.error(f"Error in outlier analysis: {str(e)}")
            return None


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

class Visualizer:
    """Create and save visualizations"""
    
    def __init__(self, config: EDAConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        
        # Set matplotlib style
        try:
            plt.style.use('seaborn-v0_8-darkgrid')
        except:
            self.logger.warning("Could not set matplotlib style")
        
        sns.set_palette("husl")
    
    def save_plot(self, fig: plt.Figure, filename: str) -> bool:
        """
        Save plot to file with error handling
        
        Args:
            fig: Matplotlib figure
            filename: Output filename
            
        Returns:
            Success status
        """
        try:
            output_path = Path(self.config.plots_dir) / filename
            fig.savefig(output_path, dpi=self.config.dpi, bbox_inches='tight')
            self.logger.info(f"✓ Saved plot: {output_path}")
            plt.close(fig)
            return True
        except Exception as e:
            self.logger.error(f"Error saving plot {filename}: {str(e)}")
            return False
    
    def plot_crisis_over_time(self, crisis_by_year: pd.DataFrame) -> Optional[plt.Figure]:
        """
        Plot crisis distribution over time
        
        Args:
            crisis_by_year: DataFrame with yearly crisis stats
            
        Returns:
            Figure object or None
        """
        try:
            self.logger.info("Creating crisis over time plot...")
            
            fig, axes = plt.subplots(2, 1, figsize=(15, 10))
            
            # Plot 1: Crisis count
            axes[0].bar(crisis_by_year.index, crisis_by_year['crisis_count'], 
                       color='darkred', alpha=0.7, edgecolor='black')
            axes[0].axvline(x=2008, color='red', linestyle='--', linewidth=2, 
                           label='2008 Financial Crisis')
            axes[0].axvline(x=2020, color='orange', linestyle='--', linewidth=2, 
                           label='2020 COVID Crisis')
            axes[0].set_xlabel('Year', fontsize=12)
            axes[0].set_ylabel('Number of Crisis Cases', fontsize=12)
            axes[0].set_title('Crisis Cases by Year', fontsize=14, fontweight='bold')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # Plot 2: Crisis rate
            axes[1].plot(crisis_by_year.index, crisis_by_year['crisis_pct'], 
                        marker='o', linewidth=2, markersize=6, color='darkred')
            axes[1].axhline(y=crisis_by_year['crisis_pct'].mean(), 
                           color='blue', linestyle='--', linewidth=2, 
                           label=f'Average: {crisis_by_year["crisis_pct"].mean():.2f}%')
            axes[1].axvline(x=2008, color='red', linestyle='--', linewidth=2, alpha=0.5)
            axes[1].axvline(x=2020, color='orange', linestyle='--', linewidth=2, alpha=0.5)
            axes[1].fill_between(crisis_by_year.index, 0, crisis_by_year['crisis_pct'], 
                                alpha=0.3, color='red')
            axes[1].set_xlabel('Year', fontsize=12)
            axes[1].set_ylabel('Crisis Rate (%)', fontsize=12)
            axes[1].set_title('Crisis Rate Over Time', fontsize=14, fontweight='bold')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating crisis over time plot: {str(e)}")
            return None
    
    def plot_crisis_vs_normal_distributions(
        self, 
        df: pd.DataFrame, 
        features: List[str],
        crisis_column: str,
        comparison_df: pd.DataFrame
    ) -> Optional[plt.Figure]:
        """
        Plot distribution comparisons between crisis and normal
        
        Args:
            df: Input dataframe
            features: Features to plot
            crisis_column: Name of crisis flag column
            comparison_df: Comparison statistics dataframe
            
        Returns:
            Figure object or None
        """
        try:
            self.logger.info(f"Creating crisis vs normal distributions for {len(features)} features...")
            
            crisis_df = df[df[crisis_column] == 1]
            normal_df = df[df[crisis_column] == 0]
            
            n_features = min(12, len(features))
            n_cols = 4
            n_rows = (n_features + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
            axes = axes.flatten() if n_features > 1 else [axes]
            
            for idx, feature in enumerate(features[:n_features]):
                ax = axes[idx]
                
                normal_data = normal_df[feature].dropna()
                crisis_data = crisis_df[feature].dropna()
                
                if len(crisis_data) > 0 and len(normal_data) > 0:
                    # Plot histograms
                    ax.hist(normal_data, bins=50, alpha=0.6, 
                           label=f'Normal (n={len(normal_data)})', 
                           color='green', density=True, edgecolor='black', linewidth=0.5)
                    ax.hist(crisis_data, bins=50, alpha=0.6, 
                           label=f'Crisis (n={len(crisis_data)})', 
                           color='red', density=True, edgecolor='black', linewidth=0.5)
                    
                    # Add mean lines
                    ax.axvline(normal_data.mean(), color='darkgreen', 
                              linestyle='--', linewidth=2)
                    ax.axvline(crisis_data.mean(), color='darkred', 
                              linestyle='--', linewidth=2)
                    
                    # Add title with % difference
                    feature_stats = comparison_df[comparison_df['Feature'] == feature]
                    if not feature_stats.empty:
                        pct_diff = feature_stats['Pct_Diff'].values[0]
                        significance = feature_stats['Significant'].values[0]
                        ax.set_title(f'{feature}\nΔ = {pct_diff:+.1f}% {significance}', 
                                   fontsize=11, fontweight='bold')
                    else:
                        ax.set_title(feature, fontsize=11, fontweight='bold')
                    
                    ax.set_xlabel('Value', fontsize=9)
                    ax.set_ylabel('Density', fontsize=9)
                    ax.legend(fontsize=8, loc='best')
                    ax.grid(True, alpha=0.3)
                else:
                    ax.text(0.5, 0.5, 'Insufficient Data', 
                           ha='center', va='center', fontsize=12)
                    ax.set_title(feature, fontsize=11, fontweight='bold')
            
            # Hide unused subplots
            for idx in range(n_features, len(axes)):
                axes[idx].axis('off')
            
            plt.suptitle('Crisis vs Normal Feature Distributions', 
                        fontsize=16, fontweight='bold', y=1.001)
            plt.tight_layout()
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating distributions plot: {str(e)}")
            self.logger.debug(traceback.format_exc())
            return None
    
    def plot_correlation_heatmap(self, corr_matrix: pd.DataFrame) -> Optional[plt.Figure]:
        """
        Plot correlation heatmap
        
        Args:
            corr_matrix: Correlation matrix
            
        Returns:
            Figure object or None
        """
        try:
            self.logger.info("Creating correlation heatmap...")
            
            fig, ax = plt.subplots(figsize=(16, 14))
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            
            sns.heatmap(corr_matrix, mask=mask, annot=False, fmt='.2f', 
                       cmap='coolwarm', center=0, square=True,
                       linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax)
            
            ax.set_title('Feature Correlation Heatmap', 
                        fontsize=16, fontweight='bold', pad=20)
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating correlation heatmap: {str(e)}")
            return None


# ============================================================================
# REPORT GENERATION
# ============================================================================

class ReportGenerator:
    """Generate comprehensive EDA reports"""
    
    def __init__(self, config: EDAConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
    
    def generate_summary_report(
        self,
        df: pd.DataFrame,
        validation_results: Dict,
        crisis_stats: Optional[Dict],
        comparison_df: Optional[pd.DataFrame],
        outlier_df: Optional[pd.DataFrame],
        high_corr_pairs: List[Dict]
    ) -> str:
        """
        Generate comprehensive summary report
        
        Args:
            df: Input dataframe
            validation_results: Validation results
            crisis_stats: Crisis statistics
            comparison_df: Crisis vs normal comparison
            outlier_df: Outlier analysis results
            high_corr_pairs: High correlation pairs
            
        Returns:
            Report text
        """
        try:
            self.logger.info("Generating summary report...")
            
            report = f"""
{'='*80}
FINANCIAL STRESS TEST - EDA SUMMARY REPORT
Model 3: Anomaly Detection & Risk Scoring
{'='*80}
Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Author: Parth Saraykar

1. DATASET OVERVIEW
{'='*80}
Total Samples: {len(df):,}
Date Range: {df[self.config.date_column].min()} to {df[self.config.date_column].max()}
Companies: {df[self.config.company_column].nunique()}
Sectors: {df[self.config.sector_column].nunique()}
Total Features: {len(df.columns)}
Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB

2. DATA VALIDATION
{'='*80}
✓ Passed Checks: {len(validation_results['passed'])}
"""
            
            for check in validation_results['passed'][:10]:
                report += f"  ✓ {check}\n"
            
            if validation_results['warnings']:
                report += f"\n⚠ Warnings: {len(validation_results['warnings'])}\n"
                for warning in validation_results['warnings'][:10]:
                    report += f"  ⚠ {warning}\n"
            
            if validation_results['errors']:
                report += f"\n✗ Errors: {len(validation_results['errors'])}\n"
                for error in validation_results['errors']:
                    report += f"  ✗ {error}\n"
            
            # Crisis statistics
            if crisis_stats:
                report += f"""
3. CRISIS DISTRIBUTION
{'='*80}
Crisis Samples: {crisis_stats['crisis_samples']:,} ({crisis_stats['crisis_rate']:.2f}%)
Normal Samples: {crisis_stats['normal_samples']:,} ({100-crisis_stats['crisis_rate']:.2f}%)
Realistic Rate: {'✓ Yes' if crisis_stats['is_realistic'] else '✗ No'}
"""
            
            # Top differentiating features
            if comparison_df is not None and len(comparison_df) > 0:
                report += f"""
4. TOP DIFFERENTIATING FEATURES (Crisis vs Normal)
{'='*80}
"""
                for i, row in comparison_df.head(10).iterrows():
                    report += f"{row['Feature']:30s} | Δ = {row['Pct_Diff']:+7.2f}% {row['Significant']}\n"
            
            # Data quality issues
            report += f"""
5. DATA QUALITY SUMMARY
{'='*80}
Missing Values: {df.isnull().sum().sum():,} total
High Correlation Pairs: {len(high_corr_pairs)} pairs with |corr| > {self.config.correlation_threshold}
"""
            
            if outlier_df is not None:
                high_outliers = outlier_df[outlier_df['ZScore_Pct'] > 5]
                report += f"Features with >5% Outliers: {len(high_outliers)}\n"
            
            # Recommendations
            report += f"""
6. KEY RECOMMENDATIONS
{'='*80}
"""
            if crisis_stats and crisis_stats['is_realistic']:
                report += "✓ Crisis rate is realistic - proceed with crisis_flag as target\n"
            else:
                report += "⚠ Review crisis labeling strategy - may need adjustment\n"
            
            if comparison_df is not None and len(comparison_df) > 0:
                report += f"✓ Focus on top {min(15, len(comparison_df))} differentiating features\n"
            
            if len(high_corr_pairs) > 0:
                report += f"⚠ Consider removing {len(high_corr_pairs)} highly correlated features\n"
            
            report += f"\n7. NEXT STEPS\n{'='*80}\n"
            report += "1. Review generated visualizations in outputs/plots/\n"
            report += "2. Examine crisis_vs_normal_comparison.csv for detailed statistics\n"
            report += "3. Design labeling strategy based on findings\n"
            report += "4. Proceed to feature selection and model training\n"
            
            report += f"\n{'='*80}\nEND OF REPORT\n{'='*80}\n"
            
            self.logger.info("✓ Summary report generated")
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating report: {str(e)}")
            self.logger.debug(traceback.format_exc())
            return f"Error generating report: {str(e)}"
    
    def save_report(self, report_text: str, filename: str = "eda_summary_report.txt") -> bool:
        """
        Save report to file
        
        Args:
            report_text: Report content
            filename: Output filename
            
        Returns:
            Success status
        """
        try:
            output_path = Path(self.config.reports_dir) / filename
            with open(output_path, 'w') as f:
                f.write(report_text)
            self.logger.info(f"✓ Report saved: {output_path}")
            return True
        except Exception as e:
            self.logger.error(f"Error saving report: {str(e)}")
            return False


# ============================================================================
# MAIN PIPELINE
# ============================================================================

class EDAPipeline:
    """Main EDA pipeline orchestrator"""
    
    def __init__(self, config_path: str = "configs/eda_config.yaml"):
        """Initialize pipeline with configuration"""
        # Load configuration
        self.config = load_config(config_path)
        
        # Initialize logger
        logger_instance = EDALogger(log_level=self.config.log_level)
        self.logger = logger_instance.get_logger()
        
        self.logger.info("="*80)
        self.logger.info("FINANCIAL STRESS TEST EDA PIPELINE")
        self.logger.info("="*80)
        
        # Setup directories
        setup_directories(self.config, self.logger)
        
        # Initialize results storage
        self.results = {}
    
    def run(self) -> bool:
        """
        Execute complete EDA pipeline
        
        Returns:
            Success status
        """
        try:
            self.logger.info("Starting EDA pipeline execution...")
            
            # Step 1: Load data
            self.logger.info("\n" + "="*80)
            self.logger.info("STEP 1: DATA LOADING")
            self.logger.info("="*80)
            
            df = load_data(self.config, self.logger)
            self.results['dataframe'] = df
            
            # Step 2: Validate data
            self.logger.info("\n" + "="*80)
            self.logger.info("STEP 2: DATA VALIDATION")
            self.logger.info("="*80)
            
            validator = DataValidator(self.logger)
            is_valid, validation_results = validator.validate_dataframe(df, self.config)
            validator.log_validation_results()
            self.results['validation_results'] = validation_results
            
            if not is_valid and validation_results['errors']:
                self.logger.error("Critical validation errors found. Stopping pipeline.")
                return False
            
            # Step 3: Crisis analysis
            self.logger.info("\n" + "="*80)
            self.logger.info("STEP 3: CRISIS ANALYSIS")
            self.logger.info("="*80)
            
            crisis_analyzer = CrisisAnalyzer(df, self.config, self.logger)
            crisis_stats = crisis_analyzer.analyze_crisis_distribution()
            crisis_by_year = crisis_analyzer.analyze_temporal_distribution()
            
            self.results['crisis_stats'] = crisis_stats
            self.results['crisis_by_year'] = crisis_by_year
            
            # Step 4: Feature comparison
            self.logger.info("\n" + "="*80)
            self.logger.info("STEP 4: FEATURE COMPARISON (Crisis vs Normal)")
            self.logger.info("="*80)
            
            comparison_df = crisis_analyzer.compare_crisis_vs_normal(self.config.risk_features)
            self.results['comparison_df'] = comparison_df
            
            if comparison_df is not None:
                # Save comparison CSV
                output_path = Path(self.config.data_dir) / "crisis_vs_normal_comparison.csv"
                comparison_df.to_csv(output_path, index=False)
                self.logger.info(f"✓ Saved: {output_path}")
            
            # Step 5: Correlation analysis
            self.logger.info("\n" + "="*80)
            self.logger.info("STEP 5: CORRELATION ANALYSIS")
            self.logger.info("="*80)
            
            corr_analyzer = CorrelationAnalyzer(df, self.config, self.logger)
            corr_matrix = corr_analyzer.compute_correlation_matrix(self.config.risk_features)
            high_corr_pairs = corr_analyzer.find_high_correlations(corr_matrix) if corr_matrix is not None else []
            
            self.results['corr_matrix'] = corr_matrix
            self.results['high_corr_pairs'] = high_corr_pairs
            
            # Step 6: Outlier analysis
            self.logger.info("\n" + "="*80)
            self.logger.info("STEP 6: OUTLIER ANALYSIS")
            self.logger.info("="*80)
            
            outlier_analyzer = OutlierAnalyzer(df, self.config, self.logger)
            outlier_df = outlier_analyzer.analyze_outliers(self.config.risk_features)
            self.results['outlier_df'] = outlier_df
            
            if outlier_df is not None:
                output_path = Path(self.config.data_dir) / "outlier_analysis.csv"
                outlier_df.to_csv(output_path, index=False)
                self.logger.info(f"✓ Saved: {output_path}")
            
            # Step 7: Visualizations
            self.logger.info("\n" + "="*80)
            self.logger.info("STEP 7: VISUALIZATIONS")
            self.logger.info("="*80)
            
            visualizer = Visualizer(self.config, self.logger)
            
            # Crisis over time
            if crisis_by_year is not None:
                fig = visualizer.plot_crisis_over_time(crisis_by_year)
                if fig:
                    visualizer.save_plot(fig, "crisis_rate_by_year.png")
            
            # Crisis vs normal distributions
            if comparison_df is not None and self.config.crisis_flag_column in df.columns:
                top_features = comparison_df.head(12)['Feature'].tolist()
                fig = visualizer.plot_crisis_vs_normal_distributions(
                    df, top_features, self.config.crisis_flag_column, comparison_df
                )
                if fig:
                    visualizer.save_plot(fig, "eda_crisis_vs_normal.png")
            
            # Correlation heatmap
            if corr_matrix is not None:
                fig = visualizer.plot_correlation_heatmap(corr_matrix)
                if fig:
                    visualizer.save_plot(fig, "feature_correlations.png")
            
            # Step 8: Generate report
            self.logger.info("\n" + "="*80)
            self.logger.info("STEP 8: REPORT GENERATION")
            self.logger.info("="*80)
            
            report_gen = ReportGenerator(self.config, self.logger)
            report_text = report_gen.generate_summary_report(
                df=df,
                validation_results=validation_results,
                crisis_stats=crisis_stats,
                comparison_df=comparison_df,
                outlier_df=outlier_df,
                high_corr_pairs=high_corr_pairs
            )
            
            report_gen.save_report(report_text)
            
            # Print report to console
            print("\n" + report_text)
            
            self.logger.info("\n" + "="*80)
            self.logger.info("✓ EDA PIPELINE COMPLETED SUCCESSFULLY!")
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
        # Initialize and run pipeline
        pipeline = EDAPipeline(config_path="configs/eda_config.yaml")
        success = pipeline.run()
        
        if success:
            print("\n✓ EDA completed successfully!")
            print("Check the following directories for outputs:")
            print("  - outputs/plots/     - Visualizations")
            print("  - outputs/data/      - Analysis results (CSV)")
            print("  - outputs/reports/   - Summary report")
            print("  - logs/              - Execution logs")
            sys.exit(0)
        else:
            print("\n✗ EDA pipeline failed. Check logs for details.")
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