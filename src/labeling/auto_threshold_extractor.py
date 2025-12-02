
"""
============================================================================
Automatic Threshold Extraction from EDA Results (FIXED - No Numpy Objects)
============================================================================
Author: Parth Saraykar
Purpose: Intelligently extract thresholds from EDA outputs and generate
         Snorkel labeling function thresholds automatically
============================================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, Tuple, Optional, Any
import yaml
import json
from datetime import datetime


def clean_for_yaml(obj: Any) -> Any:
    """
    Convert numpy types to Python native types for clean YAML serialization
    
    Args:
        obj: Object to convert (can be numpy type, dict, list, etc.)
        
    Returns:
        Clean Python native type
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: clean_for_yaml(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_for_yaml(i) for i in obj]
    elif pd.isna(obj):
        return None
    else:
        return obj


class AutoThresholdExtractor:
    """
    Automatically extract thresholds from EDA results for Snorkel labeling
    """
    
    def __init__(
        self,
        eda_comparison_path: str = "outputs/eda/data/crisis_vs_normal_comparison.csv",
        config_path: str = "configs/eda_config.yaml",
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize threshold extractor
        
        Args:
            eda_comparison_path: Path to EDA comparison CSV file
            config_path: Path to configuration file
            logger: Optional logger instance
        """
        self.eda_comparison_path = eda_comparison_path
        self.config_path = config_path
        self.logger = logger or self._setup_logger()
        
        # Load EDA results
        self.comparison_df = None
        self.thresholds = {}
        
    def _setup_logger(self) -> logging.Logger:
        """Setup basic logger"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def load_eda_results(self) -> pd.DataFrame:
        """Load EDA comparison results"""
        self.logger.info(f"Loading EDA results from: {self.eda_comparison_path}")
        
        try:
            if not Path(self.eda_comparison_path).exists():
                raise FileNotFoundError(
                    f"EDA comparison file not found: {self.eda_comparison_path}\n"
                    "Please run EDA first: python src/eda/eda.py"
                )
            
            self.comparison_df = pd.read_csv(self.eda_comparison_path)
            self.logger.info(f"✓ Loaded {len(self.comparison_df)} features from EDA results")
            
            return self.comparison_df
            
        except Exception as e:
            self.logger.error(f"Error loading EDA results: {str(e)}")
            raise
    
    def extract_thresholds(self) -> Dict:
        """
        Extract thresholds intelligently from EDA results
        
        Strategy:
        1. Use crisis mean + 0.5 * std for upper thresholds (VIX, stress indices)
        2. Use crisis mean - 0.5 * std for lower thresholds (revenue, margins)
        3. Use percentile-based thresholds for key metrics
        4. Apply domain knowledge constraints
        
        Returns:
            Dictionary of thresholds (all Python native types)
        """
        self.logger.info("Extracting thresholds from EDA results...")
        
        if self.comparison_df is None:
            self.load_eda_results()
        
        thresholds = {}
        
        # ====================================================================
        # MARKET STRESS INDICATORS (Higher during crisis)
        # ====================================================================
        
        # VIX thresholds
        vix_row = self._get_feature_row('vix_q_mean')
        if vix_row is not None:
            crisis_mean = float(vix_row['Crisis_Mean'])
            crisis_std = float(vix_row['Crisis_Std'])
            normal_mean = float(vix_row['Normal_Mean'])
            
            # VIX > 30 = extreme fear (use crisis mean as baseline)
            thresholds['vix_high'] = float(max(30.0, crisis_mean - 0.3 * crisis_std))
            # VIX > 25 = elevated volatility (between normal and crisis)
            thresholds['vix_elevated'] = float(normal_mean + 0.5 * (crisis_mean - normal_mean))
            # VIX < 20 = calm markets
            thresholds['vix_calm'] = float(min(20.0, normal_mean))
            
            self.logger.info(f"VIX thresholds: high={thresholds['vix_high']:.2f}, "
                           f"elevated={thresholds['vix_elevated']:.2f}, "
                           f"calm={thresholds['vix_calm']:.2f}")
        
        # Financial Stress Index
        fsi_row = self._get_feature_row('Financial_Stress_Index_mean')
        if fsi_row is not None:
            crisis_mean = float(fsi_row['Crisis_Mean'])
            normal_mean = float(fsi_row['Normal_Mean'])
            
            # Midpoint between normal and crisis
            thresholds['financial_stress_high'] = float(normal_mean + 0.6 * (crisis_mean - normal_mean))
            
            self.logger.info(f"Financial Stress Index threshold: "
                           f"{thresholds['financial_stress_high']:.2f}")
        
        # Composite Stress Score
        comp_row = self._get_feature_row('composite_stress_score')
        if comp_row is not None:
            crisis_mean = float(comp_row['Crisis_Mean'])
            normal_mean = float(comp_row['Normal_Mean'])
            
            thresholds['composite_stress_high'] = float(normal_mean + 0.5 * (crisis_mean - normal_mean))
            
            self.logger.info(f"Composite Stress threshold: "
                           f"{thresholds['composite_stress_high']:.2f}")
        
        # ====================================================================
        # REVENUE & GROWTH (Lower during crisis)
        # ====================================================================
        
        # Revenue growth 1Q
        rev1q_row = self._get_feature_row('Revenue_growth_1q')
        if rev1q_row is not None:
            crisis_mean = float(rev1q_row['Crisis_Mean'])
            crisis_std = float(rev1q_row['Crisis_Std'])
            normal_mean = float(rev1q_row['Normal_Mean'])
            
            # Convert from percentage to decimal if needed
            if abs(crisis_mean) > 1.0:
                crisis_mean = crisis_mean / 100.0
                crisis_std = crisis_std / 100.0
                normal_mean = normal_mean / 100.0
            
            # Severe decline: crisis mean
            thresholds['revenue_severe_decline'] = float(crisis_mean)
            # Moderate decline: crisis mean + 0.5 std (less severe)
            thresholds['revenue_decline_1q'] = float(crisis_mean + 0.5 * crisis_std)
            # Strong growth (for NOT_AT_RISK)
            thresholds['revenue_growth_strong'] = float(max(0.10, normal_mean * 1.5))
            
            self.logger.info(f"Revenue 1Q thresholds: severe={thresholds['revenue_severe_decline']:.2%}, "
                           f"decline={thresholds['revenue_decline_1q']:.2%}, "
                           f"strong={thresholds['revenue_growth_strong']:.2%}")
        
        # Revenue growth 4Q
        rev4q_row = self._get_feature_row('Revenue_growth_4q')
        if rev4q_row is not None:
            crisis_mean = float(rev4q_row['Crisis_Mean'])
            
            if abs(crisis_mean) > 1.0:
                crisis_mean = crisis_mean / 100.0
            
            thresholds['revenue_decline_4q'] = float(crisis_mean + 0.5 * abs(crisis_mean))
            
            self.logger.info(f"Revenue 4Q threshold: {thresholds['revenue_decline_4q']:.2%}")
        
        # ====================================================================
        # PROFITABILITY METRICS (Lower during crisis)
        # ====================================================================
        
        # Net margin
        margin_row = self._get_feature_row('net_margin')
        if margin_row is not None:
            crisis_mean = float(margin_row['Crisis_Mean'])
            crisis_std = float(margin_row['Crisis_Std'])
            normal_mean = float(margin_row['Normal_Mean'])
            
            # Convert from percentage to decimal if needed
            if abs(crisis_mean) > 1.0:
                crisis_mean = crisis_mean / 100.0
                crisis_std = crisis_std / 100.0
                normal_mean = normal_mean / 100.0
            
            # Critical: crisis mean
            thresholds['net_margin_critical'] = float(max(0.03, crisis_mean))
            # Low: crisis mean + 0.5 std
            thresholds['net_margin_low'] = float(max(0.05, crisis_mean + 0.5 * crisis_std))
            # Healthy: normal mean * 1.2
            thresholds['net_margin_healthy'] = float(normal_mean * 1.2)
            
            self.logger.info(f"Net margin thresholds: critical={thresholds['net_margin_critical']:.2%}, "
                           f"low={thresholds['net_margin_low']:.2%}, "
                           f"healthy={thresholds['net_margin_healthy']:.2%}")
        
        # ROA
        roa_row = self._get_feature_row('roa')
        if roa_row is not None:
            crisis_mean = float(roa_row['Crisis_Mean'])
            
            if abs(crisis_mean) > 1.0:
                crisis_mean = crisis_mean / 100.0
            
            thresholds['roa_low'] = float(max(0.01, crisis_mean + 0.5 * abs(crisis_mean)))
            
            self.logger.info(f"ROA threshold: {thresholds['roa_low']:.2%}")
        
        # ROE
        thresholds['roe_low'] = 0.05  # Default 5%
        
        # ====================================================================
        # LEVERAGE METRICS
        # ====================================================================
        
        # Debt-to-Equity (use domain knowledge due to outliers)
        thresholds['debt_to_equity_critical'] = 3.5
        thresholds['debt_to_equity_high'] = 2.5
        thresholds['debt_to_equity_healthy'] = 1.5
        thresholds['current_ratio_low'] = 1.2
        thresholds['debt_to_assets_high'] = 0.6
        
        self.logger.info(f"Leverage thresholds: D/E high=2.5, critical=3.5")
        
        # ====================================================================
        # MACRO INDICATORS
        # ====================================================================
        
        # Unemployment
        unemp_row = get_feature('Unemployment_Rate_last')
        if unemp_row is not None:
            crisis_mean = float(unemp_row['Crisis_Mean'])
            normal_mean = float(unemp_row['Normal_Mean'])
            
            thresholds['unemployment_critical'] = float(crisis_mean)
            thresholds['unemployment_high'] = float(normal_mean + 0.6 * (crisis_mean - normal_mean))
            
            self.logger.info(f"Unemployment: critical={thresholds['unemployment_critical']:.1f}%, "
                           f"high={thresholds['unemployment_high']:.1f}%")
        
        # Yield Curve
        thresholds['yield_curve_inverted'] = 0.0
        thresholds['gdp_decline'] = 0.0
        
        # ====================================================================
        # COMPOSITE RULES
        # ====================================================================
        
        thresholds['perfect_storm_conditions'] = 3
        
        self.logger.info(f"\n✓ Extracted {len(thresholds)} thresholds")
        
        return thresholds

# Extract thresholds
thresholds = {}

try:
    # Load EDA results
    comparison_df = pd.read_csv('outputs/eda/data/crisis_vs_normal_comparison.csv')
    
    def get_feature(name):
        match = comparison_df[comparison_df['Feature'] == name]
        return match.iloc[0] if len(match) > 0 else None
    
    # VIX
    vix = get_feature('vix_q_mean')
    if vix is not None:
        crisis_mean = float(vix['Crisis_Mean'])
        normal_mean = float(vix['Normal_Mean'])
        crisis_std = float(vix['Crisis_Std'])
        thresholds['vix_high'] = float(max(30.0, crisis_mean - 0.3 * crisis_std))
        thresholds['vix_elevated'] = float(normal_mean + 0.5 * (crisis_mean - normal_mean))
        thresholds['vix_calm'] = float(min(20.0, normal_mean))
    
    # Financial Stress Index
    fsi = get_feature('Financial_Stress_Index_mean')
    if fsi is not None:
        thresholds['financial_stress_high'] = float(fsi['Normal_Mean'] + 0.6 * (fsi['Crisis_Mean'] - fsi['Normal_Mean']))
    
    # Composite Stress
    comp = get_feature('composite_stress_score')
    if comp is not None:
        thresholds['composite_stress_high'] = float(comp['Normal_Mean'] + 0.5 * (comp['Crisis_Mean'] - comp['Normal_Mean']))
    
    # Revenue 1Q
    rev1q = get_feature('Revenue_growth_1q')
    if rev1q is not None:
        crisis_mean = float(rev1q['Crisis_Mean'])
        crisis_std = float(rev1q['Crisis_Std'])
        normal_mean = float(rev1q['Normal_Mean'])
        
        if abs(crisis_mean) > 1.0:
            crisis_mean /= 100.0
            crisis_std /= 100.0
            normal_mean /= 100.0
        
        thresholds['revenue_severe_decline'] = float(crisis_mean)
        thresholds['revenue_decline_1q'] = float(crisis_mean + 0.5 * crisis_std)
        thresholds['revenue_growth_strong'] = float(max(0.10, normal_mean * 1.5))
    
    # Revenue 4Q
    rev4q = get_feature('Revenue_growth_4q')
    if rev4q is not None:
        crisis_mean = float(rev4q['Crisis_Mean'])
        if abs(crisis_mean) > 1.0:
            crisis_mean /= 100.0
        thresholds['revenue_decline_4q'] = float(crisis_mean + 0.5 * abs(crisis_mean))
    
    # Net margin
    margin = get_feature('net_margin')
    if margin is not None:
        crisis_mean = float(margin['Crisis_Mean'])
        crisis_std = float(margin['Crisis_Std'])
        normal_mean = float(margin['Normal_Mean'])
        
        if abs(crisis_mean) > 1.0:
            crisis_mean /= 100.0
            crisis_std /= 100.0
            normal_mean /= 100.0
        
        thresholds['net_margin_critical'] = float(max(0.03, crisis_mean))
        thresholds['net_margin_low'] = float(max(0.05, crisis_mean + 0.5 * crisis_std))
        thresholds['net_margin_healthy'] = float(normal_mean * 1.2)
    
    # ROA
    roa = get_feature('roa')
    if roa is not None:
        crisis_mean = float(roa['Crisis_Mean'])
        if abs(crisis_mean) > 1.0:
            crisis_mean /= 100.0
        thresholds['roa_low'] = float(max(0.01, crisis_mean + 0.5 * abs(crisis_mean)))
    
    # ROE
    thresholds['roe_low'] = 0.05
    
    # Leverage (domain knowledge)
    thresholds['debt_to_equity_critical'] = 3.5
    thresholds['debt_to_equity_high'] = 2.5
    thresholds['debt_to_equity_healthy'] = 1.5
    thresholds['current_ratio_low'] = 1.2
    thresholds['debt_to_assets_high'] = 0.6
    
    # Unemployment
    unemp = get_feature('Unemployment_Rate_last')
    if unemp is not None:
        crisis_mean = float(unemp['Crisis_Mean'])
        normal_mean = float(unemp['Normal_Mean'])
        thresholds['unemployment_critical'] = float(crisis_mean)
        thresholds['unemployment_high'] = float(normal_mean + 0.6 * (crisis_mean - normal_mean))
    
    # Yield curve and GDP
    thresholds['yield_curve_inverted'] = 0.0
    thresholds['gdp_decline'] = 0.0
    
    # Composite
    thresholds['perfect_storm_conditions'] = 3
    
    print(f"\n✓ Extracted {len(thresholds)} thresholds")
    
    # Save to YAML
    output_data = {
        'metadata': {
            'generated_at': datetime.now().isoformat(),
            'source': 'outputs/eda/data/crisis_vs_normal_comparison.csv',
            'extraction_method': 'automatic',
            'num_thresholds': len(thresholds)
        },
        'thresholds': thresholds
    }
    
    # Ensure clean Python types (no numpy)
    output_data = clean_for_yaml(output_data)
    
    # Save
    Path("outputs/snorkel").mkdir(exist_ok=True, parents=True)
    with open('outputs/snorkel/thresholds_auto.yaml', 'w') as f:
        yaml.dump(output_data, f, default_flow_style=False, sort_keys=False)
    
    print(f"✓ Saved to: outputs/snorkel/thresholds_auto.yaml")
    
    # Print summary
    print("\n" + "="*80)
    print("EXTRACTED THRESHOLDS:")
    print("="*80)
    for key, value in sorted(thresholds.items()):
        if isinstance(value, float) and abs(value) < 1:
            print(f"  {key:35s}: {value:8.2%}")
        else:
            print(f"  {key:35s}: {value:8.4f}" if isinstance(value, float) else f"  {key:35s}: {value}")
    
    print("="*80)
    print("✓ Threshold extraction complete!")
    print("\nNext step: python src/labeling/snorkel_pipeline.py")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
