"""
STEP 5: BIAS DETECTION WITH EXPLICIT DATA SLICING

FIXED FOR MODIFIED PIPELINE:
- 50 companies (was 25)
- Date range: 1990-2025 (was 2005-2025)
- Quarterly data (was daily)
- Updated temporal periods
- Fixed SliceAnalyzer bug

Implements data slicing for bias detection as required by MLOps assignment.

Data Slicing Methodology:
- Creates meaningful slices of data based on categorical features
- Analyzes each slice independently  
- Compares slices to detect bias
- Generates mitigation strategies

Slicing Dimensions:
1. Company (50 companies)
2. Sector (categorical grouping)
3. Time Period (1990-2025, multiple crisis periods)
4. Market Regime (volatility-based slicing)

Usage:
    python step5_bias_detection_with_explicit_slicing.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Tuple
from datetime import datetime
import json
from scipy import stats

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# DATA SLICER - Creates slices from dataset
# ============================================================================

class DataSlicer:
<<<<<<< HEAD:src/data/step5_bias_detection_with_explicit_slicing.py
    """
    Creates slices of data for bias analysis.

    Implements data slicing similar to TensorFlow Model Analysis (TFMA).
    """

=======
    """Creates slices of data for bias analysis."""
    
>>>>>>> 8d1ecdb96c99c112cf86a9fd5fc949c4961a8f42:src/data/step3_bias_detection_with_explicit_slicing.py
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.slices = {}

    def create_slices_by_feature(self, feature: str) -> Dict[str, pd.DataFrame]:
<<<<<<< HEAD:src/data/step5_bias_detection_with_explicit_slicing.py
        """
        Create slices based on a categorical feature.

        Args:
            feature: Column name to slice by (e.g., 'Company', 'Sector')

        Returns:
            Dictionary of {slice_name: slice_dataframe}
        """
=======
        """Create slices based on a categorical feature."""
>>>>>>> 8d1ecdb96c99c112cf86a9fd5fc949c4961a8f42:src/data/step3_bias_detection_with_explicit_slicing.py
        logger.info(f"\n   Creating slices by '{feature}'...")

        if feature not in self.df.columns:
            logger.warning(f"   ⚠️  Feature '{feature}' not found")
            return {}

        slices = {}
        unique_values = self.df[feature].unique()

        for value in unique_values:
            slice_key = f"{feature}={value}"
            slice_df = self.df[self.df[feature] == value].copy()
            slices[slice_key] = slice_df

        logger.info(f"   ✓ Created {len(slices)} slices")

        self.slices[feature] = slices
        return slices

    def create_temporal_slices(self, periods: Dict[str, Tuple[int, int]]) -> Dict[str, pd.DataFrame]:
<<<<<<< HEAD:src/data/step5_bias_detection_with_explicit_slicing.py
        """
        Create slices based on time periods.

        Args:
            periods: Dictionary of {period_name: (start_year, end_year)}

        Returns:
            Dictionary of {period_name: slice_dataframe}
        """
=======
        """Create slices based on time periods."""
>>>>>>> 8d1ecdb96c99c112cf86a9fd5fc949c4961a8f42:src/data/step3_bias_detection_with_explicit_slicing.py
        logger.info(f"\n   Creating temporal slices...")

        if 'Date' not in self.df.columns:
            logger.warning(f"   ⚠️  No Date column")
            return {}
<<<<<<< HEAD:src/data/step5_bias_detection_with_explicit_slicing.py

=======
        
        # Ensure Date is datetime
        if not pd.api.types.is_datetime64_any_dtype(self.df['Date']):
            self.df['Date'] = pd.to_datetime(self.df['Date'], format='mixed', errors='coerce')
        
>>>>>>> 8d1ecdb96c99c112cf86a9fd5fc949c4961a8f42:src/data/step3_bias_detection_with_explicit_slicing.py
        self.df['Year'] = self.df['Date'].dt.year
        slices = {}

        for period_name, (start_year, end_year) in periods.items():
            period_mask = (self.df['Year'] >= start_year) & (
                self.df['Year'] <= end_year)
            slice_df = self.df[period_mask].copy()
            slices[period_name] = slice_df

        logger.info(f"   ✓ Created {len(slices)} temporal slices")

        self.slices['temporal'] = slices
        return slices

    def create_regime_slices(self, regime_column: str = 'VIX_Regime') -> Dict[str, pd.DataFrame]:
<<<<<<< HEAD:src/data/step5_bias_detection_with_explicit_slicing.py
        """
        Create slices based on market regime.

        Args:
            regime_column: Column defining market regime

        Returns:
            Dictionary of {regime: slice_dataframe}
        """
=======
        """Create slices based on market regime."""
>>>>>>> 8d1ecdb96c99c112cf86a9fd5fc949c4961a8f42:src/data/step3_bias_detection_with_explicit_slicing.py
        logger.info(f"\n   Creating market regime slices...")

        if regime_column in self.df.columns:
            return self.create_slices_by_feature(regime_column)
        elif 'VIX' in self.df.columns:
            # Create regime from VIX
            logger.info(f"   Creating regime from VIX values...")

            self.df['Market_Regime'] = pd.cut(
                self.df['VIX'],
                bins=[0, 15, 25, 100],
                labels=['Low_Vol', 'Medium_Vol', 'High_Vol']
            )

            return self.create_slices_by_feature('Market_Regime')
        else:
            logger.warning(f"   ⚠️  Cannot create regime slices - no VIX column")
            return {}

    def get_all_slices(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        """Get all created slices."""
        return self.slices


# ============================================================================
# SLICE ANALYZER - Analyzes each slice for bias
# ============================================================================

class SliceAnalyzer:
<<<<<<< HEAD:src/data/step5_bias_detection_with_explicit_slicing.py
    """
    Analyzes slices to detect bias.

    Implements fairness metrics across slices.
    """

=======
    """Analyzes slices to detect bias."""
    
>>>>>>> 8d1ecdb96c99c112cf86a9fd5fc949c4961a8f42:src/data/step3_bias_detection_with_explicit_slicing.py
    def __init__(self):
        self.slice_metrics = {}

    def analyze_slice(self, slice_name: str, slice_df: pd.DataFrame) -> Dict:
        """
        Analyze a single slice.

        Computes:
        - Sample size
        - Missing value percentage
        - Key feature statistics
        - Data quality metrics
        """
        metrics = {
            'slice_name': str(slice_name),
            'n_samples': int(len(slice_df))
        }
<<<<<<< HEAD:src/data/step5_bias_detection_with_explicit_slicing.py

        # Add feature statistics
        numeric_cols = slice_df.select_dtypes(include=[np.number]).columns

        if 'Stock_Return_1D' in numeric_cols:
            metrics['return_mean'] = float(
                round(slice_df['Stock_Return_1D'].mean(), 4))
            metrics['return_std'] = float(
                round(slice_df['Stock_Return_1D'].std(), 4))

        if 'Revenue' in numeric_cols:
            metrics['revenue_mean'] = float(slice_df['Revenue'].mean())

        if 'Date' in slice_df.columns:
            metrics['date_range_days'] = int(
                (slice_df['Date'].max() - slice_df['Date'].min()).days)

        return metrics

    def compare_slices(self, slices: Dict[str, pd.DataFrame],
                       feature: str = 'Stock_Return_1D') -> List[Dict]:
=======
        
        # Handle empty slices
        if len(slice_df) == 0:
            metrics['missing_pct'] = 0.0
            metrics['date_range_days'] = 0
            return metrics
        
        # Calculate missing percentage (handle division by zero)
        total_elements = slice_df.size
        if total_elements > 0:
            missing_count = slice_df.isna().sum().sum()
            metrics['missing_pct'] = float(round((missing_count / total_elements) * 100, 2))
        else:
            metrics['missing_pct'] = 0.0
        
        # Add feature statistics
        numeric_cols = slice_df.select_dtypes(include=[np.number]).columns
        
        # Use Stock_Price instead of Stock_Return_1D (we don't have derived features)
        if 'Stock_Price' in numeric_cols:
            stock_mean = slice_df['Stock_Price'].mean()
            stock_std = slice_df['Stock_Price'].std()
            if not pd.isna(stock_mean):
                metrics['stock_price_mean'] = float(round(stock_mean, 2))
            if not pd.isna(stock_std):
                metrics['stock_price_std'] = float(round(stock_std, 2))
        
        if 'Revenue' in numeric_cols:
            revenue_mean = slice_df['Revenue'].mean()
            if not pd.isna(revenue_mean):
                metrics['revenue_mean'] = float(revenue_mean)
        
        # Handle date range calculation
        if 'Date' in slice_df.columns:
            date_min = slice_df['Date'].min()
            date_max = slice_df['Date'].max()
            
            if pd.notna(date_min) and pd.notna(date_max):
                metrics['date_range_days'] = int((date_max - date_min).days)
            else:
                metrics['date_range_days'] = 0
        
        return metrics
    
    def compare_slices(self, slices: Dict[str, pd.DataFrame], 
                       feature: str = 'Stock_Price') -> List[Dict]:
>>>>>>> 8d1ecdb96c99c112cf86a9fd5fc949c4961a8f42:src/data/step3_bias_detection_with_explicit_slicing.py
        """
        Compare feature distributions across slices.

        Uses Kolmogorov-Smirnov test to detect distribution differences.
        """
        comparisons = []
<<<<<<< HEAD:src/data/step5_bias_detection_with_explicit_slicing.py

        if feature not in self.df.columns:
=======
        
        # Check if feature exists in any slice
        feature_exists = False
        for slice_df in slices.values():
            if feature in slice_df.columns:
                feature_exists = True
                break
        
        if not feature_exists:
            logger.warning(f"   ⚠️  Feature '{feature}' not found in slices")
>>>>>>> 8d1ecdb96c99c112cf86a9fd5fc949c4961a8f42:src/data/step3_bias_detection_with_explicit_slicing.py
            return comparisons

        slice_names = list(slices.keys())

        # Compare each pair of slices
        for i in range(len(slice_names)):
            for j in range(i + 1, len(slice_names)):
                slice1_name = slice_names[i]
                slice2_name = slice_names[j]
<<<<<<< HEAD:src/data/step5_bias_detection_with_explicit_slicing.py

=======
                
                if feature not in slices[slice1_name].columns or feature not in slices[slice2_name].columns:
                    continue
                
>>>>>>> 8d1ecdb96c99c112cf86a9fd5fc949c4961a8f42:src/data/step3_bias_detection_with_explicit_slicing.py
                slice1_data = slices[slice1_name][feature].dropna()
                slice2_data = slices[slice2_name][feature].dropna()

                if len(slice1_data) < 30 or len(slice2_data) < 30:
                    continue

                # KS test
                ks_stat, p_value = stats.ks_2samp(slice1_data, slice2_data)

                if p_value < 0.05:  # Significant difference
                    comparisons.append({
                        'feature': str(feature),
                        'slice1': str(slice1_name),
                        'slice2': str(slice2_name),
                        'ks_statistic': float(ks_stat),
                        'p_value': float(p_value),
                        'mean_diff': float(slice1_data.mean() - slice2_data.mean())
                    })

        return comparisons


# ============================================================================
# BIAS DETECTOR WITH EXPLICIT SLICING
# ============================================================================

class BiasDetectorWithSlicing:
<<<<<<< HEAD:src/data/step5_bias_detection_with_explicit_slicing.py
    """
    Bias detector with explicit data slicing implementation.

    Assignment Requirement: "perform data slicing and analyze performance 
    across different subgroups"
    """

=======
    """Bias detector with explicit data slicing implementation."""
    
>>>>>>> 8d1ecdb96c99c112cf86a9fd5fc949c4961a8f42:src/data/step3_bias_detection_with_explicit_slicing.py
    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        self.bias_report = {
            'dataset': dataset_name,
            'timestamp': datetime.now().isoformat(),
            'slicing_summary': {},
            'biases_detected': [],
            'slice_comparisons': [],
            'mitigation_recommendations': []
        }

    def run_bias_detection(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
<<<<<<< HEAD:src/data/step5_bias_detection_with_explicit_slicing.py
        """
        Run complete bias detection with data slicing.

        Steps:
        1. Create slices (by Company, Sector, Time Period, Regime)
        2. Analyze each slice
        3. Compare slices to detect bias
        4. Generate mitigation recommendations
        """
=======
        """Run complete bias detection with data slicing."""
>>>>>>> 8d1ecdb96c99c112cf86a9fd5fc949c4961a8f42:src/data/step3_bias_detection_with_explicit_slicing.py
        logger.info("\n" + "="*80)
        logger.info("BIAS DETECTION WITH DATA SLICING")
        logger.info("="*80)
        logger.info(f"Dataset: {self.dataset_name}")
        logger.info(f"Shape: {df.shape}")
        logger.info("="*80)

        # === STEP 1: CREATE SLICES ===
        logger.info("\n" + "="*80)
        logger.info("STEP 1: CREATING DATA SLICES")
        logger.info("="*80)

        slicer = DataSlicer(df)

        # Slice by Company
        company_slices = slicer.create_slices_by_feature('Company')
        logger.info(f"   ✓ Company slices: {len(company_slices)}")

        # Slice by Sector
        sector_slices = slicer.create_slices_by_feature('Sector')
        logger.info(f"   ✓ Sector slices: {len(sector_slices)}")
<<<<<<< HEAD:src/data/step5_bias_detection_with_explicit_slicing.py

        # Slice by Time Period
=======
        
        # UPDATED: Temporal periods for 1990-2025
>>>>>>> 8d1ecdb96c99c112cf86a9fd5fc949c4961a8f42:src/data/step3_bias_detection_with_explicit_slicing.py
        periods = {
            'Early Years (1990-1999)': (1990, 1999),
            'Dot-com (2000-2002)': (2000, 2002),
            'Pre-Crisis (2003-2007)': (2003, 2007),
            'Financial Crisis (2008-2009)': (2008, 2009),
            'Recovery (2010-2019)': (2010, 2019),
            'COVID (2020-2021)': (2020, 2021),
            'Recent (2022-2025)': (2022, 2025)
        }
        temporal_slices = slicer.create_temporal_slices(periods)
        logger.info(f"   ✓ Temporal slices: {len(temporal_slices)}")

        # Slice by Market Regime
        regime_slices = slicer.create_regime_slices()
        logger.info(f"   ✓ Market regime slices: {len(regime_slices)}")

        # === STEP 2: ANALYZE EACH SLICE ===
        logger.info("\n" + "="*80)
        logger.info("STEP 2: ANALYZING SLICES")
        logger.info("="*80)

        analyzer = SliceAnalyzer()

        # Analyze company slices
        logger.info("\n[A] Company Slice Analysis (50 companies):")
        company_metrics = []

        for slice_name, slice_df in company_slices.items():
            metrics = analyzer.analyze_slice(slice_name, slice_df)
            company_metrics.append(metrics)

        # Show top/bottom companies by sample size
        company_df = pd.DataFrame(company_metrics)
        company_df = company_df.sort_values('n_samples', ascending=False)

        logger.info("\n   Top 5 companies by sample size:")
        print(company_df.head(5).to_string(index=False))

        logger.info("\n   Bottom 5 companies by sample size:")
        print(company_df.tail(5).to_string(index=False))

        self.bias_report['slicing_summary']['company_slices'] = company_metrics

        # Analyze sector slices
        logger.info("\n[B] Sector Slice Analysis:")
        sector_metrics = []

        for slice_name, slice_df in sector_slices.items():
            metrics = analyzer.analyze_slice(slice_name, slice_df)
            sector_metrics.append(metrics)

        sector_df = pd.DataFrame(sector_metrics)
        print(sector_df.to_string(index=False))

        self.bias_report['slicing_summary']['sector_slices'] = sector_metrics

        # Analyze temporal slices
        logger.info("\n[C] Temporal Slice Analysis:")
        temporal_metrics = []

        for slice_name, slice_df in temporal_slices.items():
            metrics = analyzer.analyze_slice(slice_name, slice_df)
            temporal_metrics.append(metrics)

        temporal_df = pd.DataFrame(temporal_metrics)
        print(temporal_df.to_string(index=False))

        self.bias_report['slicing_summary']['temporal_slices'] = temporal_metrics

        # === STEP 3: DETECT BIAS ACROSS SLICES ===
        logger.info("\n" + "="*80)
        logger.info("STEP 3: DETECTING BIAS ACROSS SLICES")
        logger.info("="*80)

        biases = self._detect_bias_from_slices(
            company_metrics, sector_metrics, temporal_metrics)

        # === STEP 4: GENERATE RECOMMENDATIONS ===
        logger.info("\n" + "="*80)
        logger.info("STEP 4: MITIGATION RECOMMENDATIONS")
        logger.info("="*80)

        recommendations = self._generate_mitigation_recommendations(biases)

        # === SUMMARY ===
        self._print_summary()

        # Save report
        self._save_report()

        return df, self.bias_report

    def _detect_bias_from_slices(self, company_metrics: List[Dict],
                                 sector_metrics: List[Dict],
                                 temporal_metrics: List[Dict]) -> List[Dict]:
        """Detect bias by comparing slice metrics."""

        biases = []
<<<<<<< HEAD:src/data/step5_bias_detection_with_explicit_slicing.py

        # === BIAS 1: Representation Bias ===
        logger.info("\n   [A] Checking representation bias...")

        # Company representation
=======
        
        # === BIAS 1: Representation Bias (UPDATED for 50 companies) ===
        logger.info("\n   [A] Checking representation bias (50 companies)...")
        
>>>>>>> 8d1ecdb96c99c112cf86a9fd5fc949c4961a8f42:src/data/step3_bias_detection_with_explicit_slicing.py
        total_samples = sum(m['n_samples'] for m in company_metrics)
        expected_per_company = total_samples / len(company_metrics)

        underrep = []
        overrep = []

        for metrics in company_metrics:
            samples = metrics['n_samples']
            deviation = (samples - expected_per_company) / expected_per_company
<<<<<<< HEAD:src/data/step5_bias_detection_with_explicit_slicing.py

            if deviation < -0.3:  # 30% below expected
=======
            
            # UPDATED: More lenient for quarterly data (40% vs 30%)
            if deviation < -0.4:  # 40% below expected
>>>>>>> 8d1ecdb96c99c112cf86a9fd5fc949c4961a8f42:src/data/step3_bias_detection_with_explicit_slicing.py
                underrep.append(metrics['slice_name'])
            elif deviation > 0.4:  # 40% above expected
                overrep.append(metrics['slice_name'])

        if underrep or overrep:
            logger.warning(f"   ⚠️  Representation bias detected:")
            logger.warning(
                f"      Underrepresented: {len(underrep)} companies")
            logger.warning(f"      Overrepresented: {len(overrep)} companies")

            biases.append({
                'type': 'Representation Bias',
                'dimension': 'Company',
                'severity': 'HIGH',
                'underrepresented': underrep,
                'overrepresented': overrep
            })
        else:
<<<<<<< HEAD:src/data/step5_bias_detection_with_explicit_slicing.py
            logger.info(f"   ✓ No representation bias")

=======
            logger.info(f"   ✓ No representation bias (within 40% deviation)")
        
>>>>>>> 8d1ecdb96c99c112cf86a9fd5fc949c4961a8f42:src/data/step3_bias_detection_with_explicit_slicing.py
        # === BIAS 2: Sector Imbalance ===
        logger.info("\n   [B] Checking sector imbalance...")

        sector_samples = [m['n_samples'] for m in sector_metrics]
<<<<<<< HEAD:src/data/step5_bias_detection_with_explicit_slicing.py
        max_samples = max(sector_samples)
        min_samples = min(sector_samples)
        imbalance_ratio = max_samples / \
            min_samples if min_samples > 0 else float('inf')

=======
        max_samples = max(sector_samples) if sector_samples else 0
        min_samples = min(sector_samples) if sector_samples else 0
        imbalance_ratio = max_samples / min_samples if min_samples > 0 else float('inf')
        
>>>>>>> 8d1ecdb96c99c112cf86a9fd5fc949c4961a8f42:src/data/step3_bias_detection_with_explicit_slicing.py
        if imbalance_ratio > 5:
            logger.warning(
                f"   ⚠️  Sector imbalance: {imbalance_ratio:.1f}x difference")

            biases.append({
                'type': 'Sector Imbalance',
                'severity': 'MEDIUM',
                'imbalance_ratio': float(imbalance_ratio)
            })
        else:
            logger.info(f"   ✓ Sector balance OK ({imbalance_ratio:.1f}x)")
<<<<<<< HEAD:src/data/step5_bias_detection_with_explicit_slicing.py

        # === BIAS 3: Temporal Data Quality Bias ===
        logger.info("\n   [C] Checking temporal data quality bias...")

        for metrics in temporal_metrics:
            if metrics['missing_pct'] > 5:
                logger.warning(
                    f"   ⚠️  {metrics['slice_name']}: {metrics['missing_pct']:.1f}% missing")

=======
        
        # === BIAS 3: Temporal Data Quality Bias (UPDATED threshold for quarterly) ===
        logger.info("\n   [C] Checking temporal data quality bias...")
        
        biases_found = False
        for metrics in temporal_metrics:
            # UPDATED: More lenient for quarterly data (30% vs 5%)
            if metrics['missing_pct'] > 30:
                logger.warning(f"   ⚠️  {metrics['slice_name']}: {metrics['missing_pct']:.1f}% missing")
                
>>>>>>> 8d1ecdb96c99c112cf86a9fd5fc949c4961a8f42:src/data/step3_bias_detection_with_explicit_slicing.py
                biases.append({
                    'type': 'Temporal Quality Bias',
                    'period': metrics['slice_name'],
                    'severity': 'MEDIUM',
                    'missing_pct': float(metrics['missing_pct'])
                })
<<<<<<< HEAD:src/data/step5_bias_detection_with_explicit_slicing.py

=======
                biases_found = True
        
        if not biases_found:
            logger.info(f"   ✓ No temporal quality bias (all periods < 30% missing)")
        
>>>>>>> 8d1ecdb96c99c112cf86a9fd5fc949c4961a8f42:src/data/step3_bias_detection_with_explicit_slicing.py
        self.bias_report['biases_detected'] = biases

        return biases

    def _generate_mitigation_recommendations(self, biases: List[Dict]) -> List[str]:
        """Generate mitigation recommendations based on detected biases."""

        recommendations = []

        if not biases:
            logger.info("\n   ✓ No biases detected - no mitigation needed")
            recommendations.append(
                "No bias mitigation required - data is well-balanced")
            return recommendations

        # Check for representation bias
        rep_biases = [b for b in biases if 'Representation' in b['type']]
        if rep_biases:
            logger.info("\n   📋 Recommendation 1: Stratified Sampling")
            logger.info(
                "      Use stratified train/test split to ensure all companies represented")
            recommendations.append(
                "Stratified sampling by Company in train/test split")

            logger.info("\n   📋 Recommendation 2: Weighted Loss Function")
            logger.info(
                "      Apply sample weights inversely proportional to company size")
            recommendations.append(
                "Weighted loss: weight = 1 / company_sample_count")

        # Check for sector imbalance
        sector_biases = [b for b in biases if 'Sector' in b['type']]
        if sector_biases:
            logger.info("\n   📋 Recommendation 3: Sector Stratification")
            logger.info(
                "      Ensure each sector proportionally in train/val/test")
            recommendations.append("Stratified split by Sector")

        # Check for temporal bias
        temporal_biases = [b for b in biases if 'Temporal' in b['type']]
        if temporal_biases:
            logger.info("\n   📋 Recommendation 4: Crisis Data Handling")
<<<<<<< HEAD:src/data/step5_bias_detection_with_explicit_slicing.py
            logger.info(
                "      Ensure crisis periods (2008-2009, 2020) in validation set")
            recommendations.append(
                "Include crisis periods in validation for realistic stress testing")

=======
            logger.info("      Ensure crisis periods (1990s, 2000-2002, 2008-2009, 2020) in validation")
            recommendations.append("Include all crisis periods in validation for realistic stress testing")
        
>>>>>>> 8d1ecdb96c99c112cf86a9fd5fc949c4961a8f42:src/data/step3_bias_detection_with_explicit_slicing.py
        self.bias_report['mitigation_recommendations'] = recommendations

        return recommendations

    def _print_summary(self):
        """Print bias detection summary."""
        logger.info("\n" + "="*80)
        logger.info("BIAS DETECTION SUMMARY")
        logger.info("="*80)

        # Count slices created
        total_slices = 0
        for dimension, metrics in self.bias_report['slicing_summary'].items():
            total_slices += len(metrics)

        logger.info(f"\n📊 Data Slicing:")
        logger.info(f"   Total slices created: {total_slices}")
        for dimension, metrics in self.bias_report['slicing_summary'].items():
            logger.info(f"   {dimension}: {len(metrics)} slices")

        # Count biases
        biases = self.bias_report.get('biases_detected', [])
        logger.info(f"\n📊 Biases Detected: {len(biases)}")

        for bias in biases:
            severity_icon = "🚨" if bias['severity'] == 'CRITICAL' else "⚠️" if bias['severity'] == 'HIGH' else "ℹ️"
            logger.info(
                f"   {severity_icon} {bias['type']} ({bias['severity']})")

        # Print recommendations
        recommendations = self.bias_report.get(
            'mitigation_recommendations', [])
        if recommendations:
            logger.info(f"\n💡 Mitigation Recommendations:")
            for i, rec in enumerate(recommendations, 1):
                logger.info(f"   {i}. {rec}")

    def _save_report(self):
        """Save bias detection report."""
        output_dir = Path("data/bias_reports")
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Custom JSON encoder
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, (np.integer, np.int64)):
                    return int(obj)
                elif isinstance(obj, (np.floating, np.float64)):
                    return float(obj)
                elif pd.isna(obj):
                    return None
                return super().default(obj)

        # Save JSON report
        json_path = output_dir / \
            f"bias_report_{self.dataset_name}_{timestamp}.json"

        with open(json_path, 'w') as f:
            json.dump(self.bias_report, f, indent=2, cls=NumpyEncoder)

        logger.info(f"\n💾 Bias report saved: {json_path}")

        # Save slice statistics as CSV
        for slice_type, metrics in self.bias_report['slicing_summary'].items():
            if metrics:
                csv_path = output_dir / \
                    f"{slice_type}_statistics_{timestamp}.csv"
                metrics_df = pd.DataFrame(metrics)
                metrics_df.to_csv(csv_path, index=False)
                logger.info(f"💾 {slice_type} statistics: {csv_path}")


def main():
    """Execute bias detection with data slicing."""
<<<<<<< HEAD:src/data/step5_bias_detection_with_explicit_slicing.py

    features_dir = Path("data/features")

=======
    
    features_dir = Path("data/processed/")
    
>>>>>>> 8d1ecdb96c99c112cf86a9fd5fc949c4961a8f42:src/data/step3_bias_detection_with_explicit_slicing.py
    # Find dataset
    candidates = [
        "features_engineered.csv"
    ]

    filepath = None
    for candidate in candidates:
        if (features_dir / candidate).exists():
            filepath = features_dir / candidate
            logger.info(f"Found: {candidate}")
            break

    if not filepath:
        logger.error("❌ No merged features found!")
        logger.error("Run Step 3 or Step 4 first")
        return

    logger.info(f"Loading: {filepath}")
    df = pd.read_csv(filepath)

    # Ensure Date is datetime
    if 'Date' in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df['Date']):
            df['Date'] = pd.to_datetime(
                df['Date'], format='mixed', errors='coerce')
            logger.info(f"   ✓ Converted Date to datetime: {df['Date'].dtype}")
<<<<<<< HEAD:src/data/step5_bias_detection_with_explicit_slicing.py

=======
    
>>>>>>> 8d1ecdb96c99c112cf86a9fd5fc949c4961a8f42:src/data/step3_bias_detection_with_explicit_slicing.py
    # Run bias detection
    detector = BiasDetectorWithSlicing(dataset_name=filepath.stem)
    df_analyzed, report = detector.run_bias_detection(df)

    # Final status
    logger.info("\n" + "="*80)
    logger.info("✅ BIAS DETECTION WITH DATA SLICING COMPLETE")
    logger.info("="*80)

    total_biases = len(report.get('biases_detected', []))

    if total_biases == 0:
        logger.info("\n✅ No significant biases detected")
        logger.info("✅ Data is well-balanced across slices")
    else:
<<<<<<< HEAD:src/data/step5_bias_detection_with_explicit_slicing.py
        logger.warning(
            f"\n⚠️  {total_biases} biases detected - apply mitigation")

    logger.info("\n➡️  Next: Apply mitigation in train/test split code")
=======
        logger.warning(f"\n⚠️  {total_biases} biases detected")
        logger.warning("   Review bias report and apply mitigation strategies")
    
    logger.info("\n📁 Reports saved to: data/bias_reports/")
    logger.info("\n➡️  Next: Apply stratified sampling in model training")
>>>>>>> 8d1ecdb96c99c112cf86a9fd5fc949c4961a8f42:src/data/step3_bias_detection_with_explicit_slicing.py


if __name__ == "__main__":
    main()
