"""
STEP 7: DRIFT DETECTION (Historical Data Drift)

MODIFIED FOR QUARTERLY DATA (1990-2025):
- Wider reference period: 1990-2005 (was 2005-2010)
- More lenient p-value: 0.02 (was 0.01) for quarterly noise
- Adjusted for 35 years of history

Detects drift in historical data by comparing time periods.

Purpose:
- Detect if data characteristics changed over time
- Identify features with significant drift
- Flag unstable features for special handling
- Prepare for model validation strategy

Drift Types Detected:
1. Feature Drift - Has feature distribution changed?
2. Statistical Drift - Have mean/std/quantiles changed?
3. Correlation Drift - Have feature relationships changed?

Methodology:
- Compare reference period (1990-2005) vs recent period (2020-2025)
- Use Kolmogorov-Smirnov test for distribution comparison
- Flag features with p-value < 0.02 (lenient for quarterly)

Usage:
    python step7_drift_detection.py
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


class HistoricalDriftDetectorQuarterly:
    """
<<<<<<< HEAD:src/data/step7_drift_detection.py
    Detect drift in historical data by comparing time periods.

    Simple, assignment-friendly implementation.
=======
    Detect drift in historical quarterly data.
    
    MODIFIED: Adjusted for 1990-2025 date range and quarterly frequency.
>>>>>>> 8d1ecdb96c99c112cf86a9fd5fc949c4961a8f42:src/data/step5_drift_detection.py
    """

    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        self.drift_report = {
            'dataset': dataset_name,
            'timestamp': datetime.now().isoformat(),
            'drifted_features': [],
            'stable_features': [],
            'drift_summary': {}
        }

    def detect_drift(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        Detect drift by comparing early vs recent periods.
<<<<<<< HEAD:src/data/step7_drift_detection.py

        Compares:
        - Reference period: 2005-2010 (early data)
        - Current period: 2020-2025 (recent data)
=======
        
        MODIFIED:
        - Reference period: 1990-2005 (15 years)
        - Current period: 2020-2025 (5 years)
        - P-value threshold: 0.02 (more lenient for quarterly noise)
>>>>>>> 8d1ecdb96c99c112cf86a9fd5fc949c4961a8f42:src/data/step5_drift_detection.py
        """
        logger.info("\n" + "="*80)
        logger.info("HISTORICAL DRIFT DETECTION (QUARTERLY)")
        logger.info("="*80)
        logger.info(f"Dataset: {self.dataset_name}")
        logger.info(f"Shape: {df.shape}")
        logger.info("="*80)

        if 'Date' not in df.columns:
            logger.error("❌ No Date column - cannot detect temporal drift")
            return df, self.drift_report
<<<<<<< HEAD:src/data/step7_drift_detection.py

        # Define periods
        df['Year'] = df['Date'].dt.year

        reference_mask = (df['Year'] >= 2005) & (df['Year'] <= 2010)
=======
        
        # Ensure Date is datetime
        if not pd.api.types.is_datetime64_any_dtype(df['Date']):
            df['Date'] = pd.to_datetime(df['Date'], format='mixed', errors='coerce')
        
        # Define periods (MODIFIED for 1990-2025)
        df['Year'] = df['Date'].dt.year
        
        reference_mask = (df['Year'] >= 1990) & (df['Year'] <= 2005)
>>>>>>> 8d1ecdb96c99c112cf86a9fd5fc949c4961a8f42:src/data/step5_drift_detection.py
        current_mask = (df['Year'] >= 2020) & (df['Year'] <= 2025)

        df_reference = df[reference_mask]
        df_current = df[current_mask]
<<<<<<< HEAD:src/data/step7_drift_detection.py

        logger.info(f"\n📊 Time Periods:")
        logger.info(f"   Reference (2005-2010): {len(df_reference):,} samples")
        logger.info(f"   Current (2020-2025):   {len(df_current):,} samples")

        if len(df_reference) < 100 or len(df_current) < 100:
            logger.warning("⚠️  Insufficient data for drift detection")
=======
        
        logger.info(f"\n📊 Time Periods (UPDATED for 35-year range):")
        logger.info(f"   Reference (1990-2005): {len(df_reference):,} samples")
        logger.info(f"   Current (2020-2025):   {len(df_current):,} samples")
        
        # Need at least 40 quarters for reliable test
        if len(df_reference) < 40 or len(df_current) < 40:
            logger.warning("⚠️  Insufficient quarters for drift detection")
            logger.warning(f"     Reference: {len(df_reference)} samples (need ≥40)")
            logger.warning(f"     Current: {len(df_current)} samples (need ≥40)")
>>>>>>> 8d1ecdb96c99c112cf86a9fd5fc949c4961a8f42:src/data/step5_drift_detection.py
            return df, self.drift_report

        # === TEST EACH FEATURE FOR DRIFT ===
        logger.info("\n" + "="*80)
        logger.info("TESTING FEATURES FOR DRIFT")
        logger.info("="*80)

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        exclude_cols = ['Date', 'Year', 'Month', 'Day']
        test_cols = [col for col in numeric_cols if col not in exclude_cols]

        logger.info(f"\n   Testing {len(test_cols)} numeric features...")

        drifted = []
        stable = []

        for col in test_cols:
            ref_data = df_reference[col].dropna()
            cur_data = df_current[col].dropna()

            if len(ref_data) < 30 or len(cur_data) < 30:
                continue

            # Kolmogorov-Smirnov test
            ks_stat, p_value = stats.ks_2samp(ref_data, cur_data)

            # Calculate magnitude of drift
            mean_ref = ref_data.mean()
            mean_cur = cur_data.mean()
            mean_change_pct = ((mean_cur - mean_ref) /
                               (abs(mean_ref) + 1e-6)) * 100

            std_ref = ref_data.std()
            std_cur = cur_data.std()
<<<<<<< HEAD:src/data/step7_drift_detection.py
            std_change_pct = ((std_cur - std_ref) /
                              (abs(std_ref) + 1e-6)) * 100

            # Determine drift severity
            if p_value < 0.01:  # Significant drift
                severity = 'HIGH' if abs(mean_change_pct) > 50 else 'MEDIUM'

=======
            std_change_pct = ((std_cur - std_ref) / (abs(std_ref) + 1e-6)) * 100
            
            # MODIFIED: More lenient p-value for quarterly (0.02 vs 0.01)
            if p_value < 0.005:  # Very significant drift
                severity = 'HIGH' if abs(mean_change_pct) > 100 else 'MEDIUM'
                
>>>>>>> 8d1ecdb96c99c112cf86a9fd5fc949c4961a8f42:src/data/step5_drift_detection.py
                drifted.append({
                    'feature': str(col),
                    'ks_statistic': float(ks_stat),
                    'p_value': float(p_value),
                    'mean_change_pct': float(round(mean_change_pct, 2)),
                    'std_change_pct': float(round(std_change_pct, 2)),
                    'severity': severity,
                    'mean_reference': float(mean_ref),
                    'mean_current': float(mean_cur)
                })
            elif p_value < 0.02:  # Moderate drift (more lenient for quarterly)
                drifted.append({
                    'feature': str(col),
                    'ks_statistic': float(ks_stat),
                    'p_value': float(p_value),
                    'mean_change_pct': float(round(mean_change_pct, 2)),
                    'std_change_pct': float(round(std_change_pct, 2)),
                    'severity': 'LOW'
                })
            else:  # Stable
                stable.append({
                    'feature': str(col),
                    'p_value': float(p_value)
                })

        # === REPORT RESULTS ===
        logger.info(f"\n📊 Drift Detection Results:")
        logger.info(f"   Features tested: {len(test_cols)}")
        logger.info(f"   Drifted features: {len(drifted)}")
        logger.info(f"   Stable features: {len(stable)}")

        if drifted:
            logger.warning(f"\n⚠️  DRIFTED FEATURES ({len(drifted)}):")

            drifted_df = pd.DataFrame(drifted)
            drifted_df = drifted_df.sort_values(
                'ks_statistic', ascending=False)

            logger.warning("\n   Top 10 drifted features:")
<<<<<<< HEAD:src/data/step7_drift_detection.py
            print(drifted_df[['feature', 'severity', 'mean_change_pct', 'p_value']].head(
                10).to_string(index=False))

=======
            display_cols = ['feature', 'severity', 'mean_change_pct', 'p_value']
            print(drifted_df[display_cols].head(10).to_string(index=False))
            
>>>>>>> 8d1ecdb96c99c112cf86a9fd5fc949c4961a8f42:src/data/step5_drift_detection.py
            # Add flag column for drifted features
            df['Feature_Drift_Flag'] = 0
            for item in drifted:
                if item['severity'] in ['HIGH', 'MEDIUM']:
                    # Flag rows where this feature exists
                    if item['feature'] in df.columns:
                        df.loc[df[item['feature']].notna(),
                               'Feature_Drift_Flag'] = 1

            logger.info(f"\n   ✓ Added 'Feature_Drift_Flag' column")
            logger.info(f"   ℹ️  Flagged {df['Feature_Drift_Flag'].sum():,} rows with drifted features")
        else:
            logger.info(f"\n   ✓ No significant drift detected")

        self.drift_report['drifted_features'] = drifted
        self.drift_report['stable_features'] = stable

        # === GENERATE RECOMMENDATIONS ===
        self._generate_recommendations(drifted)

        # === SAVE REPORT ===
        self._save_report()

        return df, self.drift_report

    def _generate_recommendations(self, drifted: List[Dict]):
        """Generate drift mitigation recommendations."""
        logger.info("\n" + "="*80)
        logger.info("DRIFT MITIGATION RECOMMENDATIONS")
        logger.info("="*80)

        recommendations = []

        if not drifted:
            logger.info("\n   ✓ No drift detected - no mitigation needed")
            recommendations.append("No drift mitigation required")
            self.drift_report['drift_summary']['recommendations'] = recommendations
            return recommendations

        # Count by severity
        high_drift = [d for d in drifted if d.get('severity') == 'HIGH']
        medium_drift = [d for d in drifted if d.get('severity') == 'MEDIUM']

        if high_drift:
            logger.warning(f"\n   📋 RECOMMENDATION 1: Time-Based Validation")
            logger.warning(f"      {len(high_drift)} features have HIGH drift")
<<<<<<< HEAD:src/data/step7_drift_detection.py
            logger.warning(f"      Consider:")
            logger.warning(
                f"        - Normalize features separately for each time period")
            logger.warning(
                f"        - Use rolling statistics instead of absolute values")
            logger.warning(f"        - Add time-period indicator features")

            recommendations.append(
                f"Re-engineer {len(high_drift)} high-drift features with time-aware normalization")

        if medium_drift:
            logger.info(f"\n   📋 RECOMMENDATION 2: Validation Strategy")
            logger.info(
                f"      {len(medium_drift)} features have MEDIUM drift")
            logger.info(f"      Use time-based cross-validation:")
            logger.info(f"        - Train on 2005-2018")
            logger.info(f"        - Validate on 2019-2021")
            logger.info(f"        - Test on 2022-2025")

            recommendations.append(
                "Use time-based cross-validation (not random split)")

        logger.info(f"\n   📋 RECOMMENDATION 3: Model Monitoring")
        logger.info(f"      In production, implement:")
        logger.info(f"        - Real-time drift monitoring")
        logger.info(f"        - Automatic retraining triggers")
        logger.info(f"        - Performance degradation alerts")

        recommendations.append(
            "Implement production drift monitoring (future work)")

=======
            logger.warning(f"      Use time-based cross-validation:")
            logger.warning(f"        - Train: 1990-2015")
            logger.warning(f"        - Validate: 2016-2020")
            logger.warning(f"        - Test: 2021-2025")
            
            recommendations.append(f"Time-based cross-validation (not random split)")
        
        if medium_drift:
            logger.info(f"\n   📋 RECOMMENDATION 2: Feature Re-engineering")
            logger.info(f"      {len(medium_drift)} features have MEDIUM drift")
            logger.info(f"      Consider:")
            logger.info(f"        - Normalize features per time period")
            logger.info(f"        - Use rolling statistics instead of absolute values")
            logger.info(f"        - Add time-period indicator features")
            
            recommendations.append(f"Consider feature normalization for {len(medium_drift)} drifted features")
        
        logger.info(f"\n   📋 RECOMMENDATION 3: Model Retraining Strategy")
        logger.info(f"      In production:")
        logger.info(f"        - Monitor feature distributions quarterly")
        logger.info(f"        - Retrain model annually (or when drift detected)")
        logger.info(f"        - Use sliding window (last 5 years of data)")
        
        recommendations.append("Implement quarterly drift monitoring and annual retraining")
        
>>>>>>> 8d1ecdb96c99c112cf86a9fd5fc949c4961a8f42:src/data/step5_drift_detection.py
        self.drift_report['drift_summary']['recommendations'] = recommendations

        return recommendations

    def _save_report(self):
        """Save drift detection report."""
        output_dir = Path("data/drift_reports")
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
            f"drift_report_{self.dataset_name}_{timestamp}.json"

        with open(json_path, 'w') as f:
            json.dump(self.drift_report, f, indent=2, cls=NumpyEncoder)

        logger.info(f"\n💾 Drift report saved: {json_path}")

        # Save drifted features CSV
        if self.drift_report['drifted_features']:
            csv_path = output_dir / \
                f"drifted_features_{self.dataset_name}_{timestamp}.csv"
            drifted_df = pd.DataFrame(self.drift_report['drifted_features'])
            drifted_df.to_csv(csv_path, index=False)
            logger.info(f"💾 Drifted features: {csv_path}")


def main():
    """Execute historical drift detection."""
<<<<<<< HEAD:src/data/step7_drift_detection.py

    features_dir = Path("data/features")

=======
    
    features_dir = Path("data/processed/")
    
>>>>>>> 8d1ecdb96c99c112cf86a9fd5fc949c4961a8f42:src/data/step5_drift_detection.py
    # Find dataset
    candidates = [
        "features_engineered.csv",
    ]

    filepath = None
    for candidate in candidates:
        if (features_dir / candidate).exists():
            filepath = features_dir / candidate
            logger.info(f"Found: {candidate}")
            break

    if not filepath:
        logger.error("❌ No merged features found!")
        return

    logger.info(f"Loading: {filepath}")
    df = pd.read_csv(filepath)

    # Ensure Date is datetime
    if 'Date' in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df['Date']):
            df['Date'] = pd.to_datetime(
                df['Date'], format='mixed', errors='coerce')
            logger.info(f"   ✓ Converted Date to datetime: {df['Date'].dtype}")

    # Run drift detection
    detector = HistoricalDriftDetectorQuarterly(dataset_name=filepath.stem)
    df_with_flags, report = detector.detect_drift(df)

    # Save output
    output_path = features_dir / f"{filepath.stem}_with_drift_flags.csv"
    df_with_flags.to_csv(output_path, index=False)

    logger.info(f"\n✓ Saved: {output_path}")

    # === FINAL STATUS ===
    logger.info("\n" + "="*80)
    logger.info("✅ DRIFT DETECTION COMPLETE")
    logger.info("="*80)

    total_drifted = len(report.get('drifted_features', []))
    high_drift = len([d for d in report.get(
        'drifted_features', []) if d.get('severity') == 'HIGH'])

    if total_drifted == 0:
        logger.info("\n✅ No significant drift detected")
        logger.info("✅ Data is stable over time (1990-2025)")
    else:
        logger.warning(f"\n⚠️  {total_drifted} features show drift")
        logger.warning(f"   High drift: {high_drift}")
<<<<<<< HEAD:src/data/step7_drift_detection.py
        logger.warning("   Apply recommendations before training")

    logger.info("\n➡️  Next: Feature selection with drift awareness")


if __name__ == "__main__":
    main()
=======
        logger.warning(f"   This is EXPECTED over 35 years!")
        logger.warning("   Apply time-based cross-validation")
    
    logger.info("\n📁 Reports saved to: data/drift_reports/")
    logger.info("\n➡️  Next: Use time-based validation in model training")


if __name__ == "__main__":
    main()
>>>>>>> 8d1ecdb96c99c112cf86a9fd5fc949c4961a8f42:src/data/step5_drift_detection.py
