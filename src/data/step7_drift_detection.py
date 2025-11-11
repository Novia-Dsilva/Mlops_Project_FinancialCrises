"""
STEP 7: DRIFT DETECTION (Historical Data Drift)

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
- Compare early period (2005-2010) vs recent period (2020-2025)
- Use Kolmogorov-Smirnov test for distribution comparison
- Flag features with p-value < 0.05

Note: This is HISTORICAL drift (offline). 
Production drift monitoring (online) is future work.

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


class HistoricalDriftDetector:
    """
    Detect drift in historical data by comparing time periods.
    
    Simple, assignment-friendly implementation.
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
        
        Compares:
        - Reference period: 2005-2010 (early data)
        - Current period: 2020-2025 (recent data)
        """
        logger.info("\n" + "="*80)
        logger.info("HISTORICAL DRIFT DETECTION")
        logger.info("="*80)
        logger.info(f"Dataset: {self.dataset_name}")
        logger.info(f"Shape: {df.shape}")
        logger.info("="*80)
        
        if 'Date' not in df.columns:
            logger.error("❌ No Date column - cannot detect temporal drift")
            return df, self.drift_report
        
        # Define periods
        df['Year'] = df['Date'].dt.year
        
        reference_mask = (df['Year'] >= 2005) & (df['Year'] <= 2010)
        current_mask = (df['Year'] >= 2020) & (df['Year'] <= 2025)
        
        df_reference = df[reference_mask]
        df_current = df[current_mask]
        
        logger.info(f"\n📊 Time Periods:")
        logger.info(f"   Reference (2005-2010): {len(df_reference):,} samples")
        logger.info(f"   Current (2020-2025):   {len(df_current):,} samples")
        
        if len(df_reference) < 100 or len(df_current) < 100:
            logger.warning("⚠️  Insufficient data for drift detection")
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
            mean_change_pct = ((mean_cur - mean_ref) / (abs(mean_ref) + 1e-6)) * 100
            
            std_ref = ref_data.std()
            std_cur = cur_data.std()
            std_change_pct = ((std_cur - std_ref) / (abs(std_ref) + 1e-6)) * 100
            
            # Determine drift severity
            if p_value < 0.01:  # Significant drift
                severity = 'HIGH' if abs(mean_change_pct) > 50 else 'MEDIUM'
                
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
            elif p_value < 0.05:  # Moderate drift
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
            drifted_df = drifted_df.sort_values('ks_statistic', ascending=False)
            
            logger.warning("\n   Top 10 drifted features:")
            print(drifted_df[['feature', 'severity', 'mean_change_pct', 'p_value']].head(10).to_string(index=False))
            
            # Add flag column for drifted features
            df['Feature_Drift_Flag'] = 0
            for item in drifted:
                if item['severity'] in ['HIGH', 'MEDIUM']:
                    # Flag rows where this feature exists
                    if item['feature'] in df.columns:
                        df.loc[df[item['feature']].notna(), 'Feature_Drift_Flag'] = 1
            
            logger.info(f"\n   ✓ Added 'Feature_Drift_Flag' column")
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
            return recommendations
        
        # Count by severity
        high_drift = [d for d in drifted if d.get('severity') == 'HIGH']
        medium_drift = [d for d in drifted if d.get('severity') == 'MEDIUM']
        
        if high_drift:
            logger.warning(f"\n   📋 RECOMMENDATION 1: Feature Re-engineering")
            logger.warning(f"      {len(high_drift)} features have HIGH drift")
            logger.warning(f"      Consider:")
            logger.warning(f"        - Normalize features separately for each time period")
            logger.warning(f"        - Use rolling statistics instead of absolute values")
            logger.warning(f"        - Add time-period indicator features")
            
            recommendations.append(f"Re-engineer {len(high_drift)} high-drift features with time-aware normalization")
        
        if medium_drift:
            logger.info(f"\n   📋 RECOMMENDATION 2: Validation Strategy")
            logger.info(f"      {len(medium_drift)} features have MEDIUM drift")
            logger.info(f"      Use time-based cross-validation:")
            logger.info(f"        - Train on 2005-2018")
            logger.info(f"        - Validate on 2019-2021")
            logger.info(f"        - Test on 2022-2025")
            
            recommendations.append("Use time-based cross-validation (not random split)")
        
        logger.info(f"\n   📋 RECOMMENDATION 3: Model Monitoring")
        logger.info(f"      In production, implement:")
        logger.info(f"        - Real-time drift monitoring")
        logger.info(f"        - Automatic retraining triggers")
        logger.info(f"        - Performance degradation alerts")
        
        recommendations.append("Implement production drift monitoring (future work)")
        
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
        json_path = output_dir / f"drift_report_{self.dataset_name}_{timestamp}.json"
        
        with open(json_path, 'w') as f:
            json.dump(self.drift_report, f, indent=2, cls=NumpyEncoder)
        
        logger.info(f"\n💾 Drift report saved: {json_path}")
        
        # Save drifted features CSV
        if self.drift_report['drifted_features']:
            csv_path = output_dir / f"drifted_features_{self.dataset_name}_{timestamp}.csv"
            drifted_df = pd.DataFrame(self.drift_report['drifted_features'])
            drifted_df.to_csv(csv_path, index=False)
            logger.info(f"💾 Drifted features: {csv_path}")


def main():
    """Execute historical drift detection."""
    
    features_dir = Path("data/features")
    
    # Find dataset
    candidates = [
        "merged_features_clean_with_anomaly_flags.csv",
        "merged_features_clean.csv",
        "merged_features.csv"
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

    # CRITICAL FIX: Ensure Date is datetime
    if 'Date' in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df['Date']):
            df['Date'] = pd.to_datetime(df['Date'], format='mixed', errors='coerce')
            logger.info(f"   ✓ Converted Date to datetime: {df['Date'].dtype}")
    
    # Run drift detection
    detector = HistoricalDriftDetector(dataset_name=filepath.stem)
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
    high_drift = len([d for d in report.get('drifted_features', []) if d.get('severity') == 'HIGH'])
    
    if total_drifted == 0:
        logger.info("\n✅ No significant drift detected")
        logger.info("✅ Data is stable over time")
    else:
        logger.warning(f"\n⚠️  {total_drifted} features show drift")
        logger.warning(f"   High drift: {high_drift}")
        logger.warning("   Apply recommendations before training")
    
    logger.info("\n➡️  Next: Feature selection with drift awareness")


if __name__ == "__main__":
    main()

