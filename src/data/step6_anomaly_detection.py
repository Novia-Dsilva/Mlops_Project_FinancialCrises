"""
STEP 6: ANOMALY DETECTION (FLAG ONLY - NO DATA MODIFICATION)

Detects anomalies and adds flag columns for model awareness.
FIXED: Proper JSON serialization for all numpy/pandas types.

Usage:
    python src/validation/step6_anomaly_detection.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Any
from datetime import datetime
import json
import warnings

warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# JSON SERIALIZATION FIX
# ============================================================================

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy/pandas types."""
    
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        elif pd.isna(obj):
            return None
        return super().default(obj)


# ============================================================================
# ANOMALY DETECTOR
# ============================================================================

class AnomalyDetectorFlagOnly:
    """
    Anomaly detector that only flags anomalies without modifying data.
    """
    
    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        self.anomaly_report = {
            'dataset': dataset_name,
            'timestamp': datetime.now().isoformat(),
            'anomalies': [],
            'critical_count': 0,
            'total_count': 0,
            'flags_created': []
        }
    
    # ========== METHOD 1: STATISTICAL OUTLIERS (IQR) ==========
    
    def detect_statistical_outliers(self, df: pd.DataFrame, group_col: str = None) -> pd.DataFrame:
        """
        Detect statistical outliers using IQR method.
        Adds flag columns: {column}_Outlier_Flag
        """
        logger.info("\n[1/3] Statistical Outlier Detection (IQR)...")
        
        df_flagged = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        exclude_cols = ['Date', 'Year', 'Month', 'Day', 'Quarter']
        numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        outlier_summary = []
        
        # Select important columns to flag
        important_cols = [
            'Stock_Return_1D', 'Stock_Return_22D', 'Stock_Volatility_22D',
            'Revenue_Growth_YoY', 'Profit_Margin', 'ROE', 'ROA', 
            'Debt_to_Equity', 'VIX', 'SP500_Return_1D'
        ]
        
        cols_to_flag = [col for col in important_cols if col in numeric_cols]
        
        logger.info(f"   Analyzing {len(numeric_cols)} numeric columns...")
        logger.info(f"   Will create flags for {len(cols_to_flag)} key columns...")
        
        for col in numeric_cols:
            if group_col and group_col in df.columns:
                # Detect per group
                for group in df[group_col].unique():
                    group_mask = df[group_col] == group
                    group_data = df.loc[group_mask, col].dropna()
                    
                    if len(group_data) < 10:
                        continue
                    
                    Q1 = group_data.quantile(0.25)
                    Q3 = group_data.quantile(0.75)
                    IQR = Q3 - Q1
                    
                    lower = Q1 - 3 * IQR
                    upper = Q3 + 3 * IQR
                    
                    outlier_mask_group = (group_data < lower) | (group_data > upper)
                    outlier_count = outlier_mask_group.sum()
                    
                    if outlier_count > 0:
                        outlier_summary.append({
                            'column': str(col),
                            'group': str(group),
                            'count': int(outlier_count),
                            'percentage': float(round((outlier_count / len(group_data)) * 100, 2))
                        })
                        
                        # Add flag column
                        if col in cols_to_flag:
                            flag_col = f"{col}_Outlier_Flag"
                            
                            if flag_col not in df_flagged.columns:
                                df_flagged[flag_col] = 0
                            
                            outlier_indices = group_data[outlier_mask_group].index
                            df_flagged.loc[outlier_indices, flag_col] = 1
            else:
                # Detect globally
                data = df[col].dropna()
                
                if len(data) < 10:
                    continue
                
                Q1 = data.quantile(0.25)
                Q3 = data.quantile(0.75)
                IQR = Q3 - Q1
                
                lower = Q1 - 3 * IQR
                upper = Q3 + 3 * IQR
                
                outlier_mask = (df[col] < lower) | (df[col] > upper)
                outlier_count = outlier_mask.sum()
                
                if outlier_count > 0:
                    outlier_summary.append({
                        'column': str(col),
                        'count': int(outlier_count),
                        'percentage': float(round((outlier_count / len(data)) * 100, 2))
                    })
                    
                    if col in cols_to_flag:
                        flag_col = f"{col}_Outlier_Flag"
                        df_flagged[flag_col] = outlier_mask.astype(int)
        
        # Log results
        if outlier_summary:
            outlier_df = pd.DataFrame(outlier_summary)
            outlier_df = outlier_df.sort_values('count', ascending=False)
            
            logger.info(f"   Found outliers in {len(outlier_summary)} column-group combinations:")
            print(outlier_df.head(10).to_string(index=False))
            
            flags_created = [col for col in df_flagged.columns if col.endswith('_Outlier_Flag')]
            logger.info(f"\n   ✓ Created {len(flags_created)} outlier flag columns")
            
            self.anomaly_report['statistical_outliers'] = outlier_summary
            self.anomaly_report['flags_created'].extend(flags_created)
            self.anomaly_report['total_count'] += sum(item['count'] for item in outlier_summary)
        else:
            logger.info(f"   ✓ No statistical outliers detected")
        
        return df_flagged
    
    # ========== METHOD 2: BUSINESS RULE VIOLATIONS ==========
    
    def detect_business_rules(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect business rule violations.
        Adds flag columns: {column}_Violation_Flag
        """
        logger.info("\n[2/3] Business Rule Violation Detection...")
        
        df_flagged = df.copy()
        violations = []
        
        # === RULE 1: Negative values where impossible ===
        non_negative_rules = {
            'VIX': 'VIX cannot be negative',
            'Volume': 'Trading volume cannot be negative',
            'CPI': 'CPI cannot be negative',
            'Total_Assets': 'Total assets cannot be negative',
            'Stock_Price': 'Stock price cannot be negative',
            'Close': 'Close price cannot be negative'
        }
        
        for col, rule in non_negative_rules.items():
            if col in df.columns:
                negative_mask = df[col] < 0
                negative_count = negative_mask.sum()
                
                if negative_count > 0:
                    violations.append({
                        'rule': str(rule),
                        'column': str(col),
                        'count': int(negative_count),
                        'severity': 'CRITICAL'
                    })
                    
                    flag_col = f"{col}_Negative_Flag"
                    df_flagged[flag_col] = negative_mask.astype(int)
                    self.anomaly_report['flags_created'].append(flag_col)
                    
                    self.anomaly_report['critical_count'] += int(negative_count)
        
        # === RULE 2: Impossible ratios ===
        ratio_rules = {
            'Profit_Margin': (-100, 100, 'Profit margin must be between -100% and 100%'),
            'ROE': (-200, 200, 'ROE must be between -200% and 200%'),
            'Debt_to_Equity': (0, 100, 'Debt-to-equity typically < 100'),
        }
        
        for col, (min_val, max_val, rule) in ratio_rules.items():
            if col in df.columns:
                violation_mask = (df[col] < min_val) | (df[col] > max_val)
                violation_count = violation_mask.sum()
                
                if violation_count > 0:
                    violations.append({
                        'rule': str(rule),
                        'column': str(col),
                        'count': int(violation_count),
                        'severity': 'HIGH',
                        'range': [float(min_val), float(max_val)]
                    })
                    
                    flag_col = f"{col}_Extreme_Flag"
                    df_flagged[flag_col] = violation_mask.astype(int)
                    self.anomaly_report['flags_created'].append(flag_col)
        
        # === RULE 3: Economic impossibilities ===
        if 'GDP_Growth_90D' in df.columns:
            extreme_mask = df['GDP_Growth_90D'].abs() > 20
            extreme_count = extreme_mask.sum()
            
            if extreme_count > 0:
                violations.append({
                    'rule': 'GDP growth > 20% quarterly is impossible',
                    'column': 'GDP_Growth_90D',
                    'count': int(extreme_count),
                    'severity': 'CRITICAL'
                })
                
                flag_col = "GDP_Growth_Extreme_Flag"
                df_flagged[flag_col] = extreme_mask.astype(int)
                self.anomaly_report['flags_created'].append(flag_col)
                
                self.anomaly_report['critical_count'] += int(extreme_count)
        
        # Log violations
        if violations:
            logger.info(f"   Found {len(violations)} business rule violations:")
            
            for v in violations:
                severity_icon = "🚨" if v['severity'] == 'CRITICAL' else "⚠️"
                logger.info(f"   {severity_icon} {v['rule']}")
                logger.info(f"      Column: {v['column']}, Count: {v['count']}")
            
            self.anomaly_report['business_rule_violations'] = violations
            self.anomaly_report['total_count'] += sum(v['count'] for v in violations)
        else:
            logger.info(f"   ✓ No business rule violations")
        
        return df_flagged
    
    # ========== METHOD 3: TEMPORAL ANOMALIES ==========
    
    def detect_temporal_anomalies(self, df: pd.DataFrame, group_col: str = None) -> pd.DataFrame:
        """
        Detect temporal anomalies (sudden jumps).
        Adds flag columns: {column}_Jump_Flag
        """
        logger.info("\n[3/3] Temporal Anomaly Detection...")
        
        if 'Date' not in df.columns:
            logger.info("   No Date column - skipping temporal detection")
            return df
        
        df_flagged = df.copy()
        temporal_anomalies = []
        
        # Check key columns
        check_cols = ['Stock_Price', 'Close', 'Revenue']
        available_cols = [col for col in check_cols if col in df.columns]
        
        if not available_cols:
            logger.info("   No suitable columns for temporal detection")
            return df_flagged
        
        logger.info(f"   Checking {len(available_cols)} columns for sudden jumps...")
        
        for col in available_cols:
            flag_col = f"{col}_Jump_Flag"
            df_flagged[flag_col] = 0
            
            if group_col and group_col in df.columns:
                for company in df[group_col].unique():
                    company_df = df[df[group_col] == company].sort_values('Date').copy()
                    
                    if len(company_df) < 10:
                        continue
                    
                    pct_change = company_df[col].pct_change().abs() * 100
                    jump_mask = pct_change > 50
                    jump_count = jump_mask.sum()
                    
                    if jump_count > 0:
                        jump_indices = company_df[jump_mask].index
                        df_flagged.loc[jump_indices, flag_col] = 1
                        
                        jump_years = [int(year) for year in company_df.loc[jump_mask, 'Date'].dt.year.unique()]
                        crisis_years = [2008, 2009, 2020]
                        is_crisis = any(year in crisis_years for year in jump_years)
                        
                        temporal_anomalies.append({
                            'column': str(col),
                            'company': str(company),
                            'count': int(jump_count),
                            'years': jump_years,
                            'is_crisis': bool(is_crisis),
                            'severity': 'LOW' if is_crisis else 'HIGH'
                        })
                        
                        if not is_crisis:
                            self.anomaly_report['critical_count'] += int(jump_count)
            else:
                df_sorted = df.sort_values('Date').copy()
                pct_change = df_sorted[col].pct_change().abs() * 100
                
                jump_mask = pct_change > 50
                jump_count = jump_mask.sum()
                
                if jump_count > 0:
                    jump_indices = df_sorted[jump_mask].index
                    df_flagged.loc[jump_indices, flag_col] = 1
                    
                    jump_years = [int(year) for year in df_sorted.loc[jump_mask, 'Date'].dt.year.unique()]
                    
                    temporal_anomalies.append({
                        'column': str(col),
                        'count': int(jump_count),
                        'years': jump_years
                    })
        
        if temporal_anomalies:
            logger.info(f"   Found {len(temporal_anomalies)} temporal anomaly patterns:")
            
            for anomaly in temporal_anomalies[:10]:
                severity_icon = "📊" if anomaly.get('is_crisis', False) else "⚠️"
                company_str = f" ({anomaly['company']})" if 'company' in anomaly else ""
                logger.info(f"   {severity_icon} {anomaly['column']}{company_str}: " +
                          f"{anomaly['count']} jumps in years {anomaly['years']}")
            
            flags_created = [col for col in df_flagged.columns if col.endswith('_Jump_Flag')]
            logger.info(f"\n   ✓ Created {len(flags_created)} jump flag columns")
            
            self.anomaly_report['temporal_anomalies'] = temporal_anomalies
            self.anomaly_report['flags_created'].extend(flags_created)
        else:
            logger.info(f"   ✓ No temporal anomalies detected")
        
        return df_flagged
    
    # ========== ALERTING ==========
    
    def send_alert(self):
        """Send alert if critical anomalies found."""
        
        if self.anomaly_report['critical_count'] == 0:
            logger.info("\n✓ No critical anomalies - no alert needed")
            return
        
        logger.warning(f"\n🚨 CRITICAL ANOMALIES DETECTED!")
        
        alert_message = f"""
╔════════════════════════════════════════════════════════════════╗
║           CRITICAL ANOMALIES DETECTED                          ║
╚════════════════════════════════════════════════════════════════╝

Dataset: {self.dataset_name}
Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Critical Anomalies: {self.anomaly_report['critical_count']:,}
Total Anomalies: {self.anomaly_report['total_count']:,}

ACTION REQUIRED:
1. Review anomaly report in data/anomaly_reports/
2. Verify data sources for critical issues
3. Check if anomalies are valid (crisis periods) or errors

════════════════════════════════════════════════════════════════
        """
        
        logger.warning(alert_message)
    
    # ========== MAIN PIPELINE ==========
    
    def run_detection(self, df: pd.DataFrame, group_col: str = None) -> Tuple[pd.DataFrame, Dict]:
        """Run all anomaly detection methods."""
        
        logger.info("\n" + "="*80)
        logger.info("ANOMALY DETECTION (FLAG-ONLY MODE)")
        logger.info("="*80)
        logger.info(f"Dataset: {self.dataset_name}")
        logger.info(f"Shape: {df.shape}")
        logger.info(f"Mode: FLAG ONLY - No data modification")
        logger.info("="*80)
        
        # Run all detections
        df_result = self.detect_statistical_outliers(df, group_col)
        df_result = self.detect_business_rules(df_result)
        df_result = self.detect_temporal_anomalies(df_result, group_col)
        
        # Summary
        logger.info("\n" + "="*80)
        logger.info("ANOMALY DETECTION SUMMARY")
        logger.info("="*80)
        
        logger.info(f"\n📊 Anomalies Detected:")
        logger.info(f"   Total: {self.anomaly_report['total_count']:,}")
        logger.info(f"   Critical: {self.anomaly_report['critical_count']:,}")
        
        logger.info(f"\n🏷️  Flags Created: {len(self.anomaly_report['flags_created'])}")
        
        # Show flags created
        if self.anomaly_report['flags_created']:
            logger.info(f"\n   Flag columns created:")
            unique_flags = list(set(self.anomaly_report['flags_created']))
            for flag in unique_flags[:15]:
                logger.info(f"     - {flag}")
            if len(unique_flags) > 15:
                logger.info(f"     ... and {len(unique_flags) - 15} more")
        
        # Data integrity check
        original_cols = len(df.columns)
        final_cols = len(df_result.columns)
        flags_added = final_cols - original_cols
        
        logger.info(f"\n📊 Data Integrity:")
        logger.info(f"   Original columns: {original_cols}")
        logger.info(f"   Final columns: {final_cols}")
        logger.info(f"   Flags added: {flags_added}")
        logger.info(f"   Original rows: {len(df):,}")
        logger.info(f"   Final rows: {len(df_result):,}")
        logger.info(f"   ✓ No rows removed (flag-only mode)")
        
        # Send alert
        self.send_alert()
        
        # Save report
        self._save_report()
        
        return df_result, self.anomaly_report
    
    def _save_report(self):
        """Save anomaly report with proper JSON serialization."""
        output_dir = Path("data/anomaly_reports")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save JSON report with custom encoder
        json_path = output_dir / f"anomaly_report_{self.dataset_name}_{timestamp}.json"
        
        try:
            with open(json_path, 'w') as f:
                json.dump(self.anomaly_report, f, indent=2, cls=NumpyEncoder)
            
            logger.info(f"\n💾 Anomaly report saved: {json_path}")
        except Exception as e:
            logger.error(f"   ❌ Failed to save JSON report: {e}")
            logger.error(f"   Error type: {type(e).__name__}")
        
        # Save CSV summary
        if self.anomaly_report.get('statistical_outliers'):
            csv_path = output_dir / f"outlier_summary_{self.dataset_name}_{timestamp}.csv"
            
            try:
                outlier_df = pd.DataFrame(self.anomaly_report['statistical_outliers'])
                outlier_df.to_csv(csv_path, index=False)
                
                logger.info(f"💾 Outlier summary saved: {csv_path}")
            except Exception as e:
                logger.error(f"   ❌ Failed to save CSV summary: {e}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Run anomaly detection on cleaned merged data."""
    
    features_dir = Path("data/features")
    
    # Find the latest cleaned file
    candidates = [
        "merged_features_clean.csv",
        "macro_features_clean.csv",
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
        logger.error("Run Step 3c first: python step3c_post_merge_cleaning.py")
        return
    
    logger.info(f"Loading: {filepath}")
    df = pd.read_csv(filepath)

    # CRITICAL FIX: Ensure Date is datetime
    if 'Date' in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df['Date']):
            df['Date'] = pd.to_datetime(df['Date'], format='mixed', errors='coerce')
        logger.info(f"   Date column: {df['Date'].dtype}")
    
    # Run detection
    detector = AnomalyDetectorFlagOnly(dataset_name=filepath.stem)
    
    group_col = 'Company' if 'Company' in df.columns else None
    
    df_flagged, report = detector.run_detection(df, group_col=group_col)
    
    # Save output with flags
    output_path = features_dir / f"{filepath.stem}_with_anomaly_flags.csv"
    df_flagged.to_csv(output_path, index=False)
    
    logger.info(f"\n✓ Saved flagged data: {output_path}")
    logger.info(f"  Original shape: {df.shape}")
    logger.info(f"  Final shape: {df_flagged.shape}")
    logger.info(f"  Flags added: {df_flagged.shape[1] - df.shape[1]}")
    
    logger.info("\n" + "="*80)
    logger.info("✅ ANOMALY DETECTION COMPLETE")
    logger.info("="*80)
    
    if report['critical_count'] > 0:
        logger.warning(f"\n⚠️  {report['critical_count']:,} CRITICAL anomalies found!")
        logger.warning("   Review anomaly report before proceeding")
    else:
        logger.info("\n✅ No critical anomalies detected")
    
    logger.info("\n➡️  Next Steps:")
    logger.info(f"   1. Review: data/anomaly_reports/")
    logger.info(f"   2. Use: {output_path}")
    logger.info(f"   3. Next: python step4_bias_detection.py")


if __name__ == "__main__":
    main()