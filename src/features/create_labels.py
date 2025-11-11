"""
Create Financial Distress Labels
Automatic labeling using data-driven methods (no domain expertise required)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime


def create_financial_distress_labels(df):
    """
    Automatic labeling using multiple objective methods

    Methods:
    1. Future performance deterioration
    2. Known historical crisis periods
    3. Current financial health indicators
    4. Market stress signals

    No domain expertise required - all data-driven!
    """

    print("\n" + "="*70)
    print("AUTOMATIC FINANCIAL DISTRESS LABELING")
    print("="*70)
    print(f"Input: {len(df):,} rows")
    print("="*70)

    df = df.copy()
    df = df.sort_values(['Company', 'Date'])
    df['Date'] = pd.to_datetime(df['Date'])

    # Initialize distress indicators
    df['Distress_Score'] = 0

    # ========================================================================
    # METHOD 1: FUTURE PERFORMANCE (Main Method)
    # ========================================================================
    print("\n1️⃣  Future Performance Analysis")
    print("-" * 70)
    print("  Logic: If next quarter shows significant deterioration → distressed")

    for company in df['Company'].unique():
        mask = df['Company'] == company
        company_df = df[mask].copy()

        # Shift to get next quarter values
        df.loc[mask, 'Revenue_Next'] = company_df['Revenue'].shift(-1)
        df.loc[mask, 'Net_Income_Next'] = company_df['Net_Income'].shift(-1)
        df.loc[mask, 'EPS_Next'] = company_df.get('EPS', pd.Series()).shift(-1)
        df.loc[mask, 'Close_Next'] = company_df['Close'].shift(-1)

    # Calculate performance changes
    df['Revenue_Change_Pct'] = (
        (df['Revenue_Next'] - df['Revenue']) /
        df['Revenue'].replace(0, np.nan) * 100
    )
    df['Stock_Change_Pct'] = (
        (df['Close_Next'] - df['Close']) / df['Close'].replace(0, np.nan) * 100
    )

    # Flag future distress
    future_distress = (
        (df['Revenue_Change_Pct'] < -10) |       # Revenue drops >10%
        (df['Net_Income_Next'] < 0) |            # Losses next quarter
        (df['EPS_Next'] < 0) |                   # Negative EPS
        (df['Stock_Change_Pct'] < -20)           # Stock crashes >20%
    )

    df['Distress_Score'] += future_distress.astype(int)
    print(
        f"  ✓ Samples with poor future performance: {future_distress.sum():,}")

    # ========================================================================
    # METHOD 2: KNOWN CRISIS PERIODS
    # ========================================================================
    print("\n2️⃣  Historical Crisis Periods")
    print("-" * 70)
    print("  Using documented financial crises (no expertise needed)")

    crisis_periods = [
        ('2008-09-15', '2009-03-31', ['Financials'], '2008 Financial Crisis'),
        ('2020-03-01', '2020-06-30', ['all'], 'COVID-19 Crisis'),
    ]

    for start, end, sectors, name in crisis_periods:
        start_date = pd.to_datetime(start)
        end_date = pd.to_datetime(end)

        date_mask = (df['Date'] >= start_date) & (df['Date'] <= end_date)

        if sectors == ['all']:
            crisis_mask = date_mask
        else:
            crisis_mask = date_mask & df['Sector'].isin(sectors)

        df.loc[crisis_mask, 'Distress_Score'] += 1
        print(f"  ✓ {name}: {crisis_mask.sum():,} samples marked")

    # ========================================================================
    # METHOD 3: CURRENT FINANCIAL HEALTH
    # ========================================================================
    print("\n3️⃣  Current Financial Health Indicators")
    print("-" * 70)
    print("  Using objective accounting thresholds")

    # Objective financial distress indicators
    health_checks = []

    # Check 1: High leverage
    if 'Debt_to_Equity' in df.columns:
        high_leverage = df['Debt_to_Equity'] > 2.5
        health_checks.append(('High Leverage', high_leverage))

    # Check 2: Liquidity problems
    if 'Current_Ratio' in df.columns:
        low_liquidity = df['Current_Ratio'] < 1.0
        health_checks.append(('Low Liquidity', low_liquidity))

    # Check 3: Unprofitable
    if 'Net_Income' in df.columns:
        unprofitable = df['Net_Income'] < 0
        health_checks.append(('Unprofitable', unprofitable))

    # Check 4: Negative margins
    if 'Profit_Margin' in df.columns:
        negative_margins = df['Profit_Margin'] < -5
        health_checks.append(('Negative Margins', negative_margins))

    # Count failed health checks
    if health_checks:
        health_distress = sum([check[1]
                              for check in health_checks]) >= 2  # 2+ failures
        df['Distress_Score'] += health_distress.astype(int)

        for name, check in health_checks:
            print(f"  {name}: {check.sum():,} samples")
        print(
            f"  ✓ Samples failing 2+ health checks: {health_distress.sum():,}")

    # ========================================================================
    # METHOD 4: EXTREME POOR PERFORMERS (Statistical)
    # ========================================================================
    print("\n4️⃣  Statistical Threshold (Bottom Performers)")
    print("-" * 70)

    # Create composite score for each quarter
    df['Performance_Rank'] = df.groupby('Date')['Revenue'].rank(pct=True)

    # Bottom 15% performers each quarter
    bottom_performers = df['Performance_Rank'] < 0.15
    df['Distress_Score'] += bottom_performers.astype(int)
    print(f"  ✓ Bottom 15% performers: {bottom_performers.sum():,}")

    # ========================================================================
    # COMBINE ALL METHODS
    # ========================================================================
    print("\n" + "="*70)
    print("FINAL LABEL CREATION")
    print("="*70)

    # Label as distressed if 2+ methods flagged it
    df['Financial_Distress'] = (df['Distress_Score'] >= 2).astype(int)

    print(f"\nDistress Score Distribution:")
    for score in sorted(df['Distress_Score'].unique()):
        count = (df['Distress_Score'] == score).sum()
        pct = count / len(df) * 100
        print(f"  Score {score}: {count:,} samples ({pct:.1f}%)")

    print(f"\n📊 FINAL LABELS:")
    label_counts = df['Financial_Distress'].value_counts()
    healthy_count = label_counts.get(0, 0)
    distressed_count = label_counts.get(1, 0)

    print(
        f"  Healthy (0):    {healthy_count:,} ({healthy_count/len(df)*100:.1f}%)")
    print(
        f"  Distressed (1): {distressed_count:,} ({distressed_count/len(df)*100:.1f}%)")

    # Validate balance
    distress_pct = distressed_count / len(df) * 100

    if distress_pct < 5:
        print("\n⚠ WARNING: Very few distressed samples (<5%)")
        print("  Recommendation: Lower threshold to Distress_Score >= 1")
    elif distress_pct > 50:
        print("\n⚠ WARNING: Too many distressed samples (>50%)")
        print("  Recommendation: Raise threshold to Distress_Score >= 3")
    else:
        print(
            f"\n✓ Good balance: {distress_pct:.1f}% distressed (acceptable range)")

    # Clean up temporary columns
    temp_cols = [
        'Revenue_Next', 'Net_Income_Next', 'EPS_Next', 'Close_Next',
        'Revenue_Change_Pct', 'Stock_Change_Pct', 'Performance_Rank',
        'Distress_Score', 'Stock_Volatility_4Q'
    ]
    df = df.drop(columns=[col for col in temp_cols if col in df.columns])

    # Remove rows without "next quarter" data
    df = df[df['Financial_Distress'].notna()].copy()

    print(f"\n✅ Labeling complete!")
    print(f"  Final dataset: {len(df):,} rows with labels")

    return df


def validate_labels(df):
    """Validate that labels make sense"""

    print("\n" + "="*70)
    print("LABEL VALIDATION")
    print("="*70)

    # Check by time period
    print("\nLabels by Time Period:")
    df['Year'] = pd.to_datetime(df['Date']).dt.year
    period_labels = df.groupby('Year')['Financial_Distress'].agg([
        'sum', 'count', 'mean'])
    period_labels.columns = ['Distressed', 'Total', 'Distress_Rate']
    print(period_labels)

    # Check by sector
    if 'Sector' in df.columns:
        print("\nLabels by Sector:")
        sector_labels = df.groupby('Sector')['Financial_Distress'].agg([
            'sum', 'count', 'mean'])
        sector_labels.columns = ['Distressed', 'Total', 'Distress_Rate']
        print(sector_labels)

    # Sanity checks
    print("\n✅ Sanity Checks:")

    # 2008-2009 should have high distress rate
    crisis_2008 = df[(df['Year'] == 2008) | (df['Year'] == 2009)]
    if len(crisis_2008) > 0:
        crisis_rate = crisis_2008['Financial_Distress'].mean()
        print(f"  2008-2009 distress rate: {crisis_rate*100:.1f}%")
        if crisis_rate > 0.30:
            print("    ✓ High distress during 2008 crisis (expected)")
        else:
            print("    ⚠ Low distress during 2008 crisis (unexpected)")

    # 2020 COVID should have high distress
    covid_2020 = df[df['Year'] == 2020]
    if len(covid_2020) > 0:
        covid_rate = covid_2020['Financial_Distress'].mean()
        print(f"  2020 COVID distress rate: {covid_rate*100:.1f}%")
        if covid_rate > 0.25:
            print("    ✓ High distress during COVID (expected)")
        else:
            print("    ⚠ Low distress during COVID (unexpected)")

    # Normal periods should have low distress
    normal_periods = df[(df['Year'] >= 2015) & (df['Year'] <= 2019)]
    if len(normal_periods) > 0:
        normal_rate = normal_periods['Financial_Distress'].mean()
        print(
            f"  2015-2019 normal period distress rate: {normal_rate*100:.1f}%")
        if normal_rate < 0.15:
            print("    ✓ Low distress during normal period (expected)")
        else:
            print("    ⚠ High distress during normal period (check labeling)")


def main():
    # 1. INPUT: Reads your merged dataset
    input_file = 'data/features/merged_features_clean_with_anomaly_flags_with_drift_flags.csv'
    df = pd.read_csv(input_file)

    # 2. PROCESS: Creates Financial_Distress column (0 or 1)
    df_labeled = create_financial_distress_labels(df)

    # 3. OUTPUT: Saves to a NEW file
    output_file = 'data/processed/labeled_data.csv'  # ← NEW FILE HERE
    df_labeled.to_csv(output_file, index=False)


if __name__ == "__main__":
    import sys

    try:
        df_labeled = main()
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Labeling failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
