import pandas as pd
import numpy as np

# Load data
df = pd.read_csv(
    'data/features/merged_features_clean_with_anomaly_flags_with_drift_flags.csv')

print("="*70)
print("DATA STRUCTURE ANALYSIS")
print("="*70)

# Basic structure
print(f"\nShape: {df.shape} (rows × columns)")
print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
print(f"Unique companies: {df['Company'].nunique()}")
print(f"Unique dates: {df['Date'].nunique()}")

# Check one company's timeline
print("\n" + "="*70)
print("AAPL TIMELINE (First 15 rows)")
print("="*70)
aapl = df[df['Company'] == 'AAPL'].sort_values('Date').head(15)

# Check if target columns exist
if 'Revenue_Growth' in df.columns and 'Debt_to_Equity' in df.columns and 'Stock_Return' in df.columns:
    print(aapl[['Date', 'Company', 'Revenue_Growth',
          'Debt_to_Equity', 'Stock_Return']])
else:
    # Show whatever columns are available
    key_cols = ['Date', 'Company']
    available_targets = [col for col in ['Revenue_Growth', 'Debt_to_Equity',
                                         'Stock_Return', 'Revenue', 'Total_Assets'] if col in df.columns]
    print(aapl[key_cols + available_targets[:3]])

# Show how many quarters per company
print("\n" + "="*70)
print("QUARTERS PER COMPANY")
print("="*70)
quarters_per_company = df.groupby('Company').size()
print(f"Min quarters: {quarters_per_company.min()}")
print(f"Max quarters: {quarters_per_company.max()}")
print(f"Average quarters: {quarters_per_company.mean():.1f}")

# Check temporal frequency
print("\n" + "="*70)
print("TEMPORAL FREQUENCY")
print("="*70)
df_sorted = df.sort_values(['Company', 'Date'])
df_sorted['Date'] = pd.to_datetime(df_sorted['Date'])
aapl_dates = df_sorted[df_sorted['Company'] == 'AAPL']['Date']
if len(aapl_dates) > 1:
    date_diffs = aapl_dates.diff().dropna()
    if len(date_diffs) > 0:
        mode_val = date_diffs.mode().values[0] if len(
            date_diffs.mode()) > 0 else date_diffs.median()
        print(f"Typical gap between dates: {mode_val}")
        print(f"Is quarterly? {date_diffs.median().days / 30:.0f} months")

# Show sample of all columns
print("\n" + "="*70)
print("SAMPLE ROWS (first 3)")
print("="*70)
print(df.head(3))

# Show all column names
print("\n" + "="*70)
print(f"ALL COLUMNS ({len(df.columns)} total)")
print("="*70)
print(df.columns.tolist())
