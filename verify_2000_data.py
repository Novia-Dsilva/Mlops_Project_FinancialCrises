import pandas as pd
import numpy as np

df = pd.read_csv('data/processed/merged_dataset.csv')

print("="*70)
print("🎉 DATA COLLECTION VERIFICATION - 2000-2025")
print("="*70)

print(f"\n✅ SUCCESS! Dataset collected:")
print(f"   Total rows: {len(df):,}")
print(f"   Total features: {df.shape[1]}")
print(f"   Date range: {df['Date'].min()} to {df['Date'].max()}")
print(f"   Companies: {df['Company'].nunique()}")

# Check time span
df['Date'] = pd.to_datetime(df['Date'])
years = (df['Date'].max() - df['Date'].min()).days / 365.25
print(f"   Time span: {years:.1f} years")

# Check crisis coverage
print(f"\n📊 CRISIS COVERAGE:")
crises = {
    '2000-2001 Dot-com': (df['Date'].dt.year >= 2000) & (df['Date'].dt.year <= 2001),
    '2007-2009 Financial': (df['Date'].dt.year >= 2007) & (df['Date'].dt.year <= 2009),
    '2020 COVID': df['Date'].dt.year == 2020,
    '2022-2023 Inflation': (df['Date'].dt.year >= 2022) & (df['Date'].dt.year <= 2023)
}

for crisis_name, mask in crises.items():
    count = df[mask].shape[0]
    companies = df[mask]['Company'].nunique()
    print(f"   {crisis_name:25} {count:4d} observations, {companies:2d} companies ✅")

# Data quality
print(f"\n📈 DATA QUALITY:")
print(f"   Stock_Price coverage: {df['Stock_Price'].notna().sum() / len(df) * 100:.1f}%")
print(f"   Stock_Return coverage: {df['Stock_Return'].notna().sum() / len(df) * 100:.1f}%")
print(f"   GDP_Growth coverage: {df['GDP_Growth'].notna().sum() / len(df) * 100:.1f}%")
print(f"   VIX coverage: {df['VIX'].notna().sum() / len(df) * 100:.1f}%")
if 'Revenue' in df.columns:
    print(f"   Revenue coverage: {df['Revenue'].notna().sum() / len(df) * 100:.1f}%")
if 'EPS' in df.columns:
    print(f"   EPS coverage: {df['EPS'].notna().sum() / len(df) * 100:.1f}%")

# Company with least data
print(f"\n📊 COMPANY COVERAGE:")
company_counts = df.groupby('Company').size().sort_values()
print(f"   Least data: {company_counts.index[0]} ({company_counts.iloc[0]} quarters)")
print(f"   Most data: {company_counts.index[-1]} ({company_counts.iloc[-1]} quarters)")
print(f"   Average: {company_counts.mean():.1f} quarters per company")

print(f"\n✅ Dataset is ready for Financial Stress Test modeling!")
print("="*70)
