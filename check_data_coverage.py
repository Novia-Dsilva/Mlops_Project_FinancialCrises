import pandas as pd
import numpy as np

print("="*70)
print("📊 DATA COVERAGE ANALYSIS")
print("="*70)

# Load dataset
df = pd.read_csv('data/processed/merged_dataset.csv')

print(f"\n1️⃣ OVERALL DATASET:")
print(f"   Total rows: {len(df):,}")
print(f"   Date range: {df['Date'].min()} to {df['Date'].max()}")
print(f"   Companies: {df['Company'].nunique()}")

print(f"\n2️⃣ DATE COVERAGE BY COMPANY:")
print(f"{'Company':<10} {'Sector':<25} {'First Date':<15} {'Last Date':<15} {'Quarters':<10}")
print("-"*85)

for company in sorted(df['Company'].unique()):
    company_data = df[df['Company'] == company]
    sector = company_data['Sector'].iloc[0]
    first_date = company_data['Date'].min()
    last_date = company_data['Date'].max()
    quarters = len(company_data)
    print(f"{company:<10} {sector:<25} {first_date:<15} {last_date:<15} {quarters:<10}")

print(f"\n3️⃣ DATA AVAILABILITY BY YEAR:")
df['Year'] = pd.to_datetime(df['Date']).dt.year
year_counts = df.groupby('Year').size().sort_index()

print(f"{'Year':<10} {'Observations':<15} {'Companies with Data':<25}")
print("-"*50)
for year, count in year_counts.items():
    companies_with_data = df[df['Year'] == year]['Company'].nunique()
    print(f"{year:<10} {count:<15} {companies_with_data:<25}")

print(f"\n4️⃣ FRED DATA COVERAGE:")
fred_cols = ['GDP_Growth', 'CPI_Inflation', 'Unemployment_Rate', 'VIX']
print(f"   Checking FRED columns: {', '.join(fred_cols)}")
for col in fred_cols:
    if col in df.columns:
        non_null = df[col].notna().sum()
        print(f"   {col:<25} {non_null:,} non-null values")

print(f"\n5️⃣ MISSING DATA ANALYSIS:")
print(f"   Total cells: {df.size:,}")
print(f"   Missing cells: {df.isnull().sum().sum():,}")
print(f"   Missing percentage: {df.isnull().sum().sum() / df.size * 100:.2f}%")

# Check specific companies
print(f"\n6️⃣ SAMPLE: JPMorgan Chase (JPM) - First 5 records:")
jpm = df[df['Company'] == 'JPM'].head(5)
print(jpm[['Date', 'Company', 'Revenue', 'EPS', 'GDP_Growth', 'VIX']].to_string())

print("\n" + "="*70)
