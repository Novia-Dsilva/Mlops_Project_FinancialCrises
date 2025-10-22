import pandas as pd

df = pd.read_csv('data/processed/merged_dataset.csv')
df['Date'] = pd.to_datetime(df['Date'])

print("="*60)
print("QUICK DATA CHECK")
print("="*60)

# Check macro
macro = ['GDP_Growth', 'VIX', 'CPI_Inflation']
print(f"\nMacro Indicators (sample of 3):")
for m in macro:
    pct = df[m].notna().sum() / len(df) * 100
    print(f"  {m}: {pct:.1f}% complete")

# Check companies
print(f"\nCompanies:")
print(f"  Total: {df['Company'].nunique()}")
print(f"  Stock data: {df['Stock_Price'].notna().sum() / len(df) * 100:.1f}% complete")
print(f"  Financial data: {df['Revenue'].notna().sum() / len(df) * 100:.1f}% complete")

# Check crises
print(f"\nCrisis Coverage:")
print(f"  2008: {len(df[df['Date'].dt.year.isin([2007,2008,2009])])} obs")
print(f"  2020: {len(df[df['Date'].dt.year == 2020])} obs")

print(f"\n✅ You have: {len(df):,} total observations")
print("="*60)
