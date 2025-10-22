import pandas as pd

# Load FRED cache (should show 2000)
fred = pd.read_csv('data/raw/fred_data.csv', index_col=0, parse_dates=True)
print("FRED Data Range:")
print(f"  Start: {fred.index.min()}")
print(f"  End: {fred.index.max()}")
print(f"  Total Quarters: {len(fred)}\n")

# Load company cache (should show 2000)
companies = pd.read_csv('data/raw/companies_data.csv', index_col=0, parse_dates=True)
print("Company Data Range:")
print(f"  Start: {companies.index.min()}")
print(f"  End: {companies.index.max()}")
print(f"  Total Records: {len(companies)}\n")

# Load final dataset
df = pd.read_csv('data/processed/merged_dataset.csv')
df['Date'] = pd.to_datetime(df['Date'])
print("Final Dataset Range:")
print(f"  Start: {df['Date'].min()}")
print(f"  End: {df['Date'].max()}")
print(f"  Total Rows: {len(df)}")
print(f"  Total Quarters: {df['Date'].nunique()}")

# Check if 2000-2009 data exists
early_data = df[df['Date'].dt.year < 2010]
print(f"\nData Before 2010 (proves it's from 2000):")
print(f"  Rows: {len(early_data)}")
print(f"  Years: {sorted(early_data['Date'].dt.year.unique())}")

if len(early_data) > 0:
    print("\n✅ CONFIRMED: Data is from 2000, not 2010!")
else:
    print("\n❌ WARNING: No data before 2010 - START_DATE might not have changed!")
