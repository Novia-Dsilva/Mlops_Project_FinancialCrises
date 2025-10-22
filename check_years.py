import pandas as pd
import yfinance as yf

# Test a few companies to see historical data availability
test_companies = {
    'JPM': 'JPMorgan (old company)',
    'AAPL': 'Apple (2000s IPO)',
    'GOOGL': 'Google (2004 IPO)',
    'TSLA': 'Tesla (2010 IPO)'
}

print("="*70)
print("📊 HISTORICAL DATA AVAILABILITY TEST")
print("="*70)

for ticker, name in test_companies.items():
    print(f"\n{name} ({ticker}):")
    
    # Test stock prices (goes back far)
    prices = yf.download(ticker, start='2000-01-01', end='2025-10-22', progress=False)
    if not prices.empty:
        print(f"   Stock Prices: {prices.index.min()} to {prices.index.max()}")
        print(f"   Years: {(prices.index.max() - prices.index.min()).days / 365:.1f}")
    
    # Test quarterly financials (limited)
    stock = yf.Ticker(ticker)
    income = stock.quarterly_income_stmt
    if income is not None and not income.empty:
        dates = pd.to_datetime(income.columns)
        print(f"   Financials: {dates.min()} to {dates.max()}")
        print(f"   Quarters: {len(dates)}")
    else:
        print(f"   Financials: None available")

print("\n" + "="*70)
