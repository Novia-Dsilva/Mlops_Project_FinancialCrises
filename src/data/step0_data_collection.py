"""
Financial Stress Test Generator - Complete Data Loader
FETCHES: FRED Macro + Market + Company Prices + Company Fundamentals
SAVES TO: data/raw/ (RAW data, no processing)
SOURCES: FRED API, Yahoo Finance, Alpha Vantage
DATE RANGE: 2005-01-01 to present
"""

import pandas as pd
import numpy as np
import yfinance as yf
from pandas_datareader import data as pdr
import requests
import time
from pathlib import Path
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# Configuration
START_DATE = '2005-01-01'
END_DATE = datetime.now().strftime('%Y-%m-%d')

RAW_DIR = Path('data/raw')
RAW_DIR.mkdir(parents=True, exist_ok=True)

# Alpha Vantage API Keys
API_KEYS = [
    'XBAUMM6ATPHUYXTD'
]
current_key_index = 0

def get_api_key():
    global current_key_index
    return API_KEYS[current_key_index % len(API_KEYS)]

def switch_api_key():
    global current_key_index
    current_key_index += 1
    print(f"   Switched to API key #{current_key_index + 1}")

DELAY_BETWEEN_CALLS = 20
MAX_RETRIES = 3

# Data Sources
FRED_SERIES = {
    'GDPC1': 'GDP',
    'CPIAUCSL': 'CPI',
    'UNRATE': 'Unemployment_Rate',
    'FEDFUNDS': 'Federal_Funds_Rate',
    'T10Y3M': 'Yield_Curve_Spread',
    'UMCSENT': 'Consumer_Confidence',
    'DCOILWTICO': 'Oil_Price',
    'BOPGSTB': 'Trade_Balance',
    'BAA10Y': 'Corporate_Bond_Spread',
    'TEDRATE': 'TED_Spread',
    'DGS10': 'Treasury_10Y_Yield',
    'STLFSI4': 'Financial_Stress_Index',
    'BAMLH0A0HYM2': 'High_Yield_Spread'
}

MARKET_TICKERS = {
    '^VIX': 'VIX',
    '^GSPC': 'SP500'
}

COMPANIES = {
    'JPM': {'name': 'JPMorgan Chase', 'sector': 'Financials'},
    'BAC': {'name': 'Bank of America', 'sector': 'Financials'},
    'C': {'name': 'Citigroup', 'sector': 'Financials'},
    'GS': {'name': 'Goldman Sachs', 'sector': 'Financials'},
    'WFC': {'name': 'Wells Fargo', 'sector': 'Financials'},
    'AAPL': {'name': 'Apple', 'sector': 'Technology'},
    'MSFT': {'name': 'Microsoft', 'sector': 'Technology'},
    'GOOGL': {'name': 'Alphabet', 'sector': 'Technology'},
    'AMZN': {'name': 'Amazon', 'sector': 'Technology'},
    'NVDA': {'name': 'NVIDIA', 'sector': 'Technology'},
    'DIS': {'name': 'Disney', 'sector': 'Communication Services'},
    'NFLX': {'name': 'Netflix', 'sector': 'Communication Services'},
    'TSLA': {'name': 'Tesla', 'sector': 'Consumer Discretionary'},
    'HD': {'name': 'Home Depot', 'sector': 'Consumer Discretionary'},
    'MCD': {'name': 'McDonalds', 'sector': 'Consumer Discretionary'},
    'WMT': {'name': 'Walmart', 'sector': 'Consumer Staples'},
    'PG': {'name': 'Procter & Gamble', 'sector': 'Consumer Staples'},
    'COST': {'name': 'Costco', 'sector': 'Consumer Staples'},
    'XOM': {'name': 'ExxonMobil', 'sector': 'Energy'},
    'CVX': {'name': 'Chevron', 'sector': 'Energy'},
    'UNH': {'name': 'UnitedHealth', 'sector': 'Healthcare'},
    'JNJ': {'name': 'Johnson & Johnson', 'sector': 'Healthcare'},
    'BA': {'name': 'Boeing', 'sector': 'Industrials'},
    'CAT': {'name': 'Caterpillar', 'sector': 'Industrials'},
    'LIN': {'name': 'Linde', 'sector': 'Materials'}
}

# STEP 1: FETCH FRED MACRO DATA
def fetch_fred_raw():
    """Fetch FRED macroeconomic data - save RAW (no processing)"""

    print("\n" + "="*70)
    print("STEP 1/4: FETCHING FRED MACROECONOMIC DATA")
    print("="*70)
    print(f"Period: {START_DATE} to {END_DATE}")
    print(f"Indicators: {len(FRED_SERIES)}")
    print()

    fred_data = {}
    successful = 0
    failed = []

    for series_id, col_name in FRED_SERIES.items():
        try:
            print(f"  {col_name:30} ({series_id})...", end=" ", flush=True)
            df = pdr.DataReader(series_id, 'fred', START_DATE, END_DATE)
            fred_data[col_name] = df.iloc[:, 0]
            print(f"OK {len(df):,} records")
            successful += 1
            time.sleep(0.5)
        except Exception as e:
            print(f"FAILED {str(e)[:40]}")
            failed.append(series_id)

    if not fred_data:
        raise ValueError("ERROR: No FRED data collected")

    df_fred = pd.DataFrame(fred_data)

    print(f"\nFRED Data Summary:")
    print(f"  Shape: {df_fred.shape[0]:,} rows x {df_fred.shape[1]} columns")
    print(f"  Success: {successful}/{len(FRED_SERIES)}")
    if failed:
        print(f"  Failed: {', '.join(failed)}")
    print(f"  Date range: {df_fred.index.min()} to {df_fred.index.max()}")
    print(f"  Missing values: {df_fred.isna().sum().sum():,}")

    output_path = RAW_DIR / 'fred_raw.csv'
    df_fred.to_csv(output_path)
    print(f"\nSaved: {output_path}")
    print(f"Size: {output_path.stat().st_size / (1024*1024):.2f} MB")

    return df_fred

# STEP 2: FETCH MARKET DATA
def fetch_market_raw():
    """Fetch market data (VIX, S&P 500) - save RAW (no processing)"""

    print("\n" + "="*70)
    print("STEP 2/4: FETCHING MARKET DATA")
    print("="*70)
    print(f"Period: {START_DATE} to {END_DATE}")
    print(f"Indicators: VIX, S&P 500")
    print()

    market_data = {}
    successful = 0
    failed = []

    for ticker, name in MARKET_TICKERS.items():
        try:
            print(f"  {name:30} ({ticker})...", end=" ", flush=True)
            data = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False)

            if not data.empty and 'Close' in data.columns:
                close_data = data['Close']
                if isinstance(close_data, pd.DataFrame):
                    close_data = close_data.iloc[:, 0]

                market_data[name] = close_data
                print(f"OK {len(data):,} records")
                successful += 1
            else:
                print(f"FAILED: No data")
                failed.append(ticker)

            time.sleep(1)
        except Exception as e:
            print(f"FAILED: {str(e)[:40]}")
            failed.append(ticker)

    if not market_data:
        raise ValueError("ERROR: No market data collected")

    df_market = pd.DataFrame(market_data)

    print(f"\nMarket Data Summary:")
    print(f"  Shape: {df_market.shape[0]:,} rows x {df_market.shape[1]} columns")
    print(f"  Success: {successful}/{len(MARKET_TICKERS)}")
    if failed:
        print(f"  Failed: {', '.join(failed)}")
    print(f"  Date range: {df_market.index.min()} to {df_market.index.max()}")
    print(f"  Missing values: {df_market.isna().sum().sum():,}")

    output_path = RAW_DIR / 'market_raw.csv'
    df_market.to_csv(output_path)
    print(f"\nSaved: {output_path}")
    print(f"Size: {output_path.stat().st_size / (1024*1024):.2f} MB")

    return df_market

# STEP 3: FETCH COMPANY PRICES
def fetch_company_prices_raw():
    """Fetch company stock prices - save RAW OHLCV data"""

    print("\n" + "="*70)
    print("STEP 3/4: FETCHING COMPANY PRICE DATA")
    print("="*70)
    print(f"Period: {START_DATE} to {END_DATE}")
    print(f"Companies: {len(COMPANIES)}")
    print()

    all_data = []
    successful = 0
    failed = []

    for i, (ticker, info) in enumerate(COMPANIES.items(), 1):
        try:
            print(f"  [{i:2d}/25] {ticker:6} {info['name']:25}...", end=" ", flush=True)

            prices = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False)

            if prices.empty:
                print(f"FAILED: No data")
                failed.append(ticker)
                continue

            if isinstance(prices.columns, pd.MultiIndex):
                prices.columns = prices.columns.get_level_values(0)

            df = pd.DataFrame(index=prices.index)
            df['Open'] = prices['Open']
            df['High'] = prices['High']
            df['Low'] = prices['Low']
            df['Close'] = prices['Close']
            df['Volume'] = prices['Volume']
            df['Adj_Close'] = prices.get('Adj Close', prices['Close'])
            df['Company'] = ticker
            df['Company_Name'] = info['name']
            df['Sector'] = info['sector']

            all_data.append(df)
            print(f"OK {len(df):,} days")
            successful += 1

            time.sleep(0.5)

        except Exception as e:
            print(f"FAILED: {str(e)[:30]}")
            failed.append(ticker)

    if not all_data:
        raise ValueError("ERROR: No company price data collected")

    df_all = pd.concat(all_data, axis=0)

    print(f"\nCompany Prices Summary:")
    print(f"  Total records: {len(df_all):,}")
    print(f"  Companies: {successful}/{len(COMPANIES)}")
    if failed:
        print(f"  Failed: {', '.join(failed)}")
    print(f"  Date range: {df_all.index.min()} to {df_all.index.max()}")
    print(f"  Columns: {list(df_all.columns)}")

    output_path = RAW_DIR / 'company_prices_raw.csv'
    df_all.to_csv(output_path)
    print(f"\nSaved: {output_path}")
    print(f"Size: {output_path.stat().st_size / (1024*1024):.2f} MB")

    return df_all

# STEP 4: ALPHA VANTAGE FUNDAMENTALS
def fetch_alpha_vantage(ticker, function, retry_count=0):
    """Fetch data from Alpha Vantage with retry logic"""
    url = "https://www.alphavantage.co/query"
    params = {
        'function': function,
        'symbol': ticker,
        'apikey': get_api_key(),
        'datatype': 'json',
        'type': 'quarterly'
    }

    try:
        response = requests.get(url, params=params, timeout=30)
        data = response.json()

        if not data:
            print(f"   WARNING: Empty response", end=" ")
            return None

        if 'Note' in data:
            print(f"   WARNING: Rate limit, rotating...", end=" ")
            switch_api_key()
            time.sleep(5)
            return fetch_alpha_vantage(ticker, function, retry_count)

        if 'Error Message' in data or 'Information' in data:
            msg = data.get('Error Message') or data.get('Information', '')[:50]
            print(f"   WARNING: {msg}", end=" ")
            return None

        if 'quarterlyReports' not in data:
            print(f"   WARNING: No quarterlyReports", end=" ")
            return None

        return data['quarterlyReports']

    except requests.exceptions.Timeout:
        if retry_count < MAX_RETRIES:
            print(f"   Timeout, retry {retry_count+1}...", end=" ")
            time.sleep(30)
            return fetch_alpha_vantage(ticker, function, retry_count + 1)
        print(f"   FAILED: Timeout", end=" ")
        return None
    except Exception as e:
        print(f"   FAILED: {str(e)[:20]}", end=" ")
        return None


def fetch_fmp(ticker, endpoint="income-statement", limit=5):
    """Fallback: fetch from Financial Modeling Prep"""
    api_key = "demo"
    url = f"https://financialmodelingprep.com/api/v3/{endpoint}/{ticker}?period=quarter&limit={limit}&apikey={api_key}"
    try:
        r = requests.get(url, timeout=30)
        data = r.json()
        if isinstance(data, list) and len(data) > 0:
            return data
        return None
    except:
        return None


def parse_income(data):
    """Parse income statement data"""
    recs = []
    for r in data:
        recs.append({
            'Date': r.get('fiscalDateEnding') or r.get('date'),
            'Revenue': r.get('totalRevenue') or r.get('revenue'),
            'Net_Income': r.get('netIncome'),
            'Gross_Profit': r.get('grossProfit'),
            'Operating_Income': r.get('operatingIncome'),
            'EBITDA': r.get('ebitda'),
            'EPS': r.get('reportedEPS') or r.get('eps')
        })
    df = pd.DataFrame(recs)
    for col in ['Revenue', 'Net_Income', 'Gross_Profit', 'Operating_Income', 'EBITDA', 'EPS']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    return df


def parse_balance(data):
    """Parse balance sheet data"""
    recs = []
    for r in data:
        recs.append({
            'Date': r.get('fiscalDateEnding') or r.get('date'),
            'Total_Assets': r.get('totalAssets'),
            'Total_Liabilities': r.get('totalLiabilities'),
            'Total_Equity': r.get('totalShareholderEquity') or r.get('totalEquity'),
            'Current_Assets': r.get('totalCurrentAssets') or r.get('currentAssets'),
            'Current_Liabilities': r.get('totalCurrentLiabilities') or r.get('currentLiabilities'),
            'Long_Term_Debt': r.get('longTermDebt'),
            'Short_Term_Debt': r.get('shortTermDebt'),
            'Cash': r.get('cashAndCashEquivalentsAtCarryingValue') or r.get('cashAndCashEquivalents')
        })
    df = pd.DataFrame(recs)
    for col in ['Total_Assets', 'Total_Liabilities', 'Total_Equity', 'Current_Assets',
                'Current_Liabilities', 'Long_Term_Debt', 'Short_Term_Debt', 'Cash']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['Debt_to_Equity'] = df['Total_Liabilities'] / df['Total_Equity'].replace(0, 1)
    df['Current_Ratio'] = df['Current_Assets'] / df['Current_Liabilities'].replace(0, 1)
    return df


def fetch_company_fundamentals_raw():
    """Fetch company fundamentals from Alpha Vantage (quarterly data)"""

    print("\n" + "="*70)
    print("STEP 4/4: FETCHING COMPANY FUNDAMENTALS (ALPHA VANTAGE)")
    print("="*70)
    print(f"Companies: {len(COMPANIES)}")
    print(f"API Keys: {len(API_KEYS)}")
    print(f"Delay: {DELAY_BETWEEN_CALLS}s between calls")
    print(f"Estimated time: ~{len(COMPANIES) * 2 * DELAY_BETWEEN_CALLS / 60:.0f} minutes")
    print()

    cache_file = RAW_DIR / 'financials_cache.txt'
    if cache_file.exists():
        cached = set(cache_file.read_text().strip().split(','))
        if cached and '' in cached:
            cached.remove('')
        if cached:
            print(f"Cache found: {len(cached)} companies already fetched")
            print(f"Cached: {', '.join(sorted(cached))}")

            user_input = input("\nClear cache and fetch fresh? (y/n): ")
            if user_input.lower() == 'y':
                cache_file.unlink()
                cached = set()
                print("Cache cleared!")
            else:
                print("Using cache")
            print()
    else:
        cached = set()

    all_income = []
    all_balance = []
    failed = []
    start_time = time.time()

    for i, (ticker, info) in enumerate(COMPANIES.items(), 1):
        if ticker in cached:
            print(f"[{i:2d}/25] {ticker:6} {info['name']:25} CACHED")
            continue

        print(f"[{i:2d}/25] {ticker:6} {info['name']:25}")

        # Income Statement
        print("   Income...", end=" ", flush=True)
        income_data = fetch_alpha_vantage(ticker, 'INCOME_STATEMENT')

        if not income_data:
            print("trying FMP...", end=" ", flush=True)
            income_data = fetch_fmp(ticker, 'income-statement')

        if not income_data:
            print("FAILED")
            failed.append(ticker)
            continue

        df_income = parse_income(income_data)
        df_income['Company'] = ticker
        df_income['Company_Name'] = info['name']
        df_income['Sector'] = info['sector']
        all_income.append(df_income)
        print(f"OK {len(df_income)}Q")

        time.sleep(DELAY_BETWEEN_CALLS)

        # Balance Sheet
        print("   Balance...", end=" ", flush=True)
        balance_data = fetch_alpha_vantage(ticker, 'BALANCE_SHEET')

        if not balance_data:
            print("trying FMP...", end=" ", flush=True)
            balance_data = fetch_fmp(ticker, 'balance-sheet-statement')

        if balance_data:
            df_balance = parse_balance(balance_data)
            df_balance['Company'] = ticker
            df_balance['Company_Name'] = info['name']
            df_balance['Sector'] = info['sector']
            all_balance.append(df_balance)
            print(f"OK {len(df_balance)}Q")
        else:
            print("SKIPPED")

        cached.add(ticker)
        cache_file.write_text(','.join(cached))

        time.sleep(DELAY_BETWEEN_CALLS)

    elapsed = (time.time() - start_time) / 60

    print(f"\nCompany Fundamentals Summary:")
    print(f"  Elapsed: {elapsed:.1f} minutes")
    print(f"  Success: {len(all_income)}/{len(COMPANIES)}")
    if failed:
        print(f"  Failed: {', '.join(failed)}")

    # Save income statements
    if all_income:
        df_inc = pd.concat(all_income, ignore_index=True)
        output_path = RAW_DIR / 'company_income_raw.csv'
        df_inc.to_csv(output_path, index=False)
        print(f"\nIncome Statements Saved: {output_path}")
        print(f"  Records: {len(df_inc):,} quarters")
        print(f"  Companies: {df_inc['Company'].nunique()}")
        print(f"  Size: {output_path.stat().st_size / (1024*1024):.2f} MB")

    # Save balance sheets
    if all_balance:
        df_bal = pd.concat(all_balance, ignore_index=True)
        output_path = RAW_DIR / 'company_balance_raw.csv'
        df_bal.to_csv(output_path, index=False)
        print(f"\nBalance Sheets Saved: {output_path}")
        print(f"  Records: {len(df_bal):,} quarters")
        print(f"  Companies: {df_bal['Company'].nunique()}")
        print(f"  Size: {output_path.stat().st_size / (1024*1024):.2f} MB")

    if len(cached) == 25:
        print(f"\nALL 25 COMPANIES COMPLETE!")
    else:
        remaining = 25 - len(cached)
        print(f"\nProgress: {len(cached)}/25 companies")
        print(f"  Remaining: {remaining} companies")
        print(f"  NOTE: Run script again to continue fetching")

    return df_inc if all_income else None, df_bal if all_balance else None

# MAIN PIPELINE
def main():
    """
    Complete data collection pipeline
    Saves all data to data/raw/ folder
    """

    print("\n" + "="*70)
    print("FINANCIAL STRESS TEST - COMPLETE DATA LOADER")
    print("="*70)
    print(f"Period: {START_DATE} to {END_DATE}")
    print(f"Output: {RAW_DIR}/")
    print(f"Alpha Vantage Keys: {len(API_KEYS)}")
    print("="*70)

    overall_start = time.time()

    try:
        # STEP 1: FRED Macro Data
        df_fred = fetch_fred_raw()

        # STEP 2: Market Data
        df_market = fetch_market_raw()

        # STEP 3: Company Prices
        df_prices = fetch_company_prices_raw()

        # STEP 4: Company Fundamentals
        df_income, df_balance = fetch_company_fundamentals_raw()

        # Final Summary
        elapsed = time.time() - overall_start

        print("\n" + "="*70)
        print("DATA COLLECTION COMPLETE")
        print("="*70)

        print(f"\nDATA COLLECTED:")
        print(f"  1. FRED Macro:          {df_fred.shape[0]:,} rows x {df_fred.shape[1]} cols")
        print(f"  2. Market:              {df_market.shape[0]:,} rows x {df_market.shape[1]} cols")
        print(f"  3. Company Prices:      {df_prices.shape[0]:,} rows (25 companies)")
        if df_income is not None:
            print(f"  4. Income Statements:   {len(df_income):,} quarters ({df_income['Company'].nunique()} companies)")
        if df_balance is not None:
            print(f"  5. Balance Sheets:      {len(df_balance):,} quarters ({df_balance['Company'].nunique()} companies)")

        print(f"\nOUTPUT FILES (data/raw/):")
        print(f"  - fred_raw.csv")
        print(f"  - market_raw.csv")
        print(f"  - company_prices_raw.csv")
        if df_income is not None:
            print(f"  - company_income_raw.csv")
        if df_balance is not None:
            print(f"  - company_balance_raw.csv")

        print(f"\nTotal Time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
        print("="*70)

    except Exception as e:
        print(f"\nERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
    