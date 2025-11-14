

"""
Financial Stress Test - Complete Data Loader
ENHANCED: Multi-method fallback + Smart caching + Robust error handling

Features:
- 3 fallback methods for stock data (Yahoo Chart API -> yfinance -> pandas_datareader)
- Smart caching for fundamentals (90-day refresh)
- Exponential backoff retry logic
- Comprehensive error handling
- Detailed logging

Author: MLOps Team
"""

import pandas as pd
import numpy as np
import requests
import time
import calendar
import json
from pandas_datareader import data as pdr
from datetime import datetime
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION
# =============================================================================

START_DATE = "2005-01-01"
END_DATE = datetime.now().strftime("%Y-%m-%d")

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

API_KEYS = ["XBAUMM6ATPHUYXTD"]
current_key_index = 0

def get_api_key():
    global current_key_index
    return API_KEYS[current_key_index % len(API_KEYS)]

def switch_api_key():
    global current_key_index
    current_key_index += 1
    print(f"   Switched to API key #{current_key_index + 1}")


# =============================================================================
# DATA SOURCES
# =============================================================================

FRED_SERIES = {
    "GDPC1": "GDP",
    "CPIAUCSL": "CPI",
    "UNRATE": "Unemployment_Rate",
    "FEDFUNDS": "Federal_Funds_Rate",
    "T10Y3M": "Yield_Curve_Spread",
    "UMCSENT": "Consumer_Confidence",
    "DCOILWTICO": "Oil_Price",
    "BOPGSTB": "Trade_Balance",
    "BAA10Y": "Corporate_Bond_Spread",
    "TEDRATE": "TED_Spread",
    "DGS10": "Treasury_10Y_Yield",
    "STLFSI4": "Financial_Stress_Index",
    "BAMLH0A0HYM2": "High_Yield_Spread",
}

MARKET_TICKERS = {
    "^VIX": "VIX",
    "^GSPC": "SP500"
}

COMPANIES = {
    "JPM": {"name": "JPMorgan Chase", "sector": "Financials"},
    "BAC": {"name": "Bank of America", "sector": "Financials"},
    "C": {"name": "Citigroup", "sector": "Financials"},
    "GS": {"name": "Goldman Sachs", "sector": "Financials"},
    "WFC": {"name": "Wells Fargo", "sector": "Financials"},
    "AAPL": {"name": "Apple", "sector": "Technology"},
    "MSFT": {"name": "Microsoft", "sector": "Technology"},
    "GOOGL": {"name": "Alphabet", "sector": "Technology"},
    "AMZN": {"name": "Amazon", "sector": "Technology"},
    "NVDA": {"name": "NVIDIA", "sector": "Technology"},
    "DIS": {"name": "Disney", "sector": "Communication Services"},
    "NFLX": {"name": "Netflix", "sector": "Communication Services"},
    "TSLA": {"name": "Tesla", "sector": "Consumer Discretionary"},
    "HD": {"name": "Home Depot", "sector": "Consumer Discretionary"},
    "MCD": {"name": "McDonalds", "sector": "Consumer Discretionary"},
    "WMT": {"name": "Walmart", "sector": "Consumer Staples"},
    "PG": {"name": "Procter & Gamble", "sector": "Consumer Staples"},
    "COST": {"name": "Costco", "sector": "Consumer Staples"},
    "XOM": {"name": "ExxonMobil", "sector": "Energy"},
    "CVX": {"name": "Chevron", "sector": "Energy"},
    "UNH": {"name": "UnitedHealth", "sector": "Healthcare"},
    "JNJ": {"name": "Johnson & Johnson", "sector": "Healthcare"},
    "BA": {"name": "Boeing", "sector": "Industrials"},
    "CAT": {"name": "Caterpillar", "sector": "Industrials"},
    "LIN": {"name": "Linde", "sector": "Materials"},
}


# =============================================================================
# ENHANCED YAHOO FINANCE CHART API (Solution 2)
# =============================================================================

def yahoo_chart_api(ticker, start="2005-01-01", end=None, interval="1d", timeout=20):
    """
    Enhanced Yahoo Finance Chart API with:
    - Configurable timeout (default 20s)
    - Retry logic with exponential backoff
    - Better error messages
    - Connection error handling
    """
    if end is None:
        end = datetime.now().strftime("%Y-%m-%d")

    def to_unix(date_str):
        d = datetime.strptime(date_str, "%Y-%m-%d")
        return calendar.timegm(d.timetuple())

    start_ts = to_unix(start)
    end_ts = to_unix(end)
    
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
        f"?period1={start_ts}&period2={end_ts}&interval={interval}&events=history"
    )
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Accept": "application/json",
        "Accept-Language": "en-US,en;q=0.9",
    }
    
    # Retry logic with exponential backoff
    for attempt in range(3):
        try:
            # Create session with timeout
            session = requests.Session()
            session.headers.update(headers)
            
            response = session.get(url, timeout=timeout)
            
            if response.status_code != 200:
                if attempt < 2:
                    time.sleep(2 ** attempt)  # 1s, 2s
                    continue
                raise ValueError(f"HTTP {response.status_code}")
            
            data = response.json()
            
            # Validate response structure
            if "chart" not in data:
                raise ValueError("No chart data in response")
            
            if data["chart"].get("error"):
                error = data["chart"]["error"]
                raise ValueError(f"Yahoo error: {error.get('description', 'Unknown')}")
            
            if not data["chart"]["result"]:
                raise ValueError("Empty result")
            
            chart = data["chart"]["result"][0]
            ts = chart.get("timestamp", [])
            
            if not ts:
                raise ValueError("No timestamps")
            
            quote = chart["indicators"]["quote"][0]
            
            # Build dataframe
            df = pd.DataFrame({
                "Date": pd.to_datetime(ts, unit="s"),
                "Open": quote.get("open"),
                "High": quote.get("high"),
                "Low": quote.get("low"),
                "Close": quote.get("close"),
                "Volume": quote.get("volume"),
            })
            
            # Add adjusted close
            if "adjclose" in chart["indicators"]:
                df["Adj_Close"] = chart["indicators"]["adjclose"][0]["adjclose"]
            else:
                df["Adj_Close"] = df["Close"]
            
            df = df.set_index("Date").dropna(how="any")
            
            if df.empty:
                raise ValueError("All data was NaN")
            
            return df
            
        except requests.exceptions.Timeout:
            if attempt < 2:
                time.sleep(2 ** attempt)
            else:
                raise ValueError(f"Timeout after {timeout}s")
        
        except requests.exceptions.ConnectionError:
            if attempt < 2:
                time.sleep(2 ** attempt)
            else:
                raise ValueError("Connection error")
        
        except Exception as e:
            if attempt < 2:
                time.sleep(2 ** attempt)
            else:
                raise

    raise ValueError("Failed after 3 attempts")


# =============================================================================
# FALLBACK METHOD 2: YFINANCE LIBRARY
# =============================================================================

def fetch_with_yfinance(ticker, start, end, interval="1wk"):
    """
    Fallback method using yfinance library
    More reliable than Chart API in some cases
    """
    try:
        import yfinance as yf
        
        ticker_obj = yf.Ticker(ticker)
        df = ticker_obj.history(start=start, end=end, interval=interval)
        
        if df.empty:
            raise ValueError("Empty dataframe")
        
        # Standardize column names
        df = df.reset_index()
        if 'Date' not in df.columns:
            df = df.rename(columns={'index': 'Date'})
        df = df.set_index('Date')
        
        # Ensure we have required columns
        required = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Missing column: {col}")
        
        # Add Adj_Close if not present
        if 'Adj Close' in df.columns:
            df['Adj_Close'] = df['Adj Close']
        elif 'Adj_Close' not in df.columns:
            df['Adj_Close'] = df['Close']
        
        return df[['Open', 'High', 'Low', 'Close', 'Volume', 'Adj_Close']]
        
    except ImportError:
        raise ValueError("yfinance library not installed")
    except Exception as e:
        raise ValueError(f"yfinance failed: {str(e)}")


# =============================================================================
# FALLBACK METHOD 3: PANDAS_DATAREADER
# =============================================================================

def fetch_with_datareader(ticker, start, end):
    """
    Fallback method using pandas_datareader
    Most compatible but slower
    """
    try:
        df = pdr.get_data_yahoo(ticker, start=start, end=end)
        
        if df.empty:
            raise ValueError("Empty dataframe")
        
        # Ensure we have Adj_Close
        if 'Adj Close' in df.columns:
            df['Adj_Close'] = df['Adj Close']
        elif 'Adj_Close' not in df.columns:
            df['Adj_Close'] = df['Close']
        
        return df[['Open', 'High', 'Low', 'Close', 'Volume', 'Adj_Close']]
        
    except Exception as e:
        raise ValueError(f"datareader failed: {str(e)}")


# =============================================================================
# STEP 1: FRED DATA
# =============================================================================

def fetch_fred_raw():
    """Fetch FRED macroeconomic data"""
    print("\n" + "=" * 70)
    print("STEP 1/4: FETCHING FRED MACROECONOMIC DATA")
    print("=" * 70)
    
    fred_data = {}
    successful = 0
    failed = []
    
    for series_id, col_name in FRED_SERIES.items():
        try:
            print(f"  {col_name:30} ({series_id})...", end=" ", flush=True)
            df = pdr.DataReader(series_id, "fred", START_DATE, END_DATE)
            fred_data[col_name] = df.iloc[:, 0]
            print(f"OK {len(df):,} records")
            successful += 1
            time.sleep(0.5)
        except Exception as e:
            print(f"FAILED ({str(e)[:40]})")
            failed.append(series_id)
    
    if not fred_data:
        raise ValueError("ERROR: No FRED data collected")
    
    df_fred = pd.DataFrame(fred_data)
    df_fred.index.name = 'Date'
    out = RAW_DIR / "fred_raw.csv"
    df_fred.to_csv(out)
    
    print(f"\nSaved: {out}")
    print(f"Success: {successful}/{len(FRED_SERIES)}")
    if failed:
        print(f"Failed: {', '.join(failed)}")
    
    return df_fred


# =============================================================================
# STEP 2: MARKET DATA (VIX, S&P 500) - FIXED FOR TIMESTAMP ALIGNMENT
# =============================================================================

def fetch_market_raw():
    """Fetch market data with multi-method fallback and timestamp normalization"""
    print("\n" + "=" * 70)
    print("STEP 2/4: FETCHING MARKET DATA")
    print("=" * 70)
    
    market_data = {}
    successful = 0
    
    for ticker, name in MARKET_TICKERS.items():
        print(f"  {name:25} ({ticker})...", end=" ", flush=True)
        
        df = None
        method_used = None
        
        # Try Method 1: Yahoo Chart API
        try:
            df = yahoo_chart_api(ticker, START_DATE, END_DATE, "1d")
            method_used = "ChartAPI"
        except Exception as e:
            print(f"ChartAPI failed ({str(e)[:30]})...", end=" ", flush=True)
        
        # Try Method 2: yfinance
        if df is None or df.empty:
            try:
                df = fetch_with_yfinance(ticker, START_DATE, END_DATE, "1d")
                method_used = "yfinance"
            except Exception as e:
                print(f"yfinance failed ({str(e)[:30]})...", end=" ", flush=True)
        
        # Try Method 3: pandas_datareader
        if df is None or df.empty:
            try:
                df = fetch_with_datareader(ticker, START_DATE, END_DATE)
                method_used = "datareader"
            except Exception as e:
                print(f"datareader failed ({str(e)[:30]})...", end=" ", flush=True)
        
        # Process successful fetch
        if df is not None and not df.empty:
            # CRITICAL FIX: Convert to date-only (removes time component completely)
            df.index = pd.to_datetime(df.index.date)
            df.index.name = 'Date'
            
            market_data[name] = df["Close"]
            print(f"OK {len(df):,} records ({method_used})")
            successful += 1
        else:
            print("FAILED all methods")
        
        time.sleep(1)
    
    if not market_data:
        raise ValueError("ERROR: No market data collected")
    
    # Combine market data
    df_market = pd.DataFrame(market_data)
    
    # Check for and remove duplicate dates
    if df_market.index.duplicated().any():
        duplicates = df_market.index.duplicated().sum()
        print(f"  WARNING: {duplicates} duplicate dates found, removing duplicates")
        df_market = df_market[~df_market.index.duplicated(keep='first')]
    
    # Verify both columns have data
    vix_nulls = df_market['VIX'].isna().sum()
    sp500_nulls = df_market['SP500'].isna().sum()
    total_rows = len(df_market)
    
    print(f"\n  Data quality check:")
    print(f"    VIX nulls: {vix_nulls}/{total_rows} ({vix_nulls/total_rows*100:.1f}%)")
    print(f"    SP500 nulls: {sp500_nulls}/{total_rows} ({sp500_nulls/total_rows*100:.1f}%)")
    
    if vix_nulls > total_rows * 0.5 or sp500_nulls > total_rows * 0.5:
        raise ValueError("ERROR: More than 50% null values in market data")
    
    # Ensure index is named
    df_market.index.name = 'Date'
    
    out = RAW_DIR / "market_raw.csv"
    df_market.to_csv(out)
    
    print(f"\nSaved: {out}")
    print(f"Success: {successful}/{len(MARKET_TICKERS)}")
    print(f"Rows: {len(df_market):,}")
    
    return df_market


# =============================================================================
# STEP 3: COMPANY PRICES (MULTI-METHOD FALLBACK - Solution 1)
# =============================================================================

def fetch_company_prices_raw():
    """
    Fetch company stock prices with multi-method fallback
    
    For each company, tries 3 methods:
    1. Yahoo Chart API (fastest, 20s timeout)
    2. yfinance library (more reliable)
    3. pandas_datareader (most compatible)
    """
    print("\n" + "=" * 70)
    print("STEP 3/4: FETCHING COMPANY PRICE DATA")
    print("=" * 70)
    print("Using multi-method fallback strategy for maximum reliability\n")
    
    all_data = []
    failed_companies = []
    method_stats = {"ChartAPI": 0, "yfinance": 0, "datareader": 0}
    
    for i, (ticker, info) in enumerate(COMPANIES.items(), 1):
        print(f"  [{i:2d}/25] {ticker:6} {info['name']:25}...", end=" ", flush=True)
        
        df = None
        method_used = None
        
        # METHOD 1: Yahoo Chart API (fastest)
        try:
            df = yahoo_chart_api(ticker, START_DATE, END_DATE, "1wk")
            method_used = "ChartAPI"
        except Exception as e:
            print(f"Chart({str(e)[:20]})...", end=" ", flush=True)
        
        # METHOD 2: yfinance library (fallback)
        if df is None or df.empty:
            try:
                df = fetch_with_yfinance(ticker, START_DATE, END_DATE, "1wk")
                method_used = "yfinance"
            except Exception as e:
                print(f"yf({str(e)[:20]})...", end=" ", flush=True)
        
        # METHOD 3: pandas_datareader (last resort)
        if df is None or df.empty:
            try:
                df = fetch_with_datareader(ticker, START_DATE, END_DATE)
                # Resample to weekly if needed
                if len(df) > 1200:  # If daily data
                    df = df.resample('W').agg({
                        'Open': 'first',
                        'High': 'max',
                        'Low': 'min',
                        'Close': 'last',
                        'Volume': 'sum',
                        'Adj_Close': 'last'
                    }).dropna()
                method_used = "datareader"
            except Exception as e:
                print(f"dr({str(e)[:20]})...", end=" ", flush=True)
        
        # Process successful fetch
        if df is not None and not df.empty:
            df = df.copy()
            df["Company"] = ticker
            df["Company_Name"] = info["name"]
            df["Sector"] = info["sector"]
            all_data.append(df)
            method_stats[method_used] += 1
            print(f"OK {len(df):,} records ({method_used})")
        else:
            print("FAILED all methods - SKIP")
            failed_companies.append(ticker)
        
        time.sleep(1)  # Rate limiting
    
    # Check if we got enough data
    if not all_data:
        raise ValueError("ERROR: No company price data collected")
    
    if len(failed_companies) > 10:
        raise ValueError(f"ERROR: Too many failures ({len(failed_companies)}/25)")
    
    # Combine all data - preserves Date index
    df_all = pd.concat(all_data, axis=0)
    df_all.index.name = 'Date'
    
    out = RAW_DIR / "company_prices_raw.csv"
    df_all.to_csv(out, index=True)
    
    print(f"\nSaved: {out}")
    print(f"Success: {len(all_data)}/{len(COMPANIES)} companies")
    
    if failed_companies:
        print(f"WARNING: Failed: {', '.join(failed_companies)}")
    else:
        print("SUCCESS: All 25 companies captured!")
    
    print(f"\nMethod Statistics:")
    for method, count in method_stats.items():
        if count > 0:
            pct = (count / len(all_data)) * 100
            print(f"  {method:12}: {count:2d}/25 ({pct:5.1f}%)")
    
    return df_all


# =============================================================================
# STEP 4: COMPANY FUNDAMENTALS (WITH SMART CACHING)
# =============================================================================

def should_fetch_fundamentals():
    """
    Check if fundamentals need updating
    Returns True if: no cache OR cache older than 90 days
    """
    cache_file = Path('data/fundamentals_cache_state.json')
    
    if not cache_file.exists():
        print("  No cache found - will fetch fundamentals")
        return True
    
    try:
        with open(cache_file) as f:
            state = json.load(f)
        
        last_fetch = datetime.fromisoformat(state['last_fetch_date'])
        days_old = (datetime.now() - last_fetch).days

        if days_old >= 90:
            print(f"  Fundamentals cache: {days_old} days old (STALE - fetching fresh)")
            return True
        else:
            print(f"  Fundamentals cache: {days_old} days old (FRESH - using cached files)")
            return False
    except Exception as e:
        print(f"  Cache read error - will fetch: {e}")
        return True


def mark_fundamentals_fetched():
    """Save cache state"""
    state = {
        'last_fetch_date': datetime.now().isoformat(),
        'companies_count': len(COMPANIES),
        'fetch_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    with open('data/fundamentals_cache_state.json', 'w') as f:
        json.dump(state, f, indent=2)
    print(f"  Cache updated: {state['fetch_timestamp']}")


def fetch_alpha_vantage(ticker, function, retry_count=0):
    """Fetch from Alpha Vantage with retry logic"""
    url = "https://www.alphavantage.co/query"
    params = {
        "function": function,
        "symbol": ticker,
        "apikey": get_api_key(),
        "datatype": "json"
    }
    
    try:
        r = requests.get(url, params=params, timeout=30)
        data = r.json()
        
        # Check for rate limiting
        if "Note" in data or "Information" in data:
            if retry_count < 3:
                switch_api_key()
                time.sleep(10)
                return fetch_alpha_vantage(ticker, function, retry_count + 1)
            return None
        
        if "quarterlyReports" not in data:
            return None
        
        return data["quarterlyReports"]
        
    except Exception as e:
        if retry_count < 2:
            time.sleep(5)
            return fetch_alpha_vantage(ticker, function, retry_count + 1)
        return None


def parse_financials(data, mapping):
    """Parse financial data"""
    if not data:
        return pd.DataFrame()
    
    df = pd.DataFrame([{k: r.get(v) for k, v in mapping.items()} for r in data])
    
    for col in df.columns:
        if col != "Date":
            df[col] = pd.to_numeric(df[col], errors="coerce")
    
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    
    return df


def fetch_company_fundamentals_raw():
    """Fetch fundamentals with smart caching"""
    print("\n" + "=" * 70)
    print("STEP 4/4: COMPANY FUNDAMENTALS")
    print("=" * 70)
    
    # Check cache
    if not should_fetch_fundamentals():
        print("\n  SKIPPING fundamentals fetch - using cached files")
        print("  (Fundamentals are quarterly data - daily fetch not needed)")
        
        income_file = RAW_DIR / 'company_income_raw.csv'
        balance_file = RAW_DIR / 'company_balance_raw.csv'
        
        if income_file.exists() and balance_file.exists():
            print(f"  Cached income: {income_file}")
            print(f"  Cached balance: {balance_file}")
            return None, None
        else:
            print("  WARNING: Cached files missing - will fetch anyway")
    
    # Fetch fresh data
    print("\n  Fetching fresh fundamentals...")
    print(f"  Estimated time: ~{len(COMPANIES) * 40 / 60:.0f} minutes")
    print()
    
    all_income = []
    all_balance = []
    failed = []
    
    for i, (ticker, info) in enumerate(COMPANIES.items(), 1):
        print(f"  [{i:2d}/25] {ticker:6} {info['name']:25}", end=" ", flush=True)
        
        # Fetch Income Statement
        income_data = fetch_alpha_vantage(ticker, "INCOME_STATEMENT")
        if income_data:
            df_inc = parse_financials(
                income_data,
                {
                    "Date": "fiscalDateEnding",
                    "Revenue": "totalRevenue",
                    "Net_Income": "netIncome",
                    "Gross_Profit": "grossProfit",
                    "Operating_Income": "operatingIncome",
                    "EBITDA": "ebitda",
                    "EPS": "reportedEPS",
                },
            )
            if not df_inc.empty:
                df_inc["Company"] = ticker
                df_inc["Company_Name"] = info["name"]
                df_inc["Sector"] = info["sector"]
                all_income.append(df_inc)
                print("Inc:OK", end=" ", flush=True)
            else:
                print("Inc:Empty", end=" ", flush=True)
        else:
            print("Inc:FAIL", end=" ", flush=True)
            failed.append(ticker)
        
        time.sleep(20)
        
        # Fetch Balance Sheet
        balance_data = fetch_alpha_vantage(ticker, "BALANCE_SHEET")
        if balance_data:
            df_bal = parse_financials(
                balance_data,
                {
                    "Date": "fiscalDateEnding",
                    "Total_Assets": "totalAssets",
                    "Total_Liabilities": "totalLiabilities",
                    "Total_Equity": "totalShareholderEquity",
                    "Current_Assets": "totalCurrentAssets",
                    "Current_Liabilities": "totalCurrentLiabilities",
                    "Long_Term_Debt": "longTermDebt",
                    "Short_Term_Debt": "shortTermDebt",
                    "Cash": "cashAndCashEquivalentsAtCarryingValue",
                },
            )
            if not df_bal.empty:
                df_bal["Company"] = ticker
                df_bal["Company_Name"] = info["name"]
                df_bal["Sector"] = info["sector"]
                
                # Calculate ratios
                df_bal["Debt_to_Equity"] = (
                    df_bal["Total_Liabilities"] / df_bal["Total_Equity"].replace(0, 1)
                )
                df_bal["Current_Ratio"] = (
                    df_bal["Current_Assets"] / df_bal["Current_Liabilities"].replace(0, 1)
                )
                
                all_balance.append(df_bal)
                print("Bal:OK")
            else:
                print("Bal:Empty")
        else:
            print("Bal:FAIL")
        
        time.sleep(20)
    
    # Save results
    print()
    
    if all_income:
        df_inc = pd.concat(all_income, ignore_index=True)
        df_inc.to_csv(RAW_DIR / "company_income_raw.csv", index=False)
        print(f"  Income saved: {len(all_income)} companies, {len(df_inc)} quarters")
    else:
        print("  WARNING: No income data collected")
    
    if all_balance:
        df_bal = pd.concat(all_balance, ignore_index=True)
        df_bal.to_csv(RAW_DIR / "company_balance_raw.csv", index=False)
        print(f"  Balance saved: {len(all_balance)} companies, {len(df_bal)} quarters")
    else:
        print("  WARNING: No balance data collected")
    
    if failed:
        print(f"  WARNING: Failed companies: {', '.join(failed)}")
    
    # Mark as fetched
    mark_fundamentals_fetched()
    
    return (df_inc if all_income else None), (df_bal if all_balance else None)


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main data collection pipeline"""
    print("\n" + "=" * 70)
    print("FINANCIAL DATA LOADER - ENHANCED")
    print("=" * 70)
    print(f"Period: {START_DATE} to {END_DATE}")
    print("Features: Multi-method fallback + Smart caching")
    print("=" * 70)
    
    overall_start = time.time()
    
    try:
        # Step 1: FRED (always fetch - fast)
        df_fred = fetch_fred_raw()
        
        # Step 2: Market (always fetch - fast)
        df_market = fetch_market_raw()
        
        # Step 3: Company Prices (always fetch - moderate)
        df_prices = fetch_company_prices_raw()
        
        # Step 4: Fundamentals (conditional - slow)
        df_income, df_balance = fetch_company_fundamentals_raw()
        
        elapsed = (time.time() - overall_start) / 60
        
        # Summary
        print("\n" + "=" * 70)
        print("DATA COLLECTION COMPLETE")
        print("=" * 70)
        print(f"Total time: {elapsed:.1f} minutes")
        print(f"\nFiles created in {RAW_DIR}/:")
        
        for f in sorted(RAW_DIR.glob("*.csv")):
            size = f.stat().st_size / (1024 * 1024)
            rows = sum(1 for _ in open(f)) - 1  # Count rows (minus header)
            print(f"  {f.name:30} {size:>6.2f} MB  ({rows:>6,} rows)")
        
        print("=" * 70)
        print("Ready for validation and processing!")
        print("=" * 70)
        
    except KeyboardInterrupt:
        print("\n\nWARNING: Process interrupted by user")
        print("Progress has been saved where possible")
        raise
        
    except Exception as e:
        print(f"\n\nERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()