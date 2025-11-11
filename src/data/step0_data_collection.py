

"""
<<<<<<< HEAD
Financial Stress Test Generator - Complete Data Loader (AIRFLOW-READY)
FETCHES: FRED Macro + Market + Company Prices + Company Fundamentals
SAVES TO: data/raw/ (RAW data, no processing)
SOURCES: FRED API, Yahoo Finance, Alpha Vantage
DATE RANGE: 2005-01-01 to present

FIXES FOR AIRFLOW:
- Proper User-Agent headers to prevent Yahoo Finance blocking
- Retry logic with exponential backoff
- Fallback data sources (ETFs instead of indices)
- No interactive prompts
- Continues with partial data instead of crashing
=======
Financial Stress Test - Complete Data Loader
ENHANCED: Multi-method fallback + Smart caching + Robust error handling

Features:
- 3 fallback methods for stock data (Yahoo Chart API -> yfinance -> pandas_datareader)
- Smart caching for fundamentals (90-day refresh)
- Exponential backoff retry logic
- Comprehensive error handling
- Detailed logging

Author: MLOps Team
>>>>>>> origin/main
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
import os

warnings.filterwarnings("ignore")

<<<<<<< HEAD
# Configure yfinance with proper headers
session = requests.Session()
session.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
    'Connection': 'keep-alive',
})

# Configuration
START_DATE = '2005-01-01'
END_DATE = datetime.now().strftime('%Y-%m-%d')
=======
# =============================================================================
# CONFIGURATION
# =============================================================================
>>>>>>> origin/main

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

<<<<<<< HEAD

DELAY_BETWEEN_CALLS = 20
MAX_RETRIES = 3
=======
>>>>>>> origin/main

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

# UPDATED: Use ETFs as primary, indices as fallback
MARKET_TICKERS = {
<<<<<<< HEAD
    'SPY': 'SP500',   # ETF - More reliable than ^GSPC
    'VIXY': 'VIX',    # ETF - More reliable than ^VIX
=======
    "^VIX": "VIX",
    "^GSPC": "SP500"
>>>>>>> origin/main
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

<<<<<<< HEAD
# STEP 1: FETCH FRED MACRO DATA (UNCHANGED)


=======

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

>>>>>>> origin/main
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
            print(f"✓ OK {len(df):,} records")
            successful += 1
            time.sleep(0.5)
        except Exception as e:
<<<<<<< HEAD
            print(f"✗ FAILED {str(e)[:40]}")
=======
            print(f"FAILED ({str(e)[:40]})")
>>>>>>> origin/main
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
<<<<<<< HEAD
        print(f"  ⚠ Failed: {', '.join(failed)}")
    print(f"  Date range: {df_fred.index.min()} to {df_fred.index.max()}")
    print(f"  Missing values: {df_fred.isna().sum().sum():,}")

    output_path = RAW_DIR / 'fred_raw.csv'
    df_fred.to_csv(output_path)
    print(f"\n✓ Saved: {output_path}")
    print(f"  Size: {output_path.stat().st_size / (1024*1024):.2f} MB")

    return df_fred


# STEP 2: FETCH MARKET DATA (COMPLETELY REWRITTEN)
def fetch_market_raw():
    """Fetch market data with robust error handling and fallbacks"""

    print("\n" + "="*70)
    print("STEP 2/4: FETCHING MARKET DATA (ROBUST VERSION)")
    print("="*70)
    print(f"Period: {START_DATE} to {END_DATE}")
    print(f"Using ETFs (SPY, VIXY) - more reliable than indices")
    print(f"Retry logic: Up to 3 attempts with exponential backoff")
    print()

=======
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
    
>>>>>>> origin/main
    market_data = {}
    successful = 0
    
    for ticker, name in MARKET_TICKERS.items():
<<<<<<< HEAD
        retry_count = 0
        max_retries = 3

        while retry_count < max_retries:
            try:
                attempt_msg = f"attempt {retry_count + 1}/{max_retries}" if retry_count > 0 else ""
                print(f"  {name:30} ({ticker})... {attempt_msg}",
                      end=" ", flush=True)

                # Use Ticker object with session for proper headers
                ticker_obj = yf.Ticker(ticker, session=session)
                data = ticker_obj.history(
                    start=START_DATE,
                    end=END_DATE,
                    auto_adjust=True,
                    actions=False,
                    timeout=30
                )

                if data.empty:
                    raise ValueError(f"Empty dataframe for {ticker}")

                if 'Close' in data.columns:
                    market_data[name] = data['Close']
                    print(f"✓ OK {len(data):,} records")
                    successful += 1
                    break
                else:
                    raise ValueError(f"No 'Close' column for {ticker}")

            except Exception as e:
                retry_count += 1
                error_msg = str(e)[:50]

                if retry_count < max_retries:
                    wait_time = 5 * (2 ** (retry_count - 1))
                    print(f"✗ FAILED ({error_msg})")
                    print(
                        f"    Retrying in {wait_time}s...", end=" ", flush=True)
                    time.sleep(wait_time)
                else:
                    print(f"✗ FAILED after {max_retries} attempts")
                    failed.append(ticker)

        if retry_count < max_retries:
            time.sleep(2)

    # Fallback: Try index symbols if ETFs failed
    if 'SPY' in failed or 'VIXY' in failed:
        print("\n  Trying index symbol fallbacks...")

        fallback_map = {
            'SPY': ('^GSPC', 'SP500'),
            'VIXY': ('^VIX', 'VIX')
        }

        for etf_ticker in list(failed):
            if etf_ticker in fallback_map:
                fallback_ticker, name = fallback_map[etf_ticker]
                try:
                    print(f"    {fallback_ticker}...", end=" ", flush=True)
                    ticker_obj = yf.Ticker(fallback_ticker, session=session)
                    data = ticker_obj.history(
                        start=START_DATE, end=END_DATE, timeout=30)

                    if not data.empty and 'Close' in data.columns:
                        market_data[name] = data['Close']
                        print(
                            f"✓ OK (using {fallback_ticker}, {len(data):,} records)")
                        failed.remove(etf_ticker)
                        successful += 1
                    else:
                        print(f"✗ Also failed")
                except Exception as e:
                    print(f"✗ {str(e)[:30]}")

    # CRITICAL CHANGE: Don't crash if market data fails
    if not market_data:
        print("\n⚠ WARNING: No market data collected!")
        print("⚠ Creating placeholder with NaN values")
        print("⚠ You should fetch this data separately and replace market_raw.csv")

        # Create placeholder so pipeline doesn't crash
        date_range = pd.date_range(start=START_DATE, end=END_DATE, freq='D')
        market_data['SP500'] = pd.Series(index=date_range, dtype=float)
        market_data['VIX'] = pd.Series(index=date_range, dtype=float)

    df_market = pd.DataFrame(market_data)

    print(f"\nMarket Data Summary:")
    print(
        f"  Shape: {df_market.shape[0]:,} rows x {df_market.shape[1]} columns")
    print(f"  Success: {successful}/{len(MARKET_TICKERS)}")
    if failed:
        print(f"  ⚠ Failed: {', '.join(failed)}")
    print(f"  Date range: {df_market.index.min()} to {df_market.index.max()}")
    print(f"  Missing values: {df_market.isna().sum().sum():,}")

    output_path = RAW_DIR / 'market_raw.csv'
    df_market.to_csv(output_path)
    print(f"\n✓ Saved: {output_path}")
    print(f"  Size: {output_path.stat().st_size / (1024*1024):.2f} MB")

    return df_market


# STEP 3: FETCH COMPANY PRICES (IMPROVED)
def fetch_company_prices_raw():
    """Fetch company stock prices with retry logic"""

    print("\n" + "="*70)
=======
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
>>>>>>> origin/main
    print("STEP 3/4: FETCHING COMPANY PRICE DATA")
    print("=" * 70)
    print("Using multi-method fallback strategy for maximum reliability\n")
    
    all_data = []
    failed_companies = []
    method_stats = {"ChartAPI": 0, "yfinance": 0, "datareader": 0}
    
    for i, (ticker, info) in enumerate(COMPANIES.items(), 1):
<<<<<<< HEAD
        retry_count = 0
        max_retries = 2

        while retry_count < max_retries:
            try:
                attempt_msg = f"retry {retry_count}" if retry_count > 0 else ""
                print(
                    f"  [{i:2d}/25] {ticker:6} {info['name']:25} {attempt_msg}...", end=" ", flush=True)

                ticker_obj = yf.Ticker(ticker, session=session)
                prices = ticker_obj.history(
                    start=START_DATE,
                    end=END_DATE,
                    auto_adjust=False,
                    actions=True,
                    timeout=30
                )

                if prices.empty:
                    raise ValueError(f"No data for {ticker}")

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
                print(f"✓ OK {len(df):,} days")
                successful += 1
                break

            except Exception as e:
                retry_count += 1
                if retry_count < max_retries:
                    print(f"✗ retry...", end=" ", flush=True)
                    time.sleep(3)
                else:
                    print(f"✗ FAILED: {str(e)[:30]}")
                    failed.append(ticker)

        time.sleep(1)

=======
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
>>>>>>> origin/main
    if not all_data:
        raise ValueError("ERROR: No company price data collected")
    
    if len(failed_companies) > 10:
        raise ValueError(f"ERROR: Too many failures ({len(failed_companies)}/25)")
    
    # Combine all data - preserves Date index
    df_all = pd.concat(all_data, axis=0)
<<<<<<< HEAD

    print(f"\nCompany Prices Summary:")
    print(f"  Total records: {len(df_all):,}")
    print(f"  Companies: {successful}/{len(COMPANIES)}")
    if failed:
        print(f"  ⚠ Failed: {', '.join(failed)}")
    print(f"  Date range: {df_all.index.min()} to {df_all.index.max()}")
    print(f"  Columns: {list(df_all.columns)}")

    output_path = RAW_DIR / 'company_prices_raw.csv'
    df_all.to_csv(output_path)
    print(f"\n✓ Saved: {output_path}")
    print(f"  Size: {output_path.stat().st_size / (1024*1024):.2f} MB")

    return df_all


# STEP 4: ALPHA VANTAGE (NO INPUT PROMPTS)
=======
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


>>>>>>> origin/main
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


<<<<<<< HEAD
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
    df['Debt_to_Equity'] = df['Total_Liabilities'] / \
        df['Total_Equity'].replace(0, 1)
    df['Current_Ratio'] = df['Current_Assets'] / \
        df['Current_Liabilities'].replace(0, 1)
=======
def parse_financials(data, mapping):
    """Parse financial data"""
    if not data:
        return pd.DataFrame()
    
    df = pd.DataFrame([{k: r.get(v) for k, v in mapping.items()} for r in data])
    
    for col in df.columns:
        if col != "Date":
            df[col] = pd.to_numeric(df[col], errors="coerce")
    
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    
>>>>>>> origin/main
    return df


def fetch_company_fundamentals_raw():
<<<<<<< HEAD
    """Fetch company fundamentals - NO PROMPTS FOR AIRFLOW"""

    print("\n" + "="*70)
    print("STEP 4/4: FETCHING COMPANY FUNDAMENTALS (ALPHA VANTAGE)")
    print("="*70)
    print(f"Companies: {len(COMPANIES)}")
    print(f"API Keys: {len(API_KEYS)}")
    print(f"Delay: {DELAY_BETWEEN_CALLS}s between calls")
    print(
        f"Estimated time: ~{len(COMPANIES) * 2 * DELAY_BETWEEN_CALLS / 60:.0f} minutes")
    print()

    cache_file = RAW_DIR / 'financials_cache.txt'
    cached = set()

    if cache_file.exists():
        cached = set(cache_file.read_text().strip().split(','))
        if cached and '' in cached:
            cached.remove('')
        if cached:
            print(f"Cache found: {len(cached)} companies already fetched")
            print(f"Cached: {', '.join(sorted(cached))}")
            print("Using existing cache (no prompt in automated mode)")
            print()

=======
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
    
>>>>>>> origin/main
    all_income = []
    all_balance = []
    failed = []
    
    for i, (ticker, info) in enumerate(COMPANIES.items(), 1):
<<<<<<< HEAD
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
        print(f"✓ OK {len(df_income)}Q")

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
            print(f"✓ OK {len(df_balance)}Q")
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
        print(f"  ⚠ Failed: {', '.join(failed)}")

    # Save income statements
    if all_income:
        df_inc = pd.concat(all_income, ignore_index=True)
        output_path = RAW_DIR / 'company_income_raw.csv'
        df_inc.to_csv(output_path, index=False)
        print(f"\n✓ Income Statements Saved: {output_path}")
        print(f"  Records: {len(df_inc):,} quarters")
        print(f"  Companies: {df_inc['Company'].nunique()}")
        print(f"  Size: {output_path.stat().st_size / (1024*1024):.2f} MB")

    # Save balance sheets
    if all_balance:
        df_bal = pd.concat(all_balance, ignore_index=True)
        output_path = RAW_DIR / 'company_balance_raw.csv'
        df_bal.to_csv(output_path, index=False)
        print(f"\n✓ Balance Sheets Saved: {output_path}")
        print(f"  Records: {len(df_bal):,} quarters")
        print(f"  Companies: {df_bal['Company'].nunique()}")
        print(f"  Size: {output_path.stat().st_size / (1024*1024):.2f} MB")

    if len(cached) == 25:
        print(f"\n✓ ALL 25 COMPANIES COMPLETE!")
    else:
        remaining = 25 - len(cached)
        print(f"\nProgress: {len(cached)}/25 companies")
        print(f"  Remaining: {remaining} companies")
=======
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
>>>>>>> origin/main


<<<<<<< HEAD

# MAIN PIPELINE (NO PROMPTS)
def main():
    """Complete data collection pipeline - AIRFLOW READY"""

    print("\n" + "="*70)
    print("FINANCIAL STRESS TEST - COMPLETE DATA LOADER (AIRFLOW-READY)")
    print("="*70)
=======
# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main data collection pipeline"""
    print("\n" + "=" * 70)
    print("FINANCIAL DATA LOADER - ENHANCED")
    print("=" * 70)
>>>>>>> origin/main
    print(f"Period: {START_DATE} to {END_DATE}")
    print("Features: Multi-method fallback + Smart caching")
    print("=" * 70)
    
    overall_start = time.time()
    
    try:
        # Step 1: FRED (always fetch - fast)
        df_fred = fetch_fred_raw()
<<<<<<< HEAD

        # STEP 2: Market Data (won't crash if it fails)
=======
        
        # Step 2: Market (always fetch - fast)
>>>>>>> origin/main
        df_market = fetch_market_raw()
        
        # Step 3: Company Prices (always fetch - moderate)
        df_prices = fetch_company_prices_raw()
        
        # Step 4: Fundamentals (conditional - slow)
        df_income, df_balance = fetch_company_fundamentals_raw()
<<<<<<< HEAD

        # Final Summary
        elapsed = time.time() - overall_start

        print("\n" + "="*70)
        print("✓ DATA COLLECTION COMPLETE")
        print("="*70)

        print(f"\nDATA COLLECTED:")
        print(
            f"  1. FRED Macro:          {df_fred.shape[0]:,} rows x {df_fred.shape[1]} cols")
        print(
            f"  2. Market:              {df_market.shape[0]:,} rows x {df_market.shape[1]} cols")
        print(
            f"  3. Company Prices:      {df_prices.shape[0]:,} rows (25 companies)")
        if df_income is not None:
            print(
                f"  4. Income Statements:   {len(df_income):,} quarters ({df_income['Company'].nunique()} companies)")
        if df_balance is not None:
            print(
                f"  5. Balance Sheets:      {len(df_balance):,} quarters ({df_balance['Company'].nunique()} companies)")

        print(f"\nOUTPUT FILES (data/raw/):")
        print(f"  ✓ fred_raw.csv")
        print(f"  ✓ market_raw.csv")
        print(f"  ✓ company_prices_raw.csv")
        if df_income is not None:
            print(f"  ✓ company_income_raw.csv")
        if df_balance is not None:
            print(f"  ✓ company_balance_raw.csv")

        print(f"\nTotal Time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
        print("="*70)
        print("\n✓ SUCCESS: Pipeline completed!")

    except Exception as e:
        print(f"\n✗ ERROR: {str(e)}")
=======
        
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
>>>>>>> origin/main
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
<<<<<<< HEAD
    main()
=======
    main()
>>>>>>> origin/main
