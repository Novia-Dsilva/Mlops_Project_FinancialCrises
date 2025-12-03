"""
MODIFIED STEP 0: Data Collection (Cleaned Version)

Changes vs original:
- Start date: 1990-01-01 (was 2005)
- Companies: 100 (was 25)
- Frequency: Quarterly for company prices (was weekly/daily)
- Fundamentals: Direct Alpha Vantage API fetch (no cache)
"""

import pandas as pd
import numpy as np
import requests
import time
import calendar
from pandas_datareader import data as pdr
from datetime import datetime
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION - MODIFIED
# =============================================================================

START_DATE = "1990-01-01"
END_DATE = datetime.now().strftime("%Y-%m-%d")

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

# Alpha Vantage keys (can add more keys if needed)
API_KEYS = ["XBAUMM6ATPHUYXTD"]
current_key_index = 0


def get_api_key():
    """Return current Alpha Vantage API key."""
    global current_key_index
    return API_KEYS[current_key_index % len(API_KEYS)]


def switch_api_key():
    """Rotate to next API key (for rate limiting)."""
    global current_key_index
    current_key_index += 1


# =============================================================================
# EXPANDED COMPANIES - 100 TOTAL
# =============================================================================

COMPANIES = {
    # === FINANCIALS (18) ===
    "JPM":   {"name": "JPMorgan Chase",        "sector": "Financials"},
    "BAC":   {"name": "Bank of America",       "sector": "Financials"},
    "C":     {"name": "Citigroup",             "sector": "Financials"},
    "WFC":   {"name": "Wells Fargo",           "sector": "Financials"},
    "USB":   {"name": "U.S. Bancorp",          "sector": "Financials"},
    "PNC":   {"name": "PNC Financial",         "sector": "Financials"},
    "TFC":   {"name": "Truist Financial",      "sector": "Financials"},
    "BK":    {"name": "BNY Mellon",            "sector": "Financials"},
    "STT":   {"name": "State Street",          "sector": "Financials"},
    "GS":    {"name": "Goldman Sachs",         "sector": "Financials"},
    "MS":    {"name": "Morgan Stanley",        "sector": "Financials"},
    "SCHW":  {"name": "Charles Schwab",        "sector": "Financials"},
    "AXP":   {"name": "American Express",      "sector": "Financials"},
    "COF":   {"name": "Capital One",           "sector": "Financials"},
    "V":     {"name": "Visa",                  "sector": "Financials"},
    "MA":    {"name": "Mastercard",            "sector": "Financials"},
    "BRK.B": {"name": "Berkshire Hathaway",    "sector": "Financials"},
    "PGR":   {"name": "Progressive",           "sector": "Financials"},

    # === TECHNOLOGY (15) ===
    "AAPL":  {"name": "Apple",                "sector": "Technology"},
    "MSFT":  {"name": "Microsoft",            "sector": "Technology"},
    "IBM":   {"name": "IBM",                  "sector": "Technology"},
    "ORCL":  {"name": "Oracle",               "sector": "Technology"},
    "INTC":  {"name": "Intel",                "sector": "Technology"},
    "CSCO":  {"name": "Cisco",                "sector": "Technology"},
    "QCOM":  {"name": "Qualcomm",             "sector": "Technology"},
    "TXN":   {"name": "Texas Instruments",    "sector": "Technology"},
    "ADI":   {"name": "Analog Devices",       "sector": "Technology"},
    "AMZN":  {"name": "Amazon",               "sector": "Technology"},
    "GOOGL": {"name": "Alphabet",             "sector": "Technology"},
    "META":  {"name": "Meta",                 "sector": "Technology"},
    "NVDA":  {"name": "NVIDIA",               "sector": "Technology"},
    "NFLX":  {"name": "Netflix",              "sector": "Technology"},
    "CRM":   {"name": "Salesforce",           "sector": "Technology"},

    # === ENERGY (10) ===
    "XOM":   {"name": "ExxonMobil",           "sector": "Energy"},
    "CVX":   {"name": "Chevron",              "sector": "Energy"},
    "COP":   {"name": "ConocoPhillips",       "sector": "Energy"},
    "SLB":   {"name": "Schlumberger",         "sector": "Energy"},
    "EOG":   {"name": "EOG Resources",        "sector": "Energy"},
    "PXD":   {"name": "Pioneer Natural",      "sector": "Energy"},
    "MPC":   {"name": "Marathon Petroleum",   "sector": "Energy"},
    "PSX":   {"name": "Phillips 66",          "sector": "Energy"},
    "VLO":   {"name": "Valero",               "sector": "Energy"},
    "OXY":   {"name": "Occidental Petroleum", "sector": "Energy"},

    # === HEALTHCARE (10) ===
    "JNJ":   {"name": "Johnson & Johnson",     "sector": "Healthcare"},
    "UNH":   {"name": "UnitedHealth",          "sector": "Healthcare"},
    "PFE":   {"name": "Pfizer",                "sector": "Healthcare"},
    "MRK":   {"name": "Merck",                 "sector": "Healthcare"},
    "ABBV":  {"name": "AbbVie",                "sector": "Healthcare"},
    "TMO":   {"name": "Thermo Fisher",         "sector": "Healthcare"},
    "ABT":   {"name": "Abbott Labs",           "sector": "Healthcare"},
    "LLY":   {"name": "Eli Lilly",             "sector": "Healthcare"},
    "BMY":   {"name": "Bristol Myers",         "sector": "Healthcare"},
    "AMGN":  {"name": "Amgen",                 "sector": "Healthcare"},

    # === CONSUMER STAPLES (10) ===
    "WMT":   {"name": "Walmart",              "sector": "Consumer Staples"},
    "PG":    {"name": "Procter & Gamble",     "sector": "Consumer Staples"},
    "KO":    {"name": "Coca-Cola",            "sector": "Consumer Staples"},
    "PEP":   {"name": "PepsiCo",              "sector": "Consumer Staples"},
    "COST":  {"name": "Costco",               "sector": "Consumer Staples"},
    "PM":    {"name": "Philip Morris",        "sector": "Consumer Staples"},
    "MO":    {"name": "Altria",               "sector": "Consumer Staples"},
    "CL":    {"name": "Colgate-Palmolive",    "sector": "Consumer Staples"},
    "MDLZ":  {"name": "Mondelez",             "sector": "Consumer Staples"},
    "KMB":   {"name": "Kimberly-Clark",       "sector": "Consumer Staples"},

    # === CONSUMER DISCRETIONARY (10) ===
    "HD":    {"name": "Home Depot",           "sector": "Consumer Discretionary"},
    "LOW":   {"name": "Lowes",                "sector": "Consumer Discretionary"},
    "MCD":   {"name": "McDonalds",            "sector": "Consumer Discretionary"},
    "NKE":   {"name": "Nike",                 "sector": "Consumer Discretionary"},
    "SBUX":  {"name": "Starbucks",            "sector": "Consumer Discretionary"},
    "TGT":   {"name": "Target",               "sector": "Consumer Discretionary"},
    "TJX":   {"name": "TJX Companies",        "sector": "Consumer Discretionary"},
    "GM":    {"name": "General Motors",       "sector": "Consumer Discretionary"},
    "F":     {"name": "Ford",                 "sector": "Consumer Discretionary"},
    "MAR":   {"name": "Marriott",             "sector": "Consumer Discretionary"},

    # === INDUSTRIALS (10) ===
    "BA":    {"name": "Boeing",               "sector": "Industrials"},
    "CAT":   {"name": "Caterpillar",          "sector": "Industrials"},
    "GE":    {"name": "General Electric",     "sector": "Industrials"},
    "HON":   {"name": "Honeywell",            "sector": "Industrials"},
    "MMM":   {"name": "3M",                   "sector": "Industrials"},
    "UPS":   {"name": "United Parcel Service","sector": "Industrials"},
    "UNP":   {"name": "Union Pacific",        "sector": "Industrials"},
    "LMT":   {"name": "Lockheed Martin",      "sector": "Industrials"},
    "RTX":   {"name": "Raytheon Technologies","sector": "Industrials"},
    "DE":    {"name": "Deere & Company",      "sector": "Industrials"},

    # === COMMUNICATIONS (9) ===
    "DIS":   {"name": "Disney",               "sector": "Communications"},
    "CMCSA": {"name": "Comcast",              "sector": "Communications"},
    "VZ":    {"name": "Verizon",              "sector": "Communications"},
    "T":     {"name": "AT&T",                 "sector": "Communications"},
    "TMUS":  {"name": "T-Mobile",             "sector": "Communications"},
    "CHTR":  {"name": "Charter Communications","sector": "Communications"},
    "EA":    {"name": "Electronic Arts",      "sector": "Communications"},
    "ATVI":  {"name": "Activision Blizzard",  "sector": "Communications"},
    "TTWO":  {"name": "Take-Two Interactive", "sector": "Communications"},

    # === UTILITIES (5) ===
    "NEE":   {"name": "NextEra Energy",        "sector": "Utilities"},
    "DUK":   {"name": "Duke Energy",           "sector": "Utilities"},
    "SO":    {"name": "Southern Company",      "sector": "Utilities"},
    "D":     {"name": "Dominion Energy",       "sector": "Utilities"},
    "AEP":   {"name": "American Electric Power","sector": "Utilities"},

    # === REAL ESTATE (3) ===
    "AMT":   {"name": "American Tower",        "sector": "Real Estate"},
    "PLD":   {"name": "Prologis",              "sector": "Real Estate"},
    "SPG":   {"name": "Simon Property",        "sector": "Real Estate"},
}

NUM_COMPANIES = len(COMPANIES)

# =============================================================================
# FRED & MARKET CONFIG
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
    "^GSPC": "SP500",
}

# =============================================================================
# YAHOO FINANCE API - QUARTERLY
# =============================================================================

def yahoo_chart_api(ticker, start="1990-01-01", end=None, interval="3mo", timeout=20):
    """
    Wrapper around Yahoo Finance chart API.

    Default:
    - interval="3mo" for quarterly data
    - Used for both stocks (3mo) and daily market indices (1d)
    """
    if end is None:
        end = datetime.now().strftime("%Y-%m-%d")

    def to_unix(date_str: str) -> int:
        d = datetime.strptime(date_str, "%Y-%m-%d")
        return calendar.timegm(d.timetuple())

    start_ts = to_unix(start)
    end_ts = to_unix(end)

    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
        f"?period1={start_ts}&period2={end_ts}&interval={interval}&events=history"
    )

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "application/json",
    }

    last_exception = None

    for attempt in range(3):
        try:
            session = requests.Session()
            session.headers.update(headers)
            response = session.get(url, timeout=timeout)

            if response.status_code != 200:
                last_exception = ValueError(f"HTTP {response.status_code}")
                time.sleep(2 ** attempt)
                continue

            data = response.json()
            if "chart" not in data or data["chart"].get("error"):
                last_exception = ValueError("Invalid response from Yahoo chart API")
                time.sleep(2 ** attempt)
                continue

            chart = data["chart"]["result"][0]
            ts = chart.get("timestamp", [])
            if not ts:
                last_exception = ValueError("No timestamps in Yahoo chart API response")
                time.sleep(2 ** attempt)
                continue

            quote = chart["indicators"]["quote"][0]

            df = pd.DataFrame({
                "Date": pd.to_datetime(ts, unit="s"),
                "Open": quote.get("open"),
                "High": quote.get("high"),
                "Low": quote.get("low"),
                "Close": quote.get("close"),
                "Volume": quote.get("volume"),
            })

            if "adjclose" in chart["indicators"]:
                df["Adj_Close"] = chart["indicators"]["adjclose"][0]["adjclose"]
            else:
                df["Adj_Close"] = df["Close"]

            df = df.set_index("Date").dropna(how="any")

            if df.empty:
                last_exception = ValueError("All data was NaN after cleaning")
                time.sleep(2 ** attempt)
                continue

            return df

        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            last_exception = e
            time.sleep(2 ** attempt)
        except Exception as e:
            last_exception = e
            time.sleep(2 ** attempt)

    raise ValueError(f"Yahoo chart API failed after 3 attempts: {last_exception}")


# =============================================================================
# FALLBACK METHODS - QUARTERLY
# =============================================================================

def fetch_with_yfinance(ticker, start, end, interval="3mo"):
    """Fetch data via yfinance, defaulting to quarterly."""
    try:
        import yfinance as yf

        ticker_obj = yf.Ticker(ticker)
        df = ticker_obj.history(start=start, end=end, interval=interval)

        if df.empty:
            raise ValueError("Empty dataframe from yfinance")

        df = df.reset_index()
        if "Date" not in df.columns:
            df = df.rename(columns={"index": "Date"})
        df = df.set_index("Date")

        required = ["Open", "High", "Low", "Close", "Volume"]
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Missing column from yfinance: {col}")

        if "Adj Close" in df.columns:
            df["Adj_Close"] = df["Adj Close"]
        elif "Adj_Close" not in df.columns:
            df["Adj_Close"] = df["Close"]

        return df[["Open", "High", "Low", "Close", "Volume", "Adj_Close"]]

    except ImportError:
        raise ValueError("yfinance library not installed")
    except Exception as e:
        raise ValueError(f"yfinance failed: {str(e)}")


def fetch_with_datareader(ticker, start, end):
    """
    Fetch via pandas_datareader, resample to quarterly if daily.
    """
    try:
        df = pdr.get_data_yahoo(ticker, start=start, end=end)

        if df.empty:
            raise ValueError("Empty dataframe from datareader")

        # If long time span with daily data, resample to quarterly.
        if len(df) > 200:
            df = (
                df.resample("Q")
                .agg(
                    {
                        "Open": "first",
                        "High": "max",
                        "Low": "min",
                        "Close": "last",
                        "Volume": "sum",
                    }
                )
                .dropna()
            )

        if "Adj Close" in df.columns:
            df["Adj_Close"] = df["Adj Close"]
        elif "Adj_Close" not in df.columns:
            df["Adj_Close"] = df["Close"]

        return df[["Open", "High", "Low", "Close", "Volume", "Adj_Close"]]

    except Exception as e:
        raise ValueError(f"datareader failed: {str(e)}")


# =============================================================================
# STEP 1: FRED MACRO DATA
# =============================================================================

def fetch_fred_raw():
    """FRED macroeconomic data collection (unchanged frequency)."""
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
            print(f"FAILED ({str(e)[:60]})")
            failed.append(series_id)

    if not fred_data:
        raise ValueError("ERROR: No FRED data collected")

    df_fred = pd.DataFrame(fred_data)
    df_fred.index.name = "Date"

    out = RAW_DIR / "fred_raw.csv"
    df_fred.to_csv(out)

    print(f"\nSaved: {out}")
    print(f"Success: {successful}/{len(FRED_SERIES)} FRED series")
    if failed:
        print(f"Warning - failed series: {', '.join(failed)}")

    return df_fred


# =============================================================================
# STEP 2: MARKET DATA (VIX, S&P 500) - DAILY
# =============================================================================

def fetch_market_raw():
    """Market data (VIX, S&P 500) - daily frequency."""
    print("\n" + "=" * 70)
    print("STEP 2/4: FETCHING MARKET DATA (DAILY)")
    print("=" * 70)

    market_data = {}
    successful = 0

    for ticker, name in MARKET_TICKERS.items():
        print(f"  {name:25} ({ticker})...", end=" ", flush=True)

        df = None
        method_used = None

        # Method 1: Yahoo chart (daily)
        try:
            df = yahoo_chart_api(ticker, START_DATE, END_DATE, "1d")
            method_used = "ChartAPI"
        except Exception:
            print("ChartAPI failed...", end=" ", flush=True)

        # Method 2: yfinance (daily)
        if df is None or df.empty:
            try:
                df = fetch_with_yfinance(ticker, START_DATE, END_DATE, "1d")
                method_used = "yfinance"
            except Exception:
                pass

        if df is not None and not df.empty:
            df.index = pd.to_datetime(df.index.date)
            df.index.name = "Date"
            market_data[name] = df["Close"]
            print(f"OK {len(df):,} records ({method_used})")
            successful += 1
        else:
            print("FAILED all methods")

        time.sleep(1)

    if not market_data:
        raise ValueError("ERROR: No market data collected")

    df_market = pd.DataFrame(market_data)
    df_market.index.name = "Date"

    out = RAW_DIR / "market_raw.csv"
    df_market.to_csv(out)

    print(f"\nSaved: {out}")
    print(f"Success: {successful}/{len(MARKET_TICKERS)} market series")

    return df_market


# =============================================================================
# STEP 3: COMPANY PRICES - QUARTERLY
# =============================================================================

def fetch_company_prices_raw():
    """
    Fetch quarterly price data for all companies.

    - Frequency: Quarterly (3mo)
    - Universe: 100 companies (len(COMPANIES))
    - Start date: 1990-01-01
    """
    print("\n" + "=" * 70)
    print("STEP 3/4: FETCHING COMPANY PRICE DATA (QUARTERLY)")
    print("=" * 70)
    print(f"Companies: {NUM_COMPANIES}")
    print("Frequency: Quarterly (3mo)")
    print(f"Start date: {START_DATE}")
    print()

    all_data = []
    failed_companies = []
    method_stats = {"ChartAPI": 0, "yfinance": 0, "datareader": 0}

    for i, (ticker, info) in enumerate(COMPANIES.items(), 1):
        print(f"  [{i:3d}/{NUM_COMPANIES}] {ticker:6} {info['name'][:30]:30}...", end=" ", flush=True)

        df = None
        method_used = None

        # Method 1: Yahoo chart API (quarterly)
        try:
            df = yahoo_chart_api(ticker, START_DATE, END_DATE, "3mo")
            method_used = "ChartAPI"
        except Exception as e:
            print(f"ChartAPI({str(e)[:18]})...", end=" ", flush=True)

        # Method 2: yfinance (quarterly)
        if df is None or df.empty:
            try:
                df = fetch_with_yfinance(ticker, START_DATE, END_DATE, "3mo")
                method_used = "yfinance"
            except Exception as e:
                print(f"yf({str(e)[:18]})...", end=" ", flush=True)

        # Method 3: datareader (resampled to quarterly)
        if df is None or df.empty:
            try:
                df = fetch_with_datareader(ticker, START_DATE, END_DATE)
                method_used = "datareader"
            except Exception as e:
                print(f"dr({str(e)[:18]})...", end=" ", flush=True)

        # Process successful fetch
        if df is not None and not df.empty:
            df = df.copy()
            df["Company"] = ticker
            df["Company_Name"] = info["name"]
            df["Sector"] = info["sector"]
            all_data.append(df)
            if method_used in method_stats:
                method_stats[method_used] += 1
            print(f"OK {len(df):3d} quarters ({method_used})")
        else:
            print("FAILED all methods - SKIP")
            failed_companies.append(ticker)

        time.sleep(1)

    if not all_data:
        raise ValueError("ERROR: No company price data collected")

    # Allow up to 40% failures (APIs can be flaky)
    max_failures = int(NUM_COMPANIES * 0.4)
    if len(failed_companies) > max_failures:
        raise ValueError(
            f"ERROR: Too many failures ({len(failed_companies)}/{NUM_COMPANIES})"
        )

    df_all = pd.concat(all_data, axis=0)
    df_all.index.name = "Date"

    out = RAW_DIR / "company_prices_raw.csv"
    df_all.to_csv(out, index=True)

    print(f"\nSaved: {out}")
    print(f"Success: {len(all_data)}/{NUM_COMPANIES} companies")
    if failed_companies:
        print(f"WARNING: Failed companies: {', '.join(failed_companies)}")
    else:
        print("SUCCESS: All companies captured!")

    print("\nMethod statistics (by company):")
    for method, count in method_stats.items():
        if count > 0:
            pct = (count / NUM_COMPANIES) * 100
            print(f"  {method:12}: {count:3d}/{NUM_COMPANIES} ({pct:5.1f}%)")

    return df_all


# =============================================================================
# STEP 4: COMPANY FUNDAMENTALS - QUARTERLY (ALPHA VANTAGE)
# =============================================================================

def fetch_alpha_vantage(ticker, function, retry_count=0):
    """Fetch from Alpha Vantage with simple retry & rate-limit handling."""
    url = "https://www.alphavantage.co/query"
    params = {
        "function": function,
        "symbol": ticker,
        "apikey": get_api_key(),
        "datatype": "json",
    }

    try:
        r = requests.get(url, params=params, timeout=30)
        data = r.json()

        # Rate limit messages
        if "Note" in data or "Information" in data:
            if retry_count < 3:
                switch_api_key()
                time.sleep(10)
                return fetch_alpha_vantage(ticker, function, retry_count + 1)
            return None

        if "quarterlyReports" not in data:
            return None

        return data["quarterlyReports"]

    except Exception:
        if retry_count < 2:
            time.sleep(5)
            return fetch_alpha_vantage(ticker, function, retry_count + 1)
        return None


def parse_financials(data, mapping):
    """
    Parse financial JSON into a DataFrame using a key mapping dict.
    mapping: {output_col: json_field}
    """
    if not data:
        return pd.DataFrame()

    df = pd.DataFrame([{k: r.get(v) for k, v in mapping.items()} for r in data])

    for col in df.columns:
        if col != "Date":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    return df


def fetch_company_fundamentals_raw():
    """
    Fetch company fundamentals directly from Alpha Vantage API (no cache).

    - For each company:
      - Income statement (quarterly)
      - Balance sheet (quarterly + ratios)
    """
    print("\n" + "=" * 70)
    print("STEP 4/4: FETCHING COMPANY FUNDAMENTALS (QUARTERLY)")
    print("=" * 70)
    print(f"Companies: {NUM_COMPANIES}")
    est_minutes = (NUM_COMPANIES * 40) / 60.0  # ~40 seconds/company (2 calls + sleeps)
    print(f"Estimated time: ~{est_minutes:.0f} minutes (Alpha Vantage rate limits)")
    print()

    all_income = []
    all_balance = []
    failed_any = []

    for i, (ticker, info) in enumerate(COMPANIES.items(), 1):
        print(f"  [{i:3d}/{NUM_COMPANIES}] {ticker:6} {info['name'][:25]:25}", end=" ", flush=True)

        failed_this = False

        # Income Statement
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
                failed_this = True
        else:
            print("Inc:FAIL", end=" ", flush=True)
            failed_this = True

        time.sleep(20)

        # Balance Sheet
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

                # Ratios
                df_bal["Debt_to_Equity"] = df_bal["Total_Liabilities"] / df_bal[
                    "Total_Equity"
                ].replace(0, np.nan)
                df_bal["Current_Ratio"] = df_bal["Current_Assets"] / df_bal[
                    "Current_Liabilities"
                ].replace(0, np.nan)

                all_balance.append(df_bal)
                print("Bal:OK")
            else:
                print("Bal:Empty")
                failed_this = True
        else:
            print("Bal:FAIL")
            failed_this = True

        if failed_this:
            failed_any.append(ticker)

        time.sleep(20)

    print()

    # Save Income
    df_inc = None
    if all_income:
        df_inc = pd.concat(all_income, ignore_index=True)
        out_inc = RAW_DIR / "company_income_raw.csv"
        df_inc.to_csv(out_inc, index=False)
        print(f"  ✓ Income saved: {len(all_income)} companies, {len(df_inc)} quarters")
    else:
        print("  WARNING: No income data collected")

    # Save Balance
    df_bal = None
    if all_balance:
        df_bal = pd.concat(all_balance, ignore_index=True)
        out_bal = RAW_DIR / "company_balance_raw.csv"
        df_bal.to_csv(out_bal, index=False)
        print(f"  ✓ Balance saved: {len(all_balance)} companies, {len(df_bal)} quarters")
    else:
        print("  WARNING: No balance data collected")

    if failed_any:
        print(f"  WARNING: Fundamentals failed for: {', '.join(sorted(set(failed_any)))}")

    print("\n✓ Fundamentals collection complete!")
    return df_inc, df_bal


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    """Main data collection pipeline (FRED, Market, Prices, Fundamentals)."""
    print("\n" + "=" * 70)
    print("MODIFIED FINANCIAL DATA LOADER - QUARTERLY & 100 COMPANIES")
    print("=" * 70)
    print("CHANGES:")
    print(f"  - Start date: {START_DATE} (was 2005)")
    print(f"  - Companies: {NUM_COMPANIES} (expanded from 25)")
    print("  - Company prices: Quarterly (3mo, was weekly/daily)")
    print("  - Fundamentals: Direct Alpha Vantage API fetch (no cache)")
    print("=" * 70)
    print("\n⚠️  ALPHA VANTAGE API KEY REQUIRED:")
    print(f"  - Current key: {API_KEYS[0]}")
    print("  - Free tier: 25 calls/day (we need 200+ calls for 100 companies)")
    print("  - Consider: Premium key OR run in chunks over multiple days")
    print("=" * 70)

    overall_start = time.time()

    try:
        df_fred = fetch_fred_raw()
        df_market = fetch_market_raw()
        df_prices = fetch_company_prices_raw()
        df_income, df_balance = fetch_company_fundamentals_raw()

        elapsed = (time.time() - overall_start) / 60

        print("\n" + "=" * 70)
        print("DATA COLLECTION COMPLETE")
        print("=" * 70)
        print(f"Total time: {elapsed:.1f} minutes\n")
        print("Modifications applied:")
        print(f"  ✓ {NUM_COMPANIES} companies")
        print("  ✓ Quarterly company price data")
        print(f"  ✓ Data from {START_DATE}")
        print("\nFiles created in data/raw/:")
        for f in sorted(RAW_DIR.glob("*.csv")):
            size = f.stat().st_size / (1024 * 1024)
            try:
                rows = sum(1 for _ in open(f)) - 1
            except Exception:
                rows = -1
            print(f"  {f.name:30} {size:6.2f} MB  ({rows:7,} rows)")
        print("=" * 70)
        print("Ready for Step 1: Data Cleaning!")
        print("=" * 70)

    except Exception as e:
        print(f"\nFATAL ERROR in data collection: {str(e)}")
        raise


if __name__ == "__main__":
    main()
