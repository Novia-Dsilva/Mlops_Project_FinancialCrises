"""
Financial Stress Test Generator - Enhanced Data Loader
✨ FETCHES: Price Data (Yahoo) + Financial Statements (Alpha Vantage)
✨ VALIDATES: All data sources with Great Expectations
✨ OUTPUTS: Complete dataset with EPS, Revenue, Debt ratios
"""

import pandas as pd
import numpy as np
import yfinance as yf
from pandas_datareader import data as pdr
from datetime import datetime
import warnings
import time
import os
import requests
from typing import Dict, Tuple
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from validation.great_expectations_validator import (
        validate_fred_with_ge,
        validate_market_with_ge,
        validate_company_with_ge
    )
    GE_AVAILABLE = True
    print("✅ Great Expectations validator loaded")
except ImportError as e:
    print(f"⚠️  Great Expectations not available: {e}")
    GE_AVAILABLE = False

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

START_DATE = '2000-01-01'  # Alpha Vantage works best from 2010+
END_DATE = datetime.now().strftime('%Y-%m-%d')

RAW_DATA_DIR = 'data/raw'
PROCESSED_DATA_DIR = 'data/processed'

os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs('data/validation_reports', exist_ok=True)

# Alpha Vantage API Key
ALPHA_VANTAGE_API_KEY = '3T2BMC1FFD1R078D'

# Rate Limiting
MAX_COMPANIES_PER_RUN = 12  # Alpha Vantage: 25 calls/day, use ~12 for safety
API_CALL_DELAY = 15  # seconds between calls (5 calls/minute limit)

# ============================================================================
# DATA SOURCES (Same as before)
# ============================================================================

FRED_SERIES = {
    'GDPC1': 'GDP_Growth',
    'CPIAUCSL': 'CPI_Inflation',
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

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def print_section(title: str):
    print("\n" + "="*70)
    print(f"📊 {title}")
    print("="*70)

def send_alert(message: str):
    print(f"\n🚨 ALERT: {message}")

# ============================================================================
# ALPHA VANTAGE FUNCTIONS
# ============================================================================

def fetch_alpha_vantage_financials(ticker: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fetch quarterly financial statements from Alpha Vantage
    Returns: (income_statement, balance_sheet)
    """
    try:
        # 1. Income Statement
        print(f"      → Income statement...", end=" ")
        url_income = "https://www.alphavantage.co/query"
        params_income = {
            'function': 'INCOME_STATEMENT',
            'symbol': ticker,
            'apikey': ALPHA_VANTAGE_API_KEY
        }
        
        response = requests.get(url_income, params=params_income, timeout=30)
        data_income = response.json()
        
        if 'quarterlyReports' not in data_income:
            print(f"❌ No income data")
            return pd.DataFrame(), pd.DataFrame()
        
        # Parse income statement
        income_reports = []
        for report in data_income['quarterlyReports']:
            income_reports.append({
                'Date': pd.to_datetime(report['fiscalDateEnding']),
                'Revenue': float(report.get('totalRevenue', 0)),
                'Net_Income': float(report.get('netIncome', 0)),
                'EBITDA': float(report.get('ebitda', 0)),
                'EPS': float(report.get('reportedEPS', 0)),
                'Operating_Income': float(report.get('operatingIncome', 0)),
                'Gross_Profit': float(report.get('grossProfit', 0)),
            })
        
        df_income = pd.DataFrame(income_reports)
        print(f"✅ {len(df_income)}Q", end=" ")
        
        time.sleep(API_CALL_DELAY)  # Rate limit
        
        # 2. Balance Sheet
        print(f"| Balance sheet...", end=" ")
        params_balance = {
            'function': 'BALANCE_SHEET',
            'symbol': ticker,
            'apikey': ALPHA_VANTAGE_API_KEY
        }
        
        response = requests.get(url_income, params=params_balance, timeout=30)
        data_balance = response.json()
        
        if 'quarterlyReports' not in data_balance:
            print(f"❌ No balance data")
            return df_income, pd.DataFrame()
        
        # Parse balance sheet
        balance_reports = []
        for report in data_balance['quarterlyReports']:
            total_assets = float(report.get('totalAssets', 0))
            total_liabilities = float(report.get('totalLiabilities', 0))
            total_equity = float(report.get('totalShareholderEquity', 1))
            current_assets = float(report.get('totalCurrentAssets', 0))
            current_liabilities = float(report.get('totalCurrentLiabilities', 1))
            
            balance_reports.append({
                'Date': pd.to_datetime(report['fiscalDateEnding']),
                'Total_Assets': total_assets,
                'Total_Liabilities': total_liabilities,
                'Total_Equity': total_equity,
                'Current_Assets': current_assets,
                'Current_Liabilities': current_liabilities,
                'Debt_to_Equity': total_liabilities / total_equity if total_equity > 0 else np.nan,
                'Current_Ratio': current_assets / current_liabilities if current_liabilities > 0 else np.nan,
            })
        
        df_balance = pd.DataFrame(balance_reports)
        print(f"✅ {len(df_balance)}Q")
        
        return df_income, df_balance
        
    except requests.exceptions.Timeout:
        print(f"❌ API timeout")
        return pd.DataFrame(), pd.DataFrame()
    except Exception as e:
        print(f"❌ Error: {str(e)[:50]}")
        return pd.DataFrame(), pd.DataFrame()

def merge_financials(df_income: pd.DataFrame, df_balance: pd.DataFrame) -> pd.DataFrame:
    """Merge income statement and balance sheet on Date"""
    if df_income.empty or df_balance.empty:
        return pd.DataFrame()
    
    df_income.set_index('Date', inplace=True)
    df_balance.set_index('Date', inplace=True)
    
    df_merged = df_income.join(df_balance, how='outer')
    
    # Calculate derived metrics
    df_merged['Profit_Margin'] = (df_merged['Net_Income'] / df_merged['Revenue']) * 100
    df_merged['Revenue_Growth'] = df_merged['Revenue'].pct_change() * 100
    df_merged['EPS_Growth'] = df_merged['EPS'].pct_change() * 100
    df_merged['Asset_Turnover'] = df_merged['Revenue'] / df_merged['Total_Assets']
    df_merged['ROE'] = (df_merged['Net_Income'] / df_merged['Total_Equity']) * 100
    
    return df_merged

# ============================================================================
# ENHANCED COMPANY DATA FETCHING
# ============================================================================

def fetch_company_price_data(ticker: str, company_info: Dict) -> pd.DataFrame:
    """Fetch daily price data with enhanced features"""
    try:
        print(f"      → Price data...", end=" ")
        prices = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False)
        
        if prices.empty:
            print(f"❌ No price data")
            return pd.DataFrame()
        
        # Handle MultiIndex
        if isinstance(prices.columns, pd.MultiIndex):
            prices.columns = prices.columns.get_level_values(0)
        
        # Create dataframe with enhanced features
        data = pd.DataFrame(index=prices.index)
        
        # Basic price metrics
        data['Stock_Price'] = prices['Close']
        data['Stock_Return'] = prices['Close'].pct_change() * 100
        data['Stock_Volume'] = prices['Volume']
        
        # Technical indicators
        data['Volatility_20D'] = prices['Close'].rolling(window=20).std()
        data['Volatility_60D'] = prices['Close'].rolling(window=60).std()
        data['MA_50'] = prices['Close'].rolling(window=50).mean()
        data['MA_200'] = prices['Close'].rolling(window=200).mean()
        data['Volume_MA_20'] = prices['Volume'].rolling(window=20).mean()
        data['Price_Range'] = (prices['High'] - prices['Low']) / prices['Low'] * 100
        
        # Momentum
        data['Returns_5D'] = prices['Close'].pct_change(5) * 100
        data['Returns_20D'] = prices['Close'].pct_change(20) * 100
        data['Returns_60D'] = prices['Close'].pct_change(60) * 100
        
        # Company metadata
        data['Company'] = ticker
        data['Company_Name'] = company_info['name']
        data['Sector'] = company_info['sector']
        
        print(f"✅ {len(data)} days")
        
        return data
        
    except Exception as e:
        print(f"❌ Error: {str(e)[:50]}")
        return pd.DataFrame()

def fetch_company_complete_data(ticker: str, company_info: Dict, 
                                fetch_financials: bool = True) -> pd.DataFrame:
    """
    Fetch COMPLETE company data:
    - Daily price data (Yahoo Finance)
    - Quarterly financials (Alpha Vantage)
    - Merged and forward-filled
    """
    
    # 1. Get price data
    df_prices = fetch_company_price_data(ticker, company_info)
    
    if df_prices.empty:
        return pd.DataFrame()
    
    # 2. Get financial statements (if enabled)
    if fetch_financials:
        df_income, df_balance = fetch_alpha_vantage_financials(ticker)
        df_financials = merge_financials(df_income, df_balance)
        
        if not df_financials.empty:
            # Merge quarterly financials with daily prices
            # Forward fill quarterly data to daily frequency
            df_complete = df_prices.join(df_financials, how='left')
            df_complete.fillna(method='ffill', inplace=True)
            
            print(f"      ✅ Complete dataset: {len(df_complete)} days with financials")
            return df_complete
        else:
            print(f"      ⚠️  Price data only (no financials)")
            return df_prices
    else:
        return df_prices

# ============================================================================
# MAIN DATA COLLECTION (Modified from your original)
# ============================================================================

def fetch_and_validate_fred() -> pd.DataFrame:
    """Same as your original - kept unchanged"""
    
    print_section("STEP 1: FETCHING FRED MACROECONOMIC DATA")
    
    fred_data = {}
    successful = 0
    
    for series_id, col_name in FRED_SERIES.items():
        try:
            print(f"  Fetching {col_name:30} ({series_id})...", end=" ")
            df = pdr.DataReader(series_id, 'fred', START_DATE, END_DATE)
            fred_data[col_name] = df.iloc[:, 0]
            print(f"✅ {len(df)} records")
            successful += 1
            time.sleep(0.5)
        except Exception as e:
            print(f"❌ Failed: {str(e)}")
    
    if not fred_data:
        raise ValueError("No FRED data collected")
    
    df_fred = pd.DataFrame(fred_data)
    
    # Calculate growth rates
    if 'GDP_Growth' in df_fred.columns:
        df_fred['GDP_Growth'] = df_fred['GDP_Growth'].pct_change() * 100
    
    if 'CPI_Inflation' in df_fred.columns:
        df_fred['CPI_Inflation'] = df_fred['CPI_Inflation'].pct_change(4) * 100
    
    print(f"\n✅ FRED data collected")
    print(f"   Shape: {df_fred.shape}")
    print(f"   Date range: {df_fred.index[0]} to {df_fred.index[-1]}")
    
    # Save
    output_path = os.path.join(PROCESSED_DATA_DIR, 'fred_data_with_financials.csv')
    df_fred.to_csv(output_path)
    print(f"   💾 Saved: {output_path}")
    
    return df_fred

def fetch_and_validate_market() -> pd.DataFrame:
    """Same as your original - kept unchanged"""
    
    print_section("STEP 2: FETCHING MARKET DATA")
    
    market_data = {}
    
    for ticker, name in MARKET_TICKERS.items():
        try:
            print(f"  Fetching {name:30} ({ticker})...", end=" ")
            data = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False)
            
            if not data.empty and 'Close' in data.columns:
                close_data = data['Close']
                if isinstance(close_data, pd.DataFrame):
                    close_data = close_data.iloc[:, 0]
                
                market_data[name] = close_data
                print(f"✅ {len(data)} records")
            else:
                print(f"❌ No data")
            
            time.sleep(1)
        except Exception as e:
            print(f"❌ Failed: {str(e)}")
    
    if not market_data:
        raise ValueError("No market data collected")
    
    df_market = pd.DataFrame(market_data)
    
    # Calculate returns
    if 'SP500' in df_market.columns:
        df_market['SP500_Return'] = df_market['SP500'].pct_change() * 100
    
    print(f"\n✅ Market data collected")
    print(f"   Shape: {df_market.shape}")
    
    # Save
    output_path = os.path.join(PROCESSED_DATA_DIR, 'market_data_with_financials.csv')
    df_market.to_csv(output_path)
    print(f"   💾 Saved: {output_path}")
    
    return df_market

def fetch_and_validate_companies() -> pd.DataFrame:
    """
    ENHANCED VERSION: Fetches price + financials
    Uses Alpha Vantage rate limits intelligently
    """
    
    print_section(f"STEP 3: FETCHING COMPANY DATA (PRICE + FINANCIALS)")
    print(f"⚠️  Alpha Vantage Rate Limit: {MAX_COMPANIES_PER_RUN} companies/run")
    print(f"⚠️  API Call Delay: {API_CALL_DELAY} seconds between calls")
    print(f"⏱️  Estimated time: {(MAX_COMPANIES_PER_RUN * API_CALL_DELAY * 2) / 60:.1f} minutes\n")
    
    all_company_data = []
    successful = 0
    failed_companies = []
    companies_with_financials = 0
    companies_price_only = 0
    
    # Check if we already have some data cached
    cache_file = os.path.join(PROCESSED_DATA_DIR, 'company_collection_progress.txt')
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            collected_companies = set(f.read().strip().split(','))
        print(f"📋 Found cache: {len(collected_companies)} companies already collected")
    else:
        collected_companies = set()
    
    companies_processed = 0
    
    for i, (ticker, info) in enumerate(COMPANIES.items(), 1):
        # Skip if already collected (for incremental collection)
        if ticker in collected_companies:
            print(f"  [{i:2d}/25] {info['name']:30} ({ticker})... ⏭️  Cached")
            continue
        
        print(f"  [{i:2d}/25] {info['name']:30} ({ticker})...")
        
        # Check if we've hit rate limit for this run
        fetch_financials = companies_processed < MAX_COMPANIES_PER_RUN
        
        if not fetch_financials:
            print(f"      ⚠️  Rate limit reached - collecting price data only")
        
        # Fetch complete data
        df = fetch_company_complete_data(ticker, info, fetch_financials=fetch_financials)
        
        if not df.empty:
            all_company_data.append(df)
            successful += 1
            
            # Track what we got
            if fetch_financials and 'EPS' in df.columns and df['EPS'].notna().any():
                companies_with_financials += 1
            else:
                companies_price_only += 1
            
            # Update cache
            collected_companies.add(ticker)
            with open(cache_file, 'w') as f:
                f.write(','.join(collected_companies))
            
            companies_processed += 1
        else:
            print(f"      ❌ Failed completely")
            failed_companies.append(ticker)
        
        time.sleep(1)
    
    if not all_company_data:
        raise ValueError("No company data collected")
    
    df_companies = pd.concat(all_company_data, axis=0)
    
    print(f"\n✅ Company data collected")
    print(f"   Total companies: {successful}/25")
    print(f"   With financials: {companies_with_financials}")
    print(f"   Price only: {companies_price_only}")
    if failed_companies:
        print(f"   Failed: {', '.join(failed_companies)}")
    print(f"   Total records: {len(df_companies):,}")
    print(f"   Date range: {df_companies.index.min()} to {df_companies.index.max()}")
    
    # Show feature summary
    financial_features = ['EPS', 'Revenue', 'Net_Income', 'Debt_to_Equity', 'Profit_Margin']
    available_financial = [f for f in financial_features if f in df_companies.columns]
    print(f"   Financial features available: {', '.join(available_financial)}")
    
    # Save
    df_save = df_companies.reset_index().rename(columns={'index': 'Date'})
    output_path = os.path.join(PROCESSED_DATA_DIR, 'company_data_with_financials.csv')
    df_save.to_csv(output_path, index=False)
    print(f"   💾 Saved: {output_path}")
    print(f"   Size: {os.path.getsize(output_path) / (1024*1024):.1f} MB")
    
    return df_companies

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """Complete pipeline with financial statements"""
    start_time = time.time()
    
    print("\n" + "="*70)
    print("📊 FINANCIAL STRESS TEST - ENHANCED DATA COLLECTION")
    print("="*70)
    print(f"📅 Period: {START_DATE} to {END_DATE}")
    print(f"🏢 Companies: 25 (12 with financials per run)")
    print(f"💾 Output: data/processed/*_with_financials.csv")
    print(f"🔑 Alpha Vantage: Enabled")
    print("="*70)
    
    try:
        # STEP 1: FRED Data
        df_fred = fetch_and_validate_fred()
        
        # STEP 2: Market Data
        df_market = fetch_and_validate_market()
        
        # STEP 3: Company Data (ENHANCED)
        df_companies = fetch_and_validate_companies()
        
        # Summary
        elapsed = time.time() - start_time
        print("\n" + "="*70)
        print("✅ ENHANCED DATA COLLECTION COMPLETE")
        print("="*70)
        
        print(f"\n📊 COLLECTED DATA:")
        print(f"   FRED: {df_fred.shape[0]} observations, {df_fred.shape[1]} indicators")
        print(f"   Market: {df_market.shape[0]} observations, {df_market.shape[1]} indicators")
        print(f"   Companies: {len(df_companies):,} observations")
        print(f"   Companies: {df_companies['Company'].nunique()} unique tickers")
        print(f"   Features per company: {df_companies.shape[1]}")
        
        print(f"\n💾 OUTPUT FILES:")
        print(f"   📄 data/processed/fred_data_with_financials.csv")
        print(f"   📄 data/processed/market_data_with_financials.csv")
        print(f"   📄 data/processed/company_data_with_financials.csv")
        
        print(f"\n⏱️  Total time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
        
        # Check if we need another run
        remaining = 25 - df_companies['Company'].nunique()
        if remaining > 0:
            print(f"\n📋 NEXT STEPS:")
            print(f"   Run this script {remaining // MAX_COMPANIES_PER_RUN + 1} more time(s) to collect remaining companies")
            print(f"   Progress is saved - already collected companies will be skipped")
        else:
            print(f"\n🎉 SUCCESS! All 25 companies collected with financial data.")
        
        print("="*70)
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()

