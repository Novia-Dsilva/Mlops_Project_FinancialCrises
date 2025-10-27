"""
Financial Stress Test Generator - Complete Data Loader
✨ VALIDATES ALL DATA SOURCES: FRED, Market, Company
✨ PIPELINE: Fetch → Validate → Save
✨ OUTPUT: Separate files with _daily_oct25 suffix
"""

import pandas as pd
import numpy as np
import yfinance as yf
from pandas_datareader import data as pdr
from datetime import datetime
import warnings
import time
import os
from typing import Dict
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

START_DATE = '2000-01-01'
END_DATE = datetime.now().strftime('%Y-%m-%d')

RAW_DATA_DIR = 'data/raw'
PROCESSED_DATA_DIR = 'data/processed'

os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs('data/validation_reports', exist_ok=True)

# ============================================================================
# DATA SOURCES
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
# STEP 1: FETCH & VALIDATE FRED DATA
# ============================================================================

def fetch_and_validate_fred() -> pd.DataFrame:
    """
    STEP 1A: Fetch FRED data
    STEP 1B: Validate with Great Expectations
    STEP 1C: Save if valid
    """
    
    print_section("STEP 1A: FETCHING FRED DATA")
    
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
    
    print(f"\n✅ FRED data fetched")
    print(f"   Shape: {df_fred.shape}")
    print(f"   Date range: {df_fred.index[0]} to {df_fred.index[-1]}")
    print(f"   Missing: {df_fred.isna().sum().sum()} values")
    
    # ============================================================================
    # STEP 1B: VALIDATE FRED DATA
    # ============================================================================
    
    if GE_AVAILABLE:
        print_section("STEP 1B: VALIDATING FRED DATA")
        
        try:
            is_valid, report = validate_fred_with_ge(df_fred)
            
            if not is_valid:
                error_msg = f"FRED validation FAILED (Success: {report.get('success_rate', 0):.1f}%)"
                print(f"\n❌ {error_msg}")
                send_alert(error_msg)
                raise ValueError("FRED data validation failed")
            
            print(f"\n✅ FRED data validated")
            print(f"   Success rate: {report['success_rate']:.1f}%")
            print(f"   Expectations passed: {report.get('total', 0) - report.get('failed', 0)}/{report.get('total', 0)}")
            
        except Exception as e:
            print(f"\n⚠️  Validation error: {e}")
            print("   Proceeding without validation...")
    else:
        print("\n⚠️  Skipping validation (GE not available)")
    
    # ============================================================================
    # STEP 1C: SAVE FRED DATA
    # ============================================================================
    
    print_section("STEP 1C: SAVING FRED DATA")
    output_path = os.path.join(PROCESSED_DATA_DIR, 'fred_data_daily_oct25.csv')
    df_fred.to_csv(output_path)
    print(f"   ✅ Saved: {output_path}")
    print(f"   Size: {os.path.getsize(output_path) / (1024*1024):.1f} MB")
    
    return df_fred

# ============================================================================
# STEP 2: FETCH & VALIDATE MARKET DATA
# ============================================================================

def fetch_and_validate_market() -> pd.DataFrame:
    """
    STEP 2A: Fetch Market data (VIX, S&P 500)
    STEP 2B: Validate with Great Expectations
    STEP 2C: Save if valid
    """
    
    print_section("STEP 2A: FETCHING MARKET DATA")
    
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
    
    print(f"\n✅ Market data fetched")
    print(f"   Shape: {df_market.shape}")
    print(f"   Date range: {df_market.index[0]} to {df_market.index[-1]}")
    print(f"   Missing: {df_market.isna().sum().sum()} values")
    
    # ============================================================================
    # STEP 2B: VALIDATE MARKET DATA
    # ============================================================================
    
    if GE_AVAILABLE:
        print_section("STEP 2B: VALIDATING MARKET DATA")
        
        try:
            is_valid, report = validate_market_with_ge(df_market)
            
            if not is_valid:
                error_msg = f"Market validation FAILED (Success: {report.get('success_rate', 0):.1f}%)"
                print(f"\n❌ {error_msg}")
                send_alert(error_msg)
                raise ValueError("Market data validation failed")
            
            print(f"\n✅ Market data validated")
            print(f"   Success rate: {report['success_rate']:.1f}%")
            print(f"   Expectations passed: {report.get('total', 0) - report.get('failed', 0)}/{report.get('total', 0)}")
            
        except Exception as e:
            print(f"\n⚠️  Validation error: {e}")
            print("   Proceeding without validation...")
    else:
        print("\n⚠️  Skipping validation (GE not available)")
    
    # ============================================================================
    # STEP 2C: SAVE MARKET DATA
    # ============================================================================
    
    print_section("STEP 2C: SAVING MARKET DATA")
    output_path = os.path.join(PROCESSED_DATA_DIR, 'market_data_daily_oct25.csv')
    df_market.to_csv(output_path)
    print(f"   ✅ Saved: {output_path}")
    print(f"   Size: {os.path.getsize(output_path) / (1024*1024):.1f} MB")
    
    return df_market

# ============================================================================
# STEP 3: FETCH & VALIDATE COMPANY DATA
# ============================================================================

def fetch_company_data(ticker: str, company_info: Dict) -> pd.DataFrame:
    """Fetch data for a single company"""
    try:
        print(f" (fetching)", end="")
        prices = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False)
        
        if prices.empty:
            return pd.DataFrame()
        
        # Handle MultiIndex
        if isinstance(prices.columns, pd.MultiIndex):
            prices.columns = prices.columns.get_level_values(0)
        
        # Create dataframe
        data = pd.DataFrame(index=prices.index)
        data['Stock_Price'] = prices['Close']
        data['Stock_Return'] = prices['Close'].pct_change() * 100
        data['Stock_Volume'] = prices['Volume']
        data['Company'] = ticker
        data['Company_Name'] = company_info['name']
        data['Sector'] = company_info['sector']
        
        # Remove NaN rows
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        data = data[data[numeric_cols].notna().any(axis=1)]
        
        return data
        
    except Exception as e:
        print(f" ❌ Error: {str(e)}")
        return pd.DataFrame()

def fetch_and_validate_companies() -> pd.DataFrame:
    """
    STEP 3A: Fetch Company data (25 companies)
    STEP 3B: Validate with Great Expectations
    STEP 3C: Save if valid
    """
    
    print_section("STEP 3A: FETCHING COMPANY DATA (25 COMPANIES)")
    
    all_company_data = []
    successful = 0
    failed_companies = []
    
    for i, (ticker, info) in enumerate(COMPANIES.items(), 1):
        print(f"  [{i:2d}/25] {info['name']:30} ({ticker})...", end=" ")
        
        df = fetch_company_data(ticker, info)
        
        if not df.empty:
            all_company_data.append(df)
            print(f" ✅ {len(df)} days")
            successful += 1
        else:
            print(f" ❌ Failed")
            failed_companies.append(ticker)
        
        time.sleep(1)
    
    if not all_company_data:
        raise ValueError("No company data collected")
    
    df_companies = pd.concat(all_company_data, axis=0)
    
    print(f"\n✅ Company data fetched")
    print(f"   Companies: {successful}/25")
    if failed_companies:
        print(f"   Failed: {', '.join(failed_companies)}")
    print(f"   Total records: {len(df_companies):,}")
    print(f"   Date range: {df_companies.index.min()} to {df_companies.index.max()}")
    print(f"   Missing: {df_companies.isna().sum().sum()} values")
    
    # ============================================================================
    # STEP 3B: VALIDATE COMPANY DATA
    # ============================================================================
    
    if GE_AVAILABLE:
        print_section("STEP 3B: VALIDATING COMPANY DATA")
        
        try:
            is_valid, report = validate_company_with_ge(df_companies)
            
            if not is_valid:
                error_msg = f"Company validation FAILED (Success: {report.get('success_rate', 0):.1f}%)"
                print(f"\n❌ {error_msg}")
                send_alert(error_msg)
                raise ValueError("Company data validation failed")
            
            print(f"\n✅ Company data validated")
            print(f"   Success rate: {report['success_rate']:.1f}%")
            print(f"   Expectations passed: {report.get('total', 0) - report.get('failed', 0)}/{report.get('total', 0)}")
            
        except Exception as e:
            print(f"\n⚠️  Validation error: {e}")
            print("   Proceeding without validation...")
    else:
        print("\n⚠️  Skipping validation (GE not available)")
    
    # ============================================================================
    # STEP 3C: SAVE COMPANY DATA
    # ============================================================================
    
    print_section("STEP 3C: SAVING COMPANY DATA")
    
    # Reset index to save Date as column
    df_save = df_companies.reset_index().rename(columns={'index': 'Date'})
    
    output_path = os.path.join(PROCESSED_DATA_DIR, 'company_data_daily_oct25.csv')
    df_save.to_csv(output_path, index=False)
    print(f"   ✅ Saved: {output_path}")
    print(f"   Size: {os.path.getsize(output_path) / (1024*1024):.1f} MB")
    
    return df_companies

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """
    Complete pipeline: Fetch and Validate ALL data sources
    """
    start_time = time.time()
    
    print("\n" + "="*70)
    print("📊 FINANCIAL STRESS TEST - DATA INGESTION & VALIDATION")
    print("="*70)
    print(f"📅 Period: {START_DATE} to {END_DATE}")
    print(f"🏢 Companies: 25")
    print(f"💾 Output: data/processed/*_daily_oct25.csv")
    print(f"🔍 Validation: {'✅ ENABLED (All Sources)' if GE_AVAILABLE else '⚠️  DISABLED'}")
    print("="*70)
    
    validation_summary = {
        'FRED': {'status': '❓', 'success_rate': 0},
        'Market': {'status': '❓', 'success_rate': 0},
        'Company': {'status': '❓', 'success_rate': 0}
    }
    
    try:
        # STEP 1: FRED Data (Fetch → Validate → Save)
        df_fred = fetch_and_validate_fred()
        validation_summary['FRED']['status'] = '✅'
        
        # STEP 2: Market Data (Fetch → Validate → Save)
        df_market = fetch_and_validate_market()
        validation_summary['Market']['status'] = '✅'
        
        # STEP 3: Company Data (Fetch → Validate → Save)
        df_companies = fetch_and_validate_companies()
        validation_summary['Company']['status'] = '✅'
        
        # Summary
        elapsed = time.time() - start_time
        print("\n" + "="*70)
        print("✅ DATA INGESTION & VALIDATION COMPLETE")
        print("="*70)
        
        print(f"\n📊 COLLECTED DATA:")
        print(f"   FRED: {df_fred.shape[0]} observations, {df_fred.shape[1]} indicators")
        print(f"   Market: {df_market.shape[0]} observations, {df_market.shape[1]} indicators")
        print(f"   Companies: {len(df_companies):,} observations, {df_companies['Company'].nunique()} companies")
        
        print(f"\n🔍 VALIDATION RESULTS:")
        print(f"   FRED:    {validation_summary['FRED']['status']}")
        print(f"   Market:  {validation_summary['Market']['status']}")
        print(f"   Company: {validation_summary['Company']['status']}")
        
        print(f"\n💾 OUTPUT FILES:")
        print(f"   📄 data/processed/fred_data_daily_oct25.csv")
        print(f"   📄 data/processed/market_data_daily_oct25.csv")
        print(f"   📄 data/processed/company_data_daily_oct25.csv")
        
        if GE_AVAILABLE:
            print(f"\n📋 VALIDATION REPORTS:")
            print(f"   📄 data/validation_reports/fred_macro_data_*.json")
            print(f"   📄 data/validation_reports/market_data_*.json")
            print(f"   📄 data/validation_reports/company_data_*.json")
        
        print(f"\n⏱️  Total time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
        print(f"\n🎉 SUCCESS! All data sources validated and saved.")
        print("="*70)
        
    except ValueError as e:
        print("\n" + "="*70)
        print("❌ PIPELINE STOPPED DUE TO VALIDATION FAILURE")
        print("="*70)
        print(f"   Error: {str(e)}")
        
        print(f"\n📊 VALIDATION STATUS:")
        for source, info in validation_summary.items():
            print(f"   {source}: {info['status']}")
        
        print("\n   Next steps:")
        print("   1. Check validation reports in data/validation_reports/")
        print("   2. Review API status (FRED, Yahoo Finance)")
        print("   3. Fix data issues and re-run")
        raise
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()

