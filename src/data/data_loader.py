"""
Financial Stress Test Generator - Data Loader (HISTORICAL DATA FIX)
Collects macro data from FRED and company data from Yahoo Finance
Period: 1990-2025 | Companies: 25 | Sources: FRED + Yahoo Finance
✨ FIXED: Now fetches historical company data (2010-2025, ~60 quarters)
"""

import pandas as pd
import numpy as np
import yfinance as yf
from pandas_datareader import data as pdr
from datetime import datetime
import warnings
import time
import os
from typing import Dict, List

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

# Date range
START_DATE = '2000-01-01'  # Changed from 1990 to 2010 for better company data
END_DATE = datetime.now().strftime('%Y-%m-%d')

# Data directories
RAW_DATA_DIR = 'data/raw'
PROCESSED_DATA_DIR = 'data/processed'

# Ensure directories exist
os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

# ============================================================================
# FRED INDICATORS (13 Series)
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

# ============================================================================
# 25 COMPANIES
# ============================================================================

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
    """Print formatted section header"""
    print("\n" + "="*70)
    print(f"📊 {title}")
    print("="*70)

def save_intermediate(df: pd.DataFrame, name: str):
    """Save intermediate results"""
    path = os.path.join(RAW_DATA_DIR, f'{name}_data.csv')
    df.to_csv(path)
    print(f"   💾 Saved cache: {path}")

def load_cached(name: str) -> pd.DataFrame:
    """Load cached data if exists"""
    path = os.path.join(RAW_DATA_DIR, f'{name}_data.csv')
    if os.path.exists(path):
        print(f"   📂 Loading cached {name} data from: {path}")
        return pd.read_csv(path, index_col=0, parse_dates=True)
    return None

def fetch_fred_data() -> pd.DataFrame:
    """Fetch all FRED indicators"""
    cached = load_cached('fred')
    if cached is not None:
        print_section("USING CACHED FRED DATA")
        print(f"   Shape: {cached.shape}")
        print(f"   Date range: {cached.index[0]} to {cached.index[-1]}")
        return cached
    
    print_section("FETCHING FRED DATA (13 SERIES)")
    
    fred_data = {}
    successful = 0
    
    for series_id, col_name in FRED_SERIES.items():
        try:
            print(f"  Fetching {col_name:30} ({series_id})...", end=" ")
            df = pdr.DataReader(series_id, 'fred', START_DATE, END_DATE)
            fred_data[col_name] = df.iloc[:, 0]
            print(f" {len(df)} records")
            successful += 1
            time.sleep(0.5)
        except Exception as e:
            print(f"❌ Failed: {str(e)}")
    
    df_fred = pd.DataFrame(fred_data)
    df_fred_quarterly = df_fred.resample('Q').last()
    
    if 'GDP_Growth' in df_fred_quarterly.columns:
        df_fred_quarterly['GDP_Growth'] = df_fred_quarterly['GDP_Growth'].pct_change() * 100
    
    if 'CPI_Inflation' in df_fred_quarterly.columns:
        df_fred_quarterly['CPI_Inflation'] = df_fred_quarterly['CPI_Inflation'].pct_change(4) * 100
    
    print(f"\n✅ FRED DATA COMPLETE")
    print(f"   Shape: {df_fred_quarterly.shape}")
    print(f"   Quarters: {len(df_fred_quarterly)}")
    print(f"   Date range: {df_fred_quarterly.index[0]} to {df_fred_quarterly.index[-1]}")
    
    save_intermediate(df_fred_quarterly, 'fred')
    return df_fred_quarterly

def fetch_market_data() -> pd.DataFrame:
    """Fetch VIX and S&P 500 data"""
    cached = load_cached('market')
    if cached is not None:
        print_section("USING CACHED MARKET DATA")
        print(f"   Shape: {cached.shape}")
        return cached
    
    print_section("FETCHING MARKET DATA (VIX, S&P 500)")
    
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
                print(f"✅ {len(data)} daily records")
            else:
                print(f"❌ No data returned")
                
            time.sleep(1)
        except Exception as e:
            print(f"❌ Failed: {str(e)}")
    
    if not market_data:
        print("⚠️  WARNING: No market data collected.")
        return pd.DataFrame()
    
    df_market = pd.DataFrame(market_data)
    df_market_quarterly = df_market.resample('Q').agg({
        'VIX': 'mean',
        'SP500': 'last'
    })
    
    if 'SP500' in df_market_quarterly.columns:
        df_market_quarterly['SP500_Return'] = df_market_quarterly['SP500'].pct_change() * 100
    
    print(f"\n✅ MARKET DATA COMPLETE")
    print(f"   Shape: {df_market_quarterly.shape}")
    
    save_intermediate(df_market_quarterly, 'market')
    return df_market_quarterly

def fetch_company_financials_historical(ticker: str, company_info: Dict) -> pd.DataFrame:
    """
    Fetch HISTORICAL company data using stock prices and calculated ratios
    This fetches 10+ years of data instead of just recent quarters
    """
    try:
        stock = yf.Ticker(ticker)
        
        # Get historical stock prices (goes back further!)
        print(f" (prices)", end="")
        prices = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False)
        
        # Check if data is empty
        if prices.empty:
            print(f" ❌ No price data")
            return pd.DataFrame()
        
        # Handle different DataFrame formats from yfinance
        # Sometimes columns are MultiIndex, sometimes not
        if isinstance(prices.columns, pd.MultiIndex):
            # MultiIndex format: flatten it
            prices.columns = prices.columns.get_level_values(0)
        
        # Verify we have the required columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_cols = [col for col in required_cols if col not in prices.columns]
        
        if missing_cols:
            # Try alternative column names (lowercase)
            prices.columns = [col.capitalize() if isinstance(col, str) else col for col in prices.columns]
            missing_cols = [col for col in required_cols if col not in prices.columns]
            
            if missing_cols:
                print(f" ❌ Missing columns: {missing_cols}")
                return pd.DataFrame()
        
        # Resample to quarterly
        prices_quarterly = prices.resample('Q').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        })
        
        # Check if resampled data is valid
        if prices_quarterly.empty or prices_quarterly['Close'].isna().all():
            print(f" ❌ Empty after resampling")
            return pd.DataFrame()
        
        # Create dataframe
        data = pd.DataFrame(index=prices_quarterly.index)
        data['Stock_Price'] = prices_quarterly['Close']
        data['Stock_Return'] = prices_quarterly['Close'].pct_change() * 100
        data['Volatility'] = prices_quarterly['Close'].pct_change().rolling(4).std() * 100
        
        # Try to get quarterly financials (if available)
        print(f" (financials)", end="")
        try:
            # Get quarterly income statement
            income_stmt = stock.quarterly_income_stmt
            
            if income_stmt is not None and not income_stmt.empty:
                # Transpose to have dates as index
                income_stmt = income_stmt.T
                income_stmt.index = pd.to_datetime(income_stmt.index)
                income_quarterly = income_stmt.resample('Q').last()
                
                # Map common column names
                revenue_cols = ['Total Revenue', 'Revenue']
                for col in revenue_cols:
                    if col in income_quarterly.columns:
                        data['Revenue'] = income_quarterly[col]
                        break
                
                netincome_cols = ['Net Income', 'Net Income Common Stockholders']
                for col in netincome_cols:
                    if col in income_quarterly.columns:
                        data['Net_Income'] = income_quarterly[col]
                        break
                
                operating_cols = ['Operating Income', 'Operating Revenue']
                for col in operating_cols:
                    if col in income_quarterly.columns:
                        data['Operating_Income'] = income_quarterly[col]
                        break
                
                eps_cols = ['Basic EPS', 'Diluted EPS']
                for col in eps_cols:
                    if col in income_quarterly.columns:
                        data['EPS'] = income_quarterly[col]
                        break
            
            # Get quarterly balance sheet
            balance_sheet = stock.quarterly_balance_sheet
            
            if balance_sheet is not None and not balance_sheet.empty:
                balance_sheet = balance_sheet.T
                balance_sheet.index = pd.to_datetime(balance_sheet.index)
                balance_quarterly = balance_sheet.resample('Q').last()
                
                debt_cols = ['Total Debt', 'Long Term Debt', 'Total Liabilities Net Minority Interest']
                for col in debt_cols:
                    if col in balance_quarterly.columns:
                        data['Total_Debt'] = balance_quarterly[col]
                        break
                
                equity_cols = ['Total Equity Gross Minority Interest', 'Stockholders Equity', 'Total Equity']
                for col in equity_cols:
                    if col in balance_quarterly.columns:
                        data['Total_Equity'] = balance_quarterly[col]
                        break
                
                current_assets_cols = ['Current Assets']
                for col in current_assets_cols:
                    if col in balance_quarterly.columns:
                        data['Current_Assets'] = balance_quarterly[col]
                        break
                
                current_liab_cols = ['Current Liabilities']
                for col in current_liab_cols:
                    if col in balance_quarterly.columns:
                        data['Current_Liabilities'] = balance_quarterly[col]
                        break
        except Exception as e:
            # If quarterly data fails, that's OK - we still have stock prices
            print(f" (no-fin:{str(e)[:20]})", end="")
        
        # Calculate Market Cap from price if not available
        if 'Market_Cap' not in data.columns or data['Market_Cap'].isna().all():
            try:
                info = stock.info
                if 'sharesOutstanding' in info and info['sharesOutstanding']:
                    shares = info['sharesOutstanding']
                    data['Market_Cap'] = data['Stock_Price'] * shares
            except:
                data['Market_Cap'] = np.nan
        
        # Add metadata
        data['Company'] = ticker
        data['Company_Name'] = company_info['name']
        data['Sector'] = company_info['sector']
        
        # Remove rows where ALL numeric columns are NaN
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        data = data[data[numeric_cols].notna().any(axis=1)]
        
        if data.empty:
            print(f" ❌ All data is NaN")
            return pd.DataFrame()
        
        return data
        
    except Exception as e:
        print(f" ❌ Error: {str(e)}")
        return pd.DataFrame()

def fetch_all_companies() -> pd.DataFrame:
    """Fetch financial data for all 25 companies"""
    cached = load_cached('companies')
    if cached is not None:
        print_section("USING CACHED COMPANY DATA")
        print(f"   Shape: {cached.shape}")
        print(f"   Companies: {cached['Company'].nunique()}")
        if 'Date' in cached.columns:
            cached['Date'] = pd.to_datetime(cached['Date'])
        return cached
    
    print_section("FETCHING HISTORICAL COMPANY DATA (25 COMPANIES)")
    print("⚠️  Using historical stock prices (2010-2025)")
    
    all_company_data = []
    successful = 0
    failed_companies = []
    
    for i, (ticker, info) in enumerate(COMPANIES.items(), 1):
        print(f"  [{i:2d}/25] Fetching {info['name']:30} ({ticker})...", end=" ")
        
        df = fetch_company_financials_historical(ticker, info)
        
        if not df.empty:
            all_company_data.append(df)
            print(f" ✅ {len(df)} quarters")
            successful += 1
        else:
            print(f" ❌ Failed")
            failed_companies.append(ticker)
        
        time.sleep(1)  # Rate limiting
    
    # Check if we have any data
    if not all_company_data:
        print(f"\n❌ ERROR: No company data collected!")
        print(f"   All {len(COMPANIES)} companies failed.")
        print(f"   This might be a Yahoo Finance API issue or network problem.")
        print(f"\n💡 Suggestions:")
        print(f"   1. Check your internet connection")
        print(f"   2. Wait a few minutes and try again (rate limiting)")
        print(f"   3. Try with fewer companies first")
        raise ValueError("No company data collected - all companies failed")
    
    # Combine all company data
    df_companies = pd.concat(all_company_data, axis=0)
    
    print(f"\n✅ COMPANY DATA COMPLETE")
    print(f"   Companies: {successful}/25")
    if failed_companies:
        print(f"   ⚠️  Failed: {', '.join(failed_companies)}")
    print(f"   Total records: {len(df_companies)}")
    print(f"   Date range: {df_companies.index.min()} to {df_companies.index.max()}")
    
    # Save to cache
    save_intermediate(df_companies, 'companies')
    
    return df_companies

def merge_datasets(df_fred: pd.DataFrame, df_market: pd.DataFrame, 
                   df_companies: pd.DataFrame) -> pd.DataFrame:
    """Merge macro, market, and company data"""
    print_section("MERGING DATASETS")
    
    df_macro = df_fred.join(df_market, how='outer')
    df_companies = df_companies.reset_index().rename(columns={'index': 'Date'})
    df_final = df_companies.merge(df_macro, left_on='Date', right_index=True, how='left')
    
    print(f"   Merged shape: {df_final.shape}")
    print(f"   Date range: {df_final['Date'].min()} to {df_final['Date'].max()}")
    
    return df_final

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create engineered features"""
    print_section("ENGINEERING FEATURES")
    
    df = df.sort_values(['Company', 'Date'])
    
    for company in df['Company'].unique():
        mask = df['Company'] == company
        
        # Lag features
        df.loc[mask, 'GDP_Lag1'] = df.loc[mask, 'GDP_Growth'].shift(1)
        df.loc[mask, 'CPI_Lag1'] = df.loc[mask, 'CPI_Inflation'].shift(1)
        df.loc[mask, 'UNRATE_Lag1'] = df.loc[mask, 'Unemployment_Rate'].shift(1)
        df.loc[mask, 'VIX_Lag1'] = df.loc[mask, 'VIX'].shift(1)
        
        # Moving averages
        df.loc[mask, 'VIX_MA_2Q'] = df.loc[mask, 'VIX'].rolling(2).mean()
        df.loc[mask, 'SP500_MA_2Q'] = df.loc[mask, 'SP500_Return'].rolling(2).mean()
        df.loc[mask, 'Oil_MA_2Q'] = df.loc[mask, 'Oil_Price'].rolling(2).mean()
        
        # Growth rates (if Revenue/EPS exist)
        if 'Revenue' in df.columns:
            df.loc[mask, 'Revenue_Growth'] = df.loc[mask, 'Revenue'].pct_change() * 100
        if 'EPS' in df.columns:
            df.loc[mask, 'EPS_Growth'] = df.loc[mask, 'EPS'].pct_change() * 100
        if 'Total_Debt' in df.columns:
            df.loc[mask, 'Debt_Growth'] = df.loc[mask, 'Total_Debt'].pct_change() * 100
        
        # Financial ratios (if data exists)
        if 'Total_Debt' in df.columns and 'Total_Equity' in df.columns:
            df.loc[mask, 'Debt_to_Equity'] = df.loc[mask, 'Total_Debt'] / df.loc[mask, 'Total_Equity']
        if 'Current_Assets' in df.columns and 'Current_Liabilities' in df.columns:
            df.loc[mask, 'Current_Ratio'] = df.loc[mask, 'Current_Assets'] / df.loc[mask, 'Current_Liabilities']
        if 'Net_Income' in df.columns and 'Revenue' in df.columns:
            df.loc[mask, 'Profit_Margin'] = (df.loc[mask, 'Net_Income'] / df.loc[mask, 'Revenue']) * 100
        
        # Volatility
        if 'SP500_Return' in df.columns:
            df.loc[mask, 'Return_Volatility'] = df.loc[mask, 'SP500_Return'].rolling(4).std()
        if 'Debt_to_Equity' in df.columns:
            df.loc[mask, 'Debt_Ratio_Volatility'] = df.loc[mask, 'Debt_to_Equity'].rolling(4).std()
        
        # Targets
        if 'EPS' in df.columns:
            df.loc[mask, 'EPS_Next_Q'] = df.loc[mask, 'EPS'].shift(-1)
        if 'Revenue' in df.columns:
            df.loc[mask, 'Revenue_Next_Q'] = df.loc[mask, 'Revenue'].shift(-1)
        if 'Stock_Return' in df.columns:
            df.loc[mask, 'Return_Next_Q'] = df.loc[mask, 'Stock_Return'].shift(-1)
    
    # Interactions
    df['GDP_x_UNRATE'] = df['GDP_Growth'] * df['Unemployment_Rate']
    df['Inflation_x_Interest'] = df['CPI_Inflation'] * df['Federal_Funds_Rate']
    if 'Debt_to_Equity' in df.columns:
        df['VIX_x_Debt_Ratio'] = df['VIX'] * df['Debt_to_Equity']
    
    # Crisis dummies
    df['Crisis_2008'] = ((df['Date'].dt.year >= 2007) & (df['Date'].dt.year <= 2009)).astype(int)
    df['Crisis_2020'] = (df['Date'].dt.year == 2020).astype(int)
    df['Crisis_2022'] = ((df['Date'].dt.year >= 2022) & (df['Date'].dt.year <= 2023)).astype(int)
    
    print(f"   Engineered features created")
    print(f"   Final shape: {df.shape}")
    
    return df

def save_datasets(df: pd.DataFrame):
    """Save datasets"""
    print_section("SAVING DATASETS")
    
    processed_path = os.path.join(PROCESSED_DATA_DIR, 'merged_dataset.csv')
    df.to_csv(processed_path, index=False)
    print(f"   ✅ Saved: {processed_path}")
    print(f"      Size: {os.path.getsize(processed_path) / 1024:.1f} KB")
    
    summary_path = os.path.join(PROCESSED_DATA_DIR, 'dataset_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("FINANCIAL STRESS TEST GENERATOR - DATASET SUMMARY\n")
        f.write("="*70 + "\n\n")
        f.write(f"Collection Date: {datetime.now()}\n")
        f.write(f"Period: {START_DATE} to {END_DATE}\n\n")
        f.write(f"Dataset Shape: {df.shape}\n")
        f.write(f"Total Features: {df.shape[1]}\n")
        f.write(f"Total Observations: {df.shape[0]}\n")
        f.write(f"Companies: {df['Company'].nunique()}\n")
        f.write(f"Sectors: {df['Sector'].nunique()}\n\n")
        f.write(f"Date Range: {df['Date'].min()} to {df['Date'].max()}\n")
        f.write(f"Missing Values: {df.isnull().sum().sum()} ({df.isnull().sum().sum() / df.size * 100:.1f}%)\n\n")
        f.write("FEATURES:\n")
        for col in df.columns:
            f.write(f"  - {col}\n")
    
    print(f"   ✅ Saved: {summary_path}")

def main():
    """Main data collection pipeline"""
    start_time = time.time()
    
    print("\n" + "="*70)
    print("📊 FINANCIAL STRESS TEST GENERATOR - DATA COLLECTION")
    print("="*70)
    print(f"📅 Period: {START_DATE} to {END_DATE}")
    print(f"🏢 Companies: 25")
    print(f"📈 Sources: FRED + Yahoo Finance (HISTORICAL)")
    print(f"💾 Cache enabled: data/raw/")
    print("="*70)
    
    try:
        df_fred = fetch_fred_data()
        df_market = fetch_market_data()
        df_companies = fetch_all_companies()
        df_merged = merge_datasets(df_fred, df_market, df_companies)
        df_final = engineer_features(df_merged)
        save_datasets(df_final)
        
        elapsed = time.time() - start_time
        print_section("✅ DATA COLLECTION COMPLETE")
        print(f"   Final dataset shape: {df_final.shape}")
        print(f"   Total features: {df_final.shape[1]}")
        print(f"   Missing values: {df_final.isnull().sum().sum()} ({df_final.isnull().sum().sum() / df_final.size * 100:.1f}%)")
        print(f"   Date range: {df_final['Date'].min()} to {df_final['Date'].max()}")
        print(f"   Companies: {df_final['Company'].nunique()}")
        print(f"\n   ⏱️  Total execution time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
        print(f"\n   💾 Cached data saved in: data/raw/")
        print(f"   📊 Final dataset saved in: data/processed/")
        print(f"\n   🎉 SUCCESS! Dataset ready for modeling.")
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        raise

if __name__ == "__main__":
    main()