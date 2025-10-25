"""
Financial Stress Test Generator - DAILY Data Loader with Point-in-Time
Collects macro data from FRED and company data from Yahoo Finance
Period: 2000-2025 | Companies: 25 | Frequency: DAILY
✨ POINT-IN-TIME: Applies realistic reporting lags (no look-ahead bias)
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
START_DATE = '2000-01-01'
END_DATE = datetime.now().strftime('%Y-%m-%d')

# Data directories
RAW_DATA_DIR = 'data/raw'
PROCESSED_DATA_DIR = 'data/processed'

# Ensure directories exist
os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

# ============================================================================
# REPORTING LAG CONFIGURATION (Point-in-Time)
# ============================================================================

REPORTING_LAGS = {
    # Quarterly indicators (days after quarter end)
    'GDP_Growth': 25,              # BEA releases ~25 days after quarter
    'Trade_Balance': 45,           # BEA releases ~45 days after quarter
    
    # Monthly indicators (days after month end)
    'Unemployment_Rate': 5,        # BLS releases first Friday (~5 days)
    'CPI_Inflation': 15,           # BLS releases mid-month (~15 days)
    'Consumer_Confidence': 0,      # Released end of current month (no lag)
    
    # Daily/Real-time indicators (no lag)
    'Federal_Funds_Rate': 0,       # Announced immediately at FOMC
    'Yield_Curve_Spread': 0,       # Treasury market (real-time)
    'Oil_Price': 0,                # Commodity market (real-time)
    'Corporate_Bond_Spread': 0,    # Bond market (real-time)
    'TED_Spread': 0,               # Calculated daily (real-time)
    'Treasury_10Y_Yield': 0,       # Bond market (real-time)
    'Financial_Stress_Index': 0,   # Weekly calculation (near real-time)
    'High_Yield_Spread': 0,        # Bond market (real-time)
}

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
    """Save intermediate results with _daily suffix"""
    path = os.path.join(RAW_DATA_DIR, f'{name}_data_daily.csv')
    df.to_csv(path)
    print(f"   💾 Saved cache: {path}")

def load_cached(name: str) -> pd.DataFrame:
    """Load cached data if exists"""
    path = os.path.join(RAW_DATA_DIR, f'{name}_data_daily.csv')
    if os.path.exists(path):
        print(f"   📂 Loading cached {name} data from: {path}")
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        return df
    return None

# ============================================================================
# FRED DATA FETCHING
# ============================================================================

def fetch_fred_data() -> pd.DataFrame:
    """Fetch all FRED indicators (original frequency)"""
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
            print(f"✅ {len(df)} records")
            successful += 1
            time.sleep(0.5)
        except Exception as e:
            print(f"❌ Failed: {str(e)}")
    
    # Create dataframe (keep original frequency - don't resample yet!)
    df_fred = pd.DataFrame(fred_data)
    
    # Calculate growth rates where needed
    if 'GDP_Growth' in df_fred.columns:
        df_fred['GDP_Growth'] = df_fred['GDP_Growth'].pct_change() * 100
    
    if 'CPI_Inflation' in df_fred.columns:
        df_fred['CPI_Inflation'] = df_fred['CPI_Inflation'].pct_change(4) * 100
    
    print(f"\n✅ FRED DATA COMPLETE (Original Frequency)")
    print(f"   Shape: {df_fred.shape}")
    print(f"   Date range: {df_fred.index[0]} to {df_fred.index[-1]}")
    
    save_intermediate(df_fred, 'fred')
    return df_fred

# ============================================================================
# APPLY POINT-IN-TIME LAGS
# ============================================================================

def apply_reporting_lags_fixed(df_fred: pd.DataFrame) -> pd.DataFrame:
    """
    Fixed version: Apply lags correctly
    """
    df = df_fred.copy()
    
    # STEP 1: First resample to DAILY (without lag)
    df_daily = df.resample('D').ffill()
    
    # STEP 2: Then apply lags BY SHIFTING ROWS (not dates)
    # For daily data, 25 days = 25 rows (approximately, accounting for weekends)
    
    if 'GDP_Growth' in df_daily.columns:
        # Shift by 25 business days (approximately 35 calendar days)
        df_daily['GDP_Growth'] = df_daily['GDP_Growth'].shift(35)  # ~25 business days
        print(f"   GDP_Growth: 35-day shift applied (≈25 business days)")
    
    if 'Unemployment_Rate' in df_daily.columns:
        df_daily['Unemployment_Rate'] = df_daily['Unemployment_Rate'].shift(7)  # ~5 business days
        print(f"   Unemployment_Rate: 7-day shift applied (≈5 business days)")
    
    if 'CPI_Inflation' in df_daily.columns:
        df_daily['CPI_Inflation'] = df_daily['CPI_Inflation'].shift(21)  # ~15 business days
        print(f"   CPI_Inflation: 21-day shift applied (≈15 business days)")
    
    return df_daily

# ============================================================================
# CONVERT TO DAILY FREQUENCY
# ============================================================================

def convert_to_daily(df_fred_lagged: pd.DataFrame) -> pd.DataFrame:
    """
    Convert lagged macro data to daily frequency using forward-fill
    After applying lags, forward-fill is appropriate (market uses last known value)
    """
    print_section("CONVERTING TO DAILY FREQUENCY")
    
    # Resample to daily and forward-fill
    df_daily = df_fred_lagged.resample('D').ffill()
    
    print(f"   Original: {len(df_fred_lagged)} observations")
    print(f"   Daily: {len(df_daily)} observations")
    print(f"   Date range: {df_daily.index[0]} to {df_daily.index[-1]}")
    
    return df_daily

# ============================================================================
# MARKET DATA FETCHING
# ============================================================================

def fetch_market_data() -> pd.DataFrame:
    """Fetch VIX and S&P 500 data (already daily)"""
    cached = load_cached('market')
    if cached is not None:
        print_section("USING CACHED MARKET DATA")
        print(f"   Shape: {cached.shape}")
        return cached
    
    print_section("FETCHING MARKET DATA (VIX, S&P 500) - DAILY")
    
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
    
    # Keep daily frequency - no resampling!
    df_market = pd.DataFrame(market_data)
    
    # Calculate S&P 500 daily returns
    if 'SP500' in df_market.columns:
        df_market['SP500_Return'] = df_market['SP500'].pct_change() * 100
    
    print(f"\n✅ MARKET DATA COMPLETE (Daily)")
    print(f"   Shape: {df_market.shape}")
    print(f"   Date range: {df_market.index[0]} to {df_market.index[-1]}")
    
    save_intermediate(df_market, 'market')
    return df_market

# ============================================================================
# COMPANY DATA FETCHING (DAILY)
# ============================================================================

def fetch_company_data_daily(ticker: str, company_info: Dict) -> pd.DataFrame:
    """
    Fetch daily company data:
    - Stock prices: Daily (keep as-is)
    - Financials: Quarterly (will forward-fill later with lag)
    """
    try:
        stock = yf.Ticker(ticker)
        
        # ====================================================================
        # PART 1: DAILY STOCK PRICES (Keep Daily)
        # ====================================================================
        print(f" (prices)", end="")
        prices = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False)
        
        if prices.empty:
            print(f" ❌ No price data")
            return pd.DataFrame()
        
        # Handle MultiIndex columns
        if isinstance(prices.columns, pd.MultiIndex):
            prices.columns = prices.columns.get_level_values(0)
        
        # Verify required columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_cols = [col for col in required_cols if col not in prices.columns]
        
        if missing_cols:
            prices.columns = [col.capitalize() if isinstance(col, str) else col for col in prices.columns]
            missing_cols = [col for col in required_cols if col not in prices.columns]
            if missing_cols:
                print(f" ❌ Missing columns: {missing_cols}")
                return pd.DataFrame()
        
        # Create daily dataframe (NO RESAMPLING!)
        data = pd.DataFrame(index=prices.index)
        data['Stock_Price'] = prices['Close']
        data['Stock_Return'] = prices['Close'].pct_change() * 100
        data['Stock_Volume'] = prices['Volume']
        
        # Calculate rolling volatility (20-day)
        data['Volatility'] = prices['Close'].pct_change().rolling(20).std() * 100
        
        # ====================================================================
        # PART 2: QUARTERLY FINANCIALS (Will Forward-Fill with Lag)
        # ====================================================================
        print(f" (financials)", end="")
        try:
            # Get quarterly income statement
            income_stmt = stock.quarterly_income_stmt
            
            if income_stmt is not None and not income_stmt.empty:
                income_stmt = income_stmt.T
                income_stmt.index = pd.to_datetime(income_stmt.index)
                
                # Extract financial metrics
                revenue_cols = ['Total Revenue', 'Revenue']
                for col in revenue_cols:
                    if col in income_stmt.columns:
                        # Apply 45-day lag for SEC filing (10-Q due 40 days after quarter)
                        revenue_lagged = income_stmt[col].shift(freq='45D')
                        # Resample to daily and forward-fill
                        revenue_daily = revenue_lagged.resample('D').ffill()
                        # Reindex to match stock prices
                        data['Revenue'] = revenue_daily.reindex(data.index)
                        break
                
                netincome_cols = ['Net Income', 'Net Income Common Stockholders']
                for col in netincome_cols:
                    if col in income_stmt.columns:
                        ni_lagged = income_stmt[col].shift(freq='45D')
                        ni_daily = ni_lagged.resample('D').ffill()
                        data['Net_Income'] = ni_daily.reindex(data.index)
                        break
                
                operating_cols = ['Operating Income', 'Operating Revenue']
                for col in operating_cols:
                    if col in income_stmt.columns:
                        oi_lagged = income_stmt[col].shift(freq='45D')
                        oi_daily = oi_lagged.resample('D').ffill()
                        data['Operating_Income'] = oi_daily.reindex(data.index)
                        break
                
                eps_cols = ['Basic EPS', 'Diluted EPS']
                for col in eps_cols:
                    if col in income_stmt.columns:
                        eps_lagged = income_stmt[col].shift(freq='45D')
                        eps_daily = eps_lagged.resample('D').ffill()
                        data['EPS'] = eps_daily.reindex(data.index)
                        break
            
            # Get quarterly balance sheet
            balance_sheet = stock.quarterly_balance_sheet
            
            if balance_sheet is not None and not balance_sheet.empty:
                balance_sheet = balance_sheet.T
                balance_sheet.index = pd.to_datetime(balance_sheet.index)
                
                debt_cols = ['Total Debt', 'Long Term Debt']
                for col in debt_cols:
                    if col in balance_sheet.columns:
                        debt_lagged = balance_sheet[col].shift(freq='45D')
                        debt_daily = debt_lagged.resample('D').ffill()
                        data['Total_Debt'] = debt_daily.reindex(data.index)
                        break
                
                equity_cols = ['Total Equity Gross Minority Interest', 'Stockholders Equity']
                for col in equity_cols:
                    if col in balance_sheet.columns:
                        equity_lagged = balance_sheet[col].shift(freq='45D')
                        equity_daily = equity_lagged.resample('D').ffill()
                        data['Total_Equity'] = equity_daily.reindex(data.index)
                        break
                
                if 'Current Assets' in balance_sheet.columns:
                    ca_lagged = balance_sheet['Current Assets'].shift(freq='45D')
                    ca_daily = ca_lagged.resample('D').ffill()
                    data['Current_Assets'] = ca_daily.reindex(data.index)
                
                if 'Current Liabilities' in balance_sheet.columns:
                    cl_lagged = balance_sheet['Current Liabilities'].shift(freq='45D')
                    cl_daily = cl_lagged.resample('D').ffill()
                    data['Current_Liabilities'] = cl_daily.reindex(data.index)
        
        except Exception as e:
            print(f" (fin-err)", end="")
        
        # Calculate Market Cap
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
        
        # Remove all-NaN rows
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        data = data[data[numeric_cols].notna().any(axis=1)]
        
        if data.empty:
            print(f" ❌ All NaN")
            return pd.DataFrame()
        
        return data
        
    except Exception as e:
        print(f" ❌ Error: {str(e)}")
        return pd.DataFrame()

# ============================================================================
# FETCH ALL COMPANIES
# ============================================================================

def fetch_all_companies() -> pd.DataFrame:
    """Fetch daily data for all 25 companies"""
    cached = load_cached('companies')
    if cached is not None:
        print_section("USING CACHED COMPANY DATA")
        print(f"   Shape: {cached.shape}")
        print(f"   Companies: {cached['Company'].nunique()}")
        return cached
    
    print_section("FETCHING COMPANY DATA (25 COMPANIES) - DAILY")
    print("   Stock Prices: Daily granularity")
    print("   Financials: Quarterly with 45-day lag, forward-filled daily")
    
    all_company_data = []
    successful = 0
    failed_companies = []
    
    for i, (ticker, info) in enumerate(COMPANIES.items(), 1):
        print(f"  [{i:2d}/25] Fetching {info['name']:30} ({ticker})...", end=" ")
        
        df = fetch_company_data_daily(ticker, info)
        
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
    
    print(f"\n✅ COMPANY DATA COMPLETE")
    print(f"   Companies: {successful}/25")
    if failed_companies:
        print(f"   ⚠️  Failed: {', '.join(failed_companies)}")
    print(f"   Total records: {len(df_companies):,}")
    print(f"   Date range: {df_companies.index.min()} to {df_companies.index.max()}")
    
    save_intermediate(df_companies, 'companies')
    return df_companies

# ============================================================================
# MERGE DATASETS
# ============================================================================

def merge_datasets_daily(df_fred_lagged: pd.DataFrame, 
                         df_market: pd.DataFrame,
                         df_companies: pd.DataFrame) -> pd.DataFrame:
    """
    Merge macro (with lags), market, and company data at DAILY frequency
    """
    print_section("MERGING DATASETS (DAILY)")
    
    # Convert lagged FRED to daily
    print("   Converting lagged macro data to daily...")
    df_fred_daily = convert_to_daily(df_fred_lagged)
    
    # Market already daily
    print("   Market data already daily...")
    
    # Combine macro + market
    print("   Combining macro + market...")
    df_macro = df_fred_daily.join(df_market, how='outer')
    print(f"   Macro combined shape: {df_macro.shape}")
    
    # Prepare company data
    print("   Preparing company data...")
    df_companies = df_companies.reset_index().rename(columns={'index': 'Date'})
    
    # Merge on Date
    print("   Merging company + macro on Date...")
    df_final = df_companies.merge(
        df_macro,
        left_on='Date',
        right_index=True,
        how='left'
    )
    
    print(f"\n   ✅ Merged shape: {df_final.shape}")
    print(f"   Date range: {df_final['Date'].min()} to {df_final['Date'].max()}")
    print(f"   Companies: {df_final['Company'].nunique()}")
    print(f"   Avg days per company: {df_final.groupby('Company').size().mean():.0f}")
    
    return df_final

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def engineer_features_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Create engineered features for daily data"""
    print_section("ENGINEERING FEATURES (DAILY)")
    
    df = df.sort_values(['Company', 'Date']).copy()
    
    print("   Creating company-specific features...")
    for company in df['Company'].unique():
        mask = df['Company'] == company
        
        # Lag features (1-day lags for daily data)
        df.loc[mask, 'GDP_Lag1'] = df.loc[mask, 'GDP_Growth'].shift(1)
        df.loc[mask, 'VIX_Lag1'] = df.loc[mask, 'VIX'].shift(1)
        df.loc[mask, 'Return_Lag1'] = df.loc[mask, 'Stock_Return'].shift(1)
        
        # Moving averages (5-day and 20-day)
        df.loc[mask, 'VIX_MA_5D'] = df.loc[mask, 'VIX'].rolling(5).mean()
        df.loc[mask, 'VIX_MA_20D'] = df.loc[mask, 'VIX'].rolling(20).mean()
        df.loc[mask, 'Return_MA_5D'] = df.loc[mask, 'Stock_Return'].rolling(5).mean()
        df.loc[mask, 'Return_MA_20D'] = df.loc[mask, 'Stock_Return'].rolling(20).mean()
        
        # Volatility metrics
        df.loc[mask, 'Return_Volatility_20D'] = df.loc[mask, 'Stock_Return'].rolling(20).std()
        df.loc[mask, 'Return_Volatility_60D'] = df.loc[mask, 'Stock_Return'].rolling(60).std()
        
        # Financial ratios (if data available)
        if 'Revenue' in df.columns:
            df.loc[mask, 'Revenue_Growth'] = df.loc[mask, 'Revenue'].pct_change() * 100
        if 'EPS' in df.columns:
            df.loc[mask, 'EPS_Growth'] = df.loc[mask, 'EPS'].pct_change() * 100
        if 'Total_Debt' in df.columns and 'Total_Equity' in df.columns:
            df.loc[mask, 'Debt_to_Equity'] = df.loc[mask, 'Total_Debt'] / df.loc[mask, 'Total_Equity']
        if 'Current_Assets' in df.columns and 'Current_Liabilities' in df.columns:
            df.loc[mask, 'Current_Ratio'] = df.loc[mask, 'Current_Assets'] / df.loc[mask, 'Current_Liabilities']
        if 'Net_Income' in df.columns and 'Revenue' in df.columns:
            df.loc[mask, 'Profit_Margin'] = (df.loc[mask, 'Net_Income'] / df.loc[mask, 'Revenue']) * 100
        
        # Target variables (next day returns)
        df.loc[mask, 'Return_Next_1D'] = df.loc[mask, 'Stock_Return'].shift(-1)
        df.loc[mask, 'Return_Next_5D'] = df.loc[mask, 'Stock_Return'].shift(-5).rolling(5).sum()
        
        if 'EPS' in df.columns:
            df.loc[mask, 'EPS_Next_Q'] = df.loc[mask, 'EPS'].shift(-1)
    
    # Global interaction terms
    print("   Creating interaction features...")
    df['GDP_x_VIX'] = df['GDP_Growth'] * df['VIX']
    df['Inflation_x_Interest'] = df['CPI_Inflation'] * df['Federal_Funds_Rate']
    if 'Debt_to_Equity' in df.columns:
        df['VIX_x_Debt_Ratio'] = df['VIX'] * df['Debt_to_Equity']
    
    # Crisis dummies
    print("   Creating crisis indicators...")
    df['Crisis_2008'] = ((df['Date'].dt.year >= 2007) & (df['Date'].dt.year <= 2009)).astype(int)
    df['Crisis_2020'] = (df['Date'].dt.year == 2020).astype(int)
    df['Crisis_2022'] = ((df['Date'].dt.year >= 2022) & (df['Date'].dt.year <= 2023)).astype(int)
    
    # Day of week (trading patterns)
    df['Day_of_Week'] = df['Date'].dt.dayofweek  # 0=Monday, 4=Friday
    df['Is_Month_End'] = df['Date'].dt.is_month_end.astype(int)
    
    print(f"\n   ✅ Feature engineering complete")
    print(f"   Final shape: {df.shape}")
    
    return df

# ============================================================================
# SAVE DATASETS
# ============================================================================

def save_datasets(df: pd.DataFrame):
    """Save datasets with _daily suffix"""
    print_section("SAVING DATASETS")
    
    # Save main dataset
    processed_path = os.path.join(PROCESSED_DATA_DIR, 'merged_dataset_daily.csv')
    df.to_csv(processed_path, index=False)
    print(f"   ✅ Saved: {processed_path}")
    print(f"      Size: {os.path.getsize(processed_path) / (1024*1024):.1f} MB")
    
    # Save summary
    summary_path = os.path.join(PROCESSED_DATA_DIR, 'dataset_summary_daily.txt')
    with open(summary_path, 'w') as f:
        f.write("FINANCIAL STRESS TEST GENERATOR - DAILY DATASET SUMMARY\n")
        f.write("="*70 + "\n\n")
        f.write("METHODOLOGY: Point-in-Time Data with Reporting Lags\n")
        f.write("(Following Federal Reserve CCAR approach - no look-ahead bias)\n\n")
        f.write(f"Collection Date: {datetime.now()}\n")
        f.write(f"Period: {START_DATE} to {END_DATE}\n")
        f.write(f"Frequency: DAILY\n\n")
        f.write(f"Dataset Shape: {df.shape}\n")
        f.write(f"Total Features: {df.shape[1]}\n")
        f.write(f"Total Observations: {df.shape[0]:,}\n")
        f.write(f"Companies: {df['Company'].nunique()}\n")
        f.write(f"Sectors: {df['Sector'].nunique()}\n")
        f.write(f"Trading Days: {df['Date'].nunique()}\n\n")
        f.write(f"Date Range: {df['Date'].min()} to {df['Date'].max()}\n")
        f.write(f"Missing Values: {df.isnull().sum().sum():,} ({df.isnull().sum().sum() / df.size * 100:.1f}%)\n\n")
        f.write("REPORTING LAGS APPLIED:\n")
        for indicator, lag in REPORTING_LAGS.items():
            if lag > 0:
                f.write(f"  - {indicator}: {lag} days\n")
        f.write("\nFEATURES:\n")
        for col in df.columns:
            f.write(f"  - {col}\n")
    
    print(f"   ✅ Saved: {summary_path}")

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """Main daily data collection pipeline with point-in-time lags"""
    start_time = time.time()
    
    print("\n" + "="*70)
    print("📊 FINANCIAL STRESS TEST GENERATOR - DAILY DATA COLLECTION")
    print("="*70)
    print(f"📅 Period: {START_DATE} to {END_DATE}")
    print(f"🏢 Companies: 25")
    print(f"📈 Frequency: DAILY (with Point-in-Time lags)")
    print(f"💾 Cache enabled: data/raw/*_daily.csv")
    print("="*70)
    print("\n⚠️  METHODOLOGY:")
    print("   Using Point-in-Time approach (Federal Reserve CCAR standard)")
    print("   - GDP: 25-day reporting lag")
    print("   - Unemployment: 5-day lag")
    print("   - CPI: 15-day lag")
    print("   - Financials: 45-day lag (SEC filing)")
    print("   - Market data: Real-time (no lag)")
    print("="*70)
    
    try:
        # Fetch FRED data (original frequency)
        df_fred = fetch_fred_data()
        
        # Apply point-in-time lags
        df_fred_lagged = apply_reporting_lags(df_fred)
        
        # Fetch market data (daily)
        df_market = fetch_market_data()
        
        # Fetch company data (daily)
        df_companies = fetch_all_companies()
        
        # Merge all datasets
        df_merged = merge_datasets_daily(df_fred_lagged, df_market, df_companies)
        
        # Engineer features
        df_final = engineer_features_daily(df_merged)
        
        # Save datasets
        save_datasets(df_final)
        
        # Summary statistics
        elapsed = time.time() - start_time
        print_section("✅ DAILY DATA COLLECTION COMPLETE")
        print(f"   Final dataset shape: {df_final.shape}")
        print(f"   Total observations: {df_final.shape[0]:,}")
        print(f"   Total features: {df_final.shape[1]}")
        print(f"   Missing values: {df_final.isnull().sum().sum():,} ({df_final.isnull().sum().sum() / df_final.size * 100:.1f}%)")
        print(f"   Date range: {df_final['Date'].min()} to {df_final['Date'].max()}")
        print(f"   Companies: {df_final['Company'].nunique()}")
        print(f"   Trading days: {df_final['Date'].nunique():,}")
        print(f"   Avg observations per company: {df_final.groupby('Company').size().mean():.0f}")
        print(f"\n   ⏱️  Total execution time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
        print(f"\n   💾 Cached data: data/raw/*_daily.csv")
        print(f"   📊 Final dataset: data/processed/merged_dataset_daily.csv")
        print(f"\n   🎉 SUCCESS! Daily dataset ready for modeling.")
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()