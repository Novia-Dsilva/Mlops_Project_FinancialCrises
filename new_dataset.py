"""
Fetch Company Financials from Alpha Vantage (2025 Fixed Version)
+ Automatic fallback to Financial Modeling Prep (FMP)
"""
 
import pandas as pd
import requests
import time
from pathlib import Path
 
# ============================================================================
# CONFIGURATION
# ============================================================================
 
API_KEYS = [
    'J92KY8K2F85826MO',
    'UUHCSDJMGPBLH5EB',
    'QXSKA19IF0N1XHRD',
    'QS7HJPQ0OM7YBP53','LVRR2S4WWFBKIFJF'
]
current_key_index = 0
 
def get_api_key():
    global current_key_index
    return API_KEYS[current_key_index % len(API_KEYS)]
 
def switch_api_key():
    global current_key_index
    current_key_index += 1
    print(f"   🔄 Switched to API key #{current_key_index + 1}")
 
RAW_DIR = Path('data/raw')
RAW_DIR.mkdir(parents=True, exist_ok=True)
 
DELAY_BETWEEN_CALLS = 20
MAX_RETRIES = 3
 
COMPANIES = {    'JPM': {'name': 'JPMorgan Chase', 'sector': 'Financials'},    'BAC': {'name': 'Bank of America', 'sector': 'Financials'},    'C': {'name': 'Citigroup', 'sector': 'Financials'},    'GS': {'name': 'Goldman Sachs', 'sector': 'Financials'},    'WFC': {'name': 'Wells Fargo', 'sector': 'Financials'},    'AAPL': {'name': 'Apple', 'sector': 'Technology'},    'MSFT': {'name': 'Microsoft', 'sector': 'Technology'},    'GOOGL': {'name': 'Alphabet', 'sector': 'Technology'},    'AMZN': {'name': 'Amazon', 'sector': 'Technology'},    'NVDA': {'name': 'NVIDIA', 'sector': 'Technology'},    'DIS': {'name': 'Disney', 'sector': 'Communication Services'},    'NFLX': {'name': 'Netflix', 'sector': 'Communication Services'},    'TSLA': {'name': 'Tesla', 'sector': 'Consumer Discretionary'},    'HD': {'name': 'Home Depot', 'sector': 'Consumer Discretionary'},    'MCD': {'name': 'McDonalds', 'sector': 'Consumer Discretionary'},    'WMT': {'name': 'Walmart', 'sector': 'Consumer Staples'},    'PG': {'name': 'Procter & Gamble', 'sector': 'Consumer Staples'},    'COST': {'name': 'Costco', 'sector': 'Consumer Staples'},    'XOM': {'name': 'ExxonMobil', 'sector': 'Energy'},    'CVX': {'name': 'Chevron', 'sector': 'Energy'},    'UNH': {'name': 'UnitedHealth', 'sector': 'Healthcare'},    'JNJ': {'name': 'Johnson & Johnson', 'sector': 'Healthcare'},    'BA': {'name': 'Boeing', 'sector': 'Industrials'},    'CAT': {'name': 'Caterpillar', 'sector': 'Industrials'},    'LIN': {'name': 'Linde', 'sector': 'Materials'}}
 
# ============================================================================
# FETCHERS
# ============================================================================
 
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
 
        # Debug: print unexpected empty responses
        if not data:
            print(f"   ⚠️  Empty response for {ticker}")
            return None
 
        if 'Note' in data:
            print(f"   ⚠️  Rate limit hit on key #{current_key_index + 1}, rotating...")
            switch_api_key()
            time.sleep(5)
            return fetch_alpha_vantage(ticker, function, retry_count)
 
        if 'Error Message' in data or 'Information' in data:
            print(f"   ⚠️  API error: {data.get('Error Message') or data.get('Information')}")
            return None
 
        if 'quarterlyReports' not in data:
            print(f"   ⚠️  No quarterlyReports field in response → AlphaVantage may not support {ticker}")
            print("   🔍 Raw response:", str(data)[:200])
            return None
 
        return data['quarterlyReports']
 
    except requests.exceptions.Timeout:
        if retry_count < MAX_RETRIES:
            print(f"   ⚠️  Timeout (retry {retry_count + 1}/{MAX_RETRIES}), waiting 30s...")
            time.sleep(30)
            return fetch_alpha_vantage(ticker, function, retry_count + 1)
        print(f"   ❌ Timeout after {MAX_RETRIES} attempts")
        return None
    except Exception as e:
        print(f"   ❌ Exception: {e}")
        return None
 
 
def fetch_fmp(ticker, endpoint="income-statement", limit=5):
    """Fallback: fetch from Financial Modeling Prep"""
    api_key = "demo"  # Replace with your FMP key if available
    url = f"https://financialmodelingprep.com/api/v3/{endpoint}/{ticker}?period=quarter&limit={limit}&apikey={api_key}"
    try:
        r = requests.get(url, timeout=30)
        data = r.json()
        if isinstance(data, list) and len(data) > 0:
            return data
        return None
    except Exception as e:
        print(f"   ⚠️  FMP fallback failed: {e}")
        return None
 
# ============================================================================
# DATA PARSERS
# ============================================================================
 
def parse_income(data):
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
    for col in ['Total_Assets', 'Total_Liabilities', 'Total_Equity', 'Current_Assets', 'Current_Liabilities', 'Long_Term_Debt', 'Short_Term_Debt', 'Cash']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['Debt_to_Equity'] = df['Total_Liabilities'] / df['Total_Equity'].replace(0, 1)
    df['Current_Ratio'] = df['Current_Assets'] / df['Current_Liabilities'].replace(0, 1)
    return df
 
# ============================================================================
# MAIN
# ============================================================================
 
def main():
    cache_file = RAW_DIR / 'financials_cache.txt'
    if cache_file.exists():
        cached = set(cache_file.read_text().split(','))
    else:
        cached = set()
 
    print("="*70)
    print("📊 FETCHING COMPANY FINANCIALS (2025 FIXED VERSION)")
    print("="*70)
    print(f"Companies: {len(COMPANIES)} | API Keys: {len(API_KEYS)} | Delay: {DELAY_BETWEEN_CALLS}s\n")
 
    all_income, all_balance, failed = [], [], []
    start = time.time()
 
    for i, (ticker, info) in enumerate(COMPANIES.items(), 1):
        if ticker in cached:
            print(f"[{i:2d}/{len(COMPANIES)}] {ticker:6} {info['name']:25} ⏭️ Cached")
            continue
 
        print(f"\n[{i:2d}/{len(COMPANIES)}] {ticker:6} {info['name']:25}")
        print("   📄 Income...", end=" ", flush=True)
        income_data = fetch_alpha_vantage(ticker, 'INCOME_STATEMENT')
 
        if not income_data:
            print("⚠️  No data from Alpha → trying FMP")
            income_data = fetch_fmp(ticker, 'income-statement')
 
        if not income_data:
            print("❌ Failed to fetch income statement")
            failed.append(ticker)
            continue
 
        df_income = parse_income(income_data)
        df_income['Company'], df_income['Sector'] = ticker, info['sector']
        all_income.append(df_income)
        print(f"✅ {len(df_income)}Q")
 
        time.sleep(DELAY_BETWEEN_CALLS)
 
        print("   📄 Balance...", end=" ", flush=True)
        balance_data = fetch_alpha_vantage(ticker, 'BALANCE_SHEET')
        if not balance_data:
            print("⚠️  No data from Alpha → trying FMP")
            balance_data = fetch_fmp(ticker, 'balance-sheet-statement')
 
        if balance_data:
            df_balance = parse_balance(balance_data)
            df_balance['Company'], df_balance['Sector'] = ticker, info['sector']
            all_balance.append(df_balance)
            print(f"✅ {len(df_balance)}Q")
        else:
            print("⚠️  Skipped balance sheet")
 
        cached.add(ticker)
        cache_file.write_text(','.join(cached))
 
        time.sleep(DELAY_BETWEEN_CALLS)
 
    # Save results
    print("\n" + "="*70)
    print("✅ FETCH COMPLETE")
    print("="*70)
    print(f"Elapsed: {((time.time()-start)/60):.1f} min | Success: {len(all_income)}/{len(COMPANIES)}")
 
    if all_income:
        df_all = pd.concat(all_income, ignore_index=True)
        df_all.to_csv(RAW_DIR / 'company_income_raw.csv', index=False)
        print(f"💾 Saved income statements: {len(df_all)} rows")
 
    if all_balance:
        df_bal = pd.concat(all_balance, ignore_index=True)
        df_bal.to_csv(RAW_DIR / 'company_balance_raw.csv', index=False)
        print(f"💾 Saved balance sheets: {len(df_bal)} rows")
 
    if failed:
        print(f"\n⚠️ Failed tickers: {failed}")
 
    print("="*70)
    print("🎯 Done!")
 
if __name__ == "__main__":
    main()