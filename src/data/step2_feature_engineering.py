"""
STEP 2: FEATURE ENGINEERING + QUARTERLY TO DAILY CONVERSION

This script engineers features from cleaned data and converts quarterly 
company financials to daily frequency.

Pipeline:
1. Load cleaned data from data/clean/
2. Engineer features for each dataset separately
3. Convert quarterly company financials → daily (forward fill with PIT)
4. Save feature-engineered datasets to data/features/

Key Features:
- Time-based features (lags, moving averages, volatility)
- Financial ratios (profit margin, ROE, ROA)
- Growth rates (QoQ, YoY)
- Technical indicators (RSI, MACD for stocks)
- Quarterly → Daily conversion preserving PIT correctness

Input:  data/clean/*.csv (5 files)
Output: data/features/*.csv (3 files)
    - fred_features.csv (daily macro features)
    - market_features.csv (daily market features)
    - company_features.csv (daily company features - prices + financials)

Usage:
    python step2_feature_engineering.py

Next Step:
    python step3_data_merging.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, Tuple
import time
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Engineer features for each dataset before merging."""

    def __init__(self, clean_dir: str = "data/clean", features_dir: str = "data/features"):
        self.clean_dir = Path(clean_dir)
        self.features_dir = Path(features_dir)
        self.features_dir.mkdir(parents=True, exist_ok=True)

    # ========== LOAD CLEANED DATA ==========

    def load_cleaned_data(self) -> Dict[str, pd.DataFrame]:
        """Load all cleaned datasets."""
        logger.info("="*80)
        logger.info("LOADING CLEANED DATASETS")
        logger.info("="*80)

        data = {}

        # Load FRED
        fred_path = self.clean_dir / 'fred_clean.csv'
        if fred_path.exists():
            data['fred'] = pd.read_csv(fred_path, parse_dates=['Date'])
            logger.info(f"  ✓ FRED: {data['fred'].shape}")
        else:
            logger.warning(f"  ⚠️  fred_clean.csv not found")

        # Load Market
        market_path = self.clean_dir / 'market_clean.csv'
        if market_path.exists():
            data['market'] = pd.read_csv(market_path, parse_dates=['Date'])
            logger.info(f"  ✓ Market: {data['market'].shape}")
        else:
            logger.warning(f"  ⚠️  market_clean.csv not found")

        # Load Prices
        prices_path = self.clean_dir / 'company_prices_clean.csv'
        if prices_path.exists():
            data['prices'] = pd.read_csv(prices_path, parse_dates=['Date'])
            logger.info(f"  ✓ Prices: {data['prices'].shape}")
        else:
            logger.warning(f"  ⚠️  company_prices_clean.csv not found")

        # Load Balance
        balance_path = self.clean_dir / 'company_balance_clean.csv'
        if balance_path.exists():
            data['balance'] = pd.read_csv(balance_path, parse_dates=['Date'])
            logger.info(f"  ✓ Balance: {data['balance'].shape}")
        else:
            logger.warning(f"  ⚠️  company_balance_clean.csv not found")

        # Load Income
        income_path = self.clean_dir / 'company_income_clean.csv'
        if income_path.exists():
            data['income'] = pd.read_csv(income_path, parse_dates=['Date'])
            logger.info(f"  ✓ Income: {data['income'].shape}")
        else:
            logger.warning(f"  ⚠️  company_income_clean.csv not found")

        return data

    # ========== ENGINEER FRED FEATURES ==========

    def engineer_fred_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer features from FRED macroeconomic data.

        Features created:
        - Lagged variables (1, 5, 22 days)
        - Growth rates (quarterly pct change)
        - Moving averages (30, 90 days)
        - Volatility measures (rolling std)
        - Inflation (CPI pct change)
        """
        logger.info("\n" + "="*80)
        logger.info("ENGINEERING FRED FEATURES")
        logger.info("="*80)

        df = df.copy()
        df.sort_values('Date', inplace=True)
        
        original_cols = len(df.columns)

        # Define feature groups
        macro_indicators = ['GDP', 'CPI', 'Unemployment_Rate', 'Federal_Funds_Rate',
                           'Yield_Curve_Spread', 'Oil_Price', 'Consumer_Confidence']

        # Filter to existing columns
        macro_indicators = [col for col in macro_indicators if col in df.columns]

        logger.info(f"\n1. Creating lagged features (1d, 5d, 22d)...")
        # Lags: 1 day, 1 week, 1 month
        for col in macro_indicators:
            df[f'{col}_Lag1'] = df[col].shift(1)      # Yesterday
            df[f'{col}_Lag5'] = df[col].shift(5)      # ~1 week
            df[f'{col}_Lag22'] = df[col].shift(22)    # ~1 month

        logger.info(f"2. Creating growth rates...")
        # GDP growth (quarterly = ~90 trading days)
        if 'GDP' in df.columns:
            df['GDP_Growth_90D'] = df['GDP'].pct_change(periods=90) * 100  # Percentage
            df['GDP_Growth_252D'] = df['GDP'].pct_change(periods=252) * 100  # YoY

        # Inflation (CPI pct change)
        if 'CPI' in df.columns:
            df['Inflation'] = df['CPI'].pct_change(periods=12) * 100  # YoY inflation
            df['Inflation_MA3M'] = df['Inflation'].rolling(window=90, min_periods=1).mean()

        logger.info(f"3. Creating moving averages...")
        # Moving averages
        for col in ['Unemployment_Rate', 'Federal_Funds_Rate', 'Oil_Price']:
            if col in df.columns:
                df[f'{col}_MA30'] = df[col].rolling(window=30, min_periods=1).mean()
                df[f'{col}_MA90'] = df[col].rolling(window=90, min_periods=1).mean()

        logger.info(f"4. Creating volatility measures...")
        # Volatility (rolling standard deviation)
        for col in ['Oil_Price', 'Unemployment_Rate', 'Federal_Funds_Rate']:
            if col in df.columns:
                df[f'{col}_Volatility_30D'] = df[col].rolling(window=30, min_periods=1).std()

        logger.info(f"5. Creating economic stress indicators...")
        # Economic stress indicators
        if 'Yield_Curve_Spread' in df.columns:
            df['Yield_Curve_Inverted'] = (df['Yield_Curve_Spread'] < 0).astype(int)

        if 'TED_Spread' in df.columns:
            df['TED_Spread_High'] = (df['TED_Spread'] > df['TED_Spread'].quantile(0.75)).astype(int)

        new_cols = len(df.columns)
        logger.info(f"\n✓ FRED features engineered: {df.shape}")
        logger.info(f"  Original columns: {original_cols}")
        logger.info(f"  New columns: {new_cols}")
        logger.info(f"  Features added: {new_cols - original_cols}")

        return df

    # ========== ENGINEER MARKET FEATURES ==========

    def engineer_market_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer features from market data (VIX, S&P500).

        Features created:
        - Returns (daily, weekly, monthly)
        - Volatility measures
        - Moving averages
        - Momentum indicators
        - Technical indicators (RSI)
        """
        logger.info("\n" + "="*80)
        logger.info("ENGINEERING MARKET FEATURES")
        logger.info("="*80)

        df = df.copy()
        df.sort_values('Date', inplace=True)
        
        original_cols = len(df.columns)

        # === VIX FEATURES ===
        logger.info(f"\n1. Creating VIX features...")
        if 'VIX' in df.columns:
            # Lags
            df['VIX_Lag1'] = df['VIX'].shift(1)
            df['VIX_Lag5'] = df['VIX'].shift(5)
            df['VIX_Lag22'] = df['VIX'].shift(22)
            
            # Moving averages
            df['VIX_MA5'] = df['VIX'].rolling(window=5, min_periods=1).mean()
            df['VIX_MA22'] = df['VIX'].rolling(window=22, min_periods=1).mean()
            df['VIX_MA90'] = df['VIX'].rolling(window=90, min_periods=1).mean()
            
            # Volatility of volatility
            df['VIX_Std22'] = df['VIX'].rolling(window=22, min_periods=1).std()

            # VIX regime (low/medium/high volatility)
            df['VIX_Regime'] = pd.cut(df['VIX'], bins=[0, 15, 25, 100],
                                     labels=['Low', 'Medium', 'High'])

        # === S&P 500 FEATURES ===
        logger.info(f"2. Creating S&P500 features...")
        sp500_col = 'SP500_Close' if 'SP500_Close' in df.columns else 'SP500'
        
        if sp500_col in df.columns:
            # Returns (different time horizons)
            df['SP500_Return_1D'] = df[sp500_col].pct_change(periods=1) * 100
            df['SP500_Return_5D'] = df[sp500_col].pct_change(periods=5) * 100
            df['SP500_Return_22D'] = df[sp500_col].pct_change(periods=22) * 100
            df['SP500_Return_90D'] = df[sp500_col].pct_change(periods=90) * 100

            # Moving averages (trend indicators)
            df['SP500_MA50'] = df[sp500_col].rolling(window=50, min_periods=1).mean()
            df['SP500_MA200'] = df[sp500_col].rolling(window=200, min_periods=1).mean()

            # Price relative to moving average (momentum)
            df['SP500_vs_MA50'] = df[sp500_col] / df['SP500_MA50']
            df['SP500_vs_MA200'] = df[sp500_col] / df['SP500_MA200']

            # Volatility (annualized)
            df['SP500_Volatility_22D'] = df['SP500_Return_1D'].rolling(window=22, min_periods=1).std() * np.sqrt(252)
            df['SP500_Volatility_90D'] = df['SP500_Return_1D'].rolling(window=90, min_periods=1).std() * np.sqrt(252)

        logger.info(f"3. Creating momentum indicators...")
        # Momentum (Rate of change)
        if sp500_col in df.columns:
            df['SP500_Momentum_22D'] = df[sp500_col].pct_change(periods=22) * 100
            df['SP500_Momentum_90D'] = df[sp500_col].pct_change(periods=90) * 100

        logger.info(f"4. Creating RSI (Relative Strength Index)...")
        # RSI (14-day)
        if sp500_col in df.columns:
            delta = df[sp500_col].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
            rs = gain / loss.replace(0, np.nan)
            df['SP500_RSI_14D'] = 100 - (100 / (1 + rs))

        new_cols = len(df.columns)
        logger.info(f"\n✓ Market features engineered: {df.shape}")
        logger.info(f"  Original columns: {original_cols}")
        logger.info(f"  Features added: {new_cols - original_cols}")

        return df

    # ========== CONVERT QUARTERLY TO DAILY ==========

    def quarterly_to_daily(self, df: pd.DataFrame, company_col: str = 'Company') -> pd.DataFrame:
        """
        Convert quarterly financial data to daily using forward fill.

        CRITICAL: This preserves point-in-time correctness because quarterly dates
        were already shifted by +45 days in Step 1 (cleaning).

        Logic:
        - Q1 data (available 5/15 after PIT shift) applies to all days 5/15 → 8/14
        - Q2 data (available 8/15) takes over on 8/15
        
        Args:
            df: Quarterly DataFrame
            company_col: Column name for company grouping
            
        Returns:
            Daily DataFrame
        """
        logger.info("\n" + "="*80)
        logger.info("CONVERTING QUARTERLY → DAILY")
        logger.info("="*80)
        logger.info("Method: Forward fill (each quarter's values persist until next quarter)")
        logger.info("Point-in-Time: Already ensured by 45-day shift in Step 1 ✓")

        df = df.copy()
        df.sort_values([company_col, 'Date'], inplace=True)

        # Get date range
        start_date = df['Date'].min()
        end_date = df['Date'].max()

        logger.info(f"\nOriginal quarterly data:")
        logger.info(f"  Date range: {start_date.date()} to {end_date.date()}")
        logger.info(f"  Total rows: {len(df):,}")
        logger.info(f"  Companies: {df[company_col].nunique()}")

        # Create daily date range
        daily_dates = pd.date_range(start=start_date, end=end_date, freq='D')
        logger.info(f"\nExpanding to daily:")
        logger.info(f"  Daily dates: {len(daily_dates):,}")

        # Process each company separately
        daily_dfs = []

        for i, company in enumerate(df[company_col].unique(), 1):
            company_df = df[df[company_col] == company].copy()

            # Set date as index for reindexing
            company_df.set_index('Date', inplace=True)

            # Reindex to daily (creates NaN for non-quarter dates)
            company_daily = company_df.reindex(daily_dates)

            # Forward fill all columns (quarterly values persist)
            company_daily = company_daily.ffill()

            # Fill metadata columns (Company, Sector, Company_Name)
            company_daily[company_col] = company

            # Get sector from original data
            if 'Sector' in company_df.columns:
                sector = company_df['Sector'].iloc[0] if not company_df['Sector'].isna().all() else 'Unknown'
                company_daily['Sector'] = sector

            # Get company name if exists
            if 'Company_Name' in company_df.columns:
                company_name = company_df['Company_Name'].iloc[0] if not company_df['Company_Name'].isna().all() else company
                company_daily['Company_Name'] = company_name

            company_daily.reset_index(inplace=True)
            company_daily.rename(columns={'index': 'Date'}, inplace=True)

            daily_dfs.append(company_daily)

            if i <= 5 or i % 5 == 0:  # Log first 5 and every 5th
                logger.info(f"  [{i:2d}/{df[company_col].nunique()}] {company}: {len(company_df)} quarters → {len(company_daily):,} days")

        # Combine all companies
        result = pd.concat(daily_dfs, ignore_index=True)
        result.sort_values([company_col, 'Date'], inplace=True)

        logger.info(f"\n✓ Conversion complete: {result.shape}")
        logger.info(f"  Total rows: {len(result):,}")
        logger.info(f"  Rows per company: ~{len(result) / result[company_col].nunique():.0f}")

        return result

    # ========== ENGINEER COMPANY FEATURES ==========

    def engineer_company_features(self, prices_df: pd.DataFrame,
                                  financials_daily_df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer company-specific features after converting to daily.

        Features created:
        - Stock returns and volatility
        - Financial ratios (Profit Margin, ROE, ROA)
        - Growth rates (QoQ, YoY)
        - Leverage and liquidity metrics
        - Technical indicators
        
        Args:
            prices_df: Daily stock prices
            financials_daily_df: Daily financial data (converted from quarterly)
            
        Returns:
            DataFrame with all company features
        """
        logger.info("\n" + "="*80)
        logger.info("ENGINEERING COMPANY FEATURES")
        logger.info("="*80)

        # First merge prices + financials (both are now daily)
        logger.info("\n1. Merging prices + financials (both daily)...")
        
        # Common columns to merge on
        merge_cols = ['Date', 'Company']
        if 'Sector' in prices_df.columns and 'Sector' in financials_daily_df.columns:
            merge_cols.append('Sector')
        
        company_full = pd.merge(
            prices_df,
            financials_daily_df,
            on=merge_cols,
            how='outer',
            suffixes=('', '_fin')
        )

        # Drop duplicate columns from suffix
        dup_cols = [col for col in company_full.columns if col.endswith('_fin')]
        if dup_cols:
            company_full.drop(columns=dup_cols, inplace=True)

        company_full.sort_values(['Company', 'Date'], inplace=True)
        logger.info(f"  Merged shape: {company_full.shape}")

        df = company_full.copy()

        # === STOCK PRICE FEATURES ===
        logger.info(f"\n2. Creating stock price features...")
        
        stock_col = 'Stock_Price' if 'Stock_Price' in df.columns else 'Close'
        
        if stock_col in df.columns:
            # Returns (different horizons)
            df['Stock_Return_1D'] = df.groupby('Company')[stock_col].pct_change(periods=1) * 100
            df['Stock_Return_5D'] = df.groupby('Company')[stock_col].pct_change(periods=5) * 100
            df['Stock_Return_22D'] = df.groupby('Company')[stock_col].pct_change(periods=22) * 100
            df['Stock_Return_90D'] = df.groupby('Company')[stock_col].pct_change(periods=90) * 100

            # Cumulative returns
            df['Stock_Return_YTD'] = df.groupby('Company')[stock_col].pct_change(periods=252) * 100

            # Volatility (annualized)
            df['Stock_Volatility_22D'] = df.groupby('Company')['Stock_Return_1D'].rolling(22, min_periods=1).std().reset_index(0, drop=True) * np.sqrt(252)
            df['Stock_Volatility_90D'] = df.groupby('Company')['Stock_Return_1D'].rolling(90, min_periods=1).std().reset_index(0, drop=True) * np.sqrt(252)

            # Moving averages
            df['Stock_MA20'] = df.groupby('Company')[stock_col].rolling(20, min_periods=1).mean().reset_index(0, drop=True)
            df['Stock_MA50'] = df.groupby('Company')[stock_col].rolling(50, min_periods=1).mean().reset_index(0, drop=True)
            df['Stock_MA200'] = df.groupby('Company')[stock_col].rolling(200, min_periods=1).mean().reset_index(0, drop=True)

            # Price vs MA (momentum indicators)
            df['Stock_vs_MA50'] = df[stock_col] / df['Stock_MA50']
            df['Stock_vs_MA200'] = df[stock_col] / df['Stock_MA200']

            # RSI (14-day)
            logger.info(f"3. Creating technical indicators (RSI, MACD)...")
            for company in df['Company'].unique():
                company_mask = df['Company'] == company
                company_prices = df.loc[company_mask, stock_col]
                
                # RSI
                delta = company_prices.diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
                rs = gain / loss.replace(0, np.nan)
                df.loc[company_mask, 'Stock_RSI_14D'] = 100 - (100 / (1 + rs))
                
                # MACD (12-26-9)
                ema12 = company_prices.ewm(span=12, adjust=False).mean()
                ema26 = company_prices.ewm(span=26, adjust=False).mean()
                df.loc[company_mask, 'Stock_MACD'] = ema12 - ema26
                df.loc[company_mask, 'Stock_MACD_Signal'] = df.loc[company_mask, 'Stock_MACD'].ewm(span=9, adjust=False).mean()

        # === VOLUME FEATURES ===
        if 'Volume' in df.columns:
            df['Volume_MA20'] = df.groupby('Company')['Volume'].rolling(20, min_periods=1).mean().reset_index(0, drop=True)
            df['Volume_vs_MA20'] = df['Volume'] / df['Volume_MA20']

        # === FINANCIAL STATEMENT FEATURES ===
        logger.info(f"\n4. Creating financial ratios...")

        # Profitability ratios
        if 'Net_Income' in df.columns and 'Revenue' in df.columns:
            df['Profit_Margin'] = (df['Net_Income'] / df['Revenue']) * 100
            df['Profit_Margin'] = df['Profit_Margin'].replace([np.inf, -np.inf], np.nan)

        if 'Net_Income' in df.columns and 'Total_Assets' in df.columns:
            df['ROA'] = (df['Net_Income'] / df['Total_Assets']) * 100  # Return on Assets
            df['ROA'] = df['ROA'].replace([np.inf, -np.inf], np.nan)

        if 'Net_Income' in df.columns and 'Total_Equity' in df.columns:
            df['ROE'] = (df['Net_Income'] / df['Total_Equity']) * 100  # Return on Equity
            df['ROE'] = df['ROE'].replace([np.inf, -np.inf], np.nan)

        # Leverage ratios
        if 'Total_Debt' in df.columns and 'Total_Assets' in df.columns:
            df['Debt_to_Assets'] = (df['Total_Debt'] / df['Total_Assets']) * 100
            df['Debt_to_Assets'] = df['Debt_to_Assets'].replace([np.inf, -np.inf], np.nan)

        # Liquidity ratios
        if 'Cash' in df.columns and 'Current_Liabilities' in df.columns:
            df['Cash_Ratio'] = df['Cash'] / df['Current_Liabilities']
            df['Cash_Ratio'] = df['Cash_Ratio'].replace([np.inf, -np.inf], np.nan)

        # === GROWTH RATES ===
        logger.info(f"5. Creating growth rates (QoQ, YoY)...")
        
        for col in ['Revenue', 'Net_Income', 'Total_Assets']:
            if col in df.columns:
                # Quarter-over-quarter growth (~90 days)
                df[f'{col}_Growth_QoQ'] = df.groupby('Company')[col].pct_change(periods=90) * 100
                # Year-over-year growth (~252 days)
                df[f'{col}_Growth_YoY'] = df.groupby('Company')[col].pct_change(periods=252) * 100

        # === LAGGED FINANCIAL METRICS ===
        logger.info(f"6. Creating lagged financial metrics...")
        
        for col in ['Revenue', 'Net_Income', 'Total_Assets', 'Total_Debt']:
            if col in df.columns:
                df[f'{col}_Lag90'] = df.groupby('Company')[col].shift(90)   # Last quarter
                df[f'{col}_Lag252'] = df.groupby('Company')[col].shift(252) # Last year

        # === VALUATION METRICS ===
        logger.info(f"7. Creating valuation metrics...")
        
        # Price to Book (if we have market cap proxy)
        if stock_col in df.columns and 'Total_Equity' in df.columns:
            # Simple P/B approximation
            df['Price_to_Book_Proxy'] = df[stock_col] / (df['Total_Equity'] / 1e9)  # Normalize
            df['Price_to_Book_Proxy'] = df['Price_to_Book_Proxy'].replace([np.inf, -np.inf], np.nan)

        # === TREND INDICATORS ===
        logger.info(f"8. Creating trend indicators...")
        
        # Revenue trend (is it increasing?)
        if 'Revenue' in df.columns:
            df['Revenue_Trend_90D'] = df.groupby('Company')['Revenue'].rolling(90, min_periods=1).apply(
                lambda x: 1 if len(x) > 1 and x.iloc[-1] > x.iloc[0] else 0, raw=False
            ).reset_index(0, drop=True)

        new_cols = len(df.columns)
        logger.info(f"\n✓ Company features engineered: {df.shape}")
        logger.info(f"  Features added: {new_cols - company_full.shape[1]}")

        return df

    # ========== MAIN PIPELINE ==========

    def run_feature_engineering(self) -> Dict[str, pd.DataFrame]:
        """Execute complete feature engineering pipeline."""
        logger.info("\n" + "="*80)
        logger.info("STEP 2: FEATURE ENGINEERING PIPELINE")
        logger.info("="*80)

        overall_start = time.time()

        # Load cleaned data
        data = self.load_cleaned_data()

        if not data:
            logger.error("\n❌ No cleaned data found!")
            logger.error("Run Step 1 first: python step1_data_cleaning.py")
            return {}

        # === ENGINEER FRED FEATURES ===
        logger.info("\n" + "="*80)
        logger.info("[1/3] FRED FEATURE ENGINEERING")
        logger.info("="*80)
        
        if 'fred' in data:
            fred_features = self.engineer_fred_features(data['fred'])
            output_path = self.features_dir / 'fred_features.csv'
            fred_features.to_csv(output_path, index=False)
            logger.info(f"\n✓ Saved: {output_path}")
            logger.info(f"  Size: {output_path.stat().st_size / (1024*1024):.2f} MB")
        else:
            logger.error("❌ FRED data not loaded")
            return {}

        # === ENGINEER MARKET FEATURES ===
        logger.info("\n" + "="*80)
        logger.info("[2/3] MARKET FEATURE ENGINEERING")
        logger.info("="*80)
        
        if 'market' in data:
            market_features = self.engineer_market_features(data['market'])
            output_path = self.features_dir / 'market_features.csv'
            market_features.to_csv(output_path, index=False)
            logger.info(f"\n✓ Saved: {output_path}")
            logger.info(f"  Size: {output_path.stat().st_size / (1024*1024):.2f} MB")
        else:
            logger.error("❌ Market data not loaded")
            return {}

        # === CONVERT QUARTERLY FINANCIALS TO DAILY ===
        logger.info("\n" + "="*80)
        logger.info("[3/3] COMPANY FEATURE ENGINEERING")
        logger.info("="*80)

        if 'balance' in data and 'income' in data and 'prices' in data:
            # Merge balance + income first (both quarterly)
            logger.info("\nMerging balance sheet + income statement (quarterly)...")
            
            financials_quarterly = pd.merge(
                data['balance'],
                data['income'],
                on=['Date', 'Company'],
                how='outer',
                suffixes=('', '_dup')
            )

            # Drop duplicate columns
            dup_cols = [col for col in financials_quarterly.columns if col.endswith('_dup')]
            if dup_cols:
                financials_quarterly.drop(columns=dup_cols, inplace=True)

            # Ensure Sector is present
            if 'Sector' not in financials_quarterly.columns:
                if 'Sector' in data['balance'].columns:
                    sector_map = data['balance'][['Company', 'Sector']].drop_duplicates().set_index('Company')['Sector'].to_dict()
                    financials_quarterly['Sector'] = financials_quarterly['Company'].map(sector_map)
                elif 'Sector' in data['income'].columns:
                    sector_map = data['income'][['Company', 'Sector']].drop_duplicates().set_index('Company')['Sector'].to_dict()
                    financials_quarterly['Sector'] = financials_quarterly['Company'].map(sector_map)

            logger.info(f"  Merged quarterly financials: {financials_quarterly.shape}")

            # Convert to daily
            financials_daily = self.quarterly_to_daily(financials_quarterly)

            # Engineer company features
            company_features = self.engineer_company_features(data['prices'], financials_daily)
            
            output_path = self.features_dir / 'company_features.csv'
            company_features.to_csv(output_path, index=False)
            logger.info(f"\n✓ Saved: {output_path}")
            logger.info(f"  Size: {output_path.stat().st_size / (1024*1024):.2f} MB")
        else:
            logger.error("❌ Company data not fully loaded")
            return {}

        # === FINAL SUMMARY ===
        elapsed = time.time() - overall_start

        logger.info("\n" + "="*80)
        logger.info("FEATURE ENGINEERING COMPLETE - SUMMARY")
        logger.info("="*80)

        summary_data = [
            {
                'Dataset': 'fred_features.csv',
                'Rows': len(fred_features),
                'Columns': len(fred_features.columns),
                'Frequency': 'Daily',
                'Use': 'Pipeline 1 (VAE) + Pipeline 2'
            },
            {
                'Dataset': 'market_features.csv',
                'Rows': len(market_features),
                'Columns': len(market_features.columns),
                'Frequency': 'Daily',
                'Use': 'Pipeline 1 (VAE) + Pipeline 2'
            },
            {
                'Dataset': 'company_features.csv',
                'Rows': len(company_features),
                'Columns': len(company_features.columns),
                'Frequency': 'Daily',
                'Use': 'Pipeline 2 (XGBoost/LSTM)'
            }
        ]

        summary_df = pd.DataFrame(summary_data)
        print("\n" + summary_df.to_string(index=False))

        logger.info(f"\n⏱️  Total Time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
        logger.info("="*80)

        logger.info("\n✅ Step 2 Complete!")
        logger.info("\n➡️  Next Steps:")
        logger.info("   1. Run Step 3: python step3_data_merging.py")
        logger.info("   2. Then validate: python src/validation/validate_checkpoint_3_merged.py")

        return {
            'fred_features': fred_features,
            'market_features': market_features,
            'company_features': company_features
        }


def main():
    """Execute Step 2: Feature Engineering."""

    engineer = FeatureEngineer(clean_dir="data/clean", features_dir="data/features")

    try:
        features = engineer.run_feature_engineering()

        if not features:
            logger.error("\n❌ Feature engineering failed!")
            return None

        # Show sample data
        logger.info("\n" + "="*80)
        logger.info("SAMPLE DATA PREVIEW")
        logger.info("="*80)

        logger.info("\n1. FRED FEATURES (first 5 rows, first 10 cols):")
        print(features['fred_features'].iloc[:5, :10].to_string())

        logger.info("\n2. MARKET FEATURES (first 5 rows, key cols):")
        market_cols = ['Date', 'VIX', 'SP500_Close', 'SP500_Return_1D', 'VIX_Regime']
        available_market_cols = [col for col in market_cols if col in features['market_features'].columns]
        print(features['market_features'][available_market_cols].head().to_string())

        logger.info("\n3. COMPANY FEATURES (first 5 rows, key cols):")
        company_cols = ['Date', 'Company', 'Stock_Price', 'Revenue', 'Net_Income', 
                       'Stock_Return_1D', 'Profit_Margin']
        available_company_cols = [col for col in company_cols if col in features['company_features'].columns]
        print(features['company_features'][available_company_cols].head().to_string())

        logger.info("\n" + "="*80)
        logger.info("FEATURE COUNTS BY DATASET")
        logger.info("="*80)
        logger.info(f"FRED features:    {len(features['fred_features'].columns)} columns")
        logger.info(f"Market features:  {len(features['market_features'].columns)} columns")
        logger.info(f"Company features: {len(features['company_features'].columns)} columns")

        return features

    except FileNotFoundError as e:
        logger.error(f"\n❌ ERROR: {e}")
        logger.error("Make sure cleaned data exists in data/clean/")
        logger.error("Run Step 1 first: python step1_data_cleaning.py")
        return None
    except Exception as e:
        logger.error(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    features = main()