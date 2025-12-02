
"""
============================================================================
BALANCED Labeling Functions - Equal AT_RISK and NOT_AT_RISK voting
============================================================================
Strategy: Balance AT_RISK and NOT_AT_RISK LFs to achieve realistic labels
Uses VERY STRICT conditions for AT_RISK (only extreme cases)
Uses LENIENT conditions for NOT_AT_RISK (most normal cases)
============================================================================
"""

from snorkel.labeling import labeling_function
import pandas as pd
import yaml
from pathlib import Path
from typing import Dict

# ============================================================================
# LOAD THRESHOLDS
# ============================================================================

def load_thresholds() -> Dict:
    auto_file = Path("outputs/snorkel/thresholds_auto.yaml")
    if auto_file.exists():
        with open(auto_file, 'r') as f:
            data = yaml.safe_load(f)
            return data['thresholds']
    return {}

THRESHOLDS = load_thresholds()
ABSTAIN = -1
NOT_AT_RISK = 0
AT_RISK = 1

def safe_check(value, threshold, comparison='gt'):
    if pd.isna(value):
        return False
    return (value > threshold) if comparison == 'gt' else (value < threshold)

def get_threshold(key: str, default: float) -> float:
    return THRESHOLDS.get(key, default)

# ============================================================================
# AT_RISK LABELING FUNCTIONS (VERY STRICT - Only 5 LFs)
# ============================================================================

@labeling_function()
def lf_extreme_market_panic_with_debt(row):
    """VIX > 45 + High debt + Revenue decline (3 conditions)"""
    extreme_vix = safe_check(row.vix_q_mean, 45, 'gt')
    high_debt = safe_check(row.Debt_to_Equity, 5.0, 'gt')
    revenue_down = safe_check(row.Revenue_growth_1q, -0.15, 'lt')
    
    if extreme_vix and high_debt and revenue_down:
        return AT_RISK
    return ABSTAIN


@labeling_function()
def lf_severe_profitability_crisis(row):
    """Negative or near-zero profitability + recession (3 conditions)"""
    no_profit = safe_check(row.net_margin, 0.01, 'lt')
    no_roa = safe_check(row.roa, 0.01, 'lt')
    recession = safe_check(row.Unemployment_Rate_last, 8.0, 'gt')
    
    if no_profit and no_roa and recession:
        return AT_RISK
    return ABSTAIN


@labeling_function()
def lf_2008_crisis_severe_impact(row):
    """2008 crisis + extreme vulnerability"""
    crisis_2008 = (row.Year == 2008 and row.Quarter >= 3) or (row.Year == 2009 and row.Quarter <= 2)
    
    if not crisis_2008:
        return ABSTAIN
    
    # Very strict: need 3+ vulnerabilities during 2008
    conditions = [
        safe_check(row.Debt_to_Equity, 4.0, 'gt'),
        safe_check(row.Revenue_growth_1q, -0.20, 'lt'),
        safe_check(row.net_margin, 0.03, 'lt'),
        safe_check(row.vix_q_mean, 35, 'gt')
    ]
    
    if sum(conditions) >= 3:
        return AT_RISK
    return ABSTAIN


@labeling_function()
def lf_2020_covid_collapse(row):
    """2020 COVID + revenue collapse"""
    covid_2020 = (row.Year == 2020) and (row.Quarter in [1, 2])
    
    if not covid_2020:
        return ABSTAIN
    
    # Severe impact only
    severe = safe_check(row.Revenue_growth_1q, -0.25, 'lt')
    
    if severe:
        return AT_RISK
    return ABSTAIN


@labeling_function()
def lf_debt_death_spiral(row):
    """Extreme debt + collapsing revenue + no profitability"""
    extreme_debt = safe_check(row.Debt_to_Equity, 6.0, 'gt')
    revenue_collapse = safe_check(row.Revenue_growth_1q, -0.20, 'lt')
    no_profit = safe_check(row.net_margin, 0.02, 'lt')
    
    if extreme_debt and revenue_collapse and no_profit:
        return AT_RISK
    return ABSTAIN


# ============================================================================
# NOT_AT_RISK LABELING FUNCTIONS (LENIENT - 10 LFs to balance)
# ============================================================================

@labeling_function()
def lf_positive_revenue_growth(row):
    """Simple positive revenue growth"""
    if safe_check(row.Revenue_growth_1q, 0.0, 'gt'):
        return NOT_AT_RISK
    return ABSTAIN


@labeling_function()
def lf_profitable_company(row):
    """Company is profitable (margin > 5%)"""
    if safe_check(row.net_margin, 0.05, 'gt'):
        return NOT_AT_RISK
    return ABSTAIN


@labeling_function()
def lf_healthy_debt_levels(row):
    """Low debt-to-equity ratio"""
    if safe_check(row.Debt_to_Equity, 2.0, 'lt'):
        return NOT_AT_RISK
    return ABSTAIN


@labeling_function()
def lf_calm_market_environment(row):
    """Low market volatility (VIX < 25)"""
    if safe_check(row.vix_q_mean, 25, 'lt'):
        return NOT_AT_RISK
    return ABSTAIN


@labeling_function()
def lf_low_financial_stress(row):
    """Low systemic financial stress"""
    if safe_check(row.Financial_Stress_Index_mean, 0.5, 'lt'):
        return NOT_AT_RISK
    return ABSTAIN


@labeling_function()
def lf_strong_economy(row):
    """Low unemployment (< 6%)"""
    if safe_check(row.Unemployment_Rate_last, 6.0, 'lt'):
        return NOT_AT_RISK
    return ABSTAIN


@labeling_function()
def lf_positive_yield_curve(row):
    """Positive yield curve (not inverted)"""
    if safe_check(row.Yield_Curve_Spread_mean, 0.5, 'gt'):
        return NOT_AT_RISK
    return ABSTAIN


@labeling_function()
def lf_good_liquidity(row):
    """Healthy current ratio"""
    if safe_check(row.Current_Ratio, 1.5, 'gt'):
        return NOT_AT_RISK
    return ABSTAIN


@labeling_function()
def lf_stable_growth_profitable(row):
    """Positive growth + profitable"""
    growth = safe_check(row.Revenue_growth_1q, 0.0, 'gt')
    profit = safe_check(row.net_margin, 0.05, 'gt')
    
    if growth and profit:
        return NOT_AT_RISK
    return ABSTAIN


@labeling_function()
def lf_normal_period_healthy(row):
    """Normal period (not 2008/2020) + basic health"""
    crisis_years = row.Year in [2008, 2009, 2020]
    
    if crisis_years:
        return ABSTAIN
    
    # Basic health during normal times
    positive_revenue = safe_check(row.Revenue_growth_1q, -0.05, 'gt')
    positive_margin = safe_check(row.net_margin, 0.03, 'gt')
    
    if positive_revenue and positive_margin:
        return NOT_AT_RISK
    return ABSTAIN


# ============================================================================
# BALANCED REGISTRY: 5 AT_RISK + 10 NOT_AT_RISK
# ============================================================================

ALL_LFS = [
    # AT_RISK (5 strict LFs)
    lf_extreme_market_panic_with_debt,
    lf_severe_profitability_crisis,
    lf_2008_crisis_severe_impact,
    lf_2020_covid_collapse,
    lf_debt_death_spiral,
    
    # NOT_AT_RISK (10 lenient LFs)
    lf_positive_revenue_growth,
    lf_profitable_company,
    lf_healthy_debt_levels,
    lf_calm_market_environment,
    lf_low_financial_stress,
    lf_strong_economy,
    lf_positive_yield_curve,
    lf_good_liquidity,
    lf_stable_growth_profitable,
    lf_normal_period_healthy
]

def get_enabled_lfs(config_path: str = "configs/eda_config.yaml"):
    return ALL_LFS

if __name__ == "__main__":
    print(f"Balanced LFs: {len(ALL_LFS)} total")
    print(f"  AT_RISK: 5 (strict multi-condition)")
    print(f"  NOT_AT_RISK: 10 (lenient single/dual condition)")
    print("\nThis 1:2 ratio should produce ~1-5% at-risk rate")
