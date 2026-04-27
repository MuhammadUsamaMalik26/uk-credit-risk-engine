
import pandas as pd

def get_boe_scenarios():
    """
    BoE-style macro scenarios for consumer credit stress testing.
    Values are illustrative anchors for a UK-style scenario framework.
    """
    return pd.DataFrame([
        {"scenario_name": "Base", "unemployment_rate": 4.5, "gdp_growth": 1.0, "bank_rate": 4.75, "inflation_cpi": 2.5, "scenario_weight": 0.60},
        {"scenario_name": "Adverse", "unemployment_rate": 6.0, "gdp_growth": -0.5, "bank_rate": 5.25, "inflation_cpi": 3.5, "scenario_weight": 0.30},
        {"scenario_name": "Stress", "unemployment_rate": 8.0, "gdp_growth": -2.0, "bank_rate": 5.75, "inflation_cpi": 4.5, "scenario_weight": 0.10},
    ])

def apply_sidebar_shocks(base_scenarios: pd.DataFrame, unemployment_shift=0.0, gdp_shift=0.0, rate_shift_bps=0.0, inflation_shift=0.0):
    shocked = base_scenarios.copy()
    shocked["unemployment_rate"] = shocked["unemployment_rate"] + unemployment_shift
    shocked["gdp_growth"] = shocked["gdp_growth"] + gdp_shift
    shocked["bank_rate"] = shocked["bank_rate"] + rate_shift_bps / 100.0
    shocked["inflation_cpi"] = shocked["inflation_cpi"] + inflation_shift
    return shocked
