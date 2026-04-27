"""
Dynamic LGD and Basel III AIRB Capital Module
=============================================
Replaces static stage-based LGD with a model driven by:
  1. Loan type / collateral quality
  2. ONS HPI movement per scenario
  3. IFRS 9 stage (workout cost uplift)
  4. Macro scenario floor

Basel III AIRB capital requirement per BCBS §328:
  - Retail asset correlation formula (prescribed, not estimated)
  - K = stressed expected loss - normal expected loss
  - RWA = K × 12.5 × EAD
  - Basel 3.1 output floor at 72.5% of standardised approach
"""

import numpy as np
import pandas as pd
from scipy.stats import norm

N = norm.cdf
G = norm.ppf

# Base LGD by loan type — collateral quality anchor
BASE_LGD = {
    "unsecured_personal": 0.65,
    "secured_mortgage":   0.20,
    "credit_card":        0.75,
    "auto":               0.40,
}

# HPI change by scenario name (% house price movement)
HPI_BY_SCENARIO = {
    "Base":    2.0,
    "Adverse": -5.0,
    "Stress":  -12.0,
}


def dynamic_lgd(
    df: pd.DataFrame,
    stage: pd.Series,
    scenario_name: str,
) -> pd.Series:
    """
    Dynamic LGD driven by loan type, ONS HPI, IFRS 9 stage, and macro scenario.
    Falls back to debt_to_income proxy if loan_type column absent.
    """
    # --- Base LGD from loan type (or proxy) ---
    if "loan_type" in df.columns:
        base = df["loan_type"].map(BASE_LGD).fillna(0.55)
    else:
        # LendingClub / synthetic data has no loan_type — proxy from purpose + home_ownership
        purpose = df.get("purpose", pd.Series(["other"] * len(df), index=df.index))
        home = df.get("home_ownership", pd.Series(["RENT"] * len(df), index=df.index))
        base = pd.Series(0.60, index=df.index)  # unsecured default
        base = base.where(~purpose.isin(["home_improvement"]) | home.ne("MORTGAGE"), 0.30)

    # --- HPI adjustment (mortgages / home-secured only) ---
    hpi_change = HPI_BY_SCENARIO.get(scenario_name, 0.0)
    if "loan_type" in df.columns:
        is_mortgage = df["loan_type"] == "secured_mortgage"
    else:
        is_mortgage = (
            df.get("purpose", pd.Series(["other"] * len(df), index=df.index)).isin(["home_improvement"])
            & df.get("home_ownership", pd.Series(["RENT"] * len(df), index=df.index)).eq("MORTGAGE")
        )
    hpi_adjustment = np.where(is_mortgage, -hpi_change * 0.003, 0.0)

    # --- Stage adjustment (workout cost uplift) ---
    stage_adjustment = stage.map({1: 0.0, 2: 0.05, 3: 0.12}).fillna(0.0)

    # --- Macro scenario floor ---
    scenario_floor = {"Base": 0.0, "Adverse": 0.03, "Stress": 0.07}.get(scenario_name, 0.0)

    lgd = (base + hpi_adjustment + stage_adjustment + scenario_floor).clip(0.05, 0.95)
    return lgd


def retail_asset_correlation(pd_series: pd.Series) -> pd.Series:
    """Basel III retail asset correlation — BCBS §328. Range: 0.03 to 0.16."""
    exp_term = np.exp(-35 * pd_series)
    weight = (1 - exp_term) / (1 - np.exp(-35))
    return 0.03 * weight + 0.16 * (1 - weight)


def airb_capital_requirement(
    pd_series: pd.Series,
    lgd_series: pd.Series,
    ead_series: pd.Series,
) -> pd.DataFrame:
    """
    Basel III AIRB capital for retail exposures.
    Returns asset_correlation, K (capital %), RWA, CET1 Pillar 1 required.
    Applies Basel 3.1 output floor at 72.5% of standardised RWA.
    """
    pd_s = pd_series.clip(0.0003, 0.999)
    R = retail_asset_correlation(pd_s)

    # K = stressed expected loss - normal expected loss
    stressed = lgd_series * N(
        (1 - R) ** -0.5 * G(pd_s) + (R / (1 - R)) ** 0.5 * G(0.999)
    )
    normal = pd_s * lgd_series
    K = (stressed - normal).clip(0)

    # RWA: 12.5 = 1/0.08 (minimum 8% capital ratio inverted)
    rwa = K * 12.5 * ead_series

    # Basel 3.1 output floor — IRB cannot fall below 72.5% of SA
    sa_rwa = ead_series * 0.75          # Basel retail SA risk weight
    floor_rwa = sa_rwa * 0.725
    rwa_floored = rwa.clip(lower=floor_rwa)

    return pd.DataFrame({
        "asset_correlation": R.values,
        "capital_requirement_pct": K.values,
        "rwa": rwa_floored.values,
        "cet1_pillar1_required": (rwa_floored * 0.045).values,
    }, index=pd_series.index)


def icaap_waterfall(
    total_rwa: float,
    cet1_ratio: float = 0.15,
    pillar2a_pct: float = 0.025,
    stress_ecl_uplift: float = 0.16,
) -> dict:
    """
    CET1 waterfall: Pillar 1 → Pillar 2A → conservation buffer → stress depletion.
    Returns headroom at each stage and whether the bank passes the stress test.
    """
    cet1 = total_rwa * cet1_ratio
    p1 = total_rwa * 0.045
    p2a = total_rwa * pillar2a_pct
    buffer = total_rwa * 0.025          # capital conservation buffer

    stress_loss = total_rwa * stress_ecl_uplift * 0.08
    post_stress = cet1 - stress_loss

    return {
        "cet1_capital":           round(cet1, 0),
        "pillar1_requirement":    round(p1, 0),
        "pillar2a_requirement":   round(p2a, 0),
        "conservation_buffer":    round(buffer, 0),
        "headroom_after_p1":      round(cet1 - p1, 0),
        "headroom_after_p2a":     round(cet1 - p1 - p2a, 0),
        "headroom_after_buffer":  round(cet1 - p1 - p2a - buffer, 0),
        "post_stress_cet1":       round(post_stress, 0),
        "post_stress_ratio":      round(post_stress / total_rwa, 4),
        "passes_stress":          post_stress / total_rwa >= 0.045,
    }


def fca_vulnerability_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    FCA FG21/1 Consumer Duty vulnerability framework.
    Four outcome drivers: Price & Value, Products & Services,
    Consumer Understanding, Consumer Support.
    """
    out = df.copy()

    dti = df.get("debt_to_income", pd.Series(0.3, index=df.index))
    rate = df.get("interest_rate", pd.Series(13.0, index=df.index))
    score = df.get("credit_score", pd.Series(650, index=df.index))
    dpd = df.get("days_past_due", pd.Series(0, index=df.index))
    missed = df.get("missed_payments", df.get("prior_missed_payments",
                    pd.Series(0, index=df.index)))

    # Outcome 1: Price & Value — affordability
    out["flag_dti_breach"] = (dti > 0.45).astype(int)

    # Outcome 2: Products & Services — high rate + weak credit
    out["flag_high_rate_vulnerable"] = (
        (rate > 25) & (score < 580)
    ).astype(int)

    # Outcome 3: Consumer Understanding — proxy via credit score band
    out["flag_low_financial_literacy"] = (score < 580).astype(int)

    # Outcome 4: Consumer Support — payment distress signals
    out["flag_payment_distress"] = (
        (dpd > 0) | (missed > 1)
    ).astype(int)

    # Composite vulnerability indicator (FG21/1 §4.14)
    out["vulnerable_indicator"] = (
        out["flag_dti_breach"] |
        out["flag_high_rate_vulnerable"] |
        out["flag_payment_distress"]
    ).astype(int)

    # Hard policy override — FCA reject regardless of PD score
    out["fca_hard_reject"] = (
        (dti > 0.60) | (dpd >= 60)
    ).astype(int)

    return out
