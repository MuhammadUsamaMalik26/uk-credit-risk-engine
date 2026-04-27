
import numpy as np
import pandas as pd
from models.basel_lgd import dynamic_lgd, airb_capital_requirement, icaap_waterfall, fca_vulnerability_flags

STAGE_LGD = {1: 0.35, 2: 0.45, 3: 0.60}

def monthly_payment(principal: pd.Series, annual_rate: pd.Series, term_months: pd.Series) -> pd.Series:
    monthly_rate = (annual_rate / 100.0) / 12.0
    payment = np.where(
        monthly_rate > 0,
        principal * (monthly_rate / (1 - (1 + monthly_rate) ** (-term_months))),
        principal / term_months
    )
    return pd.Series(payment, index=principal.index)

def projected_ead_path(df: pd.DataFrame, horizon_months: int | None = None) -> pd.DataFrame:
    remaining_term = (df["term_months"] - df["months_on_book"]).clip(lower=1)
    horizon = int(horizon_months or remaining_term.max())
    payment = monthly_payment(df["loan_amount"], df["interest_rate"], df["term_months"])
    monthly_rate = (df["interest_rate"] / 100.0) / 12.0

    records = []
    for idx, row in df.iterrows():
        bal = float(row["current_balance"])
        max_m = int(min(horizon, remaining_term.loc[idx]))
        for m in range(1, max_m + 1):
            interest = bal * monthly_rate.loc[idx]
            principal_paid = max(float(payment.loc[idx]) - interest, 0)
            bal = max(bal - principal_paid, 0)
            records.append((idx, m, bal))
    return pd.DataFrame(records, columns=["row_id", "month", "ead"])

def scenario_factor(unemployment, gdp_growth, bank_rate, inflation):
    return max(
        0.75,
        1.0
        + 0.08 * (unemployment - 4.5)
        + 0.05 * (inflation - 2.5)
        + 0.03 * (bank_rate - 4.75)
        - 0.04 * (gdp_growth - 1.0),
    )

def stressed_pd(pd_12m: pd.Series, unemployment, gdp_growth, bank_rate, inflation):
    factor = scenario_factor(unemployment, gdp_growth, bank_rate, inflation)
    return (pd_12m * factor).clip(0.0001, 0.999)

def lifetime_pd_from_12m(pd_12m: pd.Series, remaining_months: pd.Series) -> pd.Series:
    monthly_hazard = 1 - (1 - pd_12m.clip(0.0001, 0.999)) ** (1 / 12.0)
    return (1 - (1 - monthly_hazard) ** remaining_months.clip(lower=1)).clip(0.0001, 0.999)

def assign_ifrs9_stages(df: pd.DataFrame, current_lifetime_pd: pd.Series, origination_lifetime_pd: pd.Series, sicr_threshold: float = 2.0):
    sicr_ratio = (current_lifetime_pd / origination_lifetime_pd.clip(lower=1e-4)).clip(lower=0.01)
    stage = pd.Series(1, index=df.index)
    stage[(df["days_past_due"] >= 30) | (sicr_ratio >= sicr_threshold)] = 2
    stage[(df["default_flag"] == 1) | (df["days_past_due"] >= 90)] = 3
    return stage, sicr_ratio

def stage_lgd(stage: pd.Series, scenario_multiplier: float = 1.0) -> pd.Series:
    base = stage.map(STAGE_LGD).astype(float)
    return (base * scenario_multiplier).clip(0.05, 0.95)

def discount_factor(annual_rate_pct: pd.Series, month: pd.Series) -> pd.Series:
    eir = (annual_rate_pct / 100.0) / 12.0
    return 1 / ((1 + eir) ** month)

def compute_stage_ecl(df: pd.DataFrame, stage: pd.Series, pd_12m_stressed: pd.Series, scenario_name: str):
    remaining = (df["term_months"] - df["months_on_book"]).clip(lower=1)
    lifetime_pd = lifetime_pd_from_12m(pd_12m_stressed, remaining)
    lgd = dynamic_lgd(df, stage, scenario_name=scenario_name)

    stage1_ead = df["current_balance"]
    stage1_ecl = stage1_ead * pd_12m_stressed * lgd * discount_factor(df["interest_rate"], pd.Series(12, index=df.index))

    horizon_df = projected_ead_path(df)
    results = pd.DataFrame(index=df.index)
    results["stage"] = stage
    results["pd_12m"] = pd_12m_stressed
    results["lifetime_pd"] = lifetime_pd
    results["lgd"] = lgd
    results["ead"] = df["current_balance"]

    lifetime_losses = []
    for idx, row in df.iterrows():
        path = horizon_df[horizon_df["row_id"] == idx].copy()
        if path.empty:
            lifetime_losses.append(0.0)
            continue
        rem = len(path)
        marginal_hazard = 1 - (1 - float(pd_12m_stressed.loc[idx])) ** (1 / 12.0)
        path["marginal_pd"] = (1 - marginal_hazard) ** (path["month"] - 1) * marginal_hazard
        path["df"] = 1 / ((1 + (float(df.loc[idx, "interest_rate"]) / 100.0) / 12.0) ** path["month"])
        loss = (path["ead"] * float(lgd.loc[idx]) * path["marginal_pd"] * path["df"]).sum()
        lifetime_losses.append(loss)

    results["ecl"] = np.where(stage == 1, stage1_ecl, lifetime_losses)
    results["scenario_name"] = scenario_name
    return results

def run_scenarios(df: pd.DataFrame, pd_12m: pd.Series, origination_pd_12m: pd.Series, scenarios: pd.DataFrame, sicr_threshold: float = 2.0):
    remaining = (df["term_months"] - df["months_on_book"]).clip(lower=1)
    origination_lifetime = lifetime_pd_from_12m(origination_pd_12m, remaining + df["months_on_book"])
    scenario_results = []
    for _, scen in scenarios.iterrows():
        pd_stressed = stressed_pd(
            pd_12m, scen["unemployment_rate"], scen["gdp_growth"], scen["bank_rate"], scen["inflation_cpi"]
        )
        current_lifetime = lifetime_pd_from_12m(pd_stressed, remaining)
        stage, sicr_ratio = assign_ifrs9_stages(df, current_lifetime, origination_lifetime, sicr_threshold=sicr_threshold)
        res = compute_stage_ecl(df, stage, pd_stressed, scen["scenario_name"])
        # Basel AIRB capital
        capital = airb_capital_requirement(pd_stressed, res["lgd"], df["current_balance"].reset_index(drop=True) if hasattr(df["current_balance"], "reset_index") else df["current_balance"])
        res["rwa"] = capital["rwa"].values
        res["cet1_required"] = capital["cet1_pillar1_required"].values
        res["asset_correlation"] = capital["asset_correlation"].values
        res["loan_id"] = df["loan_id"].values
        res["scenario_weight"] = float(scen["scenario_weight"])
        res["sicr_ratio"] = sicr_ratio.values
        res["baseline_pd_12m"] = pd_12m.values
        res["origination_pd_12m"] = origination_pd_12m.values
        scenario_results.append(res.reset_index(drop=True))
    return pd.concat(scenario_results, ignore_index=True)

def probability_weighted_summary(scenario_results: pd.DataFrame) -> pd.DataFrame:
    grouped = scenario_results.groupby("scenario_name", as_index=False).agg(
        total_ecl=("ecl", "sum"),
        total_ead=("ead", "sum"),
        avg_pd_12m=("pd_12m", "mean"),
        avg_lifetime_pd=("lifetime_pd", "mean"),
        avg_sicr=("sicr_ratio", "mean"),
        weight=("scenario_weight", "first"),
    )
    weighted_ecl = float((grouped["total_ecl"] * grouped["weight"]).sum())
    grouped["weighted_contribution"] = grouped["total_ecl"] * grouped["weight"]
    grouped.attrs["weighted_ecl"] = weighted_ecl
    return grouped
