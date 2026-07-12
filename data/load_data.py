
import os
import glob
import numpy as np
import pandas as pd

REAL_DATA_PATTERNS = [
    "accepted_*.csv",
    "lendingclub*.csv",
    "loan*.csv",
]

INTERNAL_COLUMNS = [
    "loan_id", "origination_date", "reporting_date", "loan_amount", "current_balance",
    "term_months", "months_on_book", "interest_rate", "annual_income", "debt_to_income",
    "credit_score", "missed_payments", "days_past_due", "grade_band", "home_ownership",
    "purpose", "loan_status", "default_flag", "stage_at_source"
]


def _find_real_csv(data_dir: str):
    for pattern in REAL_DATA_PATTERNS:
        matches = glob.glob(os.path.join(data_dir, pattern))
        if matches:
            return sorted(matches)[0]
    return None


def _parse_term(term_series: pd.Series) -> pd.Series:
    return term_series.astype(str).str.extract(r"(\d+)").astype(float).iloc[:, 0]


def _parse_pct(series: pd.Series, default: float = 0.0) -> pd.Series:
    cleaned = series.astype(str).str.replace("%", "", regex=False).str.strip()
    return pd.to_numeric(cleaned, errors="coerce").fillna(default)


def _map_lendingclub(csv_path: str, sample_n: int | None = None, random_state: int = 42):
    raw = pd.read_csv(csv_path, low_memory=False)
    if sample_n is not None and len(raw) > sample_n:
        raw = raw.sample(sample_n, random_state=random_state).copy()
    else:
        raw = raw.copy()

    if "origination_date" in raw.columns:
        origination_date = pd.to_datetime(raw["origination_date"], errors="coerce")
    elif "issue_d" in raw.columns:
        origination_date = pd.to_datetime(raw["issue_d"], errors="coerce")
    else:
        origination_date = pd.Series([pd.Timestamp("2018-01-01")] * len(raw))

    reporting_date = pd.Timestamp("2026-04-08")
    months_on_book = ((reporting_date - origination_date).dt.days / 30.4).clip(lower=1)
    term_months = _parse_term(raw.get("term", pd.Series(np.repeat(36, len(raw)))))
    months_on_book = np.minimum(months_on_book.fillna(12), term_months.fillna(36) - 1).clip(lower=1)

    loan_amount = pd.to_numeric(
        raw.get("funded_amnt", raw.get("loan_amnt", pd.Series(np.repeat(10000, len(raw))))),
        errors="coerce"
    ).fillna(10000)

    installment = pd.to_numeric(raw.get("installment", pd.Series(np.repeat(300, len(raw)))), errors="coerce").fillna(300)
    remaining_months = (term_months - months_on_book).clip(lower=1)
    current_balance = (installment * remaining_months).clip(lower=0.05 * loan_amount, upper=loan_amount)

    fico = pd.to_numeric(raw.get("fico_range_high", raw.get("fico_range_low", pd.Series(np.repeat(680, len(raw))))), errors="coerce").fillna(680)
    dti = pd.to_numeric(raw.get("dti", pd.Series(np.repeat(0.18, len(raw)))), errors="coerce").fillna(18.0) / 100.0
    missed_payments = pd.to_numeric(raw.get("delinq_2yrs", pd.Series(np.zeros(len(raw)))), errors="coerce").fillna(0).clip(lower=0)
    annual_income = pd.to_numeric(raw.get("annual_inc", pd.Series(np.repeat(60000, len(raw)))), errors="coerce").fillna(60000)
    interest_rate = _parse_pct(raw.get("int_rate", pd.Series(np.repeat("13.5%", len(raw)))), default=13.5)
    loan_status = raw.get("loan_status", pd.Series(np.repeat("Current", len(raw)))).astype(str)

    default_statuses = {
        "Charged Off", "Default", "Late (31-120 days)", "Late (16-30 days)", "Does not meet the credit policy. Status:Charged Off"
    }
    stage3_statuses = {"Charged Off", "Default"}
    default_flag = loan_status.isin(default_statuses).astype(int)
    days_past_due = np.select(
        [
            loan_status.eq("Late (31-120 days)"),
            loan_status.eq("Late (16-30 days)"),
            loan_status.isin(stage3_statuses),
        ],
        [60, 30, 90],
        default=0
    )

    home_ownership = raw.get("home_ownership", pd.Series(np.repeat("OTHER", len(raw)))).fillna("OTHER").astype(str)
    purpose = raw.get("purpose", pd.Series(np.repeat("other", len(raw)))).fillna("other").astype(str)
    grade_band = raw.get("grade", pd.Series(np.repeat("C", len(raw)))).fillna("C").astype(str)

    stage_at_source = np.select(
        [days_past_due >= 90, days_past_due >= 30, default_flag.eq(1)],
        [3, 2, 3],
        default=1
    )

    df = pd.DataFrame({
        "loan_id": raw.get("id", pd.Series(range(1, len(raw) + 1))).astype(str),
        "origination_date": pd.to_datetime(origination_date).fillna(pd.Timestamp("2018-01-01")),
        "reporting_date": reporting_date,
        "loan_amount": loan_amount,
        "current_balance": current_balance,
        "term_months": term_months.fillna(36).astype(int),
        "months_on_book": pd.Series(months_on_book, index=raw.index).replace([np.inf, -np.inf], np.nan).fillna(1).clip(lower=1).round().astype(int),
        "interest_rate": interest_rate,
        "annual_income": annual_income,
        "debt_to_income": dti.clip(lower=0, upper=1.2),
        "credit_score": fico.clip(lower=300, upper=850).astype(int),
        "missed_payments": missed_payments.astype(int),
        "days_past_due": pd.Series(days_past_due, index=raw.index).astype(int),
        "grade_band": grade_band,
        "home_ownership": home_ownership,
        "purpose": purpose,
        "loan_status": loan_status,
        "default_flag": default_flag,
        "stage_at_source": stage_at_source
    })

    meta = {"source": "LendingClub (real)", "path": csv_path}
    return df[INTERNAL_COLUMNS], meta


def _generate_synthetic(n: int = 1500, random_state: int = 42):
    rng = np.random.default_rng(random_state)
    term_months = rng.choice([12, 24, 36, 48, 60], size=n, p=[0.08, 0.18, 0.38, 0.22, 0.14])
    months_on_book = np.array([rng.integers(1, term) for term in term_months])
    credit_score = np.clip(rng.normal(665, 70, n), 520, 820).round().astype(int)
    debt_to_income = np.clip(rng.beta(2, 5, n) * 0.9, 0.02, 0.95)
    annual_income = np.clip(rng.lognormal(np.log(55000), 0.45, n), 18000, 180000)
    interest_rate = np.clip(9 + (720 - credit_score) / 28 + rng.normal(0, 1.5, n), 5, 29)
    loan_amount = np.clip(rng.lognormal(np.log(8500), 0.55, n), 1000, 35000)
    missed_payments = rng.poisson(0.5, n)
    days_past_due = np.select([missed_payments >= 4, missed_payments >= 2], [90, 30], default=0)
    grade_band = pd.cut(credit_score, bins=[0, 600, 660, 720, 900], labels=["D", "C", "B", "A"]).astype(str)
    home_ownership = rng.choice(["RENT", "MORTGAGE", "OWN"], size=n, p=[0.45, 0.45, 0.10])
    purpose = rng.choice(["debt_consolidation", "home_improvement", "car", "medical", "other"], size=n)
    base_logit = (
        -5.2
        + (700 - credit_score) / 65
        + 2.6 * debt_to_income
        + 0.35 * missed_payments
        + 0.012 * (interest_rate - 12)
    )
    default_prob = 1 / (1 + np.exp(-base_logit))
    default_flag = (rng.random(n) < default_prob).astype(int)
    days_past_due = np.where(default_flag == 1, np.maximum(days_past_due, 90), days_past_due)
    stage_at_source = np.select([default_flag == 1, days_past_due >= 30], [3, 2], default=1)

    origination_dates = pd.Timestamp("2020-01-01") + pd.to_timedelta(rng.integers(0, 1800, n), unit="D")
    reporting_date = pd.Timestamp("2026-04-08")
    current_balance = loan_amount * np.clip(1 - months_on_book / term_months, 0.08, 1.0)

    df = pd.DataFrame({
        "loan_id": [f"LN{str(i).zfill(6)}" for i in range(1, n + 1)],
        "origination_date": origination_dates,
        "reporting_date": reporting_date,
        "loan_amount": loan_amount.round(2),
        "current_balance": current_balance.round(2),
        "term_months": term_months.astype(int),
        "months_on_book": months_on_book.astype(int),
        "interest_rate": interest_rate.round(2),
        "annual_income": annual_income.round(2),
        "debt_to_income": debt_to_income.round(4),
        "credit_score": credit_score.astype(int),
        "missed_payments": missed_payments.astype(int),
        "days_past_due": days_past_due.astype(int),
        "grade_band": grade_band,
        "home_ownership": home_ownership,
        "purpose": purpose,
        "loan_status": np.where(default_flag == 1, "Charged Off", "Current"),
        "default_flag": default_flag.astype(int),
        "stage_at_source": stage_at_source.astype(int)
    })
    meta = {"source": "Synthetic fallback", "path": None}
    return df[INTERNAL_COLUMNS], meta


def load_portfolio(data_dir: str | None = None, n: int = 1500, random_state: int = 42):
    data_dir = data_dir or os.path.dirname(__file__)
    # Check for sample_loans.csv first (bundled demo data)
    sample_path = os.path.join(data_dir, "sample_loans.csv")
    if os.path.exists(sample_path):
        # Check if already in synthetic format
        df_check = pd.read_csv(sample_path, nrows=2)
        if "default_flag" in df_check.columns and "loan_id" in df_check.columns:
            df = pd.read_csv(sample_path)
            meta = {"source": "synthetic_csv", "original_rows": len(df)}
            return df, meta
        return _map_lendingclub(sample_path, sample_n=n, random_state=random_state)
    csv_path = _find_real_csv(data_dir)
    if csv_path:
        return _map_lendingclub(csv_path, sample_n=n, random_state=random_state)
    return _generate_synthetic(n=n, random_state=random_state)
