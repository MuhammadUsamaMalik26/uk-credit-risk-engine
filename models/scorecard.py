"""
PD Scorecard & Model Validation
===============================
Champion: WoE logistic regression scorecard
  - Monotonic fine/coarse binning -> Weight of Evidence (WoE)
  - Information Value (IV) feature selection (IV > 0.5 flagged for leakage review)
  - Logistic regression with NO class weighting -> PDs calibrated to observed default rates
  - Points-to-Double-Odds (PDO) scaling: 600 points at 50:1 odds, PDO = 20

Challengers: Logistic Regression (raw features), Random Forest, Gradient Boosting

Validation: out-of-time (OOT) split on origination_date; AUC, Gini, KS, Brier,
PSI (score stability) and decile calibration tables.

Feature policy
--------------
This PD feeds IFRS 9 staging on a live book, so behavioural fields
(current_balance, months_on_book) are legitimate inputs (behavioural PD).
Lender-assigned risk fields (grade_band, interest_rate) are excluded from the
scorecard to avoid circularity; delinquency fields (days_past_due,
missed_payments) are excluded to avoid target leakage.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, recall_score, roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

EPS = 1e-6

NUMERIC_FEATURES = [
    "loan_amount", "current_balance", "term_months", "months_on_book", "interest_rate",
    "annual_income", "debt_to_income", "credit_score",
]
CATEGORICAL_FEATURES = ["grade_band", "home_ownership", "purpose"]

# Scorecard excludes lender-assigned risk fields (circular)
SCORECARD_EXCLUDE = {"interest_rate", "grade_band", "rate_minus_score"}


# --------------------------------------------------------------------------- #
# Feature engineering (shared)
# --------------------------------------------------------------------------- #
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["utilization_proxy"] = (out["current_balance"] / out["loan_amount"]).clip(0, 1.5)
    out["score_x_dti"] = (out["credit_score"] / 850.0) * out["debt_to_income"]
    out["seasoning_ratio"] = (out["months_on_book"] / out["term_months"]).clip(0, 1)
    out["rate_minus_score"] = out["interest_rate"] - (out["credit_score"] / 100)
    return out


def get_feature_columns(df: pd.DataFrame):
    engineered = engineer_features(df)
    numeric = NUMERIC_FEATURES + ["utilization_proxy", "score_x_dti", "seasoning_ratio", "rate_minus_score"]
    categorical = [c for c in CATEGORICAL_FEATURES if c in engineered.columns]
    return engineered, numeric, categorical


# --------------------------------------------------------------------------- #
# Binning / WoE / IV
# --------------------------------------------------------------------------- #
def _woe_table(bins: pd.Series, y: pd.Series) -> pd.DataFrame:
    t = pd.DataFrame({"bin": bins, "y": y}).groupby("bin", observed=True)["y"].agg(["count", "sum"])
    t.columns = ["n", "bad"]
    t["good"] = t["n"] - t["bad"]
    k = len(t)
    t["dist_bad"] = (t["bad"] + 0.5) / (t["bad"].sum() + 0.5 * k)
    t["dist_good"] = (t["good"] + 0.5) / (t["good"].sum() + 0.5 * k)
    t["woe"] = np.log(t["dist_good"] / t["dist_bad"])
    t["iv"] = (t["dist_good"] - t["dist_bad"]) * t["woe"]
    t["bad_rate"] = t["bad"] / t["n"]
    return t


def _monotonic_edges(x: pd.Series, y: pd.Series, max_bins: int = 10, min_share: float = 0.05):
    """Quantile fine-classing, then merge adjacent bins until the bad rate is monotonic."""
    qs = np.unique(np.nanquantile(x, np.linspace(0, 1, max_bins + 1)))
    edges = [-np.inf] + list(qs[1:-1]) + [np.inf]
    while len(edges) > 2:
        t = _woe_table(pd.cut(x, edges, include_lowest=True), y)
        small = ((t["n"] / t["n"].sum()) < min_share).values
        diffs = np.diff(t["bad_rate"].values)
        if ((diffs >= 0).all() or (diffs <= 0).all()) and not small.any():
            break
        if small.any():
            i = int(np.argmax(small))
            j = i if i < len(edges) - 2 else i - 1
        else:
            j = int(np.argmin(np.abs(diffs)))
        del edges[j + 1]
    return edges


class WoEBinner:
    def __init__(self, max_bins: int = 10, min_share: float = 0.05, min_cat_share: float = 0.02):
        self.max_bins, self.min_share, self.min_cat_share = max_bins, min_share, min_cat_share
        self.spec: dict = {}

    def fit(self, X: pd.DataFrame, y: pd.Series, numeric: list, categorical: list):
        for c in numeric:
            med = X[c].median()
            x = X[c].fillna(med)
            edges = _monotonic_edges(x, y, self.max_bins, self.min_share)
            t = _woe_table(pd.cut(x, edges, include_lowest=True), y)
            self.spec[c] = {"type": "num", "edges": edges, "median": med, "table": t,
                            "map": dict(zip(t.index.astype(str), t["woe"]))}
        for c in categorical:
            x = X[c].fillna("MISSING").astype(str)
            share = x.value_counts(normalize=True)
            rare = set(share[share < self.min_cat_share].index)
            x = x.where(~x.isin(rare), "OTHER")
            t = _woe_table(x, y)
            self.spec[c] = {"type": "cat", "rare": rare, "table": t, "map": dict(zip(t.index, t["woe"]))}
        return self

    def iv(self) -> pd.Series:
        return pd.Series({c: s["table"]["iv"].sum() for c, s in self.spec.items()}).sort_values(ascending=False)

    def transform(self, X: pd.DataFrame, cols: list | None = None) -> pd.DataFrame:
        cols = cols or list(self.spec)
        out = {}
        for c in cols:
            s = self.spec[c]
            if s["type"] == "num":
                b = pd.cut(X[c].fillna(s["median"]), s["edges"], include_lowest=True).astype(str)
            else:
                b = X[c].fillna("MISSING").astype(str)
                b = b.where(~b.isin(s["rare"]), "OTHER")
            out[c] = b.map(s["map"]).fillna(0.0).astype(float)
        return pd.DataFrame(out, index=X.index)


# --------------------------------------------------------------------------- #
# WoE scorecard (sklearn-style: predict_proba on the raw feature frame)
# --------------------------------------------------------------------------- #
class WoEScorecard:
    def __init__(self, base_score=600, base_odds=50, pdo=20, iv_min=0.02, iv_review=0.5):
        self.base_score, self.base_odds, self.pdo = base_score, base_odds, pdo
        self.iv_min, self.iv_review = iv_min, iv_review
        self.factor = pdo / np.log(2)
        self.offset = base_score - self.factor * np.log(base_odds)

    def fit(self, X: pd.DataFrame, y: pd.Series, numeric: list, categorical: list):
        numeric = [c for c in numeric if c not in SCORECARD_EXCLUDE]
        categorical = [c for c in categorical if c not in SCORECARD_EXCLUDE]
        self.binner = WoEBinner().fit(X, y, numeric, categorical)
        self.iv_table = self.binner.iv()
        self.features = self.iv_table[self.iv_table >= self.iv_min].index.tolist()
        self.leakage_review = self.iv_table[self.iv_table > self.iv_review].index.tolist()
        W = self.binner.transform(X, self.features)
        self.lr = LogisticRegression(max_iter=1000).fit(W, y)  # no class weighting
        # WoE = ln(good/bad): every coefficient should be negative when predicting PD
        self.sign_flags = [f for f, b in zip(self.features, self.lr.coef_[0]) if b > 0]
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        p = self.lr.predict_proba(self.binner.transform(X, self.features))[:, 1]
        return np.column_stack([1 - p, p])

    def score(self, X: pd.DataFrame) -> pd.Series:
        p = np.clip(self.predict_proba(X)[:, 1], EPS, 1 - EPS)
        return pd.Series(np.round(self.offset + self.factor * np.log((1 - p) / p)), index=X.index, name="score")

    def points_table(self) -> pd.DataFrame:
        n, a = len(self.features), self.lr.intercept_[0]
        rows = []
        for f, beta in zip(self.features, self.lr.coef_[0]):
            t = self.binner.spec[f]["table"]
            for b, woe in t["woe"].items():
                pts = -(beta * woe + a / n) * self.factor + self.offset / n
                rows.append({"feature": f, "bin": str(b), "n": int(t.loc[b, "n"]),
                             "bad_rate": round(float(t.loc[b, "bad_rate"]), 4),
                             "woe": round(float(woe), 3), "points": round(float(pts), 1)})
        return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Validation metrics
# --------------------------------------------------------------------------- #
def ks_statistic(y, p) -> float:
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def psi(expected, actual, bins: int = 10) -> float:
    expected, actual = pd.Series(np.asarray(expected)), pd.Series(np.asarray(actual))
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    e = pd.cut(expected, edges).value_counts(normalize=True, sort=False) + EPS
    a = pd.cut(actual, edges).value_counts(normalize=True, sort=False) + EPS
    return float(((a - e) * np.log(a / e)).sum())


def calibration_table(y, p, bins: int = 10) -> pd.DataFrame:
    d = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)})
    d["band"] = pd.qcut(d["p"].rank(method="first"), bins, labels=[f"D{i}" for i in range(1, bins + 1)])
    t = d.groupby("band", observed=True).agg(n=("y", "size"), predicted_pd=("p", "mean"), observed_dr=("y", "mean"))
    return t.reset_index()


def _metrics(y, p, train_time=0.0) -> dict:
    auc = roc_auc_score(y, p)
    return {
        "AUC": float(auc), "Gini": float(2 * auc - 1), "KS": ks_statistic(y, p),
        "Recall": float(recall_score(y, (p >= 0.5).astype(int), zero_division=0)),
        "Brier": float(brier_score_loss(y, p)),
        "Predicted DR": float(np.mean(p)), "Observed DR": float(np.mean(y)),
        "Training Time (s)": float(train_time),
    }


def out_of_time_split(df: pd.DataFrame, date_col: str = "origination_date", oot_share: float = 0.25):
    """Train on older vintages, validate on the most recent ones."""
    order = pd.to_datetime(df[date_col], errors="coerce").sort_values(kind="mergesort").index
    cut = int(len(order) * (1 - oot_share))
    return order[:cut], order[cut:]


# --------------------------------------------------------------------------- #
# Champion / challenger training (API used by app.py and tests)
# --------------------------------------------------------------------------- #
def train_and_compare_models(df: pd.DataFrame, target_col: str = "default_flag", random_state: int = 42):
    df_eng, numeric, categorical = get_feature_columns(df)
    X = df_eng[numeric + categorical]
    y = df_eng[target_col].astype(int)

    tr_idx, oot_idx = out_of_time_split(df_eng)
    X_train, X_test, y_train, y_test = X.loc[tr_idx], X.loc[oot_idx], y.loc[tr_idx], y.loc[oot_idx]

    prep = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), numeric),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ])
    challengers = {
        "Logistic Regression": LogisticRegression(max_iter=1000),  # no class weighting -> calibrated
        "Random Forest": RandomForestClassifier(n_estimators=250, max_depth=8, min_samples_leaf=15,
                                                random_state=random_state, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(random_state=random_state),
    }

    results, fitted = {}, {}

    start = time.perf_counter()
    sc = WoEScorecard().fit(X_train, y_train, numeric, categorical)
    p = sc.predict_proba(X_test)[:, 1]
    results["WoE Scorecard"] = _metrics(y_test, p, time.perf_counter() - start)
    results["WoE Scorecard"]["PSI"] = psi(sc.score(X_train), sc.score(X_test))
    fitted["WoE Scorecard"] = sc

    for name, est in challengers.items():
        pipe = Pipeline([("prep", prep), ("model", est)])
        start = time.perf_counter()
        pipe.fit(X_train, y_train)
        p_tr, p = pipe.predict_proba(X_train)[:, 1], pipe.predict_proba(X_test)[:, 1]
        results[name] = _metrics(y_test, p, time.perf_counter() - start)
        results[name]["PSI"] = psi(p_tr, p)
        fitted[name] = pipe

    metrics = pd.DataFrame(results).T.sort_values("Gini", ascending=False)
    return fitted, metrics, X_test, y_test


def score_portfolio(model, df: pd.DataFrame) -> pd.Series:
    engineered, numeric, categorical = get_feature_columns(df)
    X = engineered[numeric + categorical]
    return pd.Series(model.predict_proba(X)[:, 1], index=df.index, name="pd_12m")


def approximate_origination_pd(current_pd: pd.Series, months_on_book: pd.Series) -> pd.Series:
    """Simplified proxy: back out an origination PD from current PD using a seasoning factor.
    In production this would come from the PD recorded at origination."""
    improvement = 1 + (months_on_book.clip(lower=1) / 120.0)
    return (current_pd / improvement).clip(0.0001, 0.999)


def build_feature_importance(model, df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(model, WoEScorecard):
        iv = model.iv_table.loc[model.features]
        return pd.DataFrame({"feature": iv.index, "importance": iv.values}).reset_index(drop=True)
    prep, step = model.named_steps["prep"], model.named_steps["model"]
    names = prep.get_feature_names_out()
    if hasattr(step, "coef_"):
        raw = np.abs(step.coef_[0])
    elif hasattr(step, "feature_importances_"):
        raw = step.feature_importances_
    else:
        raw = np.zeros(len(names))
    out = pd.DataFrame({"feature": names, "importance": raw}).sort_values("importance", ascending=False)
    return out.head(15).reset_index(drop=True)

