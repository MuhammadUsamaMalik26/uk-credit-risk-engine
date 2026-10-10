"""PD Scorecard & Model Validation.

Champion: WoE logistic regression scorecard with monotonic binning, information
value feature selection and points-to-double-odds scaling. Challengers include
raw-feature logistic regression, random forest and gradient boosting.
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
NUMERIC_FEATURES = ["loan_amount", "current_balance", "term_months", "months_on_book",
                    "interest_rate", "annual_income", "debt_to_income", "credit_score"]
CATEGORICAL_FEATURES = ["grade_band", "home_ownership", "purpose"]
SCORECARD_EXCLUDE = {"interest_rate", "grade_band", "rate_minus_score"}


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
            self.spec[c] = {"type": "cat", "rare": rare, "table": t,
                            "map": dict(zip(t.index, t["woe"]))}
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
        self.lr = LogisticRegression(max_iter=1000).fit(W, y)
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
    return {"AUC": float(auc), "Gini": float(2 * auc - 1), "KS": ks_statistic(y, p),
            "Recall": float(recall_score(y, np.asarray(p) >= 0.5, zero_division=0)),
            "Brier": float(brier_score_loss(y, p)), "Training Time (s)": round(train_time, 2),
            "Predicted DR": float(np.mean(p)), "Observed DR": float(np.mean(y))}


def _oot_split(df: pd.DataFrame, test_size: float = 0.25):
    d = df.sort_values("origination_date").reset_index(drop=True)
    split = int(len(d) * (1 - test_size))
    return d.iloc[:split].copy(), d.iloc[split:].copy()


def train_and_benchmark(df: pd.DataFrame):
    """Train champion & challenger models with chronological OOT validation."""
    train, test = _oot_split(df)
    tr, numeric, categorical = get_feature_columns(train)
    te, _, _ = get_feature_columns(test)
    X_tr, y_tr = tr[numeric + categorical], train["default_flag"].astype(int)
    X_te, y_te = te[numeric + categorical], test["default_flag"].astype(int)
    results = {}
    predictions = {}
    fitted = {}

    start = time.perf_counter()
    champion = WoEScorecard().fit(X_tr, y_tr, numeric, categorical)
    elapsed = time.perf_counter() - start
    p_tr, p_te = champion.predict_proba(X_tr)[:, 1], champion.predict_proba(X_te)[:, 1]
    results["WoE Scorecard"] = dict(_metrics(y_te, p_te, elapsed), PSI=psi(p_tr, p_te))
    predictions["WoE Scorecard"] = p_te
    fitted["WoE Scorecard"] = champion

    preprocessor = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                          ("scaler", StandardScaler())]), numeric),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ])
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=100, min_samples_leaf=10, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
    }
    for name, estimator in models.items():
        model = Pipeline([("preprocessor", preprocessor), ("model", estimator)])
        start = time.perf_counter()
        model.fit(X_tr, y_tr)
        elapsed = time.perf_counter() - start
        p_tr, p_te = model.predict_proba(X_tr)[:, 1], model.predict_proba(X_te)[:, 1]
        results[name] = dict(_metrics(y_te, p_te, elapsed), PSI=psi(p_tr, p_te))
        predictions[name] = p_te
        fitted[name] = model

    metrics = pd.DataFrame.from_dict(results, orient="index")
    metrics.index.name = "model_name"
    return metrics, fitted, predictions, (train, test)


def _importance_from_model(model, name: str, X: pd.DataFrame):
    if name == "WoE Scorecard":
        return pd.DataFrame({"feature": model.features, "importance": abs(model.lr.coef_[0])})
    pre = model.named_steps["preprocessor"]
    estimator = model.named_steps["model"]
    features = pre.get_feature_names_out()
    if hasattr(estimator, "feature_importances_"):
        imp = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        imp = abs(estimator.coef_[0])
    else:
        imp = np.zeros(len(features))
    return pd.DataFrame({"feature": features, "importance": imp})


def run_model_benchmark(df: pd.DataFrame):
    metrics, fitted, predictions, (train, test) = train_and_benchmark(df)
    champion = fitted["WoE Scorecard"]
    features_test, _, _ = get_feature_columns(test)
    pd_12m = champion.predict_proba(get_feature_columns(df)[0])[:, 1]
    importance = {name: _importance_from_model(model, name, features_test) for name, model in fitted.items()}
    cal = calibration_table(test["default_flag"], predictions["WoE Scorecard"])
    return metrics, pd_12m, importance, cal, champion
