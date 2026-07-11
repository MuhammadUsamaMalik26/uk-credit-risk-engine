
import time
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_FEATURES = [
    "loan_amount", "current_balance", "term_months", "months_on_book", "interest_rate",
    "annual_income", "debt_to_income", "credit_score"
]
CATEGORICAL_FEATURES = ["grade_band", "home_ownership", "purpose"]

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

def _build_preprocessor(numeric_features, categorical_features):
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

def train_and_compare_models(df: pd.DataFrame, target_col: str = "default_flag", random_state: int = 42):
    df_eng, numeric_features, categorical_features = get_feature_columns(df)
    X = df_eng[numeric_features + categorical_features]
    y = df_eng[target_col].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=random_state, stratify=y
    )

    preprocessor = _build_preprocessor(numeric_features, categorical_features)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=250, max_depth=8, min_samples_leaf=15, random_state=random_state, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(random_state=random_state),
    }

    results = {}
    fitted = {}

    for name, estimator in models.items():
        pipe = Pipeline(steps=[("prep", preprocessor), ("model", estimator)])
        start = time.perf_counter()
        pipe.fit(X_train, y_train)
        train_time = time.perf_counter() - start

        proba = pipe.predict_proba(X_test)[:, 1]
        pred = (proba >= 0.5).astype(int)
        auc = roc_auc_score(y_test, proba)
        recall = recall_score(y_test, pred, zero_division=0)
        brier = brier_score_loss(y_test, proba)

        results[name] = {
            "AUC": float(auc),
            "Recall": float(recall),
            "Brier": float(brier),
            "Training Time (s)": float(train_time),
        }
        fitted[name] = pipe

    metrics = pd.DataFrame(results).T.sort_values("AUC", ascending=False)
    return fitted, metrics, X_test, y_test

def score_portfolio(model, df: pd.DataFrame) -> pd.Series:
    engineered, numeric_features, categorical_features = get_feature_columns(df)
    X = engineered[numeric_features + categorical_features]
    return pd.Series(model.predict_proba(X)[:, 1], index=df.index, name="pd_12m")

def approximate_origination_pd(current_pd: pd.Series, months_on_book: pd.Series) -> pd.Series:
    improvement = 1 + (months_on_book.clip(lower=1) / 120.0)
    return (current_pd / improvement).clip(0.0001, 0.999)

def build_feature_importance(model, df: pd.DataFrame) -> pd.DataFrame:
    engineered, numeric_features, categorical_features = get_feature_columns(df)
    prep = model.named_steps["prep"]
    model_step = model.named_steps["model"]
    feature_names = prep.get_feature_names_out()
    if hasattr(model_step, "coef_"):
        raw = np.abs(model_step.coef_[0])
    elif hasattr(model_step, "feature_importances_"):
        raw = model_step.feature_importances_
    else:
        raw = np.zeros(len(feature_names))
    out = pd.DataFrame({"feature": feature_names, "importance": raw}).sort_values("importance", ascending=False)
    return out.head(15).reset_index(drop=True)
