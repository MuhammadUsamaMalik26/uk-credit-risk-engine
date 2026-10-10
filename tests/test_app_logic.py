
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.load_data import load_portfolio
from data.macro_loader import get_boe_scenarios
from models.scorecard import (
    train_and_compare_models, score_portfolio, approximate_origination_pd, WoEScorecard, psi
)
from models.ecl_engine import run_scenarios, probability_weighted_summary

def test_load_portfolio():
    df, meta = load_portfolio(n=300)
    assert len(df) >= 300
    assert "loan_amount" in df.columns
    assert meta["source"] in {"Synthetic fallback", "LendingClub (real)", "synthetic_csv"}

def test_model_training_and_scoring():
    df, _ = load_portfolio(n=500)
    models, metrics, _, _ = train_and_compare_models(df)
    assert len(models) == 4
    assert "WoE Scorecard" in models
    assert metrics["AUC"].max() > 0.55

    best_model = models[metrics.index[0]]
    pd_12m = score_portfolio(best_model, df)
    assert ((pd_12m >= 0) & (pd_12m <= 1)).all()

def test_ecl_pipeline():
    df, _ = load_portfolio(n=350)
    models, metrics, _, _ = train_and_compare_models(df)
    best_model = models[metrics.index[0]]
    pd_12m = score_portfolio(best_model, df)
    orig_pd = approximate_origination_pd(pd_12m, df["months_on_book"])
    scenarios = get_boe_scenarios()
    results = run_scenarios(df, pd_12m, orig_pd, scenarios, sicr_threshold=2.0)
    assert set(results["stage"].unique()).issubset({1,2,3})
    assert (results["ecl"] >= 0).all()
    summary = probability_weighted_summary(results)
    assert summary.attrs["weighted_ecl"] > 0


def test_scorecard_build():
    df, _ = load_portfolio(n=1500)
    models, metrics, X_test, y_test = train_and_compare_models(df)
    sc = models["WoE Scorecard"]
    assert isinstance(sc, WoEScorecard)
    assert "interest_rate" not in sc.features and "grade_band" not in sc.features  # no circular features
    assert not sc.sign_flags  # all coefficients have the expected sign
    # monotonic bad rate within every numeric WoE bin set
    for f in sc.features:
        spec = sc.binner.spec[f]
        if spec["type"] == "num":
            d = np.diff(spec["table"]["bad_rate"].values)
            assert (d >= 0).all() or (d <= 0).all()
    # calibrated: predicted vs observed default rate within 2 percentage points out-of-time
    m = metrics.loc["WoE Scorecard"]
    assert abs(m["Predicted DR"] - m["Observed DR"]) < 0.02
    assert m["Gini"] > 0.2
    # higher score = lower risk
    scores = sc.score(X_test)
    pds = sc.predict_proba(X_test)[:, 1]
    assert np.corrcoef(scores, pds)[0, 1] < 0
    assert not sc.points_table().empty


def test_psi_stable_for_same_distribution():
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=5000), rng.normal(size=5000)
    assert psi(a, b) < 0.1
    assert psi(a, b + 1.0) > 0.25
