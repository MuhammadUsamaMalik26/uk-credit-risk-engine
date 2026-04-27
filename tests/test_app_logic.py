
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.load_data import load_portfolio
from data.macro_loader import get_boe_scenarios
from models.scorecard import train_and_compare_models, score_portfolio, approximate_origination_pd
from models.ecl_engine import run_scenarios, probability_weighted_summary

def test_load_portfolio():
    df, meta = load_portfolio(n=300)
    assert len(df) == 300
    assert "loan_amount" in df.columns
    assert meta["source"] in {"Synthetic fallback", "LendingClub (real)"}

def test_model_training_and_scoring():
    df, _ = load_portfolio(n=500)
    models, metrics, _, _ = train_and_compare_models(df)
    assert len(models) == 3
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
