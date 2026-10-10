
import uuid
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.calibration import calibration_curve

from data.load_data import load_portfolio
from data.macro_loader import get_boe_scenarios, apply_sidebar_shocks
from models.scorecard import (
    train_and_compare_models, score_portfolio, approximate_origination_pd, build_feature_importance,
    WoEScorecard, calibration_table,
)
from models.ecl_engine import run_scenarios, probability_weighted_summary
from models.basel_lgd import icaap_waterfall, fca_vulnerability_flags
from database.db import get_connection, init_db, persist_frame, create_views, query_df

st.set_page_config(page_title="Credit Risk Stress-Testing & IFRS 9 Simulator", layout="wide")
st.title("Credit Risk Stress-Testing & IFRS 9 Simulator")
st.caption(
    "Bank-style unsecured lending simulator using LendingClub-ready data ingestion, "
    "BoE-style macro scenarios, IFRS 9 SICR staging, discounted ECL, and SQL-backed analytics."
)

with st.sidebar:
    st.header("Control Room")
    selected_model = st.selectbox("Model selector", ["WoE Scorecard", "Logistic Regression", "Random Forest", "Gradient Boosting"])
    unemployment_shift = st.slider("Unemployment rate shock (%)", -1.0, 4.0, 0.0, 0.25)
    gdp_shift = st.slider("GDP growth shock (%)", -4.0, 2.0, 0.0, 0.25)
    rate_shift_bps = st.slider("Bank Rate shock (bps)", -150, 250, 0, 25)
    inflation_shift = st.slider("Inflation shock (%)", -1.0, 3.0, 0.0, 0.25)
    sicr_threshold = st.slider("SICR sensitivity (x deterioration)", 1.1, 4.0, 2.0, 0.1)
    n_rows = st.slider("Rows to load", 500, 4000, 1500, 250)
    run_button = st.button("Run simulation", type="primary", use_container_width=True)

conn = get_connection("risk_app.db")
init_db(conn)
create_views(conn)

if run_button or "run_id" not in st.session_state:
    df, meta = load_portfolio(n=n_rows)
    scenarios = apply_sidebar_shocks(
        get_boe_scenarios(),
        unemployment_shift=unemployment_shift,
        gdp_shift=gdp_shift,
        rate_shift_bps=rate_shift_bps,
        inflation_shift=inflation_shift,
    )

    models, metrics, X_test, y_test = train_and_compare_models(df)
    model = models[selected_model]
    selected_proba = model.predict_proba(X_test)[:, 1]
    prob_true, prob_pred = calibration_curve(y_test, selected_proba, n_bins=10, strategy="uniform")
    calibration_df = pd.DataFrame({"predicted_pd": prob_pred, "actual_default_rate": prob_true})

    pd_12m = score_portfolio(model, df)
    orig_pd = approximate_origination_pd(pd_12m, df["months_on_book"])
    scenario_results = run_scenarios(df, pd_12m, orig_pd, scenarios, sicr_threshold=sicr_threshold)
    summary = probability_weighted_summary(scenario_results)

    run_id = str(uuid.uuid4())[:8]
    st.session_state["run_id"] = run_id
    st.session_state["df"] = df
    st.session_state["meta"] = meta
    st.session_state["metrics"] = metrics
    st.session_state["feature_importance"] = build_feature_importance(model, df)
    st.session_state["scenario_results"] = scenario_results
    st.session_state["scenario_summary"] = summary
    st.session_state["scenarios"] = scenarios
    st.session_state["selected_model"] = selected_model
    st.session_state["sicr_threshold"] = sicr_threshold
    st.session_state["calibration_df"] = calibration_df
    st.session_state["scorecard"] = models["WoE Scorecard"]
    st.session_state["scorecard_calibration"] = calibration_table(
        y_test, models["WoE Scorecard"].predict_proba(X_test)[:, 1]
    )

    persist_frame(conn, df, "loans_raw")
    persist_frame(conn, scenarios, "macro_scenarios")
    persist_frame(conn, scenario_results.assign(run_id=run_id), "loan_results")
    model_metrics = (
        metrics.reset_index()
        .rename(
            columns={
                "index": "model_name",
                "Training Time (s)": "training_time_s",
                "Recall": "recall",
                "AUC": "auc",
                "Brier": "brier",
            }
        )
    )
    model_metrics["run_id"] = run_id
    persist_frame(
        conn,
        model_metrics[["run_id", "model_name", "auc", "recall", "brier", "training_time_s"]],
        "model_metrics",
    )
    create_views(conn)

df = st.session_state["df"]
meta = st.session_state["meta"]
metrics = st.session_state["metrics"]
feature_importance = st.session_state["feature_importance"]
scenario_results = st.session_state["scenario_results"]
scenario_summary = st.session_state["scenario_summary"]
scenarios = st.session_state["scenarios"]
selected_model = st.session_state["selected_model"]
calibration_df = st.session_state["calibration_df"]

st.sidebar.success(f"Data source: {meta['source']}")
st.sidebar.info(f"Current run: {st.session_state['run_id']}")

tabs = st.tabs(
    ["Portfolio Health", "Interactive Stress Test", "Model Benchmarking", "Basel III Capital & ICAAP", "FCA Consumer Duty", "Methodology & SQL Analytics"]
)

with tabs[0]:
    base = scenario_results[scenario_results["scenario_name"] == "Base"]
    weighted_ecl = scenario_summary.attrs["weighted_ecl"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Portfolio Exposure", f"£{base['ead'].sum():,.0f}")
    c2.metric("Probability-weighted ECL", f"£{weighted_ecl:,.0f}")
    c3.metric("Portfolio 12m PD", f"{base['pd_12m'].mean()*100:.2f}%")
    c4.metric("Avg Lifetime PD", f"{base['lifetime_pd'].mean()*100:.2f}%")

    stage_counts = base["stage"].value_counts().sort_index().rename_axis("Stage").reset_index(name="Loans")
    fig_stage = px.pie(stage_counts, names="Stage", values="Loans", hole=0.55, title="Stage distribution — Base scenario")
    st.plotly_chart(fig_stage, use_container_width=True)

    seg = query_df(
        conn,
        f"""
        SELECT grade_band, total_ecl, avg_pd
        FROM vw_segment_ecl
        WHERE run_id = '{st.session_state["run_id"]}' AND scenario_name = 'Base'
        ORDER BY total_ecl DESC
        """,
    )
    if not seg.empty:
        st.plotly_chart(
            px.bar(seg, x="grade_band", y="total_ecl", title="ECL by grade band (SQL view)"),
            use_container_width=True,
        )

with tabs[1]:
    comp = scenario_summary.copy()
    fig = px.bar(comp, x="scenario_name", y="total_ecl", title="ECL by scenario", text_auto=".2s")
    st.plotly_chart(fig, use_container_width=True)

    base_ecl = float(comp.loc[comp["scenario_name"] == "Base", "total_ecl"].iloc[0])
    stress_ecl = float(comp.loc[comp["scenario_name"] == "Stress", "total_ecl"].iloc[0])
    uplift = (stress_ecl / base_ecl - 1) * 100 if base_ecl else 0

    base_stage = scenario_results[scenario_results["scenario_name"] == "Base"][["loan_id", "stage"]].rename(columns={"stage": "base_stage"})
    stress_stage = scenario_results[scenario_results["scenario_name"] == "Stress"][["loan_id", "stage"]].rename(columns={"stage": "stress_stage"})
    migration = base_stage.merge(stress_stage, on="loan_id")
    migration_table = migration.groupby(["base_stage", "stress_stage"]).size().reset_index(name="loan_count")
    stage12_migration = migration_table.loc[
        (migration_table["base_stage"] == 1) & (migration_table["stress_stage"] == 2), "loan_count"
    ].sum()

    st.info(
        f"In this stress setup, Stress ECL is {uplift:.1f}% above Base. "
        f"{int(stage12_migration):,} loans migrate from Stage 1 to Stage 2, "
        "driving more accounts onto lifetime ECL through SICR and delinquency backstops."
    )
    st.dataframe(migration_table, use_container_width=True)

with tabs[2]:
    st.subheader("Champion / challenger — out-of-time validation")
    st.caption(
        "Models are trained on older vintages and validated on the most recent 25% of originations "
        "(out-of-time). PSI compares the score distribution between the two samples (< 0.10 = stable)."
    )
    st.dataframe(
        metrics.style.format(
            {"AUC": "{:.3f}", "Gini": "{:.3f}", "KS": "{:.3f}", "Recall": "{:.3f}", "Brier": "{:.4f}",
             "Predicted DR": "{:.2%}", "Observed DR": "{:.2%}", "PSI": "{:.3f}", "Training Time (s)": "{:.2f}"}
        ),
        use_container_width=True,
    )

    st.plotly_chart(
        px.bar(feature_importance.sort_values("importance"), x="importance", y="feature", orientation="h",
               title=f"Feature importance — {selected_model}"),
        use_container_width=True,
    )

    cal_fig = go.Figure()
    cal_fig.add_trace(
        go.Scatter(
            x=calibration_df["predicted_pd"],
            y=calibration_df["actual_default_rate"],
            mode="lines+markers",
            name="Model",
        )
    )
    cal_fig.add_trace(
        go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect calibration")
    )
    cal_fig.update_layout(title=f"Calibration curve — {selected_model}", xaxis_title="Predicted PD", yaxis_title="Observed default rate")
    st.plotly_chart(cal_fig, use_container_width=True)

    scorecard = st.session_state["scorecard"]
    st.subheader("WoE Scorecard — build and validation")
    st.caption(
        f"Points scaling: {scorecard.base_score} points at {scorecard.base_odds}:1 good:bad odds, "
        f"PDO = {scorecard.pdo}. Lender-assigned grade and interest rate are excluded to avoid circularity; "
        "no class weighting, so PDs stay calibrated to observed default rates."
    )
    c1, c2 = st.columns(2)
    iv_df = scorecard.iv_table.rename("IV").reset_index().rename(columns={"index": "feature"})
    iv_df["selected"] = iv_df["feature"].isin(scorecard.features)
    c1.markdown("**Information Value (feature selection, IV ≥ 0.02)**")
    c1.dataframe(iv_df.style.format({"IV": "{:.3f}"}), use_container_width=True)
    c2.markdown("**Out-of-time calibration by PD decile**")
    c2.dataframe(
        st.session_state["scorecard_calibration"].style.format({"predicted_pd": "{:.2%}", "observed_dr": "{:.2%}"}),
        use_container_width=True,
    )
    if scorecard.leakage_review:
        st.warning(f"IV > 0.5 — reviewed for leakage: {', '.join(scorecard.leakage_review)}")
    if scorecard.sign_flags:
        st.warning(f"Unexpected coefficient sign: {', '.join(scorecard.sign_flags)}")
    else:
        st.success("All scorecard coefficients have the expected sign; WoE bins are monotonic in bad rate.")
    st.markdown("**Scorecard points table**")
    st.dataframe(scorecard.points_table(), use_container_width=True)

with tabs[3]:
    st.subheader('Basel III AIRB Capital & ICAAP Waterfall')
    base_capital = scenario_results[scenario_results['scenario_name'] == 'Base']
    stress_capital = scenario_results[scenario_results['scenario_name'] == 'Stress']

    if 'rwa' in base_capital.columns:
        total_rwa_base = base_capital['rwa'].sum()
        total_rwa_stress = stress_capital['rwa'].sum()
        total_cet1_base = base_capital['cet1_required'].sum()

        c1, c2, c3 = st.columns(3)
        c1.metric('Total RWA (Base)', f'£{total_rwa_base:,.0f}')
        c2.metric('Total RWA (Stress)', f'£{total_rwa_stress:,.0f}', delta=f'{((total_rwa_stress/total_rwa_base)-1)*100:.1f}%')
        c3.metric('CET1 Required (Base)', f'£{total_cet1_base:,.0f}')

        st.subheader('CET1 Waterfall — ICAAP')
        icaap = icaap_waterfall(total_rwa_base)
        waterfall_data = {
            'Component': ['CET1 Capital', 'Less: Pillar 1', 'Less: Pillar 2A', 'Less: Buffer', 'Post-Stress CET1'],
            'Amount': [icaap['cet1_capital'], -icaap['pillar1_requirement'], -icaap['pillar2a_requirement'], -icaap['conservation_buffer'], icaap['post_stress_cet1']]
        }
        fig_wf = px.bar(pd.DataFrame(waterfall_data), x='Component', y='Amount', title='CET1 Capital Waterfall', color='Amount', color_continuous_scale='RdYlGn')
        st.plotly_chart(fig_wf, use_container_width=True)
        stress_result = '✅ PASSES' if icaap['passes_stress'] else '❌ FAILS'
        st.info(f'Post-stress CET1 ratio: {icaap["post_stress_ratio"]*100:.2f}% — Minimum 4.5% — {stress_result}')

        st.subheader('RWA by Scenario')
        rwa_summary = scenario_results.groupby('scenario_name')['rwa'].sum().reset_index()
        st.plotly_chart(px.bar(rwa_summary, x='scenario_name', y='rwa', title='Total RWA by Scenario', text_auto='.2s'), use_container_width=True)
    else:
        st.info('Run simulation to see Basel capital results.')

with tabs[4]:
    st.subheader('FCA Consumer Duty — Vulnerability Dashboard (FG21/1)')
    fca_df = fca_vulnerability_flags(df)
    total = len(fca_df)
    vulnerable = fca_df['vulnerable_indicator'].sum()
    hard_rejects = fca_df['fca_hard_reject'].sum()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric('Total Borrowers', f'{total:,}')
    c2.metric('Vulnerable Indicators', f'{vulnerable:,}', delta=f'{vulnerable/total*100:.1f}%')
    c3.metric('FCA Hard Rejects', f'{hard_rejects:,}', delta=f'{hard_rejects/total*100:.1f}%')
    c4.metric('DTI Breaches', f"{fca_df['flag_dti_breach'].sum():,}")

    flag_cols = ['flag_dti_breach','flag_high_rate_vulnerable','flag_low_financial_literacy','flag_payment_distress']
    flag_labels = ['DTI Breach (>45%)','High Rate + Weak Credit','Low Financial Literacy Proxy','Payment Distress']
    flag_counts = [fca_df[c].sum() for c in flag_cols]
    fig_fca = px.bar(x=flag_labels, y=flag_counts, title='FCA Consumer Duty — Outcome Flag Counts', labels={'x':'Outcome Driver','y':'Borrower Count'})
    st.plotly_chart(fig_fca, use_container_width=True)
    st.subheader('Vulnerable Borrower Sample')
    st.dataframe(fca_df[fca_df['vulnerable_indicator']==1][['loan_id','credit_score','debt_to_income','interest_rate','days_past_due','vulnerable_indicator','fca_hard_reject']].head(20), use_container_width=True)

with tabs[5]:
    st.subheader("Dataset used")
    st.write(
        "This app ingests a LendingClub-style consumer lending portfolio when a real CSV is present in the "
        "`data/` folder. If no file is available, it falls back to a calibrated synthetic portfolio with the same schema."
    )

    st.subheader("Default definition")
    st.write(
        "Default is proxied using LendingClub-style loan statuses. Stage 3 / default loans include statuses such as "
        "`Charged Off` and `Default`. Late statuses are used as delinquency proxies for 30+ and 60+ DPD staging backstops."
    )

    st.subheader("Assumptions")
    st.markdown(
        """
        - **PD model**: 12-month behavioural PD from a WoE logistic regression scorecard (monotonic binning, IV selection, PDO points scaling), benchmarked against Logistic Regression, Random Forest and Gradient Boosting challengers.
        - **Validation**: out-of-time split on origination date; AUC, Gini, KS, Brier, PSI and decile calibration. No class weighting, so PDs remain calibrated for ECL and capital.
        - **SICR logic**: Stage 2 is triggered when lifetime PD deterioration breaches the selected SICR threshold or when 30+ DPD backstops are hit.
        - **EAD**: Exposure is approximated with an amortizing balance path over the remaining term.
        - **LGD**: LGD is stage-based and stressed upward in adverse and stress scenarios.
        - **Discounting**: Losses are discounted monthly using the contractual interest rate as a proxy for EIR.
        - **Macro design**: BoE-style macro variables drive scenario overlays: unemployment, GDP growth, Bank Rate, and CPI inflation.
        """
    )

    st.subheader("Limitations")
    st.markdown(
        """
        - This is a **bank-style simulator**, not a production or fully compliant IFRS 9 engine.
        - Lifetime PD is approximated from 12-month PD rather than estimated from a full term-structure model.
        - Macroeconomic linkages are simplified overlays, not a fully econometric satellite model.
        - LGD and EAD are simplified for demonstration and interview use.
        """
    )

    st.subheader("SQL-backed analytics")
    st.write(
        "SQL is a core layer in this project. SQLite stores the raw portfolio, scenario inputs, model metrics, "
        "and loan-level scenario results. The dashboard then reads SQL views for portfolio summary, segmentation, and benchmarking."
    )
    sql_summary = query_df(conn, f"SELECT * FROM vw_portfolio_summary WHERE run_id = '{st.session_state['run_id']}'")
    st.dataframe(sql_summary, use_container_width=True)
    sql_metrics = query_df(
        conn,
        f"SELECT model_name, auc, recall, brier, training_time_s FROM vw_model_metrics WHERE run_id = '{st.session_state['run_id']}'",
    )
    st.dataframe(sql_metrics, use_container_width=True)
