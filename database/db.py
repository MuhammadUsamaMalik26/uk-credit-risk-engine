
import sqlite3
import pandas as pd

def get_connection(db_path="risk_app.db"):
    return sqlite3.connect(db_path, check_same_thread=False)

def init_db(conn):
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS loans_raw (
        loan_id TEXT PRIMARY KEY,
        origination_date TEXT,
        reporting_date TEXT,
        loan_amount REAL,
        current_balance REAL,
        term_months INTEGER,
        months_on_book INTEGER,
        interest_rate REAL,
        annual_income REAL,
        debt_to_income REAL,
        credit_score INTEGER,
        missed_payments INTEGER,
        days_past_due INTEGER,
        grade_band TEXT,
        home_ownership TEXT,
        purpose TEXT,
        loan_status TEXT,
        default_flag INTEGER,
        stage_at_source INTEGER
    );

    CREATE TABLE IF NOT EXISTS macro_scenarios (
        scenario_name TEXT PRIMARY KEY,
        unemployment_rate REAL,
        gdp_growth REAL,
        bank_rate REAL,
        inflation_cpi REAL,
        scenario_weight REAL
    );

    CREATE TABLE IF NOT EXISTS model_runs (
        run_id TEXT PRIMARY KEY,
        run_ts TEXT,
        data_source TEXT,
        model_name TEXT,
        sicr_threshold REAL,
        unemployment_shift REAL,
        gdp_shift REAL,
        rate_shift_bps REAL,
        inflation_shift REAL
    );

    CREATE TABLE IF NOT EXISTS loan_results (
        run_id TEXT,
        loan_id TEXT,
        scenario_name TEXT,
        stage INTEGER,
        ead REAL,
        lgd REAL,
        pd_12m REAL,
        lifetime_pd REAL,
        ecl REAL,
        sicr_ratio REAL,
        baseline_pd_12m REAL,
        origination_pd_12m REAL
    );

    CREATE TABLE IF NOT EXISTS model_metrics (
        run_id TEXT,
        model_name TEXT,
        auc REAL,
        recall REAL,
        brier REAL,
        training_time_s REAL
    );
    """)

def persist_frame(conn, df: pd.DataFrame, table_name: str, if_exists="replace"):
    df.to_sql(table_name, conn, if_exists=if_exists, index=False)

def create_views(conn):
    conn.executescript("""
    DROP VIEW IF EXISTS vw_portfolio_summary;
    CREATE VIEW vw_portfolio_summary AS
    SELECT
        run_id,
        scenario_name,
        COUNT(*) AS loan_count,
        SUM(ead) AS total_exposure,
        SUM(ecl) AS total_ecl,
        AVG(pd_12m) AS avg_pd_12m,
        AVG(lifetime_pd) AS avg_lifetime_pd
    FROM loan_results
    GROUP BY run_id, scenario_name;

    DROP VIEW IF EXISTS vw_stage_distribution;
    CREATE VIEW vw_stage_distribution AS
    SELECT
        run_id,
        scenario_name,
        stage,
        COUNT(*) AS loan_count,
        SUM(ecl) AS total_ecl,
        SUM(ead) AS total_ead
    FROM loan_results
    GROUP BY run_id, scenario_name, stage;

    DROP VIEW IF EXISTS vw_segment_ecl;
    CREATE VIEW vw_segment_ecl AS
    SELECT
        lr.run_id,
        lr.scenario_name,
        l.grade_band,
        COUNT(*) AS loan_count,
        SUM(lr.ecl) AS total_ecl,
        AVG(lr.pd_12m) AS avg_pd
    FROM loan_results lr
    JOIN loans_raw l
      ON l.loan_id = lr.loan_id
    GROUP BY lr.run_id, lr.scenario_name, l.grade_band;

    DROP VIEW IF EXISTS vw_model_metrics;
    CREATE VIEW vw_model_metrics AS
    SELECT * FROM model_metrics;
    """)

def query_df(conn, sql: str) -> pd.DataFrame:
    return pd.read_sql_query(sql, conn)
