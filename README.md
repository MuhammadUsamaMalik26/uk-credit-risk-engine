# Credit Risk Stress-Testing & IFRS 9 Simulator

A bank-grade UK retail credit risk platform built for interview demonstration and portfolio use.

## Live Demo
🔗 **Deploy link** — add your Streamlit Cloud URL here after deployment

## What This Project Does

End-to-end credit risk modelling pipeline covering:

- **IFRS 9 ECL** — Three-stage classification with SICR logic, lifetime PD, discounted cashflow ECL
- **Dynamic LGD** — Driven by loan type, ONS HPI movement, IFRS 9 stage, and macro scenario
- **Basel III AIRB** — Retail asset correlation (BCBS §328), capital requirement K, RWA, Basel 3.1 output floor
- **ICAAP Waterfall** — Pillar 1, Pillar 2A, conservation buffer, post-stress CET1 ratio
- **FCA Consumer Duty** — FG21/1 four-outcome vulnerability framework with hard reject engine
- **PD Scorecard** — WoE logistic regression scorecard (monotonic binning, IV selection, PDO points scaling) as champion vs Logistic Regression, Random Forest and Gradient Boosting challengers, validated out-of-time (AUC, Gini, KS, Brier, PSI, decile calibration)
- **SQL Analytics** — SQLite backend with portfolio views, segment ECL, model metrics
- **Macro Scenarios** — BoE-style Base, Adverse, Stress with interactive sidebar shocks

## Key Metrics
Out-of-time validation on the bundled 5,000-loan portfolio (latest 25% of originations held out):

| Model | AUC | Gini | KS | Brier | PSI |
|---|---|---|---|---|---|
| WoE Scorecard (champion) | 0.70 | 0.41 | 0.30 | 0.030 | 0.01 |
| Logistic Regression | 0.72 | 0.43 | 0.36 | 0.029 | 0.01 |
| Random Forest | 0.72 | 0.44 | 0.41 | 0.029 | 0.01 |
| Gradient Boosting | 0.67 | 0.33 | 0.28 | 0.033 | 0.01 |

The scorecard is kept as champion: within ~0.03 Gini of the best challenger while excluding lender-assigned grade and interest rate (circular features), fully interpretable via a points table, and calibrated (predicted 3.6% vs observed 3.1% default rate out-of-time).

- Stress ECL uplift: **~70%** (Base → Stress, default sidebar settings)
- Basel 3.1 output floor applied at **72.5% of SA RWA**

## Project Structure

```
app.py                  # Streamlit dashboard — 6 interactive tabs
models/
  ecl_engine.py         # IFRS 9 staging, PD, ECL calculation
  scorecard.py          # WoE scorecard, PD challengers, out-of-time validation (KS, PSI, calibration)
  basel_lgd.py          # Dynamic LGD, Basel AIRB capital, ICAAP, FCA
data/
  load_data.py          # LendingClub ingestion + synthetic fallback
  macro_loader.py       # BoE macro scenarios
database/
  db.py                 # SQLite persistence and SQL views
tests/
  test_app_logic.py     # Unit tests
```

## Dashboard Tabs

| Tab | Contents |
|-----|----------|
| Portfolio Health | Exposure, ECL, stage distribution, grade-band SQL view |
| Interactive Stress Test | Scenario comparison, stage migration matrix |
| Model Benchmarking | OOT AUC/Gini/KS/Brier/PSI, calibration, IV table, scorecard points table |
| Basel III Capital & ICAAP | RWA by scenario, CET1 waterfall, stress test pass/fail |
| FCA Consumer Duty | FG21/1 vulnerability flags, hard reject engine |
| Methodology & SQL | Assumptions, limitations, live SQL analytics |

## Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

Place a LendingClub CSV (`loan*.csv` or `accepted_*.csv`) in `data/` for real data. Falls back to calibrated synthetic data automatically.

## Regulatory Framework

- **IFRS 9** — IASB ED/2013/3, three-stage expected credit loss model
- **Basel III AIRB** — BCBS §328, retail asset correlation formula
- **Basel 3.1** — Output floor at 72.5% of standardised approach (PRA CP16/22)
- **FCA Consumer Duty** — PS22/9, FG21/1 vulnerability guidance

## Author
Muhammad Usama | MSc FinTech (Distinction), University of Exeter
GitHub: MuhammadUsamaMalik26
