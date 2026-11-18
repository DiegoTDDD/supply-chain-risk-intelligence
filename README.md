# Global Supply Chain Risk & Delay Intelligence

[![Tests](https://github.com/DiegoTDDD/supply-chain-risk-intelligence/actions/workflows/tests.yml/badge.svg)](https://github.com/DiegoTDDD/supply-chain-risk-intelligence/actions/workflows/tests.yml)
[![Notebooks](https://github.com/DiegoTDDD/supply-chain-risk-intelligence/actions/workflows/notebooks.yml/badge.svg)](https://github.com/DiegoTDDD/supply-chain-risk-intelligence/actions/workflows/notebooks.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**Problem.** Late deliveries cost money, but "we are often late" is not something operations can act on. They need to know what causes the delay, when it happens, what volume is coming, which orders are at risk, and what to change.

**Solution.** Five connected analyses of 180,519 real orders (the DataCo dataset): causal inference, survival analysis, forecasting, risk scoring with explanations, and an optimization that turns the risk scores into a cheaper shipping allocation.

**🔗 Live dashboard: [diego-supply-chain-risk.streamlit.app](https://diego-supply-chain-risk.streamlit.app)** · **Design decisions: [`docs/adr/`](docs/adr/README.md)**

## Results

| Module | Business result | How it was measured |
|---|---|---|
| 1. Causal diagnosis | Second Class runs **1.99 days later** against its promise than Standard Class (95% CI 1.97 to 2.01) | Propensity score matching, 35,216 matched pairs, \|SMD\| < 0.1 |
| 2. Survival analysis | Shipping mode is the main driver of time to ship; the effect is stable across both halves of the data | Kaplan-Meier, log-rank, Cox PH (C-index 0.69) |
| 3. Demand forecasting | **33% lower error** than the naive baseline: MAPE 5.9% vs 8.9% | Prophet vs SARIMA vs lag-7 naive, 30-day holdout, 93% interval coverage |
| 4. Delay risk | **94.8% of the top 10% riskiest orders are late**; AUC 0.73 | Calibrated XGBoost, chronological split, leakage-audited features, SHAP |
| 5. Optimization | **1.01% lower expected cost** than the historical allocation, under the same capacity | Binary LP (PuLP) + 10,000-run Monte Carlo + ±20% penalty sensitivity |

## Business narrative

Delivery delay is not a single phenomenon — it has a cause, a timeline, a forecastable
pattern, an individual risk profile, and a cost that can be optimized against. This project
treats it that way, through five connected modules:

| # | Module | Question answered | Core method |
|---|--------|--------------------|-------------|
| 1 | Causal Diagnosis | **Why** does it delay? | Propensity Score Matching, Difference-in-Differences |
| 2 | Survival Analysis | **When** does it delay? | Kaplan-Meier, Cox Proportional Hazards |
| 3 | Demand Forecasting | **How much** is coming? | STL decomposition, SARIMA, Prophet |
| 4 | Delay Risk Classification | **Who** is at risk, and why? | XGBoost, SHAP, calibration |
| 5 | Network Optimization | **What** should we do about it? | Linear Programming, Monte Carlo simulation |

Each module produces a business-facing metric — not just a model score. The final output is
a data-driven allocation strategy with an estimated cost reduction versus the historical
baseline.

## Dataset

**DataCo Smart Supply Chain for Big Data Analysis** (Kaggle)
https://www.kaggle.com/datasets/shashwatwork/dataco-smart-supply-chain-for-big-data-analysis

Real order-level data covering shipping mode, delivery status, region, category, customer
segment, and financials across a global operation.

Download the CSV manually (Kaggle requires authentication) and place it at:
```
data/raw/DataCoSupplyChainDataset.csv
```

## Project structure

```
supply-chain-risk-intelligence/
├── data/
│   ├── raw/                  # Original dataset (not versioned)
│   └── processed/            # Cleaned, feature-engineered parquet files
├── notebooks/
│   ├── 01_causal_delay_diagnosis.ipynb
│   ├── 02_survival_analysis.ipynb
│   ├── 03_demand_forecasting.ipynb
│   ├── 04_delay_classification_shap.ipynb
│   └── 05_network_optimization.ipynb
├── src/
│   ├── data_loader.py        # Loading, cleaning, feature engineering
│   ├── causal.py              # Module 1 support functions
│   ├── survival.py            # Module 2 support functions
│   ├── forecasting.py         # Module 3 support functions
│   ├── classification.py      # Module 4 support functions
│   └── optimization.py        # Module 5 support functions
├── app/
│   ├── streamlit_app.py       # Landing page
│   └── pages/                 # One page per module
├── reports/
│   └── figures/                # Exported charts for the README / portfolio
├── tests/
│   ├── test_modules.py         # unit tests for every function in src/
│   └── make_sample_dataco.py   # synthetic DataCo-shaped file for running the notebooks in CI
├── docs/adr/                   # architecture decision records
├── .github/                # CI: tests, notebooks, Windows, security audit, app health
├── pyproject.toml          # project metadata, pytest and ruff settings
├── requirements.txt        # pinned versions for the notebooks and the dashboard
├── requirements-test.txt   # light set for the unit tests
└── README.md
```

## Methodology highlights

- **Causal rigor**: delay drivers are estimated with matched, balance-checked causal methods
  — not naive correlation — and every effect is explicitly labeled quasi-causal given the
  observational nature of the data.
- **No data leakage**: Module 4's classifier excludes every post-outcome variable (actual
  shipping date, delivery status) that would not exist at prediction time. This is documented
  explicitly in `src/data_loader.py::LEAKAGE_COLUMNS`.
- **Uncertainty is first-class**: forecasts carry confidence intervals with evaluated
  coverage; the optimization module reports a cost *distribution* via Monte Carlo, not a
  single point estimate.
- **Every module ties to money**: each analysis closes with a financial translation — cost of
  delay, expected savings from intervention, or percentage cost reduction versus the
  historical baseline.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # macOS / Linux
# source .venv/Scripts/activate    # Windows (Git Bash)
# .venv\Scripts\Activate.ps1       # Windows (PowerShell)
pip install -r requirements.txt
```

Place the raw dataset as described above, then run the notebooks in order (01 → 05), since
later modules consume outputs from earlier ones (Module 5 uses the delay probabilities from
Module 4 and the cost effects from Module 1).

## Running the dashboard

Live version: [diego-supply-chain-risk.streamlit.app](https://diego-supply-chain-risk.streamlit.app)

To run locally instead:
```bash
streamlit run app/streamlit_app.py
```

## Tests

```bash
pip install -r requirements-test.txt
pytest
```

17 tests cover every function in `src/` on data with known answers. For example, matching must recover a planted +2-day effect that the naive comparison overstates, the LP must respect capacity, and the Monte Carlo mean must equal its expectation.

## Limitations

This is an observational, single-period dataset. Causal estimates are quasi-causal (matching
reduces but does not eliminate confounding bias), and long-horizon seasonality in the
forecasting module cannot be fully validated with the available time range. These limits are
restated inside each notebook's conclusion section. In addition:

- Covariate balance after matching is checked numerically only for order value. Categorical covariates are a follow-up.
- Freight costs and the SLA penalty in Module 5 are stated assumptions, so the 1.01% saving shows the method, not a quote.
- Forecasts cover only the period before the 2017-10-02 volume break ([ADR 0004](docs/adr/0004-forecast-only-the-pre-break-regime.md)).

## Automated checks

- **Tests** (`tests.yml`) run `ruff` (type hints required in `src/`) and the 17 unit tests on every push and pull request.
- **Notebooks** (`notebooks.yml`) runs all five notebooks top to bottom with the pinned requirements, on a synthetic file with the DataCo schema, because the real dataset needs a Kaggle login ([ADR 0006](docs/adr/0006-notebooks-run-in-ci-on-synthetic-data.md)). It runs when the notebooks, `src/` or the requirements change, and monthly.
- **Security audit** (`security-audit.yml`) audits every `requirements*.txt` with `pip-audit` whenever dependencies change and every Monday. It fails only when a vulnerable package has a fixed version available.
- **App health** (`app-health.yml`) opens the [live app](https://diego-supply-chain-risk.streamlit.app) in a real browser after each push and once a day (waking it if it was asleep) and fails if the app shows an error.

## Author

Diego Leonardo Alves de Andrade  
GitHub: [DiegoTDDD](https://github.com/DiegoTDDD)
