# Global Supply Chain Risk & Delay Intelligence

An end-to-end data science project analyzing delivery delay in a global supply chain,
moving from statistical **understanding** to operational **decision**.

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
├── requirements.txt
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
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Place the raw dataset as described above, then run the notebooks in order (01 → 05), since
later modules consume outputs from earlier ones (Module 5 uses the delay probabilities from
Module 4 and the cost effects from Module 1).

## Running the dashboard

```bash
streamlit run app/streamlit_app.py
```

## Limitations

This is an observational, single-period dataset. Causal estimates are quasi-causal (matching
reduces but does not eliminate confounding bias), and long-horizon seasonality in the
forecasting module cannot be fully validated with the available time range. These limits are
restated inside each notebook's conclusion section.

## Author

Diego Leonardo Alves de Andrade — Data Analyst / Data Scientist
GitHub: [DiegoTDDD](https://github.com/DiegoTDDD)
