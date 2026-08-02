import streamlit as st
from pathlib import Path

st.set_page_config(
    page_title="Global Supply Chain Risk & Delay Intelligence",
    page_icon="🚚",
    layout="wide",
)

st.title("🚚 Global Supply Chain Risk & Delay Intelligence")

st.markdown("""
An end-to-end data science project on delivery delay in a global supply chain — moving from
statistical **understanding** to operational **decision**, on 180,000+ real orders
(DataCo Smart Supply Chain dataset).
""")

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Orders analyzed", "180,519")
col2.metric("Causal delay effect", "1.99 days", help="Second Class vs Standard Class, PSM-adjusted")
col3.metric("Forecast improvement", "33.3%", help="Prophet vs naive baseline, MAPE")
col4.metric("Risk model precision", "94.8%", help="Top 10% highest-risk orders")
col5.metric("Cost savings", "1.01%", help="Optimized vs historical allocation, capacity-constrained")

st.markdown("""
### Five modules, one narrative

| # | Module | Question | Core method |
|---|--------|----------|-------------|
| 1 | **Causal Diagnosis** | *Why* does it delay? | Propensity Score Matching |
| 2 | **Survival Analysis** | *When* does it delay? | Kaplan-Meier, Cox Proportional Hazards |
| 3 | **Demand Forecasting** | *How much* is coming? | STL, SARIMA, Prophet |
| 4 | **Delay Risk Classification** | *Who* is at risk, and why? | XGBoost, SHAP, calibration |
| 5 | **Network Optimization** | *What* should we do about it? | Linear Programming, Monte Carlo |

Each module feeds the next: the causal finding from Module 1 is tested against absolute time in
Module 2, turned into an order-level score in Module 4, and finally acted on under a real
capacity constraint in Module 5.

Use the sidebar to explore each module in depth.
""")

st.info(
    "Full code, notebooks, and methodology: "
    "[github.com/DiegoTDDD/supply-chain-risk-intelligence](https://github.com/DiegoTDDD/supply-chain-risk-intelligence)"
)
