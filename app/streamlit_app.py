import streamlit as st

st.set_page_config(
    page_title="Global Supply Chain Risk & Delay Intelligence",
    page_icon="🚚",
    layout="wide",
)

st.title("🚚 Global Supply Chain Risk & Delay Intelligence")

st.markdown("""
This project analyzes delivery delay across a global supply chain dataset through five
complementary lenses, moving from **understanding** to **decision**:

1. **Causal Diagnosis** — which factors *cause* delay, not just correlate with it
2. **Survival Analysis** — the time-to-delivery dynamics and hazard windows
3. **Demand Forecasting** — expected volume by category and region, with uncertainty
4. **Delay Risk Classification** — which orders are at risk, and why (SHAP)
5. **Network Optimization** — how to allocate orders to minimize expected total cost

Use the sidebar to navigate between modules.
""")

st.info(
    "Dataset: DataCo Smart Supply Chain for Big Data Analysis. "
    "Place the raw CSV at `data/raw/DataCoSupplyChainDataset.csv` before running any module."
)
