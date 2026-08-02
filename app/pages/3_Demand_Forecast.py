import streamlit as st
import pandas as pd
from pathlib import Path

FIG_DIR = Path(__file__).resolve().parents[2] / "reports" / "figures"

st.title("Demand Forecasting")
st.caption("Module 3 — STL Decomposition, SARIMA & Prophet")

st.markdown("""
**Business question:** how much order volume should we expect in the coming weeks — to size
inventory, fleet, and distribution centers without excess or stockouts?

A critical data-quality finding came first: starting **2017-10-03**, daily order volume
collapses into a perfectly deterministic alternating pattern — a structural break consistent
with a data-generation artifact, not real demand. The forecasting exercise below uses only the
clean, pre-break regime.
""")

st.image(str(FIG_DIR / "module3_full_series.png"), caption="Daily order volume — full series (structural break visible)")
st.image(str(FIG_DIR / "module3_break_zoom.png"), caption="Zoom on the structural break (2017-10-03)")
st.image(str(FIG_DIR / "module3_stl_decomposition.png"), caption="STL decomposition — trend, weekly seasonality, residual")

st.markdown("### Model comparison (30-day holdout, clean regime)")
comparison = pd.DataFrame({
    "Model": ["Naive (lag-7)", "SARIMA(1,0,1)(1,0,1,7)", "Prophet"],
    "MAPE (%)": [8.90, 6.00, 5.94],
    "RMSE": [18.38, 12.87, 12.78],
    "Improvement vs naive": ["—", "32.6%", "33.3%"],
})
st.dataframe(comparison, use_container_width=True, hide_index=True)

st.image(str(FIG_DIR / "module3_forecast_comparison.png"), caption="Holdout forecast comparison — clean (pre-break) regime")

st.markdown("""
### Reading the result
Both SARIMA and Prophet beat the naive baseline by a clear margin, cutting MAPE by roughly a
third. Prophet's 95% interval coverage lands close to its nominal target (93.3%), while
SARIMA's is comparatively conservative (100% coverage, wider bands).

**Limitation:** only ~2.7 years of clean data are available, limiting validation of true annual
seasonality.
""")
