import streamlit as st
from pathlib import Path

FIG_DIR = Path(__file__).resolve().parents[2] / "reports" / "figures"

st.title("Delay Risk Classification & Explainability")
st.caption("Module 4 — XGBoost, Calibration & SHAP")

st.markdown("""
**Business question:** given an order at creation time, what is its probability of being
delayed — and why — so it can become operational action?

**Leakage audit:** any variable that only exists after delivery (`Days for shipping (real)`,
`Delivery Status`, `shipping date`) was excluded, with an explicit assertion in code guarding
against reintroducing it.
""")

col1, col2, col3 = st.columns(3)
col1.metric("AUC-ROC (XGBoost)", "0.725")
col2.metric("Precision @ top 10% risk", "94.8%", help="Calibrated model")
col3.metric("Brier score (calibrated)", "0.1978")

st.image(str(FIG_DIR / "module4_shap_summary.png"), caption="SHAP global feature importance")
st.image(str(FIG_DIR / "module4_calibration.png"), caption="Calibration curves — uncalibrated vs isotonic")
st.image(str(FIG_DIR / "module4_shap_local_example.png"), caption="Local SHAP explanation — a single high-risk order")

st.markdown("""
### Reading the result
XGBoost does not outperform a plain logistic regression by a meaningful margin here — an honest
result, since the dominant signal (`Shipping Mode`) is largely linear. SHAP confirms shipping
mode as the dominant driver, converging with the causal (Module 1) and survival (Module 2)
findings — three independent methods point to the same lever.

**Limitation:** the financial translation elsewhere in this project uses a descriptive profit
gap, not a causally estimated dollar effect of intervention.
""")
