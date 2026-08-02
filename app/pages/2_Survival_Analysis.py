import streamlit as st
from pathlib import Path

FIG_DIR = Path(__file__).resolve().parents[2] / "reports" / "figures"

st.title("Survival Analysis of Order Fulfillment")
st.caption("Module 2 — Kaplan-Meier & Cox Proportional Hazards")

st.markdown("""
**Business question:** what is the "life expectancy" of an order until delivery, and which
factors accelerate or delay that time-to-event?
""")

col1, col2, col3 = st.columns(3)
col1.metric("Log-rank p-value", "0.41", help="Second Class vs Standard Class — not significant")
col2.metric("Cox C-index", "0.689")
col3.metric("Hazard ratio drift (2 periods)", "< 1%", help="Robustness check across chronological halves")

st.image(str(FIG_DIR / "module2_km_by_mode.png"), caption="Kaplan-Meier survival curves by shipping mode")
st.image(str(FIG_DIR / "module2_overall_survival.png"), caption="Overall survival curve — all shipping modes combined")

st.markdown("""
### Reading the result
In **absolute time**, Second Class and Standard Class orders take a statistically
indistinguishable number of days to ship (~4 days on average, log-rank p ≈ 0.41) — even though
Module 1 found Second Class breaks its *own, tighter promise* by ~2 more days. Together, the two
findings point to a **scheduling-promise problem**, not a difference in real operational speed.

Shipping mode dominates the hazard of fulfillment time (C-index ≈ 0.69); region and customer
segment contribute comparatively little.

**Limitation:** this historical extract has zero missing shipment dates, so no real censoring
exists — a live system would have in-flight orders genuinely censored.
""")
