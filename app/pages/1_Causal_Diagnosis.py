import streamlit as st
from pathlib import Path

FIG_DIR = Path(__file__).resolve().parents[2] / "reports" / "figures"

st.title("Causal Diagnosis of Delivery Delay")
st.caption("Module 1 — Propensity Score Matching")

st.markdown("""
**Business question:** which factors *cause* delivery delay, not merely correlate with it?

A critical data-quality check came first: the promised delivery window
(`Days for shipment (scheduled)`) has **zero variance** within every shipping mode — it is
fixed by label, not by any real assessment of the order. This makes the raw `Late_delivery_risk`
flag mechanically dominated by how tight each mode's own promise is, rather than a genuine
performance signal. The causal analysis instead compares **Second Class vs Standard Class** —
the two modes with real, order-to-order variation in delay — using days of delay relative to
promise as the outcome.
""")

col1, col2, col3 = st.columns(3)
col1.metric("Naive (unadjusted) gap", "1.99 days")
col2.metric("Matched ATE (PSM)", "1.99 days", help="95% CI: [1.97, 2.01]")
col3.metric("SMD post-matching", "< 0.03", help="Balance target: < 0.1")

st.image(str(FIG_DIR / "module1_naive_delay_boxplot.png"), caption="Delay days by shipping mode (unadjusted)")
st.image(str(FIG_DIR / "module1_propensity_overlap.png"), caption="Propensity score overlap between treatment and control")
st.image(str(FIG_DIR / "module1_ate_comparison.png"), caption="Effect of Second Class vs Standard Class on delivery delay")

st.markdown("""
### Reading the result
The ATE barely moves after matching — from 1.995 to 1.99 days — even though the propensity
score range is narrow (0.12–0.35), meaning region, category, segment, month, and order value
explain very little of the mode assignment. That stability is itself informative: the delay gap
is a **structural property of the shipping mode**, not a confound of who orders what, from where.

**Limitation:** this remains an observational, quasi-causal estimate. First Class and Same Day
were excluded because their scheduled window has zero variance, making them incomparable outcomes.
""")
