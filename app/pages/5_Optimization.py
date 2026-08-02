import streamlit as st
from pathlib import Path

FIG_DIR = Path(__file__).resolve().parents[2] / "reports" / "figures"

st.title("Network & Allocation Optimization")
st.caption("Module 5 — Linear Programming & Monte Carlo Simulation")

st.markdown("""
**Business question:** given the delay risk from Module 4, and a fixed warehouse/dock capacity,
how should orders be allocated between Second Class and Standard Class to minimize total
expected cost?

Every order was re-scored under **both** candidate shipping modes using Module 4's classifier
(counterfactual risk scoring), then optimized under a capacity constraint matching the
historical Standard Class share — without that constraint, the optimizer would trivially
assign everyone to the cheaper mode, which is not operationally realistic.
""")

col1, col2, col3 = st.columns(3)
col1.metric("Historical cost", "$61,525")
col2.metric("Optimized cost", "$60,905")
col3.metric("Savings", "1.01%", help="Under the same capacity constraint")

st.image(str(FIG_DIR / "module5_monte_carlo.png"), caption="Monte Carlo cost distribution — optimized allocation")

st.markdown("""
### Reading the result
Under the *same* capacity the operation already has, simply re-prioritizing which orders get
the scarce Standard Class slots — by risk, not by historical happenstance — produces real,
if modest, savings. Total cost scales roughly linearly with the assumed SLA penalty
(sensitivity tested at ±20%), as expected from the cost structure.

**How this closes the project:** Module 1 found *why* Second Class breaks its promise
(a structural scheduling gap). Module 2 showed both modes take similar *absolute* time.
Module 4 turned that into an order-level risk score. This module acts on that score under a
real operational constraint, rather than assuming unlimited flexibility.

**Limitation:** freight cost ($5/$8) and SLA penalty ($20) are illustrative assumptions — the
dataset has no real carrier billing data. The pipeline transfers to real cost figures; these
specific dollar values do not.
""")
