import streamlit as st

st.title("Demand Forecasting")

st.markdown("""
This page will render the results of the corresponding analysis notebook
(`notebooks/3_Demand_Forecast.ipynb`, adjusted to the numeric prefix).

Populate this page after running the notebook: load the processed outputs
(figures, tables, model artifacts) and display them here with `st.pyplot`,
`st.plotly_chart`, and `st.dataframe`.
""")
