import streamlit as st

st.title("Delay Risk Classification")

st.markdown("""
This page will render the results of the corresponding analysis notebook
(`notebooks/4_Delay_Risk.ipynb`, adjusted to the numeric prefix).

Populate this page after running the notebook: load the processed outputs
(figures, tables, model artifacts) and display them here with `st.pyplot`,
`st.plotly_chart`, and `st.dataframe`.
""")
