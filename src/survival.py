"""
Module 2 support functions — survival analysis of order fulfillment.
"""
from __future__ import annotations

import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import StatisticalResult, logrank_test


def fit_km_by_group(df: pd.DataFrame, duration_col: str, event_col: str,
                    group_col: str) -> dict[object, KaplanMeierFitter]:
    """Fit a Kaplan-Meier estimator per group, returning a dict of fitted KMF objects."""
    fitters = {}
    for name, group in df.groupby(group_col):
        kmf = KaplanMeierFitter()
        kmf.fit(group[duration_col], group[event_col], label=str(name))
        fitters[name] = kmf
    return fitters


def logrank_between_two_groups(df: pd.DataFrame, duration_col: str, event_col: str,
                                group_col: str, group_a: object, group_b: object
                                ) -> StatisticalResult:
    """Log-rank test of whether two groups have the same survival curve."""
    a = df[df[group_col] == group_a]
    b = df[df[group_col] == group_b]
    result = logrank_test(
        a[duration_col], b[duration_col],
        event_observed_A=a[event_col], event_observed_B=b[event_col],
    )
    return result


def fit_cox_model(df: pd.DataFrame, duration_col: str, event_col: str,
                  covariates: list) -> CoxPHFitter:
    """Fit a multivariate Cox Proportional Hazards model."""
    cph = CoxPHFitter()
    model_df = df[[duration_col, event_col] + covariates].dropna()
    cph.fit(model_df, duration_col=duration_col, event_col=event_col)
    return cph
