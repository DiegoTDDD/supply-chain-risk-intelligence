"""
Module 1 support functions — causal diagnosis of delivery delay.
Propensity score matching, balance diagnostics, and ATE estimation.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


def estimate_propensity_scores(df: pd.DataFrame, treatment_col: str, covariates: list) -> pd.Series:
    """Fit a logistic regression of treatment assignment on covariates."""
    X = pd.get_dummies(df[covariates], drop_first=True)
    y = df[treatment_col]
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    return pd.Series(model.predict_proba(X)[:, 1], index=df.index, name="propensity_score")


def nearest_neighbor_match(df: pd.DataFrame, treatment_col: str, ps_col: str = "propensity_score",
                            caliper: float = 0.05) -> pd.DataFrame:
    """Greedy 1:1 nearest-neighbor matching on propensity score within a caliper."""
    treated = df[df[treatment_col] == 1].sort_values(ps_col)
    control = df[df[treatment_col] == 0].sort_values(ps_col)

    control_ps = control[ps_col].to_numpy()
    matched_idx = []
    used = set()

    for idx, ps in zip(treated.index, treated[ps_col]):
        diffs = np.abs(control_ps - ps)
        order = np.argsort(diffs)
        for pos in order:
            c_idx = control.index[pos]
            if c_idx not in used and diffs[pos] <= caliper:
                matched_idx.append((idx, c_idx))
                used.add(c_idx)
                break

    treated_idx = [t for t, c in matched_idx]
    control_idx = [c for t, c in matched_idx]
    return df.loc[treated_idx + control_idx]


def standardized_mean_difference(df: pd.DataFrame, treatment_col: str, covariate: str) -> float:
    """Compute SMD for a single covariate between treated and control groups."""
    t = df[df[treatment_col] == 1][covariate]
    c = df[df[treatment_col] == 0][covariate]
    pooled_std = np.sqrt((t.var() + c.var()) / 2)
    if pooled_std == 0:
        return 0.0
    return (t.mean() - c.mean()) / pooled_std


def average_treatment_effect(df: pd.DataFrame, treatment_col: str, outcome_col: str,
                              n_bootstrap: int = 1000, seed: int = 42) -> dict:
    """ATE with bootstrap confidence interval on a matched sample."""
    rng = np.random.default_rng(seed)
    treated = df[df[treatment_col] == 1][outcome_col].to_numpy()
    control = df[df[treatment_col] == 0][outcome_col].to_numpy()

    ate = treated.mean() - control.mean()
    boot_ates = []
    for _ in range(n_bootstrap):
        t_sample = rng.choice(treated, size=len(treated), replace=True)
        c_sample = rng.choice(control, size=len(control), replace=True)
        boot_ates.append(t_sample.mean() - c_sample.mean())

    ci_low, ci_high = np.percentile(boot_ates, [2.5, 97.5])
    return {"ate": ate, "ci_low": ci_low, "ci_high": ci_high}
