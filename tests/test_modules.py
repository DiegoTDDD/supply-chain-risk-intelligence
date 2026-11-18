"""The functions in src/ that the five notebooks import, on small data with known answers."""

import numpy as np
import pandas as pd
import pytest

import causal
import classification
import data_loader
import forecasting
import optimization
import survival

# --- Module 1: causal


@pytest.fixture
def confounded():
    """Treatment depends on region; the outcome gets +2 days from treatment and +1 from region."""
    rng = np.random.default_rng(0)
    n = 4000
    region = rng.choice(["North", "South"], n)
    value = rng.uniform(10, 100, n)
    p_treat = np.where(region == "North", 0.7, 0.3)
    treatment = rng.binomial(1, p_treat)
    outcome = 2.0 * treatment + 1.0 * (region == "North") + rng.normal(0, 0.5, n)
    return pd.DataFrame({"region": region, "value": value, "treatment": treatment, "y": outcome})


def test_propensity_scores_are_probabilities_that_track_the_confounder(confounded):
    ps = causal.estimate_propensity_scores(confounded, "treatment", ["region", "value"])
    assert ps.between(0, 1).all()
    by_region = ps.groupby(confounded["region"]).mean()
    assert by_region["North"] == pytest.approx(0.7, abs=0.05)
    assert by_region["South"] == pytest.approx(0.3, abs=0.05)


def test_matching_is_one_to_one_within_the_caliper(confounded):
    df = confounded.assign(
        propensity_score=causal.estimate_propensity_scores(confounded, "treatment", ["region"])
    )
    matched = causal.nearest_neighbor_match(df, "treatment", caliper=0.01)
    treated, control = matched[matched.treatment == 1], matched[matched.treatment == 0]
    assert len(treated) == len(control) > 0
    assert control.index.is_unique  # no control used twice
    gaps = np.abs(treated["propensity_score"].to_numpy() - control["propensity_score"].to_numpy())
    assert (gaps <= 0.01).all()


def test_matching_removes_the_confounding_bias(confounded):
    naive = causal.average_treatment_effect(confounded, "treatment", "y", n_bootstrap=200)
    assert naive["ate"] > 2.3  # biased upward by region
    df = confounded.assign(
        propensity_score=causal.estimate_propensity_scores(confounded, "treatment", ["region"])
    )
    matched = causal.nearest_neighbor_match(df, "treatment", caliper=0.01)
    result = causal.average_treatment_effect(matched, "treatment", "y", n_bootstrap=200)
    assert result["ci_low"] < 2.0 < result["ci_high"]
    assert result["ate"] == pytest.approx(2.0, abs=0.1)


def test_standardized_mean_difference():
    df = pd.DataFrame({"t": [1, 1, 0, 0], "x": [3.0, 5.0, 1.0, 3.0]})
    # means 4 and 2, both variances 2 -> SMD = 2 / sqrt(2)
    assert causal.standardized_mean_difference(df, "t", "x") == pytest.approx(2 / np.sqrt(2))
    flat = pd.DataFrame({"t": [1, 1, 0, 0], "x": [1.0, 1.0, 1.0, 1.0]})  # no spread: SMD 0
    assert causal.standardized_mean_difference(flat, "t", "x") == 0.0


# --- Module 2: survival


@pytest.fixture
def shipments():
    rng = np.random.default_rng(1)
    fast = pd.DataFrame({"days": rng.exponential(2, 500), "group": "fast"})
    slow = pd.DataFrame({"days": rng.exponential(6, 500), "group": "slow"})
    df = pd.concat([fast, slow], ignore_index=True)
    df["shipped"] = 1
    df["is_slow"] = (df["group"] == "slow").astype(int)
    return df


def test_kaplan_meier_per_group_and_logrank(shipments):
    fitters = survival.fit_km_by_group(shipments, "days", "shipped", "group")
    assert set(fitters) == {"fast", "slow"}
    assert fitters["fast"].median_survival_time_ < fitters["slow"].median_survival_time_
    test = survival.logrank_between_two_groups(
        shipments, "days", "shipped", "group", "fast", "slow"
    )
    assert test.p_value < 0.001


def test_cox_model_finds_the_slower_group(shipments):
    cph = survival.fit_cox_model(shipments, "days", "shipped", ["is_slow"])
    # Slower group = lower hazard of being shipped on any given day: true ratio is 2/6.
    assert cph.hazard_ratios_["is_slow"] == pytest.approx(1 / 3, abs=0.08)


# --- Module 3: forecasting


def test_error_metrics():
    y = np.array([10.0, 0.0, 20.0])
    pred = np.array([12.0, 5.0, 15.0])
    assert forecasting.mape(y, pred) == pytest.approx((20 + 25) / 2)  # the zero day is skipped
    rmse = forecasting.rmse(np.array([1.0, 3.0]), np.array([2.0, 1.0]))
    assert rmse == pytest.approx(np.sqrt(2.5))


def test_naive_forecast_and_interval_coverage():
    s = pd.Series(range(10), dtype=float)
    assert forecasting.naive_forecast(s, lag=7).iloc[7:].tolist() == [0.0, 1.0, 2.0]
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert forecasting.interval_coverage(y, y - 0.5, np.array([1.5, 2.5, 2.9, 4.5])) == 0.75


def test_stl_recovers_a_weekly_pattern():
    days = pd.date_range("2024-01-01", periods=70, freq="D")
    pattern = np.tile([5, 0, 0, 0, 0, -2, -3], 10).astype(float)
    series = pd.Series(100 + pattern, index=days)
    result = forecasting.stl_decompose(series, period=7)
    np.testing.assert_allclose(result.trend + result.seasonal + result.resid, series)
    assert result.seasonal.iloc[14:21].round(0).tolist() == pattern[:7].tolist()


# --- Module 4: classification


def test_precision_at_k_and_savings():
    y = np.array([1, 0, 1, 0, 0])
    scores = np.array([0.9, 0.8, 0.7, 0.2, 0.1])
    assert classification.precision_at_k(y, scores, 2) == 0.5
    assert classification.precision_at_k(y, scores, 3) == pytest.approx(2 / 3)
    assert classification.expected_savings(y, scores, 3, penalty_per_delay=10.0) == 20.0


def test_evaluate_classifier_reports_every_metric():
    y = np.array([0, 0, 1, 1] * 25)
    scores = np.array([0.1, 0.4, 0.35, 0.8] * 25)
    metrics = classification.evaluate_classifier(y, scores, k_fraction=0.25)
    assert set(metrics) == {"auc_roc", "auc_pr", "brier_score", "precision_at_top_25pct"}
    assert metrics["auc_roc"] == pytest.approx(0.75)
    assert metrics["precision_at_top_25pct"] == 1.0


# --- Module 5: optimization


def test_allocation_picks_the_cheapest_mode_per_order():
    cost = {(0, "fast"): 5, (0, "slow"): 9, (1, "fast"): 7, (1, "slow"): 3}
    result = optimization.solve_allocation([0, 1], ["fast", "slow"], cost)
    assert result["status"] == "Optimal"
    assert result["assignment"] == {0: "fast", 1: "slow"}
    assert result["total_cost"] == 8


def test_allocation_respects_capacity():
    # Both orders prefer "slow", but it can take only one; the cheaper switch is order 1.
    cost = {(0, "fast"): 10, (0, "slow"): 1, (1, "fast"): 4, (1, "slow"): 2}
    result = optimization.solve_allocation([0, 1], ["fast", "slow"], cost, capacity={"slow": 1})
    assert result["assignment"] == {0: "slow", 1: "fast"}
    assert result["total_cost"] == 5


def test_monte_carlo_cost_matches_its_expectation():
    probs = np.array([0.1, 0.5, 0.9])
    costs = optimization.monte_carlo_cost_distribution(100.0, probs, penalty=10.0, n_sims=20000)
    assert costs.min() >= 100.0 and costs.max() <= 130.0
    assert costs.mean() == pytest.approx(100.0 + probs.sum() * 10.0, abs=0.2)
    again = optimization.monte_carlo_cost_distribution(100.0, probs, penalty=10.0, n_sims=20000)
    np.testing.assert_array_equal(costs, again)  # seeded


# --- data loading


def test_clean_parses_dates_and_drops_duplicates():
    raw = pd.DataFrame({
        " order date (DateOrders) ": ["1/31/2018 22:56", "1/31/2018 22:56"],
        "shipping date (DateOrders)": ["2/3/2018 22:56", "2/3/2018 22:56"],
        "Shipping Mode": ["Standard Class", "Standard Class"],
    })
    out = data_loader.engineer_core_fields(data_loader.clean(raw))
    assert len(out) == 1
    assert out["duration_days"].tolist() == [3]
    assert out["event_observed"].tolist() == [1]


def test_feature_columns_drop_leakage():
    df = pd.DataFrame(columns=["Shipping Mode", "Delivery Status", "Days for shipping (real)"])
    assert data_loader.feature_columns(df) == ["Shipping Mode"]
    assert len(data_loader.feature_columns(df, exclude_leakage=False)) == 3


def test_sample_dataset_has_every_column_the_notebooks_use():
    from make_sample_dataco import make_sample

    df = make_sample()
    needed = {"Shipping Mode", "Days for shipping (real)", "Days for shipment (scheduled)",
              "Late_delivery_risk", "Delivery Status", "Order Region", "Category Name",
              "Customer Segment", "Market", "Order Item Total", "Order Profit Per Order",
              "order date (DateOrders)", "shipping date (DateOrders)"}
    assert needed <= set(df.columns)
    assert (df["Late_delivery_risk"] == (df["Days for shipping (real)"]
                                         > df["Days for shipment (scheduled)"])).all()
    assert df.groupby("Shipping Mode")["Days for shipment (scheduled)"].nunique().eq(1).all()
