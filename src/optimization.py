"""
Module 5 support functions — network and allocation optimization.
"""
import numpy as np
import pandas as pd
import pulp


def solve_allocation(orders: list, modes: list, cost_matrix: dict,
                      capacity: dict = None) -> dict:
    """
    Solve a minimum-cost assignment of orders to shipping modes.

    cost_matrix: dict[(order_id, mode)] -> expected total cost (freight + SLA penalty)
    capacity: optional dict[mode] -> max number of orders assignable to that mode
    """
    prob = pulp.LpProblem("order_allocation", pulp.LpMinimize)
    x = pulp.LpVariable.dicts("assign", (orders, modes), cat="Binary")

    prob += pulp.lpSum(cost_matrix[(o, m)] * x[o][m] for o in orders for m in modes)

    for o in orders:
        prob += pulp.lpSum(x[o][m] for m in modes) == 1

    if capacity:
        for m in modes:
            if m in capacity:
                prob += pulp.lpSum(x[o][m] for o in orders) <= capacity[m]

    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    assignment = {o: m for o in orders for m in modes if pulp.value(x[o][m]) == 1}
    total_cost = pulp.value(prob.objective)
    return {"assignment": assignment, "total_cost": total_cost, "status": pulp.LpStatus[prob.status]}


def monte_carlo_cost_distribution(base_cost: float, delay_prob: np.ndarray, penalty: float,
                                   n_sims: int = 10000, seed: int = 42) -> np.ndarray:
    """Simulate total cost distribution given per-order delay probabilities."""
    rng = np.random.default_rng(seed)
    n_orders = len(delay_prob)
    draws = rng.random((n_sims, n_orders)) < delay_prob
    penalty_costs = draws.sum(axis=1) * penalty
    return base_cost + penalty_costs
