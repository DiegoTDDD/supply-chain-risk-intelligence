# 0005. Allocation as a capacity-constrained LP, with Monte Carlo for the cost range

**Status:** Accepted

## Context
The risk model gives each order a delay probability under each mode. Choosing a mode per order to minimise freight plus expected SLA penalty is an assignment problem. Without a limit, every order would move to the cheapest mode, which real capacity would not allow.

## Decision
- Binary LP in PuLP (CBC): each order gets exactly one mode, and Standard Class can take at most its historical share of orders.
- Cost per order and mode = freight + P(late) × SLA penalty.
- A Monte Carlo simulation (10,000 runs) of which orders actually end up late gives a cost distribution instead of a single number. The SLA penalty is varied ±20% as a sensitivity check.

## Consequences
- On a 4,000-order sample the optimized allocation saves 1.01% of expected cost against the historical one, under the same capacity.
- The freight costs and penalty are assumptions, stated in the notebook. Real rates would change the size of the saving, not the method.
