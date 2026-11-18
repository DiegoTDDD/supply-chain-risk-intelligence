# 0002. Propensity score matching with a 0.01 caliper and a bootstrap interval

**Status:** Accepted

## Context
Orders are not randomly assigned to shipping modes. Region, category, segment, month and order value could all drive both the mode and the delay.

## Decision
- Fit a logistic propensity model on those covariates (`src/causal.py`).
- Match each treated order 1:1 to the nearest control within 0.01 of propensity, without reuse.
- Check balance with the standardized mean difference (target |SMD| < 0.1).
- Report the ATE with a 1,000-draw bootstrap 95% interval.

## Alternatives considered
- **Regression adjustment.** Simpler, but it depends on getting the outcome model's functional form right.
- **Inverse probability weighting.** Uses every order, but is unstable when propensities near 0 or 1 get extreme weights. Here propensities lie between 0.12 and 0.35, so matching loses little.
- **Difference-in-differences.** Needs a policy change over time, which the data does not have.

## Consequences
- All 35,216 Second Class orders found a match. The ATE moved from 1.995 (unadjusted) to 1.99 days (95% CI 1.97 to 2.01): the gap is structural to the mode, not a confound.
- The estimate is quasi-causal: it adjusts only for observed covariates. The notebook checks SMD only for order value. Balance on the categorical covariates is a follow-up.
- `tests/test_modules.py` builds data with a known +2-day effect and a confounder, and checks that matching recovers it while the naive comparison is biased.
