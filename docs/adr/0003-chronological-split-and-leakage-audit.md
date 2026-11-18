# 0003. Risk model: chronological split and an explicit leakage list

**Status:** Accepted

## Context
The goal is to score delay risk when an order is created. Several columns only exist after shipping: actual shipping days, delivery status and the shipping date. Any of them would give a near-perfect but useless model. A random split would also let the model see the future.

## Decision
- `LEAKAGE_COLUMNS` (`src/data_loader.py`) lists the post-outcome columns. Notebook 04 asserts that none of them is a feature.
- Train on the earliest 80% of orders by order date and test on the latest 20%.
- Compare logistic regression and XGBoost. Calibrate XGBoost with isotonic regression, and evaluate AUC, PR-AUC, Brier score and precision in the top 10%.

## Consequences
- Precision among the top 10% riskiest orders is 94.8% (calibrated), with AUC 0.73. Most of the signal comes from the shipping mode, which is known at order time.
- The savings estimate (top 10% intervention) is an upper bound: it assumes every intervention prevents the delay.
