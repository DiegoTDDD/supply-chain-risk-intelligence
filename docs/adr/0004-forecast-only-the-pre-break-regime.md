# 0004. Forecast only the series before the 2017-10-02 volume break

**Status:** Accepted

## Context
Daily order volume drops abruptly on 2017-10-02 and stays at the new level. That is a structural break. Whatever its cause, a model fit across it would learn a level that no longer exists.

## Decision
Use the series up to 2017-10-01. Hold out the last 30 days, and compare a lag-7 naive baseline, SARIMA(1,0,1)(1,0,1,7) and Prophet on MAPE, RMSE and 95% interval coverage.

## Consequences
- Prophet reaches 5.9% MAPE against 8.9% for the naive baseline (33.3% better), with 93% interval coverage. SARIMA is almost as good.
- The models say nothing about the post-break regime. Forecasting it would need enough post-break history to fit.
