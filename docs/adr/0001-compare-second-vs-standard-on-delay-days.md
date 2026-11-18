# 0001. Causal question: Second vs Standard Class, outcome = days late relative to promise

**Status:** Accepted

## Context
The dataset's `Late_delivery_risk` flag compares actual shipping days with `Days for shipment (scheduled)`. Within each shipping mode the scheduled value never varies: it is set by the mode's label, not by the order. First Class always promises 1 day and Same Day 0. So the flag mostly measures how tight each mode's promise is, and modes with a zero-variance outcome cannot be compared causally.

## Decision
Compare only Second Class (treatment) with Standard Class (control), the two modes with real order-to-order variation. Use `delay_days` = real minus scheduled days as the outcome.

## Alternatives considered
- **All four modes, with `Late_delivery_risk` as the outcome.** The effect would mostly reflect the promise rules.
- **Real shipping days alone.** Ignores the promise, which is what the customer experiences as late.

## Consequences
- 142,968 orders are in scope, and the estimate (1.99 days) answers a question that can be acted on.
- No causal statement is made about First Class or Same Day.
