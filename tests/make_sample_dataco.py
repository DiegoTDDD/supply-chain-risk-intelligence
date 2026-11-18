"""Write a synthetic file with the schema of the DataCo dataset, for running the notebooks in CI.

The real dataset needs a Kaggle login, so CI cannot download it. This sample has
the same columns, formats and value ranges the notebooks use, about 12,000 orders
from 2015-01-01 to 2018-01-31, so every notebook runs top to bottom. Its numbers
mean nothing: the results in the README come from the real file.

Usage:
    python tests/make_sample_dataco.py [output_csv]
    (default: data/raw/DataCoSupplyChainDataset.csv)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_OUT = Path(__file__).resolve().parents[1] / "data" / "raw" / "DataCoSupplyChainDataset.csv"
MODES = {  # mode: (share of orders, scheduled days, possible real days)
    "Standard Class": (0.60, 4, [2, 3, 4, 5, 6]),
    "Second Class": (0.20, 2, [2, 3, 4, 5, 6]),
    "First Class": (0.15, 1, [1, 2]),
    "Same Day": (0.05, 0, [0, 1]),
}
REGIONS = ["Western Europe", "Central America", "South America", "Southeast Asia", "Oceania"]
MARKETS = ["Europe", "LATAM", "Pacific Asia", "USCA", "Africa"]
CATEGORIES = ["Cleats", "Men's Footwear", "Women's Apparel", "Indoor/Outdoor Games", "Fishing"]
SEGMENTS = ["Consumer", "Corporate", "Home Office"]


def make_sample(seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    days = pd.date_range("2015-01-01", "2018-01-31", freq="D")
    weekly = np.array([1.1, 1.0, 1.0, 1.0, 1.05, 0.9, 0.95])  # Monday..Sunday
    rate = 11 * weekly[days.dayofweek] * np.where(days >= "2017-10-02", 0.6, 1.0)
    per_day = rng.poisson(rate)
    order_dates = np.repeat(days.values, per_day) + pd.to_timedelta(
        rng.integers(0, 24 * 60, per_day.sum()), unit="min"
    ).values
    n = len(order_dates)

    names = list(MODES)
    mode = rng.choice(names, n, p=[MODES[m][0] for m in names])
    scheduled = np.array([MODES[m][1] for m in mode])
    real = np.array([rng.choice(MODES[m][2]) for m in mode])
    late = (real > scheduled).astype(int)
    order_dt = pd.to_datetime(order_dates)
    ship_dt = order_dt + pd.to_timedelta(real, unit="D")
    total = rng.uniform(20, 500, n).round(2)
    profit = (rng.normal(22, 40, n) - 1.5 * late).round(2)
    status = np.where(late == 1, "Late delivery",
                      np.where(real < scheduled, "Advance shipping", "Shipping on time"))

    fmt = "%m/%d/%Y %H:%M"
    return pd.DataFrame({
        "Type": rng.choice(["DEBIT", "TRANSFER", "PAYMENT", "CASH"], n),
        "Days for shipping (real)": real,
        "Days for shipment (scheduled)": scheduled,
        "Delivery Status": status,
        "Late_delivery_risk": late,
        "Category Name": rng.choice(CATEGORIES, n),
        "Customer Segment": rng.choice(SEGMENTS, n),
        "Market": rng.choice(MARKETS, n),
        "Order Region": rng.choice(REGIONS, n),
        "order date (DateOrders)": order_dt.strftime(fmt),
        "Order Item Total": total,
        "Order Profit Per Order": profit,
        "shipping date (DateOrders)": ship_dt.strftime(fmt),
        "Shipping Mode": mode,
    })


def main() -> None:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    df = make_sample()
    df.to_csv(out, index=False, encoding="latin1")
    sys.stdout.write(f"Wrote {len(df):,} synthetic orders to {out}\n")


if __name__ == "__main__":
    main()
