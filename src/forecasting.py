"""
Module 3 support functions — demand forecasting.
"""
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL


def naive_forecast(series: pd.Series, lag: int = 7) -> pd.Series:
    """Naive baseline: value from `lag` periods ago."""
    return series.shift(lag)


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def stl_decompose(series: pd.Series, period: int = 7):
    """Run STL decomposition and return the fitted result object."""
    return STL(series, period=period).fit()


def interval_coverage(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    """Fraction of true values falling within the predicted interval."""
    within = (y_true >= lower) & (y_true <= upper)
    return float(np.mean(within))
