"""
Module 4 support functions — delay risk classification & explainability.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss


def precision_at_k(y_true: np.ndarray, y_scores: np.ndarray, k: int) -> float:
    """Precision among the top-k highest-risk predictions."""
    order = np.argsort(y_scores)[::-1][:k]
    return float(np.mean(np.asarray(y_true)[order]))


def evaluate_classifier(y_true: np.ndarray, y_scores: np.ndarray, k_fraction: float = 0.1) -> dict:
    k = max(1, int(len(y_true) * k_fraction))
    return {
        "auc_roc": roc_auc_score(y_true, y_scores),
        "auc_pr": average_precision_score(y_true, y_scores),
        "brier_score": brier_score_loss(y_true, y_scores),
        f"precision_at_top_{int(k_fraction*100)}pct": precision_at_k(y_true, y_scores, k),
    }


def expected_savings(y_true: np.ndarray, y_scores: np.ndarray, k: int, penalty_per_delay: float) -> float:
    """Estimated savings from intervening on the top-k highest-risk orders,
    assuming intervention prevents the delay with certainty (upper bound estimate)."""
    order = np.argsort(y_scores)[::-1][:k]
    prevented_delays = np.asarray(y_true)[order].sum()
    return float(prevented_delays * penalty_per_delay)
