from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def energy_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    true = np.asarray(y_true, dtype=float).reshape(-1)
    pred = np.asarray(y_pred, dtype=float).reshape(-1)
    denominator = np.abs(true) + np.abs(pred)
    smape = np.mean(np.divide(2.0 * np.abs(pred - true), denominator, out=np.zeros_like(denominator), where=denominator > 0))
    return {
        "rmse": float(mean_squared_error(true, pred) ** 0.5),
        "mae": float(mean_absolute_error(true, pred)),
        "smape": float(smape),
        "r2": float(r2_score(true, pred)),
        "samples": int(len(true)),
    }
