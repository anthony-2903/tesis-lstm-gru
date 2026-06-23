from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, mean_squared_error, precision_score, recall_score


def classification_metrics(y_true, y_pred) -> dict[str, object]:
    labels = [0, 1]
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=labels).ravel()
    return {
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "confusionMatrix": {"tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)},
        "detectedCount": int(np.sum(y_pred)),
    }


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    return {"rmse": float(rmse)}
