from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def phishing_classification_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    *,
    threshold: float = 0.5,
) -> dict[str, float | int]:
    labels = np.asarray(y_true, dtype=np.int32).reshape(-1)
    scores = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if len(labels) != len(scores) or not len(labels):
        raise ValueError("Etiquetas y probabilidades deben tener la misma longitud no vacía.")
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("Las métricas requieren ambas clases binarias.")
    predictions = (scores >= threshold).astype(np.int32)
    negative_mask = labels == 0
    false_positives = int(((predictions == 1) & negative_mask).sum())
    negatives = int(negative_mask.sum())
    return {
        "prAuc": float(average_precision_score(labels, scores)),
        "rocAuc": float(roc_auc_score(labels, scores)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "mcc": float(matthews_corrcoef(labels, predictions)),
        "balancedAccuracy": float(balanced_accuracy_score(labels, predictions)),
        "falsePositiveRate": float(false_positives / negatives) if negatives else 0.0,
        "threshold": float(threshold),
        "samples": int(len(labels)),
    }
