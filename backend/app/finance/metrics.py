from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def finance_classification_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    *,
    threshold: float = 0.5,
) -> dict[str, float | int]:
    labels = np.asarray(y_true, dtype=np.int32).reshape(-1)
    scores = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if not len(labels) or len(labels) != len(scores):
        raise ValueError("Etiquetas y probabilidades financieras deben tener igual longitud no vacía.")
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("Las métricas financieras requieren ambas clases binarias.")
    if not np.isfinite(scores).all() or np.any((scores < 0.0) | (scores > 1.0)):
        raise ValueError("Las probabilidades financieras están fuera del intervalo [0, 1].")

    predictions = (scores >= threshold).astype(np.int32)
    true_positive = int(((predictions == 1) & (labels == 1)).sum())
    true_negative = int(((predictions == 0) & (labels == 0)).sum())
    false_positive = int(((predictions == 1) & (labels == 0)).sum())
    false_negative = int(((predictions == 0) & (labels == 1)).sum())
    negatives = true_negative + false_positive
    positives = true_positive + false_negative
    clipped = np.clip(scores, 1e-7, 1.0 - 1e-7)
    prevalence = float(labels.mean())
    return {
        "prAuc": float(average_precision_score(labels, scores)),
        "prAucBaseline": prevalence,
        "rocAuc": float(roc_auc_score(labels, scores)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "mcc": float(matthews_corrcoef(labels, predictions)),
        "balancedAccuracy": float(balanced_accuracy_score(labels, predictions)),
        "brierScore": float(brier_score_loss(labels, scores)),
        "logLoss": float(log_loss(labels, clipped, labels=[0, 1])),
        "falsePositiveRate": float(false_positive / negatives) if negatives else 0.0,
        "fraudPrevalence": prevalence,
        "truePositive": true_positive,
        "trueNegative": true_negative,
        "falsePositive": false_positive,
        "falseNegative": false_negative,
        "positiveRows": positives,
        "negativeRows": negatives,
        "threshold": float(threshold),
        "samples": int(len(labels)),
    }
