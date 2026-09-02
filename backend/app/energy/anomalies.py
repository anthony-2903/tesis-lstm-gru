from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd


ANOMALY_METHODS = ("percentile", "mad", "mean_std")


@dataclass(frozen=True)
class AnomalyThreshold:
    method: str
    center: float
    scale: float
    decision_limit: float
    absolute_threshold: float
    calibration_samples: int
    parameters: dict[str, float]


def calibrate_anomaly_threshold(
    residuals,
    *,
    method: str,
    percentile: float = 0.995,
    mad_z_limit: float = 3.5,
    std_multiplier: float = 3.0,
) -> AnomalyThreshold:
    values = np.asarray(residuals, dtype=float).reshape(-1)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        raise ValueError("Se requieren al menos dos residuos previos para calibrar anomalías.")
    if method not in ANOMALY_METHODS:
        raise ValueError(f"Metodo de anomalías no soportado: {method}")

    epsilon = float(np.finfo(np.float64).eps)
    if method == "percentile":
        if not 0.5 < percentile < 1.0:
            raise ValueError("percentile debe estar entre 0.5 y 1.0.")
        magnitudes = np.abs(values)
        threshold = max(float(np.quantile(magnitudes, percentile)), epsilon)
        return AnomalyThreshold(
            method=method,
            center=0.0,
            scale=threshold,
            decision_limit=1.0,
            absolute_threshold=threshold,
            calibration_samples=int(len(values)),
            parameters={"percentile": percentile},
        )

    if method == "mad":
        center = float(np.median(values))
        raw_mad = float(np.median(np.abs(values - center)))
        robust_scale = 1.4826 * raw_mad
        if robust_scale <= epsilon:
            robust_scale = float(np.std(values, ddof=1))
        robust_scale = max(robust_scale, epsilon)
        return AnomalyThreshold(
            method=method,
            center=center,
            scale=robust_scale,
            decision_limit=mad_z_limit,
            absolute_threshold=mad_z_limit * robust_scale,
            calibration_samples=int(len(values)),
            parameters={"madZLimit": mad_z_limit},
        )

    magnitudes = np.abs(values)
    center = float(np.mean(magnitudes))
    scale = float(np.std(magnitudes, ddof=1))
    scale = max(scale, epsilon)
    absolute_threshold = max(center + std_multiplier * scale, epsilon)
    return AnomalyThreshold(
        method=method,
        center=0.0,
        scale=absolute_threshold,
        decision_limit=1.0,
        absolute_threshold=absolute_threshold,
        calibration_samples=int(len(values)),
        parameters={"stdMultiplier": std_multiplier, "magnitudeMean": center, "magnitudeStd": scale},
    )


def apply_anomaly_threshold(residuals, threshold: AnomalyThreshold) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(residuals, dtype=float).reshape(-1)
    if threshold.method == "mad":
        raw_score = np.abs(values - threshold.center) / threshold.scale
        normalized_score = raw_score / threshold.decision_limit
    else:
        normalized_score = np.abs(values) / threshold.absolute_threshold
    is_anomaly = normalized_score > 1.0
    severity = np.asarray([_severity(float(score)) for score in normalized_score], dtype=object)
    return normalized_score.astype(float), is_anomaly.astype(bool), severity


def detect_walk_forward_anomalies(
    predictions: pd.DataFrame,
    *,
    methods: tuple[str, ...] = ANOMALY_METHODS,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    required = {"seed", "fold", "timestamp", "actual", "prediction", "residual", "predictorId", "predictorFamily"}
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"Faltan columnas para anomalías walk-forward: {', '.join(missing)}")

    records: list[dict[str, Any]] = []
    threshold_history: list[dict[str, Any]] = []
    group_columns = ["seed", "predictorFamily", "predictorId"]
    for group_values, group in predictions.groupby(group_columns, sort=False):
        seed, family, predictor_id = group_values
        folds = sorted(int(value) for value in group["fold"].unique())
        for fold in folds[1:]:
            calibration = group[group["fold"] < fold]
            evaluation = group[group["fold"] == fold].reset_index(drop=True)
            for method in methods:
                threshold = calibrate_anomaly_threshold(calibration["residual"], method=method)
                scores, anomalies, severities = apply_anomaly_threshold(evaluation["residual"], threshold)
                threshold_history.append(
                    {
                        "seed": int(seed),
                        "predictorFamily": str(family),
                        "predictorId": str(predictor_id),
                        "evaluationFold": fold,
                        "calibrationMaxFold": fold - 1,
                        **asdict(threshold),
                    }
                )
                for index, source in evaluation.iterrows():
                    records.append(
                        {
                            "seed": int(seed),
                            "fold": fold,
                            "timestamp": source["timestamp"],
                            "predictorFamily": str(family),
                            "predictorId": str(predictor_id),
                            "method": method,
                            "actual": float(source["actual"]),
                            "prediction": float(source["prediction"]),
                            "residual": float(source["residual"]),
                            "score": float(scores[index]),
                            "isAnomaly": bool(anomalies[index]),
                            "severity": str(severities[index]),
                            "threshold": float(threshold.absolute_threshold),
                            "calibrationMaxFold": fold - 1,
                            "labelType": "estimated_from_prior_validation_residuals",
                        }
                    )

    result = pd.DataFrame(records)
    if result.empty:
        raise ValueError("No hay suficientes folds para detectar anomalías walk-forward.")
    summaries = []
    for keys, group in result.groupby(["predictorFamily", "predictorId", "method"], sort=False):
        family, predictor_id, method = keys
        summaries.append(
            {
                "predictorFamily": str(family),
                "predictorId": str(predictor_id),
                "method": str(method),
                "evaluatedSamples": int(len(group)),
                "estimatedAnomalies": int(group["isAnomaly"].sum()),
                "estimatedAnomalyRate": float(group["isAnomaly"].mean()),
                "meanScore": float(group["score"].mean()),
                "maxScore": float(group["score"].max()),
            }
        )
    report = {
        "labelType": "estimated",
        "classificationMetricsAvailable": False,
        "reason": "No independent anomaly labels were provided; precision, recall and F1 are intentionally omitted.",
        "testSetUsed": False,
        "recommendedMethod": "mad",
        "methods": list(methods),
        "summaries": summaries,
        "thresholdHistory": threshold_history,
    }
    return result, report


def _severity(score: float) -> str:
    if score <= 1.0:
        return "normal"
    if score <= 1.5:
        return "low"
    if score <= 2.5:
        return "medium"
    return "high"
