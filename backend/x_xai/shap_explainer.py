"""
Explicabilidad XAI para modelos secuenciales.

El metodo implementa una aproximacion tipo SHAP por enmascaramiento:
se reemplaza una variable o un paso temporal por un valor de referencia y se
mide cuanto cambia la probabilidad de anomalia. Es liviano, reproducible y
adecuado para exportar un `xai.json` que luego visualiza el frontend.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _prediction_delta(model: Any, original: np.ndarray, masked: np.ndarray) -> float:
    original_pred = model.predict(original, verbose=0).ravel()
    masked_pred = model.predict(masked, verbose=0).ravel()
    return float(np.mean(np.abs(original_pred - masked_pred)))


def _top_items(items: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    return sorted(items, key=lambda item: item["importance"], reverse=True)[:limit]


def explain_sequence_model(
    model: Any,
    model_key: str,
    model_name: str,
    x_background: np.ndarray,
    x_explain: np.ndarray,
    feature_names: list[str],
    max_features: int = 20,
) -> dict[str, Any]:
    """
    Genera importancias globales por variable y por paso temporal.

    Args:
        model: modelo Keras entrenado.
        model_key: clave tecnica, por ejemplo `lstm`.
        model_name: nombre legible, por ejemplo `LSTM`.
        x_background: muestras de entrenamiento para calcular baseline.
        x_explain: muestras de test a explicar.
        feature_names: nombres de variables en el ultimo eje de X.
        max_features: limite de variables a evaluar para controlar costo.
    """
    if len(x_explain) == 0:
        return {
            "model_key": model_key,
            "model": model_name,
            "feature_importance": [],
            "temporal_importance": [],
            "method": "temporal_masking_shap_approximation",
        }

    sample = x_explain[: min(len(x_explain), 128)].copy()
    background = x_background[: min(len(x_background), 512)]
    baseline_by_feature = np.median(background, axis=(0, 1))
    baseline_by_step = np.median(background, axis=0)

    feature_variance = np.var(background, axis=(0, 1))
    candidate_features = np.argsort(feature_variance)[::-1][: min(max_features, sample.shape[-1])]

    feature_importance = []
    for feature_idx in candidate_features:
        masked = sample.copy()
        masked[:, :, feature_idx] = baseline_by_feature[feature_idx]
        importance = _prediction_delta(model, sample, masked)
        feature_importance.append(
            {
                "feature": feature_names[feature_idx] if feature_idx < len(feature_names) else f"feature_{feature_idx}",
                "feature_index": int(feature_idx),
                "importance": round(importance, 8),
            }
        )

    temporal_importance = []
    sequence_length = sample.shape[1]
    for step_idx in range(sequence_length):
        masked = sample.copy()
        masked[:, step_idx, :] = baseline_by_step[step_idx]
        importance = _prediction_delta(model, sample, masked)
        temporal_importance.append(
            {
                "step": f"t-{sequence_length - step_idx}",
                "step_index": int(step_idx),
                "importance": round(importance, 8),
            }
        )

    top_feature = _top_items(feature_importance, 1)
    top_step = _top_items(temporal_importance, 1)

    return {
        "model_key": model_key,
        "model": model_name,
        "method": "temporal_masking_shap_approximation",
        "description": (
            "Aproximacion tipo SHAP basada en enmascaramiento de variables y pasos temporales; "
            "mide la variacion media absoluta de la probabilidad de anomalia."
        ),
        "top_feature": top_feature[0]["feature"] if top_feature else None,
        "top_step": top_step[0]["step"] if top_step else None,
        "feature_importance": _top_items(feature_importance, max_features),
        "temporal_importance": _top_items(temporal_importance, sequence_length),
    }


def build_xai_report(
    explanations: dict[str, dict[str, Any]],
    dataset_name: str,
    feature_count: int,
    sequence_length: int,
) -> dict[str, Any]:
    model_comparison = []
    feature_totals: dict[str, float] = {}
    temporal_totals: dict[str, float] = {}

    for model_key, explanation in explanations.items():
        model_comparison.append(
            {
                "model_key": model_key,
                "model": explanation["model"],
                "top_feature": explanation.get("top_feature"),
                "top_step": explanation.get("top_step"),
            }
        )

        for item in explanation.get("feature_importance", []):
            feature_totals[item["feature"]] = feature_totals.get(item["feature"], 0.0) + float(item["importance"])
        for item in explanation.get("temporal_importance", []):
            temporal_totals[item["step"]] = temporal_totals.get(item["step"], 0.0) + float(item["importance"])

    model_count = max(len(explanations), 1)
    global_feature_importance = [
        {"feature": feature, "importance": round(value / model_count, 8)}
        for feature, value in feature_totals.items()
    ]
    global_temporal_importance = [
        {"step": step, "importance": round(value / model_count, 8)}
        for step, value in temporal_totals.items()
    ]

    return {
        "dataset": dataset_name,
        "method": "temporal_masking_shap_approximation",
        "feature_count": feature_count,
        "sequence_length": sequence_length,
        "global_feature_importance": _top_items(global_feature_importance, 20),
        "global_temporal_importance": _top_items(global_temporal_importance, sequence_length),
        "model_comparison": model_comparison,
        "models": explanations,
    }
