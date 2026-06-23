from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance


MODEL_LABELS = {
    "lstm": "LSTM",
    "gru": "GRU",
    "brnn": "BRNN",
    "transformer": "Transformer",
    "tcn": "TCN",
}


def _normalize(items: list[dict[str, object]], key: str) -> list[dict[str, object]]:
    if not items:
        return []
    max_value = max(float(item["importance"]) for item in items) or 1.0
    normalized = [
        {
            key: item[key],
            "importance": round(max(float(item["importance"]) / max_value, 0.0), 4),
            **({"feature_index": item["feature_index"]} if "feature_index" in item else {}),
            **({"step_index": item["step_index"]} if "step_index" in item else {}),
        }
        for item in items
    ]
    return sorted(normalized, key=lambda item: float(item["importance"]), reverse=True)


def explain_classifier(estimator, x_test: pd.DataFrame, y_test: pd.Series) -> list[dict[str, object]]:
    if x_test.empty:
        return []
    repeats = 5 if len(x_test) >= 10 else 3
    result = permutation_importance(
        estimator,
        x_test,
        y_test,
        scoring="accuracy",
        n_repeats=repeats,
        random_state=42,
    )
    items = [
        {"feature": str(feature), "feature_index": idx, "importance": float(value)}
        for idx, (feature, value) in enumerate(zip(x_test.columns, result.importances_mean))
    ]
    return _normalize(items, "feature")


def explain_regressor_temporal(estimator, x_test: np.ndarray, y_test: np.ndarray) -> list[dict[str, object]]:
    if x_test.size == 0:
        return []
    repeats = 5 if len(x_test) >= 10 else 3
    result = permutation_importance(
        estimator,
        x_test,
        y_test,
        scoring="neg_root_mean_squared_error",
        n_repeats=repeats,
        random_state=42,
    )
    window = x_test.shape[1]
    items = []
    for idx, value in enumerate(result.importances_mean):
        lag = window - idx
        items.append({"step": f"t-{lag}", "step_index": int(idx), "importance": float(value)})
    return _normalize(items, "step")


def average_importance(model_items: Iterable[list[dict[str, object]]], item_key: str) -> list[dict[str, object]]:
    totals: dict[str, list[float]] = defaultdict(list)
    indexes: dict[str, int] = {}
    for items in model_items:
        for item in items:
            name = str(item[item_key])
            totals[name].append(float(item["importance"]))
            if item_key == "feature" and "feature_index" in item:
                indexes[name] = int(item["feature_index"])
            if item_key == "step" and "step_index" in item:
                indexes[name] = int(item["step_index"])
    averaged = [
        {
            item_key: name,
            "importance": float(np.mean(values)),
            **({"feature_index": indexes[name]} if item_key == "feature" and name in indexes else {}),
            **({"step_index": indexes[name]} if item_key == "step" and name in indexes else {}),
        }
        for name, values in totals.items()
    ]
    return _normalize(averaged, item_key)


def build_xai_report(
    filename: str,
    feature_importance_by_model: dict[str, list[dict[str, object]]],
    temporal_importance_by_model: dict[str, list[dict[str, object]]],
) -> dict[str, object]:
    global_features = average_importance(feature_importance_by_model.values(), "feature")
    global_temporal = average_importance(temporal_importance_by_model.values(), "step")
    models = {}
    comparison = []

    for key, label in MODEL_LABELS.items():
        features = feature_importance_by_model.get(key, [])
        temporal = temporal_importance_by_model.get(key, [])
        top_feature = features[0]["feature"] if features else None
        top_step = temporal[0]["step"] if temporal else None
        models[key] = {
            "model_key": key,
            "model": label,
            "method": "permutation_importance",
            "description": "Importancia calculada por permutacion sobre datos de validacion. Para series temporales, cada paso t-n mide sensibilidad del modelo al historial.",
            "top_feature": top_feature,
            "top_step": top_step,
            "feature_importance": features[:12],
            "temporal_importance": temporal[:12],
        }
        comparison.append(
            {
                "model_key": key,
                "model": label,
                "top_feature": top_feature,
                "top_step": top_step,
            }
        )

    return {
        "dataset": filename,
        "method": "permutation_importance_and_temporal_sensitivity",
        "feature_count": len(global_features),
        "sequence_length": len(global_temporal),
        "global_feature_importance": global_features[:12],
        "global_temporal_importance": global_temporal[:12],
        "model_comparison": comparison,
        "models": models,
    }
