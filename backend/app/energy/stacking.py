from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge

from app.energy.metrics import energy_regression_metrics


ENSEMBLE_IDS = ("mean", "weighted_mean", "stacking_gradient_boosting")


def evaluate_energy_ensembles(
    oof_predictions: pd.DataFrame,
    *,
    base_model_ids: tuple[str, ...],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    required = {"seed", "fold", "timestamp", "actual", *base_model_ids}
    missing = sorted(required - set(oof_predictions.columns))
    if missing:
        raise ValueError(f"Faltan columnas OOF para stacking: {', '.join(missing)}")
    if len(base_model_ids) < 2:
        raise ValueError("El stacking requiere al menos dos modelos base.")

    prediction_rows: list[dict[str, Any]] = []
    fold_metrics: list[dict[str, Any]] = []
    for seed in sorted(oof_predictions["seed"].unique()):
        seed_data = oof_predictions[oof_predictions["seed"] == seed].copy()
        folds = sorted(int(value) for value in seed_data["fold"].unique())
        for fold in folds[1:]:
            meta_train = seed_data[seed_data["fold"] < fold].copy()
            meta_validation = seed_data[seed_data["fold"] == fold].copy()
            x_train, x_validation, feature_columns = _meta_features(meta_train, meta_validation, base_model_ids)
            y_train = meta_train["actual"].to_numpy(dtype=float)
            y_validation = meta_validation["actual"].to_numpy(dtype=float)

            candidates = {
                "mean": meta_validation[list(base_model_ids)].mean(axis=1).to_numpy(dtype=float),
                "weighted_mean": _fit_predict(
                    Ridge(alpha=1.0, positive=True),
                    meta_train[list(base_model_ids)].to_numpy(dtype=float),
                    y_train,
                    meta_validation[list(base_model_ids)].to_numpy(dtype=float),
                ),
                "stacking_gradient_boosting": _fit_predict(
                    GradientBoostingRegressor(
                        n_estimators=80,
                        learning_rate=0.05,
                        max_depth=2,
                        random_state=int(seed),
                    ),
                    x_train,
                    y_train,
                    x_validation,
                ),
            }
            for ensemble_id, predictions in candidates.items():
                metrics = energy_regression_metrics(y_validation, predictions)
                fold_metrics.append(
                    {
                        "seed": int(seed),
                        "fold": fold,
                        "ensembleId": ensemble_id,
                        "metaTrainMaxFold": fold - 1,
                        "metaFeatures": feature_columns if ensemble_id == "stacking_gradient_boosting" else list(base_model_ids),
                        **metrics,
                    }
                )
                for row_index, (_, source) in enumerate(meta_validation.reset_index(drop=True).iterrows()):
                    prediction_rows.append(
                        {
                            "seed": int(seed),
                            "fold": fold,
                            "timestamp": source["timestamp"],
                            "actual": float(source["actual"]),
                            "ensemble": ensemble_id,
                            "prediction": float(predictions[row_index]),
                            "residual": float(source["actual"] - predictions[row_index]),
                        }
                    )

    if not fold_metrics:
        raise ValueError("No existen suficientes folds OOF para entrenar y validar el stacking.")
    metrics_frame = pd.DataFrame(fold_metrics)
    aggregate = []
    for ensemble_id, group in metrics_frame.groupby("ensembleId", sort=False):
        aggregate.append(
            {
                "ensembleId": ensemble_id,
                "foldEvaluations": int(len(group)),
                "rmseMean": float(group["rmse"].mean()),
                "rmseStd": float(group["rmse"].std(ddof=1)) if len(group) > 1 else 0.0,
                "maeMean": float(group["mae"].mean()),
                "smapeMean": float(group["smape"].mean()),
                "r2Mean": float(group["r2"].mean()),
            }
        )
    winner = min(aggregate, key=lambda item: item["rmseMean"])["ensembleId"]
    report = {
        "selectionScope": "walk_forward_oof_meta_validation",
        "testSetUsed": False,
        "baseModels": list(base_model_ids),
        "winner": winner,
        "aggregate": aggregate,
        "foldMetrics": fold_metrics,
    }
    return pd.DataFrame(prediction_rows), report


def _meta_features(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    base_model_ids: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    feature_columns = [*base_model_ids, "prediction_mean", "prediction_std", "prediction_range"]
    return (
        _build_meta_matrix(train, base_model_ids),
        _build_meta_matrix(validation, base_model_ids),
        feature_columns,
    )


def _build_meta_matrix(frame: pd.DataFrame, base_model_ids: tuple[str, ...]) -> np.ndarray:
    base = frame[list(base_model_ids)].to_numpy(dtype=float)
    summaries = np.column_stack(
        [
            base.mean(axis=1),
            base.std(axis=1),
            base.max(axis=1) - base.min(axis=1),
        ]
    )
    return np.column_stack([base, summaries])


def _fit_predict(estimator, x_train: np.ndarray, y_train: np.ndarray, x_validation: np.ndarray) -> np.ndarray:
    estimator.fit(x_train, y_train)
    return np.asarray(estimator.predict(x_validation), dtype=float).reshape(-1)
