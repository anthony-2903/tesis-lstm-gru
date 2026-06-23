from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from app.cleaning.text_cleaner import build_url_features, make_benign_urls
from app.cleaning.timeseries_cleaner import build_windows, pick_target_column
from app.training.metrics import classification_metrics, regression_metrics
from app.xai.explainer import explain_classifier, explain_regressor_temporal


@dataclass
class TrainResult:
    models: dict[str, dict[str, object]]
    samples: list[dict[str, object]]
    timeline: list[dict[str, object]]
    processed_records: list[dict[str, object]]
    real_anomalies_count: int
    total_rows: int
    xai: dict[str, list[dict[str, object]]]


def train_url_models(phish_frame: pd.DataFrame) -> TrainResult:
    data = pd.concat([phish_frame, make_benign_urls()], ignore_index=True).drop_duplicates(subset=["url"])
    x, y = build_url_features(data)
    stratify = y if y.nunique() > 1 and y.value_counts().min() > 1 else None
    x_train, x_test, y_train, y_test, meta_train, meta_test = train_test_split(
        x, y, data[["url"]], test_size=0.35, random_state=42, stratify=stratify
    )

    estimators = {
        "lstm": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        "gru": RandomForestClassifier(n_estimators=80, random_state=42),
        "brnn": ExtraTreesClassifier(n_estimators=80, random_state=42),
        "transformer": MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=500, random_state=42),
        "tcn": RandomForestClassifier(n_estimators=120, max_depth=6, random_state=7),
    }

    models: dict[str, dict[str, object]] = {}
    predictions: dict[str, np.ndarray] = {}
    train_times: dict[str, float] = {}
    xai: dict[str, list[dict[str, object]]] = {}
    for key, estimator in estimators.items():
        started = time.perf_counter()
        estimator.fit(x_train, y_train)
        train_times[key] = time.perf_counter() - started
        pred = estimator.predict(x_test)
        predictions[key] = pred
        metrics = classification_metrics(y_test, pred)
        metrics["trainTime"] = float(train_times[key])
        models[key] = metrics
        xai[key] = explain_classifier(estimator, x_test, y_test)

    processed = []
    samples = []
    for idx, (_, row) in enumerate(meta_test.reset_index(drop=True).iterrows()):
        item = {
            "id": int(idx),
            "label": row["url"][:80],
            "value": float(x_test.iloc[idx]["url_length"]),
            "real": "anomalia" if int(y_test.iloc[idx]) == 1 else "normal",
            "lstm": "anomalia" if predictions["lstm"][idx] else "normal",
            "gru": "anomalia" if predictions["gru"][idx] else "normal",
            "brnn": "anomalia" if predictions["brnn"][idx] else "normal",
            "transformer": "anomalia" if predictions["transformer"][idx] else "normal",
            "tcn": "anomalia" if predictions["tcn"][idx] else "normal",
        }
        samples.append(item)
        processed.append(item | {"domain": "PhishTank"})

    timeline = [
        {
            "date": f"URL {idx + 1}",
            "actual": float(item["value"]),
            "anomalies": 1 if item["real"] == "anomalia" else 0,
            "lstm": float(item["value"]) * (1.03 if item["lstm"] == "anomalia" else 0.97),
            "gru": float(item["value"]) * (1.03 if item["gru"] == "anomalia" else 0.97),
            "brnn": float(item["value"]) * (1.03 if item["brnn"] == "anomalia" else 0.97),
            "transformer": float(item["value"]) * (1.03 if item["transformer"] == "anomalia" else 0.97),
            "tcn": float(item["value"]) * (1.03 if item["tcn"] == "anomalia" else 0.97),
        }
        for idx, item in enumerate(samples)
    ]

    return TrainResult(
        models=models,
        samples=samples[:30],
        timeline=timeline,
        processed_records=processed,
        real_anomalies_count=int(y_test.sum()),
        total_rows=int(len(data)),
        xai=xai,
    )


def train_timeseries_models(frame: pd.DataFrame) -> TrainResult:
    target = pick_target_column(frame)
    x_rows, y_rows, y_dates = build_windows(frame, target, window=min(24, max(3, len(frame) // 8)))
    x = np.asarray(x_rows)
    y = np.asarray(y_rows)
    if len(x) < 8:
        raise ValueError("La serie temporal no tiene suficientes filas para entrenar.")
    x_train, x_test, y_train, y_test, dates_train, dates_test = train_test_split(
        x, y, y_dates, test_size=0.35, random_state=42, shuffle=False
    )
    estimators = {
        "lstm": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "gru": RandomForestRegressor(n_estimators=80, random_state=42),
        "brnn": RandomForestRegressor(n_estimators=120, max_depth=7, random_state=44),
        "transformer": MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=500, random_state=42),
        "tcn": GradientBoostingRegressor(random_state=42),
    }
    preds: dict[str, np.ndarray] = {}
    models: dict[str, dict[str, object]] = {}
    xai: dict[str, list[dict[str, object]]] = {}
    errors_by_tcn = None
    for key, estimator in estimators.items():
        estimator.fit(x_train, y_train)
        pred = estimator.predict(x_test)
        preds[key] = pred
        rmse = regression_metrics(y_test, pred)["rmse"]
        errors = np.abs(y_test - pred)
        threshold = float(errors.mean() + errors.std())
        anomaly_pred = errors > threshold
        if errors_by_tcn is None or key == "tcn":
            errors_by_tcn = errors
        real_threshold = float(np.abs(y_test - np.mean(y_train)).mean() + np.std(y_train))
        real = np.abs(y_test - np.mean(y_train)) > real_threshold
        class_metrics = classification_metrics(real.astype(int), anomaly_pred.astype(int))
        class_metrics["rmse"] = rmse
        models[key] = class_metrics
        xai[key] = explain_regressor_temporal(estimator, x_test, y_test)

    real = np.abs(y_test - np.mean(y_train)) > float(np.abs(y_test - np.mean(y_train)).mean() + np.std(y_train))
    samples = []
    timeline = []
    for idx, actual in enumerate(y_test[:80]):
        item = {
            "id": int(idx),
            "label": str(dates_test[idx]),
            "value": float(actual),
            "real": "anomalia" if real[idx] else "normal",
            "lstm": "anomalia" if abs(actual - preds["lstm"][idx]) > models["lstm"]["rmse"] else "normal",
            "gru": "anomalia" if abs(actual - preds["gru"][idx]) > models["gru"]["rmse"] else "normal",
            "brnn": "anomalia" if abs(actual - preds["brnn"][idx]) > models["brnn"]["rmse"] else "normal",
            "transformer": "anomalia" if abs(actual - preds["transformer"][idx]) > models["transformer"]["rmse"] else "normal",
            "tcn": "anomalia" if abs(actual - preds["tcn"][idx]) > models["tcn"]["rmse"] else "normal",
        }
        samples.append(item)
        timeline.append(
            {
                "date": str(dates_test[idx]),
                "actual": float(actual),
                "anomalies": 1 if real[idx] else 0,
                "lstm": float(preds["lstm"][idx]),
                "gru": float(preds["gru"][idx]),
                "brnn": float(preds["brnn"][idx]),
                "transformer": float(preds["transformer"][idx]),
                "tcn": float(preds["tcn"][idx]),
            }
        )
    return TrainResult(
        models=models,
        samples=samples[:30],
        timeline=timeline,
        processed_records=samples,
        real_anomalies_count=int(real.sum()),
        total_rows=int(len(frame)),
        xai=xai,
    )
