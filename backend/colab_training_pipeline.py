"""
Pipeline portable para Google Colab.

Uso esperado:
1. Subir un CSV/XLSX a Colab.
2. Ejecutar este script o copiar sus celdas al notebook.
3. Entrenar LSTM, GRU, Transformer y TCN.
4. Exportar metricas JSON, predicciones CSV y modelos .keras.

Ejemplo:
    python colab_training_pipeline.py --dataset /content/datos.csv --target anomaly
"""

from __future__ import annotations

import argparse
import json
import re
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras

from e_entrenamiento.model_architectures import MODEL_TRAINING_CONFIGS, build_model
from x_xai.shap_explainer import build_xai_report, explain_sequence_model


DEFAULT_SEQUENCE_LENGTH = 24
DEFAULT_TEST_SIZE = 0.2
DEFAULT_OUTPUT_DIR = "training_outputs"
DEFAULT_POSITIVE_LABELS = {
    "1",
    "true",
    "si",
    "sí",
    "yes",
    "y",
    "anomaly",
    "anomalia",
    "anomalía",
    "fraude",
    "problema",
    "con problema",
    "observado",
    "paralizado",
    "retrasado",
}
DEFAULT_NEGATIVE_LABELS = {
    "",
    ".",
    "-",
    "0",
    "false",
    "no",
    "none",
    "nan",
    "ninguno",
    "ninguna",
    "ningun problema",
    "ningún problema",
    "no corresponde",
    "no presenta",
    "no presenta problemas",
    "no presenta restricciones",
    "sin restricciones",
    "sin restricciones.",
    "sin restricciones relevantes",
    "sin acciones a implementar",
    "ninguna accion a implementar",
    "ninguna acción a implementar",
    "no se identificaron problemas",
}


def load_dataset(dataset_path: str) -> pd.DataFrame:
    path = Path(dataset_path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError("Formato no soportado. Usa CSV, XLSX o XLS.")


def encode_target(series: pd.Series) -> np.ndarray:
    if pd.api.types.is_numeric_dtype(series):
        return (series.fillna(0).astype(float) > 0).astype(int).to_numpy()

    normalized = series.fillna("").astype(str).str.strip().str.lower()
    explicit_positive = normalized.isin(DEFAULT_POSITIVE_LABELS)
    explicit_negative = normalized.isin(DEFAULT_NEGATIVE_LABELS)
    return (explicit_positive | ~explicit_negative).astype(int).to_numpy()


def infer_target(df: pd.DataFrame, target_column: str | None) -> tuple[pd.DataFrame, np.ndarray]:
    if target_column and target_column in df.columns:
        y = encode_target(df[target_column])
        x_df = df.drop(columns=[target_column]).copy()
        return x_df, y

    cleaned = df.dropna().copy()
    numeric = cleaned.select_dtypes(include=["number"])
    if numeric.empty:
        raise ValueError("El dataset necesita al menos una columna numerica o una columna objetivo.")

    reference = numeric.iloc[:, 0]
    threshold = reference.mean() + 1.7 * reference.std()
    lower_threshold = reference.mean() - 1.7 * reference.std()
    y = ((reference > threshold) | (reference < lower_threshold)).astype(int).to_numpy()
    return cleaned, y


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    features = df.copy()
    date_cols = []

    for col in features.columns:
        if features[col].dtype == "object":
            sample = features[col].dropna().astype(str).head(30)
            col_name_suggests_date = bool(re.search(r"date|fecha|time|hora|timestamp", col, re.IGNORECASE))
            values_suggest_date = sample.str.contains(
                r"^\d{4}-\d{1,2}-\d{1,2}|^\d{1,2}/\d{1,2}/\d{2,4}",
                regex=True,
            ).mean() > 0.7 if not sample.empty else False

            if not col_name_suggests_date and not values_suggest_date:
                continue

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                parsed = pd.to_datetime(features[col], errors="coerce")
            if parsed.notna().mean() > 0.7:
                date_cols.append(col)
                features[f"{col}_hour"] = parsed.dt.hour.fillna(0)
                features[f"{col}_dayofweek"] = parsed.dt.dayofweek.fillna(0)
                features[f"{col}_month"] = parsed.dt.month.fillna(0)

    features = features.drop(columns=date_cols, errors="ignore")
    features = pd.get_dummies(features, dummy_na=False)
    features = features.select_dtypes(include=["number", "bool"]).astype(float)
    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.fillna(features.median(numeric_only=True)).fillna(0)
    return features


def make_sequences(
    features: np.ndarray,
    labels: np.ndarray,
    sequence_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    x_seq = []
    y_seq = []
    for idx in range(sequence_length, len(features)):
        x_seq.append(features[idx - sequence_length : idx])
        y_seq.append(labels[idx])
    return np.asarray(x_seq, dtype=np.float32), np.asarray(y_seq, dtype=np.float32)


def prepare_data(
    df: pd.DataFrame,
    target_column: str | None,
    sequence_length: int,
    test_size: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    raw_features, labels = infer_target(df, target_column)
    feature_df = build_feature_frame(raw_features)

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(feature_df)
    x, y = make_sequences(scaled, labels, sequence_length)

    if len(x) < 20:
        raise ValueError("Dataset insuficiente para entrenar. Usa mas filas o reduce sequence_length.")

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        shuffle=False,
    )
    return x_train, x_test, y_train, y_test, feature_df.columns.tolist()


def train_single_model(
    model_key: str,
    x_train: np.ndarray,
    x_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str],
    output_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    config = MODEL_TRAINING_CONFIGS[model_key]
    model = build_model(model_key, input_shape=x_train.shape[1:])

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=config.patience,
            restore_best_weights=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=max(2, config.patience // 2),
            min_lr=1e-6,
        ),
    ]

    started = time.perf_counter()
    history = model.fit(
        x_train,
        y_train,
        validation_split=0.2,
        epochs=config.epochs,
        batch_size=config.batch_size,
        callbacks=callbacks,
        verbose=1,
    )
    training_time = time.perf_counter() - started

    probabilities = model.predict(x_test, verbose=0).ravel()
    predictions = (probabilities >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, predictions, labels=[0, 1]).ravel()

    model_path = output_dir / f"{model_key}_model.keras"
    predictions_path = output_dir / f"{model_key}_predictions.csv"
    model.save(model_path)
    pd.DataFrame(
        {
            "sample_index": np.arange(len(y_test)),
            "real_label": y_test.astype(int),
            "predicted_label": predictions.astype(int),
            "anomaly_probability": probabilities,
        }
    ).to_csv(predictions_path, index=False)

    xai_explanation = explain_sequence_model(
        model=model,
        model_key=model_key,
        model_name=config.name,
        x_background=x_train,
        x_explain=x_test,
        feature_names=feature_names,
    )

    metrics = {
        "model": config.name,
        "model_key": model_key,
        "model_path": str(model_path),
        "predictions_path": str(predictions_path),
        "epochs_ran": len(history.history["loss"]),
        "training_time_seconds": round(training_time, 3),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "auc": float(roc_auc_score(y_test, probabilities)) if len(np.unique(y_test)) > 1 else 0.0,
        "confusion_matrix": {
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(tn),
        },
        "history": {
            "loss": [float(v) for v in history.history.get("loss", [])],
            "val_loss": [float(v) for v in history.history.get("val_loss", [])],
            "precision": [float(v) for v in history.history.get("precision", [])],
            "val_precision": [float(v) for v in history.history.get("val_precision", [])],
            "recall": [float(v) for v in history.history.get("recall", [])],
            "val_recall": [float(v) for v in history.history.get("val_recall", [])],
        },
    }
    return metrics, xai_explanation


def train_all_models(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(args.dataset)
    x_train, x_test, y_train, y_test, feature_names = prepare_data(
        df=df,
        target_column=args.target,
        sequence_length=args.sequence_length,
        test_size=args.test_size,
    )

    results = {
        "dataset": Path(args.dataset).name,
        "sequence_length": args.sequence_length,
        "feature_count": len(feature_names),
        "features": feature_names,
        "train_samples": int(len(x_train)),
        "test_samples": int(len(x_test)),
        "models": {},
    }
    xai_explanations = {}

    for model_key in args.models:
        model_metrics, model_xai = train_single_model(
            model_key=model_key,
            x_train=x_train,
            x_test=x_test,
            y_train=y_train,
            y_test=y_test,
            feature_names=feature_names,
            output_dir=output_dir,
        )
        results["models"][model_key] = model_metrics
        xai_explanations[model_key] = model_xai

    xai_report = build_xai_report(
        explanations=xai_explanations,
        dataset_name=Path(args.dataset).name,
        feature_count=len(feature_names),
        sequence_length=args.sequence_length,
    )

    results_path = output_dir / "metrics.json"
    xai_path = output_dir / "xai.json"
    results["xai_path"] = str(xai_path)
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    xai_path.write_text(json.dumps(xai_report, indent=2), encoding="utf-8")
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Ruta del CSV/XLSX subido a Colab.")
    parser.add_argument("--target", default=None, help="Columna objetivo. Si se omite, se infiere anomalia por desviacion.")
    parser.add_argument("--sequence-length", type=int, default=DEFAULT_SEQUENCE_LENGTH)
    parser.add_argument("--test-size", type=float, default=DEFAULT_TEST_SIZE)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["lstm", "gru", "transformer", "tcn"],
        choices=["lstm", "gru", "transformer", "tcn"],
    )
    return parser.parse_args()


if __name__ == "__main__":
    tf.keras.utils.set_random_seed(42)
    final_results = train_all_models(parse_args())
    print(json.dumps(final_results, indent=2))
