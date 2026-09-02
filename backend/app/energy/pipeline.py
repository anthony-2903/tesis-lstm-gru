from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import joblib
import numpy as np
import pandas as pd

from app.config import EXPERIMENTS_DIR
from app.energy.data import PreparedEnergyData, SequencePartition, prepare_energy_sequences, validate_energy_frame
from app.energy.metrics import energy_regression_metrics
from app.energy.models import THESIS_MODEL_IDS, build_energy_models
from app.ingestion.samples import sample_opsd


def run_energy_experiment(
    frame: pd.DataFrame,
    *,
    output_dir: Path | None = None,
    timestamp_column: str = "timestamp",
    target_column: str = "DE_load_actual_entsoe_transparency",
    window: int = 24,
    horizon: int = 1,
    gap_steps: int = 24,
    epochs: int = 20,
    batch_size: int = 32,
    seed: int = 42,
    model_ids: tuple[str, ...] = THESIS_MODEL_IDS,
    strict: bool = True,
    evaluate_test: bool = False,
) -> dict[str, Any]:
    normalized = frame.rename(columns={"utc_timestamp": timestamp_column}) if timestamp_column not in frame and "utc_timestamp" in frame else frame.copy()
    clean, validation_report = validate_energy_frame(
        normalized,
        timestamp_column=timestamp_column,
        target_column=target_column,
        strict=strict,
    )
    prepared = prepare_energy_sequences(
        clean,
        timestamp_column=timestamp_column,
        target_column=target_column,
        feature_columns=validation_report.feature_columns,
        window=window,
        horizon=horizon,
        gap_steps=gap_steps,
    )

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_energy")
    destination = output_dir or (EXPERIMENTS_DIR / run_id / "energy_v1")
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "predictions").mkdir(exist_ok=True)
    (destination / "models").mkdir(exist_ok=True)

    preprocess_path = destination / "models" / "preprocessing.joblib"
    joblib.dump(
        {
            "feature_columns": prepared.feature_columns,
            "target_column": prepared.target_column,
            "feature_imputer": prepared.feature_imputer,
            "feature_scaler": prepared.feature_scaler,
            "target_scaler": prepared.target_scaler,
            "window": window,
            "horizon": horizon,
        },
        preprocess_path,
    )

    models = build_energy_models(
        input_shape=(prepared.train.x.shape[1], prepared.train.x.shape[2]),
        target_feature_index=prepared.target_feature_index,
        model_ids=model_ids,
        epochs=epochs,
        batch_size=batch_size,
        seed=seed,
    )
    model_results = []
    for model in models:
        training = model.fit(prepared.train.x, prepared.train.y, prepared.validation.x, prepared.validation.y)
        validation_result = _evaluate_model(model, prepared.validation, prepared, destination, "validation")
        test_result = _evaluate_model(model, prepared.test, prepared, destination, "test") if evaluate_test else None
        model_path = model.save(destination / "models" / f"{model.model_id}.keras")
        model_size = model_path.stat().st_size if model_path and model_path.exists() else 0
        model_results.append(
            {
                "modelId": model.model_id,
                "family": "baseline" if model.model_id == "naive_persistence" else "base",
                "architectureVerified": model.model_id in THESIS_MODEL_IDS,
                "architecture": getattr(getattr(model, "model", None), "name", "last_observation"),
                "seed": seed,
                "hyperparameters": {
                    "window": window,
                    "horizon": horizon,
                    "epochsRequested": epochs,
                    "batchSize": batch_size,
                    "parameterCount": int(model.model.count_params()) if hasattr(model, "model") else 0,
                },
                "training": training,
                "metrics": {
                    "validation": validation_result["metrics"],
                    **({"test": test_result["metrics"]} if test_result else {}),
                },
                "operationalMetrics": {
                    "train_time_seconds": float(training["train_time_seconds"]),
                    "inference_time_ms_per_sample": float(validation_result["inference_time_ms_per_sample"]),
                    "model_size_bytes": int(model_size),
                },
                "predictionsArtifact": validation_result["predictions_artifact"],
                "testPredictionsArtifact": test_result["predictions_artifact"] if test_result else None,
                "modelArtifact": str(model_path) if model_path else None,
            }
        )

    selection_scope = "test" if evaluate_test else "validation"
    winner = min(model_results, key=lambda item: float(item["metrics"][selection_scope]["rmse"]))["modelId"]
    result = {
        "schemaVersion": "1.0.0",
        "runId": run_id,
        "protocolVersion": "1.0.0",
        "status": "demo" if not strict else "thesis_candidate",
        "domain": "energia",
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "id": "opsd_or_user_energy_series",
            "version": _frame_hash(clean),
            "rows": int(len(clean)),
            "features": prepared.feature_columns,
            "target": target_column,
            "validation": validation_report.__dict__,
        },
        "split": prepared.split_metadata,
        "preprocessingArtifact": str(preprocess_path),
        "models": model_results,
        "aggregate": {
            "primaryMetric": "rmse",
            "selectionScope": selection_scope,
            "winner": winner,
            "confidenceIntervals": {},
            "statisticalTests": [],
            "testSetEvaluatedOnce": bool(evaluate_test),
            "stackingReady": False,
            "baseModelsComplete": set(model_ids) == set(THESIS_MODEL_IDS),
        },
    }
    result_path = destination / "result.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def _evaluate_model(model, partition: SequencePartition, prepared: PreparedEnergyData, destination: Path, split_name: str) -> dict[str, Any]:
    model.predict(partition.x[: min(len(partition.x), 8)])
    started = time.perf_counter()
    prediction_scaled = model.predict(partition.x)
    duration = time.perf_counter() - started
    predictions = prepared.target_scaler.inverse_transform(prediction_scaled.reshape(-1, 1)).reshape(-1)
    actual = prepared.target_scaler.inverse_transform(partition.y.reshape(-1, 1)).reshape(-1)
    metrics = energy_regression_metrics(actual, predictions)
    artifact = destination / "predictions" / f"{model.model_id}_{split_name}.csv"
    pd.DataFrame({"timestamp": partition.timestamps, "actual": actual, "prediction": predictions, "residual": actual - predictions}).to_csv(artifact, index=False)
    return {
        "metrics": metrics,
        "inference_time_ms_per_sample": duration * 1000.0 / max(len(predictions), 1),
        "predictions_artifact": str(artifact),
    }


def _frame_hash(frame: pd.DataFrame) -> str:
    content = frame.to_csv(index=False).encode("utf-8")
    return hashlib.sha256(content).hexdigest()[:16]


def main() -> None:
    parser = argparse.ArgumentParser(description="Leakage-safe energy experiment for five neural architectures")
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--target", default="DE_load_actual_entsoe_transparency")
    parser.add_argument("--timestamp", default="timestamp")
    parser.add_argument("--window", type=int, default=24)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--gap", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--models",
        default=",".join(THESIS_MODEL_IDS),
        help="Comma-separated subset: lstm,gru,brnn,tcn,transformer",
    )
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--evaluate-test", action="store_true")
    args = parser.parse_args()
    if args.demo:
        frame = sample_opsd()
    elif args.dataset:
        frame = pd.read_csv(args.dataset)
    else:
        parser.error("Use --dataset PATH for a real experiment or --demo for a non-scientific smoke run.")
    model_ids = tuple(dict.fromkeys(item.strip().lower() for item in args.models.split(",") if item.strip()))
    if not model_ids:
        parser.error("--models debe contener al menos un modelo.")
    result = run_energy_experiment(
        frame,
        output_dir=args.output_dir,
        timestamp_column=args.timestamp,
        target_column=args.target,
        window=args.window,
        horizon=args.horizon,
        gap_steps=args.gap,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        model_ids=model_ids,
        strict=not args.demo,
        evaluate_test=args.evaluate_test,
    )
    print(json.dumps({"runId": result["runId"], "status": result["status"], "winner": result["aggregate"]["winner"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
