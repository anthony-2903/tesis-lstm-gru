from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Callable

import numpy as np
import pandas as pd
import tensorflow as tf

from app.config import EXPERIMENTS_DIR
from app.energy.anomalies import detect_walk_forward_anomalies
from app.energy.data import PreparedEnergyFold, prepare_energy_oof_folds, validate_energy_frame
from app.energy.metrics import energy_regression_metrics
from app.energy.models import THESIS_MODEL_IDS, build_energy_models
from app.energy.stacking import evaluate_energy_ensembles
from app.ingestion.samples import sample_opsd


def run_energy_oof_experiment(
    frame: pd.DataFrame,
    *,
    output_dir: Path | None = None,
    timestamp_column: str = "timestamp",
    target_column: str = "DE_load_actual_entsoe_transparency",
    window: int = 24,
    horizon: int = 1,
    gap_steps: int = 24,
    n_splits: int = 5,
    epochs: int = 20,
    batch_size: int = 32,
    seeds: tuple[int, ...] = (42, 101, 202, 303, 404),
    model_ids: tuple[str, ...] = THESIS_MODEL_IDS,
    strict: bool = True,
    source_lineage: dict[str, Any] | None = None,
    execution_run_id: str | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    domain: str = "energia",
    artifact_namespace: str = "energy_oof_v1",
    minimum_rows: int = 8760,
    expected_frequency: str = "hourly",
) -> dict[str, Any]:
    normalized = frame.rename(columns={"utc_timestamp": timestamp_column}) if timestamp_column not in frame and "utc_timestamp" in frame else frame.copy()
    clean, validation_report = validate_energy_frame(
        normalized,
        timestamp_column=timestamp_column,
        target_column=target_column,
        strict=strict,
        minimum_rows=minimum_rows,
        expected_frequency=expected_frequency,
    )
    folds = prepare_energy_oof_folds(
        clean,
        timestamp_column=timestamp_column,
        target_column=target_column,
        feature_columns=validation_report.feature_columns,
        window=window,
        horizon=horizon,
        n_splits=n_splits,
        gap_steps=gap_steps,
    )
    run_id = execution_run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_energy_oof")
    destination = output_dir or (EXPERIMENTS_DIR / run_id / artifact_namespace)
    destination.mkdir(parents=True, exist_ok=True)

    frame_sha256 = hashlib.sha256(
        pd.util.hash_pandas_object(
            clean[[timestamp_column, *validation_report.feature_columns]],
            index=False,
        ).to_numpy(dtype=np.uint64).tobytes()
    ).hexdigest()
    execution_contract = {
        "rows": int(len(clean)),
        "startAt": validation_report.start_at,
        "endAt": validation_report.end_at,
        "target": target_column,
        "features": validation_report.feature_columns,
        "window": int(window),
        "horizon": int(horizon),
        "gapSteps": int(gap_steps),
        "folds": int(n_splits),
        "epochs": int(epochs),
        "batchSize": int(batch_size),
        "seeds": [int(value) for value in seeds],
        "modelIds": list(model_ids),
        "strict": bool(strict),
        "domain": domain,
        "artifactNamespace": artifact_namespace,
        "minimumRows": int(minimum_rows),
        "expectedFrequency": expected_frequency,
        "sourceSha256": (source_lineage or {}).get("sha256"),
        "validatedFrameSha256": frame_sha256,
    }
    fingerprint = hashlib.sha256(
        json.dumps(execution_contract, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    checkpoint = _open_checkpoint(destination, run_id, fingerprint, execution_contract)

    oof_rows: list[dict[str, Any]] = []
    base_fold_metrics: list[dict[str, Any]] = []
    total_training_units = len(seeds) * len(folds) * len(model_ids)
    completed_training_units = 0
    resumed_training_units = 0
    _notify_progress(
        progress_callback,
        stage="training",
        event="prepared",
        completedUnits=0,
        totalUnits=total_training_units,
        message="Datos validados y folds temporales preparados.",
    )
    for seed in seeds:
        for fold in folds:
            actual = fold.target_scaler.inverse_transform(fold.validation.y.reshape(-1, 1)).reshape(-1)
            predictions_by_model: dict[str, np.ndarray] = {}
            for model_id in model_ids:
                restored = _load_checkpoint_unit(
                    destination,
                    fingerprint=fingerprint,
                    seed=int(seed),
                    fold=int(fold.fold_id),
                    model_id=model_id,
                    actual=actual,
                    timestamps=fold.validation.timestamps,
                )
                if restored is not None:
                    predictions_by_model[model_id] = restored["prediction"]
                    base_fold_metrics.append(restored["metrics"])
                    completed_training_units += 1
                    resumed_training_units += 1
                    _notify_progress(
                        progress_callback,
                        stage="training",
                        event="model_resumed",
                        completedUnits=completed_training_units,
                        resumedUnits=resumed_training_units,
                        totalUnits=total_training_units,
                        seed=int(seed),
                        fold=int(fold.fold_id),
                        modelId=model_id,
                        message=f"{model_id.upper()} recuperado de checkpoint verificado en fold {fold.fold_id + 1}.",
                    )
                    continue
                _notify_progress(
                    progress_callback,
                    stage="training",
                    event="model_started",
                    completedUnits=completed_training_units,
                    totalUnits=total_training_units,
                    seed=int(seed),
                    fold=int(fold.fold_id),
                    modelId=model_id,
                    message=f"Entrenando {model_id.upper()} en fold {fold.fold_id + 1}/{len(folds)}.",
                )
                tf.keras.backend.clear_session()
                model = build_energy_models(
                    input_shape=(fold.train.x.shape[1], fold.train.x.shape[2]),
                    target_feature_index=fold.target_feature_index,
                    model_ids=(model_id,),
                    epochs=epochs,
                    batch_size=batch_size,
                    seed=seed,
                    include_naive=False,
                )[0]
                training = model.fit(fold.train.x, fold.train.y, fold.validation.x, fold.validation.y)
                started = time.perf_counter()
                prediction_scaled = model.predict(fold.validation.x)
                inference_seconds = time.perf_counter() - started
                prediction = fold.target_scaler.inverse_transform(prediction_scaled.reshape(-1, 1)).reshape(-1)
                predictions_by_model[model_id] = prediction
                metrics_row = {
                    "seed": int(seed),
                    "fold": int(fold.fold_id),
                    "modelId": model_id,
                    "trainTimeSeconds": float(training["train_time_seconds"]),
                    "inferenceTimeMsPerSample": max(inference_seconds, 0.0) * 1000.0 / max(len(prediction), 1),
                    **energy_regression_metrics(actual, prediction),
                }
                _save_checkpoint_unit(
                    destination,
                    checkpoint=checkpoint,
                    fingerprint=fingerprint,
                    seed=int(seed),
                    fold=int(fold.fold_id),
                    model_id=model_id,
                    actual=actual,
                    prediction=prediction,
                    timestamps=fold.validation.timestamps,
                    metrics=metrics_row,
                )
                base_fold_metrics.append(metrics_row)
                completed_training_units += 1
                _notify_progress(
                    progress_callback,
                    stage="training",
                    event="model_completed",
                    completedUnits=completed_training_units,
                    totalUnits=total_training_units,
                    seed=int(seed),
                    fold=int(fold.fold_id),
                    modelId=model_id,
                    message=f"{model_id.upper()} completado en fold {fold.fold_id + 1}/{len(folds)}.",
                )
            for index, timestamp in enumerate(fold.validation.timestamps):
                oof_rows.append(
                    {
                        "seed": seed,
                        "fold": fold.fold_id,
                        "timestamp": str(timestamp),
                        "actual": float(actual[index]),
                        **{model_id: float(predictions_by_model[model_id][index]) for model_id in model_ids},
                    }
                )

    oof_frame = pd.DataFrame(oof_rows)
    oof_path = destination / "oof_predictions.csv"
    oof_frame.to_csv(oof_path, index=False)
    _notify_progress(
        progress_callback,
        stage="stacking",
        event="started",
        completedUnits=completed_training_units,
        totalUnits=total_training_units,
        message="Entrenando y evaluando los metamodelos con predicciones OOF.",
    )
    ensemble_predictions, ensemble_report = evaluate_energy_ensembles(oof_frame, base_model_ids=model_ids)
    ensemble_path = destination / "ensemble_oof_predictions.csv"
    ensemble_predictions.to_csv(ensemble_path, index=False)

    anomaly_input = _build_common_anomaly_input(oof_frame, ensemble_predictions, model_ids)
    anomaly_predictions, anomaly_report = detect_walk_forward_anomalies(anomaly_input)
    anomaly_path = destination / "anomaly_predictions.csv"
    anomaly_report_path = destination / "anomaly_report.json"
    anomaly_predictions.to_csv(anomaly_path, index=False)
    anomaly_report_path.write_text(json.dumps(anomaly_report, ensure_ascii=False, indent=2), encoding="utf-8")

    common_evaluation_pairs = _common_evaluation_pairs(ensemble_predictions)
    full_protocol = len(folds) >= 5 and len(seeds) >= 5 and set(model_ids) == set(THESIS_MODEL_IDS) and strict
    artifact_integrity = {
        key: _artifact_descriptor(path)
        for key, path in {
            "oofPredictions": oof_path,
            "ensemblePredictions": ensemble_path,
            "anomalyPredictions": anomaly_path,
            "anomalyReport": anomaly_report_path,
        }.items()
    }
    manifest = {
        "schemaVersion": "1.0.0",
        "runId": run_id,
        "protocolVersion": "1.0.0",
        "status": "thesis_candidate" if full_protocol else "demo",
        "domain": domain,
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "rows": int(len(clean)),
            "features": validation_report.feature_columns,
            "target": target_column,
            "validation": validation_report.__dict__,
            "sourceLineage": source_lineage or {"type": "in_memory_or_demo", "verified": False},
        },
        "walkForward": {
            "folds": [fold.metadata for fold in folds],
            "seeds": list(seeds),
            "window": window,
            "horizon": horizon,
            "testSetUsed": False,
        },
        "baseModels": list(model_ids),
        "baseFoldMetrics": base_fold_metrics,
        "ensembles": ensemble_report,
        "comparisonScope": {
            "strategy": "common_seed_fold_pairs_available_to_base_and_ensemble_predictors",
            "pairs": common_evaluation_pairs,
            "fairComparison": True,
        },
        "anomalies": {
            "labelType": anomaly_report["labelType"],
            "classificationMetricsAvailable": anomaly_report["classificationMetricsAvailable"],
            "recommendedMethod": anomaly_report["recommendedMethod"],
            "summaries": anomaly_report["summaries"],
            "commonEvaluationPairs": common_evaluation_pairs[1:],
            "testSetUsed": False,
        },
        "artifacts": {
            "oofPredictions": str(oof_path),
            "ensemblePredictions": str(ensemble_path),
            "anomalyPredictions": str(anomaly_path),
            "anomalyReport": str(anomaly_report_path),
        },
        "artifactIntegrity": artifact_integrity,
        "execution": {
            "fingerprint": fingerprint,
            "checkpointPath": str((destination / "checkpoint_manifest.json").resolve()),
            "resumable": True,
            "resumeCount": int(checkpoint.get("resumeCount", 0)),
            "resumedUnits": resumed_training_units,
            "trainedUnitsThisInvocation": total_training_units - resumed_training_units,
        },
        "aggregate": {
            "primaryMetric": "rmse",
            "winner": ensemble_report["winner"],
            "testSetEvaluatedOnce": False,
            "stackingReady": full_protocol,
        },
    }
    manifest_path = destination / "oof_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    checkpoint.update(
        status="completed",
        completedAt=datetime.now(timezone.utc).isoformat(),
        completedUnits=total_training_units,
        finalManifest=_artifact_descriptor(manifest_path),
    )
    _atomic_json(destination / "checkpoint_manifest.json", checkpoint)
    _notify_progress(
        progress_callback,
        stage="completed",
        event="completed",
        completedUnits=total_training_units,
        totalUnits=total_training_units,
        message="Experimento energético completado y artefactos persistidos.",
    )
    return manifest


def _build_common_anomaly_input(
    oof_frame: pd.DataFrame,
    ensemble_predictions: pd.DataFrame,
    model_ids: tuple[str, ...],
) -> pd.DataFrame:
    eligible_pairs = {(int(row.seed), int(row.fold)) for row in ensemble_predictions[["seed", "fold"]].drop_duplicates().itertuples()}
    base_prediction_rows = []
    for _, row in oof_frame.iterrows():
        if (int(row["seed"]), int(row["fold"])) not in eligible_pairs:
            continue
        for model_id in model_ids:
            prediction = float(row[model_id])
            base_prediction_rows.append(
                {
                    "seed": int(row["seed"]),
                    "fold": int(row["fold"]),
                    "timestamp": row["timestamp"],
                    "actual": float(row["actual"]),
                    "prediction": prediction,
                    "residual": float(row["actual"] - prediction),
                    "predictorFamily": "base",
                    "predictorId": model_id,
                }
            )
    ensemble_long = ensemble_predictions.rename(columns={"ensemble": "predictorId"}).copy()
    ensemble_long["predictorFamily"] = "ensemble"
    return pd.concat(
        [pd.DataFrame(base_prediction_rows), ensemble_long],
        ignore_index=True,
        sort=False,
    )


def _common_evaluation_pairs(ensemble_predictions: pd.DataFrame) -> list[dict[str, int]]:
    pairs = ensemble_predictions[["seed", "fold"]].drop_duplicates().sort_values(["seed", "fold"])
    return [{"seed": int(row.seed), "fold": int(row.fold)} for row in pairs.itertuples()]


def _open_checkpoint(
    destination: Path,
    run_id: str,
    fingerprint: str,
    contract: dict[str, Any],
) -> dict[str, Any]:
    path = destination / "checkpoint_manifest.json"
    if path.exists():
        checkpoint = json.loads(path.read_text(encoding="utf-8"))
        if checkpoint.get("fingerprint") != fingerprint or checkpoint.get("contract") != contract:
            raise ValueError("El checkpoint energetico no coincide con el dataset o la configuracion solicitada.")
        checkpoint["resumeCount"] = int(checkpoint.get("resumeCount", 0)) + 1
        checkpoint["status"] = "in_progress"
        checkpoint["lastResumedAt"] = datetime.now(timezone.utc).isoformat()
    else:
        checkpoint = {
            "schemaVersion": "1.0.0",
            "runId": run_id,
            "fingerprint": fingerprint,
            "contract": contract,
            "status": "in_progress",
            "createdAt": datetime.now(timezone.utc).isoformat(),
            "resumeCount": 0,
            "completedUnits": 0,
            "completedUnitKeys": [],
            "testSetLocked": True,
            "testSetUsed": False,
        }
    _atomic_json(path, checkpoint)
    return checkpoint


def _load_checkpoint_unit(
    destination: Path,
    *,
    fingerprint: str,
    seed: int,
    fold: int,
    model_id: str,
    actual: np.ndarray,
    timestamps: np.ndarray,
) -> dict[str, Any] | None:
    unit_path = destination / "checkpoints" / f"seed_{seed}" / f"fold_{fold}" / f"{model_id}.json"
    if not unit_path.exists():
        return None
    try:
        unit = json.loads(unit_path.read_text(encoding="utf-8"))
        if unit.get("fingerprint") != fingerprint:
            return None
        artifact = unit["artifact"]
        path = Path(artifact["path"])
        observed_artifact = _artifact_descriptor(path)
        if observed_artifact["sha256"] != artifact.get("sha256") or observed_artifact["bytes"] != artifact.get("bytes"):
            return None
        with np.load(path, allow_pickle=False) as payload:
            restored_actual = payload["actual"].astype(float)
            prediction = payload["prediction"].astype(float)
            restored_timestamps = payload["timestamps"].astype(str)
        if not np.allclose(restored_actual, actual, rtol=0.0, atol=1e-8):
            return None
        if restored_timestamps.tolist() != np.asarray(timestamps).astype(str).tolist():
            return None
        if len(prediction) != len(actual) or not np.isfinite(prediction).all():
            return None
        return {"prediction": prediction, "metrics": unit["metrics"]}
    except (OSError, KeyError, ValueError, json.JSONDecodeError):
        return None


def _save_checkpoint_unit(
    destination: Path,
    *,
    checkpoint: dict[str, Any],
    fingerprint: str,
    seed: int,
    fold: int,
    model_id: str,
    actual: np.ndarray,
    prediction: np.ndarray,
    timestamps: np.ndarray,
    metrics: dict[str, Any],
) -> None:
    unit_root = destination / "checkpoints" / f"seed_{seed}" / f"fold_{fold}"
    unit_root.mkdir(parents=True, exist_ok=True)
    artifact_path = unit_root / f"{model_id}.npz"
    temporary = artifact_path.with_name(f"{artifact_path.stem}.tmp.npz")
    np.savez_compressed(
        temporary,
        actual=np.asarray(actual, dtype=np.float64),
        prediction=np.asarray(prediction, dtype=np.float64),
        timestamps=np.asarray(timestamps).astype(str),
    )
    os.replace(temporary, artifact_path)
    key = f"seed={seed}|fold={fold}|model={model_id}"
    unit = {
        "schemaVersion": "1.0.0",
        "unitKey": key,
        "fingerprint": fingerprint,
        "completedAt": datetime.now(timezone.utc).isoformat(),
        "metrics": metrics,
        "artifact": _artifact_descriptor(artifact_path),
    }
    _atomic_json(unit_root / f"{model_id}.json", unit)
    completed = set(checkpoint.get("completedUnitKeys", []))
    completed.add(key)
    checkpoint["completedUnitKeys"] = sorted(completed)
    checkpoint["completedUnits"] = len(completed)
    checkpoint["updatedAt"] = datetime.now(timezone.utc).isoformat()
    _atomic_json(destination / "checkpoint_manifest.json", checkpoint)


def _artifact_descriptor(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return {"path": str(path.resolve()), "sha256": digest.hexdigest(), "bytes": size}


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def rebuild_energy_postprocessing(manifest_path: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    model_ids = tuple(manifest["baseModels"])
    artifacts = manifest["artifacts"]
    oof_frame = pd.read_csv(artifacts["oofPredictions"])
    ensemble_predictions, ensemble_report = evaluate_energy_ensembles(oof_frame, base_model_ids=model_ids)
    ensemble_predictions.to_csv(artifacts["ensemblePredictions"], index=False)
    anomaly_input = _build_common_anomaly_input(oof_frame, ensemble_predictions, model_ids)
    anomaly_predictions, anomaly_report = detect_walk_forward_anomalies(anomaly_input)
    anomaly_predictions.to_csv(artifacts["anomalyPredictions"], index=False)
    Path(artifacts["anomalyReport"]).write_text(json.dumps(anomaly_report, ensure_ascii=False, indent=2), encoding="utf-8")
    common_pairs = _common_evaluation_pairs(ensemble_predictions)
    manifest["ensembles"] = ensemble_report
    manifest["comparisonScope"] = {
        "strategy": "common_seed_fold_pairs_available_to_base_and_ensemble_predictors",
        "pairs": common_pairs,
        "fairComparison": True,
    }
    manifest["anomalies"] = {
        "labelType": anomaly_report["labelType"],
        "classificationMetricsAvailable": anomaly_report["classificationMetricsAvailable"],
        "recommendedMethod": anomaly_report["recommendedMethod"],
        "summaries": anomaly_report["summaries"],
        "commonEvaluationPairs": common_pairs[1:],
        "testSetUsed": False,
    }
    manifest["aggregate"]["winner"] = ensemble_report["winner"]
    manifest["artifactIntegrity"] = {
        key: _artifact_descriptor(Path(value))
        for key, value in artifacts.items()
        if value and Path(value).is_file()
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def _notify_progress(callback: Callable[[dict[str, Any]], None] | None, **event: Any) -> None:
    if callback is not None:
        callback(event)


def main() -> None:
    parser = argparse.ArgumentParser(description="Energy walk-forward OOF and stacking experiment")
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--target", default="DE_load_actual_entsoe_transparency")
    parser.add_argument("--timestamp", default="timestamp")
    parser.add_argument("--window", type=int, default=24)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--gap", type=int, default=24)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seeds", default="42,101,202,303,404")
    parser.add_argument("--models", default=",".join(THESIS_MODEL_IDS))
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()
    if args.demo:
        frame = sample_opsd()
    elif args.dataset:
        frame = pd.read_csv(args.dataset)
    else:
        parser.error("Use --dataset PATH o --demo.")
    seeds = tuple(dict.fromkeys(int(item.strip()) for item in args.seeds.split(",") if item.strip()))
    model_ids = tuple(dict.fromkeys(item.strip().lower() for item in args.models.split(",") if item.strip()))
    result = run_energy_oof_experiment(
        frame,
        output_dir=args.output_dir,
        timestamp_column=args.timestamp,
        target_column=args.target,
        window=args.window,
        horizon=args.horizon,
        gap_steps=args.gap,
        n_splits=args.folds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seeds=seeds,
        model_ids=model_ids,
        strict=not args.demo,
    )
    print(json.dumps({"runId": result["runId"], "status": result["status"], "winner": result["aggregate"]["winner"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
