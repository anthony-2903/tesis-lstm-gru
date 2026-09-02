from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss

from app.config import EXPERIMENTS_DIR, SILVER_DIR
from app.finance.metrics import finance_classification_metrics
from app.finance.models import FINANCE_MODEL_IDS, FinanceModelConfig, KerasFinanceClassifier
from app.finance.sequences import FEATURE_COLUMNS, fit_finance_scaler, scale_finance_sequences
from app.finance.stacking import build_finance_meta_features, build_finance_meta_models
from app.phishing.ingestion import sha256_file
from app.utils import read_json


ClassifierFactory = Callable[..., Any]
SIX_CANDIDATE_IDS = (*FINANCE_MODEL_IDS, "stacking")
METRIC_NAMES = (
    "prAuc",
    "rocAuc",
    "f1",
    "precision",
    "recall",
    "mcc",
    "balancedAccuracy",
    "brierScore",
    "logLoss",
    "falsePositiveRate",
)


def run_finance_temporal_validation(
    *,
    base_manifest_path: Path | None = None,
    stacking_manifest_path: Path | None = None,
    sequence_manifest_path: Path | None = None,
    sequences_path: Path | None = None,
    features_path: Path | None = None,
    output_dir: Path | None = None,
    demo_max_train_rows: int | None = 8_000,
    demo_max_validation_rows: int | None = None,
    bootstrap_iterations: int = 500,
    classifier_factory: ClassifierFactory = KerasFinanceClassifier,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Select and calibrate six financial candidates without touching final test."""
    if demo_max_train_rows is not None and demo_max_train_rows < 500:
        raise ValueError("El piloto de validacion requiere al menos 500 filas de train.")
    if demo_max_validation_rows is not None and demo_max_validation_rows < 500:
        raise ValueError("El piloto de validacion requiere al menos 500 filas de validation.")
    if bootstrap_iterations < 100:
        raise ValueError("Se requieren al menos 100 remuestreos bootstrap.")

    base_path = base_manifest_path or _latest_base_manifest()
    if base_path is None or not base_path.is_file():
        raise FileNotFoundError("No existe una corrida OOF financiera para validar.")
    base_manifest = read_json(base_path)
    model_ids = tuple(base_manifest.get("baseModels", []))
    seeds = tuple(int(value) for value in base_manifest.get("configuration", {}).get("seeds", []))
    if set(model_ids) != set(FINANCE_MODEL_IDS) or not seeds:
        raise ValueError("La validacion financiera requiere los cinco modelos base y al menos una semilla.")
    base_validation = base_manifest.get("validation", {})
    if not base_validation.get("testSetLocked") or base_validation.get("testSetUsed") or base_validation.get("testSetEncoded"):
        raise ValueError("La corrida base no conserva el bloqueo del test financiero.")

    oof_artifact = base_manifest.get("artifacts", {}).get("oofProbabilities", {})
    oof_path = Path(oof_artifact.get("path", ""))
    _verify_file(oof_path, oof_artifact, "probabilidades OOF")
    oof = pd.read_csv(oof_path)
    probability_columns = [f"probability_{model_id}" for model_id in model_ids]
    required_oof = {"transaction_id", "seed", "fold", "is_fraud", *probability_columns}
    if required_oof - set(oof.columns) or oof.duplicated(["seed", "transaction_id"]).any():
        raise ValueError("El artefacto OOF financiero esta incompleto o duplicado.")
    if oof[probability_columns].isna().any().any():
        raise ValueError("El artefacto OOF financiero contiene probabilidades faltantes.")

    stacking_path = stacking_manifest_path or (base_path.parent / "stacking_v1" / "stacking_manifest.json")
    if not stacking_path.is_file():
        raise FileNotFoundError("Primero construya el Stacking financiero temporal.")
    stacking_manifest = read_json(stacking_path)
    stacking_source_id = stacking_manifest.get("recommendation", {}).get("leadingStackingCandidateId")
    if stacking_source_id not in {"stacking_logistic", "stacking_gradient_boosting"}:
        raise ValueError("El manifiesto de Stacking no congela un meta-learner valido.")
    if stacking_manifest.get("validation", {}).get("testSetUsed"):
        raise ValueError("El Stacking de origen utilizo test y no puede validarse.")

    sequence_manifest_source = sequence_manifest_path or (SILVER_DIR / "finance_sequence_manifest.json")
    sequence_manifest = read_json(sequence_manifest_source)
    test_lock = sequence_manifest.get("testLock", {})
    if not test_lock.get("locked") or test_lock.get("evaluated") or test_lock.get("encoded"):
        raise ValueError("El test financiero debe seguir bloqueado, no codificado y no evaluado.")
    sequence_source = sequences_path or Path(sequence_manifest.get("artifacts", {}).get("sequences", {}).get("path", SILVER_DIR / "finance_sequences.npz"))
    feature_source = features_path or Path(sequence_manifest.get("artifacts", {}).get("features", {}).get("path", SILVER_DIR / "finance_causal_features.csv"))
    _verify_file(sequence_source, sequence_manifest.get("artifacts", {}).get("sequences", {}), "secuencias train-validation")
    _verify_file(feature_source, sequence_manifest.get("artifacts", {}).get("features", {}), "variables causales")
    sequence_hash, sequence_bytes = sha256_file(sequence_source)
    feature_hash, feature_bytes = sha256_file(feature_source)

    with np.load(sequence_source, allow_pickle=False) as arrays:
        x = arrays["x"].astype(np.float32, copy=True)
        labels = arrays["y"].astype(np.int32, copy=True)
        transaction_ids = arrays["transaction_id"].astype(np.int64, copy=True)
        timestamps = arrays["timestamp_ns"].astype(np.int64, copy=True)
        split_codes = arrays["split_code"].astype(np.int8, copy=True)
    if set(np.unique(split_codes)) - {0, 1}:
        raise ValueError("El tensor contiene un split no autorizado; test debe permanecer ausente.")
    features = pd.read_csv(feature_source)
    if len(features) != len(x) or not np.array_equal(features["transaction_id"].to_numpy(dtype=np.int64), transaction_ids):
        raise ValueError("Las variables causales y las secuencias no comparten cobertura y orden.")

    train_indices = np.flatnonzero(split_codes == 0).astype(np.int64)
    validation_indices = np.flatnonzero(split_codes == 1).astype(np.int64)
    train_indices = train_indices[np.argsort(timestamps[train_indices], kind="stable")]
    validation_indices = validation_indices[np.argsort(timestamps[validation_indices], kind="stable")]
    if demo_max_train_rows is not None:
        train_indices = train_indices[-demo_max_train_rows:]
    if demo_max_validation_rows is not None:
        validation_indices = validation_indices[:demo_max_validation_rows]
    if timestamps[train_indices].max() >= timestamps[validation_indices].min():
        raise ValueError("Train y validation no respetan el orden temporal.")
    _require_binary(labels[train_indices], "train seleccionado")
    calibration_positions, selection_positions = chronological_validation_split(labels[validation_indices])
    calibration_indices = validation_indices[calibration_positions]
    selection_indices = validation_indices[selection_positions]

    scaler = fit_finance_scaler(features.iloc[train_indices])
    scaler["fitPolicy"] = "selected_training_rows_only_never_validation_or_test"
    scaler_hash = _json_hash(scaler)
    x_train = scale_finance_sequences(x[train_indices], scaler)
    y_train = labels[train_indices]
    x_validation = scale_finance_sequences(x[validation_indices], scaler)
    y_validation = labels[validation_indices]
    validation_transaction_ids = transaction_ids[validation_indices]
    class_weight = _balanced_class_weights(y_train)

    destination = output_dir or (base_path.parent / "temporal_validation_v1")
    destination.mkdir(parents=True, exist_ok=True)
    scaler_path = destination / "scaler" / "train_only_scaler.json"
    _atomic_write_json(scaler_path, scaler)
    scaler_file_hash, scaler_file_bytes = sha256_file(scaler_path)
    total_units = len(seeds) * (len(model_ids) + 1) + len(SIX_CANDIDATE_IDS)
    completed_units = 0
    resumed_units = 0
    base_predictions_by_seed: dict[int, dict[str, np.ndarray]] = {}
    fitted_artifacts: list[dict[str, Any]] = []
    training_records: list[dict[str, Any]] = []
    _notify(progress_callback, stage="refit", event="prepared", completedUnits=0, resumedUnits=0, totalUnits=total_units, message="Train y validation verificados; test permanece bloqueado.")

    selected_train_hash = _ordered_hash(transaction_ids[train_indices])
    selected_validation_hash = _ordered_hash(validation_transaction_ids)
    for seed in seeds:
        base_predictions_by_seed[seed] = {}
        for model_id in model_ids:
            selected_epochs = _selected_epoch_count(base_manifest, seed=seed, model_id=model_id)
            fingerprint = _json_hash({
                "schemaVersion": "1.0.0",
                "modelId": model_id,
                "seed": seed,
                "epochs": selected_epochs,
                "batchSize": int(base_manifest.get("configuration", {}).get("batchSize", 128)),
                "trainTransactionsSha256": selected_train_hash,
                "validationTransactionsSha256": selected_validation_hash,
                "sequenceSha256": sequence_hash,
                "scalerSha256": scaler_hash,
                "modelConfiguration": FinanceModelConfig().__dict__,
            })
            restored = _load_refit_checkpoint(
                destination,
                seed=seed,
                model_id=model_id,
                fingerprint=fingerprint,
                transaction_ids=validation_transaction_ids,
                labels=y_validation,
            )
            if restored is not None:
                probabilities, record, artifact = restored
                base_predictions_by_seed[seed][model_id] = probabilities
                training_records.append(record)
                fitted_artifacts.append(artifact)
                completed_units += 1
                resumed_units += 1
                _notify(progress_callback, stage="refit", event="model_resumed", completedUnits=completed_units, resumedUnits=resumed_units, totalUnits=total_units, seed=seed, modelId=model_id, message=f"{model_id.upper()} recuperado de un checkpoint verificado.")
                continue

            _notify(progress_callback, stage="refit", event="model_started", completedUnits=completed_units, resumedUnits=resumed_units, totalUnits=total_units, seed=seed, modelId=model_id, message=f"Reentrenando {model_id.upper()} solo con train; semilla {seed}.")
            tf.keras.backend.clear_session()
            classifier = classifier_factory(
                model_id=model_id,
                input_shape=(int(x.shape[1]), int(x.shape[2])),
                epochs=selected_epochs,
                batch_size=int(base_manifest.get("configuration", {}).get("batchSize", 128)),
                patience=-1,
                seed=seed,
                config=FinanceModelConfig(),
            )
            training = classifier.fit_full(x_train, y_train, class_weight=class_weight)
            probabilities, inference_seconds = classifier.predict_proba(x_validation)
            probabilities = np.asarray(probabilities, dtype=np.float64)
            _validate_probabilities(probabilities, len(validation_indices), model_id)
            model_path = destination / "checkpoints" / f"seed_{seed}" / model_id / "model.keras"
            model_hash, model_bytes = classifier.save(model_path)
            record = {
                "modelId": model_id,
                "seed": seed,
                "selectedEpochsFromOof": selected_epochs,
                "epochsCompleted": int(training["epochsCompleted"]),
                "fitRows": int(len(train_indices)),
                "validationRowsPredicted": int(len(validation_indices)),
                "trainTimeSeconds": float(training["trainTimeSeconds"]),
                "inferenceTimeMsPerSample": float(inference_seconds * 1000.0 / max(len(validation_indices), 1)),
                "modelSizeBytes": int(model_bytes),
                "resumed": False,
            }
            artifact = _save_refit_checkpoint(
                destination,
                seed=seed,
                model_id=model_id,
                fingerprint=fingerprint,
                transaction_ids=validation_transaction_ids,
                labels=y_validation,
                probabilities=probabilities,
                record=record,
                model_path=model_path,
                model_hash=model_hash,
                model_bytes=model_bytes,
            )
            base_predictions_by_seed[seed][model_id] = probabilities
            training_records.append(record)
            fitted_artifacts.append(artifact)
            completed_units += 1
            _notify(progress_callback, stage="refit", event="model_completed", completedUnits=completed_units, resumedUnits=resumed_units, totalUnits=total_units, seed=seed, modelId=model_id, message=f"{model_id.upper()} produjo probabilidades de validation.")
            del classifier
            tf.keras.backend.clear_session()

    rows_by_seed: list[dict[str, Any]] = []
    feature_names: list[str] = []
    for seed in seeds:
        seed_oof = oof[oof["seed"] == seed].sort_values(["fold", "transaction_id"], kind="stable")
        if seed_oof.empty:
            raise ValueError(f"No existen probabilidades OOF para la semilla {seed}.")
        x_oof_base = seed_oof[probability_columns].to_numpy(dtype=np.float64)
        y_oof = seed_oof["is_fraud"].to_numpy(dtype=np.int32)
        x_oof_meta, feature_names = build_finance_meta_features(x_oof_base, probability_columns)
        x_validation_base = np.column_stack([base_predictions_by_seed[seed][model_id] for model_id in model_ids])
        x_validation_meta, _ = build_finance_meta_features(x_validation_base, probability_columns)
        meta_model = build_finance_meta_models(seed=seed)[stacking_source_id]
        if stacking_source_id == "stacking_gradient_boosting":
            meta_model.fit(x_oof_meta, y_oof, sample_weight=_balanced_sample_weights(y_oof))
        else:
            meta_model.fit(x_oof_meta, y_oof)
        stacking_probabilities = np.clip(meta_model.predict_proba(x_validation_meta)[:, 1], 0.0, 1.0)
        meta_path = destination / "meta_models" / f"seed_{seed}" / f"{stacking_source_id}.joblib"
        _atomic_joblib_dump(meta_path, meta_model)
        meta_hash, meta_bytes = sha256_file(meta_path)
        fitted_artifacts.append({"kind": "meta_model", "candidateId": stacking_source_id, "seed": seed, "path": str(meta_path.resolve()), "sha256": meta_hash, "bytes": meta_bytes})
        for position, row_index in enumerate(validation_indices):
            rows_by_seed.append({
                "transaction_id": int(transaction_ids[row_index]),
                "timestamp_ns": int(timestamps[row_index]),
                "seed": int(seed),
                "is_fraud": int(labels[row_index]),
                **{f"probability_{model_id}": float(base_predictions_by_seed[seed][model_id][position]) for model_id in model_ids},
                "probability_stacking": float(stacking_probabilities[position]),
            })
        completed_units += 1
        _notify(progress_callback, stage="meta_model", event="seed_completed", completedUnits=completed_units, resumedUnits=resumed_units, totalUnits=total_units, seed=seed, modelId="stacking", message=f"Stacking ajustado exclusivamente con OOF para la semilla {seed}.")

    predictions = pd.DataFrame(rows_by_seed).sort_values(["seed", "timestamp_ns", "transaction_id"], kind="stable")
    probability_output_columns = [f"probability_{candidate_id}" for candidate_id in SIX_CANDIDATE_IDS]
    if len(predictions) != len(validation_indices) * len(seeds) or predictions.duplicated(["seed", "transaction_id"]).any():
        raise ValueError("Las predicciones de validation no cubren exactamente cada transaccion y semilla.")
    mean_predictions = predictions.groupby(["transaction_id", "timestamp_ns", "is_fraud"], as_index=False)[probability_output_columns].mean()
    mean_predictions = mean_predictions.sort_values(["timestamp_ns", "transaction_id"], kind="stable").reset_index(drop=True)
    if len(mean_predictions) != len(validation_indices):
        raise ValueError("El promedio entre semillas altero la cobertura de validation.")
    mean_predictions["validation_partition"] = "selection"
    mean_predictions.loc[calibration_positions, "validation_partition"] = "calibration"

    comparison: list[dict[str, Any]] = []
    calibration_records: list[dict[str, Any]] = []
    calibrated_selection_scores: dict[str, np.ndarray] = {}
    y_calibration = mean_predictions.loc[calibration_positions, "is_fraud"].to_numpy(dtype=np.int32)
    y_selection = mean_predictions.loc[selection_positions, "is_fraud"].to_numpy(dtype=np.int32)
    for candidate_id in SIX_CANDIDATE_IDS:
        raw_scores = mean_predictions[f"probability_{candidate_id}"].to_numpy(dtype=np.float64)
        calibrator = fit_temporal_calibrator(y_calibration, raw_scores[calibration_positions])
        calibrated_calibration = apply_temporal_calibrator(calibrator, raw_scores[calibration_positions])
        threshold = select_fbeta_threshold(y_calibration, calibrated_calibration, beta=2.0)
        calibrated_selection = apply_temporal_calibrator(calibrator, raw_scores[selection_positions])
        calibrated_selection_scores[candidate_id] = calibrated_selection
        metrics = finance_classification_metrics(y_selection, calibrated_selection, threshold=threshold)
        raw_metrics = finance_classification_metrics(y_selection, raw_scores[selection_positions], threshold=0.5)
        calibration_path = destination / "calibration" / f"{candidate_id}.joblib"
        _atomic_joblib_dump(calibration_path, calibrator)
        calibration_hash, calibration_bytes = sha256_file(calibration_path)
        calibration_record = {
            "candidateId": candidate_id,
            "method": calibrator["method"],
            "methodSelection": calibrator["methodSelection"],
            "fitRows": int(len(y_calibration)),
            "fitEndTimestampNs": int(mean_predictions.loc[calibration_positions, "timestamp_ns"].max()),
            "selectionStartTimestampNs": int(mean_predictions.loc[selection_positions, "timestamp_ns"].min()),
            "threshold": threshold,
            "thresholdObjective": "f2",
            "selectionLabelsUsedForCalibration": False,
            "selectionLabelsUsedForThreshold": False,
            "path": str(calibration_path.resolve()),
            "sha256": calibration_hash,
            "bytes": calibration_bytes,
        }
        calibration_records.append(calibration_record)
        comparison.append({
            "candidateId": candidate_id,
            "sourceCandidateId": stacking_source_id if candidate_id == "stacking" else None,
            "family": "stacking" if candidate_id == "stacking" else "base",
            "validationSelectionRows": int(len(y_selection)),
            "calibrationMethod": calibrator["method"],
            "calibratedThreshold": threshold,
            "thresholdObjective": "f2",
            **{name: metrics[name] for name in METRIC_NAMES},
            "rawFixedThresholdMetrics": {name: raw_metrics[name] for name in METRIC_NAMES},
            "confusionMatrix": {key: metrics[key] for key in ("truePositive", "trueNegative", "falsePositive", "falseNegative")},
        })
        mean_predictions.loc[selection_positions, f"calibrated_probability_{candidate_id}"] = calibrated_selection
        completed_units += 1
        _notify(progress_callback, stage="calibration", event="candidate_completed", completedUnits=completed_units, resumedUnits=resumed_units, totalUnits=total_units, modelId=candidate_id, message=f"{candidate_id.upper()} calibrado y evaluado en la particion de seleccion.")

    comparison.sort(key=lambda row: (row["prAuc"], row["mcc"], row["f1"]), reverse=True)
    winner = comparison[0]
    leading_stacking = next(row for row in comparison if row["candidateId"] == "stacking")
    best_base = next(row for row in comparison if row["family"] == "base")
    bootstrap = paired_stratified_pr_auc_bootstrap(
        y_selection,
        calibrated_selection_scores["stacking"],
        calibrated_selection_scores[best_base["candidateId"]],
        iterations=bootstrap_iterations,
        seed=42,
    )

    predictions_path = destination / "validation_probabilities_by_seed.csv"
    mean_path = destination / "validation_probabilities_mean_calibrated.csv"
    metrics_path = destination / "validation_selection_metrics.json"
    freeze_path = destination / "frozen_selection.json"
    _atomic_write_csv(predictions_path, predictions)
    _atomic_write_csv(mean_path, mean_predictions)
    _atomic_write_json(metrics_path, {"items": comparison, "bootstrap": bootstrap})
    real_dataset = bool(sequence_manifest.get("readiness", {}).get("readyForThesisTraining"))
    full_thesis_validation = (
        real_dataset
        and base_manifest.get("protocol") == "thesis"
        and base_manifest.get("status") != "demo"
        and len(seeds) >= 5
        and demo_max_train_rows is None
        and demo_max_validation_rows is None
    )
    frozen_selection = {
        "schemaVersion": "1.0.0",
        "candidateId": winner["candidateId"],
        "sourceCandidateId": winner.get("sourceCandidateId"),
        "primaryMetric": "prAuc",
        "calibrationMethod": winner["calibrationMethod"],
        "threshold": winner["calibratedThreshold"],
        "thresholdObjective": "f2",
        "selectedOn": "chronological_second_half_of_validation",
        "syntheticBenchmark": not real_dataset,
        "testSetLocked": True,
        "testSetUsed": False,
        "eligibleForFinalThesisClaim": False,
        "eligibleForFinalTestEvaluation": full_thesis_validation,
    }
    _atomic_write_json(freeze_path, frozen_selection)
    artifacts = {}
    for key, path, rows in (
        ("predictionsBySeed", predictions_path, len(predictions)),
        ("meanPredictions", mean_path, len(mean_predictions)),
        ("metrics", metrics_path, len(comparison)),
        ("frozenSelection", freeze_path, 1),
    ):
        digest, size = sha256_file(path)
        artifacts[key] = {"path": str(path.resolve()), "sha256": digest, "bytes": size, "rows": int(rows)}

    manifest = {
        "schemaVersion": "1.0.0",
        "runId": f"{base_manifest.get('runId')}_temporal_validation",
        "domain": "finanzas",
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "status": "demo_validation_selection" if base_manifest.get("status") == "demo" else "validation_selection_candidate",
        "baseRun": {"runId": base_manifest.get("runId"), "manifestPath": str(base_path.resolve()), "oofPath": str(oof_path.resolve()), "models": list(model_ids), "seeds": list(seeds)},
        "stackingRun": {"manifestPath": str(stacking_path.resolve()), "sourceCandidateId": stacking_source_id, "selectionPolicy": "leading_stacking_candidate_selected_on_temporal_oof_only"},
        "dataset": {
            "datasetId": sequence_manifest.get("datasetId"),
            "kind": "real_world_curated_financial_dataset" if real_dataset else "synthetic_transactional_benchmark",
            "sequencePath": str(sequence_source.resolve()),
            "sequenceSha256": sequence_hash,
            "sequenceBytes": sequence_bytes,
            "featurePath": str(feature_source.resolve()),
            "featureSha256": feature_hash,
            "featureBytes": feature_bytes,
            "trainRowsAvailable": int((split_codes == 0).sum()),
            "trainRowsUsed": int(len(train_indices)),
            "validationRowsAvailable": int((split_codes == 1).sum()),
            "validationRowsUsed": int(len(validation_indices)),
            "demoTrainSubset": demo_max_train_rows is not None,
            "demoValidationSubset": demo_max_validation_rows is not None,
        },
        "training": {
            "epochSelection": "median_epochs_completed_across_oof_folds_per_seed_and_model",
            "validationUsedForEarlyStopping": False,
            "records": training_records,
            "resumable": True,
            "resumedUnits": resumed_units,
            "trainedUnitsThisInvocation": len(seeds) * len(model_ids) - resumed_units,
            "scaler": {"path": str(scaler_path.resolve()), "sha256": scaler_file_hash, "bytes": scaler_file_bytes, "fitRows": int(len(train_indices)), "validationRowsUsed": 0, "testRowsUsed": 0},
        },
        "metaFeatures": {"columns": feature_names, "fitSplit": "train_oof_only", "validationLabelsUsedForMetaFit": False, "selectedCandidateId": stacking_source_id},
        "calibration": {
            "strategy": "chronological_first_half_of_validation_with_internal_fit_check_split",
            "rows": int(len(calibration_positions)),
            "candidateMethods": ["identity", "platt", "isotonic"],
            "methodSelectionMetric": "brier_then_log_loss",
            "thresholdObjective": "f2",
            "records": calibration_records,
            "selectionLabelsUsed": False,
            "testLabelsUsed": False,
        },
        "selection": {
            "strategy": "chronological_second_half_of_validation",
            "rows": int(len(selection_positions)),
            "candidatePrimaryMetric": "prAuc",
            "winnerCandidateId": winner["candidateId"],
            "winnerThreshold": winner["calibratedThreshold"],
            "winnerCalibrationMethod": winner["calibrationMethod"],
            "leadingStackingCandidateId": stacking_source_id,
            "stackingRank": next(index + 1 for index, row in enumerate(comparison) if row["candidateId"] == "stacking"),
            "stackingBeatsBestBase": bool(leading_stacking["prAuc"] > best_base["prAuc"]),
            "status": "frozen_real_validation_pending_single_final_test" if full_thesis_validation else "frozen_pilot_pending_real_full_protocol_and_final_test",
        },
        "comparison": comparison,
        "statisticalComparison": {"stackingVersusBestBase": {"bestBaseCandidateId": best_base["candidateId"], **bootstrap}},
        "validation": {
            "sameRowsForAllSixCandidates": True,
            "chronologicalCalibrationBeforeSelection": True,
            "calibrationEndTimestampNs": int(mean_predictions.loc[calibration_positions, "timestamp_ns"].max()),
            "selectionStartTimestampNs": int(mean_predictions.loc[selection_positions, "timestamp_ns"].min()),
            "futureLeakagePassed": True,
            "externalRealDatasetUsed": real_dataset,
            "testSetLocked": True,
            "testSetEncoded": False,
            "testSetUsed": False,
            "readyForFinalTestEvaluation": full_thesis_validation,
            "interpretation": "real_temporal_selection_pending_single_final_test" if full_thesis_validation else "temporal_selection_not_final_thesis_performance",
        },
        "freeze": frozen_selection,
        "artifacts": {**artifacts, "fittedObjects": fitted_artifacts},
    }
    manifest_path = destination / "temporal_validation_manifest.json"
    _atomic_write_json(manifest_path, manifest)
    manifest_hash, manifest_bytes = sha256_file(manifest_path)
    _notify(progress_callback, stage="completed", event="completed", completedUnits=total_units, resumedUnits=resumed_units, totalUnits=total_units, message="Validacion temporal completada; configuracion congelada y test bloqueado.")
    return {**manifest, "manifest": {"path": str(manifest_path.resolve()), "sha256": manifest_hash, "bytes": manifest_bytes}}


def chronological_validation_split(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(labels, dtype=np.int32).reshape(-1)
    if len(values) < 20:
        raise ValueError("Validation es insuficiente para calibracion y seleccion temporales.")
    split = len(values) // 2
    calibration = np.arange(0, split, dtype=np.int64)
    selection = np.arange(split, len(values), dtype=np.int64)
    _require_binary(values[calibration], "calibracion")
    _require_binary(values[selection], "seleccion")
    return calibration, selection


def fit_temporal_calibrator(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    y_true = np.asarray(labels, dtype=np.int32).reshape(-1)
    scores = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    _validate_probabilities(scores, len(y_true), "calibration")
    fit_positions, check_positions = _chronological_calibration_check_split(y_true)
    candidates: list[tuple[str, Any]] = [("identity", None)]
    platt = LogisticRegression(C=1.0, max_iter=2_000, solver="lbfgs", random_state=42)
    platt.fit(_logits(scores[fit_positions]).reshape(-1, 1), y_true[fit_positions])
    candidates.append(("platt", platt))
    isotonic = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    isotonic.fit(scores[fit_positions], y_true[fit_positions])
    candidates.append(("isotonic", isotonic))
    evaluations = []
    for method, model in candidates:
        checked = _apply_calibration_model(method, model, scores[check_positions])
        evaluations.append({
            "method": method,
            "brierScore": float(brier_score_loss(y_true[check_positions], checked)),
            "logLoss": float(log_loss(y_true[check_positions], np.clip(checked, 1e-7, 1 - 1e-7), labels=[0, 1])),
        })
    evaluations.sort(key=lambda row: (row["brierScore"], row["logLoss"], {"identity": 0, "platt": 1, "isotonic": 2}[row["method"]]))
    method = evaluations[0]["method"]
    if method == "identity":
        final_model = None
    elif method == "platt":
        final_model = LogisticRegression(C=1.0, max_iter=2_000, solver="lbfgs", random_state=42)
        final_model.fit(_logits(scores).reshape(-1, 1), y_true)
    else:
        final_model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        final_model.fit(scores, y_true)
    return {
        "method": method,
        "model": final_model,
        "methodSelection": {
            "fitRows": int(len(fit_positions)),
            "checkRows": int(len(check_positions)),
            "chronological": True,
            "metricOrder": ["brierScore", "logLoss"],
            "evaluations": evaluations,
        },
    }


def apply_temporal_calibrator(calibrator: dict[str, Any], probabilities: np.ndarray) -> np.ndarray:
    return _apply_calibration_model(calibrator["method"], calibrator.get("model"), np.asarray(probabilities, dtype=np.float64))


def select_fbeta_threshold(labels: np.ndarray, probabilities: np.ndarray, *, beta: float = 2.0) -> float:
    y_true = np.asarray(labels, dtype=np.int32).reshape(-1)
    scores = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    _validate_probabilities(scores, len(y_true), "threshold")
    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    sorted_labels = y_true[order]
    tp = np.cumsum(sorted_labels == 1)
    fp = np.cumsum(sorted_labels == 0)
    total_positive = int((y_true == 1).sum())
    change_positions = np.flatnonzero(np.r_[sorted_scores[1:] != sorted_scores[:-1], True])
    best_threshold = 0.5
    best_key = (-np.inf, -np.inf, -np.inf, -np.inf)
    beta_squared = beta * beta
    negatives = max(int((y_true == 0).sum()), 1)
    for position in change_positions:
        true_positive = int(tp[position])
        false_positive = int(fp[position])
        false_negative = total_positive - true_positive
        denominator = (1 + beta_squared) * true_positive + beta_squared * false_negative + false_positive
        fbeta = ((1 + beta_squared) * true_positive / denominator) if denominator else 0.0
        recall = true_positive / total_positive if total_positive else 0.0
        fpr = false_positive / negatives
        threshold = float(sorted_scores[position])
        key = (fbeta, recall, -fpr, threshold)
        if key > best_key:
            best_key = key
            best_threshold = threshold
    return float(np.clip(best_threshold, 0.0, 1.0))


def paired_stratified_pr_auc_bootstrap(
    labels: np.ndarray,
    stacking_probabilities: np.ndarray,
    base_probabilities: np.ndarray,
    *,
    iterations: int = 500,
    seed: int = 42,
) -> dict[str, Any]:
    y_true = np.asarray(labels, dtype=np.int32).reshape(-1)
    stacking = np.asarray(stacking_probabilities, dtype=np.float64).reshape(-1)
    base = np.asarray(base_probabilities, dtype=np.float64).reshape(-1)
    _require_binary(y_true, "bootstrap")
    if not (len(y_true) == len(stacking) == len(base)):
        raise ValueError("El bootstrap pareado requiere igual cobertura.")
    positive = np.flatnonzero(y_true == 1)
    negative = np.flatnonzero(y_true == 0)
    generator = np.random.default_rng(seed)
    deltas = np.empty(iterations, dtype=np.float64)
    for iteration in range(iterations):
        sampled = np.concatenate([
            generator.choice(positive, size=len(positive), replace=True),
            generator.choice(negative, size=len(negative), replace=True),
        ])
        sampled_labels = y_true[sampled]
        deltas[iteration] = average_precision_score(sampled_labels, stacking[sampled]) - average_precision_score(sampled_labels, base[sampled])
    observed = float(average_precision_score(y_true, stacking) - average_precision_score(y_true, base))
    return {
        "metric": "prAuc",
        "method": "paired_stratified_percentile_bootstrap",
        "iterations": int(iterations),
        "seed": int(seed),
        "observedDelta": observed,
        "confidenceLevel": 0.95,
        "ciLower": float(np.quantile(deltas, 0.025)),
        "ciUpper": float(np.quantile(deltas, 0.975)),
        "probabilityStackingBetter": float(np.mean(deltas > 0.0)),
        "statisticallyClearAt95Percent": bool(np.quantile(deltas, 0.025) > 0.0 or np.quantile(deltas, 0.975) < 0.0),
    }


def get_latest_finance_temporal_validation() -> dict[str, Any]:
    paths = sorted(
        EXPERIMENTS_DIR.glob("*/finance_oof_v1/temporal_validation_v1/temporal_validation_manifest.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not paths:
        return {"available": False, "message": "Todavia no existe una validacion temporal financiera."}
    return {"available": True, **read_json(paths[0])}


def _chronological_calibration_check_split(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(labels, dtype=np.int32).reshape(-1)
    preferred = int(len(values) * 0.60)
    candidates = sorted(range(max(2, int(len(values) * 0.4)), min(len(values) - 2, int(len(values) * 0.8)) + 1), key=lambda value: abs(value - preferred))
    for split in candidates:
        if set(np.unique(values[:split])) == {0, 1} and set(np.unique(values[split:])) == {0, 1}:
            return np.arange(split, dtype=np.int64), np.arange(split, len(values), dtype=np.int64)
    raise ValueError("La calibracion temporal no permite fit/check con ambas clases.")


def _apply_calibration_model(method: str, model: Any, scores: np.ndarray) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if method == "identity":
        calibrated = values
    elif method == "platt":
        calibrated = model.predict_proba(_logits(values).reshape(-1, 1))[:, 1]
    elif method == "isotonic":
        calibrated = model.predict(values)
    else:
        raise ValueError(f"Metodo de calibracion no soportado: {method}")
    return np.clip(np.asarray(calibrated, dtype=np.float64), 0.0, 1.0)


def _logits(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    return np.log(clipped / (1.0 - clipped))


def _selected_epoch_count(base_manifest: dict[str, Any], *, seed: int, model_id: str) -> int:
    values = [
        int(row.get("epochsCompleted", 1))
        for row in base_manifest.get("baseFoldMetrics", [])
        if int(row.get("seed", -1)) == seed and row.get("modelId") == model_id
    ]
    fallback = int(base_manifest.get("configuration", {}).get("epochs", 1))
    return max(1, int(round(float(np.median(values))))) if values else max(1, fallback)


def _balanced_class_weights(labels: np.ndarray) -> dict[int, float]:
    values, counts = np.unique(labels, return_counts=True)
    return {int(value): float(len(labels) / (len(values) * count)) for value, count in zip(values, counts)}


def _balanced_sample_weights(labels: np.ndarray) -> np.ndarray:
    weights = _balanced_class_weights(labels)
    return np.asarray([weights[int(value)] for value in labels], dtype=np.float64)


def _require_binary(labels: np.ndarray, label: str) -> None:
    if set(np.unique(np.asarray(labels, dtype=np.int32))) != {0, 1}:
        raise ValueError(f"La particion {label} debe contener ambas clases.")


def _validate_probabilities(probabilities: np.ndarray, expected_rows: int, label: str) -> None:
    values = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if len(values) != expected_rows or not np.isfinite(values).all() or np.any((values < 0.0) | (values > 1.0)):
        raise ValueError(f"Las probabilidades de {label} no cumplen el contrato.")


def _verify_file(path: Path, artifact: dict[str, Any], label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"No existe el artefacto de {label}.")
    digest, size = sha256_file(path)
    if digest != artifact.get("sha256") or size != artifact.get("bytes"):
        raise ValueError(f"El artefacto de {label} no supera SHA-256.")


def _save_refit_checkpoint(
    destination: Path,
    *,
    seed: int,
    model_id: str,
    fingerprint: str,
    transaction_ids: np.ndarray,
    labels: np.ndarray,
    probabilities: np.ndarray,
    record: dict[str, Any],
    model_path: Path,
    model_hash: str,
    model_bytes: int,
) -> dict[str, Any]:
    root = destination / "checkpoints" / f"seed_{seed}" / model_id
    prediction_path = root / "validation_predictions.npz"
    _atomic_write_npz(prediction_path, transaction_id=transaction_ids, y=labels, probability=probabilities)
    prediction_hash, prediction_bytes = sha256_file(prediction_path)
    metadata_path = root / "checkpoint.json"
    metadata = {
        "schemaVersion": "1.0.0",
        "fingerprint": fingerprint,
        "record": record,
        "model": {"path": str(model_path.resolve()), "sha256": model_hash, "bytes": int(model_bytes)},
        "predictions": {"path": str(prediction_path.resolve()), "sha256": prediction_hash, "bytes": prediction_bytes},
    }
    _atomic_write_json(metadata_path, metadata)
    metadata_hash, metadata_bytes = sha256_file(metadata_path)
    return {"kind": "base_model", "candidateId": model_id, "seed": seed, "path": str(model_path.resolve()), "sha256": model_hash, "bytes": int(model_bytes), "checkpointPath": str(metadata_path.resolve()), "checkpointSha256": metadata_hash, "checkpointBytes": metadata_bytes}


def _load_refit_checkpoint(
    destination: Path,
    *,
    seed: int,
    model_id: str,
    fingerprint: str,
    transaction_ids: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]] | None:
    root = destination / "checkpoints" / f"seed_{seed}" / model_id
    metadata_path = root / "checkpoint.json"
    if not metadata_path.is_file():
        return None
    try:
        metadata = read_json(metadata_path)
        if metadata.get("fingerprint") != fingerprint:
            return None
        model = metadata["model"]
        predictions = metadata["predictions"]
        model_path = Path(model["path"])
        prediction_path = Path(predictions["path"])
        _verify_file(model_path, model, f"modelo {model_id}")
        _verify_file(prediction_path, predictions, f"predicciones {model_id}")
        with np.load(prediction_path, allow_pickle=False) as arrays:
            restored_ids = arrays["transaction_id"].astype(np.int64)
            restored_labels = arrays["y"].astype(np.int32)
            probabilities = arrays["probability"].astype(np.float64)
        if not np.array_equal(restored_ids, transaction_ids) or not np.array_equal(restored_labels, labels):
            return None
        _validate_probabilities(probabilities, len(labels), model_id)
        metadata_hash, metadata_bytes = sha256_file(metadata_path)
        record = {**metadata["record"], "resumed": True}
        artifact = {"kind": "base_model", "candidateId": model_id, "seed": seed, "path": str(model_path.resolve()), "sha256": model["sha256"], "bytes": int(model["bytes"]), "checkpointPath": str(metadata_path.resolve()), "checkpointSha256": metadata_hash, "checkpointBytes": metadata_bytes}
        return probabilities, record, artifact
    except (OSError, KeyError, ValueError, json.JSONDecodeError):
        return None


def _json_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _ordered_hash(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype=np.int64).tobytes()).hexdigest()


def _latest_base_manifest() -> Path | None:
    paths = sorted(EXPERIMENTS_DIR.glob("*/finance_oof_v1/oof_manifest.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    return paths[0] if paths else None


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _atomic_write_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def _atomic_joblib_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    joblib.dump(value, temporary)
    os.replace(temporary, path)


def _notify(callback: Callable[[dict[str, Any]], None] | None, **event: Any) -> None:
    if callback is not None:
        callback(event)
