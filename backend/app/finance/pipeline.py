from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import tensorflow as tf

from app.config import EXPERIMENTS_DIR, SILVER_DIR
from app.finance.checkpoints import FinanceOOFCheckpointStore
from app.finance.metrics import finance_classification_metrics
from app.finance.models import FINANCE_MODEL_IDS, FinanceModelConfig, KerasFinanceClassifier
from app.finance.sequences import scale_finance_sequences
from app.phishing.ingestion import sha256_file
from app.utils import read_json, write_json


ClassifierFactory = Callable[..., Any]


def run_finance_oof_experiment(
    *,
    output_dir: Path | None = None,
    sequences_path: Path | None = None,
    assignments_path: Path | None = None,
    sequence_manifest_path: Path | None = None,
    protocol: str = "demo",
    epochs: int = 1,
    batch_size: int = 128,
    patience: int = 1,
    seeds: tuple[int, ...] = (42,),
    model_ids: tuple[str, ...] = FINANCE_MODEL_IDS,
    demo_max_rows_per_fold: int | None = 1_000,
    execution_run_id: str | None = None,
    classifier_factory: ClassifierFactory = KerasFinanceClassifier,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    if protocol not in {"demo", "thesis"}:
        raise ValueError("El protocolo financiero debe ser demo o thesis.")
    unknown = sorted(set(model_ids) - set(FINANCE_MODEL_IDS))
    if unknown:
        raise ValueError(f"Modelos financieros no soportados: {', '.join(unknown)}")
    if not model_ids or not seeds:
        raise ValueError("Se requiere al menos un modelo y una semilla financiera.")
    if demo_max_rows_per_fold is not None and demo_max_rows_per_fold < 100:
        raise ValueError("El piloto requiere al menos 100 filas holdout por fold.")
    if protocol == "thesis" and (
        set(model_ids) != set(FINANCE_MODEL_IDS) or len(seeds) < 5 or demo_max_rows_per_fold is not None
    ):
        raise ValueError("El protocolo thesis exige cinco modelos, cinco semillas y todos los holdouts.")

    sequence_source = sequences_path or (SILVER_DIR / "finance_sequences.npz")
    assignment_source = assignments_path or (SILVER_DIR / "finance_oof_assignments.csv")
    manifest_source = sequence_manifest_path or (SILVER_DIR / "finance_sequence_manifest.json")
    if not sequence_source.is_file() or not assignment_source.is_file() or not manifest_source.is_file():
        raise FileNotFoundError("Primero prepare las secuencias y folds OOF financieros.")
    sequence_manifest = read_json(manifest_source)
    readiness = sequence_manifest.get("readiness", {})
    if not readiness.get("readyForBaseModelPilot"):
        raise ValueError("El protocolo secuencial financiero no está listo para modelos base.")
    if protocol == "thesis" and not readiness.get("readyForThesisTraining"):
        raise ValueError("El benchmark financiero actual no está habilitado para resultados de tesis.")
    test_lock = sequence_manifest.get("testLock", {})
    if not test_lock.get("locked") or test_lock.get("evaluated") or test_lock.get("encoded"):
        raise ValueError("La política de bloqueo del test financiero no es válida.")

    sequence_hash, sequence_bytes = sha256_file(sequence_source)
    assignment_hash, assignment_bytes = sha256_file(assignment_source)
    manifest_hash, manifest_bytes = sha256_file(manifest_source)
    artifacts = sequence_manifest.get("artifacts", {})
    if artifacts.get("sequences", {}).get("sha256") != sequence_hash:
        raise ValueError("El tensor financiero cambió respecto de su manifiesto.")
    if artifacts.get("assignments", {}).get("sha256") != assignment_hash:
        raise ValueError("Las asignaciones financieras cambiaron respecto de su manifiesto.")

    with np.load(sequence_source, allow_pickle=False) as arrays:
        x = arrays["x"].astype(np.float32, copy=True)
        labels = arrays["y"].astype(np.int32, copy=True)
        transaction_ids = arrays["transaction_id"].astype(np.int64, copy=True)
        timestamps = arrays["timestamp_ns"].astype(np.int64, copy=True)
        split_codes = arrays["split_code"].astype(np.int8, copy=True)
    if len(x) != len(labels) or len(x) != len(transaction_ids) or len(x) != len(timestamps):
        raise ValueError("Los tensores financieros no comparten la misma cobertura.")
    if set(np.unique(split_codes)) - {0, 1}:
        raise ValueError("El tensor financiero contiene un split no autorizado; test debe permanecer ausente.")
    assignments = pd.read_csv(assignment_source)
    required_assignment_columns = {"transaction_id", "row_index", "oof_fold"}
    if required_assignment_columns - set(assignments.columns):
        raise ValueError("Las asignaciones OOF financieras no cumplen el contrato.")
    assignments = assignments.astype({"transaction_id": "int64", "row_index": "int64", "oof_fold": "int64"})
    if assignments["transaction_id"].duplicated().any() or assignments["row_index"].duplicated().any():
        raise ValueError("Las asignaciones OOF financieras no son únicas.")
    if np.any(assignments["row_index"].to_numpy() >= len(x)) or np.any(assignments["row_index"].to_numpy() < 0):
        raise ValueError("Una asignación OOF apunta fuera del tensor financiero.")
    assigned_indices = assignments["row_index"].to_numpy(dtype=np.int64)
    if not np.array_equal(transaction_ids[assigned_indices], assignments["transaction_id"].to_numpy(dtype=np.int64)):
        raise ValueError("Los identificadores de asignaciones y secuencias no coinciden.")
    if np.any(split_codes[assigned_indices] != 0):
        raise ValueError("OOF solo puede cubrir filas de train.")

    fold_manifests = {int(item["fold"]): item for item in sequence_manifest.get("oof", {}).get("folds", [])}
    available_folds = sorted(int(value) for value in assignments["oof_fold"].unique())
    if available_folds != [0, 1, 2, 3, 4] or set(fold_manifests) != set(available_folds):
        raise ValueError("El protocolo financiero requiere exactamente los cinco folds versionados.")

    run_id = execution_run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_finance_oof")
    destination = output_dir or (EXPERIMENTS_DIR / run_id / "finance_oof_v1")
    destination.mkdir(parents=True, exist_ok=True)
    selected_holdouts: dict[int, np.ndarray] = {}
    fit_indices_by_fold: dict[int, np.ndarray] = {}
    for fold in available_folds:
        holdout_indices = assignments.loc[assignments["oof_fold"] == fold, "row_index"].to_numpy(dtype=np.int64)
        if protocol == "demo" and demo_max_rows_per_fold is not None:
            holdout_indices = _deterministic_stratified_indices(
                holdout_indices,
                labels,
                transaction_ids,
                maximum_rows=demo_max_rows_per_fold,
                salt=f"holdout:{fold}",
            )
        selected_holdouts[fold] = holdout_indices
        fit_end_ns = pd.Timestamp(fold_manifests[fold]["fitEndAt"]).value
        fit_indices = np.flatnonzero((split_codes == 0) & (timestamps <= fit_end_ns)).astype(np.int64)
        fit_indices = fit_indices[np.argsort(timestamps[fit_indices], kind="stable")]
        if protocol == "demo" and demo_max_rows_per_fold is not None:
            fit_cap = max(2_000, demo_max_rows_per_fold * 4)
            fit_indices = fit_indices[-fit_cap:]
        if not len(fit_indices) or timestamps[fit_indices].max() >= timestamps[holdout_indices].min():
            raise ValueError(f"El fold financiero {fold} presenta fuga temporal.")
        fit_indices_by_fold[fold] = fit_indices

    execution_contract = {
        "protocol": protocol,
        "epochs": int(epochs),
        "batchSize": int(batch_size),
        "patience": int(patience),
        "seeds": [int(seed) for seed in seeds],
        "modelIds": list(model_ids),
        "demoMaxRowsPerFold": demo_max_rows_per_fold,
        "datasetId": sequence_manifest.get("datasetId"),
        "sequenceSha256": sequence_hash,
        "assignmentsSha256": assignment_hash,
        "sequenceManifestSha256": manifest_hash,
        "selectedTransactionsSha256": _ordered_values_hash(
            [int(transaction_ids[index]) for fold in available_folds for index in selected_holdouts[fold]]
        ),
        "modelConfiguration": FinanceModelConfig().__dict__,
    }
    fingerprint = hashlib.sha256(json.dumps(execution_contract, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    total_units = len(seeds) * len(available_folds) * len(model_ids)
    checkpoint_store = FinanceOOFCheckpointStore(
        root=destination,
        fingerprint=fingerprint,
        run_id=run_id,
        expected_units=total_units,
        contract=execution_contract,
    )
    completed_units = 0
    resumed_units = 0
    metrics_rows: list[dict[str, Any]] = []
    oof_rows: list[dict[str, Any]] = []
    fold_protocols: list[dict[str, Any]] = []
    _notify(
        progress_callback,
        stage="training",
        event="prepared",
        completedUnits=0,
        resumedUnits=0,
        totalUnits=total_units,
        executionRunId=run_id,
        message="Secuencias, folds, escaladores y checkpoints financieros verificados.",
    )

    for seed in seeds:
        for fold in available_folds:
            fit_indices = fit_indices_by_fold[fold]
            holdout_indices = selected_holdouts[fold]
            inner_train_indices, early_validation_indices = _chronological_early_split(fit_indices, labels)
            scaler = fold_manifests[fold]["scaler"]
            scaler_sha256 = hashlib.sha256(
                json.dumps(scaler, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
            x_train = scale_finance_sequences(x[inner_train_indices], scaler)
            y_train = labels[inner_train_indices]
            x_early_validation = scale_finance_sequences(x[early_validation_indices], scaler)
            y_early_validation = labels[early_validation_indices]
            x_holdout = scale_finance_sequences(x[holdout_indices], scaler)
            y_holdout = labels[holdout_indices]
            if set(np.unique(y_holdout)) != {0, 1}:
                raise ValueError(f"El holdout financiero del fold {fold} no contiene ambas clases.")
            class_weight = _balanced_class_weights(y_train)
            fold_protocols.append(
                {
                    "seed": int(seed),
                    "fold": int(fold),
                    "fitRowsAvailable": int(fold_manifests[fold]["fitRows"]),
                    "fitRowsUsed": int(len(fit_indices)),
                    "innerTrainRows": int(len(inner_train_indices)),
                    "earlyValidationRows": int(len(early_validation_indices)),
                    "oofHoldoutRowsAvailable": int(fold_manifests[fold]["holdoutRows"]),
                    "oofHoldoutRowsUsed": int(len(holdout_indices)),
                    "fitEndAt": fold_manifests[fold]["fitEndAt"],
                    "holdoutStartAt": fold_manifests[fold]["holdoutStartAt"],
                    "futureRowsUsedForFit": False,
                    "scalerFitPolicy": scaler.get("fitPolicy"),
                    "scalerFitRows": int(scaler.get("fitRows", 0)),
                    "scalerSha256": scaler_sha256,
                }
            )
            predictions_by_model: dict[str, np.ndarray] = {}
            holdout_transaction_ids = transaction_ids[holdout_indices].astype(np.int64).tolist()
            for model_id in model_ids:
                restored = checkpoint_store.load_unit(
                    seed=seed,
                    fold=fold,
                    model_id=model_id,
                    transaction_ids=holdout_transaction_ids,
                    labels=y_holdout,
                    scaler_sha256=scaler_sha256,
                )
                if restored is not None:
                    predictions_by_model[model_id] = restored["probabilities"]
                    metrics_rows.append(restored["metrics"])
                    completed_units += 1
                    resumed_units += 1
                    _notify(
                        progress_callback,
                        stage="training",
                        event="model_resumed",
                        completedUnits=completed_units,
                        resumedUnits=resumed_units,
                        totalUnits=total_units,
                        seed=int(seed),
                        fold=int(fold),
                        modelId=model_id,
                        message=f"{model_id.upper()} recuperado del checkpoint financiero verificado.",
                    )
                    continue
                _notify(
                    progress_callback,
                    stage="training",
                    event="model_started",
                    completedUnits=completed_units,
                    resumedUnits=resumed_units,
                    totalUnits=total_units,
                    seed=int(seed),
                    fold=int(fold),
                    modelId=model_id,
                    message=f"Entrenando {model_id.upper()} — fold {fold + 1}/5 — semilla {seed}.",
                )
                tf.keras.backend.clear_session()
                classifier = classifier_factory(
                    model_id=model_id,
                    input_shape=(int(x.shape[1]), int(x.shape[2])),
                    epochs=epochs,
                    batch_size=batch_size,
                    patience=patience,
                    seed=seed,
                    config=FinanceModelConfig(),
                )
                training = classifier.fit(
                    x_train,
                    y_train,
                    x_early_validation,
                    y_early_validation,
                    class_weight=class_weight,
                )
                probabilities, inference_seconds = classifier.predict_proba(x_holdout)
                predictions_by_model[model_id] = probabilities
                model_path = checkpoint_store.model_path(seed, fold, model_id)
                model_hash, model_bytes = classifier.save(model_path)
                metrics_row = {
                    "seed": int(seed),
                    "fold": int(fold),
                    "modelId": model_id,
                    "fitRows": int(len(inner_train_indices)),
                    "earlyValidationRows": int(len(early_validation_indices)),
                    "holdoutRows": int(len(holdout_indices)),
                    "trainTimeSeconds": float(training["trainTimeSeconds"]),
                    "epochsCompleted": int(training["epochsCompleted"]),
                    "inferenceTimeMsPerSample": float(inference_seconds * 1000.0 / max(len(probabilities), 1)),
                    "modelPath": str(model_path.resolve()),
                    "modelSha256": model_hash,
                    "modelSizeBytes": int(model_bytes),
                    **finance_classification_metrics(y_holdout, probabilities, threshold=0.5),
                }
                checkpoint_store.save_unit(
                    seed=seed,
                    fold=fold,
                    model_id=model_id,
                    transaction_ids=holdout_transaction_ids,
                    labels=y_holdout,
                    probabilities=probabilities,
                    metrics=metrics_row,
                    scaler_sha256=scaler_sha256,
                    model_sha256=model_hash,
                    model_bytes=model_bytes,
                )
                metrics_rows.append(metrics_row)
                completed_units += 1
                _notify(
                    progress_callback,
                    stage="training",
                    event="model_completed",
                    completedUnits=completed_units,
                    resumedUnits=resumed_units,
                    totalUnits=total_units,
                    seed=int(seed),
                    fold=int(fold),
                    modelId=model_id,
                    message=f"{model_id.upper()} completado en fold {fold + 1}.",
                )
                del classifier
                tf.keras.backend.clear_session()

            for row_position, row_index in enumerate(holdout_indices):
                oof_rows.append(
                    {
                        "transaction_id": int(transaction_ids[row_index]),
                        "seed": int(seed),
                        "fold": int(fold),
                        "is_fraud": int(labels[row_index]),
                        **{
                            f"probability_{model_id}": float(predictions_by_model[model_id][row_position])
                            for model_id in model_ids
                        },
                    }
                )

    oof = pd.DataFrame(oof_rows).sort_values(["seed", "transaction_id"], kind="stable")
    expected_rows = sum(len(value) for value in selected_holdouts.values()) * len(seeds)
    if len(oof) != expected_rows or oof.duplicated(["seed", "transaction_id"]).any():
        raise ValueError("Las probabilidades OOF financieras no cubren exactamente el piloto seleccionado.")
    probability_columns = [f"probability_{model_id}" for model_id in model_ids]
    if oof[probability_columns].isna().any().any():
        raise ValueError("Existen probabilidades OOF financieras faltantes.")
    oof_path = destination / "oof_base_probabilities.csv"
    oof.to_csv(oof_path, index=False)
    oof_hash, oof_bytes = sha256_file(oof_path)
    metrics_path = destination / "base_fold_metrics.json"
    write_json(metrics_path, {"items": metrics_rows})
    metrics_hash, metrics_bytes = sha256_file(metrics_path)
    aggregate = _aggregate_metrics(metrics_rows)
    full_protocol = (
        protocol == "thesis"
        and len(seeds) >= 5
        and set(model_ids) == set(FINANCE_MODEL_IDS)
        and demo_max_rows_per_fold is None
    )
    manifest = {
        "schemaVersion": "1.0.0",
        "runId": run_id,
        "domain": "finanzas",
        "status": "thesis_base_models_candidate" if full_protocol else "demo",
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "protocol": protocol,
        "configuration": {
            "epochs": epochs,
            "batchSize": batch_size,
            "patience": patience,
            "seeds": list(seeds),
            "modelIds": list(model_ids),
            "demoMaxRowsPerFold": demo_max_rows_per_fold,
            "primaryMetric": "prAuc",
            "threshold": 0.5,
            "thresholdPolicy": "Fixed exploratory threshold; final threshold must be selected on external validation.",
        },
        "execution": {
            "fingerprint": fingerprint,
            "checkpointSchemaVersion": "1.0.0",
            "checkpointPath": str(checkpoint_store.manifest_path.resolve()),
            "resumable": True,
            "resumeCount": checkpoint_store.resume_count,
            "resumedUnits": resumed_units,
            "trainedUnitsThisInvocation": total_units - resumed_units,
            "invalidCheckpointsRetrained": checkpoint_store.invalid_units,
        },
        "dataset": {
            "datasetId": sequence_manifest.get("datasetId"),
            "sequencePath": str(sequence_source.resolve()),
            "sequenceSha256": sequence_hash,
            "sequenceBytes": sequence_bytes,
            "assignmentsSha256": assignment_hash,
            "assignmentsBytes": assignment_bytes,
            "sequenceManifestSha256": manifest_hash,
            "sequenceManifestBytes": manifest_bytes,
            "oofRowsAvailablePerSeed": int(len(assignments)),
            "oofRowsUsedPerSeed": int(sum(len(value) for value in selected_holdouts.values())),
            "demoSubset": protocol == "demo" and demo_max_rows_per_fold is not None,
        },
        "validation": {
            "strategy": "expanding_temporal_oof_plus_chronological_inner_early_stopping",
            "oofFolds": available_folds,
            "foldProtocols": fold_protocols,
            "futureLeakagePassed": True,
            "externalValidationUsed": False,
            "externalValidationReserved": True,
            "testSetLocked": True,
            "testSetEncoded": False,
            "testSetUsed": False,
        },
        "baseModels": list(model_ids),
        "baseFoldMetrics": metrics_rows,
        "aggregate": aggregate,
        "artifacts": {
            "oofProbabilities": {"path": str(oof_path.resolve()), "sha256": oof_hash, "bytes": oof_bytes, "rows": int(len(oof))},
            "foldMetrics": {"path": str(metrics_path.resolve()), "sha256": metrics_hash, "bytes": metrics_bytes, "rows": int(len(metrics_rows))},
        },
        "stacking": {
            "ready": len(model_ids) >= 2 and not oof[probability_columns].isna().any().any(),
            "featureColumns": probability_columns,
            "trained": False,
        },
    }
    final_manifest_path = destination / "oof_manifest.json"
    write_json(final_manifest_path, manifest)
    final_manifest_hash, _ = sha256_file(final_manifest_path)
    checkpoint_store.mark_completed(final_manifest_path=final_manifest_path, final_manifest_sha256=final_manifest_hash)
    _notify(
        progress_callback,
        stage="completed",
        event="completed",
        completedUnits=total_units,
        resumedUnits=resumed_units,
        totalUnits=total_units,
        executionRunId=run_id,
        message="Modelos base financieros completados; probabilidades OOF listas para Stacking.",
    )
    return manifest


def _chronological_early_split(indices: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ordered = np.asarray(indices, dtype=np.int64)
    if len(ordered) < 20:
        raise ValueError("El prefijo financiero es insuficiente para early stopping cronológico.")
    for validation_fraction in (0.15, 0.20, 0.25, 0.30):
        cut = max(1, min(len(ordered) - 1, int(len(ordered) * (1.0 - validation_fraction))))
        inner_train = ordered[:cut]
        validation = ordered[cut:]
        if set(np.unique(labels[inner_train])) == {0, 1} and set(np.unique(labels[validation])) == {0, 1}:
            return inner_train, validation
    raise ValueError("No fue posible crear early stopping cronológico con ambas clases.")


def _deterministic_stratified_indices(
    indices: np.ndarray,
    labels: np.ndarray,
    transaction_ids: np.ndarray,
    *,
    maximum_rows: int,
    salt: str,
) -> np.ndarray:
    values = np.asarray(indices, dtype=np.int64)
    if len(values) <= maximum_rows:
        return values
    positive = values[labels[values] == 1]
    negative = values[labels[values] == 0]
    positive_quota = max(1, min(len(positive), int(round(maximum_rows * len(positive) / len(values)))))
    negative_quota = maximum_rows - positive_quota
    if negative_quota > len(negative):
        positive_quota += negative_quota - len(negative)
        negative_quota = len(negative)

    def select(bucket: np.ndarray, count: int) -> np.ndarray:
        ordered = sorted(
            bucket.tolist(),
            key=lambda index: hashlib.sha256(f"{salt}:{int(transaction_ids[index])}".encode("utf-8")).hexdigest(),
        )
        return np.asarray(ordered[:count], dtype=np.int64)

    selected = np.concatenate([select(negative, negative_quota), select(positive, positive_quota)])
    return selected[np.argsort(transaction_ids[selected], kind="stable")]


def _balanced_class_weights(labels: np.ndarray) -> dict[int, float]:
    values, counts = np.unique(labels, return_counts=True)
    if set(values.tolist()) != {0, 1}:
        raise ValueError("El ajuste financiero requiere ambas clases.")
    total = len(labels)
    return {int(value): float(total / (len(values) * count)) for value, count in zip(values, counts)}


def _aggregate_metrics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    frame = pd.DataFrame(rows)
    higher_is_better = ["prAuc", "rocAuc", "f1", "precision", "recall", "mcc", "balancedAccuracy"]
    lower_is_better = ["brierScore", "logLoss", "falsePositiveRate"]
    result: list[dict[str, Any]] = []
    for model_id, group in frame.groupby("modelId", sort=False):
        item: dict[str, Any] = {
            "modelId": str(model_id),
            "foldEvaluations": int(len(group)),
            "trainTimeSecondsMean": float(group["trainTimeSeconds"].mean()),
            "inferenceTimeMsPerSampleMean": float(group["inferenceTimeMsPerSample"].mean()),
            "modelSizeBytesMean": float(group["modelSizeBytes"].mean()),
        }
        for metric in [*higher_is_better, *lower_is_better]:
            item[f"{metric}Mean"] = float(group[metric].mean())
            item[f"{metric}Std"] = float(group[metric].std(ddof=1)) if len(group) > 1 else 0.0
        result.append(item)
    return sorted(result, key=lambda item: item["prAucMean"], reverse=True)


def _ordered_values_hash(values: list[int]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _notify(callback: Callable[[dict[str, Any]], None] | None, **event: Any) -> None:
    if callback is not None:
        callback(event)
