from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
import tensorflow as tf

from app.config import EXPERIMENTS_DIR, SILVER_DIR
from app.phishing.checkpoints import PhishingOOFCheckpointStore
from app.phishing.data import CharacterTokenizer
from app.phishing.ingestion import sha256_file
from app.phishing.metrics import phishing_classification_metrics
from app.phishing.models import KerasPhishingClassifier, PHISHING_MODEL_IDS, PhishingModelConfig, hash_keras_artifact
from app.phishing.runtime import inspect_phishing_training_runtime
from app.utils import read_json, write_json


ClassifierFactory = Callable[..., Any]


def run_phishing_oof_experiment(
    *,
    output_dir: Path | None = None,
    silver_path: Path | None = None,
    assignments_path: Path | None = None,
    protocol: str = "demo",
    epochs: int = 1,
    batch_size: int = 64,
    patience: int = 2,
    seeds: tuple[int, ...] = (42,),
    model_ids: tuple[str, ...] = PHISHING_MODEL_IDS,
    demo_max_rows: int | None = 2_000,
    max_vocabulary: int = 256,
    length_percentile: float = 99.0,
    max_length_cap: int = 512,
    execution_run_id: str | None = None,
    classifier_factory: ClassifierFactory = KerasPhishingClassifier,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    if protocol not in {"demo", "thesis"}:
        raise ValueError("El protocolo debe ser demo o thesis.")
    unknown = sorted(set(model_ids) - set(PHISHING_MODEL_IDS))
    if unknown:
        raise ValueError(f"Modelos de phishing no soportados: {', '.join(unknown)}")
    if not model_ids or not seeds:
        raise ValueError("Se requiere al menos un modelo y una semilla.")
    if protocol == "thesis" and (set(model_ids) != set(PHISHING_MODEL_IDS) or len(seeds) < 5 or demo_max_rows is not None):
        raise ValueError("El protocolo thesis exige los cinco modelos, cinco semillas y todas las filas.")

    source = silver_path or (SILVER_DIR / "phishing.csv")
    assignments_source = assignments_path or (SILVER_DIR / "phishing_oof_assignments.csv")
    if not source.exists() or not assignments_source.exists():
        raise FileNotFoundError("Primero prepare el silver y las asignaciones OOF de phishing.")
    source_hash, source_bytes = sha256_file(source)
    if protocol == "thesis":
        metadata_path = SILVER_DIR / "phishing.metadata.json"
        sequence_manifest_path = SILVER_DIR / "phishing_sequence_manifest.json"
        if not metadata_path.exists() or not sequence_manifest_path.exists():
            raise ValueError("El protocolo thesis exige metadatos científicos y secuencias versionadas.")
        dataset_metadata = read_json(metadata_path)
        sequence_manifest = read_json(sequence_manifest_path)
        if not dataset_metadata.get("readyForThesisTraining"):
            raise ValueError("El dataset no ha superado la puerta científica de phishing.")
        if dataset_metadata.get("silver", {}).get("sha256") != source_hash:
            raise ValueError("El silver cambió respecto de sus metadatos científicos.")
        if sequence_manifest.get("datasetId") != dataset_metadata.get("datasetId"):
            raise ValueError("Las asignaciones OOF pertenecen a una versión anterior del dataset.")
    assignments_hash, _ = sha256_file(assignments_source)
    if protocol == "thesis" and sequence_manifest.get("artifacts", {}).get("assignments", {}).get("sha256") != assignments_hash:
        raise ValueError("Las asignaciones OOF no coinciden con su manifiesto versionado.")
    frame = pd.read_csv(source)
    train = frame[frame["split"] == "train"].copy()
    train["sample_id"] = train["canonical_url"].map(_sample_id)
    assignments = pd.read_csv(assignments_source, dtype={"sample_id": str, "oof_fold": int})
    train = train.merge(assignments, on="sample_id", how="inner", validate="one_to_one")
    if len(train) != len(assignments):
        raise ValueError("Las asignaciones OOF no cubren exactamente el conjunto externo de entrenamiento.")
    if protocol == "demo" and demo_max_rows is not None and len(train) > demo_max_rows:
        train = _deterministic_demo_sample(train, demo_max_rows)
    train = train.sort_values(["oof_fold", "sample_id"], kind="stable").reset_index(drop=True)
    available_folds = sorted(int(value) for value in train["oof_fold"].unique())
    if len(available_folds) < 3:
        raise ValueError("El experimento OOF requiere al menos tres folds con datos.")

    run_id = execution_run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_phishing_oof")
    destination = output_dir or (EXPERIMENTS_DIR / run_id / "phishing_oof_v1")
    destination.mkdir(parents=True, exist_ok=True)
    tokenizers_root = destination / "tokenizers"
    total_units = len(seeds) * len(available_folds) * len(model_ids)
    execution_contract = {
        "protocol": protocol,
        "epochs": int(epochs),
        "batchSize": int(batch_size),
        "patience": int(patience),
        "seeds": [int(seed) for seed in seeds],
        "modelIds": list(model_ids),
        "demoMaxRows": demo_max_rows,
        "maxVocabulary": int(max_vocabulary),
        "lengthPercentile": float(length_percentile),
        "maxLengthCap": int(max_length_cap),
        "sourceSha256": source_hash,
        "assignmentsSha256": assignments_hash,
        "sampleIdsSha256": _ordered_values_hash(train["sample_id"]),
        "folds": available_folds,
        "modelConfiguration": PhishingModelConfig().__dict__,
    }
    fingerprint = hashlib.sha256(
        json.dumps(execution_contract, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    preflight = inspect_phishing_training_runtime(
        protocol=protocol,
        output_dir=destination,
        silver_path=source,
        assignments_path=assignments_source,
    )
    if not preflight["ready"]:
        failed = ", ".join(item["id"] for item in preflight["checks"] if not item["passed"])
        raise ValueError(f"El preflight de entrenamiento no fue superado: {failed}.")
    checkpoint_store = PhishingOOFCheckpointStore(
        root=destination,
        fingerprint=fingerprint,
        run_id=run_id,
        expected_units=total_units,
        contract=execution_contract,
        preflight=preflight,
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
        preflight=preflight,
        message="Datos, recursos y checkpoints verificados; iniciando modelos base.",
    )

    for seed in seeds:
        for fold in available_folds:
            outer_holdout = train[train["oof_fold"] == fold].copy().reset_index(drop=True)
            outer_fit = train[train["oof_fold"] != fold].copy().reset_index(drop=True)
            inner_train, early_validation = _inner_group_split(outer_fit, seed=seed + fold)
            tokenizer = CharacterTokenizer.fit(
                inner_train["canonical_url"],
                max_vocabulary=max_vocabulary,
                length_percentile=length_percentile,
                max_length_cap=max_length_cap,
            )
            tokenizer_path = tokenizers_root / f"seed_{seed}" / f"fold_{fold}.json"
            write_json(tokenizer_path, {
                **tokenizer.to_dict(),
                "fitPolicy": "inner_training_only",
                "seed": seed,
                "outerFold": fold,
                "fitRows": int(len(inner_train)),
                "earlyValidationRows": int(len(early_validation)),
                "oofHoldoutRows": int(len(outer_holdout)),
            })
            tokenizer_hash, tokenizer_bytes = sha256_file(tokenizer_path)
            x_train = tokenizer.encode(inner_train["canonical_url"])
            y_train = inner_train["is_phishing"].to_numpy(dtype=np.int32)
            x_early_validation = tokenizer.encode(early_validation["canonical_url"])
            y_early_validation = early_validation["is_phishing"].to_numpy(dtype=np.int32)
            x_holdout = tokenizer.encode(outer_holdout["canonical_url"])
            y_holdout = outer_holdout["is_phishing"].to_numpy(dtype=np.int32)
            class_weight = _balanced_class_weights(y_train)
            group_overlap = set(inner_train["registrable_domain"]) & set(early_validation["registrable_domain"])
            oof_overlap = set(outer_fit["registrable_domain"]) & set(outer_holdout["registrable_domain"])
            if group_overlap or oof_overlap:
                raise ValueError("Se detectó fuga de dominios durante la preparación de entrenamiento.")
            fold_protocols.append({
                "seed": int(seed),
                "fold": int(fold),
                "innerTrainRows": int(len(inner_train)),
                "earlyValidationRows": int(len(early_validation)),
                "oofHoldoutRows": int(len(outer_holdout)),
                "innerGroupOverlap": 0,
                "oofGroupOverlap": 0,
                "tokenizer": {
                    "path": str(tokenizer_path.resolve()),
                    "sha256": tokenizer_hash,
                    "bytes": tokenizer_bytes,
                    "vocabularySize": len(tokenizer.token_to_id),
                    "maxLength": tokenizer.max_length,
                },
            })

            predictions_by_model: dict[str, np.ndarray] = {}
            for model_id in model_ids:
                restored = checkpoint_store.load_unit(
                    seed=seed,
                    fold=fold,
                    model_id=model_id,
                    sample_ids=outer_holdout["sample_id"].astype(str).tolist(),
                    labels=y_holdout,
                    tokenizer_sha256=tokenizer_hash,
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
                        executionRunId=run_id,
                        message=f"{model_id.upper()} recuperado del checkpoint verificado del fold {fold + 1}.",
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
                    message=f"Entrenando {model_id.upper()} · fold {fold + 1}/{len(available_folds)} · semilla {seed}.",
                )
                tf.keras.backend.clear_session()
                classifier = classifier_factory(
                    model_id=model_id,
                    vocabulary_size=len(tokenizer.token_to_id),
                    sequence_length=tokenizer.max_length,
                    epochs=epochs,
                    batch_size=batch_size,
                    patience=patience,
                    seed=seed,
                    config=PhishingModelConfig(),
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
                classifier.save(model_path)
                # Enforce one hashing contract independently of the classifier
                # implementation (real Keras model or deterministic test double).
                model_hash, model_bytes = hash_keras_artifact(model_path)
                metrics_row = {
                    "seed": int(seed),
                    "fold": int(fold),
                    "modelId": model_id,
                    "fitRows": int(len(inner_train)),
                    "earlyValidationRows": int(len(early_validation)),
                    "holdoutRows": int(len(outer_holdout)),
                    "trainTimeSeconds": float(training["trainTimeSeconds"]),
                    "epochsCompleted": int(training["epochsCompleted"]),
                    "inferenceTimeMsPerSample": float(inference_seconds * 1000.0 / max(len(probabilities), 1)),
                    "modelPath": str(model_path.resolve()),
                    "modelSha256": model_hash,
                    "modelSizeBytes": int(model_bytes),
                    **phishing_classification_metrics(y_holdout, probabilities, threshold=0.5),
                }
                checkpoint_store.save_unit(
                    seed=seed,
                    fold=fold,
                    model_id=model_id,
                    sample_ids=outer_holdout["sample_id"].astype(str).tolist(),
                    labels=y_holdout,
                    probabilities=probabilities,
                    metrics=metrics_row,
                    tokenizer_sha256=tokenizer_hash,
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

            for row_index, row in outer_holdout.iterrows():
                oof_rows.append({
                    "sample_id": row["sample_id"],
                    "seed": int(seed),
                    "fold": int(fold),
                    "is_phishing": int(row["is_phishing"]),
                    **{f"probability_{model_id}": float(predictions_by_model[model_id][row_index]) for model_id in model_ids},
                })

    oof = pd.DataFrame(oof_rows).sort_values(["seed", "sample_id"], kind="stable")
    expected_oof_rows = len(train) * len(seeds)
    if len(oof) != expected_oof_rows or oof.duplicated(["seed", "sample_id"]).any():
        raise ValueError("Las probabilidades OOF no cubren exactamente cada muestra y semilla.")
    probability_columns = [f"probability_{model_id}" for model_id in model_ids]
    if oof[probability_columns].isna().any().any():
        raise ValueError("Existen probabilidades OOF faltantes.")
    oof_path = destination / "oof_base_probabilities.csv"
    oof.to_csv(oof_path, index=False)
    oof_hash, oof_bytes = sha256_file(oof_path)

    metrics_path = destination / "base_fold_metrics.json"
    write_json(metrics_path, {"items": metrics_rows})
    metrics_hash, metrics_bytes = sha256_file(metrics_path)
    aggregate = _aggregate_metrics(metrics_rows)
    full_protocol = protocol == "thesis" and len(seeds) >= 5 and set(model_ids) == set(PHISHING_MODEL_IDS) and len(train) == len(assignments)
    manifest = {
        "schemaVersion": "1.0.0",
        "runId": run_id,
        "domain": "phishing",
        "status": "thesis_base_models_candidate" if full_protocol else "demo",
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "protocol": protocol,
        "configuration": {
            "epochs": epochs,
            "batchSize": batch_size,
            "patience": patience,
            "seeds": list(seeds),
            "modelIds": list(model_ids),
            "demoMaxRows": demo_max_rows,
            "threshold": 0.5,
            "thresholdPolicy": "Fixed exploratory threshold only; final threshold must be selected on outer validation.",
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
            "preflight": preflight,
        },
        "dataset": {
            "sourcePath": str(source.resolve()),
            "sourceSha256": source_hash,
            "sourceBytes": source_bytes,
            "assignmentsSha256": assignments_hash,
            "outerTrainRowsAvailable": int(len(assignments)),
            "rowsUsed": int(len(train)),
            "demoSubset": int(len(train)) < int(len(assignments)),
        },
        "validation": {
            "strategy": "outer_group_holdout_plus_inner_group_early_stopping",
            "oofFolds": available_folds,
            "foldProtocols": fold_protocols,
            "oofCoverageRowsPerSeed": int(len(train)),
            "groupLeakagePassed": True,
            "testSetLocked": True,
            "testSetUsed": False,
            "outerValidationUsed": False,
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
    manifest_path = destination / "oof_manifest.json"
    write_json(manifest_path, manifest)
    manifest_hash, _ = sha256_file(manifest_path)
    checkpoint_store.mark_completed(final_manifest_path=manifest_path, final_manifest_sha256=manifest_hash)
    _notify(
        progress_callback,
        stage="completed",
        event="completed",
        completedUnits=total_units,
        resumedUnits=resumed_units,
        totalUnits=total_units,
        executionRunId=run_id,
        message="Modelos base OOF completados; probabilidades listas para Stacking.",
    )
    return manifest


def _inner_group_split(frame: pd.DataFrame, *, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    train_indices, validation_indices = next(splitter.split(
        frame["canonical_url"],
        frame["is_phishing"],
        frame["registrable_domain"],
    ))
    return frame.iloc[train_indices].reset_index(drop=True), frame.iloc[validation_indices].reset_index(drop=True)


def _deterministic_demo_sample(frame: pd.DataFrame, maximum_rows: int) -> pd.DataFrame:
    if maximum_rows < 500:
        raise ValueError("demo_max_rows debe ser al menos 500.")
    buckets = frame.groupby(["oof_fold", "is_phishing"], sort=True)
    quota = max(1, maximum_rows // max(len(buckets), 1))
    selected = []
    for _, bucket in buckets:
        ordered = bucket.assign(_order=bucket["sample_id"].map(
            lambda value: hashlib.sha256(f"demo:{value}".encode("utf-8")).hexdigest()
        )).sort_values("_order", kind="stable")
        selected.append(ordered.head(quota).drop(columns=["_order"]))
    result = pd.concat(selected, ignore_index=True)
    if len(result) < min(maximum_rows, len(frame)):
        remaining = frame[~frame["sample_id"].isin(result["sample_id"])].copy()
        remaining["_order"] = remaining["sample_id"].map(
            lambda value: hashlib.sha256(f"remainder:{value}".encode("utf-8")).hexdigest()
        )
        result = pd.concat([result, remaining.sort_values("_order").head(maximum_rows - len(result)).drop(columns=["_order"])])
    return result.head(maximum_rows).copy()


def _balanced_class_weights(labels: np.ndarray) -> dict[int, float]:
    values, counts = np.unique(labels, return_counts=True)
    total = len(labels)
    return {int(value): float(total / (len(values) * count)) for value, count in zip(values, counts)}


def _aggregate_metrics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    frame = pd.DataFrame(rows)
    metrics = ["prAuc", "rocAuc", "f1", "precision", "recall", "mcc", "balancedAccuracy", "falsePositiveRate"]
    result = []
    for model_id, group in frame.groupby("modelId", sort=False):
        item: dict[str, Any] = {
            "modelId": str(model_id),
            "foldEvaluations": int(len(group)),
            "trainTimeSecondsMean": float(group["trainTimeSeconds"].mean()),
            "inferenceTimeMsPerSampleMean": float(group["inferenceTimeMsPerSample"].mean()),
            "modelSizeBytesMean": float(group["modelSizeBytes"].mean()),
        }
        for metric in metrics:
            item[f"{metric}Mean"] = float(group[metric].mean())
            item[f"{metric}Std"] = float(group[metric].std(ddof=1)) if len(group) > 1 else 0.0
        result.append(item)
    return sorted(result, key=lambda item: item["prAucMean"], reverse=True)


def _sample_id(url: str) -> str:
    return hashlib.sha256(str(url).encode("utf-8")).hexdigest()


def _ordered_values_hash(values: pd.Series) -> str:
    digest = hashlib.sha256()
    for value in values.astype(str):
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _notify(callback: Callable[[dict[str, Any]], None] | None, **event: Any) -> None:
    if callback is not None:
        callback(event)
