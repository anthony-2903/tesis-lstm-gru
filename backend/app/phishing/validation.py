from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from app.config import EXPERIMENTS_DIR, SILVER_DIR
from app.phishing.data import CharacterTokenizer
from app.phishing.ingestion import sha256_file
from app.phishing.metrics import phishing_classification_metrics
from app.phishing.models import KerasPhishingClassifier, PhishingModelConfig
from app.phishing.stacking import (
    BASELINE_IDS,
    ENSEMBLE_IDS,
    build_meta_features,
    build_meta_models,
    fit_nonnegative_weights,
)
from app.utils import read_json, write_json


ClassifierFactory = Callable[..., Any]
METRIC_NAMES = (
    "prAuc",
    "rocAuc",
    "f1",
    "precision",
    "recall",
    "mcc",
    "balancedAccuracy",
    "falsePositiveRate",
)


def run_phishing_external_validation(
    *,
    base_manifest_path: Path | None = None,
    silver_path: Path | None = None,
    sequence_manifest_path: Path | None = None,
    output_dir: Path | None = None,
    classifier_factory: ClassifierFactory = KerasPhishingClassifier,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    base_path = base_manifest_path or _latest_base_manifest()
    if base_path is None or not base_path.is_file():
        raise FileNotFoundError("No existe una corrida OOF base para validar.")
    base_manifest = read_json(base_path)
    oof_artifact = base_manifest.get("artifacts", {}).get("oofProbabilities", {})
    oof_path = Path(oof_artifact.get("path", ""))
    _verify_file(oof_path, oof_artifact, "probabilidades OOF")
    oof = pd.read_csv(oof_path)

    model_ids = tuple(base_manifest.get("baseModels", []))
    seeds = tuple(int(value) for value in base_manifest.get("configuration", {}).get("seeds", []))
    probability_columns = [f"probability_{model_id}" for model_id in model_ids]
    required_oof = {"sample_id", "seed", "fold", "is_phishing", *probability_columns}
    missing_oof = required_oof.difference(oof.columns)
    if missing_oof or len(model_ids) < 2 or not seeds:
        raise ValueError("La corrida base no contiene un contrato OOF completo para validación externa.")
    if oof[probability_columns].isna().any().any() or oof.duplicated(["seed", "sample_id"]).any():
        raise ValueError("Las probabilidades OOF están incompletas o duplicadas.")

    development_ids = _common_development_ids(oof, seeds)
    source = silver_path or Path(base_manifest.get("dataset", {}).get("sourcePath", SILVER_DIR / "phishing.csv"))
    source_hash, source_bytes = sha256_file(source)
    expected_source_hash = base_manifest.get("dataset", {}).get("sourceSha256")
    if expected_source_hash and source_hash != expected_source_hash:
        raise ValueError("El dataset silver cambió después de la corrida OOF.")
    development, validation = _load_development_and_validation(source, development_ids)
    _validate_external_partitions(development, validation)

    sequence_path = sequence_manifest_path or (SILVER_DIR / "phishing_sequence_manifest.json")
    sequence_manifest = read_json(sequence_path) if sequence_path.is_file() else {}
    sequence_configuration = sequence_manifest.get("configuration", {})
    tokenizer = CharacterTokenizer.fit(
        development["canonical_url"],
        max_vocabulary=int(sequence_configuration.get("maxVocabulary", 256)),
        length_percentile=float(sequence_configuration.get("lengthPercentile", 99.0)),
        max_length_cap=int(sequence_configuration.get("maxLengthCap", 512)),
    )

    destination = output_dir or (base_path.parent / "external_validation_v1")
    destination.mkdir(parents=True, exist_ok=True)
    tokenizer_path = destination / "tokenizer" / "development.json"
    write_json(tokenizer_path, {
        **tokenizer.to_dict(),
        "fitPolicy": "development_rows_from_base_oof_run_only",
        "fitRows": int(len(development)),
        "forUseOn": ["validation"],
        "testPolicy": "locked_not_encoded_not_evaluated",
    })
    tokenizer_hash, tokenizer_bytes = sha256_file(tokenizer_path)

    x_development = tokenizer.encode(development["canonical_url"])
    y_development = development["is_phishing"].to_numpy(dtype=np.int32)
    x_validation = tokenizer.encode(validation["canonical_url"])
    y_validation = validation["is_phishing"].to_numpy(dtype=np.int32)
    class_weight = _balanced_class_weights(y_development)
    base_predictions_by_seed: dict[int, dict[str, np.ndarray]] = {}
    fitted_artifacts: list[dict[str, Any]] = []
    training_records: list[dict[str, Any]] = []
    total_units = len(seeds) * (len(model_ids) + 1)
    completed_units = 0
    _notify(progress_callback, stage="refit", event="prepared", completedUnits=0, totalUnits=total_units, message="Partición validation verificada; test continúa bloqueado.")

    for seed in seeds:
        base_predictions_by_seed[seed] = {}
        for model_id in model_ids:
            epochs = _selected_epoch_count(base_manifest, seed=seed, model_id=model_id)
            _notify(progress_callback, stage="refit", event="model_started", completedUnits=completed_units, totalUnits=total_units, seed=seed, modelId=model_id, message=f"Reentrenando {model_id.upper()} con desarrollo completo; semilla {seed}.")
            tf.keras.backend.clear_session()
            classifier = classifier_factory(
                model_id=model_id,
                vocabulary_size=len(tokenizer.token_to_id),
                sequence_length=tokenizer.max_length,
                epochs=epochs,
                batch_size=int(base_manifest.get("configuration", {}).get("batchSize", 64)),
                patience=-1,
                seed=seed,
                config=PhishingModelConfig(),
            )
            training = classifier.fit_full(x_development, y_development, class_weight=class_weight)
            probabilities, inference_seconds = classifier.predict_proba(x_validation)
            base_predictions_by_seed[seed][model_id] = probabilities
            model_path = destination / "base_models" / f"seed_{seed}" / f"{model_id}.keras"
            model_hash, model_bytes = classifier.save(model_path)
            fitted_artifacts.append({
                "kind": "base_model",
                "candidateId": model_id,
                "seed": seed,
                "path": str(model_path.resolve()),
                "sha256": model_hash,
                "bytes": int(model_bytes),
            })
            training_records.append({
                "modelId": model_id,
                "seed": seed,
                "selectedEpochsFromOof": epochs,
                "epochsCompleted": int(training["epochsCompleted"]),
                "fitRows": int(len(development)),
                "validationRowsPredicted": int(len(validation)),
                "trainTimeSeconds": float(training["trainTimeSeconds"]),
                "inferenceTimeMsPerSample": float(inference_seconds * 1000.0 / max(len(validation), 1)),
                "modelSizeBytes": int(model_bytes),
            })
            completed_units += 1
            _notify(progress_callback, stage="refit", event="model_completed", completedUnits=completed_units, totalUnits=total_units, seed=seed, modelId=model_id, message=f"{model_id.upper()} produjo probabilidades externas.")
            del classifier
            tf.keras.backend.clear_session()

    all_candidate_ids = (*model_ids, *ENSEMBLE_IDS)
    probability_output_columns = [f"probability_{candidate_id}" for candidate_id in all_candidate_ids]
    validation_prediction_rows: list[dict[str, Any]] = []
    for seed in seeds:
        seed_oof = oof[oof["seed"] == seed].sort_values("sample_id", kind="stable")
        x_oof_base = seed_oof[probability_columns].to_numpy(dtype=np.float64)
        y_oof = seed_oof["is_phishing"].to_numpy(dtype=np.int32)
        x_oof_meta, feature_names = build_meta_features(x_oof_base, probability_columns)
        x_validation_base = np.column_stack([base_predictions_by_seed[seed][model_id] for model_id in model_ids])
        x_validation_meta, _ = build_meta_features(x_validation_base, probability_columns)
        candidate_predictions: dict[str, np.ndarray] = {
            **base_predictions_by_seed[seed],
            "mean": x_validation_base.mean(axis=1),
            "voting": (x_validation_base >= 0.5).mean(axis=1),
        }
        weights = fit_nonnegative_weights(x_oof_base, y_oof)
        candidate_predictions["weighted_mean"] = x_validation_base @ weights
        weights_path = destination / "meta_models" / f"seed_{seed}" / "weighted_mean.json"
        write_json(weights_path, {
            "schemaVersion": "1.0.0",
            "fitSplit": "development_oof_only",
            "objective": "binary_log_loss",
            "constraints": "nonnegative_weights_sum_to_one",
            "seed": seed,
            "fitRows": int(len(seed_oof)),
            "weights": {model_id: float(weight) for model_id, weight in zip(model_ids, weights)},
        })
        weight_hash, weight_bytes = sha256_file(weights_path)
        fitted_artifacts.append({"kind": "ensemble_weights", "candidateId": "weighted_mean", "seed": seed, "path": str(weights_path.resolve()), "sha256": weight_hash, "bytes": weight_bytes})

        for candidate_id, model in build_meta_models(seed=seed).items():
            model.fit(x_oof_meta, y_oof)
            candidate_predictions[candidate_id] = model.predict_proba(x_validation_meta)[:, 1]
            model_path = destination / "meta_models" / f"seed_{seed}" / f"{candidate_id}.joblib"
            model_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(model, model_path)
            model_hash, model_bytes = sha256_file(model_path)
            fitted_artifacts.append({"kind": "meta_model", "candidateId": candidate_id, "seed": seed, "path": str(model_path.resolve()), "sha256": model_hash, "bytes": model_bytes})

        for row_index, row in validation.reset_index(drop=True).iterrows():
            validation_prediction_rows.append({
                "sample_id": row["sample_id"],
                "seed": seed,
                "is_phishing": int(row["is_phishing"]),
                **{f"probability_{candidate_id}": float(candidate_predictions[candidate_id][row_index]) for candidate_id in all_candidate_ids},
            })
        completed_units += 1
        _notify(progress_callback, stage="meta_models", event="seed_completed", completedUnits=completed_units, totalUnits=total_units, seed=seed, message=f"Ensambles ajustados solo con OOF; semilla {seed}.")

    predictions = pd.DataFrame(validation_prediction_rows).sort_values(["seed", "sample_id"], kind="stable")
    expected_rows = len(validation) * len(seeds)
    if len(predictions) != expected_rows or predictions.duplicated(["seed", "sample_id"]).any():
        raise ValueError("La validación externa no cubre exactamente cada URL y semilla.")
    if predictions[probability_output_columns].isna().any().any():
        raise ValueError("Existen probabilidades externas faltantes.")
    mean_predictions = predictions.groupby(["sample_id", "is_phishing"], as_index=False)[probability_output_columns].mean()
    if len(mean_predictions) != len(validation):
        raise ValueError("El promedio entre semillas no conserva la cobertura de validation.")

    comparison = _build_validation_comparison(predictions, mean_predictions, all_candidate_ids, model_ids)
    winner = comparison[0]
    leading_stacking = next((row for row in comparison if row["family"] == "stacking"), None)
    predictions_path = destination / "validation_probabilities_by_seed.csv"
    mean_predictions_path = destination / "validation_probabilities_mean.csv"
    metrics_path = destination / "validation_selection_metrics.json"
    predictions.to_csv(predictions_path, index=False)
    mean_predictions.to_csv(mean_predictions_path, index=False)
    write_json(metrics_path, {"items": comparison})
    prediction_hash, prediction_bytes = sha256_file(predictions_path)
    mean_hash, mean_bytes = sha256_file(mean_predictions_path)
    metrics_hash, metrics_bytes = sha256_file(metrics_path)

    manifest = {
        "schemaVersion": "1.0.0",
        "runId": f"{base_manifest.get('runId')}_external_validation",
        "domain": "phishing",
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "status": "demo_validation_selection" if base_manifest.get("status") == "demo" else "validation_selection_candidate",
        "baseRun": {"runId": base_manifest.get("runId"), "manifestPath": str(base_path.resolve()), "oofPath": str(oof_path.resolve()), "models": list(model_ids), "seeds": list(seeds)},
        "dataset": {
            "sourcePath": str(source.resolve()),
            "sourceSha256": source_hash,
            "sourceBytes": source_bytes,
            "developmentRows": int(len(development)),
            "validationRows": int(len(validation)),
            "demoSubset": bool(base_manifest.get("dataset", {}).get("demoSubset", False)),
            "developmentValidationDomainOverlap": 0,
        },
        "training": {"epochSelection": "median_epochs_completed_across_oof_folds_per_seed_and_model", "records": training_records},
        "metaFeatures": {"columns": feature_names, "fitSplit": "development_oof_only", "validationLabelsUsedForMetaFit": False},
        "selection": {
            "candidatePrimaryMetric": "prAuc",
            "thresholdObjective": "mcc",
            "winnerCandidateId": winner["candidateId"],
            "winnerThreshold": winner["calibratedThreshold"],
            "leadingStackingCandidateId": leading_stacking["candidateId"] if leading_stacking else None,
            "status": "configuration_and_threshold_selected_on_validation_pending_test",
        },
        "comparison": comparison,
        "validation": {
            "strategy": "base_refit_on_development_meta_fit_on_oof_select_on_external_validation",
            "outerValidationUsed": True,
            "validationRows": int(len(validation)),
            "sameRowsForAllCandidates": True,
            "seedProbabilitiesAveragedForSelection": True,
            "testSetLocked": True,
            "testSetSelected": False,
            "testFeaturesEncoded": False,
            "testSetUsed": False,
            "interpretation": "model_and_threshold_selection_not_final_test_performance",
        },
        "tokenizer": {"path": str(tokenizer_path.resolve()), "sha256": tokenizer_hash, "bytes": tokenizer_bytes, "fitRows": int(len(development)), "validationStatistics": tokenizer.statistics(validation["canonical_url"])},
        "artifacts": {
            "predictionsBySeed": {"path": str(predictions_path.resolve()), "sha256": prediction_hash, "bytes": prediction_bytes, "rows": int(len(predictions))},
            "meanPredictions": {"path": str(mean_predictions_path.resolve()), "sha256": mean_hash, "bytes": mean_bytes, "rows": int(len(mean_predictions))},
            "metrics": {"path": str(metrics_path.resolve()), "sha256": metrics_hash, "bytes": metrics_bytes, "rows": int(len(comparison))},
            "fittedObjects": fitted_artifacts,
        },
    }
    write_json(destination / "external_validation_manifest.json", manifest)
    _notify(progress_callback, stage="completed", event="completed", completedUnits=total_units, totalUnits=total_units, message="Selección externa completada; test final permanece bloqueado.")
    return manifest


def select_mcc_threshold(labels: np.ndarray, probabilities: np.ndarray) -> float:
    y_true = np.asarray(labels, dtype=np.int32).reshape(-1)
    scores = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    candidates = np.unique(np.concatenate(([0.0, 0.5, 1.0], scores)))
    best_threshold = 0.5
    best_key = (-np.inf, -np.inf, -np.inf, -np.inf)
    for threshold in candidates:
        predictions = scores >= threshold
        positive = y_true == 1
        negative = ~positive
        tp = int((predictions & positive).sum())
        fp = int((predictions & negative).sum())
        fn = int((~predictions & positive).sum())
        tn = int((~predictions & negative).sum())
        denominator = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
        mcc = ((tp * tn - fp * fn) / denominator) if denominator else 0.0
        f1 = (2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else 0.0
        fpr = fp / (fp + tn) if (fp + tn) else 0.0
        key = (mcc, f1, -fpr, -abs(float(threshold) - 0.5))
        if key > best_key:
            best_key = key
            best_threshold = float(threshold)
    return best_threshold


def confusion_matrix_counts(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, int]:
    y_true = np.asarray(labels, dtype=np.int32).reshape(-1)
    predicted = np.asarray(probabilities, dtype=np.float64).reshape(-1) >= threshold
    positive = y_true == 1
    negative = ~positive
    return {
        "trueNegative": int((~predicted & negative).sum()),
        "falsePositive": int((predicted & negative).sum()),
        "falseNegative": int((~predicted & positive).sum()),
        "truePositive": int((predicted & positive).sum()),
    }


def get_latest_phishing_external_validation() -> dict[str, Any]:
    paths = sorted(
        EXPERIMENTS_DIR.glob("*/phishing_oof_v1/external_validation_v1/external_validation_manifest.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not paths:
        return {"available": False, "message": "Todavía no existe una validación externa de phishing."}
    return {"available": True, **read_json(paths[0])}


def _build_validation_comparison(
    predictions: pd.DataFrame,
    mean_predictions: pd.DataFrame,
    candidate_ids: tuple[str, ...],
    model_ids: tuple[str, ...],
) -> list[dict[str, Any]]:
    labels = mean_predictions["is_phishing"].to_numpy(dtype=np.int32)
    rows = []
    for candidate_id in candidate_ids:
        column = f"probability_{candidate_id}"
        mean_scores = mean_predictions[column].to_numpy(dtype=np.float64)
        threshold = select_mcc_threshold(labels, mean_scores)
        calibrated = phishing_classification_metrics(labels, mean_scores, threshold=threshold)
        fixed = phishing_classification_metrics(labels, mean_scores, threshold=0.5)
        seed_metrics = []
        for seed, group in predictions.groupby("seed", sort=True):
            seed_metrics.append({"seed": int(seed), **phishing_classification_metrics(group["is_phishing"].to_numpy(dtype=np.int32), group[column].to_numpy(dtype=np.float64), threshold=threshold)})
        item: dict[str, Any] = {
            "candidateId": candidate_id,
            "family": "base" if candidate_id in model_ids else "baseline" if candidate_id in BASELINE_IDS else "stacking",
            "validationRows": int(len(mean_predictions)),
            "seeds": int(len(seed_metrics)),
            "calibratedThreshold": threshold,
            "thresholdObjective": "mcc",
            **{metric: calibrated[metric] for metric in METRIC_NAMES},
            "fixedThresholdMetrics": {metric: fixed[metric] for metric in METRIC_NAMES},
            "confusionMatrix": confusion_matrix_counts(labels, mean_scores, threshold),
            "seedMetrics": seed_metrics,
        }
        for metric in METRIC_NAMES:
            values = np.asarray([row[metric] for row in seed_metrics], dtype=np.float64)
            item[f"{metric}SeedMean"] = float(values.mean())
            item[f"{metric}SeedStd"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        rows.append(item)
    return sorted(rows, key=lambda row: (row["prAuc"], row["mcc"], row["f1"]), reverse=True)


def _common_development_ids(oof: pd.DataFrame, seeds: tuple[int, ...]) -> set[str]:
    expected: set[str] | None = None
    for seed in seeds:
        identifiers = set(oof.loc[oof["seed"] == seed, "sample_id"].astype(str))
        if not identifiers:
            raise ValueError(f"No existen probabilidades OOF para la semilla {seed}.")
        if expected is None:
            expected = identifiers
        elif identifiers != expected:
            raise ValueError("Las semillas OOF no cubren el mismo conjunto de desarrollo.")
    return expected or set()


def _load_development_and_validation(source: Path, development_ids: set[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = pd.read_csv(source, usecols=["canonical_url", "registrable_domain", "is_phishing", "split"])
    retained = frame[frame["split"].isin(["train", "validation"])].copy()
    retained["sample_id"] = retained["canonical_url"].map(lambda value: hashlib.sha256(str(value).encode("utf-8")).hexdigest())
    development = retained[(retained["split"] == "train") & retained["sample_id"].isin(development_ids)].copy().reset_index(drop=True)
    validation = retained[retained["split"] == "validation"].copy().sort_values("sample_id", kind="stable").reset_index(drop=True)
    if len(development) != len(development_ids):
        raise ValueError("No se pudieron reconstruir todas las filas de desarrollo usadas por OOF.")
    return development, validation


def _validate_external_partitions(development: pd.DataFrame, validation: pd.DataFrame) -> None:
    if development.empty or validation.empty:
        raise ValueError("Development y validation deben contener filas.")
    if set(development["is_phishing"].astype(int)) != {0, 1} or set(validation["is_phishing"].astype(int)) != {0, 1}:
        raise ValueError("Development y validation deben contener ambas clases.")
    if set(development["sample_id"]) & set(validation["sample_id"]):
        raise ValueError("Existe solapamiento de muestras entre development y validation.")
    if set(development["registrable_domain"]) & set(validation["registrable_domain"]):
        raise ValueError("Existe solapamiento de dominios entre development y validation.")


def _selected_epoch_count(base_manifest: dict[str, Any], *, seed: int, model_id: str) -> int:
    values = [
        int(row.get("epochsCompleted", 1))
        for row in base_manifest.get("baseFoldMetrics", [])
        if int(row.get("seed", -1)) == seed and row.get("modelId") == model_id
    ]
    if not values:
        return max(1, int(base_manifest.get("configuration", {}).get("epochs", 1)))
    return max(1, int(round(float(np.median(values)))))


def _balanced_class_weights(labels: np.ndarray) -> dict[int, float]:
    values, counts = np.unique(labels, return_counts=True)
    total = len(labels)
    return {int(value): float(total / (len(values) * count)) for value, count in zip(values, counts)}


def _verify_file(path: Path, artifact: dict[str, Any], label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"No existe el artefacto de {label}.")
    digest, size = sha256_file(path)
    if digest != artifact.get("sha256") or size != artifact.get("bytes"):
        raise ValueError(f"El artefacto de {label} no supera la verificación de integridad.")


def _latest_base_manifest() -> Path | None:
    paths = sorted(EXPERIMENTS_DIR.glob("*/phishing_oof_v1/oof_manifest.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    return paths[0] if paths else None


def _notify(callback: Callable[[dict[str, Any]], None] | None, **event: Any) -> None:
    if callback is not None:
        callback(event)
