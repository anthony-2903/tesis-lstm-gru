from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from app.config import EXPERIMENTS_DIR
from app.finance.metrics import finance_classification_metrics
from app.finance.models import FINANCE_MODEL_IDS
from app.finance.validation import apply_temporal_calibrator, fit_temporal_calibrator, select_fbeta_threshold
from app.phishing.ingestion import sha256_file
from app.utils import read_json


def run_finance_optimized_stacking_revalidation(
    *,
    diversity_manifest_path: Path | None = None,
    output_dir: Path | None = None,
    bootstrap_iterations: int = 500,
) -> dict[str, Any]:
    if bootstrap_iterations < 100:
        raise ValueError("La revalidacion exige al menos 100 iteraciones bootstrap.")
    diversity_path = diversity_manifest_path or _latest_diversity_manifest()
    if diversity_path is None or not diversity_path.is_file():
        raise FileNotFoundError("No existe una ablacion financiera para revalidar.")
    diversity = read_json(diversity_path)
    validation_path = Path(diversity.get("validationRun", {}).get("manifestPath", ""))
    if not validation_path.is_file():
        raise FileNotFoundError("La ablacion no referencia una validacion temporal valida.")
    validation = read_json(validation_path)
    base_path = Path(validation.get("baseRun", {}).get("manifestPath", ""))
    if not base_path.is_file():
        raise FileNotFoundError("La validacion no referencia probabilidades OOF validas.")
    base = read_json(base_path)
    seeds = tuple(int(value) for value in base.get("configuration", {}).get("seeds", []))
    if set(base.get("baseModels", [])) != set(FINANCE_MODEL_IDS) or not seeds:
        raise ValueError("La revalidacion requiere las cinco arquitecturas base.")
    if not _test_locked(validation.get("validation", {})) or not _test_locked(diversity.get("validation", {})):
        raise ValueError("El test debe permanecer bloqueado, no codificado y no usado.")

    mean_artifact = validation.get("artifacts", {}).get("meanPredictions", {})
    by_seed_artifact = validation.get("artifacts", {}).get("predictionsBySeed", {})
    raw_artifact = diversity.get("artifacts", {}).get("rawAblationPredictions", {})
    mean_path = Path(mean_artifact.get("path", ""))
    by_seed_path = Path(by_seed_artifact.get("path", ""))
    raw_path = Path(raw_artifact.get("path", ""))
    _verify(mean_path, mean_artifact, "predicciones base")
    _verify(by_seed_path, by_seed_artifact, "predicciones base por semilla")
    _verify(raw_path, raw_artifact, "predicciones de ablacion")
    mean = pd.read_csv(mean_path).sort_values(["timestamp_ns", "transaction_id"], kind="stable").reset_index(drop=True)
    by_seed = pd.read_csv(by_seed_path)
    raw = pd.read_csv(raw_path)
    required_mean = {"transaction_id", "timestamp_ns", "is_fraud", *(f"probability_{model_id}" for model_id in FINANCE_MODEL_IDS)}
    required_by_seed = {"transaction_id", "timestamp_ns", "seed", "is_fraud", *(f"probability_{model_id}" for model_id in FINANCE_MODEL_IDS)}
    required_raw = {"transaction_id", "timestamp_ns", "is_fraud", "seed", "ablation_id", "meta_model_id", "probability"}
    if required_mean - set(mean.columns) or required_by_seed - set(by_seed.columns) or required_raw - set(raw.columns):
        raise ValueError("Los artefactos no cumplen el contrato de revalidacion.")
    if set(int(value) for value in raw["seed"].unique()) != set(seeds) or set(int(value) for value in by_seed["seed"].unique()) != set(seeds):
        raise ValueError("Las predicciones por semilla no cubren todas las semillas declaradas.")

    calibration_positions, selection_positions, comparison_positions = chronological_three_way_split(mean["is_fraud"].to_numpy(dtype=np.int32))
    partition = np.full(len(mean), "", dtype=object)
    partition[calibration_positions] = "calibration"
    partition[selection_positions] = "ablation_selection"
    partition[comparison_positions] = "independent_comparison"
    mean["nested_partition"] = partition
    partition_by_id = mean.set_index("transaction_id")["nested_partition"].to_dict()

    averaged_ablation = raw.groupby(["transaction_id", "timestamp_ns", "is_fraud", "ablation_id", "meta_model_id"], as_index=False)["probability"].mean()
    averaged_ablation["nested_partition"] = averaged_ablation["transaction_id"].map(partition_by_id)
    expected_configs = {item["ablationId"] for item in diversity.get("ablation", {}).get("configurations", [])}
    observed_configs = set(averaged_ablation["ablation_id"].astype(str))
    if expected_configs != observed_configs:
        raise ValueError("La revalidacion no dispone de todas las configuraciones de ablacion.")

    destination = output_dir or (base_path.parent / "optimized_stacking_revalidation_v1")
    destination.mkdir(parents=True, exist_ok=True)
    selection_metrics: list[dict[str, Any]] = []
    stacking_scores: dict[tuple[str, str], dict[str, Any]] = {}
    stacking_calibrators: dict[tuple[str, str], dict[str, Any]] = {}
    calibrator_records: list[dict[str, Any]] = []
    for (ablation_id, meta_model_id), group in averaged_ablation.groupby(["ablation_id", "meta_model_id"], sort=True):
        ordered = group.sort_values(["timestamp_ns", "transaction_id"], kind="stable")
        if not np.array_equal(ordered["transaction_id"].to_numpy(), mean["transaction_id"].to_numpy()):
            raise ValueError("Una variante no cubre las mismas transacciones de validation.")
        record = _calibrate_nested_candidate(
            ordered["is_fraud"].to_numpy(dtype=np.int32),
            ordered["probability"].to_numpy(dtype=np.float64),
            calibration_positions,
            selection_positions,
            comparison_positions,
        )
        calibrator_path = destination / "calibrators" / str(meta_model_id) / f"{ablation_id}.joblib"
        calibrator = record.pop("calibrator")
        stacking_calibrators[(str(ablation_id), str(meta_model_id))] = calibrator
        _atomic_joblib(calibrator_path, calibrator)
        digest, size = sha256_file(calibrator_path)
        selection_metric = finance_classification_metrics(
            mean.loc[selection_positions, "is_fraud"].to_numpy(dtype=np.int32),
            record["selectionScores"],
            threshold=record["threshold"],
        )
        row = {
            "ablationId": str(ablation_id),
            "metaModelId": str(meta_model_id),
            "calibrationMethod": record["method"],
            "calibratedThreshold": record["threshold"],
            "selectionRows": int(len(selection_positions)),
            **{key: selection_metric[key] for key in ("prAuc", "rocAuc", "f1", "precision", "recall", "mcc", "balancedAccuracy", "brierScore", "logLoss", "falsePositiveRate")},
        }
        selection_metrics.append(row)
        stacking_scores[(str(ablation_id), str(meta_model_id))] = record
        calibrator_records.append({"candidateType": "stacking_variant", "ablationId": str(ablation_id), "metaModelId": str(meta_model_id), "method": record["method"], "threshold": record["threshold"], "fitRows": int(len(calibration_positions)), "selectionLabelsUsedForCalibration": False, "comparisonLabelsUsedForCalibrationOrSelection": False, "path": str(calibrator_path.resolve()), "sha256": digest, "bytes": size})

    selection_metrics.sort(key=lambda row: (row["prAuc"], row["mcc"], row["f1"]), reverse=True)
    selected = selection_metrics[0]
    selected_key = (selected["ablationId"], selected["metaModelId"])
    final_rows: list[dict[str, Any]] = []
    final_scores: dict[str, np.ndarray] = {}
    base_calibrators: dict[str, dict[str, Any]] = {}
    base_thresholds: dict[str, float] = {}
    y_final = mean.loc[comparison_positions, "is_fraud"].to_numpy(dtype=np.int32)

    for model_id in FINANCE_MODEL_IDS:
        probability = mean[f"probability_{model_id}"].to_numpy(dtype=np.float64)
        record = _calibrate_nested_candidate(mean["is_fraud"].to_numpy(dtype=np.int32), probability, calibration_positions, selection_positions, comparison_positions)
        calibrator_path = destination / "calibrators" / "base" / f"{model_id}.joblib"
        calibrator = record.pop("calibrator")
        base_calibrators[model_id] = calibrator
        base_thresholds[model_id] = float(record["threshold"])
        _atomic_joblib(calibrator_path, calibrator)
        digest, size = sha256_file(calibrator_path)
        metric = finance_classification_metrics(y_final, record["comparisonScores"], threshold=record["threshold"])
        final_scores[model_id] = record["comparisonScores"]
        final_rows.append(_final_row(model_id, "base", None, None, record, metric, len(comparison_positions)))
        calibrator_records.append({"candidateType": "base", "modelId": model_id, "method": record["method"], "threshold": record["threshold"], "fitRows": int(len(calibration_positions)), "selectionLabelsUsedForCalibration": False, "comparisonLabelsUsedForCalibrationOrSelection": False, "path": str(calibrator_path.resolve()), "sha256": digest, "bytes": size})

    selected_record = stacking_scores[selected_key]
    selected_metric = finance_classification_metrics(y_final, selected_record["comparisonScores"], threshold=selected_record["threshold"])
    final_scores["stacking"] = selected_record["comparisonScores"]
    final_rows.append(_final_row("stacking", "stacking", selected["metaModelId"], selected["ablationId"], selected_record, selected_metric, len(comparison_positions)))
    final_rows.sort(key=lambda row: (row["prAuc"], row["mcc"], row["f1"]), reverse=True)
    seed_metrics = _independent_seed_metrics(
        mean=mean,
        comparison_positions=comparison_positions,
        by_seed=by_seed,
        raw=raw,
        seeds=seeds,
        selected_key=selected_key,
        base_calibrators=base_calibrators,
        base_thresholds=base_thresholds,
        stacking_calibrator=stacking_calibrators[selected_key],
        stacking_threshold=float(selected_record["threshold"]),
    )
    best_base = next(row for row in final_rows if row["family"] == "base")
    bootstrap = _temporal_block_bootstrap(
        y_final,
        final_scores["stacking"],
        final_scores[best_base["candidateId"]],
        mean.loc[comparison_positions, "timestamp_ns"].to_numpy(dtype=np.int64),
        iterations=bootstrap_iterations,
    )

    partition_path = destination / "nested_validation_partitions.csv"
    selection_path = destination / "stacking_ablation_selection_metrics.json"
    comparison_path = destination / "independent_six_candidate_comparison.json"
    _atomic_csv(partition_path, mean[["transaction_id", "timestamp_ns", "is_fraud", "nested_partition"]])
    _atomic_json(selection_path, {"items": selection_metrics, "selected": selected})
    _atomic_json(comparison_path, {"items": final_rows, "seedMetrics": seed_metrics, "bootstrap": bootstrap})
    artifacts: dict[str, Any] = {}
    for name, path, rows in (("partitions", partition_path, len(mean)), ("selectionMetrics", selection_path, len(selection_metrics)), ("independentComparison", comparison_path, len(final_rows))):
        digest, size = sha256_file(path)
        artifacts[name] = {"path": str(path.resolve()), "sha256": digest, "bytes": size, "rows": int(rows)}

    real_full = validation.get("validation", {}).get("readyForFinalTestEvaluation") is True and len(seeds) >= 5
    manifest = {
        "schemaVersion": "1.0.0",
        "runId": f"{base.get('runId')}_optimized_stacking_revalidation",
        "domain": "finanzas",
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "status": "optimized_stacking_revalidation_candidate" if real_full else "demo_optimized_stacking_revalidation",
        "baseRun": {"runId": base.get("runId"), "manifestPath": str(base_path.resolve()), "seeds": list(seeds)},
        "validationRun": {"runId": validation.get("runId"), "manifestPath": str(validation_path.resolve())},
        "diversityRun": {"runId": diversity.get("runId"), "manifestPath": str(diversity_path.resolve())},
        "protocol": {"strategy": "three_chronological_blocks_calibration_ablation_selection_independent_comparison", "calibrationRows": int(len(calibration_positions)), "ablationSelectionRows": int(len(selection_positions)), "independentComparisonRows": int(len(comparison_positions)), "sameRowsForAllCandidates": True},
        "stackingSelection": {"candidatePrimaryMetric": "prAuc", "configurationsCompared": len(selection_metrics), "selectedAblationId": selected["ablationId"], "selectedMetaModelId": selected["metaModelId"], "selectedThreshold": selected["calibratedThreshold"], "selectedOn": "middle_chronological_validation_block", "metrics": selection_metrics},
        "independentComparison": {"candidateIds": [row["candidateId"] for row in final_rows], "winnerCandidateId": final_rows[0]["candidateId"], "stackingRank": next(index + 1 for index, row in enumerate(final_rows) if row["candidateId"] == "stacking"), "stackingBeatsBestBase": final_scores and next(row for row in final_rows if row["candidateId"] == "stacking")["prAuc"] > best_base["prAuc"], "metrics": final_rows, "seedMetrics": seed_metrics, "seedCount": len(seeds), "stackingVersusBestBase": {"bestBaseCandidateId": best_base["candidateId"], **bootstrap}},
        "calibrators": calibrator_records,
        "validation": {"calibrationLabelsUsedForCalibrationAndThreshold": True, "ablationSelectionLabelsUsedOnlyForStackingConfigurationSelection": True, "independentComparisonLabelsUsedForCalibrationThresholdOrSelection": False, "testSetLocked": True, "testSetEncoded": False, "testSetUsed": False, "eligibleForFreeze": real_full, "interpretation": "predeclared_nested_validation_before_single_test" if real_full else "synthetic_retrospective_protocol_pilot_not_final_evidence"},
        "artifacts": artifacts,
    }
    manifest_path = destination / "optimized_stacking_revalidation_manifest.json"
    _atomic_json(manifest_path, manifest)
    digest, size = sha256_file(manifest_path)
    return {**manifest, "manifest": {"path": str(manifest_path.resolve()), "sha256": digest, "bytes": size}}


def _independent_seed_metrics(
    *,
    mean: pd.DataFrame,
    comparison_positions: np.ndarray,
    by_seed: pd.DataFrame,
    raw: pd.DataFrame,
    seeds: tuple[int, ...],
    selected_key: tuple[str, str],
    base_calibrators: dict[str, dict[str, Any]],
    base_thresholds: dict[str, float],
    stacking_calibrator: dict[str, Any],
    stacking_threshold: float,
) -> list[dict[str, Any]]:
    keys = ["transaction_id", "timestamp_ns", "is_fraud"]
    expected = mean.loc[comparison_positions, keys].reset_index(drop=True)
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        seed_base = by_seed[by_seed["seed"] == seed]
        aligned_base = expected.merge(seed_base, on=keys, how="left", sort=False, validate="one_to_one")
        for model_id in FINANCE_MODEL_IDS:
            column = f"probability_{model_id}"
            if aligned_base[column].isna().any():
                raise ValueError(f"La semilla {seed} no cubre la comparación independiente para {model_id}.")
            calibrated = apply_temporal_calibrator(base_calibrators[model_id], aligned_base[column].to_numpy(dtype=np.float64))
            metric = finance_classification_metrics(expected["is_fraud"].to_numpy(dtype=np.int32), calibrated, threshold=base_thresholds[model_id])
            rows.append({"seed": seed, "candidateId": model_id, "family": "base", **{key: metric[key] for key in ("prAuc", "f1", "mcc", "recall", "falsePositiveRate")}})

        seed_stacking = raw[
            (raw["seed"] == seed)
            & (raw["ablation_id"].astype(str) == selected_key[0])
            & (raw["meta_model_id"].astype(str) == selected_key[1])
        ][[*keys, "probability"]]
        aligned_stacking = expected.merge(seed_stacking, on=keys, how="left", sort=False, validate="one_to_one")
        if aligned_stacking["probability"].isna().any():
            raise ValueError(f"La semilla {seed} no cubre la comparación independiente para Stacking.")
        calibrated_stacking = apply_temporal_calibrator(stacking_calibrator, aligned_stacking["probability"].to_numpy(dtype=np.float64))
        stacking_metric = finance_classification_metrics(expected["is_fraud"].to_numpy(dtype=np.int32), calibrated_stacking, threshold=stacking_threshold)
        rows.append({"seed": seed, "candidateId": "stacking", "family": "stacking", **{key: stacking_metric[key] for key in ("prAuc", "f1", "mcc", "recall", "falsePositiveRate")}})
    return rows


def chronological_three_way_split(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(labels, dtype=np.int32).reshape(-1)
    if len(values) < 60:
        raise ValueError("Validation requiere al menos 60 filas para tres bloques temporales.")
    first = len(values) // 3
    second = 2 * len(values) // 3
    blocks = (np.arange(first), np.arange(first, second), np.arange(second, len(values)))
    if any(set(np.unique(values[index])) != {0, 1} for index in blocks):
        raise ValueError("Cada bloque temporal de revalidacion debe contener ambas clases.")
    return blocks


def get_latest_finance_optimized_revalidation() -> dict[str, Any]:
    paths = sorted(EXPERIMENTS_DIR.glob("*/finance_oof_v1/optimized_stacking_revalidation_v1/optimized_stacking_revalidation_manifest.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not paths:
        return {"available": False, "message": "Todavia no existe una revalidacion del Stacking financiero optimizado."}
    return {"available": True, **read_json(paths[0])}


def _calibrate_nested_candidate(labels: np.ndarray, probabilities: np.ndarray, calibration: np.ndarray, selection: np.ndarray, comparison: np.ndarray) -> dict[str, Any]:
    calibrator = fit_temporal_calibrator(labels[calibration], probabilities[calibration])
    calibrated_fit = apply_temporal_calibrator(calibrator, probabilities[calibration])
    return {"calibrator": calibrator, "method": calibrator["method"], "threshold": select_fbeta_threshold(labels[calibration], calibrated_fit, beta=2.0), "selectionScores": apply_temporal_calibrator(calibrator, probabilities[selection]), "comparisonScores": apply_temporal_calibrator(calibrator, probabilities[comparison])}


def _final_row(candidate_id: str, family: str, source_id: str | None, ablation_id: str | None, record: dict[str, Any], metric: dict[str, Any], rows: int) -> dict[str, Any]:
    return {"candidateId": candidate_id, "family": family, "sourceCandidateId": source_id, "ablationId": ablation_id, "comparisonRows": rows, "calibrationMethod": record["method"], "calibratedThreshold": record["threshold"], **{key: metric[key] for key in ("prAuc", "rocAuc", "f1", "precision", "recall", "mcc", "balancedAccuracy", "brierScore", "logLoss", "falsePositiveRate")}}


def _temporal_block_bootstrap(labels: np.ndarray, stacking: np.ndarray, base: np.ndarray, timestamps: np.ndarray, *, iterations: int) -> dict[str, Any]:
    groups = pd.to_datetime(timestamps, unit="ns", utc=True).strftime("%Y-%m-%d").to_numpy(dtype=object)
    unique = np.unique(groups)
    indices = {group: np.flatnonzero(groups == group) for group in unique}
    rng = np.random.default_rng(42)
    deltas = []
    for _ in range(iterations):
        sample = np.concatenate([indices[group] for group in rng.choice(unique, size=len(unique), replace=True)])
        if len(np.unique(labels[sample])) == 2:
            deltas.append(float(average_precision_score(labels[sample], stacking[sample]) - average_precision_score(labels[sample], base[sample])))
    values = np.asarray(deltas, dtype=np.float64)
    observed = float(average_precision_score(labels, stacking) - average_precision_score(labels, base))
    return {"metric": "prAuc", "method": "paired_calendar_day_block_bootstrap", "iterations": int(len(values)), "observedDelta": observed, "ciLower": float(np.quantile(values, 0.025)), "ciUpper": float(np.quantile(values, 0.975)), "probabilityStackingBetter": float(np.mean(values > 0)), "statisticallyClearAt95Percent": bool(np.quantile(values, 0.025) > 0 or np.quantile(values, 0.975) < 0)}


def _test_locked(value: dict[str, Any]) -> bool:
    return value.get("testSetLocked") is True and value.get("testSetEncoded") is False and value.get("testSetUsed") is False


def _verify(path: Path, artifact: dict[str, Any], label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"No existe el artefacto de {label}.")
    digest, size = sha256_file(path)
    if digest != artifact.get("sha256") or size != artifact.get("bytes"):
        raise ValueError(f"El artefacto de {label} no supera SHA-256.")


def _latest_diversity_manifest() -> Path | None:
    paths = sorted(EXPERIMENTS_DIR.glob("*/finance_oof_v1/diversity_ablation_v1/diversity_ablation_manifest.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    return paths[0] if paths else None


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _atomic_joblib(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    joblib.dump(value, temporary)
    os.replace(temporary, path)
