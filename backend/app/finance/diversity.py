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
from app.finance.stacking import META_MODEL_IDS, build_finance_meta_features, build_finance_meta_models
from app.finance.validation import apply_temporal_calibrator, fit_temporal_calibrator, select_fbeta_threshold
from app.phishing.ingestion import sha256_file
from app.utils import read_json


def run_finance_diversity_ablation(
    *,
    validation_manifest_path: Path | None = None,
    output_dir: Path | None = None,
    bootstrap_iterations: int = 300,
) -> dict[str, Any]:
    if bootstrap_iterations < 100:
        raise ValueError("Se requieren al menos 100 iteraciones bootstrap financieras.")
    validation_path = validation_manifest_path or _latest_validation_manifest()
    if validation_path is None or not validation_path.is_file():
        raise FileNotFoundError("No existe una validacion temporal financiera para analizar.")
    validation_manifest = read_json(validation_path)
    validation_contract = validation_manifest.get("validation", {})
    if not validation_contract.get("testSetLocked") or validation_contract.get("testSetUsed") or validation_contract.get("testSetEncoded"):
        raise ValueError("La diversidad financiera exige test bloqueado, no codificado y no utilizado.")
    if not validation_contract.get("chronologicalCalibrationBeforeSelection"):
        raise ValueError("La validacion no contiene calibracion cronologica anterior a seleccion.")

    base_path = Path(validation_manifest.get("baseRun", {}).get("manifestPath", ""))
    if not base_path.is_file():
        raise FileNotFoundError("No existe el manifiesto OOF asociado a la validacion financiera.")
    base_manifest = read_json(base_path)
    model_ids = tuple(base_manifest.get("baseModels", []))
    seeds = tuple(int(value) for value in base_manifest.get("configuration", {}).get("seeds", []))
    if set(model_ids) != set(FINANCE_MODEL_IDS) or not seeds:
        raise ValueError("La ablacion financiera requiere las cinco arquitecturas y al menos una semilla.")

    oof_artifact = base_manifest.get("artifacts", {}).get("oofProbabilities", {})
    by_seed_artifact = validation_manifest.get("artifacts", {}).get("predictionsBySeed", {})
    mean_artifact = validation_manifest.get("artifacts", {}).get("meanPredictions", {})
    oof_path = Path(oof_artifact.get("path", ""))
    by_seed_path = Path(by_seed_artifact.get("path", ""))
    mean_path = Path(mean_artifact.get("path", ""))
    _verify_file(oof_path, oof_artifact, "probabilidades OOF")
    _verify_file(by_seed_path, by_seed_artifact, "probabilidades de validation por semilla")
    _verify_file(mean_path, mean_artifact, "probabilidades medias calibradas")
    oof = pd.read_csv(oof_path)
    validation_predictions = pd.read_csv(by_seed_path)
    mean_predictions = pd.read_csv(mean_path)
    base_columns = [f"probability_{model_id}" for model_id in model_ids]
    calibrated_columns = [f"calibrated_probability_{model_id}" for model_id in model_ids]
    required_oof = {"transaction_id", "seed", "is_fraud", *base_columns}
    required_by_seed = {"transaction_id", "timestamp_ns", "seed", "is_fraud", *base_columns}
    required_mean = {"transaction_id", "timestamp_ns", "is_fraud", "validation_partition", *base_columns, *calibrated_columns}
    if required_oof - set(oof.columns) or required_by_seed - set(validation_predictions.columns) or required_mean - set(mean_predictions.columns):
        raise ValueError("Los artefactos financieros no contienen las probabilidades o particiones requeridas.")
    if oof.duplicated(["seed", "transaction_id"]).any() or validation_predictions.duplicated(["seed", "transaction_id"]).any():
        raise ValueError("Las probabilidades financieras contienen transacciones duplicadas por semilla.")
    partition_counts = mean_predictions["validation_partition"].value_counts().to_dict()
    if set(partition_counts) != {"calibration", "selection"}:
        raise ValueError("Validation debe contener exactamente calibracion y seleccion.")
    mean_predictions = mean_predictions.sort_values(["timestamp_ns", "transaction_id"], kind="stable").reset_index(drop=True)
    calibration = mean_predictions[mean_predictions["validation_partition"] == "calibration"].copy()
    selection = mean_predictions[mean_predictions["validation_partition"] == "selection"].copy()
    if calibration["timestamp_ns"].max() >= selection["timestamp_ns"].min():
        raise ValueError("Calibracion y seleccion no respetan el orden temporal.")

    thresholds = {
        row["candidateId"]: float(row["calibratedThreshold"])
        for row in validation_manifest.get("comparison", [])
        if row.get("candidateId") in model_ids
    }
    if set(thresholds) != set(model_ids):
        raise ValueError("Faltan umbrales calibrados para uno o mas modelos financieros.")
    diversity = pairwise_finance_diversity(selection, model_ids, thresholds)

    destination = output_dir or (base_path.parent / "diversity_ablation_v1")
    destination.mkdir(parents=True, exist_ok=True)
    configurations = [("full", None, model_ids)] + [
        (f"without_{removed}", removed, tuple(model_id for model_id in model_ids if model_id != removed))
        for removed in model_ids
    ]
    raw_ablation_rows: list[dict[str, Any]] = []
    fitted_objects: list[dict[str, Any]] = []
    for seed in seeds:
        seed_oof = oof[oof["seed"] == seed].sort_values("transaction_id", kind="stable")
        seed_validation = validation_predictions[validation_predictions["seed"] == seed].sort_values(["timestamp_ns", "transaction_id"], kind="stable")
        if len(seed_validation) != len(mean_predictions) or not np.array_equal(seed_validation["transaction_id"].to_numpy(), mean_predictions["transaction_id"].to_numpy()):
            raise ValueError(f"La semilla {seed} no cubre validation en el mismo orden temporal.")
        y_oof = seed_oof["is_fraud"].to_numpy(dtype=np.int32)
        for ablation_id, removed_model_id, retained_models in configurations:
            retained_columns = [f"probability_{model_id}" for model_id in retained_models]
            x_oof, feature_names = build_finance_meta_features(seed_oof[retained_columns].to_numpy(dtype=np.float64), retained_columns)
            x_validation, _ = build_finance_meta_features(seed_validation[retained_columns].to_numpy(dtype=np.float64), retained_columns)
            for meta_model_id, meta_model in build_finance_meta_models(seed=seed).items():
                if meta_model_id == "stacking_gradient_boosting":
                    meta_model.fit(x_oof, y_oof, sample_weight=_balanced_sample_weights(y_oof))
                else:
                    meta_model.fit(x_oof, y_oof)
                probabilities = np.clip(meta_model.predict_proba(x_validation)[:, 1], 0.0, 1.0)
                model_path = destination / "models" / f"seed_{seed}" / ablation_id / f"{meta_model_id}.joblib"
                _atomic_joblib_dump(model_path, meta_model)
                digest, size = sha256_file(model_path)
                fitted_objects.append({
                    "kind": "meta_model",
                    "seed": seed,
                    "ablationId": ablation_id,
                    "removedModelId": removed_model_id,
                    "retainedModels": list(retained_models),
                    "metaModelId": meta_model_id,
                    "metaFeatureCount": len(feature_names),
                    "fitSplit": "train_oof_only",
                    "validationLabelsUsedForFit": False,
                    "path": str(model_path.resolve()),
                    "sha256": digest,
                    "bytes": size,
                })
                for position, (_, row) in enumerate(seed_validation.iterrows()):
                    raw_ablation_rows.append({
                        "transaction_id": int(row["transaction_id"]),
                        "timestamp_ns": int(row["timestamp_ns"]),
                        "is_fraud": int(row["is_fraud"]),
                        "validation_partition": str(mean_predictions.iloc[position]["validation_partition"]),
                        "seed": int(seed),
                        "ablation_id": ablation_id,
                        "removed_model_id": removed_model_id,
                        "meta_model_id": meta_model_id,
                        "probability": float(probabilities[position]),
                    })

    raw_ablation = pd.DataFrame(raw_ablation_rows).sort_values(
        ["meta_model_id", "ablation_id", "seed", "timestamp_ns", "transaction_id"], kind="stable"
    )
    expected_rows = len(mean_predictions) * len(seeds) * len(configurations) * len(META_MODEL_IDS)
    if len(raw_ablation) != expected_rows:
        raise ValueError("Las predicciones de ablacion financiera no tienen cobertura completa.")
    metrics, evaluated_selection, calibration_records = aggregate_finance_ablation_metrics(
        raw_ablation,
        configurations,
        destination=destination,
    )
    reference_meta_model = validation_manifest.get("stackingRun", {}).get("sourceCandidateId")
    if reference_meta_model not in META_MODEL_IDS:
        raise ValueError("La validacion financiera no congela un meta-learner de referencia valido.")
    contribution = finance_model_contribution(
        evaluated_selection,
        metrics,
        model_ids,
        reference_meta_model,
        bootstrap_iterations=bootstrap_iterations,
    )
    recommendation = recommend_finance_ablation(metrics, reference_meta_model, contribution)

    pairwise_path = destination / "pairwise_diversity.csv"
    raw_path = destination / "ablation_validation_probabilities_raw.csv"
    evaluated_path = destination / "ablation_selection_probabilities_calibrated.csv"
    report_path = destination / "diversity_ablation_report.json"
    _atomic_write_csv(pairwise_path, pd.DataFrame(diversity["pairs"]))
    _atomic_write_csv(raw_path, raw_ablation)
    _atomic_write_csv(evaluated_path, evaluated_selection)
    _atomic_write_json(report_path, {
        "diversity": diversity,
        "ablationMetrics": metrics,
        "contribution": contribution,
        "recommendation": recommendation,
        "calibrationRecords": calibration_records,
    })
    artifacts: dict[str, Any] = {}
    for name, path, rows in (
        ("pairwiseDiversity", pairwise_path, len(diversity["pairs"])),
        ("rawAblationPredictions", raw_path, len(raw_ablation)),
        ("evaluatedSelectionPredictions", evaluated_path, len(evaluated_selection)),
        ("report", report_path, 1),
    ):
        digest, size = sha256_file(path)
        artifacts[name] = {"path": str(path.resolve()), "sha256": digest, "bytes": size, "rows": int(rows)}

    manifest = {
        "schemaVersion": "1.0.0",
        "runId": f"{base_manifest.get('runId')}_finance_diversity_ablation",
        "domain": "finanzas",
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "status": "demo_analysis" if validation_manifest.get("status") == "demo_validation_selection" else "ablation_candidate",
        "baseRun": {"runId": base_manifest.get("runId"), "manifestPath": str(base_path.resolve()), "models": list(model_ids), "seeds": list(seeds)},
        "validationRun": {"runId": validation_manifest.get("runId"), "manifestPath": str(validation_path.resolve()), "calibrationRows": int(len(calibration)), "selectionRows": int(len(selection)), "datasetKind": validation_manifest.get("dataset", {}).get("kind")},
        "diversity": diversity,
        "ablation": {
            "strategy": "full_vs_leave_one_base_model_out_with_oof_refit_and_temporal_recalibration",
            "metaModels": list(META_MODEL_IDS),
            "configurations": [{"ablationId": item[0], "removedModelId": item[1], "retainedModels": list(item[2])} for item in configurations],
            "metrics": metrics,
            "referenceMetaModelId": reference_meta_model,
            "contribution": contribution,
            "calibrationRecords": calibration_records,
        },
        "stability": {
            "seedCount": len(seeds),
            "seedLevelInferenceAvailable": len(seeds) >= 2,
            "bootstrapUnit": "calendar_day_temporal_block",
            "bootstrapIterations": bootstrap_iterations,
            "confidenceLevel": 0.95,
            "interpretation": "selection_analysis_not_final_test_inference",
        },
        "recommendation": recommendation,
        "validation": {
            "sameRowsForAllAblations": True,
            "metaFitUsesOofOnly": True,
            "calibrationUsesFirstValidationHalfOnly": True,
            "selectionLabelsUsedForCalibration": False,
            "selectionLabelsUsedForThreshold": False,
            "calibrationBeforeSelection": True,
            "testSetLocked": True,
            "testSetEncoded": False,
            "testSetUsed": False,
        },
        "artifacts": {**artifacts, "fittedObjects": fitted_objects},
    }
    manifest_path = destination / "diversity_ablation_manifest.json"
    _atomic_write_json(manifest_path, manifest)
    manifest_hash, manifest_bytes = sha256_file(manifest_path)
    return {**manifest, "manifest": {"path": str(manifest_path.resolve()), "sha256": manifest_hash, "bytes": manifest_bytes}}


def pairwise_finance_diversity(
    selection: pd.DataFrame,
    model_ids: tuple[str, ...],
    thresholds: dict[str, float],
) -> dict[str, Any]:
    labels = selection["is_fraud"].to_numpy(dtype=np.int32)
    pairs: list[dict[str, Any]] = []
    disagreement_matrix = {model_id: {} for model_id in model_ids}
    probability_correlation_matrix = {model_id: {} for model_id in model_ids}
    for model_a in model_ids:
        for model_b in model_ids:
            scores_a = selection[f"calibrated_probability_{model_a}"].to_numpy(dtype=np.float64)
            scores_b = selection[f"calibrated_probability_{model_b}"].to_numpy(dtype=np.float64)
            predicted_a = scores_a >= thresholds[model_a]
            predicted_b = scores_b >= thresholds[model_b]
            disagreement_matrix[model_a][model_b] = float(np.mean(predicted_a != predicted_b))
            probability_correlation_matrix[model_a][model_b] = _safe_correlation(scores_a, scores_b)
    for index, model_a in enumerate(model_ids):
        for model_b in model_ids[index + 1 :]:
            scores_a = selection[f"calibrated_probability_{model_a}"].to_numpy(dtype=np.float64)
            scores_b = selection[f"calibrated_probability_{model_b}"].to_numpy(dtype=np.float64)
            predicted_a = scores_a >= thresholds[model_a]
            predicted_b = scores_b >= thresholds[model_b]
            error_a = predicted_a != labels
            error_b = predicted_b != labels
            fp_a = predicted_a & (labels == 0)
            fp_b = predicted_b & (labels == 0)
            fn_a = (~predicted_a) & (labels == 1)
            fn_b = (~predicted_b) & (labels == 1)
            pairs.append({
                "modelA": model_a,
                "modelB": model_b,
                "probabilityPearson": _safe_correlation(scores_a, scores_b),
                "probabilitySpearman": _safe_spearman(scores_a, scores_b),
                "residualPearson": _safe_correlation(np.abs(labels - scores_a), np.abs(labels - scores_b)),
                "hardPredictionDisagreementRate": float(np.mean(predicted_a != predicted_b)),
                "doubleFaultRate": float(np.mean(error_a & error_b)),
                "bothCorrectRate": float(np.mean((~error_a) & (~error_b))),
                "falsePositiveJaccard": _jaccard(fp_a, fp_b),
                "falseNegativeJaccard": _jaccard(fn_a, fn_b),
                "onlyModelACorrect": int(((~error_a) & error_b).sum()),
                "onlyModelBCorrect": int((error_a & (~error_b)).sum()),
            })
    return {
        "rows": int(len(selection)),
        "models": list(model_ids),
        "thresholdSource": "chronological_validation_calibration_f2",
        "probabilitySource": "calibrated_selection_probabilities",
        "residualDefinition": "absolute_probability_error_abs_y_minus_p",
        "pairs": pairs,
        "disagreementMatrix": disagreement_matrix,
        "probabilityCorrelationMatrix": probability_correlation_matrix,
        "meanPairwiseDisagreementRate": float(np.mean([row["hardPredictionDisagreementRate"] for row in pairs])) if pairs else 0.0,
        "mostComplementaryPair": max(pairs, key=lambda row: row["hardPredictionDisagreementRate"]) if pairs else None,
        "mostCorrelatedPair": max(pairs, key=lambda row: row["probabilityPearson"] if row["probabilityPearson"] is not None else -np.inf) if pairs else None,
    }


def aggregate_finance_ablation_metrics(
    raw_predictions: pd.DataFrame,
    configurations: list[tuple[str, str | None, tuple[str, ...]]],
    *,
    destination: Path,
) -> tuple[list[dict[str, Any]], pd.DataFrame, list[dict[str, Any]]]:
    configuration_map = {item[0]: item for item in configurations}
    metric_rows: list[dict[str, Any]] = []
    evaluated_rows: list[dict[str, Any]] = []
    calibration_records: list[dict[str, Any]] = []
    for (ablation_id, meta_model_id), group in raw_predictions.groupby(["ablation_id", "meta_model_id"], sort=False):
        averaged = group.groupby(["transaction_id", "timestamp_ns", "is_fraud", "validation_partition"], as_index=False)["probability"].mean()
        averaged = averaged.sort_values(["timestamp_ns", "transaction_id"], kind="stable")
        calibration = averaged[averaged["validation_partition"] == "calibration"]
        selection = averaged[averaged["validation_partition"] == "selection"]
        if calibration["timestamp_ns"].max() >= selection["timestamp_ns"].min():
            raise ValueError("Una ablacion financiera mezcla calibracion y seleccion.")
        y_calibration = calibration["is_fraud"].to_numpy(dtype=np.int32)
        y_selection = selection["is_fraud"].to_numpy(dtype=np.int32)
        calibrator = fit_temporal_calibrator(y_calibration, calibration["probability"].to_numpy(dtype=np.float64))
        calibrated_calibration = apply_temporal_calibrator(calibrator, calibration["probability"].to_numpy(dtype=np.float64))
        threshold = select_fbeta_threshold(y_calibration, calibrated_calibration, beta=2.0)
        calibrated_selection = apply_temporal_calibrator(calibrator, selection["probability"].to_numpy(dtype=np.float64))
        metric = finance_classification_metrics(y_selection, calibrated_selection, threshold=threshold)
        _, removed_model_id, retained_models = configuration_map[str(ablation_id)]
        calibrator_path = destination / "calibrators" / str(meta_model_id) / f"{ablation_id}.joblib"
        _atomic_joblib_dump(calibrator_path, calibrator)
        digest, size = sha256_file(calibrator_path)
        calibration_records.append({
            "ablationId": str(ablation_id),
            "metaModelId": str(meta_model_id),
            "method": calibrator["method"],
            "threshold": threshold,
            "thresholdObjective": "f2",
            "fitRows": int(len(calibration)),
            "fitEndTimestampNs": int(calibration["timestamp_ns"].max()),
            "selectionStartTimestampNs": int(selection["timestamp_ns"].min()),
            "selectionLabelsUsed": False,
            "testLabelsUsed": False,
            "path": str(calibrator_path.resolve()),
            "sha256": digest,
            "bytes": size,
        })
        metric_rows.append({
            "ablationId": str(ablation_id),
            "removedModelId": removed_model_id,
            "retainedModels": list(retained_models),
            "metaModelId": str(meta_model_id),
            "calibrationRows": int(len(calibration)),
            "selectionRows": int(len(selection)),
            "calibrationMethod": calibrator["method"],
            "calibratedThreshold": threshold,
            **{name: metric[name] for name in ("prAuc", "rocAuc", "f1", "precision", "recall", "mcc", "balancedAccuracy", "brierScore", "logLoss", "falsePositiveRate")},
            "confusionMatrix": {key: int(metric[key]) for key in ("truePositive", "trueNegative", "falsePositive", "falseNegative")},
            "seedCount": int(group["seed"].nunique()),
        })
        for position, (_, row) in enumerate(selection.iterrows()):
            evaluated_rows.append({
                "transaction_id": int(row["transaction_id"]),
                "timestamp_ns": int(row["timestamp_ns"]),
                "is_fraud": int(row["is_fraud"]),
                "ablation_id": str(ablation_id),
                "removed_model_id": removed_model_id,
                "meta_model_id": str(meta_model_id),
                "calibrated_probability": float(calibrated_selection[position]),
                "threshold": threshold,
            })
    full = {row["metaModelId"]: row for row in metric_rows if row["ablationId"] == "full"}
    for row in metric_rows:
        reference = full[row["metaModelId"]]
        row["prAucDropVsFull"] = float(reference["prAuc"] - row["prAuc"])
        row["mccDropVsFull"] = float(reference["mcc"] - row["mcc"])
        row["f1DropVsFull"] = float(reference["f1"] - row["f1"])
        row["falsePositiveRateIncreaseVsFull"] = float(row["falsePositiveRate"] - reference["falsePositiveRate"])
    evaluated = pd.DataFrame(evaluated_rows).sort_values(["meta_model_id", "ablation_id", "timestamp_ns", "transaction_id"], kind="stable")
    return sorted(metric_rows, key=lambda row: (row["metaModelId"], -row["prAuc"], row["ablationId"])), evaluated, calibration_records


def finance_model_contribution(
    evaluated: pd.DataFrame,
    metrics: list[dict[str, Any]],
    model_ids: tuple[str, ...],
    reference_meta_model: str,
    *,
    bootstrap_iterations: int,
) -> list[dict[str, Any]]:
    relevant_metrics = {row["ablationId"]: row for row in metrics if row["metaModelId"] == reference_meta_model}
    relevant = evaluated[evaluated["meta_model_id"] == reference_meta_model]
    full = relevant[relevant["ablation_id"] == "full"].sort_values(["timestamp_ns", "transaction_id"], kind="stable")
    labels = full["is_fraud"].to_numpy(dtype=np.int32)
    full_scores = full["calibrated_probability"].to_numpy(dtype=np.float64)
    day_groups = pd.to_datetime(full["timestamp_ns"], unit="ns", utc=True).dt.strftime("%Y-%m-%d").to_numpy(dtype=object)
    result: list[dict[str, Any]] = []
    for index, model_id in enumerate(model_ids):
        ablation_id = f"without_{model_id}"
        variant = relevant[relevant["ablation_id"] == ablation_id].sort_values(["timestamp_ns", "transaction_id"], kind="stable")
        if not np.array_equal(full["transaction_id"].to_numpy(), variant["transaction_id"].to_numpy()):
            raise ValueError("Las configuraciones financieras no cubren las mismas transacciones de seleccion.")
        lower, upper, probability_positive = _paired_temporal_block_bootstrap_pr_auc_drop(
            labels,
            full_scores,
            variant["calibrated_probability"].to_numpy(dtype=np.float64),
            day_groups,
            iterations=bootstrap_iterations,
            seed=42 + index,
        )
        metric = relevant_metrics[ablation_id]
        result.append({
            "modelId": model_id,
            "referenceMetaModelId": reference_meta_model,
            "prAucDropWhenRemoved": metric["prAucDropVsFull"],
            "prAucDropCi95": {"lower": lower, "upper": upper},
            "probabilityPositiveContribution": probability_positive,
            "mccDropWhenRemoved": metric["mccDropVsFull"],
            "f1DropWhenRemoved": metric["f1DropVsFull"],
            "falsePositiveRateIncreaseWhenRemoved": metric["falsePositiveRateIncreaseVsFull"],
            "interpretation": "positive_contribution" if metric["prAucDropVsFull"] > 0 else "potentially_redundant_or_harmful",
        })
    return sorted(result, key=lambda row: row["prAucDropWhenRemoved"], reverse=True)


def recommend_finance_ablation(
    metrics: list[dict[str, Any]],
    reference_meta_model: str,
    contribution: list[dict[str, Any]],
) -> dict[str, Any]:
    relevant = [row for row in metrics if row["metaModelId"] == reference_meta_model]
    full = next(row for row in relevant if row["ablationId"] == "full")
    best = max(relevant, key=lambda row: (row["prAuc"], row["mcc"], row["f1"]))
    practical_margin = 0.001
    mcc_tolerance = 0.005
    removed_contribution = next((row for row in contribution if row["modelId"] == best["removedModelId"]), None)
    statistically_clear_removal = bool(removed_contribution and removed_contribution["prAucDropCi95"]["upper"] < 0.0)
    accept_ablation = (
        best["ablationId"] != "full"
        and best["prAuc"] >= full["prAuc"] + practical_margin
        and best["mcc"] >= full["mcc"] - mcc_tolerance
        and statistically_clear_removal
    )
    selected = best if accept_ablation else full
    overall_leader = max(metrics, key=lambda row: (row["prAuc"], row["mcc"], row["f1"]))
    return {
        "referenceMetaModelId": reference_meta_model,
        "recommendedAblationId": selected["ablationId"],
        "recommendedBaseModels": selected["retainedModels"],
        "removedModelId": selected["removedModelId"],
        "fullPrAuc": full["prAuc"],
        "selectedPrAuc": selected["prAuc"],
        "ablationAccepted": accept_ablation,
        "decisionRule": {"minimumPrAucGainToRemoveModel": practical_margin, "maximumMccLoss": mcc_tolerance, "requiresBootstrapCiBelowZero": True},
        "configurationLeader": {"ablationId": overall_leader["ablationId"], "metaModelId": overall_leader["metaModelId"], "prAuc": overall_leader["prAuc"]},
        "contributionRanking": [row["modelId"] for row in contribution],
        "status": "validation_selection_only_test_locked",
    }


def _paired_temporal_block_bootstrap_pr_auc_drop(
    labels: np.ndarray,
    full_scores: np.ndarray,
    variant_scores: np.ndarray,
    groups: np.ndarray,
    *,
    iterations: int,
    seed: int,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    unique_groups = np.unique(groups)
    group_indices = {group: np.flatnonzero(groups == group) for group in unique_groups}
    differences: list[float] = []
    for _ in range(iterations):
        sampled_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        indices = np.concatenate([group_indices[group] for group in sampled_groups])
        sampled_labels = labels[indices]
        if len(np.unique(sampled_labels)) < 2:
            continue
        differences.append(float(average_precision_score(sampled_labels, full_scores[indices]) - average_precision_score(sampled_labels, variant_scores[indices])))
    if not differences:
        return 0.0, 0.0, 0.0
    values = np.asarray(differences, dtype=np.float64)
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975)), float(np.mean(values > 0.0))


def get_latest_finance_diversity_ablation() -> dict[str, Any]:
    paths = sorted(
        EXPERIMENTS_DIR.glob("*/finance_oof_v1/diversity_ablation_v1/diversity_ablation_manifest.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not paths:
        return {"available": False, "message": "Todavia no existe un estudio de diversidad y ablacion financiera."}
    return {"available": True, **read_json(paths[0])}


def _balanced_sample_weights(labels: np.ndarray) -> np.ndarray:
    values, counts = np.unique(labels, return_counts=True)
    weights = {int(value): float(len(labels) / (len(values) * count)) for value, count in zip(values, counts)}
    return np.asarray([weights[int(value)] for value in labels], dtype=np.float64)


def _safe_correlation(values_a: np.ndarray, values_b: np.ndarray) -> float | None:
    if np.std(values_a) == 0 or np.std(values_b) == 0:
        return None
    value = float(np.corrcoef(values_a, values_b)[0, 1])
    return value if np.isfinite(value) else None


def _safe_spearman(values_a: np.ndarray, values_b: np.ndarray) -> float | None:
    value = float(pd.Series(values_a).corr(pd.Series(values_b), method="spearman"))
    return value if np.isfinite(value) else None


def _jaccard(mask_a: np.ndarray, mask_b: np.ndarray) -> float | None:
    union = int((mask_a | mask_b).sum())
    return float((mask_a & mask_b).sum() / union) if union else None


def _verify_file(path: Path, artifact: dict[str, Any], label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"No existe el artefacto de {label}.")
    digest, size = sha256_file(path)
    if digest != artifact.get("sha256") or size != artifact.get("bytes"):
        raise ValueError(f"El artefacto de {label} no supera SHA-256.")


def _latest_validation_manifest() -> Path | None:
    paths = sorted(
        EXPERIMENTS_DIR.glob("*/finance_oof_v1/temporal_validation_v1/temporal_validation_manifest.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
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


def _atomic_joblib_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    joblib.dump(value, temporary)
    os.replace(temporary, path)
