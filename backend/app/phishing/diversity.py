from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from app.config import EXPERIMENTS_DIR
from app.phishing.ingestion import sha256_file
from app.phishing.metrics import phishing_classification_metrics
from app.phishing.stacking import META_MODEL_IDS, build_meta_features, build_meta_models
from app.phishing.validation import confusion_matrix_counts, select_mcc_threshold
from app.utils import read_json, write_json


def run_phishing_diversity_ablation(
    *,
    validation_manifest_path: Path | None = None,
    output_dir: Path | None = None,
    bootstrap_iterations: int = 300,
) -> dict[str, Any]:
    if bootstrap_iterations < 20:
        raise ValueError("Se requieren al menos 20 iteraciones bootstrap.")
    validation_path = validation_manifest_path or _latest_validation_manifest()
    if validation_path is None or not validation_path.is_file():
        raise FileNotFoundError("No existe una validación externa para analizar.")
    validation_manifest = read_json(validation_path)
    base_path = Path(validation_manifest.get("baseRun", {}).get("manifestPath", ""))
    if not base_path.is_file():
        raise FileNotFoundError("No existe el manifiesto OOF asociado a la validación.")
    base_manifest = read_json(base_path)
    model_ids = tuple(base_manifest.get("baseModels", []))
    seeds = tuple(int(value) for value in base_manifest.get("configuration", {}).get("seeds", []))
    if len(model_ids) < 3 or not seeds:
        raise ValueError("La ablación requiere al menos tres modelos base y una semilla.")

    oof_artifact = base_manifest.get("artifacts", {}).get("oofProbabilities", {})
    validation_artifact = validation_manifest.get("artifacts", {}).get("predictionsBySeed", {})
    mean_artifact = validation_manifest.get("artifacts", {}).get("meanPredictions", {})
    oof_path = Path(oof_artifact.get("path", ""))
    validation_predictions_path = Path(validation_artifact.get("path", ""))
    mean_predictions_path = Path(mean_artifact.get("path", ""))
    _verify_file(oof_path, oof_artifact, "probabilidades OOF")
    _verify_file(validation_predictions_path, validation_artifact, "probabilidades de validation por semilla")
    _verify_file(mean_predictions_path, mean_artifact, "probabilidades medias de validation")
    oof = pd.read_csv(oof_path)
    validation_predictions = pd.read_csv(validation_predictions_path)
    mean_predictions = pd.read_csv(mean_predictions_path)
    base_columns = [f"probability_{model_id}" for model_id in model_ids]
    required = {"sample_id", "seed", "is_phishing", *base_columns}
    if required.difference(oof.columns) or required.difference(validation_predictions.columns):
        raise ValueError("Los artefactos no contienen todas las probabilidades base requeridas.")
    if {"sample_id", "is_phishing", *base_columns}.difference(mean_predictions.columns):
        raise ValueError("El promedio de validation no contiene todas las probabilidades base.")

    thresholds = {
        row["candidateId"]: float(row["calibratedThreshold"])
        for row in validation_manifest.get("comparison", [])
        if row.get("candidateId") in model_ids
    }
    if set(thresholds) != set(model_ids):
        raise ValueError("Faltan umbrales de validation para uno o más modelos base.")
    diversity = _pairwise_diversity(mean_predictions, model_ids, thresholds)
    destination = output_dir or (base_path.parent / "diversity_ablation_v1")
    destination.mkdir(parents=True, exist_ok=True)

    ablation_rows: list[dict[str, Any]] = []
    fitted_objects: list[dict[str, Any]] = []
    configurations = [("full", None, model_ids)] + [
        (f"without_{removed}", removed, tuple(model_id for model_id in model_ids if model_id != removed))
        for removed in model_ids
    ]
    for seed in seeds:
        seed_oof = oof[oof["seed"] == seed].sort_values("sample_id", kind="stable")
        seed_validation = validation_predictions[validation_predictions["seed"] == seed].sort_values("sample_id", kind="stable")
        if len(seed_validation) != len(mean_predictions):
            raise ValueError(f"La semilla {seed} no cubre toda la partición validation.")
        for ablation_id, removed_model_id, retained_models in configurations:
            retained_columns = [f"probability_{model_id}" for model_id in retained_models]
            x_oof, feature_names = build_meta_features(seed_oof[retained_columns].to_numpy(dtype=np.float64), retained_columns)
            x_validation, _ = build_meta_features(seed_validation[retained_columns].to_numpy(dtype=np.float64), retained_columns)
            y_oof = seed_oof["is_phishing"].to_numpy(dtype=np.int32)
            for meta_model_id, model in build_meta_models(seed=seed).items():
                model.fit(x_oof, y_oof)
                probabilities = model.predict_proba(x_validation)[:, 1]
                model_path = destination / "models" / f"seed_{seed}" / ablation_id / f"{meta_model_id}.joblib"
                model_path.parent.mkdir(parents=True, exist_ok=True)
                joblib.dump(model, model_path)
                digest, size = sha256_file(model_path)
                fitted_objects.append({
                    "seed": seed,
                    "ablationId": ablation_id,
                    "removedModelId": removed_model_id,
                    "retainedModels": list(retained_models),
                    "metaModelId": meta_model_id,
                    "metaFeatureCount": len(feature_names),
                    "path": str(model_path.resolve()),
                    "sha256": digest,
                    "bytes": size,
                })
                for row_index, row in seed_validation.reset_index(drop=True).iterrows():
                    ablation_rows.append({
                        "sample_id": row["sample_id"],
                        "is_phishing": int(row["is_phishing"]),
                        "seed": seed,
                        "ablation_id": ablation_id,
                        "removed_model_id": removed_model_id,
                        "meta_model_id": meta_model_id,
                        "probability": float(probabilities[row_index]),
                    })

    ablation_predictions = pd.DataFrame(ablation_rows).sort_values(
        ["meta_model_id", "ablation_id", "seed", "sample_id"], kind="stable"
    )
    expected_rows = len(mean_predictions) * len(seeds) * len(configurations) * len(META_MODEL_IDS)
    if len(ablation_predictions) != expected_rows:
        raise ValueError("Las predicciones de ablación no tienen cobertura completa.")
    metrics = _aggregate_ablation_metrics(ablation_predictions, configurations)
    reference_meta_model = validation_manifest.get("selection", {}).get("leadingStackingCandidateId")
    if reference_meta_model not in META_MODEL_IDS:
        reference_meta_model = "stacking_ridge"
    validation_groups = _validation_group_mapping(validation_manifest, mean_predictions)
    contribution = _model_contribution(
        ablation_predictions,
        metrics,
        model_ids,
        reference_meta_model,
        validation_groups,
        bootstrap_iterations,
    )
    recommendation = _recommend_configuration(metrics, model_ids, reference_meta_model, contribution)

    pairwise_path = destination / "pairwise_diversity.csv"
    ablation_predictions_path = destination / "ablation_validation_probabilities.csv"
    report_path = destination / "diversity_ablation_report.json"
    pd.DataFrame(diversity["pairs"]).to_csv(pairwise_path, index=False)
    ablation_predictions.to_csv(ablation_predictions_path, index=False)
    write_json(report_path, {
        "diversity": diversity,
        "ablationMetrics": metrics,
        "contribution": contribution,
        "recommendation": recommendation,
    })
    pairwise_hash, pairwise_bytes = sha256_file(pairwise_path)
    prediction_hash, prediction_bytes = sha256_file(ablation_predictions_path)
    report_hash, report_bytes = sha256_file(report_path)
    manifest = {
        "schemaVersion": "1.0.0",
        "runId": f"{base_manifest.get('runId')}_diversity_ablation",
        "domain": "phishing",
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "status": "demo_analysis" if validation_manifest.get("status") == "demo_validation_selection" else "ablation_candidate",
        "baseRun": {"runId": base_manifest.get("runId"), "manifestPath": str(base_path.resolve()), "models": list(model_ids), "seeds": list(seeds)},
        "validationRun": {"runId": validation_manifest.get("runId"), "manifestPath": str(validation_path.resolve()), "rows": int(len(mean_predictions))},
        "diversity": diversity,
        "ablation": {
            "strategy": "full_vs_leave_one_base_model_out",
            "metaModels": list(META_MODEL_IDS),
            "configurations": [{"ablationId": item[0], "removedModelId": item[1], "retainedModels": list(item[2])} for item in configurations],
            "metrics": metrics,
            "referenceMetaModelId": reference_meta_model,
            "contribution": contribution,
        },
        "stability": {
            "seedCount": len(seeds),
            "seedLevelInferenceAvailable": len(seeds) >= 2,
            "bootstrapUnit": "registrable_domain",
            "bootstrapIterations": bootstrap_iterations,
            "confidenceLevel": 0.95,
            "interpretation": "validation_selection_analysis_not_final_test_inference",
        },
        "recommendation": recommendation,
        "validation": {
            "outerValidationUsed": True,
            "sameRowsForAllAblations": True,
            "testSetLocked": True,
            "testFeaturesEncoded": False,
            "testSetUsed": False,
        },
        "artifacts": {
            "pairwiseDiversity": {"path": str(pairwise_path.resolve()), "sha256": pairwise_hash, "bytes": pairwise_bytes, "rows": len(diversity["pairs"])},
            "ablationPredictions": {"path": str(ablation_predictions_path.resolve()), "sha256": prediction_hash, "bytes": prediction_bytes, "rows": int(len(ablation_predictions))},
            "report": {"path": str(report_path.resolve()), "sha256": report_hash, "bytes": report_bytes},
            "fittedObjects": fitted_objects,
        },
    }
    write_json(destination / "diversity_ablation_manifest.json", manifest)
    return manifest


def _pairwise_diversity(
    predictions: pd.DataFrame,
    model_ids: tuple[str, ...],
    thresholds: dict[str, float],
) -> dict[str, Any]:
    labels = predictions["is_phishing"].to_numpy(dtype=np.int32)
    pairs = []
    disagreement_matrix = {model_id: {} for model_id in model_ids}
    probability_correlation_matrix = {model_id: {} for model_id in model_ids}
    for model_a in model_ids:
        for model_b in model_ids:
            scores_a = predictions[f"probability_{model_a}"].to_numpy(dtype=np.float64)
            scores_b = predictions[f"probability_{model_b}"].to_numpy(dtype=np.float64)
            predicted_a = scores_a >= thresholds[model_a]
            predicted_b = scores_b >= thresholds[model_b]
            disagreement_matrix[model_a][model_b] = float(np.mean(predicted_a != predicted_b))
            probability_correlation_matrix[model_a][model_b] = _safe_correlation(scores_a, scores_b)
    for index, model_a in enumerate(model_ids):
        for model_b in model_ids[index + 1 :]:
            scores_a = predictions[f"probability_{model_a}"].to_numpy(dtype=np.float64)
            scores_b = predictions[f"probability_{model_b}"].to_numpy(dtype=np.float64)
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
    off_diagonal_disagreement = [row["hardPredictionDisagreementRate"] for row in pairs]
    return {
        "rows": int(len(predictions)),
        "models": list(model_ids),
        "thresholdSource": "external_validation_mcc_calibration",
        "residualDefinition": "absolute_probability_error_abs_y_minus_p",
        "pairs": pairs,
        "disagreementMatrix": disagreement_matrix,
        "probabilityCorrelationMatrix": probability_correlation_matrix,
        "meanPairwiseDisagreementRate": float(np.mean(off_diagonal_disagreement)) if off_diagonal_disagreement else 0.0,
        "mostComplementaryPair": max(pairs, key=lambda row: row["hardPredictionDisagreementRate"]) if pairs else None,
        "mostCorrelatedPair": max(pairs, key=lambda row: row["probabilityPearson"] if row["probabilityPearson"] is not None else -np.inf) if pairs else None,
    }


def _aggregate_ablation_metrics(
    predictions: pd.DataFrame,
    configurations: list[tuple[str, str | None, tuple[str, ...]]],
) -> list[dict[str, Any]]:
    configuration_map = {item[0]: item for item in configurations}
    rows = []
    for (ablation_id, meta_model_id), group in predictions.groupby(["ablation_id", "meta_model_id"], sort=False):
        averaged = group.groupby(["sample_id", "is_phishing"], as_index=False)["probability"].mean()
        labels = averaged["is_phishing"].to_numpy(dtype=np.int32)
        probabilities = averaged["probability"].to_numpy(dtype=np.float64)
        threshold = select_mcc_threshold(labels, probabilities)
        metric = phishing_classification_metrics(labels, probabilities, threshold=threshold)
        seed_metrics = []
        for seed, seed_group in group.groupby("seed", sort=True):
            seed_metrics.append({"seed": int(seed), **phishing_classification_metrics(seed_group["is_phishing"].to_numpy(dtype=np.int32), seed_group["probability"].to_numpy(dtype=np.float64), threshold=threshold)})
        _, removed_model_id, retained_models = configuration_map[str(ablation_id)]
        rows.append({
            "ablationId": str(ablation_id),
            "removedModelId": removed_model_id,
            "retainedModels": list(retained_models),
            "metaModelId": str(meta_model_id),
            "validationRows": int(len(averaged)),
            "calibratedThreshold": threshold,
            **{name: metric[name] for name in ("prAuc", "rocAuc", "f1", "precision", "recall", "mcc", "balancedAccuracy", "falsePositiveRate")},
            "confusionMatrix": confusion_matrix_counts(labels, probabilities, threshold),
            "seedMetrics": seed_metrics,
            "seedCount": len(seed_metrics),
        })
    full = {row["metaModelId"]: row for row in rows if row["ablationId"] == "full"}
    for row in rows:
        reference = full[row["metaModelId"]]
        row["prAucDropVsFull"] = float(reference["prAuc"] - row["prAuc"])
        row["mccDropVsFull"] = float(reference["mcc"] - row["mcc"])
        row["f1DropVsFull"] = float(reference["f1"] - row["f1"])
        row["falsePositiveRateIncreaseVsFull"] = float(row["falsePositiveRate"] - reference["falsePositiveRate"])
    return sorted(rows, key=lambda row: (row["metaModelId"], -row["prAuc"], row["ablationId"]))


def _model_contribution(
    predictions: pd.DataFrame,
    metrics: list[dict[str, Any]],
    model_ids: tuple[str, ...],
    reference_meta_model: str,
    validation_groups: dict[str, str],
    bootstrap_iterations: int,
) -> list[dict[str, Any]]:
    relevant = [row for row in metrics if row["metaModelId"] == reference_meta_model]
    metric_map = {row["ablationId"]: row for row in relevant}
    averaged = predictions[predictions["meta_model_id"] == reference_meta_model].groupby(
        ["ablation_id", "sample_id", "is_phishing"], as_index=False
    )["probability"].mean()
    full = averaged[averaged["ablation_id"] == "full"].sort_values("sample_id", kind="stable")
    labels = full["is_phishing"].to_numpy(dtype=np.int32)
    full_scores = full["probability"].to_numpy(dtype=np.float64)
    groups = np.asarray([validation_groups[sample_id] for sample_id in full["sample_id"]], dtype=object)
    result = []
    for model_id in model_ids:
        ablation_id = f"without_{model_id}"
        variant = averaged[averaged["ablation_id"] == ablation_id].sort_values("sample_id", kind="stable")
        if not np.array_equal(full["sample_id"].to_numpy(), variant["sample_id"].to_numpy()):
            raise ValueError("Las configuraciones de ablación no cubren las mismas muestras.")
        lower, upper = _paired_group_bootstrap_pr_auc_drop(
            labels,
            full_scores,
            variant["probability"].to_numpy(dtype=np.float64),
            groups,
            iterations=bootstrap_iterations,
            seed=42 + model_ids.index(model_id),
        )
        metric = metric_map[ablation_id]
        result.append({
            "modelId": model_id,
            "referenceMetaModelId": reference_meta_model,
            "prAucDropWhenRemoved": metric["prAucDropVsFull"],
            "prAucDropCi95": {"lower": lower, "upper": upper},
            "mccDropWhenRemoved": metric["mccDropVsFull"],
            "f1DropWhenRemoved": metric["f1DropVsFull"],
            "falsePositiveRateIncreaseWhenRemoved": metric["falsePositiveRateIncreaseVsFull"],
            "interpretation": "positive_contribution" if metric["prAucDropVsFull"] > 0 else "potentially_redundant_or_harmful",
        })
    return sorted(result, key=lambda row: row["prAucDropWhenRemoved"], reverse=True)


def _paired_group_bootstrap_pr_auc_drop(
    labels: np.ndarray,
    full_scores: np.ndarray,
    variant_scores: np.ndarray,
    groups: np.ndarray,
    *,
    iterations: int,
    seed: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    unique_groups = np.unique(groups)
    group_indices = {group: np.flatnonzero(groups == group) for group in unique_groups}
    differences = []
    for _ in range(iterations):
        sampled_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        indices = np.concatenate([group_indices[group] for group in sampled_groups])
        sampled_labels = labels[indices]
        if len(np.unique(sampled_labels)) < 2:
            continue
        differences.append(float(
            average_precision_score(sampled_labels, full_scores[indices])
            - average_precision_score(sampled_labels, variant_scores[indices])
        ))
    if not differences:
        return 0.0, 0.0
    return float(np.percentile(differences, 2.5)), float(np.percentile(differences, 97.5))


def _recommend_configuration(
    metrics: list[dict[str, Any]],
    model_ids: tuple[str, ...],
    reference_meta_model: str,
    contribution: list[dict[str, Any]],
) -> dict[str, Any]:
    relevant = [row for row in metrics if row["metaModelId"] == reference_meta_model]
    full = next(row for row in relevant if row["ablationId"] == "full")
    best = max(relevant, key=lambda row: (row["prAuc"], row["mcc"], row["f1"]))
    practical_margin = 0.001
    mcc_tolerance = 0.005
    accept_ablation = (
        best["ablationId"] != "full"
        and best["prAuc"] >= full["prAuc"] + practical_margin
        and best["mcc"] >= full["mcc"] - mcc_tolerance
    )
    selected = best if accept_ablation else full
    overall_leader = max(metrics, key=lambda row: (row["prAuc"], row["mcc"], row["f1"]))
    return {
        "referenceMetaModelId": reference_meta_model,
        "recommendedAblationId": selected["ablationId"],
        "recommendedBaseModels": selected["retainedModels"],
        "removedModelId": selected["removedModelId"],
        "decisionRule": {"minimumPrAucGainToRemoveModel": practical_margin, "maximumMccLoss": mcc_tolerance},
        "fullPrAuc": full["prAuc"],
        "selectedPrAuc": selected["prAuc"],
        "configurationLeader": {"ablationId": overall_leader["ablationId"], "metaModelId": overall_leader["metaModelId"], "prAuc": overall_leader["prAuc"]},
        "contributionRanking": [row["modelId"] for row in contribution],
        "status": "validation_selection_only_pending_freeze_and_test",
    }


def _validation_group_mapping(validation_manifest: dict[str, Any], mean_predictions: pd.DataFrame) -> dict[str, str]:
    source = Path(validation_manifest.get("dataset", {}).get("sourcePath", ""))
    if not source.is_file():
        raise FileNotFoundError("No existe el silver para construir el bootstrap por dominio.")
    frame = pd.read_csv(source, usecols=["canonical_url", "registrable_domain", "split"])
    validation = frame[frame["split"] == "validation"].copy()
    validation["sample_id"] = validation["canonical_url"].map(lambda value: hashlib.sha256(str(value).encode("utf-8")).hexdigest())
    mapping = dict(zip(validation["sample_id"], validation["registrable_domain"].astype(str)))
    missing = set(mean_predictions["sample_id"].astype(str)).difference(mapping)
    if missing:
        raise ValueError("Faltan dominios para muestras de validation.")
    return mapping


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


def get_latest_phishing_diversity_ablation() -> dict[str, Any]:
    paths = sorted(
        EXPERIMENTS_DIR.glob("*/phishing_oof_v1/diversity_ablation_v1/diversity_ablation_manifest.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not paths:
        return {"available": False, "message": "Todavía no existe un estudio de diversidad y ablación de phishing."}
    return {"available": True, **read_json(paths[0])}


def _verify_file(path: Path, artifact: dict[str, Any], label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"No existe el artefacto de {label}.")
    digest, size = sha256_file(path)
    if digest != artifact.get("sha256") or size != artifact.get("bytes"):
        raise ValueError(f"El artefacto de {label} no supera la verificación de integridad.")


def _latest_validation_manifest() -> Path | None:
    paths = sorted(
        EXPERIMENTS_DIR.glob("*/phishing_oof_v1/external_validation_v1/external_validation_manifest.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return paths[0] if paths else None
