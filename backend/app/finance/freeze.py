from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

from app.config import EXPERIMENTS_DIR
from app.finance.models import FINANCE_MODEL_IDS
from app.finance.stacking import META_MODEL_IDS
from app.phishing.ingestion import sha256_file
from app.utils import read_json, write_json


class FinanceFreezeGateError(ValueError):
    def __init__(self, audit: dict[str, Any]) -> None:
        self.audit = audit
        super().__init__("La corrida financiera no supera la puerta de congelamiento de tesis.")


def audit_finance_freeze_readiness(
    *,
    validation_manifest_path: Path | None = None,
    diversity_manifest_path: Path | None = None,
    verify_artifacts: bool = False,
) -> dict[str, Any]:
    lineage = _resolve_lineage(validation_manifest_path, diversity_manifest_path)
    checks: list[dict[str, Any]] = []
    if lineage.get("error"):
        _check(checks, "lineage_available", False, "linaje financiero completo", lineage["error"])
        return _audit_result(checks, lineage, verify_artifacts, [])

    base = lineage["baseManifest"]
    stacking = lineage["stackingManifest"]
    validation = lineage["validationManifest"]
    diversity = lineage["diversityManifest"]
    revalidation = lineage["revalidationManifest"]
    sequence = lineage["sequenceManifest"]
    model_ids = tuple(str(value) for value in base.get("baseModels", []))
    seeds = tuple(int(value) for value in base.get("configuration", {}).get("seeds", []))
    folds = tuple(int(value) for value in base.get("validation", {}).get("oofFolds", []))
    dataset = base.get("dataset", {})

    _check(checks, "real_dataset", validation.get("dataset", {}).get("kind") == "real_world_curated_financial_dataset" and validation.get("validation", {}).get("externalRealDatasetUsed") is True, "dataset transaccional real curado", validation.get("dataset", {}).get("kind"))
    _check(checks, "sequence_thesis_ready", sequence.get("readiness", {}).get("readyForThesisTraining") is True, "readyForThesisTraining=true", sequence.get("readiness", {}))
    _check(checks, "sequence_test_lock", _sequence_test_locked(sequence), "test bloqueado, no codificado y no evaluado", sequence.get("testLock", {}))

    _check(checks, "base_status", base.get("status") == "thesis_base_models_candidate", "thesis_base_models_candidate", base.get("status"))
    _check(checks, "base_protocol", base.get("protocol") == "thesis", "thesis", base.get("protocol"))
    _check(checks, "five_models", set(model_ids) == set(FINANCE_MODEL_IDS), list(FINANCE_MODEL_IDS), list(model_ids))
    _check(checks, "five_seeds", len(set(seeds)) >= 5, "al menos cinco semillas", list(seeds))
    _check(checks, "five_folds", len(set(folds)) == 5, "cinco folds OOF", list(folds))
    _check(checks, "full_oof_rows", int(dataset.get("oofRowsAvailablePerSeed", 0)) > 0 and int(dataset.get("oofRowsUsedPerSeed", 0)) == int(dataset.get("oofRowsAvailablePerSeed", 0)) and not dataset.get("demoSubset", True), dataset.get("oofRowsAvailablePerSeed"), dataset.get("oofRowsUsedPerSeed"))
    expected_training = {(seed, fold, model_id) for seed in seeds for fold in folds for model_id in model_ids}
    observed_training = {
        (int(row.get("seed", -1)), int(row.get("fold", -1)), str(row.get("modelId")))
        for row in base.get("baseFoldMetrics", [])
    }
    _check(checks, "complete_oof_grid", observed_training == expected_training, len(expected_training), len(observed_training))
    expected_oof_rows = int(dataset.get("oofRowsUsedPerSeed", 0)) * len(seeds)
    observed_oof_rows = int(base.get("artifacts", {}).get("oofProbabilities", {}).get("rows", 0))
    _check(checks, "complete_oof_coverage", expected_oof_rows > 0 and observed_oof_rows == expected_oof_rows, expected_oof_rows, observed_oof_rows)
    _check(checks, "base_test_lock", _test_locked(base.get("validation", {})), "test bloqueado y no usado", base.get("validation", {}))

    _check(checks, "stacking_lineage", stacking.get("baseRun", {}).get("runId") == base.get("runId"), base.get("runId"), stacking.get("baseRun", {}).get("runId"))
    _check(checks, "stacking_status", stacking.get("status") == "meta_oof_candidate", "meta_oof_candidate", stacking.get("status"))
    stacking_validation = stacking.get("validation", {})
    _check(checks, "strict_temporal_stacking", stacking_validation.get("strictTemporalCrossFit") is True and stacking_validation.get("metaDependencyLeakageControlled") is True and stacking_validation.get("futureFoldsUsed") is False, "cross-fit temporal estricto", stacking_validation)
    _check(checks, "stacking_test_lock", _test_locked(stacking_validation), "test bloqueado y no usado", stacking_validation)
    _check(checks, "both_meta_learners", set(stacking.get("metaFeatures", {}).get("baseProbabilityColumns", [])) == {f"probability_{model_id}" for model_id in FINANCE_MODEL_IDS} and set(META_MODEL_IDS).issubset({row.get("candidateId") for row in stacking.get("candidateAggregate", [])}), list(META_MODEL_IDS), [row.get("candidateId") for row in stacking.get("candidateAggregate", [])])

    _check(checks, "validation_lineage", validation.get("baseRun", {}).get("runId") == base.get("runId"), base.get("runId"), validation.get("baseRun", {}).get("runId"))
    _check(checks, "validation_status", validation.get("status") == "validation_selection_candidate", "validation_selection_candidate", validation.get("status"))
    validation_data = validation.get("dataset", {})
    _check(checks, "full_train_refit", int(validation_data.get("trainRowsUsed", 0)) > 0 and validation_data.get("trainRowsUsed") == validation_data.get("trainRowsAvailable") and validation_data.get("demoTrainSubset") is False, validation_data.get("trainRowsAvailable"), validation_data.get("trainRowsUsed"))
    _check(checks, "full_validation", int(validation_data.get("validationRowsUsed", 0)) > 0 and validation_data.get("validationRowsUsed") == validation_data.get("validationRowsAvailable") and validation_data.get("demoValidationSubset") is False, validation_data.get("validationRowsAvailable"), validation_data.get("validationRowsUsed"))
    _check(checks, "validation_temporal_protocol", validation.get("validation", {}).get("chronologicalCalibrationBeforeSelection") is True and validation.get("calibration", {}).get("selectionLabelsUsed") is False, "calibracion anterior a seleccion sin etiquetas futuras", {"validation": validation.get("validation", {}), "calibration": validation.get("calibration", {})})
    _check(checks, "validation_test_lock", _test_locked(validation.get("validation", {})), "test bloqueado, no codificado y no usado", validation.get("validation", {}))
    _check(checks, "ready_for_single_test", validation.get("validation", {}).get("readyForFinalTestEvaluation") is True and validation.get("freeze", {}).get("eligibleForFinalTestEvaluation") is True, "seleccion real completa lista para una evaluacion final", validation.get("freeze", {}))
    comparison_ids = {str(row.get("candidateId")) for row in validation.get("comparison", [])}
    expected_candidates = {*FINANCE_MODEL_IDS, "stacking"}
    _check(checks, "six_candidates", comparison_ids == expected_candidates, sorted(expected_candidates), sorted(comparison_ids))
    calibrated_ids = {str(row.get("candidateId")) for row in validation.get("comparison", []) if row.get("calibratedThreshold") is not None}
    _check(checks, "six_frozen_thresholds", calibrated_ids == expected_candidates, sorted(expected_candidates), sorted(calibrated_ids))
    validation_refits = {(int(row.get("seed", -1)), str(row.get("modelId"))) for row in validation.get("training", {}).get("records", [])}
    expected_refits = {(seed, model_id) for seed in seeds for model_id in model_ids}
    _check(checks, "complete_validation_refits", validation_refits == expected_refits, len(expected_refits), len(validation_refits))

    _check(checks, "diversity_lineage", diversity.get("validationRun", {}).get("runId") == validation.get("runId"), validation.get("runId"), diversity.get("validationRun", {}).get("runId"))
    _check(checks, "diversity_status", diversity.get("status") == "ablation_candidate", "ablation_candidate", diversity.get("status"))
    _check(checks, "diversity_test_lock", _test_locked(diversity.get("validation", {})), "test bloqueado y no usado", diversity.get("validation", {}))
    expected_ablation_ids = {"full", *(f"without_{model_id}" for model_id in FINANCE_MODEL_IDS)}
    observed_ablation_ids = {str(row.get("ablationId")) for row in diversity.get("ablation", {}).get("configurations", [])}
    _check(checks, "complete_ablation_grid", observed_ablation_ids == expected_ablation_ids, sorted(expected_ablation_ids), sorted(observed_ablation_ids))
    expected_ablation_metrics = {(ablation_id, meta_model_id) for ablation_id in expected_ablation_ids for meta_model_id in META_MODEL_IDS}
    observed_ablation_metrics = {(str(row.get("ablationId")), str(row.get("metaModelId"))) for row in diversity.get("ablation", {}).get("metrics", [])}
    _check(checks, "complete_ablation_metrics", observed_ablation_metrics == expected_ablation_metrics, len(expected_ablation_metrics), len(observed_ablation_metrics))
    stability = diversity.get("stability", {})
    _check(checks, "seed_stability", int(stability.get("seedCount", 0)) >= 5 and stability.get("seedLevelInferenceAvailable") is True, "estabilidad con cinco semillas", stability)
    _check(checks, "temporal_bootstrap", stability.get("bootstrapUnit") == "calendar_day_temporal_block" and int(stability.get("bootstrapIterations", 0)) >= 300, "bootstrap por dia con >=300 iteraciones", stability)
    recommendation = diversity.get("recommendation", {})
    nested_validation = revalidation.get("validation", {})
    _check(checks, "revalidation_lineage", revalidation.get("diversityRun", {}).get("runId") == diversity.get("runId"), diversity.get("runId"), revalidation.get("diversityRun", {}).get("runId"))
    _check(checks, "revalidation_status", revalidation.get("status") == "optimized_stacking_revalidation_candidate", "optimized_stacking_revalidation_candidate", revalidation.get("status"))
    _check(checks, "independent_nested_comparison", nested_validation.get("independentComparisonLabelsUsedForCalibrationThresholdOrSelection") is False and revalidation.get("protocol", {}).get("sameRowsForAllCandidates") is True, "tercer bloque independiente sin uso para seleccion", nested_validation)
    _check(checks, "revalidation_test_lock", _test_locked(nested_validation), "test bloqueado y no usado", nested_validation)
    revalidated_ablation = revalidation.get("stackingSelection", {}).get("selectedAblationId")
    ablation_revalidated = not recommendation.get("ablationAccepted") or recommendation.get("recommendedAblationId") == revalidated_ablation
    _check(checks, "accepted_ablation_revalidated", ablation_revalidated, "la ablacion aceptada debe volver a competir como sexto candidato", {"recommendation": recommendation.get("recommendedAblationId"), "revalidated": revalidated_ablation})
    revalidated_candidates = {row.get("candidateId") for row in revalidation.get("independentComparison", {}).get("metrics", [])}
    _check(checks, "revalidated_six_candidates", revalidated_candidates == expected_candidates, sorted(expected_candidates), sorted(str(value) for value in revalidated_candidates))
    independent_seed_pairs = {
        (int(row.get("seed", -1)), str(row.get("candidateId")))
        for row in revalidation.get("independentComparison", {}).get("seedMetrics", [])
    }
    expected_seed_pairs = {(seed, candidate_id) for seed in seeds for candidate_id in expected_candidates}
    _check(checks, "independent_seed_stability", independent_seed_pairs == expected_seed_pairs, len(expected_seed_pairs), len(independent_seed_pairs))

    inventory = _build_artifact_inventory(lineage)
    _check(checks, "artifact_inventory", bool(inventory), "inventario de artefactos no vacio", len(inventory))
    structural_failures = [item for item in checks if not item["passed"]]
    if verify_artifacts and not structural_failures:
        failures = _verify_inventory(inventory)
        _check(checks, "artifact_integrity", not failures, "todos los SHA-256 y tamanos validos", failures or "verificado")
    elif verify_artifacts:
        _check(checks, "artifact_integrity", False, "superar primero los controles estructurales", "verificacion omitida")
    return _audit_result(checks, lineage, verify_artifacts, inventory)


def create_finance_freeze_package(
    *,
    validation_manifest_path: Path | None = None,
    diversity_manifest_path: Path | None = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    audit = audit_finance_freeze_readiness(
        validation_manifest_path=validation_manifest_path,
        diversity_manifest_path=diversity_manifest_path,
        verify_artifacts=True,
    )
    if not audit["ready"]:
        raise FinanceFreezeGateError(audit)
    lineage = _resolve_lineage(validation_manifest_path, diversity_manifest_path)
    base = lineage["baseManifest"]
    validation = lineage["validationManifest"]
    diversity = lineage["diversityManifest"]
    revalidation = lineage["revalidationManifest"]
    inventory = audit["artifactInventory"]
    source_fingerprint = _source_fingerprint(lineage, inventory)
    destination = output_dir or (Path(lineage["baseManifestPath"]).parent / "frozen_pipeline_v1")
    manifest_path = destination / "frozen_pipeline_manifest.json"
    seal_path = destination / "freeze_seal.json"
    if manifest_path.is_file():
        existing = read_json(manifest_path)
        if existing.get("sourceFingerprint") != source_fingerprint or not _verify_seal(manifest_path, seal_path):
            raise ValueError("El paquete financiero congelado existente no coincide o su sello es invalido.")
        return {"available": True, "reused": True, **existing, "sealVerified": True}
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("El directorio de congelamiento financiero contiene datos parciales.")
    destination.mkdir(parents=True, exist_ok=True)

    comparison = lineage["revalidationManifest"].get("independentComparison", {}).get("metrics", [])
    configuration = {
        "schemaVersion": "1.0.0",
        "domain": "finanzas",
        "datasetId": validation.get("dataset", {}).get("datasetId"),
        "datasetKind": validation.get("dataset", {}).get("kind"),
        "baseModels": list(FINANCE_MODEL_IDS),
        "seeds": base.get("configuration", {}).get("seeds", []),
        "oofFolds": base.get("validation", {}).get("oofFolds", []),
        "selectedCandidateId": revalidation.get("independentComparison", {}).get("winnerCandidateId"),
        "selectedStackingMetaModelId": revalidation.get("stackingSelection", {}).get("selectedMetaModelId"),
        "stackingAblationId": revalidation.get("stackingSelection", {}).get("selectedAblationId"),
        "calibratedThresholds": {row["candidateId"]: float(row["calibratedThreshold"]) for row in comparison},
        "calibrationMethods": {row["candidateId"]: row["calibrationMethod"] for row in comparison},
        "primaryMetric": "prAuc",
        "thresholdObjective": "f2",
        "finalTestPolicy": "single_evaluation_of_all_six_frozen_candidates_without_reselection",
    }
    configuration_path = destination / "frozen_configuration.json"
    inventory_path = destination / "artifact_inventory.json"
    write_json(configuration_path, configuration)
    write_json(inventory_path, {"items": inventory})
    configuration_hash, configuration_bytes = sha256_file(configuration_path)
    inventory_hash, inventory_bytes = sha256_file(inventory_path)
    freeze_id = f"finance-freeze-{source_fingerprint[:16]}"
    manifest = {
        "schemaVersion": "1.0.0",
        "freezeId": freeze_id,
        "domain": "finanzas",
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "status": "ready_for_single_final_test_evaluation",
        "immutable": True,
        "sourceFingerprint": source_fingerprint,
        "lineage": {name: lineage[name] for name in ("baseManifestPath", "stackingManifestPath", "validationManifestPath", "diversityManifestPath", "revalidationManifestPath", "sequenceManifestPath")},
        "gate": {"passed": True, "checks": audit["checks"], "artifactCount": len(inventory), "dependenciesVerifiedAtFreeze": True},
        "configuration": configuration,
        "testAuthorization": {"required": True, "granted": False, "evaluated": False, "maximumEvaluations": 1, "policy": "requires_explicit_separate_authorization_after_freeze"},
        "artifacts": {
            "configuration": {"path": str(configuration_path.resolve()), "sha256": configuration_hash, "bytes": configuration_bytes},
            "inventory": {"path": str(inventory_path.resolve()), "sha256": inventory_hash, "bytes": inventory_bytes, "items": len(inventory)},
        },
    }
    write_json(manifest_path, manifest)
    manifest_hash, manifest_bytes = sha256_file(manifest_path)
    write_json(seal_path, {"schemaVersion": "1.0.0", "freezeId": freeze_id, "manifestPath": str(manifest_path.resolve()), "manifestSha256": manifest_hash, "manifestBytes": manifest_bytes, "sealedAt": datetime.now(timezone.utc).isoformat(), "immutable": True})
    return {"available": True, "reused": False, **manifest, "sealVerified": _verify_seal(manifest_path, seal_path)}


def get_latest_finance_freeze() -> dict[str, Any]:
    paths = sorted(EXPERIMENTS_DIR.glob("*/finance_oof_v1/frozen_pipeline_v1/frozen_pipeline_manifest.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not paths:
        return {"available": False, "message": "Todavia no existe un pipeline financiero congelado."}
    path = paths[0]
    return {"available": True, **read_json(path), "sealVerified": _verify_seal(path, path.parent / "freeze_seal.json")}


def _resolve_lineage(validation_path: Path | None, diversity_path: Path | None) -> dict[str, Any]:
    resolved_validation = validation_path or _latest_validation_manifest()
    if resolved_validation is None or not resolved_validation.is_file():
        return {"error": "No existe una validacion temporal financiera."}
    validation = read_json(resolved_validation)
    base_path = Path(validation.get("baseRun", {}).get("manifestPath", ""))
    stacking_path = Path(validation.get("stackingRun", {}).get("manifestPath", ""))
    if not base_path.is_file() or not stacking_path.is_file():
        return {"error": "La validacion no referencia manifiestos OOF y Stacking validos."}
    resolved_diversity = diversity_path or (base_path.parent / "diversity_ablation_v1" / "diversity_ablation_manifest.json")
    if not resolved_diversity.is_file():
        return {"error": "Falta el estudio financiero de diversidad y ablacion."}
    revalidation_path = base_path.parent / "optimized_stacking_revalidation_v1" / "optimized_stacking_revalidation_manifest.json"
    if not revalidation_path.is_file():
        return {"error": "Falta la revalidacion independiente del Stacking optimizado."}
    base = read_json(base_path)
    sequence_path = Path(base.get("dataset", {}).get("sequencePath", "")).with_name("finance_sequence_manifest.json")
    if not sequence_path.is_file():
        return {"error": "Falta el manifiesto de secuencias financieras."}
    return {
        "baseManifestPath": str(base_path.resolve()),
        "stackingManifestPath": str(stacking_path.resolve()),
        "validationManifestPath": str(resolved_validation.resolve()),
        "diversityManifestPath": str(resolved_diversity.resolve()),
        "revalidationManifestPath": str(revalidation_path.resolve()),
        "sequenceManifestPath": str(sequence_path.resolve()),
        "baseManifest": base,
        "stackingManifest": read_json(stacking_path),
        "validationManifest": validation,
        "diversityManifest": read_json(resolved_diversity),
        "revalidationManifest": read_json(revalidation_path),
        "sequenceManifest": read_json(sequence_path),
    }


def _latest_validation_manifest() -> Path | None:
    paths = sorted(EXPERIMENTS_DIR.glob("*/finance_oof_v1/temporal_validation_v1/temporal_validation_manifest.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    return paths[0] if paths else None


def _test_locked(payload: dict[str, Any]) -> bool:
    return bool(payload.get("testSetLocked")) and payload.get("testSetEncoded") is False and payload.get("testSetUsed") is False


def _sequence_test_locked(sequence: dict[str, Any]) -> bool:
    lock = sequence.get("testLock", {})
    return lock.get("locked") is True and lock.get("encoded") is False and lock.get("evaluated") is False


def _build_artifact_inventory(lineage: dict[str, Any]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for role in ("base", "stacking", "validation", "diversity", "revalidation", "sequence"):
        manifest = lineage[f"{role}Manifest"]
        items.extend(_walk_artifacts(manifest.get("artifacts", {}), role))
        manifest_path = Path(lineage[f"{role}ManifestPath"])
        digest, size = sha256_file(manifest_path)
        items.append({"role": f"{role}:manifest", "path": str(manifest_path.resolve()), "sha256": digest, "bytes": size})
    sequence = lineage["sequenceManifest"]
    source = sequence.get("source", {})
    if {"path", "sha256", "bytes"}.issubset(source):
        items.append({"role": "sequence:source", "path": source["path"], "sha256": source["sha256"], "bytes": source["bytes"]})
    unique: dict[str, dict[str, Any]] = {}
    for item in items:
        path = str(Path(item["path"]).resolve())
        normalized = {**item, "path": path, "bytes": int(item["bytes"])}
        previous = unique.get(path)
        if previous and (previous["sha256"] != normalized["sha256"] or previous["bytes"] != normalized["bytes"]):
            normalized["inventoryConflict"] = True
        unique[path] = normalized
    return sorted(unique.values(), key=lambda item: (item["role"], item["path"]))


def _walk_artifacts(value: Any, role: str) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        if {"path", "sha256", "bytes"}.issubset(value) and value.get("path"):
            return [{"role": role, "path": str(value["path"]), "sha256": str(value["sha256"]), "bytes": int(value["bytes"])}]
        result: list[dict[str, Any]] = []
        for key, child in value.items():
            result.extend(_walk_artifacts(child, f"{role}:{key}"))
        return result
    if isinstance(value, list):
        result = []
        for index, child in enumerate(value):
            result.extend(_walk_artifacts(child, f"{role}:{index}"))
        return result
    return []


def _verify_inventory(inventory: list[dict[str, Any]]) -> list[dict[str, Any]]:
    failures = []
    for item in inventory:
        path = Path(item["path"])
        if item.get("inventoryConflict") or not path.is_file():
            failures.append({"role": item["role"], "path": item["path"], "reason": "conflict_or_missing"})
            continue
        digest, size = sha256_file(path)
        if digest != item["sha256"] or size != item["bytes"]:
            failures.append({"role": item["role"], "path": item["path"], "reason": "hash_or_size_mismatch"})
    return failures


def _source_fingerprint(lineage: dict[str, Any], inventory: list[dict[str, Any]]) -> str:
    payload = {
        "runIds": [lineage[name].get("runId") for name in ("baseManifest", "stackingManifest", "validationManifest", "diversityManifest", "revalidationManifest")],
        "artifacts": [(item["path"], item["sha256"], item["bytes"]) for item in inventory],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _verify_seal(manifest_path: Path, seal_path: Path) -> bool:
    if not manifest_path.is_file() or not seal_path.is_file():
        return False
    try:
        seal = read_json(seal_path)
        digest, size = sha256_file(manifest_path)
        return digest == seal.get("manifestSha256") and size == seal.get("manifestBytes") and seal.get("immutable") is True
    except (OSError, ValueError, json.JSONDecodeError):
        return False


def _check(checks: list[dict[str, Any]], check_id: str, passed: bool, expected: Any, observed: Any) -> None:
    checks.append({"id": check_id, "passed": bool(passed), "expected": expected, "observed": observed})


def _audit_result(checks: list[dict[str, Any]], lineage: dict[str, Any], verified: bool, inventory: list[dict[str, Any]]) -> dict[str, Any]:
    failed = [item for item in checks if not item["passed"]]
    return {
        "ready": not failed,
        "status": "ready_for_freeze_creation" if not failed else "blocked",
        "summary": {"passed": len(checks) - len(failed), "failed": len(failed), "total": len(checks)},
        "checks": checks,
        "blockingCheckIds": [item["id"] for item in failed],
        "artifactVerificationRequested": verified,
        "artifactInventory": inventory,
        "lineage": {key: value for key, value in lineage.items() if key.endswith("Path")},
    }
