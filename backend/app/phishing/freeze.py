from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

from app.config import EXPERIMENTS_DIR
from app.phishing.ingestion import sha256_file
from app.phishing.models import PHISHING_MODEL_IDS
from app.phishing.stacking import ENSEMBLE_IDS, META_MODEL_IDS
from app.utils import read_json, write_json


class FreezeGateError(ValueError):
    def __init__(self, audit: dict[str, Any]) -> None:
        self.audit = audit
        super().__init__("La corrida no supera la puerta de congelación experimental.")


def audit_phishing_freeze_readiness(
    *,
    validation_manifest_path: Path | None = None,
    diversity_manifest_path: Path | None = None,
    verify_artifacts: bool = False,
) -> dict[str, Any]:
    lineage = _resolve_lineage(validation_manifest_path, diversity_manifest_path)
    checks: list[dict[str, Any]] = []
    if lineage.get("error"):
        _check(checks, "lineage_available", False, "Linaje OOF → validation → ablación disponible", lineage["error"])
        return _audit_result(checks, lineage, verify_artifacts, [])

    base = lineage["baseManifest"]
    validation = lineage["validationManifest"]
    diversity = lineage["diversityManifest"]
    model_ids = tuple(base.get("baseModels", []))
    seeds = tuple(int(value) for value in base.get("configuration", {}).get("seeds", []))
    folds = tuple(int(value) for value in base.get("validation", {}).get("oofFolds", []))
    rows_used = int(base.get("dataset", {}).get("rowsUsed", 0))
    rows_available = int(base.get("dataset", {}).get("outerTrainRowsAvailable", 0))

    _check(checks, "base_status", base.get("status") == "thesis_base_models_candidate", "thesis_base_models_candidate", base.get("status"))
    _check(checks, "base_protocol", base.get("protocol") == "thesis", "thesis", base.get("protocol"))
    _check(checks, "complete_development", rows_used > 0 and rows_used == rows_available and not base.get("dataset", {}).get("demoSubset", True), "todas las filas de desarrollo", f"{rows_used}/{rows_available}")
    _check(checks, "five_seeds", len(set(seeds)) >= 5, "al menos 5 semillas", list(seeds))
    _check(checks, "five_models", set(model_ids) == set(PHISHING_MODEL_IDS), list(PHISHING_MODEL_IDS), list(model_ids))
    _check(checks, "five_oof_folds", len(set(folds)) >= 5, "al menos 5 folds", list(folds))

    dataset_readiness = _dataset_readiness(base)
    _check(
        checks,
        "dataset_scientific_readiness",
        bool(dataset_readiness.get("readyForThesisTraining")),
        "readyForThesisTraining=true",
        dataset_readiness,
    )
    _check(
        checks,
        "dataset_lineage_current",
        bool(dataset_readiness.get("sourceHashMatches")),
        "el hash científico actual debe coincidir con la corrida OOF",
        dataset_readiness,
    )

    expected_base_combinations = {(seed, fold, model_id) for seed in seeds for fold in folds for model_id in model_ids}
    observed_base_combinations = {
        (int(row.get("seed", -1)), int(row.get("fold", -1)), str(row.get("modelId")))
        for row in base.get("baseFoldMetrics", [])
    }
    _check(checks, "complete_oof_training_grid", observed_base_combinations == expected_base_combinations, len(expected_base_combinations), len(observed_base_combinations))
    expected_fold_protocols = {(seed, fold) for seed in seeds for fold in folds}
    observed_fold_protocols = {
        (int(row.get("seed", -1)), int(row.get("fold", -1)))
        for row in base.get("validation", {}).get("foldProtocols", [])
        if {"path", "sha256", "bytes"}.issubset(row.get("tokenizer", {}))
    }
    _check(checks, "complete_oof_tokenizers", observed_fold_protocols == expected_fold_protocols, len(expected_fold_protocols), len(observed_fold_protocols))
    oof_rows = int(base.get("artifacts", {}).get("oofProbabilities", {}).get("rows", 0))
    _check(checks, "complete_oof_coverage", oof_rows == rows_used * len(seeds) and oof_rows > 0, rows_used * len(seeds), oof_rows)
    _check(checks, "base_test_lock", _test_locked(base.get("validation", {})), "test bloqueado y no usado", base.get("validation", {}))

    _check(checks, "validation_lineage", validation.get("baseRun", {}).get("runId") == base.get("runId"), base.get("runId"), validation.get("baseRun", {}).get("runId"))
    _check(checks, "validation_status", validation.get("status") == "validation_selection_candidate", "validation_selection_candidate", validation.get("status"))
    _check(checks, "validation_protocol", bool(validation.get("validation", {}).get("outerValidationUsed")) and bool(validation.get("validation", {}).get("sameRowsForAllCandidates")), "validation externa común", validation.get("validation", {}))
    _check(checks, "validation_meta_fit", validation.get("metaFeatures", {}).get("validationLabelsUsedForMetaFit") is False, "validationLabelsUsedForMetaFit=false", validation.get("metaFeatures", {}).get("validationLabelsUsedForMetaFit"))
    _check(checks, "validation_test_lock", _test_locked(validation.get("validation", {})), "test bloqueado, no codificado y no usado", validation.get("validation", {}))

    expected_training = {(seed, model_id) for seed in seeds for model_id in model_ids}
    observed_training = {(int(row.get("seed", -1)), str(row.get("modelId"))) for row in validation.get("training", {}).get("records", [])}
    _check(checks, "complete_final_refit_grid", observed_training == expected_training, len(expected_training), len(observed_training))
    required_candidates = set(model_ids) | set(ENSEMBLE_IDS)
    observed_candidates = {str(row.get("candidateId")) for row in validation.get("comparison", [])}
    _check(checks, "all_candidates_frozen", observed_candidates == required_candidates, sorted(required_candidates), sorted(observed_candidates))
    threshold_candidates = {str(row.get("candidateId")) for row in validation.get("comparison", []) if row.get("calibratedThreshold") is not None}
    _check(checks, "all_thresholds_calibrated", threshold_candidates == required_candidates, sorted(required_candidates), sorted(threshold_candidates))

    validation_objects = validation.get("artifacts", {}).get("fittedObjects", [])
    expected_validation_objects = len(seeds) * (len(model_ids) + len(META_MODEL_IDS) + 1)
    _check(checks, "validation_fitted_objects", len(validation_objects) == expected_validation_objects, expected_validation_objects, len(validation_objects))

    _check(checks, "diversity_lineage", diversity.get("validationRun", {}).get("runId") == validation.get("runId"), validation.get("runId"), diversity.get("validationRun", {}).get("runId"))
    _check(checks, "diversity_status", diversity.get("status") == "ablation_candidate", "ablation_candidate", diversity.get("status"))
    _check(checks, "diversity_test_lock", _test_locked(diversity.get("validation", {})), "test bloqueado, no codificado y no usado", diversity.get("validation", {}))
    expected_ablation_ids = {"full", *(f"without_{model_id}" for model_id in model_ids)}
    observed_ablation_ids = {str(row.get("ablationId")) for row in diversity.get("ablation", {}).get("configurations", [])}
    _check(checks, "complete_ablation_grid", observed_ablation_ids == expected_ablation_ids, sorted(expected_ablation_ids), sorted(observed_ablation_ids))
    expected_ablation_metrics = {(ablation_id, meta_model_id) for ablation_id in expected_ablation_ids for meta_model_id in META_MODEL_IDS}
    observed_ablation_metrics = {(str(row.get("ablationId")), str(row.get("metaModelId"))) for row in diversity.get("ablation", {}).get("metrics", [])}
    _check(checks, "complete_ablation_metrics", observed_ablation_metrics == expected_ablation_metrics, len(expected_ablation_metrics), len(observed_ablation_metrics))
    expected_ablation_objects = len(seeds) * len(expected_ablation_ids) * len(META_MODEL_IDS)
    diversity_objects = diversity.get("artifacts", {}).get("fittedObjects", [])
    _check(checks, "ablation_fitted_objects", len(diversity_objects) == expected_ablation_objects, expected_ablation_objects, len(diversity_objects))
    stability = diversity.get("stability", {})
    _check(checks, "seed_stability", int(stability.get("seedCount", 0)) >= 5 and bool(stability.get("seedLevelInferenceAvailable")), "estabilidad con >=5 semillas", stability)
    _check(checks, "domain_bootstrap", stability.get("bootstrapUnit") == "registrable_domain" and int(stability.get("bootstrapIterations", 0)) >= 300, "bootstrap por dominio con >=300 iteraciones", stability)

    inventory = _build_artifact_inventory(lineage)
    _check(checks, "artifact_inventory_complete", bool(inventory), "inventario no vacío", len(inventory))
    structural_failures = [check for check in checks if not check["passed"]]
    if verify_artifacts and not structural_failures:
        failures = _verify_inventory(inventory)
        _check(checks, "artifact_integrity", not failures, "todos los hashes y tamaños válidos", failures or "verificado")
    elif verify_artifacts:
        _check(checks, "artifact_integrity", False, "superar primero los controles estructurales", "verificación omitida")
    return _audit_result(checks, lineage, verify_artifacts, inventory)


def create_phishing_freeze_package(
    *,
    validation_manifest_path: Path | None = None,
    diversity_manifest_path: Path | None = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    audit = audit_phishing_freeze_readiness(
        validation_manifest_path=validation_manifest_path,
        diversity_manifest_path=diversity_manifest_path,
        verify_artifacts=True,
    )
    if not audit["ready"]:
        raise FreezeGateError(audit)
    lineage = _resolve_lineage(validation_manifest_path, diversity_manifest_path)
    base = lineage["baseManifest"]
    validation = lineage["validationManifest"]
    diversity = lineage["diversityManifest"]
    inventory = audit["artifactInventory"]
    source_fingerprint = _source_fingerprint(lineage, inventory)
    destination = output_dir or (Path(lineage["baseManifestPath"]).parent / "frozen_pipeline_v1")
    manifest_path = destination / "frozen_pipeline_manifest.json"
    seal_path = destination / "freeze_seal.json"
    if manifest_path.exists():
        existing = read_json(manifest_path)
        if existing.get("sourceFingerprint") != source_fingerprint:
            raise ValueError("Ya existe un paquete congelado diferente; no se permite sobrescribirlo.")
        if not _verify_seal(manifest_path, seal_path):
            raise ValueError("El paquete congelado existente no supera la verificación del sello.")
        return {"available": True, "reused": True, **existing, "sealVerified": True}
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("El directorio de congelación contiene datos parciales; no se permite sobrescribirlos.")
    destination.mkdir(parents=True, exist_ok=True)

    comparison = validation.get("comparison", [])
    thresholds = {row["candidateId"]: float(row["calibratedThreshold"]) for row in comparison}
    selected_epochs = {
        f"seed_{int(row['seed'])}:{row['modelId']}": int(row["selectedEpochsFromOof"])
        for row in validation.get("training", {}).get("records", [])
    }
    configuration = {
        "schemaVersion": "1.0.0",
        "domain": "phishing",
        "dataset": {
            "sourcePath": base.get("dataset", {}).get("sourcePath"),
            "sourceSha256": base.get("dataset", {}).get("sourceSha256"),
            "developmentRows": base.get("dataset", {}).get("rowsUsed"),
        },
        "baseModels": diversity.get("recommendation", {}).get("recommendedBaseModels", base.get("baseModels", [])),
        "allComparedBaseModels": base.get("baseModels", []),
        "seeds": base.get("configuration", {}).get("seeds", []),
        "oofFolds": base.get("validation", {}).get("oofFolds", []),
        "selectedEpochsBySeedAndModel": selected_epochs,
        "tokenizer": validation.get("tokenizer", {}),
        "metaFeatures": validation.get("metaFeatures", {}).get("columns", []),
        "stackingCandidateId": diversity.get("recommendation", {}).get("referenceMetaModelId"),
        "validationPrimaryCandidateId": validation.get("selection", {}).get("winnerCandidateId"),
        "frozenCandidateIds": [row["candidateId"] for row in comparison],
        "calibratedThresholds": thresholds,
        "thresholdObjective": validation.get("selection", {}).get("thresholdObjective"),
        "ablationDecision": diversity.get("recommendation", {}),
        "finalTestPolicy": "single_evaluation_of_all_frozen_candidates_no_reselection",
    }
    configuration_path = destination / "frozen_configuration.json"
    inventory_path = destination / "artifact_inventory.json"
    write_json(configuration_path, configuration)
    write_json(inventory_path, {"items": inventory})
    configuration_hash, configuration_bytes = sha256_file(configuration_path)
    inventory_hash, inventory_bytes = sha256_file(inventory_path)
    freeze_id = f"phishing-freeze-{source_fingerprint[:16]}"
    manifest = {
        "schemaVersion": "1.0.0",
        "freezeId": freeze_id,
        "domain": "phishing",
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "status": "ready_for_single_test_evaluation",
        "immutable": True,
        "sourceFingerprint": source_fingerprint,
        "lineage": {
            "baseRunId": base.get("runId"),
            "validationRunId": validation.get("runId"),
            "diversityRunId": diversity.get("runId"),
            "baseManifestPath": lineage["baseManifestPath"],
            "validationManifestPath": lineage["validationManifestPath"],
            "diversityManifestPath": lineage["diversityManifestPath"],
        },
        "gate": {"passed": True, "checks": audit["checks"], "artifactCount": len(inventory), "dependenciesVerifiedAtFreeze": True},
        "configuration": configuration,
        "lockedFields": [
            "dataset.sourceSha256",
            "baseModels",
            "seeds",
            "oofFolds",
            "selectedEpochsBySeedAndModel",
            "tokenizer",
            "metaFeatures",
            "stackingCandidateId",
            "frozenCandidateIds",
            "calibratedThresholds",
        ],
        "testAuthorization": {
            "required": True,
            "granted": False,
            "evaluated": False,
            "maximumEvaluations": 1,
            "policy": "requires_explicit_separate_authorization_after_freeze",
        },
        "artifacts": {
            "configuration": {"path": str(configuration_path.resolve()), "sha256": configuration_hash, "bytes": configuration_bytes},
            "inventory": {"path": str(inventory_path.resolve()), "sha256": inventory_hash, "bytes": inventory_bytes, "items": len(inventory)},
        },
    }
    write_json(manifest_path, manifest)
    manifest_hash, manifest_bytes = sha256_file(manifest_path)
    write_json(seal_path, {
        "schemaVersion": "1.0.0",
        "freezeId": freeze_id,
        "manifestPath": str(manifest_path.resolve()),
        "manifestSha256": manifest_hash,
        "manifestBytes": manifest_bytes,
        "sealedAt": datetime.now(timezone.utc).isoformat(),
        "immutable": True,
    })
    return {"available": True, "reused": False, **manifest, "sealVerified": _verify_seal(manifest_path, seal_path)}


def get_latest_phishing_freeze() -> dict[str, Any]:
    paths = sorted(
        EXPERIMENTS_DIR.glob("*/phishing_oof_v1/frozen_pipeline_v1/frozen_pipeline_manifest.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not paths:
        return {"available": False, "message": "Todavía no existe un pipeline de phishing congelado."}
    manifest_path = paths[0]
    manifest = read_json(manifest_path)
    return {"available": True, **manifest, "sealVerified": _verify_seal(manifest_path, manifest_path.parent / "freeze_seal.json")}


def _resolve_lineage(validation_path: Path | None, diversity_path: Path | None) -> dict[str, Any]:
    resolved_validation = validation_path or _latest_validation_manifest()
    if resolved_validation is None or not resolved_validation.is_file():
        return {"error": "No existe manifiesto de validación externa."}
    validation = read_json(resolved_validation)
    base_path = Path(validation.get("baseRun", {}).get("manifestPath", ""))
    if not base_path.is_file():
        return {"error": "La validación no referencia un manifiesto OOF válido."}
    resolved_diversity = diversity_path or (base_path.parent / "diversity_ablation_v1" / "diversity_ablation_manifest.json")
    if not resolved_diversity.is_file():
        return {"error": "No existe manifiesto de diversidad y ablación para la corrida."}
    return {
        "baseManifestPath": str(base_path.resolve()),
        "validationManifestPath": str(resolved_validation.resolve()),
        "diversityManifestPath": str(resolved_diversity.resolve()),
        "baseManifest": read_json(base_path),
        "validationManifest": validation,
        "diversityManifest": read_json(resolved_diversity),
    }


def _dataset_readiness(base: dict[str, Any]) -> dict[str, Any]:
    configured = base.get("dataset", {}).get("metadataPath")
    source = Path(base.get("dataset", {}).get("sourcePath", ""))
    metadata_path = Path(configured) if configured else source.with_suffix(".metadata.json")
    if not metadata_path.is_file():
        return {"readyForThesisTraining": False, "reason": "No existe metadata científica del dataset."}
    metadata = read_json(metadata_path)
    expected_hash = base.get("dataset", {}).get("sourceSha256")
    current_hash = metadata.get("silver", {}).get("sha256")
    return {
        "readyForThesisTraining": bool(metadata.get("readyForThesisTraining")),
        "sourceHashMatches": bool(expected_hash) and current_hash == expected_hash,
        "expectedSourceSha256": expected_hash,
        "currentSourceSha256": current_hash,
        "metadataPath": str(metadata_path.resolve()),
        "datasetId": metadata.get("datasetId"),
    }


def _test_locked(payload: dict[str, Any]) -> bool:
    return (
        bool(payload.get("testSetLocked", True))
        and payload.get("testSetUsed") is False
        and payload.get("testFeaturesEncoded", False) is False
    )


def _build_artifact_inventory(lineage: dict[str, Any]) -> list[dict[str, Any]]:
    inventory: list[dict[str, Any]] = []
    base = lineage["baseManifest"]
    validation = lineage["validationManifest"]
    diversity = lineage["diversityManifest"]
    source = Path(base.get("dataset", {}).get("sourcePath", ""))
    if source.is_file():
        inventory.append({"role": "dataset_silver", "path": str(source.resolve()), "sha256": base.get("dataset", {}).get("sourceSha256"), "bytes": base.get("dataset", {}).get("sourceBytes"), "hashMode": "file"})
    metadata_path = Path(base.get("dataset", {}).get("metadataPath", "")) if base.get("dataset", {}).get("metadataPath") else source.with_suffix(".metadata.json")
    if metadata_path.is_file():
        digest, size = sha256_file(metadata_path)
        inventory.append({"role": "dataset_metadata", "path": str(metadata_path.resolve()), "sha256": digest, "bytes": size, "hashMode": "file"})
    for role, manifest in (("base", base), ("validation", validation), ("diversity", diversity)):
        inventory.extend(_walk_artifacts(manifest.get("artifacts", {}), role))
    for protocol in base.get("validation", {}).get("foldProtocols", []):
        inventory.extend(_walk_artifacts(protocol.get("tokenizer", {}), "base:oof_tokenizer"))
    inventory.extend(_walk_artifacts(validation.get("tokenizer", {}), "validation:tokenizer"))
    for row in base.get("baseFoldMetrics", []):
        if row.get("modelPath"):
            inventory.append({"role": "base_oof_model", "path": str(Path(row["modelPath"]).resolve()), "sha256": row.get("modelSha256"), "bytes": row.get("modelSizeBytes"), "hashMode": "keras_container"})
    deduplicated: dict[tuple[str, str], dict[str, Any]] = {}
    for item in inventory:
        deduplicated[(str(item.get("path")), str(item.get("sha256")))] = item
    return sorted(deduplicated.values(), key=lambda item: (item["role"], item["path"]))


def _walk_artifacts(value: Any, role_prefix: str) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    if isinstance(value, dict):
        if {"path", "sha256", "bytes"}.issubset(value):
            kind = value.get("kind")
            result.append({
                "role": f"{role_prefix}:{kind or 'artifact'}",
                "path": str(Path(value["path"]).resolve()),
                "sha256": value["sha256"],
                "bytes": value["bytes"],
                "hashMode": "keras_container" if kind == "base_model" else "file",
            })
        for nested in value.values():
            result.extend(_walk_artifacts(nested, role_prefix))
    elif isinstance(value, list):
        for nested in value:
            result.extend(_walk_artifacts(nested, role_prefix))
    return result


def _verify_inventory(inventory: list[dict[str, Any]]) -> list[dict[str, Any]]:
    failures = []
    for item in inventory:
        path = Path(item["path"])
        if not path.exists():
            failures.append({"path": item["path"], "reason": "missing"})
            continue
        digest, size = _keras_container_hash(path) if item["hashMode"] == "keras_container" else sha256_file(path)
        if digest != item.get("sha256") or size != item.get("bytes"):
            failures.append({"path": item["path"], "reason": "hash_or_size_mismatch"})
    return failures


def _keras_container_hash(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    total = 0
    files = sorted(item for item in path.rglob("*") if item.is_file()) if path.is_dir() else [path]
    for item in files:
        relative = item.relative_to(path).as_posix() if path.is_dir() else item.name
        digest.update(relative.encode("utf-8"))
        with item.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
                total += len(chunk)
    return digest.hexdigest(), total


def _source_fingerprint(lineage: dict[str, Any], inventory: list[dict[str, Any]]) -> str:
    manifest_hashes = {}
    for key in ("baseManifestPath", "validationManifestPath", "diversityManifestPath"):
        digest, size = sha256_file(Path(lineage[key]))
        manifest_hashes[key] = {"sha256": digest, "bytes": size}
    payload = {
        "manifests": manifest_hashes,
        "artifacts": [{"path": item["path"], "sha256": item["sha256"], "bytes": item["bytes"]} for item in inventory],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _verify_seal(manifest_path: Path, seal_path: Path) -> bool:
    if not manifest_path.is_file() or not seal_path.is_file():
        return False
    seal = read_json(seal_path)
    digest, size = sha256_file(manifest_path)
    return digest == seal.get("manifestSha256") and size == seal.get("manifestBytes")


def _check(checks: list[dict[str, Any]], check_id: str, passed: bool, expected: Any, observed: Any) -> None:
    checks.append({"checkId": check_id, "passed": bool(passed), "expected": expected, "observed": observed})


def _audit_result(
    checks: list[dict[str, Any]],
    lineage: dict[str, Any],
    verified: bool,
    inventory: list[dict[str, Any]],
) -> dict[str, Any]:
    failed = [check for check in checks if not check["passed"]]
    return {
        "ready": not failed,
        "structurallyReady": not [check for check in failed if check["checkId"] != "artifact_integrity"],
        "artifactIntegrityVerified": verified and not any(check["checkId"] == "artifact_integrity" and not check["passed"] for check in checks),
        "integrityVerificationRequiredAtFreeze": not verified,
        "status": "ready_for_freeze_creation" if not failed else "blocked",
        "checks": checks,
        "failedChecks": failed,
        "reasons": [f"{check['checkId']}: esperado {check['expected']}; observado {check['observed']}" for check in failed],
        "lineage": {key: value for key, value in lineage.items() if key.endswith("Path") or key == "error"},
        "artifactInventory": inventory,
        "requirements": {
            "minimumSeeds": 5,
            "minimumFolds": 5,
            "requiredModels": list(PHISHING_MODEL_IDS),
            "completeDevelopmentRequired": True,
            "datasetScientificReadinessRequired": True,
            "testMustRemainLocked": True,
        },
    }


def _latest_validation_manifest() -> Path | None:
    paths = sorted(
        EXPERIMENTS_DIR.glob("*/phishing_oof_v1/external_validation_v1/external_validation_manifest.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return paths[0] if paths else None
