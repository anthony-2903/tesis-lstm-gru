from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

from app.config import RESULTS_DIR
from app.finance.market_pipeline import FINANCE_MARKET_MODEL_IDS, _latest_base_manifest
from app.finance.mef_market_data import get_mef_market_dataset_status


class FinanceMarketFreezeGateError(ValueError):
    def __init__(self, audit: dict[str, Any]) -> None:
        self.audit = audit
        super().__init__("La corrida financiera MEF no supera la puerta de congelacion.")


def audit_finance_market_freeze_readiness(
    *,
    base_manifest_path: Path | None = None,
    revalidation_manifest_path: Path | None = None,
    verify_artifacts: bool = False,
) -> dict[str, Any]:
    base_path = base_manifest_path or _latest_base_manifest()
    revalidation_path = revalidation_manifest_path or (base_path.parent / "finance_market_revalidation_v1" / "revalidation_manifest.json")
    checks: list[dict[str, Any]] = []
    if not base_path.is_file() or not revalidation_path.is_file():
        _check(checks, "lineage_available", False, "base y revalidacion disponibles", {"base": str(base_path), "revalidation": str(revalidation_path)})
        return _result(checks, [], verify_artifacts, base_path, revalidation_path)

    base = json.loads(base_path.read_text(encoding="utf-8"))
    revalidation = json.loads(revalidation_path.read_text(encoding="utf-8"))
    data = get_mef_market_dataset_status()
    source = base.get("dataset", {}).get("sourceLineage", {})
    seeds = {int(value) for value in base.get("walkForward", {}).get("seeds", [])}
    folds = {int(row.get("foldId", -1)) for row in base.get("walkForward", {}).get("folds", [])}
    models = {str(value) for value in base.get("baseModels", [])}
    expected_grid = {(seed, fold, model) for seed in seeds for fold in folds for model in models}
    observed_grid = {
        (int(row.get("seed", -1)), int(row.get("fold", -1)), str(row.get("modelId")))
        for row in base.get("baseFoldMetrics", [])
    }

    _check(checks, "domain_and_status", base.get("domain") == "finanzas" and base.get("status") == "thesis_candidate", "finanzas/thesis_candidate", {"domain": base.get("domain"), "status": base.get("status")})
    _check(checks, "public_dataset_ready", bool(data.get("readyForThesisTraining")), True, data.get("readyForThesisTraining"))
    _check(checks, "dataset_lineage", source.get("verified") is True and source.get("datasetId") == data.get("datasetId") and source.get("sha256") == data.get("silver", {}).get("sha256"), "datasetId/hash MEF actuales", source)
    _check(checks, "five_seeds", len(seeds) >= 5, ">=5", sorted(seeds))
    _check(checks, "five_folds", len(folds) >= 5, ">=5", sorted(folds))
    _check(checks, "five_models", models == set(FINANCE_MARKET_MODEL_IDS), list(FINANCE_MARKET_MODEL_IDS), sorted(models))
    _check(checks, "complete_training_grid", observed_grid == expected_grid, len(expected_grid), len(observed_grid))
    _check(checks, "fair_comparison", base.get("comparisonScope", {}).get("fairComparison") is True, True, base.get("comparisonScope", {}))
    expected_pairs = len(seeds) * max(len(folds) - 1, 0)
    _check(checks, "complete_meta_validation", len(base.get("comparisonScope", {}).get("pairs", [])) == expected_pairs, expected_pairs, len(base.get("comparisonScope", {}).get("pairs", [])))
    _check(checks, "test_lock", base.get("walkForward", {}).get("testSetUsed") is False and base.get("aggregate", {}).get("testSetEvaluatedOnce") is False, "test no usado", base.get("walkForward", {}))
    _check(checks, "honest_anomaly_scope", base.get("anomalies", {}).get("labelType") == "estimated" and base.get("anomalies", {}).get("classificationMetricsAvailable") is False, "anomalias estimadas; sin afirmacion de fraude", base.get("anomalies", {}))

    protocol = revalidation.get("protocol", {})
    independent = revalidation.get("independentComparison", {})
    _check(checks, "revalidation_lineage", revalidation.get("baseRun", {}).get("runId") == base.get("runId"), base.get("runId"), revalidation.get("baseRun", {}).get("runId"))
    _check(checks, "independent_fold", protocol.get("selectionUsesIndependentFold") is False and protocol.get("independentComparisonFold") not in protocol.get("selectionFolds", []), "fold independiente", protocol)
    _check(checks, "revalidation_test_lock", protocol.get("testSetUsed") is False and protocol.get("testEvaluationAuthorized") is False, "test bloqueado", protocol)
    expected_ablations = {"full", *(f"without_{model}" for model in FINANCE_MARKET_MODEL_IDS)}
    observed_ablations = {str(row.get("ablationId")) for row in revalidation.get("ablation", {}).get("configurations", [])}
    _check(checks, "complete_ablation", observed_ablations == expected_ablations, sorted(expected_ablations), sorted(observed_ablations))
    ranking = set(independent.get("ranking", []))
    _check(checks, "six_candidate_table", ranking == {*FINANCE_MARKET_MODEL_IDS, "optimized_stacking"}, sorted({*FINANCE_MARKET_MODEL_IDS, "optimized_stacking"}), sorted(ranking))
    _check(checks, "paired_bootstrap", independent.get("pairedInference", {}).get("iterations", 0) >= 300, ">=300", independent.get("pairedInference", {}))
    _check(checks, "xai_available", bool(revalidation.get("xai", {}).get("items")), True, revalidation.get("xai", {}))

    inventory = _inventory(base, revalidation)
    _check(checks, "artifact_inventory", bool(inventory), "inventario no vacio", len(inventory))
    if verify_artifacts:
        failures = _verify_inventory(inventory)
        _check(checks, "artifact_integrity", not failures, "hashes y tamanos validos", failures)
    return _result(checks, inventory, verify_artifacts, base_path, revalidation_path)


def create_finance_market_freeze_package(
    *,
    base_manifest_path: Path | None = None,
    revalidation_manifest_path: Path | None = None,
) -> dict[str, Any]:
    audit = audit_finance_market_freeze_readiness(
        base_manifest_path=base_manifest_path,
        revalidation_manifest_path=revalidation_manifest_path,
        verify_artifacts=True,
    )
    if not audit["ready"]:
        raise FinanceMarketFreezeGateError(audit)
    freeze_id = f"finance_mef_freeze_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    destination = RESULTS_DIR / "finance_market_freezes" / freeze_id
    destination.mkdir(parents=True, exist_ok=False)
    configuration_path = destination / "frozen_configuration.json"
    inventory_path = destination / "artifact_inventory.json"
    configuration = {
        "domain": "finanzas",
        "source": "MEF Indices Soberanos",
        "task": "next_session_nominal_log_return_forecasting_with_residual_anomaly_detection",
        "baseManifestPath": audit["baseManifestPath"],
        "revalidationManifestPath": audit["revalidationManifestPath"],
        "primaryMetric": "rmse",
        "lowerIsBetter": True,
        "testSetLocked": True,
        "testEvaluationAuthorized": False,
        "anomalyLabels": "estimated from out-of-sample residuals; fraud claims prohibited",
    }
    configuration_path.write_text(json.dumps(configuration, ensure_ascii=False, indent=2), encoding="utf-8")
    inventory_path.write_text(json.dumps({"items": audit["artifactInventory"]}, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = {
        "schemaVersion": "1.0.0",
        "freezeId": freeze_id,
        "domain": "finanzas",
        "status": "frozen_pre_test",
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "sourceFingerprint": _source_fingerprint(audit),
        "testPolicy": {"testSetLocked": True, "testSetUsed": False, "testEvaluationAuthorized": False},
        "audit": {"totalChecks": audit["totalChecks"], "passedChecks": audit["passedChecks"]},
        "artifacts": {"configuration": _artifact(configuration_path), "inventory": _artifact(inventory_path)},
    }
    manifest_path = destination / "freeze_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    seal_path = destination / "freeze_seal.json"
    descriptor = _artifact(manifest_path)
    seal_path.write_text(json.dumps({"freezeId": freeze_id, "manifestSha256": descriptor["sha256"], "manifestBytes": descriptor["bytes"]}, ensure_ascii=False, indent=2), encoding="utf-8")
    return {**manifest, "manifest": descriptor, "seal": _artifact(seal_path)}


def get_latest_finance_market_freeze() -> dict[str, Any]:
    paths = sorted((RESULTS_DIR / "finance_market_freezes").glob("*/freeze_manifest.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not paths:
        return {"available": False, "message": "No existe congelamiento financiero MEF."}
    return {"available": True, **json.loads(paths[0].read_text(encoding="utf-8"))}


def _inventory(base: dict[str, Any], revalidation: dict[str, Any]) -> list[dict[str, Any]]:
    items = [{"role": f"base:{key}", **value} for key, value in base.get("artifactIntegrity", {}).items()]
    for key, value in revalidation.get("artifacts", {}).items():
        values = value if isinstance(value, list) else [value]
        for artifact in values:
            if isinstance(artifact, dict) and {"path", "sha256", "bytes"}.issubset(artifact):
                items.append({"role": f"revalidation:{key}", **artifact})
    source = base.get("dataset", {}).get("sourceLineage", {})
    if {"path", "sha256", "bytes"}.issubset(source):
        items.append({"role": "dataset:silver", "path": source["path"], "sha256": source["sha256"], "bytes": source["bytes"]})
    unique = {(str(item["path"]), str(item["sha256"])): item for item in items}
    return sorted(unique.values(), key=lambda row: (row["role"], row["path"]))


def _verify_inventory(items: list[dict[str, Any]]) -> list[dict[str, str]]:
    failures = []
    for item in items:
        path = Path(item["path"])
        if not path.is_file():
            failures.append({"path": str(path), "reason": "missing"})
            continue
        actual = _artifact(path)
        if actual["sha256"] != item.get("sha256") or actual["bytes"] != item.get("bytes"):
            failures.append({"path": str(path), "reason": "hash_or_size_mismatch"})
    return failures


def _source_fingerprint(audit: dict[str, Any]) -> str:
    payload = {
        "base": _artifact(Path(audit["baseManifestPath"])),
        "revalidation": _artifact(Path(audit["revalidationManifestPath"])),
        "inventory": [{"path": row["path"], "sha256": row["sha256"], "bytes": row["bytes"]} for row in audit["artifactInventory"]],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _artifact(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return {"path": str(path.resolve()), "sha256": digest.hexdigest(), "bytes": size}


def _check(checks: list[dict[str, Any]], check_id: str, passed: bool, expected: Any, observed: Any) -> None:
    checks.append({"checkId": check_id, "passed": bool(passed), "expected": expected, "observed": observed})


def _result(checks: list[dict[str, Any]], inventory: list[dict[str, Any]], verified: bool, base_path: Path, revalidation_path: Path) -> dict[str, Any]:
    failed = [row for row in checks if not row["passed"]]
    return {
        "ready": not failed,
        "totalChecks": len(checks),
        "passedChecks": len(checks) - len(failed),
        "failedChecks": failed,
        "blockingCheckIds": [row["checkId"] for row in failed],
        "artifactIntegrityVerified": verified,
        "artifactInventory": inventory,
        "baseManifestPath": str(base_path.resolve()),
        "revalidationManifestPath": str(revalidation_path.resolve()),
        "testSetUsed": False,
        "testEvaluationAuthorized": False,
    }
