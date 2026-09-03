from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.config import EXPERIMENTS_DIR, RESULTS_DIR
from app.utils import read_json

SCIENTIFIC_SNAPSHOT_PATH = Path(__file__).resolve().parents[1] / "config" / "scientific_corpus_snapshot.json"


def get_scientific_data_summary(
    *,
    results_dir: Path = RESULTS_DIR,
    experiments_dir: Path = EXPERIMENTS_DIR,
    snapshot_path: Path | None = SCIENTIFIC_SNAPSHOT_PATH,
) -> dict[str, Any]:
    """Summarize audited thesis datasets without mixing in demo/data-lake rows."""

    live_domains = [
        _phishing_summary(results_dir, experiments_dir),
        _energy_summary(results_dir, experiments_dir),
        _finance_summary(results_dir, experiments_dir),
    ]
    snapshot = _optional_json(snapshot_path) if snapshot_path else None
    snapshot_domains = {
        str(domain.get("id")): domain
        for domain in (snapshot or {}).get("domains", [])
        if isinstance(domain, dict) and domain.get("available") is True
    }
    domains: list[dict[str, Any]] = []
    origins: set[str] = set()
    for live_domain in live_domains:
        if live_domain["available"]:
            domain = {**live_domain, "summaryOrigin": "live_manifest"}
        elif live_domain["id"] in snapshot_domains:
            domain = {**snapshot_domains[live_domain["id"]], "summaryOrigin": "versioned_snapshot"}
        else:
            domain = {**live_domain, "summaryOrigin": "unavailable"}
        origins.add(str(domain["summaryOrigin"]))
        domains.append(domain)

    available_domains = [domain for domain in domains if domain["available"]]
    if origins == {"live_manifest"}:
        data_origin = "live_manifests"
    elif origins <= {"versioned_snapshot"}:
        data_origin = "versioned_snapshot"
    elif "unavailable" in origins and not available_domains:
        data_origin = "unavailable"
    else:
        data_origin = "mixed"
    return {
        "schemaVersion": "1.0.0",
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "dataOrigin": data_origin,
        "snapshotId": (snapshot or {}).get("snapshotId") if "versioned_snapshot" in origins else None,
        "available": len(available_domains) == len(domains),
        "availableDomains": len(available_domains),
        "totalDomains": len(domains),
        "totalUsableObservations": sum(int(domain["usableRows"]) for domain in available_domains),
        "domains": domains,
        "countingPolicy": (
            "Only unique, audited observations from each scientific silver dataset are totaled. "
            "OOF repetitions across seeds, folds and models are not counted as new observations."
        ),
        "demoExcluded": True,
        "finalTestUsed": any(bool(domain.get("testSetUsed")) for domain in available_domains),
    }


def _phishing_summary(results_dir: Path, experiments_dir: Path) -> dict[str, Any]:
    audit = _optional_json(results_dir / "phishing_data_audit.json")
    if not audit:
        return _unavailable("phishing", "Phishing", "URL")

    distribution = audit.get("classDistribution", {})
    splits = audit.get("splitDistribution", {})
    readiness = audit.get("readiness", {})
    source_rows = sum(
        int(source.get("originalRows", 0))
        for source in audit.get("source", {}).values()
        if isinstance(source, dict)
    )
    train_rows = int(splits.get("train", {}).get("rows", 0))
    validation_rows = int(splits.get("validation", {}).get("rows", 0))
    test_rows = int(splits.get("test", {}).get("rows", 0))
    manifest = _latest_manifest(experiments_dir, "phishing_oof_v1", domain="phishing")
    oof_rows = int(manifest.get("dataset", {}).get("rowsUsed", train_rows)) if manifest else train_rows

    return {
        "id": "phishing",
        "label": "Phishing",
        "unit": "URL",
        "available": True,
        "datasetId": audit.get("datasetId"),
        "source": "PhishTank + fuentes académicas verificadas",
        "originalRows": source_rows,
        "usableRows": int(distribution.get("total", 0)),
        "developmentRows": train_rows + validation_rows,
        "trainingRows": train_rows,
        "validationRows": validation_rows,
        "lockedTestRows": test_rows,
        "oofUniqueRows": oof_rows,
        "testSetLocked": True,
        "testSetUsed": bool(readiness.get("testSetUsed", False)),
        "classDistribution": {
            "negative": int(distribution.get("negative", 0)),
            "positive": int(distribution.get("positive", 0)),
        },
    }


def _energy_summary(results_dir: Path, experiments_dir: Path) -> dict[str, Any]:
    audit = _optional_json(results_dir / "energy_data_audit.json")
    if not audit:
        return _unavailable("energia", "Energía", "hora")

    source = audit.get("source", {})
    selected = audit.get("selectedSegment", {})
    readiness = audit.get("readiness", {})
    usable_rows = int(selected.get("rows", 0))
    manifest = _latest_manifest(experiments_dir, "energy_oof_v1", domain="energia")
    development_rows, oof_rows = _temporal_counts(manifest)
    locked_test_rows = max(usable_rows - development_rows, 0) if development_rows else 0

    return {
        "id": "energia",
        "label": "Energía",
        "unit": "hora",
        "available": True,
        "datasetId": audit.get("datasetId"),
        "source": "Open Power System Data",
        "originalRows": int(source.get("originalRows", 0)),
        "usableRows": usable_rows,
        "developmentRows": development_rows,
        "trainingRows": None,
        "validationRows": oof_rows,
        "lockedTestRows": locked_test_rows,
        "oofUniqueRows": oof_rows,
        "testSetLocked": bool(manifest),
        "testSetUsed": bool(readiness.get("testSetUsed", False)),
        "range": [selected.get("startAt"), selected.get("endAt")],
    }


def _finance_summary(results_dir: Path, experiments_dir: Path) -> dict[str, Any]:
    audit = _optional_json(results_dir / "finance_mef_data_audit.json")
    if not audit:
        return _unavailable("finanzas", "Finanzas", "sesión financiera")

    source = audit.get("source", {})
    split = audit.get("chronologicalSplit", {})
    readiness = audit.get("readiness", {})
    manifest = _latest_manifest(experiments_dir, "finance_market_oof_v1", domain="finanzas")
    _, oof_rows = _temporal_counts(manifest)

    return {
        "id": "finanzas",
        "label": "Finanzas",
        "unit": "sesión financiera",
        "available": True,
        "datasetId": audit.get("datasetId"),
        "source": "MEF Perú — Índices soberanos",
        "originalRows": int(source.get("originalRows", 0)),
        "usableRows": int(source.get("usableRows", 0)),
        "developmentRows": int(split.get("developmentRows", 0)),
        "trainingRows": None,
        "validationRows": oof_rows,
        "lockedTestRows": int(split.get("lockedTestRows", 0)),
        "oofUniqueRows": oof_rows,
        "testSetLocked": bool(split.get("testSetLocked", False)),
        "testSetUsed": bool(readiness.get("testSetUsed", False)),
        "range": [source.get("startAt"), source.get("endAt")],
    }


def _temporal_counts(manifest: dict[str, Any] | None) -> tuple[int, int]:
    if not manifest:
        return 0, 0
    walk_forward = manifest.get("walkForward", {})
    folds = walk_forward.get("folds", [])
    if not folds:
        return 0, 0
    window = int(walk_forward.get("window", 0))
    horizon = int(walk_forward.get("horizon", 1))
    development_rows = max(
        int(fold.get("trainRows", 0))
        + int(fold.get("gapSteps", 0))
        + int(fold.get("validationRows", 0))
        for fold in folds
    )
    oof_rows = sum(
        max(int(fold.get("validationRows", 0)) - window - horizon + 1, 0)
        for fold in folds
    )
    return development_rows, oof_rows


def _latest_manifest(experiments_dir: Path, namespace: str, *, domain: str) -> dict[str, Any] | None:
    candidates = sorted(
        experiments_dir.glob(f"*/{namespace}/oof_manifest.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for path in candidates:
        manifest = _optional_json(path)
        if manifest and manifest.get("domain") == domain and manifest.get("status") in {
            "thesis_candidate",
            "thesis_base_models_candidate",
        }:
            return manifest
    return None


def _optional_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return read_json(path)


def _unavailable(domain_id: str, label: str, unit: str) -> dict[str, Any]:
    return {
        "id": domain_id,
        "label": label,
        "unit": unit,
        "available": False,
        "datasetId": None,
        "source": None,
        "originalRows": 0,
        "usableRows": 0,
        "developmentRows": 0,
        "trainingRows": None,
        "validationRows": 0,
        "lockedTestRows": 0,
        "oofUniqueRows": 0,
        "testSetLocked": False,
        "testSetUsed": False,
    }
