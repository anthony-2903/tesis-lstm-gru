from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from app.config import RESULTS_DIR


DOMAIN_SPECS = (
    {
        "id": "phishing",
        "label": "Ciberseguridad / phishing",
        "route": "/phishing",
        "jobDirectory": "phishing_thesis_jobs",
        "dependsOn": [],
    },
    {
        "id": "energia",
        "label": "Energía eléctrica",
        "route": "/energy",
        "jobDirectory": "energy_thesis_jobs",
        "dependsOn": ["phishing"],
    },
    {
        "id": "finanzas",
        "label": "Finanzas / Indices Soberanos MEF",
        "route": "/finance",
        "jobDirectory": "finance_thesis_jobs",
        "dependsOn": ["phishing", "energia"],
    },
)

ACTIVE_STATUSES = {"queued", "running"}
WAITING_STATUSES = {"waiting_for_external_data", "waiting_for_scientific_data"}
FAILED_STATUSES = {"failed", "cancelled", "interrupted"}
PAUSED_STATUSES = {"paused"}


def get_thesis_status(results_dir: Path | None = None) -> dict[str, Any]:
    """Build a read-only, disk-backed view of the three scientific protocols.

    Job JSON files are authoritative because training can run in a detached process
    that is different from the API process. This function never starts, resumes, or
    mutates an experiment.
    """

    root = results_dir or RESULTS_DIR
    domains = [_domain_status(root, spec) for spec in DOMAIN_SPECS]
    chain = _read_json(root / "thesis_chain_state.json") or {}
    progress_values = [float(item["progress"]["percent"]) for item in domains]
    overall_percent = round(sum(progress_values) / len(progress_values), 2)
    completed = sum(item["status"] == "completed" for item in domains)
    active = next((item for item in domains if item["status"] in ACTIVE_STATUSES), None)
    blockers = [
        {"domain": item["id"], **item["blocker"]}
        for item in domains
        if isinstance(item.get("blocker"), dict)
    ]
    test_used = any(bool(item["testPolicy"].get("testSetUsed")) for item in domains)
    test_authorized = any(bool(item["testPolicy"].get("testEvaluationAuthorized")) for item in domains)

    return {
        "schemaVersion": "1.0.0",
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "scope": "three_domain_scientific_execution",
        "progressMethod": {
            "formula": "arithmetic_mean_of_domain_protocol_percentages",
            "equalDomainWeight": True,
            "domainCount": len(domains),
            "includesSoftwareImplementation": False,
            "note": "Mide la ejecución experimental real, no el avance de programación ni redacción.",
        },
        "overall": {
            "percent": overall_percent,
            "remainingPercent": round(100.0 - overall_percent, 2),
            "completedDomains": completed,
            "totalDomains": len(domains),
            "status": _overall_status(domains),
            "activeDomain": active["id"] if active else None,
        },
        "domains": domains,
        "blockers": blockers,
        "chain": {
            "status": chain.get("status", "not_started"),
            "domain": chain.get("domain"),
            "message": chain.get("message", "El estado detallado se obtiene de los trabajos por dominio."),
            "updatedAt": chain.get("updatedAt"),
            "ownerPid": chain.get("ownerPid"),
        },
        "testPolicy": {
            "locked": not test_used and not test_authorized,
            "testSetUsed": test_used,
            "testEvaluationAuthorized": test_authorized,
            "maximumEvaluationsPerFrozenDomain": 1,
            "message": "El test final sigue bloqueado y requiere autorización explícita después del sellado.",
        },
    }


def _domain_status(root: Path, spec: dict[str, Any]) -> dict[str, Any]:
    job, path = _latest_job(root / str(spec["jobDirectory"]))
    if job is None:
        return {
            "id": spec["id"],
            "label": spec["label"],
            "route": spec["route"],
            "dependsOn": spec["dependsOn"],
            "status": "pending",
            "jobId": None,
            "executionRunId": None,
            "updatedAt": None,
            "progress": {"stage": "pending", "percent": 0.0, "remainingPercent": 100.0, "message": "Pendiente en la cadena experimental."},
            "blocker": None,
            "nextAction": "Se iniciará automáticamente cuando terminen sus dependencias.",
            "testPolicy": {"testSetUsed": False, "testEvaluationAuthorized": False},
        }

    status = str(job.get("status") or "unknown")
    progress = job.get("progress") if isinstance(job.get("progress"), dict) else {}
    raw_percent = 100.0 if status == "completed" else _bounded_percent(progress.get("percent"))
    message = str(progress.get("message") or _status_message(status))
    raw_blocker = job.get("blocker") if isinstance(job.get("blocker"), dict) else None
    blocker = _public_blocker(raw_blocker)
    test_policy = job.get("testPolicy") if isinstance(job.get("testPolicy"), dict) else {}
    updated_at = job.get("finishedAt") or job.get("startedAt") or job.get("createdAt")
    if path is not None:
        updated_at = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()

    return {
        "id": spec["id"],
        "label": spec["label"],
        "route": spec["route"],
        "dependsOn": spec["dependsOn"],
        "status": status,
        "jobId": job.get("jobId"),
        "executionRunId": job.get("executionRunId"),
        "updatedAt": updated_at,
        "progress": {
            "stage": str(progress.get("stage") or status),
            "percent": raw_percent,
            "remainingPercent": round(100.0 - raw_percent, 2),
            "message": message,
        },
        "blocker": blocker,
        "nextAction": _next_action(status, blocker),
        "testPolicy": {
            "testSetUsed": bool(test_policy.get("testSetUsed", False)),
            "testEvaluationAuthorized": bool(test_policy.get("testEvaluationAuthorized", False)),
        },
    }


def _latest_job(directory: Path) -> tuple[dict[str, Any] | None, Path | None]:
    if not directory.exists():
        return None, None
    candidates: list[tuple[int, dict[str, Any], Path]] = []
    for path in directory.glob("*.json"):
        payload = _read_json(path)
        if payload is not None and isinstance(payload.get("jobId"), str):
            candidates.append((path.stat().st_mtime_ns, payload, path))
    if not candidates:
        return None, None
    _, payload, path = max(candidates, key=lambda item: item[0])
    return payload, path


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _public_blocker(blocker: dict[str, Any] | None) -> dict[str, str] | None:
    """Expose the reason but never local paths or provider-specific credentials."""

    if blocker is None:
        return None
    return {
        "code": str(blocker.get("code") or "unspecified_blocker"),
        "message": str(blocker.get("message") or "Existe un requisito pendiente."),
    }


def _bounded_percent(value: Any) -> float:
    try:
        return round(max(0.0, min(100.0, float(value))), 2)
    except (TypeError, ValueError):
        return 0.0


def _overall_status(domains: list[dict[str, Any]]) -> str:
    statuses = {item["status"] for item in domains}
    if statuses == {"completed"}:
        return "protocols_frozen_test_locked"
    if statuses & ACTIVE_STATUSES:
        return "running"
    if statuses & PAUSED_STATUSES:
        return "paused"
    if statuses & FAILED_STATUSES:
        return "needs_resume"
    if statuses & WAITING_STATUSES:
        return "waiting_for_data"
    return "pending"


def _status_message(status: str) -> str:
    return {
        "completed": "Protocolo sellado; test final no evaluado.",
        "running": "Protocolo científico en ejecución.",
        "queued": "Protocolo científico en cola.",
        "failed": "La ejecución falló y conserva checkpoints para reanudación.",
        "interrupted": "La ejecución se interrumpió y puede reanudarse desde checkpoints.",
        "waiting_for_external_data": "Esperando datos externos con licencia y linaje verificables.",
        "waiting_for_scientific_data": "Esperando un dataset científicamente apto.",
        "paused": "La ejecucion esta pausada de forma segura y conserva sus checkpoints.",
    }.get(status, "Estado experimental disponible.")


def _next_action(status: str, blocker: dict[str, Any] | None) -> str:
    if status == "completed":
        return "Conservar el sellado y solicitar autorización antes de la única evaluación final."
    if status in ACTIVE_STATUSES:
        return "Esperar; el supervisor continuará automáticamente y conserva checkpoints."
    if status in PAUSED_STATUSES:
        return "Reanudar desde checkpoints cuando el equipo este listo."
    if status in FAILED_STATUSES:
        return "Corregir la causa y reanudar desde checkpoints sin abrir el test."
    if status == "waiting_for_external_data":
        code = blocker.get("code") if blocker else None
        if code == "ieee_cis_required":
            return "Aceptar personalmente las reglas de IEEE-CIS y proporcionar los archivos con licencia y hash."
        return "Proporcionar la fuente externa exigida con licencia, versión y hash."
    if status == "waiting_for_scientific_data":
        return "Preparar y auditar el dataset científico antes de entrenar."
    return "Se iniciará automáticamente cuando termine el dominio anterior."
