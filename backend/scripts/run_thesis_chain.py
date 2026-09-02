from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
from typing import Any

from app.config import RESULTS_DIR
from app.energy.thesis_orchestrator import energy_thesis_orchestrator
from app.finance.market_thesis_orchestrator import finance_market_thesis_orchestrator as finance_thesis_orchestrator
from app.phishing.thesis_orchestrator import phishing_thesis_orchestrator


STATE_PATH = RESULTS_DIR / "thesis_chain_state.json"
ACTIVE = {"queued", "running"}
RESUMABLE = {
    "failed",
    "cancelled",
    "interrupted",
    "paused",
    "waiting_for_external_data",
    "waiting_for_scientific_data",
}


def main() -> None:
    _set_conservative_process_priority()
    parser = argparse.ArgumentParser(description="Execute thesis protocols sequentially without authorizing final test.")
    parser.add_argument("--resume-phishing", help="Trabajo de phishing preferido; si se omite se usa el ultimo estado persistido.")
    args = parser.parse_args()
    try:
        _run_chain(args.resume_phishing)
    except Exception:
        previous = _read_state()
        domain = str(previous.get("domain") or "phishing")
        _state("failed", domain, previous, "El supervisor se detuvo; revise el log y reanude desde checkpoints.")
        raise


def _run_chain(preferred_phishing_job_id: str | None = None) -> None:
    _state("running", "phishing", None, "Evaluando estado persistido de phishing.")
    phishing = _run_or_resume_and_wait(phishing_thesis_orchestrator, "phishing", preferred_job_id=preferred_phishing_job_id)
    if phishing["status"] != "completed":
        status = "paused" if phishing["status"] == "paused" else "stopped"
        _state(status, "phishing", phishing, "Phishing pausado de forma segura." if status == "paused" else "Phishing requiere correccion antes de continuar.")
        return

    _state("running", "energy", phishing, "Phishing completo; iniciando energia automaticamente.")
    energy = _run_or_resume_and_wait(energy_thesis_orchestrator, "energy")
    if energy["status"] != "completed":
        status = "paused" if energy["status"] == "paused" else "stopped"
        _state(status, "energy", energy, "Energia pausada de forma segura." if status == "paused" else "Energia requiere correccion antes de continuar.")
        return

    _state("running", "finance", energy, "Energia completa; iniciando finanzas con Indices Soberanos del MEF.")
    finance = _run_or_resume_and_wait(finance_thesis_orchestrator, "finance")
    status = "completed_pre_test" if finance["status"] == "completed" else "paused" if finance["status"] == "paused" else "stopped"
    message = "Cadena pre-test finalizada." if status == "completed_pre_test" else "Finanzas pausada de forma segura." if status == "paused" else "Finanzas requiere correccion."
    _state(status, "finance", finance, message)


def _run_or_resume_and_wait(manager: Any, domain: str, *, preferred_job_id: str | None = None) -> dict[str, Any]:
    latest = manager.latest()
    preferred = manager.get(preferred_job_id) if preferred_job_id else None
    if preferred_job_id and preferred is None:
        raise ValueError(f"No existe el trabajo {preferred_job_id}.")
    candidate = _select_candidate(latest, preferred)
    if candidate and candidate["status"] == "completed":
        return candidate
    if candidate and candidate["status"] in ACTIVE:
        return _wait_for_existing_job(manager, candidate, domain)
    if candidate and candidate["status"] in RESUMABLE:
        job = manager.resume(candidate["jobId"])
    elif candidate is None:
        job = manager.submit()
    else:
        raise ValueError(f"El protocolo de {domain} tiene un estado no recuperable: {candidate['status']}.")
    if job is None:
        raise ValueError("No se pudo crear o reanudar el protocolo.")
    _state("running", domain, job, f"Protocolo de {domain} activo y persistido.")
    manager._futures[job["jobId"]].result()
    return manager.get(job["jobId"])


def _select_candidate(latest: dict[str, Any] | None, preferred: dict[str, Any] | None) -> dict[str, Any] | None:
    if preferred is None:
        return latest
    if latest is None or latest["jobId"] == preferred["jobId"]:
        return preferred
    if latest.get("status") in ACTIVE | {"completed"} and str(latest.get("createdAt", "")) > str(preferred.get("createdAt", "")):
        return latest
    return preferred


def _wait_for_existing_job(manager: Any, job: dict[str, Any], domain: str) -> dict[str, Any]:
    _state("running", domain, job, f"El protocolo de {domain} ya estaba activo; observando su estado sin duplicarlo.")
    current = job
    while current["status"] in ACTIVE:
        if not _owner_is_alive(current):
            raise RuntimeError(f"El protocolo de {domain} figura activo, pero su proceso propietario no existe.")
        time.sleep(5)
        refreshed = manager.get(current["jobId"])
        if refreshed is None:
            raise RuntimeError(f"El protocolo activo de {domain} desaparecio del almacenamiento persistente.")
        current = refreshed
    return current


def _owner_is_alive(job: dict[str, Any]) -> bool:
    owner_pid = job.get("ownerPid")
    if not isinstance(owner_pid, int) or owner_pid <= 0:
        return False
    if os.name == "nt":
        import ctypes

        handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, owner_pid)
        if not handle:
            return False
        try:
            exit_code = ctypes.c_ulong()
            return bool(ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code))) and exit_code.value == 259
        finally:
            ctypes.windll.kernel32.CloseHandle(handle)
    try:
        os.kill(owner_pid, 0)
        return True
    except OSError:
        return False


def _set_conservative_process_priority() -> None:
    """Reduce CPU scheduling priority without making execution platform-specific."""

    if os.name != "nt":
        try:
            os.nice(5)
        except OSError:
            pass
        return
    try:
        import ctypes

        process = ctypes.windll.kernel32.GetCurrentProcess()
        ctypes.windll.kernel32.SetPriorityClass(process, 0x00004000)  # BELOW_NORMAL_PRIORITY_CLASS
    except Exception:
        pass


def _state(status: str, domain: str, job: dict[str, Any] | None, message: str) -> None:
    payload = {
        "schemaVersion": "1.0.0",
        "status": status,
        "domain": domain,
        "message": message,
        "updatedAt": datetime.now(timezone.utc).isoformat(),
        "ownerPid": os.getpid() if status == "running" else None,
        "jobId": job.get("jobId") if job else None,
        "executionRunId": job.get("executionRunId") if job else None,
        "testPolicy": {"testSetUsed": False, "testEvaluationAuthorized": False},
    }
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = STATE_PATH.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, STATE_PATH)
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def _read_state() -> dict[str, Any]:
    try:
        payload = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


if __name__ == "__main__":
    main()
