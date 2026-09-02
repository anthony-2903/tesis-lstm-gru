from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from threading import Lock
from typing import Any
from uuid import uuid4

from app.config import RESULTS_DIR, ensure_dirs
from app.energy.ingestion import prepare_real_energy_dataset


class EnergyDataPreparationManager:
    def __init__(self, *, state_path: Path | None = None) -> None:
        ensure_dirs()
        self.state_path = state_path or (RESULTS_DIR / "energy_data_preparation_job.json")
        self._lock = Lock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="energy-data")
        self._job = self._restore()

    def submit(self, *, force: bool = False) -> dict[str, Any]:
        with self._lock:
            if self._job and self._job["status"] in {"queued", "running"}:
                return deepcopy(self._job)
            job = {
                "jobId": f"energy_data_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}",
                "status": "queued",
                "createdAt": _now(),
                "startedAt": None,
                "finishedAt": None,
                "force": force,
                "progress": {"stage": "queued", "percent": 0.0, "message": "Preparación en cola."},
                "result": None,
                "error": None,
            }
            self._job = job
            self._persist_unlocked()
            self._executor.submit(self._execute, job["jobId"], force)
            return deepcopy(job)

    def latest(self) -> dict[str, Any] | None:
        with self._lock:
            return deepcopy(self._job) if self._job else None

    def shutdown(self, *, wait: bool = True) -> None:
        self._executor.shutdown(wait=wait, cancel_futures=True)

    def _execute(self, job_id: str, force: bool) -> None:
        self._update(status="running", startedAt=_now())

        def progress(event: dict[str, Any]) -> None:
            self._update(progress=event)

        try:
            result = prepare_real_energy_dataset(force_download=force, progress_callback=progress)
            self._update(
                status="completed",
                finishedAt=_now(),
                progress={"stage": "completed", "percent": 100.0, "message": "Dataset real preparado."},
                result={
                    "datasetId": result["metadata"]["datasetId"],
                    "rows": result["metadata"]["silver"]["rows"],
                    "readyForThesisPilot": result["audit"]["readiness"]["ready"],
                },
            )
        except Exception as exc:
            self._update(
                status="failed",
                finishedAt=_now(),
                progress={"stage": "failed", "percent": 0.0, "message": "Falló la preparación del dataset."},
                error=str(exc),
            )

    def _update(self, **changes: Any) -> None:
        with self._lock:
            if self._job is None:
                return
            self._job.update(changes)
            self._persist_unlocked()

    def _restore(self) -> dict[str, Any] | None:
        if not self.state_path.exists():
            return None
        try:
            job = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if job.get("status") in {"queued", "running"}:
            job["status"] = "interrupted"
            job["finishedAt"] = _now()
            job["error"] = "El backend se reinició durante la preparación de datos."
            job["progress"] = {"stage": "interrupted", "percent": job.get("progress", {}).get("percent", 0), "message": job["error"]}
            self._job = job
            self._persist_unlocked()
        return job

    def _persist_unlocked(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.state_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self._job, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, self.state_path)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


energy_data_manager = EnergyDataPreparationManager()
