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
from app.phishing.academic_sources import build_academic_curation_package
from app.phishing.ingestion import prepare_real_phishing_dataset


class PhishingDataPreparationManager:
    def __init__(self, *, state_path: Path | None = None) -> None:
        ensure_dirs()
        self.state_path = state_path or (RESULTS_DIR / "phishing_data_preparation_job.json")
        self._lock = Lock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="phishing-data")
        self._job = self._restore()

    def submit(self, *, force: bool = False, per_class: int = 10_000, include_academic_sources: bool = False) -> dict[str, Any]:
        with self._lock:
            if self._job and self._job["status"] in {"queued", "running"}:
                return deepcopy(self._job)
            job = {
                "jobId": f"phishing_data_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}",
                "status": "queued",
                "createdAt": _now(),
                "startedAt": None,
                "finishedAt": None,
                "force": force,
                "perClass": per_class,
                "includeAcademicSources": include_academic_sources,
                "progress": {"stage": "queued", "percent": 0.0, "message": "Preparación en cola."},
                "result": None,
                "error": None,
            }
            self._job = job
            self._persist_unlocked()
            self._executor.submit(self._execute, job["jobId"], force, per_class, include_academic_sources)
            return deepcopy(job)

    def latest(self) -> dict[str, Any] | None:
        with self._lock:
            return deepcopy(self._job) if self._job else None

    def shutdown(self, *, wait: bool = True) -> None:
        self._executor.shutdown(wait=wait, cancel_futures=True)

    def _execute(self, job_id: str, force: bool, per_class: int, include_academic_sources: bool) -> None:
        self._update(status="running", startedAt=_now())

        def progress(event: dict[str, Any]) -> None:
            percent = float(event.get("percent", 0.0))
            if include_academic_sources:
                percent = 55.0 + percent * 0.45
            self._update(progress={**event, "percent": percent})

        try:
            curation = None
            if include_academic_sources:
                def curation_progress(event: dict[str, Any]) -> None:
                    self._update(progress={**event, "percent": float(event.get("percent", 0.0)) * 0.55})

                curation = build_academic_curation_package(
                    force_download=force,
                    progress_callback=curation_progress,
                )
            result = prepare_real_phishing_dataset(
                force_download=force,
                per_class=per_class,
                progress_callback=progress,
            )
            self._update(
                status="completed",
                finishedAt=_now(),
                progress={"stage": "completed", "percent": 100.0, "message": "Dataset de phishing preparado."},
                result={
                    "datasetId": result["metadata"]["datasetId"],
                    "rows": result["metadata"]["silver"]["rows"],
                    "readyForPipelinePilot": result["audit"]["readiness"]["readyForPipelinePilot"],
                    "readyForThesisTraining": result["audit"]["readiness"]["readyForThesisTraining"],
                    "academicCurationReady": bool(curation and curation["status"].get("readyForScientificMerge")),
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
            job["progress"] = {
                "stage": "interrupted",
                "percent": job.get("progress", {}).get("percent", 0),
                "message": job["error"],
            }
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


phishing_data_manager = PhishingDataPreparationManager()
