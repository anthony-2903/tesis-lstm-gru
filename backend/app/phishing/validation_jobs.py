from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from threading import Lock
from typing import Any
from uuid import uuid4

from app.config import RESULTS_DIR, ensure_dirs
from app.phishing.validation import run_phishing_external_validation


FINAL_STATUSES = {"completed", "failed", "cancelled", "interrupted"}


class PhishingValidationCancelled(RuntimeError):
    pass


class PhishingValidationJobManager:
    def __init__(self, *, max_workers: int = 1, job_dir: Path | None = None) -> None:
        ensure_dirs()
        self.job_dir = job_dir or (RESULTS_DIR / "phishing_validation_jobs")
        self.job_dir.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="phishing-validation")
        self._jobs: dict[str, dict[str, Any]] = {}
        self._futures: dict[str, Future[Any]] = {}
        self._restore_jobs()

    def submit(self) -> dict[str, Any]:
        job_id = f"phishing_validation_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}"
        job = {
            "jobId": job_id,
            "status": "queued",
            "createdAt": _now(),
            "startedAt": None,
            "finishedAt": None,
            "cancelRequested": False,
            "progress": {
                "stage": "queued",
                "event": "queued",
                "completedUnits": 0,
                "totalUnits": 1,
                "percent": 0.0,
                "seed": None,
                "modelId": None,
                "message": "Validación externa de phishing en cola.",
            },
            "resultRunId": None,
            "error": None,
        }
        with self._lock:
            self._jobs[job_id] = job
            self._persist_unlocked(job)
            self._futures[job_id] = self._executor.submit(self._execute, job_id)
        return deepcopy(job)

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return deepcopy(job) if job else None

    def latest(self) -> dict[str, Any] | None:
        with self._lock:
            return deepcopy(max(self._jobs.values(), key=lambda item: item["createdAt"])) if self._jobs else None

    def cancel(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job["status"] in FINAL_STATUSES:
                return deepcopy(job) if job else None
            job["cancelRequested"] = True
            future = self._futures.get(job_id)
            if job["status"] == "queued" and future is not None and future.cancel():
                job["status"] = "cancelled"
                job["finishedAt"] = _now()
            else:
                job["progress"]["message"] = "Cancelación solicitada; se detendrá al terminar el modelo actual."
            self._persist_unlocked(job)
            return deepcopy(job)

    def shutdown(self, *, wait: bool = True) -> None:
        self._executor.shutdown(wait=wait, cancel_futures=True)

    def _execute(self, job_id: str) -> None:
        self._update(job_id, status="running", startedAt=_now())

        def progress(event: dict[str, Any]) -> None:
            with self._lock:
                job = self._jobs[job_id]
                if job["cancelRequested"]:
                    raise PhishingValidationCancelled("La validación fue cancelada por el usuario.")
                completed = int(event.get("completedUnits", 0))
                total = max(int(event.get("totalUnits", 1)), 1)
                job["progress"] = {**job["progress"], **event, "percent": round(completed * 100.0 / total, 2)}
                self._persist_unlocked(job)

        try:
            result = run_phishing_external_validation(progress_callback=progress)
            self._update(
                job_id,
                status="completed",
                finishedAt=_now(),
                resultRunId=result["runId"],
                progress={**self.get(job_id)["progress"], "stage": "completed", "event": "completed", "percent": 100.0, "message": "Validación externa completada; test bloqueado."},
            )
        except PhishingValidationCancelled as exc:
            self._update(job_id, status="cancelled", finishedAt=_now(), error=str(exc), progress={**self.get(job_id)["progress"], "stage": "cancelled", "message": str(exc)})
        except Exception as exc:
            self._update(job_id, status="failed", finishedAt=_now(), error=str(exc), progress={**self.get(job_id)["progress"], "stage": "failed", "message": "La validación terminó con error."})

    def _update(self, job_id: str, **changes: Any) -> None:
        with self._lock:
            self._jobs[job_id].update(changes)
            self._persist_unlocked(self._jobs[job_id])

    def _restore_jobs(self) -> None:
        for path in self.job_dir.glob("*.json"):
            try:
                job = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if job.get("status") in {"queued", "running"}:
                job["status"] = "interrupted"
                job["finishedAt"] = _now()
                job["error"] = "El backend se reinició antes de finalizar la validación."
                job["progress"]["stage"] = "interrupted"
                job["progress"]["message"] = job["error"]
                self._persist_unlocked(job)
            self._jobs[job["jobId"]] = job

    def _persist_unlocked(self, job: dict[str, Any]) -> None:
        path = self.job_dir / f"{job['jobId']}.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, path)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


phishing_validation_job_manager = PhishingValidationJobManager(max_workers=1)
