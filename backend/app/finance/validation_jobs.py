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
from app.finance.validation import run_finance_temporal_validation


FINAL_STATUSES = {"completed", "failed", "cancelled", "interrupted"}


class FinanceValidationCancelled(RuntimeError):
    pass


class FinanceValidationJobManager:
    def __init__(self, *, max_workers: int = 1, job_dir: Path | None = None) -> None:
        ensure_dirs()
        self.job_dir = job_dir or (RESULTS_DIR / "finance_validation_jobs")
        self.job_dir.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="finance-validation")
        self._jobs: dict[str, dict[str, Any]] = {}
        self._futures: dict[str, Future[Any]] = {}
        self._restore_jobs()

    def submit(self, config: dict[str, Any]) -> dict[str, Any]:
        job_id = f"finance_validation_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}"
        total_units = len(config.get("seeds", [42])) * 6 + 6
        job = {
            "jobId": job_id,
            "status": "queued",
            "createdAt": _now(),
            "startedAt": None,
            "finishedAt": None,
            "cancelRequested": False,
            "config": deepcopy(config),
            "progress": {
                "stage": "queued",
                "event": "queued",
                "completedUnits": 0,
                "resumedUnits": 0,
                "totalUnits": total_units,
                "percent": 0.0,
                "seed": None,
                "modelId": None,
                "message": "Validacion temporal financiera en cola.",
            },
            "resultRunId": None,
            "error": None,
        }
        with self._lock:
            if any(item["status"] in {"queued", "running"} for item in self._jobs.values()):
                raise ValueError("Ya existe una validacion financiera activa.")
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
                job["progress"]["message"] = "Cancelacion solicitada; se detendra al terminar la unidad actual."
            self._persist_unlocked(job)
            return deepcopy(job)

    def shutdown(self, *, wait: bool = True) -> None:
        self._executor.shutdown(wait=wait, cancel_futures=True)

    def _execute(self, job_id: str) -> None:
        self._update(job_id, status="running", startedAt=_now())
        config = self.get(job_id)["config"]

        def progress(event: dict[str, Any]) -> None:
            with self._lock:
                job = self._jobs[job_id]
                if job["cancelRequested"]:
                    raise FinanceValidationCancelled("La validacion financiera fue cancelada por el usuario.")
                completed = int(event.get("completedUnits", 0))
                total = max(int(event.get("totalUnits", 1)), 1)
                job["progress"] = {**job["progress"], **event, "percent": round(completed * 100.0 / total, 2)}
                self._persist_unlocked(job)

        try:
            result = run_finance_temporal_validation(
                demo_max_train_rows=config.get("demoMaxTrainRows"),
                demo_max_validation_rows=config.get("demoMaxValidationRows"),
                bootstrap_iterations=int(config.get("bootstrapIterations", 500)),
                progress_callback=progress,
            )
            self._update(
                job_id,
                status="completed",
                finishedAt=_now(),
                resultRunId=result["runId"],
                progress={**self.get(job_id)["progress"], "stage": "completed", "event": "completed", "percent": 100.0, "message": "Validacion temporal completada; test bloqueado."},
            )
        except FinanceValidationCancelled as exc:
            self._update(job_id, status="cancelled", finishedAt=_now(), error=str(exc), progress={**self.get(job_id)["progress"], "stage": "cancelled", "message": str(exc)})
        except Exception as exc:
            self._update(job_id, status="failed", finishedAt=_now(), error=str(exc), progress={**self.get(job_id)["progress"], "stage": "failed", "message": "La validacion financiera termino con error."})

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
                job["error"] = "El backend se reinicio antes de finalizar la validacion financiera."
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


finance_validation_job_manager = FinanceValidationJobManager(max_workers=1)
