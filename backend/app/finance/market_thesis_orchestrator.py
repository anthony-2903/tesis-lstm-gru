from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
from threading import Lock
from typing import Any
from uuid import uuid4

from app.config import EXPERIMENTS_DIR, RESULTS_DIR, ensure_dirs
from app.finance.market_freeze import audit_finance_market_freeze_readiness, create_finance_market_freeze_package
from app.finance.market_pipeline import (
    BASE_NAMESPACE,
    FINANCE_MARKET_MODEL_IDS,
    run_finance_market_oof_experiment,
    run_finance_market_revalidation,
)
from app.finance.mef_market_data import get_mef_market_dataset_status, prepare_mef_market_dataset


STOPPED = {"completed", "failed", "cancelled", "interrupted", "paused", "waiting_for_scientific_data"}
DEFAULT_SEEDS = [42, 101, 202, 303, 404]


class FinanceMarketThesisCancelled(RuntimeError):
    pass


class FinanceMarketThesisPaused(RuntimeError):
    pass


class FinanceMarketThesisOrchestrator:
    def __init__(self, *, max_workers: int = 1, job_dir: Path | None = None) -> None:
        ensure_dirs()
        self.job_dir = job_dir or (RESULTS_DIR / "finance_thesis_jobs")
        self.job_dir.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="finance-mef-thesis")
        self._jobs: dict[str, dict[str, Any]] = {}
        self._futures: dict[str, Future[Any]] = {}
        self._restore()

    def submit(self, config: dict[str, Any] | None = None, *, execution_run_id: str | None = None, resumed_from_job_id: str | None = None) -> dict[str, Any]:
        self._refresh_jobs_from_disk()
        effective = {
            "window": 20,
            "horizon": 1,
            "gapSteps": 5,
            "folds": 5,
            "epochs": 20,
            "batchSize": 32,
            "seeds": DEFAULT_SEEDS,
            "bootstrapIterations": 500,
            **(config or {}),
        }
        if len(set(effective["seeds"])) < 5 or int(effective["folds"]) < 5:
            raise ValueError("El protocolo financiero MEF exige cinco semillas y cinco folds.")
        job_id = f"finance_mef_thesis_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}"
        job = {
            "jobId": job_id,
            "executionRunId": execution_run_id or job_id,
            "resumedFromJobId": resumed_from_job_id,
            "protocolFamily": "mef_sovereign_indices_forecasting_v1",
            "status": "queued",
            "createdAt": _now(),
            "startedAt": None,
            "finishedAt": None,
            "ownerPid": None,
            "cancelRequested": False,
            "pauseRequested": False,
            "config": effective,
            "progress": {"stage": "queued", "percent": 0.0, "message": "Protocolo financiero MEF en cola."},
            "stages": [],
            "result": None,
            "blocker": None,
            "error": None,
            "testPolicy": {"testSetUsed": False, "testEvaluationAuthorized": False},
        }
        with self._lock:
            if any(item["status"] in {"queued", "running"} for item in self._jobs.values()):
                raise ValueError("Ya existe un protocolo financiero activo.")
            self._jobs[job_id] = job
            self._persist(job)
            self._futures[job_id] = self._executor.submit(self._execute, job_id)
        return deepcopy(job)

    def latest(self) -> dict[str, Any] | None:
        self._refresh_jobs_from_disk()
        with self._lock:
            return deepcopy(max(self._jobs.values(), key=lambda item: item["createdAt"])) if self._jobs else None

    def get(self, job_id: str) -> dict[str, Any] | None:
        self._refresh_job_from_disk(job_id)
        with self._lock:
            return deepcopy(self._jobs.get(job_id))

    def cancel(self, job_id: str) -> dict[str, Any] | None:
        return self._request_control(job_id, "cancelRequested", "Cancelacion solicitada; se detendra tras la unidad segura actual.")

    def pause(self, job_id: str) -> dict[str, Any] | None:
        return self._request_control(job_id, "pauseRequested", "Pausa solicitada; se detendra tras la unidad segura actual.")

    def resume(self, job_id: str) -> dict[str, Any] | None:
        self._refresh_job_from_disk(job_id)
        with self._lock:
            previous = self._jobs.get(job_id)
            if previous is None:
                return None
            if previous["status"] not in {"failed", "cancelled", "interrupted", "paused", "waiting_for_scientific_data", "waiting_for_external_data"}:
                raise ValueError("Este protocolo no se encuentra detenido.")
            config, execution = deepcopy(previous.get("config", {})), previous.get("executionRunId") or job_id
        return self.submit(config, execution_run_id=execution, resumed_from_job_id=job_id)

    def shutdown(self, *, wait: bool = True) -> None:
        self._executor.shutdown(wait=wait, cancel_futures=True)

    def _execute(self, job_id: str) -> None:
        self._update(job_id, status="running", startedAt=_now(), ownerPid=os.getpid(), blocker=None, error=None)
        job = self.get(job_id)
        config, execution = job["config"], job["executionRunId"]
        try:
            data = get_mef_market_dataset_status()
            if not data.get("readyForThesisTraining"):
                self._progress(job_id, "data_ingestion", 1.0, "Preparando automaticamente la fuente publica Indices Soberanos del MEF.")
                prepare_mef_market_dataset(progress_callback=lambda event: self._ingestion_progress(job_id, event))
                data = get_mef_market_dataset_status()
            if not data.get("readyForThesisTraining"):
                self._update(job_id, status="waiting_for_scientific_data", finishedAt=_now(), ownerPid=None, blocker={"code": "mef_sovereign_indices_audit_failed", "message": data.get("message", "La fuente MEF no supera la auditoria.")}, progress={"stage": "data_gate", "percent": 8.0, "message": "La fuente MEF requiere correccion; test intacto."})
                return
            self._stage(job_id, "data_gate", 10.0, "Fuente publica MEF y hash cientifico verificados.")

            def training_progress(event: dict[str, Any]) -> None:
                self._check_control(job_id)
                done = int(event.get("completedUnits", 0))
                total = max(int(event.get("totalUnits", 1)), 1)
                self._progress(job_id, "base_models", 10.0 + 70.0 * done / total, str(event.get("message", "Entrenando finanzas MEF.")))

            run_finance_market_oof_experiment(
                window=int(config["window"]),
                horizon=int(config["horizon"]),
                gap_steps=int(config["gapSteps"]),
                n_splits=int(config["folds"]),
                epochs=int(config["epochs"]),
                batch_size=int(config["batchSize"]),
                seeds=tuple(int(value) for value in config["seeds"]),
                model_ids=FINANCE_MARKET_MODEL_IDS,
                execution_run_id=execution,
                progress_callback=training_progress,
            )
            base_path = EXPERIMENTS_DIR / execution / BASE_NAMESPACE / "oof_manifest.json"
            self._stage(job_id, "base_models", 80.0, "Matriz 5x5 financiera completa con checkpoints verificables.")
            revalidation = run_finance_market_revalidation(base_manifest_path=base_path, bootstrap_iterations=int(config["bootstrapIterations"]))
            revalidation_path = Path(revalidation["manifest"]["path"])
            self._stage(job_id, "revalidation", 96.0, "Ablacion, comparacion independiente, bootstrap y XAI completados.")
            audit = audit_finance_market_freeze_readiness(base_manifest_path=base_path, revalidation_manifest_path=revalidation_path, verify_artifacts=True)
            if not audit["ready"]:
                raise ValueError("La corrida financiera MEF no supera el congelamiento: " + ", ".join(audit["blockingCheckIds"]))
            frozen = create_finance_market_freeze_package(base_manifest_path=base_path, revalidation_manifest_path=revalidation_path)
            self._stage(job_id, "freeze", 100.0, "Finanzas MEF sellada antes del test.")
            self._update(job_id, status="completed", finishedAt=_now(), ownerPid=None, result={"baseRunId": execution, "revalidationRunId": revalidation["runId"], "freezeId": frozen["freezeId"]}, progress={"stage": "completed", "percent": 100.0, "message": "Protocolo financiero MEF completado; test no autorizado."})
        except FinanceMarketThesisPaused:
            percent = self.get(job_id)["progress"]["percent"]
            self._update(job_id, status="paused", finishedAt=_now(), ownerPid=None, error=None, progress={"stage": "paused", "percent": percent, "message": "Protocolo pausado en una unidad segura; reanude desde checkpoints."})
        except FinanceMarketThesisCancelled as exc:
            self._update(job_id, status="cancelled", finishedAt=_now(), ownerPid=None, error=str(exc), progress={"stage": "cancelled", "percent": self.get(job_id)["progress"]["percent"], "message": str(exc)})
        except Exception as exc:
            self._update(job_id, status="failed", finishedAt=_now(), ownerPid=None, error=str(exc), progress={"stage": "failed", "percent": self.get(job_id)["progress"]["percent"], "message": "Protocolo financiero MEF detenido; reanude desde checkpoints."})

    def _ingestion_progress(self, job_id: str, event: dict[str, Any]) -> None:
        self._check_control(job_id)
        self._progress(job_id, "data_ingestion", min(8.0, float(event.get("percent", 0.0)) * 0.08), str(event.get("message", "Preparando MEF.")))

    def _request_control(self, job_id: str, flag: str, message: str) -> dict[str, Any] | None:
        self._refresh_job_from_disk(job_id)
        with self._lock:
            job = self._jobs.get(job_id)
            if not job or job["status"] in STOPPED:
                return deepcopy(job) if job else None
            other = "pauseRequested" if flag == "cancelRequested" else "cancelRequested"
            if job.get(other):
                return deepcopy(job)
            job[flag] = True
            job["progress"]["message"] = message
            self._persist(job)
            return deepcopy(job)

    def _sync_control_flags(self, job_id: str) -> None:
        path = self.job_dir / f"{job_id}.json"
        try:
            persisted = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job["pauseRequested"] = bool(job.get("pauseRequested") or persisted.get("pauseRequested"))
            job["cancelRequested"] = bool(job.get("cancelRequested") or persisted.get("cancelRequested"))

    def _check_control(self, job_id: str) -> None:
        self._sync_control_flags(job_id)
        job = self.get(job_id)
        if job.get("pauseRequested"):
            raise FinanceMarketThesisPaused("Pausa financiera solicitada por el usuario.")
        if job.get("cancelRequested"):
            raise FinanceMarketThesisCancelled("Protocolo financiero cancelado por el usuario.")

    def _stage(self, job_id: str, stage: str, percent: float, message: str) -> None:
        self._check_control(job_id)
        with self._lock:
            job = self._jobs[job_id]
            job["stages"] = [item for item in job["stages"] if item["stage"] != stage] + [{"stage": stage, "status": "completed", "completedAt": _now(), "message": message}]
            job["progress"] = {"stage": stage, "percent": percent, "message": message}
            self._persist(job)

    def _progress(self, job_id: str, stage: str, percent: float, message: str) -> None:
        self._update(job_id, progress={"stage": stage, "percent": round(percent, 2), "message": message})

    def _update(self, job_id: str, **changes: Any) -> None:
        with self._lock:
            self._jobs[job_id].update(changes)
            self._persist(self._jobs[job_id])

    def _restore(self) -> None:
        for path in self.job_dir.glob("*.json"):
            try:
                job = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if job.get("protocolFamily") != "mef_sovereign_indices_forecasting_v1":
                continue
            job.setdefault("pauseRequested", False)
            job.setdefault("cancelRequested", False)
            recently_updated = time.time() - path.stat().st_mtime < 1_800
            if job.get("status") in {"queued", "running"} and not recently_updated and not _owner_is_alive(job):
                job.update(status="interrupted", finishedAt=_now(), ownerPid=None, error="Backend reiniciado; reanude desde checkpoints.")
                self._persist(job)
            self._jobs[job["jobId"]] = job

    def _refresh_job_from_disk(self, job_id: str) -> None:
        path = self.job_dir / f"{job_id}.json"
        try:
            job = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        if not isinstance(job, dict) or job.get("jobId") != job_id:
            return
        if job.get("protocolFamily") != "mef_sovereign_indices_forecasting_v1":
            return
        job.setdefault("pauseRequested", False)
        job.setdefault("cancelRequested", False)
        with self._lock:
            self._jobs[job_id] = job

    def _refresh_jobs_from_disk(self) -> None:
        for path in self.job_dir.glob("*.json"):
            self._refresh_job_from_disk(path.stem)

    def _persist(self, job: dict[str, Any]) -> None:
        path = self.job_dir / f"{job['jobId']}.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, path)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


finance_market_thesis_orchestrator = FinanceMarketThesisOrchestrator(max_workers=1)
