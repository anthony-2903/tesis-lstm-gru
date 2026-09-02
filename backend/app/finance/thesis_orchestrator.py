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
from app.finance.diversity import run_finance_diversity_ablation
from app.finance.freeze import audit_finance_freeze_readiness, create_finance_freeze_package
from app.finance.models import FINANCE_MODEL_IDS
from app.finance.pipeline import run_finance_oof_experiment
from app.finance.real_data import get_real_finance_data_status
from app.finance.revalidation import run_finance_optimized_stacking_revalidation
from app.finance.sequences import get_finance_sequence_status, prepare_finance_sequence_protocol
from app.finance.stacking import run_finance_stacking_experiment
from app.finance.validation import run_finance_temporal_validation


STOPPED_STATUSES = {"completed", "failed", "cancelled", "interrupted", "paused", "waiting_for_external_data"}
DEFAULT_SEEDS = [42, 101, 202, 303, 404]


class FinanceThesisCancelled(RuntimeError):
    pass


class FinanceThesisPaused(RuntimeError):
    pass


class FinanceThesisOrchestrator:
    def __init__(self, *, max_workers: int = 1, job_dir: Path | None = None) -> None:
        ensure_dirs()
        self.job_dir = job_dir or (RESULTS_DIR / "finance_thesis_jobs")
        self.job_dir.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="finance-thesis")
        self._jobs: dict[str, dict[str, Any]] = {}
        self._futures: dict[str, Future[Any]] = {}
        self._restore()

    def submit(self, config: dict[str, Any] | None = None, *, execution_run_id: str | None = None, resumed_from_job_id: str | None = None) -> dict[str, Any]:
        self._refresh_jobs_from_disk()
        effective = {"epochs": 30, "batchSize": 128, "patience": 5, "seeds": DEFAULT_SEEDS, "bootstrapIterations": 500, **(config or {})}
        if len(set(effective["seeds"])) < 5:
            raise ValueError("El orquestador de tesis exige cinco semillas distintas.")
        job_id = f"finance_thesis_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}"
        job = {
            "jobId": job_id,
            "executionRunId": execution_run_id or job_id,
            "resumedFromJobId": resumed_from_job_id,
            "status": "queued",
            "createdAt": _now(), "startedAt": None, "finishedAt": None,
            "cancelRequested": False,
            "pauseRequested": False,
            "config": effective,
            "progress": {"stage": "queued", "percent": 0.0, "message": "Protocolo financiero de tesis en cola."},
            "stages": [], "result": None, "blocker": None, "error": None,
            "testPolicy": {"testSetUsed": False, "testEvaluationAuthorized": False},
        }
        with self._lock:
            if any(value["status"] in {"queued", "running"} for value in self._jobs.values()):
                raise ValueError("Ya existe un protocolo financiero de tesis activo.")
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
        self._refresh_job_from_disk(job_id)
        with self._lock:
            job = self._jobs.get(job_id)
            if not job or job["status"] in STOPPED_STATUSES:
                return deepcopy(job) if job else None
            if job.get("pauseRequested"):
                return deepcopy(job)
            job["cancelRequested"] = True
            job["progress"]["message"] = "Cancelacion solicitada; se detendra en la siguiente unidad segura."
            self._persist(job)
            return deepcopy(job)

    def pause(self, job_id: str) -> dict[str, Any] | None:
        self._refresh_job_from_disk(job_id)
        with self._lock:
            job = self._jobs.get(job_id)
            if not job or job["status"] in STOPPED_STATUSES:
                return deepcopy(job) if job else None
            if job.get("cancelRequested"):
                return deepcopy(job)
            job["pauseRequested"] = True
            job["progress"]["message"] = "Pausa solicitada; se detendra tras la unidad segura actual."
            self._persist(job)
            return deepcopy(job)

    def resume(self, job_id: str) -> dict[str, Any] | None:
        self._refresh_job_from_disk(job_id)
        with self._lock:
            previous = self._jobs.get(job_id)
            if previous is None:
                return None
            if previous["status"] not in {"failed", "cancelled", "interrupted", "paused", "waiting_for_external_data"}:
                raise ValueError("Este protocolo no se encuentra detenido.")
            config = deepcopy(previous["config"])
            execution = previous["executionRunId"]
        return self.submit(config, execution_run_id=execution, resumed_from_job_id=job_id)

    def shutdown(self, *, wait: bool = True) -> None:
        self._executor.shutdown(wait=wait, cancel_futures=True)

    def _execute(self, job_id: str) -> None:
        self._update(job_id, status="running", startedAt=_now(), ownerPid=os.getpid())
        config = self.get(job_id)["config"]
        execution = self.get(job_id)["executionRunId"]
        try:
            real = get_real_finance_data_status()
            if not real.get("readyForThesisTraining"):
                self._update(job_id, status="waiting_for_external_data", finishedAt=_now(), ownerPid=None, blocker={"code": "ieee_cis_required", "message": real.get("message"), "manifestPath": real.get("manifestPath")}, progress={"stage": "real_data_gate", "percent": 0.0, "message": "Esperando IEEE-CIS verificado; no se inicio entrenamiento."})
                return
            self._stage(job_id, "real_data_gate", 5, "Dataset real y licencia verificados.")
            sequence = get_finance_sequence_status()
            if not sequence.get("readyForThesisTraining"):
                prepare_finance_sequence_protocol(window=10, folds=5, purge_days=1)
            self._stage(job_id, "sequences", 10, "Secuencias causales y cinco folds preparados.")

            def base_progress(event: dict[str, Any]) -> None:
                self._check_control(job_id)
                completed, total = int(event.get("completedUnits", 0)), max(int(event.get("totalUnits", 1)), 1)
                self._progress(job_id, "base_models", 10 + 45 * completed / total, str(event.get("message", "Entrenando modelos base.")))

            run_finance_oof_experiment(protocol="thesis", epochs=int(config["epochs"]), batch_size=int(config["batchSize"]), patience=int(config["patience"]), seeds=tuple(int(value) for value in config["seeds"]), model_ids=FINANCE_MODEL_IDS, demo_max_rows_per_fold=None, execution_run_id=execution, progress_callback=base_progress)
            base_path = EXPERIMENTS_DIR / execution / "finance_oof_v1" / "oof_manifest.json"
            self._stage(job_id, "base_models", 55, "Cinco modelos por cinco semillas completados.")
            stacking = run_finance_stacking_experiment(base_manifest_path=base_path)
            stacking_path = Path(stacking["manifest"]["path"])
            self._stage(job_id, "stacking", 65, "Stacking temporal OOF completado.")

            def validation_progress(event: dict[str, Any]) -> None:
                self._check_control(job_id)
                completed, total = int(event.get("completedUnits", 0)), max(int(event.get("totalUnits", 1)), 1)
                self._progress(job_id, "validation", 65 + 15 * completed / total, str(event.get("message", "Validando.")))

            validation = run_finance_temporal_validation(base_manifest_path=base_path, stacking_manifest_path=stacking_path, demo_max_train_rows=None, demo_max_validation_rows=None, bootstrap_iterations=int(config["bootstrapIterations"]), progress_callback=validation_progress)
            validation_path = Path(validation["manifest"]["path"])
            self._stage(job_id, "validation", 80, "Validacion cronologica completa.")
            diversity = run_finance_diversity_ablation(validation_manifest_path=validation_path, bootstrap_iterations=int(config["bootstrapIterations"]))
            diversity_path = Path(diversity["manifest"]["path"])
            self._stage(job_id, "diversity", 88, "Diversidad y ablacion completadas.")
            revalidation = run_finance_optimized_stacking_revalidation(diversity_manifest_path=diversity_path, bootstrap_iterations=int(config["bootstrapIterations"]))
            self._stage(job_id, "revalidation", 95, "Stacking optimizado revalidado en bloque independiente.")
            audit = audit_finance_freeze_readiness(validation_manifest_path=validation_path, diversity_manifest_path=diversity_path, verify_artifacts=True)
            if not audit["ready"]:
                raise ValueError("La corrida completa no supera el congelamiento: " + ", ".join(audit["blockingCheckIds"]))
            frozen = create_finance_freeze_package(validation_manifest_path=validation_path, diversity_manifest_path=diversity_path)
            self._stage(job_id, "freeze", 100, "Pipeline sellado; test aun sin autorizacion.")
            self._update(job_id, status="completed", finishedAt=_now(), ownerPid=None, result={"baseRunId": execution, "validationRunId": validation["runId"], "revalidationRunId": revalidation["runId"], "freezeId": frozen["freezeId"]}, progress={"stage": "completed", "percent": 100.0, "message": "Protocolo de tesis completado sin usar test."})
        except FinanceThesisPaused:
            percent = self.get(job_id)["progress"]["percent"]
            self._update(job_id, status="paused", finishedAt=_now(), ownerPid=None, error=None, progress={"stage": "paused", "percent": percent, "message": "Protocolo pausado en una unidad segura; reanude desde checkpoints."})
        except FinanceThesisCancelled as exc:
            self._update(job_id, status="cancelled", finishedAt=_now(), ownerPid=None, error=str(exc), progress={"stage": "cancelled", "percent": self.get(job_id)["progress"]["percent"], "message": str(exc)})
        except Exception as exc:
            self._update(job_id, status="failed", finishedAt=_now(), ownerPid=None, error=str(exc), progress={"stage": "failed", "percent": self.get(job_id)["progress"]["percent"], "message": "El protocolo se detuvo; puede reanudarse desde checkpoints verificados."})

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
            raise FinanceThesisPaused("Pausa financiera solicitada por el usuario.")
        if job.get("cancelRequested"):
            raise FinanceThesisCancelled("Protocolo financiero cancelado por el usuario.")

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
            job.setdefault("pauseRequested", False)
            recently_updated = time.time() - path.stat().st_mtime < 1_800
            if job.get("status") in {"queued", "running"} and not recently_updated and not _owner_is_alive(job):
                job.update(status="interrupted", finishedAt=_now(), ownerPid=None, error="El backend se reinicio; reanude para usar checkpoints.")
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
        job.setdefault("pauseRequested", False)
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


finance_thesis_orchestrator = FinanceThesisOrchestrator(max_workers=1)
