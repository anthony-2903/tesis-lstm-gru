from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time
from threading import Lock
from typing import Any
from uuid import uuid4

from app.config import EXPERIMENTS_DIR, RESULTS_DIR, ensure_dirs
from app.phishing.data import get_phishing_sequence_status, prepare_phishing_sequence_protocol
from app.phishing.diversity import run_phishing_diversity_ablation
from app.phishing.freeze import audit_phishing_freeze_readiness, create_phishing_freeze_package
from app.phishing.ingestion import get_phishing_dataset_status
from app.phishing.models import PHISHING_MODEL_IDS
from app.phishing.pipeline import run_phishing_oof_experiment
from app.phishing.stacking import run_phishing_stacking_experiment
from app.phishing.validation import run_phishing_external_validation


STOPPED = {"completed", "failed", "cancelled", "interrupted", "paused", "waiting_for_scientific_data"}
DEFAULT_SEEDS = [42, 101, 202, 303, 404]


class PhishingThesisCancelled(RuntimeError):
    pass


class PhishingThesisPaused(RuntimeError):
    pass


class PhishingThesisOrchestrator:
    def __init__(self, *, max_workers: int = 1, job_dir: Path | None = None) -> None:
        ensure_dirs()
        self.job_dir = job_dir or (RESULTS_DIR / "phishing_thesis_jobs")
        self.job_dir.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="phishing-thesis")
        self._jobs: dict[str, dict[str, Any]] = {}
        self._futures: dict[str, Future[Any]] = {}
        self._restore()

    def submit(self, config: dict[str, Any] | None = None, *, execution_run_id: str | None = None, resumed_from_job_id: str | None = None) -> dict[str, Any]:
        self._refresh_jobs_from_disk()
        effective = {"epochs": 20, "batchSize": 64, "patience": 4, "seeds": DEFAULT_SEEDS, "bootstrapIterations": 500, **(config or {})}
        if len(set(effective["seeds"])) < 5:
            raise ValueError("El protocolo de phishing exige cinco semillas distintas.")
        job_id = f"phishing_thesis_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}"
        job = {"jobId": job_id, "executionRunId": execution_run_id or job_id, "resumedFromJobId": resumed_from_job_id, "status": "queued", "createdAt": _now(), "startedAt": None, "finishedAt": None, "cancelRequested": False, "pauseRequested": False, "config": effective, "progress": {"stage": "queued", "percent": 0.0, "message": "Protocolo de phishing en cola."}, "stages": [], "result": None, "blocker": None, "error": None, "testPolicy": {"testSetUsed": False, "testEvaluationAuthorized": False}}
        with self._lock:
            if any(item["status"] in {"queued", "running"} for item in self._jobs.values()):
                raise ValueError("Ya existe un protocolo de phishing activo.")
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
            if not job or job["status"] in STOPPED:
                return deepcopy(job) if job else None
            if job.get("pauseRequested"):
                return deepcopy(job)
            job["cancelRequested"] = True
            job["progress"]["message"] = "Cancelacion solicitada; se detendra en una unidad segura."
            self._persist(job)
            return deepcopy(job)

    def pause(self, job_id: str) -> dict[str, Any] | None:
        self._refresh_job_from_disk(job_id)
        with self._lock:
            job = self._jobs.get(job_id)
            if not job or job["status"] in STOPPED:
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
            if previous["status"] not in {"failed", "cancelled", "interrupted", "paused", "waiting_for_scientific_data"}:
                raise ValueError("Este protocolo no se encuentra detenido.")
            config, execution = deepcopy(previous["config"]), previous["executionRunId"]
        return self.submit(config, execution_run_id=execution, resumed_from_job_id=job_id)

    def shutdown(self, *, wait: bool = True) -> None:
        self._executor.shutdown(wait=wait, cancel_futures=True)

    def _execute(self, job_id: str) -> None:
        self._update(job_id, status="running", startedAt=_now(), ownerPid=os.getpid())
        config, execution = self.get(job_id)["config"], self.get(job_id)["executionRunId"]
        try:
            dataset = get_phishing_dataset_status()
            if not dataset.get("readyForThesisTraining"):
                self._update(job_id, status="waiting_for_scientific_data", finishedAt=_now(), ownerPid=None, blocker={"code": "scientific_phishing_dataset_required", "message": dataset.get("message")}, progress={"stage": "data_gate", "percent": 0.0, "message": "Esperando dataset cientifico; test intacto."})
                return
            self._stage(job_id, "data_gate", 5, "Dataset multifuente cientificamente apto.")
            sequence = get_phishing_sequence_status()
            if not sequence.get("readyForBaseModelTraining") or not sequence.get("lineageCurrent"):
                prepare_phishing_sequence_protocol(folds=5, seed=42, max_vocabulary=256, length_percentile=99.0, max_length_cap=512)
            self._stage(job_id, "sequences", 10, "Cinco folds por dominio y tokenizadores verificados.")

            def base_progress(event: dict[str, Any]) -> None:
                self._check_control(job_id)
                done, total = int(event.get("completedUnits", 0)), max(int(event.get("totalUnits", 1)), 1)
                self._progress(job_id, "base_models", 10 + 50 * done / total, str(event.get("message", "Entrenando phishing.")))

            run_phishing_oof_experiment(protocol="thesis", epochs=int(config["epochs"]), batch_size=int(config["batchSize"]), patience=int(config["patience"]), seeds=tuple(int(value) for value in config["seeds"]), model_ids=PHISHING_MODEL_IDS, demo_max_rows=None, execution_run_id=execution, progress_callback=base_progress)
            base_path = EXPERIMENTS_DIR / execution / "phishing_oof_v1" / "oof_manifest.json"
            self._stage(job_id, "base_models", 60, "Modelos base OOF completos.")
            run_phishing_stacking_experiment(base_manifest_path=base_path)
            self._stage(job_id, "stacking", 70, "Stacking cross-fit completado.")

            def validation_progress(event: dict[str, Any]) -> None:
                self._check_control(job_id)
                done, total = int(event.get("completedUnits", 0)), max(int(event.get("totalUnits", 1)), 1)
                self._progress(job_id, "validation", 70 + 15 * done / total, str(event.get("message", "Validando phishing.")))

            validation_path = base_path.parent / "external_validation_v1" / "external_validation_manifest.json"
            validation = _load_verified_validation(validation_path, base_path=base_path)
            if validation is None:
                validation = run_phishing_external_validation(base_manifest_path=base_path, progress_callback=validation_progress)
                validation_path = _result_manifest_path(validation, fallback=validation_path)
            self._stage(job_id, "validation", 85, "Validacion externa y umbrales congelados.")
            diversity_path = base_path.parent / "diversity_ablation_v1" / "diversity_ablation_manifest.json"
            diversity = _load_verified_diversity(diversity_path, validation_path=validation_path)
            if diversity is None:
                diversity = run_phishing_diversity_ablation(validation_manifest_path=validation_path, bootstrap_iterations=int(config["bootstrapIterations"]))
                diversity_path = _result_manifest_path(diversity, fallback=diversity_path)
            self._stage(job_id, "diversity", 95, "Diversidad y ablacion completadas.")
            audit = audit_phishing_freeze_readiness(validation_manifest_path=validation_path, diversity_manifest_path=diversity_path, verify_artifacts=True)
            if not audit["ready"]:
                raise ValueError("La corrida no supera el congelamiento: " + ", ".join(item["checkId"] for item in audit["failedChecks"]))
            frozen = create_phishing_freeze_package(validation_manifest_path=validation_path, diversity_manifest_path=diversity_path)
            self._stage(job_id, "freeze", 100, "Pipeline sellado sin evaluar test.")
            self._update(job_id, status="completed", finishedAt=_now(), ownerPid=None, result={"baseRunId": execution, "validationRunId": validation["runId"], "diversityRunId": diversity["runId"], "freezeId": frozen["freezeId"]}, progress={"stage": "completed", "percent": 100.0, "message": "Protocolo de phishing completado; test no autorizado."})
        except PhishingThesisPaused:
            percent = self.get(job_id)["progress"]["percent"]
            self._update(job_id, status="paused", finishedAt=_now(), ownerPid=None, error=None, progress={"stage": "paused", "percent": percent, "message": "Protocolo pausado en una unidad segura; reanude desde checkpoints."})
        except PhishingThesisCancelled as exc:
            self._update(job_id, status="cancelled", finishedAt=_now(), ownerPid=None, error=str(exc), progress={"stage": "cancelled", "percent": self.get(job_id)["progress"]["percent"], "message": str(exc)})
        except Exception as exc:
            self._update(job_id, status="failed", finishedAt=_now(), ownerPid=None, error=str(exc), progress={"stage": "failed", "percent": self.get(job_id)["progress"]["percent"], "message": "Protocolo detenido; reanude desde checkpoints."})

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
            raise PhishingThesisPaused("Pausa de phishing solicitada por el usuario.")
        if job.get("cancelRequested"):
            raise PhishingThesisCancelled("Protocolo de phishing cancelado por el usuario.")

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


def _result_manifest_path(result: dict[str, Any], *, fallback: Path) -> Path:
    """Resolve both descriptor-wrapped and direct-manifest result contracts."""
    descriptor = result.get("manifest")
    if isinstance(descriptor, dict) and descriptor.get("path"):
        return Path(descriptor["path"])
    if fallback.is_file():
        return fallback
    raise ValueError(f"El resultado no contiene un manifiesto verificable: {fallback}")


def _load_verified_validation(path: Path, *, base_path: Path) -> dict[str, Any] | None:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(manifest, dict):
        return None
    try:
        recorded_base = Path(manifest.get("baseRun", {}).get("manifestPath", "")).resolve()
    except (OSError, TypeError, ValueError):
        return None
    validation = manifest.get("validation", {})
    if (
        recorded_base != base_path.resolve()
        or manifest.get("status") != "validation_selection_candidate"
        or not manifest.get("comparison")
        or validation.get("outerValidationUsed") is not True
        or validation.get("testSetLocked") is not True
        or validation.get("testFeaturesEncoded") is not False
        or validation.get("testSetUsed") is not False
    ):
        return None
    descriptors = list(_artifact_descriptors(manifest.get("tokenizer")))
    descriptors.extend(_artifact_descriptors(manifest.get("artifacts")))
    if not descriptors or any(not _artifact_matches(descriptor) for descriptor in descriptors):
        return None
    return manifest


def _load_verified_diversity(path: Path, *, validation_path: Path) -> dict[str, Any] | None:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(manifest, dict):
        return None
    try:
        recorded_validation = Path(manifest.get("validationRun", {}).get("manifestPath", "")).resolve()
    except (OSError, TypeError, ValueError):
        return None
    validation = manifest.get("validation", {})
    if (
        recorded_validation != validation_path.resolve()
        or manifest.get("status") != "ablation_candidate"
        or not manifest.get("diversity", {}).get("pairs")
        or not manifest.get("ablation", {}).get("configurations")
        or validation.get("outerValidationUsed") is not True
        or validation.get("testSetLocked") is not True
        or validation.get("testFeaturesEncoded") is not False
        or validation.get("testSetUsed") is not False
    ):
        return None
    descriptors = list(_artifact_descriptors(manifest.get("artifacts")))
    if not descriptors or any(not _artifact_matches(descriptor) for descriptor in descriptors):
        return None
    return manifest


def _artifact_descriptors(value: Any):
    if isinstance(value, dict):
        if {"path", "sha256", "bytes"}.issubset(value):
            yield value
        for nested in value.values():
            yield from _artifact_descriptors(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _artifact_descriptors(nested)


def _artifact_matches(descriptor: dict[str, Any]) -> bool:
    try:
        artifact = Path(descriptor["path"])
        if descriptor.get("kind") == "base_model":
            digest, size = _keras_container_hash(artifact)
        else:
            content = artifact.read_bytes()
            digest, size = hashlib.sha256(content).hexdigest(), len(content)
        return size == int(descriptor["bytes"]) and digest == descriptor["sha256"]
    except (OSError, TypeError, ValueError):
        return False


def _keras_container_hash(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    total = 0
    files = sorted(item for item in path.rglob("*") if item.is_file()) if path.is_dir() else [path]
    for item in files:
        relative = item.relative_to(path).as_posix() if path.is_dir() else item.name
        digest.update(relative.encode("utf-8"))
        with item.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
                total += len(chunk)
    return digest.hexdigest(), total


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


phishing_thesis_orchestrator = PhishingThesisOrchestrator(max_workers=1)
