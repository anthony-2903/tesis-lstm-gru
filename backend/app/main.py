from __future__ import annotations

import secrets
from typing import Literal

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.config import (
    API_WRITE_TOKEN,
    CORS_ALLOWED_ORIGIN_REGEX,
    CORS_ALLOWED_ORIGINS,
    EXPERIMENTS_DIR,
    RESULTS_DIR,
    ensure_dirs,
)
from app.data_lake import get_data_lake_records, get_data_lake_summary, ingest_data_lake
from app.external_sources import fetch_external_data, get_source_catalog
from app.finance.benchmark import get_finance_dataset_status, prepare_finance_benchmark
from app.finance.diversity import get_latest_finance_diversity_ablation, run_finance_diversity_ablation
from app.finance.freeze import FinanceFreezeGateError, audit_finance_freeze_readiness, create_finance_freeze_package, get_latest_finance_freeze
from app.finance.jobs import finance_job_manager
from app.finance.models import FINANCE_MODEL_IDS
from app.finance.real_data import (
    get_real_finance_data_status,
    get_real_finance_source_catalog,
    initialize_real_finance_template,
    prepare_real_finance_dataset,
)
from app.finance.revalidation import get_latest_finance_optimized_revalidation, run_finance_optimized_stacking_revalidation
from app.finance.sequences import get_finance_sequence_status, prepare_finance_sequence_protocol
from app.finance.service import get_latest_finance_experiment, list_finance_experiments
from app.finance.stacking import get_latest_finance_stacking, run_finance_stacking_experiment
from app.finance.market_freeze import (
    FinanceMarketFreezeGateError,
    audit_finance_market_freeze_readiness,
    create_finance_market_freeze_package,
    get_latest_finance_market_freeze,
)
from app.finance.market_pipeline import get_latest_finance_market_experiment, get_latest_finance_market_revalidation
from app.finance.market_thesis_orchestrator import finance_market_thesis_orchestrator as finance_thesis_orchestrator
from app.finance.mef_market_data import get_mef_market_dataset_status, prepare_mef_market_dataset
from app.finance.validation import get_latest_finance_temporal_validation
from app.finance.validation_jobs import finance_validation_job_manager
from app.energy.models import THESIS_MODEL_IDS
from app.energy.jobs import energy_job_manager
from app.energy.data_jobs import energy_data_manager
from app.energy.ingestion import get_energy_dataset_status
from app.energy.freeze import EnergyFreezeGateError, audit_energy_freeze_readiness, create_energy_freeze_package, get_latest_energy_freeze
from app.energy.revalidation import get_latest_energy_optimized_revalidation, run_energy_optimized_stacking_revalidation
from app.energy.service import get_latest_energy_experiment, list_energy_experiments
from app.energy.thesis_orchestrator import energy_thesis_orchestrator
from app.phishing.data_jobs import phishing_data_manager
from app.phishing.data import get_phishing_sequence_status, prepare_phishing_sequence_protocol
from app.phishing.curation import get_phishing_curation_status, initialize_phishing_curation_template
from app.phishing.diversity import get_latest_phishing_diversity_ablation, run_phishing_diversity_ablation
from app.phishing.freeze import FreezeGateError, audit_phishing_freeze_readiness, create_phishing_freeze_package, get_latest_phishing_freeze
from app.phishing.ingestion import get_phishing_dataset_status
from app.phishing.jobs import phishing_job_manager
from app.phishing.models import PHISHING_MODEL_IDS
from app.phishing.runtime import inspect_phishing_training_runtime
from app.phishing.service import get_latest_phishing_experiment, list_phishing_experiments
from app.phishing.stacking import get_latest_phishing_stacking, run_phishing_stacking_experiment
from app.phishing.thesis_orchestrator import phishing_thesis_orchestrator
from app.phishing.validation import get_latest_phishing_external_validation
from app.phishing.validation_jobs import phishing_validation_job_manager
from app.scientific_data_summary import get_scientific_data_summary
from app.pipeline import run_pipeline
from app.thesis_status import get_thesis_status
from app.thesis_results import get_thesis_results_summary
from app.utils import read_json


app = FastAPI(title="Tesis LSTM GRU Backend", version="0.1.0")
VALID_DOMAINS = {"phishing", "energia", "finanzas"}


@app.middleware("http")
async def protect_expensive_mutations(request: Request, call_next):
    if request.url.path.startswith("/api/") and request.method in {"POST", "PUT", "PATCH", "DELETE"} and API_WRITE_TOKEN:
        supplied = request.headers.get("x-api-key", "")
        if not supplied or not secrets.compare_digest(supplied, API_WRITE_TOKEN):
            response = JSONResponse(status_code=401, content={"detail": "Credencial de escritura invalida o ausente."})
        else:
            response = await call_next(request)
    else:
        response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    return response


class EnergyExperimentRequest(BaseModel):
    protocol: Literal["demo", "thesis"] = "demo"
    source: Literal["sample", "silver"] = "sample"
    window: int = Field(12, ge=2, le=168)
    horizon: int = Field(1, ge=1, le=24)
    gapSteps: int = Field(6, ge=0, le=168)
    folds: int = Field(3, ge=3, le=10)
    epochs: int = Field(1, ge=1, le=200)
    batchSize: int = Field(16, ge=1, le=512)
    seeds: list[int] = Field(default_factory=lambda: [42], min_length=1, max_length=5)
    modelIds: list[Literal["lstm", "gru", "brnn", "tcn", "transformer"]] = Field(
        default_factory=lambda: list(THESIS_MODEL_IDS), min_length=2, max_length=5
    )


class EnergyDataPreparationRequest(BaseModel):
    force: bool = False


class EnergyThesisProtocolRequest(BaseModel):
    window: int = Field(24, ge=2, le=168)
    horizon: int = Field(1, ge=1, le=24)
    gapSteps: int = Field(24, ge=0, le=168)
    folds: int = Field(5, ge=5, le=10)
    epochs: int = Field(20, ge=1, le=200)
    batchSize: int = Field(32, ge=1, le=512)
    seeds: list[int] = Field(default_factory=lambda: [42, 101, 202, 303, 404], min_length=5, max_length=5)
    bootstrapIterations: int = Field(500, ge=300, le=5_000)


class PhishingDataPreparationRequest(BaseModel):
    force: bool = False
    perClass: int = Field(10_000, ge=1_000, le=100_000)
    includeAcademicSources: bool = False


class PhishingSequencePreparationRequest(BaseModel):
    folds: int = Field(5, ge=3, le=10)
    seed: int = 42
    maxVocabulary: int = Field(256, ge=4, le=2_048)
    lengthPercentile: float = Field(99.0, ge=50.0, le=100.0)
    maxLengthCap: int = Field(512, ge=8, le=4_096)


class PhishingExperimentRequest(BaseModel):
    protocol: Literal["demo", "thesis"] = "demo"
    epochs: int = Field(1, ge=1, le=200)
    batchSize: int = Field(64, ge=1, le=512)
    patience: int = Field(2, ge=0, le=20)
    seeds: list[int] = Field(default_factory=lambda: [42], min_length=1, max_length=5)
    modelIds: list[Literal["lstm", "gru", "brnn", "tcn", "transformer"]] = Field(
        default_factory=lambda: list(PHISHING_MODEL_IDS), min_length=1, max_length=5
    )
    demoMaxRows: int | None = Field(2_000, ge=500, le=14_000)


class PhishingThesisProtocolRequest(BaseModel):
    epochs: int = Field(20, ge=1, le=200)
    batchSize: int = Field(64, ge=1, le=2_048)
    patience: int = Field(4, ge=0, le=20)
    seeds: list[int] = Field(default_factory=lambda: [42, 101, 202, 303, 404], min_length=5, max_length=5)
    bootstrapIterations: int = Field(500, ge=300, le=5_000)


class FinanceBenchmarkRequest(BaseModel):
    days: int = Field(100, ge=30, le=365)
    customers: int = Field(500, ge=25, le=10_000)
    terminals: int = Field(200, ge=10, le=20_000)
    seed: int = 42


class FinanceRealTemplateRequest(BaseModel):
    adapter: Literal["ieee_cis", "ulb_worldline"] = "ieee_cis"
    force: bool = False


class FinanceRealPreparationRequest(BaseModel):
    minimumRows: int = Field(100_000, ge=1_000, le=2_000_000)
    minimumFraudPerSplit: int = Field(100, ge=1, le=100_000)
    minimumEntities: int = Field(1_000, ge=1, le=1_000_000)


class FinanceSequencePreparationRequest(BaseModel):
    window: int = Field(10, ge=2, le=100)
    folds: int = Field(5, ge=5, le=5)
    purgeDays: int = Field(1, ge=0, le=14)


class FinanceExperimentRequest(BaseModel):
    protocol: Literal["demo", "thesis"] = "demo"
    epochs: int = Field(1, ge=1, le=200)
    batchSize: int = Field(128, ge=1, le=2_048)
    patience: int = Field(1, ge=0, le=20)
    seeds: list[int] = Field(default_factory=lambda: [42], min_length=1, max_length=5)
    modelIds: list[Literal["lstm", "gru", "brnn", "tcn", "transformer"]] = Field(
        default_factory=lambda: list(FINANCE_MODEL_IDS), min_length=1, max_length=5
    )
    demoMaxRowsPerFold: int | None = Field(1_000, ge=100, le=10_000)


class FinanceValidationRequest(BaseModel):
    demoMaxTrainRows: int | None = Field(8_000, ge=500, le=100_000)
    demoMaxValidationRows: int | None = Field(None, ge=500, le=100_000)
    bootstrapIterations: int = Field(500, ge=100, le=5_000)


class FinanceDiversityRequest(BaseModel):
    bootstrapIterations: int = Field(300, ge=100, le=5_000)


class FinanceThesisProtocolRequest(BaseModel):
    window: int = Field(20, ge=2, le=252)
    horizon: int = Field(1, ge=1, le=20)
    gapSteps: int = Field(5, ge=0, le=60)
    folds: int = Field(5, ge=5, le=10)
    epochs: int = Field(20, ge=1, le=200)
    batchSize: int = Field(32, ge=1, le=512)
    seeds: list[int] = Field(default_factory=lambda: [42, 101, 202, 303, 404], min_length=5, max_length=5)
    bootstrapIterations: int = Field(500, ge=300, le=5_000)

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(CORS_ALLOWED_ORIGINS),
    allow_origin_regex=CORS_ALLOWED_ORIGIN_REGEX or None,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def load_artifact(name: str) -> dict:
    ensure_dirs()
    path = RESULTS_DIR / f"{name}.json"
    if not path.exists():
        run_pipeline(mode="sample")
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"No existe artefacto {name}")
    return read_json(path)


def load_domain_artifact(name: str, domain: str | None = None) -> dict:
    if not domain:
        return load_artifact(name)
    normalized = domain.strip().lower()
    if normalized in {"phishtank", "phishing"}:
        normalized = "phishing"
    if normalized not in VALID_DOMAINS:
        raise HTTPException(status_code=400, detail=f"Dominio no soportado: {domain}")
    return load_artifact(f"{name}_{normalized}")


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api")
def api_index() -> dict[str, object]:
    return {
        "status": "ok",
        "message": "Backend activo. Usa esta URL base para el frontend: /api",
        "endpoints": [
            "/api/health",
            "/api/dashboard",
            "/api/dashboard?domain=phishing",
            "/api/analysis",
            "/api/analysis?domain=energia",
            "/api/comparison",
            "/api/history",
            "/api/xai",
            "/api/xai?domain=finanzas",
            "/api/domains",
            "/api/thesis/status",
            "/api/thesis/results-summary",
            "/api/external-sources",
            "/api/external-data?domain=phishing&limit=100",
            "/api/data-lake/summary",
            "/api/data-lake/records?domain=phishing&page=1&pageSize=100",
            "/api/training-manifest",
            "/api/metrics-summary",
            "/api/experiments",
            "/api/energy/experiments/latest",
            "/api/energy/experiments",
            "/api/energy/experiments/run",
            "/api/energy/jobs/latest",
            "/api/energy/data",
            "/api/energy/data/prepare",
            "/api/energy/revalidation/latest",
            "/api/energy/revalidation/run",
            "/api/energy/freeze/readiness",
            "/api/energy/freeze/latest",
            "/api/energy/freeze",
            "/api/energy/thesis/latest",
            "/api/energy/thesis/run",
            "/api/phishing/data",
            "/api/phishing/data/prepare",
            "/api/phishing/data/curation",
            "/api/phishing/data/curation/template",
            "/api/phishing/sequences",
            "/api/phishing/sequences/prepare",
            "/api/phishing/experiments/latest",
            "/api/phishing/experiments/run",
            "/api/phishing/runtime/preflight",
            "/api/phishing/jobs/latest",
            "/api/phishing/stacking/latest",
            "/api/phishing/stacking/run",
            "/api/phishing/validation/latest",
            "/api/phishing/validation/run",
            "/api/phishing/validation/jobs/latest",
            "/api/phishing/diversity/latest",
            "/api/phishing/diversity/run",
            "/api/phishing/freeze/readiness",
            "/api/phishing/freeze/latest",
            "/api/phishing/freeze/create",
            "/api/phishing/thesis/latest",
            "/api/phishing/thesis/run",
            "/api/finance/data",
            "/api/finance/data/prepare",
            "/api/finance/market-data",
            "/api/finance/market-data/prepare",
            "/api/finance/market-experiments/latest",
            "/api/finance/market-revalidation/latest",
            "/api/finance/market-freeze/readiness",
            "/api/finance/market-freeze/latest",
            "/api/finance/market-freeze/create",
            "/api/finance/real-data/catalog",
            "/api/finance/real-data",
            "/api/finance/real-data/template",
            "/api/finance/real-data/prepare",
            "/api/finance/sequences",
            "/api/finance/sequences/prepare",
            "/api/finance/experiments/latest",
            "/api/finance/experiments",
            "/api/finance/experiments/run",
            "/api/finance/jobs/latest",
            "/api/finance/stacking/latest",
            "/api/finance/stacking/run",
            "/api/finance/validation/latest",
            "/api/finance/validation/run",
            "/api/finance/validation/jobs/latest",
            "/api/finance/diversity/latest",
            "/api/finance/diversity/run",
            "/api/finance/revalidation/latest",
            "/api/finance/revalidation/run",
            "/api/finance/freeze/readiness",
            "/api/finance/freeze/latest",
            "/api/finance/freeze/create",
            "/api/finance/thesis/latest",
            "/api/finance/thesis/run",
            "/api/ai-analysis?type=general",
        ],
    }


@app.post("/api/pipeline/run")
def run_pipeline_endpoint(mode: Literal["sample", "remote"] = "sample", limit: int = Query(10000, ge=100, le=50000)) -> dict[str, str | int]:
    run_pipeline(mode=mode, limit=limit)
    return {"status": "completed", "mode": mode, "limit": limit}


@app.get("/api/dashboard")
def dashboard(domain: str | None = Query(None)) -> dict:
    return load_domain_artifact("dashboard", domain)


@app.get("/api/analysis")
def analysis(domain: str | None = Query(None)) -> dict:
    return load_domain_artifact("analysis", domain)


@app.get("/api/comparison")
def comparison(domain: str | None = Query(None)) -> dict:
    return load_domain_artifact("comparison", domain)


@app.get("/api/history")
def history(domain: str | None = Query(None)) -> dict:
    return load_domain_artifact("history", domain)


@app.get("/api/xai")
def xai(domain: str | None = Query(None)) -> dict:
    return load_domain_artifact("xai", domain)


@app.get("/api/domains")
def domains() -> dict:
    return load_artifact("domains")


@app.get("/api/thesis/status")
def thesis_status() -> dict:
    return get_thesis_status()


@app.get("/api/thesis/results-summary")
def thesis_results_summary() -> dict:
    return get_thesis_results_summary()


@app.get("/api/scientific-data-summary")
def scientific_data_summary() -> dict:
    return get_scientific_data_summary()


@app.get("/api/external-sources")
def external_sources(domain: str | None = Query(None)) -> dict:
    try:
        return get_source_catalog(domain)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/external-data")
def external_data(domain: str = Query(...), limit: int = Query(100, ge=1, le=20000)) -> dict:
    try:
        return fetch_external_data(domain, limit)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/data-lake/ingest")
def data_lake_ingest(domain: str = Query("all"), target: int = Query(5000, ge=100, le=20000)) -> dict:
    try:
        return ingest_data_lake(domain, target)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/data-lake/summary")
def data_lake_summary() -> dict:
    return get_data_lake_summary()


@app.get("/api/data-lake/records")
def data_lake_records(
    domain: str = Query(...),
    page: int = Query(1, ge=1),
    page_size: int = Query(100, alias="pageSize", ge=10, le=500),
) -> dict:
    try:
        return get_data_lake_records(domain, page, page_size)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/training-manifest")
def training_manifest() -> dict:
    return load_artifact("training_manifest")


@app.get("/api/metrics-summary")
def metrics_summary() -> dict:
    return load_artifact("metrics_summary")


@app.get("/api/experiments")
def experiments() -> dict:
    ensure_dirs()
    items = []
    for directory in sorted(EXPERIMENTS_DIR.iterdir(), key=lambda path: path.stat().st_mtime, reverse=True):
        if not directory.is_dir():
            continue
        manifest_path = directory / "manifest.json"
        if manifest_path.exists():
            manifest = read_json(manifest_path)
            metrics_path = directory / "metrics_summary.json"
            metrics = read_json(metrics_path) if metrics_path.exists() else manifest.get("metricsSummary", {})
            items.append(
                {
                    "runId": manifest.get("runId", directory.name),
                    "mode": manifest.get("mode"),
                    "limit": manifest.get("limit"),
                    "createdAt": manifest.get("createdAt"),
                    "domainTotals": manifest.get("domainTotals", {}),
                    "metricsSummary": metrics,
                    "path": str(directory),
                }
            )
    return {"items": items}


@app.get("/api/energy/experiments/latest")
def latest_energy_experiment() -> dict:
    return get_latest_energy_experiment()


@app.get("/api/energy/experiments")
def energy_experiments() -> dict:
    return list_energy_experiments()


@app.get("/api/energy/data")
def energy_data_status() -> dict:
    return get_energy_dataset_status()


@app.get("/api/energy/data/jobs/latest")
def latest_energy_data_job() -> dict:
    return energy_data_manager.latest() or {"available": False, "message": "No existen preparaciones de datos."}


@app.post("/api/energy/data/prepare", status_code=202)
def prepare_energy_data(request: EnergyDataPreparationRequest) -> dict:
    return energy_data_manager.submit(force=request.force)


@app.get("/api/finance/data")
def finance_data_status() -> dict:
    return get_finance_dataset_status()


@app.post("/api/finance/data/prepare")
def prepare_finance_data(request: FinanceBenchmarkRequest) -> dict:
    prepare_finance_benchmark(
        days=request.days,
        customers=request.customers,
        terminals=request.terminals,
        seed=request.seed,
    )
    return get_finance_dataset_status()


@app.get("/api/finance/market-data")
def finance_market_data_status() -> dict:
    return get_mef_market_dataset_status()


@app.post("/api/finance/market-data/prepare")
def prepare_finance_market_data(request: EnergyDataPreparationRequest) -> dict:
    try:
        prepare_mef_market_dataset(force_download=request.force)
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return get_mef_market_dataset_status()


@app.get("/api/finance/market-experiments/latest")
def latest_finance_market_experiment() -> dict:
    return get_latest_finance_market_experiment()


@app.get("/api/finance/market-revalidation/latest")
def latest_finance_market_revalidation() -> dict:
    return get_latest_finance_market_revalidation()


@app.get("/api/finance/market-freeze/readiness")
def finance_market_freeze_readiness() -> dict:
    try:
        return audit_finance_market_freeze_readiness(verify_artifacts=False)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.get("/api/finance/market-freeze/latest")
def latest_finance_market_freeze() -> dict:
    return get_latest_finance_market_freeze()


@app.post("/api/finance/market-freeze/create")
def create_finance_market_freeze() -> dict:
    try:
        return create_finance_market_freeze_package()
    except FinanceMarketFreezeGateError as exc:
        raise HTTPException(status_code=409, detail={"message": str(exc), "audit": exc.audit}) from exc
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.get("/api/finance/real-data/catalog")
def finance_real_source_catalog() -> dict:
    return get_real_finance_source_catalog()


@app.get("/api/finance/real-data")
def finance_real_data_status() -> dict:
    return get_real_finance_data_status()


@app.post("/api/finance/real-data/template")
def create_finance_real_template(request: FinanceRealTemplateRequest) -> dict:
    initialize_real_finance_template(adapter=request.adapter, force=request.force)
    return get_real_finance_data_status()


@app.post("/api/finance/real-data/prepare")
def prepare_finance_real_data(request: FinanceRealPreparationRequest) -> dict:
    try:
        prepare_real_finance_dataset(
            minimum_rows=request.minimumRows,
            minimum_fraud_per_split=request.minimumFraudPerSplit,
            minimum_entities=request.minimumEntities,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"realData": get_real_finance_data_status(), "dataset": get_finance_dataset_status()}


@app.get("/api/finance/sequences")
def finance_sequence_status() -> dict:
    return get_finance_sequence_status()


@app.post("/api/finance/sequences/prepare")
def prepare_finance_sequences(request: FinanceSequencePreparationRequest) -> dict:
    dataset = get_finance_dataset_status()
    if not dataset.get("readyForPipelinePilot"):
        raise HTTPException(status_code=409, detail="Primero debe prepararse un benchmark financiero apto para pipeline.")
    prepare_finance_sequence_protocol(
        window=request.window,
        folds=request.folds,
        purge_days=request.purgeDays,
    )
    return get_finance_sequence_status()


@app.get("/api/finance/experiments/latest")
def latest_finance_experiment() -> dict:
    return get_latest_finance_experiment()


@app.get("/api/finance/experiments")
def finance_experiments() -> dict:
    return list_finance_experiments()


@app.post("/api/finance/experiments/run", status_code=202)
def run_finance_experiment(request: FinanceExperimentRequest) -> dict:
    seeds = list(dict.fromkeys(request.seeds))
    model_ids = list(dict.fromkeys(request.modelIds))
    if request.protocol == "thesis":
        if len(seeds) < 5 or set(model_ids) != set(FINANCE_MODEL_IDS) or request.demoMaxRowsPerFold is not None:
            raise HTTPException(
                status_code=400,
                detail="El protocolo thesis exige cinco semillas, los cinco modelos y demoMaxRowsPerFold=null.",
            )
        sequence_status = get_finance_sequence_status()
        if not sequence_status.get("readyForThesisTraining"):
            raise HTTPException(
                status_code=409,
                detail="El benchmark financiero sintético permite pilotos, pero no una corrida final de tesis.",
            )
    return finance_job_manager.submit(
        {
            "protocol": request.protocol,
            "epochs": request.epochs,
            "batchSize": request.batchSize,
            "patience": request.patience,
            "seeds": seeds,
            "modelIds": model_ids,
            "demoMaxRowsPerFold": request.demoMaxRowsPerFold,
        }
    )


@app.get("/api/finance/jobs/latest")
def latest_finance_job() -> dict:
    return finance_job_manager.latest() or {"available": False, "message": "No existen trabajos financieros."}


@app.get("/api/finance/jobs")
def finance_jobs(limit: int = Query(25, ge=1, le=100)) -> dict:
    return finance_job_manager.list(limit)


@app.get("/api/finance/jobs/{job_id}")
def finance_job(job_id: str) -> dict:
    job = finance_job_manager.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Trabajo financiero no encontrado.")
    return job


@app.post("/api/finance/jobs/{job_id}/cancel")
def cancel_finance_job(job_id: str) -> dict:
    job = finance_job_manager.cancel(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Trabajo financiero no encontrado.")
    return job


@app.post("/api/finance/jobs/{job_id}/resume", status_code=202)
def resume_finance_job(job_id: str) -> dict:
    try:
        job = finance_job_manager.resume(job_id)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if job is None:
        raise HTTPException(status_code=404, detail="Trabajo financiero no encontrado.")
    return job


@app.get("/api/finance/stacking/latest")
def latest_finance_stacking() -> dict:
    return get_latest_finance_stacking()


@app.post("/api/finance/stacking/run")
def run_finance_stacking() -> dict:
    try:
        return {"available": True, **run_finance_stacking_experiment()}
    except FileNotFoundError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/api/finance/validation/latest")
def latest_finance_validation() -> dict:
    return get_latest_finance_temporal_validation()


@app.post("/api/finance/validation/run", status_code=202)
def run_finance_validation(request: FinanceValidationRequest) -> dict:
    latest_base = get_latest_finance_experiment()
    latest_stacking = get_latest_finance_stacking()
    if not latest_base.get("available") or not latest_stacking.get("available"):
        raise HTTPException(status_code=409, detail="Primero complete los cinco modelos base y el Stacking financiero.")
    try:
        return finance_validation_job_manager.submit({
            "demoMaxTrainRows": request.demoMaxTrainRows,
            "demoMaxValidationRows": request.demoMaxValidationRows,
            "bootstrapIterations": request.bootstrapIterations,
            "seeds": latest_base.get("configuration", {}).get("seeds", [42]),
        })
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.get("/api/finance/validation/jobs/latest")
def latest_finance_validation_job() -> dict:
    return finance_validation_job_manager.latest() or {"available": False, "message": "No existen validaciones financieras."}


@app.get("/api/finance/validation/jobs/{job_id}")
def finance_validation_job(job_id: str) -> dict:
    job = finance_validation_job_manager.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Validacion financiera no encontrada.")
    return job


@app.post("/api/finance/validation/jobs/{job_id}/cancel")
def cancel_finance_validation_job(job_id: str) -> dict:
    job = finance_validation_job_manager.cancel(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Validacion financiera no encontrada.")
    return job


@app.get("/api/finance/diversity/latest")
def latest_finance_diversity_ablation() -> dict:
    return get_latest_finance_diversity_ablation()


@app.post("/api/finance/diversity/run")
def run_finance_diversity(request: FinanceDiversityRequest) -> dict:
    try:
        return {
            "available": True,
            **run_finance_diversity_ablation(bootstrap_iterations=request.bootstrapIterations),
        }
    except FileNotFoundError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/api/finance/revalidation/latest")
def latest_finance_optimized_revalidation() -> dict:
    return get_latest_finance_optimized_revalidation()


@app.post("/api/finance/revalidation/run")
def run_finance_optimized_revalidation(request: FinanceDiversityRequest) -> dict:
    try:
        return {"available": True, **run_finance_optimized_stacking_revalidation(bootstrap_iterations=request.bootstrapIterations)}
    except FileNotFoundError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/api/finance/freeze/readiness")
def finance_freeze_readiness() -> dict:
    return audit_finance_freeze_readiness(verify_artifacts=False)


@app.get("/api/finance/freeze/latest")
def latest_finance_freeze() -> dict:
    return get_latest_finance_freeze()


@app.post("/api/finance/freeze/create")
def create_finance_freeze() -> dict:
    try:
        return create_finance_freeze_package()
    except FinanceFreezeGateError as exc:
        raise HTTPException(status_code=409, detail={"message": str(exc), "audit": exc.audit}) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.get("/api/finance/thesis/latest")
def latest_finance_thesis_protocol() -> dict:
    return finance_thesis_orchestrator.latest() or {"available": False, "message": "No existe un protocolo financiero de tesis."}


@app.post("/api/finance/thesis/run", status_code=202)
def run_finance_thesis_protocol(request: FinanceThesisProtocolRequest) -> dict:
    try:
        return finance_thesis_orchestrator.submit(request.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/api/finance/thesis/{job_id}/resume", status_code=202)
def resume_finance_thesis_protocol(job_id: str) -> dict:
    try:
        job = finance_thesis_orchestrator.resume(job_id)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if job is None:
        raise HTTPException(status_code=404, detail="Protocolo financiero no encontrado.")
    return job


@app.post("/api/finance/thesis/{job_id}/cancel")
def cancel_finance_thesis_protocol(job_id: str) -> dict:
    job = finance_thesis_orchestrator.cancel(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Protocolo financiero no encontrado.")
    return job


@app.post("/api/finance/thesis/{job_id}/pause")
def pause_finance_thesis_protocol(job_id: str) -> dict:
    job = finance_thesis_orchestrator.pause(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Protocolo financiero no encontrado.")
    return job


@app.get("/api/phishing/data")
def phishing_data_status() -> dict:
    return get_phishing_dataset_status()


@app.get("/api/phishing/data/jobs/latest")
def latest_phishing_data_job() -> dict:
    return phishing_data_manager.latest() or {"available": False, "message": "No existen preparaciones de datos."}


@app.post("/api/phishing/data/prepare", status_code=202)
def prepare_phishing_data(request: PhishingDataPreparationRequest) -> dict:
    return phishing_data_manager.submit(
        force=request.force,
        per_class=request.perClass,
        include_academic_sources=request.includeAcademicSources,
    )


@app.get("/api/phishing/data/curation")
def phishing_curation_status() -> dict:
    return get_phishing_curation_status()


@app.post("/api/phishing/data/curation/template")
def create_phishing_curation_template() -> dict:
    return initialize_phishing_curation_template()


@app.get("/api/phishing/sequences")
def phishing_sequence_status() -> dict:
    return get_phishing_sequence_status()


@app.post("/api/phishing/sequences/prepare")
def prepare_phishing_sequences(request: PhishingSequencePreparationRequest) -> dict:
    dataset = get_phishing_dataset_status()
    if not dataset.get("readyForPipelinePilot"):
        raise HTTPException(status_code=409, detail="Primero debe prepararse un dataset de phishing apto para pipeline.")
    return prepare_phishing_sequence_protocol(
        folds=request.folds,
        seed=request.seed,
        max_vocabulary=request.maxVocabulary,
        length_percentile=request.lengthPercentile,
        max_length_cap=request.maxLengthCap,
    )


@app.get("/api/phishing/experiments/latest")
def latest_phishing_experiment() -> dict:
    return get_latest_phishing_experiment()


@app.get("/api/phishing/experiments")
def phishing_experiments() -> dict:
    return list_phishing_experiments()


@app.get("/api/phishing/runtime/preflight")
def phishing_runtime_preflight(protocol: Literal["demo", "thesis"] = "thesis") -> dict:
    return inspect_phishing_training_runtime(protocol=protocol)


@app.post("/api/phishing/experiments/run", status_code=202)
def run_phishing_experiment(request: PhishingExperimentRequest) -> dict:
    seeds = list(dict.fromkeys(request.seeds))
    model_ids = list(dict.fromkeys(request.modelIds))
    if request.protocol == "thesis":
        if len(seeds) < 5 or set(model_ids) != set(PHISHING_MODEL_IDS) or request.demoMaxRows is not None:
            raise HTTPException(
                status_code=400,
                detail="El protocolo thesis exige cinco semillas, los cinco modelos y demoMaxRows=null.",
            )
    return phishing_job_manager.submit({
        "protocol": request.protocol,
        "epochs": request.epochs,
        "batchSize": request.batchSize,
        "patience": request.patience,
        "seeds": seeds,
        "modelIds": model_ids,
        "demoMaxRows": request.demoMaxRows,
    })


@app.get("/api/phishing/jobs/latest")
def latest_phishing_job() -> dict:
    return phishing_job_manager.latest() or {"available": False, "message": "No existen trabajos de phishing."}


@app.get("/api/phishing/jobs")
def phishing_jobs(limit: int = Query(25, ge=1, le=100)) -> dict:
    return phishing_job_manager.list(limit)


@app.get("/api/phishing/jobs/{job_id}")
def phishing_job(job_id: str) -> dict:
    job = phishing_job_manager.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Trabajo de phishing no encontrado.")
    return job


@app.post("/api/phishing/jobs/{job_id}/cancel")
def cancel_phishing_job(job_id: str) -> dict:
    job = phishing_job_manager.cancel(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Trabajo de phishing no encontrado.")
    return job


@app.post("/api/phishing/jobs/{job_id}/resume", status_code=202)
def resume_phishing_job(job_id: str) -> dict:
    try:
        job = phishing_job_manager.resume(job_id)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if job is None:
        raise HTTPException(status_code=404, detail="Trabajo de phishing no encontrado.")
    return job


@app.get("/api/phishing/stacking/latest")
def latest_phishing_stacking() -> dict:
    return get_latest_phishing_stacking()


@app.post("/api/phishing/stacking/run")
def run_phishing_stacking() -> dict:
    try:
        return {"available": True, **run_phishing_stacking_experiment()}
    except FileNotFoundError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/api/phishing/validation/latest")
def latest_phishing_external_validation() -> dict:
    return get_latest_phishing_external_validation()


@app.post("/api/phishing/validation/run", status_code=202)
def run_phishing_external_validation() -> dict:
    return phishing_validation_job_manager.submit()


@app.get("/api/phishing/validation/jobs/latest")
def latest_phishing_validation_job() -> dict:
    return phishing_validation_job_manager.latest() or {"available": False, "message": "No existen trabajos de validación externa."}


@app.get("/api/phishing/validation/jobs/{job_id}")
def phishing_validation_job(job_id: str) -> dict:
    job = phishing_validation_job_manager.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Trabajo de validación externa no encontrado.")
    return job


@app.post("/api/phishing/validation/jobs/{job_id}/cancel")
def cancel_phishing_validation_job(job_id: str) -> dict:
    job = phishing_validation_job_manager.cancel(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Trabajo de validación externa no encontrado.")
    return job


@app.get("/api/phishing/diversity/latest")
def latest_phishing_diversity_ablation() -> dict:
    return get_latest_phishing_diversity_ablation()


@app.post("/api/phishing/diversity/run")
def run_phishing_diversity() -> dict:
    try:
        return {"available": True, **run_phishing_diversity_ablation()}
    except FileNotFoundError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/api/phishing/freeze/readiness")
def phishing_freeze_readiness() -> dict:
    return audit_phishing_freeze_readiness(verify_artifacts=False)


@app.get("/api/phishing/freeze/latest")
def latest_phishing_freeze() -> dict:
    return get_latest_phishing_freeze()


@app.post("/api/phishing/freeze/create")
def create_phishing_freeze() -> dict:
    try:
        return create_phishing_freeze_package()
    except FreezeGateError as exc:
        summary = "; ".join(exc.audit.get("reasons", [])[:6])
        raise HTTPException(status_code=409, detail=f"La puerta experimental está bloqueada. {summary}") from exc
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.get("/api/phishing/thesis/latest")
def latest_phishing_thesis_protocol() -> dict:
    return phishing_thesis_orchestrator.latest() or {"available": False, "message": "No existe un protocolo de phishing de tesis."}


@app.post("/api/phishing/thesis/run", status_code=202)
def run_phishing_thesis_protocol(request: PhishingThesisProtocolRequest) -> dict:
    try:
        return phishing_thesis_orchestrator.submit(request.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/api/phishing/thesis/{job_id}/resume", status_code=202)
def resume_phishing_thesis_protocol(job_id: str) -> dict:
    try:
        job = phishing_thesis_orchestrator.resume(job_id)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if job is None:
        raise HTTPException(status_code=404, detail="Protocolo de phishing no encontrado.")
    return job


@app.post("/api/phishing/thesis/{job_id}/cancel")
def cancel_phishing_thesis_protocol(job_id: str) -> dict:
    job = phishing_thesis_orchestrator.cancel(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Protocolo de phishing no encontrado.")
    return job


@app.post("/api/phishing/thesis/{job_id}/pause")
def pause_phishing_thesis_protocol(job_id: str) -> dict:
    job = phishing_thesis_orchestrator.pause(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Protocolo de phishing no encontrado.")
    return job


@app.get("/api/energy/revalidation/latest")
def latest_energy_revalidation() -> dict:
    return get_latest_energy_optimized_revalidation()


@app.post("/api/energy/revalidation/run")
def run_energy_revalidation(bootstrapIterations: int = Query(500, ge=300, le=5_000)) -> dict:
    try:
        return run_energy_optimized_stacking_revalidation(bootstrap_iterations=bootstrapIterations)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.get("/api/energy/freeze/readiness")
def energy_freeze_readiness(verifyArtifacts: bool = Query(False)) -> dict:
    return audit_energy_freeze_readiness(verify_artifacts=verifyArtifacts)


@app.get("/api/energy/freeze/latest")
def latest_energy_freeze() -> dict:
    return get_latest_energy_freeze()


@app.post("/api/energy/freeze")
def freeze_energy_protocol() -> dict:
    try:
        return create_energy_freeze_package()
    except EnergyFreezeGateError as exc:
        raise HTTPException(status_code=409, detail=exc.audit) from exc


@app.get("/api/energy/thesis/latest")
def latest_energy_thesis_protocol() -> dict:
    return energy_thesis_orchestrator.latest() or {"available": False, "message": "No existe protocolo energetico de tesis."}


@app.post("/api/energy/thesis/run", status_code=202)
def run_energy_thesis_protocol(request: EnergyThesisProtocolRequest) -> dict:
    try:
        return energy_thesis_orchestrator.submit(request.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/api/energy/thesis/{job_id}/resume", status_code=202)
def resume_energy_thesis_protocol(job_id: str) -> dict:
    try:
        job = energy_thesis_orchestrator.resume(job_id)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if job is None:
        raise HTTPException(status_code=404, detail="Protocolo energetico no encontrado.")
    return job


@app.post("/api/energy/thesis/{job_id}/cancel")
def cancel_energy_thesis_protocol(job_id: str) -> dict:
    job = energy_thesis_orchestrator.cancel(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Protocolo energetico no encontrado.")
    return job


@app.post("/api/energy/thesis/{job_id}/pause")
def pause_energy_thesis_protocol(job_id: str) -> dict:
    job = energy_thesis_orchestrator.pause(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Protocolo energetico no encontrado.")
    return job


@app.post("/api/energy/experiments/run", status_code=202)
def run_energy_experiment(request: EnergyExperimentRequest) -> dict:
    seeds = list(dict.fromkeys(request.seeds))
    model_ids = list(dict.fromkeys(request.modelIds))
    if len(model_ids) < 2:
        raise HTTPException(status_code=400, detail="Stacking requiere al menos dos modelos base distintos.")
    if request.protocol == "thesis":
        if request.source != "silver":
            raise HTTPException(status_code=400, detail="El protocolo de tesis exige la fuente silver.")
        if request.folds < 5 or len(seeds) < 5 or set(model_ids) != set(THESIS_MODEL_IDS):
            raise HTTPException(
                status_code=400,
                detail="El protocolo de tesis exige cinco folds, cinco semillas y las cinco arquitecturas.",
            )
    return energy_job_manager.submit(
        {
            "protocol": request.protocol,
            "source": request.source,
            "window": request.window,
            "horizon": request.horizon,
            "gapSteps": request.gapSteps,
            "folds": request.folds,
            "epochs": request.epochs,
            "batchSize": request.batchSize,
            "seeds": seeds,
            "modelIds": model_ids,
        }
    )


@app.get("/api/energy/jobs/latest")
def latest_energy_job() -> dict:
    return energy_job_manager.latest() or {"available": False, "message": "No existen trabajos energéticos."}


@app.get("/api/energy/jobs")
def energy_jobs(limit: int = Query(25, ge=1, le=100)) -> dict:
    return energy_job_manager.list(limit)


@app.get("/api/energy/jobs/{job_id}")
def energy_job(job_id: str) -> dict:
    job = energy_job_manager.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Trabajo energético no encontrado.")
    return job


@app.post("/api/energy/jobs/{job_id}/cancel")
def cancel_energy_job(job_id: str) -> dict:
    job = energy_job_manager.cancel(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Trabajo energético no encontrado.")
    return job


@app.get("/api/ai-analysis")
def ai_analysis(type: Literal["general", "phishtank", "energia", "finanzas"] = Query("general")) -> dict[str, str]:
    payload = load_artifact("ai_analysis")
    return {"analysis": str(payload.get(type) or payload.get("general") or "")}
