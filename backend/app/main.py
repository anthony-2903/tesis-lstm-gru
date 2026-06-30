from __future__ import annotations

from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from app.config import RESULTS_DIR, ensure_dirs
from app.data_lake import get_data_lake_records, get_data_lake_summary, ingest_data_lake
from app.external_sources import fetch_external_data, get_source_catalog
from app.pipeline import run_pipeline
from app.utils import read_json


app = FastAPI(title="Tesis LSTM GRU Backend", version="0.1.0")
VALID_DOMAINS = {"phishing", "energia", "finanzas"}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
            "/api/external-sources",
            "/api/external-data?domain=phishing&limit=100",
            "/api/data-lake/summary",
            "/api/data-lake/records?domain=phishing&page=1&pageSize=100",
            "/api/ai-analysis?type=general",
        ],
    }


@app.post("/api/pipeline/run")
def run_pipeline_endpoint(mode: Literal["sample", "remote"] = "sample") -> dict[str, str]:
    run_pipeline(mode=mode)
    return {"status": "completed", "mode": mode}


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


@app.get("/api/external-sources")
def external_sources(domain: str | None = Query(None)) -> dict:
    try:
        return get_source_catalog(domain)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/external-data")
def external_data(domain: str = Query(...), limit: int = Query(100, ge=1, le=500)) -> dict:
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


@app.get("/api/ai-analysis")
def ai_analysis(type: Literal["general", "phishtank", "energia", "finanzas"] = Query("general")) -> dict[str, str]:
    payload = load_artifact("ai_analysis")
    return {"analysis": str(payload.get(type) or payload.get("general") or "")}
