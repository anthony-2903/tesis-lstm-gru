from __future__ import annotations

from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from app.config import RESULTS_DIR, ensure_dirs
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


@app.get("/api/ai-analysis")
def ai_analysis(type: Literal["general", "phishtank", "energia", "finanzas"] = Query("general")) -> dict[str, str]:
    payload = load_artifact("ai_analysis")
    return {"analysis": str(payload.get(type) or payload.get("general") or "")}
