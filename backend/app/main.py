from __future__ import annotations

from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from app.config import RESULTS_DIR, ensure_dirs
from app.pipeline import run_pipeline
from app.utils import read_json


app = FastAPI(title="Tesis LSTM GRU Backend", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
    ],
    allow_credentials=True,
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


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api")
def api_index() -> dict[str, object]:
    return {
        "status": "ok",
        "message": "Backend local activo. Abre el frontend en http://127.0.0.1:5173",
        "endpoints": [
            "/api/health",
            "/api/dashboard",
            "/api/analysis",
            "/api/comparison",
            "/api/history",
            "/api/xai",
            "/api/ai-analysis?type=general",
        ],
    }


@app.post("/api/pipeline/run")
def run_pipeline_endpoint(mode: Literal["sample", "remote"] = "sample") -> dict[str, str]:
    run_pipeline(mode=mode)
    return {"status": "completed", "mode": mode}


@app.get("/api/dashboard")
def dashboard() -> dict:
    return load_artifact("dashboard")


@app.get("/api/analysis")
def analysis() -> dict:
    return load_artifact("analysis")


@app.get("/api/comparison")
def comparison() -> dict:
    return load_artifact("comparison")


@app.get("/api/history")
def history() -> dict:
    return load_artifact("history")


@app.get("/api/xai")
def xai() -> dict:
    return load_artifact("xai")


@app.get("/api/ai-analysis")
def ai_analysis(type: Literal["general", "phishtank", "energia", "finanzas"] = Query("general")) -> dict[str, str]:
    payload = load_artifact("ai_analysis")
    return {"analysis": str(payload.get(type) or payload.get("general") or "")}
