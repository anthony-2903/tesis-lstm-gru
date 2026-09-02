from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from app.config import EXPERIMENTS_DIR, ensure_dirs
from app.phishing.data import get_phishing_sequence_status
from app.phishing.models import PHISHING_MODEL_IDS
from app.phishing.pipeline import run_phishing_oof_experiment
from app.utils import read_json


MODEL_LABELS = {
    "lstm": "LSTM",
    "gru": "GRU",
    "brnn": "BiRNN",
    "tcn": "TCN",
    "transformer": "Transformer",
}


def run_phishing_experiment_for_api(
    *,
    protocol: str,
    epochs: int,
    batch_size: int,
    patience: int,
    seeds: tuple[int, ...],
    model_ids: tuple[str, ...],
    demo_max_rows: int | None,
    execution_run_id: str | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    sequence_status = get_phishing_sequence_status()
    if not sequence_status.get("readyForBaseModelTraining"):
        raise ValueError("Primero prepare y verifique el protocolo de secuencias OOF.")
    manifest = run_phishing_oof_experiment(
        protocol=protocol,
        epochs=epochs,
        batch_size=batch_size,
        patience=patience,
        seeds=seeds,
        model_ids=model_ids,
        demo_max_rows=demo_max_rows,
        execution_run_id=execution_run_id,
        progress_callback=progress_callback,
    )
    return build_phishing_experiment_view_from_manifest(manifest)


def get_latest_phishing_experiment() -> dict[str, Any]:
    manifests = find_phishing_manifests()
    if not manifests:
        return {
            "available": False,
            "message": "Todavía no existen resultados OOF de los modelos base de phishing.",
        }
    return build_phishing_experiment_view(manifests[0])


def list_phishing_experiments() -> dict[str, Any]:
    items = []
    for path in find_phishing_manifests():
        manifest = read_json(path)
        items.append({
            "runId": manifest.get("runId"),
            "status": manifest.get("status"),
            "createdAt": manifest.get("createdAt"),
            "protocol": manifest.get("protocol"),
            "rowsUsed": manifest.get("dataset", {}).get("rowsUsed"),
            "models": manifest.get("baseModels", []),
            "seeds": manifest.get("configuration", {}).get("seeds", []),
            "stackingReady": manifest.get("stacking", {}).get("ready", False),
        })
    return {"items": items}


def find_phishing_manifests() -> list[Path]:
    ensure_dirs()
    return sorted(
        EXPERIMENTS_DIR.glob("*/phishing_oof_v1/oof_manifest.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )


def build_phishing_experiment_view(path: Path) -> dict[str, Any]:
    return build_phishing_experiment_view_from_manifest(read_json(path))


def build_phishing_experiment_view_from_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    comparison = [
        {**row, "displayName": MODEL_LABELS.get(row["modelId"], row["modelId"])}
        for row in manifest.get("aggregate", [])
    ]
    winner = comparison[0] if comparison else None
    return {
        "available": True,
        "run": {
            "runId": manifest.get("runId"),
            "status": manifest.get("status"),
            "createdAt": manifest.get("createdAt"),
            "protocol": manifest.get("protocol"),
        },
        "configuration": manifest.get("configuration", {}),
        "dataset": manifest.get("dataset", {}),
        "validation": manifest.get("validation", {}),
        "comparison": comparison,
        "winner": {
            "modelId": winner.get("modelId") if winner else None,
            "displayName": winner.get("displayName") if winner else None,
            "primaryMetric": "prAuc",
            "higherIsBetter": True,
        },
        "stacking": manifest.get("stacking", {}),
        "methodology": {
            "testSetLocked": manifest.get("validation", {}).get("testSetLocked", True),
            "testSetUsed": manifest.get("validation", {}).get("testSetUsed", False),
            "outerValidationUsed": manifest.get("validation", {}).get("outerValidationUsed", False),
            "thresholdStatus": "exploratory_fixed_0.5",
            "warning": (
                "Corrida demostrativa: verifica la integración técnica, pero no constituye un resultado final de tesis; además persiste el riesgo de acoplamiento entre fuente y etiqueta."
                if manifest.get("status") == "demo"
                else "Candidata de modelos base; falta selección en validación, Stacking y evaluación final bloqueada."
            ),
        },
    }
