from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from app.config import EXPERIMENTS_DIR, ensure_dirs
from app.finance.models import FINANCE_MODEL_IDS
from app.finance.pipeline import run_finance_oof_experiment
from app.finance.sequences import get_finance_sequence_status
from app.utils import read_json


MODEL_LABELS = {
    "lstm": "LSTM",
    "gru": "GRU",
    "brnn": "BiRNN",
    "tcn": "TCN",
    "transformer": "Transformer",
}


def run_finance_experiment_for_api(
    *,
    protocol: str,
    epochs: int,
    batch_size: int,
    patience: int,
    seeds: tuple[int, ...],
    model_ids: tuple[str, ...],
    demo_max_rows_per_fold: int | None,
    execution_run_id: str | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    sequence_status = get_finance_sequence_status()
    if not sequence_status.get("readyForBaseModelPilot"):
        raise ValueError("Primero prepare y verifique las secuencias OOF financieras.")
    manifest = run_finance_oof_experiment(
        protocol=protocol,
        epochs=epochs,
        batch_size=batch_size,
        patience=patience,
        seeds=seeds,
        model_ids=model_ids,
        demo_max_rows_per_fold=demo_max_rows_per_fold,
        execution_run_id=execution_run_id,
        progress_callback=progress_callback,
    )
    return build_finance_experiment_view_from_manifest(manifest)


def get_latest_finance_experiment() -> dict[str, Any]:
    manifests = find_finance_manifests()
    if not manifests:
        return {"available": False, "message": "Todavía no existen resultados OOF de modelos financieros."}
    return build_finance_experiment_view_from_manifest(read_json(manifests[0]))


def list_finance_experiments() -> dict[str, Any]:
    items = []
    for path in find_finance_manifests():
        manifest = read_json(path)
        items.append(
            {
                "runId": manifest.get("runId"),
                "status": manifest.get("status"),
                "createdAt": manifest.get("createdAt"),
                "protocol": manifest.get("protocol"),
                "rowsUsed": manifest.get("dataset", {}).get("oofRowsUsedPerSeed"),
                "models": manifest.get("baseModels", []),
                "seeds": manifest.get("configuration", {}).get("seeds", []),
                "stackingReady": manifest.get("stacking", {}).get("ready", False),
            }
        )
    return {"items": items}


def find_finance_manifests() -> list[Path]:
    ensure_dirs()
    return sorted(
        EXPERIMENTS_DIR.glob("*/finance_oof_v1/oof_manifest.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )


def build_finance_experiment_view_from_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    comparison = [
        {**row, "displayName": MODEL_LABELS.get(row["modelId"], row["modelId"])}
        for row in manifest.get("aggregate", [])
    ]
    winner = comparison[0] if comparison else None
    demo = manifest.get("status") == "demo"
    return {
        "available": True,
        "run": {
            "runId": manifest.get("runId"),
            "status": manifest.get("status"),
            "createdAt": manifest.get("createdAt"),
            "protocol": manifest.get("protocol"),
        },
        "configuration": manifest.get("configuration", {}),
        "execution": manifest.get("execution", {}),
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
            "externalValidationUsed": manifest.get("validation", {}).get("externalValidationUsed", False),
            "warning": (
                "Piloto técnico sobre un subconjunto OOF del benchmark sintético; no constituye el resultado final de tesis."
                if demo
                else "Candidata de modelos base; falta validación externa, Stacking y evaluación final bloqueada."
            ),
        },
        "requiredModels": list(FINANCE_MODEL_IDS),
    }
