from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import pandas as pd

from app.config import EXPERIMENTS_DIR
from app.energy.models import THESIS_MODEL_IDS
from app.energy.oof_pipeline import run_energy_oof_experiment
from app.energy.revalidation import run_energy_optimized_stacking_revalidation
from app.energy.service import build_energy_experiment_view
from app.finance.mef_market_data import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    TIMESTAMP_COLUMN,
    get_mef_market_dataset_status,
)


FINANCE_MARKET_MODEL_IDS = THESIS_MODEL_IDS
BASE_NAMESPACE = "finance_market_oof_v1"
REVALIDATION_NAMESPACE = "finance_market_revalidation_v1"


def run_finance_market_oof_experiment(
    *,
    window: int = 20,
    horizon: int = 1,
    gap_steps: int = 5,
    n_splits: int = 5,
    epochs: int = 20,
    batch_size: int = 32,
    seeds: tuple[int, ...] = (42, 101, 202, 303, 404),
    model_ids: tuple[str, ...] = FINANCE_MARKET_MODEL_IDS,
    execution_run_id: str | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    status = get_mef_market_dataset_status()
    if not status.get("readyForThesisTraining"):
        raise ValueError(status.get("message", "La serie MEF no esta lista para tesis."))
    silver = status["silver"]
    frame = pd.read_csv(Path(silver["path"]), usecols=[TIMESTAMP_COLUMN, *FEATURE_COLUMNS])
    lineage = {
        "type": "real_public_versioned_silver",
        "verified": True,
        "datasetId": status["datasetId"],
        "provider": status["provider"],
        "category": status["category"],
        "datasetSlug": status["datasetSlug"],
        "resourceId": status["resourceId"],
        "sourceUrl": status["sourceUrl"],
        "path": silver["path"],
        "sha256": silver["sha256"],
        "bytes": silver["bytes"],
    }
    return run_energy_oof_experiment(
        frame,
        timestamp_column=TIMESTAMP_COLUMN,
        target_column=TARGET_COLUMN,
        window=window,
        horizon=horizon,
        gap_steps=gap_steps,
        n_splits=n_splits,
        epochs=epochs,
        batch_size=batch_size,
        seeds=seeds,
        model_ids=model_ids,
        strict=True,
        source_lineage=lineage,
        execution_run_id=execution_run_id,
        progress_callback=progress_callback,
        domain="finanzas",
        artifact_namespace=BASE_NAMESPACE,
        minimum_rows=1_000,
        expected_frequency="irregular_business_day_observations",
    )


def run_finance_market_revalidation(
    *,
    base_manifest_path: Path | None = None,
    bootstrap_iterations: int = 500,
) -> dict[str, Any]:
    base_path = base_manifest_path or _latest_base_manifest()
    return run_energy_optimized_stacking_revalidation(
        base_manifest_path=base_path,
        bootstrap_iterations=bootstrap_iterations,
        domain="finanzas",
        revalidation_namespace=REVALIDATION_NAMESPACE,
        run_suffix="finance_market_revalidation",
        expected_model_ids=FINANCE_MARKET_MODEL_IDS,
        bootstrap_unit="observed_financial_session",
    )


def get_latest_finance_market_experiment() -> dict[str, Any]:
    paths = find_finance_market_manifests()
    if not paths:
        return {"available": False, "message": "No existe aun una corrida financiera MEF."}
    view = build_energy_experiment_view(paths[0])
    view["task"] = "forecasting_with_residual_anomaly_detection"
    view["domain"] = "finanzas"
    return view


def get_latest_finance_market_revalidation() -> dict[str, Any]:
    paths = sorted(
        EXPERIMENTS_DIR.glob(f"*/{BASE_NAMESPACE}/{REVALIDATION_NAMESPACE}/revalidation_manifest.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not paths:
        return {"available": False, "message": "No existe aun revalidacion financiera MEF."}
    import json

    return {"available": True, **json.loads(paths[0].read_text(encoding="utf-8"))}


def find_finance_market_manifests() -> list[Path]:
    paths = list(EXPERIMENTS_DIR.glob(f"*/{BASE_NAMESPACE}/oof_manifest.json"))
    return sorted(paths, key=lambda path: path.stat().st_mtime, reverse=True)


def _latest_base_manifest() -> Path:
    paths = find_finance_market_manifests()
    if not paths:
        raise FileNotFoundError("No existe manifiesto OOF financiero MEF.")
    return paths[0]
