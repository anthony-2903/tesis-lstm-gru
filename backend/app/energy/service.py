from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from app.cleaning.timeseries_cleaner import clean_opsd
from app.config import EXPERIMENTS_DIR, SILVER_DIR, ensure_dirs
from app.energy.ingestion import get_energy_dataset_status
from app.energy.models import THESIS_MODEL_IDS
from app.energy.oof_pipeline import run_energy_oof_experiment
from app.ingestion.samples import sample_opsd
from app.utils import read_json


MODEL_LABELS = {
    "lstm": "LSTM",
    "gru": "GRU",
    "brnn": "BiRNN",
    "tcn": "TCN",
    "transformer": "Transformer",
    "mean": "Promedio",
    "weighted_mean": "Promedio ponderado",
    "stacking_gradient_boosting": "Stacking",
}


def run_energy_experiment_for_api(
    *,
    protocol: str,
    source: str,
    window: int,
    horizon: int,
    gap_steps: int,
    n_splits: int,
    epochs: int,
    batch_size: int,
    seeds: tuple[int, ...],
    model_ids: tuple[str, ...],
    execution_run_id: str | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    if protocol not in {"demo", "thesis"}:
        raise ValueError("El protocolo debe ser demo o thesis.")
    if source not in {"sample", "silver"}:
        raise ValueError("La fuente debe ser sample o silver.")
    if protocol == "thesis" and source == "sample":
        raise ValueError("Una corrida de tesis no puede utilizar la muestra sintética.")
    unknown_models = sorted(set(model_ids) - set(THESIS_MODEL_IDS))
    if unknown_models:
        raise ValueError(f"Modelos energéticos no soportados: {', '.join(unknown_models)}")
    if len(model_ids) < 2:
        raise ValueError("El experimento con Stacking necesita al menos dos modelos base.")

    frame = _load_energy_source(source)
    if source == "silver":
        status = get_energy_dataset_status()
        silver = status.get("silver", {})
        source_lineage = {
            "type": "real_versioned_silver",
            "verified": bool(status.get("readyForThesisPilot")),
            "datasetId": status.get("datasetId"),
            "provider": status.get("provider"),
            "sourceVersion": status.get("sourceVersion"),
            "path": silver.get("path"),
            "sha256": silver.get("sha256"),
            "bytes": silver.get("bytes"),
        }
    else:
        source_lineage = {"type": "synthetic_sample", "verified": False}
    manifest = run_energy_oof_experiment(
        frame,
        timestamp_column="timestamp",
        target_column="DE_load_actual_entsoe_transparency",
        window=window,
        horizon=horizon,
        gap_steps=gap_steps,
        n_splits=n_splits,
        epochs=epochs,
        batch_size=batch_size,
        seeds=seeds,
        model_ids=model_ids,
        strict=protocol == "thesis",
        source_lineage=source_lineage,
        execution_run_id=execution_run_id,
        progress_callback=progress_callback,
    )
    manifest_path = Path(manifest["artifacts"]["oofPredictions"]).parent / "oof_manifest.json"
    return build_energy_experiment_view(manifest_path)


def get_latest_energy_experiment() -> dict[str, Any]:
    manifests = find_energy_manifests()
    if not manifests:
        return {
            "available": False,
            "message": "Todavía no existe una corrida OOF de energía. Ejecute primero una demostración.",
        }
    return build_energy_experiment_view(manifests[0])


def list_energy_experiments() -> dict[str, Any]:
    items = []
    for path in find_energy_manifests():
        manifest = read_json(path)
        items.append(
            {
                "runId": manifest.get("runId"),
                "status": manifest.get("status"),
                "createdAt": manifest.get("createdAt"),
                "rows": manifest.get("dataset", {}).get("rows", 0),
                "models": manifest.get("baseModels", []),
                "folds": len(manifest.get("walkForward", {}).get("folds", [])),
                "seeds": manifest.get("walkForward", {}).get("seeds", []),
                "ensembleWinner": manifest.get("aggregate", {}).get("winner"),
                "testSetUsed": manifest.get("walkForward", {}).get("testSetUsed", False),
            }
        )
    return {"items": items}


def find_energy_manifests() -> list[Path]:
    ensure_dirs()
    paths = list(EXPERIMENTS_DIR.glob("*/energy_oof_v1/oof_manifest.json"))
    return sorted(paths, key=lambda path: path.stat().st_mtime, reverse=True)


def build_energy_experiment_view(manifest_path: Path) -> dict[str, Any]:
    manifest = read_json(manifest_path)
    common_pairs = {
        (int(item["seed"]), int(item["fold"]))
        for item in manifest.get("comparisonScope", {}).get("pairs", [])
    }
    base_rows = _aggregate_base_metrics(manifest.get("baseFoldMetrics", []), common_pairs=common_pairs or None)
    ensemble_rows = _aggregate_ensemble_metrics(manifest.get("ensembles", {}).get("aggregate", []))
    comparison = sorted([*base_rows, *ensemble_rows], key=lambda row: row["rmseMean"])
    overall_winner = comparison[0]["predictorId"] if comparison else None
    artifacts = manifest.get("artifacts", {})

    timeline = _build_timeline(
        Path(artifacts.get("oofPredictions", "")),
        Path(artifacts.get("ensemblePredictions", "")),
        manifest.get("baseModels", []),
        manifest.get("ensembles", {}).get("winner"),
    )
    anomaly_summaries = [
        {
            **summary,
            "displayName": MODEL_LABELS.get(summary.get("predictorId", ""), summary.get("predictorId", "")),
        }
        for summary in manifest.get("anomalies", {}).get("summaries", [])
        if summary.get("method") == manifest.get("anomalies", {}).get("recommendedMethod", "mad")
    ]

    return {
        "available": True,
        "run": {
            "runId": manifest.get("runId"),
            "status": manifest.get("status"),
            "createdAt": manifest.get("createdAt"),
            "protocolVersion": manifest.get("protocolVersion"),
        },
        "dataset": manifest.get("dataset", {}),
        "validation": {
            **manifest.get("walkForward", {}),
            "classificationMetricsAvailable": manifest.get("anomalies", {}).get("classificationMetricsAvailable", False),
            "anomalyLabelType": manifest.get("anomalies", {}).get("labelType", "estimated"),
        },
        "comparison": comparison,
        "winner": {
            "overall": overall_winner,
            "overallDisplayName": MODEL_LABELS.get(overall_winner or "", overall_winner),
            "bestEnsemble": manifest.get("ensembles", {}).get("winner"),
            "bestEnsembleDisplayName": MODEL_LABELS.get(
                manifest.get("ensembles", {}).get("winner", ""),
                manifest.get("ensembles", {}).get("winner"),
            ),
            "primaryMetric": "rmse",
            "lowerIsBetter": True,
        },
        "timeline": timeline,
        "anomalies": {
            "recommendedMethod": manifest.get("anomalies", {}).get("recommendedMethod", "mad"),
            "labelType": manifest.get("anomalies", {}).get("labelType", "estimated"),
            "classificationMetricsAvailable": manifest.get("anomalies", {}).get("classificationMetricsAvailable", False),
            "summaries": anomaly_summaries,
        },
        "methodology": {
            "testSetUsed": manifest.get("walkForward", {}).get("testSetUsed", False),
            "testSetEvaluatedOnce": manifest.get("aggregate", {}).get("testSetEvaluatedOnce", False),
            "stackingReady": manifest.get("aggregate", {}).get("stackingReady", False),
            "fairComparison": manifest.get("comparisonScope", {}).get("fairComparison", False),
            "commonEvaluationPairs": manifest.get("comparisonScope", {}).get("pairs", []),
            "warning": _methodology_warning(manifest),
        },
    }


def _load_energy_source(source: str) -> pd.DataFrame:
    if source == "sample":
        return clean_opsd(sample_opsd())
    path = SILVER_DIR / "opsd.csv"
    if not path.exists():
        raise ValueError("No existe backend/storage/silver/opsd.csv. Ejecute primero la ingesta energética.")
    return pd.read_csv(path)


def _aggregate_base_metrics(
    rows: list[dict[str, Any]],
    *,
    common_pairs: set[tuple[int, int]] | None = None,
) -> list[dict[str, Any]]:
    if not rows:
        return []
    frame = pd.DataFrame(rows)
    if common_pairs and {"seed", "fold"}.issubset(frame.columns):
        mask = frame.apply(lambda row: (int(row["seed"]), int(row["fold"])) in common_pairs, axis=1)
        frame = frame[mask].copy()
    result = []
    for predictor_id, group in frame.groupby("modelId", sort=False):
        result.append(
            {
                "predictorId": str(predictor_id),
                "displayName": MODEL_LABELS.get(str(predictor_id), str(predictor_id)),
                "family": "base",
                "foldEvaluations": int(len(group)),
                "rmseMean": float(group["rmse"].mean()),
                "rmseStd": _sample_std(group["rmse"]),
                "maeMean": float(group["mae"].mean()),
                "smapeMean": float(group["smape"].mean()),
                "r2Mean": float(group["r2"].mean()),
                "trainTimeMean": float(group["trainTimeSeconds"].mean()),
                "inferenceTimeMsMean": float(group["inferenceTimeMsPerSample"].mean()),
            }
        )
    return result


def _aggregate_ensemble_metrics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "predictorId": row["ensembleId"],
            "displayName": MODEL_LABELS.get(row["ensembleId"], row["ensembleId"]),
            "family": "ensemble",
            "foldEvaluations": int(row["foldEvaluations"]),
            "rmseMean": float(row["rmseMean"]),
            "rmseStd": float(row["rmseStd"]),
            "maeMean": float(row["maeMean"]),
            "smapeMean": float(row["smapeMean"]),
            "r2Mean": float(row["r2Mean"]),
            "trainTimeMean": None,
            "inferenceTimeMsMean": None,
        }
        for row in rows
    ]


def _build_timeline(
    base_path: Path,
    ensemble_path: Path,
    base_models: list[str],
    ensemble_winner: str | None,
    max_points: int = 240,
) -> list[dict[str, Any]]:
    if not base_path.is_file() or not ensemble_path.is_file() or not ensemble_winner:
        return []
    base = pd.read_csv(base_path)
    ensembles = pd.read_csv(ensemble_path)
    winner_rows = ensembles[ensembles["ensemble"] == ensemble_winner].copy()
    if winner_rows.empty:
        return []
    seed = int(winner_rows["seed"].max())
    fold = int(winner_rows[winner_rows["seed"] == seed]["fold"].max())
    base_slice = base[(base["seed"] == seed) & (base["fold"] == fold)].copy()
    ensemble_slice = winner_rows[(winner_rows["seed"] == seed) & (winner_rows["fold"] == fold)][
        ["timestamp", "prediction"]
    ].rename(columns={"prediction": "ensemblePrediction"})
    merged = base_slice.merge(ensemble_slice, on="timestamp", how="inner").sort_values("timestamp")
    if len(merged) > max_points:
        indexes = np.linspace(0, len(merged) - 1, max_points, dtype=int)
        merged = merged.iloc[indexes]
    columns = [model for model in base_models if model in merged.columns]
    return [
        {
            "timestamp": str(row["timestamp"]),
            "actual": float(row["actual"]),
            "ensemble": float(row["ensemblePrediction"]),
            **{model: float(row[model]) for model in columns},
        }
        for _, row in merged.iterrows()
    ]


def _sample_std(values: pd.Series) -> float:
    return float(values.std(ddof=1)) if len(values) > 1 else 0.0


def _methodology_warning(manifest: dict[str, Any]) -> str:
    if manifest.get("status") != "thesis_candidate":
        return (
            "Esta corrida es demostrativa. Verifica la integración técnica, pero no debe citarse como resultado final de tesis."
        )
    return (
        "Corrida candidata de tesis: el conjunto de prueba permanece bloqueado y las anomalías no tienen etiquetas independientes."
    )
