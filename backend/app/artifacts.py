from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import shutil
from typing import Any

import joblib

from app.config import EXPERIMENTS_DIR, GOLD_DIR, MODELS_DIR, RAW_DIR, RESULTS_DIR, SILVER_DIR
from app.utils import write_json


def make_run_id(mode: str, limit: int) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{stamp}_{mode}_{limit}"


def persist_models(model_groups: dict[str, dict[str, Any]], run_id: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    run_models_dir = EXPERIMENTS_DIR / run_id / "models"
    for domain, estimators in model_groups.items():
        for model_name, estimator in estimators.items():
            latest_path = MODELS_DIR / f"{domain}_{model_name}.joblib"
            run_path = run_models_dir / f"{domain}_{model_name}.joblib"
            latest_path.parent.mkdir(parents=True, exist_ok=True)
            run_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(estimator, latest_path)
            joblib.dump(estimator, run_path)
            records.append(
                {
                    "domain": domain,
                    "model": model_name,
                    "latestPath": str(latest_path),
                    "experimentPath": str(run_path),
                }
            )
    return records


def snapshot_experiment(
    *,
    run_id: str,
    mode: str,
    limit: int,
    model_records: list[dict[str, Any]],
    artifact_names: list[str],
    domain_totals: dict[str, int],
    metrics_summary: dict[str, Any],
) -> dict[str, Any]:
    experiment_dir = EXPERIMENTS_DIR / run_id
    for stage_name, stage_dir in {
        "raw": RAW_DIR,
        "silver": SILVER_DIR,
        "gold": GOLD_DIR,
        "results": RESULTS_DIR,
    }.items():
        _copy_stage(stage_dir, experiment_dir / stage_name)

    manifest = {
        "runId": run_id,
        "mode": mode,
        "limit": limit,
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "domainTotals": domain_totals,
        "metricsSummary": metrics_summary,
        "artifacts": [f"{name}.json" for name in artifact_names],
        "models": model_records,
        "paths": {
            "experiment": str(experiment_dir),
            "latestModels": str(MODELS_DIR),
            "latestResults": str(RESULTS_DIR),
        },
    }
    write_json(experiment_dir / "metrics_summary.json", metrics_summary)
    write_json(RESULTS_DIR / "metrics_summary.json", metrics_summary)
    write_json(experiment_dir / "manifest.json", manifest)
    write_json(RESULTS_DIR / "training_manifest.json", manifest)
    return manifest


def _copy_stage(source: Path, destination: Path) -> None:
    if not source.exists():
        return
    destination.mkdir(parents=True, exist_ok=True)
    for item in source.iterdir():
        target = destination / item.name
        if item.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)


def build_metrics_summary(artifacts: dict[str, Any]) -> dict[str, Any]:
    domains = []
    for domain in ["phishing", "energia", "finanzas"]:
        analysis = artifacts.get(f"analysis_{domain}", {})
        models = analysis.get("models", {})
        if not models:
            continue
        best_model = _best_model(models)
        domains.append(
            {
                "domain": domain,
                "totalRows": int(analysis.get("totalRows", 0)),
                "realAnomaliesCount": int(analysis.get("realAnomaliesCount", 0)),
                "bestModel": best_model,
                "models": {
                    key: {
                        "f1": float(value.get("f1", 0)),
                        "precision": float(value.get("precision", 0)),
                        "recall": float(value.get("recall", 0)),
                        "rmse": float(value.get("rmse", 0) or 0),
                        "trainTime": float(value.get("trainTime", 0) or 0),
                        "detectedCount": int(value.get("detectedCount", 0) or 0),
                    }
                    for key, value in models.items()
                },
            }
        )
    return {
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "domains": domains,
        "overallBest": _overall_best(domains),
    }


def _best_model(models: dict[str, dict[str, Any]]) -> dict[str, Any]:
    scored = []
    for key, value in models.items():
        f1 = float(value.get("f1", 0) or 0)
        precision = float(value.get("precision", 0) or 0)
        recall = float(value.get("recall", 0) or 0)
        rmse = float(value.get("rmse", 0) or 0)
        train_time = float(value.get("trainTime", 0) or 0)
        score = f1 * 0.55 + precision * 0.2 + recall * 0.2
        if rmse:
            score += 0.05 / (1 + rmse)
        if train_time:
            score += 0.02 / (1 + train_time)
        scored.append((score, key, value))
    scored.sort(reverse=True, key=lambda item: item[0])
    score, key, value = scored[0]
    return {
        "model": key,
        "score": float(score),
        "f1": float(value.get("f1", 0) or 0),
        "precision": float(value.get("precision", 0) or 0),
        "recall": float(value.get("recall", 0) or 0),
        "rmse": float(value.get("rmse", 0) or 0),
        "trainTime": float(value.get("trainTime", 0) or 0),
    }


def _overall_best(domains: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not domains:
        return None
    return max(
        (
            {
                "domain": item["domain"],
                **item["bestModel"],
            }
            for item in domains
        ),
        key=lambda item: float(item.get("score", 0)),
    )
