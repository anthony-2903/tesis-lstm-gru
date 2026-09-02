from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

from app.config import EXPERIMENTS_DIR
from app.energy.metrics import energy_regression_metrics
from app.energy.models import THESIS_MODEL_IDS
from app.energy.service import find_energy_manifests
from app.energy.stacking import _build_meta_matrix, evaluate_energy_ensembles


def run_energy_optimized_stacking_revalidation(
    *,
    base_manifest_path: Path | None = None,
    bootstrap_iterations: int = 500,
    random_seed: int = 42,
    domain: str = "energia",
    revalidation_namespace: str = "energy_revalidation_v1",
    run_suffix: str = "energy_revalidation",
    expected_model_ids: tuple[str, ...] = THESIS_MODEL_IDS,
    bootstrap_unit: str = "calendar_day",
) -> dict[str, Any]:
    if bootstrap_iterations < 300:
        raise ValueError("La revalidacion energetica exige al menos 300 iteraciones bootstrap.")
    base_path = base_manifest_path or _latest_manifest()
    base = json.loads(base_path.read_text(encoding="utf-8"))
    _verify_base(base, base_path, expected_model_ids=expected_model_ids)
    model_ids = tuple(str(value) for value in base.get("baseModels", []))
    oof_path = Path(base["artifacts"]["oofPredictions"])
    oof = pd.read_csv(oof_path)
    folds = sorted(int(value) for value in oof["fold"].unique())
    if len(folds) < 5:
        raise ValueError("La revalidacion independiente exige cinco folds OOF.")
    independent_fold = folds[-1]
    selection_folds = folds[:-1]
    configurations = {
        "full": model_ids,
        **{f"without_{model_id}": tuple(value for value in model_ids if value != model_id) for model_id in model_ids},
    }

    selection_rows: list[dict[str, Any]] = []
    independent_predictions: dict[str, pd.DataFrame] = {}
    fitted_objects: list[dict[str, Any]] = []
    xai_rows: list[dict[str, Any]] = []
    for ablation_id, feature_ids in configurations.items():
        selection_source = oof[oof["fold"].isin(selection_folds)].copy()
        _, selection_report = evaluate_energy_ensembles(selection_source, base_model_ids=feature_ids)
        stacking_selection = next(
            row for row in selection_report["aggregate"] if row["ensembleId"] == "stacking_gradient_boosting"
        )
        selection_rows.append({"ablationId": ablation_id, "baseModels": list(feature_ids), **stacking_selection})
        all_predictions, _ = evaluate_energy_ensembles(oof, base_model_ids=feature_ids)
        independent_predictions[ablation_id] = all_predictions[
            (all_predictions["fold"] == independent_fold)
            & (all_predictions["ensemble"] == "stacking_gradient_boosting")
        ].copy()

    selected = min(selection_rows, key=lambda row: (float(row["rmseMean"]), float(row["rmseStd"])))
    selected_id = str(selected["ablationId"])
    selected_models = tuple(str(value) for value in selected["baseModels"])
    destination = base_path.parent / revalidation_namespace
    destination.mkdir(parents=True, exist_ok=True)

    for seed in sorted(int(value) for value in oof["seed"].unique()):
        train = oof[(oof["seed"] == seed) & (oof["fold"] < independent_fold)].copy()
        estimator = GradientBoostingRegressor(
            n_estimators=80,
            learning_rate=0.05,
            max_depth=2,
            random_state=seed,
        )
        matrix = _build_meta_matrix(train, selected_models)
        estimator.fit(matrix, train["actual"].to_numpy(dtype=float))
        model_path = destination / "models" / f"stacking_seed_{seed}.joblib"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(estimator, model_path)
        fitted_objects.append({"kind": "optimized_stacking_model", "seed": seed, **_artifact(model_path)})
        feature_names = [*selected_models, "prediction_mean", "prediction_std", "prediction_range"]
        xai_rows.extend(
            {"seed": seed, "feature": name, "importance": float(importance)}
            for name, importance in zip(feature_names, estimator.feature_importances_, strict=True)
        )

    comparison_frame = _independent_comparison_frame(
        oof=oof,
        stacking=independent_predictions[selected_id],
        model_ids=model_ids,
        independent_fold=independent_fold,
    )
    seed_metrics = _independent_seed_metrics(
        oof=oof,
        stacking=independent_predictions[selected_id],
        model_ids=model_ids,
        independent_fold=independent_fold,
    )
    metric_rows = []
    for predictor_id, group in comparison_frame.groupby("predictorId", sort=False):
        metric_rows.append({"predictorId": str(predictor_id), **energy_regression_metrics(group["actual"], group["prediction"])})
    metric_rows.sort(key=lambda row: float(row["rmse"]))
    best_base = min((row for row in metric_rows if row["predictorId"] != "optimized_stacking"), key=lambda row: row["rmse"])
    stacking_metrics = next(row for row in metric_rows if row["predictorId"] == "optimized_stacking")
    inference = _paired_day_bootstrap(
        comparison_frame,
        reference_id=str(best_base["predictorId"]),
        candidate_id="optimized_stacking",
        iterations=bootstrap_iterations,
        seed=random_seed,
    )
    xai_aggregate = (
        pd.DataFrame(xai_rows).groupby("feature", as_index=False)["importance"].agg(["mean", "std"]).reset_index()
    )
    xai_payload = [
        {"feature": str(row["feature"]), "importanceMean": float(row["mean"]), "importanceStd": float(0.0 if pd.isna(row["std"]) else row["std"])}
        for _, row in xai_aggregate.sort_values("mean", ascending=False).iterrows()
    ]

    predictions_path = destination / "independent_comparison.csv"
    comparison_frame.to_csv(predictions_path, index=False)
    xai_path = destination / "stacking_feature_importance.json"
    xai_path.write_text(json.dumps({"items": xai_payload}, ensure_ascii=False, indent=2), encoding="utf-8")
    run_id = f"{base['runId']}_{run_suffix}"
    manifest = {
        "schemaVersion": "1.0.0",
        "runId": run_id,
        "domain": domain,
        "status": "thesis_optimized_candidate" if base.get("status") == "thesis_candidate" else "demo_revalidation",
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "baseRun": {"runId": base.get("runId"), "manifest": _artifact(base_path)},
        "protocol": {
            "selectionFolds": selection_folds,
            "independentComparisonFold": independent_fold,
            "selectionUsesIndependentFold": False,
            "testSetUsed": False,
            "testEvaluationAuthorized": False,
            "bootstrapUnit": bootstrap_unit,
            "bootstrapIterations": bootstrap_iterations,
        },
        "ablation": {"configurations": [{"ablationId": key, "baseModels": list(value)} for key, value in configurations.items()], "selectionMetrics": selection_rows},
        "selection": {"selectedAblationId": selected_id, "selectedBaseModels": list(selected_models), "selectionMetric": "rmse", "selectionRow": selected},
        "independentComparison": {
            "metrics": metric_rows,
            "seedMetrics": seed_metrics,
            "seedCount": int(oof["seed"].nunique()),
            "ranking": [row["predictorId"] for row in metric_rows],
            "bestBaseModel": best_base["predictorId"],
            "optimizedStackingIsWinner": metric_rows[0]["predictorId"] == "optimized_stacking",
            "optimizedStackingRmse": stacking_metrics["rmse"],
            "pairedInference": inference,
        },
        "xai": {"method": "gradient_boosting_impurity_importance_by_seed", "items": xai_payload},
        "artifacts": {
            "independentComparison": _artifact(predictions_path),
            "xai": _artifact(xai_path),
            "fittedObjects": fitted_objects,
        },
    }
    manifest_path = destination / "revalidation_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {**manifest, "manifest": _artifact(manifest_path)}


def get_latest_energy_optimized_revalidation() -> dict[str, Any]:
    paths = sorted(EXPERIMENTS_DIR.glob("*/energy_oof_v1/energy_revalidation_v1/revalidation_manifest.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not paths:
        return {"available": False, "message": "No existe revalidacion energetica optimizada."}
    return {"available": True, **json.loads(paths[0].read_text(encoding="utf-8"))}


def _independent_comparison_frame(*, oof: pd.DataFrame, stacking: pd.DataFrame, model_ids: tuple[str, ...], independent_fold: int) -> pd.DataFrame:
    source = oof[oof["fold"] == independent_fold].copy()
    rows = []
    for model_id in model_ids:
        grouped = source.groupby("timestamp", as_index=False).agg(actual=("actual", "first"), prediction=(model_id, "mean"))
        grouped["predictorId"] = model_id
        rows.append(grouped)
    optimized = stacking.groupby("timestamp", as_index=False).agg(actual=("actual", "first"), prediction=("prediction", "mean"))
    optimized["predictorId"] = "optimized_stacking"
    rows.append(optimized)
    result = pd.concat(rows, ignore_index=True)
    result["timestamp"] = pd.to_datetime(result["timestamp"], utc=True)
    result["calendarDay"] = result["timestamp"].dt.strftime("%Y-%m-%d")
    return result.sort_values(["predictorId", "timestamp"], kind="stable")


def _independent_seed_metrics(*, oof: pd.DataFrame, stacking: pd.DataFrame, model_ids: tuple[str, ...], independent_fold: int) -> list[dict[str, Any]]:
    source = oof[oof["fold"] == independent_fold]
    rows: list[dict[str, Any]] = []
    for seed in sorted(int(value) for value in source["seed"].unique()):
        seed_source = source[source["seed"] == seed]
        for model_id in model_ids:
            rows.append({"seed": seed, "predictorId": model_id, **energy_regression_metrics(seed_source["actual"], seed_source[model_id])})
        seed_stacking = stacking[stacking["seed"] == seed]
        rows.append({"seed": seed, "predictorId": "optimized_stacking", **energy_regression_metrics(seed_stacking["actual"], seed_stacking["prediction"])})
    return rows


def _paired_day_bootstrap(frame: pd.DataFrame, *, reference_id: str, candidate_id: str, iterations: int, seed: int) -> dict[str, Any]:
    pivot_actual = frame.drop_duplicates("timestamp").set_index("timestamp")["actual"]
    pivot = frame.pivot(index="timestamp", columns="predictorId", values="prediction").dropna(subset=[reference_id, candidate_id])
    actual = pivot_actual.reindex(pivot.index)
    days = pd.Index(pivot.index.strftime("%Y-%m-%d").unique())
    rng = np.random.default_rng(seed)
    deltas = []
    day_values = pivot.index.strftime("%Y-%m-%d")
    for _ in range(iterations):
        sampled = rng.choice(days.to_numpy(), size=len(days), replace=True)
        positions = np.concatenate([np.flatnonzero(day_values == day) for day in sampled])
        y = actual.to_numpy(dtype=float)[positions]
        ref = pivot[reference_id].to_numpy(dtype=float)[positions]
        candidate = pivot[candidate_id].to_numpy(dtype=float)[positions]
        ref_rmse = float(np.sqrt(np.mean(np.square(y - ref))))
        candidate_rmse = float(np.sqrt(np.mean(np.square(y - candidate))))
        deltas.append(candidate_rmse - ref_rmse)
    values = np.asarray(deltas, dtype=float)
    return {
        "referenceId": reference_id,
        "candidateId": candidate_id,
        "deltaDefinition": "candidate_rmse_minus_reference_rmse",
        "observedDeltaRmse": float(np.sqrt(np.mean(np.square(actual - pivot[candidate_id]))) - np.sqrt(np.mean(np.square(actual - pivot[reference_id])))),
        "confidenceInterval95": [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))],
        "probabilityCandidateBetter": float(np.mean(values < 0.0)),
        "calendarDays": int(len(days)),
        "iterations": iterations,
    }


def _verify_base(base: dict[str, Any], path: Path, *, expected_model_ids: tuple[str, ...] = THESIS_MODEL_IDS) -> None:
    if base.get("walkForward", {}).get("testSetUsed"):
        raise ValueError("La corrida base ya utilizo test; no es elegible para revalidacion.")
    integrity = base.get("artifactIntegrity", {}).get("oofPredictions")
    if integrity:
        observed = _artifact(Path(integrity["path"]))
        if observed["sha256"] != integrity.get("sha256") or observed["bytes"] != integrity.get("bytes"):
            raise ValueError("Las predicciones OOF energeticas no superan la verificacion de integridad.")
    if set(base.get("baseModels", [])) != set(expected_model_ids):
        raise ValueError("La revalidacion requiere las cinco arquitecturas base.")
    if not path.is_file():
        raise FileNotFoundError(path)


def _latest_manifest() -> Path:
    paths = find_energy_manifests()
    if not paths:
        raise FileNotFoundError("No existe manifiesto energetico OOF.")
    return paths[0]


def _artifact(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return {"path": str(path.resolve()), "sha256": digest.hexdigest(), "bytes": size}
