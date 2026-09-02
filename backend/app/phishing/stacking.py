from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from app.config import EXPERIMENTS_DIR
from app.phishing.ingestion import sha256_file
from app.phishing.metrics import phishing_classification_metrics
from app.utils import read_json, write_json


BASELINE_IDS = ("mean", "voting", "weighted_mean")
META_MODEL_IDS = ("stacking_logistic", "stacking_ridge", "stacking_gradient_boosting")
ENSEMBLE_IDS = (*BASELINE_IDS, *META_MODEL_IDS)


def run_phishing_stacking_experiment(
    *,
    base_manifest_path: Path | None = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    base_path = base_manifest_path or _latest_base_manifest()
    if base_path is None or not base_path.exists():
        raise FileNotFoundError("No existe una corrida OOF base para construir Stacking.")
    base_manifest = read_json(base_path)
    oof_artifact = base_manifest.get("artifacts", {}).get("oofProbabilities", {})
    oof_path = Path(oof_artifact.get("path", ""))
    if not oof_path.is_file():
        raise FileNotFoundError("No existe el artefacto de probabilidades OOF.")
    actual_hash, actual_bytes = sha256_file(oof_path)
    if actual_hash != oof_artifact.get("sha256") or actual_bytes != oof_artifact.get("bytes"):
        raise ValueError("El artefacto OOF no supera la verificación de integridad.")

    model_ids = tuple(base_manifest.get("baseModels", []))
    probability_columns = [f"probability_{model_id}" for model_id in model_ids]
    if len(probability_columns) < 2:
        raise ValueError("Stacking requiere probabilidades de al menos dos modelos base.")
    oof = pd.read_csv(oof_path)
    required = {"sample_id", "seed", "fold", "is_phishing", *probability_columns}
    missing = required.difference(oof.columns)
    if missing:
        raise ValueError(f"El OOF no contiene columnas requeridas: {', '.join(sorted(missing))}")
    if oof[list(probability_columns)].isna().any().any():
        raise ValueError("Las probabilidades OOF contienen valores faltantes.")
    if oof.duplicated(["seed", "sample_id"]).any():
        raise ValueError("Cada muestra debe aparecer una sola vez por semilla.")

    destination = output_dir or (base_path.parent / "stacking_v1")
    destination.mkdir(parents=True, exist_ok=True)
    predictions_rows: list[dict[str, Any]] = []
    fold_metrics: list[dict[str, Any]] = []
    artifacts: list[dict[str, Any]] = []
    fold_protocols: list[dict[str, Any]] = []

    for seed, seed_frame in oof.groupby("seed", sort=True):
        folds = sorted(int(value) for value in seed_frame["fold"].unique())
        if len(folds) < 3:
            raise ValueError("La evaluación meta-OOF necesita al menos tres folds.")
        for fold in folds:
            meta_train = seed_frame[seed_frame["fold"] != fold].copy()
            meta_holdout = seed_frame[seed_frame["fold"] == fold].copy()
            if set(meta_train["sample_id"]) & set(meta_holdout["sample_id"]):
                raise ValueError("Existe fuga de muestras entre ajuste y holdout del metamodelo.")
            x_train_base = meta_train[probability_columns].to_numpy(dtype=np.float64)
            x_holdout_base = meta_holdout[probability_columns].to_numpy(dtype=np.float64)
            y_train = meta_train["is_phishing"].to_numpy(dtype=np.int32)
            y_holdout = meta_holdout["is_phishing"].to_numpy(dtype=np.int32)
            x_train_meta, feature_names = build_meta_features(x_train_base, probability_columns)
            x_holdout_meta, _ = build_meta_features(x_holdout_base, probability_columns)

            candidate_predictions: dict[str, np.ndarray] = {
                "mean": x_holdout_base.mean(axis=1),
                "voting": (x_holdout_base >= 0.5).mean(axis=1),
            }
            weights = fit_nonnegative_weights(x_train_base, y_train)
            candidate_predictions["weighted_mean"] = x_holdout_base @ weights
            weights_path = destination / "weights" / f"seed_{int(seed)}_fold_{fold}.json"
            write_json(weights_path, {
                "schemaVersion": "1.0.0",
                "seed": int(seed),
                "fold": int(fold),
                "objective": "binary_log_loss",
                "constraints": "nonnegative_weights_sum_to_one",
                "weights": {model_id: float(weight) for model_id, weight in zip(model_ids, weights)},
                "fitRows": int(len(meta_train)),
            })
            weight_hash, weight_bytes = sha256_file(weights_path)
            artifacts.append({
                "candidateId": "weighted_mean",
                "seed": int(seed),
                "fold": int(fold),
                "path": str(weights_path.resolve()),
                "sha256": weight_hash,
                "bytes": weight_bytes,
            })

            candidates = build_meta_models(seed=int(seed) + fold)
            for candidate_id, model in candidates.items():
                model.fit(x_train_meta, y_train)
                candidate_predictions[candidate_id] = model.predict_proba(x_holdout_meta)[:, 1]
                model_path = destination / "models" / f"seed_{int(seed)}" / f"fold_{fold}" / f"{candidate_id}.joblib"
                model_path.parent.mkdir(parents=True, exist_ok=True)
                joblib.dump(model, model_path)
                model_hash, model_bytes = sha256_file(model_path)
                artifacts.append({
                    "candidateId": candidate_id,
                    "seed": int(seed),
                    "fold": int(fold),
                    "path": str(model_path.resolve()),
                    "sha256": model_hash,
                    "bytes": model_bytes,
                })

            for candidate_id, probabilities in candidate_predictions.items():
                fold_metrics.append({
                    "candidateId": candidate_id,
                    "family": "baseline" if candidate_id in BASELINE_IDS else "stacking",
                    "seed": int(seed),
                    "fold": int(fold),
                    **phishing_classification_metrics(y_holdout, probabilities, threshold=0.5),
                })
            for row_index, row in meta_holdout.reset_index(drop=True).iterrows():
                predictions_rows.append({
                    "sample_id": row["sample_id"],
                    "seed": int(seed),
                    "fold": int(fold),
                    "is_phishing": int(row["is_phishing"]),
                    **{f"probability_{candidate_id}": float(candidate_predictions[candidate_id][row_index]) for candidate_id in ENSEMBLE_IDS},
                })
            fold_protocols.append({
                "seed": int(seed),
                "fold": int(fold),
                "fitRows": int(len(meta_train)),
                "holdoutRows": int(len(meta_holdout)),
                "sampleOverlap": 0,
                "featureCount": len(feature_names),
            })

    predictions = pd.DataFrame(predictions_rows).sort_values(["seed", "sample_id"], kind="stable")
    expected_rows = len(oof)
    if len(predictions) != expected_rows or predictions.duplicated(["seed", "sample_id"]).any():
        raise ValueError("La evaluación meta-OOF no cubre exactamente las muestras base.")
    ensemble_probability_columns = [f"probability_{candidate_id}" for candidate_id in ENSEMBLE_IDS]
    if predictions[ensemble_probability_columns].isna().any().any():
        raise ValueError("Existen predicciones de ensamble faltantes.")
    predictions_path = destination / "stacking_oof_probabilities.csv"
    predictions.to_csv(predictions_path, index=False)
    predictions_hash, predictions_bytes = sha256_file(predictions_path)
    metrics_path = destination / "stacking_fold_metrics.json"
    write_json(metrics_path, {"items": fold_metrics})
    metrics_hash, metrics_bytes = sha256_file(metrics_path)

    aggregate = aggregate_candidate_metrics(fold_metrics)
    base_aggregate = [
        {**row, "candidateId": row["modelId"], "family": "base"}
        for row in base_manifest.get("aggregate", [])
    ]
    comparison = sorted([*base_aggregate, *aggregate], key=lambda row: row["prAucMean"], reverse=True)
    leading_stacking = next((item for item in aggregate if item["family"] == "stacking"), None)
    overall_leader = comparison[0] if comparison else None
    manifest = {
        "schemaVersion": "1.0.0",
        "runId": f"{base_manifest.get('runId')}_stacking",
        "domain": "phishing",
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "status": "demo" if base_manifest.get("status") == "demo" else "meta_oof_candidate",
        "baseRun": {
            "runId": base_manifest.get("runId"),
            "manifestPath": str(base_path.resolve()),
            "oofPath": str(oof_path.resolve()),
            "oofSha256": actual_hash,
            "rows": int(len(oof)),
            "models": list(model_ids),
        },
        "metaFeatures": {
            "columns": feature_names,
            "baseProbabilityColumns": probability_columns,
            "summaryFeatures": ["probability_mean", "probability_std", "probability_range", "probability_min", "probability_max", "vote_fraction", "mean_absolute_disagreement"],
        },
        "validation": {
            "strategy": "cross_fitted_meta_models_using_original_group_folds",
            "foldProtocols": fold_protocols,
            "coverageRows": int(len(predictions)),
            "directSampleOverlapPassed": True,
            "strictNestedCrossFit": False,
            "metaDependencyLeakageControlled": False,
            "outerValidationUsed": False,
            "testSetLocked": True,
            "testSetUsed": False,
            "interpretation": "exploratory_oof_diagnostic_not_final_model_selection",
        },
        "candidates": list(ENSEMBLE_IDS),
        "foldMetrics": fold_metrics,
        "aggregate": aggregate,
        "comparison": comparison,
        "recommendation": {
            "leadingStackingCandidateId": leading_stacking["candidateId"] if leading_stacking else None,
            "overallOofLeaderId": overall_leader["candidateId"] if overall_leader else None,
            "primaryMetric": "prAuc",
            "status": "exploratory_only_pending_outer_validation",
        },
        "thresholdPolicy": {
            "current": 0.5,
            "status": "exploratory_only",
            "finalSelectionData": "outer_validation",
            "testSetUsed": False,
        },
        "artifacts": {
            "predictions": {"path": str(predictions_path.resolve()), "sha256": predictions_hash, "bytes": predictions_bytes, "rows": int(len(predictions))},
            "metrics": {"path": str(metrics_path.resolve()), "sha256": metrics_hash, "bytes": metrics_bytes, "rows": int(len(fold_metrics))},
            "fittedObjects": artifacts,
        },
    }
    manifest_path = destination / "stacking_manifest.json"
    write_json(manifest_path, manifest)
    return manifest


def build_meta_features(
    base_probabilities: np.ndarray,
    probability_columns: list[str],
) -> tuple[np.ndarray, list[str]]:
    values = np.asarray(base_probabilities, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(probability_columns):
        raise ValueError("La matriz de probabilidades no coincide con las columnas base.")
    mean = values.mean(axis=1, keepdims=True)
    summary = np.column_stack([
        mean.reshape(-1),
        values.std(axis=1),
        np.ptp(values, axis=1),
        values.min(axis=1),
        values.max(axis=1),
        (values >= 0.5).mean(axis=1),
        np.abs(values - mean).mean(axis=1),
    ])
    names = [
        *probability_columns,
        "probability_mean",
        "probability_std",
        "probability_range",
        "probability_min",
        "probability_max",
        "vote_fraction",
        "mean_absolute_disagreement",
    ]
    return np.column_stack([values, summary]), names


def fit_nonnegative_weights(base_probabilities: np.ndarray, labels: np.ndarray) -> np.ndarray:
    values = np.asarray(base_probabilities, dtype=np.float64)
    target = np.asarray(labels, dtype=np.int32)
    initial = np.full(values.shape[1], 1.0 / values.shape[1])

    def objective(weights: np.ndarray) -> float:
        probabilities = np.clip(values @ weights, 1e-7, 1 - 1e-7)
        return float(-np.mean(target * np.log(probabilities) + (1 - target) * np.log(1 - probabilities)))

    result = minimize(
        objective,
        initial,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * values.shape[1],
        constraints={"type": "eq", "fun": lambda weights: float(weights.sum() - 1.0)},
        options={"maxiter": 500, "ftol": 1e-10},
    )
    if not result.success or not np.isfinite(result.x).all():
        return initial
    weights = np.clip(result.x, 0.0, 1.0)
    return weights / weights.sum()


def build_meta_models(*, seed: int) -> dict[str, Any]:
    return {
        "stacking_logistic": make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=1.0,
                class_weight="balanced",
                max_iter=2_000,
                random_state=seed,
                solver="liblinear",
            ),
        ),
        "stacking_ridge": make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=0.1,
                penalty="l2",
                class_weight="balanced",
                max_iter=2_000,
                random_state=seed,
                solver="liblinear",
            ),
        ),
        "stacking_gradient_boosting": GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=2,
            subsample=0.8,
            random_state=seed,
        ),
    }


def aggregate_candidate_metrics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    frame = pd.DataFrame(rows)
    metric_names = ["prAuc", "rocAuc", "f1", "precision", "recall", "mcc", "balancedAccuracy", "falsePositiveRate"]
    result = []
    for candidate_id, group in frame.groupby("candidateId", sort=False):
        item: dict[str, Any] = {
            "candidateId": str(candidate_id),
            "family": str(group["family"].iloc[0]),
            "foldEvaluations": int(len(group)),
        }
        for metric in metric_names:
            item[f"{metric}Mean"] = float(group[metric].mean())
            item[f"{metric}Std"] = float(group[metric].std(ddof=1)) if len(group) > 1 else 0.0
        result.append(item)
    return sorted(result, key=lambda item: item["prAucMean"], reverse=True)


def get_latest_phishing_stacking() -> dict[str, Any]:
    paths = sorted(
        EXPERIMENTS_DIR.glob("*/phishing_oof_v1/stacking_v1/stacking_manifest.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not paths:
        return {"available": False, "message": "Todavía no existe una corrida Stacking de phishing."}
    manifest = read_json(paths[0])
    return {"available": True, **manifest}


def _latest_base_manifest() -> Path | None:
    paths = sorted(
        EXPERIMENTS_DIR.glob("*/phishing_oof_v1/oof_manifest.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return paths[0] if paths else None
