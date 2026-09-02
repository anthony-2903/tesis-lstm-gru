from __future__ import annotations

from datetime import datetime, timezone
import json
import os
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
from app.finance.metrics import finance_classification_metrics
from app.finance.models import FINANCE_MODEL_IDS
from app.phishing.ingestion import sha256_file
from app.utils import read_json


BASELINE_IDS = ("mean", "voting", "weighted_mean")
META_MODEL_IDS = ("stacking_logistic", "stacking_gradient_boosting")
ENSEMBLE_IDS = (*BASELINE_IDS, *META_MODEL_IDS)
SUMMARY_FEATURES = (
    "probability_mean",
    "probability_std",
    "probability_range",
    "probability_min",
    "probability_max",
    "vote_fraction",
    "mean_absolute_disagreement",
)


def run_finance_stacking_experiment(
    *,
    base_manifest_path: Path | None = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    base_path = base_manifest_path or _latest_base_manifest()
    if base_path is None or not base_path.is_file():
        raise FileNotFoundError("No existe una corrida OOF financiera para construir Stacking.")
    base_manifest = read_json(base_path)
    validation = base_manifest.get("validation", {})
    if not validation.get("testSetLocked") or validation.get("testSetUsed") or validation.get("testSetEncoded"):
        raise ValueError("La corrida base no conserva el bloqueo del test financiero.")
    if validation.get("externalValidationUsed"):
        raise ValueError("La validación externa no puede participar en el cross-fitting del Stacking.")

    model_ids = tuple(base_manifest.get("baseModels", []))
    if set(model_ids) != set(FINANCE_MODEL_IDS):
        raise ValueError("El Stacking financiero requiere las cinco arquitecturas base completas.")
    probability_columns = [f"probability_{model_id}" for model_id in model_ids]
    oof_artifact = base_manifest.get("artifacts", {}).get("oofProbabilities", {})
    oof_path = Path(oof_artifact.get("path", ""))
    if not oof_path.is_file():
        raise FileNotFoundError("No existe el artefacto financiero de probabilidades OOF.")
    oof_hash, oof_bytes = sha256_file(oof_path)
    if oof_hash != oof_artifact.get("sha256") or oof_bytes != oof_artifact.get("bytes"):
        raise ValueError("Las probabilidades OOF financieras no superan la verificación SHA-256.")
    oof = pd.read_csv(oof_path)
    required = {"transaction_id", "seed", "fold", "is_fraud", *probability_columns}
    missing = required - set(oof.columns)
    if missing:
        raise ValueError(f"El OOF financiero no contiene: {', '.join(sorted(missing))}")
    if oof.duplicated(["seed", "transaction_id"]).any():
        raise ValueError("Cada transacción OOF debe aparecer una sola vez por semilla.")
    values = oof[probability_columns].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("Las probabilidades OOF base son inválidas.")

    destination = output_dir or (base_path.parent / "stacking_v1")
    destination.mkdir(parents=True, exist_ok=True)
    prediction_rows: list[dict[str, Any]] = []
    candidate_fold_metrics: list[dict[str, Any]] = []
    base_fold_metrics: list[dict[str, Any]] = []
    fitted_artifacts: list[dict[str, Any]] = []
    fold_protocols: list[dict[str, Any]] = []
    warmup_rows = 0

    for seed, seed_frame in oof.groupby("seed", sort=True):
        folds = sorted(int(value) for value in seed_frame["fold"].unique())
        if len(folds) != 5:
            raise ValueError("El Stacking financiero requiere los cinco folds OOF base.")
        warmup_rows += int((seed_frame["fold"] == folds[0]).sum())
        for fold in folds[1:]:
            meta_train = seed_frame[seed_frame["fold"] < fold].copy().sort_values(["fold", "transaction_id"], kind="stable")
            meta_holdout = seed_frame[seed_frame["fold"] == fold].copy().sort_values("transaction_id", kind="stable")
            if meta_train.empty or meta_holdout.empty:
                raise ValueError(f"El fold meta-financiero {fold} no tiene ajuste u holdout.")
            if int(meta_train["fold"].max()) >= fold:
                raise ValueError("El metamodelo financiero recibió información futura.")
            if set(meta_train["transaction_id"]) & set(meta_holdout["transaction_id"]):
                raise ValueError("Existe fuga directa de transacciones en el metamodelo financiero.")

            x_train_base = meta_train[probability_columns].to_numpy(dtype=np.float64)
            x_holdout_base = meta_holdout[probability_columns].to_numpy(dtype=np.float64)
            y_train = meta_train["is_fraud"].to_numpy(dtype=np.int32)
            y_holdout = meta_holdout["is_fraud"].to_numpy(dtype=np.int32)
            if set(np.unique(y_train)) != {0, 1} or set(np.unique(y_holdout)) != {0, 1}:
                raise ValueError("Cada ajuste y holdout meta-financiero debe contener ambas clases.")
            x_train_meta, feature_names = build_finance_meta_features(x_train_base, probability_columns)
            x_holdout_meta, _ = build_finance_meta_features(x_holdout_base, probability_columns)

            candidate_predictions: dict[str, np.ndarray] = {
                "mean": x_holdout_base.mean(axis=1),
                "voting": (x_holdout_base >= 0.5).mean(axis=1),
            }
            weights = fit_nonnegative_finance_weights(x_train_base, y_train)
            candidate_predictions["weighted_mean"] = x_holdout_base @ weights
            weights_path = destination / "weights" / f"seed_{int(seed)}_fold_{fold}.json"
            _atomic_write_json(
                weights_path,
                {
                    "schemaVersion": "1.0.0",
                    "seed": int(seed),
                    "fold": int(fold),
                    "fitFolds": sorted(int(value) for value in meta_train["fold"].unique()),
                    "objective": "binary_log_loss",
                    "constraints": "nonnegative_weights_sum_to_one",
                    "weights": {model_id: float(weight) for model_id, weight in zip(model_ids, weights)},
                    "fitRows": int(len(meta_train)),
                    "futureFoldsUsed": False,
                },
            )
            weight_hash, weight_bytes = sha256_file(weights_path)
            fitted_artifacts.append(
                {
                    "candidateId": "weighted_mean",
                    "seed": int(seed),
                    "fold": int(fold),
                    "path": str(weights_path.resolve()),
                    "sha256": weight_hash,
                    "bytes": weight_bytes,
                }
            )

            meta_models = build_finance_meta_models(seed=int(seed) + fold)
            for candidate_id, model in meta_models.items():
                if candidate_id == "stacking_gradient_boosting":
                    model.fit(x_train_meta, y_train, sample_weight=_balanced_sample_weights(y_train))
                else:
                    model.fit(x_train_meta, y_train)
                candidate_predictions[candidate_id] = np.clip(
                    model.predict_proba(x_holdout_meta)[:, 1].astype(np.float64), 0.0, 1.0
                )
                model_path = destination / "models" / f"seed_{int(seed)}" / f"fold_{fold}" / f"{candidate_id}.joblib"
                _atomic_joblib_dump(model_path, model)
                model_hash, model_bytes = sha256_file(model_path)
                fitted_artifacts.append(
                    {
                        "candidateId": candidate_id,
                        "seed": int(seed),
                        "fold": int(fold),
                        "path": str(model_path.resolve()),
                        "sha256": model_hash,
                        "bytes": model_bytes,
                    }
                )

            for candidate_id, probabilities in candidate_predictions.items():
                candidate_fold_metrics.append(
                    {
                        "candidateId": candidate_id,
                        "family": "baseline" if candidate_id in BASELINE_IDS else "stacking",
                        "seed": int(seed),
                        "fold": int(fold),
                        **finance_classification_metrics(y_holdout, probabilities, threshold=0.5),
                    }
                )
            for model_id, probability_column in zip(model_ids, probability_columns):
                base_fold_metrics.append(
                    {
                        "candidateId": model_id,
                        "family": "base",
                        "seed": int(seed),
                        "fold": int(fold),
                        **finance_classification_metrics(
                            y_holdout,
                            meta_holdout[probability_column].to_numpy(dtype=np.float64),
                            threshold=0.5,
                        ),
                    }
                )
            for row_position, (_, row) in enumerate(meta_holdout.iterrows()):
                prediction_rows.append(
                    {
                        "transaction_id": int(row["transaction_id"]),
                        "seed": int(seed),
                        "fold": int(fold),
                        "is_fraud": int(row["is_fraud"]),
                        **{
                            f"probability_{candidate_id}": float(candidate_predictions[candidate_id][row_position])
                            for candidate_id in ENSEMBLE_IDS
                        },
                    }
                )
            fold_protocols.append(
                {
                    "seed": int(seed),
                    "holdoutFold": int(fold),
                    "fitFolds": sorted(int(value) for value in meta_train["fold"].unique()),
                    "fitRows": int(len(meta_train)),
                    "holdoutRows": int(len(meta_holdout)),
                    "transactionOverlap": 0,
                    "futureFoldsUsed": False,
                    "featureCount": len(feature_names),
                }
            )

    predictions = pd.DataFrame(prediction_rows).sort_values(["seed", "fold", "transaction_id"], kind="stable")
    expected_rows = int(sum((oof["fold"] != oof.groupby("seed")["fold"].transform("min"))))
    if len(predictions) != expected_rows or predictions.duplicated(["seed", "transaction_id"]).any():
        raise ValueError("El cross-fitting financiero no cubre exactamente los folds evaluables.")
    ensemble_columns = [f"probability_{candidate_id}" for candidate_id in ENSEMBLE_IDS]
    if predictions[ensemble_columns].isna().any().any():
        raise ValueError("Existen predicciones meta-financieras faltantes.")

    predictions_path = destination / "stacking_temporal_probabilities.csv"
    _atomic_write_csv(predictions_path, predictions)
    predictions_hash, predictions_bytes = sha256_file(predictions_path)
    metrics_path = destination / "stacking_fold_metrics.json"
    _atomic_write_json(metrics_path, {"candidateItems": candidate_fold_metrics, "baseItems": base_fold_metrics})
    metrics_hash, metrics_bytes = sha256_file(metrics_path)

    candidate_aggregate = aggregate_finance_candidate_metrics(candidate_fold_metrics)
    base_aggregate = aggregate_finance_candidate_metrics(base_fold_metrics)
    comparison = sorted([*base_aggregate, *candidate_aggregate], key=lambda item: item["prAucMean"], reverse=True)
    stacking_candidates = [item for item in candidate_aggregate if item["family"] == "stacking"]
    leading_stacking = stacking_candidates[0] if stacking_candidates else None
    base_leader = base_aggregate[0] if base_aggregate else None
    six_model_comparison = sorted(
        [
            *base_aggregate,
            *(
                [
                    {
                        **leading_stacking,
                        "candidateId": "stacking",
                        "sourceCandidateId": leading_stacking["candidateId"],
                        "family": "stacking",
                    }
                ]
                if leading_stacking
                else []
            ),
        ],
        key=lambda item: item["prAucMean"],
        reverse=True,
    )
    overall_leader = six_model_comparison[0] if six_model_comparison else None
    manifest = {
        "schemaVersion": "1.0.0",
        "runId": f"{base_manifest.get('runId')}_stacking",
        "domain": "finanzas",
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "status": "demo" if base_manifest.get("status") == "demo" else "meta_oof_candidate",
        "baseRun": {
            "runId": base_manifest.get("runId"),
            "manifestPath": str(base_path.resolve()),
            "oofPath": str(oof_path.resolve()),
            "oofSha256": oof_hash,
            "rows": int(len(oof)),
            "models": list(model_ids),
        },
        "metaFeatures": {
            "columns": feature_names,
            "baseProbabilityColumns": probability_columns,
            "summaryFeatures": list(SUMMARY_FEATURES),
            "targetIncluded": False,
        },
        "validation": {
            "strategy": "expanding_temporal_cross_fit_on_prior_oof_folds",
            "foldProtocols": fold_protocols,
            "evaluatedFolds": [1, 2, 3, 4],
            "coverageRows": int(len(predictions)),
            "warmupRowsExcluded": int(warmup_rows),
            "sameRowsForAllCandidates": True,
            "directTransactionOverlapPassed": True,
            "strictTemporalCrossFit": True,
            "metaDependencyLeakageControlled": True,
            "futureFoldsUsed": False,
            "externalValidationUsed": False,
            "testSetLocked": True,
            "testSetEncoded": False,
            "testSetUsed": False,
            "interpretation": "exploratory_temporal_oof_selection_pending_external_validation",
        },
        "candidates": list(ENSEMBLE_IDS),
        "candidateFoldMetrics": candidate_fold_metrics,
        "baseFoldMetricsOnCommonRows": base_fold_metrics,
        "candidateAggregate": candidate_aggregate,
        "baseAggregateOnCommonRows": base_aggregate,
        "comparison": comparison,
        "sixModelComparison": six_model_comparison,
        "recommendation": {
            "leadingStackingCandidateId": leading_stacking["candidateId"] if leading_stacking else None,
            "baseLeaderId": base_leader["candidateId"] if base_leader else None,
            "overallSixModelLeaderId": overall_leader["candidateId"] if overall_leader else None,
            "stackingBeatsBestBaseOnMeanPrAuc": bool(
                leading_stacking and base_leader and leading_stacking["prAucMean"] > base_leader["prAucMean"]
            ),
            "primaryMetric": "prAuc",
            "status": "provisional_only_pending_external_validation_and_statistical_testing",
        },
        "thresholdPolicy": {
            "current": 0.5,
            "status": "exploratory_only",
            "finalSelectionData": "external_validation",
            "testSetUsed": False,
        },
        "artifacts": {
            "predictions": {
                "path": str(predictions_path.resolve()),
                "sha256": predictions_hash,
                "bytes": predictions_bytes,
                "rows": int(len(predictions)),
            },
            "metrics": {
                "path": str(metrics_path.resolve()),
                "sha256": metrics_hash,
                "bytes": metrics_bytes,
                "rows": int(len(candidate_fold_metrics) + len(base_fold_metrics)),
            },
            "fittedObjects": fitted_artifacts,
        },
    }
    manifest_path = destination / "stacking_manifest.json"
    _atomic_write_json(manifest_path, manifest)
    manifest_hash, manifest_bytes = sha256_file(manifest_path)
    return {**manifest, "manifest": {"path": str(manifest_path.resolve()), "sha256": manifest_hash, "bytes": manifest_bytes}}


def build_finance_meta_features(
    base_probabilities: np.ndarray,
    probability_columns: list[str],
) -> tuple[np.ndarray, list[str]]:
    values = np.asarray(base_probabilities, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(probability_columns):
        raise ValueError("La matriz financiera no coincide con las probabilidades base.")
    if not np.isfinite(values).all() or np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("Las meta-features financieras contienen probabilidades inválidas.")
    mean = values.mean(axis=1, keepdims=True)
    summary = np.column_stack(
        [
            mean.reshape(-1),
            values.std(axis=1),
            np.ptp(values, axis=1),
            values.min(axis=1),
            values.max(axis=1),
            (values >= 0.5).mean(axis=1),
            np.abs(values - mean).mean(axis=1),
        ]
    )
    return np.column_stack([values, summary]), [*probability_columns, *SUMMARY_FEATURES]


def fit_nonnegative_finance_weights(base_probabilities: np.ndarray, labels: np.ndarray) -> np.ndarray:
    values = np.asarray(base_probabilities, dtype=np.float64)
    target = np.asarray(labels, dtype=np.int32)
    initial = np.full(values.shape[1], 1.0 / values.shape[1])

    def objective(weights: np.ndarray) -> float:
        probabilities = np.clip(values @ weights, 1e-7, 1.0 - 1e-7)
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
    total = float(weights.sum())
    return weights / total if total > 0.0 else initial


def build_finance_meta_models(*, seed: int) -> dict[str, Any]:
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
        "stacking_gradient_boosting": GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=2,
            subsample=0.8,
            random_state=seed,
        ),
    }


def aggregate_finance_candidate_metrics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    frame = pd.DataFrame(rows)
    metrics = [
        "prAuc",
        "rocAuc",
        "f1",
        "precision",
        "recall",
        "mcc",
        "balancedAccuracy",
        "brierScore",
        "logLoss",
        "falsePositiveRate",
    ]
    result: list[dict[str, Any]] = []
    for candidate_id, group in frame.groupby("candidateId", sort=False):
        item: dict[str, Any] = {
            "candidateId": str(candidate_id),
            "family": str(group["family"].iloc[0]),
            "foldEvaluations": int(len(group)),
        }
        for metric in metrics:
            item[f"{metric}Mean"] = float(group[metric].mean())
            item[f"{metric}Std"] = float(group[metric].std(ddof=1)) if len(group) > 1 else 0.0
        result.append(item)
    return sorted(result, key=lambda item: item["prAucMean"], reverse=True)


def get_latest_finance_stacking() -> dict[str, Any]:
    paths = sorted(
        EXPERIMENTS_DIR.glob("*/finance_oof_v1/stacking_v1/stacking_manifest.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not paths:
        return {"available": False, "message": "Todavía no existe una corrida de Stacking financiero."}
    manifest = read_json(paths[0])
    return {"available": True, **manifest}


def _balanced_sample_weights(labels: np.ndarray) -> np.ndarray:
    values, counts = np.unique(labels, return_counts=True)
    mapping = {int(value): float(len(labels) / (len(values) * count)) for value, count in zip(values, counts)}
    return np.asarray([mapping[int(value)] for value in labels], dtype=np.float64)


def _atomic_joblib_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    joblib.dump(value, temporary)
    os.replace(temporary, path)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _latest_base_manifest() -> Path | None:
    paths = sorted(
        EXPERIMENTS_DIR.glob("*/finance_oof_v1/oof_manifest.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return paths[0] if paths else None
