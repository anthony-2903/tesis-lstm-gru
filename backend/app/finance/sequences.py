from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.config import RESULTS_DIR, SILVER_DIR, ensure_dirs
from app.finance.benchmark import get_finance_dataset_status
from app.phishing.ingestion import sha256_file
from app.utils import read_json


FEATURE_COLUMNS = (
    "log_amount",
    "hour_sin",
    "hour_cos",
    "weekday_sin",
    "weekday_cos",
    "log_seconds_since_previous",
    "log_customer_tx_count_prior",
    "log_customer_amount_mean_prior",
    "amount_zscore_prior",
    "log_terminal_tx_count_prior",
)
MASK_COLUMN = "observed_mask"


def prepare_finance_sequence_protocol(
    *,
    window: int = 10,
    folds: int = 5,
    purge_days: int = 1,
    silver_path: Path | None = None,
    metadata_path: Path | None = None,
    audit_path: Path | None = None,
    sequences_path: Path | None = None,
    features_path: Path | None = None,
    assignments_path: Path | None = None,
    manifest_path: Path | None = None,
) -> dict[str, Any]:
    if window < 2:
        raise ValueError("La ventana financiera debe contener al menos dos transacciones.")
    if folds != 5:
        raise ValueError("El protocolo financiero de tesis usa exactamente cinco folds OOF.")
    if purge_days < 0:
        raise ValueError("purge_days no puede ser negativo.")
    ensure_dirs()
    source = silver_path or (SILVER_DIR / "finance_transactions.csv")
    metadata_source = metadata_path or (SILVER_DIR / "finance_transactions.metadata.json")
    sequence_destination = sequences_path or (SILVER_DIR / "finance_sequences.npz")
    feature_destination = features_path or (SILVER_DIR / "finance_causal_features.csv")
    assignment_destination = assignments_path or (SILVER_DIR / "finance_oof_assignments.csv")
    manifest_destination = manifest_path or (SILVER_DIR / "finance_sequence_manifest.json")
    if not source.is_file() or not metadata_source.is_file():
        raise FileNotFoundError("Primero prepare el benchmark financiero transaccional.")

    metadata = read_json(metadata_source)
    source_hash, source_bytes = sha256_file(source)
    if metadata.get("silver", {}).get("sha256") != source_hash:
        raise ValueError("El silver financiero cambió respecto de sus metadatos versionados.")
    dataset_status = get_finance_dataset_status(metadata_path=metadata_source, audit_path=audit_path)
    if not dataset_status.get("readyForPipelinePilot"):
        raise ValueError("El benchmark financiero no supera la puerta de datos para pipeline.")

    raw = pd.read_csv(source)
    development = raw[raw["split"].isin(["train", "validation"])].copy()
    test_rows = int((raw["split"] == "test").sum())
    if development.empty or test_rows == 0:
        raise ValueError("Se requieren development y test bloqueado en el silver financiero.")
    features = build_causal_finance_features(development)
    x, labels, transaction_ids, customer_ids, timestamps, split_codes = build_customer_sequences(features, window=window)
    train_mask = features["split"].eq("train").to_numpy()
    validation_mask = features["split"].eq("validation").to_numpy()
    fold_protocols, assignments = build_finance_temporal_oof(
        features,
        folds=folds,
        purge_days=purge_days,
    )

    _atomic_write_csv(feature_destination, features)
    _atomic_write_csv(assignment_destination, assignments)
    _atomic_write_npz(
        sequence_destination,
        x=x,
        y=labels,
        transaction_id=transaction_ids,
        customer_id=customer_ids,
        timestamp_ns=timestamps,
        split_code=split_codes,
    )
    feature_hash, feature_bytes = sha256_file(feature_destination)
    assignment_hash, assignment_bytes = sha256_file(assignment_destination)
    sequence_hash, sequence_bytes = sha256_file(sequence_destination)
    prepared_at = datetime.now(timezone.utc).isoformat()
    manifest = {
        "schemaVersion": "1.0.0",
        "datasetId": metadata.get("datasetId"),
        "domain": "finanzas",
        "createdAt": prepared_at,
        "source": {
            "path": str(source.resolve()),
            "sha256": source_hash,
            "bytes": source_bytes,
        },
        "configuration": {
            "window": window,
            "folds": folds,
            "purgeDays": purge_days,
            "padding": "left_zero_padding_with_observed_mask",
            "sequenceOrder": "oldest_to_current_transaction",
        },
        "features": {
            "columns": list(FEATURE_COLUMNS),
            "sequenceColumns": [*FEATURE_COLUMNS, MASK_COLUMN],
            "target": "is_fraud",
            "targetUsedAsFeature": False,
            "historicalLabelsUsedAsFeatures": False,
            "causalPolicy": "Every aggregate uses strictly previous transactions; ties are ordered by transaction_id.",
            "validationOnlinePolicy": "Validation may use earlier unlabeled validation events as causal context, never their labels.",
        },
        "sequences": {
            "rows": int(len(features)),
            "developmentRows": int(len(features)),
            "trainingRows": int(train_mask.sum()),
            "externalValidationRows": int(validation_mask.sum()),
            "testRowsAvailable": test_rows,
            "testRowsEncoded": 0,
            "shape": [int(value) for value in x.shape],
        },
        "oof": {
            "strategy": "expanding_temporal_folds_by_calendar_day",
            "folds": fold_protocols,
            "coverageRows": int(len(assignments)),
            "warmupRowsExcluded": int(train_mask.sum() - len(assignments)),
            "assignmentUniquenessPassed": not assignments["transaction_id"].duplicated().any(),
            "futureRowsUsedForFit": False,
            "scalerPolicy": "fit_on_each_fold_training_prefix_only",
        },
        "artifacts": {
            "features": {"path": str(feature_destination.resolve()), "sha256": feature_hash, "bytes": feature_bytes, "rows": int(len(features))},
            "sequences": {"path": str(sequence_destination.resolve()), "sha256": sequence_hash, "bytes": sequence_bytes, "rows": int(len(features))},
            "assignments": {"path": str(assignment_destination.resolve()), "sha256": assignment_hash, "bytes": assignment_bytes, "rows": int(len(assignments))},
        },
        "testLock": {
            "locked": True,
            "evaluated": False,
            "encoded": False,
            "rows": test_rows,
            "policy": "Test is absent from causal feature engineering, sequence tensors, scalers and OOF assignments.",
        },
        "readiness": {
            "readyForBaseModelPilot": len(fold_protocols) == 5 and len(assignments) > 0,
            "readyForThesisTraining": bool(dataset_status.get("readyForThesisTraining")) and len(fold_protocols) == 5 and len(assignments) > 0,
            "reason": (
                "Real curated dataset and temporal sequence protocol satisfy the thesis data gate."
                if dataset_status.get("readyForThesisTraining")
                else "Protocol implementation is ready; the current financial benchmark remains synthetic."
            ),
        },
    }
    _atomic_write_json(manifest_destination, manifest)
    return manifest


def build_causal_finance_features(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"transaction_id", "transaction_time", "customer_id", "terminal_id", "amount", "is_fraud", "split"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Faltan columnas para variables causales: {', '.join(sorted(missing))}")
    data = frame.copy()
    data["transaction_time"] = pd.to_datetime(data["transaction_time"], errors="coerce")
    data["amount"] = pd.to_numeric(data["amount"], errors="coerce")
    if data["transaction_time"].isna().any() or data["amount"].isna().any():
        raise ValueError("Las transacciones contienen fechas o montos inválidos.")
    data = data.sort_values(["transaction_time", "transaction_id"], kind="stable").reset_index(drop=True)

    customer_group = data.groupby("customer_id", sort=False)
    prior_count = customer_group.cumcount().astype(np.float64)
    prior_sum = customer_group["amount"].cumsum() - data["amount"]
    prior_square_sum = data.assign(_amount_square=data["amount"] ** 2).groupby("customer_id", sort=False)["_amount_square"].cumsum() - data["amount"] ** 2
    prior_mean = (prior_sum / prior_count.replace(0, np.nan)).fillna(0.0)
    prior_variance = (prior_square_sum / prior_count.replace(0, np.nan) - prior_mean ** 2).clip(lower=0.0).fillna(0.0)
    prior_std = np.sqrt(prior_variance)
    previous_time = customer_group["transaction_time"].shift(1)
    seconds_since_previous = (data["transaction_time"] - previous_time).dt.total_seconds().clip(lower=0.0).fillna(0.0)
    terminal_prior_count = data.groupby("terminal_id", sort=False).cumcount().astype(np.float64)
    hour = data["transaction_time"].dt.hour + data["transaction_time"].dt.minute / 60.0
    weekday = data["transaction_time"].dt.dayofweek.astype(np.float64)

    result = data[["transaction_id", "transaction_time", "customer_id", "terminal_id", "is_fraud", "split"]].copy()
    result["log_amount"] = np.log1p(data["amount"].clip(lower=0.0))
    result["hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
    result["hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)
    result["weekday_sin"] = np.sin(2.0 * np.pi * weekday / 7.0)
    result["weekday_cos"] = np.cos(2.0 * np.pi * weekday / 7.0)
    result["log_seconds_since_previous"] = np.log1p(seconds_since_previous)
    result["log_customer_tx_count_prior"] = np.log1p(prior_count)
    result["log_customer_amount_mean_prior"] = np.log1p(prior_mean.clip(lower=0.0))
    result["amount_zscore_prior"] = np.where(prior_count > 1, (data["amount"] - prior_mean) / np.maximum(prior_std, 1.0), 0.0)
    result["amount_zscore_prior"] = result["amount_zscore_prior"].clip(-10.0, 10.0)
    result["log_terminal_tx_count_prior"] = np.log1p(terminal_prior_count)
    if not np.isfinite(result[list(FEATURE_COLUMNS)].to_numpy(dtype=np.float64)).all():
        raise ValueError("Las variables financieras causales contienen valores no finitos.")
    return result


def build_customer_sequences(frame: pd.DataFrame, *, window: int) -> tuple[np.ndarray, ...]:
    feature_matrix = frame[list(FEATURE_COLUMNS)].to_numpy(dtype=np.float32)
    row_count = len(frame)
    x = np.zeros((row_count, window, len(FEATURE_COLUMNS) + 1), dtype=np.float32)
    history: dict[int, list[int]] = {}
    customer_values = frame["customer_id"].to_numpy(dtype=np.int64)
    for row_index, customer_id in enumerate(customer_values):
        previous = history.setdefault(int(customer_id), [])
        sequence_indices = [*previous[-(window - 1):], row_index]
        start = window - len(sequence_indices)
        x[row_index, start:, :-1] = feature_matrix[sequence_indices]
        x[row_index, start:, -1] = 1.0
        previous.append(row_index)
    timestamps = pd.to_datetime(frame["transaction_time"]).astype("int64").to_numpy(dtype=np.int64)
    split_codes = frame["split"].map({"train": 0, "validation": 1}).to_numpy(dtype=np.int8)
    return (
        x,
        frame["is_fraud"].to_numpy(dtype=np.int8),
        frame["transaction_id"].to_numpy(dtype=np.int64),
        customer_values,
        timestamps,
        split_codes,
    )


def build_finance_temporal_oof(
    frame: pd.DataFrame,
    *,
    folds: int = 5,
    purge_days: int = 1,
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    train = frame[frame["split"] == "train"].copy()
    train["calendar_day"] = pd.to_datetime(train["transaction_time"]).dt.normalize()
    unique_days = np.asarray(sorted(train["calendar_day"].unique()))
    blocks = np.array_split(unique_days, folds + 1)
    if any(len(block) == 0 for block in blocks):
        raise ValueError("No existen suficientes días para cinco folds temporales.")
    protocols: list[dict[str, Any]] = []
    assignments: list[pd.DataFrame] = []
    for fold in range(folds):
        holdout_days = blocks[fold + 1]
        holdout_start = pd.Timestamp(holdout_days[0])
        holdout_end = pd.Timestamp(holdout_days[-1])
        fit_cutoff = holdout_start - pd.Timedelta(days=purge_days)
        fit = train[train["calendar_day"] < fit_cutoff].copy()
        holdout = train[train["calendar_day"].isin(holdout_days)].copy()
        if fit.empty or holdout.empty:
            raise ValueError(f"El fold {fold} no contiene prefijo de ajuste u holdout.")
        if pd.to_datetime(fit["transaction_time"]).max() >= pd.to_datetime(holdout["transaction_time"]).min():
            raise ValueError("Se detectó fuga temporal en los folds financieros.")
        scaler = fit_finance_scaler(fit)
        protocols.append({
            "fold": fold,
            "fitRows": int(len(fit)),
            "holdoutRows": int(len(holdout)),
            "fitStartAt": str(fit["transaction_time"].min()),
            "fitEndAt": str(fit["transaction_time"].max()),
            "holdoutStartAt": str(holdout["transaction_time"].min()),
            "holdoutEndAt": str(holdout["transaction_time"].max()),
            "purgeDays": purge_days,
            "fitFraudRows": int(fit["is_fraud"].sum()),
            "holdoutFraudRows": int(holdout["is_fraud"].sum()),
            "futureRowsUsedForFit": False,
            "scaler": scaler,
        })
        assignments.append(pd.DataFrame({
            "transaction_id": holdout["transaction_id"].to_numpy(dtype=np.int64),
            "row_index": holdout.index.to_numpy(dtype=np.int64),
            "oof_fold": fold,
        }))
    result = pd.concat(assignments, ignore_index=True).sort_values("row_index", kind="stable")
    if result["transaction_id"].duplicated().any():
        raise ValueError("Una transacción recibió más de un fold OOF financiero.")
    return protocols, result


def fit_finance_scaler(fit_frame: pd.DataFrame) -> dict[str, Any]:
    values = fit_frame[list(FEATURE_COLUMNS)].to_numpy(dtype=np.float64)
    mean = values.mean(axis=0)
    scale = values.std(axis=0, ddof=0)
    scale = np.where(scale < 1e-8, 1.0, scale)
    return {
        "fitPolicy": "fold_training_prefix_only",
        "featureColumns": list(FEATURE_COLUMNS),
        "mean": [float(value) for value in mean],
        "scale": [float(value) for value in scale],
        "fitRows": int(len(fit_frame)),
    }


def scale_finance_sequences(x: np.ndarray, scaler: dict[str, Any]) -> np.ndarray:
    result = np.asarray(x, dtype=np.float32).copy()
    mask = result[:, :, -1] > 0.5
    mean = np.asarray(scaler["mean"], dtype=np.float32)
    scale = np.asarray(scaler["scale"], dtype=np.float32)
    result[:, :, :-1][mask] = (result[:, :, :-1][mask] - mean) / scale
    result[:, :, :-1][~mask] = 0.0
    return result


def get_finance_sequence_status(
    *,
    manifest_path: Path | None = None,
    metadata_path: Path | None = None,
) -> dict[str, Any]:
    manifest_source = manifest_path or (SILVER_DIR / "finance_sequence_manifest.json")
    metadata_source = metadata_path or (SILVER_DIR / "finance_transactions.metadata.json")
    if not manifest_source.is_file() or not metadata_source.is_file():
        return {
            "available": False,
            "readyForBaseModelPilot": False,
            "readyForThesisTraining": False,
            "message": "El protocolo secuencial financiero todavía no ha sido preparado.",
            "testLock": {"locked": True, "evaluated": False, "encoded": False},
        }
    manifest = read_json(manifest_source)
    metadata = read_json(metadata_source)
    lineage_current = manifest.get("datasetId") == metadata.get("datasetId")
    source_lineage_current = True
    if metadata.get("provenance", {}).get("kind") == "real_world_curated_financial_dataset":
        source_items = [metadata.get("sourceManifest", {}), *metadata.get("sourceArtifacts", [])]
        source_lineage_current = bool(metadata.get("sourceManifest")) and bool(metadata.get("sourceArtifacts"))
        for item in source_items:
            path = Path(str(item.get("path", "")))
            if not path.is_file():
                source_lineage_current = False
                break
            digest, size = sha256_file(path)
            if digest != item.get("sha256") or size != item.get("bytes"):
                source_lineage_current = False
                break
    artifacts_valid = True
    for artifact in manifest.get("artifacts", {}).values():
        path = Path(artifact.get("path", ""))
        if not path.is_file():
            artifacts_valid = False
            break
        digest, size = sha256_file(path)
        if digest != artifact.get("sha256") or size != artifact.get("bytes"):
            artifacts_valid = False
            break
    ready = bool(manifest.get("readiness", {}).get("readyForBaseModelPilot")) and lineage_current and source_lineage_current and artifacts_valid
    thesis_ready = ready and bool(manifest.get("readiness", {}).get("readyForThesisTraining")) and bool(metadata.get("readyForThesisTraining"))
    return {
        "available": True,
        "datasetId": manifest.get("datasetId"),
        "createdAt": manifest.get("createdAt"),
        "configuration": manifest.get("configuration", {}),
        "features": manifest.get("features", {}),
        "sequences": manifest.get("sequences", {}),
        "oof": manifest.get("oof", {}),
        "artifacts": manifest.get("artifacts", {}),
        "testLock": manifest.get("testLock", {}),
        "lineageCurrent": lineage_current,
        "sourceLineageCurrent": source_lineage_current,
        "artifactIntegrity": artifacts_valid,
        "readyForBaseModelPilot": ready,
        "readyForThesisTraining": thesis_ready,
        "message": "Secuencias y cinco folds OOF listos para modelos base financieros." if ready else "El protocolo financiero no supera integridad o linaje.",
    }


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


def _atomic_write_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)
