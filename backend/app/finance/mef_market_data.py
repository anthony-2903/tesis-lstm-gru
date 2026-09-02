from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

import numpy as np
import pandas as pd
import requests

from app.config import RAW_DIR, RESULTS_DIR, SILVER_DIR, SOURCE_CONFIG, ensure_dirs
from app.utils import read_json, write_json


DATASET_SLUG = "indices-soberanos"
DATASET_ID = "574f11cb-f196-4c78-9e35-5d881113fe5c"
RESOURCE_ID = "c32056c1-7f91-4d4e-bc76-0c4844ac25fe"
PROVIDER = "Ministerio de Economia y Finanzas del Peru"
CATEGORY = "Endeudamiento y Tesoro Publico"
TIMESTAMP_COLUMN = "timestamp"
TARGET_COLUMN = "nominal_log_return_1d"
SOURCE_COLUMNS = {
    "FECHA": TIMESTAMP_COLUMN,
    "INDICE_NOMINAL": "nominal_index",
    "RENT_ANUAL_IN": "nominal_annual_yield",
    "INDICE_REAL": "real_index",
    "RENT_ANUAL_IR": "real_annual_yield",
}
FEATURE_COLUMNS = (
    TARGET_COLUMN,
    "real_log_return_1d",
    "nominal_index",
    "nominal_annual_yield",
    "real_index",
    "real_annual_yield",
    "weekday_sin",
    "weekday_cos",
    "month_sin",
    "month_cos",
)


def prepare_mef_market_dataset(
    *,
    source_url: str = SOURCE_CONFIG.mef_sovereign_indices_url,
    force_download: bool = False,
    minimum_rows: int = 1_000,
    raw_directory: Path | None = None,
    silver_path: Path | None = None,
    audit_path: Path | None = None,
    metadata_path: Path | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Download, version and audit the public MEF sovereign-index series.

    The target is the next observed session's nominal log-return. Anomaly labels
    are deliberately not fabricated: later stages may only describe unusually
    large out-of-sample residuals as estimated anomalies.
    """

    ensure_dirs()
    raw_root = raw_directory or (RAW_DIR / "finance_mef")
    destination = silver_path or (SILVER_DIR / "mef_sovereign_indices.csv")
    report_destination = audit_path or (RESULTS_DIR / "finance_mef_data_audit.json")
    metadata_destination = metadata_path or (SILVER_DIR / "mef_sovereign_indices.metadata.json")
    raw_root.mkdir(parents=True, exist_ok=True)

    _notify(progress_callback, stage="downloading", percent=5.0, message="Descargando o verificando el snapshot publico del MEF.")
    snapshot = download_versioned_mef_source(source_url, raw_root, force=force_download)
    _notify(progress_callback, stage="auditing", percent=60.0, message="Validando fechas, indices, rendimientos y causalidad temporal.")
    source = load_mef_sovereign_indices(Path(snapshot["rawPath"]))
    silver, audit = build_audited_mef_market_silver(source, minimum_rows=minimum_rows)

    _notify(progress_callback, stage="persisting", percent=85.0, message="Persistiendo silver, auditoria y linaje reproducible.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    silver.to_csv(destination, index=False)
    silver_sha256, silver_bytes = sha256_file(destination)
    prepared_at = datetime.now(timezone.utc).isoformat()
    dataset_id = f"mef-{DATASET_SLUG}-{silver_sha256[:12]}"
    metadata = {
        "schemaVersion": "1.0.0",
        "datasetId": dataset_id,
        "domain": "finanzas",
        "provider": PROVIDER,
        "category": CATEGORY,
        "datasetSlug": DATASET_SLUG,
        "catalogDatasetId": DATASET_ID,
        "resourceId": RESOURCE_ID,
        "sourceUrl": source_url,
        "dictionaryUrl": SOURCE_CONFIG.mef_sovereign_indices_dictionary_url,
        "retrievedAt": snapshot["retrievedAt"],
        "preparedAt": prepared_at,
        "rawSnapshot": snapshot,
        "scientificContract": {
            "task": "forecasting_with_residual_anomaly_detection",
            "unitOfAnalysis": "one_observed_financial_session",
            "target": TARGET_COLUMN,
            "targetDefinition": "100 * ln(nominal_index_t / nominal_index_t_minus_1)",
            "forecastHorizon": "next observed session",
            "frequency": "irregular_business_day_observations",
            "featureTiming": "only values known through the end of the input window",
            "groundTruthAnomalyLabels": False,
            "prohibitedClaim": "fraud_detection",
        },
        "splitPolicy": {
            "strategy": "chronological_expanding_window",
            "developmentFraction": 0.85,
            "lockedTestFraction": 0.15,
            "innerFolds": 5,
            "gapObservedSessions": 5,
            "testSetLocked": True,
            "testSetUsed": False,
            "testEvaluationAuthorized": False,
        },
        "silver": {
            "path": str(destination.resolve()),
            "sha256": silver_sha256,
            "bytes": silver_bytes,
            "rows": int(len(silver)),
            "columns": list(silver.columns),
        },
        "auditPath": str(report_destination.resolve()),
        "readyForThesisTraining": bool(audit["readiness"]["ready"]),
    }
    audit_payload = {"schemaVersion": "1.0.0", "datasetId": dataset_id, "createdAt": prepared_at, **audit}
    write_json(report_destination, audit_payload)
    write_json(metadata_destination, metadata)
    _notify(progress_callback, stage="completed", percent=100.0, message="Serie financiera MEF preparada; test final bloqueado.")
    return {"metadata": metadata, "audit": audit_payload}


def get_mef_market_dataset_status(
    *,
    metadata_path: Path | None = None,
    audit_path: Path | None = None,
) -> dict[str, Any]:
    metadata_file = metadata_path or (SILVER_DIR / "mef_sovereign_indices.metadata.json")
    audit_file = audit_path or (RESULTS_DIR / "finance_mef_data_audit.json")
    if not metadata_file.is_file() or not audit_file.is_file():
        return {
            "available": False,
            "readyForThesisTraining": False,
            "provider": PROVIDER,
            "category": CATEGORY,
            "datasetSlug": DATASET_SLUG,
            "message": "La serie publica de Indices Soberanos del MEF aun no ha sido preparada.",
            "testPolicy": {"testSetLocked": True, "testSetUsed": False, "testEvaluationAuthorized": False},
        }
    metadata = read_json(metadata_file)
    audit = read_json(audit_file)
    silver = metadata.get("silver", {})
    path = Path(str(silver.get("path", "")))
    hash_valid = False
    if path.is_file():
        actual_hash, actual_bytes = sha256_file(path)
        hash_valid = actual_hash == silver.get("sha256") and actual_bytes == silver.get("bytes")
    ready = bool(metadata.get("readyForThesisTraining")) and bool(audit.get("readiness", {}).get("ready")) and hash_valid
    return {
        "available": path.is_file(),
        "readyForThesisTraining": ready,
        "datasetId": metadata.get("datasetId"),
        "provider": metadata.get("provider", PROVIDER),
        "category": metadata.get("category", CATEGORY),
        "datasetSlug": metadata.get("datasetSlug", DATASET_SLUG),
        "resourceId": metadata.get("resourceId", RESOURCE_ID),
        "sourceUrl": metadata.get("sourceUrl"),
        "retrievedAt": metadata.get("retrievedAt"),
        "preparedAt": metadata.get("preparedAt"),
        "scientificContract": metadata.get("scientificContract", {}),
        "splitPolicy": metadata.get("splitPolicy", {}),
        "silver": silver,
        "audit": audit,
        "integrityVerified": hash_valid,
        "testPolicy": {"testSetLocked": True, "testSetUsed": False, "testEvaluationAuthorized": False},
        "message": "Dataset MEF listo para entrenamiento." if ready else "El dataset MEF no supera aun la auditoria de integridad.",
    }


def download_versioned_mef_source(source_url: str, raw_directory: Path, *, force: bool = False) -> dict[str, Any]:
    index_path = raw_directory / "latest.json"
    if not force and index_path.is_file():
        previous = read_json(index_path)
        previous_path = Path(str(previous.get("rawPath", "")))
        if previous.get("sourceUrl") == source_url and previous_path.is_file():
            actual_hash, actual_bytes = sha256_file(previous_path)
            if actual_hash == previous.get("sha256") and actual_bytes == previous.get("bytes"):
                return {**previous, "reused": True}

    retrieved_at = datetime.now(timezone.utc).isoformat()
    temporary = raw_directory / f"download_{uuid4().hex}.part"
    digest = hashlib.sha256()
    byte_count = 0
    headers = {"User-Agent": SOURCE_CONFIG.user_agent, "Accept": "text/csv,*/*"}
    try:
        with requests.get(source_url, headers=headers, timeout=(20, 120), stream=True) as response:
            response.raise_for_status()
            with temporary.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if not chunk:
                        continue
                    handle.write(chunk)
                    digest.update(chunk)
                    byte_count += len(chunk)
            response_headers = {
                "etag": response.headers.get("ETag"),
                "lastModified": response.headers.get("Last-Modified"),
                "contentType": response.headers.get("Content-Type"),
            }
    except Exception:
        temporary.unlink(missing_ok=True)
        raise

    sha256 = digest.hexdigest()
    final_path = raw_directory / f"indices_soberanos_{sha256[:12]}.csv"
    if final_path.exists():
        temporary.unlink(missing_ok=True)
    else:
        temporary.replace(final_path)
    snapshot = {
        "sourceUrl": source_url,
        "sourceVersion": response_headers.get("lastModified") or sha256[:12],
        "retrievedAt": retrieved_at,
        "rawPath": str(final_path.resolve()),
        "sha256": sha256,
        "bytes": byte_count,
        "http": response_headers,
        "reused": False,
    }
    write_json(index_path, snapshot)
    return snapshot


def load_mef_sovereign_indices(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, encoding="utf-8-sig")
    frame.columns = [str(column).lstrip("\ufeff").strip().upper() for column in frame.columns]
    missing = sorted(set(SOURCE_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"La fuente de Indices Soberanos no contiene: {', '.join(missing)}")
    return frame[list(SOURCE_COLUMNS)].rename(columns=SOURCE_COLUMNS)


def build_audited_mef_market_silver(frame: pd.DataFrame, *, minimum_rows: int = 1_000) -> tuple[pd.DataFrame, dict[str, Any]]:
    required = {TIMESTAMP_COLUMN, "nominal_index", "nominal_annual_yield", "real_index", "real_annual_yield"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"La auditoria MEF requiere: {', '.join(missing)}")
    data = frame.copy()
    original_rows = len(data)
    data[TIMESTAMP_COLUMN] = pd.to_datetime(data[TIMESTAMP_COLUMN], errors="coerce", utc=True)
    invalid_timestamps = int(data[TIMESTAMP_COLUMN].isna().sum())
    data = data.dropna(subset=[TIMESTAMP_COLUMN]).sort_values(TIMESTAMP_COLUMN, kind="stable").reset_index(drop=True)
    duplicate_timestamps = int(data[TIMESTAMP_COLUMN].duplicated().sum())

    numeric = sorted(required - {TIMESTAMP_COLUMN})
    invalid_numeric: dict[str, int] = {}
    for column in numeric:
        raw_not_empty = data[column].notna() & data[column].astype(str).str.strip().ne("")
        converted = pd.to_numeric(data[column], errors="coerce")
        invalid_numeric[column] = int((raw_not_empty & converted.isna()).sum())
        data[column] = converted
    missing_numeric = {column: int(data[column].isna().sum()) for column in numeric}
    duplicate_groups = data[data[TIMESTAMP_COLUMN].duplicated(keep=False)]
    conflicting_duplicate_dates = 0
    target_duplicate_conflicts = 0
    if not duplicate_groups.empty:
        conflicts = duplicate_groups.groupby(TIMESTAMP_COLUMN, sort=False)[numeric].nunique(dropna=False)
        conflicting_duplicate_dates = int((conflicts.max(axis=1) > 1).sum())
        target_duplicate_conflicts = int((conflicts["nominal_index"] > 1).sum())
        # The official resource has no revision timestamp. Mean aggregation is
        # deterministic, preserves the target where duplicated nominal indices
        # agree, and is disclosed in the audit instead of silently picking a row.
        data = data.groupby(TIMESTAMP_COLUMN, as_index=False, sort=True)[numeric].mean()
    nonpositive_nominal_indices = int((data["nominal_index"] <= 0).sum())
    if nonpositive_nominal_indices:
        raise ValueError("El indice nominal objetivo debe ser positivo para calcular log-rendimientos.")
    real_index_placeholders = int((data["real_index"] <= 0).sum())
    real_placeholder_mask = data["real_index"] <= 0
    data.loc[real_placeholder_mask, "real_index"] = np.nan
    data.loc[real_placeholder_mask & data["real_annual_yield"].eq(0), "real_annual_yield"] = np.nan

    data[TARGET_COLUMN] = 100.0 * np.log(data["nominal_index"] / data["nominal_index"].shift(1))
    data["real_log_return_1d"] = 100.0 * np.log(data["real_index"] / data["real_index"].shift(1))
    data = data.dropna(subset=[TARGET_COLUMN]).reset_index(drop=True)
    timestamp = data[TIMESTAMP_COLUMN]
    data["weekday_sin"] = np.sin(2 * np.pi * timestamp.dt.dayofweek / 7.0)
    data["weekday_cos"] = np.cos(2 * np.pi * timestamp.dt.dayofweek / 7.0)
    data["month_sin"] = np.sin(2 * np.pi * (timestamp.dt.month - 1) / 12.0)
    data["month_cos"] = np.cos(2 * np.pi * (timestamp.dt.month - 1) / 12.0)
    data = data[[TIMESTAMP_COLUMN, *FEATURE_COLUMNS]]

    deltas = data[TIMESTAMP_COLUMN].diff().dropna().dt.total_seconds().div(86_400)
    reasons: list[str] = []
    if len(data) < minimum_rows:
        reasons.append(f"La serie util tiene {len(data)} filas; se requieren al menos {minimum_rows}.")
    if invalid_timestamps:
        reasons.append(f"La fuente contiene {invalid_timestamps} fechas invalidas.")
    invalid_total = sum(invalid_numeric.values())
    if invalid_total:
        reasons.append(f"La fuente contiene {invalid_total} valores numericos invalidos.")
    feature_missing_after_derivation = {
        column: int(data[column].isna().sum()) for column in FEATURE_COLUMNS if column != TARGET_COLUMN
    }
    if not data[TIMESTAMP_COLUMN].is_monotonic_increasing:
        reasons.append("Las fechas no quedaron ordenadas cronologicamente.")

    test_start = int(len(data) * 0.85)
    audit = {
        "source": {
            "originalRows": int(original_rows),
            "usableRows": int(len(data)),
            "startAt": str(data[TIMESTAMP_COLUMN].iloc[0]),
            "endAt": str(data[TIMESTAMP_COLUMN].iloc[-1]),
            "invalidTimestamps": invalid_timestamps,
            "duplicateTimestamps": duplicate_timestamps,
            "duplicateResolution": "arithmetic_mean_per_date",
            "conflictingDuplicateDates": conflicting_duplicate_dates,
            "targetDuplicateConflicts": target_duplicate_conflicts,
            "invalidNumericValues": invalid_numeric,
            "missingNumericValuesBeforeDerivation": missing_numeric,
            "realIndexZeroPlaceholdersConvertedToMissing": real_index_placeholders,
            "featureMissingValuesAfterDerivation": feature_missing_after_derivation,
            "minimumGapDays": float(deltas.min()),
            "maximumGapDays": float(deltas.max()),
            "weekendOrHolidayGaps": int((deltas > 1.0).sum()),
        },
        "target": {
            "column": TARGET_COLUMN,
            "definition": "100 * natural_log(nominal_index_t / nominal_index_t_minus_1)",
            "unit": "log_percentage_points",
            "minimum": float(data[TARGET_COLUMN].min()),
            "maximum": float(data[TARGET_COLUMN].max()),
            "mean": float(data[TARGET_COLUMN].mean()),
            "standardDeviation": float(data[TARGET_COLUMN].std(ddof=1)),
        },
        "chronologicalSplit": {
            "developmentRows": int(test_start),
            "lockedTestRows": int(len(data) - test_start),
            "developmentRange": [str(data[TIMESTAMP_COLUMN].iloc[0]), str(data[TIMESTAMP_COLUMN].iloc[test_start - 1])],
            "lockedTestRange": [str(data[TIMESTAMP_COLUMN].iloc[test_start]), str(data[TIMESTAMP_COLUMN].iloc[-1])],
            "testSetLocked": True,
            "testSetUsed": False,
            "testEvaluationAuthorized": False,
        },
        "readiness": {
            "minimumRows": int(minimum_rows),
            "ready": not reasons,
            "reasons": reasons,
            "targetImputed": False,
            "featuresImputedGlobally": False,
            "featureImputation": "median_fitted_inside_each_training_fold_only",
            "groundTruthAnomalyLabelsAvailable": False,
            "fraudClaimsAllowed": False,
            "testSetUsed": False,
        },
    }
    return data, audit


def sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _notify(callback: Callable[[dict[str, Any]], None] | None, **event: Any) -> None:
    if callback is not None:
        callback(event)


def main() -> None:
    parser = argparse.ArgumentParser(description="Descarga, versiona y audita Indices Soberanos del MEF.")
    parser.add_argument("--url", default=SOURCE_CONFIG.mef_sovereign_indices_url)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--minimum-rows", type=int, default=1_000)
    args = parser.parse_args()
    result = prepare_mef_market_dataset(source_url=args.url, force_download=args.force, minimum_rows=args.minimum_rows)
    print(json.dumps({
        "datasetId": result["metadata"]["datasetId"],
        "rows": result["metadata"]["silver"]["rows"],
        "ready": result["audit"]["readiness"]["ready"],
        "testSetUsed": False,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
