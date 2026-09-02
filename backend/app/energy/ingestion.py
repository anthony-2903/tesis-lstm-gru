from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse
from uuid import uuid4

import numpy as np
import pandas as pd
import requests

from app.config import RAW_DIR, RESULTS_DIR, SILVER_DIR, SOURCE_CONFIG, ensure_dirs
from app.utils import read_json, write_json


TARGET_COLUMN = "DE_load_actual_entsoe_transparency"
TIMESTAMP_COLUMN = "timestamp"
SOURCE_TIMESTAMP_COLUMN = "utc_timestamp"
RELEVANT_COLUMNS = (
    TARGET_COLUMN,
    "DE_load_forecast_entsoe_transparency",
    "DE_solar_generation_actual",
    "DE_wind_generation_actual",
)


def prepare_real_energy_dataset(
    *,
    source_url: str = SOURCE_CONFIG.opsd_time_series_url,
    force_download: bool = False,
    minimum_rows: int = 8760,
    raw_directory: Path | None = None,
    silver_path: Path | None = None,
    audit_path: Path | None = None,
    metadata_path: Path | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    ensure_dirs()
    raw_root = raw_directory or (RAW_DIR / "energy")
    destination = silver_path or (SILVER_DIR / "opsd.csv")
    report_destination = audit_path or (RESULTS_DIR / "energy_data_audit.json")
    metadata_destination = metadata_path or (SILVER_DIR / "opsd.metadata.json")
    raw_root.mkdir(parents=True, exist_ok=True)

    _notify(progress_callback, stage="downloading", percent=5.0, message="Descargando o verificando el snapshot OPSD.")
    snapshot = download_versioned_energy_source(source_url, raw_root, force=force_download)
    _notify(progress_callback, stage="loading", percent=55.0, message="Snapshot verificado; leyendo columnas energéticas relevantes.")
    selected_source, source_columns = load_relevant_energy_columns(Path(snapshot["rawPath"]))
    _notify(progress_callback, stage="auditing", percent=70.0, message="Auditando continuidad horaria, faltantes y variable objetivo.")
    silver, audit = build_audited_energy_silver(selected_source, minimum_rows=minimum_rows)
    _notify(progress_callback, stage="persisting", percent=85.0, message="Generando dataset silver y metadatos reproducibles.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    silver.to_csv(destination, index=False)
    silver_sha256, silver_bytes = sha256_file(destination)

    prepared_at = datetime.now(timezone.utc).isoformat()
    dataset_id = f"opsd-{snapshot['sourceVersion']}-{silver_sha256[:12]}"
    metadata = {
        "schemaVersion": "1.0.0",
        "datasetId": dataset_id,
        "domain": "energia",
        "provider": "Open Power System Data",
        "sourceUrl": source_url,
        "sourceVersion": snapshot["sourceVersion"],
        "retrievedAt": snapshot["retrievedAt"],
        "preparedAt": prepared_at,
        "rawSnapshot": snapshot,
        "sourceColumns": source_columns,
        "selectionPolicy": {
            "series": "Germany",
            "frequency": "hourly",
            "target": TARGET_COLUMN,
            "rule": "Longest contiguous hourly segment with an observed target; feature gaps remain missing for train-only imputation.",
            "targetImputation": "none",
            "featureImputation": "deferred_to_each_training_fold",
        },
        "silver": {
            "path": str(destination.resolve()),
            "sha256": silver_sha256,
            "bytes": silver_bytes,
            "rows": int(len(silver)),
            "columns": list(silver.columns),
        },
        "auditPath": str(report_destination.resolve()),
        "readyForThesisPilot": bool(audit["readiness"]["ready"]),
    }
    audit_payload = {
        "schemaVersion": "1.0.0",
        "datasetId": dataset_id,
        "createdAt": prepared_at,
        **audit,
    }
    write_json(report_destination, audit_payload)
    write_json(metadata_destination, metadata)
    _notify(progress_callback, stage="completed", percent=100.0, message="Dataset OPSD preparado y validado.")
    return {"metadata": metadata, "audit": audit_payload}


def get_energy_dataset_status() -> dict[str, Any]:
    metadata_path = SILVER_DIR / "opsd.metadata.json"
    audit_path = RESULTS_DIR / "energy_data_audit.json"
    if not metadata_path.exists() or not audit_path.exists():
        return {
            "available": False,
            "readyForThesisPilot": False,
            "message": "El dataset OPSD real todavía no ha sido preparado.",
        }
    metadata = read_json(metadata_path)
    audit = read_json(audit_path)
    silver_path = Path(metadata.get("silver", {}).get("path", ""))
    return {
        "available": silver_path.is_file(),
        "readyForThesisPilot": bool(audit.get("readiness", {}).get("ready")) and silver_path.is_file(),
        "datasetId": metadata.get("datasetId"),
        "provider": metadata.get("provider"),
        "sourceVersion": metadata.get("sourceVersion"),
        "retrievedAt": metadata.get("retrievedAt"),
        "preparedAt": metadata.get("preparedAt"),
        "silver": metadata.get("silver", {}),
        "selectedSegment": audit.get("selectedSegment", {}),
        "readiness": audit.get("readiness", {}),
        "sourceAudit": audit.get("source", {}),
    }


def download_versioned_energy_source(source_url: str, raw_directory: Path, *, force: bool = False) -> dict[str, Any]:
    index_path = raw_directory / "latest.json"
    if not force and index_path.exists():
        previous = read_json(index_path)
        previous_path = Path(previous.get("rawPath", ""))
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
        with requests.get(source_url, headers=headers, timeout=(20, 300), stream=True) as response:
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
    source_version = _source_version(source_url)
    final_path = raw_directory / f"opsd_{source_version}_{sha256[:12]}.csv"
    if final_path.exists():
        temporary.unlink(missing_ok=True)
    else:
        temporary.replace(final_path)
    snapshot = {
        "sourceUrl": source_url,
        "sourceVersion": source_version,
        "retrievedAt": retrieved_at,
        "rawPath": str(final_path.resolve()),
        "sha256": sha256,
        "bytes": byte_count,
        "http": response_headers,
        "reused": False,
    }
    write_json(index_path, snapshot)
    return snapshot


def load_relevant_energy_columns(path: Path) -> tuple[pd.DataFrame, list[str]]:
    header = pd.read_csv(path, nrows=0).columns.tolist()
    timestamp = SOURCE_TIMESTAMP_COLUMN if SOURCE_TIMESTAMP_COLUMN in header else TIMESTAMP_COLUMN
    required = [timestamp, TARGET_COLUMN]
    missing = [column for column in required if column not in header]
    if missing:
        raise ValueError(f"El archivo OPSD no contiene columnas obligatorias: {', '.join(missing)}")
    selected = [timestamp, *[column for column in RELEVANT_COLUMNS if column in header]]
    return pd.read_csv(path, usecols=selected), header


def build_audited_energy_silver(frame: pd.DataFrame, *, minimum_rows: int = 8760) -> tuple[pd.DataFrame, dict[str, Any]]:
    timestamp_source = SOURCE_TIMESTAMP_COLUMN if SOURCE_TIMESTAMP_COLUMN in frame.columns else TIMESTAMP_COLUMN
    if timestamp_source not in frame or TARGET_COLUMN not in frame:
        raise ValueError("La auditoría requiere timestamp y la carga real de Alemania.")
    data = frame.rename(columns={timestamp_source: TIMESTAMP_COLUMN}).copy()
    original_rows = len(data)
    data[TIMESTAMP_COLUMN] = pd.to_datetime(data[TIMESTAMP_COLUMN], errors="coerce", utc=True)
    invalid_timestamps = int(data[TIMESTAMP_COLUMN].isna().sum())
    data = data.dropna(subset=[TIMESTAMP_COLUMN]).sort_values(TIMESTAMP_COLUMN).reset_index(drop=True)
    duplicate_timestamps = int(data[TIMESTAMP_COLUMN].duplicated().sum())
    if duplicate_timestamps:
        raise ValueError(f"OPSD contiene {duplicate_timestamps} timestamps duplicados; no se sobrescribirán silenciosamente.")

    numeric_columns = [column for column in data.columns if column != TIMESTAMP_COLUMN]
    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    raw_missing = {column: int(data[column].isna().sum()) for column in numeric_columns}
    valid_target = data[data[TARGET_COLUMN].notna()].copy()
    if valid_target.empty:
        raise ValueError("La serie objetivo de OPSD no contiene observaciones válidas.")

    deltas = valid_target[TIMESTAMP_COLUMN].diff()
    groups = deltas.ne(pd.Timedelta(hours=1)).cumsum()
    group_sizes = valid_target.groupby(groups).size()
    longest_group = group_sizes.idxmax()
    contiguous = valid_target[groups == longest_group].copy()
    start_at = contiguous[TIMESTAMP_COLUMN].iloc[0]
    end_at = contiguous[TIMESTAMP_COLUMN].iloc[-1]
    expected_index = pd.date_range(start_at, end_at, freq="h", tz="UTC")
    selected = data[(data[TIMESTAMP_COLUMN] >= start_at) & (data[TIMESTAMP_COLUMN] <= end_at)].copy()
    selected = selected.set_index(TIMESTAMP_COLUMN).reindex(expected_index).rename_axis(TIMESTAMP_COLUMN).reset_index()
    if selected[TARGET_COLUMN].isna().any():
        raise ValueError("El segmento continuo seleccionado contiene huecos en la variable objetivo.")

    selected = _add_calendar_features(selected)
    silver_missing = {
        column: int(selected[column].isna().sum())
        for column in selected.columns
        if column != TIMESTAMP_COLUMN
    }
    total_span_hours = int((data[TIMESTAMP_COLUMN].iloc[-1] - data[TIMESTAMP_COLUMN].iloc[0]) / pd.Timedelta(hours=1)) + 1
    missing_timestamps = max(0, total_span_hours - len(data))
    non_hourly_intervals = int((data[TIMESTAMP_COLUMN].diff().dropna() != pd.Timedelta(hours=1)).sum())
    reasons = []
    if len(selected) < minimum_rows:
        reasons.append(f"El segmento continuo tiene {len(selected)} filas; se requieren al menos {minimum_rows}.")
    if invalid_timestamps:
        reasons.append(f"Se encontraron {invalid_timestamps} timestamps inválidos en la fuente.")

    audit = {
        "source": {
            "originalRows": int(original_rows),
            "validTimestampRows": int(len(data)),
            "startAt": str(data[TIMESTAMP_COLUMN].iloc[0]),
            "endAt": str(data[TIMESTAMP_COLUMN].iloc[-1]),
            "invalidTimestamps": invalid_timestamps,
            "duplicateTimestamps": duplicate_timestamps,
            "missingHourlyTimestamps": int(missing_timestamps),
            "nonHourlyIntervals": non_hourly_intervals,
            "missingValues": raw_missing,
        },
        "selectedSegment": {
            "policy": "longest_contiguous_hourly_observed_target",
            "startAt": str(start_at),
            "endAt": str(end_at),
            "rows": int(len(selected)),
            "yearsApprox": float(len(selected) / 8760.0),
            "targetMissingValues": int(selected[TARGET_COLUMN].isna().sum()),
            "featureMissingValues": silver_missing,
            "calendarFeatures": ["hour_sin", "hour_cos", "weekday_sin", "weekday_cos", "month_sin", "month_cos"],
        },
        "readiness": {
            "minimumRows": int(minimum_rows),
            "ready": not reasons,
            "reasons": reasons,
            "targetImputed": False,
            "featuresImputedGlobally": False,
            "testSetUsed": False,
        },
    }
    return selected, audit


def sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    byte_count = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
            byte_count += len(chunk)
    return digest.hexdigest(), byte_count


def _add_calendar_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    timestamp = result[TIMESTAMP_COLUMN]
    result["hour_sin"] = np.sin(2 * np.pi * timestamp.dt.hour / 24.0)
    result["hour_cos"] = np.cos(2 * np.pi * timestamp.dt.hour / 24.0)
    result["weekday_sin"] = np.sin(2 * np.pi * timestamp.dt.dayofweek / 7.0)
    result["weekday_cos"] = np.cos(2 * np.pi * timestamp.dt.dayofweek / 7.0)
    result["month_sin"] = np.sin(2 * np.pi * (timestamp.dt.month - 1) / 12.0)
    result["month_cos"] = np.cos(2 * np.pi * (timestamp.dt.month - 1) / 12.0)
    return result


def _source_version(source_url: str) -> str:
    parts = [part for part in urlparse(source_url).path.split("/") if part]
    for part in parts:
        if len(part) == 10 and part[4] == "-" and part[7] == "-":
            return part
    return "unversioned"


def _notify(callback: Callable[[dict[str, Any]], None] | None, **event: Any) -> None:
    if callback is not None:
        callback(event)


def main() -> None:
    parser = argparse.ArgumentParser(description="Descarga, versiona y audita la serie horaria real de OPSD.")
    parser.add_argument("--url", default=SOURCE_CONFIG.opsd_time_series_url)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--minimum-rows", type=int, default=8760)
    args = parser.parse_args()
    result = prepare_real_energy_dataset(source_url=args.url, force_download=args.force, minimum_rows=args.minimum_rows)
    print(json.dumps({
        "datasetId": result["metadata"]["datasetId"],
        "rows": result["metadata"]["silver"]["rows"],
        "ready": result["audit"]["readiness"]["ready"],
        "reasons": result["audit"]["readiness"]["reasons"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
