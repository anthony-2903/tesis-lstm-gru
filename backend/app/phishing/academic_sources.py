from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import requests

from app.config import RAW_DIR
from app.phishing.curation import default_curation_manifest_path, get_phishing_curation_status
from app.phishing.ingestion import download_versioned_source, sha256_file
from app.utils import write_json


MENDELEY_FILES_API = "https://data.mendeley.com/public-api/datasets/{dataset_id}/files"


@dataclass(frozen=True)
class AcademicUrlSource:
    source_id: str
    provider: str
    dataset_id: str
    version: int
    filename: str
    repository_sha256: str
    repository_bytes: int
    url_column: str
    label_column: str
    label_mapping: dict[int, int]
    doi: str
    citation: str
    license_name: str
    published_at: str
    minimum_rows_per_label: int


ACADEMIC_URL_SOURCES = (
    AcademicUrlSource(
        source_id="phiusiil_uci_v2",
        provider="UCI Machine Learning Repository / Mendeley Data",
        dataset_id="shwpxscxy2",
        version=2,
        filename="PhiUSIIL_Phishing_URL_Dataset.csv",
        repository_sha256="a236549cd369cd80bd478ff8e1779cbf44c58d5c3f79f7a51a1adbed7d06d1c6",
        repository_bytes=56_854_345,
        url_column="URL",
        label_column="label",
        label_mapping={1: 0, 0: 1},
        doi="10.17632/shwpxscxy2.2",
        citation="Prasad, A. & Chandra, S. (2023). PhiUSIIL Phishing URL Dataset, Version 2. Mendeley Data.",
        license_name="CC BY 4.0",
        published_at="2023-11-15T00:00:00Z",
        minimum_rows_per_label=30_000,
    ),
    AcademicUrlSource(
        source_id="legitphish_v2",
        provider="Mendeley Data / SGGS Institute of Engineering and Technology",
        dataset_id="hx4m73v2sf",
        version=2,
        filename="url_features_extracted1.csv",
        repository_sha256="8685e7901c3b4e73eb43c106cb166cc932dd14f76960a032c713c2c32f09dbcd",
        repository_bytes=8_963_838,
        url_column="URL",
        label_column="ClassLabel",
        label_mapping={1: 0, 0: 1},
        doi="10.17632/hx4m73v2sf.2",
        citation="Potpelwar, R., Kulkarni, U. & Waghmare, J. (2025). LegitPhish Dataset, Version 2. Mendeley Data.",
        license_name="CC BY 4.0",
        published_at="2025-05-22T00:00:00Z",
        minimum_rows_per_label=30_000,
    ),
)


def build_academic_curation_package(
    *,
    force_download: bool = False,
    max_rows_per_label_per_source: int = 20_000,
    raw_directory: Path | None = None,
    manifest_path: Path | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    if max_rows_per_label_per_source < 5_000:
        raise ValueError("La curación académica exige al menos 5000 filas por clase y fuente.")
    raw_root = (raw_directory or (RAW_DIR / "phishing" / "academic")).resolve()
    destination = (manifest_path or default_curation_manifest_path()).resolve()
    curated_root = destination.parent
    raw_root.mkdir(parents=True, exist_ok=True)
    curated_root.mkdir(parents=True, exist_ok=True)

    source_entries: list[dict[str, Any]] = []
    source_reports: list[dict[str, Any]] = []
    raw_snapshots: dict[str, Any] = {}
    total_sources = len(ACADEMIC_URL_SOURCES)
    for position, definition in enumerate(ACADEMIC_URL_SOURCES):
        start = position / total_sources * 90.0
        _notify(progress_callback, stage="resolving_academic_source", percent=start + 2.0, message=f"Resolviendo {definition.source_id}.")
        repository_file = resolve_mendeley_file(definition)
        _notify(progress_callback, stage="downloading_academic_source", percent=start + 10.0, message=f"Descargando o verificando {definition.filename}.")
        snapshot = download_versioned_source(
            name=definition.source_id,
            source_url=repository_file["downloadUrl"],
            source_version=f"mendeley-v{definition.version}",
            raw_directory=raw_root / definition.source_id,
            force=force_download,
        )
        if snapshot["sha256"] != definition.repository_sha256 or snapshot["bytes"] != definition.repository_bytes:
            raise ValueError(f"{definition.source_id}: el archivo descargado no coincide con el snapshot publicado.")
        raw_snapshots[definition.source_id] = {
            **snapshot,
            "repositoryFileId": repository_file["fileId"],
            "repositorySha256": repository_file["sha256"],
            "repositoryBytes": repository_file["bytes"],
            "doi": definition.doi,
        }

        _notify(progress_callback, stage="normalizing_academic_source", percent=start + 30.0, message=f"Normalizando etiquetas de {definition.source_id}.")
        raw_frame = pd.read_csv(
            Path(snapshot["rawPath"]),
            usecols=[definition.url_column, definition.label_column],
            low_memory=False,
        )
        curated, report = normalize_academic_source_frame(
            definition,
            raw_frame,
            max_rows_per_label=max_rows_per_label_per_source,
        )
        normalized_path = curated_root / f"{definition.source_id}.csv"
        temporary = normalized_path.with_suffix(".tmp")
        curated.to_csv(temporary, index=False)
        temporary.replace(normalized_path)
        digest, size = sha256_file(normalized_path)
        source_entries.append({
            "sourceId": definition.source_id,
            "provider": definition.provider,
            "citation": definition.citation,
            "license": definition.license_name,
            "independentAcquisition": True,
            "declaredLabels": [0, 1],
            "path": normalized_path.name,
            "sha256": digest,
        })
        source_reports.append({
            **report,
            "normalizedPath": str(normalized_path),
            "normalizedSha256": digest,
            "normalizedBytes": size,
        })

    created_at = datetime.now(timezone.utc).isoformat()
    manifest = {
        "schemaVersion": "1.0.0",
        "studyId": "thesis-phishing-academic-v1",
        "createdAt": created_at,
        "sources": source_entries,
        "rawSnapshots": raw_snapshots,
        "selectionPolicy": {
            "strategy": "deterministic_sha256_within_source_and_label",
            "maximumRowsPerLabelPerSource": max_rows_per_label_per_source,
            "usesOnlyRawUrlAndPublishedLabel": True,
        },
        "evidencePolicy": {
            "method": "published_dataset_ground_truth",
            "scope": "The published dataset label is preserved; no claim of live URL safety is made.",
        },
        "methodologyLimitations": [
            "Published datasets may share upstream feeds; exact URL duplicates and conflicting domains are removed later.",
            "Legitimate is interpreted as the dataset ground truth at publication time, not a perpetual safety guarantee.",
            "Repository source identity is retained for source-aware audit and external generalization analysis.",
        ],
    }
    write_json(destination, manifest)
    status = get_phishing_curation_status(destination)
    if not status.get("readyForScientificMerge"):
        raise ValueError("El paquete académico se generó, pero no superó la cobertura científica: " + "; ".join(status.get("scientificReasons", [])))
    _notify(progress_callback, stage="academic_curation_completed", percent=100.0, message="Paquete académico curado y verificado.")
    return {
        "manifestPath": str(destination),
        "createdAt": created_at,
        "sources": source_reports,
        "status": status,
    }


def resolve_mendeley_file(definition: AcademicUrlSource) -> dict[str, Any]:
    response = requests.get(
        MENDELEY_FILES_API.format(dataset_id=definition.dataset_id),
        params={"folder_id": "root", "version": definition.version},
        headers={"Accept": "application/vnd.mendeley-public-dataset.1+json"},
        timeout=(20, 60),
    )
    response.raise_for_status()
    payload = response.json()
    files = payload if isinstance(payload, list) else [payload]
    match = next((item for item in files if item.get("filename") == definition.filename), None)
    if match is None:
        raise ValueError(f"{definition.source_id}: Mendeley no publicó el archivo esperado {definition.filename}.")
    details = match.get("content_details", {})
    if match.get("status") != "COMPLETED" or not details.get("download_url"):
        raise ValueError(f"{definition.source_id}: el archivo publicado no está disponible para descarga.")
    if details.get("sha256_hash") != definition.repository_sha256 or int(details.get("size", 0)) != definition.repository_bytes:
        raise ValueError(f"{definition.source_id}: el snapshot remoto cambió respecto de la versión fijada.")
    return {
        "fileId": match.get("id"),
        "downloadUrl": details["download_url"],
        "sha256": details["sha256_hash"],
        "bytes": int(details["size"]),
    }


def normalize_academic_source_frame(
    definition: AcademicUrlSource,
    frame: pd.DataFrame,
    *,
    max_rows_per_label: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    missing = {definition.url_column, definition.label_column}.difference(frame.columns)
    if missing:
        raise ValueError(f"{definition.source_id}: faltan columnas publicadas: {', '.join(sorted(missing))}.")
    normalized = frame[[definition.url_column, definition.label_column]].copy()
    normalized.columns = ["url", "published_label"]
    normalized["url"] = normalized["url"].fillna("").astype(str).str.strip()
    normalized = normalized[normalized["url"].ne("")].copy()
    missing_label_rows = int(normalized["published_label"].isna().sum())
    normalized = normalized[normalized["published_label"].notna()].copy()
    normalized["published_label"] = pd.to_numeric(normalized["published_label"], errors="raise").astype(int)
    observed = set(int(value) for value in normalized["published_label"].unique())
    if observed != set(definition.label_mapping):
        raise ValueError(f"{definition.source_id}: etiquetas observadas {sorted(observed)} distintas del contrato fijado.")
    normalized["is_phishing"] = normalized["published_label"].map(definition.label_mapping)
    normalized = normalized.drop_duplicates(subset=["url", "is_phishing"], keep="first")
    original_distribution = {
        str(label): int((normalized["is_phishing"] == label).sum())
        for label in (0, 1)
    }
    if any(count < definition.minimum_rows_per_label for count in original_distribution.values()):
        raise ValueError(f"{definition.source_id}: la distribución publicada es menor que el mínimo documentado.")

    parts: list[pd.DataFrame] = []
    for label, part in normalized.groupby("is_phishing"):
        part = part.copy()
        part["_order"] = part["url"].map(
            lambda url: hashlib.sha256(f"{definition.source_id}:{label}:{url}".encode("utf-8")).hexdigest()
        )
        parts.append(part.sort_values("_order", kind="stable").head(max_rows_per_label).drop(columns=["_order"]))
    selected = pd.concat(parts, ignore_index=True)
    selected["source_record_id"] = selected["url"].map(
        lambda url: hashlib.sha256(f"{definition.source_id}:{url}".encode("utf-8")).hexdigest()[:24]
    )
    selected["label_verified"] = True
    selected["verification_method"] = "published_dataset_ground_truth"
    selected["verification_reference"] = f"https://doi.org/{definition.doi}"
    selected["verified_at"] = definition.published_at
    selected["label_provenance"] = f"{definition.source_id}:published_label"
    selected = selected[[
        "url",
        "is_phishing",
        "source_record_id",
        "label_verified",
        "verification_method",
        "verification_reference",
        "verified_at",
        "label_provenance",
    ]].sort_values(["is_phishing", "source_record_id"], kind="stable").reset_index(drop=True)
    return selected, {
        "sourceId": definition.source_id,
        "doi": definition.doi,
        "license": definition.license_name,
        "originalRows": int(len(frame)),
        "eligibleRowsAfterExactDeduplication": int(len(normalized)),
        "missingLabelRowsDiscarded": missing_label_rows,
        "originalClassDistribution": original_distribution,
        "selectedRows": int(len(selected)),
        "selectedClassDistribution": {
            "negative": int((selected["is_phishing"] == 0).sum()),
            "positive": int((selected["is_phishing"] == 1).sum()),
        },
        "labelMapping": {str(key): value for key, value in definition.label_mapping.items()},
    }


def _notify(callback: Callable[[dict[str, Any]], None] | None, **event: Any) -> None:
    if callback is not None:
        callback(event)
