from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from app.config import RAW_DIR
from app.phishing.ingestion import normalize_url, sha256_file
from app.utils import read_json, write_json


CURATION_SCHEMA_VERSION = "1.0.0"
CURATION_COLUMNS = {
    "url",
    "is_phishing",
    "source_record_id",
    "label_verified",
    "verification_method",
    "verification_reference",
    "verified_at",
}
WEAK_NEGATIVE_METHODS = {
    "absence_only",
    "assumed_benign",
    "not_listed",
    "presumed_from_popularity",
    "tranco_rank",
}


def default_curation_manifest_path() -> Path:
    return RAW_DIR / "phishing" / "curated" / "curation_manifest.json"


def load_curated_source_package(manifest_path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    manifest_path = manifest_path.resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"No existe el manifiesto de curación: {manifest_path}")
    manifest = read_json(manifest_path)
    if manifest.get("schemaVersion") != CURATION_SCHEMA_VERSION:
        raise ValueError(f"schemaVersion debe ser {CURATION_SCHEMA_VERSION}.")
    sources = manifest.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("El manifiesto debe declarar al menos una fuente curada.")

    package_root = manifest_path.parent
    frames: list[pd.DataFrame] = []
    source_audits: list[dict[str, Any]] = []
    source_ids: set[str] = set()
    for entry in sources:
        source_id = str(entry.get("sourceId", "")).strip().lower()
        if not source_id or source_id in source_ids:
            raise ValueError("Cada fuente curada necesita un sourceId único y no vacío.")
        if source_id in {"phishtank", "tranco"}:
            raise ValueError(f"{source_id}: el sourceId está reservado para la fuente base.")
        source_ids.add(source_id)
        provider = str(entry.get("provider", "")).strip()
        citation = str(entry.get("citation", "")).strip()
        license_name = str(entry.get("license", "")).strip()
        if not provider or not citation or not license_name:
            raise ValueError(f"{source_id}: provider, citation y license son obligatorios.")

        source_path = (package_root / str(entry.get("path", ""))).resolve()
        if not source_path.is_relative_to(package_root):
            raise ValueError(f"{source_id}: el CSV debe estar dentro del directorio del paquete.")
        if not source_path.is_file():
            raise FileNotFoundError(f"{source_id}: no existe {source_path}.")
        digest, size = sha256_file(source_path)
        if digest != entry.get("sha256"):
            raise ValueError(f"{source_id}: el SHA-256 del CSV no coincide con el manifiesto.")

        frame = pd.read_csv(source_path, low_memory=False)
        missing = CURATION_COLUMNS.difference(frame.columns)
        if missing:
            raise ValueError(f"{source_id}: faltan columnas: {', '.join(sorted(missing))}.")
        normalized_rows: list[dict[str, Any]] = []
        invalid_urls = 0
        for row_number, record in enumerate(frame.to_dict(orient="records"), start=2):
            normalized = normalize_url(record.get("url"))
            if normalized is None:
                invalid_urls += 1
                continue
            try:
                label = int(record.get("is_phishing"))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{source_id}: etiqueta inválida en la fila {row_number}.") from exc
            if label not in {0, 1}:
                raise ValueError(f"{source_id}: is_phishing solo admite 0 y 1.")
            method = _text(record.get("verification_method")).lower()
            reference = _text(record.get("verification_reference"))
            verified_at = _text(record.get("verified_at"))
            verified = _strict_bool(record.get("label_verified"), source_id, row_number)
            normalized_rows.append({
                **normalized,
                "is_phishing": label,
                "source": source_id,
                "source_record_id": _text(record.get("source_record_id")) or str(row_number - 1),
                "label_provenance": _text(record.get("label_provenance")) or f"{source_id}:{method or 'unspecified'}",
                "collected_at": verified_at or str(manifest.get("createdAt", "")),
                "label_verified": verified,
                "verification_method": method,
                "verification_reference": reference,
                "verified_at": verified_at,
            })
        normalized_frame = pd.DataFrame(normalized_rows)
        if normalized_frame.empty:
            raise ValueError(f"{source_id}: no contiene URLs válidas.")
        labels = sorted(int(value) for value in normalized_frame["is_phishing"].unique())
        declared_labels = sorted(int(value) for value in entry.get("declaredLabels", []))
        if declared_labels and declared_labels != labels:
            raise ValueError(f"{source_id}: declaredLabels no coincide con las etiquetas observadas.")
        negative = normalized_frame[normalized_frame["is_phishing"] == 0]
        negative_verified = _verified_evidence_mask(negative)
        positive = normalized_frame[normalized_frame["is_phishing"] == 1]
        positive_verified = _verified_evidence_mask(positive)
        source_audits.append({
            "sourceId": source_id,
            "provider": provider,
            "citation": citation,
            "license": license_name,
            "independentAcquisition": bool(entry.get("independentAcquisition", False)),
            "path": str(source_path),
            "sha256": digest,
            "bytes": size,
            "rows": int(len(normalized_frame)),
            "labels": labels,
            "negativeRows": int(len(negative)),
            "verifiedNegativeRows": int(negative_verified.sum()),
            "positiveRows": int(len(positive)),
            "verifiedPositiveRows": int(positive_verified.sum()),
            "invalidUrlRows": int(invalid_urls),
        })
        frames.append(normalized_frame)

    manifest_hash, manifest_bytes = sha256_file(manifest_path)
    combined = pd.concat(frames, ignore_index=True)
    sources_by_id = {
        item["sourceId"]: {
            "provider": item["provider"],
            "citation": item["citation"],
            "license": item["license"],
            "independentAcquisition": item["independentAcquisition"],
            "originalRows": item["rows"],
            "labels": item["labels"],
            "negativeRows": item["negativeRows"],
            "verifiedNegativeRows": item["verifiedNegativeRows"],
            "positiveRows": item["positiveRows"],
            "verifiedPositiveRows": item["verifiedPositiveRows"],
        }
        for item in source_audits
    }
    return combined, {
        "available": True,
        "valid": True,
        "schemaVersion": CURATION_SCHEMA_VERSION,
        "studyId": manifest.get("studyId"),
        "manifest": {"path": str(manifest_path), "sha256": manifest_hash, "bytes": manifest_bytes},
        "sources": source_audits,
        "sourcesById": sources_by_id,
        "rows": int(len(combined)),
        "labels": {
            "negative": int((combined["is_phishing"] == 0).sum()),
            "positive": int((combined["is_phishing"] == 1).sum()),
        },
    }


def get_phishing_curation_status(manifest_path: Path | None = None) -> dict[str, Any]:
    path = (manifest_path or default_curation_manifest_path()).resolve()
    if not path.exists():
        return {
            "available": False,
            "valid": False,
            "manifestPath": str(path),
            "message": "Aún no existe un paquete de fuentes curadas. La preparación continuará en modo piloto.",
            "requirements": _requirements(),
        }
    try:
        _, audit = load_curated_source_package(path)
    except (OSError, ValueError) as exc:
        return {
            "available": True,
            "valid": False,
            "manifestPath": str(path),
            "message": str(exc),
            "requirements": _requirements(),
        }
    independent = [item for item in audit["sources"] if item["independentAcquisition"]]
    curated_sources_per_label = {
        str(label): sorted(item["sourceId"] for item in independent if label in item["labels"])
        for label in (0, 1)
    }
    sources_per_label = {
        "0": curated_sources_per_label["0"],
        "1": ["phishtank", *curated_sources_per_label["1"]],
    }
    mixed_sources = sorted(item["sourceId"] for item in independent if set(item["labels"]) == {0, 1})
    negative_rows = sum(item["negativeRows"] for item in independent)
    verified_negative_rows = sum(item["verifiedNegativeRows"] for item in independent)
    positive_rows = sum(item["positiveRows"] for item in independent)
    verified_positive_rows = sum(item["verifiedPositiveRows"] for item in independent)
    scientific_reasons: list[str] = []
    if len(sources_per_label["0"]) < 2 or len(sources_per_label["1"]) < 2:
        scientific_reasons.append("El paquete todavía no aporta dos fuentes independientes para cada clase.")
    if not mixed_sources:
        scientific_reasons.append("El paquete necesita al menos una fuente que contenga ambas clases.")
    if not negative_rows or verified_negative_rows != negative_rows:
        scientific_reasons.append("Todos los negativos curados deben conservar evidencia aceptable.")
    if positive_rows and verified_positive_rows != positive_rows:
        scientific_reasons.append("Todos los positivos curados deben conservar evidencia aceptable.")
    audit["manifestPath"] = str(path)
    audit["curatedSourcesPerLabel"] = curated_sources_per_label
    audit["sourcesPerLabel"] = sources_per_label
    audit["mixedLabelSources"] = mixed_sources
    audit["negativeEvidence"] = {"verified": verified_negative_rows, "total": negative_rows}
    audit["positiveEvidence"] = {"verified": verified_positive_rows, "total": positive_rows}
    audit["readyForScientificMerge"] = not scientific_reasons
    audit["scientificReasons"] = scientific_reasons
    audit["requirements"] = _requirements()
    return audit


def initialize_phishing_curation_template(directory: Path | None = None) -> dict[str, Any]:
    root = (directory or default_curation_manifest_path().parent).resolve()
    root.mkdir(parents=True, exist_ok=True)
    csv_path = root / "verified_urls.example.csv"
    manifest_path = root / "curation_manifest.example.json"
    if not csv_path.exists():
        csv_path.write_text(
            "url,is_phishing,source_record_id,label_verified,verification_method,verification_reference,verified_at,label_provenance\n",
            encoding="utf-8",
        )
    csv_hash, _ = sha256_file(csv_path)
    if not manifest_path.exists():
        write_json(manifest_path, {
            "schemaVersion": CURATION_SCHEMA_VERSION,
            "studyId": "replace-with-thesis-study-id",
            "createdAt": datetime.now(timezone.utc).isoformat(),
            "sources": [{
                "sourceId": "replace_source_id",
                "provider": "Replace with the independent data provider",
                "citation": "Replace with a DOI, dataset URL or bibliographic citation",
                "license": "Replace with the dataset license",
                "independentAcquisition": True,
                "declaredLabels": [0, 1],
                "path": csv_path.name,
                "sha256": csv_hash,
            }],
        })
    return {
        "created": True,
        "directory": str(root),
        "exampleManifestPath": str(manifest_path),
        "exampleCsvPath": str(csv_path),
        "activation": f"Complete los archivos, actualice SHA-256 y renombre el manifiesto a {default_curation_manifest_path().name}.",
    }


def verified_evidence_mask(frame: pd.DataFrame) -> pd.Series:
    return _verified_evidence_mask(frame)


def _verified_evidence_mask(frame: pd.DataFrame) -> pd.Series:
    if frame.empty:
        return pd.Series([], dtype=bool, index=frame.index)
    verified = frame.get("label_verified", pd.Series(False, index=frame.index)).fillna(False).astype(bool)
    method = frame.get("verification_method", pd.Series("", index=frame.index)).fillna("").astype(str).str.strip().str.lower()
    reference = frame.get("verification_reference", pd.Series("", index=frame.index)).fillna("").astype(str).str.strip()
    verified_at = frame.get("verified_at", pd.Series("", index=frame.index)).fillna("").astype(str).str.strip()
    return verified & method.ne("") & ~method.isin(WEAK_NEGATIVE_METHODS) & reference.ne("") & verified_at.ne("")


def _strict_bool(value: Any, source_id: str, row_number: int) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "sí", "si"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(f"{source_id}: label_verified inválido en la fila {row_number}.")


def _text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _requirements() -> list[str]:
    return [
        "Cada CSV debe estar versionado con SHA-256 y conservar cita y licencia.",
        "Cada clase debe aparecer en al menos dos fuentes de adquisición independientes.",
        "Al menos una fuente debe contener ambas clases para romper el atajo fuente-etiqueta.",
        "Todos los negativos seleccionados deben tener método, referencia y fecha de verificación individual o ground truth curado.",
        "La mera ausencia en una blacklist o el ranking de Tranco no se acepta como verificación benigna.",
    ]
