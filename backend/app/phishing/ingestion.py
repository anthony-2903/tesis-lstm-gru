from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import ipaddress
import json
from pathlib import Path
from typing import Any, Callable
from urllib.parse import SplitResult, urlsplit, urlunsplit
from uuid import uuid4

import pandas as pd
import tldextract
import truststore

# This is application code (API/CLI), so using the operating-system trust store
# is appropriate and keeps TLS verification enabled behind managed proxies.
truststore.inject_into_ssl()

import requests

from app.config import RAW_DIR, RESULTS_DIR, SILVER_DIR, SOURCE_CONFIG, ensure_dirs
from app.utils import read_json, write_json


SCHEMA_VERSION = "1.0.0"
DEFAULT_PER_CLASS = 10_000
SPLIT_FRACTIONS = {"train": 0.70, "validation": 0.15, "test": 0.15}
SILVER_COLUMNS = [
    "url",
    "canonical_url",
    "hostname",
    "registrable_domain",
    "is_phishing",
    "source",
    "source_record_id",
    "label_provenance",
    "label_verified",
    "verification_method",
    "verification_reference",
    "verified_at",
    "collected_at",
    "split",
]

# The bundled PSL snapshot makes preparation reproducible and prevents a hidden
# network update from changing the grouping between runs.
DOMAIN_EXTRACTOR = tldextract.TLDExtract(
    suffix_list_urls=(),
    include_psl_private_domains=True,
)


def prepare_real_phishing_dataset(
    *,
    force_download: bool = False,
    per_class: int = DEFAULT_PER_CLASS,
    seed: int = 42,
    raw_directory: Path | None = None,
    silver_path: Path | None = None,
    audit_path: Path | None = None,
    metadata_path: Path | None = None,
    curation_manifest_path: Path | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    if per_class < 1:
        raise ValueError("per_class debe ser mayor que cero.")
    ensure_dirs()
    raw_root = raw_directory or (RAW_DIR / "phishing")
    destination = silver_path or (SILVER_DIR / "phishing.csv")
    audit_destination = audit_path or (RESULTS_DIR / "phishing_data_audit.json")
    metadata_destination = metadata_path or (SILVER_DIR / "phishing.metadata.json")
    raw_root.mkdir(parents=True, exist_ok=True)

    _notify(progress_callback, stage="resolving", percent=3.0, message="Resolviendo versiones oficiales de las fuentes.")
    tranco_id = resolve_tranco_list_id()
    tranco_url = SOURCE_CONFIG.tranco_download_url_template.format(list_id=tranco_id)

    _notify(progress_callback, stage="downloading_phishtank", percent=8.0, message="Descargando o verificando PhishTank.")
    phishtank = download_versioned_source(
        name="phishtank",
        source_url=SOURCE_CONFIG.phishtank_csv_url,
        source_version="online-valid",
        raw_directory=raw_root / "phishtank",
        force=force_download,
    )
    _notify(progress_callback, stage="downloading_tranco", percent=35.0, message=f"Descargando o verificando Tranco {tranco_id}.")
    tranco = download_versioned_source(
        name="tranco",
        source_url=tranco_url,
        source_version=tranco_id,
        raw_directory=raw_root / "tranco",
        force=force_download,
    )

    _notify(progress_callback, stage="normalizing", percent=62.0, message="Normalizando URLs y dominios registrados.")
    positives, positive_stats = load_phishtank(Path(phishtank["rawPath"]), phishtank["retrievedAt"])
    negatives, negative_stats = load_tranco(Path(tranco["rawPath"]), tranco["retrievedAt"])
    positive_stats.update({"provider": "OpenDNS / PhishTank", "independentAcquisition": True})
    negative_stats.update({"provider": "Tranco", "independentAcquisition": True})
    source_stats = {"phishtank": positive_stats, "tranco": negative_stats}
    curation_audit: dict[str, Any] = {
        "available": False,
        "valid": False,
        "message": "No se incorporó un paquete de fuentes curadas.",
    }
    from app.phishing.curation import load_curated_source_package

    curation_path = curation_manifest_path or (raw_root / "curated" / "curation_manifest.json")
    if curation_path.exists():
        _notify(progress_callback, stage="loading_curation", percent=55.0, message="Verificando fuentes curadas y evidencia de etiquetas.")
        curated, curation_audit = load_curated_source_package(curation_path)
        positives = pd.concat([positives, curated[curated["is_phishing"] == 1]], ignore_index=True)
        negatives = pd.concat([negatives, curated[curated["is_phishing"] == 0]], ignore_index=True)
        source_stats.update(curation_audit["sourcesById"])
    silver, audit = build_audited_phishing_silver(
        positives,
        negatives,
        per_class=per_class,
        seed=seed,
        source_stats=source_stats,
    )

    _notify(progress_callback, stage="persisting", percent=88.0, message="Guardando silver, hashes y auditoría.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    silver.to_csv(destination, index=False)
    silver_hash, silver_bytes = sha256_file(destination)
    prepared_at = _now()
    dataset_id = f"phishing-{tranco_id.lower()}-{silver_hash[:12]}"
    source_snapshots = {"phishtank": phishtank, "tranco": tranco}
    if curation_audit.get("valid"):
        source_snapshots["curatedPackage"] = curation_audit.get("manifest", {})
    curated_providers = [item["provider"] for item in curation_audit.get("sources", [])]
    metadata = {
        "schemaVersion": SCHEMA_VERSION,
        "datasetId": dataset_id,
        "domain": "phishing",
        "preparedAt": prepared_at,
        "providers": list(dict.fromkeys(["PhishTank", "Tranco", *curated_providers])),
        "sourceSnapshots": source_snapshots,
        "trancoPermanentListUrl": f"https://tranco-list.eu/list/{tranco_id}/1000000",
        "labelPolicy": {
            "positive": "Verified and online PhishTank submissions.",
            "negative": "Curated negatives require versioned verification evidence; Tranco remains presumed benign and pilot-only.",
        },
        "curation": curation_audit,
        "normalizationPolicy": {
            "allowedSchemes": ["http", "https"],
            "fragmentsRemoved": True,
            "hostnameEncoding": "lowercase IDNA A-label",
            "registrableDomain": "tldextract bundled PSL snapshot with private suffixes enabled",
            "tldextractVersion": tldextract.__version__,
        },
        "splitPolicy": {
            "strategy": "deterministic_stratified_group_holdout",
            "groupKey": "registrable_domain",
            "fractions": SPLIT_FRACTIONS,
            "seed": seed,
        },
        "silver": {
            "path": str(destination.resolve()),
            "sha256": silver_hash,
            "bytes": silver_bytes,
            "rows": int(len(silver)),
            "columns": list(silver.columns),
        },
        "auditPath": str(audit_destination.resolve()),
        "readyForPipelinePilot": bool(audit["readiness"]["readyForPipelinePilot"]),
        "readyForThesisTraining": bool(audit["readiness"]["readyForThesisTraining"]),
    }
    audit_payload = {
        "schemaVersion": SCHEMA_VERSION,
        "datasetId": dataset_id,
        "createdAt": prepared_at,
        **audit,
    }
    write_json(audit_destination, audit_payload)
    write_json(metadata_destination, metadata)
    _notify(progress_callback, stage="completed", percent=100.0, message="Dataset de phishing preparado y auditado.")
    return {"metadata": metadata, "audit": audit_payload}


def get_phishing_dataset_status() -> dict[str, Any]:
    metadata_path = SILVER_DIR / "phishing.metadata.json"
    audit_path = RESULTS_DIR / "phishing_data_audit.json"
    if not metadata_path.exists() or not audit_path.exists():
        return {
            "available": False,
            "readyForPipelinePilot": False,
            "readyForThesisTraining": False,
            "message": "El dataset binario real de phishing todavía no ha sido preparado.",
        }
    metadata = read_json(metadata_path)
    audit = read_json(audit_path)
    silver = metadata.get("silver", {})
    silver_path = Path(silver.get("path", ""))
    valid_file = silver_path.is_file()
    if valid_file:
        actual_hash, actual_bytes = sha256_file(silver_path)
        valid_file = actual_hash == silver.get("sha256") and actual_bytes == silver.get("bytes")
    readiness = audit.get("readiness", {})
    return {
        "available": valid_file,
        "integrityVerified": valid_file,
        "readyForPipelinePilot": valid_file and bool(readiness.get("readyForPipelinePilot")),
        "readyForThesisTraining": valid_file and bool(readiness.get("readyForThesisTraining")),
        "datasetId": metadata.get("datasetId"),
        "preparedAt": metadata.get("preparedAt"),
        "providers": metadata.get("providers", []),
        "sourceSnapshots": metadata.get("sourceSnapshots", {}),
        "trancoPermanentListUrl": metadata.get("trancoPermanentListUrl"),
        "silver": silver,
        "classDistribution": audit.get("classDistribution", {}),
        "splitDistribution": audit.get("splitDistribution", {}),
        "leakageAudit": audit.get("leakageAudit", {}),
        "readiness": readiness,
        "biasAudit": audit.get("biasAudit", {}),
        "curation": metadata.get("curation", {}),
    }


def resolve_tranco_list_id() -> str:
    response = requests.get(
        SOURCE_CONFIG.tranco_latest_id_url,
        headers={"User-Agent": SOURCE_CONFIG.user_agent, "Accept": "text/plain"},
        timeout=(20, 60),
    )
    response.raise_for_status()
    list_id = response.text.strip().upper()
    if not list_id or len(list_id) > 32 or not list_id.replace("-", "").isalnum():
        raise ValueError("Tranco devolvió un identificador de lista inválido.")
    return list_id


def download_versioned_source(
    *,
    name: str,
    source_url: str,
    source_version: str,
    raw_directory: Path,
    force: bool = False,
) -> dict[str, Any]:
    raw_directory.mkdir(parents=True, exist_ok=True)
    index_path = raw_directory / "latest.json"
    if not force and index_path.exists():
        previous = read_json(index_path)
        previous_path = Path(previous.get("rawPath", ""))
        if previous.get("sourceUrl") == source_url and previous_path.is_file():
            actual_hash, actual_bytes = sha256_file(previous_path)
            if actual_hash == previous.get("sha256") and actual_bytes == previous.get("bytes"):
                return {**previous, "reused": True}

    temporary = raw_directory / f"download_{uuid4().hex}.part"
    digest = hashlib.sha256()
    byte_count = 0
    headers = {
        "User-Agent": SOURCE_CONFIG.user_agent,
        "Accept": "text/csv,application/csv,text/plain,*/*",
    }
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
                "contentDisposition": response.headers.get("Content-Disposition"),
            }
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    if byte_count == 0:
        temporary.unlink(missing_ok=True)
        raise ValueError(f"La descarga de {name} está vacía.")

    sha256 = digest.hexdigest()
    final_path = raw_directory / f"{name}_{source_version}_{sha256[:12]}.csv"
    if final_path.exists():
        temporary.unlink(missing_ok=True)
    else:
        temporary.replace(final_path)
    snapshot = {
        "provider": name,
        "sourceUrl": source_url,
        "sourceVersion": source_version,
        "retrievedAt": _now(),
        "rawPath": str(final_path.resolve()),
        "sha256": sha256,
        "bytes": byte_count,
        "http": response_headers,
        "reused": False,
    }
    write_json(index_path, snapshot)
    return snapshot


def load_phishtank(path: Path, retrieved_at: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = pd.read_csv(path, low_memory=False)
    if "url" not in frame.columns:
        raise ValueError("La fuente PhishTank no contiene la columna url.")
    original_rows = len(frame)
    if "verified" in frame.columns:
        frame = frame[frame["verified"].astype(str).str.lower().eq("yes")]
    if "online" in frame.columns:
        frame = frame[frame["online"].astype(str).str.lower().eq("yes")]
    rows: list[dict[str, Any]] = []
    invalid = 0
    for record in frame.to_dict(orient="records"):
        normalized = normalize_url(record.get("url"))
        if normalized is None:
            invalid += 1
            continue
        record_id = str(record.get("phish_id", ""))
        rows.append({
            **normalized,
            "is_phishing": 1,
            "source": "phishtank",
            "source_record_id": record_id,
            "label_provenance": "phishtank_verified_online",
            "label_verified": True,
            "verification_method": "provider_community_verification",
            "verification_reference": f"https://phishtank.org/phish_detail.php?phish_id={record_id}",
            "verified_at": retrieved_at,
            "collected_at": retrieved_at,
        })
    result = pd.DataFrame(rows)
    return result, {
        "originalRows": int(original_rows),
        "eligibleRows": int(len(frame)),
        "validNormalizedRows": int(len(result)),
        "invalidUrlRows": int(invalid),
        "labelMeaning": "verified online phishing URL",
    }


def load_tranco(path: Path, retrieved_at: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = pd.read_csv(path, names=["rank", "domain"], header=None, low_memory=False)
    original_rows = len(frame)
    rows: list[dict[str, Any]] = []
    invalid = 0
    for record in frame.to_dict(orient="records"):
        domain = str(record.get("domain", "")).strip()
        normalized = normalize_url(f"https://{domain}/")
        if normalized is None:
            invalid += 1
            continue
        rows.append({
            **normalized,
            "is_phishing": 0,
            "source": "tranco",
            "source_record_id": str(record.get("rank", "")),
            "label_provenance": "tranco_presumed_benign",
            "label_verified": False,
            "verification_method": "presumed_from_popularity",
            "verification_reference": "",
            "verified_at": "",
            "collected_at": retrieved_at,
            "source_rank": int(record["rank"]),
        })
    result = pd.DataFrame(rows)
    return result, {
        "originalRows": int(original_rows),
        "validNormalizedRows": int(len(result)),
        "invalidDomainRows": int(invalid),
        "labelMeaning": "presumed benign ranked pay-level domain; not independently verified clean",
    }


def normalize_url(value: Any) -> dict[str, str] | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    if "://" not in raw:
        raw = f"https://{raw}"
    try:
        parsed = urlsplit(raw)
        scheme = parsed.scheme.lower()
        if scheme not in {"http", "https"} or not parsed.hostname:
            return None
        hostname = parsed.hostname.rstrip(".").lower().encode("idna").decode("ascii")
        if not hostname or any(character.isspace() for character in hostname):
            return None
        port = parsed.port
    except (UnicodeError, ValueError):
        return None

    userinfo = ""
    if parsed.username is not None:
        userinfo = parsed.username
        if parsed.password is not None:
            userinfo += f":{parsed.password}"
        userinfo += "@"
    display_host = f"[{hostname}]" if ":" in hostname else hostname
    default_port = (scheme == "http" and port == 80) or (scheme == "https" and port == 443)
    netloc = f"{userinfo}{display_host}{'' if port is None or default_port else f':{port}'}"
    canonical = urlunsplit(SplitResult(scheme, netloc, parsed.path or "/", parsed.query, ""))
    group = registrable_domain(hostname)
    if not group:
        return None
    return {
        "url": raw,
        "canonical_url": canonical,
        "hostname": hostname,
        "registrable_domain": group,
    }


def registrable_domain(hostname: str) -> str:
    try:
        ipaddress.ip_address(hostname)
        return hostname
    except ValueError:
        pass
    extracted = DOMAIN_EXTRACTOR(hostname)
    return extracted.top_domain_under_public_suffix.lower() or hostname.lower()


def build_audited_phishing_silver(
    positives: pd.DataFrame,
    negatives: pd.DataFrame,
    *,
    per_class: int,
    seed: int,
    source_stats: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    required = {"canonical_url", "registrable_domain", "is_phishing", "source"}
    for name, frame in (("positives", positives), ("negatives", negatives)):
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{name} no contiene columnas obligatorias: {', '.join(sorted(missing))}")

    combined = pd.concat([positives, negatives], ignore_index=True)
    combined["is_phishing"] = pd.to_numeric(combined["is_phishing"], errors="raise").astype(int)
    if not set(combined["is_phishing"].unique()).issubset({0, 1}):
        raise ValueError("is_phishing solo admite 0 y 1.")
    for column, default in (
        ("url", ""), ("hostname", ""), ("source_record_id", ""), ("label_provenance", ""),
        ("label_verified", False), ("verification_method", ""), ("verification_reference", ""),
        ("verified_at", ""), ("collected_at", ""),
    ):
        if column not in combined.columns:
            combined[column] = default
    combined["label_verified"] = combined["label_verified"].map(_coerce_evidence_bool)
    before_dedup = len(combined)
    combined = combined.drop_duplicates(subset=["canonical_url", "is_phishing"], keep="first").copy()
    exact_duplicates = before_dedup - len(combined)

    label_counts_by_group = combined.groupby("registrable_domain")["is_phishing"].nunique()
    conflict_groups = set(label_counts_by_group[label_counts_by_group > 1].index)
    conflict_rows = int(combined["registrable_domain"].isin(conflict_groups).sum())
    if conflict_groups:
        combined = combined[~combined["registrable_domain"].isin(conflict_groups)].copy()

    positive_candidates = combined[combined["is_phishing"] == 1].copy()
    negative_candidates = combined[combined["is_phishing"] == 0].copy()
    positive = _balanced_source_sample(positive_candidates, per_class, seed)
    verified_negative_candidates = negative_candidates[_verified_evidence_mask(negative_candidates)].copy()
    if len(verified_negative_candidates) >= per_class:
        negative = _balanced_source_sample(verified_negative_candidates, per_class, seed)
        negative_selection = "verified evidence only, balanced deterministically across sources"
    elif "source_rank" in negative_candidates.columns:
        negative = negative_candidates.sort_values(["source_rank", "canonical_url"], kind="stable").head(per_class).copy()
        negative_selection = "pilot fallback using top-ranked presumed-benign candidates"
    else:
        negative = _deterministic_head(negative_candidates, per_class, seed)
        negative_selection = "pilot fallback using deterministic unverified candidates"
    selected = pd.concat([positive, negative], ignore_index=True)
    if selected.empty:
        raise ValueError("No quedaron registros válidos después de la auditoría.")

    split_mapping = _assign_group_splits(selected, seed=seed)
    selected["split"] = selected["registrable_domain"].map(split_mapping)
    selected = selected.sort_values(["split", "is_phishing", "registrable_domain", "canonical_url"], kind="stable")
    selected = selected[SILVER_COLUMNS].reset_index(drop=True)

    class_distribution = {
        "negative": int((selected["is_phishing"] == 0).sum()),
        "positive": int((selected["is_phishing"] == 1).sum()),
        "total": int(len(selected)),
        "positiveRate": float(selected["is_phishing"].mean()),
    }
    split_distribution = {
        split: {
            "rows": int(len(part)),
            "negative": int((part["is_phishing"] == 0).sum()),
            "positive": int((part["is_phishing"] == 1).sum()),
            "groups": int(part["registrable_domain"].nunique()),
        }
        for split, part in selected.groupby("split", sort=False)
    }
    groups_by_split = {split: set(part["registrable_domain"].unique()) for split, part in selected.groupby("split")}
    leakage_pairs = {
        "trainValidation": len(groups_by_split.get("train", set()) & groups_by_split.get("validation", set())),
        "trainTest": len(groups_by_split.get("train", set()) & groups_by_split.get("test", set())),
        "validationTest": len(groups_by_split.get("validation", set()) & groups_by_split.get("test", set())),
    }
    source_label_counts = selected.groupby("source")["is_phishing"].nunique().to_dict()
    source_label_coupling = bool(source_label_counts) and all(count == 1 for count in source_label_counts.values())
    declared_independent = {
        source for source, details in (source_stats or {}).items()
        if bool(details.get("independentAcquisition", False))
    }
    sources_per_label = {
        str(label): sorted(
            source for source in selected.loc[selected["is_phishing"] == label, "source"].astype(str).unique()
            if source in declared_independent
        )
        for label in (0, 1)
    }
    mixed_label_sources = sorted(
        source for source, count in source_label_counts.items()
        if count > 1 and source in declared_independent
    )
    negative_selected = selected[selected["is_phishing"] == 0]
    positive_selected = selected[selected["is_phishing"] == 1]
    verified_negative_rows = int(_verified_evidence_mask(negative_selected).sum())
    verified_positive_rows = int(_verified_evidence_mask(positive_selected).sum())
    negatives_verified = bool(len(negative_selected)) and verified_negative_rows == len(negative_selected)
    positives_verified = bool(len(positive_selected)) and verified_positive_rows == len(positive_selected)
    url_shape = _url_shape_audit(selected)

    pilot_reasons: list[str] = []
    if len(selected) < 10_000:
        pilot_reasons.append(f"Solo hay {len(selected)} filas; el protocolo exige al menos 10000.")
    if class_distribution["positive"] < 1_000 or class_distribution["negative"] < 1_000:
        pilot_reasons.append("Cada clase debe contener al menos 1000 observaciones.")
    unique_groups = int(selected["registrable_domain"].nunique())
    if unique_groups < 1_000:
        pilot_reasons.append(f"Solo hay {unique_groups} dominios registrados únicos; se requieren 1000.")
    if any(leakage_pairs.values()):
        pilot_reasons.append("Existe fuga de dominios registrados entre particiones.")
    missing_splits = set(SPLIT_FRACTIONS).difference(split_distribution)
    if missing_splits:
        pilot_reasons.append(f"Faltan particiones: {', '.join(sorted(missing_splits))}.")

    thesis_reasons = list(pilot_reasons)
    if source_label_coupling:
        thesis_reasons.append("La identidad de la fuente permite predecir completamente la etiqueta; se requiere una fuente con ambas clases.")
    if len(sources_per_label["0"]) < 2 or len(sources_per_label["1"]) < 2:
        thesis_reasons.append("Cada clase debe estar representada por al menos dos fuentes de adquisición independientes.")
    if not negatives_verified:
        thesis_reasons.append(f"Solo {verified_negative_rows} de {len(negative_selected)} negativos conservan evidencia aceptable.")
    if not positives_verified:
        thesis_reasons.append(f"Solo {verified_positive_rows} de {len(positive_selected)} positivos conservan evidencia aceptable.")
    if url_shape["absoluteRootPathRateGap"] > url_shape["maximumAcceptedGap"]:
        thesis_reasons.append("La forma de la URL difiere demasiado entre clases y puede actuar como atajo de etiqueta.")

    audit = {
        "source": source_stats or {},
        "transformations": {
            "rowsBeforeDeduplication": int(before_dedup),
            "exactDuplicateRowsRemoved": int(exact_duplicates),
            "conflictingRegistrableDomainsRemoved": int(len(conflict_groups)),
            "conflictingRowsRemoved": conflict_rows,
            "requestedRowsPerClass": int(per_class),
            "selection": {
                "positive": "balanced deterministic sampling across available sources",
                "negative": negative_selection,
            },
        },
        "classDistribution": class_distribution,
        "splitDistribution": split_distribution,
        "uniqueRegistrableDomains": unique_groups,
        "leakageAudit": {"groupKey": "registrable_domain", "overlapCounts": leakage_pairs, "passed": not any(leakage_pairs.values())},
        "biasAudit": {
            "sourceLabelCoupling": source_label_coupling,
            "labelsPerSource": {source: int(count) for source, count in source_label_counts.items()},
            "declaredIndependentSources": sorted(declared_independent),
            "sourcesPerLabel": sources_per_label,
            "mixedLabelSources": mixed_label_sources,
            "minimumIndependentSourcesPerLabel": 2,
            "negativeLabelsIndependentlyVerified": negatives_verified,
            "verifiedNegativeRows": verified_negative_rows,
            "negativeRows": int(len(negative_selected)),
            "positiveLabelsIndependentlyVerified": positives_verified,
            "verifiedPositiveRows": verified_positive_rows,
            "positiveRows": int(len(positive_selected)),
            "urlShape": url_shape,
            "knownShortcutRisk": "Audited through source-label coupling, source coverage and URL-shape differences.",
        },
        "readiness": {
            "readyForPipelinePilot": not pilot_reasons,
            "readyForThesisTraining": not thesis_reasons,
            "pilotReasons": pilot_reasons,
            "thesisReasons": thesis_reasons,
            "testSetUsed": False,
        },
    }
    return selected, audit


def _legacy_build_audited_phishing_silver(
    positives: pd.DataFrame,
    negatives: pd.DataFrame,
    *,
    per_class: int,
    seed: int,
    source_stats: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    required = {"canonical_url", "registrable_domain", "is_phishing", "source"}
    for name, frame in (("positives", positives), ("negatives", negatives)):
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{name} no contiene columnas obligatorias: {', '.join(sorted(missing))}")

    combined = pd.concat([positives, negatives], ignore_index=True)
    combined["is_phishing"] = pd.to_numeric(combined["is_phishing"], errors="raise").astype(int)
    if not set(combined["is_phishing"].unique()).issubset({0, 1}):
        raise ValueError("is_phishing solo admite 0 y 1.")
    before_dedup = len(combined)
    combined = combined.drop_duplicates(subset=["canonical_url", "is_phishing"], keep="first").copy()
    exact_duplicates = before_dedup - len(combined)

    label_counts_by_group = combined.groupby("registrable_domain")["is_phishing"].nunique()
    conflict_groups = set(label_counts_by_group[label_counts_by_group > 1].index)
    conflict_rows = int(combined["registrable_domain"].isin(conflict_groups).sum())
    if conflict_groups:
        combined = combined[~combined["registrable_domain"].isin(conflict_groups)].copy()

    positive = _deterministic_head(combined[combined["is_phishing"] == 1], per_class, seed)
    negative_candidates = combined[combined["is_phishing"] == 0].copy()
    if "source_rank" in negative_candidates.columns:
        negative_candidates = negative_candidates.sort_values(["source_rank", "canonical_url"], kind="stable")
        negative = negative_candidates.head(per_class).copy()
    else:
        negative = _deterministic_head(negative_candidates, per_class, seed)
    selected = pd.concat([positive, negative], ignore_index=True)
    if selected.empty:
        raise ValueError("No quedaron registros válidos después de la auditoría.")

    split_mapping = _assign_group_splits(selected, seed=seed)
    selected["split"] = selected["registrable_domain"].map(split_mapping)
    selected = selected.sort_values(["split", "is_phishing", "registrable_domain", "canonical_url"], kind="stable")
    selected = selected[SILVER_COLUMNS].reset_index(drop=True)

    class_distribution = {
        "negative": int((selected["is_phishing"] == 0).sum()),
        "positive": int((selected["is_phishing"] == 1).sum()),
        "total": int(len(selected)),
        "positiveRate": float(selected["is_phishing"].mean()),
    }
    split_distribution = {
        split: {
            "rows": int(len(part)),
            "negative": int((part["is_phishing"] == 0).sum()),
            "positive": int((part["is_phishing"] == 1).sum()),
            "groups": int(part["registrable_domain"].nunique()),
        }
        for split, part in selected.groupby("split", sort=False)
    }
    groups_by_split = {
        split: set(part["registrable_domain"].unique())
        for split, part in selected.groupby("split")
    }
    leakage_pairs = {
        "trainValidation": len(groups_by_split.get("train", set()) & groups_by_split.get("validation", set())),
        "trainTest": len(groups_by_split.get("train", set()) & groups_by_split.get("test", set())),
        "validationTest": len(groups_by_split.get("validation", set()) & groups_by_split.get("test", set())),
    }
    source_label_counts = selected.groupby("source")["is_phishing"].nunique().to_dict()
    source_label_coupling = bool(source_label_counts) and all(count == 1 for count in source_label_counts.values())

    pilot_reasons: list[str] = []
    if len(selected) < 10_000:
        pilot_reasons.append(f"Solo hay {len(selected)} filas; el protocolo exige al menos 10000.")
    if class_distribution["positive"] < 1_000 or class_distribution["negative"] < 1_000:
        pilot_reasons.append("Cada clase debe contener al menos 1000 observaciones.")
    unique_groups = int(selected["registrable_domain"].nunique())
    if unique_groups < 1_000:
        pilot_reasons.append(f"Solo hay {unique_groups} dominios registrados únicos; se requieren 1000.")
    if any(leakage_pairs.values()):
        pilot_reasons.append("Existe fuga de dominios registrados entre particiones.")
    missing_splits = set(SPLIT_FRACTIONS).difference(split_distribution)
    if missing_splits:
        pilot_reasons.append(f"Faltan particiones: {', '.join(sorted(missing_splits))}.")

    thesis_reasons = list(pilot_reasons)
    if source_label_coupling:
        thesis_reasons.append("La etiqueta está acoplada a la fuente: PhishTank aporta positivos y Tranco negativos.")
    thesis_reasons.append("Los dominios Tranco son referencias benignas presumidas, no negativos verificados individualmente.")
    thesis_reasons.append("Falta una segunda fuente independiente por clase para medir generalización entre fuentes.")

    audit = {
        "source": source_stats or {},
        "transformations": {
            "rowsBeforeDeduplication": int(before_dedup),
            "exactDuplicateRowsRemoved": int(exact_duplicates),
            "conflictingRegistrableDomainsRemoved": int(len(conflict_groups)),
            "conflictingRowsRemoved": conflict_rows,
            "requestedRowsPerClass": int(per_class),
            "selection": "Deterministic hash sample for positives and top-ranked permanent Tranco domains for negatives.",
        },
        "classDistribution": class_distribution,
        "splitDistribution": split_distribution,
        "uniqueRegistrableDomains": unique_groups,
        "leakageAudit": {
            "groupKey": "registrable_domain",
            "overlapCounts": leakage_pairs,
            "passed": not any(leakage_pairs.values()),
        },
        "biasAudit": {
            "sourceLabelCoupling": source_label_coupling,
            "labelsPerSource": {source: int(count) for source, count in source_label_counts.items()},
            "negativeLabelsIndependentlyVerified": False,
            "knownShortcutRisk": "PhishTank contains full reported URLs while Tranco primarily contributes root domains.",
        },
        "readiness": {
            "readyForPipelinePilot": not pilot_reasons,
            "readyForThesisTraining": not thesis_reasons,
            "pilotReasons": pilot_reasons,
            "thesisReasons": thesis_reasons,
            "testSetUsed": False,
        },
    }
    return selected, audit


def _balanced_source_sample(frame: pd.DataFrame, limit: int, seed: int) -> pd.DataFrame:
    if frame.empty or limit <= 0:
        return frame.head(0).copy()
    ordered_sources = sorted(
        frame["source"].astype(str).unique(),
        key=lambda source: hashlib.sha256(f"{seed}:source:{source}".encode("utf-8")).hexdigest(),
    )
    target = min(limit, len(frame))
    base_quota, remainder = divmod(target, len(ordered_sources))
    selected_indices: list[Any] = []
    remaining_by_source: dict[str, pd.DataFrame] = {}
    for position, source in enumerate(ordered_sources):
        source_rows = frame[frame["source"].astype(str) == source]
        ordered = _deterministic_head(source_rows, len(source_rows), seed)
        quota = base_quota + (1 if position < remainder else 0)
        take = min(quota, len(ordered))
        selected_indices.extend(ordered.index[:take].tolist())
        remaining_by_source[source] = ordered.iloc[take:]
    missing = target - len(selected_indices)
    while missing > 0:
        progressed = False
        for source in ordered_sources:
            remaining = remaining_by_source[source]
            if remaining.empty:
                continue
            take = min(missing, max(1, target // len(ordered_sources)), len(remaining))
            selected_indices.extend(remaining.index[:take].tolist())
            remaining_by_source[source] = remaining.iloc[take:]
            missing -= take
            progressed = True
            if missing == 0:
                break
        if not progressed:
            break
    return frame.loc[selected_indices].copy()


def _verified_evidence_mask(frame: pd.DataFrame) -> pd.Series:
    if frame.empty:
        return pd.Series([], dtype=bool, index=frame.index)
    weak_methods = {"absence_only", "assumed_benign", "not_listed", "presumed_from_popularity", "tranco_rank"}
    verified = frame["label_verified"].fillna(False).map(_coerce_evidence_bool)
    method = frame["verification_method"].fillna("").astype(str).str.strip().str.lower()
    reference = frame["verification_reference"].fillna("").astype(str).str.strip()
    verified_at = frame["verified_at"].fillna("").astype(str).str.strip()
    return verified & method.ne("") & ~method.isin(weak_methods) & reference.ne("") & verified_at.ne("")


def _coerce_evidence_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "sí", "si"}


def _url_shape_audit(frame: pd.DataFrame) -> dict[str, Any]:
    rates: dict[str, float] = {}
    for label in (0, 1):
        rows = frame[frame["is_phishing"] == label]
        root_only = rows["canonical_url"].astype(str).map(
            lambda value: (urlsplit(value).path or "/") == "/" and not urlsplit(value).query
        )
        rates[str(label)] = float(root_only.mean()) if len(rows) else 0.0
    return {
        "rootPathRateByLabel": rates,
        "absoluteRootPathRateGap": abs(rates["0"] - rates["1"]),
        "maximumAcceptedGap": 0.60,
    }


def _deterministic_head(frame: pd.DataFrame, limit: int, seed: int) -> pd.DataFrame:
    result = frame.copy()
    result["_sample_order"] = result["canonical_url"].map(
        lambda value: hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()
    )
    return result.sort_values("_sample_order", kind="stable").head(limit).drop(columns=["_sample_order"])


def _assign_group_splits(frame: pd.DataFrame, *, seed: int) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for label, label_frame in frame.groupby("is_phishing"):
        groups = label_frame.groupby("registrable_domain").size().rename("rows").reset_index()
        groups["order"] = groups["registrable_domain"].map(
            lambda value: hashlib.sha256(f"{seed}:{label}:{value}".encode("utf-8")).hexdigest()
        )
        groups = groups.sort_values("order", kind="stable")
        total = int(groups["rows"].sum())
        cursor = 0
        for record in groups.to_dict(orient="records"):
            midpoint = cursor + int(record["rows"]) / 2
            ratio = midpoint / total
            split = "train" if ratio <= 0.70 else "validation" if ratio <= 0.85 else "test"
            mapping[str(record["registrable_domain"])] = split
            cursor += int(record["rows"])
    return mapping


def sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    byte_count = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
            byte_count += len(chunk)
    return digest.hexdigest(), byte_count


def _notify(callback: Callable[[dict[str, Any]], None] | None, **event: Any) -> None:
    if callback is not None:
        callback(event)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> None:
    parser = argparse.ArgumentParser(description="Descarga, versiona y audita el dataset binario de phishing.")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--per-class", type=int, default=DEFAULT_PER_CLASS)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = prepare_real_phishing_dataset(force_download=args.force, per_class=args.per_class, seed=args.seed)
    print(json.dumps({
        "datasetId": result["metadata"]["datasetId"],
        "rows": result["metadata"]["silver"]["rows"],
        "readyForPipelinePilot": result["audit"]["readiness"]["readyForPipelinePilot"],
        "readyForThesisTraining": result["audit"]["readiness"]["readyForThesisTraining"],
        "thesisReasons": result["audit"]["readiness"]["thesisReasons"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
