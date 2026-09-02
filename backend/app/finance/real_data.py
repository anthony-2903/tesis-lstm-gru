from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd

from app.config import RAW_DIR, RESULTS_DIR, SILVER_DIR, ensure_dirs
from app.finance.benchmark import FINANCE_COLUMNS, audit_finance_benchmark
from app.phishing.ingestion import sha256_file
from app.utils import read_json


REAL_FINANCE_PACKAGE_DIR = RAW_DIR / "finance" / "curated"
REAL_FINANCE_MANIFEST_PATH = REAL_FINANCE_PACKAGE_DIR / "real_finance_manifest.json"
IEEE_CIS_URL = "https://www.kaggle.com/c/ieee-fraud-detection/data"
ULB_WORLDLINE_URL = "https://www.kaggle.com/mlg-ulb/creditcardfraud/home"
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")

REAL_FINANCE_SOURCE_CATALOG: dict[str, dict[str, Any]] = {
    "ieee_cis": {
        "sourceId": "ieee_cis_fraud_detection",
        "displayName": "IEEE-CIS Fraud Detection",
        "provider": "IEEE Computational Intelligence Society / Vesta",
        "sourceUrl": IEEE_CIS_URL,
        "datasetKind": "real_world_anonymized_ecommerce_transactions",
        "licenseName": "Kaggle competition rules",
        "downloadPolicy": "manual_after_accepting_competition_rules",
        "redistributionAllowed": False,
        "adapter": "ieee_cis",
        "requiredFile": "train_transaction.csv",
        "requiredColumns": ["TransactionID", "TransactionDT", "TransactionAmt", "isFraud", "card1"],
        "supportsSequenceEntity": True,
        "entityPolicy": "anonymized_card_proxy_from_card1_card2_card3_card5_card6",
        "timePolicy": "TransactionDT_relative_seconds_preserved_as_ordered_relative_time",
        "thesisRole": "primary_real_finance_dataset_candidate",
    },
    "ulb_worldline": {
        "sourceId": "ulb_worldline_creditcardfraud",
        "displayName": "ULB/Worldline Credit Card Fraud",
        "provider": "Worldline and Machine Learning Group, ULB",
        "sourceUrl": ULB_WORLDLINE_URL,
        "datasetKind": "real_world_anonymized_credit_card_transactions",
        "licenseName": "Open Database License / Database Contents License",
        "downloadPolicy": "manual_or_authenticated_provider_download",
        "redistributionAllowed": False,
        "adapter": "ulb_worldline",
        "requiredFile": "creditcard.csv",
        "requiredColumns": ["Time", "Amount", "Class", *[f"V{index}" for index in range(1, 29)]],
        "supportsSequenceEntity": False,
        "entityPolicy": "customer_and_card_identifiers_removed_for_confidentiality",
        "timePolicy": "Time_seconds_from_first_transaction",
        "thesisRole": "external_tabular_validation_only",
    },
}


def get_real_finance_source_catalog() -> dict[str, Any]:
    return {
        "schemaVersion": "1.0.0",
        "items": list(REAL_FINANCE_SOURCE_CATALOG.values()),
        "selection": {
            "recommendedAdapter": "ieee_cis",
            "reason": "IEEE-CIS conserva tiempo relativo, etiqueta binaria y un proxy anonimizado de entidad compatible con secuencias.",
            "automaticDownload": False,
            "testSetPolicy": "The labeled provider train file is repartitioned chronologically; provider test without labels is never used.",
        },
    }


def initialize_real_finance_template(
    *,
    adapter: str = "ieee_cis",
    force: bool = False,
    manifest_path: Path | None = None,
) -> dict[str, Any]:
    if adapter not in REAL_FINANCE_SOURCE_CATALOG:
        raise ValueError(f"Adaptador financiero real no soportado: {adapter}")
    destination = manifest_path or REAL_FINANCE_MANIFEST_PATH
    if destination.exists() and not force:
        return read_json(destination)
    catalog = REAL_FINANCE_SOURCE_CATALOG[adapter]
    destination.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()
    manifest = {
        "schemaVersion": "1.0.0",
        "packageId": f"finance-real-{adapter}-replace-with-version",
        "adapter": adapter,
        "source": {
            "sourceId": catalog["sourceId"],
            "provider": catalog["provider"],
            "sourceUrl": catalog["sourceUrl"],
            "sourceVersion": "REPLACE_WITH_DOWNLOADED_VERSION",
            "retrievedAt": "REPLACE_WITH_ISO_8601_TIMESTAMP",
            "datasetKind": catalog["datasetKind"],
            "realWorld": True,
            "anonymized": True,
            "labelsAreProviderGroundTruth": True,
            "labelsInferredOrImputed": False,
        },
        "license": {
            "name": catalog["licenseName"],
            "url": catalog["sourceUrl"],
            "accepted": False,
            "acceptedAt": None,
            "acceptedBy": "REPLACE_WITH_RESEARCHER_NAME",
            "redistributionAllowed": bool(catalog["redistributionAllowed"]),
            "localUseOnly": True,
        },
        "files": [
            {
                "role": "transactions",
                "path": catalog["requiredFile"],
                "sha256": "REPLACE_WITH_64_CHARACTER_SHA256",
                "bytes": None,
            }
        ],
        "scientificContract": {
            "task": "transaction_level_fraud_classification",
            "eventTimeAvailable": True,
            "binaryFraudLabelAvailable": True,
            "sequenceEntityAvailable": bool(catalog["supportsSequenceEntity"]),
            "entityPolicy": catalog["entityPolicy"],
            "providerTestUsed": False,
            "targetUsedAsFeature": False,
            "futureInformationAllowed": False,
        },
        "createdAt": now,
        "instructions": [
            f"Accept the provider terms at {catalog['sourceUrl']}.",
            f"Place {catalog['requiredFile']} beside this manifest without modifying it.",
            "Replace source version, retrieval timestamp, researcher identity and the exact SHA-256.",
            "Set license.accepted=true only after personally accepting the applicable terms.",
            "Do not add the provider's unlabeled test file; the pipeline creates its own locked chronological test split.",
        ],
    }
    _atomic_write_json(destination, manifest)
    return manifest


def get_real_finance_data_status(*, manifest_path: Path | None = None) -> dict[str, Any]:
    source = manifest_path or REAL_FINANCE_MANIFEST_PATH
    if not source.is_file():
        return {
            "available": False,
            "templateAvailable": False,
            "readyForPreparation": False,
            "readyForThesisTraining": False,
            "manifestPath": str(source.resolve()),
            "message": "Genere el manifiesto curado y coloque el archivo real descargado manualmente.",
            "catalog": get_real_finance_source_catalog(),
        }
    try:
        manifest = read_json(source)
        checks = audit_real_finance_manifest(manifest, manifest_path=source, verify_files=True)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return {
            "available": True,
            "templateAvailable": True,
            "readyForPreparation": False,
            "readyForThesisTraining": False,
            "manifestPath": str(source.resolve()),
            "message": str(exc),
            "checks": {"passed": False, "reasons": [str(exc)]},
            "catalog": get_real_finance_source_catalog(),
        }
    active = False
    active_dataset_id = None
    metadata_path = SILVER_DIR / "finance_transactions.metadata.json"
    if metadata_path.is_file():
        metadata = read_json(metadata_path)
        active = metadata.get("provenance", {}).get("kind") == "real_world_curated_financial_dataset" and metadata.get("sourceManifest", {}).get("sha256") == checks.get("manifestSha256")
        active_dataset_id = metadata.get("datasetId") if active else None
    return {
        "available": True,
        "templateAvailable": True,
        "adapter": manifest.get("adapter"),
        "packageId": manifest.get("packageId"),
        "manifestPath": str(source.resolve()),
        "readyForPreparation": checks["passed"],
        "readyForThesisTraining": bool(active and checks["passed"] and checks["sequenceEntityAvailable"]),
        "active": active,
        "activeDatasetId": active_dataset_id,
        "checks": checks,
        "source": manifest.get("source", {}),
        "license": manifest.get("license", {}),
        "message": "Paquete financiero real verificado y listo para activarse." if checks["passed"] else "El paquete financiero real tiene controles pendientes.",
        "catalog": get_real_finance_source_catalog(),
    }


def audit_real_finance_manifest(
    manifest: dict[str, Any],
    *,
    manifest_path: Path,
    verify_files: bool,
) -> dict[str, Any]:
    reasons: list[str] = []
    adapter = str(manifest.get("adapter", ""))
    catalog = REAL_FINANCE_SOURCE_CATALOG.get(adapter)
    if catalog is None:
        reasons.append("El adaptador declarado no esta soportado.")
    source = manifest.get("source", {})
    license_contract = manifest.get("license", {})
    scientific = manifest.get("scientificContract", {})
    if not source.get("realWorld"):
        reasons.append("La fuente no declara evidencia transaccional real.")
    if not source.get("labelsAreProviderGroundTruth") or source.get("labelsInferredOrImputed"):
        reasons.append("Las etiquetas deben proceder del proveedor y no pueden inferirse ni imputarse.")
    if not license_contract.get("accepted") or not license_contract.get("acceptedAt"):
        reasons.append("El investigador debe aceptar y fechar explicitamente la licencia o reglas aplicables.")
    if str(source.get("sourceVersion", "")).startswith("REPLACE_") or not source.get("sourceVersion"):
        reasons.append("Falta fijar la version de la fuente.")
    retrieved_at = str(source.get("retrievedAt", ""))
    if retrieved_at.startswith("REPLACE_") or not _valid_iso_timestamp(retrieved_at):
        reasons.append("Falta un retrievedAt ISO-8601 valido.")
    if not scientific.get("eventTimeAvailable") or not scientific.get("binaryFraudLabelAvailable"):
        reasons.append("El contrato cientifico requiere tiempo y etiqueta binaria.")
    if scientific.get("providerTestUsed") or scientific.get("targetUsedAsFeature") or scientific.get("futureInformationAllowed"):
        reasons.append("El paquete autoriza test del proveedor, target como variable o informacion futura.")
    files = manifest.get("files", [])
    transaction_files = [item for item in files if item.get("role") == "transactions"]
    if len(transaction_files) != 1:
        reasons.append("Debe existir exactamente un archivo de transacciones.")
    verified_files: list[dict[str, Any]] = []
    package_root = manifest_path.parent.resolve()
    for item in transaction_files:
        relative = Path(str(item.get("path", "")))
        if relative.is_absolute() or ".." in relative.parts:
            reasons.append("La ruta del archivo debe ser relativa y permanecer dentro del paquete.")
            continue
        resolved = (package_root / relative).resolve()
        try:
            resolved.relative_to(package_root)
        except ValueError:
            reasons.append("La ruta del archivo escapa del paquete curado.")
            continue
        expected_hash = str(item.get("sha256", "")).lower()
        if not SHA256_PATTERN.fullmatch(expected_hash):
            reasons.append("El SHA-256 del archivo de transacciones es invalido o sigue como placeholder.")
            continue
        if verify_files:
            if not resolved.is_file():
                reasons.append(f"No existe el archivo requerido: {relative.as_posix()}.")
                continue
            actual_hash, actual_bytes = sha256_file(resolved)
            if actual_hash != expected_hash:
                reasons.append("El archivo real no coincide con el SHA-256 declarado.")
                continue
            expected_bytes = item.get("bytes")
            if expected_bytes is not None and int(expected_bytes) != actual_bytes:
                reasons.append("El tamano del archivo real no coincide con el manifiesto.")
                continue
            verified_files.append({"role": "transactions", "path": str(resolved), "sha256": actual_hash, "bytes": actual_bytes})
    manifest_hash, manifest_bytes = sha256_file(manifest_path)
    sequence_entity = bool(catalog and catalog["supportsSequenceEntity"] and scientific.get("sequenceEntityAvailable"))
    if catalog and not catalog["supportsSequenceEntity"]:
        reasons.append("La fuente no conserva una entidad compatible con secuencias por cliente; solo puede usarse como validacion tabular externa.")
    return {
        "passed": not reasons,
        "reasons": reasons,
        "adapterSupported": catalog is not None,
        "realWorldDeclared": bool(source.get("realWorld")),
        "providerLabelsDeclared": bool(source.get("labelsAreProviderGroundTruth") and not source.get("labelsInferredOrImputed")),
        "licenseAccepted": bool(license_contract.get("accepted") and license_contract.get("acceptedAt")),
        "sourceVersionFixed": bool(source.get("sourceVersion") and not str(source.get("sourceVersion")).startswith("REPLACE_")),
        "filesVerified": len(verified_files) == 1,
        "verifiedFiles": verified_files,
        "sequenceEntityAvailable": sequence_entity,
        "providerTestUsed": bool(scientific.get("providerTestUsed")),
        "targetUsedAsFeature": bool(scientific.get("targetUsedAsFeature")),
        "futureInformationAllowed": bool(scientific.get("futureInformationAllowed")),
        "manifestSha256": manifest_hash,
        "manifestBytes": manifest_bytes,
    }


def prepare_real_finance_dataset(
    *,
    manifest_path: Path | None = None,
    silver_path: Path | None = None,
    metadata_path: Path | None = None,
    audit_path: Path | None = None,
    minimum_rows: int = 100_000,
    minimum_fraud_per_split: int = 100,
    minimum_entities: int = 1_000,
) -> dict[str, Any]:
    ensure_dirs()
    manifest_source = manifest_path or REAL_FINANCE_MANIFEST_PATH
    if not manifest_source.is_file():
        raise FileNotFoundError("Primero genere y complete el manifiesto financiero real.")
    manifest = read_json(manifest_source)
    manifest_checks = audit_real_finance_manifest(manifest, manifest_path=manifest_source, verify_files=True)
    if not manifest_checks["passed"]:
        raise ValueError("Paquete financiero real rechazado: " + " ".join(manifest_checks["reasons"]))
    transaction_path = Path(manifest_checks["verifiedFiles"][0]["path"])
    adapter = manifest["adapter"]
    if adapter == "ieee_cis":
        canonical, adapter_audit = adapt_ieee_cis_transactions(transaction_path)
    else:
        raise ValueError("ULB/Worldline queda reservado para validacion tabular porque no contiene entidad secuencial.")
    audit = audit_real_finance_dataset(
        canonical,
        adapter_audit=adapter_audit,
        minimum_rows=minimum_rows,
        minimum_fraud_per_split=minimum_fraud_per_split,
        minimum_entities=minimum_entities,
    )
    if not audit["readiness"]["readyForThesisTraining"]:
        raise ValueError("Dataset financiero real rechazado por la puerta de tesis: " + " ".join(audit["readiness"]["thesisReasons"]))

    destination = silver_path or (SILVER_DIR / "finance_transactions.csv")
    metadata_destination = metadata_path or (SILVER_DIR / "finance_transactions.metadata.json")
    audit_destination = audit_path or (RESULTS_DIR / "finance_data_audit.json")
    _atomic_write_csv(destination, canonical)
    silver_hash, silver_bytes = sha256_file(destination)
    source_file = manifest_checks["verifiedFiles"][0]
    prepared_at = datetime.now(timezone.utc).isoformat()
    dataset_id = f"finance-real-{adapter}-{manifest_checks['manifestSha256'][:8]}-{silver_hash[:12]}"
    metadata = {
        "schemaVersion": "2.0.0",
        "datasetId": dataset_id,
        "domain": "finanzas",
        "task": "transaction_level_fraud_classification",
        "preparedAt": prepared_at,
        "provider": manifest["source"]["provider"],
        "citation": {
            "title": REAL_FINANCE_SOURCE_CATALOG[adapter]["displayName"],
            "authors": manifest["source"]["provider"],
            "year": None,
            "url": manifest["source"]["sourceUrl"],
            "repository": manifest["source"]["sourceUrl"],
            "repositoryCommit": manifest["source"]["sourceVersion"],
        },
        "provenance": {
            "kind": "real_world_curated_financial_dataset",
            "adapter": adapter,
            "realWorld": True,
            "anonymized": bool(manifest["source"].get("anonymized")),
            "providerLabels": True,
            "labelsInferredOrImputed": False,
            "manualAuthenticatedAcquisition": True,
            "redistributionAllowed": bool(manifest["license"].get("redistributionAllowed")),
            "licenseName": manifest["license"]["name"],
            "licenseAcceptedAt": manifest["license"]["acceptedAt"],
        },
        "contract": {
            "eventId": "transaction_id",
            "eventTime": "transaction_time",
            "entityId": "customer_id",
            "counterpartyId": "terminal_id",
            "amount": "amount",
            "target": "is_fraud",
            "targetMeaning": {"0": "legitimate", "1": "fraudulent"},
            "columns": list(canonical.columns),
            "entityPolicy": adapter_audit["entityPolicy"],
            "timePolicy": adapter_audit["timePolicy"],
        },
        "configuration": {
            "adapter": adapter,
            "sourceVersion": manifest["source"]["sourceVersion"],
            "splitPolicy": "chronological_unique_time_60_20_20",
            "providerTestUsed": False,
            "selectedSourceFeatureColumns": adapter_audit["selectedSourceFeatureColumns"],
            "sequenceFeaturePolicy": "causal_amount_time_and_prior_entity_aggregates; source features retained for future ablation",
        },
        "sourceManifest": {
            "path": str(manifest_source.resolve()),
            "sha256": manifest_checks["manifestSha256"],
            "bytes": manifest_checks["manifestBytes"],
        },
        "sourceArtifacts": [source_file],
        "silver": {"path": str(destination.resolve()), "sha256": silver_hash, "bytes": silver_bytes, "rows": int(len(canonical)), "columns": list(canonical.columns)},
        "auditPath": str(audit_destination.resolve()),
        "readyForPipelinePilot": True,
        "readyForThesisTraining": True,
        "thesisLimitation": "Dataset real habilitado para entrenamiento; las conclusiones finales aun requieren cinco semillas, validacion, congelamiento y una unica evaluacion de test.",
        "testLock": {
            "locked": True,
            "evaluated": False,
            "encoded": False,
            "maximumEvaluations": 1,
            "policy": "El 20% cronologico final se excluye de variables, OOF, seleccion, calibracion y Stacking hasta autorizacion separada.",
        },
    }
    audit_payload = {"schemaVersion": "2.0.0", "datasetId": dataset_id, "createdAt": prepared_at, **audit}
    _atomic_write_json(audit_destination, audit_payload)
    _atomic_write_json(metadata_destination, metadata)
    return {"metadata": metadata, "audit": audit_payload}


def adapt_ieee_cis_transactions(source: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    required = set(REAL_FINANCE_SOURCE_CATALOG["ieee_cis"]["requiredColumns"])
    source_columns = list(pd.read_csv(source, nrows=0).columns)
    missing = required - set(source_columns)
    if missing:
        raise ValueError(f"IEEE-CIS no contiene columnas requeridas: {', '.join(sorted(missing))}")
    optional_columns = {
        "card2", "card3", "card5", "card6", "ProductCD", "addr1", "addr2", "P_emaildomain", "dist1", "dist2",
        *[f"C{index}" for index in range(1, 15)],
        *[f"D{index}" for index in range(1, 6)],
    }
    selected_read_columns = [column for column in source_columns if column in required or column in optional_columns]
    frame = pd.read_csv(source, usecols=selected_read_columns)
    data = frame.copy()
    for column in ("TransactionID", "TransactionDT", "TransactionAmt", "isFraud", "card1"):
        data[column] = pd.to_numeric(data[column], errors="coerce")
    if data[["TransactionID", "TransactionDT", "TransactionAmt", "isFraud", "card1"]].isna().any().any():
        raise ValueError("IEEE-CIS contiene nulos o valores no numericos en el contrato obligatorio.")
    if not set(data["isFraud"].astype(int).unique()).issubset({0, 1}):
        raise ValueError("IEEE-CIS contiene etiquetas fuera de {0,1}.")
    if data["TransactionID"].duplicated().any():
        raise ValueError("IEEE-CIS contiene TransactionID duplicado.")
    data = data.sort_values(["TransactionDT", "TransactionID"], kind="stable").reset_index(drop=True)
    entity_columns = [column for column in ("card1", "card2", "card3", "card5", "card6") if column in data.columns]
    terminal_columns = [column for column in ("ProductCD", "addr1", "P_emaildomain") if column in data.columns]
    entity_key = _composite_key(data, entity_columns)
    terminal_key = _composite_key(data, terminal_columns) if terminal_columns else pd.Series("unknown_terminal", index=data.index)
    customer_id = _stable_category_codes(entity_key)
    terminal_id = _stable_category_codes(terminal_key)
    relative_seconds = data["TransactionDT"].astype(np.int64)
    transaction_time = pd.Timestamp("2000-01-01", tz="UTC") + pd.to_timedelta(relative_seconds, unit="s")
    split = chronological_split_from_relative_time(relative_seconds.to_numpy(dtype=np.int64))
    selected_source_columns = [
        column for column in (
            "card2", "card3", "card5", "addr1", "addr2", "dist1", "dist2",
            *[f"C{index}" for index in range(1, 15)],
            *[f"D{index}" for index in range(1, 6)],
        ) if column in data.columns
    ]
    canonical = pd.DataFrame({
        "transaction_id": data["TransactionID"].astype(np.int64),
        "transaction_time": transaction_time.map(lambda value: value.isoformat()),
        "customer_id": customer_id,
        "terminal_id": terminal_id,
        "amount": data["TransactionAmt"].astype(np.float64),
        "is_fraud": data["isFraud"].astype(np.int8),
        "fraud_scenario": data["isFraud"].astype(np.int8),
        "split": split,
        "source_relative_time_seconds": relative_seconds,
    })
    for column in selected_source_columns:
        canonical[f"source_feature__{column}"] = pd.to_numeric(data[column], errors="coerce").astype(np.float32)
    return canonical, {
        "adapter": "ieee_cis",
        "sourceRows": int(len(data)),
        "sourceColumns": int(len(source_columns)),
        "columnsReadByAdapter": int(len(selected_read_columns)),
        "unusedSourceColumnsLoaded": False,
        "selectedSourceFeatureColumns": selected_source_columns,
        "entityColumns": entity_columns,
        "terminalColumns": terminal_columns,
        "entityPolicy": "stable categorical code of anonymized card1/card2/card3/card5/card6 composite",
        "timePolicy": "TransactionDT relative seconds anchored to a neutral synthetic epoch; order and deltas preserved",
        "originalAbsoluteTimeKnown": False,
        "providerTestUsed": False,
        "labelsModified": False,
    }


def chronological_split_from_relative_time(relative_time: np.ndarray) -> np.ndarray:
    values = np.asarray(relative_time, dtype=np.int64).reshape(-1)
    unique = np.unique(values)
    if len(unique) < 10:
        raise ValueError("La fuente real no contiene suficiente diversidad temporal.")
    train_cut = unique[max(1, int(len(unique) * 0.60))]
    validation_cut = unique[max(2, int(len(unique) * 0.80))]
    if train_cut >= validation_cut:
        raise ValueError("No se pudieron fijar cortes temporales estrictos.")
    result = np.where(values < train_cut, "train", np.where(values < validation_cut, "validation", "test"))
    if set(result) != {"train", "validation", "test"}:
        raise ValueError("La particion temporal real no contiene los tres splits.")
    return result


def audit_real_finance_dataset(
    frame: pd.DataFrame,
    *,
    adapter_audit: dict[str, Any],
    minimum_rows: int,
    minimum_fraud_per_split: int,
    minimum_entities: int,
) -> dict[str, Any]:
    core = frame[list(FINANCE_COLUMNS)].copy()
    base = audit_finance_benchmark(core, minimum_rows=minimum_rows)
    thesis_reasons = list(base["readiness"]["reasons"])
    split_fraud_minimum_passed = True
    for split in ("train", "validation", "test"):
        fraud_rows = int(frame.loc[frame["split"] == split, "is_fraud"].sum())
        if fraud_rows < minimum_fraud_per_split:
            split_fraud_minimum_passed = False
            thesis_reasons.append(f"{split} contiene {fraud_rows} fraudes; se requieren al menos {minimum_fraud_per_split}.")
    entity_count = int(frame["customer_id"].nunique())
    if entity_count < minimum_entities:
        thesis_reasons.append(f"Solo existen {entity_count} entidades; se requieren al menos {minimum_entities}.")
    source_features = [column for column in frame.columns if column.startswith("source_feature__")]
    all_missing_source_features = [column for column in source_features if frame[column].isna().all()]
    if all_missing_source_features:
        thesis_reasons.append("Existen variables fuente completamente vacias: " + ", ".join(all_missing_source_features))
    ready = not thesis_reasons
    return {
        **base,
        "sourceAudit": adapter_audit,
        "customers": entity_count,
        "sourceFeatureQuality": {
            "retainedColumns": source_features,
            "missingValues": {column: int(frame[column].isna().sum()) for column in source_features},
            "allMissingColumns": all_missing_source_features,
            "fitStatisticsComputedGlobally": False,
            "imputedGlobally": False,
        },
        "labelAudit": {
            "providerGroundTruth": True,
            "labelsModified": False,
            "labelsInferred": False,
            "labelsImputed": False,
            "minimumFraudPerSplit": minimum_fraud_per_split,
            "minimumFraudPerSplitPassed": split_fraud_minimum_passed,
        },
        "readiness": {
            "readyForPipelinePilot": ready,
            "readyForThesisTraining": ready,
            "reasons": thesis_reasons,
            "thesisReasons": thesis_reasons,
        },
    }


def _composite_key(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    if not columns:
        raise ValueError("No existen columnas para construir el proxy de entidad.")
    normalized = frame[columns].copy()
    for column in columns:
        normalized[column] = normalized[column].astype("string").fillna("<missing>")
    return normalized.agg("|".join, axis=1)


def _stable_category_codes(values: pd.Series) -> np.ndarray:
    categories = sorted(str(value) for value in values.unique())
    mapping = {value: index for index, value in enumerate(categories)}
    return values.astype(str).map(mapping).to_numpy(dtype=np.int64)


def _valid_iso_timestamp(value: str) -> bool:
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
        return True
    except (TypeError, ValueError):
        return False


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
