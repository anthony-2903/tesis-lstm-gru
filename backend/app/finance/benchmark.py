from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from app.config import RESULTS_DIR, SILVER_DIR, ensure_dirs
from app.phishing.ingestion import sha256_file
from app.utils import read_json, write_json


FINANCE_COLUMNS = (
    "transaction_id",
    "transaction_time",
    "customer_id",
    "terminal_id",
    "amount",
    "is_fraud",
    "fraud_scenario",
    "split",
)
DEFAULT_START_DATE = "2018-04-01"
HANDBOOK_URL = "https://fraud-detection-handbook.github.io/fraud-detection-handbook/Chapter_3_GettingStarted/SimulatedDataset.html"
HANDBOOK_REPOSITORY = "https://github.com/Fraud-Detection-Handbook/fraud-detection-handbook"
HANDBOOK_COMMIT = "81cf7d1714bb7b2f5b496407d9055d91dc68dc25"


def prepare_finance_benchmark(
    *,
    days: int = 100,
    customers: int = 500,
    terminals: int = 200,
    seed: int = 42,
    minimum_rows: int = 10_000,
    silver_path: Path | None = None,
    metadata_path: Path | None = None,
    audit_path: Path | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Build a deterministic temporal fraud benchmark without loading remote pickle files."""
    if days < 30:
        raise ValueError("El benchmark financiero requiere al menos 30 días.")
    if customers < 25 or terminals < 10:
        raise ValueError("Se requieren al menos 25 clientes y 10 terminales.")
    ensure_dirs()
    destination = silver_path or (SILVER_DIR / "finance_transactions.csv")
    metadata_destination = metadata_path or (SILVER_DIR / "finance_transactions.metadata.json")
    audit_destination = audit_path or (RESULTS_DIR / "finance_data_audit.json")

    _notify(progress_callback, stage="generating", percent=10.0, message="Generando transacciones y escenarios de fraude reproducibles.")
    frame = generate_finance_transactions(days=days, customers=customers, terminals=terminals, seed=seed)
    _notify(progress_callback, stage="splitting", percent=65.0, message="Aplicando partición cronológica train, validation y test bloqueado.")
    silver = apply_chronological_finance_split(frame, days=days)
    audit = audit_finance_benchmark(silver, minimum_rows=minimum_rows)
    _notify(progress_callback, stage="persisting", percent=85.0, message="Guardando silver y manifiestos con SHA-256.")

    destination.parent.mkdir(parents=True, exist_ok=True)
    silver.to_csv(destination, index=False)
    silver_hash, silver_bytes = sha256_file(destination)
    prepared_at = datetime.now(timezone.utc).isoformat()
    configuration = {
        "days": days,
        "customers": customers,
        "terminals": terminals,
        "seed": seed,
        "startDate": DEFAULT_START_DATE,
        "splitPolicy": "chronological_60_20_20",
    }
    fingerprint = hashlib.sha256(
        repr(sorted(configuration.items())).encode("utf-8")
    ).hexdigest()
    dataset_id = f"finance-fdh-{fingerprint[:8]}-{silver_hash[:12]}"
    metadata = {
        "schemaVersion": "1.0.0",
        "datasetId": dataset_id,
        "domain": "finanzas",
        "task": "transaction_level_fraud_classification",
        "preparedAt": prepared_at,
        "provider": "Fraud Detection Handbook - Machine Learning Group, ULB",
        "citation": {
            "title": "Reproducible Machine Learning for Credit Card Fraud Detection - Practical Handbook",
            "authors": "Le Borgne, Siblini, Lebichot and Bontempi",
            "year": 2022,
            "url": HANDBOOK_URL,
            "repository": HANDBOOK_REPOSITORY,
            "repositoryCommit": HANDBOOK_COMMIT,
        },
        "provenance": {
            "kind": "locally_generated_synthetic_benchmark",
            "upstreamDataFilesExecuted": False,
            "upstreamPicklesLoaded": False,
            "generatorPolicy": "Deterministic clean implementation of the public transaction and time-dependent fraud scenario specification.",
            "methodologyLicense": "Handbook prose CC BY-SA 4.0; upstream notebook code GPL-3.0 was not imported or executed.",
        },
        "contract": {
            "eventId": "transaction_id",
            "eventTime": "transaction_time",
            "entityId": "customer_id",
            "counterpartyId": "terminal_id",
            "amount": "amount",
            "target": "is_fraud",
            "targetMeaning": {"0": "legitimate", "1": "fraudulent"},
            "columns": list(FINANCE_COLUMNS),
        },
        "configuration": configuration,
        "silver": {
            "path": str(destination.resolve()),
            "sha256": silver_hash,
            "bytes": silver_bytes,
            "rows": int(len(silver)),
            "columns": list(silver.columns),
        },
        "auditPath": str(audit_destination.resolve()),
        "readyForPipelinePilot": bool(audit["readiness"]["readyForPipelinePilot"]),
        "readyForThesisTraining": False,
        "thesisLimitation": "Este benchmark es sintético. La corrida final de tesis requiere además validación externa con transacciones reales etiquetadas y licenciadas.",
        "testLock": {
            "locked": True,
            "evaluated": False,
            "maximumEvaluations": 1,
            "policy": "El split test no participa en ingeniería de variables, OOF, selección, calibración ni Stacking.",
        },
    }
    audit_payload = {"schemaVersion": "1.0.0", "datasetId": dataset_id, "createdAt": prepared_at, **audit}
    write_json(audit_destination, audit_payload)
    write_json(metadata_destination, metadata)
    _notify(progress_callback, stage="completed", percent=100.0, message="Benchmark financiero preparado; test permanece bloqueado.")
    return {"metadata": metadata, "audit": audit_payload}


def generate_finance_transactions(
    *,
    days: int,
    customers: int,
    terminals: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    customer_mean = rng.uniform(10.0, 120.0, size=customers)
    customer_frequency = rng.uniform(0.5, 3.0, size=customers)
    terminal_choices = [rng.choice(terminals, size=min(10, terminals), replace=False) for _ in range(customers)]
    terminal_compromised_until = np.full(terminals, -1, dtype=np.int32)
    customer_compromised_until = np.full(customers, -1, dtype=np.int32)
    origin = datetime.fromisoformat(DEFAULT_START_DATE)
    rows: list[dict[str, Any]] = []
    transaction_id = 0

    for day in range(days):
        compromised_terminal = int(rng.integers(0, terminals))
        terminal_compromised_until[compromised_terminal] = max(terminal_compromised_until[compromised_terminal], day + 3)
        compromised_customers = rng.choice(customers, size=min(2, customers), replace=False)
        customer_compromised_until[compromised_customers] = np.maximum(customer_compromised_until[compromised_customers], day + 5)

        for customer_id in range(customers):
            transaction_count = int(rng.poisson(customer_frequency[customer_id]))
            for _ in range(transaction_count):
                amount = max(0.5, float(rng.normal(customer_mean[customer_id], customer_mean[customer_id] / 2.0)))
                terminal_id = int(rng.choice(terminal_choices[customer_id]))
                scenario = 0
                if amount > 220.0:
                    scenario = 1
                elif terminal_compromised_until[terminal_id] >= day:
                    scenario = 2
                elif customer_compromised_until[customer_id] >= day and rng.random() < 0.25:
                    amount *= 4.0
                    scenario = 3
                seconds = int(rng.integers(0, 86_400))
                transaction_time = origin + timedelta(days=day, seconds=seconds)
                rows.append({
                    "transaction_id": transaction_id,
                    "transaction_time": transaction_time.isoformat(),
                    "customer_id": customer_id,
                    "terminal_id": terminal_id,
                    "amount": round(amount, 2),
                    "is_fraud": int(scenario > 0),
                    "fraud_scenario": scenario,
                    "day_index": day,
                })
                transaction_id += 1
    return pd.DataFrame(rows).sort_values(["transaction_time", "transaction_id"], kind="stable").reset_index(drop=True)


def apply_chronological_finance_split(frame: pd.DataFrame, *, days: int) -> pd.DataFrame:
    required = {"transaction_id", "transaction_time", "customer_id", "terminal_id", "amount", "is_fraud", "fraud_scenario", "day_index"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Faltan columnas financieras: {', '.join(sorted(missing))}")
    train_end = max(1, int(days * 0.60))
    validation_end = max(train_end + 1, int(days * 0.80))
    result = frame.copy()
    result["split"] = np.select(
        [result["day_index"] < train_end, result["day_index"] < validation_end],
        ["train", "validation"],
        default="test",
    )
    return result[list(FINANCE_COLUMNS)].sort_values(["transaction_time", "transaction_id"], kind="stable").reset_index(drop=True)


def audit_finance_benchmark(frame: pd.DataFrame, *, minimum_rows: int = 10_000) -> dict[str, Any]:
    missing = set(FINANCE_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"El silver financiero no cumple el contrato: {', '.join(sorted(missing))}")
    timestamps = pd.to_datetime(frame["transaction_time"], errors="coerce")
    invalid_timestamps = int(timestamps.isna().sum())
    duplicate_transactions = int(frame["transaction_id"].duplicated().sum())
    invalid_labels = int((~frame["is_fraud"].isin([0, 1])).sum())
    invalid_amounts = int((pd.to_numeric(frame["amount"], errors="coerce") <= 0).sum())
    split_items = []
    split_has_both_labels = True
    for split in ("train", "validation", "test"):
        subset = frame[frame["split"] == split]
        labels = sorted(int(value) for value in subset["is_fraud"].unique())
        split_has_both_labels = split_has_both_labels and labels == [0, 1]
        split_items.append({
            "split": split,
            "rows": int(len(subset)),
            "fraudRows": int(subset["is_fraud"].sum()),
            "fraudRate": float(subset["is_fraud"].mean()) if len(subset) else 0.0,
            "startAt": str(subset["transaction_time"].min()) if len(subset) else None,
            "endAt": str(subset["transaction_time"].max()) if len(subset) else None,
            "labels": labels,
        })
    chronological = all(
        split_items[index]["endAt"] < split_items[index + 1]["startAt"]
        for index in range(len(split_items) - 1)
        if split_items[index]["endAt"] and split_items[index + 1]["startAt"]
    )
    reasons = []
    if len(frame) < minimum_rows:
        reasons.append(f"Se generaron {len(frame)} filas; se requieren al menos {minimum_rows} para el piloto.")
    if invalid_timestamps or duplicate_transactions or invalid_labels or invalid_amounts:
        reasons.append("Existen valores inválidos en tiempo, identidad, etiqueta o monto.")
    if not split_has_both_labels:
        reasons.append("Cada split debe contener casos legítimos y fraudulentos.")
    if not chronological:
        reasons.append("Los splits no respetan el orden temporal estricto.")
    ready = not reasons
    return {
        "rows": int(len(frame)),
        "customers": int(frame["customer_id"].nunique()),
        "terminals": int(frame["terminal_id"].nunique()),
        "fraudRows": int(frame["is_fraud"].sum()),
        "fraudRate": float(frame["is_fraud"].mean()),
        "quality": {
            "invalidTimestamps": invalid_timestamps,
            "duplicateTransactions": duplicate_transactions,
            "invalidLabels": invalid_labels,
            "invalidAmounts": invalid_amounts,
        },
        "splits": split_items,
        "temporalAudit": {
            "chronological": chronological,
            "testLocked": True,
            "testEvaluated": False,
            "futureInformationUsed": False,
        },
        "readiness": {
            "readyForPipelinePilot": ready,
            "readyForThesisTraining": False,
            "reasons": reasons,
            "thesisReasons": [
                "El benchmark es sintético y sirve para validar la arquitectura, no para sostener por sí solo la conclusión final.",
                "Falta incorporar una fuente real, etiquetada, licenciada y temporalmente versionada para validación externa.",
            ],
        },
    }


def get_finance_dataset_status(
    *,
    metadata_path: Path | None = None,
    audit_path: Path | None = None,
) -> dict[str, Any]:
    metadata_source = metadata_path or (SILVER_DIR / "finance_transactions.metadata.json")
    audit_source = audit_path or (RESULTS_DIR / "finance_data_audit.json")
    if not metadata_source.exists() or not audit_source.exists():
        return {
            "available": False,
            "readyForPipelinePilot": False,
            "readyForThesisTraining": False,
            "message": "El benchmark transaccional financiero todavía no ha sido preparado.",
            "task": "transaction_level_fraud_classification",
            "testLock": {"locked": True, "evaluated": False},
        }
    metadata = read_json(metadata_source)
    audit = read_json(audit_source)
    silver = metadata.get("silver", {})
    silver_path = Path(silver.get("path", ""))
    lineage_current = False
    if silver_path.is_file():
        actual_hash, actual_bytes = sha256_file(silver_path)
        lineage_current = actual_hash == silver.get("sha256") and actual_bytes == silver.get("bytes")
    source_lineage_current = _verify_real_source_lineage(metadata) if metadata.get("provenance", {}).get("kind") == "real_world_curated_financial_dataset" else True
    pilot_ready = bool(audit.get("readiness", {}).get("readyForPipelinePilot")) and lineage_current and source_lineage_current
    thesis_ready = (
        pilot_ready
        and bool(metadata.get("readyForThesisTraining"))
        and bool(audit.get("readiness", {}).get("readyForThesisTraining"))
        and metadata.get("provenance", {}).get("kind") == "real_world_curated_financial_dataset"
        and bool(metadata.get("provenance", {}).get("providerLabels"))
        and not bool(metadata.get("provenance", {}).get("labelsInferredOrImputed"))
    )
    return {
        "available": silver_path.is_file(),
        "datasetId": metadata.get("datasetId"),
        "task": metadata.get("task"),
        "provider": metadata.get("provider"),
        "citation": metadata.get("citation", {}),
        "provenance": metadata.get("provenance", {}),
        "contract": metadata.get("contract", {}),
        "configuration": metadata.get("configuration", {}),
        "silver": silver,
        "audit": audit,
        "lineageCurrent": lineage_current,
        "sourceLineageCurrent": source_lineage_current,
        "readyForPipelinePilot": pilot_ready,
        "readyForThesisTraining": thesis_ready,
        "testLock": metadata.get("testLock", {"locked": True, "evaluated": False}),
        "message": "Benchmark listo para implementar secuencias y OOF financiero." if pilot_ready else "El benchmark financiero no supera todavía la puerta de datos.",
    }


def _verify_real_source_lineage(metadata: dict[str, Any]) -> bool:
    source_manifest = metadata.get("sourceManifest", {})
    artifacts = metadata.get("sourceArtifacts", [])
    items = [source_manifest, *artifacts]
    if not source_manifest or not artifacts:
        return False
    for item in items:
        path = Path(str(item.get("path", "")))
        if not path.is_file():
            return False
        digest, size = sha256_file(path)
        if digest != item.get("sha256") or size != item.get("bytes"):
            return False
    return True


def _notify(callback: Callable[[dict[str, Any]], None] | None, **event: Any) -> None:
    if callback is not None:
        callback(event)
