from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from app.finance.benchmark import get_finance_dataset_status
from app.finance.real_data import (
    adapt_ieee_cis_transactions,
    audit_real_finance_manifest,
    get_real_finance_data_status,
    initialize_real_finance_template,
    prepare_real_finance_dataset,
)
from app.finance.sequences import get_finance_sequence_status, prepare_finance_sequence_protocol
from app.phishing.ingestion import sha256_file
from app.utils import read_json, write_json


class FinanceRealDataTests(unittest.TestCase):
    def test_template_requires_license_version_hash_and_local_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = Path(directory) / "real_finance_manifest.json"
            initialize_real_finance_template(manifest_path=manifest_path)
            status = get_real_finance_data_status(manifest_path=manifest_path)
            self.assertFalse(status["readyForPreparation"])
            reasons = " ".join(status["checks"]["reasons"])
            self.assertIn("licencia", reasons)
            self.assertIn("SHA-256", reasons)
            self.assertFalse(status["readyForThesisTraining"])

    def test_ieee_adapter_preserves_order_labels_entities_and_locked_test(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "train_transaction.csv"
            self._write_ieee_fixture(source)
            canonical, audit = adapt_ieee_cis_transactions(source)
            self.assertEqual(len(canonical), 300)
            self.assertEqual(set(canonical["split"]), {"train", "validation", "test"})
            self.assertEqual(set(canonical["is_fraud"]), {0, 1})
            self.assertGreater(canonical["customer_id"].nunique(), 10)
            self.assertFalse(audit["providerTestUsed"])
            for current, following in (("train", "validation"), ("validation", "test")):
                self.assertLess(canonical.loc[canonical["split"] == current, "transaction_time"].max(), canonical.loc[canonical["split"] == following, "transaction_time"].min())

    def test_real_package_activates_thesis_gate_and_detects_source_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = root / "real_finance_manifest.json"
            source = root / "train_transaction.csv"
            silver = root / "finance_transactions.csv"
            metadata = root / "finance_transactions.metadata.json"
            audit_path = root / "finance_data_audit.json"
            sequence_path = root / "finance_sequences.npz"
            feature_path = root / "finance_features.csv"
            assignment_path = root / "finance_oof_assignments.csv"
            sequence_manifest_path = root / "finance_sequence_manifest.json"
            self._write_ieee_fixture(source)
            initialize_real_finance_template(manifest_path=manifest_path)
            manifest = read_json(manifest_path)
            digest, size = sha256_file(source)
            manifest["packageId"] = "finance-real-ieee-cis-fixture-v1"
            manifest["source"]["sourceVersion"] = "fixture-v1"
            manifest["source"]["retrievedAt"] = datetime.now(timezone.utc).isoformat()
            manifest["license"]["accepted"] = True
            manifest["license"]["acceptedAt"] = datetime.now(timezone.utc).isoformat()
            manifest["license"]["acceptedBy"] = "University Researcher"
            manifest["files"][0]["sha256"] = digest
            manifest["files"][0]["bytes"] = size
            write_json(manifest_path, manifest)

            prepared = prepare_real_finance_dataset(
                manifest_path=manifest_path,
                silver_path=silver,
                metadata_path=metadata,
                audit_path=audit_path,
                minimum_rows=100,
                minimum_fraud_per_split=3,
                minimum_entities=20,
            )
            status = get_finance_dataset_status(metadata_path=metadata, audit_path=audit_path)
            self.assertTrue(prepared["metadata"]["readyForThesisTraining"])
            self.assertEqual(prepared["metadata"]["provenance"]["kind"], "real_world_curated_financial_dataset")
            self.assertTrue(status["lineageCurrent"])
            self.assertTrue(status["sourceLineageCurrent"])
            self.assertTrue(status["readyForThesisTraining"])
            self.assertTrue(status["testLock"]["locked"])
            self.assertFalse(status["testLock"]["evaluated"])
            self.assertFalse(prepared["audit"]["sourceAudit"]["providerTestUsed"])

            prepare_finance_sequence_protocol(
                window=5,
                purge_days=0,
                silver_path=silver,
                metadata_path=metadata,
                audit_path=audit_path,
                sequences_path=sequence_path,
                features_path=feature_path,
                assignments_path=assignment_path,
                manifest_path=sequence_manifest_path,
            )
            sequence_status = get_finance_sequence_status(manifest_path=sequence_manifest_path, metadata_path=metadata)
            self.assertTrue(sequence_status["readyForBaseModelPilot"])
            self.assertTrue(sequence_status["readyForThesisTraining"])
            self.assertTrue(sequence_status["sourceLineageCurrent"])
            self.assertEqual(sequence_status["sequences"]["testRowsEncoded"], 0)

            source.write_text(source.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            stale = get_finance_dataset_status(metadata_path=metadata, audit_path=audit_path)
            self.assertFalse(stale["sourceLineageCurrent"])
            self.assertFalse(stale["readyForThesisTraining"])

    def test_rejects_unaccepted_license_without_overwriting_destinations(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = root / "real_finance_manifest.json"
            source = root / "train_transaction.csv"
            silver = root / "protected.csv"
            metadata = root / "protected.json"
            audit_path = root / "protected_audit.json"
            self._write_ieee_fixture(source)
            initialize_real_finance_template(manifest_path=manifest_path)
            manifest = read_json(manifest_path)
            digest, size = sha256_file(source)
            manifest["source"]["sourceVersion"] = "fixture-v1"
            manifest["source"]["retrievedAt"] = datetime.now(timezone.utc).isoformat()
            manifest["files"][0]["sha256"] = digest
            manifest["files"][0]["bytes"] = size
            write_json(manifest_path, manifest)
            with self.assertRaisesRegex(ValueError, "licencia"):
                prepare_real_finance_dataset(manifest_path=manifest_path, silver_path=silver, metadata_path=metadata, audit_path=audit_path, minimum_rows=100, minimum_fraud_per_split=3, minimum_entities=20)
            self.assertFalse(silver.exists())
            self.assertFalse(metadata.exists())
            self.assertFalse(audit_path.exists())

    def test_manifest_rejects_path_escape(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = root / "real_finance_manifest.json"
            initialize_real_finance_template(manifest_path=manifest_path)
            manifest = read_json(manifest_path)
            manifest["files"][0]["path"] = "../outside.csv"
            manifest["files"][0]["sha256"] = "a" * 64
            write_json(manifest_path, manifest)
            checks = audit_real_finance_manifest(manifest, manifest_path=manifest_path, verify_files=True)
            self.assertFalse(checks["passed"])
            self.assertIn("ruta", " ".join(checks["reasons"]))

    @staticmethod
    def _write_ieee_fixture(path: Path) -> None:
        rows = []
        for index in range(300):
            rows.append({
                "TransactionID": 1_000_000 + index,
                "TransactionDT": (index // 10) * 86_400,
                "TransactionAmt": 10.0 + (index % 50),
                "isFraud": int(index % 10 == 0),
                "card1": 10_000 + (index % 50),
                "card2": 100 + (index % 5),
                "card3": 150,
                "card5": 200 + (index % 3),
                "card6": "credit" if index % 2 else "debit",
                "ProductCD": "W" if index % 2 else "C",
                "addr1": 200 + (index % 7),
                "P_emaildomain": "example.test",
                "C1": float(index % 9),
                "D1": float(index % 11),
            })
        pd.DataFrame(rows).to_csv(path, index=False)


if __name__ == "__main__":
    unittest.main()
