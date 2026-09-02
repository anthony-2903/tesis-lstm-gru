from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from app.phishing.data_jobs import PhishingDataPreparationManager
from app.phishing.curation import get_phishing_curation_status, load_curated_source_package
from app.phishing.ingestion import (
    build_audited_phishing_silver,
    download_versioned_source,
    normalize_url,
    sha256_file,
)


def _record(url: str, label: int, source: str, record_id: int) -> dict:
    normalized = normalize_url(url)
    assert normalized is not None
    return {
        **normalized,
        "is_phishing": label,
        "source": source,
        "source_record_id": str(record_id),
        "label_provenance": "test",
        "collected_at": "2026-01-01T00:00:00Z",
        "source_rank": record_id,
    }


def _verified_record(label: int, source: str, record_id: int) -> dict:
    hostname = f"{source}-{label}-{record_id}.test"
    return {
        "url": f"https://{hostname}/account/{record_id}",
        "canonical_url": f"https://{hostname}/account/{record_id}",
        "hostname": hostname,
        "registrable_domain": hostname,
        "is_phishing": label,
        "source": source,
        "source_record_id": str(record_id),
        "label_provenance": f"{source}:expert_consensus",
        "label_verified": True,
        "verification_method": "expert_consensus",
        "verification_reference": f"https://evidence.test/{source}/{record_id}",
        "verified_at": "2026-01-01T00:00:00Z",
        "collected_at": "2026-01-01T00:00:00Z",
    }


class PhishingIngestionTests(unittest.TestCase):
    def test_accepts_scientific_dataset_only_with_verified_multisource_evidence(self) -> None:
        positives = [
            *(_verified_record(1, "mixed_benchmark", index) for index in range(2_500)),
            *(_verified_record(1, "positive_registry", index) for index in range(2_500)),
        ]
        negatives = [
            *(_verified_record(0, "mixed_benchmark", index + 10_000) for index in range(2_500)),
            *(_verified_record(0, "negative_registry", index + 10_000) for index in range(2_500)),
        ]
        source_stats = {
            "mixed_benchmark": {"independentAcquisition": True},
            "positive_registry": {"independentAcquisition": True},
            "negative_registry": {"independentAcquisition": True},
        }

        silver, audit = build_audited_phishing_silver(
            pd.DataFrame(positives),
            pd.DataFrame(negatives),
            per_class=5_000,
            seed=42,
            source_stats=source_stats,
        )

        self.assertEqual(len(silver), 10_000)
        self.assertTrue(audit["readiness"]["readyForThesisTraining"])
        self.assertTrue(audit["biasAudit"]["negativeLabelsIndependentlyVerified"])
        self.assertEqual(audit["biasAudit"]["mixedLabelSources"], ["mixed_benchmark"])
        self.assertEqual(len(audit["biasAudit"]["sourcesPerLabel"]["0"]), 2)
        self.assertEqual(len(audit["biasAudit"]["sourcesPerLabel"]["1"]), 2)

    def test_curated_package_verifies_hash_and_row_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "verified.csv"
            pd.DataFrame([{
                "url": "https://example.test/account",
                "is_phishing": 0,
                "source_record_id": "n-1",
                "label_verified": True,
                "verification_method": "expert_consensus",
                "verification_reference": "https://evidence.test/n-1",
                "verified_at": "2026-01-01T00:00:00Z",
            }]).to_csv(source, index=False)
            digest, _ = sha256_file(source)
            manifest = root / "curation_manifest.json"
            manifest.write_text(json.dumps({
                "schemaVersion": "1.0.0",
                "studyId": "thesis-fixture",
                "sources": [{
                    "sourceId": "curated_fixture",
                    "provider": "Fixture provider",
                    "citation": "https://dataset.test/fixture",
                    "license": "CC-BY-4.0",
                    "independentAcquisition": True,
                    "declaredLabels": [0],
                    "path": source.name,
                    "sha256": digest,
                }],
            }), encoding="utf-8")

            frame, audit = load_curated_source_package(manifest)

            self.assertEqual(len(frame), 1)
            self.assertTrue(audit["valid"])
            self.assertEqual(audit["sources"][0]["verifiedNegativeRows"], 1)
            status = get_phishing_curation_status(manifest)
            self.assertFalse(status["readyForScientificMerge"])
            self.assertTrue(status["scientificReasons"])
            source.write_text(source.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                load_curated_source_package(manifest)

    def test_normalizes_idna_and_uses_public_suffix_for_grouping(self) -> None:
        normalized = normalize_url("HTTPS://Sub.Example.CO.UK:443/path#fragment")

        self.assertIsNotNone(normalized)
        self.assertEqual(normalized["canonical_url"], "https://sub.example.co.uk/path")
        self.assertEqual(normalized["registrable_domain"], "example.co.uk")

    def test_removes_label_conflicts_and_prevents_group_leakage(self) -> None:
        positives = [_record(f"https://p{index}.bad{index}.test/login", 1, "phishtank", index) for index in range(100)]
        negatives = [_record(f"https://good{index}.com/", 0, "tranco", index) for index in range(100)]
        positives.append(_record("https://shop.conflict.com/a", 1, "phishtank", 501))
        negatives.append(_record("https://www.conflict.com/", 0, "tranco", 502))
        positives.append(positives[0].copy())

        first, audit = build_audited_phishing_silver(
            pd.DataFrame(positives),
            pd.DataFrame(negatives),
            per_class=100,
            seed=42,
        )
        second, _ = build_audited_phishing_silver(
            pd.DataFrame(positives),
            pd.DataFrame(negatives),
            per_class=100,
            seed=42,
        )

        self.assertNotIn("conflict.com", set(first["registrable_domain"]))
        self.assertEqual(audit["transformations"]["conflictingRegistrableDomainsRemoved"], 1)
        self.assertEqual(audit["transformations"]["exactDuplicateRowsRemoved"], 1)
        self.assertTrue(audit["leakageAudit"]["passed"])
        self.assertEqual(first.to_dict(orient="records"), second.to_dict(orient="records"))
        for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
            left_groups = set(first.loc[first["split"] == left, "registrable_domain"])
            right_groups = set(first.loc[first["split"] == right, "registrable_domain"])
            self.assertFalse(left_groups & right_groups)

    def test_verified_snapshot_is_reused_without_network(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "source_v1_test.csv"
            raw.write_text("rank,domain\n1,example.com\n", encoding="utf-8")
            digest, size = sha256_file(raw)
            source_url = "https://example.test/source.csv"
            metadata = {
                "sourceUrl": source_url,
                "sourceVersion": "v1",
                "retrievedAt": "2026-01-01T00:00:00Z",
                "rawPath": str(raw),
                "sha256": digest,
                "bytes": size,
                "http": {},
                "reused": False,
            }
            (root / "latest.json").write_text(json.dumps(metadata), encoding="utf-8")

            reused = download_versioned_source(
                name="source",
                source_url=source_url,
                source_version="v1",
                raw_directory=root,
            )

            self.assertTrue(reused["reused"])
            self.assertEqual(reused["sha256"], digest)

    def test_data_preparation_job_persists_scientific_readiness(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manager = PhishingDataPreparationManager(state_path=Path(directory) / "job.json")
            fake_result = {
                "metadata": {"datasetId": "phishing-test", "silver": {"rows": 20_000}},
                "audit": {"readiness": {"readyForPipelinePilot": True, "readyForThesisTraining": False}},
            }
            try:
                with patch("app.phishing.data_jobs.prepare_real_phishing_dataset", return_value=fake_result):
                    manager.submit(per_class=10_000)
                    manager._executor.shutdown(wait=True)
                completed = manager.latest()
                self.assertEqual(completed["status"], "completed")
                self.assertTrue(completed["result"]["readyForPipelinePilot"])
                self.assertFalse(completed["result"]["readyForThesisTraining"])
            finally:
                manager.shutdown()

    def test_data_job_builds_academic_package_before_scientific_silver(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manager = PhishingDataPreparationManager(state_path=Path(directory) / "job.json")
            curation_result = {"status": {"readyForScientificMerge": True}}
            dataset_result = {
                "metadata": {"datasetId": "scientific-dataset", "silver": {"rows": 20_000}},
                "audit": {"readiness": {"readyForPipelinePilot": True, "readyForThesisTraining": True}},
            }
            try:
                with (
                    patch("app.phishing.data_jobs.build_academic_curation_package", return_value=curation_result) as build,
                    patch("app.phishing.data_jobs.prepare_real_phishing_dataset", return_value=dataset_result) as prepare,
                ):
                    manager.submit(per_class=10_000, include_academic_sources=True)
                    manager._executor.shutdown(wait=True)
                completed = manager.latest()
                build.assert_called_once()
                prepare.assert_called_once()
                self.assertTrue(completed["result"]["academicCurationReady"])
                self.assertTrue(completed["result"]["readyForThesisTraining"])
            finally:
                manager.shutdown()


if __name__ == "__main__":
    unittest.main()
