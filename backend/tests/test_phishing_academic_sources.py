from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from app.phishing.academic_sources import (
    ACADEMIC_URL_SOURCES,
    build_academic_curation_package,
    normalize_academic_source_frame,
)
from app.phishing.ingestion import sha256_file


class PhishingAcademicSourceTests(unittest.TestCase):
    def test_maps_published_legitimate_and_phishing_labels_to_internal_contract(self) -> None:
        definition = replace(ACADEMIC_URL_SOURCES[0], minimum_rows_per_label=1)
        frame = pd.DataFrame({
            "URL": ["https://legitimate.test/page", "https://phishing.test/login", "https://missing.test/"],
            "label": [1, 0, None],
        })

        curated, report = normalize_academic_source_frame(definition, frame, max_rows_per_label=5)

        by_url = dict(zip(curated["url"], curated["is_phishing"]))
        self.assertEqual(by_url["https://legitimate.test/page"], 0)
        self.assertEqual(by_url["https://phishing.test/login"], 1)
        self.assertTrue(curated["label_verified"].all())
        self.assertEqual(set(curated["verification_method"]), {"published_dataset_ground_truth"})
        self.assertEqual(report["labelMapping"], {"1": 0, "0": 1})
        self.assertEqual(report["missingLabelRowsDiscarded"], 1)

    def test_rejects_an_unexpected_published_label(self) -> None:
        definition = replace(ACADEMIC_URL_SOURCES[0], minimum_rows_per_label=1)
        frame = pd.DataFrame({"URL": ["https://unknown.test/"], "label": [2]})

        with self.assertRaisesRegex(ValueError, "etiquetas observadas"):
            normalize_academic_source_frame(definition, frame, max_rows_per_label=5)

    def test_builds_a_hashed_two_source_package_ready_for_scientific_merge(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw_files: dict[str, Path] = {}
            definitions = []
            for index, source_id in enumerate(("academic_a", "academic_b")):
                raw = root / f"{source_id}.csv"
                pd.DataFrame({
                    "URL": [
                        f"https://{source_id}-legit-{row}.test/page"
                        for row in range(2)
                    ] + [
                        f"https://{source_id}-phish-{row}.test/login"
                        for row in range(2)
                    ],
                    "label": [1, 1, 0, 0],
                }).to_csv(raw, index=False)
                digest, size = sha256_file(raw)
                raw_files[source_id] = raw
                definitions.append(replace(
                    ACADEMIC_URL_SOURCES[0],
                    source_id=source_id,
                    filename=raw.name,
                    repository_sha256=digest,
                    repository_bytes=size,
                    minimum_rows_per_label=1,
                ))

            def fake_resolve(definition):
                return {
                    "fileId": f"file-{definition.source_id}",
                    "downloadUrl": f"https://example.test/{definition.filename}",
                    "sha256": definition.repository_sha256,
                    "bytes": definition.repository_bytes,
                }

            def fake_download(*, name, **_kwargs):
                raw = raw_files[name]
                digest, size = sha256_file(raw)
                return {"rawPath": str(raw), "sha256": digest, "bytes": size}

            manifest = root / "curated" / "curation_manifest.json"
            with (
                patch("app.phishing.academic_sources.ACADEMIC_URL_SOURCES", tuple(definitions)),
                patch("app.phishing.academic_sources.resolve_mendeley_file", side_effect=fake_resolve),
                patch("app.phishing.academic_sources.download_versioned_source", side_effect=fake_download),
            ):
                result = build_academic_curation_package(
                    raw_directory=root / "raw",
                    manifest_path=manifest,
                    max_rows_per_label_per_source=5_000,
                )

            self.assertTrue(manifest.is_file())
            self.assertTrue(result["status"]["readyForScientificMerge"])
            self.assertEqual(result["status"]["sourcesPerLabel"]["0"], ["academic_a", "academic_b"])
            self.assertEqual(result["status"]["negativeEvidence"], {"verified": 4, "total": 4})


if __name__ == "__main__":
    unittest.main()
