from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from app.phishing.data import (
    CharacterTokenizer,
    UNKNOWN_ID,
    get_phishing_sequence_status,
    prepare_phishing_sequence_protocol,
    validate_phishing_sequence_frame,
)
from app.phishing.ingestion import sha256_file


class PhishingSequenceDataTests(unittest.TestCase):
    def test_sequence_status_rejects_stale_dataset_lineage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            silver_dir = root / "silver"
            results_dir = root / "results"
            silver_dir.mkdir()
            results_dir.mkdir()
            silver_path = silver_dir / "phishing.csv"
            metadata_path = silver_dir / "phishing.metadata.json"
            rows = []
            for label in (0, 1):
                for split, count in (("train", 18), ("validation", 4), ("test", 4)):
                    for index in range(count):
                        rows.append(_row(
                            f"https://{split}-{label}-{index}.lineage-{split}-{label}-{index}.test/path",
                            f"lineage-{split}-{label}-{index}.test",
                            label,
                            split,
                        ))
            pd.DataFrame(rows).to_csv(silver_path, index=False)
            digest, size = sha256_file(silver_path)
            metadata_path.write_text(json.dumps({
                "datasetId": "dataset-v1",
                "readyForThesisTraining": True,
                "silver": {"sha256": digest, "bytes": size},
            }), encoding="utf-8")
            prepare_phishing_sequence_protocol(
                silver_path=silver_path,
                metadata_path=metadata_path,
                assignments_path=silver_dir / "phishing_oof_assignments.csv",
                tokenizer_directory=root / "tokenizers",
                audit_path=results_dir / "phishing_sequence_audit.json",
                manifest_path=silver_dir / "phishing_sequence_manifest.json",
                folds=3,
            )

            with (
                patch("app.phishing.data.SILVER_DIR", silver_dir),
                patch("app.phishing.data.RESULTS_DIR", results_dir),
            ):
                self.assertTrue(get_phishing_sequence_status()["readyForBaseModelTraining"])
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                metadata["datasetId"] = "dataset-v2"
                metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
                stale = get_phishing_sequence_status()

            self.assertFalse(stale["lineageCurrent"])
            self.assertFalse(stale["readyForBaseModelTraining"])
            self.assertIn("versión anterior", stale["message"])

    def test_tokenizer_learns_only_provided_training_characters(self) -> None:
        tokenizer = CharacterTokenizer.fit(["https://aaaa.test/", "https://bbbb.test/"], max_vocabulary=64)

        encoded = tokenizer.encode(["https://zzzz.test/"])

        self.assertNotIn("z", tokenizer.token_to_id)
        self.assertIn(UNKNOWN_ID, encoded[0])
        self.assertEqual(tokenizer.statistics(["https://zzzz.test/"])["unknownCharacters"], 4)

    def test_head_tail_truncation_preserves_both_url_ends(self) -> None:
        tokenizer = CharacterTokenizer.fit(["abcdefghij"], max_vocabulary=32, max_length_cap=8)

        encoded = tokenizer.encode(["abcdefghij"])[0]
        decoded = "".join(
            next(character for character, identifier in tokenizer.token_to_id.items() if identifier == token)
            for token in encoded
        )

        self.assertEqual(decoded, "abcdehij")

    def test_external_domain_overlap_is_rejected(self) -> None:
        frame = pd.DataFrame([
            _row("https://a.example.com/", "example.com", 0, "train"),
            _row("https://b.bad.test/", "bad.test", 1, "train"),
            _row("https://c.example.com/", "example.com", 0, "validation"),
            _row("https://d.phish.test/", "phish.test", 1, "validation"),
            _row("https://e.clean.test/", "clean.test", 0, "test"),
            _row("https://f.attack.test/", "attack.test", 1, "test"),
        ])

        with self.assertRaisesRegex(ValueError, "fuga"):
            validate_phishing_sequence_frame(frame)

    def test_protocol_covers_outer_train_once_and_keeps_test_locked(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            silver_path = root / "phishing.csv"
            metadata_path = root / "phishing.metadata.json"
            assignments_path = root / "assignments.csv"
            tokenizers = root / "tokenizers"
            audit_path = root / "audit.json"
            manifest_path = root / "manifest.json"
            rows = []
            for label in (0, 1):
                for index in range(50):
                    rows.append(_row(
                        f"https://train-{label}-{index}.domain-{label}-{index}.com/path",
                        f"domain-{label}-{index}.com",
                        label,
                        "train",
                    ))
                for index in range(10):
                    rows.append(_row(
                        f"https://validation-{label}-{index}.validation-{label}-{index}.org/~",
                        f"validation-{label}-{index}.org",
                        label,
                        "validation",
                    ))
                    rows.append(_row(
                        f"https://test-{label}-{index}.test-{label}-{index}.net/^",
                        f"test-{label}-{index}.net",
                        label,
                        "test",
                    ))
            pd.DataFrame(rows).to_csv(silver_path, index=False)
            digest, size = sha256_file(silver_path)
            metadata_path.write_text(json.dumps({
                "datasetId": "phishing-fixture",
                "silver": {"sha256": digest, "bytes": size},
            }), encoding="utf-8")

            result = prepare_phishing_sequence_protocol(
                silver_path=silver_path,
                metadata_path=metadata_path,
                assignments_path=assignments_path,
                tokenizer_directory=tokenizers,
                audit_path=audit_path,
                manifest_path=manifest_path,
                folds=5,
                seed=42,
                max_vocabulary=256,
            )

            assignments = pd.read_csv(assignments_path)
            outer_tokenizer = json.loads((tokenizers / "outer_train.json").read_text(encoding="utf-8"))
            self.assertEqual(len(assignments), 100)
            self.assertEqual(set(assignments["oof_fold"]), {0, 1, 2, 3, 4})
            self.assertTrue(result["audit"]["oof"]["groupLeakagePassed"])
            self.assertEqual(result["audit"]["oof"]["coverageRows"], 100)
            self.assertTrue(result["audit"]["testLock"]["locked"])
            self.assertFalse(result["audit"]["testLock"]["evaluated"])
            self.assertNotIn("~", outer_tokenizer["tokenToId"])
            self.assertNotIn("^", outer_tokenizer["tokenToId"])
            self.assertTrue(result["manifest"]["readyForBaseModelTraining"])


def _row(url: str, group: str, label: int, split: str) -> dict:
    return {
        "canonical_url": url,
        "registrable_domain": group,
        "is_phishing": label,
        "split": split,
    }


if __name__ == "__main__":
    unittest.main()
