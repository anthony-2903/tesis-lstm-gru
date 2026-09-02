from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from app.phishing.ingestion import sha256_file
from app.phishing.validation import (
    confusion_matrix_counts,
    run_phishing_external_validation,
    select_mcc_threshold,
)
from app.utils import write_json


class _FakeFinalClassifier:
    def __init__(self, *, model_id: str, epochs: int, **kwargs) -> None:
        self.model_id = model_id
        self.epochs = epochs

    def fit_full(self, x_train, y_train, *, class_weight=None):
        assert len(x_train) == len(y_train)
        assert class_weight is not None
        return {"trainTimeSeconds": 0.01, "epochsCompleted": self.epochs}

    def predict_proba(self, x):
        offset = 0.01 if self.model_id == "gru" else 0.0
        return np.linspace(0.05 + offset, 0.95 - offset, len(x)), 0.001

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = f"{self.model_id}:{self.epochs}".encode("utf-8")
        path.write_bytes(payload)
        return hashlib.sha256(payload).hexdigest(), len(payload)


class PhishingExternalValidationTests(unittest.TestCase):
    def test_threshold_selection_and_confusion_matrix(self) -> None:
        labels = np.asarray([0, 0, 1, 1])
        scores = np.asarray([0.1, 0.4, 0.6, 0.9])
        threshold = select_mcc_threshold(labels, scores)
        matrix = confusion_matrix_counts(labels, scores, threshold)

        self.assertEqual(sum(matrix.values()), 4)
        self.assertEqual(matrix["falsePositive"], 0)
        self.assertEqual(matrix["falseNegative"], 0)

    def test_external_validation_uses_oof_for_meta_fit_and_keeps_test_locked(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            silver_path = root / "phishing.csv"
            rows = []
            train_sample_ids = []
            for split, per_class in (("train", 30), ("validation", 10), ("test", 10)):
                for label in (0, 1):
                    for index in range(per_class):
                        url = f"https://{split}-{label}-{index}.domain-{split}-{label}-{index}.example/path"
                        sample_id = hashlib.sha256(url.encode("utf-8")).hexdigest()
                        rows.append({
                            "canonical_url": url,
                            "registrable_domain": f"domain-{split}-{label}-{index}.example",
                            "is_phishing": label,
                            "split": split,
                        })
                        if split == "train":
                            train_sample_ids.append((sample_id, label, index))
            pd.DataFrame(rows).to_csv(silver_path, index=False)
            source_hash, source_bytes = sha256_file(silver_path)

            oof_rows = []
            for sample_id, label, index in train_sample_ids:
                signal = 0.2 if label == 0 else 0.8
                oof_rows.append({
                    "sample_id": sample_id,
                    "seed": 42,
                    "fold": index % 5,
                    "is_phishing": label,
                    "probability_lstm": signal,
                    "probability_gru": signal + (0.02 if label == 0 else -0.02),
                })
            oof_path = root / "oof.csv"
            pd.DataFrame(oof_rows).to_csv(oof_path, index=False)
            oof_hash, oof_bytes = sha256_file(oof_path)
            base_manifest_path = root / "oof_manifest.json"
            write_json(base_manifest_path, {
                "schemaVersion": "1.0.0",
                "runId": "validation-fixture",
                "status": "demo",
                "configuration": {"seeds": [42], "batchSize": 16, "epochs": 2},
                "dataset": {
                    "sourcePath": str(silver_path.resolve()),
                    "sourceSha256": source_hash,
                    "sourceBytes": source_bytes,
                    "demoSubset": False,
                },
                "baseModels": ["lstm", "gru"],
                "baseFoldMetrics": [
                    {"seed": 42, "fold": fold, "modelId": model_id, "epochsCompleted": 2}
                    for fold in range(5) for model_id in ("lstm", "gru")
                ],
                "artifacts": {"oofProbabilities": {"path": str(oof_path.resolve()), "sha256": oof_hash, "bytes": oof_bytes}},
            })
            sequence_manifest_path = root / "sequence_manifest.json"
            write_json(sequence_manifest_path, {"configuration": {"maxVocabulary": 128, "lengthPercentile": 99.0, "maxLengthCap": 128}})

            manifest = run_phishing_external_validation(
                base_manifest_path=base_manifest_path,
                silver_path=silver_path,
                sequence_manifest_path=sequence_manifest_path,
                output_dir=root / "validation_run",
                classifier_factory=_FakeFinalClassifier,
            )

            predictions = pd.read_csv(manifest["artifacts"]["predictionsBySeed"]["path"])
            test_urls = {row["canonical_url"] for row in rows if row["split"] == "test"}
            test_ids = {hashlib.sha256(url.encode("utf-8")).hexdigest() for url in test_urls}
            self.assertEqual(len(predictions), 20)
            self.assertFalse(set(predictions["sample_id"]) & test_ids)
            self.assertEqual(len(manifest["comparison"]), 2 + 6)
            self.assertEqual(manifest["metaFeatures"]["fitSplit"], "development_oof_only")
            self.assertFalse(manifest["metaFeatures"]["validationLabelsUsedForMetaFit"])
            self.assertTrue(manifest["validation"]["outerValidationUsed"])
            self.assertTrue(manifest["validation"]["testSetLocked"])
            self.assertFalse(manifest["validation"]["testFeaturesEncoded"])
            self.assertFalse(manifest["validation"]["testSetUsed"])
            for candidate in manifest["comparison"]:
                self.assertEqual(sum(candidate["confusionMatrix"].values()), 20)
                self.assertGreaterEqual(candidate["calibratedThreshold"], 0.0)
                self.assertLessEqual(candidate["calibratedThreshold"], 1.0)


if __name__ == "__main__":
    unittest.main()
