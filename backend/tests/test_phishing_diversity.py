from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from app.phishing.diversity import run_phishing_diversity_ablation
from app.phishing.ingestion import sha256_file
from app.utils import write_json


MODEL_IDS = ("lstm", "gru", "brnn")


class PhishingDiversityAblationTests(unittest.TestCase):
    def test_diversity_and_leave_one_out_cover_the_same_validation_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            silver_path = root / "phishing.csv"
            silver_rows = []
            train_rows = []
            validation_rows = []
            for split, per_class in (("train", 20), ("validation", 8), ("test", 5)):
                for label in (0, 1):
                    for index in range(per_class):
                        url = f"https://{split}-{label}-{index}.group-{split}-{label}-{index}.example/path"
                        sample_id = hashlib.sha256(url.encode("utf-8")).hexdigest()
                        silver_rows.append({
                            "canonical_url": url,
                            "registrable_domain": f"group-{split}-{label}-{index}.example",
                            "is_phishing": label,
                            "split": split,
                        })
                        if split == "train":
                            train_rows.append((sample_id, label, index))
                        elif split == "validation":
                            validation_rows.append((sample_id, label, index))
            pd.DataFrame(silver_rows).to_csv(silver_path, index=False)

            oof = []
            for sample_id, label, index in train_rows:
                center = 0.18 if label == 0 else 0.82
                oof.append({
                    "sample_id": sample_id,
                    "seed": 42,
                    "fold": index % 5,
                    "is_phishing": label,
                    "probability_lstm": center,
                    "probability_gru": center + (0.04 if index % 3 == 0 else -0.01),
                    "probability_brnn": center + (0.08 if label == 0 and index % 4 == 0 else -0.02),
                })
            oof_path = root / "oof.csv"
            pd.DataFrame(oof).to_csv(oof_path, index=False)
            oof_hash, oof_bytes = sha256_file(oof_path)
            base_manifest_path = root / "oof_manifest.json"
            write_json(base_manifest_path, {
                "runId": "diversity-fixture",
                "status": "demo",
                "baseModels": list(MODEL_IDS),
                "configuration": {"seeds": [42]},
                "artifacts": {"oofProbabilities": {"path": str(oof_path.resolve()), "sha256": oof_hash, "bytes": oof_bytes}},
            })

            external_rows = []
            for sample_id, label, index in validation_rows:
                center = 0.2 if label == 0 else 0.8
                external_rows.append({
                    "sample_id": sample_id,
                    "seed": 42,
                    "is_phishing": label,
                    "probability_lstm": center,
                    "probability_gru": center + (0.15 if index % 5 == 0 else -0.02),
                    "probability_brnn": center + (-0.18 if label == 1 and index % 4 == 0 else 0.01),
                })
            by_seed_path = root / "validation_by_seed.csv"
            mean_path = root / "validation_mean.csv"
            pd.DataFrame(external_rows).to_csv(by_seed_path, index=False)
            pd.DataFrame(external_rows).drop(columns=["seed"]).to_csv(mean_path, index=False)
            by_seed_hash, by_seed_bytes = sha256_file(by_seed_path)
            mean_hash, mean_bytes = sha256_file(mean_path)
            validation_manifest_path = root / "external_validation_manifest.json"
            write_json(validation_manifest_path, {
                "runId": "external-fixture",
                "status": "demo_validation_selection",
                "baseRun": {"manifestPath": str(base_manifest_path.resolve())},
                "dataset": {"sourcePath": str(silver_path.resolve())},
                "selection": {"leadingStackingCandidateId": "stacking_ridge"},
                "comparison": [
                    {"candidateId": model_id, "calibratedThreshold": 0.5}
                    for model_id in MODEL_IDS
                ],
                "artifacts": {
                    "predictionsBySeed": {"path": str(by_seed_path.resolve()), "sha256": by_seed_hash, "bytes": by_seed_bytes},
                    "meanPredictions": {"path": str(mean_path.resolve()), "sha256": mean_hash, "bytes": mean_bytes},
                },
            })

            manifest = run_phishing_diversity_ablation(
                validation_manifest_path=validation_manifest_path,
                output_dir=root / "analysis",
                bootstrap_iterations=20,
            )

            predictions = pd.read_csv(manifest["artifacts"]["ablationPredictions"]["path"])
            self.assertEqual(len(manifest["diversity"]["pairs"]), 3)
            self.assertEqual(len(manifest["ablation"]["configurations"]), 4)
            self.assertEqual(len(manifest["ablation"]["metrics"]), 4 * 3)
            self.assertEqual(len(manifest["ablation"]["contribution"]), 3)
            self.assertEqual(len(predictions), 16 * 4 * 3)
            self.assertTrue(manifest["validation"]["sameRowsForAllAblations"])
            self.assertTrue(manifest["validation"]["testSetLocked"])
            self.assertFalse(manifest["validation"]["testFeaturesEncoded"])
            self.assertFalse(manifest["validation"]["testSetUsed"])
            self.assertEqual(len(manifest["artifacts"]["fittedObjects"]), 4 * 3)


if __name__ == "__main__":
    unittest.main()
